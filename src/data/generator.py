"""
generator.py — 確定性場景生成器 scene(i) = f(i, config)

機制(整理規則 §2)
-------------------
- **純函數**:給定 scene index `i` + `GenConfig` + `Assets`,輸出唯一確定的場景。
  不依賴時間 / 全域 RNG / per-run 狀態。
- **兩條正交軸**:
    1. 確定性 —— 同 `i` + 同 config → 同場景(RNG 由 SeedSequence([seed, i]) 決定)。
    2. 不重複 —— 每個 `i` 餵不同 distinct 場景;組合空間極大,以 total steps 思考。
- 與舊 composite.py 的差異:舊版用單一 RandomState 串流跑 1000 張(scene i 依賴前面
  所有抽樣、且寫死存檔)。這裡每個 i 開獨立 RNG、回傳記憶體 arrays,可在訓練迴圈
  當場生成,**不存訓練圖**。

合成演算法本身忠實沿用 composite.py(base layer → 增強 → distractors → 貼零件 +
產生 semantic/instance mask + meta),只是改成 config 驅動。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from src.data.assets import Assets, REPO_ROOT
from src.data.config import (
    CLS_DEFECT, CLS_NORMAL, DEFAULT_CONFIG, DEFECT_STATES_DEFECTIVE, GenConfig,
)


@dataclass
class SceneSample:
    """一張當場生成的場景(記憶體,不落地)。

    rgb:      (H, W, 3) uint8
    semantic: (H, W)    uint8  0=bg / 1=normal / 2=defect(與 Stage 3 schema 可比)
    instance: (H, W)    uint8  0=bg / 1..N=instance id
    meta:     dict      含 instances[](每個有 defect_state),供 schema 編碼 label
    """
    rgb: np.ndarray
    semantic: np.ndarray
    instance: np.ndarray
    meta: dict


# ── per-scene RNG ──────────────────────────────────────────
def scene_rng(i: int, config: GenConfig = DEFAULT_CONFIG) -> np.random.RandomState:
    """由 (config.seed, i) 決定的獨立 RNG。同輸入必同輸出,不同 i 必不同流。"""
    state = np.random.SeedSequence([int(config.seed), int(i)]).generate_state(4)
    return np.random.RandomState(state)


# ── 抽樣 helper(config 驅動)────────────────────────────────
def _categorical(items, rng):
    """從 {k:p} 或 [(k,p),...] 抽一個 key(cumulative thresholding)。"""
    if isinstance(items, dict):
        items = list(items.items())
    r = rng.random()
    cum = 0.0
    for k, p in items:
        cum += p
        if r < cum:
            return k
    return items[-1][0]


def _sample_parts(n, normal_pool, defect_pool, defect_prob, rng):
    chosen = []
    for _ in range(n):
        pool = defect_pool if rng.random() < defect_prob else normal_pool
        chosen.append(pool[rng.randint(len(pool))])
    return chosen


def _generate_base_layer(bg_pools, cfg, rng):
    kind = _categorical(cfg.base_layer_probs, rng)
    if kind in ("real", "procedural") and bg_pools[kind]:
        path = bg_pools[kind][rng.randint(len(bg_pools[kind]))]
        img = Image.open(path).convert("RGB")
        source = os.path.basename(path)
    else:  # solid(或對應 pool 空時的 fallback)
        kind = "solid"
        color = cfg.solid_color_palette[rng.randint(len(cfg.solid_color_palette))]
        arr = np.full((cfg.out_size * 2, cfg.out_size * 2, 3), color, dtype=np.float32)
        arr = np.clip(arr + rng.normal(0, cfg.solid_noise_std, arr.shape), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        source = f"solid_rgb{color}"
    if img.width < cfg.out_size or img.height < cfg.out_size:
        scale = cfg.out_size / min(img.width, img.height) * 1.1
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    return img, kind, source


def _augment_background(img, cfg, rng):
    if rng.random() < cfg.bg_aug_flip_h_prob:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < cfg.bg_aug_flip_v_prob:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(*cfg.bg_aug_brightness))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(*cfg.bg_aug_contrast))
    img = ImageEnhance.Color(img).enhance(rng.uniform(*cfg.bg_aug_saturation))
    return img


def _sample_distractor_count(cfg, rng):
    r = rng.random()
    cum = 0.0
    for (lo, hi), p in cfg.distractor_count_buckets:
        cum += p
        if r < cum:
            return rng.randint(lo, hi + 1) if hi > lo else lo
    lo, hi = cfg.distractor_count_buckets[-1][0]
    return rng.randint(lo, hi + 1) if hi > lo else lo


def _random_color(cfg, rng):
    if rng.random() < cfg.distractor_color_metallic_prob:
        return cfg.distractor_metallic_palette[rng.randint(len(cfg.distractor_metallic_palette))]
    return (int(rng.randint(0, 256)), int(rng.randint(0, 256)), int(rng.randint(0, 256)))


def _draw_distractors(canvas, cfg, rng):
    n = _sample_distractor_count(cfg, rng)
    if n == 0:
        return 0, []
    H, W = canvas.shape[:2]
    pil = Image.fromarray(canvas).convert("RGBA")
    shape_log = []
    for _ in range(n):
        shape = _categorical(cfg.distractor_shape_probs, rng)
        shape_log.append(shape)
        color = _random_color(cfg, rng)
        alpha = rng.uniform(*cfg.distractor_alpha_range)
        rgba_color = (color[0], color[1], color[2], int(alpha * 255))
        size = rng.randint(cfg.distractor_size_range[0], cfg.distractor_size_range[1] + 1)
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        x0, y0 = cx - size // 2, cy - size // 2
        x1, y1 = x0 + size, y0 + size
        overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        if shape == "ellipse":
            d.ellipse([x0, y0, x1, y1], fill=rgba_color)
        elif shape == "rectangle":
            d.rectangle([x0, y0, x1, y1], fill=rgba_color)
        elif shape == "ring":
            d.ellipse([x0, y0, x1, y1], outline=rgba_color, width=max(1, size // 12))
        elif shape == "line":
            angle = rng.uniform(0, 2 * math.pi)
            length = size * 2
            x2 = int(cx + math.cos(angle) * length / 2)
            y2 = int(cy + math.sin(angle) * length / 2)
            x3 = int(cx - math.cos(angle) * length / 2)
            y3 = int(cy - math.sin(angle) * length / 2)
            d.line([x2, y2, x3, y3], fill=rgba_color, width=max(1, size // 8))
        elif shape == "polygon":
            n_sides = rng.randint(3, 7)
            pts = []
            for k in range(n_sides):
                ang = 2 * math.pi * k / n_sides + rng.uniform(-0.3, 0.3)
                r_ = size / 2 * rng.uniform(0.6, 1.0)
                pts.append((cx + r_ * math.cos(ang), cy + r_ * math.sin(ang)))
            d.polygon(pts, fill=rgba_color)
        pil = Image.alpha_composite(pil, overlay)
    canvas[:] = np.array(pil.convert("RGB"))
    return n, shape_log


def _transform_part(rgba, cfg, rng):
    rot = rng.uniform(*cfg.rot_range)
    scale = rng.uniform(*cfg.scale_range)
    img = Image.fromarray(rgba, mode="RGBA")
    img = img.rotate(rot, resample=Image.BILINEAR, expand=True)
    nw, nh = img.size
    img = img.resize((max(1, int(nw * scale)), max(1, int(nh * scale))), Image.BILINEAR)
    return np.array(img), scale, rot


def _alpha_bbox(alpha, threshold=10):
    ys, xs = np.where(alpha > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ── 主入口 ─────────────────────────────────────────────────
def generate_scene(i: int, assets: Assets, config: GenConfig = DEFAULT_CONFIG) -> SceneSample:
    """生成第 i 張場景(純函數,記憶體輸出)。演算法同 composite.py。"""
    cfg = config
    rng = scene_rng(i, cfg)
    S = cfg.out_size

    # 0) 先抽 HDRI subset(同 scene 反射物理一致;Stage 4 修正)
    hdri_keys = assets.hdri_keys
    hdri_idx = hdri_keys[rng.randint(len(hdri_keys))]
    normal_pool = assets.part_buckets[hdri_idx]["normal"]
    defect_pool = assets.part_buckets[hdri_idx]["defect"]

    # 1) base layer + 增強 + 隨機裁切
    bg_pil, base_kind, base_source = _generate_base_layer(assets.bg_pools, cfg, rng)
    bg_pil = _augment_background(bg_pil, cfg, rng)
    bg = np.array(bg_pil)
    bh, bw = bg.shape[:2]
    bx = rng.randint(0, max(1, bw - S))
    by = rng.randint(0, max(1, bh - S))
    canvas = bg[by:by + S, bx:bx + S].copy()

    # 2) distractors(貼零件前;不落 semantic mask)
    n_distractors, distractor_shapes = _draw_distractors(canvas, cfg, rng)

    semantic = np.zeros((S, S), dtype=np.uint8)
    instance = np.zeros((S, S), dtype=np.uint8)

    # 3) 抽零件並逐一貼上
    n_parts = rng.randint(cfg.parts_range[0], cfg.parts_range[1] + 1)
    chosen = _sample_parts(n_parts, normal_pool, defect_pool, cfg.defect_prob, rng)

    placed_meta = []
    for inst_id, (part_path, defect_state) in enumerate(chosen, start=1):
        rgba = np.array(Image.open(part_path).convert("RGBA"))
        rgba, scale, rot = _transform_part(rgba, cfg, rng)

        bb = _alpha_bbox(rgba[:, :, 3])
        if bb is None:
            continue
        bx0, by0, bx1, by1 = bb
        part_w, part_h = bx1 - bx0, by1 - by0
        if part_w < 4 or part_h < 4:
            continue
        part_crop = rgba[by0:by1, bx0:bx1]

        cx, cy = rng.randint(0, S), rng.randint(0, S)
        px, py = cx - part_w // 2, cy - part_h // 2
        x_a, y_a = max(0, px), max(0, py)
        x_b, y_b = min(S, px + part_w), min(S, py + part_h)
        if x_a >= x_b or y_a >= y_b:
            continue

        cx0, cy0 = x_a - px, y_a - py
        cx1, cy1 = cx0 + (x_b - x_a), cy0 + (y_b - y_a)
        sub_rgba = part_crop[cy0:cy1, cx0:cx1]
        sub_alpha = sub_rgba[:, :, 3].astype(np.float32) / 255.0
        sub_rgb = sub_rgba[:, :, :3].astype(np.float32)

        region = canvas[y_a:y_b, x_a:x_b].astype(np.float32)
        blended = sub_rgb * sub_alpha[..., None] + region * (1 - sub_alpha[..., None])
        canvas[y_a:y_b, x_a:x_b] = np.clip(blended, 0, 255).astype(np.uint8)

        is_part = sub_alpha > 0.5
        cls_val = CLS_DEFECT if defect_state in DEFECT_STATES_DEFECTIVE else CLS_NORMAL
        semantic[y_a:y_b, x_a:x_b][is_part] = cls_val
        instance[y_a:y_b, x_a:x_b][is_part] = inst_id

        placed_meta.append({
            "instance_id": inst_id,
            "source_part": os.path.relpath(part_path, REPO_ROOT).replace("\\", "/"),
            "defect_state": defect_state,
            "is_defective": defect_state in DEFECT_STATES_DEFECTIVE,
            "bbox": [x_a, y_a, x_b, y_b],
            "scale": round(float(scale), 3),
            "rotation_deg": round(float(rot), 1),
        })

    meta = {
        "scene_id": int(i),
        "hdri_idx": int(hdri_idx),
        "bg_kind": base_kind,
        "bg_source": base_source,
        "bg_crop": [int(bx), int(by)],
        "n_distractors": int(n_distractors),
        "distractor_shapes": distractor_shapes,
        "instances": placed_meta,
        "n_parts": len(placed_meta),
        "n_defective": sum(1 for m in placed_meta if m["is_defective"]),
    }
    return SceneSample(rgb=canvas, semantic=semantic, instance=instance, meta=meta)


def write_scene(sample: SceneSample, scene_dir: str | os.PathLike) -> None:
    """把一張場景落地成磁碟格式(與舊 composite.py 一致:四個檔)。

    給 freeze-eval / samples 用 —— 訓練不落地,只有需要『凍結/可比/配圖』時才寫。
    """
    import json
    from pathlib import Path

    d = Path(scene_dir)
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sample.rgb).save(d / "rgb.png")
    Image.fromarray(sample.semantic).save(d / "semantic_mask.png")
    Image.fromarray(sample.instance).save(d / "instance_mask.png")
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(sample.meta, f, indent=2, ensure_ascii=False)


def freeze_scene_set(
    out_root: str | os.PathLike,
    n: int,
    assets: Assets,
    config: GenConfig = DEFAULT_CONFIG,
    index_offset: int = 0,
) -> list[int]:
    """生成並落地 n 張場景到 out_root/{00000..}/。回傳寫出的 scene id 清單。

    用 index_offset 把 eval / samples 推到保留高位區段(見 config 的 *_INDEX_OFFSET),
    確保與訓練場景互斥。落地後的目錄即 EvalSetDataset 可直接讀的格式。
    """
    from pathlib import Path

    out_root = Path(out_root)
    ids = []
    for k in range(n):
        gi = index_offset + k
        sample = generate_scene(gi, assets, config)
        write_scene(sample, out_root / f"{k:05d}")
        ids.append(k)
    return ids


if __name__ == "__main__":
    # 自檢:確定性(同 i 兩次必相同)+ 不重複(不同 i 必不同)。
    from src.data.assets import load_assets

    assets = load_assets()
    a1 = generate_scene(0, assets)
    a2 = generate_scene(0, assets)
    b = generate_scene(1, assets)
    same = np.array_equal(a1.rgb, a2.rgb) and a1.meta == a2.meta
    diff = not np.array_equal(a1.rgb, b.rgb)
    print(f"determinism (scene 0 == scene 0): {same}")
    print(f"non-repetition (scene 0 != scene 1): {diff}")
    print(f"scene 0: {a1.meta['n_parts']} parts, hdri={a1.meta['hdri_idx']}, "
          f"bg={a1.meta['bg_kind']}, distractors={a1.meta['n_distractors']}")
