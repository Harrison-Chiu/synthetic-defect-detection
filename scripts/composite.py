"""
Stage 2/3/4 — Scene composite

從 output/parts_stage2/ 取單零件 PNG（含 alpha），隨機 rotate/scale 後貼到背景上組合場景。
合成過程同時產生 ground truth：semantic mask（背景/正常/瑕疵）+ instance mask + meta.json。
允許零件重疊；instance ID 由貼上順序 assign。

Stage 3 升級：
  - 三層 base layer 抽樣（real AmbientCG / procedural / solid+noise）
  - 程序化 distractors（幾何圖案，非均勻數量分布）— 模擬工廠雜物，避免 stationary-texture shortcut
  - 詳見 docs/stage3_plan.md「背景配方」段

Stage 4 升級：
  - 每 scene 開頭 sample 一個 HDRI index，所有 instance 只從該 HDRI 渲染的 part subset 抽
    （修正 Stage 3 物理不一致 bug：同 scene 不同 instance 來自不同 HDRI 反射）
  - parts 檔名格式 `pan_head_az###_el±###_h#_<state>.png`，解析 h# 取 HDRI index

執行：
    conda activate dl_final
    python scripts/composite.py
"""

import os
import re
import math
import json
import glob
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw

# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
BASE_DIR        = r"D:\Harrison\中山\大四下\深度學習期末報告"
PARTS_DIR       = os.path.join(BASE_DIR, "output", "parts_stage2")
BG_DIR_REAL     = os.path.join(BASE_DIR, "assets", "backgrounds_real")        # AmbientCG (5)
BG_DIR_PROC     = os.path.join(BASE_DIR, "assets", "backgrounds_procedural")  # 12 procedural
SCENES_DIR      = os.path.join(BASE_DIR, "output", "scenes")

OUT_SIZE     = 256
N_SCENES     = 1000
PARTS_RANGE  = (3, 7)          # 每場景零件數區間
SCALE_RANGE  = (0.5, 1.0)      # 縮放比例
ROT_RANGE    = (0, 360)        # in-plane rotation
SEED         = 42

# 每個 instance 是 defect 的機率（真實 QC 通常 < 10%；我們選 20% 以保留足夠訓練樣本）
# 影響：1000 場景 × 平均 5 instance ≈ 5000，其中 ~1000 defective、~4000 normal
DEFECT_PROB  = 0.20

# Semantic class 編碼
CLS_BG     = 0
CLS_NORMAL = 1
CLS_DEFECT = 2

DEFECT_STATES_DEFECTIVE = {
    "bend_light", "bend_heavy",
    "displace_light", "displace_heavy",
    "remesh_light", "remesh_heavy",   # Stage 3 新增
}

# ───────── Stage 3 背景配方（recipe）─────────
# Base layer 抽樣（互斥，per scene）：
BASE_LAYER_PROBS = {
    "real":       0.50,   # AmbientCG (5 張)
    "procedural": 0.25,   # Stage 2 殘留 procedural smooth-noise (12 張)
    "solid":      0.25,   # 純色 + 輕微 noise
}
SOLID_COLOR_PALETTE = [
    (60, 60, 60), (90, 90, 90), (120, 120, 120), (160, 160, 160),  # neutral grey
    (110, 90, 70), (140, 110, 80), (80, 70, 60),                    # warm industrial
    (100, 90, 100), (90, 100, 110),                                 # cool industrial
]
SOLID_NOISE_STD = 6.0   # 0-255 範圍 Gaussian noise

# Distractor 數量分布（非均勻、per scene）：
# 每 bucket: ((count_lo, count_hi_inclusive), prob)
DISTRACTOR_COUNT_BUCKETS = [
    ((0,  0),  0.25),    # 25%：clean，無 distractor
    ((1,  4),  0.35),    # 35%：light
    ((5, 10),  0.25),    # 25%：medium
    ((11, 20), 0.15),    # 15%：heavy
]
# Distractor 形狀類型機率：
DISTRACTOR_SHAPE_PROBS = [
    ("ellipse",   0.40),
    ("rectangle", 0.25),
    ("polygon",   0.15),
    ("line",      0.15),
    ("ring",      0.05),
]
DISTRACTOR_SIZE_RANGE  = (5, 80)     # px
DISTRACTOR_ALPHA_RANGE = (0.50, 1.0) # 0.5=半透 ~ 1.0=實心

# Distractor 顏色：金屬灰調 vs 全隨機
DISTRACTOR_COLOR_METALLIC_PROB = 0.4
DISTRACTOR_METALLIC_PALETTE = [
    (70, 70, 75), (90, 90, 95), (110, 105, 100), (45, 45, 50),
    (130, 125, 120), (60, 65, 70),
]

# Bg augmentation 範圍（在 base layer 上）：
BG_AUG_BRIGHTNESS = (0.75, 1.25)   # ±25%
BG_AUG_CONTRAST   = (0.80, 1.20)   # ±20%
BG_AUG_SATURATION = (0.70, 1.30)   # ±30%
BG_AUG_FLIP_H_PROB = 0.5
BG_AUG_FLIP_V_PROB = 0.5

# ─────────────────────────────────────────────
# 載入素材
# ─────────────────────────────────────────────

_HDRI_RE = re.compile(r"_h(\d+)_")

def _hdri_idx_from_filename(path):
    m = _HDRI_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"Cannot parse hdri index from filename: {path}")
    return int(m.group(1))


def load_part_index_by_hdri():
    """掃 parts_stage2/，按 hdri_idx 分桶。
    回傳 dict: {hdri_idx: {"normal": [...], "defect": [...]}}
    每 pool 為 list of (path, defect_state)。
    """
    buckets = {}
    for state_dir in sorted(os.listdir(PARTS_DIR)):
        full = os.path.join(PARTS_DIR, state_dir)
        if not os.path.isdir(full):
            continue
        key = "defect" if state_dir in DEFECT_STATES_DEFECTIVE else "normal"
        for png in sorted(glob.glob(os.path.join(full, "*.png"))):
            h = _hdri_idx_from_filename(png)
            if h not in buckets:
                buckets[h] = {"normal": [], "defect": []}
            buckets[h][key].append((png, state_dir))
    return buckets


def sample_parts(n, normal_pool, defect_pool, rng):
    """每個 instance 以 DEFECT_PROB 機率從 defect_pool 抽，否則從 normal_pool 抽"""
    chosen = []
    for _ in range(n):
        pool = defect_pool if rng.random() < DEFECT_PROB else normal_pool
        chosen.append(pool[rng.randint(len(pool))])
    return chosen


def load_background_pools():
    """回傳 {'real': [paths], 'procedural': [paths]}"""
    pools = {"real": [], "procedural": []}
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        pools["real"].extend(glob.glob(os.path.join(BG_DIR_REAL, ext)))
        pools["procedural"].extend(glob.glob(os.path.join(BG_DIR_PROC, ext)))
    pools["real"]       = sorted(set(pools["real"]))
    pools["procedural"] = sorted(set(pools["procedural"]))
    return pools


def _categorical(probs_dict_or_list, rng):
    """從 {key: prob} 或 [(key, prob), ...] 抽一個 key（cumulative thresholding）"""
    if isinstance(probs_dict_or_list, dict):
        items = list(probs_dict_or_list.items())
    else:
        items = probs_dict_or_list
    r = rng.random(); cum = 0.0
    for k, p in items:
        cum += p
        if r < cum:
            return k
    return items[-1][0]


def generate_base_layer(pools, rng):
    """
    抽 base layer 來源並產生足夠大（≥ OUT_SIZE）的 PIL Image。
    回傳 (PIL_Image_RGB, source_kind: str)。
    """
    kind = _categorical(BASE_LAYER_PROBS, rng)
    if kind == "real":
        path = pools["real"][rng.randint(len(pools["real"]))]
        img = Image.open(path).convert("RGB")
        source = os.path.basename(path)
    elif kind == "procedural":
        path = pools["procedural"][rng.randint(len(pools["procedural"]))]
        img = Image.open(path).convert("RGB")
        source = os.path.basename(path)
    else:  # solid
        color = SOLID_COLOR_PALETTE[rng.randint(len(SOLID_COLOR_PALETTE))]
        arr = np.full((OUT_SIZE * 2, OUT_SIZE * 2, 3), color, dtype=np.float32)
        noise = rng.normal(0, SOLID_NOISE_STD, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        source = f"solid_rgb{color}"
    # 確保 ≥ OUT_SIZE
    if img.width < OUT_SIZE or img.height < OUT_SIZE:
        scale = OUT_SIZE / min(img.width, img.height) * 1.1
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    return img, kind, source


def augment_background(bg_pil, rng):
    """亮度/對比/彩度 + H/V flip。隨機數從 numpy RandomState 控制，可復現。"""
    img = bg_pil
    if rng.random() < BG_AUG_FLIP_H_PROB:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < BG_AUG_FLIP_V_PROB:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(*BG_AUG_BRIGHTNESS))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(*BG_AUG_CONTRAST))
    img = ImageEnhance.Color(img).enhance(rng.uniform(*BG_AUG_SATURATION))
    return img


def sample_distractor_count(rng):
    """非均勻抽 distractor 數量"""
    r = rng.random(); cum = 0.0
    for (lo, hi), p in DISTRACTOR_COUNT_BUCKETS:
        cum += p
        if r < cum:
            return rng.randint(lo, hi + 1) if hi > lo else lo
    lo, hi = DISTRACTOR_COUNT_BUCKETS[-1][0]
    return rng.randint(lo, hi + 1) if hi > lo else lo


def _random_color(rng):
    if rng.random() < DISTRACTOR_COLOR_METALLIC_PROB:
        return DISTRACTOR_METALLIC_PALETTE[rng.randint(len(DISTRACTOR_METALLIC_PALETTE))]
    return (int(rng.randint(0, 256)), int(rng.randint(0, 256)), int(rng.randint(0, 256)))


def draw_distractors(canvas, rng):
    """
    在 canvas (np.uint8 H×W×3) 上疊 N 個幾何 distractors。原地修改 canvas。
    回傳 (n_distractors, list[shape_type])。
    """
    n = sample_distractor_count(rng)
    if n == 0:
        return 0, []
    H, W = canvas.shape[:2]
    pil = Image.fromarray(canvas).convert("RGBA")
    shape_log = []
    for _ in range(n):
        shape = _categorical(DISTRACTOR_SHAPE_PROBS, rng)
        shape_log.append(shape)
        color = _random_color(rng)
        alpha = rng.uniform(*DISTRACTOR_ALPHA_RANGE)
        rgba_color = (color[0], color[1], color[2], int(alpha * 255))
        size = rng.randint(DISTRACTOR_SIZE_RANGE[0], DISTRACTOR_SIZE_RANGE[1] + 1)
        cx = rng.randint(0, W); cy = rng.randint(0, H)
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
            for i in range(n_sides):
                ang = 2 * math.pi * i / n_sides + rng.uniform(-0.3, 0.3)
                r_ = size / 2 * rng.uniform(0.6, 1.0)
                pts.append((cx + r_ * math.cos(ang), cy + r_ * math.sin(ang)))
            d.polygon(pts, fill=rgba_color)
        pil = Image.alpha_composite(pil, overlay)
    canvas[:] = np.array(pil.convert("RGB"))
    return n, shape_log


# ─────────────────────────────────────────────
# 變換 + 合成
# ─────────────────────────────────────────────

def transform_part(rgba, rng):
    """Rotation + scale，回傳 (rgba_new, scale, rotation_deg)"""
    rot = rng.uniform(*ROT_RANGE)
    scale = rng.uniform(*SCALE_RANGE)

    img = Image.fromarray(rgba, mode="RGBA")
    # rotate 用 expand 保留整個零件
    img = img.rotate(rot, resample=Image.BILINEAR, expand=True)
    # scale
    nw, nh = img.size
    img = img.resize((max(1, int(nw * scale)), max(1, int(nh * scale))), Image.BILINEAR)
    return np.array(img), scale, rot


def alpha_bbox(alpha, threshold=10):
    """回傳 (x0, y0, x1, y1) 緊密包圍 alpha > threshold 的區域"""
    ys, xs = np.where(alpha > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def composite_scene(scene_idx, part_buckets, bg_pools, rng):
    """產生一張 scene。Stage 4：先抽 HDRI subset，instance 只從該 subset 抽（同 scene 反射物理一致）"""
    # 0) Stage 4：先抽 HDRI subset
    hdri_keys = sorted(part_buckets.keys())
    hdri_idx = hdri_keys[rng.randint(len(hdri_keys))]
    normal_pool = part_buckets[hdri_idx]["normal"]
    defect_pool = part_buckets[hdri_idx]["defect"]

    # 1) 抽 base layer
    bg_pil, base_kind, base_source = generate_base_layer(bg_pools, rng)
    bg_pil = augment_background(bg_pil, rng)
    bg = np.array(bg_pil)
    bh, bw = bg.shape[:2]
    x0 = rng.randint(0, max(1, bw - OUT_SIZE))
    y0 = rng.randint(0, max(1, bh - OUT_SIZE))
    canvas = bg[y0:y0 + OUT_SIZE, x0:x0 + OUT_SIZE].copy()
    # 2) 疊 distractors（在貼零件之前；shape 不會落到 semantic mask 上）
    n_distractors, distractor_shapes = draw_distractors(canvas, rng)

    semantic = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.uint8)
    instance = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.uint8)

    # 2) 隨機選 N 個零件（per-instance DEFECT_PROB 抽樣）
    n_parts = rng.randint(PARTS_RANGE[0], PARTS_RANGE[1] + 1)
    chosen  = sample_parts(n_parts, normal_pool, defect_pool, rng)

    placed_meta = []

    for inst_id, (part_path, defect_state) in enumerate(chosen, start=1):
        rgba = np.array(Image.open(part_path).convert("RGBA"))
        rgba, scale, rot = transform_part(rgba, rng)
        h, w = rgba.shape[:2]

        # tight bbox of part within its image (alpha-driven)
        bb = alpha_bbox(rgba[:, :, 3])
        if bb is None:
            continue
        bx0, by0, bx1, by1 = bb
        part_w = bx1 - bx0
        part_h = by1 - by0
        if part_w < 4 or part_h < 4:
            continue

        # 把零件 tight crop 出來
        part_crop = rgba[by0:by1, bx0:bx1]

        # 隨機位置（讓 tight bbox 大部分落在畫面內，允許部分超出）
        # 用「中心點」隨機在 0..OUT_SIZE 內
        cx = rng.randint(0, OUT_SIZE)
        cy = rng.randint(0, OUT_SIZE)
        px = cx - part_w // 2
        py = cy - part_h // 2

        # 計算落點交集
        x_a = max(0, px); y_a = max(0, py)
        x_b = min(OUT_SIZE, px + part_w); y_b = min(OUT_SIZE, py + part_h)
        if x_a >= x_b or y_a >= y_b:
            continue

        # part_crop 對應切片
        cx0 = x_a - px; cy0 = y_a - py
        cx1 = cx0 + (x_b - x_a); cy1 = cy0 + (y_b - y_a)
        sub_rgba = part_crop[cy0:cy1, cx0:cx1]
        sub_alpha = sub_rgba[:, :, 3].astype(np.float32) / 255.0
        sub_rgb   = sub_rgba[:, :, :3].astype(np.float32)

        # alpha composite
        canvas_region = canvas[y_a:y_b, x_a:x_b].astype(np.float32)
        blended = sub_rgb * sub_alpha[..., None] + canvas_region * (1 - sub_alpha[..., None])
        canvas[y_a:y_b, x_a:x_b] = np.clip(blended, 0, 255).astype(np.uint8)

        # 更新 masks（用 alpha threshold 決定 "屬於零件"）
        is_part = sub_alpha > 0.5
        cls_val = CLS_DEFECT if defect_state in DEFECT_STATES_DEFECTIVE else CLS_NORMAL
        sem_slice = semantic[y_a:y_b, x_a:x_b]
        ins_slice = instance[y_a:y_b, x_a:x_b]
        sem_slice[is_part] = cls_val      # 後貼上的 overwrites 前面（與 RGB 合成一致）
        ins_slice[is_part] = inst_id

        placed_meta.append({
            "instance_id":   inst_id,
            "source_part":   os.path.relpath(part_path, BASE_DIR).replace("\\", "/"),
            "defect_state":  defect_state,
            "is_defective":  defect_state in DEFECT_STATES_DEFECTIVE,
            "bbox":          [x_a, y_a, x_b, y_b],  # 場景座標
            "scale":         round(float(scale), 3),
            "rotation_deg":  round(float(rot), 1),
        })

    # 3) 存檔
    scene_dir = os.path.join(SCENES_DIR, f"{scene_idx:05d}")
    os.makedirs(scene_dir, exist_ok=True)
    Image.fromarray(canvas).save(os.path.join(scene_dir, "rgb.png"))
    Image.fromarray(semantic).save(os.path.join(scene_dir, "semantic_mask.png"))
    Image.fromarray(instance).save(os.path.join(scene_dir, "instance_mask.png"))
    meta = {
        "scene_id":      scene_idx,
        "hdri_idx":      int(hdri_idx),    # Stage 4：scene HDRI 一致性
        "bg_kind":       base_kind,       # real / procedural / solid
        "bg_source":     base_source,
        "bg_crop":       [int(x0), int(y0)],
        "n_distractors": int(n_distractors),
        "distractor_shapes": distractor_shapes,
        "instances":     placed_meta,
        "n_parts":       len(placed_meta),
        "n_defective":   sum(1 for m in placed_meta if m["is_defective"]),
    }
    with open(os.path.join(scene_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    rng = np.random.RandomState(SEED)
    part_buckets = load_part_index_by_hdri()
    bg_pools = load_background_pools()
    for h in sorted(part_buckets.keys()):
        print(f"  HDRI {h}: {len(part_buckets[h]['normal'])} normal + "
              f"{len(part_buckets[h]['defect'])} defect")
    print(f"Background pools: real={len(bg_pools['real'])}, procedural={len(bg_pools['procedural'])}, "
          f"+ solid (procedural in-script)")
    print(f"DEFECT_PROB={DEFECT_PROB}, base layer probs={BASE_LAYER_PROBS}, "
          f"distractor count buckets={DISTRACTOR_COUNT_BUCKETS}")
    os.makedirs(SCENES_DIR, exist_ok=True)

    for i in range(N_SCENES):
        composite_scene(i, part_buckets, bg_pools, rng)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{N_SCENES}] scenes done")

    # 統計實際分布
    n_total, n_def, n_def_scenes = 0, 0, 0
    bg_kind_count = {"real": 0, "procedural": 0, "solid": 0}
    hdri_count = {}
    distractor_total = 0
    for sid in range(N_SCENES):
        mp = os.path.join(SCENES_DIR, f"{sid:05d}", "meta.json")
        m = json.load(open(mp, encoding="utf-8"))
        n_total += m["n_parts"]; n_def += m["n_defective"]
        if m["n_defective"] > 0:
            n_def_scenes += 1
        bg_kind_count[m.get("bg_kind", "real")] += 1
        h = m.get("hdri_idx", -1)
        hdri_count[h] = hdri_count.get(h, 0) + 1
        distractor_total += m.get("n_distractors", 0)
    print(f"\nDone. {N_SCENES} scenes → {SCENES_DIR}")
    print(f"Defect ratio: {n_def}/{n_total} instances defective ({n_def/n_total*100:.1f}%); "
          f"{n_def_scenes}/{N_SCENES} scenes contain >=1 defect ({n_def_scenes/N_SCENES*100:.1f}%)")
    print(f"Base layer distribution: {bg_kind_count}")
    print(f"HDRI distribution: {dict(sorted(hdri_count.items()))}")
    print(f"Avg distractors/scene: {distractor_total/N_SCENES:.2f}")


if __name__ == "__main__":
    main()
