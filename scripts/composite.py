"""
Stage 2 — Scene composite

從 output/parts_stage2/ 取單零件 PNG（含 alpha），隨機 rotate/scale 後貼到背景上組合場景。
合成過程同時產生 ground truth：semantic mask（背景/正常/瑕疵）+ instance mask + meta.json。
允許零件重疊；instance ID 由貼上順序 assign。

執行：
    conda activate dl_final
    python scripts/composite.py
"""

import os
import json
import glob
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
BASE_DIR    = r"D:\Harrison\中山\大四下\深度學習期末報告"
PARTS_DIR   = os.path.join(BASE_DIR, "output", "parts_stage2")
BG_DIR      = os.path.join(BASE_DIR, "assets",  "backgrounds")
SCENES_DIR  = os.path.join(BASE_DIR, "output",  "scenes")

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

DEFECT_STATES_DEFECTIVE = {"bend_light", "bend_heavy", "displace_light", "displace_heavy"}

# ─────────────────────────────────────────────
# 載入素材
# ─────────────────────────────────────────────

def load_part_index():
    """掃 parts_stage2/，分成 normal / defective 兩個 pool。回傳 (normal_list, defect_list)"""
    normal_pool, defect_pool = [], []
    for state_dir in sorted(os.listdir(PARTS_DIR)):
        full = os.path.join(PARTS_DIR, state_dir)
        if not os.path.isdir(full):
            continue
        target = defect_pool if state_dir in DEFECT_STATES_DEFECTIVE else normal_pool
        for png in sorted(glob.glob(os.path.join(full, "*.png"))):
            target.append((png, state_dir))
    return normal_pool, defect_pool


def sample_parts(n, normal_pool, defect_pool, rng):
    """每個 instance 以 DEFECT_PROB 機率從 defect_pool 抽，否則從 normal_pool 抽"""
    chosen = []
    for _ in range(n):
        pool = defect_pool if rng.random() < DEFECT_PROB else normal_pool
        chosen.append(pool[rng.randint(len(pool))])
    return chosen


def load_backgrounds():
    return sorted(glob.glob(os.path.join(BG_DIR, "*.png")))


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


def composite_scene(scene_idx, normal_pool, defect_pool, bgs, rng):
    """產生一張 scene"""
    # 1) 選背景並 crop 256
    bg_path = bgs[rng.randint(len(bgs))]
    bg = np.array(Image.open(bg_path).convert("RGB"))
    bh, bw = bg.shape[:2]
    x0 = rng.randint(0, max(1, bw - OUT_SIZE))
    y0 = rng.randint(0, max(1, bh - OUT_SIZE))
    canvas = bg[y0:y0 + OUT_SIZE, x0:x0 + OUT_SIZE].copy()

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
        "scene_id":  scene_idx,
        "bg_source": os.path.basename(bg_path),
        "bg_crop":   [int(x0), int(y0)],
        "instances": placed_meta,
        "n_parts":   len(placed_meta),
        "n_defective": sum(1 for m in placed_meta if m["is_defective"]),
    }
    with open(os.path.join(scene_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    rng = np.random.RandomState(SEED)
    normal_pool, defect_pool = load_part_index()
    bgs   = load_backgrounds()
    print(f"Loaded {len(normal_pool)} normal + {len(defect_pool)} defect parts, "
          f"{len(bgs)} backgrounds (DEFECT_PROB={DEFECT_PROB})")
    os.makedirs(SCENES_DIR, exist_ok=True)

    for i in range(N_SCENES):
        composite_scene(i, normal_pool, defect_pool, bgs, rng)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{N_SCENES}] scenes done")

    # 統計實際分布
    n_total, n_def, n_def_scenes = 0, 0, 0
    for sid in range(N_SCENES):
        mp = os.path.join(SCENES_DIR, f"{sid:05d}", "meta.json")
        m = json.load(open(mp, encoding="utf-8"))
        n_total += m["n_parts"]; n_def += m["n_defective"]
        if m["n_defective"] > 0:
            n_def_scenes += 1
    print(f"\nDone. {N_SCENES} scenes → {SCENES_DIR}")
    print(f"Actual ratio: {n_def}/{n_total} instances defective ({n_def/n_total*100:.1f}%); "
          f"{n_def_scenes}/{N_SCENES} scenes contain ≥1 defect ({n_def_scenes/N_SCENES*100:.1f}%)")


if __name__ == "__main__":
    main()
