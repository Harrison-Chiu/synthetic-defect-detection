"""
Stage 3 — 黑底對照組 test set 生成

從現有 output/scenes/ 抽 test split（100 張，跟 train_stage3.py seed=42 的 split 一致），
把每張的背景區（instance_mask==0）整片塗黑，產生 output/scenes_black/ 對應 100 張。

目的：跑模型對「textured bg test」vs「黑底 test」的 defect IoU 對比 → ablation 證明背景策略影響。

執行：
    & "C:\\Users\\Harrison\\miniconda3\\envs\\dl_final\\python.exe" scripts/gen_black_bg_test.py
"""

import os
import json
import glob
import shutil
import numpy as np
from PIL import Image

BASE_DIR    = r"D:\Harrison\中山\大四下\深度學習期末報告"
SCENES_DIR  = os.path.join(BASE_DIR, "output", "scenes")
OUT_DIR     = os.path.join(BASE_DIR, "output", "scenes_black")
SEED        = 42

# 跟 train_stage3.py 一致：80/10/10 split, seed=42
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1


def get_test_ids():
    all_ids  = sorted(int(os.path.basename(d))
                      for d in glob.glob(os.path.join(SCENES_DIR, "[0-9]*")))
    rng      = np.random.RandomState(SEED)
    shuffled = list(all_ids); rng.shuffle(shuffled)
    n_train = int(TRAIN_FRAC * len(shuffled))
    n_val   = int(VAL_FRAC * len(shuffled))
    return shuffled[n_train + n_val:]


def blackify_scene(sid):
    src_dir = os.path.join(SCENES_DIR, f"{sid:05d}")
    dst_dir = os.path.join(OUT_DIR,    f"{sid:05d}")
    os.makedirs(dst_dir, exist_ok=True)

    rgb  = np.array(Image.open(os.path.join(src_dir, "rgb.png")).convert("RGB"))
    inst = np.array(Image.open(os.path.join(src_dir, "instance_mask.png")))
    # 背景區（instance_mask == 0）整片塗黑
    bg_mask = (inst == 0)
    rgb_black = rgb.copy()
    rgb_black[bg_mask] = 0

    Image.fromarray(rgb_black).save(os.path.join(dst_dir, "rgb.png"))
    # semantic / instance mask 直接複製（GT 不變）
    for fname in ("semantic_mask.png", "instance_mask.png", "meta.json"):
        shutil.copy(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))


def main():
    test_ids = get_test_ids()
    print(f"Test split (seed={SEED}, 80/10/10): {len(test_ids)} scenes")
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, sid in enumerate(test_ids, 1):
        blackify_scene(sid)
        if i % 20 == 0:
            print(f"  [{i}/{len(test_ids)}] {sid:05d}")
    print(f"Done. {len(test_ids)} black-bg test scenes → {OUT_DIR}")


if __name__ == "__main__":
    main()
