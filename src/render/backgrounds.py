"""
backgrounds.py — 程序化工業風背景(不下載外部素材)→ assets/backgrounds/

移植 gen_backgrounds.py。**純 PIL/numpy/cv2,不依賴 bpy**(可在 conda 直接跑/測)。
4 種風格 × N 張,512×512,讓 composite 隨機 crop。seed 由 (name, i) 決定 → 可復現。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.render.config import REPO_ROOT

DEFAULT_BG_DIR = REPO_ROOT / "assets" / "backgrounds"
SIZE = 512
N_PER_TYPE = 3


def _smooth_noise(h, w, scale, rng):
    # cv2 延遲 import:沒裝 opencv 也能 import 本模組(只有真的生成背景才需要)
    import cv2
    n = rng.rand(h, w).astype(np.float32)
    k = max(3, int(scale) | 1)  # odd
    return cv2.GaussianBlur(n, (k, k), 0)


def gen_concrete(seed):
    rng = np.random.RandomState(seed)
    base_color = np.clip(0.5 + 0.1 * rng.randn(), 0.35, 0.65)
    noise_fine = _smooth_noise(SIZE, SIZE, 3, rng)
    noise_med = _smooth_noise(SIZE, SIZE, 21, rng)
    intensity = np.clip(base_color + 0.15 * (noise_med - 0.5) + 0.08 * (noise_fine - 0.5), 0, 1)
    tint = np.array([1.0, 0.98 + 0.04 * rng.randn(), 0.96 + 0.05 * rng.randn()])
    return np.clip(np.stack([intensity * tint[c] for c in range(3)], axis=-1), 0, 1)


def gen_metal_brushed(seed):
    rng = np.random.RandomState(seed)
    base = np.clip(0.55 + 0.1 * rng.randn(), 0.45, 0.7)
    stripes = np.broadcast_to(_smooth_noise(1, SIZE, 5, rng).squeeze(), (SIZE, SIZE))
    h_noise = _smooth_noise(SIZE, SIZE, 3, rng)
    intensity = np.clip(base + 0.18 * (stripes - 0.5) + 0.05 * (h_noise - 0.5), 0, 1)
    return np.clip(np.stack([intensity, intensity * 1.0, intensity * 1.02], axis=-1), 0, 1)


def gen_rubber_mat(seed):
    rng = np.random.RandomState(seed)
    base = np.clip(0.18 + 0.05 * rng.randn(), 0.12, 0.28)
    cellular = _smooth_noise(SIZE, SIZE, 11, rng)
    fine = _smooth_noise(SIZE, SIZE, 3, rng)
    intensity = np.clip(base + 0.06 * (cellular - 0.5) + 0.03 * (fine - 0.5), 0, 1)
    return np.clip(np.stack([intensity, intensity * 0.95, intensity * 0.92], axis=-1), 0, 1)


def gen_wood(seed):
    rng = np.random.RandomState(seed)
    grain = _smooth_noise(SIZE, SIZE, 35, rng)
    grain_fine = _smooth_noise(SIZE, SIZE, 5, rng)
    horiz = np.tile(_smooth_noise(SIZE, 1, 51, rng), (1, SIZE))
    intensity = np.clip(0.5 + 0.18 * (grain - 0.5) + 0.06 * (grain_fine - 0.5) + 0.08 * (horiz - 0.5), 0.2, 0.9)
    rgb = np.stack([
        intensity * (0.7 + 0.15 * rng.rand()),
        intensity * (0.5 + 0.1 * rng.rand()),
        intensity * (0.35 + 0.08 * rng.rand()),
    ], axis=-1)
    return np.clip(rgb, 0, 1)


_GENERATORS = (("concrete", gen_concrete), ("metal", gen_metal_brushed),
               ("rubber", gen_rubber_mat), ("wood", gen_wood))


def gen_backgrounds(out_dir: str | Path = DEFAULT_BG_DIR, n_per_type: int = N_PER_TYPE) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for name, gen in _GENERATORS:
        for i in range(n_per_type):
            seed = hash((name, i)) & 0xFFFFFFFF
            arr = (gen(seed) * 255).astype(np.uint8)
            Image.fromarray(arr).save(out_dir / f"bg_{idx:02d}_{name}_{i}.png")
            idx += 1
    print(f"Generated {idx} backgrounds → {out_dir}")
    return idx


if __name__ == "__main__":
    gen_backgrounds()
