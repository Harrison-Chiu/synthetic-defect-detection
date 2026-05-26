"""
產出工業風背景紋理（程序化生成，不下載外部素材）。

執行：
    conda activate dl_final
    python scripts/gen_backgrounds.py

輸出：assets/backgrounds/bg_XX.png（512×512 PNG），讓 composite 隨機 crop 256×256
"""

import os
import numpy as np
from PIL import Image
import cv2

OUT_DIR = r"D:\Harrison\中山\大四下\深度學習期末報告\assets\backgrounds"
SIZE    = 512
N_PER_TYPE = 3   # 每種風格 3 張，共 12 張


def _smooth_noise(h, w, scale, rng):
    """White noise + Gaussian blur → 雲狀紋理"""
    n = rng.rand(h, w).astype(np.float32)
    k = max(3, int(scale) | 1)  # ensure odd
    return cv2.GaussianBlur(n, (k, k), 0)


def gen_concrete(seed):
    """灰泥/混凝土：低飽和、中亮度、細顆粒"""
    rng = np.random.RandomState(seed)
    base = 0.5 + 0.1 * rng.randn()
    base_color = np.clip(base, 0.35, 0.65)
    noise_fine = _smooth_noise(SIZE, SIZE, 3, rng)
    noise_med  = _smooth_noise(SIZE, SIZE, 21, rng)
    intensity = base_color + 0.15 * (noise_med - 0.5) + 0.08 * (noise_fine - 0.5)
    intensity = np.clip(intensity, 0, 1)
    # 略帶暖色或冷色偏
    tint = np.array([1.0, 0.98 + 0.04 * rng.randn(), 0.96 + 0.05 * rng.randn()])
    rgb = np.stack([intensity * tint[c] for c in range(3)], axis=-1)
    return np.clip(rgb, 0, 1)


def gen_metal_brushed(seed):
    """金屬刷面：水平條紋 + 細顆粒、銀灰調"""
    rng = np.random.RandomState(seed)
    base = 0.55 + 0.1 * rng.randn()
    base = np.clip(base, 0.45, 0.7)
    # 水平條紋
    stripes_1d = _smooth_noise(1, SIZE, 5, rng).squeeze()
    stripes = np.broadcast_to(stripes_1d, (SIZE, SIZE))
    h_noise = _smooth_noise(SIZE, SIZE, 3, rng)
    intensity = base + 0.18 * (stripes - 0.5) + 0.05 * (h_noise - 0.5)
    intensity = np.clip(intensity, 0, 1)
    # 銀灰色調
    rgb = np.stack([intensity, intensity * 1.0, intensity * 1.02], axis=-1)
    return np.clip(rgb, 0, 1)


def gen_rubber_mat(seed):
    """橡膠墊：暗色、低變化、可能有蜂巢圖案"""
    rng = np.random.RandomState(seed)
    base = 0.18 + 0.05 * rng.randn()
    base = np.clip(base, 0.12, 0.28)
    cellular = _smooth_noise(SIZE, SIZE, 11, rng)
    fine = _smooth_noise(SIZE, SIZE, 3, rng)
    intensity = base + 0.06 * (cellular - 0.5) + 0.03 * (fine - 0.5)
    intensity = np.clip(intensity, 0, 1)
    rgb = np.stack([intensity, intensity * 0.95, intensity * 0.92], axis=-1)
    return np.clip(rgb, 0, 1)


def gen_wood(seed):
    """木質工作台：暖棕色 + 木紋條紋"""
    rng = np.random.RandomState(seed)
    base_r = 0.45 + 0.08 * rng.randn()
    # 木紋是長條紋（垂直方向）+ 不規則
    grain = _smooth_noise(SIZE, SIZE, 35, rng)
    grain_fine = _smooth_noise(SIZE, SIZE, 5, rng)
    # 加上水平方向的低頻 variation 模擬木紋走向
    horiz = np.tile(_smooth_noise(SIZE, 1, 51, rng), (1, SIZE))
    intensity = 0.5 + 0.18 * (grain - 0.5) + 0.06 * (grain_fine - 0.5) + 0.08 * (horiz - 0.5)
    intensity = np.clip(intensity, 0.2, 0.9)
    # 暖棕色
    rgb = np.stack([
        intensity * (0.7 + 0.15 * rng.rand()),
        intensity * (0.5 + 0.1 * rng.rand()),
        intensity * (0.35 + 0.08 * rng.rand()),
    ], axis=-1)
    return np.clip(rgb, 0, 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    generators = [
        ("concrete",     gen_concrete),
        ("metal",        gen_metal_brushed),
        ("rubber",       gen_rubber_mat),
        ("wood",         gen_wood),
    ]
    idx = 0
    for name, gen in generators:
        for i in range(N_PER_TYPE):
            seed = hash((name, i)) & 0xFFFFFFFF
            rgb = gen(seed)
            arr = (rgb * 255).astype(np.uint8)
            fname = f"bg_{idx:02d}_{name}_{i}.png"
            Image.fromarray(arr).save(os.path.join(OUT_DIR, fname))
            idx += 1
    print(f"Generated {idx} backgrounds → {OUT_DIR}")


if __name__ == "__main__":
    main()
