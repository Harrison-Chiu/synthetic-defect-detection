"""diag_diff_threshold.py — 一次性:看 Route A 的 normal vs defect 同 pose 相減 + 不同閾值切出的變形區。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "output" / "patches"
POSE = "pan_head_az000_el-030_h0"           # 任挑一個 pose
STATES = ["bend_light", "bend_heavy", "displace_heavy", "remesh_heavy"]
THRS = [15, 30, 50]                          # 0-255 的 RGB 平均絕對差閾值

def load_rgb_alpha(p):
    a = np.array(Image.open(p).convert("RGBA"))
    return a[..., :3].astype(np.float32), a[..., 3] > 10

def main():
    normal_rgb, normal_a = load_rgb_alpha(PATCH / "normal" / f"{POSE}_normal.png")
    ncol = 3 + len(THRS)
    fig, axes = plt.subplots(len(STATES), ncol, figsize=(2.0 * ncol, 2.2 * len(STATES)))
    for r, st in enumerate(STATES):
        drgb, da = load_rgb_alpha(PATCH / st / f"{POSE}_{st}.png")
        part = normal_a | da                       # 兩版任一有零件處
        diff = np.abs(drgb - normal_rgb).mean(axis=2)
        diff_in = np.where(part, diff, 0)
        axes[r, 0].imshow(normal_rgb.astype(np.uint8)); axes[r, 0].set_ylabel(st, fontsize=8)
        axes[r, 1].imshow(drgb.astype(np.uint8))
        axes[r, 2].imshow(diff_in, cmap="inferno")
        if r == 0:
            axes[r, 0].set_title("normal", fontsize=8); axes[r, 1].set_title("defect", fontsize=8)
            axes[r, 2].set_title("|diff|", fontsize=8)
        for c, thr in enumerate(THRS):
            mask = (diff_in > thr) & part
            frac = mask.sum() / max(1, part.sum())
            axes[r, 3 + c].imshow(mask, cmap="gray")
            axes[r, 3 + c].set_title(f"thr={thr}  {frac*100:.0f}%區", fontsize=7)
            print(f"  {st:14s} thr={thr:3d}: 變形區佔零件 {frac*100:5.1f}%  ({int(mask.sum())} px)")
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    plt.suptitle(f"Route A: {POSE}  normal vs defect 相減 + 閾值", fontsize=10)
    plt.tight_layout()
    out = REPO / "docs/figures/route_a_diff_threshold.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
