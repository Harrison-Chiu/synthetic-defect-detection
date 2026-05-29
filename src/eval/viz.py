"""
viz.py — 預測快照(報告 / 訓練監看用)

移植 train_stage4.py 的 mask_to_color + save_epoch_snapshot,改成 step-based 命名
與 dict 輸出。matplotlib 只在這裡 import,核心訓練迴圈不依賴它。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.eval.metrics import infer_3class

# 0=bg(黑) / 1=normal(綠) / 2=defect(紅)
_CMAP = np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]], dtype=np.uint8)


def mask_to_color(m: np.ndarray) -> np.ndarray:
    return _CMAP[m]


@torch.no_grad()
def save_prediction_snapshot(model, rgb_batch, step: int, out_dir: str | Path, device) -> str:
    """對固定一批 rgb 做 3-class 預測並存成一張對照圖。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    outputs = model(rgb_batch.to(device))
    pred, _, _ = infer_3class(outputs)
    pred = pred.cpu().numpy()

    n = pred.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(mask_to_color(pred[i]))
        axes[i].axis("off")
        axes[i].set_title(f"sample {i}", fontsize=8)
    fig.suptitle(f"step {step} predictions", fontsize=10)
    plt.tight_layout()
    out = out_dir / f"step_{step:06d}.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close()
    return str(out)
