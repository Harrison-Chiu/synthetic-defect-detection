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


def _denorm_rgb(rgb_t):
    """(3,H,W) [-1,1] → (H,W,3) uint8。"""
    import numpy as np
    a = (rgb_t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (a * 255).astype(np.uint8)


@torch.no_grad()
def save_prediction_snapshot(model, snapshot_batch, step: int, out_dir: str | Path, device) -> str:
    """固定一批做預測,每列 RGB | GT | Pred | P(defect),存成對照圖。

    snapshot_batch: (rgb_tensor (N,3,H,W), gt_tensor (N,H,W) 3-class) —— 有 GT 才看得出對錯。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb_batch, gt_batch = snapshot_batch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    outputs = model(rgb_batch.to(device))
    pred, _, defect_prob = infer_3class(outputs)
    pred = pred.cpu().numpy()
    defect_prob = defect_prob.cpu().numpy()
    gt = gt_batch.cpu().numpy()

    n = pred.shape[0]
    cols = ["RGB", "GT", "Pred", "P(defect)"]
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        axes[i, 0].imshow(_denorm_rgb(rgb_batch[i]))
        axes[i, 1].imshow(mask_to_color(gt[i]))
        axes[i, 2].imshow(mask_to_color(pred[i]))
        axes[i, 3].imshow(defect_prob[i], cmap="hot", vmin=0, vmax=1)
        for j, name in enumerate(cols):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(name, fontsize=9)
    fig.suptitle(f"step {step}  (GT/Pred: 黑=bg 綠=normal 紅=defect)", fontsize=10)
    plt.tight_layout()
    out = out_dir / f"step_{step:06d}.png"
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close()
    return str(out)
