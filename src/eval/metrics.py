"""
metrics.py — S5 評估指標(單次前向,一次算齊)

雙頭(part + defect-sigmoid)→ 合成與 Stage 3 可比的 3-class(bg/normal/defect):
- part 頭 argmax 給 0=bg / 1=part;
- defect 頭 sigmoid > thr 且落在 part 上 → 升級為 2=defect。

GT 來源用 encode_targets 的 "sem3"(整顆壞件=2,與 S3 schema 一致)。

`evaluate_all`:**一次前向**算齊 3-class 指標 + loss + per-state confusion
(取代舊版 evaluate_3class / evaluate_loss / evaluate_per_state 三次掃 loader)。
per-state 改用 **confusion 比例**(該 state 零件像素被判 bg/normal/defect 的比例),
det_rate = 被判 defect 的比例;normal 列的 det_rate 即「好件誤報率」。
"""

from __future__ import annotations

import numpy as np
import torch

from src import schema

_PART_KEY = schema.HEADS_BY_NAME["part"].key
_DEFECT_KEY = schema.HEADS_BY_NAME["defect"].key
_STATE_CLASSES = schema.STATE_CLASSES
_NORMAL_IDX = schema.STATE_TO_IDX["normal"]


def infer_3class(outputs: dict[str, torch.Tensor], defect_thr: float = 0.5):
    """dict logits → (pred 3-class, part_prob, defect_prob)。

    pred: 0=bg / 1=normal_part / 2=defect。part_prob=P(part)、defect_prob=sigmoid。
    """
    part_logits = outputs[_PART_KEY]
    part_pred = part_logits.argmax(dim=1)
    part_prob = torch.softmax(part_logits, dim=1)[:, 1]
    defect_logit = outputs[_DEFECT_KEY]
    if defect_logit.dim() == 4:
        defect_logit = defect_logit.squeeze(1)
    defect_prob = torch.sigmoid(defect_logit)
    out = torch.zeros_like(part_pred)
    is_part = part_pred == 1
    out[is_part] = 1
    out[is_part & (defect_prob > defect_thr)] = 2
    return out, part_prob, defect_prob


@torch.no_grad()
def evaluate_all(model, loader, device, defect_thr: float = 0.5,
                 head_weights=None, pos_weight: float = 8.0, num_classes: int = 3):
    """單次前向算齊:3-class IoU/acc + 平均 loss + per-state confusion。

    回傳 {pixel_acc, mIoU, IoU_per_class, loss:{...}, per_state:{name:{det_rate,conf,n_px}}}。
    """
    from src.train.losses import multihead_loss
    model.eval()
    inter = np.zeros(num_classes)
    union = np.zeros(num_classes)
    correct = total = 0
    n_st = len(_STATE_CLASSES)
    conf = np.zeros((n_st, 3))  # per-state × pred class 像素數
    loss_acc: dict[str, float] = {}
    n_batches = 0

    for rgb, targets in loader:
        rgb = rgb.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        outputs = model(rgb)
        _, comps = multihead_loss(outputs, targets, head_weights, pos_weight)
        for k, v in comps.items():
            loss_acc[k] = loss_acc.get(k, 0.0) + v
        n_batches += 1

        pred, _, _ = infer_3class(outputs, defect_thr)
        gt = targets["sem3"]
        for c in range(num_classes):
            p, t = (pred == c), (gt == c)
            inter[c] += (p & t).sum().item()
            union[c] += (p | t).sum().item()
        correct += (pred == gt).sum().item()
        total += gt.numel()

        state_px = targets["state_px"]
        for c in range(n_st):
            m = state_px == c
            if m.any():
                for cp in range(3):
                    conf[c, cp] += ((pred == cp) & m).sum().item()

    ious = [float(inter[c] / union[c]) if union[c] > 0 else float("nan") for c in range(num_classes)]
    per_state = {}
    for c, name in enumerate(_STATE_CLASSES):
        n = conf[c].sum()
        frac = (conf[c] / n).tolist() if n > 0 else [float("nan")] * 3
        per_state[name] = {"det_rate": frac[2], "conf": frac, "n_px": int(n)}
    return {
        "pixel_acc": float(correct / total) if total else float("nan"),
        "mIoU": float(np.nanmean(ious)),
        "IoU_per_class": ious,
        "loss": {k: v / max(1, n_batches) for k, v in loss_acc.items()},
        "per_state": per_state,
    }
