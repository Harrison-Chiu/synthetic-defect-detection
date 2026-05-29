"""
metrics.py — 評估指標(移植 train_stage4.py)

泛化成吃「model 回傳 dict[head_key->logits]」+「targets dict[head_key->tensor]」。
- evaluate_3class：合成 head A(part) + head B(state) → bg/normal/defect 3 類,
  與 Stage 3 schema 可比的主指標(mIoU / pixel_acc)。
- evaluate_per_state：head B 的 7 類 per-state IoU(只在零件像素),報告用。

3-class GT 由 state target 重建:<0→bg(0)、==0→normal_part(1)、>0→defect(2)。
"""

from __future__ import annotations

import numpy as np
import torch

from src import schema

# 主/狀態 head key(從 schema 取,不寫死)
_PART_KEY = schema.HEADS_BY_NAME["part"].key
_STATE_KEY = schema.HEADS_BY_NAME["state"].key
_N_STATES = schema.HEADS_BY_NAME["state"].num_classes


def _state_gt_to_3class(state_gt: torch.Tensor) -> torch.Tensor:
    return torch.where(
        state_gt < 0, torch.zeros_like(state_gt),
        torch.where(state_gt == 0, torch.ones_like(state_gt), torch.full_like(state_gt, 2)),
    )


def infer_3class(outputs: dict[str, torch.Tensor]):
    """dict logits → 3-class pred(0=bg,1=normal_part,2=defect)。"""
    part_pred = outputs[_PART_KEY].argmax(dim=1)
    state_pred = outputs[_STATE_KEY].argmax(dim=1)
    out = torch.zeros_like(part_pred)
    is_part = part_pred == 1
    out[is_part] = 1
    out[is_part & (state_pred > 0)] = 2
    return out, part_pred, state_pred


@torch.no_grad()
def evaluate_3class(model, loader, device, num_classes=3):
    model.eval()
    inter = np.zeros(num_classes)
    union = np.zeros(num_classes)
    correct = total = 0
    for rgb, targets in loader:
        outputs = model(rgb.to(device))
        pred, _, _ = infer_3class(outputs)
        pred = pred.cpu()
        gt = _state_gt_to_3class(targets[_STATE_KEY])
        for c in range(num_classes):
            p, t = (pred == c), (gt == c)
            inter[c] += (p & t).sum().item()
            union[c] += (p | t).sum().item()
        correct += (pred == gt).sum().item()
        total += gt.numel()
    ious = [float(inter[c] / union[c]) if union[c] > 0 else float("nan") for c in range(num_classes)]
    return {"pixel_acc": float(correct / total), "mIoU": float(np.nanmean(ious)), "IoU_per_class": ious}


@torch.no_grad()
def evaluate_per_state(model, loader, device):
    """head B 的 7 類 per-state IoU(只在零件像素)。"""
    model.eval()
    inter = np.zeros(_N_STATES)
    union = np.zeros(_N_STATES)
    for rgb, targets in loader:
        outputs = model(rgb.to(device))
        state_pred = outputs[_STATE_KEY].argmax(dim=1).cpu()
        state_gt = targets[_STATE_KEY]
        is_part_gt = state_gt >= 0
        for c in range(_N_STATES):
            p = (state_pred == c) & is_part_gt
            t = state_gt == c
            inter[c] += (p & t).sum().item()
            union[c] += (p | t).sum().item()
    return [float(inter[c] / union[c]) if union[c] > 0 else float("nan") for c in range(_N_STATES)]
