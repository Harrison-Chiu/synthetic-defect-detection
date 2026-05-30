"""
metrics.py — S5 評估指標

雙頭(part + defect-sigmoid)→ 合成與 Stage 3 可比的 3-class(bg/normal/defect):
- part 頭 argmax 給 0=bg / 1=part;
- defect 頭 sigmoid > thr 且落在 part 上 → 升級為 2=defect。

GT 來源改用 encode_targets 的 "sem3"(整顆壞件=2,與 S3 schema 一致),
**不再**從訓練 target 重建 —— 訓練 target(T)是「變形區」而非整顆,兩者用途不同。

- evaluate_3class：主指標 mIoU / pixel_acc / per-class IoU。
- evaluate_per_state：每個 defect_state 的「偵測率」(該 state 零件像素中被判 defect 的比例)
  + per-state defect IoU,對齊 S3 results 的 per-state 表(報告「三種瑕疵各自表現」用)。
- evaluate_loss：在 val/test 上算平均 loss(供 train+val loss 曲線)。
"""

from __future__ import annotations

import numpy as np
import torch

from src import schema

_PART_KEY = schema.HEADS_BY_NAME["part"].key
_DEFECT_KEY = schema.HEADS_BY_NAME["defect"].key
_STATE_CLASSES = schema.STATE_CLASSES


def infer_3class(outputs: dict[str, torch.Tensor], defect_thr: float = 0.5):
    """dict logits → 3-class pred(0=bg,1=normal_part,2=defect)。

    回傳 (pred, part_pred, defect_prob)。
    """
    part_pred = outputs[_PART_KEY].argmax(dim=1)
    defect_logit = outputs[_DEFECT_KEY]
    if defect_logit.dim() == 4:
        defect_logit = defect_logit.squeeze(1)
    defect_prob = torch.sigmoid(defect_logit)
    out = torch.zeros_like(part_pred)
    is_part = part_pred == 1
    out[is_part] = 1
    out[is_part & (defect_prob > defect_thr)] = 2
    return out, part_pred, defect_prob


@torch.no_grad()
def evaluate_3class(model, loader, device, num_classes=3, defect_thr: float = 0.5):
    model.eval()
    inter = np.zeros(num_classes)
    union = np.zeros(num_classes)
    correct = total = 0
    for rgb, targets in loader:
        outputs = model(rgb.to(device))
        pred, _, _ = infer_3class(outputs, defect_thr)
        pred = pred.cpu()
        gt = targets["sem3"]
        for c in range(num_classes):
            p, t = (pred == c), (gt == c)
            inter[c] += (p & t).sum().item()
            union[c] += (p | t).sum().item()
        correct += (pred == gt).sum().item()
        total += gt.numel()
    ious = [float(inter[c] / union[c]) if union[c] > 0 else float("nan") for c in range(num_classes)]
    return {"pixel_acc": float(correct / total), "mIoU": float(np.nanmean(ious)), "IoU_per_class": ious}


@torch.no_grad()
def evaluate_per_state(model, loader, device, defect_thr: float = 0.5):
    """每個 defect_state 的偵測率 + defect IoU(只在該 state 的零件像素上)。

    回傳 {state_name: {"det_rate": float, "iou": float, "n_px": int}}。
    - det_rate:該 state 零件像素中被判 defect(pred==2)的比例(= S3 表的「比例」)。
    - iou:把該 state 零件視為 GT-defect,與 pred==2 算 IoU(僅該 state 像素參與)。
    """
    model.eval()
    n_st = len(_STATE_CLASSES)
    pos = np.zeros(n_st)      # 被判 defect 的像素數
    npx = np.zeros(n_st)      # 該 state 零件總像素數
    inter = np.zeros(n_st)
    union = np.zeros(n_st)
    for rgb, targets in loader:
        outputs = model(rgb.to(device))
        pred, _, _ = infer_3class(outputs, defect_thr)
        pred = pred.cpu()
        state_px = targets["state_px"]
        pred_def = pred == 2
        for c in range(n_st):
            m = state_px == c
            cnt = m.sum().item()
            if cnt == 0:
                continue
            npx[c] += cnt
            pos[c] += (pred_def & m).sum().item()
            # 該 state 是否「應」為 defect:normal(idx0)→ GT 非 defect;其餘 → GT defect
            gt_def = m if c != schema.STATE_TO_IDX["normal"] else torch.zeros_like(m)
            inter[c] += (pred_def & gt_def).sum().item()
            union[c] += (pred_def | gt_def).sum().item()
    out = {}
    for c, name in enumerate(_STATE_CLASSES):
        out[name] = {
            "det_rate": float(pos[c] / npx[c]) if npx[c] > 0 else float("nan"),
            "iou": float(inter[c] / union[c]) if union[c] > 0 else float("nan"),
            "n_px": int(npx[c]),
        }
    return out


@torch.no_grad()
def evaluate_loss(model, loader, device, head_weights=None, pos_weight: float = 8.0):
    """val/test 平均 loss(供 train+val loss 曲線)。回傳 dict 各 component 的平均。"""
    from src.train.losses import multihead_loss
    model.eval()
    acc: dict[str, float] = {}
    n = 0
    for rgb, targets in loader:
        rgb = rgb.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        outputs = model(rgb)
        _, comps = multihead_loss(outputs, targets, head_weights, pos_weight)
        for k, v in comps.items():
            acc[k] = acc.get(k, 0.0) + v
        n += 1
    return {k: v / max(1, n) for k, v in acc.items()}
