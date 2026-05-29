"""
losses.py — schema 驅動的多頭 loss

移植 train_stage4.py 的 multiclass_dice_loss + three_head_loss,但泛化成
「掃 schema.HEADS,每個 head 依其 `losses` 套 CE / Dice,加權求和」。
新增/改 head 只動 schema,loss 自動跟上;權重數值在 TrainConfig。

注意:這裡正是 schema guardrail 警告的所在 —— instance head(state/type)目前
仍套逐像素 CE+Dice(忠實沿用 S4)。要改成 instance-level 監督是 S5 的事。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src import schema


def multiclass_dice_loss(logits, target, is_part_mask, num_classes, eps=1e-6):
    """Macro per-class Dice,只在零件像素上計算(移植自 S4)。"""
    probs = F.softmax(logits, dim=1)
    is_part = is_part_mask.float().unsqueeze(1)
    target_clamped = target.clone()
    target_clamped[target_clamped < 0] = 0
    onehot = F.one_hot(target_clamped, num_classes).permute(0, 3, 1, 2).float()
    probs = probs * is_part
    onehot = onehot * is_part
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def multihead_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    head_weights: dict[str, float] | None = None,
    heads: tuple[schema.Head, ...] = schema.HEADS,
):
    """加權多頭 loss。

    outputs / targets：{head_key -> tensor}(與 model.forward / encode_targets 對齊)。
    head_weights：{head_key -> float},預設全 1.0(= S4 ALPHA_B=ALPHA_C=1.0)。
    回傳 (total_loss, components dict[str,float])。
    """
    if head_weights is None:
        head_weights = {h.key: 1.0 for h in heads}

    # is_part:零件像素遮罩,給 instance head 的 Dice 用。取主 head 的 target==1。
    main_key = schema.main_heads(heads)[0].key
    is_part = targets[main_key] == 1

    total = 0.0
    comps: dict[str, float] = {}
    for h in heads:
        logits = outputs[h.key]
        target = targets[h.key]
        ignore = h.ignore_index if h.ignore_index is not None else -100
        head_loss = 0.0
        if schema.Loss.CE in h.losses:
            ce = F.cross_entropy(logits, target, ignore_index=ignore)
            comps[f"L_{h.key}_ce"] = float(ce.item())
            head_loss = head_loss + ce
        if schema.Loss.DICE in h.losses:
            dice = multiclass_dice_loss(logits, target, is_part, h.num_classes)
            comps[f"L_{h.key}_dice"] = float(dice.item())
            head_loss = head_loss + dice
        total = total + head_weights.get(h.key, 1.0) * head_loss

    comps["L_total"] = float(total.item())
    return total, comps
