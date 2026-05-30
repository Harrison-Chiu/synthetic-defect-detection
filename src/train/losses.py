"""
losses.py — S5 雙頭 loss(part CE + defect 加權 BCE+Dice)

S5 機制(見 docs/stage5_plan.md §3):
- **part 頭(A)**:逐像素 CE,part/bg,與 S3 相同。
- **defect 頭(B,單通道 sigmoid)**:對 soft target T 套 **加權** BCE + **加權** soft-Dice,
  權重圖 W 來自 encode_targets(變形區附近高、壞件內部/遠背景≈0)。
  「interior ignore」就靠 W≈0 自然達成 —— loss 在那些像素貢獻趨零,矛盾消失。
  pos_weight ρ 補變形區正類稀少(bend_light 僅 ~5%)。

相對 S4 multihead_loss:拿掉 state/type(已否決),defect 改單通道加權路徑。
數值權重(head_weights / pos_weight)由 TrainConfig 注入。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src import schema

_EPS = 1e-6


def _weighted_defect_loss(logit, T, W, pos_weight: float):
    """單通道 sigmoid defect 頭:加權 BCE + 加權 soft-Dice。

    logit: (N,1,H,W) 或 (N,H,W);T/W: (N,H,W) float。回傳 (loss, bce, dice)。
    """
    if logit.dim() == 4:
        logit = logit.squeeze(1)
    pw = torch.tensor(float(pos_weight), device=logit.device)
    bce_map = F.binary_cross_entropy_with_logits(logit, T, reduction="none", pos_weight=pw)
    wbce = (W * bce_map).sum() / (W.sum() + _EPS)

    p = torch.sigmoid(logit)
    num = 2.0 * (W * p * T).sum()
    den = (W * p).sum() + (W * T).sum()
    dice = 1.0 - (num + _EPS) / (den + _EPS)
    return wbce + dice, wbce, dice


def multihead_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    head_weights: dict[str, float] | None = None,
    pos_weight: float = 8.0,
    heads: tuple[schema.Head, ...] = schema.HEADS,
):
    """S5 雙頭加權 loss。

    outputs: {"A": part_logits (N,2,H,W), "B": defect_logit (N,1,H,W)}。
    targets: encode_targets 的輸出(用到 "A","B_T","B_W")。
    head_weights: {head_key -> float},預設全 1.0(B 即 λ)。
    回傳 (total_loss, components dict[str,float])。
    """
    if head_weights is None:
        head_weights = {h.key: 1.0 for h in heads}

    comps: dict[str, float] = {}
    total = 0.0
    for h in heads:
        if h.name == "part":
            ce = F.cross_entropy(outputs[h.key], targets["A"])
            comps[f"L_{h.key}_ce"] = float(ce.item())
            total = total + head_weights.get(h.key, 1.0) * ce
        elif h.name == "defect":
            loss, bce, dice = _weighted_defect_loss(
                outputs[h.key], targets["B_T"], targets["B_W"], pos_weight)
            comps[f"L_{h.key}_bce"] = float(bce.item())
            comps[f"L_{h.key}_dice"] = float(dice.item())
            total = total + head_weights.get(h.key, 1.0) * loss
        else:
            raise ValueError(f"未知的 head 名稱:{h.name!r}(S5 只支援 part / defect)")

    comps["L_total"] = float(total.item())
    return total, comps
