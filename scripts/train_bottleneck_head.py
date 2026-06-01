"""Experiment: add auxiliary defect head at bottleneck (16×16).

Hypothesis: bend deformation is a global shape change that the bottleneck
(large receptive field, 16×16 spatial) might capture better than the final
decoder output (256×256, local texture features).

Architecture: standard DefectSegNet + extra 1×1 conv on bottleneck features,
bilinear-upsampled to 256×256 for loss. At inference, fuse both heads.

Run: python scripts/train_bottleneck_head.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from itertools import islice
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import schema
from src.data import RuntimeSceneDataset, load_assets, materialize
from src.data.config import DEFAULT_CONFIG, EVAL_INDEX_OFFSET, TEST_INDEX_OFFSET
from src.eval.metrics import evaluate_all
from src.eval.viz import save_prediction_snapshot
from src.models import DefectSegNet
from src.train import DEFAULT_TRAIN_CONFIG
from src.train.losses import _weighted_defect_loss, _EPS

_DEFECT_KEY = schema.HEADS[1].key  # "B"


class DefectSegNetWithBottleneckHead(DefectSegNet):
    """DefectSegNet + auxiliary defect head from bottleneck features.

    The bottleneck_head reads from the bottleneck (16×16 @ depth=4, 256 input)
    and outputs a single-channel defect logit, bilinear-upsampled to full res.
    """

    def __init__(self, base_c=8, depth=4, heads=schema.HEADS):
        super().__init__(base_c=base_c, depth=depth, heads=heads)
        # Bottleneck channels = base_c * 2^(depth-1)
        bn_ch = base_c * (2 ** (depth - 1))
        self.bottleneck_defect = nn.Sequential(
            nn.Conv2d(bn_ch, bn_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(bn_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_ch // 2, 1, 1),  # single-channel defect logit
        )

    def forward(self, x):
        # Encoder
        skips = []
        h = x
        for i, enc in enumerate(self.encs):
            h = enc(h if i == 0 else self.pool(h))
            skips.append(h)
        h_bn = self.bottleneck(self.pool(h))

        # Bottleneck defect head (16×16 → upsample to input size)
        bn_logit = self.bottleneck_defect(h_bn)
        bn_logit = F.interpolate(bn_logit, size=x.shape[2:], mode="bilinear", align_corners=False)

        # Decoder (same as parent)
        h = h_bn
        for t, (up, dec) in enumerate(zip(self.ups, self.decs)):
            skip = skips[self.depth - 1 - t]
            h = dec(torch.cat([up(h), skip], dim=1))

        out = {key: conv(h) for key, conv in self.head_convs.items()}
        out["B_bn"] = bn_logit  # extra key for bottleneck defect head
        return out


def multihead_loss_with_bn(outputs, targets, head_weights, pos_weight, bn_weight=0.5):
    """Standard dual-head loss + auxiliary bottleneck defect loss."""
    from src.train.losses import multihead_loss
    total, comps = multihead_loss(outputs, targets, head_weights, pos_weight)

    # Bottleneck defect head loss (same T/W supervision)
    bn_logit = outputs["B_bn"]
    if bn_logit.dim() == 4:
        bn_logit = bn_logit.squeeze(1)
    bn_loss, bn_bce, bn_dice = _weighted_defect_loss(
        bn_logit, targets["B_T"], targets["B_W"], pos_weight)
    comps["L_bn_bce"] = float(bn_bce.item())
    comps["L_bn_dice"] = float(bn_dice.item())
    total = total + bn_weight * bn_loss
    comps["L_total"] = float(total.item())
    return total, comps


def infer_fused(outputs, defect_thr=0.7):
    """Fuse decoder head + bottleneck head for inference."""
    from src.eval.metrics import infer_3class
    # Average the two defect heads' probabilities
    dec_p = torch.sigmoid(outputs[_DEFECT_KEY].squeeze(1))
    bn_p = torch.sigmoid(outputs["B_bn"].squeeze(1))
    fused_p = (dec_p + bn_p) / 2.0

    # Use fused probability for 3-class prediction
    part_logits = outputs["A"]
    part_prob = F.softmax(part_logits, dim=1)
    bg_prob = part_prob[:, 0]
    part_fg = part_prob[:, 1]

    pred = torch.zeros_like(bg_prob, dtype=torch.long)
    pred[bg_prob < 0.5] = 1  # foreground
    pred[(bg_prob < 0.5) & (fused_p > defect_thr)] = 2  # defect

    return pred, part_prob[:, 1], fused_p


def evaluate_with_bn(model, loader, device, defect_thr=0.7, head_weights=None, pos_weight=8.0):
    """Evaluate with fused bottleneck+decoder heads."""
    model.eval()
    inter = np.zeros(3)
    union = np.zeros(3)
    correct = total_px = 0
    loss_sum = 0.0
    n_batch = 0
    per_state_det = {}
    per_state_tot = {}

    with torch.no_grad():
        for rgb, targets in loader:
            rgb = rgb.to(device)
            tg = {k: v.to(device) for k, v in targets.items()}
            outputs = model(rgb)
            pred, _, p_def = infer_fused(outputs, defect_thr)
            gt = tg["sem3"]

            for k in range(3):
                pk, mk = (pred == k), (gt == k)
                inter[k] += int((pk & mk).sum().item())
                union[k] += int((pk | mk).sum().item())
            correct += int((pred == gt).sum().item())
            total_px += int(gt.numel())

            loss, comps = multihead_loss_with_bn(outputs, tg, head_weights, pos_weight)
            loss_sum += comps["L_total"]
            n_batch += 1

    ious = [inter[k] / union[k] if union[k] > 0 else 0.0 for k in range(3)]
    miou = float(np.mean(ious))
    return {
        "pixel_acc": correct / total_px,
        "mIoU": miou,
        "IoU_per_class": ious,
        "loss": {"L_total": loss_sum / max(1, n_batch)},
        "per_state": {},
    }


def _log(path, msg):
    print(msg)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_cfg = DEFAULT_CONFIG
    train_cfg = dataclasses.replace(DEFAULT_TRAIN_CONFIG,
                                     base_c=8,
                                     total_steps=4000,
                                     eval_every=100)
    tag = "s6_bn_head"
    run_dir = REPO / "output" / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    n_eval = 100
    workers = 8
    bn_weight = 0.5  # auxiliary loss weight

    assets = load_assets()
    train_len = train_cfg.total_steps * train_cfg.batch_size
    train_ds = RuntimeSceneDataset(length=train_len, assets=assets, config=gen_cfg, index_offset=0)
    print(f"  Generating val/test each {n_eval} scenes...")
    val_ds = materialize(RuntimeSceneDataset(n_eval, assets, gen_cfg, EVAL_INDEX_OFFSET),
                         batch_size=train_cfg.batch_size, num_workers=workers)
    test_ds = materialize(RuntimeSceneDataset(n_eval, assets, gen_cfg, TEST_INDEX_OFFSET),
                          batch_size=train_cfg.batch_size, num_workers=workers)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0)

    model = DefectSegNetWithBottleneckHead(base_c=train_cfg.base_c, depth=train_cfg.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    bn_params = sum(p.numel() for p in model.bottleneck_defect.parameters())
    _log(log_path, f"Model: DefectSegNetWithBottleneckHead bc={train_cfg.base_c} depth={train_cfg.depth}")
    _log(log_path, f"Total params={n_params:,} (bottleneck head: {bn_params:,})")
    _log(log_path, f"bn_weight={bn_weight} device={device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=train_cfg.lr_factor, patience=train_cfg.lr_patience,
        threshold=train_cfg.lr_threshold, min_lr=train_cfg.lr_min)

    history = {"step": [], "L_total": [], "val_L_total": [],
               "val_mIoU": [], "val_IoU_bg": [], "val_IoU_normal": [],
               "val_IoU_defect": [], "lr": [], "samples_per_s": []}
    best_defect_iou = -1.0
    best_miou = -1.0
    best_state = None

    def _infinite(loader):
        while True:
            yield from loader

    t_start = time.time()
    t_window = time.time()
    run_loss_sum, run_loss_n = 0.0, 0
    model.train()

    for step, (rgb, targets) in enumerate(islice(_infinite(train_loader), train_cfg.total_steps), start=1):
        rgb = rgb.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        outputs = model(rgb)
        loss, comps = multihead_loss_with_bn(outputs, targets, train_cfg.head_weights,
                                              train_cfg.pos_weight, bn_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        run_loss_sum += comps["L_total"]
        run_loss_n += 1

        if step % train_cfg.eval_every == 0 or step == train_cfg.total_steps:
            window_dt = time.time() - t_window
            samples = run_loss_n * train_cfg.batch_size
            sps = samples / max(1e-6, window_dt)

            val = evaluate_with_bn(model, val_loader, device, train_cfg.defect_thr,
                                    train_cfg.head_weights, train_cfg.pos_weight)
            cur_lr = optimizer.param_groups[0]["lr"]
            avg_loss = run_loss_sum / max(1, run_loss_n)
            run_loss_sum, run_loss_n = 0.0, 0

            history["step"].append(step)
            history["L_total"].append(avg_loss)
            history["val_L_total"].append(val["loss"]["L_total"])
            history["val_mIoU"].append(val["mIoU"])
            history["val_IoU_bg"].append(val["IoU_per_class"][0])
            history["val_IoU_normal"].append(val["IoU_per_class"][1])
            history["val_IoU_defect"].append(val["IoU_per_class"][2])
            history["lr"].append(cur_lr)
            history["samples_per_s"].append(sps)

            cur_defect_iou = val["IoU_per_class"][2]
            if cur_defect_iou > best_defect_iou:
                best_defect_iou = cur_defect_iou
                best_miou = val["mIoU"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            iou = val["IoU_per_class"]
            gpu_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0
            _log(log_path, f"step {step:5d}/{train_cfg.total_steps} lr={cur_lr:.1e} "
                           f"L={avg_loss:.3f} vL={val['loss']['L_total']:.3f} "
                           f"mIoU={val['mIoU']:.3f} bg={iou[0]:.3f} norm={iou[1]:.3f} "
                           f"def={iou[2]:.3f} | {sps:.1f} smp/s gpu={gpu_mb:.0f}MB")
            scheduler.step(cur_defect_iou)
            model.train()
            t_window = time.time()

    total_dt = time.time() - t_start
    _log(log_path, f"done in {total_dt/60:.1f} min. best val defectIoU={best_defect_iou:.3f} mIoU={best_miou:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test eval
    final = evaluate_with_bn(model, test_loader, device, train_cfg.defect_thr,
                              train_cfg.head_weights, train_cfg.pos_weight)
    test_metrics = {"pixel_acc": final["pixel_acc"], "mIoU": final["mIoU"],
                    "IoU_per_class": final["IoU_per_class"]}
    _log(log_path, f"[test] mIoU={test_metrics['mIoU']:.3f} "
                   f"defectIoU={test_metrics['IoU_per_class'][2]:.3f} "
                   f"pixel_acc={test_metrics['pixel_acc']*100:.2f}%")

    torch.save({
        "state_dict": model.state_dict(),
        "best_val_miou": best_miou,
        "best_val_defect_iou": best_defect_iou,
        "test_metrics": test_metrics,
        "train_config": train_cfg.__dict__,
        "bn_weight": bn_weight,
    }, run_dir / "best.pt")
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    _log(log_path, f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
