"""
loop.py — step-based 訓練迴圈

機制(整理規則 §2/§3)
---------------------
- 以 **total steps** 驅動(不是 epoch);每 `eval_every` 步在凍結 eval set 上評估 +
  checkpoint,每 `snapshot_every` 步存預測快照。
- train_loader 來自 RuntimeSceneDataset(shuffle=False),每步一張 distinct 場景;
  不需 shuffle,因為生成器本身就不重複。
- 產出寫進 run_dir(對齊 runs/<stage_tag>/):best.pt / history.json / train.log /
  snapshots/。

資料集構建 / 路徑由 cli 注入,本檔只負責「怎麼訓練」。
"""

from __future__ import annotations

import json
import time
from itertools import islice
from pathlib import Path

import torch

from src.eval.metrics import evaluate_3class, evaluate_per_state
from src.eval.viz import save_prediction_snapshot
from src.train.config import DEFAULT_TRAIN_CONFIG, TrainConfig
from src.train.losses import multihead_loss


def _infinite(loader):
    while True:
        yield from loader


def _log(log_path: Path, msg: str):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def train_loop(
    model,
    train_loader,
    eval_loader,
    device,
    run_dir: str | Path,
    config: TrainConfig = DEFAULT_TRAIN_CONFIG,
    snapshot_batch=None,
):
    """跑完整 step-based 訓練,回傳 (history, best_metrics)。"""
    cfg = config
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = run_dir / "snapshots"
    log_path = run_dir / "train.log"

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience,
        threshold=cfg.lr_threshold, min_lr=cfg.lr_min,
    )

    history: dict[str, list] = {"step": [], "L_total": [], "val_mIoU": [],
                                "val_IoU_bg": [], "val_IoU_normal": [], "val_IoU_defect": [], "lr": []}
    best_miou = -1.0
    best_state = None

    n_params = sum(p.numel() for p in model.parameters())
    _log(log_path, f"params={n_params:,} device={device} total_steps={cfg.total_steps} "
                   f"eval_every={cfg.eval_every} batch={cfg.batch_size}")

    t_start = time.time()
    run_loss_sum, run_loss_n = 0.0, 0
    model.train()
    for step, (rgb, targets) in enumerate(islice(_infinite(train_loader), cfg.total_steps), start=1):
        rgb = rgb.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        outputs = model(rgb)
        loss, comps = multihead_loss(outputs, targets, cfg.head_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        run_loss_sum += comps["L_total"]
        run_loss_n += 1

        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            val = evaluate_3class(model, eval_loader, device)
            cur_lr = optimizer.param_groups[0]["lr"]
            avg_loss = run_loss_sum / max(1, run_loss_n)
            run_loss_sum, run_loss_n = 0.0, 0

            history["step"].append(step)
            history["L_total"].append(avg_loss)
            history["val_mIoU"].append(val["mIoU"])
            history["val_IoU_bg"].append(val["IoU_per_class"][0])
            history["val_IoU_normal"].append(val["IoU_per_class"][1])
            history["val_IoU_defect"].append(val["IoU_per_class"][2])
            history["lr"].append(cur_lr)

            if val["mIoU"] > best_miou:
                best_miou = val["mIoU"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            iou = val["IoU_per_class"]
            _log(log_path, f"step {step:5d}/{cfg.total_steps} lr={cur_lr:.1e} "
                           f"L={avg_loss:.3f} mIoU={val['mIoU']:.3f} "
                           f"bg={iou[0]:.3f} norm={iou[1]:.3f} def={iou[2]:.3f}")
            scheduler.step(val["mIoU"])
            model.train()

        if snapshot_batch is not None and (step % cfg.snapshot_every == 0 or step == cfg.total_steps):
            save_prediction_snapshot(model, snapshot_batch, step, snap_dir, device)
            model.train()

    total_dt = time.time() - t_start
    _log(log_path, f"done in {total_dt/60:.1f} min. best val mIoU={best_miou:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_3class(model, eval_loader, device)
    per_state_iou = evaluate_per_state(model, eval_loader, device)

    torch.save({
        "state_dict": model.state_dict(),
        "best_val_miou": best_miou,
        "test_metrics": test_metrics,
        "per_state_iou": per_state_iou,
        "train_config": cfg.__dict__,
    }, run_dir / "best.pt")
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_metrics = {"best_val_miou": best_miou, "test_metrics": test_metrics, "per_state_iou": per_state_iou}
    return history, best_metrics
