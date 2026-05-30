"""
loop.py — step-based 訓練迴圈(S5)

機制(整理規則 §2/§3)
---------------------
- 以 **total steps** 驅動;每 `eval_every` 步在 val set 上評估 + checkpoint,
  每 `snapshot_every` 步存預測快照。
- train_loader 來自 RuntimeSceneDataset(shuffle=False),每步一張 distinct 場景。
- **val / test 分離**(S5 步驟6):val 選 best,test 報數字(disjoint index 區段)。
- 產出:best.pt / history.json / train.log / snapshots/。
- history 同時記 **train + val loss**(報告要求),per-class val IoU,lr。
- train.log 含 **效能紀錄**(samples/s、GPU 峰值記憶體),供日後調 workers/batch 參考。

資料集構建 / 路徑由 cli 注入,本檔只負責「怎麼訓練」。
"""

from __future__ import annotations

import json
import time
from itertools import islice
from pathlib import Path

import torch

from src.eval.metrics import evaluate_all
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
    val_loader,
    device,
    run_dir: str | Path,
    config: TrainConfig = DEFAULT_TRAIN_CONFIG,
    snapshot_batch=None,
    test_loader=None,
):
    """跑完整 step-based 訓練,回傳 (history, best_metrics)。

    val_loader 選 best;test_loader(可選)只在最後報數字,不參與選 best。
    """
    cfg = config
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = run_dir / "snapshots"
    log_path = run_dir / "train.log"

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience,
        threshold=cfg.lr_threshold, min_lr=cfg.lr_min,
    )

    history: dict[str, list] = {"step": [], "L_total": [], "val_L_total": [],
                                "val_mIoU": [], "val_IoU_bg": [], "val_IoU_normal": [],
                                "val_IoU_defect": [], "lr": [], "samples_per_s": []}
    best_miou = -1.0
    best_state = None

    n_params = sum(p.numel() for p in model.parameters())
    _log(log_path, f"params={n_params:,} device={device} base_c={cfg.base_c} "
                   f"total_steps={cfg.total_steps} eval_every={cfg.eval_every} "
                   f"batch={cfg.batch_size} workers={cfg.num_workers} "
                   f"pos_weight={cfg.pos_weight} defect_thr={cfg.defect_thr}")

    t_start = time.time()
    t_window = time.time()
    run_loss_sum, run_loss_n = 0.0, 0
    model.train()
    for step, (rgb, targets) in enumerate(islice(_infinite(train_loader), cfg.total_steps), start=1):
        rgb = rgb.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        outputs = model(rgb)
        loss, comps = multihead_loss(outputs, targets, cfg.head_weights, cfg.pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        run_loss_sum += comps["L_total"]
        run_loss_n += 1

        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            # 效能:本窗 train 牆鐘(含資料生成,排除下方 eval)
            window_dt = time.time() - t_window
            samples = run_loss_n * cfg.batch_size
            sps = samples / max(1e-6, window_dt)

            val = evaluate_all(model, val_loader, device, defect_thr=cfg.defect_thr,
                               head_weights=cfg.head_weights, pos_weight=cfg.pos_weight)
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

            if val["mIoU"] > best_miou:
                best_miou = val["mIoU"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            iou = val["IoU_per_class"]
            gpu_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0
            _log(log_path, f"step {step:5d}/{cfg.total_steps} lr={cur_lr:.1e} "
                           f"L={avg_loss:.3f} vL={val['loss']['L_total']:.3f} "
                           f"mIoU={val['mIoU']:.3f} bg={iou[0]:.3f} norm={iou[1]:.3f} "
                           f"def={iou[2]:.3f} | {sps:.1f} smp/s gpu={gpu_mb:.0f}MB")
            scheduler.step(val["mIoU"])
            model.train()
            t_window = time.time()

        if snapshot_batch is not None and (step % cfg.snapshot_every == 0 or step == cfg.total_steps):
            save_prediction_snapshot(model, snapshot_batch, step, snap_dir, device)
            model.train()
            t_window = time.time()

    total_dt = time.time() - t_start
    _log(log_path, f"done in {total_dt/60:.1f} min. best val mIoU={best_miou:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # 最終數字:test set(獨立)優先;無則退回 val
    report_loader = test_loader if test_loader is not None else val_loader
    split_name = "test" if test_loader is not None else "val"
    final = evaluate_all(model, report_loader, device, defect_thr=cfg.defect_thr,
                         head_weights=cfg.head_weights, pos_weight=cfg.pos_weight)
    test_metrics = {"pixel_acc": final["pixel_acc"], "mIoU": final["mIoU"],
                    "IoU_per_class": final["IoU_per_class"]}
    per_state = final["per_state"]
    _log(log_path, f"[{split_name}] mIoU={test_metrics['mIoU']:.3f} "
                   f"defectIoU={test_metrics['IoU_per_class'][2]:.3f} "
                   f"pixel_acc={test_metrics['pixel_acc']*100:.2f}%")

    torch.save({
        "state_dict": model.state_dict(),
        "best_val_miou": best_miou,
        "test_metrics": test_metrics,
        "test_split": split_name,
        "per_state": per_state,
        "train_config": cfg.__dict__,
    }, run_dir / "best.pt")
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_metrics = {"best_val_miou": best_miou, "test_metrics": test_metrics, "per_state": per_state}
    return history, best_metrics
