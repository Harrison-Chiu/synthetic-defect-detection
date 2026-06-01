"""Train with bend-only defects — experiment to isolate bend detection.

Filters assets to only keep bend_45 in defect pool (displace/remesh patches
become normal). Everything else identical to s6_final.

Run: python scripts/train_bend_only.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from torch.utils.data import DataLoader

from src.data import RuntimeSceneDataset, load_assets, materialize
from src.data.config import DEFAULT_CONFIG, EVAL_INDEX_OFFSET, TEST_INDEX_OFFSET
from src.models import DefectSegNet
from src.train import DEFAULT_TRAIN_CONFIG, train_loop


def filter_assets_bend_only(assets):
    """Filter part_buckets: only bend_45 stays in 'defect', others move to 'normal'."""
    new_buckets = {}
    for hdri_idx, pools in assets.part_buckets.items():
        new_normal = list(pools["normal"])
        new_defect = []
        for path, state in pools["defect"]:
            if state == "bend_45":
                new_defect.append((path, state))
            else:
                # displace/remesh → treat as normal
                new_normal.append((path, "normal"))
        new_buckets[hdri_idx] = {"normal": new_normal, "defect": new_defect}
    # Assets is frozen dataclass → use object.__setattr__ to bypass
    object.__setattr__(assets, 'part_buckets', new_buckets)
    return assets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_cfg = DEFAULT_CONFIG
    train_cfg = dataclasses.replace(DEFAULT_TRAIN_CONFIG,
                                     base_c=8,  # S5 recommended
                                     total_steps=4000,
                                     eval_every=100)
    tag = "s6_bend_only"
    run_dir = REPO / "output" / "runs" / tag
    n_eval = 100
    workers = 8

    assets = load_assets()
    # Count before filter
    total_def = sum(len(p["defect"]) for p in assets.part_buckets.values())
    total_bend = sum(1 for p in assets.part_buckets.values()
                     for _, s in p["defect"] if s == "bend_45")
    print(f"Before filter: {total_def} defect patches, {total_bend} bend_45")

    assets = filter_assets_bend_only(assets)
    total_def_after = sum(len(p["defect"]) for p in assets.part_buckets.values())
    print(f"After filter:  {total_def_after} defect patches (bend_45 only)")

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

    n_snap = min(4, n_eval)
    snap_batch = (torch.stack([val_ds[i][0] for i in range(n_snap)]),
                  torch.stack([val_ds[i][1]["sem3"] for i in range(n_snap)])) if n_snap else None

    model = DefectSegNet(base_c=train_cfg.base_c, depth=train_cfg.depth)
    print(f"train tag={tag} device={device} base_c={train_cfg.base_c} "
          f"depth={train_cfg.depth} run_dir={run_dir}")
    _, best = train_loop(model, train_loader, val_loader, device, run_dir, train_cfg,
                         snap_batch, test_loader=test_loader)
    print(f"best val mIoU={best['best_val_miou']:.3f}  test mIoU={best['test_metrics']['mIoU']:.3f}  "
          f"test defectIoU={best['test_metrics']['IoU_per_class'][2]:.3f}")


if __name__ == "__main__":
    main()
