"""bench_dataloader.py — 一次性:量 runtime 生成的吞吐,比較 num_workers。"""
from __future__ import annotations
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from torch.utils.data import DataLoader
from src.data import RuntimeSceneDataset, load_assets
from src.data.config import DEFAULT_CONFIG

N_BATCH = 40
BS = 8

def bench(workers, assets):
    ds = RuntimeSceneDataset(length=N_BATCH * BS, assets=assets, config=DEFAULT_CONFIG)
    dl = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=workers,
                    persistent_workers=(workers > 0))
    it = iter(dl)
    next(it)  # 暖機(spawn workers / 第一批)
    t0 = time.perf_counter()
    seen = 1
    for _ in it:
        seen += 1
    dt = time.perf_counter() - t0
    n = (seen - 1) * BS
    return dt, n / dt

if __name__ == "__main__":
    assets = load_assets()
    print(f"CPU count={torch.get_num_threads()} (torch threads)")
    for w in [0, 4, 8]:
        dt, sps = bench(w, assets)
        full = 24000 / sps
        print(f"  workers={w:>2}: {dt:5.1f}s / {(N_BATCH-1)*BS} samples "
              f"= {sps:6.1f} samp/s  → 24000 樣本約 {full/60:4.1f} min(純生成,不含 GPU)")
