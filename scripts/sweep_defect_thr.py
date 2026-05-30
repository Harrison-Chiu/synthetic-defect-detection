"""sweep_defect_thr.py — 後處理掃推論閾值 defect_thr(免重訓,壓 FP 用)。

對已訓練的 run,固定 100 張 test 場景,掃多個 defect_thr,印每個閾值的:
  defect IoU、好件誤報率(normal FP)、bend/displace/remesh 偵測率。
看 precision/recall 隨閾值的取捨,挑推論端最佳 thr。

用法:python scripts/sweep_defect_thr.py s5_bc8 [thr1 thr2 ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import schema  # noqa: E402
from src.eval.report import _build_cache, _per_type_stats  # noqa: E402
from src.models import DefectSegNet, load_state_dict_flexible  # noqa: E402


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "s5_bc8"
    thrs = [float(x) for x in sys.argv[2:]] or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(REPO / "output/runs" / tag / "best.pt", map_location=dev, weights_only=False)
    tc = ck["train_config"]
    m = DefectSegNet(base_c=tc["base_c"], depth=tc.get("depth", 4)).to(dev)
    load_state_dict_flexible(m, ck["state_dict"])
    print(f"{tag}: base_c={tc['base_c']} depth={tc.get('depth',4)} pos_weight={tc.get('pos_weight')}\n")
    print(f"{'thr':>5} {'defIoU':>7} {'normalFP':>9} {'bend':>7} {'displace':>9} {'remesh':>7}")
    for thr in thrs:
        cache = _build_cache(m, dev, 100, thr)
        # 3-class defect IoU
        inter = union = 0
        nfp_pos = nfp_tot = 0
        for c in cache:
            pd, gd = c["pred"] == 2, c["gt"] == 2
            inter += int((pd & gd).sum()); union += int((pd | gd).sum())
            nm = c["gt"] == 1
            nfp_pos += int((pd & nm).sum()); nfp_tot += int(nm.sum())
        iou = inter / union if union else float("nan")
        fp = nfp_pos / nfp_tot if nfp_tot else float("nan")
        pt = _per_type_stats(cache)
        print(f"{thr:>5.2f} {iou:>7.3f} {fp*100:>8.1f}% "
              f"{pt['bend']['det_rate']*100:>6.1f}% {pt['displace']['det_rate']*100:>8.1f}% "
              f"{pt['remesh']['det_rate']*100:>6.1f}%")


if __name__ == "__main__":
    main()
