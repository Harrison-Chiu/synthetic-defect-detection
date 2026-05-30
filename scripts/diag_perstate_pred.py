"""
diag_perstate_pred.py — 一次性診斷:s4_repro 模型對各 defect_state 的逐像素預測 vs GT

目的(bend 診斷):看模型在每種瑕疵上『實際標在哪』。
預期(若 label 矛盾假設成立):
  - bend：GT 整顆塗 defect(紅),但 pred 內部判 normal(綠)、頂多輪廓零星紅 → recall 崩。
  - displace/remesh：表面紋理可逐像素辨識，pred 較能填滿 → 較高 recall。

輸出：docs/figures/s4_repro_perstate_pred.png（每列一個 state：rgb | GT 3-class | pred 3-class）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import load_assets
from src.data.config import DEFAULT_CONFIG
from src.data.generator import generate_scene
from src.data.dataset import encode_rgb
from src.models import DefectSegNet
from src.eval.metrics import infer_3class

STATES = ["bend_light", "bend_heavy", "displace_light", "displace_heavy",
          "remesh_light", "remesh_heavy"]
N_PER_STATE = 1  # 每 state 取幾個範例

# 3-class 上色：0 bg=灰黑, 1 normal=綠, 2 defect=紅
PALETTE = np.array([[40, 40, 40], [40, 180, 60], [220, 40, 40]], dtype=np.uint8)


def colorize(mask3):
    return PALETTE[np.clip(mask3, 0, 2)]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(REPO / "output/runs/s4_repro/best.pt", map_location=device,
                      weights_only=False)
    base_c = ckpt.get("train_config", {}).get("base_c", 32)
    model = DefectSegNet(base_c=base_c).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    assets = load_assets()

    # 掃場景，收集每個 state 的實例範例
    found = {s: [] for s in STATES}
    i = 0
    while any(len(v) < N_PER_STATE for v in found.values()) and i < 4000:
        sample = generate_scene(i, assets, DEFAULT_CONFIG)
        with torch.no_grad():
            out = model(encode_rgb(sample.rgb).unsqueeze(0).to(device))
        pred3, _, _ = infer_3class(out)
        pred3 = pred3[0].cpu().numpy().astype(np.uint8)
        gt3 = sample.semantic  # 已是 0/1/2
        for ins in sample.meta["instances"]:
            st = ins["defect_state"]
            if st in found and len(found[st]) < N_PER_STATE:
                x0, y0, x1, y1 = ins["bbox"]
                pad = 6
                x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
                x1, y1 = min(256, x1 + pad), min(256, y1 + pad)
                found[st].append({
                    "rgb": sample.rgb[y0:y1, x0:x1],
                    "gt": gt3[y0:y1, x0:x1],
                    "pred": pred3[y0:y1, x0:x1],
                    "scene": i, "iid": ins["instance_id"],
                })
        i += 1

    rows = [(s, ex) for s in STATES for ex in found[s]]
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.5 * n))
    if n == 1:
        axes = axes[None, :]
    for r, (st, ex) in enumerate(rows):
        # defect 像素 recall（GT==2 中被 pred 判 2 的比例）
        gtm = ex["gt"] == 2
        rec = float((ex["pred"][gtm] == 2).mean()) if gtm.any() else float("nan")
        axes[r, 0].imshow(ex["rgb"]); axes[r, 0].set_ylabel(f"{st}\nscene{ex['scene']}", fontsize=8)
        axes[r, 1].imshow(colorize(ex["gt"]))
        axes[r, 2].imshow(colorize(ex["pred"]))
        axes[r, 2].set_title(f"defect recall={rec:.2f}", fontsize=8)
        print(f"  {st:16s} scene{ex['scene']:>4} iid{ex['iid']}  "
              f"GT defect px={int(gtm.sum()):5d}  pred-defect recall={rec:.3f}")
        if r == 0:
            axes[r, 0].set_title("RGB", fontsize=9)
            axes[r, 1].set_title("GT (3-class)", fontsize=9)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    plt.suptitle("s4_repro 預測 vs GT｜綠=normal_part 紅=defect_part", fontsize=10)
    plt.tight_layout()
    out = REPO / "docs/figures/s4_repro_perstate_pred.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"scanned {i} scenes")
    for s in STATES:
        print(f"  {s}: {len(found[s])} examples")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
