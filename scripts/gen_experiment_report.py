"""生成 S6 ablation 實驗比較報告（中文）。

比較三組實驗：
1. s6_bc8 — baseline（全瑕疵，標準雙頭）
2. s6_bend_only — 只有 bend 瑕疵
3. s6_bn_head — 加 bottleneck auxiliary head

產出：
- 訓練曲線比較圖
- Per-state 偵測率比較（需要跑 inference）
- 實驗摘要表格圖

Run: python scripts/gen_experiment_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 嘗試使用支援中文的字型
for font_name in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [font_name] + matplotlib.rcParams["font.sans-serif"]
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import schema
from src.data import load_assets, generate_scene, encode_rgb, encode_targets
from src.data.config import DEFAULT_CONFIG, TEST_INDEX_OFFSET
from src.models import DefectSegNet, load_state_dict_flexible
from src.eval.metrics import infer_3class, _DEFECT_KEY

OUT = REPO / "docs" / "figures" / "stage6" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = REPO / "output" / "runs"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THR = 0.7
NCACHE = 100

CMAP3 = np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]], np.uint8)

EXPERIMENTS = {
    "s6_bc8": {"label": "Baseline (bc=8)", "color": "#1a73e8"},
    "s6_bend_only": {"label": "Bend-only", "color": "#e65100"},
    "s6_bn_head": {"label": "Bottleneck Head", "color": "#2e7d32"},
}


def load_history(tag):
    return json.loads((RUNS / tag / "history.json").read_text(encoding="utf-8"))


def load_model_standard(tag):
    ck = torch.load(RUNS / tag / "best.pt", map_location=DEV, weights_only=False)
    tc = ck["train_config"]
    m = DefectSegNet(base_c=tc["base_c"], depth=tc.get("depth", 4)).to(DEV)
    load_state_dict_flexible(m, ck["state_dict"])
    m.eval()
    return m, ck


def load_model_bn(tag):
    """Load bottleneck head model."""
    from scripts.train_bottleneck_head import DefectSegNetWithBottleneckHead, infer_fused
    ck = torch.load(RUNS / tag / "best.pt", map_location=DEV, weights_only=False)
    tc = ck["train_config"]
    m = DefectSegNetWithBottleneckHead(base_c=tc["base_c"], depth=tc.get("depth", 4)).to(DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck, infer_fused


def build_cache(model, infer_fn=None):
    """Build prediction cache. infer_fn overrides default infer_3class."""
    assets = load_assets()
    cache = []
    model.eval()
    with torch.no_grad():
        for k in range(NCACHE):
            s = generate_scene(TEST_INDEX_OFFSET + k, assets, DEFAULT_CONFIG)
            rgb_t = encode_rgb(s.rgb).unsqueeze(0).to(DEV)
            out = model(rgb_t)
            if infer_fn:
                pred, _, pdef = infer_fn(out, THR)
            else:
                pred, _, pdef = infer_3class(out, THR)
            cache.append({
                "rgb": s.rgb,
                "gt": s.semantic.astype(np.uint8),
                "pred": pred[0].cpu().numpy().astype(np.uint8),
                "p_def": pdef[0].cpu().numpy(),
                "instance": s.instance,
                "meta": s.meta,
            })
    return cache


def per_state_rates(cache):
    """Per defect_state detection rate."""
    states = list(schema.STATE_CLASSES)
    det = {s: [0, 0] for s in states}
    for c in cache:
        pd = c["pred"] == 2
        for ins in c["meta"]["instances"]:
            st = ins["defect_state"]
            pix = c["instance"] == ins["instance_id"]
            ps = int(pix.sum())
            if ps < 50:
                continue
            det[st][0] += int((pd & pix).sum())
            det[st][1] += ps
    return {s: det[s][0] / det[s][1] if det[s][1] else 0 for s in states}


# ── Fig 1: 訓練曲線比較 ──────────────────────────────────────
def fig_training_comparison():
    print("  [1] 訓練曲線比較 ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for tag, info in EXPERIMENTS.items():
        h = load_history(tag)
        s = h["step"]
        axes[0].plot(s, h["L_total"], "-", color=info["color"], lw=1.5, alpha=0.7, label=f'{info["label"]} train')
        axes[0].plot(s, h["val_L_total"], "--", color=info["color"], lw=1.5, label=f'{info["label"]} val')
        axes[1].plot(s, h["val_IoU_defect"], "-o", color=info["color"], lw=2, markersize=2, label=info["label"])
        axes[2].plot(s, h["lr"], "-", color=info["color"], lw=2, label=info["label"])

    axes[0].set_title("Loss（實線=train, 虛線=val）")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("L_total")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    axes[1].set_title("Val Defect IoU")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Defect IoU")
    axes[1].set_ylim(-0.02, 0.55); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Step"); axes[2].set_ylabel("LR")
    axes[2].set_yscale("log"); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    fig.suptitle("S6 Ablation 實驗 — 訓練動態比較", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "s6_exp_training_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> s6_exp_training_comparison.png")


# ── Fig 2: Per-state 偵測率比較 ──────────────────────────────
def fig_per_state_comparison(caches):
    print("  [2] Per-state 偵測率比較 ...")
    states = ["normal", "bend_45", "displace", "remesh"]
    state_labels = ["normal\n(FP 率)", "bend 45°", "displace", "remesh"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(states))
    n_exp = len(caches)
    w = 0.22
    offsets = np.linspace(-(n_exp-1)*w/2, (n_exp-1)*w/2, n_exp)

    for i, (tag, cache) in enumerate(caches.items()):
        rates = per_state_rates(cache)
        vals = [rates.get(s, 0) for s in states]
        info = EXPERIMENTS[tag]
        bars = ax.bar(x + offsets[i], vals, w, label=info["label"], color=info["color"], alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{v:.1%}", ha="center", fontsize=7, color=info["color"])

    ax.set_xticks(x)
    ax.set_xticklabels(state_labels)
    ax.set_ylabel("偵測率")
    ax.set_title("Per-State 偵測率比較（thr=0.7）")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "s6_exp_per_state.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> s6_exp_per_state.png")


# ── Fig 3: 實驗摘要表格 ──────────────────────────────────────
def fig_summary_table(caches):
    print("  [3] 摘要表格 ...")
    rows = []
    for tag, cache in caches.items():
        ck = torch.load(RUNS / tag / "best.pt", map_location="cpu", weights_only=False)
        tm = ck["test_metrics"]
        tc = ck["train_config"]
        ps = per_state_rates(cache)
        n_params = sum(p.numel() for p in DefectSegNet(base_c=tc["base_c"], depth=tc.get("depth", 4)).parameters())
        if tag == "s6_bn_head":
            n_params += 2000  # approx bottleneck head params

        rows.append({
            "實驗": EXPERIMENTS[tag]["label"],
            "參數量": f"{n_params/1000:.1f}K",
            "test mIoU": f"{tm['mIoU']:.3f}",
            "test defect IoU": f"{tm['IoU_per_class'][2]:.3f}",
            "pixel acc": f"{tm['pixel_acc']*100:.1f}%",
            "bend 偵測率": f"{ps.get('bend_45', 0):.1%}",
            "displace 偵測率": f"{ps.get('displace', 0):.1%}",
            "remesh 偵測率": f"{ps.get('remesh', 0):.1%}",
            "normal FP 率": f"{ps.get('normal', 0):.1%}",
        })

    # Also add bc=32 for reference
    if (RUNS / "s6_final" / "best.pt").exists():
        ck32 = torch.load(RUNS / "s6_final" / "best.pt", map_location="cpu", weights_only=False)
        tm32 = ck32["test_metrics"]
        rows.append({
            "實驗": "bc=32 (參考)",
            "參數量": "3,313K",
            "test mIoU": f"{tm32['mIoU']:.3f}",
            "test defect IoU": f"{tm32['IoU_per_class'][2]:.3f}",
            "pixel acc": f"{tm32['pixel_acc']*100:.1f}%",
            "bend 偵測率": "—",
            "displace 偵測率": "—",
            "remesh 偵測率": "—",
            "normal FP 率": "—",
        })

    cols = list(rows[0].keys())
    cell_text = [[r[c] for c in cols] for r in rows]

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=cols, loc="center",
                      cellLoc="center", colColours=["#e8eaf6"] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Highlight best values
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            cell = table[i + 1, j]
            if "bn_head" in list(caches.keys())[i] if i < len(caches) else False:
                cell.set_facecolor("#e8f5e9")

    fig.suptitle("S6 Ablation 實驗結果摘要", fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT / "s6_exp_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   -> s6_exp_summary_table.png")


# ── Fig 4: Bottleneck head 效果示意（pred 比較）──────────────
def fig_pred_comparison(cache_baseline, cache_bn):
    print("  [4] 預測比較面板 ...")
    # Find scenes with defects for comparison
    pairs = []
    for i in range(min(len(cache_baseline), len(cache_bn))):
        cb = cache_baseline[i]
        cn = cache_bn[i]
        if (cb["gt"] == 2).sum() > 200:
            pairs.append((cb, cn))
        if len(pairs) >= 3:
            break

    if not pairs:
        print("   ! No defective scenes found")
        return

    n = len(pairs)
    fig, axes = plt.subplots(n, 5, figsize=(18, 3.5 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["RGB", "GT", "Baseline 預測", "Bottleneck 預測", "差異（綠=BN更好）"]
    for r, (cb, cn) in enumerate(pairs):
        axes[r, 0].imshow(cb["rgb"])
        axes[r, 1].imshow(CMAP3[cb["gt"]])
        axes[r, 2].imshow(CMAP3[cb["pred"]])
        axes[r, 3].imshow(CMAP3[cn["pred"]])

        # Difference: green where BN correct & baseline wrong, red where opposite
        gt = cb["gt"]
        diff_img = np.full((*gt.shape, 3), 128, dtype=np.uint8)
        bn_better = (cn["pred"] == gt) & (cb["pred"] != gt)
        base_better = (cb["pred"] == gt) & (cn["pred"] != gt)
        diff_img[bn_better] = [0, 200, 0]
        diff_img[base_better] = [200, 0, 0]
        axes[r, 4].imshow(diff_img)

        for j in range(5):
            axes[r, j].axis("off")
            if r == 0:
                axes[0, j].set_title(col_titles[j], fontsize=10)

    fig.suptitle("Baseline vs Bottleneck Head 預測比較（綠=BN更好，紅=Baseline更好）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "s6_exp_pred_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> s6_exp_pred_comparison.png")


# ── Main ──────────────────────────────────────────────────────
def main():
    print(f"=== S6 Ablation 實驗報告生成 ===")
    print(f"device={DEV}  output={OUT}\n")

    # Fig 1: Training curves (no model loading needed)
    fig_training_comparison()

    # Build caches for per-state analysis
    print("\n  建立 prediction cache ...")
    caches = {}

    print(f"  [s6_bc8] loading ...")
    m1, _ = load_model_standard("s6_bc8")
    caches["s6_bc8"] = build_cache(m1)
    del m1

    print(f"  [s6_bend_only] loading ...")
    m2, _ = load_model_standard("s6_bend_only")
    caches["s6_bend_only"] = build_cache(m2)
    del m2

    print(f"  [s6_bn_head] loading ...")
    m3, _, infer_fn = load_model_bn("s6_bn_head")
    caches["s6_bn_head"] = build_cache(m3, infer_fn)

    # Fig 2-4
    fig_per_state_comparison(caches)
    fig_summary_table(caches)
    fig_pred_comparison(caches["s6_bc8"], caches["s6_bn_head"])

    print(f"\n=== Done! {len(list(OUT.glob('*.png')))} figures in {OUT} ===")


if __name__ == "__main__":
    main()
