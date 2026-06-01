"""Generate Stage-6 figure set into docs/figures/stage6/training/.

Produces training curves, prediction panel, per-type performance, and confusion
matrix from the s6_final run. ALL labels in English.

Run (conda env dl_final):  python scripts/gen_stage6_figures.py
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import schema  # noqa: E402
from src.data import generate_scene, load_assets, encode_targets, encode_rgb  # noqa: E402
from src.data.config import DEFAULT_CONFIG, TEST_INDEX_OFFSET  # noqa: E402
from src.models import DefectSegNet, load_state_dict_flexible  # noqa: E402
from src.eval.metrics import infer_3class, _DEFECT_KEY  # noqa: E402

OUT = REPO / "docs" / "figures" / "stage6" / "training"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = REPO / "output" / "runs"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAG = "s6_final"
THR = 0.7
NCACHE = 100

CMAP3 = np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]], np.uint8)


def load_model(tag):
    ck = torch.load(RUNS / tag / "best.pt", map_location=DEV, weights_only=False)
    tc = ck["train_config"]
    m = DefectSegNet(base_c=tc["base_c"], depth=tc.get("depth", 4)).to(DEV)
    load_state_dict_flexible(m, ck["state_dict"])
    m.eval()
    return m, ck


def build_cache(model, n=NCACHE):
    assets = load_assets()
    cache = []
    model.eval()
    with torch.no_grad():
        for k in range(n):
            s = generate_scene(TEST_INDEX_OFFSET + k, assets, DEFAULT_CONFIG)
            rgb_t = encode_rgb(s.rgb).unsqueeze(0).to(DEV)
            out = model(rgb_t)
            pred, ppart, pdef = infer_3class(out, THR)
            tg = encode_targets(s.semantic, s.instance, s.meta, s.defect_region)
            dl = out[_DEFECT_KEY]
            if dl.dim() == 4:
                dl = dl.squeeze(1)
            cache.append({
                "rgb": s.rgb,
                "gt": s.semantic.astype(np.uint8),
                "pred": pred[0].cpu().numpy().astype(np.uint8),
                "p_def": pdef[0].cpu().numpy(),
                "instance": s.instance,
                "meta": s.meta,
            })
    return cache


# ── Fig: Training Curves (2×2) ────────────────────────────────
def fig_training_curves(history, ckpt):
    print("  [curves] training curves ...")
    s = history["step"]
    best_def_iou = ckpt.get("best_val_defect_iou", max(history["val_IoU_defect"]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    ax_loss, ax_iou, ax_lr, ax_tput = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Loss
    ax_loss.plot(s, history["L_total"], "-", color="#d4a017", lw=1.5, label="Train")
    ax_loss.plot(s, history["val_L_total"], "-", color="#8e44ad", lw=1.5, label="Val")
    ax_loss.set_ylabel("L_total")
    ax_loss.set_title("Loss (train vs val)")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)

    # Per-class IoU
    ax_iou.plot(s, history["val_IoU_bg"], "--", color="#888", lw=1.2, label="bg")
    ax_iou.plot(s, history["val_IoU_normal"], "--", color="#198754", lw=1.2, label="normal")
    ax_iou.plot(s, history["val_IoU_defect"], "-o", color="#c0392b", lw=2,
                markersize=3, label="defect (ckpt sel.)")
    ax_iou.set_ylabel("IoU")
    ax_iou.set_title("Val IoU per Class")
    ax_iou.set_ylim(-0.02, 1.02)
    ax_iou.legend(fontsize=9)
    ax_iou.grid(alpha=0.3)
    # Mark peak defect IoU
    def_ious = history["val_IoU_defect"]
    peak_idx = int(np.argmax(def_ious))
    ax_iou.annotate(f"peak: {def_ious[peak_idx]:.3f}\n@step {s[peak_idx]}",
                    xy=(s[peak_idx], def_ious[peak_idx]),
                    xytext=(s[peak_idx] + 300, def_ious[peak_idx] - 0.08),
                    fontsize=7, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))

    # LR
    ax_lr.plot(s, history["lr"], "-", color="#198754", lw=2)
    ax_lr.set_yscale("log")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_xlabel("Step")
    ax_lr.set_title("Learning Rate Schedule")
    ax_lr.grid(alpha=0.3)
    lr_arr = np.array(history["lr"])
    for i in range(1, len(lr_arr)):
        if lr_arr[i] < lr_arr[i - 1] * 0.9:
            ax_lr.axvline(s[i], color="#198754", alpha=0.3, ls=":")

    # Throughput
    ax_tput.plot(s, history["samples_per_s"], "-", color="#e65100", lw=1.5)
    ax_tput.set_ylabel("Samples/s")
    ax_tput.set_xlabel("Step")
    ax_tput.set_title("Training Throughput")
    ax_tput.grid(alpha=0.3)

    tc = ckpt.get("train_config", {})
    fig.suptitle(f"S6 Training — {TAG} (bc={tc.get('base_c',32)}, pw={tc.get('pos_weight',8)}, "
                 f"{tc.get('total_steps',4000)} steps, best defIoU={best_def_iou:.3f})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "s6_training_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> s6_training_curves.png")


# ── Fig: Prediction Panel ─────────────────────────────────────
def fig_pred_panel(cache):
    print("  [pred] prediction panel ...")
    # Select scenes: one per defect type + one clean
    type_picks = {"remesh": None, "displace": None, "bend": None, "normal": None}

    for i, c in enumerate(cache):
        types_in = set()
        for ins in c["meta"]["instances"]:
            if ins["is_defective"]:
                types_in.add(schema.state_to_type(ins["defect_state"]))
        has_def = (c["gt"] == 2).sum() > 100

        if not types_in and not has_def and type_picks["normal"] is None:
            type_picks["normal"] = i
        for t in ["remesh", "displace", "bend"]:
            if t in types_in and type_picks[t] is None:
                type_picks[t] = i
        if all(v is not None for v in type_picks.values()):
            break

    rows = [(t, type_picks[t]) for t in ["remesh", "displace", "bend", "normal"]
            if type_picks[t] is not None]
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[None, :]

    for r, (tname, idx) in enumerate(rows):
        c = cache[idx]
        axes[r, 0].imshow(c["rgb"])
        axes[r, 1].imshow(CMAP3[c["gt"]])
        axes[r, 2].imshow(CMAP3[c["pred"]])
        im = axes[r, 3].imshow(c["p_def"], cmap="jet", vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[r, 3], fraction=0.046, pad=0.04)
        for j in range(4):
            axes[r, j].axis("off")
        axes[r, 0].set_ylabel(tname, fontsize=10, rotation=0, labelpad=55, va="center")
        # Label defective instances
        for ins in c["meta"]["instances"]:
            if ins["is_defective"]:
                ys, xs = np.where(c["instance"] == ins["instance_id"])
                if len(xs) > 0:
                    t = schema.state_to_type(ins["defect_state"])
                    axes[r, 1].text(xs.mean(), ys.mean(), t, color="yellow", fontsize=7,
                                    ha="center", va="center",
                                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, ec="none"))

    for j, t in enumerate(["RGB", "GT (3-class)", "Prediction", "P(defect) heatmap"]):
        axes[0, j].set_title(t, fontsize=10)

    fig.suptitle("S6 Prediction Panel (green=normal, red=defect)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "s6_pred_panel.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> s6_pred_panel.png")


# ── Fig: Per-State Performance ────────────────────────────────
def fig_per_state(cache):
    print("  [per-state] performance breakdown ...")
    states = list(schema.STATE_CLASSES)  # normal, bend_45, displace, remesh
    det = {s: [0, 0] for s in states}  # [defect_pred_px, total_px]
    inst_ious = {s: [] for s in states if s != "normal"}

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
            if ins["is_defective"]:
                inter = int((pd & pix).sum())
                union = int((pd | pix).sum())
                if union > 0:
                    inst_ious[st].append(inter / union)

    det_rates = {s: det[s][0] / det[s][1] if det[s][1] else 0 for s in states}
    normal_fp = det_rates["normal"]

    # Bar chart: det_rate + delta
    defect_states = [s for s in states if s != "normal"]
    x = np.arange(len(defect_states))
    w = 0.3
    drs = [det_rates[s] for s in defect_states]
    deltas = [det_rates[s] - normal_fp for s in defect_states]
    mean_ious = [np.mean(inst_ious[s]) if inst_ious[s] else 0 for s in defect_states]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: detection rate + delta
    ax1.bar(x - w / 2, drs, w, label="Detection rate", color="#1a73e8")
    ax1.bar(x + w / 2, deltas, w, label=f"Delta (det - FP={normal_fp:.1%})", color="#2e7d32")
    ax1.axhline(normal_fp, color="#ef9a9a", ls="--", lw=1.5, label=f"Normal FP = {normal_fp:.1%}")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace("_", " ") for s in defect_states])
    ax1.set_ylabel("Rate")
    ax1.set_title("Per-State Detection Rate")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    for i, (dr, d) in enumerate(zip(drs, deltas)):
        ax1.text(i - w / 2, dr + 0.01, f"{dr:.1%}", ha="center", fontsize=8)
        ax1.text(i + w / 2, d + 0.01, f"{d:.1%}", ha="center", fontsize=8)

    # Right: per-instance IoU boxplot
    data = [inst_ious[s] for s in defect_states]
    labels = [f"{s.replace('_', ' ')}\n(n={len(inst_ious[s])})" for s in defect_states]
    ax2.boxplot(data, labels=labels, showmeans=True)
    ax2.set_ylabel("Per-Instance Defect IoU")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("Per-Instance Defect IoU Distribution")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"S6 Per-State Performance (thr={THR})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "s6_per_state.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> s6_per_state.png")

    return det_rates, inst_ious


# ── Fig: Confusion Matrix ─────────────────────────────────────
def fig_confusion(cache):
    print("  [confusion] matrix ...")
    states = list(schema.STATE_CLASSES)
    class_names = ["background", "normal_part", "defective_part"]
    mat = np.zeros((len(states), 3))
    for c in cache:
        for ins in c["meta"]["instances"]:
            ri = schema.STATE_TO_IDX[ins["defect_state"]]
            pix = c["instance"] == ins["instance_id"]
            for cp in range(3):
                mat[ri, cp] += int(((c["pred"] == cp) & pix).sum())
    norm = mat / mat.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(class_names, rotation=15)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([s.replace("_", " ") for s in states])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("GT State")
    for i in range(len(states)):
        for j in range(3):
            val = norm[i, j]
            ax.text(j, i, f"{val * 100:.1f}%", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=9)
    ax.set_title(f"S6 Confusion Matrix — row-normalized (thr={THR})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "s6_confusion.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> s6_confusion.png")


# ── Fig: KPI Summary Table ───────────────────────────────────
def fig_kpi_summary(cache, ckpt):
    print("  [kpi] summary ...")
    inter = np.zeros(3)
    union = np.zeros(3)
    correct = total = 0
    for c in cache:
        p, m = c["pred"], c["gt"]
        for k in range(3):
            pk, mk = (p == k), (m == k)
            inter[k] += int((pk & mk).sum())
            union[k] += int((pk | mk).sum())
        correct += int((p == m).sum())
        total += int(m.size)
    ious = [inter[k] / union[k] if union[k] > 0 else 0 for k in range(3)]
    miou = float(np.mean(ious))
    pixel_acc = correct / total

    # S3 baseline for comparison
    s3 = {"mIoU": 0.712, "defect_IoU": 0.362, "pixel_acc": 0.941}
    # S5 headline
    s5 = {"mIoU": 0.722, "defect_IoU": 0.390}

    metrics = ["mIoU", "Defect IoU", "Pixel Acc"]
    s3_vals = [s3["mIoU"], s3["defect_IoU"], s3["pixel_acc"]]
    s5_vals = [s5["mIoU"], s5["defect_IoU"], None]
    s6_vals = [miou, ious[2], pixel_acc]

    x = np.arange(len(metrics))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, s3_vals, w, label="S3 baseline", color="#9aa0a6")
    s5_plot = [v if v is not None else 0 for v in s5_vals]
    ax.bar(x, s5_plot, w, label="S5 (prev best)", color="#fbbc04")
    ax.bar(x + w, s6_vals, w, label="S6 (this run)", color="#1a73e8")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"S6 vs Previous Stages (test set, thr={THR})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bars in ax.containers:
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "s6_kpi_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> s6_kpi_comparison.png")

    return {"pixel_acc": pixel_acc, "mIoU": miou, "IoU": ious}


# ── Main ──────────────────────────────────────────────────────
def main():
    print(f"=== Stage 6 Figure Generation ===")
    print(f"device={DEV}  tag={TAG}  thr={THR}  output={OUT}\n")

    run_dir = RUNS / TAG
    if not (run_dir / "best.pt").exists():
        sys.exit(f"Run not found: {run_dir / 'best.pt'}")

    model, ckpt = load_model(TAG)
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))

    # Training curves
    fig_training_curves(history, ckpt)

    # Build prediction cache
    print(f"\n  Building cache ({NCACHE} scenes) ...")
    cache = build_cache(model, NCACHE)

    # KPI summary
    m = fig_kpi_summary(cache, ckpt)
    print(f"  -> mIoU={m['mIoU']:.3f} defIoU={m['IoU'][2]:.3f} pxAcc={m['pixel_acc']:.3f}\n")

    # Prediction panel
    fig_pred_panel(cache)

    # Per-state performance
    fig_per_state(cache)

    # Confusion matrix
    fig_confusion(cache)

    print(f"\n=== Done! {len(list(OUT.glob('*.png')))} figures in {OUT} ===")


if __name__ == "__main__":
    main()
