"""Generate Stage-5 figure set (15-16 figures) into docs/figures/stage5/.

Matches the finalized design in docs/stage5_figure_plan.md.
ALL labels in English (no CJK — avoids matplotlib tofu on some systems).

Run (conda env dl_final):  python scripts/gen_stage5_figures.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import schema  # noqa: E402
from src.data import generate_scene, load_assets, encode_targets, encode_rgb  # noqa: E402
from src.data.config import DEFAULT_CONFIG, TEST_INDEX_OFFSET  # noqa: E402
from src.models import DefectSegNet, load_state_dict_flexible  # noqa: E402
from src.eval.report import _build_cache, _per_type_stats  # noqa: E402
from src.eval.metrics import infer_3class, _DEFECT_KEY  # noqa: E402

OUT = REPO / "docs" / "figures" / "stage5"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = REPO / "output" / "runs"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THR = 0.7
NCACHE = 100

# ── Utilities ──────────────────────────────────────────────────
CMAP3 = np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]], np.uint8)


def load_model(tag):
    ck = torch.load(RUNS / tag / "best.pt", map_location=DEV, weights_only=False)
    tc = ck["train_config"]
    m = DefectSegNet(base_c=tc["base_c"], depth=tc.get("depth", 4)).to(DEV)
    load_state_dict_flexible(m, ck["state_dict"])
    m.eval()
    return m, tc


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def cache_scores(cache):
    """defect IoU + normal FP rate from cache."""
    inter = union = 0
    fp_pos = fp_tot = 0
    for c in cache:
        pd, gd, nm = c["pred"] == 2, c["gt"] == 2, c["gt"] == 1
        inter += int((pd & gd).sum())
        union += int((pd | gd).sum())
        fp_pos += int((pd & nm).sum())
        fp_tot += int(nm.sum())
    return {
        "defIoU": inter / union if union else float("nan"),
        "normalFP": fp_pos / fp_tot if fp_tot else float("nan"),
    }


def per_state_from_cache(cache):
    """Per defect_state detection rate."""
    agg = {}
    for c in cache:
        pd = c["pred"] == 2
        for ins in c["meta"]["instances"]:
            st = ins["defect_state"]
            if st not in agg:
                agg[st] = [0, 0]
            pix = c["instance"] == ins["instance_id"]
            ps = int(pix.sum())
            if ps < 50:
                continue
            agg[st][0] += int((pd & pix).sum())
            agg[st][1] += ps
    out = {}
    for st, (det, tot) in agg.items():
        out[st] = det / tot if tot else 0.0
    return out


# ── Fig 00: Dataset Showcase ──────────────────────────────────
def fig_00_dataset_showcase():
    """3 sub-figures: defect states, domain randomization, part types."""
    print("  [00] dataset showcase ...")
    assets = load_assets()

    # Sub-A: Defect states (2R x 5C) — patch row + scene row
    # We use specific indices to get each defect state
    states_order = ["normal", "remesh_heavy", "displace_heavy", "bend_heavy", "bend_light"]
    # Find scenes with specific states
    state_scenes = {s: None for s in states_order}
    for idx in range(500):
        s = generate_scene(TEST_INDEX_OFFSET + 500 + idx, assets, DEFAULT_CONFIG)
        for ins in s.meta["instances"]:
            st = ins["defect_state"]
            if st in state_scenes and state_scenes[st] is None:
                state_scenes[st] = (s, ins)
        if all(v is not None for v in state_scenes.values()):
            break

    fig = plt.figure(figsize=(14, 10))
    # Sub-A: 2R x 5C
    gs_a = fig.add_gridspec(2, 5, top=0.95, bottom=0.62, hspace=0.05, wspace=0.05)
    for col, st_name in enumerate(states_order):
        pair = state_scenes.get(st_name)
        if pair is None:
            continue
        scene, ins = pair
        inst_mask = scene.instance == ins["instance_id"]
        ys, xs = np.where(inst_mask)
        if len(ys) == 0:
            continue
        y0, y1 = max(0, ys.min() - 5), min(scene.rgb.shape[0], ys.max() + 5)
        x0, x1 = max(0, xs.min() - 5), min(scene.rgb.shape[1], xs.max() + 5)
        # Row 0: patch crop
        ax = fig.add_subplot(gs_a[0, col])
        ax.imshow(scene.rgb[y0:y1, x0:x1])
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Patch", fontsize=9)
        ax.set_title(st_name.replace("_", " "), fontsize=8)
        # Row 1: full scene
        ax2 = fig.add_subplot(gs_a[1, col])
        ax2.imshow(scene.rgb)
        ax2.axis("off")
        if col == 0:
            ax2.set_ylabel("Scene", fontsize=9)

    # Sub-B: Domain Randomization (1R x 4C) — same scene, different backgrounds
    # Use 4 consecutive indices (DR is inherent in the generator)
    gs_b = fig.add_gridspec(1, 4, top=0.56, bottom=0.32, hspace=0.05, wspace=0.05)
    dr_labels = ["HDRI env A", "HDRI env B", "HDRI env C", "HDRI env D"]
    for col in range(4):
        s = generate_scene(TEST_INDEX_OFFSET + 1100 + col, assets, DEFAULT_CONFIG)
        ax = fig.add_subplot(gs_b[0, col])
        ax.imshow(s.rgb)
        ax.axis("off")
        ax.set_title(dr_labels[col], fontsize=8)

    # Sub-C: Part Types (1R x 4C)
    # Part type is extracted from source_part path, e.g.
    # "output/patches/normal/pan_head_az120_el+010_h0_normal.png"
    gs_c = fig.add_gridspec(1, 4, top=0.28, bottom=0.02, hspace=0.05, wspace=0.05)
    part_names = ["socket_head", "pan_head", "hex_nut", "flange_nut"]

    def _part_type_from_meta(ins_meta):
        sp = ins_meta.get("source_part", "")
        fname = sp.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]  # basename
        for pn in part_names:
            if fname.startswith(pn):
                return pn
        return ""

    for col in range(4):
        found = False
        for idx in range(300):
            s = generate_scene(TEST_INDEX_OFFSET + 1200 + col * 300 + idx, assets, DEFAULT_CONFIG)
            for ins in s.meta["instances"]:
                if _part_type_from_meta(ins) == part_names[col] and not ins["is_defective"]:
                    inst_mask = s.instance == ins["instance_id"]
                    ys, xs = np.where(inst_mask)
                    if len(ys) > 100:
                        y0 = max(0, ys.min() - 3)
                        y1 = min(s.rgb.shape[0], ys.max() + 3)
                        x0 = max(0, xs.min() - 3)
                        x1 = min(s.rgb.shape[1], xs.max() + 3)
                        ax = fig.add_subplot(gs_c[0, col])
                        ax.imshow(s.rgb[y0:y1, x0:x1])
                        ax.axis("off")
                        ax.set_title(part_names[col].replace("_", " "), fontsize=8)
                        found = True
                        break
            if found:
                break
        if not found:
            ax = fig.add_subplot(gs_c[0, col])
            ax.text(0.5, 0.5, part_names[col], ha="center", va="center", fontsize=9)
            ax.axis("off")

    fig.text(0.5, 0.97, "A. Defect States", ha="center", fontsize=10, weight="bold")
    fig.text(0.5, 0.58, "B. Domain Randomization", ha="center", fontsize=10, weight="bold")
    fig.text(0.5, 0.30, "C. Part Types", ha="center", fontsize=10, weight="bold")
    fig.savefig(OUT / "00_dataset_showcase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   -> 00_dataset_showcase.png")


# ── Fig 01: Supervision T/W ───────────────────────────────────
def fig_01_supervision_TW():
    """3R x 4C: strategically selected (remesh / bend / displace)."""
    print("  [01] supervision T/W ...")
    assets = load_assets()

    # Find one scene per target type.
    # IMPORTANT: pick scenes where ALL defective instances are the SAME type,
    # so T/W visualization cleanly shows only that type's deformation.
    target_types = ["remesh", "bend", "displace"]
    picks = {t: None for t in target_types}
    for idx in range(1000):
        s = generate_scene(TEST_INDEX_OFFSET + 2000 + idx, assets, DEFAULT_CONFIG)
        def_types = set()
        for ins in s.meta["instances"]:
            if ins["is_defective"]:
                def_types.add(schema.state_to_type(ins["defect_state"]))
        if len(def_types) != 1:
            continue  # skip mixed-type or all-normal scenes
        t = def_types.pop()
        if t in picks and picks[t] is None:
            tg = encode_targets(s.semantic, s.instance, s.meta, s.defect_region)
            picks[t] = (s, tg["B_T"].numpy(), tg["B_W"].numpy())
        if all(v is not None for v in picks.values()):
            break

    rows = [(t, picks[t]) for t in target_types if picks[t] is not None]
    n = len(rows)
    fig, ax = plt.subplots(n, 4, figsize=(12, 3.0 * n))
    if n == 1:
        ax = ax[None, :]

    col_titles = ["RGB Scene", "Semantic GT", "Target T", "Weight W"]
    for r, (tname, (scene, T, W)) in enumerate(rows):
        ax[r, 0].imshow(scene.rgb)
        # Semantic GT: bg=black, normal=green, defect=red
        ax[r, 1].imshow(CMAP3[scene.semantic.astype(int)])
        ax[r, 2].imshow(T, cmap="hot", vmin=0, vmax=1)
        w_max = max(1e-6, float(W.max()))
        im_w = ax[r, 3].imshow(W, cmap="viridis", vmin=0, vmax=w_max)
        for j in range(4):
            ax[r, j].axis("off")
        ax[r, 0].set_ylabel(tname, fontsize=10, rotation=0, labelpad=50, va="center")

    for j, t in enumerate(col_titles):
        ax[0, j].set_title(t, fontsize=10)

    # Annotation on W column: "W~0: interior ignored"
    # Point to the center of a defective part's interior
    if n > 0:
        ax[0, 3].annotate("W≈0: interior\nignored", xy=(0.5, 0.5),
                          xycoords="axes fraction", fontsize=8,
                          color="white", ha="center", va="center",
                          bbox=dict(fc="black", alpha=0.6, ec="none", pad=2))

    fig.suptitle("Interior-Ignore Supervision: T marks deformation region, "
                 "W≈0 inside defective parts", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "01_supervision_TW.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 01_supervision_TW.png")


# ── Fig 03: KPI vs S3 ─────────────────────────────────────────
def fig_03_kpi_vs_s3(head_cache):
    """Grouped bar chart: S3 vs S5."""
    print("  [03] KPI vs S3 ...")
    sc = cache_scores(head_cache)
    pt = _per_type_stats(head_cache)

    s3_vals = {"mIoU": 0.712, "defect IoU": 0.362, "remesh inst IoU": 0.276}
    s5_vals = {
        "mIoU": 0.722,  # from best_val_miou
        "defect IoU": sc["defIoU"],
        "remesh inst IoU": pt["remesh"]["inst_iou"],
    }
    keys = list(s3_vals)
    x = np.arange(len(keys))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w/2, [s3_vals[k] for k in keys], w,
                label="S3 baseline", color="#9aa0a6")
    b2 = ax.bar(x + w/2, [s5_vals[k] for k in keys], w,
                label="S5 (interior-ignore)", color="#1a73e8")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score")
    ax.set_title(f"S5 vs S3 Baseline (test set, thr={THR})")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Bar value labels
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=8)

    # Annotation arrow on defect IoU
    diff = s5_vals["defect IoU"] - s3_vals["defect IoU"]
    ax.annotate(f"+{diff:.3f}", xy=(1 + w/2, s5_vals["defect IoU"] + 0.02),
                fontsize=10, color="#1a73e8", weight="bold", ha="center")

    fig.tight_layout()
    fig.savefig(OUT / "03_kpi_vs_s3.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 03_kpi_vs_s3.png")


# ── Fig 04+05: Capacity Sweep (combined) ──────────────────────
def fig_0405_capacity_sweep():
    """1R x 2C: width line + depth bar."""
    print("  [04+05] capacity sweep ...")

    # Left: width sweep
    width_runs = ["s5_bc2", "s5_bc4", "s5_bc6", "s5_bc8", "s5_bc16", "s5_bc24", "s5_bc32"]
    bc_list, params_list, di_list = [], [], []
    for tag in width_runs:
        if not (RUNS / tag / "best.pt").exists():
            continue
        m, tc = load_model(tag)
        sc = cache_scores(_build_cache(m, DEV, NCACHE, THR))
        bc_list.append(tc["base_c"])
        params_list.append(n_params(m) / 1000)
        di_list.append(sc["defIoU"])
        print(f"     bc{tc['base_c']:>2}: {params_list[-1]:.0f}K  defIoU={di_list[-1]:.3f}")

    # Right: depth sweep
    depth_runs = ["s5_bc8_d2", "s5_bc8_d3", "s5_bc8"]
    d_list, dp_list, dd_list = [], [], []
    for tag in depth_runs:
        if not (RUNS / tag / "best.pt").exists():
            continue
        m, tc = load_model(tag)
        sc = cache_scores(_build_cache(m, DEV, NCACHE, THR))
        d_list.append(tc.get("depth", 4))
        dp_list.append(n_params(m) / 1000)
        dd_list.append(sc["defIoU"])
        print(f"     depth{d_list[-1]}: {dp_list[-1]:.0f}K  defIoU={dd_list[-1]:.3f}")

    order = np.argsort(d_list)
    d_list = [d_list[i] for i in order]
    dp_list = [dp_list[i] for i in order]
    dd_list = [dd_list[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left subplot: width line chart
    ax1.plot(bc_list, di_list, "o-", color="#1a73e8", lw=2, markersize=7)
    ax1.set_xlabel("base_c (channel width)")
    ax1.set_ylabel(f"Defect IoU (thr={THR})")
    ax1.set_title("Width Sweep")
    for xi, yi, pp in zip(bc_list, di_list, params_list):
        ax1.annotate(f"{pp:.0f}K", (xi, yi), textcoords="offset points",
                     xytext=(0, 9), fontsize=7, ha="center")
    # Saturation line at bc=8
    if 8 in bc_list:
        ax1.axvline(8, color="gray", ls="--", alpha=0.6)
        ax1.text(8.5, ax1.get_ylim()[1] * 0.95, "saturation", fontsize=8,
                 color="gray", va="top")
    # Gray band for bc8-bc32 flatness
    if len(di_list) >= 4:
        flat_vals = [d for b, d in zip(bc_list, di_list) if b >= 8]
        if flat_vals:
            ax1.axhspan(min(flat_vals) - 0.005, max(flat_vals) + 0.005,
                        alpha=0.1, color="gray")
    ax1.grid(alpha=0.3)
    ax1.text(0.98, 0.02, "thr=0.5, pw=8, 3000 steps\n(sweep conditions)",
             transform=ax1.transAxes, fontsize=7, ha="right", va="bottom",
             color="gray", style="italic")

    # Right subplot: depth bar chart
    colors = ["#1a73e8"] * len(d_list)
    bars = ax2.bar([f"depth={d}" for d in d_list], dd_list, color=colors, width=0.5)
    ax2.set_ylabel(f"Defect IoU (thr={THR})")
    ax2.set_title("Depth Sweep (bc=8)")
    for b, pp, dval in zip(bars, dp_list, dd_list):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                 f"{dval:.3f}\n{pp:.0f}K", ha="center", fontsize=8)
    # Reference line: bc=4 at same param count as depth=3
    if len(di_list) >= 2 and 4 in bc_list:
        bc4_idx = bc_list.index(4)
        bc4_iou = di_list[bc4_idx]
        ax2.axhline(bc4_iou, color="red", ls="--", alpha=0.7)
        ax2.text(0.95, bc4_iou + 0.005, f"bc=4 (same 52K): {bc4_iou:.3f}",
                 transform=ax2.get_yaxis_transform(), fontsize=7, color="red",
                 ha="right", va="bottom")
    ax2.set_ylim(0, max(dd_list) * 1.3 if dd_list else 0.5)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Model Capacity: width saturates at bc=8; depth more efficient than width",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "0405_capacity_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 0405_capacity_sweep.png")


# ── Fig 06: Threshold Sweep ───────────────────────────────────
def fig_06_thr_sweep(model):
    """Dual Y-axis: defect IoU vs normal FP rate."""
    print("  [06] threshold sweep ...")
    thrs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    di, fp = [], []
    for t in thrs:
        sc = cache_scores(_build_cache(model, DEV, NCACHE, t))
        di.append(sc["defIoU"])
        fp.append(sc["normalFP"])
        print(f"     thr={t}: defIoU={di[-1]:.3f} normalFP={fp[-1]:.3f}")

    fig, a1 = plt.subplots(figsize=(7.5, 4.5))
    l1 = a1.plot(thrs, di, "o-", color="#1a73e8", lw=2, label="Defect IoU")
    a1.set_xlabel("defect_thr")
    a1.set_ylabel("Defect IoU", color="#1a73e8")
    a1.tick_params(axis="y", labelcolor="#1a73e8")

    a2 = a1.twinx()
    l2 = a2.plot(thrs, fp, "s--", color="#d62728", lw=2, label="Normal FP rate")
    a2.set_ylabel("Normal FP Rate", color="#d62728")
    a2.tick_params(axis="y", labelcolor="#d62728")

    a1.axvline(0.7, color="gray", ls=":", alpha=0.7, lw=1.5)
    a1.text(0.71, max(di) * 0.95, "selected", fontsize=8, color="gray")

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    a1.legend(lines, labels, loc="center right")
    a1.set_title("Threshold Trade-off: raise thr suppresses FP but reduces IoU")
    a1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "06_thr_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 06_thr_sweep.png")


# ── Fig 07: Pos_weight Sweep ──────────────────────────────────
def fig_07_posweight_sweep():
    """Bar chart, pw=5 highlighted."""
    print("  [07] pos_weight sweep ...")
    candidates = ["s5_pw3", "s5_pw5", "s5_pw8"]
    pw_list, di_list = [], []
    for tag in candidates:
        if not (RUNS / tag / "best.pt").exists():
            print(f"     (skip {tag}: missing)")
            continue
        m, tc = load_model(tag)
        sc = cache_scores(_build_cache(m, DEV, NCACHE, THR))
        pw_list.append(tc.get("pos_weight", float("nan")))
        di_list.append(sc["defIoU"])
        print(f"     pw={pw_list[-1]}: defIoU={di_list[-1]:.3f}")

    order = np.argsort(pw_list)
    pw_list = [pw_list[i] for i in order]
    di_list = [di_list[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = ["#1a73e8" if pw == 5 else "#b0bec5" for pw in pw_list]
    bars = ax.bar([f"pw={int(p)}" for p in pw_list], di_list, color=colors, width=0.5)
    ax.set_xlabel("pos_weight")
    ax.set_ylabel(f"Defect IoU (thr={THR})")
    ax.set_title(f"Pos_weight Sweep (thr={THR}, bc=8, 3000 steps)")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004,
                f"{b.get_height():.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(di_list) * 1.25 if di_list else 0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "07_posweight_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 07_posweight_sweep.png")


# ── Fig 08: Training Curves (2x2 enhanced) ────────────────────
def fig_08_training_curves(history):
    """2x2 grid: loss / per-class IoU / LR / throughput."""
    print("  [08] training curves ...")
    s = history["step"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    ax_loss, ax_iou, ax_lr, ax_tput = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Top-left: Loss
    ax_loss.plot(s, history["L_total"], "-", color="#d4a017", lw=1.5, label="Train")
    ax_loss.plot(s, history["val_L_total"], "-", color="#8e44ad", lw=1.5, label="Val")
    ax_loss.set_ylabel("L_total")
    ax_loss.set_title("Loss (train vs val)")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)
    # Annotation: val <= train
    ax_loss.text(0.97, 0.95, "val ≤ train: BN eval mode",
                 transform=ax_loss.transAxes, fontsize=7, ha="right", va="top",
                 color="#8e44ad", style="italic")

    # Top-right: Val IoU per class (no mIoU line)
    ax_iou.plot(s, history["val_IoU_bg"], "--", color="#888", lw=1.2, label="bg")
    ax_iou.plot(s, history["val_IoU_normal"], "--", color="#198754", lw=1.2, label="normal")
    ax_iou.plot(s, history["val_IoU_defect"], "-o", color="#c0392b", lw=2,
                markersize=3, label="defect")
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
                    xytext=(s[peak_idx] + 1000, def_ious[peak_idx] - 0.08),
                    fontsize=7, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))

    # Bottom-left: Learning rate (log scale)
    ax_lr.plot(s, history["lr"], "-", color="#198754", lw=2)
    ax_lr.set_yscale("log")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_xlabel("Step")
    ax_lr.set_title("Learning Rate Schedule")
    ax_lr.grid(alpha=0.3)
    # Mark LR drops
    lr_arr = np.array(history["lr"])
    for i in range(1, len(lr_arr)):
        if lr_arr[i] < lr_arr[i-1] * 0.9:
            ax_lr.axvline(s[i], color="#198754", alpha=0.3, ls=":")

    # Bottom-right: Throughput (since loss components not in history)
    ax_tput.plot(s, history["samples_per_s"], "-", color="#e65100", lw=1.5)
    ax_tput.set_ylabel("Samples/s")
    ax_tput.set_xlabel("Step")
    ax_tput.set_title("Training Throughput")
    ax_tput.grid(alpha=0.3)
    ax_tput.text(0.97, 0.05, "(loss components not recorded\nin this run's history)",
                 transform=ax_tput.transAxes, fontsize=7, ha="right", va="bottom",
                 color="gray", style="italic")

    fig.suptitle("Training Dynamics — s5_long (bc=8, pw=5, 15000 steps)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "08_training_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 08_training_curves.png")


# ── Fig 09: Per-Type Performance ──────────────────────────────
def fig_09_per_type(head_cache):
    """Horizontal grouped bar: det_rate / normal FP / delta."""
    print("  [09] per-type performance ...")
    ps = per_state_from_cache(head_cache)
    normal_fp = ps.get("normal", 0.0)

    types_display = ["remesh_heavy", "displace_heavy", "bend_heavy"]
    det_rates = [ps.get(t, 0.0) for t in types_display]
    deltas = [dr - normal_fp for dr in det_rates]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(types_display))
    h = 0.25

    bars_det = ax.barh(y - h, det_rates, h, label="det_rate", color="#1a73e8")
    bars_fp = ax.barh(y, [normal_fp] * len(types_display), h,
                      label=f"normal FP ({normal_fp:.1%})", color="#ef9a9a")
    bars_delta = ax.barh(y + h, deltas, h, label="delta (det - FP)", color="#2e7d32")

    ax.set_yticks(y)
    ax.set_yticklabels([t.replace("_", " ") for t in types_display])
    ax.set_xlabel("Rate")
    ax.set_title(f"Per-Type Detection Performance (thr={THR})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # Value labels
    for bars in [bars_det, bars_fp, bars_delta]:
        for b in bars:
            w = b.get_width()
            ax.text(w + 0.005, b.get_y() + b.get_height()/2,
                    f"{w:.1%}", va="center", fontsize=7)

    # Annotation on bend delta
    bend_idx = types_display.index("bend_heavy")
    ax.text(deltas[bend_idx] + 0.02, y[bend_idx] + h,
            "≈ noise floor", fontsize=8, color="#2e7d32",
            va="center", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "09_per_type.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 09_per_type.png")


# ── Fig 10: Prediction Panel (strategic) ──────────────────────
def fig_10_pred_panel(head_cache):
    """4R x 4C: remesh success / displace success / bend miss / all-normal."""
    print("  [10] prediction panel ...")
    # Categorize cache entries by dominant defect type
    remesh_scenes = []
    displace_scenes = []
    bend_scenes = []
    normal_scenes = []

    for i, c in enumerate(head_cache):
        has_defect = (c["gt"] == 2).sum() > 100
        types_in_scene = set()
        for ins in c["meta"]["instances"]:
            if ins["is_defective"]:
                types_in_scene.add(schema.state_to_type(ins["defect_state"]))

        if not has_defect and not types_in_scene:
            # Compute FP to find clean scene
            fp_px = int(((c["pred"] == 2) & (c["gt"] == 1)).sum())
            normal_scenes.append((i, fp_px))
        elif "remesh" in types_in_scene:
            # Compute defect IoU for this scene
            gd = c["gt"] == 2
            pd = c["pred"] == 2
            inter = int((gd & pd).sum())
            union = int((gd | pd).sum())
            iou = inter / union if union > 0 else 0
            remesh_scenes.append((i, iou))
        elif "displace" in types_in_scene:
            gd = c["gt"] == 2
            pd = c["pred"] == 2
            inter = int((gd & pd).sum())
            union = int((gd | pd).sum())
            iou = inter / union if union > 0 else 0
            displace_scenes.append((i, iou))
        elif "bend" in types_in_scene:
            gd = c["gt"] == 2
            pd = c["pred"] == 2
            inter = int((gd & pd).sum())
            union = int((gd | pd).sum())
            iou = inter / union if union > 0 else 0
            bend_scenes.append((i, iou))

    # Select: best remesh, best displace, worst bend, cleanest normal
    selections = []
    row_labels = []

    if remesh_scenes:
        best = max(remesh_scenes, key=lambda x: x[1])
        selections.append(head_cache[best[0]])
        row_labels.append(f"remesh (success, IoU={best[1]:.2f})")
    if displace_scenes:
        best = max(displace_scenes, key=lambda x: x[1])
        selections.append(head_cache[best[0]])
        row_labels.append(f"displace (success, IoU={best[1]:.2f})")
    if bend_scenes:
        worst = min(bend_scenes, key=lambda x: x[1])
        selections.append(head_cache[worst[0]])
        row_labels.append(f"bend (miss, IoU={worst[1]:.2f})")
    if normal_scenes:
        cleanest = min(normal_scenes, key=lambda x: x[1])
        selections.append(head_cache[cleanest[0]])
        row_labels.append("normal (no false alarm)")

    n = len(selections)
    if n == 0:
        print("   ! No suitable scenes found for pred panel")
        return

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["RGB", "GT (3-class)", "Prediction", "P(defect) heatmap"]
    for r, c in enumerate(selections):
        axes[r, 0].imshow(c["rgb"])
        axes[r, 1].imshow(CMAP3[c["gt"]])
        axes[r, 2].imshow(CMAP3[c["pred"]])
        im = axes[r, 3].imshow(c["p_def"], cmap="jet", vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[r, 3], fraction=0.046, pad=0.04)
        for j in range(4):
            axes[r, j].axis("off")
        axes[r, 0].set_ylabel(row_labels[r], fontsize=8, rotation=0,
                              labelpad=120, va="center")

    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)

    fig.suptitle("Prediction Panel — Strategic Selection (green=normal, red=defect)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "10_pred_panel.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 10_pred_panel.png")


# ── Fig 11: FP/FN Cases ───────────────────────────────────────
def fig_11_fpfn(head_cache):
    """2R x 3C: FP case + FN case with overlays."""
    print("  [11] FP/FN cases ...")

    def make_overlay(rgb, mask, color, alpha=0.5):
        out = rgb.astype(float).copy()
        out[mask] = out[mask] * (1 - alpha) + np.array(color) * alpha
        return out.clip(0, 255).astype(np.uint8)

    # Find worst FP scene (normal scene with most false positives)
    fp_stats = []
    fn_stats = []
    for i, c in enumerate(head_cache):
        pd = c["pred"] == 2
        nm = c["gt"] == 1
        gd = c["gt"] == 2
        fp_px = int((pd & nm).sum())
        fn_px = int((gd & ~pd).sum())
        has_defect = gd.sum() > 100
        if not has_defect:
            fp_stats.append((i, fp_px))
        else:
            fn_stats.append((i, fn_px))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    col_titles = ["RGB", "GT overlay", "Pred overlay"]

    # Row 0: FP case
    if fp_stats:
        worst_fp = max(fp_stats, key=lambda x: x[1])
        c = head_cache[worst_fp[0]]
        axes[0, 0].imshow(c["rgb"])
        # GT: green on normal regions
        gt_ov = make_overlay(c["rgb"], c["gt"] == 1, [0, 200, 0], 0.3)
        axes[0, 1].imshow(gt_ov)
        # Pred: red on predicted defect (= false alarm on normal parts)
        pred_ov = make_overlay(c["rgb"], c["pred"] == 2, [220, 0, 0], 0.5)
        axes[0, 2].imshow(pred_ov)
        axes[0, 0].set_ylabel("FP case\n(normal scene)", fontsize=9)

    # Row 1: FN case (bend miss)
    if fn_stats:
        worst_fn = max(fn_stats, key=lambda x: x[1])
        c = head_cache[worst_fn[0]]
        axes[1, 0].imshow(c["rgb"])
        # GT: red on defect GT
        gt_ov = make_overlay(c["rgb"], c["gt"] == 2, [220, 0, 0], 0.4)
        axes[1, 1].imshow(gt_ov)
        # Pred: green on predicted normal (missed defect shows as green)
        pred_ov = make_overlay(c["rgb"], (c["pred"] == 1) & (c["gt"] == 2), [0, 200, 0], 0.5)
        pred_ov2 = make_overlay(pred_ov, c["pred"] == 2, [220, 0, 0], 0.4)
        axes[1, 2].imshow(pred_ov2)
        axes[1, 0].set_ylabel("FN case\n(missed defect)", fontsize=9)

    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Color legend for overlays
    fig.text(0.5, 0.01,
             "GT overlay: green = normal parts, red = defect GT  |  "
             "Pred overlay: red = predicted defect, green = predicted normal (on GT-defect pixels = miss)",
             ha="center", fontsize=8, style="italic", color="gray")
    fig.suptitle("Representative FP and FN Cases", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(OUT / "11_fp_fn_cases.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 11_fp_fn_cases.png")


# ── Fig 12: Confusion Matrix (5 rows) ─────────────────────────
def fig_12_confusion(head_cache):
    """5-row heatmap: normal/bend_light/bend_heavy/displace_heavy/remesh_heavy."""
    print("  [12] confusion matrix ...")
    row_states = ["normal", "bend_light", "bend_heavy", "displace_heavy", "remesh_heavy"]
    col_names = ["background", "normal_part", "defective_part"]
    mat = np.zeros((len(row_states), 3))

    for c in head_cache:
        for ins in c["meta"]["instances"]:
            st = ins["defect_state"]
            if st not in row_states:
                continue
            ri = row_states.index(st)
            pix = c["instance"] == ins["instance_id"]
            for cp in range(3):
                mat[ri, cp] += int(((c["pred"] == cp) & pix).sum())

    # Row-normalize
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    norm = mat / row_sums

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(col_names, rotation=15)
    ax.set_yticks(range(len(row_states)))
    ax.set_yticklabels([s.replace("_", " ") for s in row_states])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("GT State")

    for i in range(len(row_states)):
        for j in range(3):
            val = norm[i, j]
            ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=9)

    ax.set_title(f"Confusion Matrix — row-normalized (thr={THR})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "12_confusion.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 12_confusion.png")


# ── Fig 14: Bend Honesty ──────────────────────────────────────
def fig_14_bend_honesty(model):
    """3 lines across thr showing parallel descent → bend = FP spillover."""
    print("  [14] bend honesty ...")
    thrs = [0.5, 0.6, 0.7]
    bend_h_rates = []
    bend_l_rates = []
    normal_rates = []

    for t in thrs:
        cache = _build_cache(model, DEV, NCACHE, t)
        ps = per_state_from_cache(cache)
        bend_h_rates.append(ps.get("bend_heavy", 0.0))
        bend_l_rates.append(ps.get("bend_light", 0.0))
        normal_rates.append(ps.get("normal", 0.0))
        print(f"     thr={t}: bend_h={bend_h_rates[-1]:.3f} "
              f"bend_l={bend_l_rates[-1]:.3f} normal={normal_rates[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thrs, bend_h_rates, "o-", color="#c0392b", lw=2, markersize=8,
            label="bend_heavy det_rate")
    ax.plot(thrs, bend_l_rates, "s--", color="#e74c3c", lw=2, markersize=7,
            label="bend_light det_rate")
    ax.plot(thrs, normal_rates, "^-", color="#7f8c8d", lw=2, markersize=7,
            label="normal det_rate (= FP baseline)")

    ax.set_xlabel("defect_thr")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Bend Honesty: detection rate tracks FP baseline")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Annotate deltas
    for i, t in enumerate(thrs):
        delta = bend_h_rates[i] - normal_rates[i]
        ax.annotate(f"Δ={delta:.1%}", xy=(t, bend_h_rates[i]),
                    xytext=(t + 0.02, bend_h_rates[i] + 0.02),
                    fontsize=8, color="#c0392b")

    # Bottom caption
    ax.text(0.5, -0.12,
            "Bend detection rate tracks normal FP rate — "
            "no real discriminative power for bend deformation.",
            transform=ax.transAxes, fontsize=9, ha="center", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "14_bend_honesty.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 14_bend_honesty.png")


# ── Fig 15a: PR Curve (overall) ───────────────────────────────
def fig_15a_pr_curve(model):
    """Sweep sigmoid threshold, compute defect-class precision & recall."""
    print("  [15a] PR curve (overall) ...")
    cache = _build_cache(model, DEV, NCACHE, 0.5)  # thr doesn't matter for raw logits

    # Collect all defect pixels and predictions
    all_gt_def = []
    all_p_def = []
    for c in cache:
        # Only on part pixels (gt != 0)
        part_mask = c["gt"] != 0
        all_gt_def.append((c["gt"][part_mask] == 2).astype(np.float32))
        all_p_def.append(c["p_def"][part_mask])

    gt_flat = np.concatenate(all_gt_def)
    prob_flat = np.concatenate(all_p_def)

    thrs_sweep = np.linspace(0.01, 0.99, 50)
    precisions = []
    recalls = []
    for t in thrs_sweep:
        pred_pos = prob_flat >= t
        tp = int((pred_pos & (gt_flat == 1)).sum())
        fp = int((pred_pos & (gt_flat == 0)).sum())
        fn = int((~pred_pos & (gt_flat == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    # AUC (trapezoidal)
    sorted_idx = np.argsort(recalls)
    r_sorted = np.array(recalls)[sorted_idx]
    p_sorted = np.array(precisions)[sorted_idx]
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x compat
    auc = float(_trapz(p_sorted, r_sorted))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(recalls, precisions, "-", color="#1a73e8", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Precision-Recall Curve (defect class, AUC={auc:.3f})")
    ax.grid(alpha=0.3)

    # Mark thr=0.7 operating point
    idx_07 = np.argmin(np.abs(thrs_sweep - 0.7))
    ax.plot(recalls[idx_07], precisions[idx_07], "o", color="#c0392b", markersize=10)
    ax.annotate(f"thr=0.7\n(P={precisions[idx_07]:.2f}, R={recalls[idx_07]:.2f})",
                xy=(recalls[idx_07], precisions[idx_07]),
                xytext=(recalls[idx_07] + 0.08, precisions[idx_07] - 0.1),
                fontsize=8, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b"))

    # Mark thr=0.5
    idx_05 = np.argmin(np.abs(thrs_sweep - 0.5))
    ax.plot(recalls[idx_05], precisions[idx_05], "s", color="#7f8c8d", markersize=8)
    ax.annotate(f"thr=0.5", xy=(recalls[idx_05], precisions[idx_05]),
                xytext=(recalls[idx_05] + 0.05, precisions[idx_05] + 0.05),
                fontsize=7, color="#7f8c8d")

    fig.tight_layout()
    fig.savefig(OUT / "15a_pr_curve.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 15a_pr_curve.png")


# ── Fig 15b: PR Curve (per-type, scene-scoped) ───────────────
def fig_15b_pr_curve_pertype(model):
    """Per-type PR with scene-scoped negatives.

    For type T: only count normal-instance pixels from scenes that CONTAIN
    type-T defective instances.  This avoids class imbalance where global
    normal pixels overwhelm per-type positives.
    """
    print("  [15b] PR curve (per-type, scene-scoped) ...")
    cache = _build_cache(model, DEV, NCACHE, 0.5)

    # Identify which types each scene contains
    type_gt = {"remesh": [], "displace": [], "bend": []}
    type_prob = {"remesh": [], "displace": [], "bend": []}

    for c in cache:
        # Which defect types are in this scene?
        scene_types = set()
        for ins in c["meta"]["instances"]:
            if ins["is_defective"]:
                scene_types.add(schema.state_to_type(ins["defect_state"]))

        if not scene_types:
            continue  # all-normal scene → skip entirely

        # Collect positives (defective instance pixels)
        for ins in c["meta"]["instances"]:
            if not ins["is_defective"]:
                continue
            t = schema.state_to_type(ins["defect_state"])
            if t not in type_gt:
                continue
            pix = c["instance"] == ins["instance_id"]
            if pix.sum() < 50:
                continue
            type_gt[t].append(np.ones(int(pix.sum()), dtype=np.float32))
            type_prob[t].append(c["p_def"][pix])

        # Collect negatives: normal instance pixels, only for types present
        for ins in c["meta"]["instances"]:
            if ins["is_defective"]:
                continue
            pix = c["instance"] == ins["instance_id"]
            ps = int(pix.sum())
            if ps < 50:
                continue
            neg_prob = c["p_def"][pix]
            for t in scene_types:
                if t in type_gt:
                    type_gt[t].append(np.zeros(ps, dtype=np.float32))
                    type_prob[t].append(neg_prob)

    colors = {"remesh": "#1a73e8", "displace": "#e65100", "bend": "#c0392b"}
    thrs_sweep = np.linspace(0.01, 0.99, 50)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for t in ["remesh", "displace", "bend"]:
        if not type_gt[t]:
            continue
        gt_flat = np.concatenate(type_gt[t])
        prob_flat = np.concatenate(type_prob[t])
        precs, recs = [], []
        for thr in thrs_sweep:
            pred_pos = prob_flat >= thr
            tp = int((pred_pos & (gt_flat == 1)).sum())
            fp = int((pred_pos & (gt_flat == 0)).sum())
            fn = int((~pred_pos & (gt_flat == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precs.append(p)
            recs.append(r)
        ax.plot(recs, precs, "-", color=colors[t], lw=2, label=t)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Per-Type Precision-Recall (scene-scoped negatives)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.02, "Negatives = normal parts from same scenes only",
            transform=ax.transAxes, fontsize=7, color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "15b_pr_curve_pertype.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("   -> 15b_pr_curve_pertype.png")


# ── Main ──────────────────────────────────────────────────────
def main():
    print(f"=== Stage 5 Figure Generation ===")
    print(f"device={DEV}  thr={THR}  output={OUT}\n")

    # Fig 02: reuse existing diff-threshold figure
    src02 = REPO / "docs" / "figures" / "route_a_diff_threshold.png"
    if src02.exists():
        shutil.copy(src02, OUT / "02_diff_threshold.png")
        print("  [02] diff_threshold (copied existing)")
    else:
        print("  [02] ! source not found:", src02)

    # Fig 00: Dataset showcase
    fig_00_dataset_showcase()

    # Fig 01: Supervision T/W
    fig_01_supervision_TW()

    # Load headline model (s5_long) + cache for multiple figures
    print("\n  [loading] headline model: s5_long ...")
    m_head, tc_head = load_model("s5_long")
    head_cache = _build_cache(m_head, DEV, NCACHE, THR)
    sc = cache_scores(head_cache)
    print(f"  headline: defIoU={sc['defIoU']:.3f} normalFP={sc['normalFP']:.3f}\n")

    # Fig 03: KPI vs S3
    fig_03_kpi_vs_s3(head_cache)

    # Fig 08: Training curves
    hist = json.loads((RUNS / "s5_long" / "history.json").read_text(encoding="utf-8"))
    fig_08_training_curves(hist)

    # Fig 09: Per-type performance
    fig_09_per_type(head_cache)

    # Fig 10: Prediction panel
    fig_10_pred_panel(head_cache)

    # Fig 11: FP/FN cases
    fig_11_fpfn(head_cache)

    # Fig 12: Confusion matrix
    fig_12_confusion(head_cache)

    # Sweep figures (heavy: reload models per run)
    print("\n  [sweeps] ...")
    fig_0405_capacity_sweep()
    fig_06_thr_sweep(m_head)
    fig_07_posweight_sweep()

    # Fig 14: Bend honesty (reuses headline model, builds 3 caches)
    fig_14_bend_honesty(m_head)

    # Fig 15a/b: PR curves
    fig_15a_pr_curve(m_head)
    fig_15b_pr_curve_pertype(m_head)

    print(f"\n=== Done! {len(list(OUT.glob('*.png')))} figures in {OUT} ===")


if __name__ == "__main__":
    main()
