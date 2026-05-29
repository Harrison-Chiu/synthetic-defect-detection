"""
Stage 4 — 三組對照 evaluation：
  (1) Stage 3 baseline      (來自 stage3_results.md 紀錄, 寫死)
  (2) Stage 3 arch + S4 data (output/stage4_singlehead_best.pt) — 隔離資料效應
  (3) Stage 4 multi-head     (output/stage4_best_model.pt)

產生：
  output/stage4_eval_compare.json
  docs/figures/stage4/compare_per_state_bar.png
  docs/figures/stage4/compare_overall_bar.png
"""

import os, json, glob, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_stage4 import (DefectSegNetV4, STATE_TO_IDX, IDX_TO_STATE, IGNORE_IDX,
                          N_STATES, BASE_DIR, SCENES_DIR, SEED)
from train_stage3 import DefectSegNetV2

SAVE_DIR  = os.path.join(BASE_DIR, "output")
FIG_DIR   = os.path.join(BASE_DIR, "docs", "figures", "stage4")
BLACK_DIR = os.path.join(BASE_DIR, "output", "scenes_black")
MULTI_PT  = os.path.join(SAVE_DIR, "stage4_best_model.pt")
SINGLE_PT = os.path.join(SAVE_DIR, "stage4_singlehead_best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_split():
    all_ids  = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(SCENES_DIR, "[0-9]*")))
    rng      = np.random.RandomState(SEED)
    shuffled = list(all_ids); rng.shuffle(shuffled)
    n_train = int(0.8 * len(shuffled)); n_val = int(0.1 * len(shuffled))
    return shuffled[n_train+n_val:]


def load_scene_for_eval(scene_dir):
    rgb  = np.array(Image.open(os.path.join(scene_dir, "rgb.png")).convert("RGB"))
    inst = np.array(Image.open(os.path.join(scene_dir, "instance_mask.png")))
    with open(os.path.join(scene_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    gt_state = np.full(inst.shape, IGNORE_IDX, dtype=np.int64)
    for ins in meta["instances"]:
        pix = inst == ins["instance_id"]
        if pix.any():
            gt_state[pix] = STATE_TO_IDX[ins["defect_state"]]
    rgb_t = torch.from_numpy(rgb).permute(2,0,1).float()/255.0
    rgb_t = ((rgb_t - 0.5)/0.5).unsqueeze(0).to(device)
    return rgb_t, gt_state


@torch.no_grad()
def predict_defect_mask(model, model_kind, rgb_t):
    """回傳 (part_pred [H,W] 0/1, defect_pred [H,W] 0/1)"""
    if model_kind == "single":
        p_l, d_l = model(rgb_t)
        part = p_l.argmax(1)[0].cpu().numpy()
        d_prob = torch.sigmoid(d_l).squeeze(1)[0].cpu().numpy()
        defect = ((part == 1) & (d_prob > 0.5)).astype(np.int64)
        return part, defect
    else:  # multi (Stage 4 three-head)
        p_l, s_l, _ = model(rgb_t)
        part = p_l.argmax(1)[0].cpu().numpy()
        state = s_l.argmax(1)[0].cpu().numpy()
        defect = ((part == 1) & (state > 0)).astype(np.int64)
        return part, defect


def eval_3class(model, model_kind, scene_root, test_ids):
    inter = np.zeros(3); union = np.zeros(3); correct = total = 0
    for sid in test_ids:
        rgb_t, gt_state = load_scene_for_eval(os.path.join(scene_root, f"{sid:05d}"))
        part, defect = predict_defect_mask(model, model_kind, rgb_t)
        pred = np.zeros_like(part)
        pred[part == 1] = 1
        pred[defect == 1] = 2
        gt = np.where(gt_state < 0, 0, np.where(gt_state == 0, 1, 2))
        for c in range(3):
            p = (pred == c); t = (gt == c)
            inter[c] += int((p & t).sum()); union[c] += int((p | t).sum())
        correct += int((pred == gt).sum()); total += int(gt.size)
    ious = [inter[c]/union[c] if union[c]>0 else float("nan") for c in range(3)]
    return {"pixel_acc": correct/total, "mIoU": float(np.nanmean(ious)),
            "IoU_per_class": ious}


def eval_per_state_recall(model, model_kind, scene_root, test_ids):
    """每個 state 的 binary defect recall on its own pixels"""
    n_correct = np.zeros(N_STATES); n_total = np.zeros(N_STATES)
    for sid in test_ids:
        rgb_t, gt_state = load_scene_for_eval(os.path.join(scene_root, f"{sid:05d}"))
        _, defect = predict_defect_mask(model, model_kind, rgb_t)
        for c in range(N_STATES):
            mask_c = (gt_state == c)
            if not mask_c.any(): continue
            n_total[c] += int(mask_c.sum())
            if c == 0:   # normal — recall of "not predicted as defect"
                n_correct[c] += int(((defect == 0) & mask_c).sum())
            else:
                n_correct[c] += int(((defect == 1) & mask_c).sum())
    return {IDX_TO_STATE[c]: float(n_correct[c]/n_total[c]) if n_total[c] > 0 else 0.0
            for c in range(N_STATES)}


def main():
    test_ids = get_split()
    print(f"Test: {len(test_ids)} scenes")

    # Load models
    multi = DefectSegNetV4(base_c=32).to(device)
    multi.load_state_dict(torch.load(MULTI_PT, map_location=device, weights_only=False)["state_dict"])
    multi.eval()

    single = DefectSegNetV2(base_c=32).to(device)
    single.load_state_dict(torch.load(SINGLE_PT, map_location=device, weights_only=False)["state_dict"])
    single.eval()

    print("\n--- Eval: Stage 3 arch (single-head) + S4 data ---")
    m_single = eval_3class(single, "single", SCENES_DIR, test_ids)
    s_single = eval_per_state_recall(single, "single", SCENES_DIR, test_ids)
    b_single = eval_3class(single, "single", BLACK_DIR, test_ids)
    print(f"  mIoU={m_single['mIoU']:.3f} defect={m_single['IoU_per_class'][2]:.3f}  "
          f"black mIoU={b_single['mIoU']:.3f}")
    for s in s_single: print(f"  {s:18s} = {s_single[s]:.3f}")

    print("\n--- Eval: Stage 4 multi-head (3-head) + S4 data ---")
    m_multi = eval_3class(multi, "multi", SCENES_DIR, test_ids)
    s_multi = eval_per_state_recall(multi, "multi", SCENES_DIR, test_ids)
    b_multi = eval_3class(multi, "multi", BLACK_DIR, test_ids)
    print(f"  mIoU={m_multi['mIoU']:.3f} defect={m_multi['IoU_per_class'][2]:.3f}  "
          f"black mIoU={b_multi['mIoU']:.3f}")
    for s in s_multi: print(f"  {s:18s} = {s_multi[s]:.3f}")

    # 寫死 Stage 3 baseline 數字（來自 stage3_results.md）
    stage3_baseline = {
        "test_3class": {"pixel_acc": 0.941, "mIoU": 0.712,
                        "IoU_per_class": [0.983, 0.790, 0.362]},
        "test_3class_blackbg": {"pixel_acc": 0.928, "mIoU": 0.678,
                                "IoU_per_class": [0.979, 0.737, 0.319]},
        # 註：Stage 3 報告 §3 用「像素中被預測為 defect 的比例」，schema 一致
        "per_state_recall": {
            "normal": float("nan"),
            "bend_light": 0.034, "bend_heavy": 0.047,
            "displace_light": 0.780, "displace_heavy": 0.932,
            "remesh_light": 0.324, "remesh_heavy": 0.523,
        },
    }

    compare = {
        "stage3_baseline_on_s3_data": stage3_baseline,
        "stage3_arch_on_s4_data": {
            "test_3class": m_single, "test_3class_blackbg": b_single,
            "per_state_recall": s_single,
        },
        "stage4_multihead_on_s4_data": {
            "test_3class": m_multi, "test_3class_blackbg": b_multi,
            "per_state_recall": s_multi,
        },
    }
    with open(os.path.join(SAVE_DIR, "stage4_eval_compare.json"), "w", encoding="utf-8") as f:
        json.dump(compare, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → output/stage4_eval_compare.json")

    # Plot per-state bar
    states = ["bend_light", "bend_heavy", "displace_light", "displace_heavy",
              "remesh_light", "remesh_heavy"]
    b = stage3_baseline["per_state_recall"]
    s = s_single
    m = s_multi
    vb = [b[k] for k in states]; vs = [s[k] for k in states]; vm = [m[k] for k in states]
    x = np.arange(len(states)); w = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w, vb, w, label="S3 baseline (S3 arch + S3 data)", color="#7eb0d5")
    ax.bar(x,     vs, w, label="S3 arch + S4 data (isolated data fix)", color="#198754")
    ax.bar(x + w, vm, w, label="S4 multi-head (S4 arch + S4 data)",     color="#fd7f6f")
    ax.set_xticks(x); ax.set_xticklabels(states, rotation=15)
    ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Defect recall on this state's pixels")
    ax.set_title("Per-state defect detection — 3 experiments compared")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "compare_per_state_bar.png"), dpi=120)
    plt.close()

    # Plot overall bar
    metrics = ["mIoU (textured)", "defect IoU (textured)", "mIoU (black)", "defect IoU (black)"]
    vb = [stage3_baseline["test_3class"]["mIoU"], stage3_baseline["test_3class"]["IoU_per_class"][2],
          stage3_baseline["test_3class_blackbg"]["mIoU"], stage3_baseline["test_3class_blackbg"]["IoU_per_class"][2]]
    vs = [m_single["mIoU"], m_single["IoU_per_class"][2],
          b_single["mIoU"], b_single["IoU_per_class"][2]]
    vm = [m_multi["mIoU"],  m_multi["IoU_per_class"][2],
          b_multi["mIoU"],  b_multi["IoU_per_class"][2]]
    x = np.arange(len(metrics)); w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w, vb, w, label="S3 baseline",        color="#7eb0d5")
    ax.bar(x,     vs, w, label="S3 arch + S4 data",  color="#198754")
    ax.bar(x + w, vm, w, label="S4 multi-head",      color="#fd7f6f")
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=10)
    ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("IoU")
    ax.set_title("Overall comparison — 3 experiments")
    for i in range(len(metrics)):
        for off, v in zip([-w, 0, +w], [vb[i], vs[i], vm[i]]):
            ax.text(i + off, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "compare_overall_bar.png"), dpi=120)
    plt.close()

    print(f"Figures → docs/figures/stage4/compare_*.png")


if __name__ == "__main__":
    main()
