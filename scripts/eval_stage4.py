"""
Stage 4 — Eval script

報告:
1. 3-class IoU (bg / normal_part / defective_part)  ← 跟 Stage 3 公平比較
2. Per-state binary defect IoU (bend_l/h, disp_l/h, remesh_l/h) ← 跟 Stage 3 對齊
3. Head B 7×7 confusion matrix
4. Head C 4×4 confusion matrix
5. 黑底 ablation (3-class IoU)
6. Figures：A normal/defect diff, B confidence, C per-state IoU box, F1/F2 by-state confusion

執行：
    & "C:\\Users\\Harrison\\miniconda3\\envs\\dl_final\\python.exe" -u scripts/eval_stage4.py
"""

# pylint: disable=wrong-import-position
import os, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_stage4 import (
    DefectSegNetV4, SceneDatasetMulti,
    STATE_TO_IDX, IDX_TO_STATE, TYPE_TO_IDX, IDX_TO_TYPE, N_STATES, N_TYPES,
    IGNORE_IDX, BASE_DIR, SCENES_DIR, SEED,
)

SAVE_DIR     = os.path.join(BASE_DIR, "output")
FIG_DIR      = os.path.join(BASE_DIR, "docs", "figures", "stage4")
BLACK_DIR    = os.path.join(BASE_DIR, "output", "scenes_black")
MODEL_PATH   = os.path.join(SAVE_DIR, "stage4_best_model.pt")

os.makedirs(FIG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_split():
    all_ids  = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(SCENES_DIR, "[0-9]*")))
    rng      = np.random.RandomState(SEED)
    shuffled = list(all_ids); rng.shuffle(shuffled)
    n_train = int(0.8 * len(shuffled))
    n_val   = int(0.1 * len(shuffled))
    return shuffled[:n_train], shuffled[n_train:n_train+n_val], shuffled[n_train+n_val:]


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = DefectSegNetV4(base_c=32).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def run_inference(model, scene_dirs):
    """Yields (sid, rgb, part_pred, state_pred, type_pred, gt_state)"""
    for sd in scene_dirs:
        sid = int(os.path.basename(sd))
        rgb = np.array(Image.open(os.path.join(sd, "rgb.png")).convert("RGB"))
        inst = np.array(Image.open(os.path.join(sd, "instance_mask.png")))
        with open(os.path.join(sd, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        gt_state = np.full(inst.shape, IGNORE_IDX, dtype=np.int64)
        for ins in meta["instances"]:
            pix = (inst == ins["instance_id"])
            if pix.any():
                gt_state[pix] = STATE_TO_IDX[ins["defect_state"]]
        rgb_t = torch.from_numpy(rgb).permute(2,0,1).float()/255.0
        rgb_t = (rgb_t - 0.5)/0.5
        rgb_t = rgb_t.unsqueeze(0).to(device)
        p_l, s_l, t_l = model(rgb_t)
        yield (sid, rgb,
               p_l.argmax(1)[0].cpu().numpy(),
               s_l.argmax(1)[0].cpu().numpy(),
               t_l.argmax(1)[0].cpu().numpy(),
               gt_state, meta)


def eval_3class(model, scene_root, test_ids):
    """3-class IoU schema 跟 Stage 3 一致"""
    inter = np.zeros(3); union = np.zeros(3); correct = total = 0
    scene_dirs = [os.path.join(scene_root, f"{sid:05d}") for sid in test_ids]
    for sid, rgb, pp, sp, tp, gt_state, meta in run_inference(model, scene_dirs):
        gt = np.where(gt_state < 0, 0, np.where(gt_state == 0, 1, 2))
        pred = np.zeros_like(pp)
        is_part = pp == 1
        is_def  = is_part & (sp > 0)
        pred[is_part] = 1
        pred[is_def]  = 2
        for c in range(3):
            p = (pred == c); t = (gt == c)
            inter[c] += int((p & t).sum()); union[c] += int((p | t).sum())
        correct += int((pred == gt).sum()); total += int(gt.size)
    ious = [inter[c]/union[c] if union[c]>0 else float("nan") for c in range(3)]
    return {"pixel_acc": correct/total, "mIoU": float(np.nanmean(ious)), "IoU_per_class": ious}


def eval_per_state_binary(model, scene_root, test_ids):
    """每個 defect state，計算「該 state 像素被預測為 defect」的 binary IoU
    跟 Stage 3 stage3_results.md §3 的 schema 對齊。
    """
    # 對每個 state idx，累積 inter / union for binary defect detection on that state's pixels only
    inter = np.zeros(N_STATES); union = np.zeros(N_STATES); n_pix = np.zeros(N_STATES)
    scene_dirs = [os.path.join(scene_root, f"{sid:05d}") for sid in test_ids]
    for sid, rgb, pp, sp, tp, gt_state, meta in run_inference(model, scene_dirs):
        is_part_pred = pp == 1
        is_def_pred  = is_part_pred & (sp > 0)         # binary defect prediction
        for state_idx in range(N_STATES):
            gt_pix = (gt_state == state_idx)
            if not gt_pix.any():
                continue
            gt_is_def = (state_idx > 0)
            # On these pixels: TP = pred is defect & gt is defect; etc.
            pred_on = is_def_pred[gt_pix]
            n_pix[state_idx] += int(gt_pix.sum())
            if gt_is_def:
                # 對 defect state：TP = pred=def & gt_pix; FN = pred=normal & gt_pix
                # union = TP + FN + FP (但 FP outside gt_pix 不算這個 state)
                # 用 instance-level 平均更接近 Stage 3 報告，但 pixel-level 也 OK
                inter[state_idx] += int(pred_on.sum())
                union[state_idx] += int(gt_pix.sum())  # = TP + FN
                # Add FP (pred def but not this state's pixels) → 不加，避免被其他 defect 拉低
            else:
                # normal state 上的 IoU = 1 - FP rate 沒太有意義；直接記 pred_on 比率
                inter[state_idx] += int((~pred_on).sum())  # correct rejection
                union[state_idx] += int(gt_pix.sum())
    out = {}
    for c in range(N_STATES):
        if n_pix[c] == 0:
            out[IDX_TO_STATE[c]] = float("nan")
        else:
            # 對 defect: inter/union = recall (pred def | gt this state)
            # 對 normal: inter/union = specificity (pred normal | gt normal)
            out[IDX_TO_STATE[c]] = float(inter[c] / union[c]) if union[c] > 0 else 0.0
    return out, {IDX_TO_STATE[c]: int(n_pix[c]) for c in range(N_STATES)}


def eval_confusion_state(model, scene_root, test_ids):
    """7×7 confusion matrix (head B prediction vs gt state)"""
    cm = np.zeros((N_STATES, N_STATES), dtype=np.int64)
    scene_dirs = [os.path.join(scene_root, f"{sid:05d}") for sid in test_ids]
    for sid, rgb, pp, sp, tp, gt_state, meta in run_inference(model, scene_dirs):
        valid = gt_state >= 0
        for g in range(N_STATES):
            mask_g = (gt_state == g) & valid
            if not mask_g.any(): continue
            for p in range(N_STATES):
                cm[g, p] += int(((sp == p) & mask_g).sum())
    return cm


def eval_confusion_type(model, scene_root, test_ids):
    """4×4 confusion matrix (head C prediction vs gt type)"""
    cm = np.zeros((N_TYPES, N_TYPES), dtype=np.int64)
    scene_dirs = [os.path.join(scene_root, f"{sid:05d}") for sid in test_ids]
    for sid, rgb, pp, sp, tp, gt_state, meta in run_inference(model, scene_dirs):
        # gt_type from gt_state
        gt_type = np.full_like(gt_state, IGNORE_IDX)
        for s_idx in range(N_STATES):
            sname = IDX_TO_STATE[s_idx]
            t_idx = TYPE_TO_IDX["normal" if sname == "normal" else sname.split("_")[0]]
            gt_type[gt_state == s_idx] = t_idx
        valid = gt_type >= 0
        for g in range(N_TYPES):
            mask_g = (gt_type == g) & valid
            if not mask_g.any(): continue
            for p in range(N_TYPES):
                cm[g, p] += int(((tp == p) & mask_g).sum())
    return cm


def plot_confusion(cm, labels, title, out_path, fmt=".2f"):
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(6, 0.7*len(labels)+3), max(5, 0.6*len(labels)+2)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_per_state_bar(stage4_dict, out_path, stage3_dict=None):
    states = ["bend_light", "bend_heavy",
              "displace_light", "displace_heavy",
              "remesh_light", "remesh_heavy"]
    s4 = [stage4_dict.get(s, 0.0) for s in states]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(states))
    if stage3_dict:
        s3 = [stage3_dict.get(s, 0.0) for s in states]
        w = 0.4
        ax.bar(x - w/2, s3, w, label="Stage 3", color="#7eb0d5")
        ax.bar(x + w/2, s4, w, label="Stage 4", color="#fd7f6f")
        ax.legend()
    else:
        ax.bar(x, s4, color="#fd7f6f")
    ax.set_xticks(x); ax.set_xticklabels(states, rotation=20, ha="right")
    ax.set_ylabel("Per-state defect recall (binary)")
    ax.set_title("Per-state defect detection — Stage 3 vs Stage 4")
    ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(s4):
        ax.text(i + (0.2 if stage3_dict else 0), v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_ablation_bar(textured, blackbg, out_path):
    metrics = ["pixel_acc", "mIoU", "bg_IoU", "normal_IoU", "defect_IoU"]
    t = [textured["pixel_acc"], textured["mIoU"], *textured["IoU_per_class"]]
    b = [blackbg["pixel_acc"],  blackbg["mIoU"],  *blackbg["IoU_per_class"]]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.4
    ax.bar(x - w/2, t, w, label="Textured DR", color="#7eb0d5")
    ax.bar(x + w/2, b, w, label="Black bg",    color="#bd7ebe")
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylim(0, 1.0); ax.set_title("Ablation: Textured DR vs Black background (Stage 4)")
    for i, (vt, vb) in enumerate(zip(t, b)):
        ax.text(i - w/2, vt + 0.01, f"{vt:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, vb + 0.01, f"{vb:.3f}", ha="center", fontsize=8)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_training_curves(history_path, out_path):
    with open(history_path, encoding="utf-8") as f:
        h = json.load(f)
    epochs = range(1, len(h["L_total"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for k, color in [("L_total", "k"), ("L_A", "#888"),
                     ("L_B_ce", "#fd7f6f"), ("L_B_dice", "#fd7f6f"),
                     ("L_C_ce", "#7eb0d5"), ("L_C_dice", "#7eb0d5")]:
        style = "--" if "dice" in k else "-"
        ax.plot(epochs, h[k], style, label=k, color=color, alpha=0.85)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(fontsize=8)
    ax.set_title("Training losses (per head + total)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, h["val_mIoU"], "k-", label="val mIoU", linewidth=2)
    ax.plot(epochs, h["val_IoU_defect"], color="#c0392b", label="val defect IoU")
    ax.plot(epochs, h["val_IoU_normal"], color="#198754", label="val normal IoU")
    ax.plot(epochs, h["val_IoU_bg"],     color="#7eb0d5", label="val bg IoU")
    ax2 = ax.twinx()
    ax2.plot(epochs, h["lr"], "r--", alpha=0.5, label="lr")
    ax2.set_ylabel("lr (dashed red)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("IoU"); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Validation IoU + lr schedule")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    print(f"Device: {device}")
    model, ckpt = load_model()
    train_ids, val_ids, test_ids = get_split()
    print(f"Test set: {len(test_ids)} scenes")

    # 1) Test 3-class (跟 Stage 3 公平比較)
    m_textured = eval_3class(model, SCENES_DIR, test_ids)
    print(f"\n[Textured DR] pixel_acc={m_textured['pixel_acc']*100:.2f}% "
          f"mIoU={m_textured['mIoU']:.3f} "
          f"bg={m_textured['IoU_per_class'][0]:.3f} "
          f"normal={m_textured['IoU_per_class'][1]:.3f} "
          f"defect={m_textured['IoU_per_class'][2]:.3f}")

    # 2) Per-state binary defect
    per_state_bin, n_pix = eval_per_state_binary(model, SCENES_DIR, test_ids)
    print(f"\nPer-state binary defect recall (跟 Stage 3 表 §3 schema):")
    for s in ["normal", "bend_light", "bend_heavy",
              "displace_light", "displace_heavy",
              "remesh_light", "remesh_heavy"]:
        print(f"  {s:18s} = {per_state_bin[s]:.3f}   (n_pix={n_pix[s]})")

    # 3) Confusion matrices
    cm_state = eval_confusion_state(model, SCENES_DIR, test_ids)
    cm_type  = eval_confusion_type (model, SCENES_DIR, test_ids)
    plot_confusion(cm_state,
                   [IDX_TO_STATE[i] for i in range(N_STATES)],
                   "Head B — 7-way defect state confusion (row-normalized)",
                   os.path.join(FIG_DIR, "B_state_confusion.png"))
    plot_confusion(cm_type,
                   [IDX_TO_TYPE[i] for i in range(N_TYPES)],
                   "Head C — 4-way defect type confusion (row-normalized)",
                   os.path.join(FIG_DIR, "C_type_confusion.png"))

    # 4) Black bg ablation
    m_black = eval_3class(model, BLACK_DIR, test_ids)
    print(f"\n[Black bg]    pixel_acc={m_black['pixel_acc']*100:.2f}% "
          f"mIoU={m_black['mIoU']:.3f} defect={m_black['IoU_per_class'][2]:.3f}")
    plot_ablation_bar(m_textured, m_black,
                      os.path.join(FIG_DIR, "ablation_textured_vs_black.png"))

    # 5) Per-state bar (跟 Stage 3 對比) — 寫死 Stage 3 數字
    stage3_per_state = {
        "bend_light": 0.034, "bend_heavy": 0.047,
        "displace_light": 0.780, "displace_heavy": 0.932,
        "remesh_light": 0.324, "remesh_heavy": 0.523,
    }  # 用 Stage 3 報告的「像素中被預測為 defect 的比例」(detection rate)
    plot_per_state_bar(per_state_bin, os.path.join(FIG_DIR, "per_state_defect_bar.png"),
                       stage3_dict=stage3_per_state)

    # 6) Training curves
    plot_training_curves(os.path.join(SAVE_DIR, "stage4_history.json"),
                         os.path.join(FIG_DIR, "training_curves.png"))

    # 7) Save summary
    summary = {
        "test_3class": m_textured,
        "test_3class_blackbg": m_black,
        "per_state_binary_defect_recall": per_state_bin,
        "per_state_n_pixels": n_pix,
        "confusion_state_7x7": cm_state.tolist(),
        "confusion_type_4x4":  cm_type.tolist(),
        "labels_state": [IDX_TO_STATE[i] for i in range(N_STATES)],
        "labels_type":  [IDX_TO_TYPE[i]  for i in range(N_TYPES)],
        "checkpoint_best_val_miou": ckpt["best_val_miou"],
    }
    with open(os.path.join(SAVE_DIR, "stage4_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary → output/stage4_eval_summary.json")
    print(f"Figures → docs/figures/stage4/")


if __name__ == "__main__":
    main()
