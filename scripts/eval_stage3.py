"""
Stage 3 — Two-head 模型評估 + 6 張 debug 視覺化

讀 output/stage3_best_model.pt，eval test split（textured + black-bg ablation）
產出：
    docs/figures/stage3/
      A_normal_vs_defect_diff.png      ← 同 pose normal vs defect 對照
      B_confidence_heatmap.png         ← 模型 part_prob / defect_prob heatmap
      C_per_state_iou_box.png          ← 每種 defect state 的 per-instance IoU 分布
      D_fp_fn_overlay.png              ← 4 個代表案例的 FP/FN overlay
      E_epoch_snapshots.png            ← 訓練過程預測演化（從 stage3_epoch_snapshots/ 組合）
      F_by_state_confusion.png         ← 7 defect_state × 3 pred class confusion matrix
      ablation_textured_vs_black.png   ← textured bg vs black bg 的 defect IoU 對比
      training_curves.png              ← loss / per-class IoU 沿 epoch
      summary.json                     ← 數據摘要

執行：
    & "C:\\Users\\Harrison\\miniconda3\\envs\\dl_final\\python.exe" -u scripts/eval_stage3.py
"""

import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
BASE_DIR     = r"D:\Harrison\中山\大四下\深度學習期末報告"
SCENES_DIR   = os.path.join(BASE_DIR, "output", "scenes")
SCENES_BLACK = os.path.join(BASE_DIR, "output", "scenes_black")
PARTS_DIR    = os.path.join(BASE_DIR, "output", "parts_stage2")
MODEL_PT     = os.path.join(BASE_DIR, "output", "stage3_best_model.pt")
HIST_JSON    = os.path.join(BASE_DIR, "output", "stage3_history.json")
SNAP_DIR     = os.path.join(BASE_DIR, "output", "stage3_epoch_snapshots")
FIG_DIR      = os.path.join(BASE_DIR, "docs", "figures", "stage3")
os.makedirs(FIG_DIR, exist_ok=True)

NUM_CLASSES = 3
CLASS_NAMES = ["background", "normal_part", "defective_part"]
SHORT_NAMES = ["bg", "normal", "defect"]
CMAP3 = np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]], dtype=np.uint8)
DEFECT_STATES = ["normal", "bend_light", "bend_heavy",
                 "displace_light", "displace_heavy",
                 "remesh_light", "remesh_heavy"]
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[eval] device={device}")


# ─── Model（同 train_stage3.py）───────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x

class DefectSegNetV2(nn.Module):
    def __init__(self, base_c=32):
        super().__init__()
        c = base_c
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c*2)
        self.enc3 = ConvBlock(c*2, c*4)
        self.enc4 = ConvBlock(c*4, c*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c*8, c*8)
        self.up4  = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)
        self.dec4 = ConvBlock(c*4 + c*8, c*4)
        self.up3  = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec3 = ConvBlock(c*2 + c*4, c*2)
        self.up2  = nn.ConvTranspose2d(c*2, c, 2, stride=2)
        self.dec2 = ConvBlock(c + c*2, c)
        self.up1  = nn.ConvTranspose2d(c, c//2, 2, stride=2)
        self.dec1 = ConvBlock(c//2 + c, c//2)
        self.head_part   = nn.Conv2d(c//2, 2, 1)
        self.head_defect = nn.Conv2d(c//2, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head_part(d1), self.head_defect(d1)


def infer(model, rgb):
    pl, dl = model(rgb)
    part_prob   = F.softmax(pl, dim=1)[:, 1]            # P(part)
    defect_prob = torch.sigmoid(dl).squeeze(1)
    part_pred = pl.argmax(dim=1)
    out = torch.zeros_like(part_pred)
    is_part   = part_pred == 1
    is_defect = is_part & (defect_prob > 0.5)
    out[is_part]   = 1
    out[is_defect] = 2
    return out, part_prob, defect_prob


# ─── Dataset ─────────────────────────────────
class SceneDataset(Dataset):
    def __init__(self, root, ids):
        self.root = root; self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        sid = self.ids[idx]
        rgb = np.array(Image.open(os.path.join(self.root, f"{sid:05d}", "rgb.png")).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.root, f"{sid:05d}", "semantic_mask.png")))
        r = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        r = (r - 0.5) / 0.5
        return r, torch.from_numpy(mask).long(), sid


# ─── Load model + reconstruct split ──────────
ckpt = torch.load(MODEL_PT, map_location=device, weights_only=False)
model = DefectSegNetV2(base_c=32).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"[eval] loaded model, best_val_miou={ckpt.get('best_val_miou', 'n/a'):.4f}")

all_ids = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(SCENES_DIR, "[0-9]*")))
rng = np.random.RandomState(SEED); shuffled = list(all_ids); rng.shuffle(shuffled)
n_train = int(0.8 * len(shuffled)); n_val = int(0.1 * len(shuffled))
test_ids = shuffled[n_train + n_val:]
print(f"[eval] test set: {len(test_ids)} scenes")


# ─── Inference cache（textured 與 black-bg 各跑一次）───
def run_inference(root):
    cache = {}
    loader = DataLoader(SceneDataset(root, test_ids), batch_size=8, shuffle=False, num_workers=0)
    with torch.no_grad():
        for rgb, mask, sids in loader:
            pred, ppart, pdef = infer(model, rgb.to(device))
            for b in range(rgb.shape[0]):
                cache[int(sids[b])] = {
                    "pred":   pred[b].cpu().numpy().astype(np.uint8),
                    "p_part": ppart[b].cpu().numpy(),
                    "p_def":  pdef[b].cpu().numpy(),
                    "gt":     mask[b].numpy().astype(np.uint8),
                    "rgb":    ((rgb[b]*0.5+0.5).clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8),
                }
    return cache

print("[eval] inference on textured bg ...")
cache_tex = run_inference(SCENES_DIR)
print("[eval] inference on black bg ...")
cache_blk = run_inference(SCENES_BLACK) if os.path.isdir(SCENES_BLACK) else None


# ─── Metrics ──────────────────────────────────
def compute_metrics(cache):
    conf = np.zeros((3, 3), dtype=np.int64)
    inter = np.zeros(3); union = np.zeros(3)
    correct = total = 0
    for c in cache.values():
        p, m = c["pred"], c["gt"]
        for ct in range(3):
            for cp in range(3):
                conf[ct, cp] += int(((m == ct) & (p == cp)).sum())
        for k in range(3):
            pk = (p == k); mk = (m == k)
            inter[k] += int((pk & mk).sum()); union[k] += int((pk | mk).sum())
        correct += int((p == m).sum()); total += int(m.size)
    ious = [inter[k]/union[k] if union[k] > 0 else float("nan") for k in range(3)]
    return {"pixel_acc": correct/total, "mIoU": float(np.nanmean(ious)),
            "IoU": ious, "conf": conf}

m_tex = compute_metrics(cache_tex)
print(f"[textured] mIoU={m_tex['mIoU']:.3f} IoU={[f'{v:.3f}' for v in m_tex['IoU']]} pacc={m_tex['pixel_acc']:.3f}")
if cache_blk:
    m_blk = compute_metrics(cache_blk)
    print(f"[black-bg] mIoU={m_blk['mIoU']:.3f} IoU={[f'{v:.3f}' for v in m_blk['IoU']]} pacc={m_blk['pixel_acc']:.3f}")


# ─── FIG A: normal vs defect same-pose diff ─────────
# 從 parts_stage2/ 挑同 pose 7 個 state 的 png 並列
def fig_A():
    pose_az, pose_el = 0, 30; h = 0
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    normal_path = os.path.join(PARTS_DIR, "normal",
                               f"pan_head_az{pose_az:03d}_el{pose_el:+04d}_h{h}_normal.png")
    normal_img = np.array(Image.open(normal_path).convert("RGBA"))
    axes = axes.flatten()
    axes[0].imshow(normal_img); axes[0].set_title("normal (reference)", fontsize=10); axes[0].axis("off")
    for i, state in enumerate(DEFECT_STATES[1:], start=1):
        p = os.path.join(PARTS_DIR, state,
                         f"pan_head_az{pose_az:03d}_el{pose_el:+04d}_h{h}_{state}.png")
        img = np.array(Image.open(p).convert("RGBA"))
        # diff overlay：absdiff of RGB then highlight
        diff = np.abs(img[..., :3].astype(int) - normal_img[..., :3].astype(int)).sum(-1)
        diff_norm = (diff / max(1, diff.max()) * 255).astype(np.uint8)
        axes[i].imshow(img)
        axes[i].imshow(diff_norm, cmap="hot", alpha=0.35)
        axes[i].set_title(state, fontsize=10); axes[i].axis("off")
    plt.suptitle(f"FIG A — same pose (az={pose_az}, el={pose_el}) normal vs defect (diff heatmap overlay)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "A_normal_vs_defect_diff.png"), dpi=110, bbox_inches="tight")
    plt.close()
    print("  saved A_normal_vs_defect_diff.png")

# ─── FIG B: confidence heatmap ────────────────
def fig_B():
    # 挑 3 個有 defect 的 test scene
    defect_sids = []
    for sid in test_ids:
        if (cache_tex[sid]["gt"] == 2).sum() > 200:
            defect_sids.append(sid)
        if len(defect_sids) >= 3: break
    fig, axes = plt.subplots(len(defect_sids), 4, figsize=(14, 3.4 * len(defect_sids)))
    if len(defect_sids) == 1: axes = axes[None, :]
    for r, sid in enumerate(defect_sids):
        c = cache_tex[sid]
        axes[r, 0].imshow(c["rgb"]);    axes[r, 0].set_title(f"scene {sid:05d} RGB"); axes[r, 0].axis("off")
        axes[r, 1].imshow(CMAP3[c["gt"]]); axes[r, 1].set_title("GT"); axes[r, 1].axis("off")
        im2 = axes[r, 2].imshow(c["p_part"], cmap="viridis", vmin=0, vmax=1)
        axes[r, 2].set_title("P(part)  head 1"); axes[r, 2].axis("off")
        plt.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)
        im3 = axes[r, 3].imshow(c["p_def"], cmap="hot", vmin=0, vmax=1)
        axes[r, 3].set_title("P(defect)  head 2"); axes[r, 3].axis("off")
        plt.colorbar(im3, ax=axes[r, 3], fraction=0.046, pad=0.04)
    plt.suptitle("FIG B — Per-head confidence heatmap", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "B_confidence_heatmap.png"), dpi=110, bbox_inches="tight")
    plt.close()
    print("  saved B_confidence_heatmap.png")

# ─── FIG C: per defect_state per-instance IoU box ─────
def fig_C():
    # 對每個 instance：從 instance_mask 取該 instance 像素，跟 pred 算 binary IoU（pred==2 vs gt==2）
    by_state = {s: [] for s in DEFECT_STATES if s != "normal"}
    for sid in test_ids:
        meta = json.load(open(os.path.join(SCENES_DIR, f"{sid:05d}", "meta.json"), encoding="utf-8"))
        inst_mask = np.array(Image.open(os.path.join(SCENES_DIR, f"{sid:05d}", "instance_mask.png")))
        c = cache_tex[sid]
        pred_def = (c["pred"] == 2)
        for ins in meta["instances"]:
            if not ins["is_defective"]: continue
            state = ins["defect_state"]
            if state not in by_state: continue
            ins_pix = (inst_mask == ins["instance_id"])
            if ins_pix.sum() < 50: continue
            # gt_def restricted to this instance
            gt = ins_pix    # 全是 defect（因為 is_defective）
            inter = (pred_def & gt).sum()
            union = (pred_def | gt).sum()
            if union > 0:
                by_state[state].append(inter/union)
    data = [by_state[s] for s in by_state]
    labels = [f"{s}\n(n={len(by_state[s])})" for s in by_state]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#FFB570","#FF8533","#5BC0DE","#0099CC","#9B59B6","#7D3C98"]):
        patch.set_facecolor(color)
    ax.set_ylabel("Per-instance IoU (defect pixels)"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("FIG C — Per-instance defect IoU by ground-truth defect state")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "C_per_state_iou_box.png"), dpi=110, bbox_inches="tight")
    plt.close()
    print("  saved C_per_state_iou_box.png")
    return by_state

# ─── FIG D: FP / FN overlay 代表案例 ───────────
def diff_overlay(rgb, gt, pred):
    out = rgb.astype(float).copy()
    gt_def = (gt == 2); pred_def = (pred == 2)
    tp = gt_def & pred_def; fp = ~gt_def & pred_def; fn = gt_def & ~pred_def
    out[tp] = out[tp] * 0.3 + np.array([0, 220, 0]) * 0.7
    out[fp] = out[fp] * 0.3 + np.array([255, 0, 0]) * 0.7
    out[fn] = out[fn] * 0.3 + np.array([255, 160, 0]) * 0.7
    return out.clip(0, 255).astype(np.uint8)

def fig_D():
    stats = []
    for sid in test_ids:
        c = cache_tex[sid]
        gt_d = (c["gt"] == 2); pr_d = (c["pred"] == 2)
        stats.append({"sid": sid,
                      "tp": int((gt_d & pr_d).sum()),
                      "fp": int((~gt_d & pr_d).sum()),
                      "fn": int((gt_d & ~pr_d).sum()),
                      "n_gt": int(gt_d.sum())})
    with_def = [s for s in stats if s["n_gt"] > 0]
    best  = max(with_def, key=lambda s: s["tp"])
    miss  = max(with_def, key=lambda s: s["fn"])
    fa    = max(stats,    key=lambda s: s["fp"])
    clean = [s for s in stats if s["n_gt"] == 0]
    clean_best = min(clean, key=lambda s: s["fp"]) if clean else stats[0]
    cases = [("Best detection",  best),
             ("Worst miss (FN)", miss),
             ("Worst false alarm (FP)", fa),
             ("Clean (no GT defect)",    clean_best)]
    fig, axes = plt.subplots(len(cases), 4, figsize=(15, 3.4 * len(cases)))
    for r, (title, s) in enumerate(cases):
        c = cache_tex[s["sid"]]
        sub = f"\nTP={s['tp']} FP={s['fp']} FN={s['fn']}"
        axes[r,0].imshow(c["rgb"]); axes[r,0].set_title(f"{title}\nscene {s['sid']:05d}", fontsize=9); axes[r,0].axis("off")
        axes[r,1].imshow(CMAP3[c["gt"]]);  axes[r,1].set_title("GT"); axes[r,1].axis("off")
        axes[r,2].imshow(CMAP3[c["pred"]]); axes[r,2].set_title("Pred" + sub, fontsize=9); axes[r,2].axis("off")
        axes[r,3].imshow(diff_overlay(c["rgb"], c["gt"], c["pred"]))
        axes[r,3].set_title("Diff: green=TP red=FP orange=FN", fontsize=9); axes[r,3].axis("off")
    plt.suptitle("FIG D — FP / FN overlay representative cases", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "D_fp_fn_overlay.png"), dpi=110, bbox_inches="tight")
    plt.close()
    print("  saved D_fp_fn_overlay.png")

# ─── FIG E: epoch snapshots 組合 ───────────────
def fig_E():
    snaps = sorted(glob.glob(os.path.join(SNAP_DIR, "epoch_*.png")))
    if not snaps:
        print("  skip E (no snapshots)"); return
    n = len(snaps)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n))
    if n == 1: axes = [axes]
    for ax, p in zip(axes, snaps):
        ax.imshow(np.array(Image.open(p)))
        ax.set_title(os.path.basename(p), fontsize=9); ax.axis("off")
    plt.suptitle("FIG E — Validation prediction evolution across training", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "E_epoch_snapshots.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print("  saved E_epoch_snapshots.png")

# ─── FIG F: by-defect-state confusion matrix ─────
def fig_F():
    # rows: GT defect_state (7); cols: predicted class on that instance's pixels (3)
    mat = np.zeros((len(DEFECT_STATES), 3), dtype=np.int64)
    for sid in test_ids:
        meta = json.load(open(os.path.join(SCENES_DIR, f"{sid:05d}", "meta.json"), encoding="utf-8"))
        inst = np.array(Image.open(os.path.join(SCENES_DIR, f"{sid:05d}", "instance_mask.png")))
        c = cache_tex[sid]
        for ins in meta["instances"]:
            state = ins["defect_state"]
            if state not in DEFECT_STATES: continue
            ri = DEFECT_STATES.index(state)
            pix = (inst == ins["instance_id"])
            for cp in range(3):
                mat[ri, cp] += int(((c["pred"] == cp) & pix).sum())
    mat_norm = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES, rotation=15)
    ax.set_yticks(range(len(DEFECT_STATES))); ax.set_yticklabels(DEFECT_STATES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Ground-truth defect_state")
    ax.set_title("FIG F — Pixel-level confusion: defect_state × predicted (row-norm)")
    for i in range(len(DEFECT_STATES)):
        for j in range(3):
            ax.text(j, i, f"{mat_norm[i,j]*100:.1f}%",
                    ha="center", va="center",
                    color="white" if mat_norm[i,j] > 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "F_by_state_confusion.png"), dpi=110, bbox_inches="tight")
    plt.close()
    print("  saved F_by_state_confusion.png")
    return mat

# ─── ablation: textured vs black-bg ────────────
def fig_ablation():
    if not cache_blk:
        print("  skip ablation (no black-bg)"); return
    labels = ["bg", "normal", "defect"]
    tex_iou = m_tex["IoU"]; blk_iou = m_blk["IoU"]
    x = np.arange(3); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w/2, tex_iou, w, label=f"Textured (mIoU={m_tex['mIoU']:.3f})", color="#4C72B0")
    b2 = ax.bar(x + w/2, blk_iou, w, label=f"Black bg (mIoU={m_blk['mIoU']:.3f})", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Test IoU"); ax.set_ylim(0, 1.05)
    ax.set_title("Ablation: Textured DR bg vs Black bg (control)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.02, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ablation_textured_vs_black.png"), dpi=110, bbox_inches="tight")
    plt.close()
    print("  saved ablation_textured_vs_black.png")

# ─── training curves ───────────────────────────
def fig_training():
    if not os.path.exists(HIST_JSON):
        print("  skip training_curves (no history)"); return
    h = json.load(open(HIST_JSON, encoding="utf-8"))
    epochs = range(1, len(h["L_total"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].plot(epochs, h["L_total"], "k-", lw=2,   label="L_total")
    axes[0].plot(epochs, h["L_part"],  "C0--",        label="L_part (CE)")
    axes[0].plot(epochs, h["L_dice"],  "C3--",        label="L_dice")
    axes[0].plot(epochs, h["L_bce"],   "C2--",        label="L_bce")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(alpha=.3)
    axes[0].set_title("Training Loss Components"); axes[0].legend()
    axes[1].plot(epochs, h["val_mIoU"],       "k-", lw=2.2, label="mIoU")
    axes[1].plot(epochs, h["val_IoU_bg"],     "C0--", label="bg")
    axes[1].plot(epochs, h["val_IoU_normal"], "C2--", label="normal")
    axes[1].plot(epochs, h["val_IoU_defect"], "C3--", label="defect")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation IoU")
    axes[1].set_ylim(-.02, 1.02); axes[1].legend(loc="center right"); axes[1].grid(alpha=.3)
    axes[1].set_title("Validation IoU per Class")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  saved training_curves.png")


# ─── Run all ────────────────────────────────────
print("[fig] A normal/defect diff ...");        fig_A()
print("[fig] B confidence heatmap ...");        fig_B()
print("[fig] C per-state IoU box ...");         by_state = fig_C()
print("[fig] D FP/FN overlay ...");             fig_D()
print("[fig] E epoch snapshots ...");           fig_E()
print("[fig] F by-state confusion ...");        F_mat = fig_F()
print("[fig] ablation textured vs black ...");  fig_ablation()
print("[fig] training curves ...");             fig_training()

# ─── summary ────────────────────────────────────
summary = {
    "model": "DefectSegNetV2 (two-head, 3.31M params)",
    "test_set_size": len(test_ids),
    "textured_bg": {
        "pixel_acc": float(m_tex["pixel_acc"]),
        "mIoU": float(m_tex["mIoU"]),
        "IoU_per_class": {n: float(v) for n, v in zip(CLASS_NAMES, m_tex["IoU"])},
        "confusion_matrix": m_tex["conf"].tolist(),
    },
    "by_defect_state_mean_IoU": {s: (float(np.mean(v)) if v else None) for s, v in by_state.items()},
    "by_state_confusion_matrix": F_mat.tolist(),
    "ckpt_best_val_miou": float(ckpt.get("best_val_miou", float("nan"))),
}
if cache_blk:
    summary["black_bg_ablation"] = {
        "pixel_acc": float(m_blk["pixel_acc"]),
        "mIoU": float(m_blk["mIoU"]),
        "IoU_per_class": {n: float(v) for n, v in zip(CLASS_NAMES, m_blk["IoU"])},
    }
with open(os.path.join(FIG_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nAll figures + summary → {FIG_DIR}")
