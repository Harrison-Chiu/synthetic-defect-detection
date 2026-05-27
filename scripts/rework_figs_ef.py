"""
Rework FIG E (加 GT 欄) + FIG F (拆 head 1 / head 2)

讀 stage3_best_model.pt + test split + val snapshot 樣本，產生：
  docs/figures/stage3/E_epoch_snapshots.png   ← 加上 GT/RGB 欄
  docs/figures/stage3/F1_head1_part_bg.png    ← head 1 by-state confusion
  docs/figures/stage3/F2_head2_defect.png     ← head 2 by-state confusion
保留原 F_by_state_confusion.png 不動（已合併版本）。

執行：
    & "C:\\Users\\Harrison\\miniconda3\\envs\\dl_final\\python.exe" -u scripts/rework_figs_ef.py
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

# ───── paths ─────
BASE_DIR     = r"D:\Harrison\中山\大四下\深度學習期末報告"
SCENES_DIR   = os.path.join(BASE_DIR, "output", "scenes")
MODEL_PT     = os.path.join(BASE_DIR, "output", "stage3_best_model.pt")
SNAP_DIR     = os.path.join(BASE_DIR, "output", "stage3_epoch_snapshots")
FIG_DIR      = os.path.join(BASE_DIR, "docs", "figures", "stage3")

DEFECT_STATES = ["normal", "bend_light", "bend_heavy",
                 "displace_light", "displace_heavy",
                 "remesh_light", "remesh_heavy"]
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CMAP3 = np.array([[0,0,0], [0,200,0], [200,0,0]], dtype=np.uint8)


# ───── model（與 train/eval 一致）─────
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
        self.enc1 = ConvBlock(3, c); self.enc2 = ConvBlock(c, c*2)
        self.enc3 = ConvBlock(c*2, c*4); self.enc4 = ConvBlock(c*4, c*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c*8, c*8)
        self.up4 = nn.ConvTranspose2d(c*8, c*4, 2, stride=2); self.dec4 = ConvBlock(c*4 + c*8, c*4)
        self.up3 = nn.ConvTranspose2d(c*4, c*2, 2, stride=2); self.dec3 = ConvBlock(c*2 + c*4, c*2)
        self.up2 = nn.ConvTranspose2d(c*2, c,   2, stride=2); self.dec2 = ConvBlock(c + c*2, c)
        self.up1 = nn.ConvTranspose2d(c, c//2,  2, stride=2); self.dec1 = ConvBlock(c//2 + c, c//2)
        self.head_part   = nn.Conv2d(c//2, 2, 1)
        self.head_defect = nn.Conv2d(c//2, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head_part(d1), self.head_defect(d1)


# ───── load ─────
ckpt = torch.load(MODEL_PT, map_location=device, weights_only=False)
model = DefectSegNetV2(base_c=32).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"[load] best_val_miou={ckpt.get('best_val_miou', 'n/a'):.4f}")

# ───── split（同 train/eval seed=42）─────
all_ids = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(SCENES_DIR, "[0-9]*")))
rng = np.random.RandomState(SEED); shuffled = list(all_ids); rng.shuffle(shuffled)
n_train = int(0.8 * len(shuffled)); n_val = int(0.1 * len(shuffled))
val_ids   = shuffled[n_train: n_train+n_val]
test_ids  = shuffled[n_train+n_val:]
print(f"[split] val={len(val_ids)} test={len(test_ids)}")


def load_scene(sid, root=SCENES_DIR):
    rgb = np.array(Image.open(os.path.join(root, f"{sid:05d}", "rgb.png")).convert("RGB"))
    mask = np.array(Image.open(os.path.join(root, f"{sid:05d}", "semantic_mask.png")))
    return rgb, mask


# ════════════════════════════════════════════
# FIG E: 加 GT 欄
# ════════════════════════════════════════════
def fig_E_with_gt():
    """4 個固定 val sample，第一 row 顯示 RGB + GT，後面 row 是每個 epoch 的預測."""
    snap_sids = val_ids[:4]   # 同 train script 的 FIXED_VAL_SNAPSHOT_N
    snap_pngs = sorted(glob.glob(os.path.join(SNAP_DIR, "epoch_*.png")))
    n_epoch = len(snap_pngs)
    n_sample = len(snap_sids)

    # 2 個 header row（RGB、GT）+ n_epoch row
    fig, axes = plt.subplots(2 + n_epoch, n_sample, figsize=(3.0*n_sample, 2.6*(2+n_epoch)))

    # row 0: RGB
    # row 1: GT colored
    for i, sid in enumerate(snap_sids):
        rgb, gt = load_scene(sid)
        axes[0, i].imshow(rgb); axes[0, i].axis("off")
        axes[0, i].set_title(f"sample {i}  (scene {sid:05d})", fontsize=9)
        axes[1, i].imshow(CMAP3[gt]); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("RGB", fontsize=10)
    axes[1, 0].set_ylabel("GT",  fontsize=10)
    # 用 text 在左邊標 row name
    for r, lbl in enumerate(["RGB", "GT"]):
        fig.text(0.01, 1 - (r+0.5)/(2+n_epoch), lbl, fontsize=11, fontweight="bold",
                 va="center", ha="left", rotation=90)

    # rows 2..: epoch pred snapshots — 每張 epoch_xxx.png 已是 1×4 並排，需切成 4 個
    for ei, p in enumerate(snap_pngs):
        img = np.array(Image.open(p))
        H, W = img.shape[:2]
        # 圖內 4 個 subplot 由 train script 用 plt.subplots(1,4,...) 產生
        # 直接均分 W 取 4 塊，先嘗試裁掉左右白邊
        # 為簡單起見：找出非白邊界後均分
        gray = img[..., :3].mean(axis=2) if img.ndim == 3 else img
        col_nonwhite = (gray < 250).any(axis=0)
        if col_nonwhite.any():
            x0 = np.argmax(col_nonwhite); x1 = W - np.argmax(col_nonwhite[::-1])
        else:
            x0, x1 = 0, W
        row_nonwhite = (gray < 250).any(axis=1)
        if row_nonwhite.any():
            y0 = np.argmax(row_nonwhite); y1 = H - np.argmax(row_nonwhite[::-1])
        else:
            y0, y1 = 0, H
        cropped = img[y0:y1, x0:x1]
        Hc, Wc = cropped.shape[:2]
        seg_w = Wc / n_sample
        epoch_name = os.path.basename(p).replace(".png", "")
        for i in range(n_sample):
            sub = cropped[:, int(i*seg_w):int((i+1)*seg_w)]
            ax = axes[2 + ei, i]
            ax.imshow(sub); ax.axis("off")
            if i == 0:
                fig.text(0.01, 1 - (2+ei+0.5)/(2+n_epoch),
                         epoch_name, fontsize=10, fontweight="bold",
                         va="center", ha="left", rotation=90)

    plt.suptitle("FIG E — Validation prediction evolution (with GT reference)", fontsize=12, y=0.995)
    plt.tight_layout(rect=[0.025, 0, 1, 0.985])
    out = os.path.join(FIG_DIR, "E_epoch_snapshots.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ════════════════════════════════════════════
# FIG F1 + F2: 拆 head
# ════════════════════════════════════════════
def fig_F_split():
    """
    對 test set 每個 defect instance 的像素統計：
      F1 — head 1 把這些 pixel 預測成 {bg, part} 的分布（看 head 1 有沒有看到該 instance）
      F2 — head 2 (sigmoid > 0.5) 在這些 pixel 上判定為 {normal, defect} 的分布
            （獨立評估 head 2，不受 head 1 gating 影響）
    """
    # accumulators: rows = DEFECT_STATES (含 normal)
    mat_h1 = np.zeros((len(DEFECT_STATES), 2), dtype=np.int64)  # cols = [bg, part]
    mat_h2 = np.zeros((len(DEFECT_STATES), 2), dtype=np.int64)  # cols = [pred_normal, pred_defect]

    with torch.no_grad():
        for sid in test_ids:
            rgb_np = np.array(Image.open(os.path.join(SCENES_DIR, f"{sid:05d}", "rgb.png")).convert("RGB"))
            r = torch.from_numpy(rgb_np).permute(2,0,1).float()/255.0
            r = ((r-0.5)/0.5).unsqueeze(0).to(device)
            pl, dl = model(r)
            h1_pred = pl.argmax(dim=1).squeeze(0).cpu().numpy()           # 0=bg, 1=part
            h2_pred = (torch.sigmoid(dl).squeeze().cpu().numpy() > 0.5).astype(np.uint8)  # 0/1

            inst = np.array(Image.open(os.path.join(SCENES_DIR, f"{sid:05d}", "instance_mask.png")))
            meta = json.load(open(os.path.join(SCENES_DIR, f"{sid:05d}", "meta.json"), encoding="utf-8"))
            for ins in meta["instances"]:
                state = ins["defect_state"]
                if state not in DEFECT_STATES: continue
                ri = DEFECT_STATES.index(state)
                pix = (inst == ins["instance_id"])
                if pix.sum() == 0: continue
                # head 1
                mat_h1[ri, 0] += int(((h1_pred == 0) & pix).sum())
                mat_h1[ri, 1] += int(((h1_pred == 1) & pix).sum())
                # head 2
                mat_h2[ri, 0] += int(((h2_pred == 0) & pix).sum())
                mat_h2[ri, 1] += int(((h2_pred == 1) & pix).sum())

    def plot_mat(mat, col_labels, title, fname, cmap):
        norm = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
        im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(DEFECT_STATES))); ax.set_yticklabels(DEFECT_STATES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("GT defect_state")
        ax.set_title(title, fontsize=11)
        for i in range(len(DEFECT_STATES)):
            for j in range(len(col_labels)):
                ax.text(j, i, f"{norm[i,j]*100:.1f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if norm[i,j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        out = os.path.join(FIG_DIR, fname)
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  saved {out}")
        return norm

    n1 = plot_mat(mat_h1, ["pred bg", "pred part"],
                  "FIG F1 — Head 1 (part vs bg) on each defect_state's pixels",
                  "F1_head1_part_bg.png", "Greens")
    n2 = plot_mat(mat_h2, ["pred normal", "pred defect"],
                  "FIG F2 — Head 2 (defect vs normal) on each defect_state's pixels  [independent of head 1]",
                  "F2_head2_defect.png", "Reds")
    return mat_h1.tolist(), mat_h2.tolist(), n1.tolist(), n2.tolist()


# ───── run ─────
print("[fig E] reworking with GT ...")
fig_E_with_gt()
print("[fig F1/F2] splitting heads ...")
m_h1, m_h2, n1, n2 = fig_F_split()

# 寫進 summary 補充
summary_path = os.path.join(FIG_DIR, "summary.json")
if os.path.exists(summary_path):
    summary = json.load(open(summary_path, encoding="utf-8"))
else:
    summary = {}
summary["head1_part_bg_by_state"] = {
    "rows": DEFECT_STATES, "cols": ["bg", "part"],
    "counts": m_h1, "row_normalized": n1,
}
summary["head2_defect_by_state"] = {
    "rows": DEFECT_STATES, "cols": ["normal", "defect"],
    "counts": m_h2, "row_normalized": n2,
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[summary] updated {summary_path}")
