"""
Stage 3 — Two-head Defect Segmentation 訓練腳本

設計：
    - 共用 encoder-decoder（同 Stage 2 架構 + skip connection）
    - Head 1: part vs bg（2-class softmax + CE）
    - Head 2: defect vs normal，**只在 part 像素算 loss**（Dice + BCE）
    - 推論時兩 head 組合回 3-class mask 跟 Stage 2 可比

執行：
    & "C:\\Users\\Harrison\\miniconda3\\envs\\dl_final\\python.exe" -u scripts/train_stage3.py

輸出：
    output/stage3_best_model.pt
    output/stage3_history.json
    output/stage3_epoch_snapshots/  ← debug viz E 用
"""

# pylint: disable=wrong-import-position
import os
import json
import time
import glob

import matplotlib
matplotlib.use("Agg")   # 防 plt.show 阻塞
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
BASE_DIR    = r"D:\Harrison\中山\大四下\深度學習期末報告"
SCENES_DIR  = os.path.join(BASE_DIR, "output", "scenes")
SAVE_DIR    = os.path.join(BASE_DIR, "output")
SNAPSHOT_DIR = os.path.join(SAVE_DIR, "stage3_epoch_snapshots")

SEED        = 42
BATCH_SIZE  = 8
EPOCHS      = 30
LR          = 1e-3                    # Stage 2 用 0.01 SGD，Stage 3 改 Adam 1e-3
SNAPSHOT_EVERY = 5                    # 每 5 epoch 存 val 預測 snapshot

# Two-head loss 權重
LAMBDA_DEFECT  = 1.0                  # L_total = L_part + λ * L_defect
DEFECT_DICE_W  = 0.5                  # L_defect = w * Dice + (1-w) * BCE
DEFECT_BCE_W   = 1 - DEFECT_DICE_W

# 視覺檢查 snapshot 用的 fixed val 索引（不隨機，固定看同幾張）
FIXED_VAL_SNAPSHOT_N = 4

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────
# Dataset（跟 Stage 2 同格式）
# ─────────────────────────────────────────────
class SceneDataset(Dataset):
    def __init__(self, root, scene_ids):
        self.root = root
        self.scene_ids = scene_ids

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        sid = self.scene_ids[idx]
        scene_dir = os.path.join(self.root, f"{sid:05d}")
        rgb  = np.array(Image.open(os.path.join(scene_dir, "rgb.png")).convert("RGB"))
        mask = np.array(Image.open(os.path.join(scene_dir, "semantic_mask.png")))
        rgb_t  = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        rgb_t  = (rgb_t - 0.5) / 0.5
        mask_t = torch.from_numpy(mask).long()
        return rgb_t, mask_t


# ─────────────────────────────────────────────
# 模型：Two-head Encoder-Decoder
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    """跟 Stage 2 完全一樣，方便 weight 概念延續"""
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
    """
    Encoder-Decoder 完全沿用 Stage 2 設計，只在最後分兩個 1×1 conv head。
    - head_part:    輸出 (B, 2, H, W) → part vs bg 的 2-class logits
    - head_defect:  輸出 (B, 1, H, W) → defect 的 per-pixel logit (sigmoid 後是機率)
    """
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
        # Two heads
        self.head_part   = nn.Conv2d(c//2, 2, 1)   # bg / part
        self.head_defect = nn.Conv2d(c//2, 1, 1)   # defect logit

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


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
def two_head_loss(part_logits, defect_logits, mask_3class):
    """
    part_logits:   (B, 2, H, W)
    defect_logits: (B, 1, H, W)
    mask_3class:   (B, H, W) long，值 0/1/2
    """
    # GT 拆解
    part_gt   = (mask_3class > 0).long()           # 0=bg, 1=part
    defect_gt = (mask_3class == 2).float()         # 1=defect, 0=else（含 bg，但 loss 只算 part 區）
    is_part   = (mask_3class > 0).float()          # mask for head 2 loss

    # Head 1: part vs bg (CE)
    L_part = F.cross_entropy(part_logits, part_gt)

    # Head 2: defect/normal，only on part pixels
    d_logit = defect_logits.squeeze(1)             # (B, H, W)
    d_prob  = torch.sigmoid(d_logit)

    # Masked Dice
    p_masked = d_prob * is_part
    g_masked = defect_gt * is_part
    eps = 1e-6
    intersect = (p_masked * g_masked).sum()
    L_dice = 1 - (2 * intersect + eps) / (p_masked.sum() + g_masked.sum() + eps)

    # Masked BCE（per-pixel BCE 後乘 is_part，再對 is_part 數量平均）
    bce = F.binary_cross_entropy_with_logits(d_logit, defect_gt, reduction="none")
    L_bce = (bce * is_part).sum() / (is_part.sum() + eps)

    L_defect = DEFECT_DICE_W * L_dice + DEFECT_BCE_W * L_bce
    L_total  = L_part + LAMBDA_DEFECT * L_defect

    return L_total, {
        "L_part":   float(L_part.item()),
        "L_dice":   float(L_dice.item()),
        "L_bce":    float(L_bce.item()),
        "L_total":  float(L_total.item()),
    }


# ─────────────────────────────────────────────
# 推論 + 評估
# ─────────────────────────────────────────────
def infer_3class(model, rgb):
    """組合兩 head → 3-class mask: 0=bg, 1=normal, 2=defect"""
    part_logits, defect_logits = model(rgb)
    part_pred   = part_logits.argmax(dim=1)            # (B, H, W)
    defect_prob = torch.sigmoid(defect_logits).squeeze(1)
    # part 區內 defect_prob>0.5 → 2，否則 → 1；非 part 區 → 0
    out = torch.zeros_like(part_pred)
    is_part = part_pred == 1
    is_defect = is_part & (defect_prob > 0.5)
    out[is_part]   = 1
    out[is_defect] = 2
    return out, part_pred, defect_prob


def evaluate(model, loader, num_classes=3):
    model.eval()
    inter = np.zeros(num_classes); union = np.zeros(num_classes)
    correct = total = 0
    with torch.no_grad():
        for rgb, mask in loader:
            rgb, mask = rgb.to(device), mask.to(device)
            pred, _, _ = infer_3class(model, rgb)
            for c in range(num_classes):
                p = (pred == c); t = (mask == c)
                inter[c] += (p & t).sum().item()
                union[c] += (p | t).sum().item()
            correct += (pred == mask).sum().item()
            total   += mask.numel()
    ious = [inter[c] / union[c] if union[c] > 0 else float("nan") for c in range(num_classes)]
    return {
        "pixel_acc": correct / total,
        "mIoU":      float(np.nanmean(ious)),
        "IoU_per_class": ious,
    }


# ─────────────────────────────────────────────
# Snapshot 工具（給 debug viz E 用）
# ─────────────────────────────────────────────
def mask_to_color(m):
    cmap = np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]], dtype=np.uint8)
    return cmap[m]


def save_epoch_snapshot(model, snapshot_rgb, snapshot_mask, epoch):
    """訓練過程中每 N epoch 對固定的 val sample 跑推論存圖"""
    model.eval()
    with torch.no_grad():
        rgb_d = snapshot_rgb.to(device)
        pred, _, _ = infer_3class(model, rgb_d)
        pred = pred.cpu().numpy()
    n = pred.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i in range(n):
        # 並排：GT 在上半、Pred 在下半疊加 → 簡單做法是直接看 pred
        axes[i].imshow(mask_to_color(pred[i]))
        axes[i].axis("off")
        axes[i].set_title(f"sample {i}", fontsize=8)
    fig.suptitle(f"Epoch {epoch} predictions", fontsize=10)
    plt.tight_layout()
    out = os.path.join(SNAPSHOT_DIR, f"epoch_{epoch:03d}.png")
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close()
    return out


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # Dataset split（跟 Stage 2 同 seed=42 → 同 train/val/test scene ids）
    all_ids  = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(SCENES_DIR, "[0-9]*")))
    rng      = np.random.RandomState(SEED)
    shuffled = list(all_ids); rng.shuffle(shuffled)
    n_train = int(0.8 * len(shuffled))
    n_val   = int(0.1 * len(shuffled))
    train_ids = shuffled[:n_train]
    val_ids   = shuffled[n_train:n_train + n_val]
    test_ids  = shuffled[n_train + n_val:]
    print(f"Train / Val / Test: {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")

    train_loader = DataLoader(SceneDataset(SCENES_DIR, train_ids),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(SceneDataset(SCENES_DIR, val_ids),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(SceneDataset(SCENES_DIR, test_ids),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 取固定 val snapshot
    snap_ds = SceneDataset(SCENES_DIR, val_ids[:FIXED_VAL_SNAPSHOT_N])
    snap_rgb  = torch.stack([snap_ds[i][0] for i in range(len(snap_ds))])
    snap_mask = torch.stack([snap_ds[i][1] for i in range(len(snap_ds))])

    # Model + Optimizer
    model = DefectSegNetV2(base_c=32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    history = {"L_total": [], "L_part": [], "L_dice": [], "L_bce": [],
               "val_mIoU": [], "val_IoU_bg": [], "val_IoU_normal": [], "val_IoU_defect": []}
    best_miou  = -1.0
    best_state = None

    t_start = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        sum_losses = {"L_total": 0.0, "L_part": 0.0, "L_dice": 0.0, "L_bce": 0.0}
        n_batches = 0
        t0 = time.time()
        for rgb, mask in train_loader:
            rgb, mask = rgb.to(device), mask.to(device)
            p_logits, d_logits = model(rgb)
            loss, comps = two_head_loss(p_logits, d_logits, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k in sum_losses:
                sum_losses[k] += comps[k]
            n_batches += 1
        for k in sum_losses:
            sum_losses[k] /= max(1, n_batches)
        train_dt = time.time() - t0

        val_metrics = evaluate(model, val_loader)
        history["L_total"].append(sum_losses["L_total"])
        history["L_part"].append(sum_losses["L_part"])
        history["L_dice"].append(sum_losses["L_dice"])
        history["L_bce"].append(sum_losses["L_bce"])
        history["val_mIoU"].append(val_metrics["mIoU"])
        history["val_IoU_bg"].append(val_metrics["IoU_per_class"][0])
        history["val_IoU_normal"].append(val_metrics["IoU_per_class"][1])
        history["val_IoU_defect"].append(val_metrics["IoU_per_class"][2])

        if val_metrics["mIoU"] > best_miou:
            best_miou  = val_metrics["mIoU"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        ious = val_metrics["IoU_per_class"]
        print(f"Ep {ep:2d}/{EPOCHS} [{train_dt:.1f}s] "
              f"L={sum_losses['L_total']:.3f} (part={sum_losses['L_part']:.3f} "
              f"dice={sum_losses['L_dice']:.3f} bce={sum_losses['L_bce']:.3f}) "
              f"mIoU={val_metrics['mIoU']:.3f} "
              f"bg={ious[0]:.3f} norm={ious[1]:.3f} def={ious[2]:.3f}")

        # 存 snapshot
        if ep == 1 or ep % SNAPSHOT_EVERY == 0 or ep == EPOCHS:
            save_epoch_snapshot(model, snap_rgb, snap_mask, ep)

    total_dt = time.time() - t_start
    print(f"\nDone training in {total_dt/60:.1f} min. Best val mIoU = {best_miou:.3f}")

    # Load best, eval test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)
    print(f"\nTest pixel_acc = {test_metrics['pixel_acc']*100:.2f}%")
    print(f"Test mIoU      = {test_metrics['mIoU']:.3f}")
    for c, name in enumerate(["background", "normal_part", "defective_part"]):
        print(f"  {name:18s} IoU = {test_metrics['IoU_per_class'][c]:.3f}")

    # 存 best model + history
    save_path = os.path.join(SAVE_DIR, "stage3_best_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "best_val_miou": best_miou,
        "test_metrics":  test_metrics,
        "config": {
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "lambda_defect": LAMBDA_DEFECT,
            "defect_dice_w": DEFECT_DICE_W, "defect_bce_w": DEFECT_BCE_W,
            "seed": SEED,
        },
    }, save_path)
    print(f"\nSaved model → {save_path}")

    with open(os.path.join(SAVE_DIR, "stage3_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history → output/stage3_history.json")


if __name__ == "__main__":
    main()
