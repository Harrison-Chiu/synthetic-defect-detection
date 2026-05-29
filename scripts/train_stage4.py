"""
Stage 4 — Three-head Defect Segmentation 訓練腳本

設計（vs Stage 3 two-head）：
    A part/bg     2-way softmax  CE                            (gate + 主指標)
    B defect_state 7-way softmax CE + multi-class Dice         (細粒度監督)
    C defect_type 4-way softmax  CE + multi-class Dice         (中介層級監督, aux)

GT gate：B / C 的 loss 只在 GT part 像素上算（ignore_index=-1）。
共用 encoder-decoder（沿用 Stage 2/3 ConvBlock），只在最後分歧成 3 個 1×1 conv。

額外：加 ReduceLROnPlateau（解 Stage 3 epoch 30 突崩問題）。

執行：
    & "C:\\Users\\Harrison\\miniconda3\\envs\\dl_final\\python.exe" -u scripts/train_stage4.py

輸出：
    output/stage4_best_model.pt
    output/stage4_history.json
    output/stage4_epoch_snapshots/
"""

# pylint: disable=wrong-import-position
import os
import json
import time
import glob

import matplotlib
matplotlib.use("Agg")
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
BASE_DIR     = r"D:\Harrison\中山\大四下\深度學習期末報告"
SCENES_DIR   = os.path.join(BASE_DIR, "output", "scenes")
SAVE_DIR     = os.path.join(BASE_DIR, "output")
SNAPSHOT_DIR = os.path.join(SAVE_DIR, "stage4_epoch_snapshots")

SEED        = 42
BATCH_SIZE  = 8
EPOCHS      = 30
LR          = 1e-3
SNAPSHOT_EVERY = 5

# Loss 權重（all 1.0 起步）
ALPHA_B = 1.0
ALPHA_C = 1.0

# Label spaces
STATE_TO_IDX = {
    "normal":         0,
    "bend_light":     1,
    "bend_heavy":     2,
    "displace_light": 3,
    "displace_heavy": 4,
    "remesh_light":   5,
    "remesh_heavy":   6,
}
IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
N_STATES = len(STATE_TO_IDX)

TYPE_TO_IDX = {"normal": 0, "bend": 1, "displace": 2, "remesh": 3}
IDX_TO_TYPE = {v: k for k, v in TYPE_TO_IDX.items()}
N_TYPES = len(TYPE_TO_IDX)

IGNORE_IDX = -100   # PyTorch CE 預設 ignore_index

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
# Dataset — 多頭 label
# ─────────────────────────────────────────────
def state_to_type_idx(state_name):
    if state_name == "normal":
        return TYPE_TO_IDX["normal"]
    prefix = state_name.split("_")[0]
    return TYPE_TO_IDX[prefix]


class SceneDatasetMulti(Dataset):
    def __init__(self, root, scene_ids):
        self.root = root
        self.scene_ids = scene_ids

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        sid = self.scene_ids[idx]
        scene_dir = os.path.join(self.root, f"{sid:05d}")
        rgb       = np.array(Image.open(os.path.join(scene_dir, "rgb.png")).convert("RGB"))
        sem       = np.array(Image.open(os.path.join(scene_dir, "semantic_mask.png")))
        inst      = np.array(Image.open(os.path.join(scene_dir, "instance_mask.png")))
        with open(os.path.join(scene_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)

        # Head A: part vs bg
        part_mask  = (sem > 0).astype(np.int64)

        # Head B/C: per-pixel state / type label, -100 outside part
        state_mask = np.full(inst.shape, IGNORE_IDX, dtype=np.int64)
        type_mask  = np.full(inst.shape, IGNORE_IDX, dtype=np.int64)
        for ins in meta["instances"]:
            iid   = ins["instance_id"]
            state = ins["defect_state"]
            pix   = (inst == iid)
            if not pix.any():
                continue
            state_mask[pix] = STATE_TO_IDX[state]
            type_mask[pix]  = state_to_type_idx(state)

        rgb_t  = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        rgb_t  = (rgb_t - 0.5) / 0.5
        return (rgb_t,
                torch.from_numpy(part_mask),
                torch.from_numpy(state_mask),
                torch.from_numpy(type_mask))


# ─────────────────────────────────────────────
# 模型：三頭
# ─────────────────────────────────────────────
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


class DefectSegNetV4(nn.Module):
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
        # 三頭
        self.head_part  = nn.Conv2d(c//2, 2,        1)   # A
        self.head_state = nn.Conv2d(c//2, N_STATES, 1)   # B
        self.head_type  = nn.Conv2d(c//2, N_TYPES,  1)   # C

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
        return self.head_part(d1), self.head_state(d1), self.head_type(d1)


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
def multiclass_dice_loss(logits, target, is_part_mask, num_classes, eps=1e-6):
    """
    Macro per-class Dice，only over part pixels。
    logits: (B, C, H, W); target: (B, H, W) long, ignore_index for bg;
    is_part_mask: (B, H, W) bool/float (1 inside part).
    """
    probs = F.softmax(logits, dim=1)                          # (B, C, H, W)
    is_part = is_part_mask.float().unsqueeze(1)               # (B, 1, H, W)
    # Build one-hot, but bg pixels get 0 everywhere → won't affect dice numer/denom thanks to is_part mask
    target_clamped = target.clone()
    target_clamped[target_clamped < 0] = 0
    onehot = F.one_hot(target_clamped, num_classes).permute(0, 3, 1, 2).float()
    probs  = probs  * is_part
    onehot = onehot * is_part
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)                        # (C,)
    denom = probs.sum(dims) + onehot.sum(dims)                # (C,)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def three_head_loss(p_logits, s_logits, t_logits, part_gt, state_gt, type_gt):
    """
    p_logits (B,2,H,W), s_logits (B,7,H,W), t_logits (B,4,H,W).
    part_gt (B,H,W) long 0/1, state_gt/type_gt long with IGNORE_IDX outside part.
    """
    is_part = (part_gt == 1)

    # Head A
    L_A = F.cross_entropy(p_logits, part_gt)

    # Head B
    L_B_ce   = F.cross_entropy(s_logits, state_gt, ignore_index=IGNORE_IDX)
    L_B_dice = multiclass_dice_loss(s_logits, state_gt, is_part, N_STATES)

    # Head C
    L_C_ce   = F.cross_entropy(t_logits, type_gt, ignore_index=IGNORE_IDX)
    L_C_dice = multiclass_dice_loss(t_logits, type_gt, is_part, N_TYPES)

    L_B = L_B_ce + L_B_dice
    L_C = L_C_ce + L_C_dice
    L_total = L_A + ALPHA_B * L_B + ALPHA_C * L_C
    return L_total, {
        "L_A":      float(L_A.item()),
        "L_B_ce":   float(L_B_ce.item()),
        "L_B_dice": float(L_B_dice.item()),
        "L_C_ce":   float(L_C_ce.item()),
        "L_C_dice": float(L_C_dice.item()),
        "L_total":  float(L_total.item()),
    }


# ─────────────────────────────────────────────
# 推論 + 評估
# ─────────────────────────────────────────────
def infer_3class(model, rgb):
    """組合 head A + head B → bg/normal/defect 3-class mask（跟 Stage 3 同 schema 可比）"""
    p_logits, s_logits, _ = model(rgb)
    part_pred  = p_logits.argmax(dim=1)               # (B, H, W) 0=bg, 1=part
    state_pred = s_logits.argmax(dim=1)               # (B, H, W) 0..6
    out = torch.zeros_like(part_pred)
    is_part = part_pred == 1
    is_defect = is_part & (state_pred > 0)
    out[is_part]   = 1
    out[is_defect] = 2
    return out, part_pred, state_pred


def evaluate_3class(model, loader, num_classes=3):
    model.eval()
    inter = np.zeros(num_classes); union = np.zeros(num_classes)
    correct = total = 0
    with torch.no_grad():
        for rgb, part_gt, state_gt, type_gt in loader:
            rgb = rgb.to(device)
            pred, _, _ = infer_3class(model, rgb)
            pred = pred.cpu()
            # 重建 3-class GT (0=bg, 1=normal_part, 2=defect_part)
            gt = torch.where(state_gt < 0, torch.zeros_like(state_gt),
                             torch.where(state_gt == 0,
                                         torch.ones_like(state_gt),
                                         torch.full_like(state_gt, 2)))
            for c in range(num_classes):
                p = (pred == c); t = (gt == c)
                inter[c] += (p & t).sum().item()
                union[c] += (p | t).sum().item()
            correct += (pred == gt).sum().item()
            total   += gt.numel()
    ious = [inter[c] / union[c] if union[c] > 0 else float("nan") for c in range(num_classes)]
    return {
        "pixel_acc": correct / total,
        "mIoU":      float(np.nanmean(ious)),
        "IoU_per_class": ious,
    }


def evaluate_per_state(model, loader):
    """7-class per-state IoU（only on part pixels），給報告用"""
    model.eval()
    inter = np.zeros(N_STATES); union = np.zeros(N_STATES)
    with torch.no_grad():
        for rgb, part_gt, state_gt, _ in loader:
            rgb = rgb.to(device)
            _, s_logits, _ = model(rgb)
            state_pred = s_logits.argmax(dim=1).cpu()
            is_part_gt = (state_gt >= 0)
            for c in range(N_STATES):
                p = (state_pred == c) & is_part_gt
                t = (state_gt == c)
                inter[c] += (p & t).sum().item()
                union[c] += (p | t).sum().item()
    return [inter[c] / union[c] if union[c] > 0 else float("nan") for c in range(N_STATES)]


# ─────────────────────────────────────────────
# Snapshot
# ─────────────────────────────────────────────
def mask_to_color(m):
    cmap = np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]], dtype=np.uint8)
    return cmap[m]


def save_epoch_snapshot(model, snapshot_rgb, epoch):
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

    all_ids  = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(SCENES_DIR, "[0-9]*")))
    rng      = np.random.RandomState(SEED)
    shuffled = list(all_ids); rng.shuffle(shuffled)
    n_train = int(0.8 * len(shuffled))
    n_val   = int(0.1 * len(shuffled))
    train_ids = shuffled[:n_train]
    val_ids   = shuffled[n_train:n_train + n_val]
    test_ids  = shuffled[n_train + n_val:]
    print(f"Train / Val / Test: {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")

    train_loader = DataLoader(SceneDatasetMulti(SCENES_DIR, train_ids),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(SceneDatasetMulti(SCENES_DIR, val_ids),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(SceneDatasetMulti(SCENES_DIR, test_ids),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    snap_ds   = SceneDatasetMulti(SCENES_DIR, val_ids[:FIXED_VAL_SNAPSHOT_N])
    snap_rgb  = torch.stack([snap_ds[i][0] for i in range(len(snap_ds))])

    model = DefectSegNetV4(base_c=32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, threshold=0.005, min_lr=1e-5
    )

    history = {"L_total": [], "L_A": [], "L_B_ce": [], "L_B_dice": [],
               "L_C_ce": [], "L_C_dice": [],
               "val_mIoU": [], "val_IoU_bg": [], "val_IoU_normal": [], "val_IoU_defect": [],
               "lr": []}
    best_miou  = -1.0
    best_state = None

    t_start = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        sum_losses = {k: 0.0 for k in ["L_total", "L_A", "L_B_ce", "L_B_dice", "L_C_ce", "L_C_dice"]}
        n_batches = 0
        t0 = time.time()
        for rgb, part_gt, state_gt, type_gt in train_loader:
            rgb      = rgb.to(device)
            part_gt  = part_gt.to(device)
            state_gt = state_gt.to(device)
            type_gt  = type_gt.to(device)
            p_logits, s_logits, t_logits = model(rgb)
            loss, comps = three_head_loss(p_logits, s_logits, t_logits, part_gt, state_gt, type_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k in sum_losses:
                sum_losses[k] += comps[k]
            n_batches += 1
        for k in sum_losses:
            sum_losses[k] /= max(1, n_batches)
        train_dt = time.time() - t0

        val_metrics = evaluate_3class(model, val_loader)
        cur_lr = optimizer.param_groups[0]["lr"]
        for k in sum_losses:
            history[k].append(sum_losses[k])
        history["val_mIoU"].append(val_metrics["mIoU"])
        history["val_IoU_bg"].append(val_metrics["IoU_per_class"][0])
        history["val_IoU_normal"].append(val_metrics["IoU_per_class"][1])
        history["val_IoU_defect"].append(val_metrics["IoU_per_class"][2])
        history["lr"].append(cur_lr)

        if val_metrics["mIoU"] > best_miou:
            best_miou  = val_metrics["mIoU"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        ious = val_metrics["IoU_per_class"]
        print(f"Ep {ep:2d}/{EPOCHS} [{train_dt:.1f}s] lr={cur_lr:.1e} "
              f"L={sum_losses['L_total']:.3f} "
              f"(A={sum_losses['L_A']:.3f} Bce={sum_losses['L_B_ce']:.3f} Bdc={sum_losses['L_B_dice']:.3f} "
              f"Cce={sum_losses['L_C_ce']:.3f} Cdc={sum_losses['L_C_dice']:.3f}) "
              f"mIoU={val_metrics['mIoU']:.3f} "
              f"bg={ious[0]:.3f} norm={ious[1]:.3f} def={ious[2]:.3f}")

        scheduler.step(val_metrics["mIoU"])

        if ep == 1 or ep % SNAPSHOT_EVERY == 0 or ep == EPOCHS:
            save_epoch_snapshot(model, snap_rgb, ep)

    total_dt = time.time() - t_start
    print(f"\nDone training in {total_dt/60:.1f} min. Best val mIoU = {best_miou:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_3class(model, test_loader)
    per_state_iou = evaluate_per_state(model, test_loader)
    print(f"\nTest pixel_acc = {test_metrics['pixel_acc']*100:.2f}%")
    print(f"Test mIoU      = {test_metrics['mIoU']:.3f}")
    for c, name in enumerate(["background", "normal_part", "defective_part"]):
        print(f"  {name:18s} IoU = {test_metrics['IoU_per_class'][c]:.3f}")
    print(f"\nPer-state IoU (head B, 7-class on part pixels):")
    for c, iou in enumerate(per_state_iou):
        print(f"  {IDX_TO_STATE[c]:18s} IoU = {iou:.3f}")

    save_path = os.path.join(SAVE_DIR, "stage4_best_model.pt")
    torch.save({
        "state_dict":      model.state_dict(),
        "best_val_miou":   best_miou,
        "test_metrics":    test_metrics,
        "per_state_iou":   per_state_iou,
        "state_to_idx":    STATE_TO_IDX,
        "type_to_idx":     TYPE_TO_IDX,
        "config": {
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "alpha_B": ALPHA_B, "alpha_C": ALPHA_C, "seed": SEED,
            "base_c": 32,
        },
    }, save_path)
    print(f"\nSaved model → {save_path}")

    with open(os.path.join(SAVE_DIR, "stage4_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history → output/stage4_history.json")


if __name__ == "__main__":
    main()
