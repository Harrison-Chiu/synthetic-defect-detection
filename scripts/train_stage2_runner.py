"""
Stage 2 訓練 runner — 跑完 segmentation baseline，存模型 + history JSON。
之後 notebook 直接 load 結果視覺化即可。

優化點 vs notebook：
- 全部 scene 一次預載入 RAM（~200MB）→ 0 IO bottleneck
- 較小模型 (base_c=16) → 約 1.2M params，速度約 4 倍
- 每 epoch print timing 便於診斷
"""
import os, json, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

BASE_DIR    = r"D:\Harrison\中山\大四下\深度學習期末報告"
SCENES_DIR  = os.path.join(BASE_DIR, "output", "scenes")
OUT_PATH    = os.path.join(BASE_DIR, "output", "stage2_best_model.pt")
HIST_PATH   = os.path.join(BASE_DIR, "output", "stage2_history.json")

NUM_CLASSES = 3
SEED = 42
BATCH_SIZE = 16
EPOCHS = 25
LR = 0.01
BASE_C = 16   # 模型起始通道數（總 params ~1.2M）

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


# ── 預載入全部 scene 到 RAM ──
class CachedSceneDataset(Dataset):
    def __init__(self, root, scene_ids):
        print(f"  Loading {len(scene_ids)} scenes into RAM...")
        t0 = time.time()
        self.rgb  = np.empty((len(scene_ids), 3, 256, 256), dtype=np.float32)
        self.mask = np.empty((len(scene_ids), 256, 256), dtype=np.int64)
        for i, sid in enumerate(scene_ids):
            sd = os.path.join(root, f"{sid:05d}")
            r = np.array(Image.open(os.path.join(sd, "rgb.png")).convert("RGB"))
            m = np.array(Image.open(os.path.join(sd, "semantic_mask.png")))
            self.rgb[i]  = ((r.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
            self.mask[i] = m
        print(f"  Loaded in {time.time()-t0:.1f}s, total RAM ~{(self.rgb.nbytes + self.mask.nbytes)/1e6:.0f}MB")

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.rgb[idx]), torch.from_numpy(self.mask[idx])


# ── 自製 Encoder-Decoder ──
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

class DefectSegNet(nn.Module):
    def __init__(self, num_classes=3, base_c=16):
        super().__init__()
        c = base_c
        self.enc1 = ConvBlock(3,    c)
        self.enc2 = ConvBlock(c,    c*2)
        self.enc3 = ConvBlock(c*2,  c*4)
        self.enc4 = ConvBlock(c*4,  c*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c*8, c*8)
        self.up4  = nn.ConvTranspose2d(c*8, c*4, 2, stride=2); self.dec4 = ConvBlock(c*4+c*8, c*4)
        self.up3  = nn.ConvTranspose2d(c*4, c*2, 2, stride=2); self.dec3 = ConvBlock(c*2+c*4, c*2)
        self.up2  = nn.ConvTranspose2d(c*2, c,   2, stride=2); self.dec2 = ConvBlock(c+c*2,   c)
        self.up1  = nn.ConvTranspose2d(c,   max(c//2,4), 2, stride=2)
        self.dec1 = ConvBlock(max(c//2,4)+c, max(c//2,4))
        self.out_conv = nn.Conv2d(max(c//2,4), num_classes, 1)
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
        return self.out_conv(d1)


def evaluate(model, loader):
    model.eval()
    inter = np.zeros(NUM_CLASSES); union = np.zeros(NUM_CLASSES)
    correct = total = 0
    with torch.no_grad():
        for rgb, mask in loader:
            rgb, mask = rgb.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            pred = model(rgb).argmax(dim=1)
            for c in range(NUM_CLASSES):
                p = (pred == c); t = (mask == c)
                inter[c] += (p & t).sum().item()
                union[c] += (p | t).sum().item()
            correct += (pred == mask).sum().item()
            total   += mask.numel()
    ious = [inter[c]/union[c] if union[c]>0 else 0.0 for c in range(NUM_CLASSES)]
    return {'pixel_acc': correct/total, 'mIoU': np.mean(ious), 'iou': ious}


def main():
    # Split
    all_ids = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(SCENES_DIR, '[0-9]*')))
    rng = np.random.RandomState(SEED)
    shuffled = list(all_ids); rng.shuffle(shuffled)
    n_tr = int(0.8 * len(shuffled)); n_va = int(0.1 * len(shuffled))
    train_ids = shuffled[:n_tr]
    val_ids   = shuffled[n_tr:n_tr+n_va]
    test_ids  = shuffled[n_tr+n_va:]
    print(f"Train/Val/Test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    print("Loading data...")
    train_ds = CachedSceneDataset(SCENES_DIR, train_ids)
    val_ds   = CachedSceneDataset(SCENES_DIR, val_ids)
    test_ds  = CachedSceneDataset(SCENES_DIR, test_ids)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Class weights
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i in range(min(200, len(train_ds))):
        m = train_ds.mask[i]
        for c in range(NUM_CLASSES):
            counts[c] += int((m == c).sum())
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    print(f"Class freq: {freq}, weights: {weights}")
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # Model
    model = DefectSegNet(num_classes=NUM_CLASSES, base_c=BASE_C).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

    history = {'train_loss': [], 'val_miou': [], 'val_iou': [], 'val_pacc': []}
    best_miou = 0.0
    best_state = None

    for ep in range(EPOCHS):
        t0 = time.time()
        model.train()
        running = 0.0
        for rgb, mask in train_loader:
            rgb, mask = rgb.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            logits = model(rgb)
            loss = criterion(logits, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)
        val = evaluate(model, val_loader)
        history['train_loss'].append(train_loss)
        history['val_miou'].append(val['mIoU'])
        history['val_iou'].append(val['iou'])
        history['val_pacc'].append(val['pixel_acc'])
        if val['mIoU'] > best_miou:
            best_miou = val['mIoU']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        dt = time.time() - t0
        print(f"Epoch {ep+1:2d}/{EPOCHS} [{dt:5.1f}s] loss={train_loss:.4f} pix_acc={val['pixel_acc']*100:.1f}% "
              f"mIoU={val['mIoU']:.3f}  [bg={val['iou'][0]:.3f} norm={val['iou'][1]:.3f} def={val['iou'][2]:.3f}]")

    # Restore best & final test
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nBest val mIoU: {best_miou:.3f}")
    test = evaluate(model, test_loader)
    print(f"\nTest pix_acc={test['pixel_acc']*100:.2f}%  mIoU={test['mIoU']:.3f}")
    for c, name in enumerate(['background','normal_part','defective_part']):
        print(f"  {name:18s} IoU: {test['iou'][c]:.3f}")

    # Save
    torch.save({
        'state_dict': model.state_dict(),
        'base_c': BASE_C,
        'num_classes': NUM_CLASSES,
        'best_val_miou': best_miou,
        'test_metrics': test,
    }, OUT_PATH)
    with open(HIST_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved model → {OUT_PATH}")
    print(f"Saved history → {HIST_PATH}")


if __name__ == "__main__":
    main()
