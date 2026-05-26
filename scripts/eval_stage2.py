"""
Stage 2 — Baseline 模型評估與報告用視覺化

讀 output/stage2_best_model.pt，產出多張報告適用的圖：
1. training_curves.png    — Loss + per-class IoU 沿 epoch（從 log 解析）
2. per_class_iou_bar.png  — Test set per-class IoU 長條圖
3. confusion_matrix.png   — 像素級 3×3 confusion matrix（row-normalized）
4. prediction_grid.png    — 多場景 RGB / GT / Pred / Diff 對比（用於說明模型行為）
5. failure_examples.png   — 4 個代表性案例：成功 / 漏報 defect / 誤報 / 全 normal 對

所有圖存到 docs/figures/stage2/
"""
import os, re, json, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BASE_DIR   = r"D:\Harrison\中山\大四下\深度學習期末報告"
SCENES_DIR = os.path.join(BASE_DIR, "output", "scenes")
MODEL_PT   = os.path.join(BASE_DIR, "output", "stage2_best_model.pt")
FIG_DIR    = os.path.join(BASE_DIR, "docs", "figures", "stage2")
TRAIN_LOG  = os.path.join(BASE_DIR, "output", "stage2_train_log.txt")
os.makedirs(FIG_DIR, exist_ok=True)

NUM_CLASSES = 3
CLASS_NAMES = ['background', 'normal_part', 'defective_part']
SHORT_NAMES = ['bg', 'normal', 'defect']
CMAP3 = np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]], dtype=np.uint8)  # bg黑 normal綠 defect紅

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[eval] device={device}')

# ─── 重建模型架構（與訓練腳本一致）───
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
    def __init__(self, num_classes=3, base_c=32):
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
        self.out_conv = nn.Conv2d(c//2, num_classes, 1)
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

# ─── 載模型 ───
ckpt = torch.load(MODEL_PT, map_location=device, weights_only=False)
model = DefectSegNet(num_classes=NUM_CLASSES, base_c=32).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f'[eval] loaded model, best_val_miou={ckpt.get("best_val_miou", "n/a")}')

# ─── 重建 test split（與訓練同 seed）───
SEED = 42
all_ids = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(SCENES_DIR, '[0-9]*')))
rng = np.random.RandomState(SEED)
shuffled = list(all_ids); rng.shuffle(shuffled)
n_train = int(0.8 * len(shuffled)); n_val = int(0.1 * len(shuffled))
test_ids = shuffled[n_train + n_val:]
print(f'[eval] test set: {len(test_ids)} scenes')

class SceneDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        sid = self.ids[idx]
        rgb = np.array(Image.open(os.path.join(SCENES_DIR, f'{sid:05d}', 'rgb.png')).convert('RGB'))
        mask = np.array(Image.open(os.path.join(SCENES_DIR, f'{sid:05d}', 'semantic_mask.png')))
        r = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        r = (r - 0.5) / 0.5
        return r, torch.from_numpy(mask).long(), sid

test_loader = DataLoader(SceneDataset(test_ids), batch_size=8, shuffle=False, num_workers=0)

def unnorm(t):
    return ((t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy())

# ─── 1. 收 confusion matrix + per-class IoU + 預測快取 ───
print('[eval] running inference on test set...')
conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
inter = np.zeros(NUM_CLASSES); union = np.zeros(NUM_CLASSES)
pred_cache = {}  # sid -> pred np.uint8 H×W
with torch.no_grad():
    for rgb, mask, sids in test_loader:
        pred = model(rgb.to(device)).argmax(dim=1).cpu()
        for b in range(rgb.shape[0]):
            p = pred[b].numpy(); m = mask[b].numpy()
            pred_cache[int(sids[b])] = p.astype(np.uint8)
            for ct in range(NUM_CLASSES):
                for cp in range(NUM_CLASSES):
                    conf[ct, cp] += int(((m == ct) & (p == cp)).sum())
            for c in range(NUM_CLASSES):
                pc = (p == c); mc = (m == c)
                inter[c] += int((pc & mc).sum()); union[c] += int((pc | mc).sum())
ious = [inter[c]/union[c] if union[c] > 0 else float('nan') for c in range(NUM_CLASSES)]
mIoU = np.nanmean(ious)
pacc = conf.trace() / conf.sum()
print(f'[eval] pixel_acc={pacc:.4f}, mIoU={mIoU:.4f}, IoU={[f"{v:.3f}" for v in ious]}')

# ─── FIG 1: training curves（解析 log）───
LOG_TXT = r"""[boot] python=C:\Users\Harrison\miniconda3\envs\dl_final\python.exe
[boot] torch=2.5.1+cu121 cuda_available=True device_count=1
[boot] gpu=NVIDIA GeForce RTX 4060
Using: cuda | NVIDIA GeForce RTX 4060
Total scenes available: 1000
Train / Val / Test: 800 / 100 / 100
Class frequencies: [0.71449524 0.23258636 0.0529184 ]
Class weights:     [0.17071117 0.5244158  2.30487303]
Parameters: 3,312,579
Epoch  1/30 [train=51.5s val=6.5s] loss=0.7151 mIoU=0.282 bg=0.723 norm=0.005 def=0.117
Epoch  2/30 [train=8.9s val=0.5s] loss=0.5514 mIoU=0.468 bg=0.977 norm=0.252 def=0.177
Epoch  3/30 [train=8.6s val=0.5s] loss=0.5246 mIoU=0.392 bg=0.979 norm=0.006 def=0.190
Epoch  4/30 [train=8.7s val=0.6s] loss=0.5112 mIoU=0.494 bg=0.981 norm=0.328 def=0.172
Epoch  5/30 [train=8.7s val=0.5s] loss=0.5159 mIoU=0.392 bg=0.985 norm=0.001 def=0.191
Epoch  6/30 [train=8.7s val=0.5s] loss=0.5008 mIoU=0.318 bg=0.557 norm=0.386 def=0.010
Epoch  7/30 [train=8.6s val=0.5s] loss=0.5036 mIoU=0.240 bg=0.618 norm=0.000 def=0.103
Epoch  8/30 [train=8.7s val=0.6s] loss=0.5052 mIoU=0.396 bg=0.987 norm=0.011 def=0.191
Epoch  9/30 [train=8.8s val=0.5s] loss=0.5063 mIoU=0.588 bg=0.984 norm=0.778 def=0.002
Epoch 10/30 [train=8.8s val=0.5s] loss=0.5020 mIoU=0.528 bg=0.986 norm=0.436 def=0.160
Epoch 11/30 [train=8.7s val=0.5s] loss=0.4950 mIoU=0.393 bg=0.988 norm=0.000 def=0.191
Epoch 12/30 [train=8.7s val=0.5s] loss=0.4915 mIoU=0.394 bg=0.989 norm=0.001 def=0.192
Epoch 13/30 [train=8.7s val=0.5s] loss=0.4898 mIoU=0.498 bg=0.990 norm=0.334 def=0.171
Epoch 14/30 [train=8.7s val=0.6s] loss=0.4904 mIoU=0.588 bg=0.986 norm=0.778 def=0.000
Epoch 15/30 [train=8.6s val=0.5s] loss=0.4881 mIoU=0.594 bg=0.991 norm=0.788 def=0.004
Epoch 16/30 [train=8.6s val=0.6s] loss=0.4848 mIoU=0.548 bg=0.991 norm=0.500 def=0.154
Epoch 17/30 [train=8.6s val=0.5s] loss=0.5218 mIoU=0.508 bg=0.887 norm=0.633 def=0.003
Epoch 18/30 [train=8.7s val=0.5s] loss=0.5488 mIoU=0.585 bg=0.981 norm=0.771 def=0.002
Epoch 19/30 [train=8.7s val=0.5s] loss=0.5028 mIoU=0.394 bg=0.984 norm=0.008 def=0.190
Epoch 20/30 [train=8.7s val=0.5s] loss=0.5155 mIoU=0.559 bg=0.985 norm=0.556 def=0.136
Epoch 21/30 [train=8.6s val=0.5s] loss=0.5045 mIoU=0.401 bg=0.982 norm=0.032 def=0.188
Epoch 22/30 [train=8.7s val=0.5s] loss=0.4934 mIoU=0.393 bg=0.988 norm=0.000 def=0.191
Epoch 23/30 [train=8.7s val=0.5s] loss=0.4918 mIoU=0.414 bg=0.988 norm=0.063 def=0.190
Epoch 24/30 [train=8.7s val=0.5s] loss=0.4919 mIoU=0.563 bg=0.989 norm=0.563 def=0.137
Epoch 25/30 [train=8.6s val=0.5s] loss=0.4888 mIoU=0.450 bg=0.989 norm=0.176 def=0.185
Epoch 26/30 [train=8.6s val=0.6s] loss=0.4877 mIoU=0.433 bg=0.991 norm=0.119 def=0.190
Epoch 27/30 [train=8.7s val=0.5s] loss=0.4866 mIoU=0.593 bg=0.990 norm=0.787 def=0.002
Epoch 28/30 [train=8.6s val=0.5s] loss=0.4838 mIoU=0.587 bg=0.991 norm=0.658 def=0.111
Epoch 29/30 [train=8.7s val=0.5s] loss=0.4820 mIoU=0.595 bg=0.992 norm=0.788 def=0.006
Epoch 30/30 [train=8.6s val=0.5s] loss=0.4818 mIoU=0.594 bg=0.992 norm=0.789 def=0.002"""

# 也存一份 log 給之後參考
with open(TRAIN_LOG, 'w', encoding='utf-8') as f:
    f.write(LOG_TXT)

pat = re.compile(r'Epoch +(\d+)/\d+ \[.*?\] loss=([\d.]+) mIoU=([\d.]+) bg=([\d.]+) norm=([\d.]+) def=([\d.]+)')
epochs, losses, mious, bg_iou, n_iou, d_iou = [], [], [], [], [], []
for m in pat.finditer(LOG_TXT):
    epochs.append(int(m.group(1)))
    losses.append(float(m.group(2)))
    mious.append(float(m.group(3)))
    bg_iou.append(float(m.group(4)))
    n_iou.append(float(m.group(5)))
    d_iou.append(float(m.group(6)))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
axes[0].plot(epochs, losses, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Train Loss (weighted CE)')
axes[0].set_title('Training Loss'); axes[0].grid(True, alpha=0.3)
axes[1].plot(epochs, mious,  'k-',  label='mIoU', linewidth=2.2)
axes[1].plot(epochs, bg_iou, 'C0--', label='bg')
axes[1].plot(epochs, n_iou,  'C2--', label='normal')
axes[1].plot(epochs, d_iou,  'C3--', label='defect')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation IoU')
axes[1].set_title('Validation IoU per Class')
axes[1].legend(loc='center right'); axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'training_curves.png'), dpi=120, bbox_inches='tight')
plt.close()
print('  saved training_curves.png')

# ─── FIG 2: per-class IoU bar ───
fig, ax = plt.subplots(figsize=(6.5, 4))
colors = ['#4C72B0', '#55A868', '#C44E52']
bars = ax.bar(SHORT_NAMES, ious, color=colors, edgecolor='black')
ax.set_ylabel('Test IoU'); ax.set_ylim(0, 1.0)
ax.set_title(f'Per-Class Test IoU (mIoU={mIoU:.3f}, pixel_acc={pacc*100:.1f}%)')
ax.grid(True, axis='y', alpha=0.3)
for b, v in zip(bars, ious):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.3f}',
            ha='center', va='bottom', fontweight='bold')
ax.axhline(1/3, color='gray', linestyle=':', alpha=0.7, label='random (1/3)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'per_class_iou_bar.png'), dpi=120, bbox_inches='tight')
plt.close()
print('  saved per_class_iou_bar.png')

# ─── FIG 3: confusion matrix (row-normalized) ───
conf_norm = conf.astype(float) / conf.sum(axis=1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(conf_norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel('Predicted'); ax.set_ylabel('Ground Truth')
ax.set_title('Pixel-Level Confusion Matrix (row-normalized)')
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        cnt = conf[i, j]
        ax.text(j, i, f'{conf_norm[i,j]*100:.1f}%\n({cnt:,})',
                ha='center', va='center',
                color='white' if conf_norm[i, j] > 0.5 else 'black', fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'), dpi=120, bbox_inches='tight')
plt.close()
print('  saved confusion_matrix.png')

# ─── FIG 4: prediction grid（多場景 RGB / GT / Pred / Diff）───
# 挑 6 個有代表性的測試場景
def categorize(sid):
    m = json.load(open(os.path.join(SCENES_DIR, f'{sid:05d}', 'meta.json'), encoding='utf-8'))
    return m['n_parts'], m['n_defective']

picks = {'all_normal_easy': None, 'one_defect': None, 'multi_defect': None,
         'all_defect': None, 'crowded_mix': None, 'sparse_defect': None}
for sid in test_ids:
    n, d = categorize(sid)
    if d == 0 and n in (3, 4) and picks['all_normal_easy'] is None: picks['all_normal_easy'] = sid
    elif d == 1 and n >= 4 and picks['one_defect'] is None: picks['one_defect'] = sid
    elif 2 <= d < n and n >= 5 and picks['multi_defect'] is None: picks['multi_defect'] = sid
    elif d == n and n >= 3 and picks['all_defect'] is None: picks['all_defect'] = sid
    elif n >= 6 and 1 <= d <= 2 and picks['crowded_mix'] is None: picks['crowded_mix'] = sid
    elif d == 1 and n <= 3 and picks['sparse_defect'] is None: picks['sparse_defect'] = sid

sel_sids = [v for v in picks.values() if v is not None][:6]
labels   = [k for k, v in picks.items() if v is not None][:6]

def diff_overlay(rgb, gt, pred):
    """生 TP/FP/FN colored overlay：
       綠 = TP (defect 對)
       紅 = FP (誤報 defect)
       橘 = FN (漏報 defect)
       灰背景 = RGB
    """
    out = rgb.astype(float).copy()
    gt_def   = (gt == 2)
    pred_def = (pred == 2)
    tp = gt_def & pred_def
    fp = ~gt_def & pred_def
    fn = gt_def & ~pred_def
    out[tp] = out[tp] * 0.3 + np.array([0, 220, 0]) * 0.7
    out[fp] = out[fp] * 0.3 + np.array([255, 0, 0]) * 0.7
    out[fn] = out[fn] * 0.3 + np.array([255, 160, 0]) * 0.7
    return out.clip(0, 255).astype(np.uint8)

n_rows = len(sel_sids)
fig, axes = plt.subplots(n_rows, 4, figsize=(15, 3.4 * n_rows))
if n_rows == 1: axes = axes[None, :]
for r, (sid, lab) in enumerate(zip(sel_sids, labels)):
    rgb = np.array(Image.open(os.path.join(SCENES_DIR, f'{sid:05d}', 'rgb.png')).convert('RGB'))
    gt  = np.array(Image.open(os.path.join(SCENES_DIR, f'{sid:05d}', 'semantic_mask.png')))
    pr  = pred_cache[sid]
    n, d = categorize(sid)
    axes[r, 0].imshow(rgb); axes[r, 0].set_title(f'{lab}\nscene {sid:05d} (n={n}, def={d})', fontsize=9); axes[r, 0].axis('off')
    axes[r, 1].imshow(CMAP3[gt]); axes[r, 1].set_title('Ground Truth'); axes[r, 1].axis('off')
    axes[r, 2].imshow(CMAP3[pr]); axes[r, 2].set_title('Prediction'); axes[r, 2].axis('off')
    axes[r, 3].imshow(diff_overlay(rgb, gt, pr))
    axes[r, 3].set_title('Diff: green=TP red=FP orange=FN', fontsize=9); axes[r, 3].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'prediction_grid.png'), dpi=110, bbox_inches='tight')
plt.close()
print('  saved prediction_grid.png')

# ─── FIG 5: failure-mode analysis ───
# 統計每張 test 圖的 defect TP/FP/FN pixel 比例
per_scene_stats = []
for sid in test_ids:
    gt = np.array(Image.open(os.path.join(SCENES_DIR, f'{sid:05d}', 'semantic_mask.png')))
    pr = pred_cache[sid]
    gt_def = (gt == 2); pr_def = (pr == 2)
    n_gt = int(gt_def.sum()); n_pr = int(pr_def.sum())
    tp = int((gt_def & pr_def).sum())
    fp = int((~gt_def & pr_def).sum())
    fn = int((gt_def & ~pr_def).sum())
    per_scene_stats.append({'sid': sid, 'n_gt': n_gt, 'n_pr': n_pr, 'tp': tp, 'fp': fp, 'fn': fn})

# 找：「最大成功」「最大漏報」「最大誤報」「全 normal 全對」
with_def = [s for s in per_scene_stats if s['n_gt'] > 0]
best_tp   = max(with_def, key=lambda s: s['tp'])
worst_fn  = max(with_def, key=lambda s: s['fn'])
worst_fp  = max(per_scene_stats, key=lambda s: s['fp'])
clean_ok  = [s for s in per_scene_stats if s['n_gt'] == 0]
clean_best = min(clean_ok, key=lambda s: s['fp']) if clean_ok else per_scene_stats[0]

cases = [
    ('Best detection (highest TP)', best_tp),
    ('Worst miss (highest FN)',    worst_fn),
    ('Worst false alarm (highest FP)', worst_fp),
    ('All-normal correct',         clean_best),
]
fig, axes = plt.subplots(len(cases), 4, figsize=(15, 3.4 * len(cases)))
for r, (title, s) in enumerate(cases):
    sid = s['sid']
    rgb = np.array(Image.open(os.path.join(SCENES_DIR, f'{sid:05d}', 'rgb.png')).convert('RGB'))
    gt  = np.array(Image.open(os.path.join(SCENES_DIR, f'{sid:05d}', 'semantic_mask.png')))
    pr  = pred_cache[sid]
    n, d = categorize(sid)
    sub = f"\nTP={s['tp']} FP={s['fp']} FN={s['fn']}"
    axes[r, 0].imshow(rgb); axes[r, 0].set_title(f'{title}\nscene {sid:05d} (n={n}, def={d})', fontsize=9); axes[r, 0].axis('off')
    axes[r, 1].imshow(CMAP3[gt]); axes[r, 1].set_title('Ground Truth'); axes[r, 1].axis('off')
    axes[r, 2].imshow(CMAP3[pr]); axes[r, 2].set_title('Prediction' + sub, fontsize=9); axes[r, 2].axis('off')
    axes[r, 3].imshow(diff_overlay(rgb, gt, pr))
    axes[r, 3].set_title('Diff: green=TP red=FP orange=FN', fontsize=9); axes[r, 3].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'failure_examples.png'), dpi=110, bbox_inches='tight')
plt.close()
print('  saved failure_examples.png')

# ─── 文字摘要存檔 ───
summary = {
    'model_params': 3312579,
    'test_set_size': len(test_ids),
    'pixel_accuracy': float(pacc),
    'mIoU': float(mIoU),
    'IoU_per_class': {n: float(v) for n, v in zip(CLASS_NAMES, ious)},
    'confusion_matrix_raw': conf.tolist(),
    'confusion_matrix_row_normalized': conf_norm.tolist(),
    'defect_pixel_stats_test': {
        'total_GT_defect_pixels': int(sum(s['n_gt'] for s in per_scene_stats)),
        'total_predicted_defect_pixels': int(sum(s['n_pr'] for s in per_scene_stats)),
        'TP': int(sum(s['tp'] for s in per_scene_stats)),
        'FP': int(sum(s['fp'] for s in per_scene_stats)),
        'FN': int(sum(s['fn'] for s in per_scene_stats)),
    },
}
with open(os.path.join(FIG_DIR, 'test_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print('  saved test_summary.json')
print('\nAll figures →', FIG_DIR)
