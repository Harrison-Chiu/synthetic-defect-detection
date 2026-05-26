"""快速可視化 output/scenes 內容 — 把 0/1/2 mask 上色看得到。"""
import os, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 挑 4 個有代表性的：全 normal / 1 defect / 多 defect / 全 defect
import json as _json
_picks = {'all_normal': None, 'one_def': None, 'multi_def': None, 'all_def': None}
for sid in range(1000):
    m = _json.load(open(f'output/scenes/{sid:05d}/meta.json', encoding='utf-8'))
    nd, nt = m['n_defective'], m['n_parts']
    if nd == 0 and _picks['all_normal'] is None: _picks['all_normal'] = f'{sid:05d}'
    elif nd == 1 and nt >= 4 and _picks['one_def'] is None: _picks['one_def'] = f'{sid:05d}'
    elif nd >= 2 and nd < nt and _picks['multi_def'] is None: _picks['multi_def'] = f'{sid:05d}'
    elif nd == nt and nt >= 3 and _picks['all_def'] is None: _picks['all_def'] = f'{sid:05d}'
ids = [v for v in _picks.values() if v]
fig, axes = plt.subplots(len(ids), 4, figsize=(14, 3.2 * len(ids)))

for r, sid in enumerate(ids):
    rgb = np.array(Image.open(f'output/scenes/{sid}/rgb.png'))
    sem = np.array(Image.open(f'output/scenes/{sid}/semantic_mask.png'))
    ins = np.array(Image.open(f'output/scenes/{sid}/instance_mask.png'))
    meta = json.load(open(f'output/scenes/{sid}/meta.json'))

    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    overlay = rgb.astype(float).copy()
    overlay[sem == 1] = overlay[sem == 1] * 0.5 + np.array([0, 200, 0]) * 0.5
    overlay[sem == 2] = overlay[sem == 2] * 0.5 + np.array([220, 0, 0]) * 0.5
    overlay = overlay.clip(0, 255).astype(np.uint8)

    title = f"scene {sid}  n={meta['n_parts']} def={meta['n_defective']}"
    axes[r, 0].imshow(rgb); axes[r, 0].set_title(title); axes[r, 0].axis('off')
    axes[r, 1].imshow(sem, cmap='viridis', vmin=0, vmax=2); axes[r, 1].set_title('semantic (0/1/2)'); axes[r, 1].axis('off')
    axes[r, 2].imshow(ins, cmap='tab10', vmin=0, vmax=10); axes[r, 2].set_title('instance ID'); axes[r, 2].axis('off')
    axes[r, 3].imshow(overlay); axes[r, 3].set_title('overlay G=normal R=defect'); axes[r, 3].axis('off')

plt.tight_layout()
plt.savefig('output/scenes_preview.png', dpi=90, bbox_inches='tight')
print('saved output/scenes_preview.png')

n_def, n_total = 0, 0
for sid in os.listdir('output/scenes'):
    p = f'output/scenes/{sid}/meta.json'
    if not os.path.isfile(p):
        continue
    m = json.load(open(p))
    n_total += m['n_parts']
    n_def += m['n_defective']
print(f'1000 scenes: total parts={n_total}, defective={n_def} ({n_def/n_total*100:.1f}%)')
