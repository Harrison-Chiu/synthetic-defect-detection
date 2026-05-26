"""可視化 parts_stage2 — 5 個 defect state 各抽 4 張對比。"""
import os, random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

states = ['normal', 'bend_light', 'bend_heavy', 'displace_light', 'displace_heavy']
random.seed(0)
fig, axes = plt.subplots(len(states), 4, figsize=(12, 3 * len(states)))
for r, st in enumerate(states):
    files = sorted(os.listdir(f'output/parts_stage2/{st}'))
    pick = random.sample(files, 4)
    for c, fn in enumerate(pick):
        im = np.array(Image.open(f'output/parts_stage2/{st}/{fn}'))
        bg = np.ones((im.shape[0], im.shape[1], 3), dtype=np.uint8) * 200
        if im.shape[-1] == 4:
            alpha = im[:, :, 3:4] / 255.0
            rgb = im[:, :, :3]
            comp = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
        else:
            comp = im
        axes[r, c].imshow(comp)
        axes[r, c].set_title(f'{st}\n{fn[-30:]}', fontsize=7)
        axes[r, c].axis('off')
plt.tight_layout()
plt.savefig('output/parts_preview.png', dpi=90, bbox_inches='tight')
print('saved output/parts_preview.png')
