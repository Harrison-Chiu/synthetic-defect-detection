"""
dataset.py — 訓練/評估的 torch Dataset

兩種來源,同一套 schema 標籤編碼:
- RuntimeSceneDataset：訓練用。當場 generate_scene(i),**不存訓練圖**。
  map-style + DataLoader(shuffle=False) → 每個 index 一張 distinct 場景,
  以 total steps 思考(length = 要跑的樣本數),天然不重複、可復現。
- EvalSetDataset：評估用。讀**凍結**在磁碟的場景(output/eval_set/ 或現有 scenes
  的 test split),保跨 stage 數字可比。

標籤編碼由 src.schema 驅動:每個 head 一個 target,新增/改 head 只動 schema。
取代 scripts/train_stage4.py 寫死的 (part_mask, state_mask, type_mask) 三元組。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src import schema
from src.data.assets import Assets, load_assets
from src.data.config import (
    CLS_DEFECT, CLS_NORMAL, DEFAULT_CONFIG,
    DEFECT_DILATE_PX, DEFECT_GAUSS_SIGMA, DEFECT_W_BG, DEFECT_W_NORM, GenConfig,
)
from src.data.generator import SceneSample, generate_scene


# ── 影像 / 標籤編碼(兩個 dataset 共用)──────────────────────
def encode_rgb(rgb: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 → (3,H,W) float,normalize 到 [-1,1](同 Stage 4)。"""
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (t - 0.5) / 0.5


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask.astype(bool)
    try:
        from scipy import ndimage
        return ndimage.binary_dilation(mask, iterations=px)
    except ImportError:
        return mask.astype(bool)


def _gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    try:
        from scipy import ndimage
        return ndimage.gaussian_filter(x, sigma=sigma)
    except ImportError:
        return x


def encode_targets(
    semantic: np.ndarray,
    instance: np.ndarray,
    meta: dict,
    defect_region: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """產生 S5 雙頭的 target + eval 所需的輔助 GT。

    回傳 dict:
      - "A"        part head GT,(H,W) int64,0=bg / 1=part(CE)。
      - "B_T"      defect 頭 soft target T,(H,W) float32:變形區膨脹帶=1,其餘=0
                   (壞件內部/背景的值無所謂,因 W≈0)。
      - "B_W"      defect 頭權重 W,(H,W) float32 = max(Gauss(D⁺,σ), w_norm·N);
                   遠背景 & 壞件內部 → ≈0(= ignore,**由高斯自然衰減,非硬條件**)。
      - "sem3"     eval 用 3-class GT,(H,W) int64,0=bg/1=normal/2=defect(整顆壞件)。
      - "state_px" eval 用 per-pixel state idx,(H,W) int64,零件外 = IGNORE_INDEX。

    defect_region 缺省(disk eval set 未存)→ T/W 退化為全 0(eval 不吃 T/W)。
    """
    part_mask = (instance > 0).astype(np.int64)
    sem3 = semantic.astype(np.int64)
    N = (semantic == CLS_NORMAL)  # 好件剪影(逼模型學「好件→不該亮」)

    # ── defect 頭 T / W ──
    if defect_region is None:
        D = np.zeros(semantic.shape, dtype=bool)
    else:
        D = defect_region.astype(bool)
    D_plus = _dilate(D, DEFECT_DILATE_PX)
    # S6: T = 壞件 silhouette ∪ D+(bend 差異在邊緣,D+ 可能延伸到 silhouette 外)
    defect_sil = (semantic == CLS_DEFECT)
    T = (defect_sil | D_plus).astype(np.float32)
    w_gauss = _gaussian(D_plus.astype(np.float32), DEFECT_GAUSS_SIGMA)
    if w_gauss.max() > 0:
        w_gauss = w_gauss / w_gauss.max()  # 正規化到 [0,1],變形區中心≈1
    W = np.maximum(w_gauss, DEFECT_W_NORM * N.astype(np.float32))
    W = np.maximum(W, DEFECT_W_BG)  # 背景底權重(預設 0)

    # ── eval per-state(從 meta 重建,不再是訓練頭)──
    state_px = np.full(instance.shape, schema.IGNORE_INDEX, dtype=np.int64)
    for ins in meta["instances"]:
        pix = instance == ins["instance_id"]
        if pix.any():
            state_px[pix] = schema.STATE_TO_IDX[ins["defect_state"]]

    return {
        "A": torch.from_numpy(part_mask),
        "B_T": torch.from_numpy(T),
        "B_W": torch.from_numpy(W),
        "sem3": torch.from_numpy(sem3),
        "state_px": torch.from_numpy(state_px),
    }


def sample_to_tensors(sample: SceneSample):
    return encode_rgb(sample.rgb), encode_targets(
        sample.semantic, sample.instance, sample.meta, sample.defect_region)


# ── 訓練:runtime 生成 ──────────────────────────────────────
class RuntimeSceneDataset(Dataset):
    """當場確定性生成。index → generate_scene(index_offset + index)。

    用法:length 設為這個 run 要消化的樣本數(total_steps × batch_size);
    DataLoader(shuffle=False, num_workers>0 OK —— 生成是純函數,worker 各算各的)。
    """

    def __init__(
        self,
        length: int,
        assets: Assets | None = None,
        config: GenConfig = DEFAULT_CONFIG,
        index_offset: int = 0,
    ):
        self.length = int(length)
        self.assets = assets if assets is not None else load_assets()
        self.config = config
        self.index_offset = int(index_offset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = generate_scene(self.index_offset + idx, self.assets, self.config)
        return sample_to_tensors(sample)


# ── 評估:一次生成 → 常駐記憶體(val/test 每輪重用,不重生)──────
class ListDataset(Dataset):
    """把已生成的 (rgb_tensor, targets_dict) 清單包成 Dataset。

    val/test 固定不變,跑前用 workers 平行生成一次、收進 list,之後每次 eval
    直接從 RAM 取(num_workers=0)→ 消除「每次 eval 重生場景」的大宗開銷。
    """

    def __init__(self, items: list):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def materialize(dataset: Dataset, batch_size: int = 8, num_workers: int = 8) -> ListDataset:
    """用 DataLoader 平行跑完一個 dataset,把每筆 (rgb, targets) 收進 ListDataset。"""
    from torch.utils.data import DataLoader
    items = []
    for rgb, targets in DataLoader(dataset, batch_size=batch_size, num_workers=num_workers):
        for i in range(rgb.shape[0]):
            items.append((rgb[i], {k: v[i] for k, v in targets.items()}))
    return ListDataset(items)


# ── 評估:凍結磁碟場景 ──────────────────────────────────────
class EvalSetDataset(Dataset):
    """讀凍結場景目錄(每張一個子資料夾,含 rgb/semantic/instance/meta)。"""

    def __init__(self, root: str | os.PathLike, scene_ids: list[int] | None = None):
        self.root = Path(root)
        if scene_ids is None:
            scene_ids = sorted(
                int(p.name) for p in self.root.iterdir()
                if p.is_dir() and p.name.isdigit()
            )
        self.scene_ids = scene_ids

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        sid = self.scene_ids[idx]
        d = self.root / f"{sid:05d}"
        rgb = np.array(Image.open(d / "rgb.png").convert("RGB"))
        semantic = np.array(Image.open(d / "semantic_mask.png"))
        instance = np.array(Image.open(d / "instance_mask.png"))
        dr_path = d / "defect_region.png"
        defect_region = (np.array(Image.open(dr_path).convert("L")) > 127).astype(np.uint8) \
            if dr_path.exists() else None
        with open(d / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        return encode_rgb(rgb), encode_targets(semantic, instance, meta, defect_region)


if __name__ == "__main__":
    ds = RuntimeSceneDataset(length=8)
    rgb_t, targets = ds[0]
    print(f"rgb: {tuple(rgb_t.shape)} {rgb_t.dtype} range=[{rgb_t.min():.2f},{rgb_t.max():.2f}]")
    for k, v in targets.items():
        uniq = torch.unique(v).tolist()
        print(f"  target {k}: {tuple(v.shape)} {v.dtype} uniq={uniq[:8]}{'...' if len(uniq) > 8 else ''}")
