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
from src.data.config import DEFAULT_CONFIG, GenConfig
from src.data.generator import SceneSample, generate_scene


# ── 影像 / 標籤編碼(兩個 dataset 共用)──────────────────────
def encode_rgb(rgb: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 → (3,H,W) float,normalize 到 [-1,1](同 Stage 4)。"""
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (t - 0.5) / 0.5


def encode_targets(semantic: np.ndarray, instance: np.ndarray, meta: dict) -> dict[str, torch.Tensor]:
    """依 schema.HEADS 產生每個 head 的 target tensor。

    - level=PIXEL 的 part head：(instance>0) → 0/1 dense mask。
    - level=INSTANCE 的 state/type head：每實例一個標籤,廣播到該實例像素,
      實例外填 ignore_index。
    """
    part_mask = (instance > 0).astype(np.int64)

    # 預建 instance → state/type idx 查表
    state_full = np.full(instance.shape, schema.IGNORE_INDEX, dtype=np.int64)
    type_full = np.full(instance.shape, schema.IGNORE_INDEX, dtype=np.int64)
    for ins in meta["instances"]:
        iid = ins["instance_id"]
        pix = instance == iid
        if not pix.any():
            continue
        s_idx = schema.STATE_TO_IDX[ins["defect_state"]]
        state_full[pix] = s_idx
        type_full[pix] = schema.state_idx_to_type_idx(s_idx)

    by_name = {
        "part": torch.from_numpy(part_mask),
        "state": torch.from_numpy(state_full),
        "type": torch.from_numpy(type_full),
    }
    # 以 head.key 為鍵回傳,與 model.forward 輸出對齊
    out = {}
    for h in schema.HEADS:
        if h.name not in by_name:
            raise KeyError(f"encode_targets 沒有對應 head {h.name!r} 的編碼邏輯")
        out[h.key] = by_name[h.name]
    return out


def sample_to_tensors(sample: SceneSample):
    return encode_rgb(sample.rgb), encode_targets(sample.semantic, sample.instance, sample.meta)


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
        with open(d / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        return encode_rgb(rgb), encode_targets(semantic, instance, meta)


if __name__ == "__main__":
    ds = RuntimeSceneDataset(length=8)
    rgb_t, targets = ds[0]
    print(f"rgb: {tuple(rgb_t.shape)} {rgb_t.dtype} range=[{rgb_t.min():.2f},{rgb_t.max():.2f}]")
    for k, v in targets.items():
        uniq = torch.unique(v).tolist()
        print(f"  target {k}: {tuple(v.shape)} {v.dtype} uniq={uniq[:8]}{'...' if len(uniq) > 8 else ''}")
