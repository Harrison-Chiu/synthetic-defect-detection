"""
assets.py — 生成器的素材載入(零件 patch 池 + 背景池)

把 composite.py 的 load_part_index_by_hdri / load_background_pools 抽出來,
並去掉寫死的 BASE_DIR:路徑改由 repo root 推導 + 可覆寫。
素材只載入一次(回傳輕量索引:檔案路徑清單),不在這裡讀圖。

目錄名已正名為 `output/patches/`(原 `parts_stage2`,Task #8 完成)。
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.data.config import DEFECT_STATES_DEFECTIVE

# repo root = 本檔的 .../src/data/assets.py 往上三層
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PARTS_DIR = REPO_ROOT / "output" / "patches"
DEFAULT_BG_REAL_DIR = REPO_ROOT / "assets" / "backgrounds_real"
DEFAULT_BG_PROC_DIR = REPO_ROOT / "assets" / "backgrounds_procedural"

_HDRI_RE = re.compile(r"_h(\d+)_")
_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


def _hdri_idx_from_filename(path: str) -> int:
    m = _HDRI_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"檔名無法解析 hdri index: {path}")
    return int(m.group(1))


@dataclass(frozen=True)
class Assets:
    """生成器需要的素材索引(只存路徑,不存影像)。

    part_buckets: {hdri_idx: {"normal": [(path, state), ...], "defect": [...]}}
    bg_pools:     {"real": [path, ...], "procedural": [path, ...]}
    """
    part_buckets: dict[int, dict[str, list[tuple[str, str]]]]
    bg_pools: dict[str, list[str]]

    @property
    def hdri_keys(self) -> list[int]:
        return sorted(self.part_buckets.keys())


def load_part_index_by_hdri(parts_dir: str | os.PathLike = DEFAULT_PARTS_DIR):
    """掃 parts 目錄,按 hdri_idx 分桶,再分 normal / defect。"""
    parts_dir = str(parts_dir)
    buckets: dict[int, dict[str, list[tuple[str, str]]]] = {}
    for state_dir in sorted(os.listdir(parts_dir)):
        full = os.path.join(parts_dir, state_dir)
        if not os.path.isdir(full):
            continue
        key = "defect" if state_dir in DEFECT_STATES_DEFECTIVE else "normal"
        for png in sorted(glob.glob(os.path.join(full, "*.png"))):
            if png.endswith("_defectmask.png"):
                continue  # S5 離線產的變形區遮罩,不是零件 patch
            h = _hdri_idx_from_filename(png)
            buckets.setdefault(h, {"normal": [], "defect": []})
            buckets[h][key].append((png, state_dir))
    if not buckets:
        raise FileNotFoundError(f"在 {parts_dir} 找不到任何零件 patch")
    return buckets


def defectmask_path_for(part_path: str) -> str:
    """由 defect patch 路徑推出對應的變形區遮罩路徑(gen_defectmask.py 產)。

    `.../<state>/<pose>_<state>.png` → `.../<state>/<pose>_<state>_defectmask.png`
    normal patch 無變形區,呼叫端自行判斷(is_defective)後才查。
    """
    return part_path[:-4] + "_defectmask.png"


def load_background_pools(
    bg_real_dir: str | os.PathLike = DEFAULT_BG_REAL_DIR,
    bg_proc_dir: str | os.PathLike = DEFAULT_BG_PROC_DIR,
):
    """回傳 {'real': [...], 'procedural': [...]}(solid 由生成器程序產生,不在此)。"""
    pools: dict[str, list[str]] = {"real": [], "procedural": []}
    for ext in _IMG_EXTS:
        pools["real"].extend(glob.glob(os.path.join(str(bg_real_dir), ext)))
        pools["procedural"].extend(glob.glob(os.path.join(str(bg_proc_dir), ext)))
    pools["real"] = sorted(set(pools["real"]))
    pools["procedural"] = sorted(set(pools["procedural"]))
    return pools


def load_assets(
    parts_dir: str | os.PathLike = DEFAULT_PARTS_DIR,
    bg_real_dir: str | os.PathLike = DEFAULT_BG_REAL_DIR,
    bg_proc_dir: str | os.PathLike = DEFAULT_BG_PROC_DIR,
) -> Assets:
    return Assets(
        part_buckets=load_part_index_by_hdri(parts_dir),
        bg_pools=load_background_pools(bg_real_dir, bg_proc_dir),
    )
