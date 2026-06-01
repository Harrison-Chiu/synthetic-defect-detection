"""
config.py — Blender 渲染側參數(RenderConfig)

把 render_stage1.py / render_pan_head.py 散落的渲染常數收斂成單一結構。
純資料、不 import bpy(可在任何地方 import)。數值忠實沿用 Stage 4。

grid 採樣為 pure 確定性(無 random);defect 參數的 random 由 deterministic seed 控制。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# 4 零件命名(硬依賴,見 CLAUDE.md 命名約定)
PARTS = ("socket_head", "pan_head", "hex_nut", "flange_nut")
CLASS_ID = {"socket_head": 0, "pan_head": 1, "hex_nut": 2, "flange_nut": 3}

# S6 defect_state（與 schema.STATE_CLASSES 對齊）
DEFECT_STATES = (
    "normal",
    "bend_45",
    "displace",
    "remesh",
)

# 舊版 defect states（保留供舊 script 參照）
DEFECT_STATES_LEGACY = (
    "normal",
    "bend_light", "bend_heavy",
    "displace_light", "displace_heavy",
    "remesh_light", "remesh_heavy",
)


@dataclass(frozen=True)
class RenderConfig:
    # 相機(Stage 1 起沿用,已踩過所有雷:near clip 必須 < radius)
    camera_radius: float = 0.08
    camera_fl_mm: float = 85.0
    render_res: int = 256
    clip_start: float = 0.001
    clip_end: float = 10.0

    # HDRI(Stage 4 補回 4 個;Stage 3 寫死 2 個是 bug)
    hdri_dir: Path = REPO_ROOT / "assets" / "hdri"
    hdris: tuple[str, ...] = (
        "university_workshop_4k.exr",
        "crossfit_gym_4k.exr",
        "monochrome_studio_02_4k.exr",
        "pretoria_gardens_4k.exr",
    )

    # patch 預渲 grid
    # S6: elevation 非線性 5 個(±30 每 10°太密,改 -30,-20,0,20,30)
    patch_elevations: tuple[int, ...] = (-30, -20, 0, 20, 30)
    patch_azimuths: tuple[int, ...] = (0, 120, 240)
    # S6: HDRI rotation(Z 軸旋轉)+ strength 變化
    hdri_rotations: tuple[int, ...] = (0, 90, 180, 270)
    hdri_strengths: tuple[float, ...] = (0.5, 1.0, 2.0)

    # 分類 grid(render_stage1:12×3 = 36 視角)
    classify_elevations: tuple[int, ...] = (-85, -70, -55, -40, -25, -10, 10, 25, 40, 55, 70, 85)
    classify_azimuths: tuple[int, ...] = (0, 120, 240)

    # S6 defect 參數（固定值，不再用 range random）
    bend_angle_deg: float = 45.0
    bend_axis_jitter_deg_range: tuple[float, float] = (-30, 30)  # 彎曲軸向仍保留 jitter
    displace_strength: float = 0.4
    displace_noise_scale: float = 0.7
    remesh_mode: str = "SMOOTH"   # S6: SHARP → SMOOTH
    remesh_octree_depth: int = 7  # S6: d5 → d7
    remesh_scale: float = 0.99

    # 舊版 defect 參數（保留供參照）
    bend_light_range: tuple[float, float] = (10, 20)
    bend_heavy_range: tuple[float, float] = (25, 45)
    displace_light_range: tuple[float, float] = (0.15, 0.30)
    displace_heavy_range: tuple[float, float] = (0.40, 0.80)
    displace_noise_scale_range: tuple[float, float] = (0.55, 0.85)
    remesh_light_octree: tuple[int, int] = (5, 5)
    remesh_heavy_octree: tuple[int, int] = (4, 4)

    # 輸出路徑
    patches_dir: Path = REPO_ROOT / "output" / "patches"       # 舊版(S5)
    patches_s6_dir: Path = REPO_ROOT / "output" / "patches_s6" # S6 新渲
    classify_dir: Path = REPO_ROOT / "output" / "dataset" / "raw"
    labels_dir: Path = REPO_ROOT / "labels"


DEFAULT_RENDER_CONFIG = RenderConfig()
