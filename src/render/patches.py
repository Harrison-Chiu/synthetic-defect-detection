"""
patches.py — 零件 patch 預渲(含 bend/displace/remesh 變體)→ output/patches/

移植 render_pan_head.py 的主迴圈。每 (elevation, azimuth, HDRI, defect_state) 一張;
defect 參數由 deterministic seed 隨機化,整體可復現。

在 Blender 內執行:
    1) Script Editor 開此檔按 ▶,或
    2) blender --background "blender/...blend" --python src/render/patches.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

# 讓 `blender --python src/render/patches.py` 找得到 src 套件
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import bpy  # noqa: E402

from src.render.blender_setup import configure_render, set_camera, set_hdri, show_only  # noqa: E402
from src.render.config import DEFAULT_RENDER_CONFIG, DEFECT_STATES, RenderConfig  # noqa: E402
from src.render.defects import apply_defect, clear_defect_modifiers  # noqa: E402


def render_patches(part: str = "pan_head", cfg: RenderConfig = DEFAULT_RENDER_CONFIG):
    """對單一零件跑完整 patch grid。回傳寫出的 record 數。"""
    configure_render(cfg)
    show_only(part)
    obj = bpy.data.objects[part]
    scene = bpy.context.scene

    out_dir = Path(cfg.patches_dir)
    for state in DEFECT_STATES:
        (out_dir / state).mkdir(parents=True, exist_ok=True)

    records = []
    total = len(cfg.patch_elevations) * len(cfg.patch_azimuths) * len(cfg.hdris) * len(DEFECT_STATES)
    idx = 0
    for el in cfg.patch_elevations:
        for az in cfg.patch_azimuths:
            set_camera(az, el, cfg)
            for hdri_idx, hdri_file in enumerate(cfg.hdris):
                set_hdri(hdri_file, cfg)
                for defect_state in DEFECT_STATES:
                    idx += 1
                    seed = abs(hash((el, az, hdri_idx, defect_state))) & 0xFFFFFFFF
                    rng = random.Random(seed)
                    params = apply_defect(obj, defect_state, az, rng, cfg)

                    fname = f"{part}_az{az:03d}_el{el:+04d}_h{hdri_idx}_{defect_state}.png"
                    scene.render.filepath = str(out_dir / defect_state / fname)
                    # 強制標 dirty,否則連續 render 下 displace 會被 depsgraph 漏掉
                    obj.update_tag(refresh={"OBJECT", "DATA"})
                    bpy.context.view_layer.update()
                    bpy.ops.render.render(write_still=True)

                    records.append({
                        "filename": f"{defect_state}/{fname}",
                        "elevation": el, "azimuth": az,
                        "hdri": hdri_file.replace("_4k.exr", ""),
                        "defect_state": defect_state, "defect_params": params, "seed": seed,
                    })
                    if idx % 20 == 0 or idx == total:
                        print(f"  [{idx}/{total}] {fname}")

    clear_defect_modifiers(obj)
    obj.rotation_euler = (0, 0, 0)

    meta_path = out_dir / "parts_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Done. {len(records)} renders → {out_dir}\nMetadata → {meta_path}")
    return len(records)


if __name__ == "__main__":
    render_patches()
