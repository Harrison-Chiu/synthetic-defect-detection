"""
patches.py — S6 零件 patch 預渲

渲染 grid: (elevation × azimuth × defect_state × HDRI × rotation × strength)
= 5 × 3 × 4 × 4 × 4 × 3 = 2,880 patches

迴圈順序：pose+defect 在外層，光照在內層。
同 (az, el, defect_state) 的 defect seed 固定 → modifier 只建一次，
所有光照變體共用同一組 defect 參數 → defectmask 可通用。

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
import time
from pathlib import Path

# 讓 `blender --python src/render/patches.py` 找得到 src 套件
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import bpy  # noqa: E402

from src.render.blender_setup import configure_render, set_camera, set_hdri, show_only  # noqa: E402
from src.render.config import DEFAULT_RENDER_CONFIG, DEFECT_STATES, RenderConfig  # noqa: E402
from src.render.defects import apply_defect, clear_defect_modifiers  # noqa: E402


def render_patches_s6(part: str = "pan_head", cfg: RenderConfig = DEFAULT_RENDER_CONFIG):
    """S6 完整 patch grid 渲染。回傳寫出的 record 數。

    迴圈外層 = pose + defect（modifier 只建一次）
    迴圈內層 = 光照變化（只換 HDRI/rotation/strength，不動 modifier）
    → 同 pose+defect 的所有光照共用同一組 defect 參數
    → defectmask per (az, el, defect_state) 可正確通用
    """
    configure_render(cfg)
    show_only(part)
    obj = bpy.data.objects[part]
    scene = bpy.context.scene

    out_dir = Path(cfg.patches_s6_dir)
    for state in DEFECT_STATES:
        (out_dir / state).mkdir(parents=True, exist_ok=True)

    total = (len(cfg.patch_elevations) * len(cfg.patch_azimuths) *
             len(DEFECT_STATES) *
             len(cfg.hdris) * len(cfg.hdri_rotations) * len(cfg.hdri_strengths))

    records = []
    idx = 0
    t_start = time.time()

    for el in cfg.patch_elevations:
        for az in cfg.patch_azimuths:
            set_camera(az, el, cfg)
            for defect_state in DEFECT_STATES:
                # ★ seed 只依賴 (el, az, defect_state)，不含光照
                # → 同 pose 同 defect = 同彎曲方向/同位移，defectmask 可通用
                seed = abs(hash((el, az, defect_state))) & 0xFFFFFFFF
                rng = random.Random(seed)
                params = apply_defect(obj, defect_state, az, rng, cfg)

                # 同一組 defect modifier 下，遍歷所有光照
                for hdri_idx, hdri_file in enumerate(cfg.hdris):
                    for rot in cfg.hdri_rotations:
                        for strength in cfg.hdri_strengths:
                            set_hdri(hdri_file, cfg, rotation_deg=rot, strength=strength)
                            idx += 1

                            str_tag = f"{int(strength * 10):02d}"
                            fname = (f"{part}_az{az:03d}_el{el:+04d}"
                                     f"_h{hdri_idx}_r{rot:03d}_s{str_tag}"
                                     f"_{defect_state}.png")
                            scene.render.filepath = str(out_dir / defect_state / fname)
                            obj.update_tag(refresh={"OBJECT", "DATA"})
                            bpy.context.view_layer.update()
                            bpy.ops.render.render(write_still=True)

                            records.append({
                                "filename": f"{defect_state}/{fname}",
                                "elevation": el, "azimuth": az,
                                "hdri": hdri_file.replace("_4k.exr", ""),
                                "hdri_idx": hdri_idx,
                                "hdri_rotation": rot,
                                "hdri_strength": strength,
                                "defect_state": defect_state,
                                "defect_params": params,
                                "seed": seed,
                            })
                            if idx % 200 == 0 or idx == total:
                                elapsed = time.time() - t_start
                                eta = elapsed / idx * (total - idx) if idx > 0 else 0
                                print(f"  [{idx}/{total}] {elapsed:.0f}s elapsed, "
                                      f"ETA {eta:.0f}s | {fname}")

    clear_defect_modifiers(obj)
    obj.rotation_euler = (0, 0, 0)
    # 復原 HDRI 設定
    set_hdri(cfg.hdris[0], cfg, rotation_deg=0, strength=1.0)

    meta_path = out_dir / "parts_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    elapsed = time.time() - t_start
    print(f"Done. {len(records)} renders in {elapsed:.0f}s → {out_dir}\n"
          f"Metadata → {meta_path}")
    return len(records)


if __name__ == "__main__":
    render_patches_s6()
