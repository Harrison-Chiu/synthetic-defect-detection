"""
classify.py — Stage 1 分類資料集 grid 渲染(4 零件 × 視角 × HDRI)→ output/dataset/raw/

移植 render_stage1.py。pure grid、無 random、完全可復現。產出 labels/metadata.csv。
屬早期分類資料(現行管線已轉 patch + runtime 合成),保留以便重生。

在 Blender 內執行(同 patches.py)。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import bpy  # noqa: E402

from src.render.blender_setup import configure_render, set_camera, set_hdri, show_only  # noqa: E402
from src.render.config import CLASS_ID, DEFAULT_RENDER_CONFIG, PARTS, RenderConfig  # noqa: E402


def render_classification(cfg: RenderConfig = DEFAULT_RENDER_CONFIG):
    configure_render(cfg)
    scene = bpy.context.scene
    out_dir = Path(cfg.classify_dir)
    records = []

    for part in PARTS:
        (out_dir / part).mkdir(parents=True, exist_ok=True)
        show_only(part)
        for el in cfg.classify_elevations:
            for az in cfg.classify_azimuths:
                set_camera(az, el, cfg)
                for hdri_idx, hdri_file in enumerate(cfg.hdris):
                    set_hdri(hdri_file, cfg)
                    fname = f"{part}_az{az:03d}_el{el:+04d}_h{hdri_idx}.png"
                    scene.render.filepath = str(out_dir / part / fname)
                    bpy.ops.render.render(write_still=True)
                    records.append({
                        "filename": f"raw/{part}/{fname}",
                        "class_id": CLASS_ID[part], "class_name": part,
                        "azimuth_deg": az, "elevation_deg": el,
                        "hdri_name": hdri_file.replace("_4k.exr", ""), "split": "",
                    })

    labels_dir = Path(cfg.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    csv_path = labels_dir / "metadata.csv"
    fieldnames = ["filename", "class_id", "class_name", "azimuth_deg", "elevation_deg", "hdri_name", "split"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Done. {len(records)} renders → {csv_path}")
    return len(records)


if __name__ == "__main__":
    render_classification()
