"""
Blender batch render script — 深度學習期末專題 合成資料集 (MVP)
Usage:
    blender --background "blender/114-2_DLcourse_FinalProject-01.blend" --python scripts/render.py
Or run inside Blender's Script Editor (skip the --background flag).

Pure grid sampling, no randomness — 完全可復現。
"""

import bpy
import math
import csv
import os
from mathutils import Vector

# ─────────────────────────────────────────────
# 路徑設定
# ─────────────────────────────────────────────
BASE_DIR   = r"D:\Harrison\中山\大四下\深度學習期末報告"
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "dataset", "raw")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
HDRI_DIR   = os.path.join(BASE_DIR, "assets", "hdri")

# ─────────────────────────────────────────────
# 渲染參數（pure grid）
# ─────────────────────────────────────────────
PARTS = ["socket_head", "pan_head", "hex_nut", "flange_nut"]

CLASS_ID = {
    "socket_head": 0,
    "pan_head":    1,
    "hex_nut":     2,
    "flange_nut":  3,
}

# 相機 elevation：12 個，覆蓋上下半球，避開 0° 退化、±90° degenerate
# 涵蓋仰視角度可看到 flange_nut 凸緣底面，提供獨特鑑別資訊
ELEVATIONS = [-85, -70, -55, -40, -25, -10, 10, 25, 40, 55, 70, 85]

# 相機 azimuth：3 個 (每 120°)，零件繞主軸近對稱，azimuth 主要貢獻光照變化
AZIMUTHS = [0, 120, 240]

HDRI_FILES = [
    "university_workshop_4k.exr",
    "crossfit_gym_4k.exr",
    "monochrome_studio_02_4k.exr",
    "pretoria_gardens_4k.exr",
]

# 零件統一固定 canonical pose（直立，無旋轉）
PART_POSE_EULER_DEG = (0, 0, 0)

CAMERA_RADIUS = 0.08    # 相機到原點距離 (m)
CAMERA_FL_MM  = 85      # 焦距 (mm)
RENDER_RES    = 256     # 渲染解析度

# ─────────────────────────────────────────────
# 內部工具函式
# ─────────────────────────────────────────────

def set_hdri(hdri_filename):
    """切換 World shader 的 HDRI 貼圖。"""
    path = os.path.join(HDRI_DIR, hdri_filename)
    env_node = bpy.context.scene.world.node_tree.nodes.get("HDRI_node")
    if env_node is None:
        raise RuntimeError("找不到 HDRI_node，請先在 Blender 場景中接好 World HDRI 節點")
    img = bpy.data.images.load(path, check_existing=True)
    env_node.image = img


def set_camera(azimuth_deg, elevation_deg):
    """依球座標移動 RenderCam 並對準原點。"""
    cam = bpy.data.objects["RenderCam"]
    az  = math.radians(azimuth_deg)
    el  = math.radians(elevation_deg)
    r   = CAMERA_RADIUS

    x = r * math.cos(el) * math.cos(az)
    y = r * math.cos(el) * math.sin(az)
    z = r * math.sin(el)
    cam.location = Vector((x, y, z))

    direction = Vector((0, 0, 0)) - cam.location
    rot_quat  = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()
    cam.data.lens = CAMERA_FL_MM


def show_only(part_name):
    """只顯示指定零件，隱藏其他三個，並移到原點。"""
    for p in PARTS:
        obj = bpy.data.objects.get(p)
        if obj:
            obj.hide_render   = (p != part_name)
            obj.hide_viewport = (p != part_name)
    obj = bpy.data.objects[part_name]
    obj.location = (0, 0, 0)
    rx, ry, rz = PART_POSE_EULER_DEG
    obj.rotation_euler = (math.radians(rx), math.radians(ry), math.radians(rz))


def configure_render():
    """設定渲染引擎、解析度、透明背景、EEVEE 反射、相機 clip。"""
    scene = bpy.context.scene
    # Blender 5.x 已把 EEVEE Next 合併回 BLENDER_EEVEE
    scene.render.engine       = "BLENDER_EEVEE"
    scene.render.resolution_x = RENDER_RES
    scene.render.resolution_y = RENDER_RES
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGBA"   # 含 alpha
    scene.render.film_transparent = True               # HDRI 不入鏡（但保留打光/反射）
    scene.camera = bpy.data.objects["RenderCam"]

    # EEVEE 金屬反射需要 raytracing（Blender 4.2+/5.x）
    if hasattr(scene.eevee, "use_raytracing"):
        scene.eevee.use_raytracing = True

    # 相機 near clip 必須小於 CAMERA_RADIUS，否則零件被裁掉
    cam = bpy.data.objects["RenderCam"]
    cam.data.clip_start = 0.001
    cam.data.clip_end   = 10.0


# ─────────────────────────────────────────────
# 主渲染迴圈
# ─────────────────────────────────────────────

def main():
    configure_render()
    scene   = bpy.context.scene
    records = []

    for part_name in PARTS:
        os.makedirs(os.path.join(OUTPUT_DIR, part_name), exist_ok=True)
        show_only(part_name)

        for el in ELEVATIONS:
            for az in AZIMUTHS:
                set_camera(az, el)

                for hdri_idx, hdri_file in enumerate(HDRI_FILES):
                    set_hdri(hdri_file)
                    hdri_name = hdri_file.replace("_4k.exr", "")

                    fname = f"{part_name}_az{az:03d}_el{el:+04d}_h{hdri_idx}.png"
                    fpath = os.path.join(OUTPUT_DIR, part_name, fname)
                    scene.render.filepath = fpath

                    bpy.ops.render.render(write_still=True)

                    records.append({
                        "filename":      os.path.join("raw", part_name, fname),
                        "class_id":      CLASS_ID[part_name],
                        "class_name":    part_name,
                        "azimuth_deg":   az,
                        "elevation_deg": el,
                        "hdri_name":     hdri_name,
                        "split":         "",
                    })

    os.makedirs(LABELS_DIR, exist_ok=True)
    csv_path = os.path.join(LABELS_DIR, "metadata.csv")
    fieldnames = [
        "filename", "class_id", "class_name",
        "azimuth_deg", "elevation_deg", "hdri_name", "split",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Done. {len(records)} renders → {csv_path}")


if __name__ == "__main__":
    main()
