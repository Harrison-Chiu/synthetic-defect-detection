"""
Sanity check: Remesh Sharp 強度候選

目的：在正式重定 remesh_heavy 的 octree_depth 前，目視比較不同 octree_depth 視覺差異。
Harrison 從候選裡挑「比舊 heavy 更明顯破碎、但還看得出是螺絲」的當新 heavy。

舊 Stage 3 設定：
    remesh_light: octree 7-8（輕微多邊形化）
    remesh_heavy: octree 5-6（明顯塊狀破損）← 視覺只達 IoU 0.28

候選新值：octree 3 / 4 / 5（越小越破碎）

執行：MCP 或 Blender Script Editor
輸出：output/sanity/remesh/octree{N}_az{az}.png
"""

import bpy, os, math
from mathutils import Vector

BASE_DIR   = r"D:\Harrison\中山\大四下\深度學習期末報告"
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "sanity", "remesh")
HDRI_DIR   = os.path.join(BASE_DIR, "assets", "hdri")
HDRI_FILE  = "university_workshop_4k.exr"

CAMERA_RADIUS = 0.08
CAMERA_FL_MM  = 85
RENDER_RES    = 256
ELEVATION     = 10

AZIMUTHS = [0, 90]
OCTREE_CANDIDATES = [3, 4, 5, 6]   # 5/6 對應舊 heavy；3/4 是新候選
REMESH_SCALE = 0.99


def setup_render():
    s = bpy.context.scene
    s.render.engine = "BLENDER_EEVEE"
    s.render.resolution_x = RENDER_RES
    s.render.resolution_y = RENDER_RES
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGBA"
    s.render.film_transparent = True
    s.camera = bpy.data.objects["RenderCam"]
    cam = bpy.data.objects["RenderCam"]
    cam.data.clip_start = 0.001
    cam.data.clip_end = 10.0
    cam.data.lens = CAMERA_FL_MM


def set_camera(az_deg, el_deg):
    cam = bpy.data.objects["RenderCam"]
    az = math.radians(az_deg); el = math.radians(el_deg); r = CAMERA_RADIUS
    cam.location = Vector((r * math.cos(el) * math.cos(az),
                           r * math.cos(el) * math.sin(az),
                           r * math.sin(el)))
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()


def set_hdri(fname):
    env = bpy.context.scene.world.node_tree.nodes.get("HDRI_node")
    img = bpy.data.images.load(os.path.join(HDRI_DIR, fname), check_existing=True)
    env.image = img


def show_only_pan_head():
    for p in ["socket_head", "pan_head", "hex_nut", "flange_nut"]:
        o = bpy.data.objects.get(p)
        if o:
            o.hide_render = (p != "pan_head")
            o.hide_viewport = (p != "pan_head")
    pan = bpy.data.objects["pan_head"]
    pan.location = (0, 0, 0)
    pan.rotation_euler = (0, 0, 0)


def clear_defect_modifiers(obj):
    for m in list(obj.modifiers):
        if m.name.startswith("Defect"):
            obj.modifiers.remove(m)


def apply_remesh(obj, octree_depth):
    clear_defect_modifiers(obj)
    m = obj.modifiers.new("DefectRemesh", "REMESH")
    m.mode = "SHARP"
    m.octree_depth = octree_depth
    m.scale = REMESH_SCALE


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_render()
    show_only_pan_head()
    set_hdri(HDRI_FILE)

    pan = bpy.data.objects["pan_head"]
    scene = bpy.context.scene

    # 對照 normal
    clear_defect_modifiers(pan)
    for az in AZIMUTHS:
        set_camera(az, ELEVATION)
        scene.render.filepath = os.path.join(OUTPUT_DIR, f"normal_az{az:03d}.png")
        pan.update_tag(refresh={"OBJECT", "DATA"})
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)

    # Remesh 候選
    count = 0
    for octree in OCTREE_CANDIDATES:
        for az in AZIMUTHS:
            set_camera(az, ELEVATION)
            apply_remesh(pan, octree)
            fname = f"octree{octree}_az{az:03d}.png"
            scene.render.filepath = os.path.join(OUTPUT_DIR, fname)
            pan.update_tag(refresh={"OBJECT", "DATA"})
            bpy.context.view_layer.update()
            bpy.ops.render.render(write_still=True)
            count += 1
            print(f"  [{count}] {fname}")

    clear_defect_modifiers(pan)
    print(f"\nDone. Output → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
