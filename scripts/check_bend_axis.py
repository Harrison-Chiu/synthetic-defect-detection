"""
Sanity check: Bend axis 對齊相機 (Empty-based)

目的：在正式改 render_pan_head.py 前，目視確認 Empty-based bend axis 邏輯
讓 bend 弧落在 image plane、剪影看得到彎曲。

執行方式（在 Blender 的 Script Editor 開啟 → Run，或 MCP 跑）：
    需先載入 blender/114-2_DLcourse_FinalProject-01.blend

輸出：output/sanity/bend_axis/*.png
命名：bend_{state}_az{az}_jit{jitter:+d}.png
"""

import bpy, os, math
from mathutils import Vector

BASE_DIR   = r"D:\Harrison\中山\大四下\深度學習期末報告"
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "sanity", "bend_axis")
HDRI_DIR   = os.path.join(BASE_DIR, "assets", "hdri")
HDRI_FILE  = "university_workshop_4k.exr"

CAMERA_RADIUS = 0.08
CAMERA_FL_MM  = 85
RENDER_RES    = 256
ELEVATION     = 10  # 固定中等仰角

# 4 個 az 探索 — 確認 bend 在所有方位都看得見
AZIMUTHS = [0, 90, 180, 270]

# Bend 強度
BEND_CASES = [
    ("bend_light", 15.0),
    ("bend_heavy", 35.0),
]

# Jitter 樣本：0 (理想), +30 (最大正偏), -30 (最大負偏)
JITTERS_DEG = [0, +30, -30]


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


def apply_bend_aligned(obj, az_deg, jit_deg, angle_deg):
    """
    讓 bend 弧在 image plane 內可見：
    - 旋轉物件本身 az_deg + jit_deg 度繞 Z 軸
    - 使用 deform_axis = "X"，這樣 bend 軸（物件 local X）轉到世界 (cos α, sin α, 0)
    - tip 位移方向 ≈ A × Z 在 image 水平方向 → 可見
    """
    clear_defect_modifiers(obj)
    alpha = math.radians(az_deg + jit_deg)
    obj.rotation_euler = (0, 0, alpha)
    m = obj.modifiers.new("DefectBend", "SIMPLE_DEFORM")
    m.deform_method = "BEND"
    m.deform_axis = "X"
    m.angle = math.radians(angle_deg)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_render()
    show_only_pan_head()
    set_hdri(HDRI_FILE)

    pan = bpy.data.objects["pan_head"]
    scene = bpy.context.scene

    # ── 同時也渲一張 normal 當對照組 ──
    clear_defect_modifiers(pan)
    pan.rotation_euler = (0, 0, 0)
    for az in AZIMUTHS:
        set_camera(az, ELEVATION)
        scene.render.filepath = os.path.join(OUTPUT_DIR, f"normal_az{az:03d}.png")
        pan.update_tag(refresh={"OBJECT", "DATA"})
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)

    # ── Bend cases ──
    count = 0
    for state, angle_deg in BEND_CASES:
        for az in AZIMUTHS:
            set_camera(az, ELEVATION)
            for jit_deg in JITTERS_DEG:
                apply_bend_aligned(pan, az, jit_deg, angle_deg)

                fname = f"{state}_az{az:03d}_jit{jit_deg:+03d}.png"
                scene.render.filepath = os.path.join(OUTPUT_DIR, fname)
                pan.update_tag(refresh={"OBJECT", "DATA"})
                bpy.context.view_layer.update()
                bpy.ops.render.render(write_still=True)
                count += 1
                print(f"  [{count}] {fname}")

    clear_defect_modifiers(pan)
    pan.rotation_euler = (0, 0, 0)
    print(f"\nDone. Output → {OUTPUT_DIR}")
    print(f"Normal: {len(AZIMUTHS)} 張 | Bend: {count} 張")


if __name__ == "__main__":
    main()
