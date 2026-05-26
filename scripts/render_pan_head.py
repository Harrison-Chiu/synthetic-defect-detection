"""
Stage 2 — pan_head 單零件渲染（含 Bend / Displace 瑕疵變體）

每 (elevation, azimuth, HDRI, defect_state) 組合渲一張，總 6×3×2×5 = 180 張。
所有 defect 參數透過 deterministic seed 隨機化，整體可復現。

執行方式：
    1) Blender Script Editor 開啟此檔，按 ▶ Run Script
    2) 或 background mode：
       blender --background "blender/114-2_DLcourse_FinalProject-01.blend" \
               --python "scripts/render_pan_head.py"
"""

import bpy
import math
import os
import json
import random
from mathutils import Vector

# ─────────────────────────────────────────────
# 路徑設定
# ─────────────────────────────────────────────
BASE_DIR   = r"D:\Harrison\中山\大四下\深度學習期末報告"
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "parts_stage2")
HDRI_DIR   = os.path.join(BASE_DIR, "assets", "hdri")

# ─────────────────────────────────────────────
# 採樣參數（grid，無 random）
# ─────────────────────────────────────────────
PART = "pan_head"

ELEVATIONS = [-60, -30, -10, 10, 30, 60]   # 6 個，含仰視
AZIMUTHS   = [0, 120, 240]                  # 3 個
HDRIS      = ["university_workshop_4k.exr", "crossfit_gym_4k.exr"]  # 2 個
# Stage 3：新增 remesh_light / remesh_heavy 取代 Bevel（Bevel 視覺幾乎無效）
DEFECT_STATES = [
    "normal",
    "bend_light", "bend_heavy",
    "displace_light", "displace_heavy",
    "remesh_light", "remesh_heavy",
]

# Defect 參數範圍（內部 random 由 seed 控制）
BEND_LIGHT_RANGE     = (10, 20)         # degrees（從 5–15 提高，5° 在 256 解析度幾乎看不到）
BEND_HEAVY_RANGE     = (25, 45)
# Stage 3：displace 強度翻倍（Stage 2 light=0.05–0.15、heavy=0.20–0.40 視覺效果過弱）
DISPLACE_LIGHT_RANGE = (0.15, 0.30)    # OBJECT LOCAL units, NOT world meters
DISPLACE_HEAVY_RANGE = (0.40, 0.80)
# Remesh Sharp：voxel 化造成「塊狀破損」感。octree_depth 越小越破碎
REMESH_LIGHT_OCTREE_RANGE = (7, 8)     # 表面變粗糙、輕微多邊形化
REMESH_HEAVY_OCTREE_RANGE = (5, 6)     # 明顯塊狀破損
REMESH_SCALE              = 0.99       # 0.99 接近原大小，1.0 會跟 boundary 同步
# 重要：Displace strength 是物件 local space 單位，不是世界 meter
# 我們零件 scale=0.001，local 1 unit = 1mm 世界，所以這裡 0.1 ≈ 0.1mm 世界 displacement
# pan_head 原始 3614 頂點已足夠 displace，不需 Subsurf

# 渲染參數（沿用 Stage 1，已踩過所有雷）
CAMERA_RADIUS = 0.08
CAMERA_FL_MM  = 85
RENDER_RES    = 256

# ─────────────────────────────────────────────
# Blender helpers
# ─────────────────────────────────────────────

def setup_render():
    scene = bpy.context.scene
    scene.render.engine       = "BLENDER_EEVEE"
    scene.render.resolution_x = RENDER_RES
    scene.render.resolution_y = RENDER_RES
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGBA"
    scene.render.film_transparent = True
    scene.camera = bpy.data.objects["RenderCam"]

    if hasattr(scene.eevee, "use_raytracing"):
        scene.eevee.use_raytracing = True

    cam = bpy.data.objects["RenderCam"]
    cam.data.clip_start = 0.001
    cam.data.clip_end   = 10.0
    cam.data.lens       = CAMERA_FL_MM


def set_camera(az_deg, el_deg):
    cam = bpy.data.objects["RenderCam"]
    az = math.radians(az_deg); el = math.radians(el_deg); r = CAMERA_RADIUS
    cam.location = Vector((r * math.cos(el) * math.cos(az),
                           r * math.cos(el) * math.sin(az),
                           r * math.sin(el)))
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()


def set_hdri(hdri_filename):
    env_node = bpy.context.scene.world.node_tree.nodes.get("HDRI_node")
    if env_node is None:
        raise RuntimeError("找不到 HDRI_node，先在 Blender 場景接好 World HDRI")
    img = bpy.data.images.load(os.path.join(HDRI_DIR, hdri_filename), check_existing=True)
    env_node.image = img


def show_only_pan_head():
    for p in ["socket_head", "pan_head", "hex_nut", "flange_nut"]:
        o = bpy.data.objects.get(p)
        if o:
            o.hide_render   = (p != PART)
            o.hide_viewport = (p != PART)
    pan = bpy.data.objects[PART]
    pan.location = (0, 0, 0)
    pan.rotation_euler = (0, 0, 0)


# ─────────────────────────────────────────────
# Defect modifier 自動化
# ─────────────────────────────────────────────

def clear_defect_modifiers(obj):
    for m in list(obj.modifiers):
        if m.name.startswith("Defect"):
            obj.modifiers.remove(m)


def get_or_create_clouds_texture(name="DefectCloudsTex", noise_scale=0.5):
    """CLOUDS 紋理：比 NOISE 大顆，displace 視覺效果明顯"""
    tex = bpy.data.textures.get(name)
    if tex is None:
        tex = bpy.data.textures.new(name, type="CLOUDS")
    tex.noise_scale = noise_scale
    return tex


def apply_defect(obj, defect_state, rng):
    """套用 defect modifier，回傳 param dict（供 metadata 記錄）"""
    clear_defect_modifiers(obj)

    if defect_state == "normal":
        return {}

    if defect_state.startswith("bend"):
        rng_range = BEND_LIGHT_RANGE if defect_state == "bend_light" else BEND_HEAVY_RANGE
        angle_deg = rng.uniform(*rng_range)
        axis      = rng.choice(["X", "Y"])
        m = obj.modifiers.new("DefectBend", "SIMPLE_DEFORM")
        m.deform_method = "BEND"
        m.deform_axis   = axis
        m.angle         = math.radians(angle_deg)
        return {"axis": axis, "angle_deg": round(angle_deg, 3)}

    if defect_state.startswith("displace"):
        rng_range = DISPLACE_LIGHT_RANGE if defect_state == "displace_light" else DISPLACE_HEAVY_RANGE
        strength    = rng.uniform(*rng_range)
        noise_scale = rng.uniform(0.55, 0.85)
        tex = get_or_create_clouds_texture(noise_scale=noise_scale)
        m = obj.modifiers.new("DefectDisplace", "DISPLACE")
        m.texture   = tex
        m.strength  = strength
        m.mid_level = 0.5
        m.direction = "NORMAL"
        return {"strength": round(strength, 6), "noise_scale": round(noise_scale, 3)}

    if defect_state.startswith("remesh"):
        rng_range = (REMESH_LIGHT_OCTREE_RANGE if defect_state == "remesh_light"
                     else REMESH_HEAVY_OCTREE_RANGE)
        octree_depth = rng.randint(rng_range[0], rng_range[1])  # inclusive both ends
        m = obj.modifiers.new("DefectRemesh", "REMESH")
        m.mode         = "SHARP"
        m.octree_depth = octree_depth
        m.scale        = REMESH_SCALE
        # threshold/sharpness 用預設即可（threshold=1.0, sharpness=1.0）
        return {"octree_depth": int(octree_depth), "scale": REMESH_SCALE}

    raise ValueError(f"Unknown defect_state: {defect_state}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    setup_render()
    show_only_pan_head()
    pan   = bpy.data.objects[PART]
    scene = bpy.context.scene

    # 建立 output 子目錄
    for state in DEFECT_STATES:
        os.makedirs(os.path.join(OUTPUT_DIR, state), exist_ok=True)

    records = []
    total   = len(ELEVATIONS) * len(AZIMUTHS) * len(HDRIS) * len(DEFECT_STATES)
    idx     = 0

    for el in ELEVATIONS:
        for az in AZIMUTHS:
            set_camera(az, el)
            for hdri_idx, hdri_file in enumerate(HDRIS):
                set_hdri(hdri_file)
                for defect_state in DEFECT_STATES:
                    idx += 1
                    # Deterministic seed for defect param randomization
                    seed = abs(hash((el, az, hdri_idx, defect_state))) & 0xFFFFFFFF
                    rng  = random.Random(seed)
                    params = apply_defect(pan, defect_state, rng)

                    fname = f"pan_head_az{az:03d}_el{el:+04d}_h{hdri_idx}_{defect_state}.png"
                    fpath = os.path.join(OUTPUT_DIR, defect_state, fname)
                    scene.render.filepath = fpath
                    # 強制標 dirty + view_layer update，否則 displace modifier 在連續 render 下會被
                    # EEVEE/depsgraph 漏掉，渲出跟 normal 一模一樣的圖
                    pan.update_tag(refresh={"OBJECT", "DATA"})
                    bpy.context.view_layer.update()
                    bpy.ops.render.render(write_still=True)

                    records.append({
                        "filename":     os.path.join(defect_state, fname).replace("\\", "/"),
                        "elevation":    el,
                        "azimuth":      az,
                        "hdri":         hdri_file.replace("_4k.exr", ""),
                        "defect_state": defect_state,
                        "defect_params": params,
                        "seed":         seed,
                    })
                    if idx % 20 == 0 or idx == total:
                        print(f"  [{idx}/{total}] {fname}")

    clear_defect_modifiers(pan)

    meta_path = os.path.join(OUTPUT_DIR, "parts_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(records)} renders → {OUTPUT_DIR}")
    print(f"Metadata → {meta_path}")


if __name__ == "__main__":
    main()
