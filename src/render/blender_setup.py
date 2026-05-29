"""
blender_setup.py — 共用 bpy 場景設定(camera / hdri / render / 可見性)

整併 render_stage1.py 與 render_pan_head.py 重複的 configure_render / set_camera /
set_hdri / show_only。**只在 Blender 內可 import**(import bpy)。

已踩雷(見 docs/render_gotchas.md):
- 相機 near clip 必須 < camera_radius,否則零件被裁成空白。
- Blender 5.x 引擎名是 BLENDER_EEVEE(不是 _NEXT)。
- HDRI 入鏡需 film_transparent=True。
"""

from __future__ import annotations

import math

import bpy
from mathutils import Vector

from src.render.config import DEFAULT_RENDER_CONFIG, PARTS, RenderConfig


def configure_render(cfg: RenderConfig = DEFAULT_RENDER_CONFIG):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = cfg.render_res
    scene.render.resolution_y = cfg.render_res
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.camera = bpy.data.objects["RenderCam"]
    if hasattr(scene.eevee, "use_raytracing"):
        scene.eevee.use_raytracing = True
    cam = bpy.data.objects["RenderCam"]
    cam.data.clip_start = cfg.clip_start
    cam.data.clip_end = cfg.clip_end
    cam.data.lens = cfg.camera_fl_mm


def set_camera(az_deg, el_deg, cfg: RenderConfig = DEFAULT_RENDER_CONFIG):
    cam = bpy.data.objects["RenderCam"]
    az, el, r = math.radians(az_deg), math.radians(el_deg), cfg.camera_radius
    cam.location = Vector((r * math.cos(el) * math.cos(az),
                           r * math.cos(el) * math.sin(az),
                           r * math.sin(el)))
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = cfg.camera_fl_mm


def set_hdri(hdri_filename, cfg: RenderConfig = DEFAULT_RENDER_CONFIG):
    env_node = bpy.context.scene.world.node_tree.nodes.get("HDRI_node")
    if env_node is None:
        raise RuntimeError("找不到 HDRI_node,先在 Blender 場景接好 World HDRI 節點")
    img = bpy.data.images.load(str(cfg.hdri_dir / hdri_filename), check_existing=True)
    env_node.image = img


def show_only(part_name, pose_euler_deg=(0, 0, 0)):
    """只顯示指定零件,其餘隱藏,並移到原點 + 指定 pose。"""
    for p in PARTS:
        o = bpy.data.objects.get(p)
        if o:
            o.hide_render = (p != part_name)
            o.hide_viewport = (p != part_name)
    obj = bpy.data.objects[part_name]
    obj.location = (0, 0, 0)
    rx, ry, rz = pose_euler_deg
    obj.rotation_euler = (math.radians(rx), math.radians(ry), math.radians(rz))
