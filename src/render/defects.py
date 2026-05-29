"""
defects.py — defect modifier 自動化(bend / displace / remesh)

移植 render_pan_head.py 的 apply_defect。**只在 Blender 內可 import**(import bpy)。

bend(見 CLAUDE.md 踩雷):用 SIMPLE_DEFORM 時直接旋轉物件繞 Z(對齊相機 az + jitter)
+ deform_axis="X",別靠 Empty origin。displace strength 是 object local 單位(零件
scale=0.001 → local 1 unit ≈ 1mm 世界)。
"""

from __future__ import annotations

import math

import bpy

from src.render.config import DEFAULT_RENDER_CONFIG, RenderConfig


def clear_defect_modifiers(obj):
    for m in list(obj.modifiers):
        if m.name.startswith("Defect"):
            obj.modifiers.remove(m)


def get_or_create_clouds_texture(name="DefectCloudsTex", noise_scale=0.5):
    """CLOUDS 紋理:比 NOISE 大顆,displace 視覺效果明顯。"""
    tex = bpy.data.textures.get(name)
    if tex is None:
        tex = bpy.data.textures.new(name, type="CLOUDS")
    tex.noise_scale = noise_scale
    return tex


def apply_defect(obj, defect_state, az_deg, rng, cfg: RenderConfig = DEFAULT_RENDER_CONFIG):
    """套 defect modifier,回傳 param dict(供 metadata)。

    bend:旋轉 obj 對齊相機 az + ±jitter + 隨機 ±方向(避免 model 學 shortcut)。
    其他 state 確保 obj.rotation_euler=(0,0,0)。
    """
    clear_defect_modifiers(obj)
    obj.rotation_euler = (0, 0, 0)

    if defect_state == "normal":
        return {}

    if defect_state.startswith("bend"):
        rng_range = cfg.bend_light_range if defect_state == "bend_light" else cfg.bend_heavy_range
        angle_deg = rng.uniform(*rng_range)
        sign = rng.choice([-1, 1])
        jit_deg = rng.uniform(*cfg.bend_axis_jitter_deg_range)
        alpha_deg = az_deg + jit_deg
        obj.rotation_euler = (0, 0, math.radians(alpha_deg))
        m = obj.modifiers.new("DefectBend", "SIMPLE_DEFORM")
        m.deform_method = "BEND"
        m.deform_axis = "X"
        m.angle = math.radians(angle_deg * sign)
        return {"angle_deg": round(angle_deg, 3), "sign": int(sign),
                "axis_jitter_deg": round(jit_deg, 3), "obj_z_rot_deg": round(alpha_deg, 3)}

    if defect_state.startswith("displace"):
        rng_range = cfg.displace_light_range if defect_state == "displace_light" else cfg.displace_heavy_range
        strength = rng.uniform(*rng_range)
        noise_scale = rng.uniform(*cfg.displace_noise_scale_range)
        tex = get_or_create_clouds_texture(noise_scale=noise_scale)
        m = obj.modifiers.new("DefectDisplace", "DISPLACE")
        m.texture = tex
        m.strength = strength
        m.mid_level = 0.5
        m.direction = "NORMAL"
        return {"strength": round(strength, 6), "noise_scale": round(noise_scale, 3)}

    if defect_state.startswith("remesh"):
        rng_range = cfg.remesh_light_octree if defect_state == "remesh_light" else cfg.remesh_heavy_octree
        octree_depth = rng.randint(rng_range[0], rng_range[1])
        m = obj.modifiers.new("DefectRemesh", "REMESH")
        m.mode = "SHARP"
        m.octree_depth = octree_depth
        m.scale = cfg.remesh_scale
        return {"octree_depth": int(octree_depth), "scale": cfg.remesh_scale}

    raise ValueError(f"Unknown defect_state: {defect_state}")
