"""
src.render — Blender 渲染側(patch 預渲、分類 grid、程序背景)

⚠️ 跨兩個不相容環境,故 __init__ **只 import 純 config**(無 bpy/PIL/cv2 依賴):
  - blender_setup / defects / patches / classify → `import bpy`,只能在 **Blender 內** import。
  - backgrounds → 需 PIL/cv2,只能在 **conda(dl_final)** import。
請依環境直接 import 對應子模組,不要期待此處 re-export 它們。
"""

from src.render.config import DEFAULT_RENDER_CONFIG, DEFECT_STATES, PARTS, RenderConfig

__all__ = ["DEFAULT_RENDER_CONFIG", "DEFECT_STATES", "PARTS", "RenderConfig"]
