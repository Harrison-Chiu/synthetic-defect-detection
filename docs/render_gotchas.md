# Render Gotchas 排查清單

> 跑批次渲染前要逐項確認的點。每項註明：誰處理、目前狀態、判斷方式。

---

## 必修項目

### ✓ 相機 near clip 把零件裁掉（已踩到）
- **症狀**：渲出全空白圖（透明背景＋零件不見）
- **原因**：相機預設 `clip_start = 0.1m`，CAMERA_RADIUS=0.08m，零件全部在 near plane 之內被裁掉
- **處理**：`cam.data.clip_start = 0.001`（render.py 已加）
- **教訓**：MVP 用 macro scale（公分等級）拍小零件時，相機 clip 一定要先調

### ✓ Blender 5.x 引擎名稱
- **症狀**：`scene.render.engine = "BLENDER_EEVEE_NEXT"` 報 enum not found
- **原因**：Blender 5.x 已把 EEVEE Next 合併回叫 `BLENDER_EEVEE`（Blender 4.2–4.5 才是 `BLENDER_EEVEE_NEXT`）
- **處理**：用 `BLENDER_EEVEE`（render.py 已改）

### ✓ HDRI 入鏡污染背景
- **風險**：模型學到 HDRI 紋路當零件特徵，分類結果不可信
- **處理**：`scene.render.film_transparent = True`（render.py 已加）
  - 同時建議 Harrison 在 Blender UI 勾掉 World → Ray Visibility → Camera 作為本機預設
- **驗證**：渲出的 PNG 開起來背景是透明（棋盤格圖案）

### ✓ EEVEE Next 金屬反射
- **風險**：不鏽鋼看起來像啞光塑膠
- **處理**：`scene.eevee.use_raytracing = True`（render.py 已加）
- **驗證**：測試渲染圖看金屬感是否合理

---

## 待視覺確認

### ☐ 構圖大小
- **參數**：CAMERA_RADIUS=0.08m, CAMERA_FL_MM=85mm
- **預計**：零件佔 frame 40–60%
- **判斷**：測試圖看零件是否清晰、不會太小或被切到
- **若需調**：改 render.py 的 `CAMERA_RADIUS`（變大→零件變小）或 `CAMERA_FL_MM`（變大→放大）

### ☐ 極端 elevation 退化
- **el=5°**：近側視，螺絲應看到完整側面、螺母看到邊緣
- **el=80°**：近俯視，螺絲變一個小點？螺母只看到頂面？
- **判斷**：兩端是否提供足夠形狀資訊；如果太退化，調整 ELEVATIONS 範圍

### ☐ HDRI 光照亮度
- **判斷**：太亮（過曝、細節消失）或太暗（黑成一片）
- **若需調**：World shader 的 Background strength（目前未設定，預設 1.0）

### ☐ 4 種 HDRI 的光照一致性
- **風險**：某一張 HDRI 過亮/過暗，導致該光照下的圖訓練上有偏
- **判斷**：渲一輪 4 種 HDRI 對比

---

## 應該沒問題（已驗證或低風險）

| 項目 | 為何安心 |
|------|---------|
| 含中文路徑寫 PNG | Blender Windows 用 UTF-8，第一張成功就驗證了 |
| 渲染時間 | EEVEE 256×256 應 <1s/張，1152 張預估 ~20min |
| `stainless_steel` 材質共用 4 零件 | 一致性是好的，不需改 |
| 球座標 azimuth 對應軸向 | 形狀視覺等價，不影響訓練 |
| 物件隱藏邏輯 | `hide_render` + `hide_viewport` 都設了 |

---

## 進度

- [x] film_transparent 加入 render.py
- [x] EEVEE raytracing 加入 render.py
- [x] 相機 clip_start 調為 0.001m
- [x] Blender 5.x 引擎名改回 BLENDER_EEVEE
- [x] 測試渲 8 張（4 零件 × 2 極端 el × 1 az × 1 hdri）→ 全部清晰
- [ ] Harrison 視覺判斷：構圖、亮度、反射真實感
- [ ] 必要時調整參數，重渲測試
- [ ] 跑完整 1152 張
