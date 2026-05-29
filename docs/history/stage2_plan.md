# Stage 2 計畫 — 瑕疵檢測 (Defect Detection)

> 開始：2026-05-19  
> Stage 1 完成記錄見 [stage1_mvp_report.md](stage1_mvp_report.md)

---

## 目標重塑

從 Stage 1 的「4 類零件分類」改為「**單一零件的瑕疵檢測**」。

| 項目 | Stage 1 | Stage 2 |
|------|---------|---------|
| 零件 | 4 種混合 | **pan_head only** |
| 任務 | 4-class image classification | **Semantic segmentation** (3 class) |
| 模型 | 手刻 CNN 全連接到 4 class | **手刻 Encoder-Decoder** (per-pixel) |
| 場景 | 單零件、黑底 | **多零件 + 重疊 + 工業背景** |
| Pretrained? | 不用 | **不用（老師指定）** |
| 預訓練架構抄? | 不抄 | **不抄 U-Net 等 SOTA，自製** |

---

## 任務定義

**輸入**：256×256 RGB 場景圖（多 pan_head 螺絲散落在工業背景上）  
**輸出**：256×256 × 3-channel logits，每像素分類為：
- `0` = 背景
- `1` = 正常零件
- `2` = 瑕疵零件（整個瑕疵零件的像素都標 2）

**後處理可導出**：
- 連通元件 (connected components) → 每個零件實例的 bbox + mask
- 若實例內含類別 2 像素 → flagged as defective
- 整圖 Pass / Fail：是否有任何類別 2 像素

**Overlap 處理**：資料端允許重疊（更真實的工廠場景），instance 分離留給組員研究（看他們要不要加 center prediction 或 embedding）。

---

## 資料生成 Pipeline

### Phase A — Blender 端：單零件 + Modifier 變體渲染

**零件**：pan_head（沿用 Stage 1 已匯入的 mesh）

**角度採樣**（縮小範圍，因為單一零件不需 144 角度的 coverage）：

| 變數 | 取值 | 數量 |
|------|------|------|
| Elevation | -60°, -30°, -10°, 10°, 30°, 60° | 6 |
| Azimuth | 0°, 120°, 240° | 3 |
| HDRI | university_workshop, crossfit_gym | 2 |

合計 6 × 3 × 2 = **36 base poses**。

**Modifier 變體**：

| Defect State | Modifier | 參數範圍 | 數量 |
|--------------|----------|---------|------|
| normal | (無) | — | 1 |
| bend_light | Simple Deform → Bend | 角度 5°–15° (隨機) | 1 |
| bend_heavy | Simple Deform → Bend | 角度 20°–35° (隨機) | 1 |
| displace_light | Displace + Noise tex | strength 0.0005–0.0015m | 1 |
| displace_heavy | Displace + Noise tex | strength 0.002–0.004m | 1 |

每 base pose × 5 defect states = **180 渲染圖**。

**輸出**：`output/parts_stage2/{defect_state}/pan_head_az{azim:03d}_el{el:+04d}_h{hdri}_{defect_state}.png`

格式：256×256 RGBA PNG（跟 Stage 1 一致，alpha = 零件 mask）

### Phase B — Python 端：場景合成

**為什麼是 Python 不是 Blender**：composite 是 2D 影像操作，純 numpy/PIL/cv2 快兩個數量級。Blender 只負責 3D 渲染（modifier 必須在 3D 端做）。

**單場景產生流程**：
1. 隨機選 1 張背景紋理（解析度 ≥ 256×256）
2. 隨機選 3–6 張零件 PNG（從 180 張 pool 隨機抽樣）
3. 每張零件：
   - 隨機 in-plane rotation (0–360°)
   - 隨機 scale (60%–100%)
   - 隨機位置（允許重疊）
4. 用 alpha 合成到背景上
5. 同步建構 ground truth：
   - `semantic_mask`：每張零件的 alpha 範圍：normal 標 1、defective 標 2
   - `instance_mask`：每張零件依貼上順序分配 ID (1, 2, 3, ...)
   - `meta.json`：每實例的 bbox / defect_state / source_part_id

**輸出 per scene**：
```
output/scenes/{scene_id:05d}/
├── rgb.png             — 合成後 RGB 場景
├── semantic_mask.png   — 3 class 灰階 (0/1/2)
├── instance_mask.png   — 各 pixel instance ID（uint16）
└── meta.json           — instance 詳細資訊
```

**規模**：1000–2000 場景，平均 ~5 零件/場景 = 5000+ instance 樣本

### 背景紋理來源

- **AmbientCG**（[ambientcg.com](https://ambientcg.com)）— 工業紋理 CC0 免費，下載 10–20 張（混凝土、金屬板、橡膠墊）
- **Procedural Perlin**：Python 端用 `noise` 套件生成，無限張，灰階為主
- **Solid color + noise**：最簡 baseline，當 fallback

---

## 模型架構（自製，從零訓練）

**設計原則**：tailored to 256×256 / 3-class output，不抄 U-Net 等開源架構名字。但結構上「encoder-decoder + skip connection」是通用設計模式，我們自己決定 channel 數、深度、激活函式等所有細節。

### 草圖

```
Input (3, 256, 256)
  │
  ├─ Encoder（4 stage 縮放，存 skip 給 decoder）
  │    Stage 1: Conv(3→32) + Conv(32→32) → MaxPool      → 32×128×128
  │    Stage 2: Conv(32→64) + Conv(64→64) → MaxPool     → 64×64×64
  │    Stage 3: Conv(64→128) + Conv(128→128) → MaxPool  → 128×32×32
  │    Stage 4: Conv(128→256) + Conv(256→256) → MaxPool → 256×16×16
  │
  ├─ Bottleneck
  │    Conv(256→256) + Conv(256→256)                    → 256×16×16
  │
  ├─ Decoder（4 stage 放大，接 encoder skip）
  │    Stage 1: Upsample + concat skip → Conv(256+256→128) → Conv(128→128) → 128×32×32
  │    Stage 2: Upsample + concat skip → Conv(128+128→64)  → Conv(64→64)   → 64×64×64
  │    Stage 3: Upsample + concat skip → Conv(64+64→32)    → Conv(32→32)   → 32×128×128
  │    Stage 4: Upsample + concat skip → Conv(32+32→16)    → Conv(16→16)   → 16×256×256
  │
  └─ Output: Conv(16→3, kernel=1)                       → 3×256×256

Output: per-pixel 3-class logits（過 softmax 才是機率）
```

預估 5–10M 參數，4060 8GB 訓練無壓力。

### Loss
- 主：pixel-wise Cross Entropy
- 改進：class weighting（背景像素多、瑕疵像素少，weight 反比）
- 進階：可選 Dice loss 或 Focal loss（之後迭代再加）

### 訓練
- Optimizer：Adam or SGD (sgd 更接近 HW9 風格)
- LR：1e-3 起手
- Epochs：~30–50
- Batch size：8–16（看 VRAM）

### 評估
- Pixel-level：per-class IoU、mean IoU、pixel accuracy
- Instance-level（連通元件後處理後）：
  - 偵測：實例 precision / recall
  - 分類：defective vs normal accuracy

---

## Modifier 自動化（Blender script 端範例）

```python
import bpy, math, random

obj = bpy.data.objects["pan_head"]

def clear_modifiers(o):
    for m in list(o.modifiers):
        o.modifiers.remove(m)

def add_bend(o, angle_deg, axis='Z'):
    m = o.modifiers.new("DefectBend", "SIMPLE_DEFORM")
    m.deform_method = 'BEND'
    m.deform_axis = axis
    m.angle = math.radians(angle_deg)

def add_displace(o, strength, noise_scale=5.0):
    tex = bpy.data.textures.new("DefectNoise", type='NOISE')
    m = o.modifiers.new("DefectDisplace", "DISPLACE")
    m.texture = tex
    m.strength = strength
    m.mid_level = 0.5

# 渲一張前
clear_modifiers(obj)
if state == "bend_light":
    add_bend(obj, random.uniform(5, 15), random.choice(['X','Y']))
elif state == "displace_heavy":
    add_displace(obj, random.uniform(0.002, 0.004))
# ... 渲染後 clear，下一張重新加
```

---

## 執行順序

| 階段 | 動作 | 預估時間 | 輸出 |
|------|------|----------|------|
| 1 | 寫 `scripts/render_pan_head.py`（modifier-enhanced renderer） | 半天 | script |
| 2 | 跑 180 base renders | <30 分鐘 | `output/parts_stage2/` |
| 3 | 下載 10 張 ambientCG 背景紋理 | 半小時 | `assets/backgrounds/` |
| 4 | 寫 `scripts/composite.py`（scene generator）| 1 天 | script |
| 5 | 跑 1000 場景生成 | ~10 分鐘 | `output/scenes/` |
| 6 | 視覺檢查 + 標注正確性確認 | 半天 | — |
| 7 | 寫 `notebooks/train_stage2.ipynb`（segmentation 訓練） | 1 天 | notebook |
| 8 | Baseline 訓練 + 評估 | 1–2 小時 | metrics + viz |

預估總時間：4–5 天 wall-clock，6/02 場輕鬆趕得上。

---

## 待研究 / 給組員的議題

- **Instance separation under overlap**：純 semantic seg 在重疊區會把多實例合成一團 connected component。可選的解法：
  - Center prediction（per-pixel offset to instance center）
  - Embedding-based（per-pixel vector clustering）
  - Watershed post-processing
- **Defect type 分類**：目前 mask 只區分 normal/defective，不區分 bend vs displace。是否要進一步分類？
- **Sim-to-real 評估**：實體拍幾張 pan_head 真實照片做測試集，看 domain gap

詳見 [future_ideas.md](../future_ideas.md)。

---

## 跟課程評分對齊

| 評分維度 | Stage 2 對應做法 |
|---------|----------------|
| Problem Value & Creativity | QC 瑕疵檢測 + 合成資料解 cold start，標準工業應用、賣點清楚 |
| Technical Challenge | 自製 dataset (Blender + Python composite) + multi-objective (segment + classify) |
| Architecture & Completeness | 自製 encoder-decoder（非 pretrained，老師指定方向） |
| Prediction Results | mean IoU + per-instance precision/recall |
| Oral Presentation | 視覺化最帥（defect mask overlay 可看） |
