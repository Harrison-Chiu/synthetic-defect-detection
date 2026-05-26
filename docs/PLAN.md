# 深度學習期末專題：工廠零件辨識系統
## Synthetic Dataset via Blender — Project Plan

> 課程：MEME552 深度學習理論與應用，NSYSU  
> 分工：Harrison 負責資料集產生，另外兩位組員負責模型訓練  
> 最後更新：2026-05-17

---

## 專題定位

**主題類別**：Sorting & Grasping Guided Systems  
**核心問題**：工廠場景中，真實零件分類資料極難取得，透過 Blender 合成資料集解決冷啟動問題  
**本週目標（Week 1 MVP）**：驗證整個 pipeline 可行，產出可供訓練的分類資料集

---

## 資料夾結構

```
project/
├── docs/
│   ├── PLAN.md           ← 本文件
│   └── weekly/
│       └── week01.md     ← 每週進度紀錄
├── assets/
│   ├── 3d_model/             ← 從 McMaster-Carr 下載的 STEP，FreeCAD 轉出的 OBJ
│   │   ├── 91292A137_18-8 Stainless Steel Socket Head Screw.obj
│   │   ├── 91292A137_18-8 Stainless Steel Socket Head Screw.STEP
│   │   ├── 91828A251_Corrosion-Resistant 18-8 Stainless Steel Hex Nut.obj
│   │   ├── 91828A251_Corrosion-Resistant 18-8 Stainless Steel Hex Nut.STEP
│   │   ├── 92000A428_Passivated 18-8 Stainless Steel Pan Head Phillips Screws.obj
│   │   ├── 92000A428_Passivated 18-8 Stainless Steel Pan Head Phillips Screws.STEP
│   │   ├── 93033A101_18-8 Stainless Steel Flange Nut.obj
│   │   └── 93033A101_18-8 Stainless Steel Flange Nut.STEP
│   └── hdri/                 ← Poly Haven 下載的 4K EXR
│       ├── university_workshop_4k.exr
│       ├── crossfit_gym_4k.exr
│       ├── monochrome_studio_02_4k.exr
│       └── pretoria_gardens_4k.exr
├── blender/
│   └── 114-2_DLcourse_FinalProject-01.blend  ← 主場景檔
├── scripts/
│   └── render.py             ← Blender Python 批量渲染腳本
├── output/
│   └── dataset/
│       ├── raw/              ← 渲染出來的原始圖（不動）
│       │   ├── socket_head/
│       │   ├── pan_head/
│       │   ├── hex_nut/
│       │   └── flange_nut/
│       ├── train/
│       ├── val/
│       └── test/
├── labels/
│   └── metadata.csv          ← 完整標注 CSV
└── README.md
```

---

## 零件規格

統一 M6 × 1mm、18-8 不鏽鋼，視覺差異如下：

| ID | 零件名 | McMaster-Carr 型號 | 頭部特徵 |
|----|--------|-------------------|----------|
| 0  | Socket Head Cap Screw | 91292A137 | 圓柱頭 10mm，六角內孔 |
| 1  | Pan Head Phillips Screw | 92000A428 | 扁圓頭 12mm，十字槽 |
| 2  | Hex Nut | 91828A251 | 六邊形，無軸，高度 5mm |
| 3  | Flange Nut | 93033A101 | 六角 + 底部凸緣 14.2mm |

螺絲固定長度 20mm，排除長度作為變數。

> **注意**：螺絲與螺母為旋轉對稱體，azimuth 角度間隔設 45°（8 個）即可，繞一圈不會帶來新資訊。

---

## HDRI 環境光

從 [Poly Haven](https://polyhaven.com) 下載，格式一律選 **4K EXR**（不選 HDR，金屬反射品質差異明顯）。

| 檔名 | 場景類型 | 光照特性 |
|------|----------|----------|
| University Workshop | 室內工業 | 柔和漫射光 |
| Crossfit Gym | 室內 | 高對比硬光 |
| Monochrome Studio 02 | 錄影棚 | 中性均勻光 |
| Pretoria Gardens | 室外 | 自然日光 |

---

## 渲染參數（Week 1, MVP）

採 **pure grid sampling**，無隨機，整個資料集完全由設計參數決定，trivially 可復現。

| 參數 | 值 | 說明 |
|------|----|------|
| 解析度 | 256 × 256 px | MVP 最低合理值，224 以上供 ResNet 直接用 |
| 渲染引擎 | EEVEE Next | 速度快，4060 可接受；Cycles 留到下週視需求 |
| 零件姿態 | 固定 (0,0,0) canonical pose | 不變動，把採樣自由度集中在相機 |
| Camera elevation | ±10°, ±25°, ±40°, ±55°, ±70°, ±85° | **12 個**，覆蓋上下半球（含仰視） |
| Camera azimuth | 0°, 120°, 240° | **3 個**，零件近軸對稱故減少 |
| HDRI 數量 | 4 | 見上表 |
| 總張數/零件 | 12 × 3 × 4 = **144 張** | 共 4 零件 = **576 張** |

### 設計依據

**為何不要 pose 變數**：4 個零件都繞主軸近似對稱（螺絲軸對稱、螺母 6 次對稱）。在世界座標系裡旋轉零件 vs 相機繞零件對「形狀視角」是冗餘的。

**為何 elevation 12 個（含仰視）**：
- 仰視對 `flange_nut` 特別有鑑別性 — 從下方可清楚看到凸緣輪廓，與 `hex_nut` 顯著區分
- 涵蓋 ±85° 等於完整上下半球，比原本只取上半球資訊量更平均

**為何 azimuth 只 3 個**：軸對稱零件繞 azimuth 旋轉對形狀視覺貢獻很小（pan head 十字槽是例外但 120° 取樣也夠覆蓋週期內變化），主要是貢獻 HDRI 反射方向變化。

**從 1152 減為 576 的取捨**：每類 144 張看起來少但對 MVP transfer-learning baseline 夠用；要加密的話 future_ideas 有列。

---

## CSV 標注欄位設計

路徑：`labels/metadata.csv`

```
filename, class_id, class_name, azimuth_deg, elevation_deg, hdri_name, split
```

| 欄位 | 類型 | 說明 |
|------|------|------|
| filename | string | 相對路徑，e.g. `raw/hex_nut/hex_nut_az030_el20_h2.png` |
| class_id | int | 0–3 |
| class_name | string | socket_head / pan_head / hex_nut / flange_nut |
| azimuth_deg | int | 相機水平角度 |
| elevation_deg | int | 相機仰角 |
| hdri_name | string | HDRI 檔名（不含 `_4k.exr`） |
| split | string | train / val / test（渲染後再填，或 script 自動分） |

**未來升級欄位**（下週 detection / 加隨機性時需要時再加）：
- `bbox_x, bbox_y, bbox_w, bbox_h, mask_path` — detection / segmentation
- `pose_euler_x/y/z` — 若加 pose 變化
- `hdri_rotation_deg` — 若加 HDRI 旋轉
- `render_seed` — 若引入 stratified random sampling

---

## Blender 工作流程

### 手動設定（114-2_DLcourse_FinalProject-01.blend）

1. 匯入 4 個 OBJ 零件（File → Import → Wavefront OBJ）
2. 每個零件加 **Principled BSDF** 材質：
   - Metallic: 1.0
   - Roughness: 0.2–0.35（不鏽鋼略有漫反射）
   - Base Color: 略帶冷灰（#B8BEC4）
3. 設定 3 種零件擺放姿態，用 **Collection** 或 Object Hide 管理
4. 新增一個 Camera，命名 `RenderCam`，對焦到場景原點
5. 在 World → Surface 節點接上 HDRI（先手動確認一張渲染看起來合理再跑 script）

### 批量渲染（scripts/render.py）

用 Blender 內建 Script Editor 執行，或從外部 terminal 背景執行：

```bash
blender --background "blender/114-2_DLcourse_FinalProject-01.blend" --python scripts/render.py
```

骨架邏輯（由 Claude Code 產生完整版）：

```python
import bpy, math, csv, os

OUTPUT_DIR = "output/dataset/raw"
PARTS = ["socket_head", "pan_head", "hex_nut", "flange_nut"]
AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
ELEVATIONS = [15, 45, 75]
POSES = [0, 1, 2]          # 0=flat, 1=upright, 2=tilted
HDRI_FILES = [...]          # 4 個 EXR 的完整路徑

records = []
cam = bpy.data.objects["RenderCam"]
scene = bpy.context.scene
scene.render.resolution_x = 256
scene.render.resolution_y = 256

for part_idx, part_name in enumerate(PARTS):
    for az in AZIMUTHS:
        for el in ELEVATIONS:
            for pose in POSES:
                for hdri_idx, hdri_path in enumerate(HDRI_FILES):
                    # 1. 顯示對應零件、隱藏其他
                    # 2. 設定零件姿態
                    # 3. 移動相機到 (az, el) 對應球座標
                    # 4. 換 HDRI
                    # 5. 設定輸出路徑
                    fname = f"{part_name}_{az:03d}_el{el}_p{pose}_h{hdri_idx}.png"
                    scene.render.filepath = os.path.join(OUTPUT_DIR, part_name, fname)
                    bpy.ops.render.render(write_still=True)
                    # 6. 寫入 CSV record
                    records.append({...})

# 寫出 metadata.csv
with open("labels/metadata.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[...])
    writer.writeheader()
    writer.writerows(records)
```

> Geometry Nodes **不適合**這個用途（無法控制渲染迴圈），下週若要加幾何變異（磨損、形變）再考慮引入。

---

## Optional：Blender MCP

若使用 Claude Desktop，可安裝 [blender-mcp](https://github.com/ahujasid/blender-mcp) addon，讓 AI 直接操控 Blender 場景（調材質、移相機、debug 用）。**不作為核心流程依賴**，適合設定場景階段加速，批量渲染仍靠 Python script。

---

## 格式轉換流程（McMaster-Carr → Blender）

McMaster-Carr 提供 STEP 格式，需透過 FreeCAD 轉換：

1. McMaster-Carr 產品頁往下滑，找 **3D Model** 區塊，下載 `.step`（不需購買）
2. 開啟 FreeCAD → File → Open `.step`
3. File → Export → 選 **Wavefront OBJ** → 儲存到 `assets/3d_model/`
4. Blender → File → Import → Wavefront OBJ

---

## 週次規劃

### Week 1（本週）：MVP 分類資料集
- [ ] 下載 4 個 STEP、FreeCAD 轉 OBJ
- [ ] 下載 4 個 HDRI（4K EXR）
- [ ] Blender 場景設定（材質、相機、3 種姿態）
- [ ] 手動確認一張渲染
- [ ] 寫 render.py，產出 1152 張
- [ ] 產出 metadata.csv
- [ ] 交付 zip 給組員，確認 ResNet18 pipeline 通

### Week 2（預計）
- 加入 Object Index pass，自動計算 bounding box → metadata.csv 補上 bbox 欄位
- 升級任務到 4-class detection（YOLOv8n）
- 加入更多幾何變異（Geometry Nodes：輕微形變模擬磨損）
- 評估加入真實拍攝樣本做 sim-to-real 驗證

### 後續擴充方向
- 更多零件種類
- 場景背景（工廠地板貼圖）
- 多零件同框（counting task）
- Cycles 渲染提升真實感

---

## 給組員的說明（模型端）

- 輸入格式：256×256 RGB PNG
- 資料夾結構：`output/dataset/train/{class_name}/`（PyTorch `ImageFolder` 相容）
- 建議起手：ResNet18 pretrained，修改最後一層為 4-class
- CSV 裡有 `split` 欄位，可改用 CSV 直接讀取取代資料夾分法，擴充性較好
- Baseline 目標：val accuracy > 90%（合成資料場景應可達到）
