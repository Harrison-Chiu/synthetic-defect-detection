# 深度學習期末專題：工廠零件辨識系統

## 專題概要

NSYSU MEME552 深度學習理論與應用 — 期末專題。
- 主題：Sorting & Grasping Guided Systems（4 類螺絲螺母分類）
- 分工：Harrison 負責 Blender 合成資料集，另 2 位組員負責模型訓練
- 報告日：2026/06/02 或 06/09，英文口頭報告
- 課程要求摘要：[docs/reference/course_requirements.md](docs/reference/course_requirements.md)

## 平台

- Windows，工作目錄含中文：`D:\Harrison\中山\大四下\深度學習期末報告`
- Blender 透過 blender-mcp 操作；物件層級修改優先用 `mcp__Blender__execute_blender_code`
- Shell 是 PowerShell（用 PowerShell 語法，不是 bash 的 `$VAR` / `/dev/null` 等）

## 關鍵路徑與命名（已固定）

| 用途 | 路徑 |
|------|------|
| Blend 檔 | `blender/114-2_DLcourse_FinalProject-01.blend` |
| 3D 模型 | `assets/3d_model/<McMaster-Carr 完整檔名>.obj` |
| HDRI | `assets/hdri/<name>_4k.exr` ← **檔名有 `_4k` 後綴** |
| 渲染輸出 | `output/dataset/raw/<class_name>/` |
| 標注 CSV | `labels/metadata.csv` |
| 渲染腳本 | `scripts/render.py` |

## Blender 場景內部命名（render.py 依賴）

- 4 個零件物件：`socket_head`, `pan_head`, `hex_nut`, `flange_nut`
- 相機：`RenderCam`（不是 Blender 預設 `Camera`）
- World shader 的 Environment Texture 節點：`HDRI_node`
- 材質：`stainless_steel`（單一材質共用 4 零件）

零件 scale 是 0.001（STEP→OBJ→Blender 把 mm 對齊到 m 的標準處理），dimensions 已校正成 14.2mm 等真實尺寸，**不要動 scale**。

## MVP 採樣方案

**Pure grid，無隨機**。每張圖的參數＝它在 grid 上的座標，trivially 可復現。

| 變數 | 取值 | 數量 |
|------|------|------|
| 零件 | 4 種 | 4 |
| Camera elevation | ±10°, ±25°, ±40°, ±55°, ±70°, ±85° | 12（含仰視） |
| Camera azimuth | 0°, 120°, 240° | 3 |
| HDRI | 4 個環境 | 4 |
| **總計** | | **576 張**（每類 144） |

仰視加入是因為 `flange_nut` 凸緣從下方看才有鑑別性。azimuth 從 12 減為 3 因軸對稱零件 azimuth 主要只貢獻光照變化。

### 為何沒有 pose 變數
4 個零件都繞主軸近似對稱（螺絲軸對稱、螺母 6 次對稱），世界座標系下旋轉零件 vs 相機 orbit 對「形狀視角」是冗餘的。MVP 固定零件 pose=(0,0,0)，採樣自由度全給相機。pose 已列入 [docs/future_ideas.md](docs/future_ideas.md)，要加再加。

### 渲染參數
- 引擎 EEVEE Next，256×256 PNG
- 相機焦距 85mm，距原點 0.08m，永遠朝向原點

## 設計原則（請持續遵守）

1. **可復現性 > 採樣多樣性**：能用 pure grid 就不要 random+seed
2. **schema 不亂加欄位**：CSV 只放 MVP 真的會用的欄位，升級欄位清單見 [PLAN.md](docs/PLAN.md)
3. **分階段思考**：MVP 通了才加變體。新點子先記到 [docs/future_ideas.md](docs/future_ideas.md) 不要直接塞進當前 pipeline

## 文件結構

```
docs/
├── PLAN.md                ← 主計畫（持續更新）
├── design_decisions.md    ← 為何選 X 不選 Y
├── future_ideas.md        ← 提到但 MVP 不做的點子
├── reference/             ← 外部參考（課程 PDF、摘要）
└── weekly/                ← 每週進度
```

## Render 觸發方式

```bash
blender --background "blender/114-2_DLcourse_FinalProject-01.blend" --python scripts/render.py
```

互動式 debug 階段建議開 Blender GUI 從 Script Editor 跑，方便看進度與場景狀態。批次大量渲染再用 `--background`。

## 已踩過的雷（不要再踩）

1. **相機 near clip**：預設 0.1m > CAMERA_RADIUS 0.08m → 零件被裁掉、渲出全空白。已修為 0.001m
2. **Blender 5.x 引擎名稱**：是 `BLENDER_EEVEE`（不是 `BLENDER_EEVEE_NEXT`）
3. **HDRI 入鏡**：需要 `scene.render.film_transparent = True` 才不會污染背景

完整排查清單見 [docs/render_gotchas.md](docs/render_gotchas.md)

## 目前進度（隨進度更新）

- [x] STEP → OBJ 轉檔，4 零件匯入 Blender
- [x] 零件分開、改短名、套金屬材質、Camera 改名 RenderCam，刪掉預設 Light
- [x] HDRI 接上 World shader
- [x] render.py 寫好 + 修完 clip / engine / transparent 三個雷
- [x] 兩批測試渲（batch_001 + batch_002 含參數對照）通過視覺檢查
- [x] 採樣方案改為 12×3×4 = 576 張
- [x] train.ipynb 寫好（手刻 CNN + skip connection，HW9 風格，4-class）
- [x] **MVP 完成**：576 張渲完、conda 環境建好、train.ipynb 跑通 → **test acc 100%**
- [x] Stage 1 報告 [docs/stage1_mvp_report.md](docs/stage1_mvp_report.md)
- [ ] **Stage 2**：增加問題難度（方向討論中，見 [docs/future_ideas.md](docs/future_ideas.md)）

## Stage 1 結論

Pipeline 整條通，但問題本身對 deep model 太簡單（黑底＋canonical pose＋4 類差異大），100% 不具鑑別意義。下一階段應增加難度。
