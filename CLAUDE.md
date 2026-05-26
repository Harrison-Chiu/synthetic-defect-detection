# 深度學習期末專題：工廠零件辨識系統

## 專題概要

NSYSU MEME552 深度學習理論與應用 — 期末專題。
- **題目方向**：Quality Control & Defect Detection（瑕疵檢測）
- **現在階段**：**Stage 2** — 單一零件（pan_head）瑕疵 segmentation
- 分工：Harrison 負責 Blender 合成資料集，另 2 位組員負責模型訓練
- 報告日：2026/06/02 或 06/09，英文口頭報告
- 課程要求：[docs/reference/course_requirements.md](docs/reference/course_requirements.md)

**Stage 1 已完成**（4-class 分類 MVP，test acc 100%，pipeline 驗證通過），記錄在 [docs/stage1_mvp_report.md](docs/stage1_mvp_report.md)。Stage 2 改打 QC 方向。

完整 Stage 2 計畫見 [docs/stage2_plan.md](docs/stage2_plan.md)。

## 平台

- Windows，工作目錄含中文：`D:\Harrison\中山\大四下\深度學習期末報告`
- Blender 透過 blender-mcp 操作；物件層級修改優先用 `mcp__Blender__execute_blender_code`
- Shell 是 PowerShell（用 PowerShell 語法，不是 bash 的 `$VAR` / `/dev/null` 等）
- 訓練環境：conda env `dl_final`（Python 3.11 + PyTorch cu121）

## 關鍵路徑與命名（已固定）

| 用途 | 路徑 |
|------|------|
| Blend 檔 | `blender/114-2_DLcourse_FinalProject-01.blend` |
| 3D 模型 | `assets/3d_model/<McMaster-Carr 完整檔名>.obj` |
| HDRI（git 不追蹤，從 Poly Haven 下載）| `assets/hdri/<name>_4k.exr` |
| Stage 1 渲染輸出（576 張） | `output/dataset/raw/<class_name>/` |
| Stage 2 單零件渲染 | `output/parts_stage2/{defect_state}/` |
| Stage 2 場景合成 | `output/scenes/{scene_id:05d}/` |

## Blender 場景內部命名（script 依賴）

- 4 個零件物件：`socket_head`, `pan_head`, `hex_nut`, `flange_nut`
- 相機：`RenderCam`（不是 Blender 預設 `Camera`）
- World shader 的 Environment Texture 節點：`HDRI_node`
- 材質：`stainless_steel`（單一材質共用 4 零件）

零件 scale 是 0.001（STEP→OBJ→Blender 把 mm 對齊到 m 的標準處理），dimensions 已校正成 14.2mm 等真實尺寸，**不要動 scale**。

## Stage 2 任務定義

**輸入**：256×256 RGB 場景圖（多 pan_head 螺絲散落在工業背景，可重疊）  
**輸出**：256×256 × 3-class per-pixel semantic segmentation
- `0` = 背景
- `1` = 正常零件
- `2` = 瑕疵零件

**瑕疵類型**：Bend（彎曲）+ Displace（表面凹凸），各 2 強度。

**模型**：自製 Encoder-Decoder + skip connections（**不用 pretrained，不抄 U-Net 名字**）。架構細節見 stage2_plan.md。

**Overlap 處理**：資料端允許重疊，instance 分離（如有需要）留給組員研究。

## 設計原則（請持續遵守）

1. **可復現性 > 採樣多樣性**：能用 pure grid 就不要 random+seed
2. **schema 不亂加欄位**：CSV / meta 只放真的會用的欄位
3. **分階段思考**：MVP 通了才加變體。新點子先記到 [docs/future_ideas.md](docs/future_ideas.md)
4. **自製優先**：老師指定，不用 pretrained model，不直接抄 SOTA 架構（U-Net / YOLO 等）

## 已踩過的雷（不要再踩）

1. **相機 near clip**：預設 0.1m > CAMERA_RADIUS 0.08m → 零件被裁掉、渲出全空白。已修為 0.001m
2. **Blender 5.x 引擎名稱**：是 `BLENDER_EEVEE`（不是 `BLENDER_EEVEE_NEXT`）
3. **HDRI 入鏡**：需要 `scene.render.film_transparent = True` 才不會污染背景

完整排查清單見 [docs/render_gotchas.md](docs/render_gotchas.md)

## 文件結構

```
docs/
├── PLAN.md                ← 主計畫
├── stage2_plan.md         ← Stage 2 詳細技術設計
├── stage1_mvp_report.md   ← Stage 1 完成報告
├── operations_stage1.md   ← Stage 1 操作指令（封存）
├── design_decisions.md    ← 為何選 X 不選 Y
├── future_ideas.md        ← 提到但目前不做的點子
├── render_gotchas.md      ← 渲染踩雷清單
├── reference/             ← 外部參考（課程 PDF、HW9）
└── weekly/                ← 每週進度
```

## 目前進度

### Stage 1（完成）
576 張 4-class 分類資料集 + 手刻 CNN baseline → test acc 100%。`scripts/render_stage1.py`、`notebooks/train_stage1.ipynb` 保留封存。

### Stage 2 — 資料端（完成）
- [x] `scripts/render_pan_head.py`：180 張 pan_head 帶 modifier 瑕疵變體（normal / bend×2 / displace×2）
- [x] `scripts/gen_backgrounds.py`：12 張程序化背景（concrete/metal/rubber/wood）
- [x] `scripts/composite.py`：1000 張合成場景，含 semantic_mask + instance_mask + meta.json，允許重疊
- [x] **defect 比例調整**：原本 80% defect instance（uniform 抽 180 parts）→ 改 `DEFECT_PROB=0.2` per-instance 抽樣 → 實際 19.4%
- [x] `scripts/preview_scenes.py`、`scripts/preview_parts.py`：mask 是 {0,1,2} 全黑值，上色才看得到
- 輸出在 `output/parts_stage2/` 與 `output/scenes/`，meta 在 `parts_meta.json` 與每場景 `meta.json`

### Stage 2 — Baseline 訓練（完成）
- [x] `notebooks/train_stage2.ipynb`：3.3M params 自製 encoder-decoder (base_c=32, 4 stage)
- [x] **GPU 訓練 30 epoch ~5 min**（4060），SGD lr=0.01 + weighted CE
- [x] Test mIoU=0.597, pixel_acc=94%；但 **defect IoU=0.004**（baseline 主要失敗）
- [x] `scripts/eval_stage2.py` + `docs/figures/stage2/`：5 張報告用圖 + summary
- [x] 完整結果 + 失敗分析 + 下一步選項見 [docs/stage2_baseline_results.md](docs/stage2_baseline_results.md)

### Stage 2 — 下一步（討論中）
**核心觀察**：confusion matrix 顯示「物件 vs 背景」幾乎完美（99.4% / 99.0%），失敗純粹在「normal vs defect」一刀（99.2% defect pixel 被預測成 normal）。
**主要候選方向**（待跟組員討論）：
1. **Two-head 架構** — 共用 encoder，part/bg binary head + defect head 解耦（Harrison 提議，confusion matrix 背書）
2. 同 single-head 換 loss：Dice / Focal 對付 minority class，LR 降到 1e-3 + Adam
3. 質疑 defect 視覺特徵強度（displace 在 parts_preview 看就很微妙）→ 回 Blender 加強參數或剔除 displace

### Stage 2 訓練排查重點 — 已解決
- 之前 nbconvert 訓練 timeout 30 分鐘的**根因**：nbconvert 沒走 notebook 的 kernelspec，跑成 Windows Store Python 3.11（CPU-only torch）。Conda env `dl_final` 本身有 cu121 GPU torch。
- **執行 notebook 訓練的正確方式**：在 VS Code/Jupyter UI 打開 → 選 kernel `Python (dl_final)` → 跑
- **要用 .py 跑訓練**：先 `jupyter nbconvert --to script`、加 `matplotlib.use('Agg')` 防 plt.show 卡住、用 `& "C:\Users\Harrison\miniconda3\envs\dl_final\python.exe" -u <script>` 跑
- `scripts/train_stage2_runner.py`（base_c=16 備案）目前**不需要**了 — 完整 baseline GPU 跑 5 分鐘輕鬆
