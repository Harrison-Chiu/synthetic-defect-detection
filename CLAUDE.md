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
├── stage3_plan.md         ← Stage 3 兩頭架構 + 資料集擴充計畫（當前）
├── stage2_baseline_results.md ← Stage 2 結果與失敗分析
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

### Stage 3 — 完成（2026-05-27）
完整計畫見 [docs/stage3_plan.md](docs/stage3_plan.md)，結果見 [docs/stage3_results.md](docs/stage3_results.md)。

**三大改動**：
1. **Two-head 架構**：共用 encoder-decoder，head 1 part/bg + head 2 defect（只在 part 像素算 loss）
2. **Loss**：head 1 CE / head 2 Dice 0.5 + BCE 0.5；Adam lr=1e-3
3. **資料**：displace 強度翻倍、新增 **Remesh Sharp** 瑕疵、背景改 **Domain Randomization 策略**（5 真實 AmbientCG + 12 procedural + 純色三選一 + 非均勻數量的程序化 distractors）+ 黑底 ablation 對照組

**結果（Test 100 場景）**：

| 指標 | Stage 2 | Stage 3 目標 | **Stage 3 實際** |
|------|--------:|-------------:|------------------:|
| mIoU | 0.597 | > 0.65 | **0.712** ✅ |
| pixel_acc | 94.0% | > 94% | **94.1%** ✅ |
| **defect IoU** | **0.004** | **> 0.30** | **0.362** ✅ (90×) |

**Per-state defect IoU**: displace 0.39–0.43 最強、remesh 0.16–0.28 中等、**bend 0.006–0.024 失敗**（256px 解析度下視覺 signal 太弱，不是模型容量問題）。

**Ablation（textured DR vs 黑底）**: mIoU 0.712 → 0.678，defect IoU 0.362 → 0.319 — 模型沒靠背景 shortcut，DR 策略成功。

**8 張報告圖在 `docs/figures/stage3/`**：A normal/defect diff、B confidence heatmap、C per-state IoU box、D FP/FN overlay、E epoch snapshots、F by-state confusion、ablation bar、training curves。

**Artifact**：`output/stage3_best_model.pt`、`output/stage3_history.json`、`output/stage3_epoch_snapshots/`、`output/scenes_black/` (100 ablation)。

**遠端檢視用 HTML 報告**：`docs/stage3_report.html`（3.2 MB 單檔，圖 base64 內嵌），生成 script `scripts/build_stage3_html.py`。已上傳到 Google Drive 共用資料夾 `G:\我的雲端硬碟\大學 中山\大四下_深度學習\深度學習_雲端共用\`（同時放了 stage3_results.md / stage3_plan.md / stage2_baseline_results.md / figures_stage3/ 完整一套給組員）。

**訓練動態踩雷**：epoch 30 val mIoU 突然從 0.71 崩到 0.49（best epoch=26, mIoU=0.716）。原因推測 Adam lr 沒 decay + Dice loss 在 zero-defect batch 不穩。test 用 best_state 跑（epoch 26 weight），所以 test=0.712 沒受影響；但 FIG E epoch snapshot 顯示的 epoch 30 是 last 不是 best，看起來變糟是真的。下次加 `ReduceLROnPlateau` 或早停。

**問題診斷小結（給之後討論用）**：
- 模型架構 ✅ Two-head 有效
- Loss ✅ Dice+BCE 解了 class imbalance
- 資料量 ✅ 1000 場景夠
- **資料 signal — bend ❌ 是剩下唯一瓶頸**：256 px 下 bend 物理上幾乎沒像素差異（FIG F 顯示 93% bend 像素被預測為 normal，等於 model 在猜 majority）。要解必須動資料端（拉相機焦距/角度 60°+/側面視角），動模型無用。
- 資料 signal — displace ✅ saturate 在 0.43
- 資料 signal — remesh 🟡 0.16–0.28 中等

### Stage 4 — 完成（2026-05-28）

完整結果見 **[docs/stage4_results.md](docs/stage4_results.md)**、HTML 報告 `docs/stage4_report.html`。計畫存檔 [docs/stage4_plan.md](docs/stage4_plan.md)。

**TL;DR**：兩條獨立路徑 — (1) 資料 bug fix + 強化、(2) Multi-head 架構嘗試。
- **路徑 1 成功**：S3 arch + S4 新資料 → **defect IoU 0.385**（vs S3 baseline 0.362, +0.023）
- **路徑 2 失敗**：Multi-head (A+B+C) → 0.342（L_A 只佔總 loss 3%，aux head 把 encoder bandwidth 吃掉）
- **最終 best 採 single-head + 新資料**，multi-head 進 ablation

**Per-state（best 模型 vs S3）**：
| State | S3 | S4 best | 變化 |
|-------|---:|---:|---:|
| bend_light | 0.034 | 0.062 | +0.028 |
| bend_heavy | 0.047 | 0.079 | +0.032 |
| **displace_light** | 0.780 | **0.235** | **-0.545 意外退步** |
| displace_heavy | 0.932 | 0.763 | -0.169 |
| remesh_light | 0.324 | **0.794** | +0.470 |
| remesh_heavy | 0.523 | **0.859** | +0.336 |

**Bend axis 修正關鍵踩雷**（前兩次都錯）：
1. 第一次 Empty origin + axis_yaw=`az+π/2` → bend 弧 ⊥ image plane = 看不到（軸算反）
2. 第二次 axis_yaw=`az` + Empty origin → 變圓錐（Empty local frame 跟 SIMPLE_DEFORM 的座標互動有坑，文件講不清）
3. **第三次成功**：放棄 Empty，**直接旋轉 pan_head 自身 α=az+jitter 度繞 Z + deform_axis="X"**，bend 弧落 image plane 可見
- **教訓**：SIMPLE_DEFORM 的 `origin` 物件如何映射 deform_axis 在 Blender 文件講不清，**直接旋轉物件比較可控**

**Multi-head 失敗模式（給 Stage 5 用）**：
- 訓練尾段 loss 分解：L_A=0.063 (3%)、L_B+L_C=2.17 (97%) → 主任務梯度被 aux 主導
- α_B = α_C = 1.0 直接 sum 是錯的，下次要 0.1 ~ 0.3 或 Uncertainty Weighting (Kendall 2018)
- 或更乾淨：完全砍 binary head，只用 B 7-way 推論時 argmax collapse

**Displace 退步是新發現**（未診斷完成，留 Stage 5）：
- 渲染參數沒改，只是 HDRI pool 2 → 4
- 推論：新加的 `monochrome_studio_02`（純白光）+ `pretoria_gardens`（室外柔光）讓 displace 反射訊號變平
- **sim-2-real 的縮影** — 同樣的 defect 在不同光照下「可學程度」差很多
- 待調查：渲 displace_heavy × 4 HDRI 對照，量化視覺強度差

**Decision point 處理**：bend IoU 落 0.05–0.15 區間（plan 規定要加碼），但因報告期限 5 天 + 現有結果已超 S3，**選擇結案 + future_ideas.md 記錄**：
- Bend elevation ∈ [-30, 30] 限縮
- Focal length 拉長 1.5×
- Bend angle 60°+ 試（單變因）
- 上述都列為 phase 2 候選但 Stage 4 不做

**Artifact**：
- `output/stage4_singlehead_best.pt`（最終 best，defect IoU 0.385）
- `output/stage4_best_model.pt`（multi-head ablation）
- `output/parts_stage2/` 504 張新 render
- `output/scenes/` 1000 + `output/scenes_black/` 100
- `docs/figures/stage4/` 報告圖（compare_overall、compare_per_state、B_state_confusion、C_type_confusion、ablation_bar、training_curves）
- Drive 已同步 `G:\我的雲端硬碟\大學 中山\大四下_深度學習\深度學習_雲端共用\`（含 stage4_plan.md、stage4_results.md、stage4_report.html、figures_stage4/、model_stage4/）

### Stage 2 訓練排查重點 — 已解決
- 之前 nbconvert 訓練 timeout 30 分鐘的**根因**：nbconvert 沒走 notebook 的 kernelspec，跑成 Windows Store Python 3.11（CPU-only torch）。Conda env `dl_final` 本身有 cu121 GPU torch。
- **執行 notebook 訓練的正確方式**：在 VS Code/Jupyter UI 打開 → 選 kernel `Python (dl_final)` → 跑
- **要用 .py 跑訓練**：先 `jupyter nbconvert --to script`、加 `matplotlib.use('Agg')` 防 plt.show 卡住、用 `& "C:\Users\Harrison\miniconda3\envs\dl_final\python.exe" -u <script>` 跑
- `scripts/train_stage2_runner.py`（base_c=16 備案）目前**不需要**了 — 完整 baseline GPU 跑 5 分鐘輕鬆
