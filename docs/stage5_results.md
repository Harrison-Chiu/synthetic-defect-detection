# Stage 5 — 結果與分析(interior-ignore 監督 + 模型瘦身)

> 訓練:2026-05-31｜計畫:[stage5_plan.md](stage5_plan.md)｜對比:[history/stage3_results.md](history/stage3_results.md)
> 根因背景:[bend_diagnosis_handoff.md](bend_diagnosis_handoff.md)(S4 坐實 bend 崩潰=監督形式不匹配)

---

## 0. 一句話
把 defect 監督從「整顆壞件塗 defect」改成「**只標變形區、壞件內部 ignore**」(由高斯權重 W≈0 自然達成),
回 S3 雙頭、只換 Head2 target。結果 **defect IoU 0.362→0.39(勝 S3)**、bend 不再崩 0(但本質仍弱),
且發現模型嚴重過參數化、合成資料無過擬合。

---

## 1. KPI 對比(test set, headline 模型 pw5/thr0.7)

| 指標 | S3 baseline | **S5** | 備註 |
|------|------|------|------|
| mIoU | 0.712 | **0.721** | ≈ |
| pixel_acc | 94.1% | 94.1% | = |
| **defect IoU** | 0.362 | **0.39** | ✓ 小勝 |
| remesh per-inst IoU | 0.276 | **0.491** | ✓ 顯著 |
| bend per-inst IoU | ~0.006 | 0.028–0.115 | 離地但仍弱 |
| 好件誤報率 (normal FP) | — | 5–7% | thr=0.7 控制後 |

---

## 2. 核心機制(已落地)

- **Head2 target**:變形區(normal vs defect 同 pose 相減,`scripts/gen_defectmask.py`,thr=30)膨脹成容忍帶 T=1;
  好件/背景 T=0;**壞件內部 + 遠背景權重 W≈0 → ignore**(高斯 σ=6 自然衰減,非硬條件)。
- **loss**:part 頭 CE + defect 頭加權 BCE+soft-Dice(吃 W)+ pos_weight。
- **變形區隨零件同 rot/scale/裁切**貼進場景空間(同原點已配準,不按質心歸零)。
- 程式:`src/schema.py`(雙頭)、`src/data/generator.py`(defect_region)、`src/data/dataset.py`(encode_targets→T/W)、
  `src/train/losses.py`、`src/eval/metrics.py`(evaluate_all 單次前向)。

---

## 3. 實驗矩陣與發現

### 3.1 base_c 寬度掃描 → 嚴重過參數化
| base_c | 2 | 4 | 6 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| params | 13K | 52K | 117K | 208K | 829K | 1.86M | 3.31M |
| defect IoU* | 0.000 | 0.177 | 0.283 | 0.296 | 0.294 | 0.299 | 0.287 |

*thr=0.5。**bc8 即飽和,bc6 達 96% @ 56% params,bc2 崩。容量非瓶頸**(計畫預測正確)。

### 3.2 深度掃描 → 深度縮優於寬度縮
| 設定 | params | defIoU(thr0.7) |
|---|---|---|
| depth=4 | 208K | 0.365 |
| **depth=3** | **52K** | **0.324** |
| depth=2 | 13K | 0.257 |
- depth3(52K)遠勝同參數量 bc4(52K, 0.177)→ **第 4 下採樣層近乎冗餘**(bottleneck 16×16 早覆蓋整顆螺絲)。
  **depth=3 = 最佳「極小模型」**(4× 小、~89% 效能)。

### 3.3 壓 FP:defect_thr(後處理免重訓)+ pos_weight
- `scripts/sweep_defect_thr.py`:bc8 上 **thr 0.5→0.7 → defIoU 0.296→0.365、normal FP 22%→7%**。
- pos_weight 掃描(thr0.7):**pw5 最佳 defIoU 0.385**(prefer over pw8=0.365、pw3=0.353 過抑)。
- 預設 `defect_thr=0.7`、headline `pos_weight=5`。

### 3.4 長版 15000 步(5×)→ 無過擬合
- train/val loss 全程同步、雙雙平台化,val 不回升 → **無過擬合**(runtime 生成=等效無限資料,無固定訓練集可記憶)。
- **收斂平台 ~step 5000–7000(best@6800)→ 實際需要步數 ≈ 5000–6000**;原 3000 步略欠訓。

---

## 4. 三種瑕疵各自表現(headline, test, thr0.7)
| type | 偵測率 | per-inst IoU | defect BCE | 解讀 |
|---|---|---|---|---|
| remesh | 88% | **0.49** | 0.35 | 最強,邊緣訊號扛得住 interior-ignore |
| displace | 39% | 0.23 | 0.69 | 中等;比 S3(0.43)回落,因 dense 內部訊號被 ignore(已知 trade-off) |
| bend | 8% | 0.028 | 0.80 | floor。interior-ignore 讓它「能表示」(不再崩 0)但 matched-FP 下偵測 ≈ FP floor |

**bend 誠實面**:thr=0.5 時的「bend 27%」是誤報灌水;matched-FP(thr0.7, FP7%)下 bend 偵測 9% vs floor 7%
→ bend 本質視覺訊號低(呼應 S3),interior-ignore 解決的是「監督矛盾致崩潰」,非「讓弱訊號變強」。

---

## 5. 最終選定
**headline:base_c=8 / depth=4 / pos_weight=5 / defect_thr=0.7 / ~6000 步**(run `output/runs/s5_long`,defIoU 0.39)。
**極小備選:base_c=8 / depth=3**(52K params)。

---

## 6. 可復現性
- 資料 100% 確定性 `scene(i)=f(seed,i)`;val/test 用 EVAL/TEST_INDEX_OFFSET 與訓練 disjoint。
- 重跑:`python cli.py train --tag X --base-c 8 --pos-weight 5 [--total-steps 6000]`。
- defectmask:`python scripts/gen_defectmask.py`(由 patch 重生)。
- 訓練數字近乎重現(cuDNN 浮點微抖);`output/runs/`、`output/patches/` 不進 git(可重生)。

---

## 7. 未完成(交付前)
- **`.ipynb` 繳交橋**(課程要求 .ipynb 原始碼;6/2 截止)—— **暫緩,待 Harrison 看結果後決定範圍**。
- 統一用 thr=0.7 重生全部 report(舊 width 掃 run 的 report 存 thr=0.5,誤報率偏高、不可並排)。
- (低優)w_norm 旋鈕未 plumb 進 TrainConfig;S3 的 FIG A(資料特性圖)未進通用 report。
- headline best.pt 未版控(交付 .zip 時需保留)。
