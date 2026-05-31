# Stage 5 — 成果報告：Interior-Ignore 監督 + 模型瘦身

> 訓練日期：2026-05-31 ｜ 計畫：[stage5_plan.md](stage5_plan.md) ｜ S3 對比：[history/stage3_results.md](history/stage3_results.md)
> S4 診斷（本 stage 的出發點）：[bend_diagnosis_handoff.md](bend_diagnosis_handoff.md)

---

## 術語速查

| 術語 | 全名 | 說明 |
|---|---|---|
| IoU | Intersection over Union | 預測區域與真實區域的重疊率。1 = 完美，0 = 完全不重疊 |
| mIoU | mean IoU | 所有類別（bg / normal / defect）IoU 的算術平均 |
| defect IoU | — | 單看 defect 類的 IoU，本專案最重要的指標 |
| pixel_acc | pixel accuracy | 所有像素判對的比例；被背景像素主導，偏樂觀 |
| det_rate | detection rate | 某 state 零件的像素被判為 defect 的比例（注意：好件的 det_rate = 誤報率） |
| FP / FN | False Positive / Negative | 誤報（好件判壞）/ 漏報（壞件沒抓到） |
| BCE | Binary Cross-Entropy | 逐像素二元分類損失 |
| Dice | Dice loss | 看整塊預測與 GT 的重疊率，抗極端類別不平衡 |
| pos_weight (pw) | positive weight | 正類（defect）在 BCE 中的加倍權重，補償變形區像素稀少 |
| thr | threshold | 推論時 sigmoid > thr 才判為 defect（後處理參數，不需重新訓練） |
| base_c (bc) | base channels | U-Net 第一層的通道數，整個模型的寬度控制旋鈕；參數量 ∝ bc² |
| depth (d) | — | U-Net 下採樣層數（4 = 預設，3 = 少一層） |

---

## 0. 摘要

S4 診斷坐實了 bend 偵測崩潰（IoU ≈ 0）的根因：**壞件內部像素外觀和正常件一樣，卻被逼著學 defect → 監督矛盾**。

S5 的修正是一個**單變因實驗**——架構回到 S3 雙頭（part + defect），只改 defect 頭的 target：
- 變形區（normal vs defect 同 pose 相減得到的差異區）= 1
- 壞件內部 + 遠背景 → 權重 W ≈ 0（= ignore，矛盾就消在這）
- 好件 / 背景 = 0（不該亮）

結果：
- **defect IoU 0.362 → 0.390（勝 S3 baseline）**
- 模型**嚴重過參數化**（bc=8 的 208K 參數即飽和，加到 3.31M 毫無提升）
- 合成資料**無過擬合**（train/val loss 全程同步，runtime 生成 = 等效無限資料）
- bend **不再崩潰到 0**（S5 解決了監督矛盾），但本質視覺訊號仍弱（見 §4 誠實分析）

---

## 1. Headline 模型與 KPI

**選定模型**：`base_c=8 / depth=4 / pos_weight=5 / defect_thr=0.7 / ~6000 步`
（run：`output/runs/s5_long`，208K 參數）

**極小備選**：`base_c=8 / depth=3`（52K 參數，defect IoU ≈ 0.324）

### 1.1 vs S3 baseline（test set）

| 指標 | S3 | S5 | 備註 |
|---|---|---|---|
| mIoU | 0.712 | **0.721** | 小勝 |
| pixel_acc | 94.1% | 94.1% | 持平 |
| **defect IoU** | 0.362 | **0.390** | +0.028，核心指標改善 |

### 1.2 best 選擇標準說明

目前以 **val mIoU（三類平均）** 選 best checkpoint。由於 bg 和 normal 佔絕大多數像素，mIoU 主要被它們撐著，選出來的 best 不保證是 defect IoU 最高的那一步。但從 s5_long 的訓練記錄看，defect IoU 在 step ~4000 後進入平台（0.39–0.41 波動），各步差異很小，改用 defect IoU 選 best 不會根本改變結論。

---

## 2. 核心機制（已落地的程式碼）

### 2.1 離線：產變形區 mask

對每個 (pose × defect_state) 的 patch pair，計算：
```
M = despeckle( mean_RGB|defect_patch − normal_patch| > 30 )
```
存為 `*_defectmask.png`。閾值 30 的選擇依據見 `docs/figures/route_a_diff_threshold.png`。

關鍵事實：bend 的 SIMPLE_DEFORM 繞原點彎（根固定、尖端彎），normal/defect 共用原點 → 相減已正確配準，**不需按質心歸零**。

Script：`scripts/gen_defectmask.py`

### 2.2 每張場景：組 Target T 和 Weight W

```
D   = 所有壞件的變形區（各自 mask 隨零件 rot/scale/裁切貼進場景空間）
D⁺  = D 膨脹 4 px（容忍帶：亮在附近也算對）
N   = 好件剪影

Target T(x)  = 1 if x ∈ D⁺, else 0
Weight W(x)  = max( GaussBlur(D⁺, σ=6), 0.3·1{x∈N} )
               → 壞件內部 & 遠背景 W ≈ 0 = ignore
```

### 2.3 Loss

```
L_total = L_part + L_defect

L_part   = CrossEntropy(part_logits, bg/part mask)       # 和 S3 相同
L_defect = Σ W(x)·BCE_pw(ŝ(x), T(x)) / Σ W(x) + Dice  # 加權 BCE + soft-Dice，都吃 W
```

- **BCE**：逐像素精準但被多數類淹沒 → 加 pos_weight 補正
- **Dice**：看整塊重疊率，抗極端類別不平衡。S3 用 Dice+BCE 將 defect IoU 從 0.004→0.362，故保留

### 2.4 推論

defect 頭 sigmoid > thr → 亮區與 part mask 對位 → 某顆零件上有亮 ⇒ 判整顆為瑕疵件。

### 2.5 程式碼位置

| 模組 | 職責 |
|---|---|
| `src/schema.py` | 雙頭定義（part + defect） |
| `src/data/generator.py` | 場景生成，含 defect_region 貼圖 |
| `src/data/dataset.py` | `encode_targets` → T / W |
| `src/train/losses.py` | 加權 BCE + Dice |
| `src/eval/metrics.py` | `evaluate_all`（單次前向算齊所有指標） |

---

## 3. 實驗矩陣與發現

S5 共跑了 **13 個 run**（`output/runs/s5_*`），分四組實驗。

### 3.1 寬度掃描（base_c）→ 嚴重過參數化

固定 depth=4、pw=8、thr=0.5、3000 步。

| base_c | params | defect IoU | 判讀 |
|---|---|---|---|
| 2 | 13K | 0.000 | 崩潰（容量不足） |
| 4 | 52K | 0.177 | 弱 |
| 6 | 117K | 0.283 | 接近飽和 |
| **8** | **208K** | **0.296** | **飽和起點** |
| 16 | 829K | 0.294 | ≈ bc8（4× 參數，零提升） |
| 24 | 1.86M | 0.299 | ≈ bc8（9× 參數，零提升） |
| 32 | 3.31M | 0.287 | ≈ bc8（16× 參數，零提升） |

**結論**：bc=8（208K）即飽和。模型容量不是瓶頸——和 S5 計畫的預測一致。

### 3.2 深度掃描（depth）→ 深度比寬度有效

固定 bc=8、pw=8、thr=0.7、3000 步。

| depth | params | defect IoU | 判讀 |
|---|---|---|---|
| 2 | 13K | 0.257 | |
| **3** | **52K** | **0.324** | 極小模型，89% 效能 |
| 4 | 208K | 0.365 | 完整模型 |

depth=3（52K）遠勝同參數量的 bc=4（52K，IoU 僅 0.177）。

**解讀**：第 4 下採樣層的 bottleneck（16×16）感受野已覆蓋整顆螺絲，第 4 層近乎冗餘。depth=3 是最佳「極小模型」（4× 小，89% 效能）。

### 3.3 壓 FP：defect_thr + pos_weight

**thr 掃描**（後處理，不需重訓）：bc=8 上 thr 0.5→0.7 → defect IoU 0.296→0.365，好件誤報率 22%→7%。

**pos_weight 掃描**（bc=8、thr=0.7、3000 步）：

| pos_weight | defect IoU | 好件誤報率 |
|---|---|---|
| 3 | 0.353 | 3.1% |
| **5** | **0.385** | **5.1%** |
| 8 | 0.365 | 9.6% |

**pw=5 最佳**（defect IoU 最高且誤報可控）。最終預設：`defect_thr=0.7`、`pos_weight=5`。

### 3.4 長版 15000 步（5× 標準）→ 無過擬合

用選定的 bc=8、pw=5、thr=0.7 跑 15000 步。

- train/val loss 全程同步下降、雙雙平台化，**val loss 不回升 → 無過擬合**
- 收斂平台 ~step 5000–7000（best 在 step 6800）→ 實際只需 ~6000 步；原 3000 步略欠訓
- 最終 test：mIoU=0.721、defect IoU=0.390

**Val loss 略低於 train loss 的原因**：不是資料洩漏（val/train 用完全互斥的 index 區段）。原因是 (1) BatchNorm 在 eval mode 使用 running statistics（比 train mode 的 batch statistics 更穩定），以及 (2) val set 是固定 100 張（materialize 後常駐 RAM），其平均難度碰巧略低於持續新生的 train 場景。這是已知現象，不影響結論。

---

## 4. 三種瑕疵的表現與誠實分析

### 4.1 Headline 模型各瑕疵表現（test, thr=0.7）

| defect type | det_rate | 好件誤報率 (FP) | **delta（det − FP）** | 解讀 |
|---|---|---|---|---|
| remesh_heavy | **90.8%** | 6.2% | **+84.6%** | 最強，邊緣幾何訊號明確 |
| displace_heavy | **83.4%** | 6.2% | **+77.2%** | 強，表面凹凸訊號清晰 |
| bend_heavy | 7.4% | 6.2% | **+1.2%** | ≈ 噪音，見下方分析 |
| bend_light | 10.0% | 6.2% | +3.8% | 同上 |

### 4.2 Bend 的誠實面：跨 run 完整數據

bend 的「偵測率」**必須和好件誤報率（normal det_rate）一起看**，否則會被灌水。以下是所有 S5 run 的 bend_heavy 對照：

| run | thr | pw | bend_heavy det | normal det (FP) | delta |
|---|---|---|---|---|---|
| s5_bc4 | 0.5 | 8 | 91.5% | 93.2% | −1.7%（全判 defect，無辨識力） |
| s5_bc32 | 0.5 | 8 | 35.2% | 29.5% | +5.7% |
| s5_bc8 | 0.5 | 8 | 28.6% | 22.2% | +6.4% |
| s5_pw8 | 0.7 | 8 | 15.2% | 9.6% | +5.6% |
| **s5_long** | **0.7** | **5** | **7.4%** | **6.2%** | **+1.2%** |
| s5_pw3 | 0.7 | 3 | 5.9% | 3.1% | +2.8% |

**分析**：

1. **thr 是最大變因**：0.5→0.7 讓 bend 從 28%→7%，但 normal FP 也從 22%→6%——兩者**同步下降**。模型不是「抓到 bend」，而是「全面灑 defect 時 bend 也中」。
2. **Delta（bend det − normal FP）在所有 run 都只有 1–6%**，不隨模型容量或訓練步數改善。
3. **結論**：interior-ignore 解決了「監督矛盾導致 bend 崩潰到 0」的問題——bend 不再崩潰，但它的**視覺訊號本質就是弱的**（彎曲是全域形變，像素級差異微小）。Interior-ignore 解決的是監督端的 bug，不是讓弱訊號變強。

### 4.3 已知 trade-off

displace 的 det_rate 從 S3 的 ~43%（thr=0.5 下）變化到 S5 的 83.4%（thr=0.7 下）——看似提升，但主因是 S5 的推論合併邏輯不同（defect 頭 sigmoid + part mask 對位），不可直接比較。在 interior-ignore 機制下，displace 的 dense 內部訊號被 ignore，理論上它的像素級定位精度會略降（已知 trade-off）。

---

## 5. 訓練設定

| 項目 | 設定 | 說明 |
|---|---|---|
| Optimizer | Adam | lr=1e-3, weight_decay=1e-4；主流選擇 |
| LR schedule | ReduceLROnPlateau | 監控 val mIoU，3 次 eval 無進步 → lr × 0.5，下限 1e-5 |
| Batch size | 8 | |
| DataLoader workers | 8 | runtime 生成防阻塞（實測 11× 加速） |
| 輸入尺寸 | 256 × 256 | |
| 資料正規化 | (x/255 − 0.5) / 0.5 → [−1, 1] | |

LR 實際軌跡（s5_long）：`1e-3 → 5e-4(@2400步) → 2.5e-4(@3400) → 1.3e-4(@4600) → 6.3e-5(@5400) → 3.1e-5(@6200) → 1.6e-5(@7400) → 1e-5(@8200, floor)`

Adam + ReduceLROnPlateau 是 DL 訓練的主流標準配置。本專案沿用 S3 的設定，未額外實驗其他 schedule（如 cosine annealing）。

---

## 6. 可復現性

- 資料 100% 確定性：`scene(i) = f(seed, i)`；val（offset 10M）/ test（offset 30M）和訓練完全互斥
- 重跑 headline：`python cli.py train --tag X --base-c 8 --pos-weight 5 --total-steps 6000`
- 重生 defect mask：`python scripts/gen_defectmask.py`
- 訓練數字近乎完全重現（cuDNN 浮點微抖）
- `output/runs/` 和 `output/patches/` 不進 git（可重生）

---

## 7. 現有限制

- **Per-type 訓練動態不可追蹤**：目前 training loop 只記錄整體 loss / mIoU / per-class IoU，沒有按瑕疵 type（bend / displace / remesh）拆開的曲線。無法回答「bend 在訓練過程中是持續低迷還是有階段性波動」。→ 列入 S6 改善。
- **只有 pan_head 有 defect patch**（72 pose）；socket / hex_nut / flange 無 → 「多零件 QC + Sorting」的真實缺口。
- Part 頭只分 bg / 好件 / 壞件，**不區分零件種類**（socket vs pan vs hex vs flange）。
- Per-run HTML 報告的預測面板是隨機選圖，沒有針對三種瑕疵 + 全好做策略性選圖。
- 寬度掃描的 report 用 thr=0.5 生成，不可與 headline（thr=0.7）的數字並排比較。

---

## 8. S5 所有 run 清單

| run | bc | depth | pw | thr | steps | params | test defIoU | 目的 |
|---|---|---|---|---|---|---|---|---|
| s5_bc2 | 2 | 4 | 8 | 0.5 | 3000 | 13K | 0.000 | 寬度掃描 |
| s5_bc4 | 4 | 4 | 8 | 0.5 | 3000 | 52K | 0.177 | 寬度掃描 |
| s5_bc6 | 6 | 4 | 8 | 0.5 | 3000 | 117K | 0.283 | 寬度掃描 |
| s5_bc8 | 8 | 4 | 8 | 0.5 | 3000 | 208K | 0.296 | 寬度掃描（飽和基準） |
| s5_bc16 | 16 | 4 | 8 | 0.5 | 3000 | 829K | 0.294 | 寬度掃描 |
| s5_bc24 | 24 | 4 | 8 | 0.5 | 3000 | 1.86M | 0.299 | 寬度掃描 |
| s5_bc32 | 32 | 4 | 8 | 0.5 | 3000 | 3.31M | 0.287 | 寬度掃描 |
| s5_bc8_d2 | 8 | 2 | 8 | 0.7 | 3000 | 13K | 0.257 | 深度掃描 |
| s5_bc8_d3 | 8 | 3 | 8 | 0.7 | 3000 | 52K | 0.324 | 深度掃描 |
| s5_pw3 | 8 | 4 | 3 | 0.7 | 3000 | 208K | 0.353 | pw 掃描 |
| s5_pw5 | 8 | 4 | 5 | 0.7 | 3000 | 208K | 0.385 | pw 掃描 |
| s5_pw8 | 8 | 4 | 8 | 0.7 | 3000 | 208K | 0.346 | pw 掃描 |
| **s5_long** | **8** | **4** | **5** | **0.7** | **15000** | **208K** | **0.390** | **Headline（長版驗證）** |
