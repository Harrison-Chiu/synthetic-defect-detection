# Active — 當前 stage 指標

> 本檔是 always-on context 的「當前 stage」指標,**永久保留**(空或極簡也留)。
> 規則:只放指標與待辦,**不複製** stage 報告內容;細節一律連到對應檔。
> stage 結案 review 完 → 整份搬 `docs/history/`,本檔重置為下一 stage。

---

## 狀態:S5 施工中(report 日 2026-06-02)｜計畫詳見 `docs/stage5_plan.md`

### 進度(2026-05-31)
- ✅ **步驟3** defectmask(`scripts/gen_defectmask.py`,432 張,thr=30)。commit `ce3c6c6`。
- ✅ **步驟1/2/4/5/6** 核心:雙頭 + interior-ignore (T,W) + 加權 loss + val/test 分離。commit `e48918c`。
- ✅ **提速 + 修 eval + 豐富 report**:ListDataset/materialize(val/test 一次生成常駐)+ 單次前向
  evaluate_all → bc8 3000步 8min→~4min;修 per-state IoU union bug;report 48KB→2.1MB
  (KPI對S3、train/val loss、預測面板、FP/FN、混淆矩陣、per-instance箱、per-type)。commit `38c8b8d`/`a395212`。
- ✅ **步驟7/8** base_c 掃描 {8,16,24,32}:**defect IoU 全平 0.287-0.299 → 已飽和,容量非瓶頸**。

### S5 結果判讀(關鍵數字,bc8≈bc24)
| type | 偵測率 | inst IoU | vs S3 |
| --- | --- | --- | --- |
| bend | ~33% | **0.115** | S3 ~0.006 → **~5×,離地了**(核心命題成立 ✅) |
| displace | ~62% | 0.23 | S3 ~0.43 → **回落約一半**(如預測 trade-off ⚠️) |
| remesh | ~93% | 0.32 | 最強,邊緣訊號扛得住 ignore |
- ⚠️ **好件誤報率(normal FP)= 22-25%**:模型過度判正 → bend 帳面被底噪灌水。
  根因 = loss 平衡(pos_weight=8 太推正 / w_norm 太弱),**多 step 修不掉**,要調旋鈕。

### base_c 飽和下邊界(下掃完成)
| base_c | 2 | 4 | 6 | 8 | 16-32 |
| --- | --- | --- | --- | --- | --- |
| params | 13K | 52K | 117K | 208K | 0.8-3.3M |
| defect IoU | 0.000崩 | 0.177 | 0.283 | 0.296 | ~0.29 |
→ 拐點 base_c≈6(bc6 達 bc8 的 96% @ 56% params);bc4 不足、bc2 崩。最小可用寬度 ≈6-8。

### ⭐ FP 突破:defect_thr 後處理(免重訓,scripts/sweep_defect_thr.py)
bc8 上掃推論閾值:**thr 0.5→0.7 → defect IoU 0.296→0.365(≈/勝 S3 0.362)、normal FP 22%→7%**。
→ TrainConfig.defect_thr 預設改 **0.7**(commit `bd38678`)。
- **誠實面**:thr=0.7(matched-FP 7%)下 **bend 偵測僅 9% vs FP floor 7% → bend 訊號仍 marginal**。
  之前 thr=0.5 的「bend 27%」是 FP 灌水。interior-ignore 讓 bend 能「表示」變形(inst IoU 0.006→0.115、5×、
  不再崩 0),但 controlled-FP 下 bend 仍弱 = 本質視覺訊號低(呼應 S3)。displace(40%)+remesh(86%)扛主力。

### 模型已可調深度(commit `4a51447`)
segnet enc/dec → ModuleList + `depth` 參數;depth=4 與原架構參數完全一致;舊 ckpt 經 flexible loader 仍可載。

### 深度探針 + pos_weight 掃描(thr=0.7 統一評估,完成)
| run | params | pos_w | defIoU | normalFP | bend | disp | rem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bc8 d4 | 208K | 8 | 0.365 | 7.0% | 9% | 40% | 86% |
| bc8 **d3** | **52K** | 8 | 0.324 | 8.8% | 14% | 40% | 78% |
| bc8 d2 | 13K | 8 | 0.257 | 12% | 17% | 35% | 68% |
| **pw5** | 208K | **5** | **0.385** | 5.1% | 6% | 40% | 85% |
| pw3 | 208K | 3 | 0.353 | 3.1% | 4% | 28% | 75% |
- **深度縮 > 寬度縮**:depth3(52K)defIoU 0.324 遠勝同參數 bc4(52K)的 0.177 → 第4下採樣層近乎冗餘
  (bottleneck 已覆蓋整顆),**depth=3 是更好的「最小模型」**(4× 小、~89% 效能)。
- **pos_weight=5 最佳**:defIoU **0.385**(勝 baseline 0.365、勝 S3 0.362)@ FP 5.1%。pw3 過抑。
- bend 各設定都在 floor(4-17%,且隨 FP 同升)→ 本質弱,確認。

### 選定 headline 模型:base_c=8 / depth=4 / pos_weight=5 / thr=0.7(defIoU 0.385)
depth=3 為「極小版」備選(52K)。

### 長版 15000 步(s5_long, pw5/thr0.7):★ 沒有過擬合 ★
- train/val loss 全程同步、雙雙平台化,val 從不回升 → **無過擬合**(runtime 生成=等效無限資料,
  無固定訓練集可記憶)。**收斂平台 ~step 5000-7000(best@6800)→ 實際需要步數 ≈ 5000-6000**;3000 略欠訓。
- **最終 headline(test, pw5, thr0.7):mIoU 0.721(S3 0.712)、pixel_acc 94.1%、defect IoU 0.39(S3 0.362)✓、
  remesh inst IoU 0.491(S3 0.276)。bend 仍 floor(本質弱,已坐實)。**

### S5 實驗階段 ≈ 收斂,剩報告/交付
1. ⬜ 用 thr=0.7 統一重生所有 report(舊 width 掃 run 存的是 thr0.5);可加 build_report thr 參數。
2. ⬜ **`.ipynb` 繳交橋(6/2 截止,優先度↑)**:薄 notebook import src + 跑 cli。
3. ⬜ 最終報告敘事彙整(飽和/深度>寬度/FP-thr/無過擬合/bend 誠實面)。
- ⬜ (低優)w_norm 旋鈕未 plumb;S3 的 A 圖(資料特性)未進 report。

### 已結案(歸檔 `docs/history/refactor_and_s4_review.md`)
- **程式大重構**(src/ 分層 + runtime 生成 + cli + output 角色分類)。
- **S4 review**:bend 崩潰根因 = defect 監督「任務形式不匹配」(整顆塗 + 逐像素 loss,對 bend 全域形狀最致命),
  已用程式 + 診斷實驗坐實;先前「物理天花板 / 柔光 HDRI」歸因已證偽作廢。

## S5 方向(已定案,機制與執行步驟見 `docs/stage5_plan.md`)
**核心單變因 = 壞件內部 ignore**:把 defect 監督裡「壞件內部」從『整顆塗 defect』改成『ignore(W≈0,不算 loss)』,
直接拔掉害死 bend 的矛盾(看起來正常的內部被逼當 defect)。
- 模型回 **S3 雙頭**(part 16→2 + defect 16→1 sigmoid)。⚠️ src 現行 `schema.HEADS` 是被否決的三頭(含七類),動工第一步要改回。
- **只改 Head2 的 target**:變形區=1(膨脹容忍帶)、好件/背景=0、內部 ignore + 高斯權重 W;loss = 加權 Dice+BCE + pos_weight。
- 變形區 = 同 pose normal vs defect patch 相減(資料已在,不需重渲;閾值圖 `docs/figures/route_a_diff_threshold.png`)。

### S5 概念性指南(沿用,避免重蹈覆轍)
1. **bend 崩 = 監督矛盾,不是容量/解析度/權重** → 修 target(interior ignore),別加參數、別調 loss 權重。
2. **一次只動一個變因**(handoff §6):核心驗證 = 「只把內部改 ignore,bend recall 從 0 起來」。workers / schema / test set / base_c 拆開做。
3. **逐像素 IoU 不是終極指標**:目標是「定位瑕疵零件」,評估轉定位式(峰值落在瑕疵件上 → per-part TP/FN/FP/TN)。
4. **可復現**:資料 `f(seed,i)` 100% 重現;訓練數字近乎重現(cuDNN 浮點微抖,要 bit 級需 deterministic flag)。
5. **runtime 生成保留**,防阻塞用 DataLoader workers(實測 workers=8 → 11×),非預存。
6. **名詞**:BCE=逐像素該不該亮(被多數類淹沒);Dice=整塊重疊率(抗不平衡);S3 用 Dice+BCE 把 defect IoU 0.004→0.362。

### 仍待辦(非 S5 主線,但截止前要顧)
- **`.ipynb` 繳交橋**:課程要 .ipynb 原始碼,我們是 src/ 套件 → 截止(6/2)前需一條橋(薄 notebook import src + 跑 cli)。

## 仍待處理 / 缺口
- 只有 `pan_head` 有 defect patch(72 張);socket/hex_nut/flange 無 → 「多零件 QC+Sorting」題目的真實缺口。
- `configs/` 數值序列化(`cli --config` 載入)尚未實作,現以 dataclass 預設為準。
