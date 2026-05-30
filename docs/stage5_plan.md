# Stage 5 計畫 — 修正 defect 監督形式 + 模型瘦身

> 定位:S4 review 已坐實 bend 崩潰根因(見 `docs/active.md` / `docs/bend_diagnosis_handoff.md`)。
> 本檔承接,規劃 S5 的修正方向、實驗與待決設計。**討論中,未定案的標 ⬜。**

---

## 0. 一句話方向

把 defect 監督從「整顆塗 + 逐像素 Dice」改成**「A:乾淨的 per-pixel 變形目標」+「c1:寬容的定位式評分」**,
讓 bend(全域形狀瑕疵)從 recall 0 拉起來;同時把模型瘦到「剛好飽和」。

---

## 1. 已定案決定

1. **保留 runtime 生成**(不改預存)。防阻塞靠 DataLoader workers —— 實測 workers=8 → 11×(純生成 30min→2.7min,且藏在 GPU 後)。
   → **TODO:** cli `--workers` 預設改 8(或進 TrainConfig);跑一次完整訓練量真實牆鐘時間。
2. **獨立 test set**:目前 val/test 共用同一批 100 張(`loop.py` L93/L126),test 偏樂觀。
   → **TODO:** 新增 `TEST_INDEX_OFFSET`(如 30_000_000),凍第二批 disjoint 場景;loop 收 val(選 best)與 test(報數字)分離。
3. **模型瘦身**:`base_c` 是寬度總旋鈕,參數 ∝ `base_c²`。
   實測:base_c 8→208K / 16→830K / 24→1.86M / 32→3.31M。詳見 §4 掃描計畫。
4. **修正路線 = A + c1 融合(同一個 defect 頭上的兩個角色,不是兩個頭)**:
   - **A(換 target)**:同 pose 的 normal vs defect patch 相減 → per-pixel「真正變形區」mask。資料已在 `output/patches/normal/`,**不需重渲**,離線算一次存 `*_defectmask.png`。
   - **c1(換評分)**:寬容、不對稱的定位 loss —— 正向寬容(抓到瑕疵區就給分,不要求填滿)、負向嚴格(背景/正常件誤亮就罰)。推論讀熱圖峰值,不分實例。
5. **暫不做** loss 動態加權(對 bend 無效甚至有害:bend 低是 label 矛盾不是權重不足)。

---

## 2. 模型逐層流動圖（`DefectSegNet`，base_c=32，輸入 256×256）

U-Net:4 層下採樣 → bottleneck → 4 層上採樣,每層 skip-concat。ConvBlock = 2×(3×3 conv+BN+ReLU)。

```
輸入  x                                  (3,   256, 256)
┌─ encoder（每層後 MaxPool2d 砍半）
│  enc1  ConvBlock 3→32        e1 =      (32,  256, 256) ─┐ skip
│  pool                                  (32,  128, 128)  │
│  enc2  ConvBlock 32→64       e2 =      (64,  128, 128) ─┼┐
│  pool                                  (64,   64,  64)  ││
│  enc3  ConvBlock 64→128      e3 =      (128,  64,  64) ─┼┼┐
│  pool                                  (128,  32,  32)  │││
│  enc4  ConvBlock 128→256     e4 =      (256,  32,  32) ─┼┼┼┐
│  pool                                  (256,  16,  16)  ││││
└─ bottleneck ConvBlock 256→256  b =     (256,  16,  16)  ││││  ← 感受野最大處
┌─ decoder（ConvTranspose ×2 放大，concat 對應 skip）
│  up4 256→128 →(128,32,32); cat e4 →(384,32,32); dec4→  (128, 32, 32) ◄┘│││
│  up3 128→64  →(64,64,64);  cat e3 →(192,64,64); dec3→  (64,  64, 64) ◄─┘││
│  up2 64→32   →(32,128,128);cat e2 →(96,128,128);dec2→  (32, 128,128) ◄──┘│
│  up1 32→16   →(16,256,256);cat e1 →(48,256,256);dec1→  (16, 256,256) ◄───┘
└─ d1 =                                   (16,  256, 256)  ← 所有 head 共用這張
   【S4 三頭】 head A part(16→2) / head B state(16→7) / head C type(16→4)
   【S5 雙頭】 head part (16→2, 留) + head defect (16→1, 新, 瑕疵定位熱圖)
              拿掉 state/type(aux、矛盾梯度來源、且非當前目標);
              backbone 完全不動,只換 d1 之後的頭。
```

### 三個關鍵認知
1. **三頭共用到最後一刻**,只在最後 1×1 conv 分岔。「aux 搶容量」不是搶參數(head 參數極少),是 **aux 的 loss 梯度回流擾動共用 backbone** → 影響的是訓練動態。
2. **bend 不是感受野/解析度問題**:bottleneck 16×16,單像素感受野早已覆蓋大半螺絲,模型「看得到」全域形狀。它崩是**輸出端被逼逐像素吐 defect 而監督矛盾**。
3. **S5 修正只動 d1 之後**:加/換 defect-localization 頭 + 換 loss,backbone 幾乎不動 → 改動範圍小、可控。

---

## 3. S5 算分設計（討論中,已收斂大方向）

**頭設計(已傾向):** 雙頭 `part`(bg/part)+ `defect`(1ch 熱圖)。state/type 拿掉,日後要分瑕疵種類再以 instance-level 接回。

**訓練 = MIL 式(用 GT instance 分零件算分),推論 = 讀熱圖峰值(不分實例)。**

**loss 形式(c1,寬容定位):**
```
L = λ_pos · sum_over_defparts( 1 − s_p )           # s_p = 該壞件區域峰值(或 top-k mean)
  + λ_neg(t) · mean_over_normalpart∪bg( s_i )        # 好件/背景響應壓低
```
- 正向用 **sum**(多抓多得,自然處理不定瑕疵數,符合 2/2 > 1/1 直覺);**只看峰值不要求填滿** → 對 bend 寬容。
- 負向**逐像素壓低** → 抑制誤報(嚴格)。
- **報告 metric** 另用 recall/precision 正規化(跨資料集才公平),與訓練 loss 分開。

**curriculum + 防退化(解「為了不扣分而不答」):**
- `λ_neg(t)` 從 ~0 線性 warm-up → 目標值:前期敢開火建 recall,後期收緊 precision。
- 不變式:warm-up 期保證「抓到一顆壞件的獎勵 > 不答省下的罰」→ 全不答必虧 → 強制開火。

**評估(per-part 定位):** 每顆零件區域峰值 >閾值 → 測為瑕疵。壞件中→TP/漏→FN;好件亮→FP/靜→TN。報 precision/recall/F-β + 混淆矩陣。

### ⬜ 仍待敲定
1. **FN vs FP 誰更糟?** 使用者初步偏「FP(誤報)更糟」(precision-favoring),與傳統 QC「FN(漏檢)更糟」相反 → 決定 λ 比重 / F-β 的 β。**待確認。**
2. **c1 target 怎麼從 A mask 生:** 直接用?膨脹幾 px?質心高斯 blob?(膨脹/模糊給位置容忍,對細邊的 bend 重要)。
3. **評估邊界:** ① 峰值橫跨兩相鄰零件算誰的;② 峰值落在零件「附近背景」算 TP 還 FP;③ 閾值怎麼定(掃 PR 曲線)。
4. **λ_pos / λ_neg 目標值與 warm-up 長度** 的具體數字(進 TrainConfig)。

---

## 4. base_c 掃描計畫

- **目的**:找最小的「飽和」寬度,降低每輪實驗成本(大模型 = 慢迭代,非好事)。
- **方法**:固定其他設定,掃 `base_c ∈ {8, 16, 24, 32}`,各跑一次,記 val mIoU / defect 指標 / 牆鐘時間。
- **判讀**:mIoU 隨 base_c 上升到某點後平緩(飽和)→ 選飽和起點。
- **前置**:先修 workers(否則每次 30min 掃 4 個太貴);掃描在「A+c1 改完」之後或之前做都行,但建議**先在現行 baseline 掃一次**當對照,改完再掃一次看修正是否改變飽和點。
- **注意**:bend 不會因減參數變差(label 問題非容量問題),所以掃描結果乾淨可信。
