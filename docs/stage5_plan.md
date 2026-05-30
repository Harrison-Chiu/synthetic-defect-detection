# Stage 5 計畫 — 修正 defect 監督形式 + 模型瘦身

> 定位:S4 review 已坐實 bend 崩潰根因(見 `docs/active.md` / `docs/bend_diagnosis_handoff.md`)。
> 本檔承接,規劃 S5 的修正方向、實驗與待決設計。**討論中,未定案的標 ⬜。**

---

## 0. 一句話方向（已定案）

**核心單變因:把 defect 監督裡「壞件內部」從『整顆塗成 defect』改成『ignore(不算 loss)』** —— 直接拔掉害死 bend 的監督矛盾(看起來正常的內部像素被逼當 defect)。
模型回到 **S3 雙頭**(乾淨基準),**只改 Head2 的 target**:變形區=1、內部 ignore、好件/背景=0,外加一張高斯權重圖 W;loss 沿用 S3 的 Dice+BCE 再加 pos_weight。同時把模型瘦到剛好飽和。

---

## 1. 已定案決定

1. **保留 runtime 生成**(不改預存)。防阻塞靠 DataLoader workers —— 實測 workers=8 → 11×(純生成 30min→2.7min,且藏在 GPU 後)。
   → **TODO:** cli `--workers` 預設改 8(或進 TrainConfig);跑一次完整訓練量真實牆鐘時間。
2. **獨立 test set**:目前 val/test 共用同一批 100 張(`loop.py` L93/L126),test 偏樂觀。
   → **TODO:** 新增 `TEST_INDEX_OFFSET`(如 30_000_000),凍第二批 disjoint 場景;loop 收 val(選 best)與 test(報數字)分離。
3. **模型瘦身**:`base_c` 是寬度總旋鈕,參數 ∝ `base_c²`。
   實測:base_c 8→208K / 16→830K / 24→1.86M / 32→3.31M。詳見 §4 掃描計畫。
4. **模型回 S3 雙頭**(part 16→2 + defect 16→1 sigmoid)。
   ⚠️ **重要事實**:src 目前的 `schema.HEADS`(A/B/C,含七類 state)是**被否決的 S4 三頭 multi-head**,不是 S3 雙頭。
   → S5 動工第一步 = 把 schema 改回 S3 雙頭(順手解決「七類不該在」)。S4-best 與 S3 同為雙頭(見 `model_architecture_handoff.md §4`)。
5. **修正 = 只改 Head2 的 target**(換 target,不動架構;見 §3 機制):
   - **A(變形區來源)**:同 pose normal vs defect patch 相減 → 變形區 mask。資料已在,**不需重渲**,離線算一次存 `*_defectmask.png`。
   - **interior ignore + 高斯權重 W**:壞件內部 ignore、好件/背景=0、變形區=1。loss = 加權 Dice+BCE + pos_weight。
6. **暫不做** loss 動態加權 / curriculum(出問題再加)。bend 低是 label 矛盾不是權重不足,動態加權對它無效甚至有害。

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

## 3. S5 機制(講解版,已定案計算流程)

**核心:壞件內部 ignore = 拔掉矛盾。** 其餘(高斯 W、pos_weight、容忍帶、推論對位)是讓它更穩/更好用的配套。

### 離線一次（每個 patch:pose × defect_state）
```
M = despeckle( mean_RGB|defect_patch − normal_patch| > thr )     # 變形區,二值。thr≈25–30
```
存成 patch 旁的 `*_defectmask.png`。閾值佐證見 `docs/figures/route_a_diff_threshold.png`
(thr=30 時:bend_heavy 36% / bend_light 5% / displace_h 4% / remesh_h 43% 區;bend 變形集中頭緣+輪廓,內部≈0)。
- bend 幾何註記:SIMPLE_DEFORM 繞原點彎(根固定、尖端彎),normal/defect 共用原點 → 相減已正確配準,**不要按質心歸零**(會把對齊的根推開、製造假 diff)。

### 每張場景（組 defect 頭的 target T 與 weight W,256×256）
```
D   = 所有壞件的變形區（各自 M 貼到場景位置）
D⁺  = D 膨脹 3–5 px（容忍帶:亮在附近也算對）
N   = 所有正常件的剪影

Target T(x) = 1   if x∈D⁺        （該亮）
            = 0   otherwise       （正常件/背景:不該亮;壞件內部 T 無所謂,因 W≈0）

Weight W(x) = max( GaussBlur(D⁺, σ=5–8px) ,  w_norm·1{x∈N} )
              遠背景 & 壞件內部 → 0   （= ignore,矛盾就消在這）
```
- 高斯 W = 你要的「漸層 ignore 圖」:變形區附近 W 高、向外淡出;遠處 W≈0 = 不在意。
- σ 太大 → 糊進內部把 ignore 變成「T=0 輕罰」(病復發);σ 太小 → 退化成硬 mask 無容忍。σ 5–8 + 膨脹 3–5 剛好。

### loss
```
L_defect = Σ_x W(x)·BCE_pw( ŝ(x), T(x) ) / Σ_x W(x)   # +可選 Dice(吃 W);pos_weight=ρ 補正類稀少
L_part   = CE( part_logits, bg/part mask )             # 不變
L_total  = L_part + λ·L_defect
```
- **BCE**:逐像素「該亮沒亮/不該亮卻亮」,精準但被多數類淹沒。**Dice**:看整塊重疊率,抗極端不平衡。S3 用 Dice+BCE 把 defect IoU 從 0.004→0.362,故保留。
- **pos_weight ρ**:固定值(非動態),補變形區像素太少(bend_light 僅 5%)。

### 推論
`ŝ` 取峰值/閾值 → 亮區與 part mask 對位 → **某顆零件上有亮 ⇒ 判整顆為瑕疵件**(「整顆是壞件」的語意在此實現,不在像素監督)。

### ⬜ 仍待敲定的數值/規則（執行時定）
1. **thr(25–30)** + despeckle 大小;**σ(5–8)** + **D⁺ 膨脹(3–5)**。
2. **pos_weight ρ**(估 5–10);**λ**(defect vs part);**w_norm**(好件權重)、背景 W(0 或留一點)。
3. **是否保留 Dice**、Dice 是否吃 W。
4. **推論**:閾值 + 「熱圖 blob 算哪顆零件」的重疊規則;評估邊界(峰值橫跨兩零件、落在零件附近背景算 TP/FP)。
5. **FN vs FP 取捨**(影響評估 F-β 的 β;訓練端先對稱,出問題再調)。

---

## 3.5 執行順序（動工步驟,compact 後照此走）

1. **修 workers**:cli `--workers` 預設 8(或進 TrainConfig),跑一次量真實牆鐘。
2. **schema 改回 S3 雙頭**:`src/schema.py` 拿掉 B(state,七類)/C(type),defect 頭改 `16→1` sigmoid(對齊 S3/S4-best)。同步改 model heads、metrics、losses 對 head 的引用。
3. **離線產變形區 mask**:寫一支 script 對 `output/patches/<state>/*` 用 §3 公式算 `*_defectmask.png`(thr 用閾值圖挑)。
4. **改 `encode_targets`**:defect 頭吐 (T, W) —— 變形區=1、膨脹容忍帶、好件/背景=0、壞件內部與遠背景 W≈0。
5. **改 loss**:加權 BCE(+可選 Dice,吃 W)+ pos_weight。
6. **獨立 test set**:加 `TEST_INDEX_OFFSET`,freeze 第二批;loop 把 val(選 best)/ test(報數字)分離。
7. **跑 S5 vs S3 baseline**:看 bend recall 是否從 0 起來(核心驗證:單變因 = interior ignore)。
8. **base_c 掃描**(見 §4):找飽和寬度。

> **驗證的核心命題**:「只把壞件內部從『塗 defect』改成『ignore』,bend recall 就該從 0 抬起」。這是單變因實驗,別一次混入多個改動。

## 4. base_c 掃描計畫

- **目的**:找最小的「飽和」寬度,降低每輪實驗成本(大模型 = 慢迭代,非好事)。
- **方法**:固定其他設定,掃 `base_c ∈ {8, 16, 24, 32}`,各跑一次,記 val mIoU / defect 指標 / 牆鐘時間。
- **判讀**:mIoU 隨 base_c 上升到某點後平緩(飽和)→ 選飽和起點。
- **前置**:先修 workers(否則每次 30min 掃 4 個太貴);掃描在「A+c1 改完」之後或之前做都行,但建議**先在現行 baseline 掃一次**當對照,改完再掃一次看修正是否改變飽和點。
- **注意**:bend 不會因減參數變差(label 問題非容量問題),所以掃描結果乾淨可信。
