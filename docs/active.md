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

### 下一步(用戶定序 2026-05-31)
1. 🔄 **再往下縮 base_c {2,4,6}** 找飽和下邊界(bc2=13K params)。`output/s5_downsweep.log`。
2. ⬜ **壓 FP**:pos_weight↓(8→3/5)、defect_thr↑(0.5→0.6/0.7,推論端免重訓可先掃)、w_norm↑(0.3→0.5/0.8)。
3. ⬜ **長版**(選定 size+旋鈕後,如 8000-10000 步)最後跑。
- ⬜ 報告:S3 的 A 圖(資料特性,與模型無關)未進 report;`.ipynb` 繳交橋(6/2)。

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
