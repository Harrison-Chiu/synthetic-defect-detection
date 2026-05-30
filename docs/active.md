# Active — 當前 stage 指標

> 本檔是 always-on context 的「當前 stage」指標,**永久保留**(空或極簡也留)。
> 規則:只放指標與待辦,**不複製** stage 報告內容;細節一律連到對應檔。
> stage 結案 review 完 → 整份搬 `docs/history/`,本檔重置為下一 stage。

---

## 狀態:S5 規劃中(report 日 2026-06-02)｜計畫詳見 `docs/stage5_plan.md`

### 已結案(歸檔 `docs/history/refactor_and_s4_review.md`)
- **程式大重構**(src/ 分層 + runtime 生成 + cli + output 角色分類)。
- **S4 review**:bend 崩潰根因 = defect 監督「任務形式不匹配」(整顆塗 + 逐像素 loss,對 bend 全域形狀最致命),
  已用程式 + 診斷實驗坐實;先前「物理天花板 / 柔光 HDRI」歸因已證偽作廢。

## S5 方向
方向已收斂為 **A + c1 融合**(同一個 defect 頭上的兩個角色,不是兩個頭):
- **A**:換 target —— 同 pose normal vs defect patch 相減 → per-pixel「真正變形區」mask(資料已在,不需重渲)。
- **c1**:換評分 —— 寬容定位 loss(正向寬容:抓到瑕疵區就給分;負向嚴格:背景/正常件誤亮就罰),推論讀熱圖峰值、不分實例。

### S5 概念性指南(沿用,避免重蹈覆轍)
1. **bend 崩 = 監督形式不匹配,不是容量/解析度/權重** → 修標籤與評分,別加參數、別調 loss 權重。
2. **一次只動一個變因**(handoff §6):A、c1、base_c、test set 拆開驗。
3. **逐像素 IoU 不是終極指標**:目標是「定位畫面中的瑕疵零件」,評估該轉成定位式(峰值落在瑕疵件上)。
4. **可復現**:資料 `f(seed,i)` 100% 重現;訓練數字近乎重現(cuDNN 浮點微抖,要 bit 級需 deterministic flag)。
5. **runtime 生成保留**,防阻塞用 DataLoader workers(實測 workers=8 → 11×),非預存。

### 仍待辦(非 S5 主線,但截止前要顧)
- **`.ipynb` 繳交橋**:課程要 .ipynb 原始碼,我們是 src/ 套件 → 截止(6/2)前需一條橋(薄 notebook import src + 跑 cli)。

## 仍待處理 / 缺口
- 只有 `pan_head` 有 defect patch(72 張);socket/hex_nut/flange 無 → 「多零件 QC+Sorting」題目的真實缺口。
- `configs/` 數值序列化(`cli --config` 載入)尚未實作,現以 dataclass 預設為準。
