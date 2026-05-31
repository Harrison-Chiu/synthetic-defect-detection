# Active — 當前 stage 指標

> 本檔是 always-on context 的「當前 stage」指標，**永久保留**（空或極簡也留）。
> 規則：只放指標與待辦，**不複製** stage 報告內容；細節一律連到對應檔。
> stage 結案 review 完 → 整份搬 `docs/history/`，本檔重置為下一 stage。

---

## 狀態：S6 交付階段（report 日 2026-06-02）

- **S5 成果**：`docs/stage5_results.md`（已改寫為可讀版本）
- **S6 計畫**：`docs/stage6_plan.md`
- **圖表規劃（待討論收斂）**：`docs/stage5_figure_plan.md`

### S5 一句話成果
interior-ignore 監督（壞件內部 W≈0）+ 回 S3 雙頭 → **defect IoU 0.362→0.39（勝 S3）**；
模型嚴重過參數化（bc8 即飽和）；合成資料**無過擬合**；
bend 不再崩 0 但本質仍弱（delta ≈ 1–6%，接近 FP floor）。

### S6 待辦（優先序）
1. ⬜ **圖表集討論** → 收斂後修改 script → 一次生成（`stage5_figure_plan.md`）
2. ⬜ **per-type eval 加進 training loop** → 指定步數驗證跑
3. ⏸ **`.ipynb` 繳交橋**（課程要求，6/2）— 待 Harrison 決定範圍
4. ⬜ **最終報告敘事**

### 結構性缺口（跨 stage）
- 只有 `pan_head` 有 defect patch（72 pose）；socket/hex_nut/flange 無
- Part 頭不區分零件種類（bg / 好件 / 壞件，非 socket / pan / hex / flange）
- `configs/` 數值序列化未實作

---

## 已結案（歸檔）
- `docs/stage5_results.md` — S5 完整成果（待 S6 結案後可移 history/）
- `docs/history/refactor_and_s4_review.md` — 程式大重構 + S4 review
