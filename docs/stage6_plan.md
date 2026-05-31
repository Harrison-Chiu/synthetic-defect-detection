# Stage 6 計畫 — 圖表交付 + 訓練可觀測性 + 最終報告

> 前置：S5 實驗已收斂（[stage5_results.md](stage5_results.md)），進入交付階段。
> 報告日：2026-06-02

---

## 0. S6 的定位

S5 完成了核心實驗（interior-ignore 監督 + 模型瘦身），但**交付物尚未到位**：
- 沒有一份完整的圖表集
- Per-run HTML 報告品質不足（隨機選圖、tofu、缺 per-type 訓練動態）
- 最終報告敘事未寫
- .ipynb 繳交橋未做

S6 不做新的模型/監督實驗，專注交付。唯一的「訓練」是加入 per-type eval 後的驗證跑。

---

## 1. 工作項目

### 1.1 S5 圖表集（需先討論收斂 → 再跑）

**狀態**：`scripts/gen_stage5_figures.py` 已寫好但未跑。`docs/stage5_figure_plan.md` 有初步規劃，但需根據以下討論結果修改後再生成。

**待討論 → 決定後更新 figure plan → 修改 script → 一次生成**：
- 選圖策略（按 type 選 vs 隨機）
- 圖表語言（英文，已確認）
- 補強圖（training curves 強化版、bend honesty 圖、dataset showcase）
- 缺口圖（架構圖、PR 曲線）

### 1.2 Per-type eval 加進 training loop

**目的**：看三種瑕疵在訓練過程中的表現走勢（bend 是持續低迷？有階段波動？）

**改動範圍**：
- `src/eval/metrics.py`：`evaluate_all` 多回傳 per-state det_rate
- `src/train/loop.py`：history 多記 per-state 欄位
- 指定步數跑一次驗證（不照跑 s5_long 15000 步，用 headline config）

**注意**：這會讓每次 eval 的時間稍增（需掃 per-instance 匹配），但不影響訓練本身。

### 1.3 .ipynb 繳交橋

課程要求 .ipynb 原始碼。範圍待 Harrison 決定（含訓練到什麼程度）。

### 1.4 最終報告敘事

可直接取材 `stage5_results.md`（已改寫為可讀版本）。

---

## 2. 優先序

1. **討論 figure plan** → 收斂後一次生成圖表
2. **per-type eval** → 驗證跑 → 把結果回填進圖表
3. .ipynb 繳交橋
4. 最終報告敘事

---

## 3. 已知事實（S5 遺留，S6 需面對）

### 3.1 Bend 的本質限制

Bend 在所有 S5 run 中 delta（det − FP）僅 1–6%，不隨容量/步數改善。
這是**視覺訊號本質弱**的問題，不是模型/監督的問題。
報告中需誠實呈現此限制。

### 3.2 寬度掃描 thr 不一致

寬度掃描的 7 個 run（s5_bc2..bc32）用 thr=0.5 訓練/報告，headline 用 thr=0.7。
數字不可直接並排。圖表中需標註此差異，或統一用 thr=0.7 重算（後處理，不需重訓）。

### 3.3 Val loss < Train loss

原因已確認：BatchNorm eval/train mode 差異 + 固定 val set。
非資料洩漏（index 區段完全互斥）。報告中需簡要解釋。
