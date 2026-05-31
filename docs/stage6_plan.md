# Stage 6 計畫 — Dataset 改進 + 監督修正 + 圖表交付

> 前置：S5 實驗已收斂（[stage5_results.md](stage5_results.md)），進入交付階段。
> 報告日：2026-06-02

---

## 0. S6 的定位

S5 完成了核心實驗（interior-ignore 監督 + 模型瘦身），但有幾個可直接改進的方向：
- Dataset 品質可提升（角度、重疊、bend 強度不足）
- 監督編碼有修正空間（T/W 設計）
- 圖表集未完成

S6 = **Dataset 改進 → 監督修正 → 重訓一次 → 圖表交付**。

---

## 1. Dataset 改進（需 Blender 重渲）

### 1.1 俯仰角限制：±30° 內

**現狀**：patches 有 6 個 elevation（-60, -30, -10, +10, +30, +60），±60° 太立。
**改動**：只保留 el = -30, -10, +10, +30（4 個），刪掉 ±60°。

**實作方式 A（不重渲）**：修改 `src/data/assets.py` 載入時 filter filename。
**實作方式 B（重渲）**：Blender script 只渲 4 個 elevation。

→ **選 A**（省時，patch 已渲好只是不用 ±60° 的）。

### 1.2 Defect states 簡化：3 種

**現狀**：6 個 defect state（light + heavy × 3 types）。
**改為**：

| State | 說明 | Blender 參數 |
|-------|------|-------------|
| `bend_45` | 45° SIMPLE_DEFORM bend（現有 light=15°/heavy=35° 都不夠強）| **需重渲** |
| `displace_heavy` | 維持現有 | 沿用 |
| `remesh_light` | 維持現有 | 沿用 |

- 刪除：bend_light, bend_heavy, displace_light, remesh_heavy
- 新增：bend_45（需 Blender 重渲 + gen_defectmask）

**需修改**：
- `src/data/config.py`：`DEFECT_STATES_DEFECTIVE` 改為 3 個
- `src/schema.py`：`STATE_CLASSES` 更新
- `src/data/assets.py`：載入時只認新的 3 個 state 目錄
- Blender：渲 bend_45 patches（3 az × 4 el × 4 hdri = 48 張 + 48 defectmask）

### 1.3 場景生成改進

**現狀**：
- `parts_range = (3, 7)` → 太多零件堆疊
- 位置完全隨機 → 大量重疊
- scale 範圍 (0.5, 1.0) → 同場景大小差異可能很大

**改為**：
- `parts_range = (2, 4)` — 最多 4 顆（Harrison：重疊不超過兩個）
- 加碰撞檢查：`min_separation_px = 8`（零件 bbox 至少間隔 8px）
- `scale_range = (0.65, 0.85)` — 同場景尺寸更一致
- 保留跨場景的隨機性

**需修改**：`src/data/config.py`（GenConfig）+ `src/data/generator.py`（碰撞檢查邏輯）

---

## 2. 監督編碼修正

### 2.1 T = 整顆壞件（非僅變形區）

**現狀**：`T = dilated defect_region`（只有差異邊框 = 1，壞件內部 T = 0）
**改為**：`T = 整顆 defective instance mask`（壞件全體 = 1）

原因：W ≈ 0 的內部像素無論 T = 0 或 1 都不影響 loss，但 T = 1 讓 W 過渡帶的目標正確（壞件就是壞件，不該是 0）。

**需修改**：`src/data/dataset.py` 的 `encode_targets` — T 改用 `(semantic == CLS_DEFECT)`。

### 2.2 Gaussian sigma 加寬：6 → 14

**現狀**：`DEFECT_GAUSS_SIGMA = 6.0` — W 衰減太快，壞件內部幾乎全 ignore。
**改為**：`DEFECT_GAUSS_SIGMA = 14.0` — 過渡帶更寬，壞件邊緣有更多梯度訊號。

效果（cross-section 已驗證）：sigma=14 讓壞件邊緣的 W 從 ~0 提升到 ~0.3，跟好件的 W_NORM=0.3 同量級。

**需修改**：`src/data/config.py` — 改一個數字。

---

## 3. 訓練

用修正後的 dataset + 監督跑一次訓練：
- 配置：bc=8, depth=4, pw=5, thr=0.7, **4000 steps**（Fig 08 確認 ~4000 步即收斂）
- Best checkpoint 用 **val defect IoU**（非 mIoU）選取
- 加 per-type eval：history 記錄 per-state det_rate

---

## 4. 圖表交付

S5 圖表已生成（15 張，`docs/figures/stage5/`），S6 訓練完後：
- 用 S6 模型重生成所有圖表
- 或保留 S5 圖表 + 追加 S6 comparison

---

## 5. 優先序（順序 = 依賴鏈）

1. ✅ S5 圖表初版已生成（15 張）
2. ⬜ **Blender 重渲 bend_45**（~48 張 patch + defectmask）
3. ⬜ **Code 修改**（elevation filter + state 簡化 + GenConfig + T/W 修正）
4. ⬜ **訓練一次**（4000 steps，~30 min）
5. ⬜ **重生成圖表**（用 S6 模型）
6. ⬜ .ipynb 繳交橋
7. ⬜ 最終報告敘事

---

## 6. S5 遺留事實（仍適用）

### 6.1 Val loss < Train loss
原因已確認：BatchNorm eval/train mode 差異 + 固定 val set。非資料洩漏。

### 6.2 寬度掃描 thr 不一致
寬度掃描用 thr=0.5，headline 用 thr=0.7。圖表已用 thr=0.7 post-hoc 重算。
