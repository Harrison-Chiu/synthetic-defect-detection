# Stage 3 — 訓練結果與分析

> 訓練：2026-05-27  
> 計畫：[stage3_plan.md](stage3_plan.md)  
> 對比：[stage2_baseline_results.md](stage2_baseline_results.md)  
> 圖：[docs/figures/stage3/](figures/stage3/)

---

## 1. KPI 對比

| 指標 | Stage 2 baseline | Stage 3 目標 | **Stage 3 實際** | 達成 |
|------|------------------|--------------|------------------|------|
| Test mIoU | 0.597 | > 0.65 | **0.712** | ✅ |
| Test pixel_acc | 94.0% | > 94% | **94.1%** | ✅ |
| Test bg IoU | 0.992 | — | 0.983 | — |
| Test normal_part IoU | 0.789 | — | 0.790 | ≈ |
| **Test defect IoU** | **0.004** | **> 0.30** | **0.362** | ✅ **(90×)** |

主 KPI（defect IoU）從 0.004 → 0.362，**90 倍改善**，超過 0.30 目標。

---

## 2. Stage 3 三大改動回顧

| 改動 | Stage 2 | Stage 3 |
|------|---------|---------|
| **架構** | 3-class softmax 單 head | Two-head Design B：part/bg + defect 分離 |
| **Loss** | weighted CE (w=[0.17, 0.52, 2.30]) | Head 1 CE + Head 2 (Dice 0.5 + BCE 0.5)，λ=1.0 |
| **資料 — defect** | bend×2 + displace×2（弱）| bend×2 + **displace×2（強度翻倍）** + **remesh sharp×2**（取代 Bevel）|
| **資料 — bg** | 12 程序化單一紋理 | **Domain Randomization v2**：5 真實 AmbientCG + 12 procedural + 9 純色，加程序化 distractors |
| **Optimizer** | SGD lr=0.01 | Adam lr=1e-3, weight_decay=1e-4 |

---

## 3. Per-defect-state 細部分析（FIG C / F）

每個 defect state 在 test set 上的 per-instance defect IoU（mean）：

| Defect state | mean IoU | 像素中被預測為 `defective_part` 的比例 | 解讀 |
|--------------|---------:|----------------------------------:|------|
| bend_light   | **0.006** |  3.4% | 模型完全看不到 |
| bend_heavy   | **0.024** |  4.7% | 同上，連 45° bend 都不行 |
| displace_light | 0.391 | 78.0% | 偵測率高 |
| displace_heavy | **0.428** | **93.2%** | 最強的 defect signal |
| remesh_light | 0.155 | 32.4% | 中等 |
| remesh_heavy | 0.276 | 52.3% | 中等 |

### 為什麼 Bend 失敗？

256×256 解析度 + 80mm 相機距離下，pan_head 螺絲身長度只有 ~50 像素。Bend modifier 沿 X/Y 軸彎曲，視覺結果是「螺絲身輪廓微微側偏」，但對單個 instance 來說：
- Silhouette 變化只發生在邊緣 1-3 像素
- 主體紋理（金屬反射、HDRI lighting）完全不變
- 跟「螺絲被旋轉到不同角度」幾乎無法區分

**這是 data signal 的問題，不是 model capacity 問題。** 後續若要拉 bend IoU：
1. 拉大 camera focal length 或縮短距離 → bend 在像素上更明顯
2. Bend angle 拉到 60°+（已接近物理斷裂，但 visually distinguishable）
3. 從旁邊（el ≈ 0）拍攝增加 → 側面看 bend 比俯視明顯

### 為什麼 Displace 最強？

Displace modifier 用 CLOUDS 紋理改變表面法線，**整個零件表面都變得粗糙**，每個像素的反射強度都受影響。signal 是 dense 的，CNN 學習很容易。

### Remesh sharp 為什麼中等？

Voxel 化會讓邊緣破碎、表面塊狀化，**邊緣變化明顯但內部紋理沒變**。signal 是 sparse 的（集中在邊緣），CNN 需要學「邊緣的 jaggedness」這個比較抽象的特徵，所以比 displace 慢但比 bend 強。

---

## 4. 背景策略 ablation（textured DR vs 黑底）

| 指標 | Textured DR bg | Black bg（控制組）| 差異 |
|------|---------------:|------------------:|------:|
| pixel_acc | 94.15% | 92.77% | -1.4% |
| mIoU | 0.712 | 0.678 | -0.034 |
| bg IoU | 0.983 | 0.979 | -0.004 |
| normal IoU | 0.790 | 0.737 | -0.053 |
| **defect IoU** | **0.362** | **0.319** | -0.043 |

**解讀**：
- 黑底 IoU 略低於 textured，差距小但一致
- 反直覺：黑底「理論上更乾淨」，model 應該預測更準，但實測反而稍差
- 推論：**模型在 textured DR bg 上訓練後，已學會零件本身的視覺特徵；黑底是 OOD（out-of-distribution）情境，model 對「純黑」沒見過，反而稍微受擾**
- 但差距很小（< 5pp），證明 DR 背景成功「讓模型不依賴背景訊號」——這正是 Domain Randomization 的目標

如果 stage 2 的 model 跑黑底，預期 defect IoU 會掉到接近 0（因為它本來就靠背景 shortcut）。

---

## 5. 訓練動態（FIG E / training_curves）

- Epoch 1–9: defect IoU = 0（model 全部預測為 normal，因 Dice + BCE warmup）
- Epoch 10: defect IoU 第一次 > 0（0.07）
- Epoch 13: 突破 0.20 — 模型「發現」defect signal
- Epoch 16–17: 穩定 0.32–0.33
- Epoch 26: 達到 best val mIoU = 0.716 (defect IoU = 0.384)
- Epoch 30: 0.347（輕微 overfit / 不穩定，best 已存）

Loss 分解：
- L_part 從 0.39 → 0.04（part/bg 很快學會）
- L_dice 從 0.76 → 0.53（dice loss 慢慢降）
- L_bce 從 0.56 → 0.42（BCE 配合 dice 一起下降）

---

## 6. 報告六張圖速查（docs/figures/stage3/）

| File | 內容 | 主要發現 |
|------|------|---------|
| `A_normal_vs_defect_diff.png` | 同 pose normal vs 6 defect state，疊 diff heatmap | 證明 displace / remesh 對人眼可辨識；bend 幾乎不可辨識 |
| `B_confidence_heatmap.png` | Head 1 P(part) + Head 2 P(defect) | 兩 head 各自學到不同東西，head 1 銳利，head 2 軟 |
| `C_per_state_iou_box.png` | 6 defect state × per-instance IoU box | bend ≈ 0, displace 最強，remesh 中等 |
| `D_fp_fn_overlay.png` | best/miss/FA/clean 4 case 的 TP/FP/FN overlay | 失敗模式可視化 |
| `E_epoch_snapshots.png` | epoch 1/5/10/15/20/25/30 的同 val sample 預測 | 故事性：從全預測 bg → 學會 part → 學會 defect |
| `F_by_state_confusion.png` | 7 defect_state × 3 pred class confusion | 量化每個 defect 被誤分到哪裡 |
| `ablation_textured_vs_black.png` | textured DR vs 黑底 IoU bar | DR 策略沒掉 IoU，model 沒靠背景 shortcut |
| `training_curves.png` | Loss 分解 + per-class val IoU | 訓練收斂分析 |

---

## 7. 結論與後續方向

### Stage 3 證明的事

1. **Two-head 架構顯著優於單 head 3-class** — defect IoU 0.004 → 0.362（90×）
2. **Dice + BCE 對 class imbalance 必要** — weighted CE 在 5% minority 上學不起來
3. **Displace 是最好的 visual defect signal** — 整面紋理變化最易學
4. **Domain Randomization 有效** — model 不依賴背景，黑底測試只掉 0.04 mIoU
5. **Bend 在當前 setup 下幾乎不可學** — 物理限制，需要拉強度或改視角

### 報告口頭重點

- **故事線**：Stage 2 失敗（defect IoU 0.004 是 majority-vote 退化）→ 三個維度同步改 → defect IoU 提升 90×
- **核心技術**：two-head 解耦難易任務 + Dice loss 對抗 class imbalance + DR 強迫 model 學零件特徵
- **誠實面**：Bend 我們承認解不開，這是 sim2real / data signal 的限制，不是模型容量問題

### 未做但記下

- 拉 bend 強度到 60°+（[future_ideas.md](future_ideas.md)）
- 加 instance 分離 head（panoptic head）— 留給組員模型方向
- 真實圖片 fine-tune 評估 sim2real gap

