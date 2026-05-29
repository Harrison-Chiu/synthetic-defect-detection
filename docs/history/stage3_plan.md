# Stage 3 計畫 — Defect Detection 強化

> 開始：2026-05-27  
> Stage 2 baseline 結果見 [stage2_baseline_results.md](stage2_baseline_results.md)

---

## 為什麼有 Stage 3

Stage 2 baseline 結果：
- mIoU = 0.597, pixel_acc = 94%
- **defect IoU = 0.004**（壓倒性失敗）
- Confusion matrix：99.2% defect pixel 被預測成 normal_part

根因分析三條線：
1. **架構**：3-class softmax 把「part vs bg」（簡單）和「normal vs defect」（困難）強塞同一個 head，weighted CE 對 5% pixel minority 無能為力
2. **資料**：defect 視覺特徵太弱（特別是 displace），背景太單調（procedural smooth noise，幾乎純色）
3. **loss**：weighted CE 是 majority-vote 友善的，對 minority class 用 Dice / Focal 才有救

Stage 3 同時動三個面向。

---

## 改動清單

### A. 模型架構：Two-head Design B

| 項目 | Stage 2 | Stage 3 |
|------|---------|---------|
| 輸出 | 1 個 head，3-class softmax | **2 個 head** |
| Head 1 | — | part vs bg（2-class softmax 或 sigmoid） |
| Head 2 | — | normal vs defect，**只在 part 像素算 loss** |
| Encoder | 共用 | 共用（同 4-stage encoder-decoder + skip） |
| Decoder | 共用 | 共用，最後分兩條 1×1 conv 輸出兩個 head |

**Loss 設計**：
```
L = L_part + λ * L_defect
  L_part   = CrossEntropy(head1_logits, part_vs_bg_mask)
  L_defect = Dice(sigmoid(head2_logits[is_part]), defect_mask[is_part])
```
- λ = 1.0 起手，視訓練曲線調
- Head 2 只算 part 像素，bg 完全不影響它

**推論時**：先用 head 1 切 part/bg，在 part 區再用 head 2 切 normal/defect，組合回 3-class mask。

### B. Loss：CE + Dice 混合（給 Head 2）

Dice loss 公式：
$$L_{\text{Dice}} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

- $p_i$ = sigmoid 後的 defect 機率，$g_i$ = ground truth (0/1)
- 對 minority class 友善（不分像素數，看 overlap）
- 跟 CE 各 0.5 混合可保留 per-pixel 梯度的好處

### C. 資料集擴充

#### C.1 Displace 強度翻倍
- light: 0.05–0.15 → **0.15–0.30**
- heavy: 0.20–0.40 → **0.40–0.80**

對應 render_gotchas.md 留下的「displace 視覺效果太弱」的修正。

#### C.2 新增瑕疵類型：Remesh Sharp
取代 Bevel（Harrison 測試 Bevel 視覺幾乎無效），用 **Remesh modifier (mode='SHARP')** 製造「voxel 化的塊狀破損」感。

| State | octree_depth | scale | 視覺效果 |
|-------|--------------|-------|---------|
| remesh_light | 7 | 0.99 | 表面變粗糙、輕微多邊形化 |
| remesh_heavy | 5 | 0.99 | 明顯塊狀破損，邊緣破碎 |

#### C.3 背景策略 v2（Domain Randomization 風格）

**理由**：AmbientCG 真實紋理雖然「看起來像工廠」，但統計上仍是 stationary（整張紋理特徵均勻），
模型只要學「stationary 區 = bg / 非 stationary 結構 = part」就能 shortcut。所以 Stage 3 改採
混合策略 + 程序化干擾，逼模型學零件本身的視覺特徵。學界依據：Tobin et al. 2017 Domain Randomization。

**檔案組織**：
- `assets/backgrounds_real/`        — 5 張 Harrison 從 AmbientCG 34 候選挑出（cand 01/04/13/23/24）
- `assets/backgrounds_procedural/`  — 12 張 Stage 2 殘留 procedural（concrete/metal/rubber/wood）
- `assets/backgrounds_candidates/`  — 34 張下載候選 + preview.png（封存）

**配方（記在 composite.py 頂部常數，重要！）**：

| 項目 | 機率 / 範圍 |
|------|------------|
| **Base layer**（互斥三選一）| |
| ├─ `real`（AmbientCG 5 張隨機）| 50% |
| ├─ `procedural`（12 張 procedural 隨機）| 25% |
| └─ `solid`（純色+輕微 noise，9 色板）| 25% |
| **背景 augmentation**（每場景）| |
| ├─ 亮度 | ±25% |
| ├─ 對比 | ±20% |
| ├─ 彩度 | ±30% |
| ├─ 水平翻轉 | 50% |
| └─ 垂直翻轉 | 50% |
| **Distractor 數量分布**（非均勻、per scene）| |
| ├─ 0 個（clean） | 25% |
| ├─ 1–4 個（light） | 35% |
| ├─ 5–10 個（medium） | 25% |
| └─ 11–20 個（heavy） | 15% |
| **Distractor 形狀**（每個獨立抽）| |
| ├─ ellipse | 40% |
| ├─ rectangle | 25% |
| ├─ polygon (3–6 邊) | 15% |
| ├─ line | 15% |
| └─ ring（只有外框）| 5% |
| **Distractor 顏色** | 40% 工業金屬灰調 / 60% 全隨機 RGB |
| **Distractor 大小** | 5–80 px |
| **Distractor 透明度** | 0.5–1.0 |
| Solid base noise σ | 6 / 255 Gaussian |

**重要設計選擇**：
- Distractor 在貼零件**之前**畫，但 alpha composite，零件還是會在最上層
- semantic_mask / instance_mask **不會**標 distractor（它們算 bg）
- Distractor 顏色刻意包含金屬灰，避免模型走「非灰=part」shortcut

#### C.4 黑底對照組（ablation）
`scripts/gen_black_bg_test.py`：把 test split 100 張的 bg 全部塗黑（instance_mask==0 處 = 0,0,0）
→ 產生 `output/scenes_black/`。eval 時跑兩個 test set 對比 defect IoU，判斷背景策略影響。

| 對比情境 | 解讀 |
|---------|------|
| 黑底 IoU >> textured | 背景太雜，模型被干擾 |
| 黑底 IoU << textured | distractors 有效，模型學會零件真實特徵 |
| 兩者差不多 | 背景不是 bottleneck，問題在 defect signal 本身 |

### D. Debug 視覺化（給報告用）

訓練完跑 6 種 figure 到 `docs/figures/stage3/`：

| 編號 | 內容 | 目的 |
|------|------|------|
| A | normal vs defect 同位置 diff | 證明 defect 對人眼可辨識 |
| B | 模型 confidence heatmap（per-class softmax）| 看模型「考慮了什麼」 |
| C | per-defect-state IoU box plot | 定位哪種 defect 特別難 |
| D | False Positive / False Negative overlay | 失敗類型分類 |
| E | 訓練過程每 5 epoch 的預測 snapshot | 故事性，看模型怎麼學會的 |
| F | by-defect-state 5×3 confusion matrix | 細分到哪種 defect 被誤分 |

---

## 執行順序

### Phase 1（不需要互動，先準備）
1. ✅ 寫 `docs/stage3_plan.md`、更新 `CLAUDE.md`
2. ✅ 修改 `scripts/render_pan_head.py`（強化 displace + 加 remesh）
3. ✅ 修改 `scripts/composite.py`（7 defect states + bg augmentation）
4. ✅ 寫 `scripts/download_ambientcg.py`（下載候選）
5. ✅ 寫 `scripts/preview_bg_candidates.py`（生 contact sheet）
6. ✅ 寫 `notebooks/train_stage3.ipynb` 草案（two-head + Dice + epoch snapshots）

### Phase 2（互動）
1. ✅ **Harrison 挑背景**：05-27 挑出 cand 01/04/13/23/24 共 5 張 → 已搬到 `assets/backgrounds_real/`
2. ✅ Stage 2 procedural 12 張搬到 `assets/backgrounds_procedural/` 當第二層 pool
3. ✅ `composite.py` 加入 base layer 三選一 + 程序化 distractors 邏輯
4. ✅ `gen_black_bg_test.py` 寫好待用
5. Blender 跑 `render_pan_head.py`（252 張 = 36 pose × 7 state）← **待 Harrison 執行**
6. 跑 `scripts/composite.py` → 1000 場景重生成
7. 跑 `scripts/preview_scenes.py` 視覺檢查
8. 跑 `scripts/gen_black_bg_test.py` 出黑底對照組

### Phase 3（訓練 + 評估）
1. 跑 `train_stage3.ipynb`（estimated ~10 min）
2. 跑 `scripts/eval_stage3.py` 生 6 個 debug figure
3. 寫 `docs/stage3_results.md` 對比 Stage 2 → Stage 3 改善
4. Commit

---

## 預期評估指標

| 指標 | Stage 2 baseline | Stage 3 目標 |
|------|-----------------|-------------|
| mIoU | 0.597 | > 0.65 |
| defect IoU | **0.004** | **> 0.30** |
| pixel_acc | 94.0% | > 94% |
| instance precision | 0.336 | > 0.5 |
| instance recall | 0.821 | > 0.7 |

defect IoU 從 0.004 → > 0.30 是主要 KPI；如果突破不了，表示問題在資料而非模型，需要回頭加強 defect 視覺強度。

---

## 風險清單

| 風險 | 對策 |
|------|------|
| Remesh sharp 視覺效果過頭，零件不像零件了 | 先渲 1–2 張預覽，視覺檢查後再批次跑 |
| Two-head 訓練不收斂（λ 沒調好） | 從 λ=1.0 起手，觀察 head 1/head 2 loss 比例調整 |
| Dice loss 在 batch 早期所有 prediction=0 時梯度炸 | 加 epsilon 平滑（公式裡的 ε）|
| AmbientCG 下載失敗 | 用備用 procedural backgrounds（保留 legacy） |
| 重渲後 stage 2 baseline 不可比 | 訓練 stage 3 時同時保留 stage 2 模型，eval 時兩個都跑 |
