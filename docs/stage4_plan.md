# Stage 4 計畫 — Multi-head 監督 + Bend 視覺信號修正

> 起草：2026-05-28（討論中，未開工）
> 前情：[stage3_results.md](stage3_results.md)、[stage3_plan.md](stage3_plan.md)
> 目的：解決 Stage 3 剩下的兩個瓶頸 — (1) **bend 看不見**（F2 顯示 head 2 只給 3-5% defect probability），(2) **單一 binary defect head 標籤利用率低**（meta 已含 defect_state / type / severity 但訓練只用 binary）

---

## 1. Stage 3 診斷回顧（為什麼要做 Stage 4）

### 1.1 KPI vs 失敗點

| 類別 | Test defect IoU |
|------|----------------:|
| displace_heavy | 0.428 ✅ |
| displace_light | 0.391 ✅ |
| remesh_heavy | 0.276 🟡 |
| remesh_light | 0.155 🟡（看起來幾乎不可辨）|
| bend_heavy | **0.024** ❌ |
| bend_light | **0.006** ❌ |

### 1.2 拆 head 後的明確診斷（FIG F1 / F2）

- **Head 1 (part/bg)**：所有 defect state 都被 96-98% 正確認為 part → **head 1 沒問題**
- **Head 2 (defect/normal, 獨立評估)**：
  - bend_light/heavy 只有 3-5% 被認為 defect ← **head 2 完全看不到**
  - displace 95%/81% ← 強
  - remesh 33%/54% ← 中等

→ Bend 失敗 **100% 源自 head 2 看不到 visual signal**，不是 head 1 漏 part，也不是模型容量。

### 1.3 兩個 root cause 推論

| Root cause | 證據 | Stage 4 對策 |
|------------|------|--------------|
| **A. Bend axis 隨機選 X/Y 導致 50% 樣本「彎進畫面」** | `render_pan_head.py` L143 `rng.choice(["X","Y"])`。沒考慮相機 az。 | Bend axis 跟相機 az 對齊（±30° jitter），讓 bend 弧落在垂直視線方向 |
| **B. Binary defect head 把 light/heavy/3 種 type 全部塞進 1 個 sigmoid 輸出**，weak signal 被強 signal 主導 | F2 數字 + 訓練 loss 行為 | 改 multi-head 監督（state + type）|

### 1.4 額外發現的 bug

- **HDRI 數量**：`render_pan_head.py` 寫死 2 個 HDRI（`university_workshop`、`crossfit_gym`），但 `assets/hdri/` 實際有 4 個 EXR（多了 `monochrome_studio_02`、`pretoria_gardens`）。Stage 4 修正 → 用全部 4 個。
- **Scene 內 HDRI 不一致**：parts 階段每張用單一 HDRI 渲染，composite 階段隨機抽 parts → 同一 scene 內不同 instance 來自不同 HDRI 的反射，物理上不合理。Stage 4 修正 → 同 scene 只從同一 HDRI 的 part subset 抽。

---

## 2. ✅ 定案內容（已確定的設計）

### 2.1 Architecture — Multi-head V4 (A + B + C + E)

| Head | Output 維度 | Label space | Loss | 推論用途 |
|------|------------|-------------|------|---------|
| **A** part/bg | 2-way softmax | bg / part | CE | gate（loss 限制 part 像素）+ 報告 |
| **B** defect_state | **7-way softmax** | normal / bend_l / bend_h / disp_l / disp_h / remesh_l / remesh_h | CE + multi-class Dice | 細粒度 confusion matrix、報告主賣點 |
| **C** defect_type | **4-way softmax** | none / bend / disp / remesh | CE + Dice | 工廠 QC 想知道哪類瑕疵 |
| **E** binary defect | 1, sigmoid | normal / defect | BCE + Dice | 主 KPI（跟 Stage 3 比較）|

- **砍掉 D severity** — light/heavy 是 per-instance label，pixel-level 強硬 broadcast 後 noise 大，不適合 pixel-level head
- **共用 encoder-decoder**，只在最後一層分歧出 4 個 head（4 個 1×1 conv）
- 額外參數成本：(2+7+4+1) × c/2 channels ≈ 多幾百 params 而已

### 2.2 Loss — 先用 weighted sum

```
L_total = L_A + α_B (L_B_CE + L_B_Dice) + α_C (L_C_CE + L_C_Dice) + α_E (L_E_BCE + L_E_Dice)
```

- α_B = α_C = α_E = 1.0 開始試
- **Uncertainty Weighting (Kendall 2018) 已記錄但不在 Stage 4 採用**，留待後續優化（見 §4 延後項）
- B/C/E 的 loss 都 gate 在 head A 預測為 part 的像素（或 GT part，看實作）

### 2.3 Bend axis 修正

**現況**：`m.deform_axis = rng.choice(["X", "Y"])` — 完全隨機，跟 camera az 無關 → 50% 機率 bend 彎進畫面看不見。

**Stage 4 改動**：
1. 給定相機 az（rad），算出「理想 bend 軸」= 垂直視線方向
2. 允許 **±30° jitter**（你的決定：不要強制必定垂直，但不超過 30° 偏離）
3. 因為 Blender SIMPLE_DEFORM 的 `deform_axis` 只接受離散 X/Y/Z，要做任意角度的彎曲需要用 **Empty 物件當 modifier `origin`** + 旋轉 Empty 的 Z 軸來指定 bend 方向

實作雛形：
```python
# 給定 camera az (rad)
ideal_bend_dir = az + math.pi/2     # 弧垂直視線
jitter = math.radians(rng.uniform(-30, 30))
actual_bend_dir = ideal_bend_dir + jitter

# 建立或重用 Empty，旋轉到 actual_bend_dir
bend_empty = bpy.data.objects.get("BendOrigin") or bpy.data.objects.new("BendOrigin", None)
bend_empty.rotation_euler = (0, 0, actual_bend_dir)
bend_empty.hide_render = True

m = obj.modifiers.new("DefectBend", "SIMPLE_DEFORM")
m.deform_method = "BEND"
m.origin = bend_empty            # 用 Empty 的座標系
m.deform_axis = "Z"              # 沿 Empty 的 Z 軸彎
m.angle = math.radians(angle_deg)
```

### 2.4 Remesh 強度重定

**現況**：
- `remesh_light`：octree_depth ∈ `REMESH_LIGHT_OCTREE_RANGE`（要查確切值，推測較高 depth ≈ 細）
- `remesh_heavy`：octree_depth ∈ `REMESH_HEAVY_OCTREE_RANGE`（推測較低）

**Stage 4 改動**：
- 新 `remesh_light` = 舊 `remesh_heavy` 強度（保留有用的等級）
- 新 `remesh_heavy` = 再降一級 octree_depth（更激進），實際數字看 preview 後決定（討論項 §3）

### 2.5 HDRI 補回 4 個 + Scene 一致性

- `render_pan_head.py` L35：`HDRIS = ["university_workshop_4k.exr", "crossfit_gym_4k.exr", "monochrome_studio_02_4k.exr", "pretoria_gardens_4k.exr"]`
- `composite.py`：每張 scene 開頭 sample 一個 hdri 名稱，後續 instance 只從 `parts_meta` 中該 hdri 的 subset 抽

### 2.6 解析度維持 256

Bend 是 macro silhouette 問題、非 pixel 細節問題。提升到 384 對 bend 沒幫助，浪費算力。**維持 256**。

### 2.7 不做的事項（已明確否決）

- ❌ Bend 角度拉到 60°（先改 axis 修正足夠）
- ❌ 零件 tilt 額外軸（self-roll 跟 az 對 pan_head 等價，無意義）
- ❌ HDRI 旋轉 augmentation
- ❌ Cycles 渲染（太慢）
- ❌ Localized defect mask（要動 Blender shader / vertex weight，留 future ideas）
- ❌ Image-level aux head（光照/角度）— scene-level label 對 segmentation 邊際效益低
- ❌ Severity head（per-instance label 不適合 pixel-level）

---

## 3. 🟡 討論中 / 待確認

> 這些項目 Harrison 想跟另一位助手討論後再定。

### 3.1 Head A/B/C/E 的 loss 細節

- α_B / α_C / α_E 起始值都 1.0 合理嗎？要不要 B 高、C 低（因為 B 已涵蓋 C 的資訊）？
- B/C/E 都該 gate 在 GT part 像素上？還是 head A 預測 part 像素上（後者讓 head A 錯誤會 propagate）？
- Dice 在 multi-class 怎麼算（per-class Dice 平均、generalized Dice、還是 macro Dice）？
- 是否加 **hierarchical consistency loss**（head B argmax collapse 到 type 應該 = head C argmax；不一致加 penalty）？實作不難，但 paper 引用稍弱。

### 3.2 Bend pose 是否也限縮 elevation

- Azimuth 修正已經是主要解法（§2.3）
- Elevation ±60° 仍會壓縮投影看不見 bend → 要不要在 bend instance 額外限縮 `|el| ≤ 30`？
- 保險起見可做，代價是 bend 樣本減少（從 6 個 el → 4 個 el = -33%）

### 3.3 新 remesh_heavy 的具體 octree_depth

- 要先 preview 幾個值看視覺差異再定
- 候選：octree_depth = {3, 2}（越小越粗糙；目前舊 heavy 推測是 4）
- Plan：先 render 3-4 張不同 depth 的 sample，挑「比舊 heavy 更明顯但還能看出零件」的那個

### 3.4 總部件數量規劃

- 原本 252 part = 6 el × 3 az × 2 hdri × 7 state
- Stage 4 拆解：
  - HDRI 2 → 4（×2 倍）
  - Azimuth 3 → ? （討論：3 維持？6 加密？因為 bend 已綁定 az，az 多樣性會直接影響 bend 多樣性）
  - Bend elevation 若限縮 → bend state 的 part 比其他少
- 預估範圍：**400 – 600 parts**
- 渲染時間：Stage 3 跑 252 張花約 5-7 分鐘，500 張預估 10-15 分鐘 OK

### 3.5 是否做小規模圖像級增強（defect ratio / focal / HDRI strength）

- 先前提的 brainstorm 被否決，但需確認**全砍**還是有選項保留
- Defect ratio 從 19.4% → 30% 是純改 composite 的常數，幾乎無代價
- Focal length、HDRI strength 加變化需要小改 render script

### 3.6 新資料集規模

- Stage 3 是 1000 scenes。Stage 4 維持？還是 1500？
- 若 head 增加但資料量不變，可能過擬合風險變高

### 3.7 報告角度的 framing

- Stage 4 故事線：「Stage 3 用拆 head F1/F2 診斷出 bend 失敗是 head 2 signal 問題 + bend axis bug → Stage 4 修 axis + 引入 multi-head 監督利用既有細粒度 label」
- 重點：診斷導向設計，不是盲改

---

## 4. 📝 已記錄但暫不採用的點子（往後優化）

### 4.1 Uncertainty Weighting (Kendall et al. 2018, CVPR)

> *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*

每個 task 學 trainable `log σᵢ²`，loss = `Σᵢ exp(-log_var_i) * L_i + 0.5 * log_var_i`。
- 自動 balance，不用手調 weight
- 3 行 code、每 head 多 1 個 scalar
- Stage 4 先用 weighted sum（α=1），跑通了之後可作為 Stage 5 升級項

### 4.2 Hierarchical / Consistency Loss

- B 的 7-class argmax collapse 到 4-class type，應該 = C 的 4-class argmax
- 不一致加 KL divergence penalty
- 是 §3.1 的延伸，先看 baseline 表現再決定

### 4.3 Localized defect mask（per-pixel 瑕疵位置）

- 目前 defect label = 整顆瑕疵零件 alpha（bend 的螺絲頭也被標 defect → noise）
- 改用 Blender Material Index pass 或 vertex weight 輸出真正彎曲區域
- 對 bend 影響可能比換 head 還大，但工作量大，留 Stage 5+

### 4.4 GradNorm / PCGrad

- 比 Uncertainty Weighting 複雜
- 報告講起來不漂亮
- 若 Uncertainty Weighting 也不夠才考慮

### 4.5 Domain Adversarial Training（光照不變性）

- 從 bottleneck 加 GRL + HDRI classifier，強迫 encoder 丟掉光照資訊
- DR 已經一定程度做到了，邊際效益不確定

---

## 5. 預計工作流程（待 §3 全部敲定後執行）

1. **修 `render_pan_head.py`**：HDRI 補 4 個、bend axis 改 Empty-based + ±30° jitter、remesh 強度重定
2. **重新渲 parts**（~500 張，10-15 min）
3. **修 `composite.py`**：同 scene HDRI 一致性、defect ratio（如改）、可選的 focal/HDRI strength augment
4. **重 composite 1000–1500 scenes**（~3-5 min）
5. **黑底 ablation 重生**（100 scenes）
6. **改 `train_stage3.py` → `train_stage4.py`**：4-head model、新 loss、early stop / ReduceLROnPlateau
7. **訓練**（30 epoch，~7-10 min）
8. **改 `eval_stage3.py` → `eval_stage4.py`**：4-head 對應的 8-10 張新圖
9. **寫 `stage4_results.md` + 更新 HTML 報告**
10. **同步上傳 Drive**

---

## 6. 給組員的速覽

- **背景**：Stage 3 兩頭模型把 defect IoU 從 0.004 拉到 0.362，但 bend 仍是 0.01-0.02。診斷出兩個 root cause（bend axis bug + binary head 標籤利用率低）。
- **Stage 4 動作**：4-head 監督（part/bg + state + type + binary）+ 修 bend axis 對齊相機 + HDRI 補完整。
- **不動的**：模型架構主體（共用 encoder-decoder）、解析度（256）、訓練 pipeline 大致同 Stage 3。
- **可討論的**：上面 §3 七項。
