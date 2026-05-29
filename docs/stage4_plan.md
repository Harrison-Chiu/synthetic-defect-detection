# Stage 4 計畫 — Bug 修復 + 細粒度監督

> 重寫：2026-05-28
> 前情：[stage3_results.md](stage3_results.md)
> 立場：只做有獨立理由的改動。不預先承諾「修 bend 的方法」，先把已知 bug 修完看數字怎麼動，再決定後續。

---

## 1. 從 Stage 3 數字獨立讀出的事實

| 指標 | 值 | 判讀 |
|------|---:|------|
| Test mIoU | 0.712 | 主架構 / loss / 資料量都 OK |
| normal_part IoU | 0.790 | 模型認得零件 |
| bg IoU | 0.983 | 背景沒問題 |
| 黑底 ablation defect IoU | 0.319 (vs 0.362) | DR 成功，模型沒靠背景 shortcut |
| displace_h / displace_l | 0.428 / 0.391 | 已 saturate |
| remesh_h / remesh_l | 0.276 / 0.155 | 中等 |
| **bend_h / bend_l** | **0.024 / 0.006** | **失敗 — 預測為 defect 率僅 3-5%，遠低於 base rate ~50%** |

**核心觀察**：三種 defect 的 IoU 排序 = 各自影響到的像素數量排序（displace 整面 > remesh 邊緣 > bend 剪影）。模型沒挑食，它就是學它看得到的東西。

**Bend 的失敗模式**是「從不預測 defect」，不是「搞混了」。

---

## 2. 已知的兩個 bug（Harrison 在 Stage 3 後抓到）

### Bug 1：Bend axis 跟相機 az 無關

`render_pan_head.py` L143：
```python
m.deform_axis = rng.choice(["X", "Y"])
```

完全隨機選 X 或 Y。當 bend 軸恰好沿視線方向時，螺絲是「彎進畫面」，silhouette 從相機角度看不到差異。

**後果**：bend 訓練樣本約有 50% 視覺上等同 normal → 等於 **data quality 50% 折扣**。

### Bug 2：HDRI 寫死 2 個 + scene 內 HDRI 不一致

- `render_pan_head.py` L35 寫死 `["university_workshop_4k.exr", "crossfit_gym_4k.exr"]`，但 `assets/hdri/` 實際有 4 個（多 `monochrome_studio_02_4k.exr`、`pretoria_gardens_4k.exr`）
- `composite.py` 隨機抽 parts → 同 scene 不同 instance 來自不同 HDRI 渲染 → 反射物理上不一致

---

## 3. Stage 4 三件事（彼此獨立，不互相聲稱因果）

### 3.1 修 Bug 1 — Bend axis 對齊相機

**目標**：讓 bend 弧落在垂直視線方向，silhouette 變化最大化。

**實作**：因為 Blender SIMPLE_DEFORM 的 `deform_axis` 只接受 X/Y/Z，要做任意角度需用 Empty 物件當 modifier `origin`：

```python
ideal_bend_dir = az + math.pi/2          # 弧垂直視線
jitter = math.radians(rng.uniform(-30, 30))
actual_bend_dir = ideal_bend_dir + jitter

bend_empty = bpy.data.objects.get("BendOrigin") or \
             bpy.data.objects.new("BendOrigin", None)
bend_empty.rotation_euler = (0, 0, actual_bend_dir)
bend_empty.hide_render = True

m = obj.modifiers.new("DefectBend", "SIMPLE_DEFORM")
m.deform_method = "BEND"
m.origin = bend_empty
m.deform_axis = "Z"
m.angle = math.radians(angle_deg)
```

**±30° jitter** 是 Harrison 的決定：不強制必定垂直，但偏離有限。

**驗證步驟（先做）**：寫 `scripts/check_bend_axis.py`，render 4-6 顆 bend pan_head 用上述邏輯，PNG 輸出讓 Harrison 目視確認幾何對了，**再**正式改 `render_pan_head.py`。

### 3.2 修 Bug 2 — HDRI 補回 4 個 + Scene 一致性

- `render_pan_head.py` 的 `HDRIS` 改成 4 個檔名
- `composite.py`：每個 scene 開頭 sample 一個 hdri，後續 instance 只從 `parts_meta` 中該 hdri 的 subset 抽

### 3.3 架構升級 — A + B(7-way) + C(4-way) 三頭

**動機**：跟 bend 無因果關係。理由是 Harrison 的細粒度監督直覺 — 「給模型更多細節 label，讓它學更好的 representation」，這在 multi-task learning 裡是 auxiliary supervision 的標準做法。

| Head | Output | Label space | Loss | 角色 |
|------|--------|-------------|------|------|
| **A** part/bg | 2-way softmax | bg / part | CE | gate + 報告主指標 |
| **B** defect_state | **7-way softmax** | normal / bend_l / bend_h / disp_l / disp_h / remesh_l / remesh_h | CE + multi-class Dice | 細粒度監督，報告 confusion matrix |
| **C** defect_type | **4-way softmax** | none / bend / disp / remesh | CE + multi-class Dice | 中介層級監督，幫 encoder 學「類別」這層抽象 |

**設計決策**：
- **保留 C（4-way）作為 auxiliary head**：資訊上 C 是 B 的 collapse、推論時可從 B 推得，但獨立監督等於給 encoder 一個比 B 容易學的中介信號，**可能幫早期收斂與 representation 學習**。代價只是多一個 1×1 conv
- **砍 binary defect head / severity head**：binary 從 argmax(B) 推、severity 不適合 pixel-level 強行 broadcast
- **B 是 7-way 而非 4-way**：按 Harrison 邏輯，l/h 是獨立視覺類別（不是把 severity 當跨類軸），給模型最完整的 label
- **GT gate**：B 和 C 的 loss 只在 GT part 像素上算（不是 pred part，避免 head A 早期不穩 propagate）
- **共用 encoder-decoder**：只在最後分歧出 3 個 1×1 conv
- **不加 hierarchical consistency loss**（強制 collapse(B)==C）：先看 baseline，如果 B/C 預測常不一致再考慮

**Loss**：
```
L_total = L_A_CE + α_B (L_B_CE + L_B_Dice) + α_C (L_C_CE + L_C_Dice)
```
α_B = α_C = 1.0 起步。Dice 在 multi-class 用 macro per-class Dice 平均（每類算一次再平均，minority 也有 voice）。

**推論時**：
- Binary defect mask = `argmax(B) != normal`（主 KPI、跟 Stage 3 比較）
- Defect type 報告：可選 C 直接輸出 或 collapse(B)；兩者比對也是有趣的分析項

---

## 4. 不做的事

| 項目 | 理由 |
|------|------|
| Bend angle 拉到 60°+ | 假設「修 axis 不夠」才需要，先看數字 |
| Camera focal / 距離 | 同上 |
| Bend instance elevation 限縮 | 同上，且修 axis 後 elevation 60° 看下去 bend 弧 cos(60°)=0.5 仍可見 |
| 解析度 384 | 已駁回，bend 是 macro 問題不是 pixel 細節問題 |
| Severity / type 獨立 head | 7-way B 已含全部資訊 |
| Uncertainty Weighting / GradNorm / PCGrad | 報告講不漂亮，先 weighted sum α=1 跑通 |
| Localized defect mask | 要動 Blender shader，工程量大，留 Stage 5+ |
| Image-level aux head | scene-level label 對 segmentation 邊際效益低 |

---

## 5. Phase 化執行 — 跑完再決定

### Phase 1（必做 — 兩個 bug fix + multi-head）

1. 寫 `scripts/check_bend_axis.py` 渲 sanity 圖
2. Harrison 確認幾何 → 改 `render_pan_head.py`（bend axis + HDRI 4 個）
3. 改 `composite.py`（scene HDRI 一致性）
4. 重 render parts（數量同 Stage 3 量級，~250–500 張）
5. 重 composite 1000 scenes
6. 黑底 ablation 重生 100 scenes
7. 改 `train_stage3.py` → `train_stage4.py`，加 7-way B head + 4-way C head
8. 訓練 + eval

### Phase 1 完成後的 **Decision Point**

看 `bend_l` / `bend_h` 的 per-state IoU：

| Bend IoU 結果 | 行動 |
|---------------|------|
| **> 0.15** | Phase 2 結案，寫報告 |
| **0.05–0.15** | 加碼一項視覺強化（候選：angle 拉強 / elevation 限縮 / focal 拉長），單一變因再跑一次 |
| **< 0.05** | 重新評估 bend 是否該留在 defect set，或承認在 256px 下不可解、報告誠實標註 |

**重點**：這個 decision point 是「拿到數字才決定」，不是「先決定下一步」。

### Phase 2（可選 — Phase 1 結果不佳才觸發）

留空。等數字。

---

## 6. 預期結果（誠實預測）

- normal / bg / displace：跟 Stage 3 持平或略升（multi-head 邊際 regularization）
- remesh：可能小幅改善（7-way 細粒度監督幫助）
- **bend**：axis 修對後從 0.02 拉到 **大約 0.05–0.15** 區間
  - 理由：effective 訓練資料量翻倍（50% 廢樣本變有效），但每樣本的 silhouette 偏移在 256px 下還是只有幾 px
  - 如果落在 > 0.15，是上限好的情況
  - 如果 < 0.05，axis fix 不足以解，要進 Phase 2

整體 defect IoU：**0.38–0.45** 區間，不會炸性提升。

---

## 7. 給組員的速覽

獨立 HTML 版（仿 `docs/stage3_report.html` 格式），訓練 / eval 跑完後再生成。
Plan 階段不放速覽。
