# 未來工作 / 靈感池

> 討論過程中提到、但 MVP 階段不做的想法。之後要加變體或升級時，從這裡挑。

---

## Stage 5 候選（Stage 4 結案時記錄，2026-05-28）

按優先順序，每項標「為何 Stage 4 沒做」+「期望效果」。

### 🔴 高優先 — Stage 4 留下的未解問題

#### 1. Displace 退步診斷與修復
- **背景**：Stage 3 displace IoU 0.78/0.93 → Stage 4 0.235/0.763。渲染參數**完全沒改**，唯一變因是 HDRI pool 從 2 → 4
- **假設**：新加的 `monochrome_studio_02`（純白光）+ `pretoria_gardens`（室外柔光）讓 displace 的 normal-perturbation 反射訊號變平。強光下 displace 凹凸面有明顯陰影，柔光下訊號散
- **驗證方法（單變因實驗）**：渲 4 顆 displace_heavy × 4 HDRI 對照，計算「每 HDRI 下的 displace pixel intensity std」量化視覺強度差
- **可能 fix**：
  - 移除 2 個柔光 HDRI（退回到 strong directional lighting only）
  - 或加大 displace strength range（補償柔光下的訊號損失）
  - 或 normalize 渲染後的 displace 強度（先測量再調 per-HDRI 參數）
- **為何 Stage 4 沒做**：訓練完才發現退步，時間壓力下選擇結案

#### 2. Bend IoU 從 0.08 → 0.15+（單變因實驗）
- **現況**：Stage 4 best bend_l=0.062, bend_h=0.079（落在 plan decision point 0.05-0.15 區間「應加碼」）
- **三個候選改動（單變因跑，看哪個影響最大）**：
  - **(a) Elevation 限縮**：bend instance 的相機 elevation 限 `|el| ≤ 30°`。原因：俯視/仰視 60° 把 bend 弧 cos(60°)=0.5 壓縮。代價 bend 樣本減少 33%
  - **(b) Focal length 拉長**：85mm → 130mm（×1.5）。每樣本 bend 弧多佔幾個像素。代價場景空間感變少，可能要少放 distractor
  - **(c) Bend angle 拉到 60°**：物理上接近斷裂但 visually distinguishable。從 `BEND_HEAVY_RANGE=(25,45)` 改成 `(50,70)` 試試
- **建議順序**：先 (a)（最便宜），不夠再 (c)，再不夠才 (b)
- **為何 Stage 4 沒做**：plan 上明確列「不做 — 等修 axis 後看數字再決定」，結果落在 grey zone 但時間壓力下沒做

#### 3. Multi-head 救活（α 重新加權 或 Uncertainty Weighting）
- **Stage 4 failure mode 量化**：L_A=0.063 (3%)，L_B+L_C ≈ 2.17 (97%)。Encoder 梯度被 aux task 主導，主任務學不好
- **三個救活方向**：
  - **(a) α_B = α_C = 0.1 ~ 0.3** — 暴力但簡單，可能 0.1 就夠
  - **(b) Uncertainty Weighting (Kendall 2018)** — 每 task 學 trainable log σ²，自動 balance。3 行 code 每 head 多 1 scalar
  - **(c) 砍 binary head，純用 7-way B 推論時 collapse**：完全沒 aux conflict
- **建議**：(c) 最乾淨 — 跟 Harrison 立場一致（B 7-way 已含全部資訊）。(b) 是學術賣點
- **為何 Stage 4 沒做**：架構失敗後選擇放棄 multi-head，沒進階修復

### 🟡 中優先 — 模型穩定性 / 訓練品質

#### 4. ReduceLROnPlateau 合併到 single-head 訓練
- **Stage 3 / Stage 4 single-head 都有 epoch 30 val mIoU 突崩問題**（從 0.73 → 0.46）
- **Stage 4 multi-head 加了 `ReduceLROnPlateau(mode="max", factor=0.5, patience=3)` 收斂穩定，沒崩**
- **Fix**：把同樣 scheduler 加進 `train_stage3.py`。3 行改動
- **為何 Stage 4 沒做**：ablation 跑時懶得改，直接複用原檔。Test 用 best_state 所以結果不受影響，但訓練曲線難看

#### 5. ResolutionLossOnPlateau 早停 / best-only test
- 跟 #4 連帶。早停可以省訓練時間（best 通常在 epoch 25-28，跑完 30 是多餘）
- **為何 Stage 4 沒做**：30 epoch 才 5 分鐘，省這個時間意義不大

### 🟢 低優先 — 報告 / 視覺化

#### 6. Per-state binary recall vs 7-way IoU metric 釐清
- **Stage 4 踩到**：`eval_stage4.py` 的「per-state IoU」是 head B 7-way argmax 嚴格 IoU，跟 Stage 3 報告的「per-state defect detection rate」（binary）不可比
- **Fix**：在報告/HTML 明確標 schema，或統一只用 binary detection rate
- **已做**：`eval_stage4_compare.py` 用 binary，數字才能跟 Stage 3 對齊

---

## Stage 4+ 已記錄但不在當前 stage 採用

### Loss / 訓練優化
- **Uncertainty Weighting (Kendall et al. 2018, CVPR)** — 每個 task 學 trainable log σ²，loss = Σᵢ exp(-log_var_i) * L_i + 0.5 * log_var_i。自動 balance 多任務 weight，3 行 code、每 head 多 1 個 scalar。Stage 4 先用 weighted sum，跑通後考慮升級
- **Hierarchical / Consistency Loss** — head B (state 7-way) argmax collapse 到 type 應該 = head C (type 4-way) argmax；不一致加 KL penalty。實作不難，是 multi-head 監督的延伸
- **GradNorm (Chen 2018)** — 動態調整 weight 平衡 gradient norm。比 Uncertainty Weighting 複雜
- **PCGrad (Yu 2020)** — project conflicting gradients。解 task conflict 不解 weight balance

### 模型架構
- **Localized defect mask** — 目前 defect label = 整顆瑕疵零件 alpha（bend 的螺絲頭也被標 defect → noise）。改用 Blender Material Index pass 或 vertex weight 輸出真正彎曲區域。對 bend 影響可能比換 head 還大，但工作量大
- **Domain Adversarial Training** — bottleneck 加 GRL + HDRI classifier，強迫 encoder 丟掉光照資訊。DR 已部分達成，邊際效益不確定
- **Image-level aux head（光照/角度）** — 從 bottleneck 接 global pool + FC 預測 HDRI class / camera az 等 scene-level label。Stage 4 暫不做（segmentation 主任務已足夠複雜）

### 資料端
- **Bend 也限縮 elevation** — Stage 4 已修 azimuth，elevation ±60° 仍會壓縮投影，可額外限縮 |el| ≤ 30
- **Cycles 渲染**、**HDRI 旋轉 aug**、**焦距 / HDRI strength jitter**、**真實 OBJ distractors** — 都是邊際改善，工作量大
- **拉 bend 強度到 60°+** — 物理上接近斷裂，但 visually distinguishable

---

## Stage 2 之後可擴充

### 任務面
- **Instance separation under overlap**：純 semantic seg 在重疊區會把多實例合成一團 connected component。解法：(1) center prediction（per-pixel 對 instance center 的 offset）、(2) embedding-based clustering、(3) watershed post-processing
- **Defect type 分類**：MVP 只區分 normal / defective，可進一步分 bend / displace / 其他
- **Localized defect mask**：MVP 的 defect mask = 整個瑕疵零件 alpha。進階：用 Material Index pass 或 AOV 輸出 per-pixel 瑕疵位置
- **零件種類擴充**：從 pan_head 擴回 4 種，多 class 分割 + 瑕疵組合任務
- **Sim-to-real 驗證**：實體拍 pan_head 真實照片做 test set，評估 domain gap

### Blender 端進階
- **AOV output 自動瑕疵 mask**：shader 端做 procedural rust，AOV 輸出 rust factor 當 mask（針對表面瑕疵的 per-pixel mask）
- **Material Index pass**：bend/displace 之外加 rust（獨立 material）+ Material Index pass 輸出每像素材質 → 自動 defect mask
- **Cycles 渲染**：取代 EEVEE，金屬反射真實感大幅提升，1000+ 張要評估時間
- **更多瑕疵幾何**：Boolean 缺角、Lattice 自由變形、Vertex weight 限制 displace 區域

### 資料端進階
- **小角度傾斜擾動**：composite 時對零件加 ±5° 旋轉模擬不完美姿態
- **HDRI 旋轉 augmentation**：同 HDRI 用多種旋轉產生 4 倍光照變體（需引入 seeded random）
- **景深 / 動態模糊 / 鏡頭雜訊**：模擬實際相機，增加 sim-to-real 真實感
- **背景紋理擴充**：MVP 用 10 張，可擴充到 50–100 張覆蓋更多工業場景

## 資料多樣性

- **Geometry Nodes 幾何變異**：模擬零件磨損、輕微形變、刮痕。注意 GN 不能直接控制渲染迴圈，要跟 Python script 配合
- **更多零件種類**：擴充到 8–10 類，例如墊片、各種螺絲頭型、不同尺寸（M4/M8）
- **混合尺寸/材質**：黑色氧化處理、黃銅、鋁件
- **場景背景**：工廠地板貼圖、輸送帶、零件盤
- **零件姿態 (pose) 變化**：MVP 為了獨立性拿掉了。之後可以加 1–2 種非典型姿態（如完全倒置）當 augmentation，但需要設計好不破壞獨立性
- **小角度傾斜擾動**：±10° 之類的隨機 tilt，增加真實感（需引入 seeded randomness）

## 渲染品質

- **Cycles 渲染引擎**：取代 EEVEE，金屬反射真實感大幅提升。但速度慢，需評估 4060 跑 1000+ 張的時間
- **更高解析度**：512×512 或 1024×1024，給更深的 backbone 用
- **更多 HDRI**：8–10 個環境，涵蓋更多光照條件
- **HDRI 旋轉 augmentation**：同一張 HDRI 用 4 個旋轉角，等於 4 倍光照變體（需引入 seeded randomness）
- **景深 / 動態模糊**：模擬實際相機拍攝特性

## 採樣方案延伸

- **加密到 1152 張**：MVP 是 576（12 elev × 3 azim × 4 hdri × 4 part）。加密做法：azim 4 個（0/90/180/270）或 elev 24 個
- **針對對稱性最佳化的不均勻取樣**：螺絲不需多 azimuth（軸對稱），螺母 6 次對稱可分零件設計
- **球面均勻取樣 (Fibonacci sphere)**：取代 elev × azim grid，視角分布更均勻
- **零件 pose 變化**：MVP 為了獨立性拿掉了。之後可以加 1–2 種非典型姿態當 augmentation

## Pipeline / 工具

- **BlenderProc 框架**：成熟的合成資料工具，下週若覺得手寫 script 不夠用可評估
- **Blender MCP addon**：debug 場景時 AI 直接操控 Blender，加速 iteration

---

## 加新想法時請註明來源
格式：`- **想法**：說明（提出時間/情境）`
