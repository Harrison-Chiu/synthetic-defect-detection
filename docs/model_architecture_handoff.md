# 模型架構與訓練流程交接文件（S1–S4）

> 用途：交給負責模型訓練的組員，快速掌握我們從 Stage 1 到 Stage 4 用過的網路架構、張量流、訓練參數，以及會影響架構/訓練判斷的關鍵機制。
> 來源：`src/models/segnet.py`、`src/schema.py`、`notebooks/train_stage1.ipynb`、各 stage 報告（`docs/history/`、`docs/stage4_results.md`）。
> 整理日：2026-05-30

---

## 0. 一分鐘總覽

- **S1 和 S2–S4 不是同一種網路。** S1 是「分類」CNN（判斷整張圖是哪種零件，4 選 1）；S2 起改成「語意分割」encoder-decoder（每個像素分類，輸出 mask）。
- **S2 → S4 共用同一個分割骨幹**（自製 U-Net 式，約 3.31M 參數），骨幹一個 box 都沒改；各 stage 只動 **輸出頭 + loss + optimizer + 資料**。
- **S4 有兩個版本**：① best（採用）＝沿用 S3 雙頭架構、只換新資料；② multi-head（三頭，否決）。S4 的主結論是「**資料修正有效、加多頭無效**」，所以最終交出的 S4 模型架構 ≈ S3 架構。
- 全程**自製、無 pretrained、無 attention、無 dropout**（呼應老師偏好：模型自己設計、不抄開源架構）。

---

## 1. 兩種架構家族

| | S1 | S2 – S4 |
|---|---|---|
| 任務 | 影像分類（whole-image classification） | 語意分割（semantic segmentation） |
| 輸出 | 1 個類別標籤（4 類零件） | 每像素一個標籤（mask） |
| 輸入解析度 | 64×64 | 256×256 |
| 結構 | 3 Conv + 2 FC（含 1 skip） | 4 層 encoder-decoder + skip-concat |
| 參數量 | ≈ 73.7K | ≈ 3.31M（骨幹） |
| 正規化 | 無 BN | 每 conv 後接 BatchNorm |
| 結果 | test acc 100%（問題 trivial） | defect IoU 逐 stage 提升（見 §5） |

> S1 的 100% 是因為背景純黑、pose 固定、四類視覺差異大 → 問題太簡單，不是有意義的鑑別指標。它的價值是「驗證資料→訓練 pipeline 整條通」，之後才轉向真正的任務：**像素級缺陷分割**。

---

## 2. S1 架構與張量流 — PartsCNN（分類）

```
Input 3×64×64
  └─ Conv1 3×3 (3→32)  → ReLU → MaxPool2  ─→ 32×32×32
  └─ Conv2 3×3 (32→16) → ReLU → MaxPool2  ─→ 16×16×16  ┐(identity)
  └─ Conv3 3×3 (16→16) → ReLU → (＋identity) → MaxPool2 ─→ 16×8×8
  └─ Flatten (1024)
  └─ FC1 1024→64 → ReLU
  └─ FC2 64→4
  └─ Softmax → 4 類 (socket_head / pan_head / hex_nut / flange_nut)
```

- 激活：ReLU（**無 BatchNorm**）；下採樣：MaxPool 2×2。
- skip：第 3 個 conv 用「相加（＋identity）」式 residual（注意：跟 S2–S4 的 concat 式 skip 不同）。
- 參數量 ≈ **73,700**。

---

## 3. S2–S4 共用骨幹與張量流 — DefectSegNet（分割）

> `ConvBlock` = **[ Conv 3×3 → BatchNorm → ReLU ] ×2**。
> skip 用 **concat**（不是相加），所以 decoder 每層輸入通道 = 「上採樣輸出 + 同層 encoder」。
> 上採樣用**學習式 ConvTranspose 2×2 stride2**（不是 bilinear）。

```
Input 3×256×256
  └─ enc1 ConvBlock(3→32)           ─→  32×256×256  ────────────────skip─┐
  └─ MaxPool2                       ─→  32×128×128                       │
  └─ enc2 ConvBlock(32→64)          ─→  64×128×128  ────────────skip─┐   │
  └─ MaxPool2                       ─→  64×64×64                     │   │
  └─ enc3 ConvBlock(64→128)         ─→  128×64×64   ────────skip─┐   │   │
  └─ MaxPool2                       ─→  128×32×32                │   │   │
  └─ enc4 ConvBlock(128→256)        ─→  256×32×32   ────skip─┐   │   │   │
  └─ MaxPool2                       ─→  256×16×16            │   │   │   │
  └─ bottleneck ConvBlock(256→256)  ─→  256×16×16            │   │   │   │
  └─ up4 ConvTranspose(256→128,s2)  ─→  128×32×32            │   │   │   │
  └─ concat ◄── enc4                ─→  384×32×32   ◄────────┘   │   │   │
  └─ dec4 ConvBlock(384→128)        ─→  128×32×32                │   │   │
  └─ up3 ConvTranspose(128→64,s2)   ─→  64×64×64                 │   │   │
  └─ concat ◄── enc3                ─→  192×64×64   ◄────────────┘   │   │
  └─ dec3 ConvBlock(192→64)         ─→  64×64×64                     │   │
  └─ up2 ConvTranspose(64→32,s2)    ─→  32×128×128                   │   │
  └─ concat ◄── enc2                ─→  96×128×128  ◄────────────────┘   │
  └─ dec2 ConvBlock(96→32)          ─→  32×128×128                       │
  └─ up1 ConvTranspose(32→16,s2)    ─→  16×256×256                       │
  └─ concat ◄── enc1                ─→  48×256×256  ◄────────────────────┘
  └─ dec1 ConvBlock(48→16)          ─→  16×256×256  ← 共用特徵圖，接下面的頭
```

### 3.1 逐層參數量（base_c=32）

| 階段 | 模組 | 運算 | 輸出 C×H×W | 參數 |
|---|---|---|---|---|
| Enc | enc1 | ConvBlock 3→32 | 32×256×256 | 10,208 |
| | enc2 | ConvBlock 32→64 | 64×128×128 | 55,552 |
| | enc3 | ConvBlock 64→128 | 128×64×64 | 221,696 |
| | enc4 | ConvBlock 128→256 | 256×32×32 | 885,760 |
| Bottleneck | bottleneck | ConvBlock 256→256 | 256×16×16 | 1,180,672 |
| Dec | up4 | ConvTranspose 256→128 | 128×32×32 | 131,200 |
| | dec4 | ConvBlock 384→128 | 128×32×32 | 590,336 |
| | up3 | ConvTranspose 128→64 | 64×64×64 | 32,832 |
| | dec3 | ConvBlock 192→64 | 64×64×64 | 147,712 |
| | up2 | ConvTranspose 64→32 | 32×128×128 | 8,224 |
| | dec2 | ConvBlock 96→32 | 32×128×128 | 36,992 |
| | up1 | ConvTranspose 32→16 | 16×256×256 | 2,064 |
| | dec1 | ConvBlock 48→16 | 16×256×256 | 9,280 |
| **骨幹合計** | | | | **≈ 3,312,528（3.31M）** |

頭部極小（1×1 Conv，每多一個輸出類別只 +17 參數）：1 類=17、2 類=34、3 類=51、4 類=68、7 類=119。

---

## 4. 頭部變體（骨幹後面換不同尾巴＝ S2/S3/S4 唯一差異）

**S2 baseline — 單頭**
```
共用特徵 16×256×256
  └─ Head Conv1×1 (16→3) → Softmax ─→ 3×256×256   (bg / normal / defect)
```

**S3 — 雙頭**
```
共用特徵 16×256×256
  ├─ Head1 Conv1×1 (16→2) → Softmax ─→ 2×256×256  (bg / part)        Loss: CE
  └─ Head2 Conv1×1 (16→1) → Sigmoid ─→ 1×256×256  (defect score)     Loss: Dice+BCE
```

**S4 ① best — 與 S3 完全相同的雙頭**（架構未改，只換新資料）
```
共用特徵 16×256×256
  ├─ Head1 Conv1×1 (16→2) → Softmax ─→ 2×256×256  (bg / part)
  └─ Head2 Conv1×1 (16→1) → Sigmoid ─→ 1×256×256  (defect score)
```

**S4 ② multi-head — 三頭（被否決，放 ablation）**
```
共用特徵 16×256×256
  ├─ Head A Conv1×1 (16→2) ─→ 2×256×256  (part:  bg / part)
  ├─ Head B Conv1×1 (16→7) ─→ 7×256×256  (state: normal + bend/displace/remesh ×{light,heavy})
  └─ Head C Conv1×1 (16→4) ─→ 4×256×256  (type:  normal / bend / displace / remesh)
       Loss: 每頭 CE+Dice 相加（α 都 1.0，GT gate）
```

> 畫圖建議：骨幹那張畫一次，S2/S3/S4 各自只把最右邊的「頭」那一小段換掉疊上去，觀眾一眼看懂「我們只動了輸出頭」。

---

## 5. 各 Stage 訓練參數與結果總表

| | S1 | S2 baseline | S3 | S4 ① best | S4 ② multi-head |
|---|---|---|---|---|---|
| 任務 | 4 類分類 | 3-class 分割 | 雙頭分割 | 雙頭分割 | 三頭分割 |
| 輸入 | 64×64 | 256×256 | 256×256 | 256×256 | 256×256 |
| 資料總量 | 576 張零件圖 | 1000 場景 | 1000 場景 | 1000 場景 | 1000 場景 |
| 切分 train/val/test | 345 / 115 / 116（60/20/20, seed 30）| 800 / 100 / 100（seed 42）| 800 / 100 / 100 | 800 / 100 / 100 | 800 / 100 / 100 |
| 頭部 | FC→4 | 16→3 | 16→2 + 16→1 | 同 S3 | 16→2 / 16→7 / 16→4 |
| Loss | CE | weighted CE | CE + (Dice+BCE) | 同 S3 | 各頭 CE+Dice 相加 |
| Optimizer | SGD 0.01, mom 0.9 | SGD 0.01, mom 0.9, wd 1e-4 | Adam 1e-3, wd 1e-4 | Adam 1e-3 + ReduceLROnPlateau | 同 ① |
| Batch / Epoch | – / 50 | 8 / 30 | 8 / 30 | 8 / 30 | 8 / 30 |
| 參數量 | 73.7K | 3.31M | 3.31M | 3.31M | 3.31M |
| **Test mIoU** | –（acc 100%） | 0.597 | 0.712 | **0.723** | 0.708 |
| **Test defect IoU** | – | **0.004** | 0.362 | **0.385** ✅ | 0.342 ❌ |

資料來源：S2 [stage2_baseline_results.md](history/stage2_baseline_results.md)、S3 [stage3_results.md](history/stage3_results.md)、S4 [stage4_results.md](stage4_results.md)。

### 資料量補充說明

- **S1（576 張）**：4 類零件 × 144 視角（12 elevation × 3 azimuth × 4 HDRI）的單一零件渲染圖，透明背景。是「每張圖一顆零件、標一個類別」的分類資料。
- **S2–S4（1000 場景）**：每張是 256×256 的**合成場景**——把預渲的零件 patch 貼到隨機背景上、一張可含多顆零件，附 per-pixel 的 semantic/instance mask。三個 stage 都是 1000 場景、800/100/100 切分、batch size 8、30 epoch。
- **場景是合成的（composite），不是逐張重渲**：底層 patch 才是 Blender 真正渲的。S3 渲了 252 張 pan_head patch（36 pose × 7 state）、S4 渲了 504 張（含 bend/HDRI 修正）；composite 階段把 patch 取樣組成 1000 場景。所以「換資料」換的是底層 patch + composite 規則，場景張數不變。
- **缺陷比例（S2 實測）**：per-instance `DEFECT_PROB=0.2` → 約 19.4% 的實例有缺陷；換算到像素只佔約 **5.3%**（bg 71.4% / normal 23.3% / defect 5.3%）。這個極端 class imbalance 是 S2 單頭 weighted-CE 學不起來（defect IoU 0.004）、S3 改用 Dice+BCE 的直接動機。

### Per-defect-state（S4 best，binary defect recall）

| State | S3 baseline | S4 best | 解讀 |
|---|---:|---:|---|
| bend_light | 0.034 | 0.062 | 物理天花板，幾乎學不到 |
| bend_heavy | 0.047 | 0.079 | 同上 |
| displace_light | 0.780 | 0.235 ❌ | S4 意外退步（見 §7） |
| displace_heavy | 0.932 | 0.763 ❌ | 同上 |
| remesh_light | 0.324 | 0.794 ✅ | S4 最大進步 |
| remesh_heavy | 0.523 | 0.859 ✅ | 同上 |

---

## 6. 「換資料」是什麼意思（S3 → S4）

S4 best 的設計是控制變因實驗：**拿 S3 那個一模一樣的網路，只餵修正過的新資料重訓**，用來隔離「純資料修正的貢獻」（結果 +0.023 defect IoU）。S3→S4 換的四項資料修正：

1. **Bend 軸修正** — 原本 bend 方向沒對齊相機，半數樣本是「彎進畫面」廢樣本；修正後 100% 有效，effective 訓練量翻倍。
2. **HDRI 2 → 4 個** — 原本寫死 2 個是 bug，實際 assets 有 4 個。
3. **同場景 HDRI 統一** — 原本同一張圖裡不同零件來自不同 HDRI，物理不一致。
4. **Remesh octree 加粗** — LIGHT (7,8)→(5,5)、HEAVY (5,6)→(4,4)，缺陷視覺更明顯。

---

## 7. 給模型組員的關鍵資訊（事實）與待驗證推測

> ⚠️ **本章的因果解釋大多尚未驗證，請當低可信度看待。** 下面刻意把「**事實**」（程式碼裡寫的、實測量到的數字）與「**推測**」（對原因的猜想，未做對照實驗證實）分開。**以事實為準；推測僅供排查方向參考，不要當結論寫進報告。**

### 7.1 事實（可直接引用）

1. **S4 的最佳模型是「雙頭」不是「三頭」。** 三頭 multi-head 經比較後 defect IoU 較低（0.342 < 雙頭 0.385），被歸入 ablation。
2. **三頭訓練的 loss 分布實測**：主任務 `L_A` 約佔總 loss **3%**（epoch 30 log），其餘由 aux 頭 B/C 的 CE+Dice 主導。這是「觀測到的數字」，不是它導致退步的證明。
3. **Schema 結構事實**：三頭裡 B(state)/C(type) 的標籤是 **per-instance**（meta 裡每個實例一個 defect_state），訓練時被廣播成 per-pixel 再套 **pixel-level loss（CE+Dice）**。`src/schema.py` 的 `check_schema()` 會對此 **warn 但放行**。
4. **Per-state 實測**（S4 best，binary recall）：bend ≈ 0.06–0.08、displace 從 S3 的 0.78/0.93 掉到 0.235/0.763、remesh 從 0.32/0.52 升到 0.79/0.86。
5. **displace 退步時，渲染參數沒動**，唯一的資料改動是 HDRI pool 2→4（含新增的柔光 HDRI）。
6. **指標單位陷阱**：S4 報告那張 per-state 是 **recall**；重構後 parity run 的 per-state 是 **IoU**，**不能逐位對齊**比較。
7. **訓練不穩定（實測現象）**：S2/S3/S4 都在 epoch 30 附近出現過 defect IoU 突然下掉一次。best_state 有存、test 用 best。ReduceLROnPlateau 當時只掛在 multi-head 版本，single/雙頭仍固定 lr。
8. **可復現性設計**：資料採樣用 pure deterministic（grid / `f(i,config)`）而非 seeded random。

### 7.2 推測（未驗證，僅供排查方向；不重要、隨時可能被推翻）

> 這些是「為什麼會這樣」的猜想，**目前沒有對照實驗背書**。Harrison 標為低可信度。

- **（猜）多頭退步可能與 loss 失衡有關**：`L_A` 只佔 3%，懷疑 encoder 容量被 aux 頭吃掉。— 待驗證：要做 loss reweight（α_B=α_C=0.1）或 uncertainty weighting 的對照才算數。
- **（猜）schema mismatch 可能是多頭失敗主因之一**：per-instance 標籤套 pixel loss 的監督訊號不乾淨。— 待驗證：改成 instance-level loss 重跑對照。
- **（猜）bend 學不到可能是 256px 的物理天花板**：螺絲身約 50px、bend 只改邊緣 1–3px。— 待驗證：拉 focal / 限縮 elevation / 加大角度的單變因實驗。
- **（猜）displace 退步可能是柔光 HDRI 把反射訊號打平**（sim2real 的縮影）。— 待驗證：固定其他條件，只比不同 HDRI 下 displace 的視覺強度。
- **（猜）remesh 變強可能因 octree 調粗後塊狀邊緣訊號更 dense、好學。** — 待驗證：octree 強度的單變因掃描。

---

## 8. 一句話交接

> S1 是分類熱身（73.7K，64×64，純 ReLU）；S2 起換成 3.31M 的 U-Net 式分割骨幹，**S2→S4 骨幹一字不改，只動最右邊的輸出頭和 loss**。S4 證明「資料修正有效、加多頭無效」，最終 S4 模型架構＝S3 架構（雙頭）。畫架構圖時骨幹畫一次，頭部畫變體疊上去即可。
</content>
</invoke>
