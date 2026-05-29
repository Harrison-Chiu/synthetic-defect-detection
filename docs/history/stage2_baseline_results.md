# Stage 2 Baseline — 訓練結果 + 失敗分析

> 訓練日：2026-05-26  
> Commit：`1245907` Stage 2: Baseline 訓練完成 + 評估視覺化  
> 對應計畫：[stage2_plan.md](stage2_plan.md)

---

## 訓練配置

| 項目 | 設定 |
|------|------|
| 模型 | 自製 4-stage encoder-decoder + skip connections，`base_c=32` |
| 參數量 | 3,312,579 (~3.3M) |
| 輸入 | 256×256 RGB 場景圖（normalize to [-1, 1]） |
| 輸出 | 3 class semantic seg (bg / normal_part / defective_part) |
| Dataset | 1000 scenes → 800 train / 100 val / 100 test，seed=42 |
| Defect instance 比例 | 19.4%（per-instance `DEFECT_PROB=0.2`） |
| Class freq (train pixels) | bg 71.4% / normal 23.3% / defect 5.3% |
| Class weight (auto) | bg 0.171 / normal 0.524 / defect 2.305 |
| Loss | weighted CrossEntropy |
| Optimizer | SGD lr=0.01, momentum=0.9, weight_decay=1e-4 |
| Batch size | 8 |
| Epochs | 30 |
| 硬體 | RTX 4060 8GB，torch 2.5.1+cu121 |
| Wall-clock | ~5 min（epoch 1 = 51.5s 含 cuda init/data cache，其餘 ~8.7s/epoch） |
| Best val mIoU | 0.595（epoch 29） |

---

## Test 結果（100 scenes）

```
pixel_acc = 94.00%
mIoU      = 0.597
  background      IoU = 0.992
  normal_part     IoU = 0.794
  defective_part  IoU = 0.004    ← baseline 主要失敗
```

**Instance-level（connected components 後處理）**：
- Total GT instances: 249, Predicted: 267
- Precision = 0.336, Recall = 0.821, F1 = 0.477

Recall 高是因為 CC 把所有非背景 blob 都當 instance、defect/normal 分類錯誤但「有偵測到零件」算對；Precision 低反映分類完全沒辦法區分 defect。

---

## 失敗模式：confusion matrix 直接定位

像素級 row-normalized confusion（row=GT, col=Pred）：

| GT \ Pred | background | normal_part | defective_part |
|-----------|-----------|-------------|----------------|
| **background**   | **99.4%** | 0.5%      | 0.1%           |
| **normal_part**  | 0.5%      | **99.0%** | 0.5%           |
| **defective_part** | 0.4%    | **99.2%** | 0.4%           |

**關鍵觀察**：
1. 「**物件 vs 背景**」分得幾乎完美（99% 等級）
2. **defect 整列 99.2% 被預測成 normal_part** — 模型壓根沒把它當另一類
3. 高 pixel accuracy（94%）是 bg + normal 主導的副產物，**不能反映任務真實表現**
4. defect 的 class weight 2.3× 不足以對抗 5% 像素 minority

訓練曲線（[training_curves.png](../figures/stage2/training_curves.png)）也顯示 val defect IoU 從來沒突破 0.20，30 epoch 完全沒有上升趨勢，模型穩定收斂到「不輸出 class 2」的 local minimum。

---

## 報告用圖檔

全在 [docs/figures/stage2/](../figures/stage2/)：

| 檔名 | 用途 |
|------|------|
| `training_curves.png` | Loss + val per-class IoU 沿 epoch |
| `per_class_iou_bar.png` | Test set per-class IoU 長條圖（一張看懂 baseline 失效） |
| `confusion_matrix.png` | 像素級 3×3 row-normalized confusion（最關鍵） |
| `prediction_grid.png` | 6 個代表性場景 RGB / GT / Pred / Diff 對比 |
| `failure_examples.png` | 最佳偵測 / 最大漏報 / 最大誤報 / 全 normal 四種代表 |
| `test_summary.json` | 所有數字機器可讀備份（pixel_acc, mIoU, confusion 等） |

重新產生：`& "C:\Users\Harrison\miniconda3\envs\dl_final\python.exe" scripts\eval_stage2.py`

---

## 下一步候選方向

### A. Two-head 架構（Harrison 主推、confusion matrix 背書）
共用 encoder，分兩個 head：
- Head 1: part/bg binary segmentation（**已知幾乎完美**，繼續沿用 CE）
- Head 2: defect score per pixel（focal/dice loss）或 per instance（接 CC + 小 classifier）

優點：解耦兩個不同難度的子任務、各自用適合的 loss、保留 baseline 強項。  
風險：分頭設計細節要想清楚，per-instance head 又會牽扯 overlap 切割。

### B. 同 single-head 換 loss + 優化器
保留現架構，只改：
- LR 0.01 SGD → 1e-3 Adam（val 曲線擺盪太大可能 LR 太高）
- weighted CE → CE + Dice（0.5/0.5 混合，Dice 對 minority class 友善）
- 可選 Focal loss γ=2 加強 hard examples

優點：改動最小、~5 min 就能跑出第二版比較。  
風險：可能還是治不了 defect 視覺特徵太弱的根本問題。

### C. 質疑資料端
- displace_light/heavy 在 [parts_preview.png](../figures/stage2/parts_preview.png) 視覺幾乎看不出來（render_gotchas Stage 2 已記錄 strength unit 已修正）
- 可能要回 Blender 加大 displace strength 重新渲，或乾脆暫時剔除 displace 兩個 state
- 對照組：純 bend defect 模型若 IoU 突然合理 → 確認 displace 是噪音

**建議順序**：A 主路線，B 當對照組同時跑（便宜），C 留給若 A/B 都 IoU < 0.3 時的退路。

---

## 訓練環境踩雷（這次踩到）

### nbconvert 沒走 notebook kernelspec
`jupyter nbconvert --execute` 不一定用 notebook metadata 裡的 kernel；曾跑成 Windows Store Python 3.11（CPU-only torch），導致 30 分鐘 timeout。

**正確做法**：
1. **首選**：在 VS Code / JupyterLab UI 開 notebook，右上選 kernel `Python (dl_final)`，跑就對了
2. 若要批次跑 .py：
   - `jupyter nbconvert --to script` 先轉
   - script 開頭加：
     ```python
     import matplotlib
     matplotlib.use('Agg')          # 防 plt.show() 阻塞
     import matplotlib.pyplot as plt
     plt.show = lambda *a, **k: None
     ```
   - 用 `& "C:\Users\Harrison\miniconda3\envs\dl_final\python.exe" -u <script>` 直接呼叫 conda Python，不要用 nbconvert

### 驗證 GPU 真的被用
腳本開頭 print：
```python
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu={torch.cuda.get_device_name(0)}')
```
看到 `cuda_available=True` + GPU 名稱才能往下跑。
