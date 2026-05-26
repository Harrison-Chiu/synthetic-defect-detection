# 操作說明

> 給 Harrison 自己 / 之後重看的執行流程。完成版 README 之後再寫。

---

## 步驟 1：渲染 576 張資料集

### 前置確認
- Blender 場景已開啟（`blender/114-2_DLcourse_FinalProject-01.blend`）
- 4 個零件物件存在（`socket_head`, `pan_head`, `hex_nut`, `flange_nut`）
- 相機 `RenderCam` 存在
- World shader 的 `HDRI_node` 已接好

### 執行
**方式 A — 從 Blender Script Editor 跑（互動，看得到進度）**：
1. 開啟 Blender
2. 切到 Scripting workspace
3. Open → 選 `scripts/render.py`
4. 按 ▶ Run Script
5. 進度看 Window → Toggle System Console

**方式 B — 背景跑（無 GUI，可關 Blender 視窗）**：
```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" `
  --background "blender\114-2_DLcourse_FinalProject-01.blend" `
  --python "scripts\render.py"
```
（Blender 安裝路徑可能不同，自己改）

### 預期結果
- 約 10 分鐘
- 576 張 PNG 在 `output/dataset/raw/{class_name}/`
- `labels/metadata.csv` 產生

---

## 步驟 2：建訓練環境（一次性）

### 用 miniconda（你機器有裝）
```powershell
conda create -n dl_final python=3.11 -y
conda activate dl_final
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install jupyter matplotlib scikit-learn pandas pillow ipykernel
python -m ipykernel install --user --name dl_final --display-name "Python (dl_final)"
```

### 驗證 CUDA 通
```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```
應該印出 `cuda: True | NVIDIA GeForce RTX 4060`

---

## 步驟 3：跑訓練

```powershell
conda activate dl_final
cd "D:\Harrison\中山\大四下\深度學習期末報告"
jupyter notebook notebooks\train.ipynb
```

或用 VS Code 開 `.ipynb` 直接跑，Kernel 選 `Python (dl_final)`。

### 預期
- 50 epochs，4060 上 ~2–5 分鐘
- 印出 train loss、val acc 曲線
- 印出 test set 準確率 + confusion matrix + 樣本預測圖
- Baseline 目標：test acc > 90%（合成資料應該很容易達標）

---

## 出問題時的檢查順序

| 症狀 | 檢查 |
|------|------|
| 渲染全空白 | 相機 `clip_start` 是否被改回 0.1？應為 0.001 |
| 載入資料找不到 | `DATA_ROOT` 路徑跟實際 raw/ 位置對嗎 |
| CUDA out of memory | `BATCH_SIZE` 調小（32 → 16） |
| 準確率永遠 25% | 看 sample 圖是不是全黑、或 label 對錯 |

---

## 檔案位置

- 渲染腳本：[scripts/render.py](../scripts/render.py)
- 訓練 notebook：[notebooks/train.ipynb](../notebooks/train.ipynb)
- 渲染輸出：`output/dataset/raw/{class_name}/`
- 標注：`labels/metadata.csv`
