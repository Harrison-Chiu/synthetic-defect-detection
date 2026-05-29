# Week 01 進度紀錄

> 期間：2026-05-17 → 2026-05-18  
> 目標：MVP 分類資料集，驗證 Blender → 訓練 pipeline 可行

---

## 進度紀錄

### 2026-05-17
- 下載 4 個 STEP、FreeCAD 轉 OBJ，匯入 Blender 並改短名
- 套金屬材質、Camera 改 RenderCam、刪 default Light
- 接 HDRI 到 World shader（4 個 4K EXR）
- 跟 Claude 討論並重設計採樣方案：
  - 從 8×3×3 pose-based 改為 pure grid（無 pose、無隨機）
  - 推理出零件軸對稱性、決定固定 pose 把自由度給相機 orbit
- 寫 render.py，跑兩批測試
- 踩雷 + 修正：相機 near clip 把零件裁掉、Blender 5.x 引擎名改回 `BLENDER_EEVEE`、`film_transparent` 設定

### 2026-05-18
- 採樣再優化：12 elev（含仰視）× 3 azim × 4 hdri = 576 張（每類 144）
- 跑完整 576 張渲染（比預期快，幾分鐘內）
- 建 conda 環境（python 3.11 + torch cu121 + jupyter）
- 寫 train.ipynb：手刻 CNN with skip connection（延續 HW9 風格）
- 跑訓練：**test acc 100%** → MVP pipeline 確認可行
- 寫 Stage 1 報告 [docs/stage1_mvp_report.md](../history/stage1_mvp_report.md)

---

## 問題與阻礙

- 相機 `clip_start` 把零件裁掉 — debug 過程詳見 [docs/render_gotchas.md](../render_gotchas.md)
- 100% acc 顯示 MVP 任務太簡單，下階段需增加難度

---

## 下週預計（Stage 2）

- 選定增加難度的方向（候選：背景雜訊 / 多零件同框 / detection 升級）
- 評估是否真的把 stage1 資料集打包給組員（取決於是否升級任務）
- 補 README
