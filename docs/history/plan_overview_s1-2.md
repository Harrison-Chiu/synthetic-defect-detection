# 深度學習期末專題：工廠零件辨識系統

> 課程：MEME552 深度學習理論與應用，NSYSU  
> 分工：Harrison 負責資料端，另外兩位組員負責模型訓練  
> 最後更新：2026-05-20

---

## 專題定位

**主題類別**：Quality Control & Defect Detection（從原本 Sorting & Grasping 改向）  
**核心問題**：工廠合成資料 + 自製神經網路做螺絲瑕疵檢測  
**現在階段**：**Stage 2 — 瑕疵檢測**（Stage 1 MVP 已完成，見 [stage1_mvp_report.md](stage1_mvp_report.md)）

---

## 兩階段總覽

### Stage 1（已完成）— Pipeline 驗證
4 類零件 × pure grid 採樣 → 576 張 RGBA PNG → 手刻 CNN 分類 → test acc 100%  
→ 證明 Blender 合成 → 訓練 pipeline 整條通。100% 不具鑑別意義，所以升級到 Stage 2。

完整記錄：[stage1_mvp_report.md](stage1_mvp_report.md)、[weekly/week01.md](../weekly/week01.md)、[operations_stage1.md](operations_stage1.md)

### Stage 2（進行中）— Defect Detection
單一零件（pan_head）+ Modifier 製造瑕疵變體 + Python composite 多零件場景（含重疊） → 自製 Encoder-Decoder semantic segmentation。

**詳細技術設計**：[stage2_plan.md](stage2_plan.md)

---

## Stage 2 重點摘要

| 項目 | 內容 |
|------|------|
| 零件 | pan_head（M6×20 十字槽螺絲） |
| 瑕疵類型 | Bend（彎曲）、Displace（表面凹凸）各 2 強度 |
| 任務 | 3-class semantic segmentation（背景 / 正常零件 / 瑕疵零件） |
| 資料規模 | 180 base parts × 1000+ composite scenes |
| 模型 | 自製 Encoder-Decoder + skip connections（不用 pretrained） |
| 評估 | mean IoU + instance-level precision/recall |

---

## 設計原則（持續遵守）

1. **可復現性 > 採樣多樣性**：能用 pure grid 就不要 random + seed
2. **MVP discipline**：當前階段聚焦最小可行，新點子寫到 [future_ideas.md](../future_ideas.md)
3. **自製優先**：不用 pretrained model、不抄 SOTA 架構，自己設計 tailored to 我們的任務
4. **資料端解決資料問題、模型端解決模型問題**：例如 overlap instance 分離有兩條路，先選資料端能解的，難題給組員研究

---

## 文件結構

```
docs/
├── PLAN.md                ← 本文件，主計畫
├── stage2_plan.md         ← Stage 2 技術設計細節
├── stage1_mvp_report.md   ← Stage 1 完成報告（封存）
├── operations_stage1.md   ← Stage 1 操作指令（封存）
├── design_decisions.md    ← 為何選 X 不選 Y
├── future_ideas.md        ← 提到但目前不做的點子
├── render_gotchas.md      ← 渲染踩雷清單（持續累積）
├── reference/             ← 課程 PDF 等外部資料
└── weekly/                ← 每週進度
```

---

## 環境需求

- **Windows + Blender 5.x**：渲染 + modifier 自動化
- **Python 3.11 + conda env `dl_final`**：訓練 + composite
- **PyTorch + CUDA**：本地 4060 8GB

詳細安裝步驟：[operations_stage1.md](operations_stage1.md)（Stage 2 操作會新增 operations_stage2.md）

---

## 報告 / 繳交

- 報告日：2026/06/02 或 06/09，英文口頭 10–15 min + Q&A 3 min
- 繳交：單一 .zip 含 .ipynb + 投影片 + 資料來源
- 課程要求摘要：[reference/course_requirements.md](../reference/course_requirements.md)
