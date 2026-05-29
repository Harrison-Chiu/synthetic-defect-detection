# 設計決策紀錄

> 記錄討論過程中的選擇理由，避免之後重複討論或忘記排除原因。

---

## 資料集產生方式

**決定：Blender 手動操作 + Python Script 批量渲染**

- 不使用 BlenderProc 等自動化框架，避免學習成本分散、流程不透明
- Harrison 有 Blender 使用經驗，手動設定場景掌控感較高
- Python Script 是 Blender 內建功能（Script Editor），不算外掛工具
- Geometry Nodes 不適合此用途：GN 操控幾何形狀，無法控制「換角度→渲染→存圖→重複」的迴圈

---

## 零件選擇

**決定：M6 × 20mm，4 種（兩種螺絲頭型 + 兩種螺母輪廓）**

- 統一 M6 尺寸：Blender 只需一套材質參數，排除尺寸作為變數
- 選法蘭螺母（而非墊片）：墊片正面是圓形薄片，側面幾乎無特徵，傾角稍變資訊量驟減；法蘭螺母與六角螺母同屬螺母類但輪廓明顯不同，是有意義的挑戰
- 固定長度 20mm：10mm 軸心太短，側面角度幾乎看不出螺絲；20mm 螺紋清晰
- 來源：McMaster-Carr（型號見 history/plan_overview_s1-2.md），STEP 格式透過 FreeCAD 轉 OBJ

---

## 3D 模型格式

**決定：OBJ（Wavefront）**

| 格式 | 排除原因 |
|------|----------|
| STL | 無 UV，匯入後需手動展 UV 才能貼材質 |
| FBX | 功能過多（動畫、骨架），靜態零件用不到 |
| USD | FreeCAD 的 USD 匯出支援尚不穩定 |
| OBJ | UV 資訊保留，FreeCAD → Blender 轉換最穩定 ✓ |

---

## HDRI 格式

**決定：4K EXR（非 HDR）**

- 不鏽鋼高反射率，HDRI 動態範圍直接影響金屬反射真實感
- EXR 為 32-bit float 無損壓縮，HDR 為 32-bit RGBE 有損，金屬材質下差異明顯
- 4K 解析度：1K/2K 在金屬反射上會有明顯模糊

---

## 渲染角度策略（Stage 1 採用版）

**決定：12 elev (含仰視) × 3 azim × 4 hdri = 576 張（從原 8×3×3×4 改）**

- 推理零件軸對稱性 → pose 跟 camera orbit 對形狀視覺冗餘
- 固定零件 pose=(0,0,0)，採樣自由度全給相機
- 仰視角度加入因為 flange_nut 凸緣從下方看才有鑑別性
- 從 1152 減為 576：每張資訊獨立、無重複，pure grid 完全可復現

---

## 解析度

**決定：256 × 256**

- 128 太小：ResNet/EfficientNet 預設 input 224，128 需多加 resize；螺紋細節消失
- 256 是 MVP 合理下限，渲染速度與資訊量的平衡點
- 4060 8GB VRAM 跑得動

---

## 任務定義 — Stage 1（已封存）

**決定：分類（4-class classification）**

- 先驗證 pipeline 通，比直接做 detection 風險低
- 結果：test acc 100%，pipeline 確認可行但問題太簡單
- 詳細結果見 [stage1_mvp_report.md](history/stage1_mvp_report.md)

---

## 任務轉向 — Stage 2（當前）

**決定：從 Sorting 改為 Quality Control / Defect Detection（瑕疵 segmentation）**

理由：
- Stage 1 的 100% acc 沒有鑑別性，需要難度更高的任務
- QC 是真實工廠核心痛點，bonus 點：「合成資料解決瑕疵罕見」很學界
- 老師明確說「不要套用開源模型架構、不要 pretrained」 → segmentation 從零實作比 YOLO 從零實作可行得多
- 跟原 Sorting 比，QC 在 grading rubric 上一樣強或更強

## 單一零件選擇

**決定：pan_head（十字槽扁圓頭螺絲）**

- 街上撿到一根螺絲就長那樣，typical screw 形象，對外講最直覺
- 頭部十字槽提供獨特鑑別特徵
- 頭 + 軸 + 螺紋三種瑕疵載體都齊全

排除原因：
- socket_head：內六角從上看有戲但對外不直觀（特殊規格）
- hex_nut：幾何最簡單但缺乏「螺絲」的故事性
- flange_nut：特徵最豐富但太特殊

## 瑕疵生成策略

**決定：Blender Modifier 為主（Bend + Displace）+ 全 Blender 渲染**

- Bend / Displace 兩種 modifier 涵蓋「幾何彎曲」+「表面凹凸」兩大物理瑕疵類別
- 純參數驅動，自動化超容易
- 全 Blender 渲染（不用 Python 後處理瑕疵）：物理光照正確、敘事乾淨
- 排除其他 modifier 的理由見 [stage2_plan.md](history/stage2_plan.md) modifier 評估表

## 多零件場景合成

**決定：Python composite（不是 Blender 多物件渲染）**

- Blender 渲單張零件、Python 端拼場景，**繼承 Stage 1 的單張渲染格式**
- composite 是純 2D 影像操作（PIL/cv2/numpy），速度比 Blender 快兩個數量級
- ground truth (semantic mask + instance ID) 在 composite 過程「順便產生」，免標注
- Blender 端只負責 3D（modifier 必須 3D 端做），責任清楚

## Overlap 允不允

**決定：允許重疊**

- 真實工廠場景零件常重疊，強制不重疊脫離 reality
- ground truth 端有完整 instance ID（composite 端記得住）
- 模型端要不要解決重疊 instance 分離 → 給組員當研究題

## 模型架構

**決定：自製 Encoder-Decoder + skip connections，不用 pretrained，不抄 U-Net 名字**

- 老師明確要求「自製優先，pretrained 扣分」
- Encoder-Decoder 是通用設計模式，不算抄 U-Net；自己決定 channel/depth/activation 才是 tailored 設計
- 從零訓練在 segmentation 比在 detection 容易得多 → 跟「no pretrained」約束更相容

---

## 工具選用（Claude 端）

**資料夾建立與 render.py 骨架：給 Claude Code 處理**

- CC 可直接在專案目錄操作，比 Cowork 精確
- Cowork 適合重複性桌面操作，不適合這個任務

**Blender MCP（可選）**：
- 安裝 blender-mcp addon 後，Claude Desktop 可直接操控 Blender 場景
- 適合設定場景、調材質、除錯用，不作為批量渲染的核心依賴

---

## Stage 4 — Multi-head 廢、single-head 留

**決定：採 Stage 3 single-head 架構 + Stage 4 新資料當最終 best；multi-head 進 ablation**

- Multi-head V4 (A part/bg 2 + B state 7 + C type 4) 跑出 defect IoU 0.342，比 single-head 0.385 差
- Loss 分解：L_A 只佔總 loss 3%，aux head (B+C) 主導梯度 → 主任務「part vs bg + defect vs normal」被邊緣化
- 沒做 loss balancing 直接 sum α=1.0 是錯的；Stage 4 plan 把這當 baseline 結果壞掉
- **Stage 5 救活方向**：(a) α_B = α_C = 0.1~0.3、(b) Uncertainty Weighting (Kendall 2018)、(c) 砍 binary head 純用 7-way B argmax collapse — (c) 最乾淨

**決定：B head 用 7-way 而非 4-way**

- 跟 Harrison 立場一致：l/h 是獨立視覺類別，不把 severity 當跨類軸
- 這 Stage 4 沒救活 multi-head，但 7-way 設計本身在 Stage 5 救活方案 (c) 可重用

**決定：不放 severity head**

- per-instance label 強行 broadcast 到 per-pixel 引入 noise
- 7-way B 已隱含 severity 區分（bend_l vs bend_h 是不同 class）

## Stage 4 — Bend 方向加隨機 ±

**決定：Bend angle 加 `rng.choice([-1, +1])` 50/50 翻轉**

- 不翻轉時所有 bend 樣本往同一側 → 模型可能學 shortcut「bend = 往某方向歪」而非「bend = silhouette 變化」
- 在 part render 階段決定，比 composite 階段翻轉乾淨（後者會連帶翻 HDRI 反射 / bg）

## Stage 4 — 結案不進 Phase 2

**決定：bend IoU 落 0.05-0.15 grey zone（plan 規定應加碼），但 Stage 4 結案**

- 實際 bend_l=0.062, bend_h=0.079
- 結案理由：(1) 報告期限剩 5 天 (2) 現有 defect IoU 0.385 已超 Stage 3 故事完整 (3) Displace 退步是更值得追的問題
- Phase 2 候選（elevation 限縮 / focal 拉長 / 60° angle）全部記到 future_ideas.md Stage 5 候選
