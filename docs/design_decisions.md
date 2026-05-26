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
- 來源：McMaster-Carr（型號見 PLAN.md），STEP 格式透過 FreeCAD 轉 OBJ

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

## 渲染角度策略

**決定：Azimuth 8 個（每 45°），Elevation 3 個（15°/45°/75°）**

- 螺絲與螺母為旋轉對稱體（六角螺母每 60° 重複），繞 16 個角度大部分是 duplicate
- 減少 azimuth 到 8 個，改增加 pose 種類（3 種擺放姿態）
- 最終每類 288 張（8×3×3×4），總計 1152 張，比原估算 1536 更少但資訊密度更高

---

## 解析度

**決定：256 × 256**

- 128 太小：ResNet/EfficientNet 預設 input 224，128 需多加 resize；螺紋細節消失
- 256 是 MVP 合理下限，渲染速度與資訊量的平衡點
- 4060 8GB VRAM 跑得動

---

## 任務定義（本週）

**決定：分類（4-class classification）**

- 先驗證 pipeline 通，比直接做 detection 風險低
- CSV schema 已預留 `bbox_x/y/w/h` 與 `mask_path` 欄位，下週升級不需改格式
- 下週目標：Object Index pass 自動計算 bbox → 升級到 YOLOv8 detection

---

## 工具選用（Claude 端）

**資料夾建立與 render.py 骨架：給 Claude Code 處理**

- CC 可直接在專案目錄操作，比 Cowork 精確
- Cowork 適合重複性桌面操作，不適合這個任務

**Blender MCP（可選）**：
- 安裝 blender-mcp addon 後，Claude Desktop 可直接操控 Blender 場景
- 適合設定場景、調材質、除錯用，不作為批量渲染的核心依賴
