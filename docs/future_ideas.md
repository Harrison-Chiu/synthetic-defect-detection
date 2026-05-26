# 未來工作 / 靈感池

> 討論過程中提到、但 MVP 階段不做的想法。之後要加變體或升級時，從這裡挑。

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
