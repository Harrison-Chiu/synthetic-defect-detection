# 未來工作 / 靈感池

> 討論過程中提到、但 MVP 階段不做的想法。之後要加變體或升級時，從這裡挑。

---

## 任務升級

- **Object Detection (YOLOv8n)**：用 Blender Object Index pass 自動算 bbox，補進 CSV 的 `bbox_x/y/w/h` 欄位。CSV schema 已預留
- **Instance Segmentation**：Object Index pass 同時可輸出 mask，CSV 已預留 `mask_path` 欄位
- **多零件同框 (counting task)**：一個畫面放多個零件，挑戰計數與多目標偵測
- **Sim-to-real 驗證**：拍少量真實照片當測試集，評估 domain gap

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
