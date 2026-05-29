# Stage 1 — MVP 完成報告

> 期間：2026-05-17 → 2026-05-18  
> 目標：驗證 Blender 合成資料集 → 訓練 pipeline 整條通

---

## 達成

| 項目 | 狀態 |
|------|------|
| 4 類零件 STEP → OBJ → Blender 場景 | ✅ |
| Pure grid 採樣方案 (12 elev × 3 azim × 4 hdri) | ✅ |
| 576 張 RGBA PNG（透明背景、HDRI 打光、金屬反射） | ✅ |
| 標注 CSV (`labels/metadata.csv`) | ✅ |
| Baseline 訓練：手刻 CNN + skip connection | ✅ |
| **Test accuracy** | **100%** |

---

## 資料集樣本

每類在三個典型仰角的代表圖（azimuth=0°, HDRI=university_workshop）：

### Socket Head Cap Screw
| el=-40° (仰視) | el=+40° (3/4 視角) | el=+70° (俯視) |
|---|---|---|
| ![](../../output/archive/dataset/raw/socket_head/socket_head_az000_el-040_h0.png) | ![](../../output/archive/dataset/raw/socket_head/socket_head_az000_el+040_h0.png) | ![](../../output/archive/dataset/raw/socket_head/socket_head_az000_el+070_h0.png) |

### Pan Head Phillips Screw
| el=-40° | el=+40° | el=+70° |
|---|---|---|
| ![](../../output/archive/dataset/raw/pan_head/pan_head_az000_el-040_h0.png) | ![](../../output/archive/dataset/raw/pan_head/pan_head_az000_el+040_h0.png) | ![](../../output/archive/dataset/raw/pan_head/pan_head_az000_el+070_h0.png) |

### Hex Nut
| el=-40° | el=+40° | el=+70° |
|---|---|---|
| ![](../../output/archive/dataset/raw/hex_nut/hex_nut_az000_el-040_h0.png) | ![](../../output/archive/dataset/raw/hex_nut/hex_nut_az000_el+040_h0.png) | ![](../../output/archive/dataset/raw/hex_nut/hex_nut_az000_el+070_h0.png) |

### Flange Nut
| el=-40° | el=+40° | el=+70° |
|---|---|---|
| ![](../../output/archive/dataset/raw/flange_nut/flange_nut_az000_el-040_h0.png) | ![](../../output/archive/dataset/raw/flange_nut/flange_nut_az000_el+040_h0.png) | ![](../../output/archive/dataset/raw/flange_nut/flange_nut_az000_el+070_h0.png) |

---

## 結果評估

**100% 是預期的**。當前任務有四個讓問題變得 trivial 的條件：

1. **完全乾淨的背景**（透明 → RGB 後變黑）
2. **完美固定的 canonical pose**
3. **4 類視覺差異大**（螺絲 vs 螺母 vs 凸緣螺母 vs hex socket 一眼可分）
4. **studio-quality HDRI**，無實際場景雜訊

→ 結論：pipeline 整條通了，但**問題本身對 deep model 太簡單**，分類準確率不是有意義的鑑別指標。

---

## 接下來方向（討論用）

詳見下方「Stage 2 方向討論」段落。
