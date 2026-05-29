# 工業零件場景理解 — Claude Code 操作手冊

> NSYSU MEME552 深度學習期末專題 · 自製 Blender 合成資料集 + 自製神經網路
> 題目橫跨：**主 QC / Defect Detection，兼具 Sorting & Grasping**（多零件辨識/定位）
> 分工：Harrison 負責 Blender 合成資料集，另 2 位組員負責模型訓練
>
> **當前 active：`docs/active.md`** ｜ 歷史 stage：`docs/history/`
> **結構重構已落地** — `src/` 分層 + runtime 生成 + `output/` 角色分類 + `cli.py` 入口。規則見 `docs/reference/整理規則.md`。

## 課程規則（成文，唯一來源 `docs/reference/course_requirements.md`）
- 報告日 / 時長 / 繳交物 / 評分 一律以該檔為準，**不要自行腦補**。
- 組內題目選擇（QC + Sorting 橫跨）記在此檔，不在 course_requirements。

## 老師偏好（非成文，Harrison 由課堂口頭線索推斷的得分方向）
- 自製資料集 > 公開資料集（本組以 Blender 合成達成）
- 模型自己設計，不直接抄開源架構（U-Net / YOLO 等）
- （推斷）pretrained 大概也不在希望使用範圍
> 這些是「往老師偏好靠」的策略 default，非硬規則；有強理由偏離可與 Harrison 討論。

## 平台
- OS：Windows，工作目錄含中文：`D:\Harrison\中山\大四下\深度學習期末報告`
- Shell：**預設 PowerShell；bash 亦可用**（git-bash）。
- 訓練環境：conda env `dl_final`（Python 3.11 + PyTorch cu121）
- Blender：透過 blender-mcp 操作；物件層級修改優先用 `mcp__Blender__execute_blender_code`

## 命名約定（script 硬依賴 — 改名 = 連帶改所有 script 字串）
- 零件物件：`socket_head` / `pan_head` / `hex_nut` / `flange_nut`
- 相機：`RenderCam`（script 用名字抓，非 active camera）
- World Environment Texture 節點：`HDRI_node`
- 材質：`stainless_steel`（4 零件共用）

## 物理事實（改了會壞，非約定）
- 零件 scale = 0.001：STEP→OBJ 的 mm→m 對齊；dimensions 已校正成真實尺寸。**別動 scale**。

## 關鍵路徑（現況；重構目標見 `docs/reference/整理規則.md`）
| 用途 | 路徑 |
| --- | --- |
| Blend 檔 | `blender/114-2_DLcourse_FinalProject-01.blend` |
| 3D 模型 | `assets/3d_model/<McMaster-Carr 完整檔名>.obj` |
| HDRI（git 不追蹤） | `assets/hdri/<name>_4k.exr` |
| 零件預渲 patch | `output/patches/{defect_state}/`（版控核心，Blender 預渲）|
| 場景 | **runtime 生成**（`src/data`，不落地）；凍結 eval：`output/eval_set/`；配圖：`output/samples/` |
| 訓練產出 | `output/runs/<stage_tag>/`（`best.pt` / `history.json` / `train.log` / `snapshots/`）|
| 封存（舊 stage） | `output/archive/`（scenes、scenes_black、S1 dataset、舊 stage models）|
| 程式碼 | `src/{schema,data,models,train,eval}/`；統一入口 `cli.py`；生成/訓練參數 `configs/` |

## 設計原則（持續遵守）
1. **可復現性 > 採樣多樣性**：能用 pure 確定性（grid / `f(i,config)`）就不要 random+seed
2. **schema 不亂加欄位**：CSV / meta 只放真的會用的欄位
3. **分階段思考**：MVP 通了才加變體；新點子先記到 `docs/future_ideas.md`
4. **自製優先**（呼應老師偏好）

## 已踩過的雷（完整見 `docs/render_gotchas.md`）
- 相機 near clip 預設 0.1m > CAMERA_RADIUS → 渲出全空白，已修為 0.001m
- Blender 5.x 引擎名是 `BLENDER_EEVEE`（不是 `_NEXT`）
- HDRI 入鏡需 `scene.render.film_transparent = True`
- bend 用 SIMPLE_DEFORM 時直接旋轉物件 + `deform_axis="X"`，別靠 Empty origin（座標互動有坑）

## 文件導覽
- `docs/active.md` — 當前 stage 指標（最常更新）
- `docs/history/` — 已結案 stage 報告（封存，需要才查）
- `docs/reference/整理規則.md` — 結構重構/版控規則（施工中）
- `docs/design_decisions.md` — 為何選 X 不選 Y（ADR）
- `docs/render_gotchas.md` — 渲染排查清單
- `docs/future_ideas.md` — 提到但目前不做的點子
