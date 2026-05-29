# 重構交接 Brief — CLAUDE.md 重組與後續議題

> 給接手的 Claude Code 助手。本文件是「意圖與約束」,**不是施工手冊**。
> 構想已定、為什麼要這樣已說明;但「在這個真實環境裡實際怎麼做、可不可行、用什麼順序」,
> 由你(CC)判斷,並與 Harrison 後續討論。請不要把本文件當成逐步指令照抄。

---

## 0. 這次重構的背景(為什麼要做)

Harrison 用 Claude Code 開這個專案討論時,發現**對話品質會變差**:討論偏侷限、有時無法理解當下的問題。
經分析,根因高度指向 `CLAUDE.md` 本身已經從「操作手冊」演變成「專案日誌(journal)」——
把 Stage 1~4 的每一個結果、ablation 數字、診斷推論、待討論項全部 inline 寫在裡面(170 行 / 11KB)。

對 LLM 的具體後果:
1. **注意力被歷史稀釋** — 已封存的歷史 stage 每次都跟「現在在做什麼」一起被讀進去,同層競爭注意力。
2. **結論多、context 少** — 記了「決定」沒記「為什麼這樣決定」,新討論只能接受結論,無法沿同一推理路徑深談。
3. **待討論與已定案混在同層** — Claude 不知道該站在哪個版本的事實上回話,於是時而把未定當已定、時而反過來。
4. **meta-instruction 污染** — 「要拿給另一位助手討論」這類備忘,對 Claude 而言讀起來是 instruction,會微妙帶偏。
5. **token 預算** — 11KB 每次先吃掉 3000+ tokens。

**重組目標**:把 always-on context 瘦身成只剩「穩定的事實 + 當前 stage」,歷史改成 on-demand 查閱。

---

## 1. 三份檔案的角色(先釐清,避免搞混)

| 檔案 | 角色 | 說明 |
| --- | --- | --- |
| `docs/history/CLAUDE_archive_001.md` | 舊版封存(留底) | 現行 CLAUDE.md 原封不動搬進來,完整保留歷史,別刪內容 |
| 本文件 + 下方草稿 | 參考稿 | 重組的意圖、規則、與精簡版草稿。**CC 不會 auto-load,正確** |
| `CLAUDE.md`(正式版) | 終點 | CC 核對環境後產出的精簡正式版,覆蓋舊的。**必須叫 CLAUDE.md**,auto-load 才生效 |

> 重點:過渡期可以用任意參考檔名,但**終點一定要回到 `CLAUDE.md`**,否則 auto-load 失效。

---

## 2. 【現在就做】Part 1

### 2.1 CLAUDE.md 重組原則
- 只保留**穩定**內容:專題定位、平台、路徑/命名 convention、設計原則、已踩雷(單行+link)、文件導覽。
- **當前 stage 的細節(head 結構、class 定義、loss、待討論)全部移到 `docs/active.md`**,CLAUDE.md 只留一行指標。
- 歷史 stage 報告移到 `docs/history/`,CLAUDE.md 不 inline 摘要,只留一行指標。
- 篇幅目標:從 ~170 行壓到 ~55 行。

### 2.2 CLAUDE.md 精簡版草稿(意圖示意,實際用字/路徑請 CC 核對環境)

```markdown
# 工廠零件辨識系統 — Claude Code 操作手冊

> NSYSU MEME552 深度學習期末專題 · 瑕疵檢測（QC / Defect Detection）
> 分工：Harrison 負責 Blender 合成資料集，另 2 位組員負責模型訓練
>
> **目前 active：Stage 4 — 詳見 `docs/active.md`**
> **歷史進度（Stage 1–3）見 `docs/history/`**

## 課程規則（成文，唯一來源 docs/reference/course_requirements.md）
- 課程：NSYSU MEME552 · 英文口頭報告
- 時長 / 報告日 / 繳交物：一律以 course_requirements.md 為準
  （成文規則僅此一份；下方「老師偏好」皆為課堂口頭線索的推斷，非成文）

## 老師偏好（非成文，Harrison 由課堂口頭線索推斷的得分方向）
- 自製資料集 > 用公開資料集（老師舉例「可自己拍」；本組以 Blender 合成達成）
  → 用網路公開資料集能過，但分數較低
- 模型自己設計，不要直接照抄開源架構（U-Net / YOLO 等）
- （推斷）pretrained 大概也在不希望使用的範圍 — 老師未明講，
  但依其「從頭到尾自己做」的一貫態度，Harrison 如此判斷
> 這些是「往老師偏好靠」的策略選擇，不是硬性規則。
> 設計時 default 朝此方向，但若有強理由偏離可與 Harrison 討論。

## 平台
- OS：Windows，工作目錄含中文：`D:\Harrison\中山\大四下\深度學習期末報告`
- Shell：PowerShell。⚠️ 是否需禁用 bash idiom 待 CC 驗證（見本 brief 待驗證清單）
- 訓練環境：conda env `dl_final`（Python 3.11 + PyTorch cu121）
- Blender：透過 blender-mcp 操作；物件層級修改優先用 `mcp__Blender__execute_blender_code`

## 命名約定（script 硬依賴 — 要改必須同步改 script）
- 零件物件：socket_head / pan_head / hex_nut / flange_nut
- 相機：RenderCam（script 用名字抓，非抓 active camera）
- World Environment Texture 節點：HDRI_node
- 材質：stainless_steel（4 零件共用）
  → 不是不能改，而是「改名 = 連帶改所有 script 的字串」，成本高才列出

## 物理事實（改了會壞，非約定）
- 零件 scale = 0.001：STEP→OBJ 的 mm→m 對齊；dimensions 已校正成真實尺寸。
  動 scale 會讓所有零件尺寸錯亂 → 別動。

## 關鍵路徑（資料集版本規則尚未定案，見本 brief Part 2）
| 用途 | 路徑 |
| --- | --- |
| Blend 檔 | blender/114-2_DLcourse_FinalProject-01.blend |
| 3D 模型 | assets/3d_model/<McMaster-Carr 完整檔名>.obj |
| HDRI（git 不追蹤） | assets/hdri/<name>_4k.exr |
| 單零件渲染 | output/parts_stage2/{defect_state}/ |
| 場景合成 | output/scenes/{scene_id:05d}/ |

## 設計原則（持續遵守）
1. 可復現性 > 採樣多樣性：能用 pure grid 就不要 random+seed
2. schema 不亂加欄位：CSV / meta 只放真的會用的欄位
3. 分階段思考：MVP 通了才加變體；新點子先記到 docs/future_ideas.md
4. 自製優先（呼應上方「老師偏好」）

## 已踩過的雷（完整見 docs/render_gotchas.md）
- 相機 near clip 預設 0.1m > CAMERA_RADIUS → 渲出全空白，已修為 0.001m
- Blender 5.x 引擎名是 BLENDER_EEVEE（不是 _NEXT）
- HDRI 入鏡需 scene.render.film_transparent = True

## 文件導覽
- docs/active.md — 當前 stage 的計畫 / 定案 / 待討論（最常更新）
- docs/history/ — 已完成 stage 的報告與結果（封存，需要才查）
- docs/design_decisions.md — 為何選 X 不選 Y（ADR）
- docs/render_gotchas.md — 渲染排查清單
- docs/future_ideas.md — 提到但目前不做的點子
```

### 2.3 docs/active.md 建立(草稿)

```markdown
# Active — Stage 4（已跑完，結果待 review，更新 2026-05-29）

> 本檔只放**當前 stage**。Stage 4 結束時整份搬到 docs/history/。
> 上半 = 定案（執行時 default 用這版事實）
> 下半 = 待討論（要找助手討論才看；未定，不要當成已定事實）
>
> V4 訓練已完成，表現不如預期，root cause 尚未審查。
> 下一步：review V4 結果 → 定位不如預期的原因 → 再決定調整方向。

## 任務定義（Stage 4 當前）
- 輸入：256×256 RGB 場景圖（多 pan_head 散落工業背景，可重疊）
- 輸出：per-pixel 多頭預測
- 解析度維持 256（bend 是 macro 問題不是 pixel 問題，已駁回 384）

## ✅ 定案
**架構：Multi-head V4**
| head | 任務 | 輸出 |
| --- | --- | --- |
| A | part / bg | 2-way |
| B | defect_state | 7-way |
| C | defect_type | 4-way |
| E | binary defect | 1 |
- 砍掉 D severity（per-instance label 不適合 pixel-level）

**Loss**：weighted sum（α=1.0）。Uncertainty Weighting (Kendall 2018) 已評估，Stage 4 不採用。

**資料端**
- Bend：deform_axis 改用 Empty `origin` + ±30° jitter（不強制垂直，偏離不超過 30°）
- Remesh：新 light = 舊 heavy；新 heavy = 再激進一級
- HDRI：補回 4 個 + composite 強制同 scene 用同一 HDRI

## 🟡 待討論（未定，找助手討論時才進此段）
- V4 不如預期的 root cause（最優先，待 review）
- α 權重、Dice 怎麼算
- bend elevation 是否也限縮
- 新 remesh_heavy 的 octree_depth 具體值
- 總部件數量、資料集規模
- defect ratio / focal / HDRI strength 微改動：全砍或部分留

## 📝 已記錄但不做（往後優化，不要主動提）
Uncertainty Weighting · Hierarchical consistency loss · Localized defect mask ·
GradNorm / PCGrad · Domain Adversarial Training · image-level aux head（光照/角度）
```
> 註:active.md 的 class 定義、per-state IoU 等實際數字,Harrison 手邊若有會再補;
> Stage 1~3 的歷史數字請查 docs/history/。

### 2.4 歷史分離(構想,實際移哪些檔由 CC 核對)
構想:把已完成 stage 的歷史性文件集中到 `docs/history/`,讓 CLAUDE.md 與 active context
只保留當前 stage,歷史改成 on-demand。哪些檔該移、移去哪,請 CC 核對環境後決定。

### Part 1 的 do / don't
- ✅ 照「意圖」重組,用字與路徑以**環境實況**為準(本草稿是示意)。
- ✅ 課程規則 vs 老師偏好「分層」這個結構要保留(這是這次重組的關鍵之一)。
- ✅ 命名「約定 vs 物理事實」分層要保留。
- ❌ **不要**把本 brief Part 2 的未來議題拿來現在做。
- ❌ **不要**自行把歷史 journal 內容補回精簡版 CLAUDE.md(那正是要消除的問題)。
- ❌ **不要**自行腦補課程規則;成文規則只有 course_requirements.md 一份。

---

## 3. 【只記不做】Part 2 — 未來議題清單

> ⚠️ 以下全是**構想備忘**,這次重構**不要動手**。等 Harrison 另開討論再逐項展開。
> 每項給的是「想達成什麼 / 為什麼 / 不可妥協的約束」,可行性與做法由 CC 與 Harrison 後續決定。

### 議題 A:`src/` package 化(連帶含 label schema,見議題 B)
- **意圖**:目前邏輯散在 `scripts/` 多個檔 + notebook cell 裡(repo 61% 是 Jupyter)。
  想抽成 `src/` 模組(render / data / models / train / eval 分層),scripts 只留 entry point。
- **為什麼**:重複 utility 多;Stage 4 要管 4 個 head + 可調 loss,不抽模組會越來越痛;
  notebook 化也是先前 nbconvert kernel 跑錯 Python 的遠因。
- **約束**:符合「自製優先」——抽模組是整理自己的程式,不是引入外部架構。

### 議題 B:label schema(獨立項 — 源自程式審查發現的潛在風險)
- **性質**:這是審查程式時發現的**風險防治**,與「專案結構更新」是兩件事,故獨立成項;
  但實作上它剛好坐落在 `src/models/` 與 `src/data/` 之間,與議題 A 相鄰,可一起規劃。
- **意圖**:把「每個 head 的定義(類別數 / level 是 pixel 或 instance / 標籤來源 / loss 形式)」
  集中成**單一事實來源**,讓 model channel、dataloader、eval 全部從這一處推導,不再各自寫死。
- **為什麼**:Stage 4 砍 severity head,正是因為 per-instance 標籤被誤用到 pixel-level loss。
  根因是「level 屬性沒有被明確寫出、缺護欄」,跑完才發現。集中定義後這類錯能在定義層攔下。
- **約束**:呼應「schema 不亂加欄位 / 可復現性優先」——讓事實只有一份、明確、機器可讀。

### 議題 C:場景改實時生成 + 資料集版控(兩者合併,因為彼此相關)
- **意圖**:場景合成想從「offline 存檔一千張」改成「runtime(DataLoader)即時組合」。
- **為什麼**:
  (a) defect ratio / 背景分佈 / 重疊程度想當成**可調參數**,而不是每次重新渲染;
  (b) 連帶解決資料集版控——若場景即時生成,需版控的就只剩「零件 patch + 生成 config」,
      範圍比「版控一千張場景圖」小很多、也乾淨;
  (c) 對先前的 bend 信號弱問題有利:相機 elevation / 焦距 / 視角若由 runtime 決定,可快速 sweep。
- **不可妥協的約束**:**零件 patch 仍要 Blender 預渲**,以保物理光照真實性;
  **不可**為了省事把零件 patch 也改成程序化生成。
- **附帶**:Harrison 目前**偏好走這個方向(實時生成)**。資料集版控的最終形式取決於此方向是否採用,
  故兩者一起評估,不分開決定。

### 議題 D:協作工作流(備忘,優先度最低)
- task brief 模式(每個任務一個 brief 檔:背景 / 目標 / 驗收)。
- decision log 用 ADR 格式,與 CLAUDE.md 解耦(CLAUDE.md 不 inline decision,只指向 design_decisions.md)。
- 討論型 vs 執行型對話用不同 context;多助手 handoff template。
- **註**:Harrison 想先驗證「CLAUDE.md 重組」是否已解決對話品質問題,再決定這層要不要做。

---

## 4. 待 CC 在環境中驗證(這是「查證」,不是「施工」)

1. **bash idiom 到底會不會壞**:Harrison 記得 CC 先前跑過 bash 指令,所以「Windows 純 PowerShell、
   不能用 bash」這個假設**不一定成立**(環境可能有 git-bash / WSL)。請實測後回報,
   再決定 CLAUDE.md 那條 shell 規則怎麼寫。先前的 bash idiom 對照(僅供參考,待驗證):
   `export VAR` / `$VAR` / `> /dev/null` / `rm -rf` / `&&` 在 PowerShell 的對應與相容性。
2. **各路徑現況核對**:Stage 1~3 報告現在實際在 docs/ 的哪裡、是否已有 history/ 子資料夾。
3. **精簡版 CLAUDE.md 的用字/路徑**:以環境實況覆蓋本 brief 的示意草稿。

---

## 5. 給 CC 的協作原則(本次交接的底層約定)

- **分工**:Harrison 與規劃端負責「想清楚要什麼、為什麼」;CC 負責「在真實環境判斷怎麼做、可不可行」。
- **本 brief 給的是意圖與約束,不是逐步操作**。實作順序、技術選型、可行性由 CC 判斷並與 Harrison 討論。
- **意圖可以開放,約束必須明確**:構想本身留白讓 CC 發揮,但標明「不可妥協」的點(如議題 C 的零件 patch 預渲)請務必守住。
- 遇到本 brief 與環境實況衝突,以**環境實況為準**,並回報 Harrison。
