# Active — 當前 stage 指標

> 本檔是 always-on context 的「當前 stage」指標,**永久保留**(空或極簡也留)。
> 規則:只放指標與待辦,**不複製** stage 報告內容;細節一律連到對應檔。
> stage 結案 review 完 → 整份搬 `docs/history/`,本檔重置為下一 stage。

---

## 狀態:程式大重構已完成 → 下一步 S4 review(report 日 2026-06-02)

### 已完成:程式大重構(2026-05-30)
- `src/` 分層:`schema`(單一事實來源)/ `data`(runtime 確定性生成 `scene(i)=f(i,config)`)/
  `models` / `train`(step-based)/ `eval` / `render`(Blender 側)。入口 `cli.py`,參數 `configs/`。
- `output/` 角色分類:`patches/`(原 parts_stage2)`runs/<tag>/` `eval_set/` `samples/` `archive/`。
- 規格與決策見 `docs/reference/整理規則.md`;舊 script 遷移對應見 `scripts/README.md`。
- 已刪 6 支被 src/ 取代的舊 script(git `f805b98` 可救回)。

### parity run:`output/runs/s4_repro/`(驗證重構是否重現 S4 行為)
- 設定:3 頭 multi-head、**runtime 生成資料**、**新生成器凍結的 eval_set**(與 S4 非精確可比)。
- 結果:test mIoU **0.751**、defect IoU **0.457**;per-state IoU(head B):
  normal 0.862 / bend_l,h **0.000** / displace_l **0.000**,h 0.352 / remesh_l 0.252,h 0.338。
- **結論:src/ 重現了 S4 的三個已知問題**(bend 崩、displace_light 崩、remesh 相對最強)
  → pipeline 可信。defect IoU 高於 S4(0.342/0.385)疑因 runtime 資料多樣性大增(24000 distinct vs 800 重用)。
- ⚠️ 注意:此 per-state 是 **IoU**,S4 報告那張是 **recall**,別逐位對齊。eval set 與訓練同分布(in-distribution)。

## S4 review 結果(2026-05-30 已查證,取代先前「物理天花板」誤判)
> 交接脈絡見 `docs/bend_diagnosis_handoff.md`;S4 原始結果 `docs/stage4_results.md`。

**根因(已用程式 + 診斷實驗坐實,非推測):defect 監督的「任務形式不匹配」。**
- mask 是**整顆零件塗**:`generator.py:251-253`(`is_part = alpha>0.5` → 整個剪影上同一 class,依據 instance 層級的 defect flag,非逐像素幾何偏離);state head 同樣廣播到整顆(`dataset.py:51-58`)。
- 這正是 `schema.py` guardrail 早標的「`level=INSTANCE` 套逐像素 loss」mismatch —— 兩條獨立線索收斂同一點。
- **致命性隨「瑕疵的逐像素局部可見度」變化**:bend 是純全域形狀(內部像素 ≈ 正常)→ 逐像素監督要求把看起來正常的像素硬標 defect → 矛盾梯度 → 崩;remesh/displace 是表面紋理/起伏,局部就看得出 → 自洽 → 學得起來。
- **診斷實驗(s4_repro best.pt,`scripts/diag_perstate_pred.py` → `docs/figures/s4_repro_perstate_pred.png`)**:
  pred-defect recall 隨局部可見度**單調**:bend_l/h **0.000** < displace_l 0.116 < displace_h 0.800 < remesh_h 0.941 < remesh_l 0.986。
  bend 圖示:GT 整顆紅、pred 整顆綠(判成正常零件)。佐證 `docs/figures/stage3/A_normal_vs_defect_diff.png`(bend 差異集中輪廓邊緣,remesh/displace 滿表面)。

**三個已知問題重新歸因:**
1. ~~displace_light「柔光 HDRI」假設~~ → 不需要。它就是「局部可見度低」同一條軸的下端(recall 0.116),與 bend 同因不同程度。
2. ~~bend「256px 物理天花板」~~ → **證偽且危險**(差異肉眼可見)。動解析度/focal/角度救不了監督矛盾。
3. **multi-head**:本次 multihead defect IoU 0.457 不差,疑 runtime 資料多樣性緩解;非主線。

## 下一步:規劃 S5(方向已收斂)
1. **核心修正方向**:把 defect 監督從「整顆塗 + 逐像素 loss」改成與瑕疵本質相符的形式 —— 候選:(a) 合成資料特權,用 normal vs defect 幾何/深度差自動生 per-pixel mask(純 segmentation);(b) 把 state/type 改成 instance-level 監督(region-pooled),不再逐像素廣播。**一次只動一個變因**(沿用 handoff §6 紀律)。
2. **新比對圖**:可對「目前新版生成器」重做 FIG A 等比對圖(bend_light 已修正方向),後續報告好用。
3. **`.ipynb` 繳交橋**:課程要 .ipynb 原始碼,我們是 src/ 套件 → 截止(6/2)前需一條橋(薄 notebook import src + 跑 cli)。

## 仍待處理 / 缺口
- 只有 `pan_head` 有 defect patch(72 張);socket/hex_nut/flange 無 → 「多零件 QC+Sorting」題目的真實缺口。
- `configs/` 數值序列化(`cli --config` 載入)尚未實作,現以 dataclass 預設為準。
