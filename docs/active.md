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

## S4 review 議程(下一階段焦點;非結論,待審)
> S4 原始結果見 `docs/stage4_results.md`(HTML `docs/stage4_report.html`),結論仍 provisional。
1. **displace_light 退步/崩潰**:S4 recall 0.78→0.235、本次 IoU 0.000。渲染參數沒改,只動 HDRI pool 2→4。
   疑因新增柔光 HDRI(monochrome_studio_02 / pretoria_gardens)讓 displace 反射訊號變平。→ 調 displace 強度 or 移除柔光 HDRI?
2. **bend 天花板**:S4 recall 0.06–0.08、本次 IoU 0.000。疑 256px 解析度物理極限。→ 限縮 elevation / 拉 focal / 加大角度?
3. **multi-head**:S4 記載 aux head 吃 encoder bandwidth(L_A 僅佔 3% 總 loss)。本次 multihead defect IoU 0.457 反而不差
   → 是 runtime 資料救了它,還是 loss 動態不同?值得查。schema guardrail 已標「instance 標籤套 pixel loss」。

## 下一步(S4 review 後才動)
1. review 上述三項 → 定位真因。
2. 才決定 **S5 方向**(可能大改 heads/loss)+ runtime 生成器的 **config 數值調校**(displace/HDRI)。
3. **`.ipynb` 繳交橋**:課程要 .ipynb 原始碼,我們是 src/ 套件 → 截止(6/2)前需一條橋(薄 notebook import src + 跑 cli)。

## 仍待處理 / 缺口
- 只有 `pan_head` 有 defect patch(72 張);socket/hex_nut/flange 無 → 「多零件 QC+Sorting」題目的真實缺口。
- `configs/` 數值序列化(`cli --config` 載入)尚未實作,現以 dataclass 預設為準。
