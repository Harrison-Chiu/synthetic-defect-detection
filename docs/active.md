# Active — 當前 stage 指標

> 本檔是 always-on context 的「當前 stage」指標,**永久保留**(空或極簡也留)。
> 規則:只放指標與待辦,**不複製** stage 報告內容;細節一律連到對應檔。
> stage 結案 review 完 → 整份搬 `docs/history/`,本檔重置為下一 stage。

---

## 狀態:Stage 4 已完成,結果待 Harrison review

- 完整結果見 **`docs/stage4_results.md`**(HTML:`docs/stage4_report.html`)。
- ⚠️ S4 已跑完但**尚未審查、據報有問題** → 結論一律視為 provisional,**不要當定案事實引用**。
- best = single-head + 新資料,defect IoU 0.385(待查證);multi-head 路線已否決進 ablation。

## review 議程(非結論,待審)
- **displace_light 意外大退步** 0.78→0.235(渲染參數沒改,只動 HDRI pool 2→4)。
- bend 仍弱(0.06–0.08),疑似 256px 物理天花板。
- multi-head 失敗已知原因:aux head 吃掉 encoder bandwidth(L_A 僅佔總 loss 3%)。

## 下一步
1. review S4 結果 → 定位問題。
2. 才決定 S5 方向(可能大改)+ runtime 生成器的 config 數值。

> 註:專案結構重構(src/ + runtime 生成 + output 整理)依 `docs/reference/整理規則.md` 進行,與 S4 審查解耦。
