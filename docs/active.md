# Active — 當前 stage 指標

> 本檔是 always-on context 的「當前 stage」指標,**永久保留**(空或極簡也留)。
> 規則:只放指標與待辦,**不複製** stage 報告內容;細節一律連到對應檔。
> stage 結案 review 完 → 整份搬 `docs/history/`,本檔重置為下一 stage。

---

## 狀態:S5 實驗收斂,進入交付(report 日 2026-06-02)
- **計畫**:`docs/stage5_plan.md`｜**完整結果與分析**:`docs/stage5_results.md`(新,歸檔級)

### S5 一句話成果
interior-ignore 監督(壞件內部 W≈0)+ 回 S3 雙頭 → **defect IoU 0.362→0.39(勝 S3)**;
模型嚴重過參數化(bc8 即飽和、depth=3 52K 近乎等效);合成資料**無過擬合**(收斂 ~6000 步);
bend 不再崩 0 但本質仍弱(matched-FP 下 ≈ floor)。數字/矩陣/判讀全在 `stage5_results.md`。

### headline 模型
`base_c=8 / depth=4 / pos_weight=5 / defect_thr=0.7 / ~6000 步`(run `output/runs/s5_long`,test defIoU 0.39)。
極小備選 `depth=3`(52K)。重跑:`python cli.py train --tag X --base-c 8 --pos-weight 5 --total-steps 6000`。

### 剩餘待辦(交付前;優先序)
1. ⏸ **`.ipynb` 繳交橋**(課程要求,6/2)—— **暫緩**,待 Harrison 看完 report 決定範圍(含訓練到什麼程度)。
2. ⬜ thr=0.7 統一重生所有 report(舊 width 掃 run 存 thr0.5,不可並排)。
3. ⬜ 最終報告敘事彙整(可直接取材 `stage5_results.md`)。
4. ⬜ (低優)w_norm 旋鈕 plumb;S3 FIG A 進 report;headline best.pt 納入交付 .zip。

### 結構性缺口(跨 stage)
- 只有 `pan_head` 有 defect patch(72 pose);socket/hex_nut/flange 無 → 「多零件 QC+Sorting」的真實缺口。
- `configs/` 數值序列化(`cli --config`)未實作,現以 dataclass 預設為準。

---

## 已結案(歸檔)
- `docs/history/refactor_and_s4_review.md` — 程式大重構 + S4 review(bend 崩=監督形式不匹配,已坐實)。
- `docs/stage5_results.md` — S5 完整結果(待整個 stage 結案後可移 history/)。
