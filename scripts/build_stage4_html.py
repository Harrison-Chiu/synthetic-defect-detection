"""
產生單一自帶資源的 docs/stage4_report.html
所有圖片 base64 內嵌，瀏覽器直接打開即可（適合遠端瀏覽）
"""
import os
import base64
import json

BASE = r"D:\Harrison\中山\大四下\深度學習期末報告"
FIG  = os.path.join(BASE, "docs", "figures", "stage4")
OUT  = os.path.join(BASE, "docs", "stage4_report.html")


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img(name, caption):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return f'<div class="note">[missing figure: {name}]</div>'
    return (f'<figure><img src="data:image/png;base64,{b64(p)}" />'
            f'<figcaption>{caption}</figcaption></figure>')


cmp_path = os.path.join(BASE, "output", "stage4_eval_compare.json")
c = json.load(open(cmp_path, encoding="utf-8"))
s3 = c["stage3_baseline_on_s3_data"]
sd = c["stage3_arch_on_s4_data"]
mh = c["stage4_multihead_on_s4_data"]

states_order = ["bend_light", "bend_heavy", "displace_light", "displace_heavy",
                "remesh_light", "remesh_heavy"]


def row(state, fmt="{:.3f}"):
    v3 = s3["per_state_recall"][state]
    vs = sd["per_state_recall"][state]
    vm = mh["per_state_recall"][state]
    # 標 best
    best = max(v3, vs, vm)
    def cell(v):
        cls = ' class="best"' if v == best else ''
        return f'<td class="num"{cls}>{fmt.format(v)}</td>'
    return f"<tr><td>{state}</td>{cell(v3)}{cell(vs)}{cell(vm)}</tr>"


html = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>Stage 4 — Defect Detection Report</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft JhengHei", sans-serif;
          max-width: 1100px; margin: 30px auto; padding: 0 20px; color: #222; line-height: 1.55; }}
  h1 {{ border-bottom: 3px solid #2a4d8f; padding-bottom: 8px; }}
  h2 {{ color: #2a4d8f; border-left: 5px solid #2a4d8f; padding-left: 10px; margin-top: 36px; }}
  h3 {{ color: #555; margin-top: 24px; }}
  table {{ border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
  th {{ background: #f0f4fa; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.best {{ background: #e8f5e9; font-weight: bold; }}
  .kpi-yes {{ color: #198754; font-weight: bold; }}
  .kpi-no  {{ color: #c0392b; font-weight: bold; }}
  figure {{ margin: 20px 0; padding: 10px; background: #fafafa;
            border: 1px solid #e0e0e0; border-radius: 6px; }}
  figure img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
  figcaption {{ font-size: 0.92em; color: #555; margin-top: 8px; text-align: center; }}
  .takeaway {{ background: #fff8dc; border-left: 5px solid #d4a017;
               padding: 10px 16px; margin: 14px 0; }}
  .note {{ background: #eef6f9; border-left: 5px solid #5b9bd5;
           padding: 10px 16px; margin: 14px 0; font-size: 0.95em; }}
  .warn {{ background: #fdf0f0; border-left: 5px solid #c0392b;
           padding: 10px 16px; margin: 14px 0; font-size: 0.95em; }}
  code {{ background: #f3f3f3; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
</style></head>
<body>

<h1>Stage 4 — Defect Detection 結果報告</h1>
<p><strong>NSYSU 深度學習期末專題</strong> ｜ pan_head 螺絲瑕疵 segmentation ｜ Stage 3 → Stage 4</p>

<div class="takeaway">
<strong>一句話結論：</strong>Stage 4 兩條獨立路徑 — (1) 資料 bug fix + 強化 (2) Multi-head 架構 — 跑下來
<strong>路徑 1 成功</strong>（defect IoU 0.362 → <strong>0.385</strong>），<strong>路徑 2 失敗</strong>（→ 0.342，
multi-head 的 aux loss 把 encoder bandwidth 吃掉）。最終 best 採「S3 arch + S4 data」。
意外發現：Displace 在新 HDRI 配置下退步，是 sim-2-real 的縮影，留 Stage 5。
</div>

<h2>1. 三組對照總覽</h2>
<table>
<tr><th>指標</th>
    <th class="num">S3 baseline<br/>(S3 arch + S3 data)</th>
    <th class="num"><strong>S3 arch + S4 data</strong><br/>(隔離資料效應 ← 最終 best)</th>
    <th class="num">S4 multi-head<br/>(S4 arch + S4 data)</th></tr>
<tr><td>Test pixel_acc</td><td class="num">94.1%</td><td class="num best">94.6%</td><td class="num">94.5%</td></tr>
<tr><td>Test mIoU</td><td class="num">{s3['test_3class']['mIoU']:.3f}</td>
    <td class="num best">{sd['test_3class']['mIoU']:.3f}</td>
    <td class="num">{mh['test_3class']['mIoU']:.3f}</td></tr>
<tr><td><strong>Test defect IoU</strong></td>
    <td class="num">{s3['test_3class']['IoU_per_class'][2]:.3f}</td>
    <td class="num best"><strong>{sd['test_3class']['IoU_per_class'][2]:.3f}</strong></td>
    <td class="num">{mh['test_3class']['IoU_per_class'][2]:.3f}</td></tr>
<tr><td>Test bg IoU</td>
    <td class="num">{s3['test_3class']['IoU_per_class'][0]:.3f}</td>
    <td class="num">{sd['test_3class']['IoU_per_class'][0]:.3f}</td>
    <td class="num">{mh['test_3class']['IoU_per_class'][0]:.3f}</td></tr>
<tr><td>Test normal_part IoU</td>
    <td class="num">{s3['test_3class']['IoU_per_class'][1]:.3f}</td>
    <td class="num">{sd['test_3class']['IoU_per_class'][1]:.3f}</td>
    <td class="num">{mh['test_3class']['IoU_per_class'][1]:.3f}</td></tr>
<tr><td>Black-bg mIoU</td>
    <td class="num">{s3['test_3class_blackbg']['mIoU']:.3f}</td>
    <td class="num">{sd['test_3class_blackbg']['mIoU']:.3f}</td>
    <td class="num">{mh['test_3class_blackbg']['mIoU']:.3f}</td></tr>
<tr><td>Black-bg defect IoU</td>
    <td class="num">{s3['test_3class_blackbg']['IoU_per_class'][2]:.3f}</td>
    <td class="num">{sd['test_3class_blackbg']['IoU_per_class'][2]:.3f}</td>
    <td class="num">{mh['test_3class_blackbg']['IoU_per_class'][2]:.3f}</td></tr>
</table>

{img("compare_overall_bar.png", "圖 1：3 組實驗在 4 個指標上的對照。S3 arch + S4 data 在 textured 上 best，但黑底 ablation 跌幅較大。")}

<h2>2. Per-state Defect Recall（細粒度比較）</h2>
<table>
<tr><th>Defect state</th>
    <th class="num">S3 baseline</th>
    <th class="num">S3 arch + S4 data</th>
    <th class="num">S4 multi-head</th></tr>
{row("bend_light")}
{row("bend_heavy")}
{row("displace_light")}
{row("displace_heavy")}
{row("remesh_light")}
{row("remesh_heavy")}
</table>
<p style="font-size:0.9em; color:#666;">綠底 = 該 state 三組中表現最佳</p>

{img("compare_per_state_bar.png", "圖 2：6 個 defect state × 3 組實驗的 binary defect recall 對照。")}

<div class="note">
<strong>逐項解讀：</strong>
<ul>
<li><strong>Bend</strong>：axis fix 把 bend 從 0.04~0.05 拉到 0.06~0.08 — 確實有效，但**沒到 plan 預期的 0.15+**。
    Stage 3 的 50% bend 樣本是「彎進畫面」廢樣本，axis 修正後 effective 訓練量翻倍；
    但每樣本本身在 256px 下的剪影變化只有幾個 px，pixel-level signal 還是太弱。</li>
<li><strong>Remesh</strong>：大躍進 +0.34 ~ +0.47，**最強的單一變因**。
    octree (5,6)→(4,4) 視覺塊狀化更明顯，model 容易學。
    Stage 3 sanity 時 octree 7-8 視覺幾乎=normal 的問題確實是癥結。</li>
<li><strong>Displace</strong>：意外退步 -0.17 ~ -0.55。**渲染參數完全沒改**，只是 HDRI pool 從 2 → 4。
    推論：新加的 <code>monochrome_studio_02</code>（純白光）與 <code>pretoria_gardens</code>（室外柔光）
    讓 displace 的 normal-perturbation 反射差異變小。
    在強光下 displace 的凹凸面有明顯陰影，柔光下訊號變平。
    <strong>這是 sim-2-real 的縮影</strong>。</li>
</ul>
</div>

<h2>3. 為什麼 Multi-head 失敗？</h2>

<p>訓練 log epoch 30 的 loss 分解：</p>
<pre style="background:#f3f3f3; padding:10px; border-radius:4px;">
L_total = 2.235
  L_A      = 0.063  (3%)   ← 主任務 (part/bg + binary defect)
  L_B_ce   = 0.538  (24%)  ← aux: 7-way state CE
  L_B_dice = 0.719  (32%)  ← aux: 7-way state Dice
  L_C_ce   = 0.418  (19%)  ← aux: 4-way type CE
  L_C_dice = 0.497  (22%)  ← aux: 4-way type Dice
</pre>

<div class="warn">
<strong>L_A 只貢獻 3% 總 loss</strong> → 梯度幾乎完全由 aux task 主導。
Encoder bandwidth 被拉去學「分清 bend_l vs bend_h」這種極弱 signal 區分，
反而傷害了主任務「part vs bg」與「defect vs normal」的學習。<br/><br/>
這是 multi-task learning 的標準失敗模式 — 沒做 loss balancing 直接 sum，
強 loss 數值的頭壓制強重要性的頭。
</div>

{img("B_state_confusion.png", "圖 3：Head B 7-way 預測 vs GT 的 row-normalized confusion。")}

{img("C_type_confusion.png", "圖 4：Head C 4-way type 預測 vs GT 的 row-normalized confusion。")}

<p><strong>可能補救</strong>（未在 Stage 4 採用）：</p>
<ul>
<li>α_B = α_C = 0.1 ~ 0.3 重新加權</li>
<li>Uncertainty Weighting (Kendall et al. 2018) — 每個 task 學 trainable log σ²</li>
<li>純拿 B 7-way 取代 A+B binary（推論時 collapse argmax）→ 完全沒 aux task overhead</li>
</ul>

<h2>4. Decision Point 處理</h2>

<p>Plan §5 規定：</p>
<ul>
<li>Bend IoU &gt; 0.15 → 結案</li>
<li>0.05–0.15 → 加碼一項視覺強化</li>
<li>&lt; 0.05 → 重新評估 bend 是否該留</li>
</ul>

<p>實際 bend IoU（best 模型 = S3 arch + S4 data）：</p>
<ul>
<li>bend_light = {sd['per_state_recall']['bend_light']:.3f}</li>
<li>bend_heavy = {sd['per_state_recall']['bend_heavy']:.3f}</li>
</ul>

<p>落在 <strong>0.05–0.15 區間 → 應加碼</strong>。但考慮到：</p>
<ol>
<li>報告期限 2026-06-02 約 5 天，時間壓力大</li>
<li>現有 best (defect IoU 0.385) 已超 Stage 3 (0.362)，故事完整</li>
<li>Displace 退步是更值得追的問題</li>
</ol>
<p><strong>決定：Stage 4 結案於現有結果</strong>，bend 在報告誠實標註「256px 解析度下的物理限制」。
Phase 2 列入 future_ideas.md 供 Stage 5 / 報告後優化。</p>

<h2>5. Stage 4 做了什麼（vs Stage 3）</h2>
<table>
<tr><th>類別</th><th>內容</th></tr>
<tr><td>資料 — bug fix</td><td>Bend axis 對齊相機（旋轉 pan_head + deform_axis=X）+ 隨機 ±方向 + ±30° jitter</td></tr>
<tr><td>資料 — bug fix</td><td>HDRI 2 → 4（Stage 3 寫死 2 個是 bug）</td></tr>
<tr><td>資料 — bug fix</td><td>composite 每 scene 統一 HDRI（修原本同 scene 物理不一致）</td></tr>
<tr><td>資料 — 強化</td><td>Remesh octree LIGHT (7,8)→(5,5), HEAVY (5,6)→(4,4)</td></tr>
<tr><td>架構（multi-head 嘗試）</td><td>A part/bg (2) + B state (7) + C type (4)，共用 encoder-decoder</td></tr>
<tr><td>Loss</td><td>L_A + α(L_B_CE+L_B_Dice) + α(L_C_CE+L_C_Dice), α=1.0, GT gate</td></tr>
<tr><td>Optimizer</td><td>Adam lr=1e-3 + ReduceLROnPlateau patience=3</td></tr>
</table>

<h2>6. 訓練動態</h2>

{img("training_curves.png", "圖 5：Multi-head 訓練曲線。L_A 收斂很快但 L_B/C 持續高 → 梯度被 aux task 主導。ReduceLROnPlateau 在 epoch 22/27 觸發。")}

{img("ablation_textured_vs_black.png", "圖 6：Multi-head 在 textured DR vs 黑底的對比。黑底跌幅比 Stage 3 大（0.20 vs 0.034），但這跟新 HDRI 分布有關，不一定是模型靠背景 shortcut。")}

<h2>7. 後續方向（Stage 5+）</h2>
<ol>
<li><strong>Displace 退步調查</strong>：渲 displace_heavy × 4 HDRI 對照圖，量化「不同 HDRI 下的 displace 視覺強度」，
    考慮把柔光 HDRI 移除或加大 displace strength</li>
<li><strong>Bend 衝 0.15+</strong>：限縮 bend instance 的 elevation ∈ [-30, 30]、拉長 focal、或試 60° bend angle（單變因）</li>
<li><strong>Multi-head 救活</strong>：α_B = α_C = 0.1 試試，或改用 Uncertainty Weighting</li>
<li><strong>取代 binary head</strong>：只用 7-way B head 推論，跳過 aux 衝突</li>
<li><strong>Epoch 30 崩潰</strong>：把 ReduceLROnPlateau 合併到 single-head 訓練</li>
</ol>

<h2>8. Artifact 列表</h2>
<pre style="background:#f3f3f3; padding:10px; border-radius:4px; font-size:0.85em;">
output/stage4_singlehead_best.pt        ← 最終 best model (defect IoU 0.385)
output/stage4_singlehead_history.json
output/stage4_best_model.pt             ← multi-head model (ablation)
output/stage4_history.json
output/stage4_eval_summary.json         ← multi-head 完整 eval
output/stage4_eval_compare.json         ← 3 組對照表
output/stage4_train.log
output/stage4_ablation_singlehead.log
output/stage4_epoch_snapshots/          ← multi-head 訓練演化

output/parts_stage2/                    ← Stage 4 新 render (504 張)
output/scenes/                          ← 1000 scenes
output/scenes_black/                    ← 100 黑底 ablation

docs/figures/stage4/                    ← 報告圖
docs/stage4_plan.md
docs/stage4_results.md                  ← 完整 markdown 版
docs/stage4_report.html                 ← 本檔（單檔內嵌圖）
</pre>

<p style="margin-top:50px; color:#888; font-size:0.85em; text-align:center;">
generated by <code>scripts/build_stage4_html.py</code>
</p>

</body></html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {OUT}  ({os.path.getsize(OUT) / 1024:.1f} KB)")
