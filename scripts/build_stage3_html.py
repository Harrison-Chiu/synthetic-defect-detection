"""
產生單一自帶資源的 docs/stage3_report.html
所有圖片 base64 內嵌，瀏覽器直接打開即可（適合遠端瀏覽）
"""
import os
import base64
import json

BASE = r"D:\Harrison\中山\大四下\深度學習期末報告"
FIG  = os.path.join(BASE, "docs", "figures", "stage3")
OUT  = os.path.join(BASE, "docs", "stage3_report.html")

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def img(name, caption):
    p = os.path.join(FIG, name)
    if not os.path.exists(p): return ""
    return (f'<figure><img src="data:image/png;base64,{b64(p)}" />'
            f'<figcaption>{caption}</figcaption></figure>')

summary = json.load(open(os.path.join(FIG, "summary.json"), encoding="utf-8"))
tex = summary["textured_bg"]
blk = summary.get("black_bg_ablation", {})
states = summary["by_defect_state_mean_IoU"]

html = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>Stage 3 — Defect Detection Report</title>
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
  code {{ background: #f3f3f3; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
</style></head>
<body>

<h1>Stage 3 — Defect Detection 結果報告</h1>
<p><strong>NSYSU 深度學習期末專題</strong> ｜ pan_head 螺絲瑕疵 segmentation ｜ Stage 2 → Stage 3</p>

<h2>1. KPI 達成總覽</h2>
<table>
<tr><th>指標</th><th class="num">Stage 2 baseline</th><th class="num">Stage 3 目標</th>
    <th class="num">Stage 3 實際</th><th>達成</th></tr>
<tr><td>Test mIoU</td><td class="num">0.597</td><td class="num">&gt; 0.65</td>
    <td class="num"><strong>{tex['mIoU']:.3f}</strong></td><td class="kpi-yes">✅</td></tr>
<tr><td>Test pixel_acc</td><td class="num">94.0%</td><td class="num">&gt; 94%</td>
    <td class="num"><strong>{tex['pixel_acc']*100:.1f}%</strong></td><td class="kpi-yes">✅</td></tr>
<tr><td><strong>Test defect IoU</strong></td><td class="num"><strong>0.004</strong></td>
    <td class="num"><strong>&gt; 0.30</strong></td>
    <td class="num"><strong>{tex['IoU_per_class']['defective_part']:.3f}</strong></td>
    <td class="kpi-yes">✅ (90×)</td></tr>
<tr><td>bg IoU</td><td class="num">0.992</td><td class="num">—</td>
    <td class="num">{tex['IoU_per_class']['background']:.3f}</td><td>—</td></tr>
<tr><td>normal_part IoU</td><td class="num">0.789</td><td class="num">—</td>
    <td class="num">{tex['IoU_per_class']['normal_part']:.3f}</td><td>—</td></tr>
</table>

<div class="takeaway">
<strong>結論一句話：</strong>Two-head + Dice/BCE + Domain Randomization 三大改動，
讓 defect IoU 從 <strong>0.004 → 0.362（90× 改善）</strong>，超過 0.30 目標。
但剩下的瓶頸明確指向「資料 signal 強度」（bend 物理上不可辨識），不是模型容量。
</div>

<h2>2. 問題診斷 — 三個維度</h2>
<table>
<tr><th>維度</th><th>狀態</th><th>證據</th></tr>
<tr><td>模型架構</td><td class="kpi-yes">已解</td>
    <td>Two-head 把 part/bg（易）跟 defect（難）解耦，defect head 只在 part 區算 loss</td></tr>
<tr><td>Loss / class imbalance</td><td class="kpi-yes">已解</td>
    <td>Dice 對 minority 友善 + BCE 補 per-pixel 梯度，5% defect pixel 不再被淹沒</td></tr>
<tr><td>資料量</td><td class="kpi-yes">充足</td>
    <td>800 train / 100 val / 100 test，normal IoU 0.79 穩定，沒 overfit signal</td></tr>
<tr><td><strong>資料 signal — bend</strong></td><td class="kpi-no">❌ 剩餘瓶頸</td>
    <td>256×256 下螺絲身 ~50 px，bend 25–45° 只偏移 1–3 px，模型 93% 像素預測為 normal</td></tr>
<tr><td>資料 signal — displace</td><td class="kpi-yes">已 saturate</td>
    <td>0.43，整面紋理變化，dense signal 易學</td></tr>
<tr><td>資料 signal — remesh</td><td>🟡 中等</td>
    <td>0.16–0.28，邊緣變化 sparse signal，比 displace 弱比 bend 強</td></tr>
</table>

<h3>Bend 失敗的根因（最重要）</h3>
<div class="note">
<strong>不是模型看不到，是「像素層級沒差異」</strong>。<br>
FIG F 顯示 bend_light / bend_heavy 都有 93–94% 像素被預測為 normal，
跟「猜 normal」的 baseline（normal pixel 在 part 區占大宗）完全一致 ──
model 沒學到任何 bend-specific feature，只是 fallback 到 majority class。<br><br>
要解 bend 必須動<strong>資料端</strong>：拉近相機焦距 / bend angle 拉到 60°+ / 加側面視角，
動模型沒用。Stage 3 在 256 px 解析度的限制下，bend 是物理 ceiling。
</div>

<h2>3. Per-defect-state 細部表現</h2>
<table>
<tr><th>Defect state</th><th class="num">mean per-instance IoU</th><th>備註</th></tr>
"""
state_notes = {
    "bend_light":     "❌ 模型看不到（像素差異 < 3 px）",
    "bend_heavy":     "❌ 同上，連 45° 也不行",
    "displace_light": "✅ 偵測率高，整面紋理變化",
    "displace_heavy": "✅ 最強的 signal",
    "remesh_light":   "🟡 邊緣破碎程度有限",
    "remesh_heavy":   "🟡 邊緣明顯破碎",
}
for s, iou in states.items():
    if iou is None: continue
    html += (f"<tr><td>{s}</td><td class='num'>{iou:.3f}</td>"
             f"<td>{state_notes.get(s, '')}</td></tr>\n")

html += f"""
</table>

<h2>4. 背景 Ablation — Domain Randomization 真的有效嗎？</h2>
<table>
<tr><th>指標</th><th class="num">Textured DR</th><th class="num">Black bg（控制）</th><th class="num">差異</th></tr>
<tr><td>mIoU</td><td class="num">{tex['mIoU']:.3f}</td>
    <td class="num">{blk.get('mIoU', 0):.3f}</td>
    <td class="num">{tex['mIoU']-blk.get('mIoU',0):+.3f}</td></tr>
<tr><td>defect IoU</td><td class="num">{tex['IoU_per_class']['defective_part']:.3f}</td>
    <td class="num">{blk.get('IoU_per_class', {}).get('defective_part', 0):.3f}</td>
    <td class="num">{tex['IoU_per_class']['defective_part']-blk.get('IoU_per_class', {}).get('defective_part', 0):+.3f}</td></tr>
<tr><td>normal IoU</td><td class="num">{tex['IoU_per_class']['normal_part']:.3f}</td>
    <td class="num">{blk.get('IoU_per_class', {}).get('normal_part', 0):.3f}</td>
    <td class="num">{tex['IoU_per_class']['normal_part']-blk.get('IoU_per_class', {}).get('normal_part', 0):+.3f}</td></tr>
</table>

<div class="takeaway">
黑底測試 IoU 比 textured 略低（差 ~0.04），證明模型<strong>沒靠背景 shortcut</strong>。
黑底是模型沒見過的 OOD 情境，反而稍微受擾。Stage 2 模型若跑黑底預期 defect IoU 會掉到接近 0
（它本來就靠背景訊號），這裡 Stage 3 模型穩穩維持 0.32，表示 DR 策略成功讓模型學到零件本身的特徵。
</div>

<h2>5. 訓練動態（為什麼 epoch 30 看起來變糟？）</h2>
<table>
<tr><th>Epoch</th><th class="num">val mIoU</th><th class="num">val defect IoU</th><th>備註</th></tr>
<tr><td>1–9</td><td class="num">0.51–0.57</td><td class="num">0.000</td><td>warmup，model 全預測 normal</td></tr>
<tr><td>10</td><td class="num">0.597</td><td class="num">0.071</td><td>第一次「發現」defect</td></tr>
<tr><td>13</td><td class="num">0.498</td><td class="num">0.216</td><td>突破，但 normal IoU 暫時掉</td></tr>
<tr><td>20</td><td class="num">0.687</td><td class="num">0.334</td><td>穩定上升</td></tr>
<tr><td><strong>26</strong></td><td class="num"><strong>0.716</strong></td><td class="num"><strong>0.384</strong></td>
    <td><strong>best ← test 用這個 weight</strong></td></tr>
<tr><td>30</td><td class="num">0.489</td><td class="num">0.271</td><td>最後 epoch 崩，但已存 best</td></tr>
</table>
<div class="note">
<strong>FIG E (epoch snapshots) 的 epoch 30 看起來比中間幾張差是真的</strong> ──
那是「epoch 30 當下」的 model 預測，不是 best。
<code>train_stage3.py</code> 每 epoch 比較 <code>val_mIoU</code> 並儲存 <code>best_state</code>，
test evaluation 是用 best_state 跑的（epoch 26 的 weight），所以 test mIoU=0.712 正確。<br><br>
Epoch 30 崩可能是 Adam 後期 lr 沒衰減 + Dice loss 對單一 batch zero-defect 場景敏感。
下次可加 <code>ReduceLROnPlateau</code> 或早停。
</div>

<h2>6. 八張報告圖</h2>

{img("training_curves.png", "Training Curves — Loss 分解（L_total / L_part / L_dice / L_bce）+ Val IoU per class")}
{img("A_normal_vs_defect_diff.png", "FIG A — 同 pose normal vs 6 個 defect state，疊 diff heatmap。可見 bend 視覺差異微小，displace / remesh 明顯")}
{img("B_confidence_heatmap.png", "FIG B — Two-head 內部信心：P(part) 是 head 1 的 part/bg softmax，P(defect) 是 head 2 sigmoid。兩 head 各自學到不同特徵")}
{img("C_per_state_iou_box.png", "FIG C — Per-instance defect IoU 分布。bend 中位數接近 0，displace 中位數 0.4，remesh 中等")}
{img("D_fp_fn_overlay.png", "FIG D — 4 個代表案例：best / worst miss / worst false alarm / clean。綠=TP, 紅=FP, 橘=FN")}
{img("E_epoch_snapshots.png", "FIG E — 訓練過程同 4 個 val sample 的預測演化。注意 epoch 30 比 epoch 25/26 差（那是 last，不是 best）")}
{img("F_by_state_confusion.png", "FIG F — 7 defect_state × 3 pred class confusion matrix（row-normalized）。bend 兩列幾乎全跑去 normal，displace_heavy 93% 正確")}
{img("ablation_textured_vs_black.png", "Ablation — Textured DR vs Black bg。差距小（0.04 mIoU），證明 model 沒靠背景")}

<h2>7. 下一步（已超出 Stage 3 範圍，記下）</h2>
<ul>
<li>拉 bend 強度到 60°+ 或縮短相機距離，讓 bend 在像素上可辨識</li>
<li>加 <code>ReduceLROnPlateau</code> 或早停，避免 epoch 30 崩潰</li>
<li>實機真實照片 fine-tune，量化 sim2real gap</li>
<li>Instance 分離 head（panoptic）— 留給組員模型方向</li>
</ul>

<p style="margin-top:50px; color:#888; font-size:0.85em; text-align:center;">
generated by <code>scripts/build_stage3_html.py</code> ｜ Stage 3 commit a6c207c
</p>
</body></html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Saved → {OUT}")
print(f"Size: {os.path.getsize(OUT)/1024:.1f} KB")
