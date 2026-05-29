"""
report.py — 通用訓練報告產生器(吃 runs/<tag>/ → 自帶資源 HTML)

定位:這是**通用**報告(任何 run 都能套),不是 S4 那種手工敘事報告
(scripts/build_stage4_html.py 是一次性,保留)。讀 run dir 的 history.json + best.pt
+ snapshots/,自動產出:訓練曲線、最終指標表、per-state IoU、內嵌預測快照。

圖片一律 base64 內嵌 → 單檔 HTML,瀏覽器/遠端直接開。
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from src import schema


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _file_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _curves_b64(history: dict) -> str | None:
    if not history.get("step"):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = history["step"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, history["val_mIoU"], "-o", label="mIoU", color="#2a4d8f")
    axes[0].plot(steps, history["val_IoU_defect"], "-o", label="defect IoU", color="#c0392b")
    axes[0].set_title("val mIoU / defect IoU"); axes[0].set_xlabel("step"); axes[0].legend()
    axes[1].plot(steps, history["L_total"], "-o", color="#d4a017")
    axes[1].set_title("train loss (L_total)"); axes[1].set_xlabel("step")
    axes[2].plot(steps, history["lr"], "-o", color="#198754")
    axes[2].set_title("learning rate"); axes[2].set_xlabel("step"); axes[2].set_yscale("log")
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def build_report(run_dir: str | Path) -> Path:
    """產生 run_dir/report.html,回傳路徑。"""
    run_dir = Path(run_dir)
    tag = run_dir.name
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8")) \
        if (run_dir / "history.json").exists() else {}

    import torch
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu") if (run_dir / "best.pt").exists() else {}
    test = ckpt.get("test_metrics", {})
    per_state = ckpt.get("per_state_iou", [])
    train_cfg = ckpt.get("train_config", {})
    best_miou = ckpt.get("best_val_miou", float("nan"))

    # 區塊
    curves = _curves_b64(history)
    curves_html = (f'<figure><img src="data:image/png;base64,{curves}"/>'
                   f'<figcaption>圖 1：訓練曲線(val mIoU/defect IoU、loss、lr)</figcaption></figure>'
                   if curves else '<div class="note">無 history,略過訓練曲線</div>')

    iou = test.get("IoU_per_class", [None, None, None])
    metrics_rows = "".join(
        f"<tr><td>{k}</td><td class='num'>{v}</td></tr>" for k, v in [
            ("best val mIoU", f"{best_miou:.3f}"),
            ("test pixel_acc", f"{test.get('pixel_acc', float('nan'))*100:.2f}%" if test else "—"),
            ("test mIoU", f"{test.get('mIoU', float('nan')):.3f}" if test else "—"),
            ("test IoU bg", f"{iou[0]:.3f}" if iou[0] is not None else "—"),
            ("test IoU normal_part", f"{iou[1]:.3f}" if iou[1] is not None else "—"),
            ("test IoU defect_part", f"{iou[2]:.3f}" if iou[2] is not None else "—"),
        ])

    per_state_rows = "".join(
        f"<tr><td>{schema.STATE_CLASSES[i]}</td><td class='num'>{v:.3f}</td></tr>"
        for i, v in enumerate(per_state)) if per_state else \
        "<tr><td colspan=2 class='note'>無 per-state IoU</td></tr>"

    cfg_rows = "".join(f"<tr><td>{k}</td><td class='num'>{v}</td></tr>"
                       for k, v in train_cfg.items()) or \
        "<tr><td colspan=2 class='note'>無 train_config</td></tr>"

    # snapshots(取最後 2 張)
    snaps = sorted((run_dir / "snapshots").glob("step_*.png")) if (run_dir / "snapshots").exists() else []
    snap_html = "".join(
        f'<figure><img src="data:image/png;base64,{_file_to_b64(p)}"/>'
        f'<figcaption>{p.stem}</figcaption></figure>' for p in snaps[-2:]) or \
        '<div class="note">無快照</div>'

    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>Run Report — {tag}</title><style>
 body{{font-family:"Segoe UI","Microsoft JhengHei",sans-serif;max-width:1000px;margin:30px auto;padding:0 20px;color:#222;line-height:1.55;}}
 h1{{border-bottom:3px solid #2a4d8f;padding-bottom:8px;}} h2{{color:#2a4d8f;border-left:5px solid #2a4d8f;padding-left:10px;margin-top:32px;}}
 table{{border-collapse:collapse;margin:12px 0;}} th,td{{border:1px solid #ccc;padding:6px 14px;text-align:left;}}
 th{{background:#f0f4fa;}} td.num{{text-align:right;font-variant-numeric:tabular-nums;}}
 figure{{margin:18px 0;padding:10px;background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;}}
 figure img{{max-width:100%;display:block;margin:0 auto;}} figcaption{{font-size:.9em;color:#555;text-align:center;margin-top:6px;}}
 .note{{background:#eef6f9;border-left:5px solid #5b9bd5;padding:8px 14px;margin:10px 0;font-size:.93em;}}
</style></head><body>
<h1>Run Report — <code>{tag}</code></h1>
<p class="note">由 <code>cli.py report</code> 自動產生(通用報告)。資料來源:<code>{run_dir}</code></p>
<h2>1. 最終指標</h2><table><tr><th>指標</th><th class="num">值</th></tr>{metrics_rows}</table>
<h2>2. 訓練曲線</h2>{curves_html}
<h2>3. Per-state IoU(head B,7 類)</h2><table><tr><th>defect_state</th><th class="num">IoU</th></tr>{per_state_rows}</table>
<h2>4. 預測快照</h2>{snap_html}
<h2>5. 訓練設定(TrainConfig)</h2><table><tr><th>參數</th><th class="num">值</th></tr>{cfg_rows}</table>
</body></html>"""

    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}  ({out.stat().st_size/1024:.1f} KB)")
    return out


if __name__ == "__main__":
    import sys
    build_report(sys.argv[1])
