"""
report.py — 通用訓練報告(吃 runs/<tag>/ → 自帶資源 HTML,含視覺化)

讀 run dir 的 history.json + best.pt,**自行生成一批 test 場景**跑模型,產出:
  1. KPI 表(對 S3 baseline 比較)
  2. 訓練曲線(train+val loss、per-class val IoU)
  3. 預測面板  RGB | GT | Pred | P(defect)(挑有瑕疵的場景)
  4. FP/FN overlay 代表案例(best / worst-miss / worst-FA / clean)
  5. per-state × predicted class 混淆矩陣(列正規化)
  6. per-instance defect IoU by state(箱形圖)
  7. per-state 偵測率表 + 最終指標表 + TrainConfig

圖片一律 base64 內嵌 → 單檔 HTML。圖表邏輯移植自 scripts/eval_stage3.py。
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np

# 0=bg(黑) / 1=normal(綠) / 2=defect(紅)
_CMAP3 = np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]], dtype=np.uint8)
_CLASS_NAMES = ["background", "normal_part", "defective_part"]
# S3 baseline(docs/history/stage3_results.md),報告對照用
_S3 = {"mIoU": 0.712, "pixel_acc": 0.941, "defect_IoU": 0.362,
       "bend_light_det": 0.034, "bend_heavy_det": 0.047,
       "displace_heavy_det": 0.932, "remesh_heavy_det": 0.523}


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=95, bbox_inches="tight")
    buf.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


# ── 跑模型 → cache ─────────────────────────────────────────
def _build_cache(model, device, n: int, defect_thr: float):
    import torch
    from src.data import encode_rgb, encode_targets, generate_scene, load_assets
    from src.data.config import DEFAULT_CONFIG, TEST_INDEX_OFFSET
    from src.eval.metrics import _DEFECT_KEY, infer_3class

    assets = load_assets()
    cache = []
    model.eval()
    with torch.no_grad():
        for k in range(n):
            s = generate_scene(TEST_INDEX_OFFSET + k, assets, DEFAULT_CONFIG)
            rgb_t = encode_rgb(s.rgb).unsqueeze(0).to(device)
            out = model(rgb_t)
            pred, ppart, pdef = infer_3class(out, defect_thr)
            tg = encode_targets(s.semantic, s.instance, s.meta, s.defect_region)
            dl = out[_DEFECT_KEY]
            if dl.dim() == 4:
                dl = dl.squeeze(1)
            cache.append({
                "rgb": s.rgb,
                "gt": s.semantic.astype(np.uint8),
                "pred": pred[0].cpu().numpy().astype(np.uint8),
                "p_part": ppart[0].cpu().numpy(),
                "p_def": pdef[0].cpu().numpy(),
                "def_logit": dl[0].cpu().numpy(),       # defect 頭原始 logit(算 per-type BCE)
                "T": tg["B_T"].numpy(),                 # defect target(變形區)
                "W": tg["B_W"].numpy(),                 # defect 權重
                "instance": s.instance,
                "meta": s.meta,
            })
    return cache


def _metrics_from_cache(cache):
    inter = np.zeros(3); union = np.zeros(3); correct = total = 0
    for c in cache:
        p, m = c["pred"], c["gt"]
        for k in range(3):
            pk, mk = (p == k), (m == k)
            inter[k] += int((pk & mk).sum()); union[k] += int((pk | mk).sum())
        correct += int((p == m).sum()); total += int(m.size)
    ious = [inter[k] / union[k] if union[k] > 0 else float("nan") for k in range(3)]
    return {"pixel_acc": correct / total, "mIoU": float(np.nanmean(ious)), "IoU": ious}


# ── 圖 ─────────────────────────────────────────────────────
def _fig_curves(history):
    if not history.get("step"):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = history["step"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(s, history["L_total"], "-o", label="train", color="#d4a017")
    if history.get("val_L_total"):
        ax[0].plot(s, history["val_L_total"], "-o", label="val", color="#8e44ad")
    ax[0].set_title("loss (train vs val)"); ax[0].set_xlabel("step"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(s, history["val_mIoU"], "-o", lw=2, label="mIoU", color="#2a4d8f")
    ax[1].plot(s, history["val_IoU_bg"], "--", label="bg", color="#888")
    ax[1].plot(s, history["val_IoU_normal"], "--", label="normal", color="#198754")
    ax[1].plot(s, history["val_IoU_defect"], "-o", label="defect", color="#c0392b")
    ax[1].set_title("val IoU per class"); ax[1].set_xlabel("step"); ax[1].set_ylim(-.02, 1.02)
    ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(s, history["lr"], "-o", color="#198754"); ax[2].set_yscale("log")
    ax[2].set_title("learning rate"); ax[2].set_xlabel("step"); ax[2].grid(alpha=.3)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _fig_pred_panel(cache, k=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = [c for c in cache if (c["gt"] == 2).sum() > 150][:k]
    if not sel:
        return None
    fig, axes = plt.subplots(len(sel), 4, figsize=(14, 3.4 * len(sel)))
    if len(sel) == 1:
        axes = axes[None, :]
    for r, c in enumerate(sel):
        axes[r, 0].imshow(c["rgb"]); axes[r, 0].set_title("RGB", fontsize=9)
        axes[r, 1].imshow(_CMAP3[c["gt"]]); axes[r, 1].set_title("GT (標 type)", fontsize=9)
        _label_instances(axes[r, 1], c)
        axes[r, 2].imshow(_CMAP3[c["pred"]]); axes[r, 2].set_title("Pred", fontsize=9)
        im = axes[r, 3].imshow(c["p_def"], cmap="hot", vmin=0, vmax=1)
        axes[r, 3].set_title("P(defect)", fontsize=9)
        plt.colorbar(im, ax=axes[r, 3], fraction=0.046, pad=0.04)
        for j in range(4):
            axes[r, j].axis("off")
    plt.suptitle("預測面板:RGB | GT | Pred | P(defect)  (綠=normal 紅=defect)", fontsize=11)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _fig_fpfn(cache):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def overlay(rgb, gt, pred):
        out = rgb.astype(float).copy()
        gd, pd = (gt == 2), (pred == 2)
        tp, fp, fn = gd & pd, ~gd & pd, gd & ~pd
        out[tp] = out[tp] * 0.3 + np.array([0, 220, 0]) * 0.7
        out[fp] = out[fp] * 0.3 + np.array([255, 0, 0]) * 0.7
        out[fn] = out[fn] * 0.3 + np.array([255, 160, 0]) * 0.7
        return out.clip(0, 255).astype(np.uint8)

    stats = []
    for i, c in enumerate(cache):
        gd, pd = (c["gt"] == 2), (c["pred"] == 2)
        stats.append({"i": i, "tp": int((gd & pd).sum()), "fp": int((~gd & pd).sum()),
                      "fn": int((gd & ~pd).sum()), "n_gt": int(gd.sum())})
    with_def = [s for s in stats if s["n_gt"] > 0]
    if not with_def:
        return None
    clean = [s for s in stats if s["n_gt"] == 0]
    cases = [("Best detection", max(with_def, key=lambda s: s["tp"])),
             ("Worst miss (FN)", max(with_def, key=lambda s: s["fn"])),
             ("Worst false alarm (FP)", max(stats, key=lambda s: s["fp"])),
             ("Clean (no GT defect)", min(clean, key=lambda s: s["fp"]) if clean else stats[0])]
    fig, axes = plt.subplots(len(cases), 4, figsize=(15, 3.4 * len(cases)))
    for r, (title, s) in enumerate(cases):
        c = cache[s["i"]]
        sub = f"TP={s['tp']} FP={s['fp']} FN={s['fn']}"
        axes[r, 0].imshow(c["rgb"]); axes[r, 0].set_title(title, fontsize=9)
        axes[r, 1].imshow(_CMAP3[c["gt"]]); axes[r, 1].set_title("GT (標 type)", fontsize=9)
        _label_instances(axes[r, 1], c)
        axes[r, 2].imshow(_CMAP3[c["pred"]]); axes[r, 2].set_title("Pred " + sub, fontsize=8)
        axes[r, 3].imshow(overlay(c["rgb"], c["gt"], c["pred"]))
        axes[r, 3].set_title("綠=TP 紅=FP 橘=FN", fontsize=9)
        for j in range(4):
            axes[r, j].axis("off")
    plt.suptitle("FP / FN overlay 代表案例", fontsize=11)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _fig_confusion(cache):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src import schema
    states = schema.STATE_CLASSES
    mat = np.zeros((len(states), 3))
    for c in cache:
        for ins in c["meta"]["instances"]:
            ri = schema.STATE_TO_IDX[ins["defect_state"]]
            pix = c["instance"] == ins["instance_id"]
            for cp in range(3):
                mat[ri, cp] += int(((c["pred"] == cp) & pix).sum())
    norm = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(_CLASS_NAMES, rotation=15)
    ax.set_yticks(range(len(states))); ax.set_yticklabels(states)
    ax.set_xlabel("Predicted"); ax.set_ylabel("GT defect_state")
    for i in range(len(states)):
        for j in range(3):
            ax.text(j, i, f"{norm[i, j]*100:.1f}%", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_title("per-state × predicted(列正規化)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _fig_inst_box(cache):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src import schema
    states = [s for s in schema.STATE_CLASSES if s != "normal"]
    by = {s: [] for s in states}
    for c in cache:
        pd = c["pred"] == 2
        for ins in c["meta"]["instances"]:
            if not ins["is_defective"]:
                continue
            st = ins["defect_state"]
            pix = c["instance"] == ins["instance_id"]
            if pix.sum() < 50:
                continue
            inter = int((pd & pix).sum()); union = int((pd | pix).sum())
            if union > 0:
                by[st].append(inter / union)
    data = [by[s] for s in states]
    labels = [f"{s}\n(n={len(by[s])})" for s in states]
    if not any(data):
        return None, by
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("per-instance defect IoU"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("per-instance defect IoU by GT state"); ax.grid(axis="y", alpha=.3)
    plt.xticks(fontsize=8); plt.tight_layout()
    return _fig_to_b64(fig), by


_TYPES = ("bend", "displace", "remesh")


def _per_type_stats(cache):
    """三種瑕疵 type(bend/displace/remesh)各自:偵測率、per-instance IoU、defect 頭 BCE。

    - det_rate:該 type 零件像素被判 defect 的比例。
    - inst_iou:該 type 每顆瑕疵零件的 defect IoU(pred==2 vs 整顆),取平均。
    - bce:defect 頭在「該 type 零件所屬像素」上的加權 BCE(W·BCE / ΣW),反映 loss 貢獻。
    """
    from src import schema
    agg = {t: {"det": [0, 0], "iou": [], "bce_num": 0.0, "bce_den": 0.0} for t in _TYPES}
    for c in cache:
        pd = c["pred"] == 2
        # 逐像素 BCE map(對 T,以 sigmoid logit)
        z = c["def_logit"]
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-6
        bce_map = -(c["T"] * np.log(p + eps) + (1 - c["T"]) * np.log(1 - p + eps))
        for ins in c["meta"]["instances"]:
            if not ins["is_defective"]:
                continue
            t = schema.state_to_type(ins["defect_state"])
            if t not in agg:
                continue
            pix = c["instance"] == ins["instance_id"]
            ps = int(pix.sum())
            if ps < 50:
                continue
            agg[t]["det"][0] += int((pd & pix).sum())
            agg[t]["det"][1] += ps
            inter = int((pd & pix).sum()); union = int((pd | pix).sum())
            if union > 0:
                agg[t]["iou"].append(inter / union)
            w = c["W"][pix]
            agg[t]["bce_num"] += float((w * bce_map[pix]).sum())
            agg[t]["bce_den"] += float(w.sum())
    out = {}
    for t in _TYPES:
        a = agg[t]
        out[t] = {
            "det_rate": a["det"][0] / a["det"][1] if a["det"][1] else float("nan"),
            "inst_iou": float(np.mean(a["iou"])) if a["iou"] else float("nan"),
            "bce": a["bce_num"] / a["bce_den"] if a["bce_den"] else float("nan"),
            "n_inst": len(a["iou"]),
        }
    return out


def _fig_per_type(pt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(_TYPES)); w = 0.27
    det = [pt[t]["det_rate"] for t in _TYPES]
    iou = [pt[t]["inst_iou"] for t in _TYPES]
    bce = [pt[t]["bce"] for t in _TYPES]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(x - w/2, det, w, label="偵測率", color="#2a4d8f")
    ax[0].bar(x + w/2, iou, w, label="inst IoU", color="#c0392b")
    ax[0].set_xticks(x); ax[0].set_xticklabels(_TYPES); ax[0].set_ylim(0, 1.02)
    ax[0].set_title("三種瑕疵:偵測率 / per-instance IoU"); ax[0].legend(); ax[0].grid(axis="y", alpha=.3)
    ax[1].bar(x, bce, color="#d4a017")
    ax[1].set_xticks(x); ax[1].set_xticklabels(_TYPES)
    ax[1].set_title("三種瑕疵:defect 頭加權 BCE(越低越好)"); ax[1].grid(axis="y", alpha=.3)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _label_instances(ax, c):
    """在每顆瑕疵零件質心標注 defect type。"""
    from src import schema
    for ins in c["meta"]["instances"]:
        if not ins["is_defective"]:
            continue
        ys, xs = np.where(c["instance"] == ins["instance_id"])
        if len(xs) == 0:
            continue
        t = schema.state_to_type(ins["defect_state"])
        ax.text(xs.mean(), ys.mean(), t, color="yellow", fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, ec="none"))


# ── 主入口 ─────────────────────────────────────────────────
def build_report(run_dir: str | Path, n_scenes: int = 100) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import torch
    from src.models import DefectSegNet, load_state_dict_flexible

    run_dir = Path(run_dir)
    tag = run_dir.name
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8")) \
        if (run_dir / "history.json").exists() else {}
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False) \
        if (run_dir / "best.pt").exists() else {}
    train_cfg = ckpt.get("train_config", {})
    best_miou = ckpt.get("best_val_miou", float("nan"))
    split_name = ckpt.get("test_split", "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_c = train_cfg.get("base_c", 32)
    depth = train_cfg.get("depth", 4)
    defect_thr = train_cfg.get("defect_thr", 0.5)
    model = DefectSegNet(base_c=base_c, depth=depth).to(device)
    if ckpt.get("state_dict"):
        load_state_dict_flexible(model, ckpt["state_dict"])
    print(f"[report] {tag}: 生成 {n_scenes} 張 test 場景跑模型 (base_c={base_c} depth={depth}) ...")
    cache = _build_cache(model, device, n_scenes, defect_thr)
    m = _metrics_from_cache(cache)

    # 圖
    curves = _fig_curves(history)
    panel = _fig_pred_panel(cache)
    fpfn = _fig_fpfn(cache)
    conf = _fig_confusion(cache)
    box, by_state = _fig_inst_box(cache)
    pt = _per_type_stats(cache)
    pt_fig = _fig_per_type(pt)

    def img(b64, cap):
        return (f'<figure><img src="data:image/png;base64,{b64}"/><figcaption>{cap}</figcaption></figure>'
                if b64 else f'<div class="note">{cap}:無資料</div>')

    # KPI 表(對 S3)
    kpi = "".join(
        f"<tr><td>{k}</td><td class='num'>{s3}</td><td class='num'><b>{v}</b></td></tr>"
        for k, s3, v in [
            ("mIoU", f"{_S3['mIoU']:.3f}", f"{m['mIoU']:.3f}"),
            ("pixel_acc", f"{_S3['pixel_acc']*100:.1f}%", f"{m['pixel_acc']*100:.2f}%"),
            ("defect IoU", f"{_S3['defect_IoU']:.3f}", f"{m['IoU'][2]:.3f}"),
            ("bg IoU", "0.983", f"{m['IoU'][0]:.3f}"),
            ("normal IoU", "0.790", f"{m['IoU'][1]:.3f}"),
        ])

    # 三種瑕疵 type 表(bend/displace/remesh)
    pt_rows = "".join(
        f"<tr><td>{t}</td><td class='num'>{pt[t]['det_rate']*100:.1f}%</td>"
        f"<td class='num'>{pt[t]['inst_iou']:.3f}</td><td class='num'>{pt[t]['bce']:.3f}</td>"
        f"<td class='num'>{pt[t]['n_inst']}</td></tr>" for t in _TYPES)

    # per-state 偵測率(從 cache 算)
    from src import schema
    ps_rows = ""
    pscount = {s: [0, 0] for s in schema.STATE_CLASSES}  # [defect_pred_px, total_px]
    for c in cache:
        for ins in c["meta"]["instances"]:
            st = ins["defect_state"]
            pix = c["instance"] == ins["instance_id"]
            pscount[st][0] += int(((c["pred"] == 2) & pix).sum())
            pscount[st][1] += int(pix.sum())
    for st in schema.STATE_CLASSES:
        dp, tp = pscount[st]
        det = dp / tp if tp else float("nan")
        ms = by_state.get(st)
        iou = f"{np.mean(ms):.3f}" if ms else "—"
        tag_fp = " (誤報率)" if st == "normal" else ""
        ps_rows += (f"<tr><td>{st}{tag_fp}</td><td class='num'>{det*100:.1f}%</td>"
                    f"<td class='num'>{iou}</td><td class='num'>{tp}</td></tr>")

    cfg_rows = "".join(f"<tr><td>{k}</td><td class='num'>{v}</td></tr>"
                       for k, v in train_cfg.items()) or \
        "<tr><td colspan=2 class='note'>無 train_config</td></tr>"

    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>Run Report — {tag}</title><style>
 body{{font-family:"Segoe UI","Microsoft JhengHei",sans-serif;max-width:1080px;margin:30px auto;padding:0 20px;color:#222;line-height:1.55;}}
 h1{{border-bottom:3px solid #2a4d8f;padding-bottom:8px;}} h2{{color:#2a4d8f;border-left:5px solid #2a4d8f;padding-left:10px;margin-top:32px;}}
 table{{border-collapse:collapse;margin:12px 0;}} th,td{{border:1px solid #ccc;padding:6px 14px;text-align:left;}}
 th{{background:#f0f4fa;}} td.num{{text-align:right;font-variant-numeric:tabular-nums;}}
 figure{{margin:18px 0;padding:10px;background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;}}
 figure img{{max-width:100%;display:block;margin:0 auto;}} figcaption{{font-size:.9em;color:#555;text-align:center;margin-top:6px;}}
 .note{{background:#eef6f9;border-left:5px solid #5b9bd5;padding:8px 14px;margin:10px 0;font-size:.93em;}}
</style></head><body>
<h1>Run Report — <code>{tag}</code></h1>
<p class="note">由 <code>cli.py report</code> 自動產生({split_name} set,{n_scenes} 張)。
 best val mIoU=<b>{best_miou:.3f}</b>｜base_c={base_c}｜depth={depth}｜defect_thr={defect_thr}</p>

<h2>1. KPI(對 S3 baseline)</h2>
<table><tr><th>指標</th><th class="num">S3</th><th class="num">本 run</th></tr>{kpi}</table>

<h2>2. 三種瑕疵各自表現(bend / displace / remesh)</h2>
<table><tr><th>defect type</th><th class="num">偵測率</th><th class="num">inst IoU</th><th class="num">defect BCE</th><th class="num">n_inst</th></tr>{pt_rows}</table>
{img(pt_fig, "三種瑕疵:偵測率/IoU(左)、defect 頭加權 BCE(右)")}
<h2>3. 訓練曲線</h2>{img(curves, "train/val loss、per-class val IoU、lr")}
<h2>4. 預測面板</h2>{img(panel, "RGB | GT(標 type) | Pred | P(defect)")}
<h2>5. FP / FN 案例</h2>{img(fpfn, "綠=TP 紅=FP(誤報) 橘=FN(漏抓);GT 欄標 type")}
<h2>6. per-state 混淆矩陣</h2>{img(conf, "每種 defect_state 的零件像素被判成哪一類")}
<h2>7. per-instance defect IoU</h2>{img(box, "每顆瑕疵零件的 defect IoU 分布(按 state)")}

<h2>8. per-state 偵測率表</h2>
<table><tr><th>defect_state</th><th class="num">偵測率</th><th class="num">inst IoU</th><th class="num">n_px</th></tr>{ps_rows}</table>

<h2>9. 訓練設定(TrainConfig)</h2><table><tr><th>參數</th><th class="num">值</th></tr>{cfg_rows}</table>
</body></html>"""

    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}  ({out.stat().st_size/1024:.1f} KB)")
    return out


if __name__ == "__main__":
    import sys
    build_report(sys.argv[1])
