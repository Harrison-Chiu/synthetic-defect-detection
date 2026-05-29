"""
cli.py — 專案統一入口(argparse 子指令)

紀律(整理規則 §4):
  - **邏輯只在 src/**,本檔只負責解析參數 + 組裝呼叫。
  - **knob 進 configs/ 不進 CLI flag**:旗標僅限「操作性」參數(tag / 路徑 / 數量 /
    device);模型 / 生成的數值旋鈕來自 src.*.config 的 dataclass 預設
    (日後序列化成 configs/*.json 由 --config 載入)。

子指令:
  freeze-eval   生成並凍結 eval set(可比、互斥於訓練)
  samples       生成 ~N 張報告配圖場景
  train         step-based 訓練,產出寫 output/runs/<tag>/
  eval          載入 run 的 best.pt,在 eval set 上評估
  render-patches  零件 patch 預渲(Blender 側,見 src/render — 尚未抽出)
  report        產生 HTML 報告(暫沿用 scripts/)

範例:
  python cli.py freeze-eval --n 100
  python cli.py samples --n 10
  python cli.py train --tag s4_repro
  python cli.py eval --tag s4_repro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_SET = REPO_ROOT / "output" / "eval_set"
DEFAULT_SAMPLES = REPO_ROOT / "output" / "samples"
DEFAULT_RUNS = REPO_ROOT / "output" / "runs"


def _device(arg: str):
    import torch
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


# ── freeze-eval ────────────────────────────────────────────
def cmd_freeze_eval(args):
    from src.data import load_assets
    from src.data.config import DEFAULT_CONFIG, EVAL_INDEX_OFFSET
    from src.data.generator import freeze_scene_set

    assets = load_assets()
    ids = freeze_scene_set(args.out, args.n, assets, DEFAULT_CONFIG, index_offset=EVAL_INDEX_OFFSET)
    print(f"凍結 {len(ids)} 張 eval 場景 → {args.out}(index offset={EVAL_INDEX_OFFSET})")


# ── samples ────────────────────────────────────────────────
def cmd_samples(args):
    from src.data import load_assets
    from src.data.config import DEFAULT_CONFIG, SAMPLE_INDEX_OFFSET
    from src.data.generator import freeze_scene_set

    assets = load_assets()
    ids = freeze_scene_set(args.out, args.n, assets, DEFAULT_CONFIG, index_offset=SAMPLE_INDEX_OFFSET)
    print(f"產生 {len(ids)} 張報告配圖 → {args.out}(index offset={SAMPLE_INDEX_OFFSET})")


# ── train ──────────────────────────────────────────────────
def cmd_train(args):
    import torch
    from torch.utils.data import DataLoader

    from src.data import RuntimeSceneDataset, EvalSetDataset, load_assets
    from src.data.config import DEFAULT_CONFIG
    from src.models import DefectSegNet
    from src.train import DEFAULT_TRAIN_CONFIG, train_loop

    eval_dir = Path(args.eval_set)
    if not eval_dir.exists():
        sys.exit(f"找不到 eval set:{eval_dir}\n請先跑 `python cli.py freeze-eval`")

    device = _device(args.device)
    gen_cfg = DEFAULT_CONFIG
    train_cfg = DEFAULT_TRAIN_CONFIG
    run_dir = Path(args.runs) / args.tag

    assets = load_assets()
    train_len = train_cfg.total_steps * train_cfg.batch_size
    train_ds = RuntimeSceneDataset(length=train_len, assets=assets, config=gen_cfg, index_offset=0)
    eval_ds = EvalSetDataset(eval_dir)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=False,
                              num_workers=args.workers)
    eval_loader = DataLoader(eval_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0)

    # 固定快照批:取 eval set 前幾張
    n_snap = min(4, len(eval_ds))
    snap_batch = torch.stack([eval_ds[i][0] for i in range(n_snap)]) if n_snap else None

    model = DefectSegNet(base_c=train_cfg.base_c)
    print(f"train tag={args.tag} device={device} run_dir={run_dir}")
    _, best = train_loop(model, train_loader, eval_loader, device, run_dir, train_cfg, snap_batch)
    print(f"best val mIoU={best['best_val_miou']:.3f}  test mIoU={best['test_metrics']['mIoU']:.3f}")


# ── eval ───────────────────────────────────────────────────
def cmd_eval(args):
    import torch
    from torch.utils.data import DataLoader

    from src.data import EvalSetDataset
    from src.models import DefectSegNet
    from src.eval import evaluate_3class, evaluate_per_state
    from src import schema

    device = _device(args.device)
    ckpt_path = Path(args.runs) / args.tag / "best.pt"
    if not ckpt_path.exists():
        sys.exit(f"找不到 checkpoint:{ckpt_path}")
    eval_dir = Path(args.eval_set)
    if not eval_dir.exists():
        sys.exit(f"找不到 eval set:{eval_dir}")

    ckpt = torch.load(ckpt_path, map_location=device)
    base_c = ckpt.get("train_config", {}).get("base_c", 32)
    model = DefectSegNet(base_c=base_c).to(device)
    model.load_state_dict(ckpt["state_dict"])

    loader = DataLoader(EvalSetDataset(eval_dir), batch_size=8, shuffle=False, num_workers=0)
    m = evaluate_3class(model, loader, device)
    per_state = evaluate_per_state(model, loader, device)
    print(f"pixel_acc={m['pixel_acc']*100:.2f}%  mIoU={m['mIoU']:.3f}")
    for c, name in enumerate(["background", "normal_part", "defective_part"]):
        print(f"  {name:18s} IoU={m['IoU_per_class'][c]:.3f}")
    print("per-state IoU (head B):")
    for c, iou in enumerate(per_state):
        print(f"  {schema.STATE_CLASSES[c]:18s} IoU={iou:.3f}")


# ── render(Blender 側,shell out)+ gen-backgrounds(純 conda)──
def cmd_gen_backgrounds(args):
    # 純 PIL/numpy/cv2,在 conda(dl_final)直接跑
    from src.render.backgrounds import gen_backgrounds
    n = gen_backgrounds(args.out, n_per_type=args.n_per_type)
    print(f"產生 {n} 張程序背景 → {args.out}")


def cmd_render_patches(args):
    # patch 預渲是 bpy 工作,conda 無 bpy → 以 background mode 叫起 Blender 跑 src/render/patches.py
    import subprocess

    blend = Path(args.blend)
    if not blend.exists():
        sys.exit(f"找不到 blend 檔:{blend}")
    script = REPO_ROOT / "src" / "render" / "patches.py"
    cmd = [args.blender, "--background", str(blend), "--python", str(script)]
    print("執行:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit(f"找不到 Blender 執行檔 '{args.blender}'。\n"
                 f"請用 --blender 指定路徑,或在 Blender Script Editor 直接跑 {script}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Blender 渲染失敗(exit {e.returncode})")


def cmd_report(args):
    from src.eval.report import build_report

    run_dir = Path(args.runs) / args.tag
    if not run_dir.exists():
        sys.exit(f"找不到 run:{run_dir}")
    build_report(run_dir)


def build_parser():
    p = argparse.ArgumentParser(prog="cli.py", description="工業零件場景理解 — 統一入口")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("freeze-eval", help="生成並凍結 eval set")
    pe.add_argument("--n", type=int, default=100)
    pe.add_argument("--out", default=str(DEFAULT_EVAL_SET))
    pe.set_defaults(func=cmd_freeze_eval)

    ps = sub.add_parser("samples", help="生成報告配圖場景")
    ps.add_argument("--n", type=int, default=10)
    ps.add_argument("--out", default=str(DEFAULT_SAMPLES))
    ps.set_defaults(func=cmd_samples)

    pt = sub.add_parser("train", help="step-based 訓練")
    pt.add_argument("--tag", required=True, help="run 標籤(輸出到 runs/<tag>/)")
    pt.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    pt.add_argument("--runs", default=str(DEFAULT_RUNS))
    pt.add_argument("--device", default="auto")
    pt.add_argument("--workers", type=int, default=0)
    pt.set_defaults(func=cmd_train)

    pv = sub.add_parser("eval", help="在 eval set 上評估某個 run")
    pv.add_argument("--tag", required=True)
    pv.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    pv.add_argument("--runs", default=str(DEFAULT_RUNS))
    pv.add_argument("--device", default="auto")
    pv.set_defaults(func=cmd_eval)

    pr = sub.add_parser("render-patches", help="零件 patch 預渲(background mode 叫起 Blender)")
    pr.add_argument("--blender", default="blender", help="Blender 執行檔路徑")
    pr.add_argument("--blend", default=str(REPO_ROOT / "blender" / "114-2_DLcourse_FinalProject-01.blend"))
    pr.set_defaults(func=cmd_render_patches)

    pb = sub.add_parser("gen-backgrounds", help="程序化背景生成(純 conda)")
    pb.add_argument("--out", default=str(REPO_ROOT / "assets" / "backgrounds"))
    pb.add_argument("--n-per-type", type=int, default=3)
    pb.set_defaults(func=cmd_gen_backgrounds)

    prep = sub.add_parser("report", help="產生 run 的 HTML 報告(通用)")
    prep.add_argument("--tag", required=True)
    prep.add_argument("--runs", default=str(DEFAULT_RUNS))
    prep.set_defaults(func=cmd_report)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
