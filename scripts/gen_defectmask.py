"""gen_defectmask.py — 離線一次性:為每個 defect patch 產生「變形區」二值遮罩。

S5 機制(見 docs/stage5_plan.md §3)的離線前置:
    M = despeckle( mean_RGB|defect_patch − normal_patch| > thr )   # 變形區,二值

- 配準:normal 與 defect 同 pose 同原點(SIMPLE_DEFORM 繞原點彎,根固定),
  相減已正確對齊 —— **不要按質心歸零**(會把對齊的根推開、製造假 diff)。
- part 區域 = normal_alpha | defect_alpha(兩版任一有零件處),只在 part 內算 diff。
- thr 預設 30:佐證 docs/figures/route_a_diff_threshold.png(diag_diff_threshold.py)。
- despeckle:移除面積 < min_blob 的雜散小點(scipy 有就用 label,否則 fallback)。

輸出:每張 defect patch 旁 `<pose>_<state>_defectmask.png`(uint8 0/255,單通道)。
normal/ 不產(無變形區)。冪等:重跑覆蓋。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "output" / "patches"
NORMAL_DIR = PATCH / "normal"
DEFECT_STATES = [
    "bend_light", "bend_heavy",
    "displace_light", "displace_heavy",
    "remesh_light", "remesh_heavy",
]


def _load_rgb_alpha(p: Path):
    a = np.array(Image.open(p).convert("RGBA"))
    return a[..., :3].astype(np.float32), a[..., 3] > 10


def _despeckle(mask: np.ndarray, min_blob: int) -> np.ndarray:
    """移除面積 < min_blob 的連通小點。scipy 有就用,否則原樣回傳。"""
    if min_blob <= 0:
        return mask
    try:
        from scipy import ndimage
    except ImportError:
        return mask
    lbl, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
    keep = np.isin(lbl, np.nonzero(sizes >= min_blob)[0] + 1)
    return keep


def make_mask(normal_path: Path, defect_path: Path, thr: float, min_blob: int):
    nrgb, na = _load_rgb_alpha(normal_path)
    drgb, da = _load_rgb_alpha(defect_path)
    part = na | da
    diff = np.abs(drgb - nrgb).mean(axis=2)
    mask = (diff > thr) & part
    mask = _despeckle(mask, min_blob)
    frac = mask.sum() / max(1, part.sum())
    return mask, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=30.0)
    ap.add_argument("--min-blob", type=int, default=8, help="despeckle 最小連通面積(px)")
    ap.add_argument("--dry-run", action="store_true", help="只印統計,不寫檔")
    args = ap.parse_args()

    print(f"thr={args.thr}  min_blob={args.min_blob}  dry_run={args.dry_run}\n")
    stats: dict[str, list[float]] = {}
    n_written = 0
    for state in DEFECT_STATES:
        ddir = PATCH / state
        if not ddir.is_dir():
            print(f"[skip] {ddir} 不存在")
            continue
        fracs = []
        for dpath in sorted(ddir.glob(f"*_{state}.png")):
            pose = dpath.name[: -(len(state) + 5)]  # 去掉 "_<state>.png"
            npath = NORMAL_DIR / f"{pose}_normal.png"
            if not npath.exists():
                print(f"  [warn] 找不到對應 normal: {npath.name}")
                continue
            mask, frac = make_mask(npath, dpath, args.thr, args.min_blob)
            fracs.append(frac)
            if not args.dry_run:
                out = ddir / f"{pose}_{state}_defectmask.png"
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(out)
                n_written += 1
        stats[state] = fracs
        if fracs:
            arr = np.array(fracs) * 100
            print(f"  {state:15s} n={len(fracs):2d}  變形區佔零件 "
                  f"mean={arr.mean():5.1f}%  min={arr.min():4.1f}%  max={arr.max():5.1f}%")
    print(f"\n寫出 {n_written} 張 defectmask" + ("(dry-run,未寫)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
