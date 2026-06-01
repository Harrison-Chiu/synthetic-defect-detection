"""gen_defectmask_s6.py — S6 defectmask 生成

S6 的 defectmask 只跟視角(az, el)和瑕疵類型有關，跟光照無關。
因此只用固定的一組光照 (h0, r000, s10) 來比較 normal vs defect，
產出 45 張 defectmask (3az × 5el × 3defect)。

每張 defectmask 存在 defect state 目錄下，檔名只含 pose 資訊：
  output/patches_s6/{state}/pan_head_az{az}_el{el}_{state}_defectmask.png

scene generator 的 defectmask_path_for() 需要對應更新。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PATCH_S6 = REPO / "output" / "patches_s6"
NORMAL_DIR = PATCH_S6 / "normal"

# S6 defect states
DEFECT_STATES = ["bend_45", "displace", "remesh"]

# 用固定光照做比較（h0, r000, s10 = strength 1.0）
REFERENCE_LIGHTING = "_h0_r000_s10_"


def _load_rgb_alpha(p: Path):
    a = np.array(Image.open(p).convert("RGBA"))
    return a[..., :3].astype(np.float32), a[..., 3] > 10


def _despeckle(mask: np.ndarray, min_blob: int) -> np.ndarray:
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
    ap.add_argument("--min-blob", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"thr={args.thr}  min_blob={args.min_blob}  dry_run={args.dry_run}")
    print(f"Reference lighting: {REFERENCE_LIGHTING}\n")

    # 找所有 reference normal patches
    normal_ref = {}  # (az, el) -> path
    for p in sorted(NORMAL_DIR.glob("*.png")):
        if REFERENCE_LIGHTING in p.name and "_defectmask" not in p.name:
            # 解析 az, el from filename
            name = p.stem  # pan_head_az000_el+00_h0_r000_s10_normal
            parts = name.split("_")
            az_str = [x for x in parts if x.startswith("az")][0]
            el_str = [x for x in parts if x.startswith("el")][0]
            normal_ref[(az_str, el_str)] = p

    print(f"Found {len(normal_ref)} reference normal patches\n")

    stats: dict[str, list[float]] = {}
    n_written = 0
    for state in DEFECT_STATES:
        ddir = PATCH_S6 / state
        if not ddir.is_dir():
            print(f"[skip] {ddir} 不存在")
            continue
        fracs = []
        for dpath in sorted(ddir.glob("*.png")):
            if "_defectmask" in dpath.name:
                continue
            if REFERENCE_LIGHTING not in dpath.name:
                continue
            # 解析 pose
            name = dpath.stem
            parts = name.split("_")
            az_str = [x for x in parts if x.startswith("az")][0]
            el_str = [x for x in parts if x.startswith("el")][0]

            npath = normal_ref.get((az_str, el_str))
            if npath is None:
                print(f"  [warn] no normal ref for ({az_str}, {el_str})")
                continue

            mask, frac = make_mask(npath, dpath, args.thr, args.min_blob)
            fracs.append(frac)
            if not args.dry_run:
                # 存為 pose-only 檔名，所有同 pose 的 lighting 變體都指向這張
                out = ddir / f"pan_head_{az_str}_{el_str}_{state}_defectmask.png"
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(out)
                n_written += 1

        stats[state] = fracs
        if fracs:
            arr = np.array(fracs) * 100
            print(f"  {state:15s} n={len(fracs):2d}  變形區佔零件 "
                  f"mean={arr.mean():5.1f}%  min={arr.min():4.1f}%  max={arr.max():5.1f}%")

    print(f"\n寫出 {n_written} 張 defectmask" + ("(dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
