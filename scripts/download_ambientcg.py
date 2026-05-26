"""
Stage 3 — 從 AmbientCG 下載 ~35 張 CC0 工業材質背景候選。

執行：
    conda activate dl_final
    python scripts/download_ambientcg.py

策略：
    - 用 ?file=<Asset>_1K-JPG.zip 直接下載 zip
    - 解壓只留 _Color.jpg（其他 normal/rough/disp 不需要）
    - 存到 assets/backgrounds_candidates/cand_<idx>_<asset_id>.jpg
    - 失敗的略過、不擋整體流程

候選清單為「工業/工作台/工地」題材，刻意避開磁磚、布料、自然石頭。
"""

import os
import io
import sys
import time
import zipfile
import urllib.request
import urllib.error

BASE_DIR = r"D:\Harrison\中山\大四下\深度學習期末報告"
OUT_DIR  = os.path.join(BASE_DIR, "assets", "backgrounds_candidates")

# 候選 asset IDs（從 AmbientCG 公開 ID 中挑「工業」題材）
# 排序大致按題材分組，下載順序對應 contact-sheet 編號
CANDIDATES = [
    # ── 混凝土 / 水泥 ──
    "Concrete001", "Concrete002", "Concrete008", "Concrete016", "Concrete017",
    "Concrete018", "Concrete020", "Concrete023", "Concrete028", "Concrete033",
    # ── 金屬板 / 鋼板 ──
    "Metal001", "Metal007", "Metal008", "Metal009", "Metal026",
    "Metal027", "Metal032",
    # ── 鏽蝕表面 ──
    "Rust001", "Rust002", "Rust004", "Rust005", "Rust006",
    # ── 木工作台 ──
    "Wood048", "Wood049", "Wood050", "Wood067",
    # ── 瀝青 / 工地地面 ──
    "Asphalt009", "Asphalt010", "Asphalt012",
    # ── 油漆牆面（工業色）──
    "PaintedPlaster001", "PaintedPlaster005",
    # ── Plastic / 工業塑膠 ──
    "Plastic001", "Plastic004",
    # ── Cardboard 紙箱（工廠常見）──
    "Cardboard002",
]

URL_TEMPLATE = "https://ambientcg.com/get?file={asset}_1K-JPG.zip"
TIMEOUT = 60  # seconds per request


def download_one(idx, asset_id):
    """下載並解出 _Color.jpg。回傳 (success, output_path or error_msg)."""
    url = URL_TEMPLATE.format(asset=asset_id)
    out_name = f"cand_{idx:02d}_{asset_id}.jpg"
    out_path = os.path.join(OUT_DIR, out_name)
    if os.path.exists(out_path):
        return True, f"(cached) {out_name}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "stage3-bg-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # 找出 _Color.jpg
            color_name = None
            for n in zf.namelist():
                low = n.lower()
                if low.endswith(".jpg") and "color" in low:
                    color_name = n
                    break
            if color_name is None:
                return False, f"no Color jpg in zip ({zf.namelist()[:3]}...)"
            with zf.open(color_name) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
        return True, out_name
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"URL error: {e.reason}"
    except zipfile.BadZipFile:
        return False, "bad zip"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading {len(CANDIDATES)} candidates → {OUT_DIR}")
    ok, fail = 0, 0
    for i, asset_id in enumerate(CANDIDATES):
        t0 = time.time()
        success, msg = download_one(i, asset_id)
        dt = time.time() - t0
        if success:
            ok += 1
            print(f"  [{i:02d}] OK   {msg:48s} ({dt:.1f}s)")
        else:
            fail += 1
            print(f"  [{i:02d}] FAIL {asset_id:25s} {msg}")
    print(f"\nDone. {ok} ok, {fail} fail. Output: {OUT_DIR}")
    print("Next: python scripts/preview_bg_candidates.py")


if __name__ == "__main__":
    main()
