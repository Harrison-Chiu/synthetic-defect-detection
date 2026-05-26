"""
Stage 3 — 把 assets/backgrounds_candidates/ 內所有 jpg 做成一張 contact sheet 給人挑。

執行：
    python scripts/preview_bg_candidates.py

輸出：
    assets/backgrounds_candidates/preview.png
    每格左上角印編號（cand_XX_）+ 下方印 asset id

Harrison 操作流程：
    1. 開 preview.png 看圖
    2. 回報要保留的編號（例：「保留 0 2 5 7 9 11 14」）
    3. Claude 把選中的 cand_*.jpg 搬進 assets/backgrounds/，
       舊的 procedural 12 張搬到 assets/backgrounds_legacy/。
"""

import os
import glob
import math
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = r"D:\Harrison\中山\大四下\深度學習期末報告"
CAND_DIR = os.path.join(BASE_DIR, "assets", "backgrounds_candidates")
OUT_PATH = os.path.join(CAND_DIR, "preview.png")

CELL_SIZE = 320           # 每張縮成 320×320
COLS      = 6             # 一列幾張
PAD       = 8             # 縫隙
LABEL_H   = 36            # 下方文字區高度


def main():
    files = sorted(glob.glob(os.path.join(CAND_DIR, "cand_*.jpg")))
    if not files:
        print(f"No candidates in {CAND_DIR}. Run download_ambientcg.py first.")
        return

    n     = len(files)
    rows  = math.ceil(n / COLS)
    sheet_w = COLS * CELL_SIZE + (COLS + 1) * PAD
    sheet_h = rows * (CELL_SIZE + LABEL_H) + (rows + 1) * PAD

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 30))
    draw  = ImageDraw.Draw(sheet)

    try:
        font_big = ImageFont.truetype("arial.ttf", 28)
        font_sm  = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font_big = ImageFont.load_default()
        font_sm  = ImageFont.load_default()

    for i, f in enumerate(files):
        r = i // COLS
        c = i % COLS
        x = PAD + c * (CELL_SIZE + PAD)
        y = PAD + r * (CELL_SIZE + LABEL_H + PAD)

        img = Image.open(f).convert("RGB")
        # center-crop 成正方形再縮放
        side = min(img.size)
        left = (img.width - side) // 2
        top  = (img.height - side) // 2
        img  = img.crop((left, top, left + side, top + side))
        img  = img.resize((CELL_SIZE, CELL_SIZE), Image.BILINEAR)
        sheet.paste(img, (x, y))

        # 編號（左上角，黃底黑字）
        basename = os.path.basename(f)
        # cand_XX_AssetName.jpg
        try:
            idx_str = basename.split("_")[1]
            asset_id = "_".join(basename.split("_")[2:]).rsplit(".", 1)[0]
        except Exception:
            idx_str = "??"
            asset_id = basename

        # 編號標籤（左上）
        badge_w, badge_h = 60, 40
        draw.rectangle([x, y, x + badge_w, y + badge_h], fill=(255, 220, 0))
        draw.text((x + 6, y + 4), idx_str, fill=(0, 0, 0), font=font_big)
        # asset id（下方文字區）
        draw.text((x + 4, y + CELL_SIZE + 4), asset_id, fill=(220, 220, 220), font=font_sm)

    sheet.save(OUT_PATH)
    print(f"Saved contact sheet: {OUT_PATH}")
    print(f"  {n} candidates in {rows}×{COLS} grid, {sheet_w}×{sheet_h} px")


if __name__ == "__main__":
    main()
