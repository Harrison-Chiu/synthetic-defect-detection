"""
schema.py — 標籤 / 模型輸出頭的「單一事實來源」(single source of truth)

定位
----
本檔**只描述結構**,不放數值、不綁框架(不 import torch)。任何模組都能 import:
  - models/  從 `output_channels()` 推導每個 head 的輸出通道數
  - data/    從各 head 的 class 表 / index map 編碼 label
  - eval/    從 `level` 決定該用 per-pixel 還是 per-instance 指標
  - train/   從 `losses` 知道每個 head 套哪些 loss(權重數值在 configs/,不在此)

為什麼要有這檔
--------------
Stage 4 的痛點:state(缺陷嚴重度)本質是 **per-instance** 屬性(meta 裡每個
實例一個 defect_state),卻被廣播成 per-pixel 後用 pixel-level loss(CE+Dice)訓練。
這種「instance 標籤套 pixel loss」的 mismatch 是 multi-head 失敗的疑兇之一,而且是
「跑完才發現」的那種錯。把每個 head 的 `level` 顯性寫下來 + 一個 guardrail,就能在
實例化時就把這類 mismatch 叫出來(目前設為 **warn 放行**,不擋 S4 重跑)。

現狀 vs 未來
------------
下方 HEADS 如實編碼 **Stage 4 程式碼的 3 頭(A/B/C)**。其中:
  - A(part)是主任務,真正的 dense pixel 標籤。
  - B(state)/ C(type)是 aux 頭,屬 S4 已**否決**的多頭路線(best 結果來自 single-head)。
S5 建模方向(改不改 head 列表 / loss)等 S4 review 後再定 —— 本檔即為當時要動的起點。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum


# ── 共用常數 ───────────────────────────────────────────────
IGNORE_INDEX = -100  # 與 torch CE 預設一致;標記「零件外 / 不計入 loss」的像素


class Level(str, Enum):
    """標籤的本質粒度(不是『預測成什麼』,是『標籤怎麼來』)。"""
    PIXEL = "pixel"        # 真·dense:每像素獨立一個標籤(如 part/bg mask)
    INSTANCE = "instance"  # 每個實例一個標籤,訓練時廣播到該實例的像素


class Loss(str, Enum):
    """目前用到的 loss。語意上皆為 *逐像素套用*(pixel-applied)。"""
    CE = "ce"      # cross entropy(逐像素,multiclass softmax head)
    DICE = "dice"  # soft dice(逐像素)
    BCE = "bce"    # 加權 binary cross-entropy(單通道 sigmoid head,S5 defect 用)


#: 哪些 loss 屬於「逐像素監督」——guardrail 用這組判斷 instance/pixel mismatch。
#: 未來若新增 instance-level loss(如 region-pooled CE),不要列進這裡。
PIXEL_APPLIED_LOSSES: frozenset[Loss] = frozenset({Loss.CE, Loss.DICE})


# ── class 表(label index 的單一來源)─────────────────────────
# index = tuple 內位置;下游一律用這裡,別在各 script 重新硬寫 dict。
PART_CLASSES: tuple[str, ...] = ("bg", "part")

#: S5 defect 頭:單通道 sigmoid 瑕疵定位熱圖(非 class softmax)。
DEFECT_CLASSES: tuple[str, ...] = ("defect",)

STATE_CLASSES: tuple[str, ...] = (
    "normal",
    "bend_light", "bend_heavy",
    "displace_light", "displace_heavy",
    "remesh_light", "remesh_heavy",
)

TYPE_CLASSES: tuple[str, ...] = ("normal", "bend", "displace", "remesh")


def _index_map(classes: tuple[str, ...]) -> dict[str, int]:
    return {name: i for i, name in enumerate(classes)}


PART_TO_IDX = _index_map(PART_CLASSES)
STATE_TO_IDX = _index_map(STATE_CLASSES)
TYPE_TO_IDX = _index_map(TYPE_CLASSES)


def state_to_type(state_name: str) -> str:
    """defect_state → defect_type(取底線前綴;normal 自成一類)。

    單一來源:取代散在 script 裡的同名邏輯。
    """
    if state_name == "normal":
        return "normal"
    return state_name.split("_")[0]


def state_idx_to_type_idx(state_idx: int) -> int:
    """state 的 class index → type 的 class index(供 label 編碼/查表用)。"""
    return TYPE_TO_IDX[state_to_type(STATE_CLASSES[state_idx])]


# ── Head 定義 ──────────────────────────────────────────────
@dataclass(frozen=True)
class Head:
    key: str                      # 短代號(沿用 S4 的 A/B/C,log/程式好對照)
    name: str                     # 語意名(part / state / type)
    classes: tuple[str, ...]      # class 表(順序即 index)
    level: Level                  # 標籤本質粒度
    source: str                   # 標籤從哪來(人讀說明,對齊資料管線)
    losses: tuple[Loss, ...]      # 套哪些 loss(數值權重在 configs/)
    role: str = "main"            # main / aux —— aux = S4 已否決的多頭路線
    ignore_index: int | None = None  # 該 head 是否有 ignore 區(零件外像素)

    @property
    def num_classes(self) -> int:
        return len(self.classes)


# S5 雙頭(回 S3 乾淨基準):part(2 類 softmax)+ defect(1 通道 sigmoid 熱圖)。
# 相對 S4 三頭:拿掉 state(B,7 類)/ type(C,4 類)—— aux、矛盾梯度來源、
# 且非當前目標。backbone 不動,只換 d1 之後的頭。state/type 的 class 表仍保留
# (上方),供 eval 的 per-state 分項分析用(從 meta 重建,不再是訓練頭)。
HEADS: tuple[Head, ...] = (
    Head(
        key="A", name="part",
        classes=PART_CLASSES,
        level=Level.PIXEL,
        source="semantic_mask: (sem > 0) → 0=bg / 1=part",
        losses=(Loss.CE,),
        role="main",
        ignore_index=None,
    ),
    Head(
        key="B", name="defect",
        classes=DEFECT_CLASSES,
        level=Level.PIXEL,  # 變形區為真·dense 像素監督(非 instance 廣播)
        source="變形區(normal vs defect 同pose相減)→ T(膨脹容忍帶) + 高斯權重 W",
        losses=(Loss.BCE, Loss.DICE),  # 加權 BCE + 加權 soft-Dice(見 losses.py)
        role="main",
        ignore_index=None,  # ignore 由 weight map W≈0 自然達成,非 index
    ),
)

HEADS_BY_KEY: dict[str, Head] = {h.key: h for h in HEADS}
HEADS_BY_NAME: dict[str, Head] = {h.name: h for h in HEADS}


# ── 下游推導 helper ────────────────────────────────────────
def get_head(key_or_name: str) -> Head:
    if key_or_name in HEADS_BY_KEY:
        return HEADS_BY_KEY[key_or_name]
    if key_or_name in HEADS_BY_NAME:
        return HEADS_BY_NAME[key_or_name]
    raise KeyError(f"未知的 head: {key_or_name!r}(可用:{list(HEADS_BY_KEY)} / {list(HEADS_BY_NAME)})")


def output_channels(heads: tuple[Head, ...] = HEADS) -> dict[str, int]:
    """model 用:每個 head 的輸出通道數 = num_classes。"""
    return {h.key: h.num_classes for h in heads}


def main_heads(heads: tuple[Head, ...] = HEADS) -> tuple[Head, ...]:
    return tuple(h for h in heads if h.role == "main")


def aux_heads(heads: tuple[Head, ...] = HEADS) -> tuple[Head, ...]:
    return tuple(h for h in heads if h.role == "aux")


# ── Guardrail ──────────────────────────────────────────────
def check_schema(heads: tuple[Head, ...] = HEADS, *, raise_on_error: bool = False) -> list[str]:
    """檢查 schema 一致性,回傳問題訊息清單(同時 warnings.warn)。

    目前唯一規則(S4 教訓):`level=INSTANCE` 的 head 套了逐像素 loss → mismatch。
    依 2026-05-30 決議設為 **warn 放行**(保留 S4 可在新 schema 下重跑);
    `raise_on_error=True` 可在未來收緊成硬擋。
    """
    issues: list[str] = []
    for h in heads:
        if h.level is Level.INSTANCE:
            bad = [l.value for l in h.losses if l in PIXEL_APPLIED_LOSSES]
            if bad:
                issues.append(
                    f"[head {h.key}:{h.name}] level=instance 卻套逐像素 loss {bad} "
                    f"—— 每實例僅一個標籤,逐像素監督是 S4 疑似失敗主因。"
                )
    for msg in issues:
        warnings.warn(msg, stacklevel=2)
    if issues and raise_on_error:
        raise ValueError("schema 違規:\n" + "\n".join(issues))
    return issues


if __name__ == "__main__":
    # 快速自檢:列印 schema 摘要 + 跑 guardrail。
    print("=== HEADS ===")
    for h in HEADS:
        print(f"  {h.key} {h.name:6s} | {h.num_classes:2d} cls | {h.level.value:8s} "
              f"| {h.role:4s} | loss={[l.value for l in h.losses]}")
    print(f"\noutput_channels = {output_channels()}")
    print("\n=== guardrail ===")
    found = check_schema()
    print("OK,無問題" if not found else f"{len(found)} 個警告(見上)")
