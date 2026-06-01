"""
config.py — runtime 場景生成器的參數(版控核心)

定位
----
`整理規則 §2`:版控 = 零件 patch + **生成 config** + 凍結 eval set。
生成器是純函數 `scene(i) = f(i, config)`,所以「config 數值」就是資料集的身分證 ——
固定 config + 固定 patch 池 → 可完全重生同一批場景,不必存 1000 張訓練圖。

數值來源 / 調校紀律
-------------------
下方數值**原樣沿用 Stage 4 `scripts/composite.py`**(忠實搬遷,不重新調)。
依 2026-05-30 決議:生成 config 的**數值調校等 S4 review 之後**(displace 退步、HDRI
pool 影響等都要等審查定位)。本檔此刻的職責只是「把散在 script 的 module 常數,
收斂成一個可被 git 追蹤、可被 hash 成資料集版本的結構」。

之後要序列化成 configs/*.yaml|json 由 cli 載入;先用 dataclass 當單一事實來源。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# semantic class 編碼(與 composite.py 一致;eval 對 Stage 3 schema 可比)
CLS_BG = 0
CLS_NORMAL = 1
CLS_DEFECT = 2

#: 屬於「瑕疵」的 defect_state(其餘視為 normal)。與 schema.STATE_CLASSES 對齊。
#: S6 簡化為 3 種（目錄名與 render config DEFECT_STATES 對齊）
DEFECT_STATES_DEFECTIVE = frozenset({
    "bend_45",
    "displace",
    "remesh",
})

# ── index 區段保留 ──────────────────────────────────────────
# scene(i) 是純函數,因此「index 互斥 = 場景互斥」。給 eval / samples 各保留高位
# 區段,確保它們的場景永遠不會與訓練(從 0 起算)撞號 → 杜絕資料洩漏。
EVAL_INDEX_OFFSET = 10_000_000     # val set 用(選 best;與訓練 0 起算互斥)
SAMPLE_INDEX_OFFSET = 20_000_000   # 報告配圖用
TEST_INDEX_OFFSET = 30_000_000     # 獨立 test set 用(報數字;與 val 互斥,S5 新增)


# ── S5 defect 頭 target 編碼參數(label 幾何,非生成;見 stage5_plan.md §3)──
# 變形區二值 mask(離線 gen_defectmask.py,thr=30)→ 場景空間後,在此組 T / W。
DEFECT_DILATE_PX = 4      # D⁺ = 變形區膨脹(容忍帶:亮在附近也算對)
DEFECT_GAUSS_SIGMA = 14.0 # S6: 加寬高斯(6→14),讓過渡帶延伸更遠,壞件邊緣有更多梯度訊號
DEFECT_W_NORM = 0.3       # 好件(normal part)像素的權重(逼模型學「好件→不該亮」)
DEFECT_W_BG = 0.0         # 遠背景權重(0 = 完全不在意;壞件內部亦由高斯自然衰減到≈0)


@dataclass(frozen=True)
class GenConfig:
    """一次生成的完整參數。frozen → 可 hash 成資料集版本指紋。"""

    # ── 種子 / 尺寸 ──
    seed: int = 42              # 與 i 一起決定 per-scene RNG;換 seed = 換整批資料
    out_size: int = 256

    # ── 零件擺放 ──
    parts_range: tuple[int, int] = (3, 6)      # S6: 3~6 顆
    scale_base_range: tuple[float, float] = (0.50, 0.75)  # S6: per-scene base scale (50~75%)
    scale_jitter: float = 0.10                 # S6: 各零件 ± jitter (base × [1-j, 1+j])
    rot_range: tuple[float, float] = (0.0, 360.0)
    defect_prob: float = 0.20                  # 每 instance 為瑕疵的機率
    collision_attempts: int = 50               # S6: 防三層堆疊的隨機嘗試次數

    # ── 背景 base layer(互斥,per scene)──
    base_layer_probs: dict[str, float] = field(default_factory=lambda: {
        "real": 0.50, "procedural": 0.25, "solid": 0.25,
    })
    solid_color_palette: tuple[tuple[int, int, int], ...] = (
        (60, 60, 60), (90, 90, 90), (120, 120, 120), (160, 160, 160),
        (110, 90, 70), (140, 110, 80), (80, 70, 60),
        (100, 90, 100), (90, 100, 110),
    )
    solid_noise_std: float = 6.0

    # ── 背景增強 ──
    bg_aug_brightness: tuple[float, float] = (0.75, 1.25)
    bg_aug_contrast: tuple[float, float] = (0.80, 1.20)
    bg_aug_saturation: tuple[float, float] = (0.70, 1.30)
    bg_aug_flip_h_prob: float = 0.5
    bg_aug_flip_v_prob: float = 0.5

    # ── distractors ──
    # 每 bucket:((count_lo, count_hi_inclusive), prob)
    distractor_count_buckets: tuple[tuple[tuple[int, int], float], ...] = (
        ((0, 0), 0.25), ((1, 4), 0.35), ((5, 10), 0.25), ((11, 20), 0.15),
    )
    distractor_shape_probs: tuple[tuple[str, float], ...] = (
        ("ellipse", 0.40), ("rectangle", 0.25),
        ("polygon", 0.15), ("line", 0.15), ("ring", 0.05),
    )
    distractor_size_range: tuple[int, int] = (5, 80)
    distractor_alpha_range: tuple[float, float] = (0.50, 1.0)
    distractor_color_metallic_prob: float = 0.4
    distractor_metallic_palette: tuple[tuple[int, int, int], ...] = (
        (70, 70, 75), (90, 90, 95), (110, 105, 100),
        (45, 45, 50), (130, 125, 120), (60, 65, 70),
    )


#: 預設 = 忠實還原 Stage 4 的生成設定。
DEFAULT_CONFIG = GenConfig()
