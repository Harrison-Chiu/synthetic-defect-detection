"""
config.py — 訓練超參數(TrainConfig)

step-based 思考(整理規則 §2):不再用 epoch,改 total_steps + 間隔 eval/checkpoint。
數值預設對齊 Stage 4(BATCH=8, LR=1e-3, SEED=42, ~3000 步 ≈ 30 epoch×100 step)。
機制就位,實際調值等 S4 review 後再定。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainConfig:
    # 模型
    base_c: int = 32

    # 優化
    seed: int = 42
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # step-based 排程(取代 epoch)
    total_steps: int = 3000      # ≈ S4 的 30 epoch × ~100 step
    eval_every: int = 100        # 每多少 step 評估 + checkpoint 一次(≈ 1 epoch)
    snapshot_every: int = 500    # 每多少 step 存一次預測快照

    # 多頭 loss 權重(預設全 1.0 = S4 ALPHA_B=ALPHA_C=1.0)
    head_weights: dict[str, float] = field(default_factory=lambda: {"A": 1.0, "B": 1.0, "C": 1.0})

    # ReduceLROnPlateau(解 S3 epoch30 突崩;以 eval mIoU 為準)
    lr_factor: float = 0.5
    lr_patience: int = 3         # 以「eval 次數」計
    lr_threshold: float = 0.005
    lr_min: float = 1e-5


DEFAULT_TRAIN_CONFIG = TrainConfig()
