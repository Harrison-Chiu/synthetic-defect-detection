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
    depth: int = 4               # encoder/decoder 層數(下採樣次數);縮小模型的第二軸

    # 優化
    seed: int = 42
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # step-based 排程(取代 epoch)
    total_steps: int = 3000      # ≈ S4 的 30 epoch × ~100 step
    eval_every: int = 100        # 每多少 step 評估 + checkpoint 一次(≈ 1 epoch)
    snapshot_every: int = 500    # 每多少 step 存一次預測快照

    # S5 雙頭 loss 權重(A=part, B=defect;B 即 λ)
    head_weights: dict[str, float] = field(default_factory=lambda: {"A": 1.0, "B": 1.0})
    pos_weight: float = 8.0        # defect 頭 BCE 正類權重 ρ(補變形區稀少)
    defect_thr: float = 0.7        # 推論 sigmoid > thr 判 defect。0.7 由 sweep_defect_thr 定:
                                   # bc8 上 thr=0.7 → defect IoU 0.365(≈S3)且 normal FP 22%→7%

    # DataLoader(runtime 生成防阻塞:workers 平行生成,實測 8 → ~11×)
    num_workers: int = 8

    # ReduceLROnPlateau(解 S3 epoch30 突崩;以 eval mIoU 為準)
    lr_factor: float = 0.5
    lr_patience: int = 3         # 以「eval 次數」計
    lr_threshold: float = 0.005
    lr_min: float = 1e-5


DEFAULT_TRAIN_CONFIG = TrainConfig()
