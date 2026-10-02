from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    bert_name: str = "bert-base-uncased"
    pretrained: bool = True
    local_files_only: bool = False
    feature_dim: int = 256
    decoder_dim: int = 256
    pool_sizes: list[int] = field(default_factory=lambda: [8, 4, 2, 1])
    local_kernel: int = 3
    use_mfie: bool = True
    use_afw: bool = True


@dataclass
class DataConfig:
    root: str = "data/droneris"
    train: str = "annotations/train.jsonl"
    val: str = "annotations/val.jsonl"
    image_size: int = 480
    max_text_length: int = 64
    evaluation_resolution: str = "original"


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 8
    lr: float = 3e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    dice_eps: float = 1e-6
    seed: int = 42
    amp: bool = False


@dataclass
class RuntimeConfig:
    output: str = "runs/droneris"
    workers: int = 4
    device: str = "cuda"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        m, d, t, r = self.model, self.data, self.training, self.runtime
        if m.feature_dim < 1 or m.decoder_dim < 1:
            raise ValueError("Feature dimensions must be positive")
        if len(m.pool_sizes) != 4 or min(m.pool_sizes) < 1:
            raise ValueError("pool_sizes must contain four positive integers")
        if m.local_kernel < 1 or m.local_kernel % 2 != 1:
            raise ValueError("local_kernel must be a positive odd integer")
        if d.image_size < 32 or d.image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        if not 2 <= d.max_text_length <= 512:
            raise ValueError("max_text_length must be between 2 and 512")
        if d.evaluation_resolution not in {"original", "resized"}:
            raise ValueError("evaluation_resolution must be original or resized")
        if t.epochs < 2 or t.batch_size < 1 or r.workers < 0:
            raise ValueError("Invalid epochs, batch_size or workers")
        if not 0 < t.min_lr <= t.lr or t.weight_decay < 0:
            raise ValueError("Invalid learning rate or weight decay")
        if t.eps <= 0 or t.dice_eps <= 0:
            raise ValueError("Loss and optimizer epsilon must be positive")
        if len(t.betas) != 2 or any(not 0 <= beta < 1 for beta in t.betas):
            raise ValueError("AdamW betas must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


def from_dict(values: dict) -> Config:
    classes = {
        "model": ModelConfig,
        "data": DataConfig,
        "training": TrainingConfig,
        "runtime": RuntimeConfig,
    }
    unknown = set(values) - classes.keys()
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    config = Config(**{key: classes[key](**value) for key, value in values.items()})
    config.validate()
    return config


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as stream:
        return from_dict(yaml.safe_load(stream) or {})
