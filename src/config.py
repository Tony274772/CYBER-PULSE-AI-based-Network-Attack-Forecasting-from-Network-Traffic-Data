"""Config system for V2 Temporal World Model."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    delta_t_seconds: float = 5.0
    window_overlap: float = 0.0
    history_windows: int = 12
    horizon_5_windows: int = 1
    horizon_30_windows: int = 6
    horizon_60_windows: int = 12


@dataclass
class ModelConfig:
    flow_input_dim: int = 28
    packet_input_dim: int = 12
    branch_dim: int = 32
    fused_dim: int = 64
    hidden_dim: int = 64


@dataclass
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 7
    gradient_clip_norm: float = 1.0
    seed: int = 42


@dataclass
class LossConfig:
    state_weight: float = 0.3
    attack_weight: float = 1.0
    stage_weight: float = 0.5


@dataclass
class ThresholdingConfig:
    target_validation_fpr: float = 0.05


@dataclass
class PathsConfig:
    cic2017_csv_dir: str = "data/raw/cic2017/csv"
    cic2017_pcap: str = "data/raw/cic2017/pcap/Wednesday-workingHours.pcap"
    ctu13_dir: str = "data/raw/ctu13"
    processed_dir: str = "data/processed/cic2017"
    checkpoint_dir: str = "checkpoints"
    metrics_dir: str = "metrics"
    mitre_mapping: str = "mitre_mapping/attack_stage_mapping.json"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    thresholding: ThresholdingConfig = field(default_factory=ThresholdingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(
            data=DataConfig(**(raw.get("data") or {})),
            model=ModelConfig(**(raw.get("model") or {})),
            training=TrainingConfig(**(raw.get("training") or {})),
            loss=LossConfig(**(raw.get("loss") or {})),
            thresholding=ThresholdingConfig(**(raw.get("thresholding") or {})),
            paths=PathsConfig(**(raw.get("paths") or {})),
        )

    def load_mitre_mapping(self) -> dict:
        with open(self.paths.mitre_mapping, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @property
    def stage_order(self) -> list[str]:
        return self.load_mitre_mapping()["stage_order"]

    def apply_overrides(self, overrides: dict[str, Any]) -> "Config":
        for key, value in overrides.items():
            if value is None:
                continue
            section, _, attr = key.partition(".")
            if not attr:
                raise KeyError(f"override '{key}' must be dotted, e.g. data.delta_t_seconds")
            target = getattr(self, section)
            if not hasattr(target, attr):
                raise KeyError(f"unknown config field '{key}'")
            current = getattr(target, attr)
            setattr(target, attr, type(current)(value) if current is not None else value)
        return self

    def to_dict(self) -> dict:
        def _d(obj):
            if is_dataclass(obj):
                return {f.name: _d(getattr(obj, f.name)) for f in fields(obj)}
            return obj
        return _d(self)

def add_config_args(parser) -> None:
    parser.add_argument("--config", default="configs/default.yaml")

def load_config_from_args(args) -> Config:
    cfg = Config.load(args.config)
    return cfg
