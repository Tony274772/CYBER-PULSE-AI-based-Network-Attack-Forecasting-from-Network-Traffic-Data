"""Config system for GT-RSSM (Section 5 of the build spec).

A single nested dataclass loaded from ``configs/default.yaml``. Every CLI
entrypoint in the pipeline accepts ``--config configs/default.yaml`` plus
optional overrides (e.g. ``--delta-t 10``) so the Δt-ablation is a one-flag
change, never a code edit. No feature/model module hardcodes these numbers;
they all read them from a ``Config`` instance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    delta_t_seconds: float = 5.0
    window_overlap: float = 0.5
    sequence_length_W: int = 20
    n_max_nodes: int = 128


@dataclass
class ModelConfig:
    gat_embed_dim: int = 64
    gat_heads: int = 4
    gat_layers: int = 2
    rssm_h_dim: int = 128
    rssm_z_dim: int = 32


@dataclass
class TrainingConfig:
    stage1_lr: float = 1.0e-3
    stage1_epochs: int = 25
    stage1_batch_size: int = 64
    stage2_encoder_lr: float = 1.0e-4
    stage2_head_lr: float = 1.0e-3
    stage2_epochs: int = 30
    focal_loss_gamma: float = 2.0
    kl_anneal_fraction: float = 0.10
    lambda_infiltration: float = 1.0
    lambda_mitre: float = 1.0


@dataclass
class PathsConfig:
    cic2017_csv_dir: str = "data/raw/cic2017/csv"
    cic2017_pcap: str = "data/raw/cic2017/pcap/Wednesday-workingHours.pcap"
    cic2017_processed_dir: str = "data/processed/cic2017"
    ctu13_dir: str = "data/raw/ctu13"
    ctu13_processed_dir: str = "data/processed/ctu13"
    nf_unsw_nb15_v2_csv: str = "data/raw/nf_unsw_nb15_v2/NF-UNSW-NB15-v2.csv"
    nf_unsw_nb15_v2_processed_dir: str = "data/processed/nf_unsw_nb15_v2"
    processed_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    mitre_mapping: str = "mitre_mapping/attack_stage_mapping.json"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # -- construction --------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(
            data=DataConfig(**(raw.get("data") or {})),
            model=ModelConfig(**(raw.get("model") or {})),
            training=TrainingConfig(**(raw.get("training") or {})),
            paths=PathsConfig(**(raw.get("paths") or {})),
        )

    # -- derived quantities --------------------------------------------------
    @property
    def window_stride_seconds(self) -> float:
        """Distance between consecutive window starts (Section 7.4 step 2)."""
        return self.data.delta_t_seconds * (1.0 - self.data.window_overlap)

    @property
    def belief_dim(self) -> int:
        """s_t = concat(h_t, z_t)."""
        return self.model.rssm_h_dim + self.model.rssm_z_dim

    def load_mitre_mapping(self) -> dict:
        with open(self.paths.mitre_mapping, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @property
    def stage_order(self) -> list[str]:
        return self.load_mitre_mapping()["stage_order"]

    # -- CLI overrides -------------------------------------------------------
    def apply_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Apply a flat dict of dotted keys, e.g. {'data.delta_t_seconds': 10}.

        Also accepts convenience aliases used by the entrypoints
        (``delta_t`` -> ``data.delta_t_seconds`` etc.).
        """
        alias = {
            "delta_t": "data.delta_t_seconds",
            "overlap": "data.window_overlap",
            "W": "data.sequence_length_W",
            "n_max": "data.n_max_nodes",
        }
        for key, value in overrides.items():
            if value is None:
                continue
            dotted = alias.get(key, key)
            section, _, attr = dotted.partition(".")
            if not attr:
                raise KeyError(f"override '{key}' must be dotted, e.g. data.delta_t_seconds")
            target = getattr(self, section)
            if not hasattr(target, attr):
                raise KeyError(f"unknown config field '{dotted}'")
            # cast to the declared field type
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
    """Register the standard --config + common override flags on an argparse
    parser. Entrypoints call this, then :func:`load_config_from_args`."""
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--delta-t", dest="delta_t", type=float, default=None,
                        help="override data.delta_t_seconds (ablation)")
    parser.add_argument("--overlap", type=float, default=None,
                        help="override data.window_overlap")
    parser.add_argument("--W", type=int, default=None, help="override sequence_length_W")
    parser.add_argument("--n-max", dest="n_max", type=int, default=None,
                        help="override data.n_max_nodes")


def load_config_from_args(args) -> Config:
    cfg = Config.load(args.config)
    cfg.apply_overrides({
        "delta_t": getattr(args, "delta_t", None),
        "overlap": getattr(args, "overlap", None),
        "W": getattr(args, "W", None),
        "n_max": getattr(args, "n_max", None),
    })
    return cfg
