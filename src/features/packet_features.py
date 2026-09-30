"""Packet-level feature handling (Section 7.3 / 6.4).

Two roles:
  1. Training path — wrap the precomputed Wednesday packet-feature parquet
     (``cic2017_wednesday_packet_features.parquet``) in a lookup keyed by
     (window_id, src_ip, dst_ip), and provide the flow->packet window
     alignment used to join packet features onto Wednesday flow edges.
  2. Live path — thin re-export of the streaming PCAP parser for the
     dashboard's fresh-upload case.

Alignment note (documented simplification): the precomputed packet features
use an integer ``window_id`` relative to the PCAP's own first-packet time,
with a fixed 5s non-overlapping bucket. Flow windows are aligned to it by the
common capture start — Wednesday flow times are shifted so the flow t_min
coincides with packet window 0, then ``packet_wid = floor(seconds_since_start /
delta_t)``. See docs/architecture.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.schema import PACKET_FEATURES
from src.ingestion.pcap_parser import parse_pcap  # noqa: F401 (re-export for live path)


class PacketFeatureIndex:
    """Fast (window_id, src_ip, dst_ip) -> 10-dim packet vector lookup."""

    def __init__(self, packet_df: pd.DataFrame, delta_t_seconds: float):
        self.delta_t = float(delta_t_seconds)
        cols = ["window_id", "src_ip", "dst_ip", *PACKET_FEATURES]
        df = packet_df[cols].copy()
        df["src_ip"] = df["src_ip"].astype(str)
        df["dst_ip"] = df["dst_ip"].astype(str)
        self._table = df.set_index(["window_id", "src_ip", "dst_ip"])[PACKET_FEATURES]
        # dict for O(1) lookup during windowing
        self._lookup = {k: v for k, v in zip(self._table.index, self._table.to_numpy())}

    @classmethod
    def from_parquet(cls, path: str, delta_t_seconds: float) -> "PacketFeatureIndex":
        return cls(pd.read_parquet(path), delta_t_seconds)

    def packet_window_id(self, window_start_epoch: float, capture_start_epoch: float) -> int:
        return int((window_start_epoch - capture_start_epoch) // self.delta_t)

    def get(self, window_id: int, src_ip: str, dst_ip: str) -> np.ndarray | None:
        return self._lookup.get((int(window_id), str(src_ip), str(dst_ip)))

    @property
    def dim(self) -> int:
        return len(PACKET_FEATURES)
