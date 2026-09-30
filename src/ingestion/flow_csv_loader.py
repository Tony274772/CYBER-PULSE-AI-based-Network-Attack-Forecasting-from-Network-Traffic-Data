"""Ingestion: CIC-IDS2017 flow CSVs -> tidy per-flow DataFrame (Section 7.1).

The raw CSVs were already cleaned to parquet by ``data_prep/clean_cic2017.py``
(header whitespace stripped, embedded header rows dropped, inf/NaN handled,
timestamps parsed). This loader reads those processed parquet files and maps
the CICFlowMeter columns onto the canonical Section 6.3 flow feature vector,
returning the tidy schema the rest of the pipeline expects:

    [timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
     <FLOW_FEATURES>, <FLOW_AVAIL_FEATURES>, raw_label, mapped_stage,
     dataset, source_file]

CIC-IDS2017 flow timestamps are minute-quantized in the source data, so a
synthetic intra-minute spread (rank/count within each (source_file, minute))
is applied to make sub-minute (Δt=5s) windowing non-degenerate. This is a
documented simplification (see docs/architecture.md).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from src.features.schema import FLOW_AVAIL_FEATURES, FLOW_FEATURES

# CICFlowMeter (whitespace-stripped) column -> canonical 6.3 name, for the
# columns that map directly. Ratios and rates are re-derived after aggregation.
_CIC_DIRECT = {
    "Flow Duration": "flow_duration_sum",
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Total Length of Fwd Packets": "tot_fwd_bytes",
    "Total Length of Bwd Packets": "tot_bwd_bytes",
    "SYN Flag Count": "flag_syn_count",
    "ACK Flag Count": "flag_ack_count",
    "FIN Flag Count": "flag_fin_count",
    "RST Flag Count": "flag_rst_count",
    "PSH Flag Count": "flag_psh_count",
    "URG Flag Count": "flag_urg_count",
    "Flow IAT Mean": "iat_mean",
    "Flow IAT Std": "iat_std",
    "Flow IAT Max": "iat_max",
    "Flow IAT Min": "iat_min",
    "Flow Bytes/s": "flow_bytes_per_sec",
    "Flow Packets/s": "flow_pkts_per_sec",
}


def _load_mapping(mapping_path: str) -> dict:
    with open(mapping_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def spread_intra_minute(df: pd.DataFrame, time_col: str = "timestamp",
                        group_col: str = "source_file") -> pd.DataFrame:
    """Add a fractional second offset within each (group, minute) bucket.

    CICFlowMeter emits flows in roughly chronological order, so the row rank
    within a minute approximates start order. Offset = (rank / count) * 60s.
    """
    df = df.copy()
    minute = df[time_col].dt.floor("min")
    keys = [group_col, minute] if group_col in df.columns else [minute]
    grp = df.groupby(keys, sort=False)
    rank = grp.cumcount()
    count = grp[time_col].transform("size").to_numpy()
    frac_seconds = (rank.to_numpy() / np.maximum(count, 1)) * 60.0
    df[time_col] = df[time_col] + pd.to_timedelta(frac_seconds, unit="s")
    return df


def to_tidy(df: pd.DataFrame, mapping_path: str) -> pd.DataFrame:
    """Map a cleaned CIC-IDS2017 flow frame to the tidy schema."""
    mapping = _load_mapping(mapping_path)
    label_map = mapping["cic_ids2017_label_map"]

    out = pd.DataFrame(index=df.index)
    out["timestamp"] = pd.to_datetime(df["Timestamp"])
    out["src_ip"] = df["Source IP"].astype(str)
    out["dst_ip"] = df["Destination IP"].astype(str)
    out["src_port"] = pd.to_numeric(df["Source Port"], errors="coerce").fillna(0).astype(int)
    out["dst_port"] = pd.to_numeric(df["Destination Port"], errors="coerce").fillna(0).astype(int)
    out["protocol"] = pd.to_numeric(df["Protocol"], errors="coerce").fillna(0).astype(int)

    for src, dst in _CIC_DIRECT.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce").astype(float) if src in df.columns else 0.0
    # CICFlowMeter Flow Duration is in microseconds; convert to seconds so the
    # duration-weighted IAT aggregation and rate re-derivation stay in seconds.
    out["flow_duration_sum"] = out["flow_duration_sum"] / 1e6

    # derived ratios (per-flow; re-derived again after window aggregation)
    out["fwd_bwd_byte_ratio"] = out["tot_fwd_bytes"] / (out["tot_bwd_bytes"] + 1.0)
    out["fwd_bwd_pkt_ratio"] = out["tot_fwd_pkts"] / (out["tot_bwd_pkts"] + 1.0)

    # every CIC flow feature is genuinely available -> availability bits all 1
    for c in FLOW_AVAIL_FEATURES:
        out[c] = 1.0

    out = out.replace([np.inf, -np.inf], np.nan)
    out[FLOW_FEATURES] = out[FLOW_FEATURES].fillna(0.0)

    raw_label = df["Label"].astype(str).str.strip()
    out["raw_label"] = raw_label
    out["mapped_stage"] = raw_label.map(label_map).fillna("Benign")
    out["dataset"] = "cic2017"
    out["source_file"] = df["source_file"].astype(str) if "source_file" in df.columns else "cic2017"
    return out


def load(path_or_dir: str, mapping_path: str = "mitre_mapping/attack_stage_mapping.json",
         spread_minutes: bool = True) -> pd.DataFrame:
    """Load a cleaned CIC-IDS2017 parquet file (or a directory of them / a split
    file) and return the tidy per-flow DataFrame, sorted by timestamp."""
    if os.path.isdir(path_or_dir):
        files = sorted(glob.glob(os.path.join(path_or_dir, "**", "*.parquet"), recursive=True))
        # avoid re-loading split shards on top of the clean file
        files = [f for f in files if os.path.basename(f) == "cic2017_clean.parquet"] or files
        frames = [pd.read_parquet(f) for f in files]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_parquet(path_or_dir)

    tidy = to_tidy(df, mapping_path)
    if spread_minutes:
        tidy = spread_intra_minute(tidy)
    return tidy.sort_values("timestamp").reset_index(drop=True)
