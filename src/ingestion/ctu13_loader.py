"""Ingestion: CTU-13 bidirectional NetFlow -> tidy per-flow DataFrame (7.1).

``data_prep/clean_ctu13.py`` already parsed the .binetflow files, applied the
Section 6.2 substring rules to produce ``mapped_stage``, and split the rows
into ``ctu13_labeled.parquet`` (Benign + mapped-malicious) and
``ctu13_background_pool.parquet`` (Background, unlabeled Stage-1 pool). This
loader reads those parquet files and maps the coarse binetflow columns onto
the canonical Section 6.3 flow feature vector.

CTU-13 is coarser than CIC-IDS2017: there is no forward/backward packet split
and no inter-arrival timing, so those feature positions are zero-filled and
their per-field availability bit set to 0 (Section 6.3). TCP flag presence is
approximated from the Argus ``State`` string (letters S/A/F/R/P/U).
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.features.schema import FLOW_AVAIL_FEATURES, FLOW_FEATURES

_PROTO_CODE = {"tcp": 6, "udp": 17, "icmp": 1, "igmp": 2, "rtp": 0, "arp": 0}
_FLAG_LETTER = {
    "flag_syn_count": "S", "flag_ack_count": "A", "flag_fin_count": "F",
    "flag_rst_count": "R", "flag_psh_count": "P", "flag_urg_count": "U",
}


def _load_rules(mapping_path: str) -> list[dict]:
    with open(mapping_path, "r", encoding="utf-8") as fh:
        return json.load(fh)["ctu13_label_substring_rules"]


def map_ctu13_label(label: str,
                    mapping_path: str = "mitre_mapping/attack_stage_mapping.json"
                    ) -> Optional[str]:
    """Section 6.2 rule engine: lower-case the free-text label, test the rules
    in array order, return the first matching stage or ``None`` for Background
    (meaning: Stage-1 pretraining only, excluded from Stage-2 supervised loss)."""
    text = str(label).lower()
    for rule in _load_rules(mapping_path):
        if "contains" in rule and rule["contains"].lower() in text:
            return rule["stage"]
        if "contains_any" in rule:
            if any(s.lower() in text for s in rule["contains_any"]):
                if any(x.lower() in text for x in rule.get("excludes", [])):
                    continue
                return rule["stage"]
    return None


def _hex_or_int_port(series: pd.Series) -> pd.Series:
    def conv(v):
        s = str(v).strip()
        try:
            return int(s, 16) if s.lower().startswith("0x") else int(float(s))
        except (ValueError, TypeError):
            return 0
    return series.map(conv).astype(int)


def to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = pd.to_datetime(df["StartTime"])
    out["src_ip"] = df["SrcAddr"].astype(str)
    out["dst_ip"] = df["DstAddr"].astype(str)
    out["src_port"] = _hex_or_int_port(df["Sport"]) if "Sport" in df.columns else 0
    out["dst_port"] = _hex_or_int_port(df["Dport"]) if "Dport" in df.columns else 0
    out["protocol"] = df["Proto"].astype(str).str.lower().map(_PROTO_CODE).fillna(0).astype(int)

    dur = pd.to_numeric(df["Dur"], errors="coerce").fillna(0.0).clip(lower=0.0)
    tot_pkts = pd.to_numeric(df["TotPkts"], errors="coerce").fillna(0.0)
    tot_bytes = pd.to_numeric(df["TotBytes"], errors="coerce").fillna(0.0)
    src_bytes = pd.to_numeric(df["SrcBytes"], errors="coerce").fillna(0.0)

    out["flow_duration_sum"] = dur
    # no fwd/bwd packet split in binetflow -> total goes in "fwd", bwd unknown
    out["tot_fwd_pkts"] = tot_pkts
    out["tot_bwd_pkts"] = 0.0
    out["tot_fwd_bytes"] = src_bytes
    out["tot_bwd_bytes"] = (tot_bytes - src_bytes).clip(lower=0.0)

    # TCP flag presence approximated from the Argus State string
    state = df["State"].astype(str).str.upper() if "State" in df.columns else pd.Series("", index=df.index)
    for feat, letter in _FLAG_LETTER.items():
        out[feat] = state.str.contains(letter, regex=False).astype(float)

    # IAT stats not derivable from a single aggregated flow row
    for c in ("iat_mean", "iat_std", "iat_max", "iat_min"):
        out[c] = 0.0

    safe_dur = dur.replace(0.0, np.nan)
    out["flow_bytes_per_sec"] = (tot_bytes / safe_dur).fillna(0.0)
    out["flow_pkts_per_sec"] = (tot_pkts / safe_dur).fillna(0.0)
    out["fwd_bwd_byte_ratio"] = out["tot_fwd_bytes"] / (out["tot_bwd_bytes"] + 1.0)
    out["fwd_bwd_pkt_ratio"] = out["tot_fwd_pkts"] / (out["tot_bwd_pkts"] + 1.0)

    # availability bits: 0 for the positions CTU-13 cannot derive
    unavailable = {"tot_bwd_pkts", "iat_mean", "iat_std", "iat_max", "iat_min"}
    for c, avail_c in zip(FLOW_FEATURES, FLOW_AVAIL_FEATURES):
        out[avail_c] = 0.0 if c in unavailable else 1.0

    out = out.replace([np.inf, -np.inf], np.nan)
    out[FLOW_FEATURES] = out[FLOW_FEATURES].fillna(0.0)

    out["raw_label"] = df["Label"].astype(str) if "Label" in df.columns else ""
    out["mapped_stage"] = df["mapped_stage"] if "mapped_stage" in df.columns else "Benign"
    out["dataset"] = "ctu13"
    out["source_file"] = df["scenario_file"].astype(str) if "scenario_file" in df.columns else "ctu13"
    return out


def load(path: str, drop_background: bool = True) -> pd.DataFrame:
    """Load a processed CTU-13 parquet (labeled split or background pool) and
    return the tidy per-flow DataFrame sorted by timestamp.

    ``ctu13_labeled.parquet`` already excludes Background rows. If a file that
    still contains Background rows is passed and ``drop_background`` is True,
    rows with a null ``mapped_stage`` are removed (they belong in the Stage-1
    pool, loaded separately)."""
    df = pd.read_parquet(path)
    if drop_background and "mapped_stage" in df.columns:
        df = df[df["mapped_stage"].notna()]
    tidy = to_tidy(df)
    return tidy.sort_values("timestamp").reset_index(drop=True)


def load_background(path: str) -> pd.DataFrame:
    """Load the Background pool as tidy rows (mapped_stage set to None). Used
    only in Stage-1 unsupervised pretraining."""
    df = pd.read_parquet(path)
    tidy = to_tidy(df)
    tidy["mapped_stage"] = None
    return tidy.sort_values("timestamp").reset_index(drop=True)
