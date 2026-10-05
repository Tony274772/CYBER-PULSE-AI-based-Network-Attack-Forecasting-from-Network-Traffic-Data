"""Builds the temporal dataset for V2.

Performs windowing, flow and packet aggregation, target creation,
and time-aware splitting with a purge gap.
"""
from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from src.features.flow_features import aggregate_flows
from src.features.packet_features import aggregate_packet_features
from src.features.window_state import join_window_states
from src.features.forecast_targets import build_forecast_targets

logger = logging.getLogger(__name__)


def create_window_states(
    flow_df: pd.DataFrame, 
    packet_df: pd.DataFrame, 
    delta_t_seconds: float
) -> pd.DataFrame:
    """Aggregates raw flow and packet data into non-overlapping temporal windows."""
    
    if flow_df.empty:
        return pd.DataFrame()

    t0 = flow_df["Timestamp"].min()
    
    # We assign window_id to both dataframes based on the same start time
    flow_df["window_id"] = ((flow_df["Timestamp"] - t0).dt.total_seconds() // delta_t_seconds).astype(int)
    
    flow_windows = []
    # Aggregate flow features per window
    for wid, group in flow_df.groupby("window_id"):
        # Most severe stage mapping
        stages = group.get("Mapped_Stage", pd.Series(["Benign"] * len(group)))
        # We need a fallback if Mapped_Stage isn't present, we'll try to find it later or assume from Label
        label = "BENIGN" if (group["Label"] == "BENIGN").all() else "ATTACK"
        
        agg = aggregate_flows(group)
        agg["window_id"] = wid
        agg["timestamp"] = group["Timestamp"].min()
        agg["Label"] = label
        
        # If 'mapped_stage' exists in raw data use it, otherwise keep 'Benign'/'Attack'
        if "mapped_stage" in group.columns:
            # simple voting or max
            agg["mapped_stage"] = group["mapped_stage"].value_counts().index[0]
        else:
            agg["mapped_stage"] = label # We will remap later
            
        agg["source_file"] = group["source_file"].iloc[0] if "source_file" in group.columns else "unknown"
        flow_windows.append(agg)
        
    flow_states = pd.DataFrame(flow_windows)
    
    # Packet windows
    packet_windows_list = []
    if packet_df is not None and not packet_df.empty:
        for wid, group in packet_df.groupby("window_id"):
            agg = aggregate_packet_features(group)
            agg["window_id"] = wid
            packet_windows_list.append(agg)
            
    packet_states = pd.DataFrame(packet_windows_list)
    
    joined = join_window_states(flow_states, packet_states)
    return joined.sort_values("window_id").reset_index(drop=True)


def split_with_purge(
    df: pd.DataFrame, 
    group_col: str, 
    purge_gap_windows: int, 
    train_frac: float = 0.70, 
    val_frac: float = 0.15
) -> dict[str, pd.DataFrame]:
    """Splits dataset temporally with a purge gap to prevent leakage."""
    parts = {"train": [], "val": [], "test": []}
    
    for _, g in df.groupby(group_col, sort=False):
        g = g.sort_values("window_id").reset_index(drop=True)
        n = len(g)
        
        i1 = int(n * train_frac)
        i2 = int(n * (train_frac + val_frac))
        
        train_end = max(0, i1 - purge_gap_windows)
        val_start = min(n, i1 + purge_gap_windows)
        val_end = max(val_start, i2 - purge_gap_windows)
        test_start = min(n, i2 + purge_gap_windows)
        
        parts["train"].append(g.iloc[:train_end])
        parts["val"].append(g.iloc[val_start:val_end])
        parts["test"].append(g.iloc[test_start:])
        
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
