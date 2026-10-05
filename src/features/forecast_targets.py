"""Future forecasting target construction for V2.
Constructs y_5, y_30, y_60 and future_stage_30 targets for each window.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def build_forecast_targets(window_df: pd.DataFrame, config) -> pd.DataFrame:
    """Constructs future targets from the window-level sequence.
    
    Expects window_df sorted chronologically by window_id.
    """
    targets = pd.DataFrame(index=window_df.index)
    
    # 1. Next state
    # Next state is simply a shifted version of the features.
    # Exclude non-feature columns.
    exclude_cols = ["window_id", "timestamp", "Label", "mapped_stage", "source_file"]
    feature_cols = [c for c in window_df.columns if c not in exclude_cols]
    
    # Shift by -1 to get the next window's features
    next_state = window_df[feature_cols].shift(-1)
    
    for c in feature_cols:
        targets[f"next_{c}"] = next_state[c]
        
    # Attack labels
    is_attack = (window_df["Label"] != "BENIGN").astype(int)
    stages = window_df.get("mapped_stage", pd.Series("Benign", index=window_df.index))
    
    # +5s target (next 1 window)
    targets["y_5"] = is_attack.shift(-1).fillna(0).astype(int)
    
    # +30s target (next 6 windows)
    y_30 = is_attack.rolling(window=config.data.horizon_30_windows, min_periods=1).max().shift(-config.data.horizon_30_windows)
    targets["y_30"] = y_30.fillna(0).astype(int)
    
    # +60s target (next 12 windows)
    y_60 = is_attack.rolling(window=config.data.horizon_60_windows, min_periods=1).max().shift(-config.data.horizon_60_windows)
    targets["y_60"] = y_60.fillna(0).astype(int)
    
    # Future stage 30 mapping
    # To find the most severe stage, we should use the stage_order from mitre mapping
    stage_order = config.stage_order
    stage_to_idx = {s: i for i, s in enumerate(stage_order)}
    idx_to_stage = {i: s for i, s in enumerate(stage_order)}
    
    stage_idx = stages.map(stage_to_idx).fillna(0).astype(int)
    future_stage_idx = stage_idx.rolling(window=config.data.horizon_30_windows, min_periods=1).max().shift(-config.data.horizon_30_windows)
    future_stage_idx = future_stage_idx.fillna(0).astype(int)
    
    targets["future_stage_30"] = future_stage_idx.map(idx_to_stage)
    
    return targets
