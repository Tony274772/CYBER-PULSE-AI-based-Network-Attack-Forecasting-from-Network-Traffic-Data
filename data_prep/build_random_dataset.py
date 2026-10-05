"""Builds a randomized temporal dataset to demonstrate data leakage."""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.features.build_temporal_dataset import create_window_states
from src.features.forecast_targets import build_forecast_targets

def random_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> dict[str, pd.DataFrame]:
    """Random split that causes massive data leakage for time-series."""
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return {
        "train": df.iloc[:i1].reset_index(drop=True),
        "val": df.iloc[i1:i2].reset_index(drop=True),
        "test": df.iloc[i2:].reset_index(drop=True)
    }

def map_labels_to_stages(df: pd.DataFrame, mapping_file: str) -> pd.DataFrame:
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    label_to_stage = mapping.get("cic_ids2017_label_map", {})
    df["mapped_stage"] = df["Label"].map(label_to_stage).fillna("Benign")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_flow", required=True)
    parser.add_argument("--packet_features", required=False)
    parser.add_argument("--mapping_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="configs/random_split.yaml")
    args = parser.parse_args()

    config = Config.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading flow data...")
    flow_df = pd.read_parquet(args.clean_flow)
    flow_df = map_labels_to_stages(flow_df, args.mapping_file)
    
    packet_df = None
    if args.packet_features and os.path.exists(args.packet_features):
        print("Loading packet data...")
        packet_df = pd.read_parquet(args.packet_features)

    print("Building window states...")
    window_states = create_window_states(flow_df, packet_df, config.data.delta_t_seconds)
    
    print("Building forecast targets...")
    targets = build_forecast_targets(window_states, config)
    
    full_df = pd.concat([window_states, targets], axis=1)
    full_df = full_df.dropna(subset=["y_5", "y_30", "y_60"])
    
    print("Splitting randomly (WARNING: CAUSES LEAKAGE)...")
    splits = random_split(full_df)
    
    print("Scaling features...")
    exclude_cols = ["window_id", "timestamp", "Label", "mapped_stage", "source_file", "packet_features_available"]
    target_cols = [c for c in full_df.columns if c.startswith("next_") or c.startswith("y_") or c.startswith("future_")]
    feature_cols = [c for c in full_df.columns if c not in exclude_cols and c not in target_cols]
    
    scaler = StandardScaler()
    
    train_df = splits["train"].copy()
    val_df = splits["val"].copy()
    test_df = splits["test"].copy()
    
    log_cols = ["total_packets", "total_fwd_bytes", "total_bwd_bytes", "mean_flow_duration", "retransmission_count", "payload_size_mean"]
    log_cols = [c for c in log_cols if c in feature_cols]
    
    for df in [train_df, val_df, test_df]:
        if df.empty: continue
        for col in log_cols:
            df[col] = np.log1p(df[col].clip(lower=0))
            
    if not train_df.empty:
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        if not val_df.empty: val_df[feature_cols] = scaler.transform(val_df[feature_cols])
        if not test_df.empty: test_df[feature_cols] = scaler.transform(test_df[feature_cols])
             
        import joblib
        joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
    
    window_states.to_parquet(os.path.join(args.output_dir, "window_states.parquet"), index=False)
    train_df.to_parquet(os.path.join(args.output_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.output_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.output_dir, "test.parquet"), index=False)
    
    print("\n=== RANDOMIZED DATASET CREATION REPORT ===")
    print(f"Total Window States   : {len(window_states)}")
    print(f"Train samples         : {len(train_df)}")
    print(f"Validation samples    : {len(val_df)}")
    print(f"Test samples          : {len(test_df)}")

if __name__ == "__main__":
    main()
