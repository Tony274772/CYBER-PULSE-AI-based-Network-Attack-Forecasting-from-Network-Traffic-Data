"""Build temporal dataset from clean CIC CSVs and packet features."""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add the project root to sys.path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.features.build_temporal_dataset import create_window_states, split_with_purge
from src.features.forecast_targets import build_forecast_targets

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
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = Config.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading flow data...")
    flow_df = pd.read_parquet(args.clean_flow)
    
    # Map stages
    flow_df = map_labels_to_stages(flow_df, args.mapping_file)
    
    packet_df = None
    if args.packet_features and os.path.exists(args.packet_features):
        print("Loading packet data...")
        packet_df = pd.read_parquet(args.packet_features)

    print("Building window states...")
    window_states = create_window_states(flow_df, packet_df, config.data.delta_t_seconds)
    
    print("Building forecast targets...")
    targets = build_forecast_targets(window_states, config)
    
    # Combine states and targets
    full_df = pd.concat([window_states, targets], axis=1)
    
    # Drop rows that don't have valid future targets (at the very end of the file)
    full_df = full_df.dropna(subset=["y_5", "y_30", "y_60"])
    
    print("Splitting with purge gap...")
    purge_windows = int(60 / config.data.delta_t_seconds)
    splits = split_with_purge(full_df, "source_file", purge_windows)
    
    # Feature Scaling (Fit only on Train)
    print("Scaling features...")
    exclude_cols = ["window_id", "timestamp", "Label", "mapped_stage", "source_file", "packet_features_available"]
    target_cols = [c for c in full_df.columns if c.startswith("next_") or c.startswith("y_") or c.startswith("future_")]
    feature_cols = [c for c in full_df.columns if c not in exclude_cols and c not in target_cols]
    
    scaler = StandardScaler()
    
    train_df = splits["train"].copy()
    val_df = splits["val"].copy()
    test_df = splits["test"].copy()
    
    # Log transform heavy tailed features before scaling
    log_cols = ["total_packets", "total_fwd_bytes", "total_bwd_bytes", "mean_flow_duration", "retransmission_count", "payload_size_mean"]
    log_cols = [c for c in log_cols if c in feature_cols]
    
    for df in [train_df, val_df, test_df]:
        if df.empty: continue
        for col in log_cols:
            df[col] = np.log1p(df[col].clip(lower=0))
            
    if not train_df.empty:
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        if not val_df.empty:
             val_df[feature_cols] = scaler.transform(val_df[feature_cols])
        if not test_df.empty:
             test_df[feature_cols] = scaler.transform(test_df[feature_cols])
             
        # Save scaler stats
        import joblib
        joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
    
    # Save
    window_states_path = os.path.join(args.output_dir, "window_states.parquet")
    window_states.to_parquet(window_states_path, index=False)
    
    train_df.to_parquet(os.path.join(args.output_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.output_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.output_dir, "test.parquet"), index=False)
    
    print("\n=== DATASET CREATION REPORT ===")
    print(f"Total Window States   : {len(window_states)}")
    print(f"Train samples         : {len(train_df)}")
    print(f"Validation samples    : {len(val_df)}")
    print(f"Test samples          : {len(test_df)}")
    print(f"Feature Dimension     : {len(feature_cols)}")
    
    packet_cov = (window_states["packet_features_available"] == 1).mean() * 100
    print(f"Packet coverage       : {packet_cov:.2f}%")
    if len(full_df):
        print(f"+30s positive rate    : {full_df['y_30'].mean()*100:.2f}%")

if __name__ == "__main__":
    main()
