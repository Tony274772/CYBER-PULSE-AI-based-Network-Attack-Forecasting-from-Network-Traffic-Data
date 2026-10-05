"""Sequence dataset for training the Temporal World Model."""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class TemporalDataset(Dataset):
    def __init__(self, df_path, config):
        """Loads a parquet file and prepares historical sequences."""
        self.df = pd.read_parquet(df_path)
        self.config = config
        self.history_windows = config.data.history_windows
        
        # Columns
        exclude_cols = ["window_id", "timestamp", "Label", "mapped_stage", "source_file", "packet_features_available"]
        target_cols = [c for c in self.df.columns if c.startswith("next_") or c.startswith("y_") or c.startswith("future_")]
        
        # Assume first 28 features are flow, remaining 11 + availability are packet
        feature_cols = [c for c in self.df.columns if c not in exclude_cols and c not in target_cols]
        
        self.flow_cols = feature_cols[:28]
        # Includes packet_features_available at the end or we can just pass the next 12
        self.packet_cols = feature_cols[28:39] + ["packet_features_available"]
        
        self.X_flow = self.df[self.flow_cols].values.astype(np.float32)
        self.X_packet = self.df[self.packet_cols].values.astype(np.float32)
        
        # Targets
        next_state_cols = [f"next_{c}" for c in feature_cols if f"next_{c}" in target_cols]
        # Include packet availability in next state if possible, or just ignore it.
        # Actually next_state_head predicts flow + packet input dim
        self.Y_next_state = self.df[next_state_cols].values.astype(np.float32)
        
        self.Y_5 = self.df["y_5"].values.astype(np.float32)
        self.Y_30 = self.df["y_30"].values.astype(np.float32)
        self.Y_60 = self.df["y_60"].values.astype(np.float32)
        
        stage_order = config.stage_order
        stage_to_idx = {s: i for i, s in enumerate(stage_order)}
        self.Y_stage = self.df["future_stage_30"].map(stage_to_idx).fillna(0).values.astype(np.int64)

        # Build valid indices where we have enough history
        # (Assuming the dataframe is sorted and consecutive)
        # Actually we should group by source_file to not cross file boundaries
        self.valid_indices = []
        for _, group in self.df.groupby("source_file"):
            idx = group.index.values
            if len(idx) >= self.history_windows:
                # the sequence is idx[i - history_windows + 1 : i + 1]
                for i in range(self.history_windows - 1, len(idx)):
                    self.valid_indices.append(idx[i])
                    
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.history_windows + 1
        
        seq_flow = self.X_flow[start_idx : end_idx + 1]
        seq_packet = self.X_packet[start_idx : end_idx + 1]
        
        y_next_state = self.Y_next_state[end_idx]
        y_5 = self.Y_5[end_idx]
        y_30 = self.Y_30[end_idx]
        y_60 = self.Y_60[end_idx]
        y_stage = self.Y_stage[end_idx]
        
        return {
            "seq_flow": torch.tensor(seq_flow),
            "seq_packet": torch.tensor(seq_packet),
            "y_next_state": torch.tensor(y_next_state),
            "y_5": torch.tensor(y_5),
            "y_30": torch.tensor(y_30),
            "y_60": torch.tensor(y_60),
            "y_stage": torch.tensor(y_stage)
        }
