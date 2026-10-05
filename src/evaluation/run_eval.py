"""Run evaluation on baseline and Temporal GRU models."""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import joblib

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import Config
from src.evaluation.metrics import calculate_metrics, calculate_lead_time
from src.training.dataset import TemporalDataset
from src.model.temporal_world_model import TemporalWorldModel

def evaluate_model(config, model, dataloader, device):
    model.eval()
    all_y_30 = []
    all_p_30 = []
    
    with torch.no_grad():
        for batch in dataloader:
            seq_flow = batch["seq_flow"].to(device)
            seq_packet = batch["seq_packet"].to(device)
            y_30 = batch["y_30"].to(device)
            
            out = model(seq_flow, seq_packet)
            p_attack = torch.sigmoid(out["attack_logits"])
            p_30 = p_attack[:, 1]
            
            all_y_30.extend(y_30.cpu().numpy())
            all_p_30.extend(p_30.cpu().numpy())
            
    return np.array(all_y_30), np.array(all_p_30)

def main():
    config = Config.load("configs/default.yaml")
    device = torch.device("cpu")
    
    # Load baselines
    lr_model = joblib.load(os.path.join(config.paths.checkpoint_dir, "baseline_lr.pkl"))
    xgb_model = joblib.load(os.path.join(config.paths.checkpoint_dir, "baseline_xgb.pkl"))
    
    test_df = pd.read_parquet(os.path.join(config.paths.processed_dir, "test.parquet"))
    exclude_cols = ["window_id", "timestamp", "Label", "mapped_stage", "source_file", "packet_features_available"]
    target_cols = [c for c in test_df.columns if c.startswith("next_") or c.startswith("y_") or c.startswith("future_")]
    feature_cols = [c for c in test_df.columns if c not in exclude_cols and c not in target_cols]
    
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["y_30"].values
    
    p_test_lr = lr_model.predict_proba(X_test)[:, 1]
    p_test_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # Base baseline thresholds at 0.5 for now (could load actual threshold)
    lr_metrics = calculate_metrics(y_test, p_test_lr)
    xgb_metrics = calculate_metrics(y_test, p_test_xgb)
    
    # Temporal GRU
    test_ds = TemporalDataset(os.path.join(config.paths.processed_dir, "test.parquet"), config)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False)
    
    ckpt_path = os.path.join(config.paths.checkpoint_dir, "temporal_world_model.pt")
    gru_metrics = {}
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        gru = TemporalWorldModel(config).to(device)
        gru.load_state_dict(ckpt["model_state_dict"])
        
        y_test_gru, p_test_gru = evaluate_model(config, gru, test_loader, device)
        gru_metrics = calculate_metrics(y_test_gru, p_test_gru, threshold=ckpt.get("best_threshold", 0.5))
        
        # calculate lead time requires full df mapping, omitted here for brevity since it needs rolling df
    else:
        gru_metrics = {"pr_auc": 0.0, "f1": 0.0, "fpr": 0.0, "precision": 0.0, "recall": 0.0, "brier": 0.0}
        
    results_md = f"""# Final Evaluation Results

## Model Comparison (+30s Horizon)

| Model | PR-AUC | F1 | Precision | Recall | FPR | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | {lr_metrics['pr_auc']:.4f} | {lr_metrics['f1']:.4f} | {lr_metrics['precision']:.4f} | {lr_metrics['recall']:.4f} | {lr_metrics['fpr']:.4f} | {lr_metrics['brier']:.4f} |
| XGBoost | {xgb_metrics['pr_auc']:.4f} | {xgb_metrics['f1']:.4f} | {xgb_metrics['precision']:.4f} | {xgb_metrics['recall']:.4f} | {xgb_metrics['fpr']:.4f} | {xgb_metrics['brier']:.4f} |
| Temporal GRU | {gru_metrics['pr_auc']:.4f} | {gru_metrics['f1']:.4f} | {gru_metrics['precision']:.4f} | {gru_metrics['recall']:.4f} | {gru_metrics['fpr']:.4f} | {gru_metrics['brier']:.4f} |
"""
    
    os.makedirs(config.paths.metrics_dir, exist_ok=True)
    with open(os.path.join(config.paths.metrics_dir, "results.md"), "w") as f:
        f.write(results_md)
        
    print("Evaluation Complete. Results saved to metrics/results.md")

if __name__ == "__main__":
    main()
