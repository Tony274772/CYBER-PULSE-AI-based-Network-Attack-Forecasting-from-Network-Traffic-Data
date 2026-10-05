"""Train and evaluate baseline models (Logistic Regression & XGBoost)."""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib

from src.config import Config
from src.evaluation.metrics import calculate_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    config = Config.load(args.config)
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(config.paths.metrics_dir, exist_ok=True)
    
    print("Loading datasets...")
    train_df = pd.read_parquet(os.path.join(config.paths.processed_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(config.paths.processed_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(config.paths.processed_dir, "test.parquet"))
    
    exclude_cols = ["window_id", "timestamp", "Label", "mapped_stage", "source_file", "packet_features_available"]
    target_cols = [c for c in train_df.columns if c.startswith("next_") or c.startswith("y_") or c.startswith("future_")]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and c not in target_cols]
    
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["y_30"].values
    
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df["y_30"].values
    
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["y_30"].values
    
    print("Training Logistic Regression...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.training.seed)
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(config.paths.checkpoint_dir, "baseline_lr.pkl"))
    
    val_probs_lr = lr.predict_proba(X_val)[:, 1]
    
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_val, val_probs_lr)
    idx = np.where(fpr <= config.thresholding.target_validation_fpr)[0]
    thresh_lr = thresholds[idx[-1]] if len(idx) > 0 else 0.5
    
    test_probs_lr = lr.predict_proba(X_test)[:, 1]
    lr_metrics = calculate_metrics(y_test, test_probs_lr, threshold=thresh_lr)
    
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        scale_pos_weight=max(1, (len(y_train)-y_train.sum())/max(1, y_train.sum())),
        random_state=config.training.seed
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, os.path.join(config.paths.checkpoint_dir, "baseline_xgb.pkl"))
    
    val_probs_xgb = xgb_model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, val_probs_xgb)
    idx = np.where(fpr <= config.thresholding.target_validation_fpr)[0]
    thresh_xgb = thresholds[idx[-1]] if len(idx) > 0 else 0.5
    
    test_probs_xgb = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = calculate_metrics(y_test, test_probs_xgb, threshold=thresh_xgb)
    
    print("\n=== BASELINE RESULTS (+30s TEST SET) ===")
    print(f"Logistic Regression: PR-AUC={lr_metrics['pr_auc']:.4f}, F1={lr_metrics['f1']:.4f}, FPR={lr_metrics['fpr']:.4f}")
    print(f"XGBoost            : PR-AUC={xgb_metrics['pr_auc']:.4f}, F1={xgb_metrics['f1']:.4f}, FPR={xgb_metrics['fpr']:.4f}")
    
    # Save results
    results = {
        "LogisticRegression": lr_metrics,
        "XGBoost": xgb_metrics
    }
    import json
    with open(os.path.join(config.paths.metrics_dir, "baseline_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
