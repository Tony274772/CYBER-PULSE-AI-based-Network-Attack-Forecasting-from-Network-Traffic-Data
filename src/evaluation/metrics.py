"""Evaluation metrics and lead-time calculation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss


def calculate_metrics(y_true, y_prob, threshold=0.5):
    """Calculates forecasting metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    # Handle cases where all true labels are negative
    if len(np.unique(y_true)) == 1:
        roc = np.nan
        pr_auc = np.nan
    else:
        roc = roc_auc_score(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # FPR
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "fpr": float(fpr),
        "brier": float(brier)
    }

def calculate_lead_time(df: pd.DataFrame, alert_threshold: float) -> dict:
    """Calculate mean and median lead time from forecasting probabilities.
    
    Assumes df has 'timestamp', 'Label', 'p_attack_30', 'source_file'.
    """
    lead_times = []
    
    for _, group in df.groupby("source_file"):
        group = group.sort_values("timestamp")
        labels = group["Label"].values
        probs = group["p_attack_30"].values
        times = group["timestamp"].values
        
        # Identify attack episodes (contiguous blocks of non-benign)
        in_attack = False
        attack_start_time = None
        
        for i in range(len(labels)):
            if labels[i] != "BENIGN":
                if not in_attack:
                    in_attack = True
                    attack_start_time = times[i]
                    
                    # Look back before i to find the first alert
                    alert_time = None
                    for j in range(i-1, -1, -1):
                        if probs[j] >= alert_threshold:
                            alert_time = times[j]
                        else:
                            break # contiguous alert before attack
                            
                    if alert_time is not None:
                        # calculate lead time in seconds
                        lt = (attack_start_time - alert_time) / np.timedelta64(1, 's')
                        lead_times.append(lt)
                    else:
                        # missed or late
                        lead_times.append(0.0)
            else:
                in_attack = False
                
    if not lead_times:
        return {"mean_lead_time": 0.0, "median_lead_time": 0.0}
        
    return {
        "mean_lead_time": float(np.mean(lead_times)),
        "median_lead_time": float(np.median(lead_times))
    }
