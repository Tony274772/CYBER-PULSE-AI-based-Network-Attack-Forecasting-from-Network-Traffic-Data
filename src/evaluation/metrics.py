"""Shared evaluation metrics (Section 7.9).

Threshold-based classification metrics plus the probabilistic/forecasting
metrics the full evaluation needs. Kept dependency-light (sklearn + numpy) so
both the LR baseline (7.7) and ``run_eval`` (7.9) import from here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Precision / recall / F1 at ``threshold`` plus ROC-AUC, PR-AUC, Brier.

    AUC metrics are skipped (NaN) when only one class is present in ``y_true``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n": int(y_true.size),
        "positives": int(y_true.sum()),
    }
    if y_true.min() != y_true.max():
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        out["brier"] = float(brier_score_loss(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        out["brier"] = float("nan")
    return out


def threshold_for_target_fpr(y_true, y_prob, target_fpr: float = 0.05) -> float:
    """Smallest score threshold whose false-positive rate on ``y_true`` is
    <= ``target_fpr`` (Section 7.9 lead-time alert threshold)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    neg = y_prob[y_true == 0]
    if neg.size == 0:
        return 0.5
    # threshold at the (1 - target_fpr) quantile of negative scores
    return float(np.quantile(neg, 1.0 - target_fpr))


def format_metrics_row(name: str, m: dict) -> str:
    """One markdown table row: name | P | R | F1 | ROC-AUC | PR-AUC | Brier."""
    def f(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"
    return (f"| {name} | {f(m.get('precision'))} | {f(m.get('recall'))} | "
            f"{f(m.get('f1'))} | {f(m.get('roc_auc'))} | {f(m.get('pr_auc'))} | "
            f"{f(m.get('brier'))} |")


METRICS_TABLE_HEADER = (
    "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |\n"
    "|---|---|---|---|---|---|---|"
)
