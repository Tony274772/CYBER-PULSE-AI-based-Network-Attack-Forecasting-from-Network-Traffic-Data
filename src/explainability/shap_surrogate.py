"""XGBoost surrogate + SHAP (Section 7.8.3).

Train a small ``XGBClassifier`` on (flattened window-level tabular features →
GT-RSSM's infiltration_head sigmoid output as a soft regression-style target).
Then compute ``shap.TreeExplainer`` values on this surrogate for per-window
explanations.

This is the standard, defensible workaround for "SHAP directly on the RSSM is
impractical" — the surrogate imitates the neural model, not the ground truth,
so the SHAP values explain *what the model is seeing*, not the data per se.
"""
from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np

from src.features.schema import EDGE_FEATURE_COLUMNS


def train_surrogate(
    X_train: np.ndarray,
    y_soft: np.ndarray,
    save_path: str | None = None,
    n_estimators: int = 200,
    max_depth: int = 5,
) -> Any:
    """Train an XGBoost surrogate on flattened edge features → soft infiltration
    probability from the GT-RSSM model.

    ``y_soft``: the sigmoid output from the model (float in [0,1]), used as a
    regression target (XGBRegressor) so the surrogate imitates the model rather
    than the ground truth.
    """
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_soft)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as fh:
            pickle.dump(model, fh)
    return model


def load_surrogate(path: str) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def compute_shap_values(
    surrogate,
    X: np.ndarray,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute SHAP values for the given samples using TreeExplainer.

    Returns:
      * ``shap_values``: [n_samples, d_e] numpy array
      * ``top_features``: list of {feature, mean_abs_shap, direction} (top-k)
      * ``base_value``: expected model output
    """
    import shap

    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[0] if len(shap_values) > 0 else np.zeros_like(X)
    shap_values = np.asarray(shap_values, dtype=float)

    # Per-feature mean absolute SHAP
    mean_abs = np.abs(shap_values).mean(axis=0) if shap_values.ndim > 1 else np.abs(shap_values)
    mean_signed = shap_values.mean(axis=0) if shap_values.ndim > 1 else shap_values
    top_idx = np.argsort(mean_abs)[::-1][:top_k]

    top_features = []
    for idx in top_idx:
        feat = EDGE_FEATURE_COLUMNS[idx] if idx < len(EDGE_FEATURE_COLUMNS) else f"feat_{idx}"
        direction = "positive" if mean_signed[idx] > 0 else "negative"
        top_features.append({
            "feature": feat,
            "mean_abs_shap": float(mean_abs[idx]),
            "direction": direction,
        })

    base_value = float(explainer.expected_value)
    if isinstance(explainer.expected_value, np.ndarray):
        base_value = float(explainer.expected_value[0])

    return {
        "shap_values": shap_values,
        "top_features": top_features,
        "base_value": base_value,
    }


def explain_window(
    surrogate,
    edge_features: np.ndarray,
    top_k: int = 5,
) -> dict[str, Any]:
    """Explain a single window's edge features.

    ``edge_features``: [n_edges, d_e] — the non-zero edge feature rows for one
    window (flattened from the N×N dense tensor by selecting real edges).
    """
    if edge_features.ndim == 1:
        edge_features = edge_features.reshape(1, -1)
    return compute_shap_values(surrogate, edge_features, top_k=top_k)


def generate_explanation_sentence(
    top_features: list[dict],
    predicted_stage: str,
    ip: str | None = None,
) -> str:
    """Build the templated natural-language explanation sentence (Section 7.11
    dashboard requirement #6).

    Example: "High SYN-flag ratio and a rising distinct-port count on 192.168.1.5
    are driving this Impact_DoS prediction."
    """
    if not top_features:
        return f"No significant features driving this {predicted_stage} prediction."

    # Human-readable feature name mapping
    _HUMAN_NAMES = {
        "flag_syn_count": "SYN-flag count",
        "flag_ack_count": "ACK-flag count",
        "flag_fin_count": "FIN-flag count",
        "flag_rst_count": "RST-flag count",
        "flag_psh_count": "PSH-flag count",
        "flag_urg_count": "URG-flag count",
        "flow_duration_sum": "flow duration",
        "tot_fwd_pkts": "forward packet count",
        "tot_bwd_pkts": "backward packet count",
        "tot_fwd_bytes": "forward byte volume",
        "tot_bwd_bytes": "backward byte volume",
        "iat_mean": "inter-arrival time",
        "iat_std": "IAT variability",
        "fwd_bwd_byte_ratio": "forward/backward byte ratio",
        "fwd_bwd_pkt_ratio": "forward/backward packet ratio",
        "flow_bytes_per_sec": "byte rate",
        "flow_pkts_per_sec": "packet rate",
        "distinct_dst_port_count": "distinct destination port count",
        "retransmission_count": "retransmission count",
        "port_scan_sequential_score": "sequential port-scan score",
        "port_scan_randomised_score": "randomised port-scan score",
        "ttl_mean": "TTL mean",
        "payload_size_mean": "payload size",
        "tcp_window_mean": "TCP window size",
    }

    parts = []
    for i, feat in enumerate(top_features[:3]):
        name = _HUMAN_NAMES.get(feat["feature"], feat["feature"].replace("_", " "))
        qualifier = "high" if feat["direction"] == "positive" else "low"
        parts.append(f"{qualifier} {name}")

    feature_desc = " and ".join(parts) if len(parts) <= 2 else \
        ", ".join(parts[:-1]) + ", and " + parts[-1]

    host_part = f" on `{ip}`" if ip else ""
    return (f"{feature_desc.capitalize()}{host_part} "
            f"are driving this **{predicted_stage}** prediction.")
