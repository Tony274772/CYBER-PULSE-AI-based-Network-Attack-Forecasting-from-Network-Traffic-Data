"""Unified inference entry point (Section 7.10).

Single function ``predict(input_path, k_steps)`` used by **both** the Streamlit
dashboard and the optional FastAPI wrapper — there is exactly one inference code
path in the whole repository, so results never disagree between the two UIs.

Loads ``checkpoints/gt_rssm_v1.pt`` once (cached at module level so repeated
calls in the Streamlit session don't reload weights), accepts a PCAP or CSV path,
runs it through the same feature pipeline as Section 7.4 (reusing
``build_dataset``'s functions directly), runs the model forward pass, runs
``imagine()`` from the last observed belief state, and returns a single dict.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.features.packet_features import PacketFeatureIndex
from src.features.schema import (
    EDGE_FEATURE_COLUMNS,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    NODE_FEATURES,
)
from src.features.windowing import (
    _epoch_seconds,
    _explode_to_windows,
    _aggregate_edges,
    _attach_packet_features,
    _node_features,
    _build_window,
    _severity_maps,
    reconstruct_dense,
)
from src.model.gt_rssm import GTRSSM

# ---- module-level cache for loaded model -----------------------------------
_CACHED_MODEL: GTRSSM | None = None
_CACHED_MODEL_PATH: str | None = None
_CACHED_CFG: Config | None = None
_CACHED_CKPT: dict | None = None

_DEFAULT_CONFIG = "configs/default.yaml"
_DEFAULT_CKPT = "checkpoints/gt_rssm_v1.pt"
_STAGE1_CKPT = "checkpoints/stage1.pt"


def _load_model(ckpt_path: str | None = None, cfg: Config | None = None,
                device: str = "cpu") -> tuple[GTRSSM, Config, dict]:
    """Load GT-RSSM from checkpoint, caching it for repeat calls."""
    global _CACHED_MODEL, _CACHED_MODEL_PATH, _CACHED_CFG, _CACHED_CKPT

    if ckpt_path is None:
        ckpt_path = _DEFAULT_CKPT if os.path.exists(_DEFAULT_CKPT) else _STAGE1_CKPT

    if _CACHED_MODEL is not None and _CACHED_MODEL_PATH == ckpt_path:
        return _CACHED_MODEL, _CACHED_CFG, _CACHED_CKPT

    if cfg is None:
        cfg = Config.load(_DEFAULT_CONFIG)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = GTRSSM(
        node_dim=ckpt.get("node_dim", NODE_FEATURE_DIM),
        edge_dim=ckpt.get("edge_dim", EDGE_FEATURE_DIM),
        n_stages=ckpt.get("n_stages", len(cfg.stage_order)),
        embed_dim=cfg.model.gat_embed_dim,
        heads=cfg.model.gat_heads,
        layers=cfg.model.gat_layers,
        h_dim=cfg.model.rssm_h_dim,
        z_dim=cfg.model.rssm_z_dim,
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    if ckpt.get("feature_stats"):
        model.set_feature_stats(ckpt["feature_stats"])
    model.to(device).eval()

    _CACHED_MODEL = model
    _CACHED_MODEL_PATH = ckpt_path
    _CACHED_CFG = cfg
    _CACHED_CKPT = ckpt
    return model, cfg, ckpt


# ---- CSV / PCAP → tidy DataFrame ------------------------------------------

def _load_input(input_path: str, cfg: Config) -> tuple[pd.DataFrame, bool]:
    """Load a CSV or PCAP into the tidy per-flow DataFrame the windowing needs.

    Returns (tidy_df, has_pcap_data).
    """
    ext = Path(input_path).suffix.lower()
    if ext in (".pcap", ".pcapng"):
        return _load_pcap(input_path, cfg), True
    elif ext in (".csv", ".parquet"):
        return _load_csv(input_path, cfg), False
    else:
        raise ValueError(f"unsupported file extension: {ext}")


def _load_csv(path: str, cfg: Config) -> pd.DataFrame:
    """Load a CIC-IDS2017-style CSV or a pre-processed parquet."""
    from src.ingestion.flow_csv_loader import load
    return load(path, cfg.paths.mitre_mapping, spread_minutes=True)


def _load_pcap(path: str, cfg: Config) -> pd.DataFrame:
    """Parse a PCAP and turn the raw packets into a tidy flow-like frame."""
    from src.ingestion.pcap_parser import parse_pcap

    pkt_df = parse_pcap(path)
    if pkt_df is None or pkt_df.empty:
        return pd.DataFrame()

    # Convert packet data to a flow-like dataframe for windowing
    from src.features.schema import FLOW_FEATURES, FLOW_AVAIL_FEATURES
    tidy = pd.DataFrame({
        "timestamp": pd.to_datetime(pkt_df["timestamp"], unit="s"),
        "src_ip": pkt_df["src_ip"].astype(str),
        "dst_ip": pkt_df["dst_ip"].astype(str),
        "src_port": pkt_df.get("src_port", 0),
        "dst_port": pkt_df.get("dst_port", 0),
        "protocol": pkt_df.get("protocol", 0),
        "raw_label": "UNKNOWN",
        "mapped_stage": "Benign",
        "dataset": "upload",
        "source_file": Path(path).name,
    })
    for c in FLOW_FEATURES:
        if c not in tidy.columns:
            tidy[c] = 0.0
    for c in FLOW_AVAIL_FEATURES:
        if c not in tidy.columns:
            tidy[c] = 0.0
    return tidy


# ---- windowing (one-shot, no sharding) ------------------------------------

def _window_input(tidy: pd.DataFrame, cfg: Config,
                  has_pcap: bool) -> list[dict]:
    """Run steps 1-7 of windowing on the input, returning a list of dense
    window dicts (no parquet I/O — everything in memory for the small
    inference-time input)."""
    stride = cfg.window_stride_seconds
    delta_t = cfg.data.delta_t_seconds
    n_max = cfg.data.n_max_nodes
    sev_rank, inv_sev = _severity_maps(cfg.stage_order)

    tidy = tidy.copy()
    tidy["_epoch"] = _epoch_seconds(tidy["timestamp"])
    t_min = float(tidy["_epoch"].min())
    t_max = float(tidy["_epoch"].max())
    n_windows = max(1, int(np.floor((t_max - t_min) / stride)) + 1)

    ex = _explode_to_windows(tidy, t_min, stride, delta_t)
    if ex.empty:
        return []

    edges = _aggregate_edges(ex, sev_rank)
    edges = _attach_packet_features(edges, None, t_min, False, stride, None)
    nodes = _node_features(ex)

    edges_by = {wid: g for wid, g in edges.groupby("window_id", sort=False)}
    nodes_by = {wid: g for wid, g in nodes.groupby(level="window_id", sort=False)}

    windows = []
    for wid in range(n_windows):
        start = t_min + wid * stride
        w = _build_window(wid, start, edges_by.get(wid), nodes_by.get(wid),
                          n_max, sev_rank, inv_sev)
        dense = {
            "window_id": wid,
            "window_start": start,
            "node_ips": list(w.node_ips),
            "n_nodes": len(w.node_ips),
            "n_edges": len(w.edge_src),
        }
        # Build dense tensors
        row = {
            "n_nodes": len(w.node_ips),
            "node_feats_flat": w.node_feats.reshape(-1).tolist(),
            "edge_src": [int(x) for x in w.edge_src],
            "edge_dst": [int(x) for x in w.edge_dst],
            "edge_feats_flat": w.edge_feats.reshape(-1).tolist(),
            "infil_label": w.infil_label,
            "stage_label": w.stage_label,
        }
        d = reconstruct_dense(row, n_max)
        dense.update(d)
        windows.append(dense)
    return windows


# ---- main inference function -----------------------------------------------

@torch.no_grad()
def predict(
    input_path: str,
    k_steps: tuple[int, ...] = (5, 10, 20),
    ckpt_path: str | None = None,
    config_path: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the full inference pipeline on a PCAP or CSV file.

    Returns a single dict:
      * ``windows``: per-window metadata
      * ``infiltration_timeline``: per-window sigmoid infiltration probability
      * ``stage_timeline``: per-window predicted MITRE stage index + probabilities
      * ``forecast``: dict keyed by K horizon → forecasted infil prob + stage probs
      * ``kl_surprise``: per-window KL divergence (anomaly signal)
      * ``attention_by_window``: dict window_id → attention info
      * ``top_features_by_window``: dict window_id → SHAP/IG top features
    """
    cfg = Config.load(config_path or _DEFAULT_CONFIG)
    model, cfg, ckpt = _load_model(ckpt_path, cfg, device)
    stage_order = cfg.stage_order

    # 1. Load and window the input
    tidy, has_pcap = _load_input(input_path, cfg)
    if tidy.empty:
        return _empty_result()
    windows = _window_input(tidy, cfg, has_pcap)
    if not windows:
        return _empty_result()

    W = len(windows)
    n_max = cfg.data.n_max_nodes

    # 2. Stack into sequence tensors
    node_feats = torch.zeros(1, W, n_max, NODE_FEATURE_DIM)
    edge_feats = torch.zeros(1, W, n_max, n_max, EDGE_FEATURE_DIM)
    adj_mask = torch.zeros(1, W, n_max, n_max)
    valid_mask = torch.zeros(1, W, n_max)

    for i, w in enumerate(windows):
        node_feats[0, i] = torch.from_numpy(w["node_feats"])
        edge_feats[0, i] = torch.from_numpy(w["edge_feats"])
        adj_mask[0, i] = torch.from_numpy(w["adjacency_mask"])
        valid_mask[0, i] = torch.from_numpy(w["node_valid_mask"])

    # 3. Forward pass
    out = model(node_feats.to(device), edge_feats.to(device),
                adj_mask.to(device), valid_mask.to(device))

    infil_probs = torch.sigmoid(out["infil_logit"])[0].cpu().numpy()  # [W]
    stage_probs = torch.softmax(out["mitre_logits"], dim=-1)[0].cpu().numpy()  # [W, 7]
    kl_vals = out["kl"][0].cpu().numpy()  # [W]
    pred_stages = stage_probs.argmax(axis=-1)  # [W]

    # 4. Build timelines
    infiltration_timeline = []
    stage_timeline = []
    window_meta = []
    for i, w in enumerate(windows):
        infiltration_timeline.append({
            "window_id": w["window_id"],
            "window_start": w["window_start"],
            "prob": float(infil_probs[i]),
        })
        stage_timeline.append({
            "window_id": w["window_id"],
            "predicted_stage_idx": int(pred_stages[i]),
            "predicted_stage": stage_order[pred_stages[i]] if pred_stages[i] < len(stage_order) else "Unknown",
            "probs": {s: float(stage_probs[i, j]) for j, s in enumerate(stage_order)},
        })
        window_meta.append({
            "window_id": w["window_id"],
            "window_start": w["window_start"],
            "n_nodes": w["n_nodes"],
            "n_edges": w["n_edges"],
            "node_ips": w["node_ips"],
            "gt_infil": w["infil_label"],
            "gt_stage": w["stage_label"],
        })

    # 5. K-step imagination from last belief
    forecast = {}
    last_h = out["last_h"]
    last_z = out["last_z"]
    for K in k_steps:
        traj = model.imagine(last_h, last_z, K)
        forecast[K] = {
            "infiltration_probs": [float(t["infiltration_prob"][0]) for t in traj],
            "stage_probs": [
                {s: float(t["stage_probs"][0, j]) for j, s in enumerate(stage_order)}
                for t in traj
            ],
        }

    # 6. Attention export for each window (lightweight)
    attention_by_window = {}
    try:
        from src.explainability.attention_export import export_attention, top_suspicious_hosts
        for i, w in enumerate(windows):
            nf_t = node_feats[:, i:i+1].squeeze(1).to(device)
            ef_t = edge_feats[:, i:i+1].squeeze(1).to(device)
            am_t = adj_mask[:, i:i+1].squeeze(1).to(device)
            vm_t = valid_mask[:, i:i+1].squeeze(1).to(device)
            attn_info = export_attention(model, nf_t, ef_t, am_t, vm_t, w["node_ips"])
            attn_info["suspicious_hosts"] = top_suspicious_hosts(attn_info)
            # Don't serialize the full numpy matrix
            attn_info.pop("attn_matrix", None)
            attention_by_window[w["window_id"]] = attn_info
    except Exception:
        pass

    # 7. Top features per window (SHAP surrogate if available)
    top_features_by_window = {}
    try:
        surr_path = os.path.join(cfg.paths.checkpoint_dir, "xgb_surrogate.pkl")
        if os.path.exists(surr_path):
            from src.explainability.shap_surrogate import load_surrogate, explain_window
            surrogate = load_surrogate(surr_path)
            for i, w in enumerate(windows):
                ef_dense = edge_feats[0, i].numpy()  # [N, N, d_e]
                am_dense = adj_mask[0, i].numpy()     # [N, N]
                real_edges = ef_dense[am_dense > 0]   # [n_edges, d_e]
                if real_edges.shape[0] > 0:
                    shap_info = explain_window(surrogate, real_edges, top_k=5)
                    shap_info.pop("shap_values", None)
                    top_features_by_window[w["window_id"]] = shap_info
    except Exception:
        pass

    return {
        "windows": window_meta,
        "infiltration_timeline": infiltration_timeline,
        "stage_timeline": stage_timeline,
        "forecast": forecast,
        "kl_surprise": [{"window_id": windows[i]["window_id"],
                         "kl": float(kl_vals[i])} for i in range(W)],
        "attention_by_window": attention_by_window,
        "top_features_by_window": top_features_by_window,
        "stage_order": stage_order,
    }


def _empty_result() -> dict:
    return {
        "windows": [], "infiltration_timeline": [], "stage_timeline": [],
        "forecast": {}, "kl_surprise": [],
        "attention_by_window": {}, "top_features_by_window": [],
        "stage_order": [],
    }
