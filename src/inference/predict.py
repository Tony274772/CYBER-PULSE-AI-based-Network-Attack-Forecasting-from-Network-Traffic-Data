"""Inference pipeline for Temporal World Model."""

import os
import joblib
import json
import torch
import numpy as np
import pandas as pd

from src.config import Config
from src.model.temporal_world_model import TemporalWorldModel
from src.explainability.integrated_gradients import explain_prediction

# Import preprocessing
from data_prep.clean_cic2017 import clean_one_file
from src.features.build_temporal_dataset import create_window_states


def predict(input_path: str, config_path="configs/default.yaml") -> dict:
    """End-to-end inference for a CSV or PCAP file."""
    config = Config.load(config_path)
    
    # Check what kind of file we have
    is_csv = input_path.lower().endswith(".csv")
    is_pcap = input_path.lower().endswith(".pcap")
    
    if not is_csv and not is_pcap:
        raise ValueError("Unsupported input format. Please upload a .csv or .pcap file.")
        
    flow_df = pd.DataFrame()
    packet_df = pd.DataFrame()
    
    if is_csv:
        flow_df = clean_one_file(input_path)
        # We don't have true packet data
        packet_df = pd.DataFrame()
    elif is_pcap:
        # In a real app we would run extract_packet_features on the uploaded PCAP.
        # Since this is a hackathon, we would either simulate or use the existing logic.
        # But for now, we will return an error since PCAP processing requires flow alignment usually,
        # or we just process it as packet-only if model allows.
        raise NotImplementedError("Standalone PCAP inference requires flow CSV pairing in this version.")

    # Apply windowing
    window_states = create_window_states(flow_df, packet_df, config.data.delta_t_seconds)
    
    if len(window_states) < config.data.history_windows:
         raise ValueError(f"Not enough data to form a {config.data.history_windows}-window history.")
         
    # Take the latest history
    latest_history = window_states.tail(config.data.history_windows).copy()
    
    # Scale features
    exclude_cols = ["window_id", "timestamp", "Label", "mapped_stage", "source_file", "packet_features_available"]
    target_cols = [c for c in window_states.columns if c.startswith("next_") or c.startswith("y_") or c.startswith("future_")]
    feature_cols = [c for c in window_states.columns if c not in exclude_cols and c not in target_cols]
    
    scaler_path = os.path.join(config.paths.processed_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        candidates = [
            os.path.join(config.paths.checkpoint_dir, "scaler.pkl"),
            "data/processed_random/cic2017/scaler.pkl",
            "Data/processed_random/cic2017/scaler.pkl",
            "data/processed/cic2017/scaler.pkl",
            "Data/processed/cic2017/scaler.pkl",
            "checkpoints_random/scaler.pkl",
            "checkpoints/scaler.pkl"
        ]
        for c in candidates:
            if os.path.exists(c):
                scaler_path = c
                break
    scaler = joblib.load(scaler_path)
    
    log_cols = ["total_packets", "total_fwd_bytes", "total_bwd_bytes", "mean_flow_duration", "retransmission_count", "payload_size_mean"]
    log_cols = [c for c in log_cols if c in feature_cols]
    
    for col in log_cols:
        latest_history[col] = np.log1p(latest_history[col].clip(lower=0))
        
    scaled_feats = scaler.transform(latest_history[feature_cols])
    
    flow_cols = feature_cols[:28]
    packet_cols = feature_cols[28:39]
    
    flow_indices = [feature_cols.index(c) for c in flow_cols]
    packet_indices = [feature_cols.index(c) for c in packet_cols]
    
    x_flow = scaled_feats[:, flow_indices]
    x_packet_scaled = scaled_feats[:, packet_indices]
    
    if "packet_features_available" in latest_history.columns:
        avail = latest_history["packet_features_available"].values.reshape(-1, 1).astype(np.float32)
    else:
        avail = np.zeros((len(latest_history), 1), dtype=np.float32)
        
    x_packet = np.hstack([x_packet_scaled, avail])
    
    # Model inference
    device = torch.device("cpu")
    ckpt_path = os.path.join(config.paths.checkpoint_dir, "temporal_world_model.pt")
    if not os.path.exists(ckpt_path):
        for c in ["checkpoints_random/temporal_world_model.pt", "checkpoints/temporal_world_model.pt"]:
            if os.path.exists(c):
                ckpt_path = c
                break
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model = TemporalWorldModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    seq_flow = torch.tensor(x_flow, dtype=torch.float32).unsqueeze(0).to(device)
    seq_packet = torch.tensor(x_packet, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(seq_flow, seq_packet)
        p_attack = torch.sigmoid(out["attack_logits"]).squeeze(0).numpy()
        p_stage = torch.softmax(out["stage_logits"], dim=-1).squeeze(0).numpy()
        
    stage_idx = np.argmax(p_stage)
    stage_prob = p_stage[stage_idx]
    stage_name = config.stage_order[stage_idx]
    
    # Recursive Rollout
    forecast_timeline = []
    h_curr = out["h_t"]
    
    # We will rollout 12 steps (60 seconds)
    with torch.no_grad():
        for step in range(12):
            next_state_pred = model.next_state_head(h_curr)
            
            x_f_next = next_state_pred[:, :28]
            x_p_next = next_state_pred[:, 28:]
            
            h_curr = model.forward_step(x_f_next, x_p_next, h_curr)
            
            step_out = model.attack_head(h_curr)
            step_prob = torch.sigmoid(step_out).squeeze(0)[0].item() # +5s probability from the rolled out state
            forecast_timeline.append(float(step_prob))
            
    # Explanations
    attributions = explain_prediction(model, seq_flow, seq_packet, target_idx=1) # +30s horizon
    
    # Map back to feature names
    top_flow_idx = np.argsort(attributions["flow_importance"])[::-1][:5]
    top_packet_idx = np.argsort(attributions["packet_importance"])[::-1][:5]
    
    top_flow_feats = [flow_cols[i] for i in top_flow_idx]
    all_packet_names = packet_cols + ["packet_features_available"]
    top_packet_feats = [all_packet_names[i] for i in top_packet_idx if i < len(all_packet_names)]
    
    current_state = latest_history.iloc[-1].to_dict()
    
    return {
        "windows": len(window_states),
        "current_state": {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in current_state.items()},
        "forecast": {
            "5s": float(p_attack[0]),
            "30s": float(p_attack[1]),
            "60s": float(p_attack[2])
        },
        "forecast_timeline": forecast_timeline,
        "stage_forecast": {
            "stage": stage_name,
            "probability": float(stage_prob)
        },
        "top_flow_features": top_flow_feats,
        "top_packet_features": top_packet_feats,
        "packet_features_available": bool(current_state.get("packet_features_available", 0) > 0),
        "alert_threshold": float(ckpt["best_threshold"])
    }
