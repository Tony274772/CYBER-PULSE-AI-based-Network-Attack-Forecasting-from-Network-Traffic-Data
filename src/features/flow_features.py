"""Per-window flow-feature aggregation for V2.

Produces a compact network state vector per time window from raw flow rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_flows(window_df: pd.DataFrame) -> pd.Series:
    """Aggregate tidy flow rows (one window) into a single network state vector.
    
    Returns a pandas Series with the 29 required flow features.
    """
    if window_df.empty:
        # Return zeros for all features if no flows in this window
        cols = [
            "flow_count", "total_packets", "total_fwd_packets", "total_bwd_packets",
            "total_fwd_bytes", "total_bwd_bytes", "total_flow_duration", "mean_flow_duration",
            "mean_iat", "std_iat", "min_iat", "max_iat", "syn_count", "ack_count",
            "fin_count", "rst_count", "psh_count", "urg_count", "mean_flow_bytes_per_sec",
            "mean_flow_packets_per_sec", "fwd_bwd_byte_ratio", "fwd_bwd_packet_ratio",
            "unique_source_ips", "unique_destination_ips", "unique_source_ports",
            "unique_destination_ports", "unique_protocols", "destination_port_entropy",
            "sequential_port_score"
        ]
        return pd.Series(0.0, index=cols)

    # Convert commonly needed columns to numeric
    fwd_pkts = pd.to_numeric(window_df.get("Total Fwd Packets", 0), errors="coerce").fillna(0)
    bwd_pkts = pd.to_numeric(window_df.get("Total Backward Packets", 0), errors="coerce").fillna(0)
    tot_pkts = fwd_pkts + bwd_pkts
    
    fwd_bytes = pd.to_numeric(window_df.get("Total Length of Fwd Packets", 0), errors="coerce").fillna(0)
    bwd_bytes = pd.to_numeric(window_df.get("Total Length of Bwd Packets", 0), errors="coerce").fillna(0)
    
    duration = pd.to_numeric(window_df.get("Flow Duration", 0), errors="coerce").fillna(0)
    
    syn = pd.to_numeric(window_df.get("SYN Flag Count", 0), errors="coerce").fillna(0)
    ack = pd.to_numeric(window_df.get("ACK Flag Count", 0), errors="coerce").fillna(0)
    fin = pd.to_numeric(window_df.get("FIN Flag Count", 0), errors="coerce").fillna(0)
    rst = pd.to_numeric(window_df.get("RST Flag Count", 0), errors="coerce").fillna(0)
    psh = pd.to_numeric(window_df.get("PSH Flag Count", 0), errors="coerce").fillna(0)
    urg = pd.to_numeric(window_df.get("URG Flag Count", 0), errors="coerce").fillna(0)
    
    iat_mean = pd.to_numeric(window_df.get("Flow IAT Mean", 0), errors="coerce").fillna(0)
    iat_std = pd.to_numeric(window_df.get("Flow IAT Std", 0), errors="coerce").fillna(0)
    iat_min = pd.to_numeric(window_df.get("Flow IAT Min", 0), errors="coerce").fillna(0)
    iat_max = pd.to_numeric(window_df.get("Flow IAT Max", 0), errors="coerce").fillna(0)
    
    # Ratios (from aggregated)
    total_fwd_bytes = fwd_bytes.sum()
    total_bwd_bytes = bwd_bytes.sum()
    total_fwd_pkts = fwd_pkts.sum()
    total_bwd_pkts = bwd_pkts.sum()
    
    fwd_bwd_byte_ratio = total_fwd_bytes / (total_bwd_bytes + 1.0)
    fwd_bwd_packet_ratio = total_fwd_pkts / (total_bwd_pkts + 1.0)
    
    # Rates
    safe_dur = duration.sum() / 1e6 # assuming microsecond duration typically
    safe_dur = max(safe_dur, 1e-6)
    tot_bytes_all = total_fwd_bytes + total_bwd_bytes
    tot_pkts_all = total_fwd_pkts + total_bwd_pkts
    mean_flow_bytes_per_sec = tot_bytes_all / safe_dur
    mean_flow_packets_per_sec = tot_pkts_all / safe_dur
    
    # Diversity
    unique_src_ips = window_df.get("Source IP", pd.Series()).nunique()
    unique_dst_ips = window_df.get("Destination IP", pd.Series()).nunique()
    unique_src_ports = window_df.get("Source Port", pd.Series()).nunique()
    unique_dst_ports = window_df.get("Destination Port", pd.Series()).nunique()
    unique_protocols = window_df.get("Protocol", pd.Series()).nunique()
    
    # Entropy & Scan Score
    dst_ports = window_df.get("Destination Port", pd.Series()).dropna().astype(int)
    if len(dst_ports) > 0:
        counts = dst_ports.value_counts(normalize=True)
        destination_port_entropy = float(-(counts * np.log2(counts + 1e-9)).sum())
        
        sorted_ports = sorted(set(dst_ports))
        if len(sorted_ports) > 1:
            diffs = np.diff(sorted_ports)
            sequential_port_score = float(np.mean(diffs == 1))
        else:
            sequential_port_score = 0.0
    else:
        destination_port_entropy = 0.0
        sequential_port_score = 0.0

    return pd.Series({
        "flow_count": float(len(window_df)),
        "total_packets": float(tot_pkts_all),
        "total_fwd_packets": float(total_fwd_pkts),
        "total_bwd_packets": float(total_bwd_pkts),
        "total_fwd_bytes": float(total_fwd_bytes),
        "total_bwd_bytes": float(total_bwd_bytes),
        "total_flow_duration": float(duration.sum()),
        "mean_flow_duration": float(duration.mean()),
        "mean_iat": float((iat_mean * (duration + 1e-9)).sum() / (duration.sum() + 1e-9)), # Duration weighted
        "std_iat": float(iat_std.mean()),
        "min_iat": float(iat_min.min()),
        "max_iat": float(iat_max.max()),
        "syn_count": float(syn.sum()),
        "ack_count": float(ack.sum()),
        "fin_count": float(fin.sum()),
        "rst_count": float(rst.sum()),
        "psh_count": float(psh.sum()),
        "urg_count": float(urg.sum()),
        "mean_flow_bytes_per_sec": float(mean_flow_bytes_per_sec),
        "mean_flow_packets_per_sec": float(mean_flow_packets_per_sec),
        "fwd_bwd_byte_ratio": float(fwd_bwd_byte_ratio),
        "fwd_bwd_packet_ratio": float(fwd_bwd_packet_ratio),
        "unique_source_ips": float(unique_src_ips),
        "unique_destination_ips": float(unique_dst_ips),
        "unique_source_ports": float(unique_src_ports),
        "unique_destination_ports": float(unique_dst_ports),
        "unique_protocols": float(unique_protocols),
        "destination_port_entropy": float(destination_port_entropy),
        "sequential_port_score": float(sequential_port_score)
    })
