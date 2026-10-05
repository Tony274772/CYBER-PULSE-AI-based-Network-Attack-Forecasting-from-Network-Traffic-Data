"""Packet-level feature aggregation for V2."""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_packet_features(window_df: pd.DataFrame) -> pd.Series:
    """Aggregate edge-level packet features into a single network state vector.
    
    If window_df is empty, returns zeros with packet_features_available=0.
    """
    cols = [
        "ttl_mean", "ttl_variance", "tcp_window_mean", "tcp_window_std",
        "ip_fragment_ratio", "payload_size_mean", "payload_size_std",
        "retransmission_count", "port_scan_sequential_score",
        "port_scan_randomised_score", "packet_count",
        "packet_features_available"
    ]
    if window_df.empty:
        return pd.Series(0.0, index=cols)

    # In V2, we aggregate the edge-level packet features produced by extract_packet_features.py
    # into a single network-wide packet vector.
    
    # Estimate total packets as proxy if missing
    pkt_count = window_df.get("packet_count", pd.Series(1.0, index=window_df.index))
    # Or derive from retransmission logic/etc if actual counts aren't in dataframe.
    # We will assume a 'packet_count' column exists or we fallback to 1 weight per row
    if "packet_count" not in window_df.columns:
         pkt_count = pd.Series(1.0, index=window_df.index)
         
    total_pkts = pkt_count.sum()
    safe_tot = total_pkts if total_pkts > 0 else 1.0

    def wavg(col_name):
        if col_name not in window_df.columns: return 0.0
        return (window_df[col_name].fillna(0) * pkt_count).sum() / safe_tot
        
    def wstd(col_name):
        if col_name not in window_df.columns: return 0.0
        # rough weighted aggregation for variance/std
        return (window_df[col_name].fillna(0) * pkt_count).sum() / safe_tot

    retransmissions = window_df.get("retransmission_count", pd.Series([0.0])).sum()

    ttl_mean = wavg("ttl_mean")
    ttl_var = wstd("ttl_var")
    tcp_mean = wavg("tcp_window_mean")
    tcp_std = wstd("tcp_window_std")
    ip_frag = wavg("ip_frag_flag_ratio")
    payload_mean = wavg("payload_size_mean")
    payload_std = wstd("payload_size_std")
    
    scan_seq = window_df.get("port_scan_sequential_score", pd.Series([0.0])).max()
    scan_rand = window_df.get("port_scan_randomised_score", pd.Series([0.0])).max()

    return pd.Series({
        "ttl_mean": float(ttl_mean),
        "ttl_variance": float(ttl_var),
        "tcp_window_mean": float(tcp_mean),
        "tcp_window_std": float(tcp_std),
        "ip_fragment_ratio": float(ip_frag),
        "payload_size_mean": float(payload_mean),
        "payload_size_std": float(payload_std),
        "retransmission_count": float(retransmissions),
        "port_scan_sequential_score": float(scan_seq),
        "port_scan_randomised_score": float(scan_rand),
        "packet_count": float(total_pkts),
        "packet_features_available": 1.0
    })
