"""Window state joining logic for V2."""
from __future__ import annotations

import pandas as pd


def join_window_states(flow_windows: pd.DataFrame, packet_windows: pd.DataFrame) -> pd.DataFrame:
    """Join flow and packet window states by window_id.
    
    If packet data is unavailable for a window, its packet features are filled with 0 
    and packet_features_available is set to 0.
    """
    if packet_windows.empty:
        packet_cols = [
            "ttl_mean", "ttl_variance", "tcp_window_mean", "tcp_window_std",
            "ip_fragment_ratio", "payload_size_mean", "payload_size_std",
            "retransmission_count", "port_scan_sequential_score",
            "port_scan_randomised_score", "packet_count", "packet_features_available"
        ]
        for col in packet_cols:
            flow_windows[col] = 0.0
        return flow_windows

    # Join
    joined = pd.merge(flow_windows, packet_windows, on="window_id", how="left")
    
    # Fill missing packet data with 0
    packet_cols = [c for c in packet_windows.columns if c != "window_id"]
    for col in packet_cols:
        joined[col] = joined[col].fillna(0.0)
        
    return joined
