"""Attention-weight explainability export (Section 7.8.1).

Pulls ``DenseGraphAttention.last_attention`` for a chosen window, averages
over heads, and returns an edge-weight overlay suitable for the dashboard's
network graph view (node positions from a spring layout, edge thickness/opacity
proportional to attention weight).

Usage (called by ``inference/predict.py`` per window):
    attn_info = export_attention(model, window_tensors, node_ips)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


@torch.no_grad()
def export_attention(
    model,
    node_feats: torch.Tensor,      # [1, N, d_n]
    edge_feats: torch.Tensor,      # [1, N, N, d_e]
    adjacency_mask: torch.Tensor,  # [1, N, N]
    node_valid_mask: torch.Tensor,  # [1, N]
    node_ips: list[str],
) -> dict[str, Any]:
    """Run the encoder on a single window and extract head-averaged attention.

    Returns a dict with:
      * ``nodes``:  list of {ip, idx, valid}
      * ``edges``:  list of {src_ip, dst_ip, weight}  (weight ∈ [0, 1])
      * ``attn_matrix``:  [N, N] numpy array (head-averaged, masked)
    """
    model.eval()
    device = next(model.parameters()).device
    nf = node_feats.to(device)
    ef = edge_feats.to(device)
    am = adjacency_mask.to(device)
    vm = node_valid_mask.to(device)

    # Normalize inputs the same way forward() does
    nf, ef = model._normalize(nf, ef)

    # Run through the encoder layers to populate last_attention
    model.encoder(nf, ef, am, vm)
    attn = model.encoder.last_attention  # [1, N, N, H]
    if attn is None:
        return {"nodes": [], "edges": [], "attn_matrix": np.array([])}

    attn = attn[0]  # [N, N, H]
    # Average over heads
    attn_avg = attn.mean(dim=-1).cpu().numpy()  # [N, N]
    valid = node_valid_mask[0].cpu().numpy()
    adj = adjacency_mask[0].cpu().numpy()

    # Build node list
    n_valid = int(valid.sum())
    nodes = []
    for i in range(n_valid):
        ip = node_ips[i] if i < len(node_ips) else f"node_{i}"
        nodes.append({"ip": ip, "idx": i, "valid": True})

    # Build edge list from real edges (adjacency mask)
    edges = []
    for i in range(n_valid):
        for j in range(n_valid):
            if adj[i, j] > 0 and i != j:
                src_ip = node_ips[i] if i < len(node_ips) else f"node_{i}"
                dst_ip = node_ips[j] if j < len(node_ips) else f"node_{j}"
                edges.append({
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "weight": float(attn_avg[i, j]),
                })

    # Sort edges by attention weight descending for easy top-k extraction
    edges.sort(key=lambda e: e["weight"], reverse=True)

    return {
        "nodes": nodes,
        "edges": edges,
        "attn_matrix": attn_avg[:n_valid, :n_valid],
    }


def top_suspicious_hosts(attn_info: dict, top_k: int = 5) -> list[dict]:
    """Identify hosts with the highest total inbound attention weight —
    these are the nodes the model 'paid the most attention to', i.e. the
    most suspicious endpoints."""
    if not attn_info["edges"]:
        return []
    host_attn: dict[str, float] = {}
    for e in attn_info["edges"]:
        host_attn[e["dst_ip"]] = host_attn.get(e["dst_ip"], 0.0) + e["weight"]
        host_attn[e["src_ip"]] = host_attn.get(e["src_ip"], 0.0) + e["weight"]
    ranked = sorted(host_attn.items(), key=lambda x: x[1], reverse=True)
    return [{"ip": ip, "total_attention": w} for ip, w in ranked[:top_k]]
