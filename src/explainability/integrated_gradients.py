"""Integrated Gradients attribution (Section 7.8.2).

Uses ``captum.attr.IntegratedGradients`` with the model's infiltration_head
output (composed through the frozen encoder+RSSM up to belief s_t) as the
target function. Baseline = all-zeros edge feature vector, attributing back to
the input edge features for a chosen window. Reports top-5 features by
absolute attribution.

The approach: build a thin wrapper that accepts the edge feature tensor for one
window and returns the infiltration sigmoid, then attribute through it. The
encoder + RSSM state up to the chosen window is "frozen" as context — we only
attribute the last step's edge features.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.features.schema import EDGE_FEATURE_COLUMNS


class _InfilWrapper(nn.Module):
    """Thin wrapper: edge_feats_flat -> infiltration_prob for one window.

    The deterministic context (h, node_feats, adjacency_mask, node_valid_mask)
    is fixed at construction time.  Only ``edge_feats`` is the attribution
    input, which is the edge feature tensor [1, N, N, d_e].
    """
    def __init__(self, model, node_feats, adjacency_mask, node_valid_mask, h_prev, z_prev):
        super().__init__()
        self.model = model
        self.register_buffer("_nf", node_feats)
        self.register_buffer("_am", adjacency_mask)
        self.register_buffer("_vm", node_valid_mask)
        self.register_buffer("_h", h_prev)
        self.register_buffer("_z", z_prev)

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        nf, ef = self.model._normalize(self._nf, edge_feats)
        e_t = self.model.encoder(nf, ef, self._am, self._vm)  # [1, E]
        h_t, z_t, _, _ = self.model.rssm.observe_step(self._h, self._z, e_t)
        s_t = torch.cat([h_t, z_t], dim=-1)
        logit = self.model.heads.infiltration_head(s_t)
        return torch.sigmoid(logit)  # [1, 1]


@torch.no_grad()
def _get_context(model, node_feats_seq, edge_feats_seq, adj_seq, valid_seq, target_step: int):
    """Run the model up to ``target_step - 1`` to get (h, z) context."""
    device = next(model.parameters()).device
    B = node_feats_seq.shape[0]
    h, z = model.rssm.init_state(B, device)

    for w in range(target_step):
        nf, ef = model._normalize(
            node_feats_seq[:, w].to(device), edge_feats_seq[:, w].to(device))
        e_t = model.encoder(nf, ef, adj_seq[:, w].to(device), valid_seq[:, w].to(device))
        h, z, _, _ = model.rssm.observe_step(h, z, e_t)
    return h, z


def compute_integrated_gradients(
    model,
    node_feats_seq: torch.Tensor,    # [1, W, N, d_n]
    edge_feats_seq: torch.Tensor,    # [1, W, N, N, d_e]
    adjacency_mask_seq: torch.Tensor,  # [1, W, N, N]
    node_valid_mask_seq: torch.Tensor,  # [1, W, N]
    target_step: int = -1,
    n_steps: int = 50,
    top_k: int = 5,
) -> dict[str, Any]:
    """Attribute the infiltration head output to edge features at ``target_step``.

    Returns:
      * ``attributions``: [N, N, d_e] numpy array of attribution values
      * ``top_features``: list of {feature, attribution, rank}
      * ``target_step``: the resolved window index
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        return {"error": "captum not installed", "top_features": [],
                "attributions": np.array([]), "target_step": target_step}

    model.eval()
    device = next(model.parameters()).device

    W = node_feats_seq.shape[1]
    if target_step < 0:
        target_step = W + target_step
    target_step = max(0, min(target_step, W - 1))

    # Get RSSM context from steps before target
    h, z = _get_context(model, node_feats_seq, edge_feats_seq,
                        adjacency_mask_seq, node_valid_mask_seq, target_step)

    nf_t = node_feats_seq[:, target_step].to(device)
    ef_t = edge_feats_seq[:, target_step].to(device)
    am_t = adjacency_mask_seq[:, target_step].to(device)
    vm_t = node_valid_mask_seq[:, target_step].to(device)

    wrapper = _InfilWrapper(model, nf_t, am_t, vm_t, h, z).to(device)

    ig = IntegratedGradients(wrapper)
    baseline = torch.zeros_like(ef_t)
    attributions = ig.attribute(ef_t, baselines=baseline, n_steps=n_steps)
    attr_np = attributions[0].cpu().numpy()  # [N, N, d_e]

    # Per-feature: sum absolute attribution over all edges
    abs_per_feat = np.abs(attr_np).sum(axis=(0, 1))  # [d_e]
    top_idx = np.argsort(abs_per_feat)[::-1][:top_k]
    top_features = []
    for rank, idx in enumerate(top_idx):
        feat_name = EDGE_FEATURE_COLUMNS[idx] if idx < len(EDGE_FEATURE_COLUMNS) else f"feat_{idx}"
        top_features.append({
            "feature": feat_name,
            "attribution": float(abs_per_feat[idx]),
            "rank": rank + 1,
        })

    return {
        "attributions": attr_np,
        "top_features": top_features,
        "target_step": target_step,
    }
