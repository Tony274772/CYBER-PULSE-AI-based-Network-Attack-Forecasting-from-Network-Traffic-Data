"""Prediction heads (Section 7.5.3).

Four small MLPs sharing the belief state ``s_t = concat(h_t, z_t)`` (dim 160):
  * reconstruction_head -> pooled edge-feature-vector summary (MSE target)
  * infiltration_head   -> sigmoid scalar (focal-loss binary target)
  * mitre_head          -> 7-way stage logits (class-weighted CE)
There is no anomaly head: ``kl_t`` from the RSSM is the surprise signal.
"""
from __future__ import annotations

import torch.nn as nn


def _head(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ELU(),
        nn.Linear(hidden, out_dim),
    )


class Heads(nn.Module):
    def __init__(self, belief_dim: int, edge_dim: int, n_stages: int,
                 hidden: int = 128):
        super().__init__()
        # reconstruction target = mean edge feature vector over valid edges
        self.reconstruction_head = _head(belief_dim, hidden, edge_dim)
        self.infiltration_head = _head(belief_dim, hidden, 1)     # logit (sigmoid at loss)
        self.mitre_head = _head(belief_dim, hidden, n_stages)     # logits (softmax at loss)
