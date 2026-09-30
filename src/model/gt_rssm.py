"""GT-RSSM assembly (Section 7.5.4).

Runs the dense graph-attention encoder per window to get e_1..e_W, unrolls the
RSSM (teacher-forced on the real e_t at every step in training), and reads out
all heads plus kl_t per step. Also exposes ``imagine`` for label-free K-step
forecasting.

Batched tensor shapes: B batch, W sequence length, N node cap, d_n/d_e node/edge
feat dims. Sequence inputs are [B,W,...].
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.dense_graph_attention import DenseGraphAttention
from src.model.heads import Heads
from src.model.rssm import RSSM


class GTRSSM(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, n_stages: int,
                 embed_dim: int = 64, heads: int = 4, layers: int = 2,
                 h_dim: int = 128, z_dim: int = 32):
        super().__init__()
        self.edge_dim = edge_dim
        self.encoder = DenseGraphAttention(node_dim, edge_dim, embed_dim, heads, layers)
        self.rssm = RSSM(embed_dim, h_dim=h_dim, z_dim=z_dim)
        self.belief_dim = h_dim + z_dim
        self.heads = Heads(self.belief_dim, edge_dim, n_stages)

        # Input standardization (z-score). Buffers so they travel with the
        # checkpoint and apply identically at train/val/inference. Identity
        # until set_feature_stats() is called with training-set statistics.
        self.register_buffer("node_mean", torch.zeros(node_dim))
        self.register_buffer("node_std", torch.ones(node_dim))
        self.register_buffer("edge_mean", torch.zeros(edge_dim))
        self.register_buffer("edge_std", torch.ones(edge_dim))
        self.norm_clip = 10.0

    def set_feature_stats(self, stats: dict) -> None:
        """Load per-feature mean/std (from ``dataset.compute_feature_stats``)."""
        self.node_mean.copy_(torch.tensor(stats["node_mean"], dtype=torch.float32))
        self.node_std.copy_(torch.tensor(stats["node_std"], dtype=torch.float32))
        self.edge_mean.copy_(torch.tensor(stats["edge_mean"], dtype=torch.float32))
        self.edge_std.copy_(torch.tensor(stats["edge_std"], dtype=torch.float32))

    def _normalize(self, node_feats, edge_feats):
        """z-score node/edge features, then clamp to ±``self.norm_clip`` std.
        The clamp tames heavy-tailed flow/packet magnitudes (byte counts, rates)
        whose rare extreme values would otherwise make the MSE recon target
        spike by orders of magnitude on an outlier window. Structural zeros
        (non-edges, padded nodes) become non-zero here but are always
        adjacency/valid-masked downstream (attention, pooling, recon target)."""
        node = (node_feats - self.node_mean) / self.node_std
        edge = (edge_feats - self.edge_mean) / self.edge_std
        c = self.norm_clip
        return node.clamp(-c, c), edge.clamp(-c, c)

    # -- convenience accessors for the individual heads (used by imagine/captum)
    @property
    def infiltration_head(self):
        return self.heads.infiltration_head

    @property
    def mitre_head(self):
        return self.heads.mitre_head

    @staticmethod
    def pooled_edge_target(edge_feats: torch.Tensor, adjacency_mask: torch.Tensor) -> torch.Tensor:
        """Mean edge feature vector over valid edges per window -> [B,W,d_e].
        This is the reconstruction_head's MSE target (7.5.3)."""
        # edge_feats [B,W,N,N,d_e]; adjacency_mask [B,W,N,N]
        m = adjacency_mask.unsqueeze(-1)                       # [B,W,N,N,1]
        s = (edge_feats * m).sum(dim=(2, 3))                   # [B,W,d_e]
        cnt = adjacency_mask.sum(dim=(2, 3)).clamp(min=1.0).unsqueeze(-1)
        return s / cnt

    def encode_sequence(self, node_feats, edge_feats, adjacency_mask, node_valid_mask):
        """Encode every window -> [B,W,E]."""
        B, W = node_feats.shape[0], node_feats.shape[1]
        e = []
        for w in range(W):
            e.append(self.encoder(node_feats[:, w], edge_feats[:, w],
                                  adjacency_mask[:, w], node_valid_mask[:, w]))
        return torch.stack(e, dim=1)                           # [B,W,E]

    def forward(self, node_feats, edge_feats, adjacency_mask, node_valid_mask):
        B, W = node_feats.shape[0], node_feats.shape[1]
        device = node_feats.device
        node_feats, edge_feats = self._normalize(node_feats, edge_feats)
        e_seq = self.encode_sequence(node_feats, edge_feats, adjacency_mask, node_valid_mask)

        h, z = self.rssm.init_state(B, device)
        s_list, kl_list = [], []
        for w in range(W):
            h, z, kl_t, _ = self.rssm.observe_step(h, z, e_seq[:, w])
            s_list.append(torch.cat([h, z], dim=-1))
            kl_list.append(kl_t)
        s = torch.stack(s_list, dim=1)                         # [B,W,belief]
        kl = torch.stack(kl_list, dim=1)                       # [B,W]

        recon = self.heads.reconstruction_head(s)              # [B,W,d_e]
        infil_logit = self.heads.infiltration_head(s).squeeze(-1)   # [B,W]
        mitre_logits = self.heads.mitre_head(s)                # [B,W,n_stages]

        return {
            "e_seq": e_seq, "s": s, "kl": kl,
            "recon": recon, "infil_logit": infil_logit, "mitre_logits": mitre_logits,
            "recon_target": self.pooled_edge_target(edge_feats, adjacency_mask),
            "last_h": h, "last_z": z,
        }

    def imagine(self, h_t, z_t, K):
        trajectory = []
        h, z = h_t, z_t
        for _ in range(1, K + 1):
            h, z = self.rssm.imagine_step(h, z)
            s = torch.cat([h, z], dim=-1)
            trajectory.append({
                "infiltration_prob": torch.sigmoid(self.infiltration_head(s)).squeeze(-1),
                "stage_probs": torch.softmax(self.mitre_head(s), dim=-1),
            })
        return trajectory
