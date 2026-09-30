"""Dense masked graph-attention encoder (Section 7.5.1).

A PyG-free replacement for the original ``GATConv`` stack: pure matmul/softmax
attention over a dense adjacency mask, so it batches cleanly and exports to
ONNX trivially. Per window it maps a padded graph (node features, dense edge
features, adjacency + validity masks) to a single graph embedding ``e_t``.

Shapes: batch ``B``, node cap ``N``, node feat dim ``d_n``, edge feat dim
``d_e``, embedding dim ``E`` (=gat_embed_dim), ``H`` heads, per-head ``dh=E/H``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _AttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, edge_dim: int, heads: int):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.heads = heads
        self.dh = embed_dim // heads
        self.embed_dim = embed_dim
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_e = nn.Linear(edge_dim, heads)   # edge features bias attention scores
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, edge_feats: torch.Tensor,
                allow: torch.Tensor) -> torch.Tensor:
        # x [B,N,E]; edge_feats [B,N,N,d_e]; allow [B,N,N] in {0,1}
        B, N, _ = x.shape
        q = self.w_q(x).view(B, N, self.heads, self.dh)
        k = self.w_k(x).view(B, N, self.heads, self.dh)
        v = self.w_v(x).view(B, N, self.heads, self.dh)
        edge_bias = self.w_e(edge_feats)                         # [B,N,N,H]

        scores = torch.einsum("bihd,bjhd->bijh", q, k) / math.sqrt(self.dh)
        scores = scores + edge_bias                              # [B,N,N,H]
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(allow.unsqueeze(-1) == 0, neg_inf)
        attn = torch.softmax(scores, dim=2)                      # over j
        self.last_attention = attn

        ctx = torch.einsum("bijh,bjhd->bihd", attn, v).reshape(B, N, self.embed_dim)
        x = self.norm1(x + ctx)                                  # residual + norm
        x = self.norm2(x + self.mlp(x))                          # FFN + residual + norm
        return x


class DenseGraphAttention(nn.Module):
    """2-layer, 4-head dense graph attention -> masked mean-pooled embedding."""

    def __init__(self, node_dim: int, edge_dim: int, embed_dim: int = 64,
                 heads: int = 4, layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(node_dim, embed_dim)
        self.layers = nn.ModuleList(
            [_AttentionLayer(embed_dim, edge_dim, heads) for _ in range(layers)])

    @property
    def last_attention(self) -> torch.Tensor | None:
        """Attention of the final layer (for explainability export)."""
        return self.layers[-1].last_attention

    def forward(self, node_feats: torch.Tensor, edge_feats: torch.Tensor,
                adjacency_mask: torch.Tensor, node_valid_mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = node_feats.shape
        # allowed attention edges: real edge AND both endpoints valid, plus a
        # self-loop on every valid node so no valid row is fully masked (-> NaN).
        valid = node_valid_mask                                  # [B,N]
        pair_valid = valid.unsqueeze(1) * valid.unsqueeze(2)     # [B,N,N]
        allow = adjacency_mask * pair_valid
        eye = torch.eye(N, device=node_feats.device).unsqueeze(0)
        allow = torch.maximum(allow, eye * pair_valid)

        x = self.input_proj(node_feats)
        for layer in self.layers:
            x = layer(x, edge_feats, allow)

        # masked mean-pool over valid nodes -> [B, E]
        vm = valid.unsqueeze(-1)                                 # [B,N,1]
        pooled = (x * vm).sum(dim=1) / vm.sum(dim=1).clamp(min=1.0)
        return pooled
