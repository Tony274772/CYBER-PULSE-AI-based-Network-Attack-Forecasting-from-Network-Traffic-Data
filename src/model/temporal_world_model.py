"""Temporal World Model architecture for V2."""

import torch
import torch.nn as nn


class TemporalWorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Branch embeddings
        self.flow_emb = nn.Sequential(
            nn.Linear(config.model.flow_input_dim, config.model.branch_dim),
            nn.ReLU(),
            nn.LayerNorm(config.model.branch_dim),
            nn.Dropout(0.1)
        )
        
        self.packet_emb = nn.Sequential(
            nn.Linear(config.model.packet_input_dim, config.model.branch_dim),
            nn.ReLU(),
            nn.LayerNorm(config.model.branch_dim),
            nn.Dropout(0.1)
        )
        
        # Temporal Dynamics
        self.gru = nn.GRUCell(
            input_size=config.model.fused_dim, 
            hidden_size=config.model.hidden_dim
        )
        
        # Heads
        self.next_state_head = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.model.hidden_dim, config.model.flow_input_dim + config.model.packet_input_dim)
        )
        
        self.attack_head = nn.Sequential(
            nn.Linear(config.model.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        self.stage_head = nn.Sequential(
            nn.Linear(config.model.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 7)
        )

    def forward_step(self, x_flow, x_packet, h):
        """Processes a single time step."""
        flow_feat = self.flow_emb(x_flow)
        packet_feat = self.packet_emb(x_packet)
        x_fused = torch.cat([flow_feat, packet_feat], dim=-1)
        
        h_next = self.gru(x_fused, h)
        return h_next
        
    def forward(self, seq_flow, seq_packet):
        """Processes a sequence of inputs.
        
        Args:
            seq_flow: [batch_size, seq_len, flow_dim]
            seq_packet: [batch_size, seq_len, packet_dim]
            
        Returns:
            h_t: [batch_size, hidden_dim] (Final hidden state)
        """
        batch_size, seq_len, _ = seq_flow.size()
        h = torch.zeros(batch_size, self.config.model.hidden_dim, device=seq_flow.device)
        
        for t in range(seq_len):
            x_flow = seq_flow[:, t, :]
            x_packet = seq_packet[:, t, :]
            h = self.forward_step(x_flow, x_packet, h)
            
        # Predict based on final hidden state
        next_state_pred = self.next_state_head(h)
        attack_logits = self.attack_head(h)
        stage_logits = self.stage_head(h)
        
        return {
            "h_t": h,
            "next_state_pred": next_state_pred,
            "attack_logits": attack_logits,
            "stage_logits": stage_logits
        }
