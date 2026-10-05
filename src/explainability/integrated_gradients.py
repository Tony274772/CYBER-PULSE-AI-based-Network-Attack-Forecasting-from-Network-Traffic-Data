"""Integrated gradients explainability for Temporal World Model."""

import torch
import numpy as np
from captum.attr import IntegratedGradients


def explain_prediction(model, seq_flow, seq_packet, target_idx=1):
    """
    Compute feature attributions using Integrated Gradients.
    target_idx=1 is for the +30s horizon (index 1 of attack_logits).
    """
    model.eval()
    
    # We need to compute gradients with respect to the inputs
    seq_flow.requires_grad_()
    seq_packet.requires_grad_()
    
    def forward_func(f, p):
        out = model(f, p)
        # return the logits for the specific horizon (1 = +30s)
        return out["attack_logits"][:, target_idx]
        
    ig = IntegratedGradients(forward_func)
    
    # Baseline is all zeros
    base_flow = torch.zeros_like(seq_flow)
    base_packet = torch.zeros_like(seq_packet)
    
    attributions = ig.attribute(
        inputs=(seq_flow, seq_packet),
        baselines=(base_flow, base_packet),
        n_steps=20
    )
    
    attr_flow, attr_packet = attributions
    
    # Aggregate absolute attribution across the 12 windows to get feature importance
    feat_importance_flow = attr_flow.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
    feat_importance_packet = attr_packet.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
    
    # Historical window importance
    # Sum absolute attributions across features for each window
    window_importance_flow = attr_flow.abs().sum(dim=2).squeeze(0).detach().cpu().numpy()
    window_importance_packet = attr_packet.abs().sum(dim=2).squeeze(0).detach().cpu().numpy()
    window_importance = window_importance_flow + window_importance_packet
    
    return {
        "flow_importance": feat_importance_flow.tolist(),
        "packet_importance": feat_importance_packet.tolist(),
        "window_importance": window_importance.tolist()
    }
