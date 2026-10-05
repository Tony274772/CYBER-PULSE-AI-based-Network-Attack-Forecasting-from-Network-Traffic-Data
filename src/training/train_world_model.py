"""Training loop for Temporal World Model."""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import random

from src.config import Config
from src.training.dataset import TemporalDataset
from src.model.temporal_world_model import TemporalWorldModel
from src.evaluation.metrics import calculate_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    config = Config.load(args.config)
    set_seed(config.training.seed)
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_ds = TemporalDataset(os.path.join(config.paths.processed_dir, "train.parquet"), config)
    val_ds = TemporalDataset(os.path.join(config.paths.processed_dir, "val.parquet"), config)
    
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False)
    
    model = TemporalWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )
    
    # Positive weights for attack loss
    # Compute from train_ds targets
    y_30_all = train_ds.Y_30
    pos = max(1, int(y_30_all.sum()))
    neg = max(1, len(y_30_all) - pos)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    huber_loss = nn.SmoothL1Loss()
    ce_loss = nn.CrossEntropyLoss()
    
    best_val_pr_auc = -1
    patience_counter = 0
    
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    
    print("Starting training...")
    for epoch in range(config.training.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            seq_flow = batch["seq_flow"].to(device)
            seq_packet = batch["seq_packet"].to(device)
            y_next_state = batch["y_next_state"].to(device)
            y_5 = batch["y_5"].to(device)
            y_30 = batch["y_30"].to(device)
            y_60 = batch["y_60"].to(device)
            y_stage = batch["y_stage"].to(device)
            
            out = model(seq_flow, seq_packet)
            
            loss_state = huber_loss(out["next_state_pred"], y_next_state)
            
            # Stack targets
            y_attack = torch.stack([y_5, y_30, y_60], dim=1)
            loss_attack = bce_loss(out["attack_logits"], y_attack)
            
            # Stage loss masked on y_30
            mask = y_30 > 0
            if mask.sum() > 0:
                loss_stage = ce_loss(out["stage_logits"][mask], y_stage[mask])
            else:
                loss_stage = torch.tensor(0.0).to(device)
                
            loss = (config.loss.state_weight * loss_state + 
                    config.loss.attack_weight * loss_attack + 
                    config.loss.stage_weight * loss_stage)
                    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_y_30 = []
        all_p_30 = []
        
        with torch.no_grad():
            for batch in val_loader:
                seq_flow = batch["seq_flow"].to(device)
                seq_packet = batch["seq_packet"].to(device)
                y_next_state = batch["y_next_state"].to(device)
                y_5 = batch["y_5"].to(device)
                y_30 = batch["y_30"].to(device)
                y_60 = batch["y_60"].to(device)
                y_stage = batch["y_stage"].to(device)
                
                out = model(seq_flow, seq_packet)
                
                loss_state = huber_loss(out["next_state_pred"], y_next_state)
                y_attack = torch.stack([y_5, y_30, y_60], dim=1)
                loss_attack = bce_loss(out["attack_logits"], y_attack)
                
                mask = y_30 > 0
                if mask.sum() > 0:
                    loss_stage = ce_loss(out["stage_logits"][mask], y_stage[mask])
                else:
                    loss_stage = torch.tensor(0.0).to(device)
                    
                loss = (config.loss.state_weight * loss_state + 
                        config.loss.attack_weight * loss_attack + 
                        config.loss.stage_weight * loss_stage)
                val_loss += loss.item()
                
                p_attack = torch.sigmoid(out["attack_logits"])
                p_30 = p_attack[:, 1]
                all_y_30.extend(y_30.cpu().numpy())
                all_p_30.extend(p_30.cpu().numpy())
                
        val_loss /= len(val_loader)
        
        metrics = calculate_metrics(np.array(all_y_30), np.array(all_p_30))
        val_pr_auc = metrics["pr_auc"]
        
        print(f"Epoch {epoch+1}/{config.training.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val PR-AUC 30s: {val_pr_auc:.4f}")
        
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            patience_counter = 0
            
            # Find threshold that gives FPR = 0.05
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(all_y_30, all_p_30)
            idx = np.where(fpr <= config.thresholding.target_validation_fpr)[0]
            if len(idx) > 0:
                best_threshold = thresholds[idx[-1]]
            else:
                best_threshold = 0.5
                
            ckpt_path = os.path.join(config.paths.checkpoint_dir, "temporal_world_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "best_val_pr_auc": best_val_pr_auc,
                "best_threshold": best_threshold
            }, ckpt_path)
            print(f"  -> Saved new best model (Threshold set to {best_threshold:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()
