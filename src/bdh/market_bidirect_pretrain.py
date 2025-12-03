import torch
import torch.nn as nn
import torch.nn.functional as F
from .market_bidirect import MarketBDHBase


class MarketBDHPretrain(MarketBDHBase):
    """
    Market BDH model for pre-training (regression task).
    
    Predicts next-step features to learn market dynamics.
    Uses MSE loss for regression.
    """
    def __init__(self, config):
        """
        Args:
            config: MarketBDHConfig instance
        """
        super().__init__(config)
        
        # Regression head: predicts next-step features
        # Projects Model Dimension [D] to flattened features [N*F]
        self.regression_head = nn.Linear(config.n_embd, self.input_dim)
        
        # Initialization
        nn.init.normal_(self.regression_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.regression_head.bias)
    
    def forward(self, x, targets=None, mask=None):
        """
        Args:
            x: Input tensor of shape [B, T, N, F]
            targets: Target tensor of shape [B, T, N, F] (next-step features)
            mask: Optional mask tensor of shape [B, T, N] (1 = valid, 0 = missing/filled)
            
        Returns:
            predicted_features: [B, T, N, F]
            loss: Scalar MSE loss (if targets provided)
        """
        # Forward through shared core
        h_out = self.forward_features(x, mask)
        
        B, T, N, num_features = x.size()
        
        # Regression head: [B, T, D] -> [B, T, N*F]
        predicted_flat = self.regression_head(h_out)
        
        # Reshape to [B, T, N, F]
        predicted_features = predicted_flat.view(B, T, N, num_features)
        
        loss = None
        if targets is not None:
            # Flatten for loss computation
            pred_flat = predicted_features.view(-1, num_features)
            target_flat = targets.view(-1, num_features)
            
            if mask is not None:
                # mask shape: [B, T, N] -> [B*T*N]
                mask_flat = mask.view(-1)
                valid_mask = mask_flat > 0.5
                
                if valid_mask.sum() > 0:
                    # Only compute loss on valid (non-filled) data points
                    loss = F.mse_loss(
                        pred_flat[valid_mask],
                        target_flat[valid_mask]
                    )
                else:
                    # All data points are invalid (shouldn't happen, but handle gracefully)
                    loss = torch.tensor(0.0, device=predicted_features.device, requires_grad=True)
            else:
                # No mask provided - assume all data is valid
                loss = F.mse_loss(pred_flat, target_flat)
        
        return predicted_features, loss

