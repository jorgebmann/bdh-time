import torch
import torch.nn as nn
import torch.nn.functional as F
from .market_attention import MarketBDHBase


class MarketBDHAttentionPretrain(MarketBDHBase):
    """
    Market BDH model for pre-training (regression task).
    
    Predicts next-step features to learn market dynamics.
    Uses MSE loss for regression.
    """
    def __init__(
        self,
        num_assets: int,
        num_features: int,
        d_model: int = 256,
        n_neurons: int = 8192,
        n_heads: int = 4,
        n_layers: int = 4
    ):
        """
        Args:
            num_assets: Number of stocks/assets (N)
            num_features: Number of features per asset (F)
            d_model: Latent driver dimension (D)
            n_neurons: Number of neurons in the brain (Should be >= num_assets)
            n_heads: Number of attention heads
            n_layers: Number of BDH layers
        """
        super().__init__(
            num_assets=num_assets,
            num_features=num_features,
            d_model=d_model,
            n_neurons=n_neurons,
            n_heads=n_heads,
            n_layers=n_layers
        )
        
        # Regression head: predicts next-step features
        # Projects Model Dimension [D] to flattened features [N*F]
        self.regression_head = nn.Linear(d_model, num_assets * num_features)
        
        # Initialization
        nn.init.normal_(self.regression_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.regression_head.bias)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None, mask: torch.Tensor = None):
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
        h_out = self.forward_features(x)
        
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
                # mask shape: [B, T, N] -> expand to [B, T, N, F] for broadcasting
                mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, num_features)
                mask_flat = mask_expanded.contiguous().view(-1, num_features)
                
                # Only compute loss on valid (non-filled) data points
                # Use first feature dimension to determine validity
                valid_mask = mask_flat[:, 0] > 0.5
                
                if valid_mask.sum() > 0:
                    # Select valid predictions and targets
                    pred_masked = pred_flat[valid_mask]
                    target_masked = target_flat[valid_mask]
                    loss = F.mse_loss(pred_masked, target_masked)
                else:
                    # All data points are invalid (shouldn't happen, but handle gracefully)
                    loss = torch.tensor(0.0, device=predicted_features.device, requires_grad=True)
            else:
                # No mask provided - assume all data is valid
                loss = F.mse_loss(pred_flat, target_flat)
        
        return predicted_features, loss

