import torch
import torch.nn as nn
import torch.nn.functional as F
from .market_attention import MarketBDHBase


class MarketBDHAttentionFinetune(MarketBDHBase):
    """
    Market BDH model for fine-tuning (classification task).
    
    Predicts binary return direction for each asset.
    Uses binary cross-entropy loss for classification.
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
        
        # Classification head: predicts binary return direction
        # Projects Model Dimension [D] to number of assets [N]
        self.classification_head = nn.Linear(d_model, num_assets)
        
        # Initialization
        nn.init.normal_(self.classification_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classification_head.bias)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None, mask: torch.Tensor = None):
        """
        Args:
            x: Input tensor of shape [B, T, N, F]
            targets: Target tensor of shape [B, T, N] (binary class labels: 0 or 1)
            mask: Optional mask tensor of shape [B, T, N] (1 = valid, 0 = missing/filled)
            
        Returns:
            probabilities: [B, T, N] (sigmoid probabilities 0-1)
            loss: Scalar BCE loss (if targets provided)
        """
        # Forward through shared core
        h_out = self.forward_features(x)
        
        B, T, N, num_features = x.size()
        
        # Classification head: [B, T, D] -> [B, T, N]
        logits = self.classification_head(h_out)
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits)
        
        loss = None
        if targets is not None:
            # Flatten for loss computation
            logits_flat = logits.view(-1)
            targets_flat = targets.view(-1).float()  # BCE requires float targets
            
            if mask is not None:
                # mask shape: [B, T, N] -> [B*T*N]
                mask_flat = mask.view(-1)
                valid_mask = mask_flat > 0.5
                
                if valid_mask.sum() > 0:
                    # Only compute loss on valid (non-filled) data points
                    loss = F.binary_cross_entropy_with_logits(
                        logits_flat[valid_mask],
                        targets_flat[valid_mask]
                    )
                else:
                    # All data points are invalid (shouldn't happen, but handle gracefully)
                    loss = torch.tensor(0.0, device=probabilities.device, requires_grad=True)
            else:
                # No mask provided - assume all data is valid
                loss = F.binary_cross_entropy_with_logits(logits_flat, targets_flat)
        
        return probabilities, loss

