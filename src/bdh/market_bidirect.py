import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
from .model import BDH, BDHConfig

@dataclasses.dataclass
class MarketBDHConfig(BDHConfig):
    """
    Configuration for Market BDH model.
    Extends BDHConfig with market-specific parameters.
    """
    num_assets: int = 28425
    input_features: int = 21
    num_classes: int = 2
    label_smoothing: float = 0.0  # Label smoothing factor (0.0 = no smoothing)
    pretrain_mode: bool = False  # If True, model is in pre-training mode (for backward compatibility)


class MarketBDHBase(nn.Module):
    """
    Base class for Market BDH models.
    Contains shared core architecture: input projection + BDH layers.
    """
    def __init__(self, config: MarketBDHConfig):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.input_dim = config.num_assets * config.input_features
        
        # 1. Input Projection
        # Projects flattened market state [N*F] to Model Dimension [D]
        self.input_proj = nn.Linear(self.input_dim, config.n_embd)
        
        # 2. Core BDH Model
        # We use the BDH model but bypass its embedding layer since we provide continuous vectors
        self.bdh = BDH(config)
        
        # Initialization
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
    
    def forward_features(self, x, mask=None):
        """
        Forward pass through shared core architecture.
        
        Args:
            x: Input tensor of shape [B, T, N, F]
            mask: Optional mask tensor (not used in base, but kept for API consistency)
            
        Returns:
            h_out: Hidden representation [B, T, D]
        """
        B, T, N, num_features = x.size()
        
        # Flatten input: [B, T, N, num_features] -> [B, T, N*num_features]
        x_flat = x.view(B, T, -1)
        
        # Project to embedding dimension: [B, T, N*F] -> [B, T, D]
        x_emb = self.input_proj(x_flat)
        
        # Pass through BDH Core
        # Access internal components of BDH
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N_internal = D * C.mlp_internal_dim_multiplier // nh
        
        # Add head dim for processing: [B, T, D] -> [B, 1, T, D]
        h = x_emb.unsqueeze(1)
        h = self.bdh.ln(h)
        
        # Loop through layers
        for _ in range(C.n_layer):
            h_res = h
            
            # Project to conceptual space
            h_latent = h @ self.bdh.encoder
            h_sparse = F.relu(h_latent)
            
            # Hook point
            h_sparse = self.bdh.x_sparse_hook_point(h_sparse)
            
            # Attention (Bidirectional: is_causal=False)
            # Allows model to see full context of the sequence window, similar to BERT
            yKV = self.bdh.attn(Q=h_sparse, K=h_sparse, V=h, is_causal=False)
            yKV = self.bdh.ln(yKV)
            
            # Modulation
            y_latent = yKV @ self.bdh.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = h_sparse * y_sparse
            xy_sparse = self.bdh.drop(xy_sparse)
            
            # Project back
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N_internal * nh) @ self.bdh.decoder
            y = self.bdh.ln(yMLP)
            
            # Residual
            h = self.bdh.ln(h_res + y)
        
        # Remove head dim: [B, 1, T, D] -> [B, T, D]
        h_out = h.squeeze(1)
        
        return h_out



