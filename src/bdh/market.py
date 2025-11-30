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
    num_assets: int = 93
    input_features: int = 4
    num_classes: int = 2
    label_smoothing: float = 0.0  # Label smoothing factor (0.0 = no smoothing)

class MarketBDH(nn.Module):
    """
    BDH model adapted for Financial Market prediction.
    
    It processes a sequence of global market states (all assets at time t)
    and predicts the next state (classification per asset).
    """
    def __init__(self, config: MarketBDHConfig):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.input_dim = config.num_assets * config.input_features
        self.output_dim = config.num_assets * config.num_classes
        
        # 1. Input Projection
        # Projects flattened market state [N*F] to Model Dimension [D]
        self.input_proj = nn.Linear(self.input_dim, config.n_embd)
        
        # 2. Core BDH Model
        # We use the BDH model but bypass its embedding layer since we provide continuous vectors
        self.bdh = BDH(config)
        
        # 3. Output Projection
        # Projects Model Dimension [D] to flattened predictions [N*C]
        self.output_proj = nn.Linear(config.n_embd, self.output_dim)
        
        # Initialization
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, targets=None):
        """
        Args:
            x: Input tensor of shape [B, T, N, F]
            targets: Target tensor of shape [B, T, N] (class indices)
            
        Returns:
            logits: [B, T, N, C]
            loss: Scalar (if targets provided)
        """
        B, T, N, num_features = x.size()
        
        # Flatten input: [B, T, N, num_features] -> [B, T, N*num_features]
        x_flat = x.view(B, T, -1)
        
        # Project to embedding dimension: [B, T, N*F] -> [B, T, D]
        x_emb = self.input_proj(x_flat)
        
        # Pass through BDH Core
        # We need to manually run the layers because BDH.forward() expects indices for embedding.
        # We'll replicate the logic from BDH.forward but starting after embedding.
        
        # Access internal components of BDH
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N_internal = D * C.mlp_internal_dim_multiplier // nh
        
        # Add head dim for processing: [B, T, D] -> [B, 1, T, D]
        # Note: BDH.forward uses x = self.embed(idx).unsqueeze(1)
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
            
            # Attention
            yKV = self.bdh.attn(Q=h_sparse, K=h_sparse, V=h)
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
        
        # Output Projection: [B, T, D] -> [B, T, N*C]
        logits_flat = self.output_proj(h_out)
        
        # Reshape to [B, T, N, C]
        logits = logits_flat.view(B, T, N, self.config.num_classes)
        
        loss = None
        if targets is not None:
            # targets shape: [B, T, N]
            # Flatten for loss computation: [B*T*N, C] vs [B*T*N]
            loss = F.cross_entropy(
                logits.view(-1, self.config.num_classes), 
                targets.view(-1),
                label_smoothing=self.config.label_smoothing if hasattr(self.config, 'label_smoothing') else 0.0
            )
            
        return logits, loss

