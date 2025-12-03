import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
from pathlib import Path
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


class MarketBDHPretrain(MarketBDHBase):
    """
    Market BDH model for pre-training (regression task).
    
    Predicts next-step features to learn market dynamics.
    Uses MSE loss for regression.
    """
    def __init__(self, config: MarketBDHConfig):
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


class MarketBDH(MarketBDHBase):
    """
    Market BDH model for fine-tuning (classification task).
    
    It processes a sequence of global market states (all assets at time t)
    and predicts the next state (classification per asset).
    """
    def __init__(self, config: MarketBDHConfig):
        super().__init__(config)
        
        # Dimensions
        self.output_dim = config.num_assets * config.num_classes
        
        # Classification head: predicts class probabilities
        # Projects Model Dimension [D] to flattened predictions [N*C]
        self.output_proj = nn.Linear(config.n_embd, self.output_dim)
        
        # Initialization
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, targets=None, mask=None):
        """
        Args:
            x: Input tensor of shape [B, T, N, F]
            targets: Target tensor of shape [B, T, N] (class indices)
            mask: Optional mask tensor of shape [B, T, N] (1 = valid, 0 = missing/filled)
            
        Returns:
            logits: [B, T, N, C]
            loss: Scalar (if targets provided)
        """
        # Forward through shared core
        h_out = self.forward_features(x, mask)
        
        B, T, N, num_features = x.size()
        
        # Output Projection: [B, T, D] -> [B, T, N*C]
        logits_flat = self.output_proj(h_out)
        
        # Reshape to [B, T, N, C]
        logits = logits_flat.view(B, T, N, self.config.num_classes)
        
        loss = None
        if targets is not None:
            # Flatten for loss computation
            logits_flat = logits.view(-1, self.config.num_classes)
            targets_flat = targets.view(-1)
            
            if mask is not None:
                # mask shape: [B, T, N] -> [B*T*N]
                mask_flat = mask.view(-1)
                valid_mask = mask_flat > 0.5
                
                if valid_mask.sum() > 0:
                    # Only compute loss on valid (non-filled) data points
                    loss = F.cross_entropy(
                        logits_flat[valid_mask],
                        targets_flat[valid_mask],
                        label_smoothing=self.config.label_smoothing if hasattr(self.config, 'label_smoothing') else 0.0
                    )
                else:
                    # All data points are invalid (shouldn't happen, but handle gracefully)
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            else:
                # No mask provided - assume all data is valid (original behavior)
                loss = F.cross_entropy(
                    logits_flat,
                    targets_flat,
                    label_smoothing=self.config.label_smoothing if hasattr(self.config, 'label_smoothing') else 0.0
                )
            
        return logits, loss


def load_pretrained_weights(finetune_model: MarketBDH, pretrain_checkpoint_path: str):
    """
    Load pre-trained weights from MarketBDHPretrain checkpoint into MarketBDH model.
    
    Transfers weights for:
    - input_proj (shared input projection)
    - bdh (core BDH layers)
    
    The output_proj (classification head) is left as-is (randomly initialized).
    
    Args:
        finetune_model: MarketBDH model to load weights into
        pretrain_checkpoint_path: Path to pre-trained checkpoint file
        
    Returns:
        finetune_model: Model with loaded weights (modified in-place)
    """
    checkpoint_path = Path(pretrain_checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pre-trained checkpoint not found: {pretrain_checkpoint_path}")
    
    print(f"Loading pre-trained weights from {pretrain_checkpoint_path}...")
    checkpoint = torch.load(pretrain_checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        pretrain_state = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'input_proj' in checkpoint:
        pretrain_state = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint format. Expected dict with 'model_state_dict' or model weights.")
    
    # Transfer input_proj weights
    if 'input_proj.weight' in pretrain_state and 'input_proj.bias' in pretrain_state:
        finetune_model.input_proj.load_state_dict({
            'weight': pretrain_state['input_proj.weight'],
            'bias': pretrain_state['input_proj.bias']
        })
        print("  ✓ Loaded input_proj weights")
    else:
        print("  ⚠ Warning: input_proj weights not found in checkpoint")
    
    # Transfer BDH core weights
    bdh_keys = [k for k in pretrain_state.keys() if k.startswith('bdh.')]
    if bdh_keys:
        bdh_state = {k.replace('bdh.', ''): pretrain_state[k] for k in bdh_keys}
        finetune_model.bdh.load_state_dict(bdh_state, strict=False)
        print(f"  ✓ Loaded {len(bdh_keys)} BDH core weights")
    else:
        print("  ⚠ Warning: BDH core weights not found in checkpoint")
    
    print("Pre-trained weights loaded successfully!")
    print("  Note: Classification head (output_proj) remains randomly initialized.")
    
    return finetune_model

