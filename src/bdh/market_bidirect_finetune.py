import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from .market_bidirect import MarketBDHBase


class MarketBDH(MarketBDHBase):
    """
    Market BDH model for fine-tuning (classification task).
    
    It processes a sequence of global market states (all assets at time t)
    and predicts the next state (classification per asset).
    """
    def __init__(self, config):
        """
        Args:
            config: MarketBDHConfig instance
        """
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

