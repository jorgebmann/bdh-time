#!/usr/bin/env python3
"""
Fine-tuning script for MarketBDH Attention model.

This script fine-tunes a pre-trained model to predict binary return direction
(classification task).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.bdh.market_attention_finetune import MarketBDHAttentionFinetune
from src.dataset.preprocess import process_market_data_from_parquet


# --- Dataset Class ---
class MarketSequenceDataset(Dataset):
    def __init__(self, X, Y, mask, seq_len):
        """
        Args:
            X: [Total_Time, N, F]
            Y: [Total_Time, N]
            mask: [Total_Time, N] (1=Valid, 0=Padding)
            seq_len: Length of window
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()  # BCE requires float targets
        self.mask = torch.from_numpy(mask).float()
        self.seq_len = seq_len

    def __len__(self):
        # Ensure we can grab a full window + 1 step for next-token prediction
        return len(self.X) - self.seq_len - 1

    def __getitem__(self, idx):
        # We grab seq_len + 1 to handle "next step" targets
        end_idx = idx + self.seq_len + 1

        x_window = self.X[idx: end_idx]
        y_window = self.Y[idx: end_idx]
        mask_window = self.mask[idx: end_idx]

        return x_window, y_window, mask_window


def masked_bce_loss(pred, target, mask):
    """Binary Cross Entropy ignoring invalid assets."""
    # pred, target, mask: [B, T, N]
    bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
    masked_bce = bce * mask
    return masked_bce.sum() / (mask.sum() + 1e-8)


def load_pretrained_weights(finetune_model: MarketBDHAttentionFinetune, pretrain_checkpoint_path: str):
    """
    Load pre-trained weights from MarketBDHAttentionPretrain checkpoint into MarketBDHAttentionFinetune model.
    
    Transfers weights for:
    - input_adapter (shared input projection)
    - layers (core BDH layers)
    
    The classification_head is left as-is (randomly initialized).
    
    Args:
        finetune_model: MarketBDHAttentionFinetune model to load weights into
        pretrain_checkpoint_path: Path to pre-trained checkpoint file
    """
    checkpoint_path = Path(pretrain_checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pre-trained checkpoint not found: {pretrain_checkpoint_path}")
    
    print(f"Loading pre-trained weights from {pretrain_checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        pretrain_state = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'input_adapter' in checkpoint:
        pretrain_state = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint format. Expected dict with 'model_state_dict' or model weights.")
    
    # Get current model state
    finetune_state = finetune_model.state_dict()
    
    # Transfer input_adapter weights
    if 'input_adapter.weight' in pretrain_state and 'input_adapter.bias' in pretrain_state:
        finetune_state['input_adapter.weight'] = pretrain_state['input_adapter.weight']
        finetune_state['input_adapter.bias'] = pretrain_state['input_adapter.bias']
        print("  ✓ Loaded input_adapter weights")
    else:
        print("  ⚠ Warning: input_adapter weights not found in checkpoint")
    
    # Transfer BDH layer weights
    layer_keys = [k for k in pretrain_state.keys() if k.startswith('layers.')]
    if layer_keys:
        for key in layer_keys:
            if key in finetune_state:
                finetune_state[key] = pretrain_state[key]
        print(f"  ✓ Loaded {len(layer_keys)} BDH layer weights")
    else:
        print("  ⚠ Warning: BDH layer weights not found in checkpoint")
    
    # Load updated state dict
    finetune_model.load_state_dict(finetune_state, strict=False)
    
    print("Pre-trained weights loaded successfully!")
    print("  Note: Classification head remains randomly initialized.")


def train_finetune(model, train_loader, val_loader, config):
    """Run fine-tuning phase."""
    print("\n=== Fine-tuning (Directional Alpha) ===")
    optimizer = optim.AdamW(model.parameters(), lr=config['finetune_lr'])

    best_acc = 0

    for epoch in range(config['finetune_epochs']):
        # Train
        model.train()
        train_loss = 0
        for batch_x, batch_y, batch_mask in tqdm(train_loader, desc=f"Finetune Epoch {epoch + 1}"):
            batch_x = batch_x.to(config['device'])
            batch_y = batch_y.to(config['device'])
            batch_mask = batch_mask.to(config['device'])

            # Use full sequence, but we focus on predicting Y at each step
            # Preprocess aligns Y[t] to be the target for X[t]
            # We slice to seq_len to match the batch logic
            inp = batch_x[:, :-1]
            tgt = batch_y[:, :-1]  # Targets for the input sequence steps
            mask = batch_mask[:, :-1]

            optimizer.zero_grad()
            pred, loss = model(inp, targets=tgt, mask=mask)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y, batch_mask in val_loader:
                batch_x = batch_x.to(config['device'])
                batch_y = batch_y.to(config['device'])
                batch_mask = batch_mask.to(config['device'])

                inp = batch_x[:, :-1]
                tgt = batch_y[:, :-1]
                mask = batch_mask[:, :-1]

                pred, loss = model(inp, targets=tgt, mask=mask)
                val_loss += loss.item()

                # Accuracy calculation (masked)
                pred_binary = (pred > 0.5).float()
                hits = (pred_binary == tgt).float() * mask
                correct += hits.sum().item()
                total += mask.sum().item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        val_acc = (correct / total * 100) if total > 0 else 0

        print(f"Epoch {epoch + 1} | Train BCE: {avg_train:.6f} | Val BCE: {avg_val:.6f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = Path(config['save_dir']) / "attention_finetune_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'config': {
                    'num_assets': model.num_assets,
                    'num_features': model.num_features,
                    'd_model': model.d_model,
                    'n_neurons': model.n_neurons,
                    'n_heads': model.n_heads,
                    'n_layers': model.n_layers
                }
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint to {checkpoint_path}")

    print(f"\nFine-tuning complete. Best validation accuracy: {best_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune MarketBDH Attention model')
    
    # Data arguments
    parser.add_argument('--parquet-path', type=str, default='./data/parquet',
                       help='Path to parquet directory')
    parser.add_argument('--library', type=str, default='nasdaq100',
                       help='Library name')
    parser.add_argument('--min-years', type=float, default=3.0,
                       help='Minimum years of data per symbol')
    parser.add_argument('--seq-len', type=int, default=64,
                       help='Context window length')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Train/validation split ratio')
    
    # Model arguments
    parser.add_argument('--d-model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--n-neurons', type=int, default=4096,
                       help='Number of neurons')
    parser.add_argument('--n-heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=4,
                       help='Number of BDH layers')
    
    # Training arguments
    parser.add_argument('--finetune-epochs', type=int, default=20,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--finetune-lr', type=float, default=1e-4,
                       help='Fine-tuning learning rate')
    parser.add_argument('--finetune-batch-size', type=int, default=32,
                       help='Fine-tuning batch size')
    
    # Pre-training checkpoint
    parser.add_argument('--pretrain-checkpoint', type=str, required=True,
                       help='Path to pre-trained checkpoint file')
    
    # System arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu), defaults to cuda if available')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Build config dict
    config = {
        'parquet_path': args.parquet_path,
        'library': args.library,
        'min_years': args.min_years,
        'seq_len': args.seq_len,
        'train_split': args.train_split,
        'd_model': args.d_model,
        'n_neurons': args.n_neurons,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'finetune_epochs': args.finetune_epochs,
        'finetune_lr': args.finetune_lr,
        'finetune_batch_size': args.finetune_batch_size,
        'device': args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_dir': args.save_dir
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)

    # 1. Load & Process Data
    print("Step 1: Loading Data...")
    try:
        data = process_market_data_from_parquet(
            config['parquet_path'],
            library=config['library'],
            min_years=config['min_years']
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    X, Y, mask = data['X'], data['Y'], data['mask']

    # 2. Split Data (Time-based split)
    split_idx = int(len(X) * config['train_split'])

    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    mask_train, mask_val = mask[:split_idx], mask[split_idx:]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    train_ds = MarketSequenceDataset(X_train, Y_train, mask_train, config['seq_len'])
    val_ds = MarketSequenceDataset(X_val, Y_val, mask_val, config['seq_len'])

    # 3. DataLoaders
    train_dl = DataLoader(train_ds, batch_size=config['finetune_batch_size'], shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=config['finetune_batch_size'], shuffle=False, num_workers=4)

    # 4. Initialize Model
    print("Step 2: Initializing Model...")
    num_assets = X.shape[1]
    num_features = X.shape[2]

    model = MarketBDHAttentionFinetune(
        num_assets=num_assets,
        num_features=num_features,
        d_model=config['d_model'],
        n_neurons=config['n_neurons'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers']
    ).to(config['device'])

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 5. Load Pre-trained Weights
    load_pretrained_weights(model, args.pretrain_checkpoint)

    # 6. Run Fine-tuning
    train_finetune(model, train_dl, val_dl, config)

    print("\nFine-tuning Complete.")


if __name__ == "__main__":
    main()

