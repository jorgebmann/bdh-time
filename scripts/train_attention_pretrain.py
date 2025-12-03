#!/usr/bin/env python3
"""
Pre-training script for MarketBDH Attention model.

This script trains the model to predict next-step features (regression task)
to learn market dynamics.
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

from src.bdh.market_attention_pretrain import MarketBDHAttentionPretrain
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


def masked_mse_loss(pred, target, mask):
    """MSE Loss ignoring invalid (padded) assets."""
    # pred, target: [B, T, N, F]
    # mask: [B, T, N] -> Broadcast to [B, T, N, 1]
    mask_b = mask.unsqueeze(-1)

    squared_error = (pred - target) ** 2
    masked_error = squared_error * mask_b

    # Avoid division by zero
    valid_elements = mask_b.sum() * pred.shape[-1]
    return masked_error.sum() / (valid_elements + 1e-8)


def train_pretrain(model, train_loader, val_loader, config):
    """Run pre-training phase."""
    print("\n=== Pre-training (Market Physics) ===")
    optimizer = optim.AdamW(model.parameters(), lr=config['pretrain_lr'])

    best_val_loss = float('inf')

    for epoch in range(config['pretrain_epochs']):
        # Train
        model.train()
        train_loss = 0
        for batch_x, _, batch_mask in tqdm(train_loader, desc=f"Pretrain Epoch {epoch + 1}"):
            batch_x, batch_mask = batch_x.to(config['device']), batch_mask.to(config['device'])

            # Input: Steps 0 to T-1
            # Target: Steps 1 to T (Next feature vector)
            inp = batch_x[:, :-1, :, :]
            tgt = batch_x[:, 1:, :, :]
            mask = batch_mask[:, 1:]  # Mask aligned to target

            optimizer.zero_grad()
            pred, loss = model(inp, targets=tgt, mask=mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _, batch_mask in val_loader:
                batch_x, batch_mask = batch_x.to(config['device']), batch_mask.to(config['device'])
                inp = batch_x[:, :-1]
                tgt = batch_x[:, 1:]
                mask = batch_mask[:, 1:]

                _, loss = model(inp, targets=tgt, mask=mask)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}")

        # Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            checkpoint_path = Path(config['save_dir']) / "attention_pretrain_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val,
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

    print(f"\nPre-training complete. Best validation loss: {best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Pre-train MarketBDH Attention model')
    
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
    parser.add_argument('--pretrain-epochs', type=int, default=50,
                       help='Number of pre-training epochs')
    parser.add_argument('--pretrain-lr', type=float, default=1e-3,
                       help='Pre-training learning rate')
    parser.add_argument('--pretrain-batch-size', type=int, default=32,
                       help='Pre-training batch size')
    
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
        'pretrain_epochs': args.pretrain_epochs,
        'pretrain_lr': args.pretrain_lr,
        'pretrain_batch_size': args.pretrain_batch_size,
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
    train_dl = DataLoader(train_ds, batch_size=config['pretrain_batch_size'], shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=config['pretrain_batch_size'], shuffle=False, num_workers=4)

    # 4. Initialize Model
    print("Step 2: Initializing Model...")
    num_assets = X.shape[1]
    num_features = X.shape[2]

    model = MarketBDHAttentionPretrain(
        num_assets=num_assets,
        num_features=num_features,
        d_model=config['d_model'],
        n_neurons=config['n_neurons'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers']
    ).to(config['device'])

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 5. Run Pre-training
    train_pretrain(model, train_dl, val_dl, config)

    print("\nPre-training Complete.")
    print(f"To fine-tune this model, run:")
    print(f"  python scripts/train_attention_finetune.py --pretrain-checkpoint {config['save_dir']}/attention_pretrain_best.pt")


if __name__ == "__main__":
    main()

