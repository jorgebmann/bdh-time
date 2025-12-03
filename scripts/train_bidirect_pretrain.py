#!/usr/bin/env python3
"""
Pre-train MarketBDH model on processed market dataset.

This script loads the processed market_dataset.pt file and pre-trains the MarketBDHPretrain model
by predicting next-step features (regression task with MSE loss).
The pre-trained weights can then be used for fine-tuning with MarketBDH.
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bdh.market_bidirect import MarketBDHConfig
from bdh.market_bidirect_pretrain import MarketBDHPretrain
from bdh.data import MarketDataset

# --- Default Configuration ---
DEFAULT_BATCH_SIZE = 32
DEFAULT_WINDOW_SIZE = 64
DEFAULT_MAX_ITERS = 5000  # More iterations for pre-training
DEFAULT_LEARNING_RATE = 1e-4  # Higher LR for pre-training
DEFAULT_EVAL_FREQ = 250
DEFAULT_WEIGHT_DECAY = 0.1 #Aggressive regularization. It penalizes large weights, preventing the model from relying too heavily on any single feature or asset.
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_EARLY_STOP_PATIENCE = 5
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_N_LAYER = 2
DEFAULT_N_EMBD = 128
DEFAULT_N_HEAD = 4
DEFAULT_DROPOUT = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset_info(data_path):
    """Load and display dataset metadata without loading full tensors."""
    import gc
    
    print(f"\nLoading dataset metadata from {data_path}...")
    
    try:
        checkpoint = torch.load(data_path, map_location='cpu', weights_only=False)
        
        # Extract shapes before clearing
        x_tensor = checkpoint.get('X', None)
        y_tensor = checkpoint.get('Y', None)
        mask_tensor = checkpoint.get('mask', None)
        
        x_shape_str = str(x_tensor.shape) if x_tensor is not None and hasattr(x_tensor, 'shape') else 'N/A'
        y_shape_str = str(y_tensor.shape) if y_tensor is not None and hasattr(y_tensor, 'shape') else 'N/A'
        mask_shape_str = str(mask_tensor.shape) if mask_tensor is not None and hasattr(mask_tensor, 'shape') else 'N/A'
        
        # Calculate coverage if mask exists
        coverage = None
        if mask_tensor is not None and hasattr(mask_tensor, 'sum'):
            coverage = mask_tensor.sum().item() / mask_tensor.numel() * 100
        
        # Extract feature count
        num_features = None
        if x_tensor is not None and hasattr(x_tensor, 'shape') and len(x_tensor.shape) >= 3:
            num_features = x_tensor.shape[-1]
        
        # Extract metadata
        info = {
            'asset_names': checkpoint.get('asset_names', []),
            'feature_names': checkpoint.get('feature_names', []),
            'asset_date_ranges': checkpoint.get('asset_date_ranges', []),
            'union_dates': checkpoint.get('union_dates', []),
            'x_shape': x_shape_str,
            'y_shape': y_shape_str,
            'mask_shape': mask_shape_str,
            'coverage': coverage,
            'num_features': num_features
        }
        
        # Immediately delete large tensors to free memory
        if 'X' in checkpoint:
            del checkpoint['X']
        if 'Y' in checkpoint:
            del checkpoint['Y']
        if 'mask' in checkpoint:
            del checkpoint['mask']
        del x_tensor, y_tensor, mask_tensor, checkpoint
        gc.collect()
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Dataset path: {data_path}")
    print(f"X shape: {info['x_shape']}")
    print(f"Y shape: {info['y_shape']}")
    if info['mask_shape'] != 'N/A':
        print(f"Mask shape: {info['mask_shape']}")
        if info['coverage'] is not None:
            print(f"Data coverage: {info['coverage']:.1f}%")
    print(f"Number of assets: {len(info['asset_names'])}")
    if info['num_features'] is not None:
        print(f"Number of features: {info['num_features']}")
    if info['feature_names']:
        print(f"Features: {info['feature_names']}")
    print("="*60 + "\n")
    
    return info

def train(args):
    # 1. Load Data
    print("="*60)
    print("MARKETBDH PRE-TRAINING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Determine dataset path
    if args.dataset:
        data_path = Path(args.dataset)
    else:
        data_path = Path(__file__).parent.parent / "data" / "pretrain_dataset.pt"
    
    if not data_path.exists():
        print(f"\nError: Processed dataset not found at {data_path}")
        print("Please run 'python scripts/build_dataset.py' first.")
        sys.exit(1)
    
    # Load and display dataset info
    dataset_info = load_dataset_info(data_path)
    
    # Create datasets with pretrain_mode=True
    print("Loading training and validation datasets (pre-training mode)...")
    train_dataset = MarketDataset(
        data_path=str(data_path),
        window_size=args.window_size,
        split='train',
        val_split=args.val_split,
        pretrain_mode=True
    )
    
    val_dataset = MarketDataset(
        data_path=str(data_path),
        window_size=args.window_size,
        split='val',
        val_split=args.val_split,
        feature_mean=train_dataset.feature_mean,
        feature_std=train_dataset.feature_std,
        pretrain_mode=True
    )
    
    print(f"\nDataset splits:")
    print(f"  Train sequences: {len(train_dataset):,}")
    print(f"  Val sequences: {len(val_dataset):,}")
    print(f"  Assets: {train_dataset.num_assets}")
    print(f"  Features: {train_dataset.num_features}")
    print(f"  Window size: {args.window_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    # 2. Initialize Model
    print("\n" + "="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    config = MarketBDHConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        dropout=args.dropout,
        vocab_size=1,  # Dummy value, not used
        num_assets=train_dataset.num_assets,
        input_features=train_dataset.num_features,
        num_classes=2,  # Not used in pre-training, but required by config
        pretrain_mode=True
    )
    
    model = MarketBDHPretrain(config).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model architecture:")
    print(f"  Layers: {args.n_layer}")
    print(f"  Embedding dim: {args.n_embd}")
    print(f"  Attention heads: {args.n_head}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Total parameters: {num_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {num_trainable / 1e6:.2f}M")
    print("="*60 + "\n")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate * 10,
        total_steps=args.max_iters,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    print("Training configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Early stopping patience: {args.early_stop_patience}")
    print(f"  Evaluation frequency: {args.eval_freq}")
    print()
    
    # 3. Training Loop with Early Stopping
    print("="*60)
    print("PRE-TRAINING")
    print("="*60)
    print(f"Starting pre-training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    model.train()
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    
    step = 0
    train_iter = iter(train_loader)
    
    for step in range(args.max_iters):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Handle mask if present
        if len(batch) == 3:
            x, x_next, mask = batch
            mask = mask.to(DEVICE)
        else:
            x, x_next = batch
            mask = None
            
        x = x.to(DEVICE)
        x_next = x_next.to(DEVICE)
        
        predicted_features, loss = model(x, targets=x_next, mask=mask)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        
        if step % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step:5d}/{args.max_iters} | Loss: {loss.item():.4f} | "
                  f"Grad: {grad_norm:.3f} | LR: {current_lr:.2e}")
            
        if step > 0 and step % args.eval_freq == 0:
            val_loss = evaluate(model, val_loader)
            val_losses.append(val_loss)
            model.train()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"✓ New best: Loss={best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f"\nEarly stopping triggered at step {step}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\nRestored best model state.")
    
    print("\nPre-training complete.")
    final_val_loss = evaluate(model, val_loader)
    
    # Save model checkpoint
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"market_bdh_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_loss': best_val_loss,
        'final_val_loss': final_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'step': step,
    }, checkpoint_path)
    print(f"\nPre-trained model checkpoint saved to: {checkpoint_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PRE-TRAINING SUMMARY")
    print("="*60)
    print(f"Total steps: {step + 1}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Pre-training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\nTo fine-tune this model, use:")
    print(f"  python scripts/train_market.py --pretrain-checkpoint {checkpoint_path}")

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            # Handle mask if present
            if len(batch) == 3:
                x, x_next, mask = batch
                mask = mask.to(DEVICE)
            else:
                x, x_next = batch
                mask = None
                
            x = x.to(DEVICE)
            x_next = x_next.to(DEVICE)
            
            predicted_features, loss = model(x, targets=x_next, mask=mask)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"\n--- Evaluation ---")
    print(f"Val Loss (MSE): {avg_loss:.4f}")
    print(f"------------------\n")
    
    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pre-train MarketBDH model on processed market dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to processed dataset (.pt file). Default: data/market_dataset.pt'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help='Sequence length (window size)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help='Fraction of data to use for validation'
    )
    
    # Model arguments
    parser.add_argument(
        '--n-layer',
        type=int,
        default=DEFAULT_N_LAYER,
        help='Number of BDH layers'
    )
    parser.add_argument(
        '--n-embd',
        type=int,
        default=DEFAULT_N_EMBD,
        help='Embedding dimension'
    )
    parser.add_argument(
        '--n-head',
        type=int,
        default=DEFAULT_N_HEAD,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=DEFAULT_DROPOUT,
        help='Dropout rate'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help='Weight decay (L2 regularization)'
    )
    parser.add_argument(
        '--grad-clip',
        type=float,
        default=DEFAULT_GRAD_CLIP,
        help='Gradient clipping threshold'
    )
    parser.add_argument(
        '--max-iters',
        type=int,
        default=DEFAULT_MAX_ITERS,
        help='Maximum number of training iterations'
    )
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=DEFAULT_EVAL_FREQ,
        help='Evaluation frequency (in steps)'
    )
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help='Early stopping patience (number of evaluations without improvement)'
    )
    
    args = parser.parse_args()
    train(args)

