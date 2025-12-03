#!/usr/bin/env python3
"""
Train MarketBDH model on processed market dataset.

This script loads the processed market_dataset.pt file and trains the MarketBDH model
with various regularization techniques to prevent overfitting.
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
from bdh.market_bidirect_finetune import MarketBDH, load_pretrained_weights
from bdh.data import MarketDataset

# --- Default Configuration ---
DEFAULT_BATCH_SIZE = 32
DEFAULT_WINDOW_SIZE = 32
DEFAULT_MAX_ITERS = 1 # 2000
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_EVAL_FREQ = 50
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_LABEL_SMOOTHING = 0.01 # Softens targets to prevent overconfidence
DEFAULT_EARLY_STOP_PATIENCE = 50
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_N_LAYER = 2
DEFAULT_N_EMBD = 128
DEFAULT_N_HEAD = 2
DEFAULT_DROPOUT = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset_info(data_path):
    """Load and display dataset metadata."""
    data = torch.load(data_path, map_location='cpu')
    
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Dataset file: {data_path}")
    print(f"Assets: {len(data['asset_names'])}")
    print(f"Time steps: {data['X'].shape[0]}")
    print(f"Features per asset: {data['X'].shape[2]}")
    print(f"Data shape X: {data['X'].shape}")
    print(f"Data shape Y: {data['Y'].shape}")
    
    if 'mask' in data:
        valid_ratio = data['mask'].sum() / data['mask'].size * 100
        print(f"Data coverage: {valid_ratio:.1f}% valid (union-based alignment)")
    
    if 'asset_date_ranges' in data:
        print(f"\nDate ranges (showing first 5 assets):")
        for i, (name, (start, end)) in enumerate(zip(data['asset_names'], data['asset_date_ranges'])):
            if i < 5:
                print(f"  {name}: {start} to {end}")
            elif i == 5:
                print(f"  ... ({len(data['asset_names']) - 5} more assets)")
                break
    
    print("="*60 + "\n")
    
    return data


def train(args):
    # 1. Load Data
    print("="*60)
    print("MARKETBDH TRAINING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Determine dataset path
    if args.dataset:
        data_path = Path(args.dataset)
    else:
        data_path = Path(__file__).parent.parent / "data" / "market_dataset.pt"
    
    if not data_path.exists():
        print(f"\nError: Processed dataset not found at {data_path}")
        print("Please run 'python scripts/build_dataset.py' first.")
        sys.exit(1)
    
    # Load and display dataset info
    dataset_info = load_dataset_info(data_path)
    
    # Create datasets
    print("Loading training and validation datasets...")
    train_dataset = MarketDataset(
        data_path=str(data_path),
        window_size=args.window_size,
        split='train',
        val_split=args.val_split
    )
    
    val_dataset = MarketDataset(
        data_path=str(data_path),
        window_size=args.window_size,
        split='val',
        val_split=args.val_split,
        feature_mean=train_dataset.feature_mean,
        feature_std=train_dataset.feature_std
    )
    
    print(f"\nDataset splits:")
    print(f"  Train sequences: {len(train_dataset):,}")
    print(f"  Val sequences: {len(val_dataset):,}")
    print(f"  Assets: {train_dataset.num_assets}")
    print(f"  Features: {train_dataset.num_features}")
    print(f"  Window size: {args.window_size}")
    
    # Check class balance
    train_labels = train_dataset.raw_Y.flatten()
    class_counts = torch.bincount(torch.from_numpy(train_labels))
    print(f"\nClass distribution:")
    print(f"  Class 0 (Down): {class_counts[0].item():,}")
    print(f"  Class 1 (Up): {class_counts[1].item():,}")
    if len(class_counts) == 2 and class_counts[1].item() > 0:
        balance_ratio = class_counts[0].item() / class_counts[1].item()
        print(f"  Balance ratio: {balance_ratio:.2f}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
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
        num_classes=2,
        label_smoothing=args.label_smoothing
    )
    
    model = MarketBDH(config).to(DEVICE)
    
    # Load pre-trained weights if provided
    if args.pretrain_checkpoint:
        try:
            load_pretrained_weights(model, args.pretrain_checkpoint)
            print(f"  Using pre-trained weights from: {args.pretrain_checkpoint}")
            if args.freeze_core:
                # Freeze core layers (input_proj + bdh)
                for param in model.input_proj.parameters():
                    param.requires_grad = False
                for param in model.bdh.parameters():
                    param.requires_grad = False
                print(f"  Core layers frozen (only classification head will be trained)")
        except Exception as e:
            print(f"  Warning: Failed to load pre-trained weights: {e}")
            print(f"  Continuing with randomly initialized weights...")
    else:
        print(f"  Using randomly initialized weights")
    
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
        max_lr=args.learning_rate * 10,  # Peak LR is 10x base
        total_steps=args.max_iters,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    print("Training configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Gradient clip: {args.grad_clip}")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Early stopping patience: {args.early_stop_patience}")
    print(f"  Evaluation frequency: {args.eval_freq}")
    print()
    
    # 3. Training Loop with Early Stopping
    print("="*60)
    print("TRAINING")
    print("="*60)
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    model.train()
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accs = []
    best_model_state = None
    
    step = 0
    train_iter = iter(train_loader)
    
    for step in range(args.max_iters):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Handle mask if present (dataset returns (x, y, mask) or (x, y))
        if len(batch) == 3:
            x, targets, mask = batch
            mask = mask.to(DEVICE)
        else:
            x, targets = batch
            mask = None
            
        x = x.to(DEVICE)
        targets = targets.to(DEVICE)
        
        logits, loss = model(x, targets, mask=mask)
        
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
            val_loss, val_acc = evaluate(model, val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            model.train()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"✓ New best: Loss={best_val_loss:.4f}, Acc={best_val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f"\nEarly stopping triggered at step {step}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    print(f"Best validation accuracy: {best_val_acc:.4f}")
                    break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\nRestored best model state.")
    
    print("\nTraining complete.")
    final_val_loss, final_val_acc = evaluate(model, val_loader)
    
    # Save model checkpoint if requested
    if args.save_checkpoint:
        checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / f"market_bdh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'final_val_loss': final_val_loss,
            'final_val_acc': final_val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'step': step,
        }, checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total steps: {step + 1}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if val_accs:
        print(f"Best validation accuracy: {max(val_accs):.4f} ({max(val_accs)*100:.2f}%)")
    print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            # Handle mask if present
            if len(batch) == 3:
                x, targets, mask = batch
                mask = mask.to(DEVICE)
            else:
                x, targets = batch
                mask = None
                
            x = x.to(DEVICE)
            targets = targets.to(DEVICE)
            
            logits, loss = model(x, targets, mask=mask)
            total_loss += loss.item()
            
            # Accuracy (only count valid predictions if mask is provided)
            preds = torch.argmax(logits, dim=-1)
            if mask is not None:
                # Only count accuracy on valid data points
                valid_mask = mask > 0.5
                correct += ((preds == targets) & valid_mask).sum().item()
                total += valid_mask.sum().item()
            else:
                correct += (preds == targets).sum().item()
                total += targets.numel()
            
    avg_loss = total_loss / len(loader)
    acc = correct / total
    print(f"\n--- Evaluation ---")
    print(f"Val Loss: {avg_loss:.4f}")
    print(f"Val Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"------------------\n")
    
    return avg_loss, acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train MarketBDH model on processed market dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to market_dataset.pt file (default: data/market_dataset.pt)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help='Sequence window size'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help='Validation split ratio'
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
        '--label-smoothing',
        type=float,
        default=DEFAULT_LABEL_SMOOTHING,
        help='Label smoothing factor'
    )
    parser.add_argument(
        '--grad-clip',
        type=float,
        default=DEFAULT_GRAD_CLIP,
        help='Gradient clipping max norm'
    )
    parser.add_argument(
        '--max-iters',
        type=int,
        default=DEFAULT_MAX_ITERS,
        help='Maximum training iterations'
    )
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=DEFAULT_EVAL_FREQ,
        help='Evaluation frequency (steps)'
    )
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help='Early stopping patience (evaluations)'
    )
    
    # Other arguments
    parser.add_argument(
        '--save-checkpoint',
        action='store_true',
        help='Save model checkpoint after training'
    )
    
    # Pre-training arguments
    parser.add_argument(
        '--pretrain-checkpoint',
        type=str,
        default=None,
        help='Path to pre-trained model checkpoint (from train_bidirect_pretrain.py). If provided, loads pre-trained weights for fine-tuning.'
    )
    parser.add_argument(
        '--freeze-core',
        action='store_true',
        help='Freeze core layers (input_proj + bdh) when using pre-trained weights. Only classification head will be trained.'
    )
    
    args = parser.parse_args()
    train(args)
