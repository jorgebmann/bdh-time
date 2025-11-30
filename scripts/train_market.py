import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bdh.market import MarketBDH, MarketBDHConfig
from bdh.data import MarketDataset

# --- Configuration ---
BATCH_SIZE = 32
WINDOW_SIZE = 32  # Reduced from 64
MAX_ITERS = 2000  # Increased with early stopping
LEARNING_RATE = 5e-5  # Reduced from 1e-4
EVAL_FREQ = 50  # More frequent evaluation
WEIGHT_DECAY = 0.01  # L2 regularization
GRAD_CLIP = 1.0  # Gradient clipping
LABEL_SMOOTHING = 0.1  # Label smoothing
EARLY_STOP_PATIENCE = 50  # Early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    # 1. Load Data
    print("Loading dataset...")
    data_path = Path(__file__).parent.parent / "data" / "market_dataset.pt"
    
    if not data_path.exists():
        print(f"Error: Processed dataset not found at {data_path}")
        print("Please run 'python scripts/build_dataset.py' first.")
        sys.exit(1)
    
    train_dataset = MarketDataset(
        data_path=str(data_path),
        window_size=WINDOW_SIZE,
        split='train',
        val_split=0.2
    )
    
    val_dataset = MarketDataset(
        data_path=str(data_path),
        window_size=WINDOW_SIZE,
        split='val',
        val_split=0.2,
        feature_mean=train_dataset.feature_mean,
        feature_std=train_dataset.feature_std
    )
    
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")
    print(f"Assets: {train_dataset.num_assets}, Features: {train_dataset.num_features}")
    
    # Check class balance
    train_labels = train_dataset.Y.flatten()
    class_counts = torch.bincount(torch.from_numpy(train_labels))
    print(f"Class distribution: {class_counts.numpy()}")
    if len(class_counts) == 2:
        balance_ratio = class_counts[0].item() / class_counts[1].item()
        print(f"Class balance ratio: {balance_ratio:.2f}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model (Reduced size to prevent overfitting)
    config = MarketBDHConfig(
        n_layer=2,  # Reduced from 4
        n_embd=128,  # Reduced from 256
        n_head=2,  # Reduced from 4
        dropout=0.3,  # Increased from 0.1
        vocab_size=1, # Dummy value, not used
        num_assets=train_dataset.num_assets,
        input_features=train_dataset.num_features,
        num_classes=2,
        label_smoothing=LABEL_SMOOTHING
    )
    
    model = MarketBDH(config).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE * 10,  # Peak LR is 10x base
        total_steps=MAX_ITERS,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    # 3. Training Loop with Early Stopping
    print("\nStarting training...")
    model.train()
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accs = []
    
    step = 0
    train_iter = iter(train_loader)
    
    for step in range(MAX_ITERS):
        try:
            x, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, targets = next(train_iter)
            
        x = x.to(DEVICE)
        targets = targets.to(DEVICE)
        
        logits, loss = model(x, targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        
        if step % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step}/{MAX_ITERS} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
            
        if step > 0 and step % EVAL_FREQ == 0:
            val_loss, val_acc = evaluate(model, val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            model.train()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"✓ New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"\nEarly stopping at step {step}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break
            
    print("\nTraining complete.")
    final_val_loss, final_val_acc = evaluate(model, val_loader)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if val_accs:
        print(f"Best validation accuracy: {max(val_accs):.4f}")
    print("="*60)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, targets in loader:
            x = x.to(DEVICE)
            targets = targets.to(DEVICE)
            
            logits, loss = model(x, targets)
            total_loss += loss.item()
            
            # Accuracy
            preds = torch.argmax(logits, dim=-1)
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
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train(args)
