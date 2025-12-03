"""
This script implements the two-phase training strategy (Pre-training via Self-Supervision

Fine-tuning via Classification) tailored for the MarketBDH architecture.
It handles:
Time-Series Windowing: Slicing continuous market data into sequence batches.
Masking: Ensuring loss is only calculated on valid asset data (ignoring zero-padded periods for assets that didn't exist yet).
Phase A (Physics): Training the model to predict next-day feature vectors.
Phase B (Alpha): Transferring the "brain" to predict binary return direction.
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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.bdh.market_pre_fine import MarketBDH
from src.dataset.preprocess import process_market_data_from_parquet

# --- Configuration ---
CONFIG = {
    # Data
    'parquet_path': './data/parquet',
    'library': 'nasdaq100',
    'min_years': 3.0,
    'seq_len': 64,  # Context window (T)
    'train_split': 0.8,

    # Model Architecture
    'd_model': 256,
    'n_neurons': 4096,  # Should be >> num_assets
    'n_heads': 4,
    'n_layers': 4,

    # Training - Phase A (Pretrain)
    'pretrain_epochs': 50,
    'pretrain_lr': 1e-3,
    'pretrain_batch_size': 32,

    # Training - Phase B (Finetune)
    'finetune_epochs': 20,
    'finetune_lr': 1e-4,
    'finetune_batch_size': 32,

    # System
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './checkpoints'
}


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


# --- Training Functions ---

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


def masked_bce_loss(pred, target, mask):
    """Binary Cross Entropy ignoring invalid assets."""
    # pred, target, mask: [B, T, N]
    bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
    masked_bce = bce * mask
    return masked_bce.sum() / (mask.sum() + 1e-8)


def run_pretrain_phase(model, train_loader, val_loader, config):
    print("\n=== Phase A: Pre-training (Market Physics) ===")
    model.set_mode('pretrain')
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
            pred = model(inp)  # Returns features

            loss = masked_mse_loss(pred, tgt, mask)
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

                pred = model(inp)
                val_loss += masked_mse_loss(pred, tgt, mask).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}")

        # Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"{config['save_dir']}/bdh_pretrain_best.pt")


def run_finetune_phase(model, train_loader, val_loader, config):
    print("\n=== Phase B: Fine-tuning (Directional Alpha) ===")

    # Load best brain structure
    print("Loading best pre-trained weights...")
    model.load_state_dict(torch.load(f"{config['save_dir']}/bdh_pretrain_best.pt"))

    model.set_mode('finetune')
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
            pred = model(inp)  # Returns probabilities [B, T, N]

            loss = masked_bce_loss(pred, tgt, mask)
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

                pred = model(inp)
                val_loss += masked_bce_loss(pred, tgt, mask).item()

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
            torch.save(model.state_dict(), f"{config['save_dir']}/bdh_finetuned_best.pt")


# --- Main ---
def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # 1. Load & Process Data
    print("Step 1: Ingesting Data...")
    try:
        data = process_market_data_from_parquet(
            CONFIG['parquet_path'],
            library=CONFIG['library'],
            min_years=CONFIG['min_years']
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    X, Y, mask = data['X'], data['Y'], data['mask']

    # 2. Split Data (Time-based split)
    split_idx = int(len(X) * CONFIG['train_split'])

    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    mask_train, mask_val = mask[:split_idx], mask[split_idx:]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    train_ds = MarketSequenceDataset(X_train, Y_train, mask_train, CONFIG['seq_len'])
    val_ds = MarketSequenceDataset(X_val, Y_val, mask_val, CONFIG['seq_len'])

    # 3. DataLoaders
    # Note: Shuffle=True for pretraining helps break correlation between batches,
    # but intra-batch sequence order is preserved.
    train_dl = DataLoader(train_ds, batch_size=CONFIG['pretrain_batch_size'], shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['pretrain_batch_size'], shuffle=False, num_workers=4)

    # 4. Initialize Model
    print("Step 2: Initializing Dragon Hatchling...")
    num_assets = X.shape[1]
    num_features = X.shape[2]

    model = MarketBDH(
        num_assets=num_assets,
        num_features=num_features,
        d_model=CONFIG['d_model'],
        n_neurons=CONFIG['n_neurons'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers']
    ).to(CONFIG['device'])

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 5. Run Pre-training
    run_pretrain_phase(model, train_dl, val_dl, CONFIG)

    # 6. Run Fine-tuning
    # Update loaders for finetune batch size if needed
    train_dl = DataLoader(train_ds, batch_size=CONFIG['finetune_batch_size'], shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['finetune_batch_size'], shuffle=False, num_workers=4)

    run_finetune_phase(model, train_dl, val_dl, CONFIG)

    print("\nTraining Complete.")


if __name__ == "__main__":
    main()