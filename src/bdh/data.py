import torch
import numpy as np
from torch.utils.data import Dataset


class MarketDataset(Dataset):
    """
    PyTorch Dataset for Market Data.
    Loads pre-processed data from a .pt file.

    Handles normalization carefully to account for temporal masking
    (stocks that did not exist in certain time periods).
    """

    def __init__(self, data_path, window_size=64, split='train', val_split=0.2,
                 feature_mean=None, feature_std=None, pretrain_mode=False):
        """
        Args:
            data_path (str): Path to the processed .pt file.
            window_size (int): Sequence length T.
            split (str): 'train' or 'val'.
            val_split (float): Fraction of data to use for validation.
            feature_mean: Mean for normalization (from training set).
            feature_std: Std for normalization (from training set).
            pretrain_mode (bool): If True, return X[t+1] as targets for regression. 
                                 If False, return Y (classification targets).
        """
        self.window_size = window_size
        self.pretrain_mode = pretrain_mode

        # Load processed data
        print(f"Loading processed dataset from {data_path}...")
        data = torch.load(data_path)

        self.raw_X = data['X']  # Shape: [Total_Time, Num_Assets, Num_Features]
        self.raw_Y = data['Y']  # Shape: [Total_Time, Num_Assets]
        self.asset_names = data['asset_names']

        # Handle mask if present
        if 'mask' in data:
            self.mask = data['mask']  # Shape: [Total_Time, Num_Assets]
        else:
            # If no mask, assume all data is valid
            self.mask = np.ones_like(self.raw_Y, dtype=np.float32)

        self.num_assets = len(self.asset_names)
        self.num_features = self.raw_X.shape[-1]

        # Train/Val Split (Time-based)
        total_timesteps = self.raw_X.shape[0]
        split_idx = int(total_timesteps * (1 - val_split))

        if split == 'train':
            self.X = self.raw_X[:split_idx]
            self.Y = self.raw_Y[:split_idx]
            self.mask = self.mask[:split_idx]

            # --- MASKED NORMALIZATION ---
            # We must compute mean/std ONLY on valid data points.
            # Including the 0-padding from pre-IPO dates would skew stats towards zero.

            # Expand mask for broadcasting: [T, N] -> [T, N, 1]
            mask_expanded = self.mask[..., None]

            # Count valid elements (sum of mask)
            valid_count = mask_expanded.sum()
            if valid_count == 0: valid_count = 1.0  # Safety

            # 1. Compute Weighted Mean
            sum_x = (self.X * mask_expanded).sum(axis=(0, 1), keepdims=True)
            self.feature_mean = sum_x / valid_count

            # 2. Compute Weighted Standard Deviation
            # Var = E[X^2] - (E[X])^2
            sum_x2 = ((self.X ** 2) * mask_expanded).sum(axis=(0, 1), keepdims=True)
            mean_x2 = sum_x2 / valid_count
            self.feature_std = np.sqrt(mean_x2 - self.feature_mean ** 2 + 1e-8)

            # 3. Apply Normalization
            # Note: (0 - Mean) / Std results in a non-zero value.
            # We must apply the mask again to force padded regions back to 0.0.
            self.X = (self.X - self.feature_mean) / self.feature_std
            self.X = self.X * mask_expanded

            print(f"Normalized training features (masked):")
            print(f"  Mean: {self.feature_mean.squeeze()}")
            print(f"  Std:  {self.feature_std.squeeze()}")

        else:
            self.X = self.raw_X[split_idx:]
            self.Y = self.raw_Y[split_idx:]
            self.mask = self.mask[split_idx:]

            # Use provided normalization statistics (from training set)
            mask_expanded = self.mask[..., None]

            if feature_mean is not None and feature_std is not None:
                self.feature_mean = feature_mean
                self.feature_std = feature_std

                # Normalize and re-mask
                self.X = (self.X - self.feature_mean) / self.feature_std
                self.X = self.X * mask_expanded
                print(f"Applied normalization from training set")
            else:
                # Fallback: compute on validation set (masked)
                valid_count = mask_expanded.sum()
                if valid_count == 0: valid_count = 1.0

                sum_x = (self.X * mask_expanded).sum(axis=(0, 1), keepdims=True)
                self.feature_mean = sum_x / valid_count

                sum_x2 = ((self.X ** 2) * mask_expanded).sum(axis=(0, 1), keepdims=True)
                mean_x2 = sum_x2 / valid_count
                self.feature_std = np.sqrt(mean_x2 - self.feature_mean ** 2 + 1e-8)

                self.X = (self.X - self.feature_mean) / self.feature_std
                self.X = self.X * mask_expanded
                print(f"Warning: Using validation set statistics for normalization")

        # For pretrain_mode, we need an extra timestep for the target (X[t+1])
        # So valid_starts should exclude the last timestep
        if pretrain_mode:
            self.valid_starts = range(len(self.X) - window_size - 1)
        else:
            self.valid_starts = range(len(self.X) - window_size)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.window_size

        # Data is likely numpy arrays from the load, convert to torch here
        x = torch.from_numpy(self.X[start:end])
        mask = torch.from_numpy(self.mask[start:end])

        if self.pretrain_mode:
            # For pre-training: return X[t+1] as regression target
            # Ensure we don't exceed bounds
            if end + 1 <= len(self.X):
                x_next = torch.from_numpy(self.X[start+1:end+1])
                mask_next = torch.from_numpy(self.mask[start+1:end+1])
                return x, x_next, mask_next
            else:
                # Fallback: use last available timestep (shouldn't happen due to valid_starts)
                x_next = torch.from_numpy(self.X[end-1:end])
                mask_next = torch.from_numpy(self.mask[end-1:end])
                return x, x_next, mask_next
        else:
            # For fine-tuning: return Y (classification targets)
            y = torch.from_numpy(self.Y[start:end])
            return x, y, mask
