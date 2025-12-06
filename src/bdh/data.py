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
        import gc
        self.window_size = window_size
        self.pretrain_mode = pretrain_mode

        # Load processed data
        print(f"Loading processed dataset from {data_path}...")
        data = torch.load(data_path, map_location='cpu')

        # Extract full tensors
        full_X = data['X']  # Shape: [Total_Time, Num_Assets, Num_Features]
        full_Y = data.get('Y', None)  # Shape: [Total_Time, Num_Assets] or None
        full_mask = data.get('mask', None)  # Shape: [Total_Time, Num_Assets] or None
        self.asset_names = data['asset_names']

        # Ensure tensors are float
        if not isinstance(full_X, torch.Tensor):
            full_X = torch.from_numpy(full_X).float()
        else:
            full_X = full_X.float()

        if full_Y is not None:
            if not isinstance(full_Y, torch.Tensor):
                full_Y = torch.from_numpy(full_Y).float()
            else:
                full_Y = full_Y.float()

        if full_mask is not None:
            if not isinstance(full_mask, torch.Tensor):
                full_mask = torch.from_numpy(full_mask).float()
            else:
                full_mask = full_mask.float()
        else:
            # Create default mask if not present
            mask_shape = (full_X.shape[0], full_X.shape[1])
            full_mask = torch.ones(mask_shape, dtype=torch.float32)

        self.num_assets = len(self.asset_names)
        self.num_features = full_X.shape[-1]

        # Calculate split indices BEFORE slicing
        total_timesteps = full_X.shape[0]
        split_idx = int(total_timesteps * (1 - val_split))

        # Extract only the needed split and immediately delete full tensors
        if split == 'train':
            # Slice training data
            self.raw_X = full_X[:split_idx].clone()
            self.raw_Y = full_Y[:split_idx].clone() if full_Y is not None else None
            self.mask = full_mask[:split_idx].clone()
            
            # Delete full tensors immediately to free memory
            del full_X, full_Y, full_mask
            del data
            gc.collect()
            
            # --- MASKED NORMALIZATION ---
            # We must compute mean/std ONLY on valid data points.
            # Including the 0-padding from pre-IPO dates would skew stats towards zero.

            # Expand mask for broadcasting: [T, N] -> [T, N, 1]
            mask_expanded = self.mask[..., None]

            # Count valid elements (sum of mask)
            valid_count = mask_expanded.sum()
            if valid_count == 0: valid_count = 1.0  # Safety

            # 1. Compute Weighted Mean
            sum_x = (self.raw_X * mask_expanded).sum(dim=(0, 1), keepdim=True)
            self.feature_mean = sum_x / valid_count

            # 2. Compute Weighted Standard Deviation
            # Var = E[X^2] - (E[X])^2
            sum_x2 = ((self.raw_X ** 2) * mask_expanded).sum(dim=(0, 1), keepdim=True)
            mean_x2 = sum_x2 / valid_count
            self.feature_std = torch.sqrt(mean_x2 - self.feature_mean ** 2 + 1e-8)

            # 3. Apply Normalization
            # Note: (0 - Mean) / Std results in a non-zero value.
            # We must apply the mask again to force padded regions back to 0.0.
            self.X = (self.raw_X - self.feature_mean) / self.feature_std
            self.X = self.X * mask_expanded
            
            # Delete raw_X after normalization to save memory (keep normalized X)
            del self.raw_X
            # Set Y for consistency (Y doesn't need normalization)
            self.Y = self.raw_Y
            gc.collect()

            print(f"Normalized training features (masked):")
            print(f"  Mean: {self.feature_mean.squeeze()}")
            print(f"  Std:  {self.feature_std.squeeze()}")

        else:
            # Slice validation data
            self.raw_X = full_X[split_idx:].clone()
            self.raw_Y = full_Y[split_idx:].clone() if full_Y is not None else None
            self.mask = full_mask[split_idx:].clone()
            
            # Delete full tensors immediately to free memory
            del full_X, full_Y, full_mask
            del data
            gc.collect()

            # Use provided normalization statistics (from training set)
            mask_expanded = self.mask[..., None]

            if feature_mean is not None and feature_std is not None:
                # Ensure normalization stats are tensors
                if not isinstance(feature_mean, torch.Tensor):
                    self.feature_mean = torch.from_numpy(feature_mean).float()
                else:
                    self.feature_mean = feature_mean.float()
                if not isinstance(feature_std, torch.Tensor):
                    self.feature_std = torch.from_numpy(feature_std).float()
                else:
                    self.feature_std = feature_std.float()

                # Normalize and re-mask
                self.X = (self.raw_X - self.feature_mean) / self.feature_std
                self.X = self.X * mask_expanded
                
                # Delete raw_X after normalization
                del self.raw_X
                # Set Y for consistency (Y doesn't need normalization)
                self.Y = self.raw_Y
                gc.collect()
                
                print(f"Applied normalization from training set")
            else:
                # Fallback: compute on validation set (masked)
                valid_count = mask_expanded.sum()
                if valid_count == 0: valid_count = 1.0

                sum_x = (self.raw_X * mask_expanded).sum(dim=(0, 1), keepdim=True)
                self.feature_mean = sum_x / valid_count

                sum_x2 = ((self.raw_X ** 2) * mask_expanded).sum(dim=(0, 1), keepdim=True)
                mean_x2 = sum_x2 / valid_count
                self.feature_std = torch.sqrt(mean_x2 - self.feature_mean ** 2 + 1e-8)

                self.X = (self.raw_X - self.feature_mean) / self.feature_std
                self.X = self.X * mask_expanded
                
                # Delete raw_X after normalization
                del self.raw_X
                # Set Y for consistency (Y doesn't need normalization)
                self.Y = self.raw_Y
                gc.collect()
                
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

        # X and mask are already tensors
        x = self.X[start:end]
        mask = self.mask[start:end]

        if self.pretrain_mode:
            # For pre-training: return X[t+1] as regression target
            # Ensure we don't exceed bounds
            if end + 1 <= len(self.X):
                x_next = self.X[start+1:end+1]
                mask_next = self.mask[start+1:end+1]
                return x, x_next, mask_next
            else:
                # Fallback: use last available timestep (shouldn't happen due to valid_starts)
                x_next = self.X[end-1:end]
                mask_next = self.mask[end-1:end]
                return x, x_next, mask_next
        else:
            # For fine-tuning: return Y (classification targets)
            if self.Y is None:
                raise ValueError("Y (targets) not found in dataset. This dataset appears to be for pre-training only.")
            # Y is already a tensor
            y = self.Y[start:end]
            return x, y, mask
