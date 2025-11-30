import torch
import numpy as np
from torch.utils.data import Dataset

class MarketDataset(Dataset):
    """
    PyTorch Dataset for Market Data.
    Loads pre-processed data from a .pt file.
    """
    def __init__(self, data_path, window_size=64, split='train', val_split=0.2, 
                 feature_mean=None, feature_std=None):
        """
        Args:
            data_path (str): Path to the processed .pt file.
            window_size (int): Sequence length T.
            split (str): 'train' or 'val'.
            val_split (float): Fraction of data to use for validation.
            feature_mean: Mean for normalization (from training set).
            feature_std: Std for normalization (from training set).
        """
        self.window_size = window_size
        
        # Load processed data
        print(f"Loading processed dataset from {data_path}...")
        data = torch.load(data_path)
        
        self.raw_X = data['X']
        self.raw_Y = data['Y']
        self.asset_names = data['asset_names']
        
        self.num_assets = len(self.asset_names)
        self.num_features = self.raw_X.shape[-1]
        
        # Train/Val Split (Time-based)
        total_timesteps = self.raw_X.shape[0]
        split_idx = int(total_timesteps * (1 - val_split))
        
        if split == 'train':
            self.X = self.raw_X[:split_idx]
            self.Y = self.raw_Y[:split_idx]
            
            # Compute normalization statistics on training data
            # Normalize per feature across all time steps and assets
            self.feature_mean = self.X.mean(axis=(0, 1), keepdims=True)
            self.feature_std = self.X.std(axis=(0, 1), keepdims=True) + 1e-8
            
            # Apply normalization
            self.X = (self.X - self.feature_mean) / self.feature_std
            print(f"Normalized training features: mean={self.feature_mean.squeeze()}, std={self.feature_std.squeeze()}")
        else:
            self.X = self.raw_X[split_idx:]
            self.Y = self.raw_Y[split_idx:]
            
            # Use provided normalization statistics (from training set)
            if feature_mean is not None and feature_std is not None:
                self.feature_mean = feature_mean
                self.feature_std = feature_std
                self.X = (self.X - self.feature_mean) / self.feature_std
                print(f"Applied normalization from training set")
            else:
                # Fallback: compute on validation set (not ideal but better than nothing)
                self.feature_mean = self.X.mean(axis=(0, 1), keepdims=True)
                self.feature_std = self.X.std(axis=(0, 1), keepdims=True) + 1e-8
                self.X = (self.X - self.feature_mean) / self.feature_std
                print(f"Warning: Using validation set statistics for normalization")
            
        self.valid_starts = range(len(self.X) - window_size)
        
    def __len__(self):
        return len(self.valid_starts)
        
    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.window_size
        
        # Use torch.from_numpy if data is numpy, or just slice if it's tensor.
        # Assuming torch.load returned the dictionary with numpy arrays from preprocess step.
        x = torch.from_numpy(self.X[start:end])
        y = torch.from_numpy(self.Y[start:end])
        
        return x, y
