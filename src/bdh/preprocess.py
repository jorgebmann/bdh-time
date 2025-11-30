import pickle
import numpy as np
import pandas as pd

def process_market_data(data_path):
    """
    Processes raw market data from pickle file into tensors.
    
    Args:
        data_path (str): Path to raw .pkl file.
        
    Returns:
        dict: Dictionary containing processed tensors and metadata.
            {
                'X': np.array [Total_Time, N, F],
                'Y': np.array [Total_Time, N],
                'asset_names': list of str
            }
    """
    print(f"Processing raw data from {data_path}...")
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
        
    dfs = []
    asset_names = []
    
    for df in raw_data:
        # df has MultiIndex columns (Price, Ticker)
        ticker = df.columns[0][1]
        
        # Simplify columns to single level
        df_simple = df.xs(ticker, axis=1, level=1).copy()
        
        # Feature Engineering
        # 1. Log Returns
        df_simple['Log_Ret'] = np.log(df_simple['Close'] / df_simple['Close'].shift(1))
        
        # 2. Positive Momentum (Excitation)
        df_simple['Pos_Mom'] = df_simple['Log_Ret'].clip(lower=0)
        
        # 3. Negative Momentum (Inhibition)
        df_simple['Neg_Mom'] = df_simple['Log_Ret'].clip(upper=0).abs()
        
        # 4. Volatility (Energy) - Rolling Std of Log Returns
        df_simple['Vol_Energy'] = df_simple['Log_Ret'].rolling(window=20).std()
        
        # 5. Volume (Activity) - Normalized
        vol_rolling_mean = df_simple['Volume'].rolling(window=20).mean()
        df_simple['Vol_Norm'] = df_simple['Volume'] / (vol_rolling_mean + 1e-8)
        
        # 6. Target: Next Step Return Direction (Binary Classification)
        next_ret = df_simple['Log_Ret'].shift(-1)
        df_simple['Target'] = (next_ret > 0).astype(int)
        
        # Drop NaNs
        df_simple = df_simple.dropna()
        
        # --- FIX: Skip empty assets ---
        if len(df_simple) == 0:
            print(f"Warning: Asset {ticker} has no valid data after preprocessing. Skipping.")
            continue
            
        asset_names.append(ticker)
        dfs.append(df_simple)
        
    if not dfs:
        raise ValueError("No valid data found after preprocessing.")
        
    # Align Assets
    # Start with the intersection of all valid indices
    common_index = dfs[0].index
    for i in range(1, len(dfs)):
        common_index = common_index.intersection(dfs[i].index)
    
    if len(common_index) == 0:
        raise ValueError("No common time steps found across assets.")
        
    print(f"Aligned {len(asset_names)} assets over {len(common_index)} time steps.")
        
    # Filter and Stack
    aligned_features = []
    aligned_targets = []
    
    for df in dfs:
        df_aligned = df.loc[common_index]
        
        feats = df_aligned[['Pos_Mom', 'Neg_Mom', 'Vol_Energy', 'Vol_Norm']].values
        targs = df_aligned['Target'].values
        
        aligned_features.append(feats)
        aligned_targets.append(targs)
        
    # Stack along asset dimension
    X = np.stack(aligned_features, axis=1).astype(np.float32)
    Y = np.stack(aligned_targets, axis=1).astype(np.int64)
    
    return {
        'X': X,
        'Y': Y,
        'asset_names': asset_names
    }
