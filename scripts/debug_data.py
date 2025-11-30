import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def debug_intersection():
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / "data" / "nasdaq100_data.pkl"
    
    with open(raw_data_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    print(f"Loaded {len(raw_data)} dataframes")
    
    # Check indices after dropna
    indices = []
    for i, df in enumerate(raw_data):
        ticker = df.columns[0][1]
        df_simple = df.xs(ticker, axis=1, level=1).copy()
        
        # Replicate processing
        df_simple['Log_Ret'] = np.log(df_simple['Close'] / df_simple['Close'].shift(1))
        df_simple['Pos_Mom'] = df_simple['Log_Ret'].clip(lower=0)
        df_simple['Neg_Mom'] = df_simple['Log_Ret'].clip(upper=0).abs()
        df_simple['Vol_Energy'] = df_simple['Log_Ret'].rolling(window=20).std()
        vol_rolling_mean = df_simple['Volume'].rolling(window=20).mean()
        df_simple['Vol_Norm'] = df_simple['Volume'] / (vol_rolling_mean + 1e-8)
        next_ret = df_simple['Log_Ret'].shift(-1)
        df_simple['Target'] = (next_ret > 0).astype(int)
        
        df_simple = df_simple.dropna()
        indices.append(df_simple.index)
        
        if i < 3:
            print(f"Asset {ticker}: {len(df_simple)} rows. Range: {df_simple.index[0]} to {df_simple.index[-1]}")

    # Compute intersection step by step
    common = indices[0]
    for i in range(1, len(indices)):
        prev_len = len(common)
        common = common.intersection(indices[i])
        curr_len = len(common)
        if curr_len == 0:
            print(f"Intersection became empty at index {i} (Asset {raw_data[i].columns[0][1]})")
            print(f"Previous common range: {indices[i-1][0]} to {indices[i-1][-1]}")
            print(f"Current asset range: {indices[i][0]} to {indices[i][-1]}")
            break
            
    print(f"Final intersection length: {len(common)}")

if __name__ == "__main__":
    debug_intersection()

