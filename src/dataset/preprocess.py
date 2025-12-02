import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from pathlib import Path


def _process_single_dataframe(df: pd.DataFrame, ticker: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Process a single DataFrame to extract both baseline and BDH-optimized features.

    Combines:
    1. The original 4 baseline features.
    2. The 10 new physical/biological features (split into Exc/Inh channels).
    3. Sentiment features (if available).

    Args:
        df: DataFrame with datetime index and OHLCV columns (may include sentiment)
        ticker: Ticker symbol for logging

    Returns:
        Tuple(Processed DataFrame, List of feature column names) or (None, [])
    """
    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Asset {ticker} missing columns {missing_cols}. Skipping.")
        return None, []

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            print(f"Warning: Asset {ticker} does not have datetime index. Skipping.")
            return None, []

    # Sort by date
    df = df.sort_index()
    
    # Handle sentiment data if present
    has_sentiment = 'sentiment' in df.columns
    if has_sentiment:
        # Interpolate missing sentiment values using time-based interpolation
        df['sentiment'] = df['sentiment'].interpolate(method='time', limit_direction='both')
        # Fill any remaining NaN values (at the beginning/end) with forward/backward fill
        df['sentiment'] = df['sentiment'].ffill().bfill().fillna(0)

    # --- 1. Original Baseline Features ---

    # Log Returns
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # Positive Momentum (Excitation)
    df['Pos_Mom'] = df['Log_Ret'].clip(lower=0)

    # Negative Momentum (Inhibition)
    df['Neg_Mom'] = df['Log_Ret'].clip(upper=0).abs()

    # Volatility (Energy) - Rolling Std of Log Returns
    df['Vol_Energy'] = df['Log_Ret'].rolling(window=20).std()

    # Volume (Activity) - Normalized
    vol_rolling_mean = df['Volume'].rolling(window=20).mean()
    df['Vol_Norm'] = df['Volume'] / (vol_rolling_mean + 1e-8)

    # --- 2. New BDH-Optimized Features (10 Physical Forces) ---

    # Pre-calculations for new features
    prev_close = df['Close'].shift(1)

    # Feature 1: Gap Force (Overnight Energy)
    gap = (df['Open'] - prev_close) / prev_close
    df['Gap_Exc'] = gap.clip(lower=0)
    df['Gap_Inh'] = gap.clip(upper=0).abs()

    # Feature 2: Intraday Kinetic Drive
    body = (df['Close'] - df['Open']) / df['Open']
    df['Body_Exc'] = body.clip(lower=0)
    df['Body_Inh'] = body.clip(upper=0).abs()

    # Feature 3 & 4: Shadows (Rejection vs Support)
    high_body = np.maximum(df['Open'], df['Close'])
    low_body = np.minimum(df['Open'], df['Close'])

    # Upper Wick -> Inhibitory (Selling pressure/Rejection)
    df['Wick_Upper_Inh'] = (df['High'] - high_body) / df['Close']

    # Lower Wick -> Excitatory (Buying support/Bounce)
    df['Wick_Lower_Exc'] = (low_body - df['Low']) / df['Close']

    # Feature 5: Trend Potential (20-Day Mean Reversion)
    ma20 = df['Close'].rolling(window=20).mean()
    dev20 = (df['Close'] / ma20) - 1
    df['Trend20_Exc'] = dev20.clip(lower=0)
    df['Trend20_Inh'] = dev20.clip(upper=0).abs()

    # Feature 6: Regime Status (200-Day Structural)
    ma200 = df['Close'].rolling(window=200).mean()
    dev200 = (df['Close'] / ma200) - 1
    df['Trend200_Exc'] = dev200.clip(lower=0)
    df['Trend200_Inh'] = dev200.clip(upper=0).abs()

    # Feature 7: Volatility Breakout (Expansion)
    tr = (df['High'] - df['Low']) / df['Close']
    avg_tr = tr.rolling(window=20).mean()
    vol_exp = (tr / avg_tr) - 1
    df['Vol_Exp_Exc'] = vol_exp.clip(lower=0)

    # Feature 8: Volume Impulse (Force Vector)
    vol_ratio = df['Volume'] / (vol_rolling_mean + 1e-8)
    direction = np.sign(df['Close'] - prev_close)
    vol_force = direction * vol_ratio
    df['Vol_Force_Exc'] = vol_force.clip(lower=0)
    df['Vol_Force_Inh'] = vol_force.clip(upper=0).abs()

    # Feature 9: Oscillator Extremes (RSI Stress)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    df['RSI_Oversold_Exc'] = (30 - rsi).clip(lower=0) / 30.0
    df['RSI_Overbought_Inh'] = (rsi - 70).clip(lower=0) / 30.0

    # Feature 10: Consecutive Strain (Streak Proxy)
    is_up = (df['Close'] > prev_close).astype(int)
    is_down = (df['Close'] < prev_close).astype(int)

    up_strain = is_up.rolling(3).sum() / 3.0
    down_strain = is_down.rolling(3).sum() / 3.0

    df['Streak_Inh'] = up_strain
    df['Streak_Exc'] = down_strain

    # --- Target Generation ---
    # Next Step Return Direction
    next_ret = np.log(df['Close'] / df['Close'].shift(1)).shift(-1)
    df['Target'] = (next_ret > 0).astype(int)

    # --- Collect Feature Columns ---
    original_cols = ['Pos_Mom', 'Neg_Mom', 'Vol_Energy', 'Vol_Norm']
    bdh_cols = [c for c in df.columns if c.endswith('_Exc') or c.endswith('_Inh')]
    
    # Add sentiment if present
    sentiment_cols = []
    if has_sentiment and 'sentiment' in df.columns:
        sentiment_cols = ['sentiment']

    feature_cols = original_cols + bdh_cols + sentiment_cols

    # Keep only relevant columns
    final_cols = feature_cols + ['Target']
    df = df[final_cols]

    # Drop NaNs (dominated by 200-day MA)
    df = df.dropna()

    # Skip empty assets
    if len(df) == 0:
        # 200 days history required for Feature 6
        return None, []

    return df, feature_cols


def process_market_data_from_dataframes(dfs: List[pd.DataFrame], asset_names: List[str],
                                        min_years: float = 3.0) -> dict:
    """
    Process market data from a list of DataFrames.
    Uses union of dates instead of intersection.
    """
    processed_dfs = []
    processed_names = []
    feature_columns = []

    # We need at least 200 days for features (Trend200) + training data
    min_history_days = 200
    min_training_days = int(min_years * 252)
    total_required = min_history_days + min_training_days

    print(f"Filtering stocks. Need >{total_required} days history...")

    for df, ticker in zip(dfs, asset_names):
        proc_df, feats = _process_single_dataframe(df.copy(), ticker)

        if proc_df is not None:
            if not feature_columns:
                feature_columns = feats

            if len(proc_df) >= min_training_days:
                processed_dfs.append(proc_df)
                processed_names.append(ticker)

    if not processed_dfs:
        raise ValueError(f"No valid data found after filtering.")

    print(f"\nProcessing {len(processed_names)} assets.")
    print(f"Total Features: {len(feature_columns)}")
    print(f"Columns: {feature_columns}")

    # Find union of all dates
    all_dates = set()
    asset_date_ranges = []

    for df in processed_dfs:
        dates = set(df.index)
        all_dates.update(dates)
        asset_date_ranges.append((df.index.min(), df.index.max()))

    # Sort dates
    union_index = pd.DatetimeIndex(sorted(all_dates))

    print(f"Union date range: {union_index.min().date()} to {union_index.max().date()}")
    print(f"Total time steps: {len(union_index)}")

    # Align all stocks to union dates
    aligned_features = []
    aligned_targets = []
    aligned_masks = []

    for i, df in enumerate(processed_dfs):
        # Reindex to union dates
        df_aligned = df.reindex(union_index)

        # Mask logic: 1 if data exists, 0 if missing. Check first col.
        mask = (~df_aligned[[feature_columns[0]]].isna().any(axis=1)).astype(np.float32)

        # Fill missing values (Forward fill state, then 0 for start gaps)
        df_aligned = df_aligned.ffill().bfill().fillna(0)

        feats = df_aligned[feature_columns].values
        targs = df_aligned['Target'].values

        aligned_features.append(feats)
        aligned_targets.append(targs)
        aligned_masks.append(mask)

        if i % 100 == 0 and i > 0:
            print(f"  Aligned {i}/{len(processed_names)} assets...")

    # Stack along asset dimension: [Time, Assets, Features]
    X = np.stack(aligned_features, axis=1).astype(np.float32)
    Y = np.stack(aligned_targets, axis=1).astype(np.int64)
    mask = np.stack(aligned_masks, axis=1).astype(np.float32)

    # Coverage stats
    total_possible = len(union_index) * len(processed_dfs)
    valid_data_points = mask.sum()
    coverage = valid_data_points / total_possible * 100

    print(f"\nData coverage: {valid_data_points:.0f}/{total_possible} ({coverage:.1f}%)")
    print(f"Final Tensor Shapes: X={X.shape}, Y={Y.shape}")

    return {
        'X': X,
        'Y': Y,
        'mask': mask,
        'asset_names': processed_names,
        'feature_names': feature_columns,
        'asset_date_ranges': [(dr[0].date(), dr[1].date()) for dr in asset_date_ranges],
        'union_dates': union_index.strftime('%Y-%m-%d').tolist()
    }


def process_market_data(data_path, min_years: float = 3.0):
    """Legacy wrapper for pickle files."""
    print(f"Processing raw data from {data_path}...")
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)

    dfs = []
    asset_names = []

    for df in raw_data:
        if isinstance(df.columns, pd.MultiIndex):
            ticker = df.columns[0][1]
            df_simple = df.xs(ticker, axis=1, level=1).copy()
            dfs.append(df_simple)
            asset_names.append(ticker)
        else:
            dfs.append(df)
            asset_names.append(f"Asset_{len(dfs)}")

    return process_market_data_from_dataframes(dfs, asset_names, min_years=min_years)


def process_market_data_from_parquet(parquet_path: str, library: str = 'nasdaq100',
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None,
                                     min_years: float = 3.0,
                                     sentiment_library: Optional[str] = None) -> dict:
    """
    Wrapper for ParquetStore that loads OHLCV and sentiment data.
    
    Args:
        parquet_path: Base path to parquet directory
        library: Library name for OHLCV data
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        min_years: Minimum years of data required per stock
        sentiment_library: Library name for sentiment data (default: same as library)
    """
    from dataset.ingestion import ParquetStore

    if sentiment_library is None:
        sentiment_library = library

    print(f"Loading data from parquet files in {parquet_path}/{library}...")
    parquet_store = ParquetStore(parquet_path)
    symbols = parquet_store.list_symbols(library)

    if not symbols:
        raise ValueError(f"No symbols found in library '{library}'")

    print(f"Found {len(symbols)} symbols...")

    dfs = []
    asset_names = []
    sentiment_loaded = 0

    for symbol in symbols:
        # Load OHLCV data
        df = parquet_store.read_symbol(symbol, start_date=start_date, end_date=end_date, library=library)
        if df is not None and not df.empty:
            # Try to load sentiment data
            sentiment_df = parquet_store.read_symbol(
                symbol, start_date=start_date, end_date=end_date, library=sentiment_library
            )
            
            if sentiment_df is not None and not sentiment_df.empty and 'sentiment' in sentiment_df.columns:
                # Merge sentiment data with OHLCV data on date index
                df = df.join(sentiment_df[['sentiment']], how='left')
                sentiment_loaded += 1
            
            ticker = symbol.replace('.US', '')
            dfs.append(df)
            asset_names.append(ticker)

    if not dfs:
        raise ValueError("No valid data found.")
    
    if sentiment_loaded > 0:
        print(f"Loaded sentiment data for {sentiment_loaded}/{len(dfs)} symbols")

    return process_market_data_from_dataframes(dfs, asset_names, min_years=min_years)