#!/usr/bin/env python3
"""
Build pre-train dataset from date-based raw daily parquet files.
Refactored for memory efficiency and speed using a Scatter-Gather approach.
"""

import argparse
import gc
import multiprocessing as mp
import shutil
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import the single dataframe processor
from dataset.preprocess import process_single_dataframe

warnings.filterwarnings('ignore')


def get_symbol_col(df: pd.DataFrame) -> str:
    """Identify the symbol column name."""
    for col in ['Code', 'Symbol', 'code', 'symbol', 'ticker', 'Ticker']:
        if col in df.columns:
            return col
    raise ValueError(f"No symbol column found. Columns: {df.columns}")


def parse_date_from_filename(filename: str) -> Optional[str]:
    import re
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return None


# -----------------------------------------------------------------------------
# PHASE 1: SCATTER
# Read daily files and partition them into N temporary buckets based on Symbol Hash.
# -----------------------------------------------------------------------------
def partition_data_to_disk(raw_daily_path: Path, temp_dir: Path, n_bins: int,
                           start_date: str, end_date: str) -> List[str]:
    """
    Reads daily files and writes them into n_bins temporary folders.
    Returns the list of valid bins created.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Setup bin directories
    for i in range(n_bins):
        (temp_dir / str(i)).mkdir(exist_ok=True)

    # Find files
    files = sorted(list(raw_daily_path.glob("*.parquet")))
    files_to_process = []

    print("Filtering files by date...")
    for f in files:
        d_str = parse_date_from_filename(f.name)
        if d_str and (not start_date or d_str >= start_date) and (not end_date or d_str <= end_date):
            files_to_process.append((d_str, f))

    print(f"Partitioning {len(files_to_process)} daily files into {n_bins} intermediate bins...")

    # Buffer for writing: buffer[bin_id] = list of dataframes
    buffer: Dict[int, List[pd.DataFrame]] = {i: [] for i in range(n_bins)}
    buffer_size_rows = 0
    FLUSH_THRESHOLD = 500_000  # Flush to disk every 500k rows total

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    pbar = tqdm(files_to_process, desc="Scattering data")
    chunk_counter = 0

    for date_str, f_path in pbar:
        try:
            df = pd.read_parquet(f_path)

            # Standardization
            sym_col = get_symbol_col(df)
            cols_needed = [sym_col] + required_cols
            if not all(c in df.columns for c in cols_needed):
                continue

            if 'date' not in df.columns:
                df['date'] = pd.to_datetime(date_str)
            else:
                df['date'] = pd.to_datetime(df['date'])

            # Keep only necessary columns to save memory
            subset = df[[sym_col, 'date'] + required_cols].copy()

            # Assign Bin ID
            # We map string symbol to integer 0..n_bins
            # Using simple hash is fast enough
            bin_ids = subset[sym_col].map(lambda x: hash(str(x)) % n_bins)

            # Split and add to buffer
            # Grouping by bin_id is faster than iterating rows
            for b_id, group in subset.groupby(bin_ids):
                buffer[b_id].append(group)
                buffer_size_rows += len(group)

            # Flush if buffer is full
            if buffer_size_rows >= FLUSH_THRESHOLD:
                _flush_buffer(buffer, temp_dir, chunk_counter)
                chunk_counter += 1
                buffer = {i: [] for i in range(n_bins)}
                buffer_size_rows = 0
                gc.collect()

        except Exception as e:
            print(f"Error processing {f_path.name}: {e}")

    # Final Flush
    if buffer_size_rows > 0:
        _flush_buffer(buffer, temp_dir, chunk_counter)

    return [str(i) for i in range(n_bins)]


def _flush_buffer(buffer, temp_dir, chunk_id):
    """Write accumulated dataframes to disk."""
    for b_id, dfs in buffer.items():
        if not dfs:
            continue

        # Concatenate all data for this bin
        bin_df = pd.concat(dfs, ignore_index=True)

        # Write to a chunk file
        # Format: temp/bin_id/chunk_X.parquet
        save_path = temp_dir / str(b_id) / f"chunk_{chunk_id}.parquet"
        bin_df.to_parquet(save_path, index=False, compression='snappy')


# -----------------------------------------------------------------------------
# PHASE 2: GATHER & PROCESS
# Process each bin independently. A bin contains full history for a subset of symbols.
# -----------------------------------------------------------------------------
def process_bin_worker(args):
    """
    Worker function to process one bin directory.
    1. Loads all chunks in the bin.
    2. Groups by symbol.
    3. Runs feature extraction.
    4. Returns list of processed data (symbol, df, feature_names).
    """
    bin_dir, min_days = args
    bin_dir = Path(bin_dir)

    # 1. Load all chunks
    files = list(bin_dir.glob("*.parquet"))
    if not files:
        return []

    try:
        # Load and concat
        full_bin_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

        if full_bin_df.empty:
            return []

        sym_col = get_symbol_col(full_bin_df)

        # 2. Group by Symbol
        results = []

        # Iterate unique symbols in this bin
        for symbol, sub_df in full_bin_df.groupby(sym_col):
            # Sort by date
            sub_df = sub_df.sort_values('date').set_index('date')

            # Filter length
            if len(sub_df) < min_days:
                continue

            # 3. Feature Extraction (using existing logic)
            # We use the imported function from preprocess.py
            # For pre-training, we don't need the Target column (we predict X[t+1] instead)
            proc_df, feats = process_single_dataframe(sub_df, str(symbol), include_target=False)

            if proc_df is not None and len(proc_df) >= min_days:
                # We return the dataframe to main process for alignment
                # For massive datasets, we might return numpy arrays, but DF is safer for alignment
                results.append((str(symbol), proc_df, feats))

        return results

    except Exception as e:
        print(f"Worker failed on {bin_dir}: {e}")
        return []


# -----------------------------------------------------------------------------
# PHASE 3: ALIGNMENT & SAVING
# -----------------------------------------------------------------------------
def align_and_save(results_flat, output_path, min_years):
    """
    Aligns processed dataframes to a global date union and saves to .pt
    """
    print("\nPhase 3: Alignment and Saving...")

    processed_dfs = []
    processed_names = []
    feature_columns = None

    # Unpack results
    for symbol, df, feats in results_flat:
        if feature_columns is None:
            feature_columns = feats
        processed_dfs.append(df)
        processed_names.append(symbol)

    if not processed_dfs:
        raise ValueError("No valid symbols processed.")

    print(f"Merging {len(processed_names)} symbols. Features: {len(feature_columns)}")

    # 1. Determine Union Dates
    all_dates = set()
    asset_date_ranges = []

    print("Calculating date union...")
    for df in processed_dfs:
        dates = set(df.index)
        all_dates.update(dates)
        asset_date_ranges.append((df.index.min(), df.index.max()))

    union_index = pd.DatetimeIndex(sorted(all_dates))
    print(f"Union date range: {union_index.min().date()} to {union_index.max().date()}")
    print(f"Total time steps: {len(union_index)}")

    # 2. Align Data
    aligned_features = []
    aligned_masks = []

    print("Aligning assets to global timeline...")
    for i, df in enumerate(tqdm(processed_dfs, desc="Aligning")):
        # Reindex
        df_aligned = df.reindex(union_index)

        # Mask: 1 where data existed, 0 where created by reindex
        mask = (~df_aligned[[feature_columns[0]]].isna().any(axis=1)).astype(np.float32)

        # Fill gaps (Forward fill then backward fill then 0)
        df_aligned = df_aligned.ffill().bfill().fillna(0)

        aligned_features.append(df_aligned[feature_columns].values.astype(np.float32))
        aligned_masks.append(mask.values)

        # Free memory of original DF
        processed_dfs[i] = None

    # 3. Stack and Convert to Tensors
    # Converting to Tensor allows torch.save to handle files > 4GB efficiently
    print("Stacking and converting to Tensors...")

    # Stack numpy arrays first (aligned_features is a list of arrays)
    X_np = np.stack(aligned_features, axis=1)  # [Time, Assets, Feats]
    mask_np = np.stack(aligned_masks, axis=1)

    # Convert to Torch Tensors
    X = torch.from_numpy(X_np)
    mask = torch.from_numpy(mask_np)

    # Free numpy memory
    del X_np, mask_np
    gc.collect()

    # 4. Save
    # For pre-training, we don't save Y (targets) since we predict X[t+1] instead
    output_data = {
        'X': X,
        'mask': mask,
        'asset_names': processed_names,
        'feature_names': feature_columns,
        'asset_date_ranges': [(dr[0].date(), dr[1].date()) for dr in asset_date_ranges],
        'union_dates': union_index.strftime('%Y-%m-%d').tolist()
    }

    print(f"Saving to {output_path} (Protocol 5)...")
    # pickle_protocol=5 allows massive files and is standard in Python 3.8+
    torch.save(output_data, output_path, pickle_protocol=5)

    # Stats
    total_cells = mask.numel()
    valid_cells = mask.sum().item()
    print("=" * 60)
    print(f"Saved to: {output_path}")
    print(f"Shape: {X.shape}")
    print(f"Coverage: {valid_cells / total_cells * 100:.2f}%")
    print("=" * 60)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def build_pretrain_dataset(raw_daily_path: str = 'data/raw_daily',
                           output_path: str = 'data/pretrain_dataset.pt',
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           min_years: float = 1.0,
                           n_jobs: int = 1,
                           temp_path: str = 'data/temp_processing'):
    project_root = Path(__file__).parent.parent

    # Resolve paths
    raw_path = Path(raw_daily_path) if Path(raw_daily_path).is_absolute() else project_root / raw_daily_path
    out_path = Path(output_path) if Path(output_path).is_absolute() else project_root / output_path
    tmp_path = Path(temp_path) if Path(temp_path).is_absolute() else project_root / temp_path

    # Clean temp
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    min_days = int(min_years * 252)

    # Set number of bins (shards).
    # Too few = large files, memory issues in workers.
    # Too many = too many file handles.
    # 100 is a good balance for typical stock markets (5k-10k symbols).
    N_BINS = 100

    print("=" * 60)
    print(f"BUILDING PRE-TRAIN DATASET (Optimized)")
    print(f"Jobs: {n_jobs if n_jobs > 0 else 'Auto'}")
    print("=" * 60)

    try:
        # Step 1: Partition (Scatter)
        print("\n--- Phase 1: Partitioning Data to Temp Storage ---")
        partition_data_to_disk(raw_path, tmp_path, N_BINS, start_date, end_date)

        # Step 2: Process (Map)
        print("\n--- Phase 2: Processing Symbols in Parallel ---")
        bin_dirs = [tmp_path / str(i) for i in range(N_BINS)]

        # Prepare args for workers: (bin_path, min_days)
        worker_args = [(str(p), min_days) for p in bin_dirs]

        all_results = []

        if n_jobs == 1:
            for args in tqdm(worker_args, desc="Processing bins (Serial)"):
                all_results.extend(process_bin_worker(args))
        else:
            cpu_count = mp.cpu_count() if n_jobs == -1 else n_jobs
            print(f"Spinning up {cpu_count} workers...")

            with mp.Pool(processes=cpu_count) as pool:
                # Use imap_unordered for better progress tracking
                for batch_result in tqdm(pool.imap_unordered(process_bin_worker, worker_args),
                                         total=len(worker_args),
                                         desc="Processing bins (Parallel)"):
                    all_results.extend(batch_result)

        # Step 3: Align & Save (Reduce)
        print("\n--- Phase 3: Finalizing Dataset ---")
        align_and_save(all_results, out_path, min_years)

    finally:
        # Cleanup
        print(f"\nCleaning up temp files in {tmp_path}...")
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-daily-path', type=str, default='data/raw_daily')
    parser.add_argument('--output', type=str, default='data/pretrain_dataset.pt')
    parser.add_argument('--temp-path', type=str, default='data/temp_build', help="Location for intermediate files")
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--min-years', type=float, default=3.0)
    parser.add_argument('--n-jobs', type=int, default=-1, help="-1 for all CPUs")

    args = parser.parse_args()

    build_pretrain_dataset(
        raw_daily_path=args.raw_daily_path,
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        min_years=args.min_years,
        n_jobs=args.n_jobs,
        temp_path=args.temp_path
    )


if __name__ == "__main__":
    main()
