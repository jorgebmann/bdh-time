#!/usr/bin/env python3
"""
Extract OHLCV values from all NASDAQ100 stocks from data/raw_daily.

This script reads daily parquet files from data/raw_daily, filters for NASDAQ100 tickers,
and extracts OHLCV data for each ticker, saving them in the same format as the
ParquetStore for consistency.
"""

import sys
import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset.ingestion import ParquetStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NASDAQ100 ticker list (as of 2024)
NASDAQ100_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "PEP", "ADBE", "CSCO", "CMCSA", "INTC", "TXN", "AMGN", "QCOM",
    "INTU", "ISRG", "VRTX", "BKNG", "ADP", "PAYX", "REGN", "CDNS", "SNPS", "CRWD",
    "MRVL", "KLAC", "NXPI", "CDW", "FTNT", "ODFL", "CTAS", "ANSS", "TEAM", "FAST",
    "PCAR", "EXPD", "IDXX", "DXCM", "ZS", "BKR", "MELI", "AEP", "GEHC", "ON",
    "TTD", "GFS", "CTSH", "DASH", "ROST", "XEL", "DLTR", "WBD", "EA", "ENPH",
    "VRSK", "CSGP", "TTWO", "ALGN", "EBAY", "ANET", "FANG", "LCID", "RIVN",
    "MCHP", "LULU", "WDAY", "CPRT", "MNST", "CHRW", "CEG", "MDB", "PANW",
    "DDOG", "NET", "OKTA", "DOCN", "ESTC", "SPLK", "NOW", "SNOW", "PLTR", "RBLX"
]

# Remove duplicates and sort
NASDAQ100_TICKERS = sorted(list(set(NASDAQ100_TICKERS)))

# Set of tickers for fast lookup
NASDAQ100_SET = set(NASDAQ100_TICKERS)


def get_symbol_col(df: pd.DataFrame) -> str:
    """Identify the symbol column name."""
    for col in ['Code', 'Symbol', 'code', 'symbol', 'ticker', 'Ticker']:
        if col in df.columns:
            return col
    raise ValueError(f"No symbol column found. Columns: {df.columns}")


def parse_date_from_filename(filename: str) -> Optional[str]:
    """Extract date from filename (e.g., us_market_1995-01-02.parquet -> 1995-01-02)."""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return None


def normalize_symbol(symbol) -> Optional[str]:
    """
    Normalize symbol to match NASDAQ100 ticker format.
    Handles cases like 'AAPL.US' -> 'AAPL' or just 'AAPL' -> 'AAPL'.
    Returns None if symbol is None, NaN, or empty.
    """
    # Handle None, NaN, or empty values
    if symbol is None or pd.isna(symbol):
        return None
    
    # Convert to string and strip whitespace
    symbol_str = str(symbol).strip()
    
    # Return None if empty after stripping
    if not symbol_str:
        return None
    
    # Remove exchange suffix if present (e.g., '.US', '.NASDAQ')
    if '.' in symbol_str:
        symbol_str = symbol_str.split('.')[0]
    
    return symbol_str.upper().strip()


def extract_ohlcv_from_raw_daily(
    raw_daily_path: Path,
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_path: Optional[str] = None,
    library: str = 'nasdaq100'
) -> Dict[str, pd.DataFrame]:
    """
    Extract OHLCV data for specified tickers from raw_daily parquet files.
    
    Args:
        raw_daily_path: Path to directory containing daily parquet files
        tickers: List of ticker symbols to extract
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        output_path: Optional path to save extracted data (uses ParquetStore)
        library: Library name for ParquetStore (default: nasdaq100)
        
    Returns:
        Dictionary mapping ticker to DataFrame with OHLCV data
    """
    # Normalize tickers and create lookup set (filter out None values)
    ticker_set = set()
    for t in tickers:
        normalized = normalize_symbol(t)
        if normalized is not None:
            ticker_set.add(normalized)
    logger.info(f"Extracting OHLCV data for {len(ticker_set)} NASDAQ100 tickers")
    
    # Dictionary to store dataframes by ticker
    ticker_data: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    
    # Required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Find all parquet files
    parquet_files = sorted(list(raw_daily_path.glob("*.parquet")))
    logger.info(f"Found {len(parquet_files)} parquet files in {raw_daily_path}")
    
    # Filter files by date if specified
    files_to_process = []
    for f in parquet_files:
        date_str = parse_date_from_filename(f.name)
        if date_str:
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            files_to_process.append((date_str, f))
    
    logger.info(f"Processing {len(files_to_process)} files after date filtering")
    
    # Process each file
    processed_files = 0
    skipped_files = 0
    
    for date_str, file_path in tqdm(files_to_process, desc="Extracting OHLCV data"):
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)
            
            if df.empty:
                skipped_files += 1
                continue
            
            # Identify symbol column
            try:
                sym_col = get_symbol_col(df)
            except ValueError as e:
                logger.debug(f"Skipping {file_path.name}: {e}")
                skipped_files += 1
                continue
            
            # Check if required columns exist
            if not all(col in df.columns for col in required_cols):
                logger.debug(f"Skipping {file_path.name}: Missing required OHLCV columns")
                skipped_files += 1
                continue
            
            # Add date column if not present
            if 'date' not in df.columns:
                df['date'] = pd.to_datetime(date_str)
            else:
                df['date'] = pd.to_datetime(df['date'])
            
            # Normalize symbols and filter for NASDAQ100 tickers
            # Filter out None/NaN values before normalization
            df_valid = df[df[sym_col].notna()].copy()
            if df_valid.empty:
                skipped_files += 1
                continue
            
            df_valid['normalized_symbol'] = df_valid[sym_col].apply(normalize_symbol)
            # Filter out rows where normalization returned None
            df_valid = df_valid[df_valid['normalized_symbol'].notna()].copy()
            if df_valid.empty:
                skipped_files += 1
                continue
            
            # Filter for NASDAQ100 tickers
            df_filtered = df_valid[df_valid['normalized_symbol'].isin(ticker_set)].copy()
            
            if df_filtered.empty:
                skipped_files += 1
                continue
            
            # Select only necessary columns
            cols_to_keep = ['normalized_symbol', 'date'] + required_cols
            df_subset = df_filtered[cols_to_keep].copy()
            
            # Group by ticker and append to ticker_data
            for ticker, group in df_subset.groupby('normalized_symbol'):
                # Remove normalized_symbol column and set date as index
                group_clean = group.drop(columns=['normalized_symbol']).copy()
                group_clean.set_index('date', inplace=True)
                ticker_data[ticker].append(group_clean)
            
            processed_files += 1
            
        except Exception as e:
            logger.warning(f"Error processing {file_path.name}: {e}")
            skipped_files += 1
            continue
    
    logger.info(f"Processed {processed_files} files, skipped {skipped_files} files")
    
    # Combine dataframes for each ticker
    logger.info("Combining data across dates for each ticker...")
    combined_data: Dict[str, pd.DataFrame] = {}
    
    for ticker in tqdm(ticker_data.keys(), desc="Combining ticker data"):
        if ticker_data[ticker]:
            # Concatenate all dataframes for this ticker
            combined_df = pd.concat(ticker_data[ticker], axis=0)
            
            # Remove duplicates (keep last occurrence)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            
            # Sort by date
            combined_df = combined_df.sort_index()
            
            # Remove any rows with NaN values
            combined_df = combined_df.dropna()
            
            if not combined_df.empty:
                combined_data[ticker] = combined_df
                logger.debug(f"Extracted {len(combined_df)} rows for {ticker} "
                           f"({combined_df.index.min().date()} to {combined_df.index.max().date()})")
    
    logger.info(f"Successfully extracted data for {len(combined_data)} tickers")
    
    # Save to ParquetStore if output_path is provided
    if output_path:
        logger.info(f"Saving extracted data to {output_path}/{library}")
        parquet_store = ParquetStore(output_path)
        
        saved_count = 0
        for ticker, df in tqdm(combined_data.items(), desc="Saving to ParquetStore"):
            try:
                # Format symbol for storage (add .US suffix to match EODHD format)
                symbol = f"{ticker}.US"
                parquet_store.write_symbol(symbol, df, library=library)
                saved_count += 1
                logger.info(f"✓ Saved {ticker}: {len(df)} rows "
                          f"({df.index.min().date()} to {df.index.max().date()})")
            except Exception as e:
                logger.error(f"✗ Failed to save {ticker}: {e}")
        
        logger.info(f"Saved {saved_count}/{len(combined_data)} tickers to ParquetStore")
    
    return combined_data


def main():
    parser = argparse.ArgumentParser(
        description='Extract OHLCV values from all NASDAQ100 stocks from data/raw_daily'
    )
    parser.add_argument(
        '--raw-daily-path',
        type=str,
        default='data/raw_daily',
        help='Path to directory containing daily parquet files (default: data/raw_daily)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='data/parquet',
        help='Path to save extracted data using ParquetStore (default: data/parquet)'
    )
    parser.add_argument(
        '--library',
        type=str,
        default='nasdaq100',
        help='Library name for ParquetStore (default: nasdaq100)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date filter (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date filter (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--tickers',
        type=str,
        default=None,
        help='Comma-separated list of specific tickers (overrides NASDAQ100 list)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save extracted data, only return in memory'
    )
    
    args = parser.parse_args()
    
    # Get ticker list
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
        logger.info(f"Using specified tickers: {len(tickers)}")
    else:
        tickers = NASDAQ100_TICKERS
        logger.info(f"Using NASDAQ100 tickers: {len(tickers)}")
    
    # Convert paths
    raw_daily_path = Path(args.raw_daily_path)
    if not raw_daily_path.exists():
        logger.error(f"Raw daily path does not exist: {raw_daily_path}")
        sys.exit(1)
    
    # Extract OHLCV data
    output_path = None if args.no_save else args.output_path
    combined_data = extract_ohlcv_from_raw_daily(
        raw_daily_path=raw_daily_path,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=output_path,
        library=args.library
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total tickers requested: {len(tickers)}")
    print(f"Tickers with data extracted: {len(combined_data)}")
    
    if combined_data:
        total_rows = sum(len(df) for df in combined_data.values())
        date_ranges = [(df.index.min(), df.index.max()) for df in combined_data.values()]
        min_date = min(d[0] for d in date_ranges)
        max_date = max(d[1] for d in date_ranges)
        
        print(f"Total rows extracted: {total_rows:,}")
        print(f"Date range: {min_date.date()} to {max_date.date()}")
        
        # Show tickers with most/least data
        ticker_counts = {ticker: len(df) for ticker, df in combined_data.items()}
        sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 5 tickers by row count:")
        for ticker, count in sorted_tickers[:5]:
            print(f"  {ticker}: {count:,} rows")
        
        print(f"\nBottom 5 tickers by row count:")
        for ticker, count in sorted_tickers[-5:]:
            print(f"  {ticker}: {count:,} rows")
    
    if not args.no_save:
        print(f"\nOutput directory: {args.output_path}/{args.library}")
    
    print("="*60)


if __name__ == "__main__":
    main()

