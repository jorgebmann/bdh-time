#!/usr/bin/env python3
"""
Build processed market dataset from parquet files.

This script reads NASDAQ100 stock data from parquet files and processes it into
a format suitable for training the MarketBDH model.
"""

import sys
import argparse
import yaml
import torch
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset.preprocess import process_market_data_from_parquet, process_market_data
from dataset.ingestion import ParquetStore

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}


def download_sentiment_batch(tickers: List[str], start_date: str, end_date: str, 
                             api_token: str, offset: int = 0, limit: int = 100) -> Dict[str, List]:
    """
    Download sentiment scores for multiple tickers in a single API call.
    
    Args:
        tickers: List of ticker symbols with .US suffix (e.g., ['AAPL.US', 'MSFT.US'])
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_token: EODHD API token
        offset: Pagination offset
        limit: Number of results per page
        
    Returns:
        Dictionary mapping ticker to list of sentiment records with date, count, normalized fields
    """
    # Join tickers with commas
    ticker_str = ','.join(tickers)
    
    url = "https://eodhd.com/api/sentiments"
    params = {
        's': ticker_str,
        'filter[date_from]': start_date,
        'filter[date_to]': end_date,
        'offset': offset,
        'limit': limit,
        'api_token': api_token,
        'fmt': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Initialize result dictionary
        result = {ticker: [] for ticker in tickers}
        
        # Parse response - API returns a dictionary with ticker keys
        if isinstance(data, dict):
            for ticker in tickers:
                if ticker in data:
                    ticker_data = data[ticker]
                    if isinstance(ticker_data, list):
                        result[ticker].extend(ticker_data)
                    elif isinstance(ticker_data, dict):
                        # Single record
                        result[ticker].append(ticker_data)
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading sentiment batch: {e}")
        return {ticker: [] for ticker in tickers}
    except Exception as e:
        print(f"Unexpected error processing sentiment batch: {e}")
        return {ticker: [] for ticker in tickers}


def download_sentiment_for_ticker(ticker: str, start_date: str, end_date: str, 
                                   api_token: str, limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Download sentiment data for a single ticker with pagination support.
    
    Args:
        ticker: Ticker symbol with .US suffix (e.g., 'AAPL.US')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_token: EODHD API token
        limit: Number of results per page
        
    Returns:
        DataFrame with date index and 'sentiment' column (normalized values) or None if failed
    """
    all_records = []
    offset = 0
    
    while True:
        batch_result = download_sentiment_batch([ticker], start_date, end_date, 
                                                api_token, offset=offset, limit=limit)
        
        records = batch_result.get(ticker, [])
        if not records:
            break
        
        all_records.extend(records)
        
        # If we got fewer than limit records, we've reached the end
        if len(records) < limit:
            break
        
        offset += limit
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    if not all_records:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    # Ensure date column exists
    if 'date' not in df.columns:
        return None
    
    # Convert date to datetime and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Use normalized sentiment value
    if 'normalized' in df.columns:
        df['sentiment'] = df['normalized']
    else:
        return None
    
    # Keep only sentiment column
    df = df[['sentiment']].copy()
    
    # Sort by date
    df = df.sort_index()
    
    # Remove duplicates (keep last)
    df = df[~df.index.duplicated(keep='last')]
    
    return df


def get_symbol_date_ranges(parquet_store: ParquetStore, symbols: List[str], 
                           library: str) -> Dict[str, Tuple[str, str]]:
    """
    Get date ranges from existing OHLCV parquet files.
    
    Args:
        parquet_store: ParquetStore instance
        symbols: List of symbol strings
        library: Library name
        
    Returns:
        Dictionary mapping symbol to (start_date, end_date) tuple as strings
    """
    date_ranges = {}
    
    for symbol in symbols:
        date_range = parquet_store.get_date_range(symbol, library)
        if date_range:
            min_date, max_date = date_range
            date_ranges[symbol] = (
                min_date.strftime('%Y-%m-%d'),
                max_date.strftime('%Y-%m-%d')
            )
    
    return date_ranges


def download_sentiment_for_symbols(symbols: List[str], start_date: Optional[str], 
                                   end_date: Optional[str], api_token: str,
                                   parquet_path: str, library: str, 
                                   sentiment_library: Optional[str] = None,
                                   batch_size: int = 10, limit: int = 100,
                                   delay: float = 0.5) -> Dict[str, int]:
    """
    Download sentiment data for multiple symbols using batching.
    
    Args:
        symbols: List of symbol strings (e.g., ['AAPL.US', 'MSFT.US'])
        start_date: Start date (YYYY-MM-DD) or None to use OHLCV date ranges
        end_date: End date (YYYY-MM-DD) or None to use OHLCV date ranges
        api_token: EODHD API token
        parquet_path: Base path for parquet storage
        library: Library name for OHLCV data
        sentiment_library: Library name for sentiment data (default: same as library)
        batch_size: Number of tickers per API call (default: 10)
        limit: Number of results per page (default: 100)
        delay: Delay between batches in seconds (default: 0.5)
        
    Returns:
        Dictionary with download statistics
    """
    if sentiment_library is None:
        sentiment_library = library
    
    parquet_store = ParquetStore(parquet_path)
    sentiment_store = ParquetStore(parquet_path)
    
    # Get date ranges from existing OHLCV data if not provided
    if start_date is None or end_date is None:
        print("Determining date ranges from existing OHLCV data...")
        date_ranges = get_symbol_date_ranges(parquet_store, symbols, library)
        
        if not date_ranges:
            print("Warning: No OHLCV data found. Cannot determine date ranges.")
            return {'successful': 0, 'failed': len(symbols), 'skipped': 0}
        
        # Use union of all date ranges
        all_starts = [pd.to_datetime(dr[0]) for dr in date_ranges.values()]
        all_ends = [pd.to_datetime(dr[1]) for dr in date_ranges.values()]
        
        if start_date is None:
            start_date = min(all_starts).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = max(all_ends).strftime('%Y-%m-%d')
        
        print(f"Using date range: {start_date} to {end_date}")
    else:
        # Use provided dates for all symbols
        date_ranges = {symbol: (start_date, end_date) for symbol in symbols}
    
    # Group symbols into batches
    batches = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        batches.append(batch)
    
    print(f"Downloading sentiment for {len(symbols)} symbols in {len(batches)} batches...")
    
    stats = {'successful': 0, 'failed': 0, 'skipped': 0}
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Downloading sentiment batches")):
        # Get date range for this batch (use union if different ranges)
        batch_starts = [pd.to_datetime(date_ranges.get(sym, (start_date, end_date))[0]) for sym in batch]
        batch_ends = [pd.to_datetime(date_ranges.get(sym, (start_date, end_date))[1]) for sym in batch]
        batch_start = min(batch_starts).strftime('%Y-%m-%d')
        batch_end = max(batch_ends).strftime('%Y-%m-%d')
        
        # Download batch with pagination
        all_batch_data = {ticker: [] for ticker in batch}
        offset = 0
        
        while True:
            batch_result = download_sentiment_batch(
                batch, batch_start, batch_end, api_token, offset=offset, limit=limit
            )
            
            # Check if we got any new data
            got_data = False
            for ticker in batch:
                records = batch_result.get(ticker, [])
                if records:
                    all_batch_data[ticker].extend(records)
                    got_data = True
            
            if not got_data:
                break
            
            # Check if we need to paginate (if any ticker got limit records)
            needs_pagination = any(
                len(batch_result.get(ticker, [])) >= limit for ticker in batch
            )
            
            if not needs_pagination:
                break
            
            offset += limit
            time.sleep(0.1)  # Small delay between pages
        
        # Process and store each ticker's data
        for ticker in batch:
            records = all_batch_data[ticker]
            
            if not records:
                stats['failed'] += 1
                continue
            
            # Convert to DataFrame
            try:
                df = pd.DataFrame(records)
                
                if 'date' not in df.columns:
                    stats['failed'] += 1
                    continue
                
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                if 'normalized' in df.columns:
                    df['sentiment'] = df['normalized']
                else:
                    stats['failed'] += 1
                    continue
                
                df = df[['sentiment']].copy()
                df = df.sort_index()
                df = df[~df.index.duplicated(keep='last')]
                
                # Store sentiment data
                sentiment_store.write_symbol(ticker, df, library=sentiment_library)
                stats['successful'] += 1
                
            except Exception as e:
                print(f"Error processing sentiment for {ticker}: {e}")
                stats['failed'] += 1
        
        # Rate limiting delay between batches
        if batch_idx < len(batches) - 1:
            time.sleep(delay)
    
    return stats


def build_dataset_from_parquet(parquet_path: str = None, library: str = 'nasdaq100',
                               start_date: str = None, end_date: str = None,
                               min_years: float = 3.0,
                               config_path: str = 'config.yaml',
                               download_sentiment: bool = False,
                               sentiment_library: Optional[str] = None,
                               sentiment_batch_size: int = 10,
                               sentiment_limit: int = 100):
    """
    Build dataset from parquet files.
    
    Args:
        parquet_path: Base path to parquet directory (default: from config)
        library: Library/directory name (default: 'nasdaq100')
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        min_years: Minimum years of data required per stock (default: 3.0)
        config_path: Path to config file
        download_sentiment: Whether to download sentiment data before building dataset
        sentiment_library: Library name for sentiment storage (default: same as library)
        sentiment_batch_size: Number of tickers per API call (default: 10)
        sentiment_limit: Number of results per page (default: 100)
    """
    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "market_dataset.pt"
    
    # Load config for parquet path and API token
    config = load_config(config_path)
    ingestion_config = config.get('Ingestion', {})
    eodhd_config = config.get('EODHD', {})
    
    if parquet_path is None:
        parquet_path = ingestion_config.get('parquet_path', 'data/parquet')
    
    # Download sentiment if requested
    if download_sentiment:
        api_token = eodhd_config.get('api_token')
        if not api_token:
            print("Warning: EODHD API token not found in config. Skipping sentiment download.")
        else:
            print("\n" + "="*60)
            print("DOWNLOADING SENTIMENT DATA")
            print("="*60)
            
            parquet_store = ParquetStore(parquet_path)
            symbols = parquet_store.list_symbols(library)
            
            if not symbols:
                print(f"Warning: No symbols found in library '{library}'. Skipping sentiment download.")
            else:
                stats = download_sentiment_for_symbols(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    api_token=api_token,
                    parquet_path=parquet_path,
                    library=library,
                    sentiment_library=sentiment_library,
                    batch_size=sentiment_batch_size,
                    limit=sentiment_limit
                )
                
                print("\n" + "="*60)
                print("SENTIMENT DOWNLOAD SUMMARY")
                print("="*60)
                print(f"Successful: {stats['successful']}")
                print(f"Failed: {stats['failed']}")
                print(f"Skipped: {stats['skipped']}")
                print("="*60 + "\n")
    
    print(f"Building dataset from parquet files...")
    print(f"Parquet path: {parquet_path}")
    print(f"Library: {library}")
    print(f"Minimum years per stock: {min_years}")
    if start_date:
        print(f"Start date: {start_date}")
    if end_date:
        print(f"End date: {end_date}")
    
    # Process data from parquet
    try:
        processed_data = process_market_data_from_parquet(
            parquet_path=parquet_path,
            library=library,
            start_date=start_date,
            end_date=end_date,
            min_years=min_years,
            sentiment_library=sentiment_library if sentiment_library else library
        )
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nMake sure you have downloaded NASDAQ100 data first:")
        print(f"  python scripts/download_nasdaq100_yfinance.py")
        sys.exit(1)
    
    # Save to .pt file
    print(f"\nSaving processed dataset to {output_path}...")
    torch.save(processed_data, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET BUILD SUMMARY")
    print("="*60)
    print(f"Assets processed: {len(processed_data['asset_names'])}")
    print(f"Time steps: {processed_data['X'].shape[0]}")
    print(f"Features per asset: {processed_data['X'].shape[2]}")
    print(f"Output shape X: {processed_data['X'].shape}")
    print(f"Output shape Y: {processed_data['Y'].shape}")
    if 'mask' in processed_data:
        valid_ratio = processed_data['mask'].sum() / processed_data['mask'].size * 100
        print(f"Data coverage: {valid_ratio:.1f}% valid")
    if 'asset_date_ranges' in processed_data:
        print(f"\nDate ranges per asset:")
        for name, (start, end) in zip(processed_data['asset_names'], processed_data['asset_date_ranges']):
            print(f"  {name}: {start} to {end}")
    print(f"Output file: {output_path}")
    print("="*60)
    print("Dataset build complete.")


def build_dataset_from_pickle(pickle_path: str = None):
    """
    Build dataset from pickle file (legacy method for backward compatibility).
    
    Args:
        pickle_path: Path to pickle file
    """
    project_root = Path(__file__).parent.parent
    
    if pickle_path is None:
        pickle_path = project_root / "data" / "nasdaq100_data.pkl"
    
    output_path = project_root / "data" / "market_dataset.pt"
    
    print(f"Building dataset from pickle file: {pickle_path}...")
    
    if not Path(pickle_path).exists():
        print(f"Error: Pickle file not found at {pickle_path}")
        sys.exit(1)
    
    # Process data
    processed_data = process_market_data(str(pickle_path))
    
    # Save to .pt file
    print(f"Saving processed dataset to {output_path}...")
    torch.save(processed_data, output_path)
    print("Dataset build complete.")


def main():
    parser = argparse.ArgumentParser(
        description='Build processed market dataset from parquet files or pickle file'
    )
    parser.add_argument(
        '--parquet-path',
        type=str,
        default=None,
        help='Base path to parquet directory (default: from config.yaml)'
    )
    parser.add_argument(
        '--library',
        type=str,
        default='nasdaq100',
        help='Parquet library/directory name (default: nasdaq100)'
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
        '--min-years',
        type=float,
        default=3.0,
        help='Minimum years of data required per stock (default: 3.0)'
    )
    parser.add_argument(
        '--pickle',
        type=str,
        default=None,
        help='Use pickle file instead of parquet (legacy mode). Path to .pkl file.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Config file path (default: config.yaml)'
    )
    parser.add_argument(
        '--download-sentiment',
        action='store_true',
        help='Download sentiment data before building dataset'
    )
    parser.add_argument(
        '--sentiment-library',
        type=str,
        default=None,
        help='Library name for sentiment storage (default: same as OHLCV library)'
    )
    parser.add_argument(
        '--sentiment-batch-size',
        type=int,
        default=10,
        help='Number of tickers per API call (default: 10, max ~20)'
    )
    parser.add_argument(
        '--sentiment-limit',
        type=int,
        default=100,
        help='Number of results per page (default: 100)'
    )
    
    args = parser.parse_args()
    
    if args.pickle:
        # Legacy mode: use pickle file
        build_dataset_from_pickle(args.pickle)
    else:
        # Default: use parquet files
        build_dataset_from_parquet(
            parquet_path=args.parquet_path,
            library=args.library,
            start_date=args.start_date,
            end_date=args.end_date,
            min_years=args.min_years,
            config_path=args.config,
            download_sentiment=args.download_sentiment,
            sentiment_library=args.sentiment_library,
            sentiment_batch_size=args.sentiment_batch_size,
            sentiment_limit=args.sentiment_limit
        )


if __name__ == "__main__":
    main()
