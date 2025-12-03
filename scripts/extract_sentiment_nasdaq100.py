#!/usr/bin/env python3
"""
Extract sentiment data for all NASDAQ100 stocks from parquet files.

This script downloads sentiment data for all NASDAQ100 stocks using date chunking
to minimize API costs. It checks for existing sentiment data and only downloads
missing date ranges, making it resumable.

The script extracts:
- sentiment: Normalized sentiment score (from 'normalized' field)
- sentiment_count: Number of sentiment data points (from 'count' field, if available)
"""

import sys
import argparse
import logging
import yaml
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
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


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        # Resolve config path relative to project root if relative path provided
        if not Path(config_path).is_absolute():
            project_root = Path(__file__).parent.parent
            config_path = project_root / config_path
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config if config is not None else {}
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except Exception as e:
        logger.warning(f"Error loading config file {config_path}: {e}")
        return {}


def _parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    if isinstance(date_str, datetime):
        return date_str
    return parse_date(date_str)


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
        logger.warning(f"Error downloading sentiment batch: {e}")
        return {ticker: [] for ticker in tickers}
    except Exception as e:
        logger.warning(f"Unexpected error processing sentiment batch: {e}")
        return {ticker: [] for ticker in tickers}


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


def _generate_date_chunks(start_date: str, end_date: str, chunk_days: int = 90) -> List[Tuple[str, str]]:
    """
    Generate date range chunks.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        chunk_days: Days per chunk
        
    Returns:
        List of (start, end) date tuples
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    
    chunks = []
    current = start
    
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((
            current.strftime('%Y-%m-%d'),
            chunk_end.strftime('%Y-%m-%d')
        ))
        current = chunk_end + timedelta(days=1)
    
    return chunks


def _get_missing_sentiment_ranges(parquet_store: ParquetStore, symbol: str, 
                                  start_date: str, end_date: str, 
                                  sentiment_library: str) -> List[Tuple[str, str]]:
    """
    Determine missing date ranges for sentiment data for a symbol.
    
    Args:
        parquet_store: ParquetStore instance
        symbol: Symbol string
        start_date: Desired start date (YYYY-MM-DD)
        end_date: Desired end date (YYYY-MM-DD)
        sentiment_library: Library name for sentiment data
        
    Returns:
        List of (start, end) date tuples that need to be fetched
    """
    status = parquet_store.get_symbol_metadata(symbol, sentiment_library)
    
    if status is None:
        # No existing data - fetch entire range
        return [(start_date, end_date)]
    
    existing_start = status['min_date']
    existing_end = status['max_date']
    
    desired_start = _parse_date(start_date)
    desired_end = _parse_date(end_date)
    
    missing_ranges = []
    
    # Check for gap before existing data
    if desired_start < existing_start:
        gap_end = min(existing_start - timedelta(days=1), desired_end)
        missing_ranges.append((
            desired_start.strftime('%Y-%m-%d'),
            gap_end.strftime('%Y-%m-%d')
        ))
    
    # Check for gap after existing data
    if desired_end > existing_end:
        gap_start = max(existing_end + timedelta(days=1), desired_start)
        missing_ranges.append((
            gap_start.strftime('%Y-%m-%d'),
            desired_end.strftime('%Y-%m-%d')
        ))
    
    # If no gaps, check if we need to extend
    if not missing_ranges:
        if desired_start < existing_start or desired_end > existing_end:
            # Need to extend range
            new_start = min(desired_start, existing_start).strftime('%Y-%m-%d')
            new_end = max(desired_end, existing_end).strftime('%Y-%m-%d')
            missing_ranges.append((new_start, new_end))
    
    return missing_ranges if missing_ranges else []


def download_sentiment_chunk(tickers: List[str], chunk_start: str, chunk_end: str,
                            api_token: str, limit: int = 100) -> Dict[str, List]:
    """
    Download sentiment data for a date chunk and batch of tickers with pagination.
    
    Args:
        tickers: List of ticker symbols
        chunk_start: Start date of chunk (YYYY-MM-DD)
        chunk_end: End date of chunk (YYYY-MM-DD)
        api_token: EODHD API token
        limit: Number of results per page
        
    Returns:
        Dictionary mapping ticker to list of sentiment records
    """
    all_batch_data = {ticker: [] for ticker in tickers}
    offset = 0
    
    while True:
        batch_result = download_sentiment_batch(
            tickers, chunk_start, chunk_end, api_token, offset=offset, limit=limit
        )
        
        # Check if we got any new data
        got_data = False
        for ticker in tickers:
            records = batch_result.get(ticker, [])
            if records:
                all_batch_data[ticker].extend(records)
                got_data = True
        
        if not got_data:
            break
        
        # Check if we need to paginate (if any ticker got limit records)
        needs_pagination = any(
            len(batch_result.get(ticker, [])) >= limit for ticker in tickers
        )
        
        if not needs_pagination:
            break
        
        offset += limit
        time.sleep(0.1)  # Small delay between pages
    
    return all_batch_data


def extract_sentiment_for_symbols(symbols: List[str], start_date: Optional[str], 
                                  end_date: Optional[str], api_token: str,
                                  parquet_path: str, library: str, 
                                  sentiment_library: str = 'nasdaq100_sentiment',
                                  chunk_days: int = 90, batch_size: int = 10, 
                                  limit: int = 100, delay: float = 0.5,
                                  resume: bool = True) -> Dict[str, int]:
    """
    Extract sentiment data for multiple symbols using date chunking and batching.
    
    Args:
        symbols: List of symbol strings (e.g., ['AAPL.US', 'MSFT.US'])
        start_date: Start date (YYYY-MM-DD) or None to use OHLCV date ranges
        end_date: End date (YYYY-MM-DD) or None to use OHLCV date ranges
        api_token: EODHD API token
        parquet_path: Base path for parquet storage
        library: Library name for OHLCV data
        sentiment_library: Library name for sentiment data
        chunk_days: Days per chunk (default: 90)
        batch_size: Number of tickers per API call (default: 10)
        limit: Number of results per page (default: 100)
        delay: Delay between batches in seconds (default: 0.5)
        resume: If True, skip already-downloaded date ranges
        
    Returns:
        Dictionary with download statistics
    """
    parquet_store = ParquetStore(parquet_path)
    sentiment_store = ParquetStore(parquet_path)
    
    # Get date ranges from existing OHLCV data if not provided
    if start_date is None or end_date is None:
        logger.info("Determining date ranges from existing OHLCV data...")
        date_ranges = get_symbol_date_ranges(parquet_store, symbols, library)
        
        if not date_ranges:
            logger.warning("No OHLCV data found. Cannot determine date ranges.")
            return {'successful': 0, 'failed': len(symbols), 'skipped': 0, 'chunks_processed': 0, 'chunks_skipped': 0, 'api_calls': 0}
        
        # Use union of all date ranges
        all_starts = [pd.to_datetime(dr[0]) for dr in date_ranges.values()]
        all_ends = [pd.to_datetime(dr[1]) for dr in date_ranges.values()]
        
        if start_date is None:
            start_date = min(all_starts).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = max(all_ends).strftime('%Y-%m-%d')
        
        logger.info(f"Using date range: {start_date} to {end_date}")
    else:
        # Use provided dates for all symbols
        date_ranges = {symbol: (start_date, end_date) for symbol in symbols}
    
    # Group symbols into batches
    batches = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        batches.append(batch)
    
    logger.info(f"Extracting sentiment for {len(symbols)} symbols in {len(batches)} batches...")
    logger.info(f"Using date chunks of {chunk_days} days")
    
    stats = {
        'successful': 0, 
        'failed': 0, 
        'skipped': 0,
        'chunks_processed': 0,
        'chunks_skipped': 0,
        'api_calls': 0
    }
    
    # Track which symbols got data written
    symbols_with_data = set()
    
    # Process each batch
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        # Get date range for this batch (use union if different ranges)
        batch_starts = [pd.to_datetime(date_ranges.get(sym, (start_date, end_date))[0]) for sym in batch]
        batch_ends = [pd.to_datetime(date_ranges.get(sym, (start_date, end_date))[1]) for sym in batch]
        batch_start = min(batch_starts).strftime('%Y-%m-%d')
        batch_end = max(batch_ends).strftime('%Y-%m-%d')
        
        # Determine date chunks for this batch
        if resume:
            # Get missing ranges per symbol, then generate chunks
            missing_ranges_per_symbol = {}
            for symbol in batch:
                symbol_start, symbol_end = date_ranges.get(symbol, (batch_start, batch_end))
                missing_ranges = _get_missing_sentiment_ranges(
                    sentiment_store, symbol, symbol_start, symbol_end, sentiment_library
                )
                if missing_ranges:
                    missing_ranges_per_symbol[symbol] = missing_ranges
                else:
                    stats['skipped'] += 1
            
            # If all symbols in batch are already complete, skip
            if not missing_ranges_per_symbol:
                continue
            
            # Generate chunks from missing ranges (union all missing ranges)
            all_missing_starts = []
            all_missing_ends = []
            for ranges in missing_ranges_per_symbol.values():
                for r_start, r_end in ranges:
                    all_missing_starts.append(_parse_date(r_start))
                    all_missing_ends.append(_parse_date(r_end))
            
            if not all_missing_starts:
                continue
            
            overall_start = min(all_missing_starts).strftime('%Y-%m-%d')
            overall_end = max(all_missing_ends).strftime('%Y-%m-%d')
            date_chunks = _generate_date_chunks(overall_start, overall_end, chunk_days)
        else:
            # Generate chunks for entire range
            date_chunks = _generate_date_chunks(batch_start, batch_end, chunk_days)
            missing_ranges_per_symbol = {symbol: [(batch_start, batch_end)] for symbol in batch}
        
        # Process each date chunk
        for chunk_start, chunk_end in tqdm(date_chunks, desc=f"Batch {batch_idx+1}/{len(batches)} chunks", leave=False):
            # Check if this chunk overlaps with any missing ranges
            chunk_start_dt = _parse_date(chunk_start)
            chunk_end_dt = _parse_date(chunk_end)
            
            # Filter symbols that need this chunk
            symbols_needing_chunk = []
            for symbol in batch:
                if symbol in missing_ranges_per_symbol:
                    for missing_start, missing_end in missing_ranges_per_symbol[symbol]:
                        missing_start_dt = _parse_date(missing_start)
                        missing_end_dt = _parse_date(missing_end)
                        # Check if chunk overlaps with missing range
                        if not (chunk_end_dt < missing_start_dt or chunk_start_dt > missing_end_dt):
                            symbols_needing_chunk.append(symbol)
                            break
            
            if not symbols_needing_chunk:
                stats['chunks_skipped'] += 1
                continue
            
            # Download sentiment for this chunk
            try:
                chunk_data = download_sentiment_chunk(
                    symbols_needing_chunk, chunk_start, chunk_end, api_token, limit
                )
                stats['api_calls'] += 1
                
                # Process and store each ticker's data
                for ticker in symbols_needing_chunk:
                    records = chunk_data.get(ticker, [])
                    
                    if not records:
                        # No sentiment data for this ticker/date range - this is normal
                        continue
                    
                    # Convert to DataFrame
                    try:
                        df = pd.DataFrame(records)
                        
                        if 'date' not in df.columns:
                            continue
                        
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        
                        if 'normalized' not in df.columns:
                            continue
                        
                        # Extract sentiment (normalized value)
                        df['sentiment'] = df['normalized']
                        
                        # Extract sentiment count if available
                        columns_to_keep = ['sentiment']
                        if 'count' in df.columns:
                            df['sentiment_count'] = df['count']
                            columns_to_keep.append('sentiment_count')
                        
                        df = df[columns_to_keep].copy()
                        df = df.sort_index()
                        df = df[~df.index.duplicated(keep='last')]
                        
                        # Store sentiment data (ParquetStore handles appending)
                        sentiment_store.write_symbol(ticker, df, library=sentiment_library)
                        symbols_with_data.add(ticker)
                        stats['chunks_processed'] += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing sentiment for {ticker} in chunk {chunk_start} to {chunk_end}: {e}")
                        stats['failed'] += 1
                
            except Exception as e:
                logger.warning(f"Error downloading chunk {chunk_start} to {chunk_end}: {e}")
                stats['failed'] += len(symbols_needing_chunk)
        
        # Rate limiting delay between batches
        if batch_idx < len(batches) - 1:
            time.sleep(delay)
    
    # Count successful symbols (those that had missing ranges and got data)
    stats['successful'] = len(symbols_with_data)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Extract sentiment data for all NASDAQ100 stocks from parquet files'
    )
    parser.add_argument(
        '--parquet-path',
        type=str,
        default="data/parquet",
        help='Base path to parquet directory (default: from config.yaml)'
    )
    parser.add_argument(
        '--library',
        type=str,
        default='nasdaq100',
        help='Source library name for OHLCV data (default: nasdaq100)'
    )
    parser.add_argument(
        '--sentiment-library',
        type=str,
        default='nasdaq100_sentiment',
        help='Target library name for sentiment data (default: nasdaq100_sentiment)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date filter (YYYY-MM-DD, optional, defaults to OHLCV min date)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date filter (YYYY-MM-DD, optional, defaults to OHLCV max date)'
    )
    parser.add_argument(
        '--chunk-days',
        type=int,
        default=90,
        help='Days per chunk (default: 90)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of tickers per API call (default: 10, max ~20)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Number of results per page (default: 100)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between batches in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume mode (re-download all data)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Config file path (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load config for parquet path and API token
    config = load_config(args.config)
    
    # Ensure config loaded successfully
    if not config:
        logger.error(f"Failed to load config from {args.config}")
        sys.exit(1)
    
    ingestion_config = config.get('Ingestion', {})
    
    # Resolve parquet path
    if args.parquet_path == "data/parquet":  # Check default value instead of None
        args.parquet_path = ingestion_config.get('parquet_path', 'data/parquet')
        # Resolve relative path
        if not Path(args.parquet_path).is_absolute():
            project_root = Path(__file__).parent.parent
            args.parquet_path = str(project_root / args.parquet_path)
    elif not Path(args.parquet_path).is_absolute():
        # Resolve relative path even if custom path provided
        project_root = Path(__file__).parent.parent
        args.parquet_path = str(project_root / args.parquet_path)
    
    # Get API token - use direct access like ingest_stocks.py for better error messages
    try:
        api_token = config['EODHD']['api_token']
    except KeyError:
        logger.error("EODHD API token not found in config. Please add it to config.yaml under EODHD.api_token")
        logger.error(f"Config keys found: {list(config.keys())}")
        sys.exit(1)
    
    # Get symbols from parquet store
    parquet_store = ParquetStore(args.parquet_path)
    symbols = parquet_store.list_symbols(args.library)
    
    if not symbols:
        logger.error(f"No symbols found in library '{args.library}'. Make sure you have downloaded NASDAQ100 data first.")
        sys.exit(1)
    
    logger.info(f"Found {len(symbols)} symbols in library '{args.library}'")
    
    # Extract sentiment data
    print("\n" + "="*60)
    print("EXTRACTING SENTIMENT DATA")
    print("="*60)
    
    stats = extract_sentiment_for_symbols(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        api_token=api_token,
        parquet_path=args.parquet_path,
        library=args.library,
        sentiment_library=args.sentiment_library,
        chunk_days=args.chunk_days,
        batch_size=args.batch_size,
        limit=args.limit,
        delay=args.delay,
        resume=not args.no_resume
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total symbols: {len(symbols)}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped (already complete): {stats['skipped']}")
    print(f"Chunks processed: {stats['chunks_processed']}")
    print(f"Chunks skipped: {stats['chunks_skipped']}")
    print(f"Total API calls: {stats['api_calls']}")
    print(f"Sentiment library: {args.sentiment_library}")
    print("="*60)


if __name__ == "__main__":
    main()

