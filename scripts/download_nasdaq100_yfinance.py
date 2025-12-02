#!/usr/bin/env python3
"""
Download NASDAQ100 stocks from yfinance for the last 20 years and save as Parquet files.

This script downloads historical OHLCV data for all NASDAQ100 stocks and saves them
in the same format as the EODHD ingestion pipeline for consistency.
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import time

import yfinance as yf
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
# Source: https://www.nasdaq.com/market-activity/quotes/nasdaq-ndx-index
# Note: This list may need periodic updates as index composition changes
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


def get_nasdaq100_tickers() -> List[str]:
    """
    Get list of NASDAQ100 tickers.
    
    Returns:
        List of ticker symbols
    """
    # Try to fetch from yfinance if possible, otherwise use hardcoded list
    try:
        # yfinance doesn't have a direct NASDAQ100 list, so we use the hardcoded one
        # You can update this list periodically from NASDAQ's website
        logger.info(f"Using hardcoded NASDAQ100 list with {len(NASDAQ100_TICKERS)} tickers")
        return NASDAQ100_TICKERS
    except Exception as e:
        logger.warning(f"Could not fetch ticker list dynamically: {e}")
        logger.info("Using hardcoded NASDAQ100 list")
        return NASDAQ100_TICKERS


def download_ticker_data(ticker: str, start_date: datetime, end_date: datetime, 
                        retries: int = 3, delay: float = 0.5) -> Optional[pd.DataFrame]:
    """
    Download historical data for a ticker using yfinance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        retries: Number of retry attempts
        delay: Delay between retries (seconds)
        
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    for attempt in range(retries):
        try:
            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=True, prepost=False)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            # yfinance returns data with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
            # We need: Open, High, Low, Close, Volume
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Check if all required columns exist
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"{ticker}: Missing columns {missing_cols}")
                return None
            
            # Select only OHLCV columns
            df = df[required_cols].copy()
            
            # Ensure index is datetime (should already be)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Sort by date
            df = df.sort_index()
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            if df.empty:
                logger.warning(f"{ticker}: No valid data after cleaning")
                return None
            
            logger.debug(f"Downloaded {len(df)} rows for {ticker} from {df.index.min()} to {df.index.max()}")
            return df
            
        except Exception as e:
            if attempt < retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Error downloading {ticker} (attempt {attempt + 1}/{retries}): {e}")
                logger.info(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download {ticker} after {retries} attempts: {e}")
                return None
    
    return None


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description='Download NASDAQ100 stocks from yfinance and save as Parquet files'
    )
    parser.add_argument(
        '--years', 
        type=int, 
        default=20, 
        help='Number of years of historical data to download (default: 20)'
    )
    parser.add_argument(
        '--start-date', 
        type=str, 
        default=None,
        help='Start date (YYYY-MM-DD). If not provided, calculates from --years'
    )
    parser.add_argument(
        '--end-date', 
        type=str, 
        default=None,
        help='End date (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--library', 
        type=str, 
        default='nasdaq100', 
        help='Parquet library/directory name (default: nasdaq100)'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml', 
        help='Config file path (default: config.yaml)'
    )
    parser.add_argument(
        '--delay', 
        type=float, 
        default=0.5, 
        help='Delay between downloads in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--skip-existing', 
        action='store_true',
        help='Skip tickers that already have data files'
    )
    parser.add_argument(
        '--tickers', 
        type=str, 
        default=None,
        help='Comma-separated list of specific tickers (overrides NASDAQ100 list)'
    )
    
    args = parser.parse_args()
    
    # Calculate date range
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = end_date - timedelta(days=args.years * 365)
    
    logger.info(f"Downloading data from {start_date.date()} to {end_date.date()}")
    
    # Load config for parquet path
    config = load_config(args.config)
    ingestion_config = config.get('Ingestion', {})
    parquet_path = ingestion_config.get('parquet_path', 'data/parquet')
    
    # Initialize ParquetStore
    parquet_store = ParquetStore(parquet_path)
    
    # Get ticker list
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
        logger.info(f"Using specified tickers: {len(tickers)}")
    else:
        tickers = get_nasdaq100_tickers()
        logger.info(f"Downloading data for {len(tickers)} NASDAQ100 tickers")
    
    # Download data for each ticker
    successful = 0
    failed = 0
    skipped = 0
    
    for ticker in tqdm(tickers, desc="Downloading NASDAQ100 stocks"):
        # Format ticker for storage (add .US suffix to match EODHD format)
        symbol = f"{ticker}.US"
        
        # Check if we should skip existing files
        if args.skip_existing and parquet_store.symbol_exists(symbol, args.library):
            logger.debug(f"Skipping {ticker} (file already exists)")
            skipped += 1
            continue
        
        # Download data
        df = download_ticker_data(ticker, start_date, end_date)
        
        if df is not None and not df.empty:
            try:
                # Save using ParquetStore
                parquet_store.write_symbol(symbol, df, library=args.library)
                successful += 1
                logger.info(f"✓ Saved {ticker}: {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")
            except Exception as e:
                logger.error(f"✗ Failed to save {ticker}: {e}")
                failed += 1
        else:
            failed += 1
        
        # Rate limiting delay
        if args.delay > 0:
            time.sleep(args.delay)
    
    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total tickers: {len(tickers)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Output directory: {parquet_path}/{args.library}")
    print("="*60)


if __name__ == "__main__":
    main()

