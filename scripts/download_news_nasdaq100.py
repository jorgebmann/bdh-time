#!/usr/bin/env python3
"""
Download news articles for all NASDAQ100 stocks from EODHD API.

This script downloads news data for all NASDAQ100 stocks using date chunking
to minimize API costs. It checks for existing news data and only downloads
missing date ranges, making it resumable.

Features:
- Batch API support testing and fallback to single-ticker requests
- Date chunking to minimize memory usage and enable resume
- Per-ticker JSON storage with deduplication
- Resume capability to avoid re-downloading existing data
"""

import sys
import argparse
import logging
import yaml
import requests
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from collections import defaultdict
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NASDAQ100 tickers (from download_nasdaq100_yfinance.py)
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
NASDAQ100_TICKERS = sorted(list(set(NASDAQ100_TICKERS)))


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
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


def test_batch_api_support(api_token: str, test_tickers: List[str] = None) -> bool:
    """
    Test if EODHD news API supports batch requests (comma-separated tickers).
    
    Args:
        api_token: EODHD API token
        test_tickers: List of tickers to test (default: ['AAPL.US', 'MSFT.US'])
        
    Returns:
        True if batch API is supported, False otherwise
    """
    if test_tickers is None:
        test_tickers = ['AAPL.US', 'MSFT.US']
    
    # Test with a recent date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    ticker_str = ','.join(test_tickers)
    url = "https://eodhd.com/api/news"
    params = {
        's': ticker_str,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'limit': 10,
        'api_token': api_token,
        'fmt': 'json'
    }
    
    try:
        logger.info(f"Testing batch API support with tickers: {ticker_str}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Check if response is a dict with ticker keys (batch format)
        if isinstance(data, dict):
            # Check if any of the test tickers are keys in the response
            if any(ticker in data for ticker in test_tickers):
                logger.info("✓ Batch API is supported (dict response with ticker keys)")
                return True
        
        # Check if response is a list (might still work but not batched)
        if isinstance(data, list):
            logger.info("⚠ Batch API test returned list (may not support batching)")
            # Could still work if articles contain multiple tickers
            # But we'll treat as not supported for safety
            return False
        
        logger.warning("⚠ Unexpected batch API response format")
        return False
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error testing batch API: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error testing batch API: {e}")
        return False


def download_news_batch(tickers: List[str], start_date: str, end_date: str,
                       api_token: str, offset: int = 0, limit: int = 100) -> Dict[str, List]:
    """
    Download news articles for multiple tickers in a single API call.
    
    Args:
        tickers: List of ticker symbols with .US suffix (e.g., ['AAPL.US', 'MSFT.US'])
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_token: EODHD API token
        offset: Pagination offset
        limit: Number of results per page
        
    Returns:
        Dictionary mapping ticker to list of news articles
    """
    ticker_str = ','.join(tickers)
    
    url = "https://eodhd.com/api/news"
    params = {
        's': ticker_str,
        'from': start_date,
        'to': end_date,
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
        
        # Parse response - API may return dict with ticker keys (batch format)
        if isinstance(data, dict):
            for ticker in tickers:
                if ticker in data:
                    ticker_data = data[ticker]
                    if isinstance(ticker_data, list):
                        result[ticker].extend(ticker_data)
                    elif isinstance(ticker_data, dict):
                        result[ticker].append(ticker_data)
        
        # If response is a list, distribute articles by symbols field
        elif isinstance(data, list):
            for article in data:
                article_symbols = article.get('symbols', [])
                # Add article to all matching tickers
                for ticker in tickers:
                    if ticker in article_symbols:
                        result[ticker].append(article)
        
        return result
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error downloading news batch: {e}")
        return {ticker: [] for ticker in tickers}
    except Exception as e:
        logger.warning(f"Unexpected error processing news batch: {e}")
        return {ticker: [] for ticker in tickers}


def download_news_single(ticker: str, start_date: str, end_date: str,
                        api_token: str, offset: int = 0, limit: int = 100) -> List[Dict]:
    """
    Download news articles for a single ticker from EODHD API.
    
    Args:
        ticker: Ticker symbol with .US suffix (e.g., 'AAPL.US')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_token: EODHD API token
        offset: Pagination offset
        limit: Number of results per page (max 100)
        
    Returns:
        List of news article dictionaries
    """
    url = "https://eodhd.com/api/news"
    params = {
        's': ticker,
        'from': start_date,
        'to': end_date,
        'offset': offset,
        'limit': limit,
        'api_token': api_token,
        'fmt': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # API returns a list of news articles
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'data' in data:
                return data['data']
            elif ticker in data:
                ticker_data = data[ticker]
                if isinstance(ticker_data, list):
                    return ticker_data
                elif isinstance(ticker_data, dict):
                    return [ticker_data]
            else:
                return [data]
        else:
            logger.warning(f"Unexpected response format: {type(data)}")
            return []
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error downloading news for {ticker}: {e}")
        return []
    except Exception as e:
        logger.warning(f"Unexpected error processing news for {ticker}: {e}")
        return []


def download_all_news_single(ticker: str, start_date: str, end_date: str,
                            api_token: str, limit: int = 100, delay: float = 0.5) -> List[Dict]:
    """
    Download all news articles for a ticker, handling pagination.
    
    Args:
        ticker: Ticker symbol with .US suffix
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_token: EODHD API token
        limit: Number of results per page
        delay: Delay between requests in seconds
        
    Returns:
        List of all news article dictionaries
    """
    all_news = []
    offset = 0
    
    while True:
        news_batch = download_news_single(ticker, start_date, end_date, api_token, offset, limit)
        
        if not news_batch:
            break
        
        all_news.extend(news_batch)
        
        # If we got fewer results than the limit, we've reached the end
        if len(news_batch) < limit:
            break
        
        offset += limit
        time.sleep(delay)
    
    return all_news


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


def get_missing_date_ranges(ticker: str, start_date: str, end_date: str,
                           output_dir: Path) -> List[Tuple[str, str]]:
    """
    Determine missing date ranges for news data for a ticker.
    
    Args:
        ticker: Ticker symbol (e.g., 'AAPL.US')
        start_date: Desired start date (YYYY-MM-DD)
        end_date: Desired end date (YYYY-MM-DD)
        output_dir: Output directory for news files
        
    Returns:
        List of (start, end) date tuples that need to be fetched
    """
    news_file = output_dir / f"{ticker}.json"
    
    if not news_file.exists():
        # No existing data - fetch entire range
        return [(start_date, end_date)]
    
    try:
        with open(news_file, 'r', encoding='utf-8') as f:
            existing_articles = json.load(f)
        
        if not existing_articles:
            return [(start_date, end_date)]
        
        # Extract dates from existing articles
        dates = []
        for article in existing_articles:
            date_str = article.get('date', '')
            if date_str:
                try:
                    dt = _parse_date(date_str)
                    dates.append(dt.date())
                except:
                    continue
        
        if not dates:
            return [(start_date, end_date)]
        
        existing_start = min(dates)
        existing_end = max(dates)
        
        desired_start = _parse_date(start_date).date()
        desired_end = _parse_date(end_date).date()
        
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
        
        return missing_ranges if missing_ranges else []
        
    except Exception as e:
        logger.warning(f"Error reading existing news file for {ticker}: {e}")
        return [(start_date, end_date)]


def deduplicate_articles(articles: List[Dict]) -> List[Dict]:
    """
    Remove duplicate articles based on URL or link field.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Deduplicated list of articles
    """
    seen_urls = set()
    deduplicated = []
    
    for article in articles:
        # Use 'link' or 'url' field as unique identifier
        url = article.get('link') or article.get('url') or article.get('id')
        
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduplicated.append(article)
        elif not url:
            # If no URL, keep article but log warning
            logger.debug(f"Article without URL/link field: {article.get('title', 'N/A')[:50]}")
            deduplicated.append(article)
    
    return deduplicated


def download_news_chunk(tickers: List[str], chunk_start: str, chunk_end: str,
                       api_token: str, use_batch: bool, limit: int = 100,
                       delay: float = 0.5) -> Dict[str, List]:
    """
    Download news data for a date chunk with batching and pagination.
    
    Args:
        tickers: List of ticker symbols
        chunk_start: Start date of chunk (YYYY-MM-DD)
        chunk_end: End date of chunk (YYYY-MM-DD)
        api_token: EODHD API token
        use_batch: Whether to use batch API
        limit: Number of results per page
        delay: Delay between requests
        
    Returns:
        Dictionary mapping ticker to list of news articles
    """
    all_data = {ticker: [] for ticker in tickers}
    
    if use_batch:
        # Use batch API
        offset = 0
        while True:
            batch_result = download_news_batch(
                tickers, chunk_start, chunk_end, api_token, offset=offset, limit=limit
            )
            
            got_data = False
            for ticker in tickers:
                records = batch_result.get(ticker, [])
                if records:
                    all_data[ticker].extend(records)
                    got_data = True
            
            if not got_data:
                break
            
            # Check if we need to paginate
            needs_pagination = any(
                len(batch_result.get(ticker, [])) >= limit for ticker in tickers
            )
            
            if not needs_pagination:
                break
            
            offset += limit
            time.sleep(delay)
    else:
        # Use single-ticker API
        for ticker in tickers:
            articles = download_all_news_single(
                ticker, chunk_start, chunk_end, api_token, limit=limit, delay=delay
            )
            all_data[ticker] = articles
    
    return all_data


def download_news_for_ticker(ticker: str, start_date: str, end_date: str,
                             api_token: str, output_dir: Path, use_batch: bool,
                             chunk_days: int = 90, batch_size: int = 10,
                             limit: int = 100, delay: float = 0.5,
                             resume: bool = True) -> int:
    """
    Download news for a single ticker with date chunking and resume support.
    
    Args:
        ticker: Ticker symbol (e.g., 'AAPL.US')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_token: EODHD API token
        output_dir: Output directory
        use_batch: Whether to use batch API
        chunk_days: Days per chunk
        batch_size: Tickers per batch (if using batch API)
        limit: Results per page
        delay: Delay between requests
        resume: Enable resume mode
        
    Returns:
        Number of articles downloaded
    """
    news_file = output_dir / f"{ticker}.json"
    
    # Load existing articles if resuming
    existing_articles = []
    if resume and news_file.exists():
        try:
            with open(news_file, 'r', encoding='utf-8') as f:
                existing_articles = json.load(f)
            logger.info(f"  Resuming: Found {len(existing_articles)} existing articles for {ticker}")
        except Exception as e:
            logger.warning(f"  Error reading existing file for {ticker}: {e}")
    
    # Determine missing date ranges
    if resume:
        missing_ranges = get_missing_date_ranges(ticker, start_date, end_date, output_dir)
    else:
        missing_ranges = [(start_date, end_date)]
    
    if not missing_ranges:
        logger.info(f"  {ticker}: Already up to date")
        return len(existing_articles)
    
    # Download missing ranges
    all_new_articles = []
    
    for range_start, range_end in missing_ranges:
        # Generate chunks for this range
        chunks = _generate_date_chunks(range_start, range_end, chunk_days)
        
        for chunk_start, chunk_end in chunks:
            logger.info(f"  Downloading {ticker}: {chunk_start} to {chunk_end}")
            
            if use_batch:
                # For single ticker, batch API still works
                articles = download_news_chunk(
                    [ticker], chunk_start, chunk_end, api_token,
                    use_batch=True, limit=limit, delay=delay
                )
                all_new_articles.extend(articles.get(ticker, []))
            else:
                articles = download_all_news_single(
                    ticker, chunk_start, chunk_end, api_token, limit=limit, delay=delay
                )
                all_new_articles.extend(articles)
    
    # Merge with existing articles and deduplicate
    all_articles = existing_articles + all_new_articles
    all_articles = deduplicate_articles(all_articles)
    
    # Save updated file
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(news_file, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False, default=str)
    
    new_count = len(all_articles) - len(existing_articles)
    logger.info(f"  {ticker}: Saved {len(all_articles)} articles ({new_count} new)")
    
    return len(all_articles)


def main():
    parser = argparse.ArgumentParser(
        description='Download news for all NASDAQ100 stocks from EODHD API'
    )
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD). Default: one year ago')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD). Default: today')
    parser.add_argument('--chunk-days', type=int, default=90,
                       help='Days per chunk (default: 90)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Tickers per batch if batch API supported (default: 10)')
    parser.add_argument('--limit', type=int, default=100,
                       help='Results per page (default: 100)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between requests in seconds (default: 0.5)')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Enable resume mode (default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='Disable resume mode')
    parser.add_argument('--output-dir', type=str, default='data/news',
                       help='Output directory (default: data/news)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file path (default: config.yaml)')
    parser.add_argument('--api-token', type=str,
                       help='EODHD API token (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get API token
    api_token = args.api_token
    if not api_token:
        try:
            api_token = config['EODHD']['api_token']
        except (KeyError, TypeError):
            logger.error("API token not found. Provide --api-token or set in config.yaml under EODHD.api_token")
            return 1
    
    # Set date range
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = end_date - timedelta(days=365)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Prepare tickers
    tickers = [f"{ticker}.US" for ticker in NASDAQ100_TICKERS]
    
    # Test batch API support
    logger.info("Testing batch API support...")
    use_batch = test_batch_api_support(api_token)
    
    output_dir = Path(args.output_dir)
    
    logger.info("="*60)
    logger.info("NASDAQ100 NEWS DOWNLOAD")
    logger.info("="*60)
    logger.info(f"Date range: {start_date_str} to {end_date_str}")
    logger.info(f"Tickers: {len(tickers)}")
    logger.info(f"Batch API: {'Enabled' if use_batch else 'Disabled (using single-ticker requests)'}")
    logger.info(f"Chunk days: {args.chunk_days}")
    logger.info(f"Resume mode: {args.resume}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    
    # Download news for each ticker
    stats = {
        'total_articles': 0,
        'failed_tickers': [],
        'api_calls': 0
    }
    
    for ticker in tqdm(tickers, desc="Downloading news"):
        try:
            count = download_news_for_ticker(
                ticker=ticker,
                start_date=start_date_str,
                end_date=end_date_str,
                api_token=api_token,
                output_dir=output_dir,
                use_batch=use_batch,
                chunk_days=args.chunk_days,
                batch_size=args.batch_size,
                limit=args.limit,
                delay=args.delay,
                resume=args.resume
            )
            stats['total_articles'] += count
        except Exception as e:
            logger.error(f"Failed to download news for {ticker}: {e}")
            stats['failed_tickers'].append(ticker)
    
    # Summary
    logger.info("="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    logger.info(f"Total articles downloaded: {stats['total_articles']}")
    logger.info(f"Successful tickers: {len(tickers) - len(stats['failed_tickers'])}")
    logger.info(f"Failed tickers: {len(stats['failed_tickers'])}")
    if stats['failed_tickers']:
        logger.info(f"Failed tickers: {', '.join(stats['failed_tickers'])}")
    logger.info("="*60)
    
    return 0 if not stats['failed_tickers'] else 1


if __name__ == '__main__':
    sys.exit(main())

