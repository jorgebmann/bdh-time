#!/usr/bin/env python3
"""
Batch filter news articles for all NASDAQ100 tickers using strict filtering.

Iterates over all {ticker}.US.json files in data/news/ and applies strict filtering
to create {ticker}_filtered_strict.json files.
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_ticker_files(news_dir: Path) -> List[Path]:
    """
    Find all {ticker}.US.json files in the news directory.
    
    Args:
        news_dir: Path to news directory
        
    Returns:
        List of Path objects for ticker files
    """
    pattern = "*.US.json"
    ticker_files = list(news_dir.glob(pattern))
    
    # Exclude already filtered files
    ticker_files = [f for f in ticker_files if not f.name.endswith('_filtered_strict.json')]
    
    return sorted(ticker_files)


def extract_ticker_from_filename(filename: str) -> str:
    """
    Extract ticker symbol from filename.
    
    Args:
        filename: Filename like "AAPL.US.json"
        
    Returns:
        Ticker symbol like "AAPL.US"
    """
    # Remove .json extension
    ticker = filename.replace('.json', '')
    return ticker


def filter_ticker_news(
    input_file: Path,
    ticker: str,
    news_dir: Path,
    max_symbols: int = 3,
    require_primary: bool = True,
    require_keywords: bool = True,
    keyword_threshold: int = 2
) -> bool:
    """
    Filter news for a single ticker using filter_news.py script.
    
    Args:
        input_file: Path to input JSON file
        ticker: Ticker symbol (e.g., "AAPL.US")
        news_dir: Directory for output files
        max_symbols: Maximum number of symbols per article
        require_primary: Require ticker to be first symbol
        require_keywords: Require keywords in article
        keyword_threshold: Minimum keyword matches
        
    Returns:
        True if successful, False otherwise
    """
    # Construct output filename
    ticker_base = ticker.replace('.US', '')
    output_file = news_dir / f"{ticker_base}_filtered_strict.json"
    
    # Build command
    script_path = Path(__file__).parent / "filter_news.py"
    cmd = [
        sys.executable,
        str(script_path),
        '--input', str(input_file),
        '--output', str(output_file),
        '--ticker', ticker,
        '--max-symbols', str(max_symbols),
        '--keyword-threshold', str(keyword_threshold)
    ]
    
    if require_primary:
        cmd.append('--require-primary')
    
    if require_keywords:
        cmd.append('--require-keywords')
    else:
        cmd.append('--no-keywords')
    
    try:
        logger.info(f"Filtering {ticker}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Log output if any
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        if result.stderr:
            logger.debug(f"Stderr: {result.stderr}")
        
        logger.info(f"✓ Filtered {ticker}: {output_file.name}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Error filtering {ticker}: {e}")
        if e.stdout:
            logger.error(f"  stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"  stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error filtering {ticker}: {e}")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch filter news articles for all NASDAQ100 tickers'
    )
    parser.add_argument('--news-dir', type=str, default='data/news',
                       help='Directory containing news JSON files (default: data/news)')
    parser.add_argument('--max-symbols', type=int, default=3,
                       help='Maximum number of symbols per article (default: 3)')
    parser.add_argument('--require-primary', action='store_true', default=True,
                       help='Require ticker to be first symbol (default: True)')
    parser.add_argument('--no-require-primary', dest='require_primary', action='store_false',
                       help='Do not require ticker to be first symbol')
    parser.add_argument('--require-keywords', action='store_true', default=True,
                       help='Require keywords in article (default: True)')
    parser.add_argument('--no-keywords', dest='require_keywords', action='store_false',
                       help='Do not require keywords')
    parser.add_argument('--keyword-threshold', type=int, default=2,
                       help='Minimum keyword matches (default: 2)')
    parser.add_argument('--skip-existing', action='store_true', default=False,
                       help='Skip tickers that already have filtered files')
    
    args = parser.parse_args()
    
    # Resolve news directory path
    news_dir = Path(args.news_dir)
    if not news_dir.is_absolute():
        project_root = Path(__file__).parent.parent
        news_dir = project_root / args.news_dir
    
    if not news_dir.exists():
        logger.error(f"News directory not found: {news_dir}")
        return 1
    
    logger.info("="*60)
    logger.info("BATCH FILTER NEWS ARTICLES")
    logger.info("="*60)
    logger.info(f"News directory: {news_dir}")
    logger.info(f"Max symbols: {args.max_symbols}")
    logger.info(f"Require primary: {args.require_primary}")
    logger.info(f"Require keywords: {args.require_keywords}")
    logger.info(f"Keyword threshold: {args.keyword_threshold}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("="*60)
    
    # Find all ticker files
    ticker_files = find_ticker_files(news_dir)
    
    if not ticker_files:
        logger.warning(f"No ticker files found in {news_dir}")
        return 1
    
    logger.info(f"Found {len(ticker_files)} ticker files to process")
    
    # Filter each ticker
    successful = 0
    failed = 0
    skipped = 0
    
    for ticker_file in ticker_files:
        ticker = extract_ticker_from_filename(ticker_file.name)
        ticker_base = ticker.replace('.US', '')
        output_file = news_dir / f"{ticker_base}_filtered_strict.json"
        
        # Skip if already exists and --skip-existing is set
        if args.skip_existing and output_file.exists():
            logger.info(f"⊘ Skipping {ticker} (output file already exists)")
            skipped += 1
            continue
        
        # Filter the ticker
        if filter_ticker_news(
            ticker_file,
            ticker,
            news_dir,
            max_symbols=args.max_symbols,
            require_primary=args.require_primary,
            require_keywords=args.require_keywords,
            keyword_threshold=args.keyword_threshold
        ):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total tickers: {len(ticker_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

