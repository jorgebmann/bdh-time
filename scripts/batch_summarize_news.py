#!/usr/bin/env python3
"""
Batch summarize news articles for all filtered ticker files.

Iterates over all {ticker}_filtered_strict.json files in data/news/ and applies
summarization to create {ticker}_summarized.json files. Skips tickers that already
have summarized files.
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


def find_filtered_files(news_dir: Path) -> List[Path]:
    """
    Find all {ticker}_filtered_strict.json files in the news directory.
    
    Args:
        news_dir: Path to news directory
        
    Returns:
        List of Path objects for filtered files
    """
    pattern = "*_filtered_strict.json"
    filtered_files = list(news_dir.glob(pattern))
    return sorted(filtered_files)


def extract_ticker_from_filename(filename: str) -> str:
    """
    Extract ticker symbol from filtered filename.
    
    Args:
        filename: Filename like "AAPL_filtered_strict.json"
        
    Returns:
        Ticker symbol like "AAPL.US"
    """
    # Remove _filtered_strict.json suffix
    ticker_base = filename.replace('_filtered_strict.json', '')
    # Add .US suffix if not present
    if not ticker_base.endswith('.US'):
        ticker = f"{ticker_base}.US"
    else:
        ticker = ticker_base
    return ticker


def summarize_ticker_news(
    input_file: Path,
    ticker: str,
    news_dir: Path,
    token_threshold: int = 10000,
    max_summary_tokens: int = 3000,
    min_summary_tokens: int = 1500,
    config_path: str = "config.yaml"
) -> bool:
    """
    Summarize news for a single ticker using summarize_news.py script.
    
    Args:
        input_file: Path to input filtered JSON file
        ticker: Ticker symbol (e.g., "AAPL.US")
        news_dir: Directory for output files
        token_threshold: Token threshold for summarization
        max_summary_tokens: Maximum tokens for summary
        min_summary_tokens: Minimum tokens for summary
        config_path: Path to config file
        
    Returns:
        True if successful, False otherwise
    """
    # Construct output filename
    ticker_base = ticker.replace('.US', '')
    output_file = news_dir / f"{ticker_base}_summarized.json"
    
    # Build command
    script_path = Path(__file__).parent / "summarize_news.py"
    cmd = [
        sys.executable,
        str(script_path),
        '--input', str(input_file),
        '--ticker', ticker,
        '--output', str(output_file),
        '--token-threshold', str(token_threshold),
        '--max-summary-tokens', str(max_summary_tokens),
        '--min-summary-tokens', str(min_summary_tokens),
        '--config', config_path
    ]
    
    try:
        logger.info(f"Summarizing {ticker}...")
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
        
        logger.info(f"✓ Summarized {ticker}: {output_file.name}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Error summarizing {ticker}: {e}")
        if e.stdout:
            logger.error(f"  stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"  stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error summarizing {ticker}: {e}")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch summarize news articles for all filtered ticker files'
    )
    parser.add_argument('--news-dir', type=str, default='data/news',
                       help='Directory containing news JSON files (default: data/news)')
    parser.add_argument('--token-threshold', type=int, default=10000,
                       help='Token threshold for summarization (default: 10000)')
    parser.add_argument('--max-summary-tokens', type=int, default=3000,
                       help='Target max tokens for summary (default: 3000)')
    parser.add_argument('--min-summary-tokens', type=int, default=1500,
                       help='Minimum tokens for summary (default: 1500)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--force', action='store_true', default=False,
                       help='Force re-summarization even if output file exists')
    
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
    logger.info("BATCH SUMMARIZE NEWS ARTICLES")
    logger.info("="*60)
    logger.info(f"News directory: {news_dir}")
    logger.info(f"Token threshold: {args.token_threshold}")
    logger.info(f"Max summary tokens: {args.max_summary_tokens}")
    logger.info(f"Min summary tokens: {args.min_summary_tokens}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Force re-summarization: {args.force}")
    logger.info("="*60)
    
    # Find all filtered files
    filtered_files = find_filtered_files(news_dir)
    
    if not filtered_files:
        logger.warning(f"No *_filtered_strict.json files found in {news_dir}")
        return 1
    
    logger.info(f"Found {len(filtered_files)} filtered files to process")
    
    # Summarize each ticker
    successful = 0
    failed = 0
    skipped = 0
    
    for filtered_file in filtered_files:
        ticker = extract_ticker_from_filename(filtered_file.name)
        ticker_base = ticker.replace('.US', '')
        output_file = news_dir / f"{ticker_base}_summarized.json"
        
        # Skip if already exists and not forcing
        if not args.force and output_file.exists():
            logger.info(f"⊘ Skipping {ticker} (summarized file already exists: {output_file.name})")
            skipped += 1
            continue
        
        # Summarize the ticker
        if summarize_ticker_news(
            filtered_file,
            ticker,
            news_dir,
            token_threshold=args.token_threshold,
            max_summary_tokens=args.max_summary_tokens,
            min_summary_tokens=args.min_summary_tokens,
            config_path=args.config
        ):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total tickers: {len(filtered_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

