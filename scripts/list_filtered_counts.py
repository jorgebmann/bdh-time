#!/usr/bin/env python3
"""
List the number of entries in each *_filtered_strict.json file.
Shows total entries and number of unique days with news per ticker.
"""

import sys
import json
from pathlib import Path
from typing import List, Set
import pandas as pd

def load_articles_from_file(file_path: Path) -> List[dict]:
    """
    Load articles from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of article dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Check if it has an 'articles' key
            if 'articles' in data:
                return data['articles']
            else:
                # Return empty list if no articles key
                return []
        else:
            return []
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}", file=sys.stderr)
        return []


def count_unique_days(articles: List[dict]) -> int:
    """
    Count unique days that have news articles.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Number of unique days with news
    """
    unique_dates = set()
    
    for article in articles:
        date_str = article.get('date', '')
        if not date_str:
            continue
        
        try:
            dt = pd.to_datetime(date_str)
            date_key = dt.strftime('%Y-%m-%d')
            unique_dates.add(date_key)
        except Exception:
            continue
    
    return len(unique_dates)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='List number of entries and unique days with news per ticker'
    )
    parser.add_argument('--news-dir', type=str, default='data/news',
                       help='Directory containing news JSON files (default: data/news)')
    parser.add_argument('--sort', choices=['name', 'count', 'days'], default='name',
                       help='Sort by name, count, or days (default: name)')
    
    args = parser.parse_args()
    
    # Resolve news directory path
    news_dir = Path(args.news_dir)
    if not news_dir.is_absolute():
        project_root = Path(__file__).parent.parent
        news_dir = project_root / args.news_dir
    
    if not news_dir.exists():
        print(f"Error: News directory not found: {news_dir}", file=sys.stderr)
        return 1
    
    # Find all *_filtered_strict.json files
    filtered_files = sorted(news_dir.glob("*_filtered_strict.json"))
    
    if not filtered_files:
        print(f"No *_filtered_strict.json files found in {news_dir}")
        return 0
    
    # Process files
    results = []
    total_entries = 0
    total_days = 0
    
    for file_path in filtered_files:
        articles = load_articles_from_file(file_path)
        if articles is None:
            continue
        
        ticker = file_path.stem.replace('_filtered_strict', '')
        entry_count = len(articles)
        unique_days = count_unique_days(articles)
        
        results.append((ticker, entry_count, unique_days))
        total_entries += entry_count
        total_days += unique_days
    
    # Sort results
    if args.sort == 'count':
        results.sort(key=lambda x: x[1], reverse=True)
    elif args.sort == 'days':
        results.sort(key=lambda x: x[2], reverse=True)
    else:
        results.sort(key=lambda x: x[0])
    
    # Print results
    print(f"{'Ticker':<20} {'Entries':>12} {'Unique Days':>12}")
    print("=" * 46)
    
    for ticker, entry_count, unique_days in results:
        print(f"{ticker:<20} {entry_count:>12,} {unique_days:>12,}")
    
    print("=" * 46)
    print(f"{'TOTAL':<20} {total_entries:>12,} {total_days:>12,}")
    print(f"\nFiles processed: {len(results)}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

