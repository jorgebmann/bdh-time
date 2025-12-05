#!/usr/bin/env python3
"""
Filter news articles to keep only those relevant to a specific ticker.

Filters out articles where the ticker is mentioned incidentally but isn't the focus.
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter

# Apple-related keywords (case-insensitive)
APPLE_KEYWORDS = {
    'apple', 'aapl', 'iphone', 'ipad', 'macbook', 'mac', 'ios', 'app store',
    'tim cook', 'cook', 'cupertino', 'apple watch', 'airpods', 'apple tv',
    'apple pay', 'icloud', 'siri', 'apple silicon', 'm-series', 'm1', 'm2', 'm3',
    'app store', 'apple music', 'apple services', 'apple store'
}

def is_apple_relevant(title: str, content: str, threshold: int = 2) -> bool:
    """
    Check if article is relevant to Apple based on keywords.
    
    Args:
        title: Article title
        content: Article content (first 500 chars checked)
        threshold: Minimum number of keyword matches
        
    Returns:
        True if article appears relevant to Apple
    """
    if not title and not content:
        return False
    
    text = f"{title} {content[:500]}".lower()
    
    # Count keyword matches
    matches = sum(1 for keyword in APPLE_KEYWORDS if keyword in text)
    
    return matches >= threshold


def filter_news_articles(
    articles: List[Dict],
    ticker: str = 'AAPL.US',
    filters: Dict = None
) -> tuple:
    """
    Filter news articles to keep only relevant ones.
    
    Args:
        articles: List of article dictionaries
        ticker: Ticker symbol to filter for
        filters: Dictionary with filter settings:
            - max_symbols: Maximum number of symbols (default: 10)
            - require_primary: AAPL must be first symbol (default: False)
            - require_keywords: Must have Apple keywords (default: True)
            - min_title_length: Minimum title length (default: 10)
            - min_content_length: Minimum content length (default: 50)
            - keyword_threshold: Min keyword matches (default: 2)
            - exclude_tags: Tags to exclude (default: [])
            
    Returns:
        Tuple of (filtered_articles, filter_stats)
    """
    if filters is None:
        filters = {}
    
    # Default filter settings
    max_symbols = filters.get('max_symbols', 10)
    require_primary = filters.get('require_primary', False)
    require_keywords = filters.get('require_keywords', True)
    min_title_length = filters.get('min_title_length', 10)
    min_content_length = filters.get('min_content_length', 50)
    keyword_threshold = filters.get('keyword_threshold', 2)
    exclude_tags = set(filters.get('exclude_tags', []))
    
    filtered = []
    stats = {
        'total': len(articles),
        'filtered_out': 0,
        'reasons': Counter()
    }
    
    for article in articles:
        symbols = article.get('symbols', [])
        title = article.get('title', '') or ''
        content = article.get('content', '') or ''
        tags = set(article.get('tags', []))
        
        # Check if ticker is in symbols
        if ticker not in symbols:
            stats['filtered_out'] += 1
            stats['reasons']['ticker_not_in_symbols'] += 1
            continue
        
        # Filter 1: Exclude articles with excluded tags
        if exclude_tags and tags.intersection(exclude_tags):
            stats['filtered_out'] += 1
            stats['reasons']['excluded_tag'] += 1
            continue
        
        # Filter 2: Too many symbols (likely generic market news)
        if len(symbols) > max_symbols:
            stats['filtered_out'] += 1
            stats['reasons'][f'too_many_symbols_{len(symbols)}'] += 1
            continue
        
        # Filter 3: Require AAPL to be primary (first symbol)
        if require_primary and symbols[0] != ticker:
            stats['filtered_out'] += 1
            stats['reasons']['not_primary_symbol'] += 1
            continue
        
        # Filter 4: Minimum content length
        if len(title) < min_title_length or len(content) < min_content_length:
            stats['filtered_out'] += 1
            stats['reasons']['too_short'] += 1
            continue
        
        # Filter 5: Keyword relevance (most important)
        if require_keywords and not is_apple_relevant(title, content, keyword_threshold):
            stats['filtered_out'] += 1
            stats['reasons']['no_keywords'] += 1
            continue
        
        # Article passed all filters
        filtered.append(article)
    
    stats['kept'] = len(filtered)
    stats['retention_rate'] = (len(filtered) / len(articles) * 100) if articles else 0
    
    return filtered, stats


def analyze_filtering_impact(
    articles: List[Dict],
    ticker: str = 'AAPL.US',
    filter_configs: List[Dict] = None
) -> None:
    """
    Analyze impact of different filter configurations.
    
    Args:
        articles: List of article dictionaries
        ticker: Ticker symbol
        filter_configs: List of filter configurations to test
    """
    if filter_configs is None:
        filter_configs = [
            {
                'name': 'Baseline (no filters)',
                'filters': {
                    'max_symbols': 1000,  # Effectively no limit
                    'require_primary': False,
                    'require_keywords': False,
                }
            },
            {
                'name': 'Keyword filter only',
                'filters': {
                    'max_symbols': 1000,
                    'require_primary': False,
                    'require_keywords': True,
                    'keyword_threshold': 2,
                }
            },
            {
                'name': 'Max 5 symbols + keywords',
                'filters': {
                    'max_symbols': 5,
                    'require_keywords': True,
                    'keyword_threshold': 2,
                }
            },
            {
                'name': 'Max 3 symbols + keywords',
                'filters': {
                    'max_symbols': 3,
                    'require_keywords': True,
                    'keyword_threshold': 2,
                }
            },
            {
                'name': 'Primary symbol + keywords',
                'filters': {
                    'max_symbols': 10,
                    'require_primary': True,
                    'require_keywords': True,
                    'keyword_threshold': 2,
                }
            },
            {
                'name': 'Strict: Primary + max 3 symbols + keywords',
                'filters': {
                    'max_symbols': 3,
                    'require_primary': True,
                    'require_keywords': True,
                    'keyword_threshold': 2,
                }
            },
        ]
    
    print(f"\n{'='*80}")
    print(f"FILTERING ANALYSIS FOR {ticker}")
    print(f"{'='*80}")
    print(f"Total articles: {len(articles)}\n")
    
    for config in filter_configs:
        filtered, stats = filter_news_articles(articles, ticker, config['filters'])
        
        print(f"{config['name']}:")
        print(f"  Kept: {stats['kept']} ({stats['retention_rate']:.1f}%)")
        print(f"  Filtered out: {stats['filtered_out']}")
        
        if stats['reasons']:
            print(f"  Top reasons for filtering:")
            for reason, count in stats['reasons'].most_common(5):
                print(f"    - {reason}: {count}")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filter news articles to keep only relevant ones for a ticker'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to JSON file with news data')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (optional)')
    parser.add_argument('--ticker', type=str, default='AAPL.US',
                       help='Ticker symbol to filter for')
    parser.add_argument('--max-symbols', type=int, default=10,
                       help='Maximum number of symbols per article')
    parser.add_argument('--require-primary', action='store_true',
                       help='Require ticker to be first symbol')
    parser.add_argument('--require-keywords', action='store_true', default=True,
                       help='Require Apple keywords in title/content')
    parser.add_argument('--no-keywords', dest='require_keywords', action='store_false',
                       help='Disable keyword requirement')
    parser.add_argument('--keyword-threshold', type=int, default=2,
                       help='Minimum keyword matches required')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze different filter configurations')
    
    args = parser.parse_args()
    
    # Load articles
    json_file = Path(args.input)
    if not json_file.exists():
        print(f"Error: File not found: {args.input}")
        return 1
    
    print(f"Loading articles from {args.input}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles")
    
    # Analyze different filter configurations
    if args.analyze:
        analyze_filtering_impact(articles, args.ticker)
        return 0
    
    # Apply filters
    filters = {
        'max_symbols': args.max_symbols,
        'require_primary': args.require_primary,
        'require_keywords': args.require_keywords,
        'keyword_threshold': args.keyword_threshold,
    }
    
    filtered, stats = filter_news_articles(articles, args.ticker, filters)
    
    print(f"\n{'='*60}")
    print("FILTERING RESULTS")
    print(f"{'='*60}")
    print(f"Total articles: {stats['total']}")
    print(f"Kept: {stats['kept']} ({stats['retention_rate']:.1f}%)")
    print(f"Filtered out: {stats['filtered_out']}")
    print(f"\nTop reasons for filtering:")
    for reason, count in stats['reasons'].most_common(10):
        print(f"  {reason}: {count}")
    
    # Save filtered articles
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved {len(filtered)} filtered articles to {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

