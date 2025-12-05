#!/usr/bin/env python3
"""
Analyze news spike on a specific date to understand why there are so many articles.

This script analyzes:
- Which stocks are mentioned in articles on that date
- Common tags/themes
- Whether articles mention multiple stocks
- Sample article titles to understand the context
"""

import sys
import json
from pathlib import Path
from collections import Counter
from datetime import datetime
import pandas as pd


def analyze_date_spike(json_path: str, target_date: str):
    """
    Analyze news articles on a specific date to understand the spike.

    Args:
        json_path: Path to JSON file with news data
        target_date: Date to analyze (YYYY-MM-DD format)
    """
    # Load news data
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"News file not found: {json_path}")

    print(f"Loading news data from {json_path}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        news_data = json.load(f)

    print(f"Total articles in file: {len(news_data)}")

    # Filter for target date
    articles_on_date = []
    for article in news_data:
        date_str = article.get('date', '')
        if date_str.startswith(target_date):
            articles_on_date.append(article)

    print(f"\n{'=' * 60}")
    print(f"ANALYSIS FOR DATE: {target_date}")
    print(f"{'=' * 60}")
    print(f"Total articles on {target_date}: {len(articles_on_date)}")

    if not articles_on_date:
        print(f"No articles found for {target_date}")
        return

    # Analyze symbols mentioned
    all_symbols = []
    for article in articles_on_date:
        symbols = article.get('symbols', [])
        all_symbols.extend(symbols)

    symbol_counts = Counter(all_symbols)
    print(f"\n{'=' * 60}")
    print("SYMBOLS MENTIONED")
    print(f"{'=' * 60}")
    print(f"Total unique symbols: {len(set(all_symbols))}")
    print(f"Total symbol mentions: {len(all_symbols)}")
    print(f"Average symbols per article: {len(all_symbols) / len(articles_on_date):.2f}")

    print(f"\nTop 30 most mentioned symbols:")
    for i, (symbol, count) in enumerate(symbol_counts.most_common(30), 1):
        percentage = (count / len(articles_on_date)) * 100
        print(f"  {i:2d}. {symbol:15s}: {count:5d} articles ({percentage:5.1f}%)")

    # Analyze multi-stock articles
    multi_stock_articles = [a for a in articles_on_date if len(a.get('symbols', [])) > 1]
    single_stock_articles = [a for a in articles_on_date if len(a.get('symbols', [])) == 1]

    print(f"\n{'=' * 60}")
    print("ARTICLE COMPOSITION")
    print(f"{'=' * 60}")
    print(f"Articles mentioning only AAPL.US: {len([a for a in articles_on_date if a.get('symbols') == ['AAPL.US']])}")
    print(f"Articles mentioning AAPL.US + other stocks: {len(multi_stock_articles)}")
    print(
        f"Articles mentioning multiple stocks (>1): {len(multi_stock_articles)} ({len(multi_stock_articles) / len(articles_on_date) * 100:.1f}%)")
    print(
        f"Articles mentioning single stock: {len(single_stock_articles)} ({len(single_stock_articles) / len(articles_on_date) * 100:.1f}%)")

    # Analyze tags
    all_tags = []
    for article in articles_on_date:
        tags = article.get('tags', [])
        all_tags.extend(tags)

    tag_counts = Counter(all_tags)
    print(f"\n{'=' * 60}")
    print("TOP TAGS/THEMES")
    print(f"{'=' * 60}")
    print(f"Top 20 most common tags:")
    for i, (tag, count) in enumerate(tag_counts.most_common(20), 1):
        percentage = (count / len(articles_on_date)) * 100
        print(f"  {i:2d}. {tag:30s}: {count:5d} articles ({percentage:5.1f}%)")

    # Show sample article titles
    print(f"\n{'=' * 60}")
    print("SAMPLE ARTICLE TITLES (first 20)")
    print(f"{'=' * 60}")
    for i, article in enumerate(articles_on_date[:20], 1):
        title = article.get('title', 'N/A')
        symbols = article.get('symbols', [])
        num_symbols = len(symbols)
        print(f"{i:2d}. [{num_symbols} symbols] {title[:80]}")
        if len(symbols) > 1:
            print(f"     Symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")

    # Analyze sentiment distribution
    sentiments = []
    for article in articles_on_date:
        sentiment = article.get('sentiment', None)
        if sentiment and isinstance(sentiment, dict):
            polarity = sentiment.get('polarity', None)
            if polarity is not None:
                sentiments.append(polarity)

    if sentiments:
        print(f"\n{'=' * 60}")
        print("SENTIMENT ANALYSIS")
        print(f"{'=' * 60}")
        print(f"Articles with sentiment data: {len(sentiments)}")
        print(f"Mean sentiment: {pd.Series(sentiments).mean():.3f}")
        print(f"Std sentiment: {pd.Series(sentiments).std():.3f}")
        print(f"Min sentiment: {pd.Series(sentiments).min():.3f}")
        print(f"Max sentiment: {pd.Series(sentiments).max():.3f}")

    # Check for common patterns in titles
    print(f"\n{'=' * 60}")
    print("COMMON KEYWORDS IN TITLES")
    print(f"{'=' * 60}")
    title_words = []
    for article in articles_on_date:
        title = article.get('title', '').lower()
        # Simple word extraction
        words = title.split()
        title_words.extend([w.strip('.,!?;:()[]{}"\'') for w in words if len(w) > 3])

    word_counts = Counter(title_words)
    print("Top 20 words in titles:")
    for i, (word, count) in enumerate(word_counts.most_common(20), 1):
        print(f"  {i:2d}. {word:20s}: {count:5d} times")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze news spike on a specific date')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to JSON file with news data')
    parser.add_argument('--date', type=str, required=True,
                        help='Date to analyze (YYYY-MM-DD format)')

    args = parser.parse_args()

    analyze_date_spike(args.input, args.date)


if __name__ == '__main__':
    main()