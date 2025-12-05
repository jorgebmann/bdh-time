#!/usr/bin/env python3
"""
Analyze news data and generate aggregated DataFrame.

Generates a dataframe with timestamp as index, and columns:
- ticker
- number of news (aggregated over date)
- number of tokens (aggregated over date)
- mean sentiment
- sd sentiment
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import re

# Simple tokenizer - split by whitespace and punctuation
def count_tokens(text: str) -> int:
    """Count tokens in text (simple whitespace/punctuation split)."""
    if not text or pd.isna(text):
        return 0
    # Split by whitespace and common punctuation
    tokens = re.findall(r'\b\w+\b', str(text))
    return len(tokens)


def analyze_news_data(json_path: str, ticker: str = None) -> pd.DataFrame:
    """
    Analyze news data and generate aggregated DataFrame.
    
    Args:
        json_path: Path to JSON file with news data
        ticker: Ticker symbol (if None, inferred from filename)
        
    Returns:
        DataFrame with timestamp index and aggregated metrics
    """
    # Load JSON data
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"News file not found: {json_path}")
    
    # Infer ticker from filename if not provided
    if ticker is None:
        ticker = json_file.stem.upper()
        if not ticker.endswith('.US'):
            ticker = f"{ticker}.US"
    
    print(f"Loading news data from {json_path}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        news_data = json.load(f)
    
    if not news_data:
        print("Warning: No news data found in file")
        return pd.DataFrame()
    
    print(f"Loaded {len(news_data)} news articles")
    
    # Parse articles and extract metrics
    records = []
    for article in news_data:
        # Parse date
        date_str = article.get('date', '')
        if not date_str:
            continue
        
        try:
            # Parse ISO format timestamp
            dt = pd.to_datetime(date_str)
            # Extract date (YYYY-MM-DD) for aggregation
            date = dt.date()
        except Exception as e:
            print(f"Warning: Could not parse date '{date_str}': {e}")
            continue
        
        # Get sentiment polarity (use polarity as sentiment score)
        sentiment = article.get('sentiment', None)
        if sentiment is None or not isinstance(sentiment, dict):
            # Skip articles without sentiment
            continue
        
        polarity = sentiment.get('polarity', None)
        if polarity is None:
            # Skip articles without polarity
            continue
        
        # Count tokens (title + content)
        title = article.get('title', '') or ''
        content = article.get('content', '') or ''
        total_text = f"{title} {content}"
        num_tokens = count_tokens(total_text)
        
        records.append({
            'date': date,
            'ticker': ticker,
            'num_news': 1,
            'num_tokens': num_tokens,
            'sentiment': polarity
        })
    
    if not records:
        print("Warning: No valid records found")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Aggregate by date
    print(f"\nAggregating {len(df)} articles by date...")
    aggregated = df.groupby('date').agg({
        'ticker': 'first',  # Should be same for all
        'num_news': 'sum',
        'num_tokens': 'sum',
        'sentiment': ['mean', 'std']
    })
    
    # Flatten column names
    aggregated.columns = ['ticker', 'number_of_news', 'number_of_tokens', 'mean_sentiment', 'sd_sentiment']
    
    # Reset index to make date a column, then set as index with timestamp
    aggregated = aggregated.reset_index()
    aggregated['timestamp'] = pd.to_datetime(aggregated['date'])
    aggregated = aggregated.set_index('timestamp')
    
    # Drop the date column (now redundant)
    aggregated = aggregated.drop(columns=['date'])
    
    # Sort by timestamp
    aggregated = aggregated.sort_index()
    
    # Fill NaN std with 0 (when only one article per date)
    aggregated['sd_sentiment'] = aggregated['sd_sentiment'].fillna(0.0)
    
    return aggregated


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze news data and generate aggregated DataFrame')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to JSON file with news data')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Ticker symbol (default: inferred from filename)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    # Analyze news data
    df = analyze_news_data(args.input, args.ticker)
    
    if df.empty:
        print("No data to display")
        return 1
    
    print("\n" + "="*60)
    print("AGGREGATED NEWS DATA")
    print("="*60)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nLast few rows:")
    print(df.tail(10))
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(df.describe())
    
    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        print(f"\nSaved DataFrame to {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

