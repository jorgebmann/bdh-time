#!/usr/bin/env python3
"""
Summarize news articles per day using Azure OpenAI GPT-4o when token count exceeds threshold.

Groups articles by date, counts tokens per day, and summarizes days exceeding 11k tokens.
Output includes daily summary objects with aggregated sentiment statistics.
"""

import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import tiktoken
import yaml
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tokenizer for GPT-4o (cl100k_base encoding)
ENCODING = tiktoken.get_encoding("cl100k_base")


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (relative or absolute)
        
    Returns:
        Dictionary with configuration values
    """
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


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken (GPT-4o encoding).
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    try:
        return len(ENCODING.encode(str(text)))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}")
        return 0


def parse_date(date_str: str) -> Optional[date]:
    """
    Parse date string to date object.
    
    Args:
        date_str: ISO format date string (e.g., "2025-12-04T01:20:57+00:00")
        
    Returns:
        Date object or None if parsing fails
    """
    if not date_str:
        return None
    try:
        dt = pd.to_datetime(date_str)
        return dt.date()
    except Exception as e:
        logger.warning(f"Could not parse date '{date_str}': {e}")
        return None


def group_articles_by_date(articles: List[Dict]) -> Dict[date, List[Dict]]:
    """
    Group articles by date.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Dictionary mapping date to list of articles for that date
    """
    grouped = defaultdict(list)
    
    for article in articles:
        date_str = article.get('date', '')
        article_date = parse_date(date_str)
        if article_date:
            grouped[article_date].append(article)
        else:
            logger.warning(f"Skipping article without valid date: {article.get('title', 'Unknown')[:50]}")
    
    return dict(grouped)


def calculate_daily_sentiment(articles: List[Dict]) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate mean and standard deviation of sentiment polarity for a list of articles.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Tuple of (mean, std) or (None, None) if no valid sentiment found
    """
    polarities = []
    
    for article in articles:
        sentiment = article.get('sentiment', None)
        if sentiment is None or not isinstance(sentiment, dict):
            continue
        
        polarity = sentiment.get('polarity', None)
        if polarity is not None:
            polarities.append(polarity)
    
    if not polarities:
        return None, None
    
    mean_polarity = np.mean(polarities)
    std_polarity = np.std(polarities) if len(polarities) > 1 else 0.0
    
    return float(mean_polarity), float(std_polarity)


def count_daily_tokens(articles: List[Dict]) -> int:
    """
    Count total tokens for all articles in a day.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Total token count
    """
    total_tokens = 0
    
    for article in articles:
        title = article.get('title', '') or ''
        content = article.get('content', '') or ''
        text = f"{title} {content}"
        total_tokens += count_tokens(text)
    
    return total_tokens


def format_articles_for_summarization(articles: List[Dict]) -> str:
    """
    Format articles as text for summarization.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Formatted text string
    """
    formatted = []
    
    for i, article in enumerate(articles, 1):
        title = article.get('title', '') or ''
        content = article.get('content', '') or ''
        date_str = article.get('date', '') or ''
        
        formatted.append(f"Article {i} ({date_str}):")
        formatted.append(f"Title: {title}")
        formatted.append(f"Content: {content}")
        formatted.append("")  # Empty line between articles
    
    return "\n".join(formatted)


def summarize_articles_azure(
    articles: List[Dict],
    ticker: str,
    config: dict,
    max_summary_tokens: int = 3000,
    min_summary_tokens: int = 1500
) -> Optional[str]:
    """
    Summarize articles using Azure OpenAI GPT-4o.
    
    Args:
        articles: List of article dictionaries to summarize
        ticker: Ticker symbol (for context in prompt)
        config: Configuration dictionary with Azure OpenAI settings
        max_summary_tokens: Maximum tokens for summary
        min_summary_tokens: Minimum tokens for summary (default: 1500)
        
    Returns:
        Summary text or None if summarization fails
    """
    try:
        # Get Azure OpenAI configuration
        azure_config = config.get('azure_openai', {})
        llm_config = config.get('llm_large', {})
        
        api_key = azure_config.get('openai_api_key')
        azure_endpoint = azure_config.get('azure_endpoint')
        api_version = azure_config.get('openai_api_version', '2024-12-01-preview')
        deployment_name = llm_config.get('deployment_name', 'gpt-4o-2024-11-20')
        timeout = azure_config.get('timeout', 90)
        max_retries = azure_config.get('max_retries', 3)
        
        if not api_key or not azure_endpoint:
            logger.error("Azure OpenAI credentials not found in config")
            return None
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Format articles for summarization
        articles_text = format_articles_for_summarization(articles)
        
        # Create system and user prompts - Updated to request longer, detailed summaries
        system_prompt = (
            f"You are a financial news analyst summarizing articles about {ticker}. "
            f"Generate a comprehensive, detailed summary that covers all key events, market impacts, "
            f"sentiment trends, and important details from the articles. "
            f"The summary must be AT LEAST {min_summary_tokens} tokens long and should be thorough and detailed. "
            f"Aim for approximately {max_summary_tokens} tokens, but ensure it is comprehensive and covers all important aspects. "
            f"Include specific details, numbers, dates, and context from the articles."
        )
        
        user_prompt = (
            f"Please provide a comprehensive, detailed summary of the following {len(articles)} articles about {ticker}. "
            f"The summary must be at least {min_summary_tokens} tokens long and should include:\n"
            f"- All key events and developments\n"
            f"- Market impact and implications\n"
            f"- Sentiment analysis and trends\n"
            f"- Specific details, numbers, and dates\n"
            f"- Context and background information\n\n"
            f"Articles:\n\n{articles_text}"
        )
        
        # Call Azure OpenAI API with retry if summary is too short
        max_attempts = 3
        summary = None
        summary_tokens = 0
        
        for attempt in range(max_attempts):
            logger.info(f"Summarizing {len(articles)} articles for {ticker} (attempt {attempt + 1}/{max_attempts}, target: {min_summary_tokens}-{max_summary_tokens} tokens)")
            
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent summaries
                max_tokens=max_summary_tokens
            )
            
            summary = response.choices[0].message.content
            summary_tokens = count_tokens(summary)
            
            logger.info(f"Generated summary: {summary_tokens} tokens")
            
            # Check if summary meets minimum length requirement
            if summary_tokens >= min_summary_tokens:
                return summary
            else:
                logger.warning(f"Summary too short ({summary_tokens} < {min_summary_tokens} tokens), retrying...")
                if attempt < max_attempts - 1:
                    # Update prompt to emphasize length requirement
                    user_prompt = (
                        f"The previous summary was too short. Please provide a MUCH LONGER and more detailed summary "
                        f"of the following {len(articles)} articles about {ticker}. "
                        f"The summary MUST be at least {min_summary_tokens} tokens long. "
                        f"Be thorough and include extensive details:\n"
                        f"- All key events and developments\n"
                        f"- Market impact and implications\n"
                        f"- Sentiment analysis and trends\n"
                        f"- Specific details, numbers, and dates\n"
                        f"- Context and background information\n"
                        f"- Quotes and specific examples from the articles\n\n"
                        f"Articles:\n\n{articles_text}"
                    )
        
        # If we get here, all attempts failed to meet minimum length
        logger.warning(f"Summary still too short after {max_attempts} attempts ({summary_tokens} tokens), returning anyway")
        return summary
        
    except Exception as e:
        logger.error(f"Error summarizing articles with Azure OpenAI: {e}")
        return None


def process_daily_articles(
    articles: List[Dict],
    date_obj: date,
    ticker: str,
    config: dict,
    token_threshold: int = 10000,
    max_summary_tokens: int = 3000,
    min_summary_tokens: int = 1500
) -> Dict:
    """
    Process articles for a single day.
    
    Args:
        articles: List of articles for the day
        date_obj: Date object for the day
        ticker: Ticker symbol
        config: Configuration dictionary
        token_threshold: Token threshold for summarization
        max_summary_tokens: Maximum tokens for summary
        min_summary_tokens: Minimum tokens for summary (default: 1500)
        
    Returns:
        Daily summary dictionary
    """
    # Count tokens
    original_tokens = count_daily_tokens(articles)
    
    # Calculate sentiment statistics
    sentiment_mean, sentiment_sd = calculate_daily_sentiment(articles)
    
    # Create base summary object
    summary_obj = {
        "date": date_obj.isoformat(),
        "num_articles": len(articles),
        "original_tokens": original_tokens,
        "sentiment_mean": sentiment_mean,
        "sentiment_sd": sentiment_sd
    }
    
    # Summarize if tokens exceed threshold
    if original_tokens > token_threshold:
        logger.info(f"Day {date_obj}: {original_tokens} tokens exceed threshold ({token_threshold}), summarizing...")
        
        summary_text = summarize_articles_azure(
            articles,
            ticker,
            config,
            max_summary_tokens=max_summary_tokens,
            min_summary_tokens=min_summary_tokens
        )
        
        if summary_text:
            summary_obj["summary"] = summary_text
            summary_obj["summary_tokens"] = count_tokens(summary_text)
        else:
            logger.warning(f"Failed to generate summary for {date_obj}, setting to null")
            summary_obj["summary"] = None
            summary_obj["summary_tokens"] = 0
    else:
        logger.debug(f"Day {date_obj}: {original_tokens} tokens <= threshold ({token_threshold}), no summarization needed")
        summary_obj["summary"] = None
        summary_obj["summary_tokens"] = 0
    
    return summary_obj


def summarize_news(
    input_path: str,
    ticker: Optional[str] = None,
    output_path: Optional[str] = None,
    token_threshold: int = 10000,
    max_summary_tokens: int = 3000,
    min_summary_tokens: int = 1500,
    config: Optional[dict] = None
) -> Dict:
    """
    Main function to summarize news articles.
    
    Args:
        input_path: Path to filtered news JSON file
        ticker: Ticker symbol (inferred from filename if not provided)
        output_path: Output file path (default: {ticker}_summarized.json)
        token_threshold: Token threshold for summarization
        max_summary_tokens: Maximum tokens for summary
        min_summary_tokens: Minimum tokens for summary (default: 1500)
        config: Configuration dictionary (loaded from config.yaml if not provided)
        
    Returns:
        Dictionary with articles and daily_summaries
    """
    # Load configuration
    if config is None:
        config = load_config()
    
    # Load input file
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Infer ticker from filename if not provided
    if ticker is None:
        ticker = input_file.stem.upper()
        # Remove _filtered_strict suffix if present
        if ticker.endswith('_FILTERED_STRICT'):
            ticker = ticker.replace('_FILTERED_STRICT', '')
        if not ticker.endswith('.US'):
            ticker = f"{ticker}.US"
    
    logger.info(f"Processing news for {ticker} from {input_path}")
    
    # Load articles
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and dict input formats
    if isinstance(data, list):
        articles = data
    elif isinstance(data, dict) and 'articles' in data:
        articles = data['articles']
    else:
        raise ValueError(f"Unexpected input format. Expected list or dict with 'articles' key, got {type(data)}")
    
    if not articles:
        logger.warning("No articles found in input file")
        return {"articles": [], "daily_summaries": []}
    
    logger.info(f"Loaded {len(articles)} articles")
    
    # Group articles by date
    logger.info("Grouping articles by date...")
    articles_by_date = group_articles_by_date(articles)
    logger.info(f"Found articles for {len(articles_by_date)} unique dates")
    
    # Process each day
    daily_summaries = []
    days_summarized = 0
    total_tokens_saved = 0
    
    for date_obj in sorted(articles_by_date.keys()):
        day_articles = articles_by_date[date_obj]
        
        summary_obj = process_daily_articles(
            day_articles,
            date_obj,
            ticker,
            config,
            token_threshold=token_threshold,
            max_summary_tokens=max_summary_tokens,
            min_summary_tokens=min_summary_tokens
        )
        
        daily_summaries.append(summary_obj)
        
        # Track statistics
        if summary_obj["summary"] is not None:
            days_summarized += 1
            tokens_saved = summary_obj["original_tokens"] - summary_obj["summary_tokens"]
            total_tokens_saved += tokens_saved
    
    # Create output structure
    output = {
        "articles": articles,
        "daily_summaries": daily_summaries
    }
    
    # Save output
    if output_path is None:
        output_path = input_file.parent / f"{ticker.replace('.US', '')}_summarized.json"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Saved output to {output_path}")
    
    # Print summary statistics
    logger.info("="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)
    logger.info(f"Total articles: {len(articles)}")
    logger.info(f"Days processed: {len(daily_summaries)}")
    logger.info(f"Days summarized: {days_summarized}")
    logger.info(f"Total tokens saved: {total_tokens_saved:,}")
    
    return output


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Summarize news articles per day using Azure OpenAI GPT-4o'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to filtered news JSON file (e.g., {ticker}_filtered_strict.json)')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Ticker symbol (default: inferred from filename)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: {ticker}_summarized.json)')
    parser.add_argument('--token-threshold', type=int, default=10000,
                       help='Token threshold for summarization (default: 10000)')
    parser.add_argument('--max-summary-tokens', type=int, default=3000,
                       help='Target max tokens for summary (default: 3000)')
    parser.add_argument('--min-summary-tokens', type=int, default=1500,
                       help='Minimum tokens for summary (default: 1500)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    try:
        # Load config
        config = load_config(args.config)
        
        # Run summarization
        summarize_news(
            input_path=args.input,
            ticker=args.ticker,
            output_path=args.output,
            token_threshold=args.token_threshold,
            max_summary_tokens=args.max_summary_tokens,
            min_summary_tokens=args.min_summary_tokens,
            config=config
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

