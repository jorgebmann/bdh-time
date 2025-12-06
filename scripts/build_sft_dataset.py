#!/usr/bin/env python3
"""
Build a supervised fine-tuning dataset for GPT-4.1 that predicts next-day
close-to-close returns from NASDAQ-100 news.

Inputs:
- News files: data/news/*_summarized.json
  Structure: {"articles": [...], "daily_summaries": [...]} per ticker.
- Prices: data/raw_daily/us_market_YYYY-MM-DD.parquet (EODHD bulk files).

Output:
- JSONL chat format at data/processed/sft_next_day_returns.jsonl
  Each line has a constant system prompt, one user message with structured
  news features, and an assistant message with the target return (%), rounded
  to 2 decimals.

Assumptions:
- Price column preference: adjusted_close > adj_close > close.
- Symbol column preference: Code > Symbol > code > symbol > ticker > Ticker.
- Date column preference: date > Date; if missing, fallback to filename date.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# -----------------------
# Configuration
# -----------------------
START_DATE = date.fromisoformat("2024-12-01")
END_DATE = date.fromisoformat("2025-12-04")
NEWS_DIR = Path("data/news")
RAW_DAILY_DIR = Path("data/raw_daily")
OUTPUT_PATH = Path("data/processed/sft_next_day_returns.jsonl")
EODHD_EXCHANGE = "US"
EODHD_API_TOKEN = os.getenv("EODHD_API_TOKEN") or os.getenv("API_TOKEN")
SYSTEM_PROMPT = (
    "You are a quantitative model predicting next-day close-to-close return (%) "
    "for NASDAQ-100 equities from prior-day news. Respond with a single number "
    "representing the percent change, rounded to 2 decimals. Do not add text."
)
MAX_ARTICLE_CHARS = 1000  # truncate long bodies to keep tokens manageable


# -----------------------
# Helpers
# -----------------------
def ensure_parquet_engine() -> None:
    """
    Ensure a parquet engine is available. pandas requires either pyarrow or
    fastparquet. If neither is installed, raise a clear error early.
    """
    try:
        import pyarrow  # type: ignore  # noqa: F401
        return
    except Exception:
        try:
            import fastparquet  # type: ignore  # noqa: F401
            return
        except Exception as exc:
            raise RuntimeError(
                "Install a parquet engine (pyarrow or fastparquet) to read/write "
                "raw daily files. Example: pip install --user pyarrow"
            ) from exc


def parse_filename_date(path: Path) -> Optional[date]:
    """Extract YYYY-MM-DD from a filename such as us_market_2024-12-01.parquet."""
    try:
        name = path.stem
        parts = name.split("_")
        maybe_date = parts[-1]
        return date.fromisoformat(maybe_date)
    except Exception:
        return None


def pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first column name present from a list of candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def pick_price_column(df: pd.DataFrame) -> str:
    """Choose the best available close/adjusted close column."""
    lowered = {c.lower(): c for c in df.columns}
    for key in [
        "adjusted_close",
        "adj_close",
        "adjclose",
        "close_adj",
        "closeadjusted",
    ]:
        if key in lowered:
            return lowered[key]
    if "close" in lowered:
        return lowered["close"]
    raise ValueError(f"No price column found. Columns: {df.columns}")


def pick_symbol_column(df: pd.DataFrame) -> str:
    col = pick_column(df, ["Code", "Symbol", "code", "symbol", "ticker", "Ticker"])
    if not col:
        raise ValueError(f"No symbol column found. Columns: {df.columns}")
    return col


def pick_date_column(df: pd.DataFrame) -> Optional[str]:
    return pick_column(df, ["date", "Date", "DATE"])


def normalize_ticker(symbol: str) -> str:
    """Normalize ticker to upper-case with exchange suffix preserved if present."""
    if symbol is None:
        return symbol
    sym = str(symbol).strip()
    return sym.upper()


def truncate_text(text: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


@dataclass
class DailyStats:
    num_articles: int
    sentiment_mean: Optional[float]
    sentiment_sd: Optional[float]
    summary: Optional[str]


# -----------------------
# Data loading
# -----------------------
def load_price_frame(file_path: Path) -> pd.DataFrame:
    """Load a single daily parquet file into a normalized dataframe."""
    df = pd.read_parquet(file_path)
    return normalize_price_df(df, fallback_date=parse_filename_date(file_path))


def normalize_price_df(
    df: pd.DataFrame, fallback_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Normalize a raw EODHD dataframe into ticker/date/close columns.
    If the date column is missing, fallback_date is used.
    """
    if df.empty:
        return df

    sym_col = pick_symbol_column(df)
    price_col = pick_price_column(df)
    date_col = pick_date_column(df)

    df = df[[sym_col, price_col]].copy()
    df.rename(columns={sym_col: "ticker", price_col: "close"}, inplace=True)

    if date_col:
        df["date"] = pd.to_datetime(df[date_col]).dt.date
    else:
        df["date"] = fallback_date

    # Keep only rows with a resolved date
    df = df.dropna(subset=["date"])
    df["ticker"] = df["ticker"].map(normalize_ticker)
    return df[["ticker", "date", "close"]]


def load_prices(
    raw_daily_dir: Path,
    start: date,
    end: date,
    buffer_days: int = 5,
    api_token: Optional[str] = None,
    exchange: str = EODHD_EXCHANGE,
) -> pd.DataFrame:
    """
    Load price data for [start, end] plus a buffer of future trading days so
    we can compute t+1 returns near the end of the range.
    """
    ensure_parquet_engine()
    end_with_buffer = end + timedelta(days=buffer_days)

    rows: List[pd.DataFrame] = []
    business_days = pd.date_range(start=start, end=end_with_buffer, freq="B").date

    for day in tqdm(business_days, desc="Loading price parquet files"):
        file_path = raw_daily_dir / f"us_market_{day.isoformat()}.parquet"
        if file_path.exists():
            df_day = load_price_frame(file_path)
            if not df_day.empty:
                rows.append(df_day)
            continue

        if not api_token:
            logger.warning(
                "Missing price file %s and no API token provided; skipping", file_path
            )
            continue

        df_raw = fetch_bulk_day(day, api_token=api_token, exchange=exchange)
        if df_raw is None or df_raw.empty:
            logger.warning("No data returned for %s", day)
            continue

        df_raw["date"] = day.isoformat()
        df_norm = normalize_price_df(df_raw, fallback_date=day)
        if not df_norm.empty:
            rows.append(df_norm)
            try:
                df_raw.to_parquet(file_path, compression="snappy")
            except Exception as exc:
                logger.warning("Failed to cache parquet for %s: %s", day, exc)

    if not rows:
        raise RuntimeError(
            f"No price data found in {raw_daily_dir} for {start}..{end_with_buffer}"
        )

    df_all = pd.concat(rows, ignore_index=True)
    logger.info(
        "Loaded %d price rows across %d files",
        len(df_all),
        len(rows),
    )
    return df_all


def fetch_bulk_day(
    day: date, api_token: str, exchange: str = EODHD_EXCHANGE
) -> Optional[pd.DataFrame]:
    """Fetch a single day of EODHD bulk data as a dataframe."""
    url = f"https://eodhd.com/api/eod-bulk-last-day/{exchange}"
    params = {"date": day.isoformat(), "api_token": api_token, "fmt": "csv"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            logger.warning("Rate limited fetching %s; retry after backoff", day)
            return None
        if resp.status_code != 200:
            logger.warning(
                "EODHD returned %s for %s: %s", resp.status_code, day, resp.text[:200]
            )
            return None
        if not resp.text or len(resp.text.splitlines()) < 2:
            return None
        return pd.read_csv(StringIO(resp.text))
    except Exception as exc:
        logger.error("Failed to fetch %s: %s", day, exc)
        return None


def compute_next_day_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute next-day close-to-close returns (percent) per ticker.
    Drops rows where the next trading day price is missing.
    """
    price_df = price_df.copy()
    price_df.sort_values(["ticker", "date"], inplace=True)

    price_df["next_close"] = price_df.groupby("ticker")["close"].shift(-1)
    price_df["next_return"] = (price_df["next_close"] / price_df["close"] - 1) * 100
    price_df["next_date"] = price_df.groupby("ticker")["date"].shift(-1)

    price_df = price_df.dropna(subset=["next_return", "next_date"])
    price_df["next_return"] = price_df["next_return"].round(2)

    return price_df[["ticker", "date", "next_date", "next_return"]]


def load_news(ticker_file: Path) -> Tuple[Dict[str, List[dict]], Dict[str, DailyStats]]:
    """Load articles grouped by date and daily stats keyed by date."""
    data = json.loads(ticker_file.read_text())
    articles = data.get("articles", [])
    daily_summaries = data.get("daily_summaries", [])

    by_date: Dict[str, List[dict]] = defaultdict(list)
    for art in articles:
        dt = pd.to_datetime(art.get("date")).date().isoformat()
        by_date[dt].append(
            {
                "title": art.get("title"),
                "content": truncate_text(art.get("content", "")),
                "link": art.get("link"),
                "tags": art.get("tags"),
                "sentiment": art.get("sentiment"),
            }
        )

    stats: Dict[str, DailyStats] = {}
    for entry in daily_summaries:
        stats[entry["date"]] = DailyStats(
            num_articles=entry.get("num_articles", 0),
            sentiment_mean=entry.get("sentiment_mean"),
            sentiment_sd=entry.get("sentiment_sd"),
            summary=entry.get("summary"),
        )

    return by_date, stats


# -----------------------
# Dataset assembly
# -----------------------
def make_user_payload(
    ticker: str,
    dt: str,
    articles: List[dict],
    stats: Optional[DailyStats],
) -> dict:
    """
    Build the user payload. If a summary is available for the date, include the
    summary and omit the articles; otherwise include the articles list.
    """
    num_articles = stats.num_articles if stats else len(articles)

    if stats and stats.summary:
        payload = {
            "ticker": ticker,
            "date": dt,
            "num_articles": num_articles,
            "summary": stats.summary,
        }
    else:
        payload = {
            "ticker": ticker,
            "date": dt,
            "num_articles": num_articles,
            "articles": articles,
        }

    if stats:
        payload["daily_sentiment_mean"] = stats.sentiment_mean
        payload["daily_sentiment_sd"] = stats.sentiment_sd
    return payload


def build_examples(
    news_dir: Path,
    returns_df: pd.DataFrame,
    start: date,
    end: date,
    output_path: Path,
) -> Dict[str, int]:
    """
    Iterate through news files, align with returns, and write JSONL.
    Returns basic stats for logging.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Map (ticker, date) -> next_return
    ret_lookup = {
        (row.ticker, row.date.isoformat()): row.next_return
        for row in returns_df.itertuples()
    }

    total_examples = 0
    skipped_no_target = 0

    with output_path.open("w", encoding="utf-8") as fout:
        news_files = sorted(news_dir.glob("*_summarized.json"))
        for news_file in tqdm(news_files, desc="Building examples"):
            base = news_file.stem.replace("_summarized", "")
            ticker = f"{base}.US"
            articles_by_date, stats_by_date = load_news(news_file)

            for dt, articles in articles_by_date.items():
                dt_obj = date.fromisoformat(dt)
                if dt_obj < start or dt_obj > end:
                    continue

                target = ret_lookup.get((ticker, dt))
                if target is None:
                    skipped_no_target += 1
                    continue

                stats = stats_by_date.get(dt)
                payload = make_user_payload(ticker, dt, articles, stats)
                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                        {"role": "assistant", "content": f"{target:.2f}"},
                    ]
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_examples += 1

    return {
        "examples": total_examples,
        "skipped_no_target": skipped_no_target,
        "news_files": len(list(news_dir.glob('*_summarized.json'))),
    }


def preview_output(path: Path, n: int = 3) -> List[dict]:
    """Return the first n examples for quick inspection."""
    preview: List[dict] = []
    if not path.exists():
        return preview
    with path.open() as fin:
        for _ in range(n):
            line = fin.readline()
            if not line:
                break
            preview.append(json.loads(line))
    return preview


def main() -> None:
    logger.info("Building SFT dataset for %s .. %s", START_DATE, END_DATE)

    logger.info("Loading prices from %s", RAW_DAILY_DIR)
    prices = load_prices(RAW_DAILY_DIR, START_DATE, END_DATE)
    returns = compute_next_day_returns(prices)

    # Restrict to returns where t (news day) lies within the desired window
    returns = returns[
        (returns["date"] >= START_DATE) & (returns["date"] <= END_DATE)
    ]

    logger.info(
        "Computed %d next-day returns covering %d tickers",
        len(returns),
        returns["ticker"].nunique(),
    )

    stats = build_examples(NEWS_DIR, returns, START_DATE, END_DATE, OUTPUT_PATH)
    logger.info(
        "Wrote %d examples to %s (skipped %d with no target)",
        stats["examples"],
        OUTPUT_PATH,
        stats["skipped_no_target"],
    )

    sample = preview_output(OUTPUT_PATH, n=2)
    if sample:
        logger.info("Sample example 1: %s", json.dumps(sample[0], ensure_ascii=False)[:800])
        if len(sample) > 1:
            logger.info(
                "Sample example 2: %s",
                json.dumps(sample[1], ensure_ascii=False)[:800],
            )
    else:
        logger.warning("No samples to preview (file empty?)")


if __name__ == "__main__":
    main()

