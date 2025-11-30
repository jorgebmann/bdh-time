import pandas as pd
import requests
import os
import time
from io import StringIO
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

# CONFIGURATION
API_TOKEN = "692b5170c9fb15.77141019"
EXCHANGE = "US"  # 'US' usually covers NYSE, NASDAQ, AMEX in EODHD bulk.
# If 'US' fails, loop through ['NYSE', 'NASDAQ']
START_DATE = "1995-01-01"
END_DATE = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = "data/raw_daily"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_bulk_data(date_str, api_token, exchange):
    """
    Fetches the bulk CSV for a specific exchange and date.
    EODHD Endpoint: https://eodhd.com/api/eod-bulk-last-day/{Exchange}?date={Date}&api_token={Token}&fmt=csv
    """
    url = f"https://eodhd.com/api/eod-bulk-last-day/{exchange}"
    params = {
        'date': date_str,
        'api_token': api_token,
        'fmt': 'csv'
    }

    try:
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            # Check if empty (non-trading day)
            if not response.text or len(response.text.splitlines()) < 2:
                return None
            return response.text
        elif response.status_code == 429:
            logger.warning(f"Rate limited on {date_str}. Waiting...")
            time.sleep(5)
            return get_bulk_data(date_str, api_token, exchange)  # Retry
        else:
            logger.error(f"Error {response.status_code} on {date_str}: {response.reason}")
            return None
    except Exception as e:
        logger.error(f"Exception on {date_str}: {e}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate business days (skips weekends, but not holidays - API will just return empty)
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B')

    logger.info(f"Starting bulk ingestion for {len(dates)} days...")

    for date_obj in tqdm(dates):
        date_str = date_obj.strftime('%Y-%m-%d')
        file_path = os.path.join(OUTPUT_DIR, f"us_market_{date_str}.parquet")

        # Skip if already downloaded
        if os.path.exists(file_path):
            continue

        csv_data = get_bulk_data(date_str, API_TOKEN, EXCHANGE)

        if csv_data:
            # Convert CSV string to DataFrame
            try:
                df = pd.read_csv(StringIO(csv_data))

                # Basic cleaning
                df['date'] = date_str
                # Save as Parquet (much smaller than CSV)
                df.to_parquet(file_path, compression='snappy')

            except Exception as parse_error:
                logger.error(f"Failed to parse CSV for {date_str}: {parse_error}")
        else:
            # Likely a holiday
            pass


if __name__ == "__main__":
    main()