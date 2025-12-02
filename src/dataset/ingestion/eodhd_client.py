import requests
import time
import logging
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class EODHDClient:
    """
    EODHD API client with rate limiting, retry logic, and error handling.
    """
    
    BASE_URL = "https://eodhd.com/api"
    
    def __init__(self, api_token: str, rate_limit_delay: float = 1.0, max_retries: int = 3, retry_backoff: float = 2.0):
        """
        Initialize EODHD API client.
        
        Args:
            api_token: EODHD API token
            rate_limit_delay: Seconds to wait between requests
            max_retries: Maximum number of retries for failed requests
            retry_backoff: Exponential backoff multiplier
        """
        self.api_token = api_token
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Make HTTP request with retry logic.
        
        Args:
            endpoint: API endpoint (e.g., 'eod', 'exchange-symbol-list')
            params: Query parameters
            
        Returns:
            Response JSON or None if failed
        """
        params['api_token'] = self.api_token
        params['fmt'] = 'json'
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Check for API-level errors in response
                        if isinstance(data, dict):
                            if 'error' in data:
                                logger.error(f"API error: {data.get('error')}")
                                logger.debug(f"Full error response: {data}")
                                return None
                            # Some APIs return error messages in different fields
                            if 'message' in data and 'error' in str(data.get('message', '')).lower():
                                logger.error(f"API error message: {data.get('message')}")
                                return None
                        return data
                    except ValueError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        logger.debug(f"Response text: {response.text[:500]}")
                        return None
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = self.retry_backoff ** attempt
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                elif response.status_code == 404:
                    # 404 typically means ticker not found - don't retry, just return None
                    # Check if it's a "Ticker Not Found" message (expected for invalid symbols)
                    response_text = response.text[:200].lower()
                    if 'ticker not found' in response_text or 'not found' in response_text:
                        logger.debug(f"Ticker not found for endpoint {endpoint} (this is expected for some symbols)")
                        return None
                    else:
                        # Unexpected 404 - log it
                        logger.error(f"HTTP 404 for endpoint {endpoint}")
                        logger.error(f"Response: {response.text[:500]}")
                        return None
                else:
                    logger.error(f"HTTP {response.status_code} for endpoint {endpoint}")
                    logger.error(f"Response: {response.text[:500]}")
                    logger.debug(f"Request URL: {url}")
                    logger.debug(f"Request params: {dict((k, v) for k, v in params.items() if k != 'api_token')}")
                    if response.status_code >= 500:
                        # Server error - retry
                        wait_time = self.retry_backoff ** attempt
                        time.sleep(wait_time)
                        continue
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_backoff ** attempt
                    time.sleep(wait_time)
                else:
                    return None
        
        return None
    
    def get_exchange_symbols(self, exchange: str) -> List[str]:
        """
        Fetch all symbols for an exchange.
        
        Args:
            exchange: Exchange code (e.g., 'NASDAQ', 'NYSE', 'US')
            
        Returns:
            List of symbol strings (e.g., ['AAPL.US', 'MSFT.US'])
        """
        logger.info(f"Fetching symbols for exchange: {exchange}")
        # Endpoint format: exchange-symbol-list/{EXCHANGE}
        endpoint = f'exchange-symbol-list/{exchange}'
        params = {}
        data = self._make_request(endpoint, params)
        
        if not data:
            logger.error(f"Failed to fetch symbols for {exchange} - no data returned")
            return []
        
        # Log response structure for debugging
        logger.debug(f"Response type: {type(data)}, length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
        if isinstance(data, dict):
            logger.debug(f"Response keys: {list(data.keys())[:10]}")
            # Check if data is nested
            if 'data' in data:
                data = data['data']
            elif 'symbols' in data:
                data = data['symbols']
        
        symbols = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Log first item structure for debugging
                    if len(symbols) == 0:
                        logger.debug(f"Sample response item keys: {list(item.keys())[:10]}")
                    
                    # Try different possible field names
                    code = item.get('Code') or item.get('code') or item.get('Symbol') or item.get('symbol') or item.get('ticker')
                    exch = item.get('Exchange') or item.get('exchange') or exchange
                    
                    if code:
                        # Format symbol properly
                        if '.' in code:
                            symbol = code
                        else:
                            symbol = f"{code}.{exch}"
                        symbols.append(symbol)
                elif isinstance(item, str):
                    symbols.append(item)
        elif isinstance(data, dict):
            logger.warning(f"Unexpected response format (dict): {list(data.keys())[:10]}")
        
        logger.info(f"Found {len(symbols)} symbols for {exchange}")
        if len(symbols) > 0:
            logger.debug(f"Sample symbols: {symbols[:5]}")
        
        return symbols
    
    def get_ohlcv(self, symbol: str, start_date: str, end_date: str, period: str = 'd') -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol and date range.
        
        Args:
            symbol: Symbol string (e.g., 'AAPL.US')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period ('d' for daily, 'w' for weekly, 'm' for monthly)
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
            Returns None if fetch failed
        """
        logger.debug(f"Fetching OHLCV for {symbol} from {start_date} to {end_date}")
        
        # Endpoint format: eod/{SYMBOL}
        endpoint = f'eod/{symbol}'
        params = {
            'from': start_date,
            'to': end_date,
            'period': period
        }
        
        data = self._make_request(endpoint, params)
        
        if not data:
            logger.warning(f"No data returned for {symbol}")
            return None
        
        if len(data) == 0:
            logger.debug(f"Empty data for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Standardize column names
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        col_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume',
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'volume': 'volume'
        }
        
        df.rename(columns=col_mapping, inplace=True)
        
        # Check for required columns
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"Missing columns for {symbol}: {missing}")
            return None
        
        # Select and order columns
        df = df[required_cols]
        df.columns = [c.capitalize() for c in required_cols]
        
        logger.debug(f"Fetched {len(df)} rows for {symbol}")
        return df
    
    def get_bulk_ohlcv(self, symbols: List[str], start_date: str, end_date: str, period: str = 'd') -> Dict[str, pd.DataFrame]:
        """
        Batch fetch OHLCV data for multiple symbols with rate limiting.
        
        Note: This method is currently unused but kept for potential future use.
        
        Args:
            symbols: List of symbol strings
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period ('d' for daily)
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        for symbol in symbols:
            df = self.get_ohlcv(symbol, start_date, end_date, period)
            if df is not None:
                results[symbol] = df
        return results

