import logging
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import pandas as pd

from .eodhd_client import EODHDClient
from .parquet_store import ParquetStore

logger = logging.getLogger(__name__)

class DateIngestionManager:
    """
    Manages date-range based ingestion with progress tracking and chunking.
    """
    
    def __init__(self, eodhd_client: EODHDClient, parquet_store: ParquetStore):
        """
        Initialize ingestion manager.
        
        Args:
            eodhd_client: EODHD API client
            parquet_store: Parquet storage instance
        """
        self.eodhd_client = eodhd_client
        self.parquet_store = parquet_store
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime."""
        if isinstance(date_str, datetime):
            return date_str
        return parse_date(date_str)
    
    def _generate_date_chunks(self, start_date: str, end_date: str, chunk_days: int = 90) -> List[Tuple[str, str]]:
        """
        Generate date range chunks.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunk_days: Days per chunk
            
        Returns:
            List of (start, end) date tuples
        """
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        
        chunks = []
        current = start
        
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            chunks.append((
                current.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d')
            ))
            current = chunk_end + timedelta(days=1)
        
        return chunks
    
    def get_ingestion_status(self, symbol: str, library: str = 'stocks') -> Optional[dict]:
        """
        Check what dates are already ingested for a symbol.
        
        Args:
            symbol: Symbol string
            library: Library name
            
        Returns:
            Dictionary with min_date, max_date, row_count or None if not found
        """
        return self.parquet_store.get_symbol_metadata(symbol, library)
    
    def _get_missing_date_ranges(self, symbol: str, start_date: str, end_date: str, 
                                 library: str = 'stocks') -> List[Tuple[str, str]]:
        """
        Determine missing date ranges for a symbol.
        
        Args:
            symbol: Symbol string
            start_date: Desired start date
            end_date: Desired end date
            library: Library name
            
        Returns:
            List of (start, end) date tuples that need to be fetched
        """
        status = self.get_ingestion_status(symbol, library)
        
        if status is None:
            # No existing data - fetch entire range
            return [(start_date, end_date)]
        
        existing_start = status['min_date']
        existing_end = status['max_date']
        
        desired_start = self._parse_date(start_date)
        desired_end = self._parse_date(end_date)
        
        missing_ranges = []
        
        # Check for gap before existing data
        if desired_start < existing_start:
            gap_end = min(existing_start - timedelta(days=1), desired_end)
            missing_ranges.append((
                desired_start.strftime('%Y-%m-%d'),
                gap_end.strftime('%Y-%m-%d')
            ))
        
        # Check for gap after existing data
        if desired_end > existing_end:
            gap_start = max(existing_end + timedelta(days=1), desired_start)
            missing_ranges.append((
                gap_start.strftime('%Y-%m-%d'),
                desired_end.strftime('%Y-%m-%d')
            ))
        
        # If no gaps, check if we need to extend
        if not missing_ranges:
            if desired_start < existing_start or desired_end > existing_end:
                # Need to extend range
                new_start = min(desired_start, existing_start).strftime('%Y-%m-%d')
                new_end = max(desired_end, existing_end).strftime('%Y-%m-%d')
                missing_ranges.append((new_start, new_end))
        
        return missing_ranges if missing_ranges else []
    
    def ingest_symbol(self, symbol: str, start_date: str, end_date: str, 
                     chunk_days: int = 90, library: str = 'stocks', 
                     resume: bool = True) -> dict:
        """
        Ingest a single symbol in date chunks.
        
        Args:
            symbol: Symbol string
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunk_days: Days per chunk
            library: Library name
            resume: If True, skip already-ingested date ranges
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting ingestion for {symbol} from {start_date} to {end_date}")
        
        stats = {
            'symbol': symbol,
            'chunks_processed': 0,
            'chunks_skipped': 0,
            'rows_added': 0,
            'errors': 0
        }
        
        # Determine date ranges to fetch
        if resume:
            date_ranges = self._get_missing_date_ranges(symbol, start_date, end_date, library)
            if not date_ranges:
                logger.info(f"{symbol}: All data already ingested, skipping")
                stats['chunks_skipped'] = 1
                return stats
        else:
            date_ranges = [(start_date, end_date)]
        
        # Process each date range in chunks
        for range_start, range_end in date_ranges:
            chunks = self._generate_date_chunks(range_start, range_end, chunk_days)
            
            for chunk_start, chunk_end in chunks:
                try:
                    # Check if chunk already exists (if resuming)
                    if resume:
                        existing = self.parquet_store.read_symbol(symbol, chunk_start, chunk_end, library)
                        if existing is not None and len(existing) > 0:
                            # Check if chunk is complete
                            chunk_start_dt = self._parse_date(chunk_start)
                            chunk_end_dt = self._parse_date(chunk_end)
                            if len(existing) >= (chunk_end_dt - chunk_start_dt).days * 0.8:  # 80% threshold
                                logger.debug(f"{symbol}: Chunk {chunk_start} to {chunk_end} already exists, skipping")
                                stats['chunks_skipped'] += 1
                                continue
                    
                    # Fetch data
                    logger.debug(f"{symbol}: Fetching chunk {chunk_start} to {chunk_end}")
                    df = self.eodhd_client.get_ohlcv(symbol, chunk_start, chunk_end)
                    
                    if df is None or df.empty:
                        # Don't log warnings for empty data - it's expected for many symbols
                        logger.debug(f"{symbol}: No data for chunk {chunk_start} to {chunk_end}")
                        stats['errors'] += 1
                        continue
                    
                    # Store data
                    self.parquet_store.write_symbol(symbol, df, library)
                    stats['chunks_processed'] += 1
                    stats['rows_added'] += len(df)
                    
                except Exception as e:
                    logger.error(f"{symbol}: Error processing chunk {chunk_start} to {chunk_end}: {e}")
                    stats['errors'] += 1
        
        logger.info(f"{symbol}: Ingestion complete. Processed {stats['chunks_processed']} chunks, "
                   f"added {stats['rows_added']} rows")
        return stats
    
    def ingest_exchange(self, exchange: str, start_date: str, end_date: str,
                       chunk_days: int = 90, library: str = 'stocks',
                       resume: bool = True) -> List[dict]:
        """
        Ingest all symbols from an exchange.
        
        Args:
            exchange: Exchange code (e.g., 'NASDAQ')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunk_days: Days per chunk
            library: Library name
            resume: Resume from last checkpoint
            
        Returns:
            List of ingestion statistics dictionaries
        """
        logger.info(f"Starting exchange ingestion for {exchange}")
        
        # Fetch symbol list
        symbols = self.eodhd_client.get_exchange_symbols(exchange)
        
        if not symbols:
            logger.error(f"No symbols found for exchange {exchange}")
            return []
        
        logger.info(f"Found {len(symbols)} symbols for {exchange}")
        
        results = []
        for symbol in symbols:
            try:
                stats = self.ingest_symbol(symbol, start_date, end_date, chunk_days, library, resume)
                results.append(stats)
            except Exception as e:
                logger.error(f"Failed to ingest {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'chunks_processed': 0,
                    'chunks_skipped': 0,
                    'rows_added': 0,
                    'errors': 1
                })
        
        return results

