import logging
from typing import Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class ParquetStore:
    """
    Parquet-based storage operations for time-series OHLCV data.
    Stores each symbol as a separate Parquet file.
    """
    
    def __init__(self, parquet_path: str):
        """
        Initialize Parquet storage.
        
        Args:
            parquet_path: Base directory path for storing Parquet files
        """
        self.parquet_path = Path(parquet_path)
        self.parquet_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized Parquet storage at {self.parquet_path}")
    
    def _get_symbol_path(self, symbol: str, library: str = 'stocks') -> Path:
        """
        Get file path for a symbol.
        
        Args:
            symbol: Symbol string (e.g., 'AAPL.US')
            library: Library/directory name (default: 'stocks')
            
        Returns:
            Path object for the symbol's Parquet file
        """
        # Sanitize symbol for filename (replace dots and special chars)
        safe_symbol = symbol.replace('.', '_').replace('/', '_')
        library_dir = self.parquet_path / library
        library_dir.mkdir(parents=True, exist_ok=True)
        return library_dir / f"{safe_symbol}.parquet"
    
    def write_symbol(self, symbol: str, df: pd.DataFrame, library: str = 'stocks'):
        """
        Write/append OHLCV data for a symbol.
        
        Args:
            symbol: Symbol string (e.g., 'AAPL.US')
            df: DataFrame with Date index and OHLCV columns
            library: Library/directory name (default: 'stocks')
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol}, skipping write")
            return
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                raise ValueError(f"DataFrame for {symbol} must have datetime index")
        
        # Sort by date
        df = df.sort_index()
        
        # Reset index to make date a column for Parquet
        df_write = df.reset_index()
        # Handle different index names - rename to 'date' for consistency
        first_col = df_write.columns[0]
        if first_col in ['index', 'Date', 'DATE'] or first_col.lower() == 'date':
            df_write.rename(columns={first_col: 'date'}, inplace=True)
        elif first_col != 'date':
            # If first column is not already 'date', rename it
            df_write.rename(columns={first_col: 'date'}, inplace=True)
        
        symbol_path = self._get_symbol_path(symbol, library)
        
        # Check if file exists
        if symbol_path.exists():
            # Read existing data
            try:
                existing_df = pd.read_parquet(symbol_path)
                if 'date' in existing_df.columns:
                    existing_df['date'] = pd.to_datetime(existing_df['date'])
                    existing_df.set_index('date', inplace=True)
                
                # Combine and deduplicate
                combined = pd.concat([existing_df, df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                
                # Reset index for writing
                combined_write = combined.reset_index()
                if combined_write.columns[0] == 'index':
                    combined_write.rename(columns={'index': 'date'}, inplace=True)
                
                logger.debug(f"Appending {len(df)} rows to existing {symbol} (total: {len(combined)})")
                combined_write.to_parquet(symbol_path, index=False, engine='pyarrow')
            except Exception as e:
                logger.warning(f"Error reading existing data for {symbol}, overwriting: {e}")
                df_write.to_parquet(symbol_path, index=False, engine='pyarrow')
        else:
            # Write new file
            logger.debug(f"Writing {len(df)} rows for new symbol {symbol}")
            df_write.to_parquet(symbol_path, index=False, engine='pyarrow')
        
        logger.info(f"Stored {len(df)} rows for {symbol}")
    
    def read_symbol(self, symbol: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, library: str = 'stocks') -> Optional[pd.DataFrame]:
        """
        Read data for a symbol with optional date filtering.
        
        Args:
            symbol: Symbol string
            start_date: Start date (YYYY-MM-DD) or None
            end_date: End date (YYYY-MM-DD) or None
            library: Library/directory name
            
        Returns:
            DataFrame with datetime index or None if symbol not found
        """
        symbol_path = self._get_symbol_path(symbol, library)
        
        if not symbol_path.exists():
            logger.warning(f"Symbol {symbol} not found at {symbol_path}")
            return None
        
        try:
            df = pd.read_parquet(symbol_path)
            
            # Debug: log column names and index info
            logger.debug(f"Reading {symbol}: columns={df.columns.tolist()}, index_name={df.index.name}, index_type={type(df.index)}")
            
            # Try to find date column - check multiple possible names
            date_col = None
            for col_name in ['date', 'Date', 'DATE', 'index', 'Date', '__index_level_0__']:
                if col_name in df.columns:
                    date_col = col_name
                    break
            
            # Also check if any column name contains 'date' (case insensitive)
            if date_col is None:
                for col_name in df.columns:
                    if 'date' in col_name.lower():
                        date_col = col_name
                        break
            
            # Set date as index
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            elif isinstance(df.index, pd.DatetimeIndex):
                # Already has datetime index
                pass
            elif df.index.name in ['date', 'Date', 'DATE']:
                # Index is named date but might not be datetime - try to convert
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    logger.warning(f"Could not convert index to datetime for {symbol}")
                    return None
            else:
                # Check if first column looks like dates
                first_col = df.columns[0]
                try:
                    # Try to parse first column as dates
                    test_dates = pd.to_datetime(df[first_col].head(5))
                    if test_dates.notna().all():
                        # First column appears to be dates
                        df[first_col] = pd.to_datetime(df[first_col])
                        df.set_index(first_col, inplace=True)
                        logger.debug(f"Using first column '{first_col}' as date index for {symbol}")
                    else:
                        logger.warning(f"Could not find date column/index in {symbol}. Columns: {df.columns.tolist()}, Index: {df.index.name}")
                        logger.info(f"File structure - Shape: {df.shape}, Dtypes: {df.dtypes.to_dict()}")
                        return None
                except Exception as parse_err:
                    logger.warning(f"Could not parse first column as dates for {symbol}: {parse_err}")
                    logger.warning(f"Columns: {df.columns.tolist()}, Index: {df.index.name}")
                    return None
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning(f"Index is not datetime for {symbol} after processing")
                return None
            
            # Apply date filtering if requested
            if start_date or end_date:
                if start_date:
                    start = pd.to_datetime(start_date)
                    df = df[df.index >= start]
                if end_date:
                    end = pd.to_datetime(end_date)
                    df = df[df.index <= end]
            
            return df
        except Exception as e:
            logger.error(f"Error reading {symbol}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def get_date_range(self, symbol: str, library: str = 'stocks') -> Optional[Tuple[datetime, datetime]]:
        """
        Get min/max dates for a symbol.
        
        Args:
            symbol: Symbol string
            library: Library/directory name
            
        Returns:
            Tuple of (min_date, max_date) or None if not found
        """
        df = self.read_symbol(symbol, library=library)
        if df is None or df.empty:
            return None
        
        return (df.index.min(), df.index.max())
    
    def get_symbol_metadata(self, symbol: str, library: str = 'stocks') -> Optional[dict]:
        """
        Get metadata for a symbol (date range, row count, last update).
        
        Args:
            symbol: Symbol string
            library: Library/directory name
            
        Returns:
            Dictionary with metadata or None
        """
        date_range = self.get_date_range(symbol, library)
        if date_range is None:
            return None
        
        df = self.read_symbol(symbol, library=library)
        symbol_path = self._get_symbol_path(symbol, library)
        
        # Get file modification time
        last_update = datetime.fromtimestamp(symbol_path.stat().st_mtime) if symbol_path.exists() else datetime.now()
        
        return {
            'symbol': symbol,
            'min_date': date_range[0],
            'max_date': date_range[1],
            'row_count': len(df) if df is not None else 0,
            'last_update': last_update
        }
    
    def list_symbols(self, library: str = 'stocks') -> List[str]:
        """
        List all symbols in a library.
        
        Args:
            library: Library/directory name
            
        Returns:
            List of symbol strings (with dots restored from underscores)
        """
        library_dir = self.parquet_path / library
        if not library_dir.exists():
            return []
        
        symbols = []
        for parquet_file in library_dir.glob("*.parquet"):
            # Restore symbol name (replace underscore with dot, remove .parquet)
            symbol = parquet_file.stem.replace('_', '.')
            symbols.append(symbol)
        
        return sorted(symbols)
    
    def symbol_exists(self, symbol: str, library: str = 'stocks') -> bool:
        """Check if symbol exists in library."""
        symbol_path = self._get_symbol_path(symbol, library)
        return symbol_path.exists()

