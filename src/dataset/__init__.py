# Dataset package for data ingestion and preprocessing

from .preprocess import process_market_data
from .ingestion import EODHDClient, ParquetStore, DateIngestionManager

__all__ = ['process_market_data', 'EODHDClient', 'ParquetStore', 'DateIngestionManager']
