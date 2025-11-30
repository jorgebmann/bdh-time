import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset.ingestion import EODHDClient, ParquetStore, DateIngestionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Ingest stock OHLCV data from EODHD API to Parquet storage')
    parser.add_argument('--exchanges', type=str, help='Comma-separated list of exchanges (e.g., "NASDAQ,NYSE")')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of specific symbols (overrides exchange)')
    parser.add_argument('--chunk-days', type=int, default=90, help='Days per chunk (default: 90)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--library', type=str, default='stocks', help='Parquet library/directory name (default: stocks)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set end date to today if not provided
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Initialize clients
    api_token = config['EODHD']['api_token']
    ingestion_config = config.get('Ingestion', {})
    
    eodhd_client = EODHDClient(
        api_token=api_token,
        rate_limit_delay=ingestion_config.get('rate_limit_delay', 1.0),
        max_retries=ingestion_config.get('max_retries', 3),
        retry_backoff=ingestion_config.get('retry_backoff', 2.0)
    )
    
    parquet_path = ingestion_config.get('parquet_path', 'data/parquet')
    parquet_store = ParquetStore(parquet_path)
    
    ingestion_manager = DateIngestionManager(eodhd_client, parquet_store)
    
    # Determine symbols to ingest
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        logger.info(f"Ingesting {len(symbols)} specified symbols")
    elif args.exchanges:
        exchanges = [e.strip() for e in args.exchanges.split(',')]
        symbols = []
        for exchange in exchanges:
            exchange_symbols = eodhd_client.get_exchange_symbols(exchange)
            symbols.extend(exchange_symbols)
        logger.info(f"Ingesting {len(symbols)} symbols from exchanges: {exchanges}")
    else:
        # Use default exchanges from config
        default_exchanges = ingestion_config.get('default_exchanges', ['NASDAQ', 'NYSE'])
        symbols = []
        for exchange in default_exchanges:
            exchange_symbols = eodhd_client.get_exchange_symbols(exchange)
            symbols.extend(exchange_symbols)
        logger.info(f"Ingesting {len(symbols)} symbols from default exchanges: {default_exchanges}")
    
    if not symbols:
        logger.error("No symbols to ingest")
        return
    
    # Ingest symbols
    logger.info(f"Starting ingestion from {args.start_date} to {args.end_date}")
    logger.info(f"Chunk size: {args.chunk_days} days, Library: {args.library}, Resume: {args.resume}")
    
    results = []
    for symbol in tqdm(symbols, desc="Ingesting symbols"):
        try:
            stats = ingestion_manager.ingest_symbol(
                symbol=symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                chunk_days=args.chunk_days,
                library=args.library,
                resume=args.resume
            )
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
    
    # Generate summary report
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    
    total_chunks = sum(r['chunks_processed'] for r in results)
    total_skipped = sum(r['chunks_skipped'] for r in results)
    total_rows = sum(r['rows_added'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    successful = sum(1 for r in results if r['rows_added'] > 0)
    
    print(f"Total symbols processed: {len(results)}")
    print(f"Successful ingestions: {successful}")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Total chunks skipped: {total_skipped}")
    print(f"Total rows added: {total_rows:,}")
    print(f"Total errors: {total_errors}")
    
    if total_errors > 0:
        print("\nSymbols with errors:")
        for r in results:
            if r['errors'] > 0:
                print(f"  - {r['symbol']}: {r['errors']} errors")
    
    print("="*60)

if __name__ == "__main__":
    main()

