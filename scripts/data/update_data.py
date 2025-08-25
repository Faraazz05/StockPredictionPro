"""
data/update_data.py

Automated incremental market data update script for StockPredictionPro.
Downloads new data since last update, merges with existing files, and maintains data integrity.
Integrates with MarketDataDownloader; works safely for production environments.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Import the data downloader
sys.path.append(str(Path(__file__).parent))
from download_data import MarketDataDownloader

RAW_DATA_DIR = Path("./data/raw")
PROCESSED_DATA_DIR = Path("./data/processed")
UPDATE_LOG = Path("./logs/update_data.log")

# Logging configuration
logging.basicConfig(
    filename=str(UPDATE_LOG),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger('StockPredictionPro.DataUpdate')

def get_latest_timestamp(symbol: str, data_dir: Path) -> datetime:
    """
    Get the latest timestamp available for a symbol in processed data.
    Returns default (2 years ago) if no file exists.
    """
    files = list(data_dir.glob(f"{symbol}_clean*.csv"))
    if not files:
        return datetime.utcnow() - timedelta(days=730)
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    df = pd.read_csv(latest_file)
    if 'timestamp' in df.columns:
        return pd.to_datetime(df['timestamp']).max()
    elif 'date' in df.columns:
        return pd.to_datetime(df['date']).max()
    else:
        return datetime.utcnow() - timedelta(days=730)

def load_existing_data(symbol: str, data_dir: Path) -> pd.DataFrame:
    """
    Load existing cleaned data for a symbol from the latest file.
    Returns empty DataFrame if not found.
    """
    files = list(data_dir.glob(f"{symbol}_clean*.csv"))
    if not files:
        return pd.DataFrame()
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return pd.read_csv(latest_file)

def merge_and_deduplicate(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new and old data, drop duplicates by symbol/timestamp, and sort.
    """
    merged = pd.concat([existing, new])
    if 'symbol' in merged.columns and 'timestamp' in merged.columns:
        merged = merged.drop_duplicates(subset=['symbol', 'timestamp'])
        merged = merged.sort_values(['symbol', 'timestamp'])
    merged = merged.reset_index(drop=True)
    return merged

def update_symbol(symbol: str, source: str = 'alpha_vantage'):
    """
    Main routine to update a given symbol's data incrementally.
    """
    logger.info(f"=== Updating symbol: {symbol} ===")
    existing_data = load_existing_data(symbol, PROCESSED_DATA_DIR)
    last_update = get_latest_timestamp(symbol, PROCESSED_DATA_DIR)
    today = datetime.utcnow()

    # Only fetch new dates
    fetch_start = last_update + timedelta(days=1)
    fetch_end = today
    if fetch_start > fetch_end:
        logger.info(f"{symbol}: Data already up to date (last: {last_update.date()})")
        return

    downloader = MarketDataDownloader(symbol=symbol, source=source)
    new_data = downloader.fetch(fetch_start, fetch_end)
    if new_data is None or new_data.empty:
        logger.warning(f"No new data fetched for {symbol}")
        return

    # Merge and deduplicate
    merged_data = merge_and_deduplicate(existing_data, new_data)
    out_path = PROCESSED_DATA_DIR / f"{symbol}_clean_updated_{today.strftime('%Y%m%d')}.csv"
    merged_data.to_csv(out_path, index=False)
    logger.info(f"{symbol}: Updated data saved to {out_path} [{len(merged_data)} rows]")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Incrementally update market data for StockPredictionPro")
    parser.add_argument('--symbols', nargs='+', required=True, help='Stock symbols (space-separated)')
    parser.add_argument('--source', default='alpha_vantage', choices=['alpha_vantage', 'polygon_io', 'yahoo_finance'], help='Data source')
    args = parser.parse_args()

    logger.info(f"Starting updates for symbols: {args.symbols} from source: {args.source}")
    for symbol in args.symbols:
        try:
            update_symbol(symbol, source=args.source)
        except Exception as e:
            logger.error(f"Error updating symbol {symbol}: {e}")

    logger.info("All symbol updates completed.")

if __name__ == '__main__':
    main()
