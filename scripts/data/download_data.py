"""
data/download_data.py

Advanced market data downloader for StockPredictionPro.
Supports multiple data sources, rate limiting, error handling, and parallel downloads.
Integrates with database storage and caching systems.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import time
import json
import logging
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockPredictionPro.DataDownloader')

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    name: str
    base_url: str
    api_key_env: str
    rate_limit_per_minute: int
    max_requests_per_day: int
    supported_intervals: List[str]
    free_tier_limit: int

# Data source configurations
DATA_SOURCES = {
    'alpha_vantage': DataSourceConfig(
        name='Alpha Vantage',
        base_url='https://www.alphavantage.co/query',
        api_key_env='ALPHA_VANTAGE_API_KEY',
        rate_limit_per_minute=5,
        max_requests_per_day=500,
        supported_intervals=['1min', '5min', '15min', '30min', '60min', '1day'],
        free_tier_limit=500
    ),
    'polygon': DataSourceConfig(
        name='Polygon.io',
        base_url='https://api.polygon.io/v2',
        api_key_env='POLYGON_API_KEY',
        rate_limit_per_minute=10,
        max_requests_per_day=1000,
        supported_intervals=['1min', '5min', '15min', '30min', '1hour', '1day'],
        free_tier_limit=1000
    ),
    'yahoo_finance': DataSourceConfig(
        name='Yahoo Finance',
        base_url='https://query1.finance.yahoo.com/v8/finance/chart',
        api_key_env='',  # No API key required
        rate_limit_per_minute=60,
        max_requests_per_day=10000,
        supported_intervals=['1m', '5m', '15m', '30m', '1h', '1d'],
        free_tier_limit=10000
    )
}

@dataclass
class MarketDataRecord:
    """Standardized market data record"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None
    split_coefficient: Optional[float] = None
    dividend_amount: Optional[float] = None

class RateLimiter:
    """Rate limiting utility"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def acquire(self) -> bool:
        """Check if request can be made within rate limit"""
        now = time.time()
        
        # Remove old requests outside time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def wait_time(self) -> float:
        """Calculate wait time until next request can be made"""
        if len(self.requests) < self.max_requests:
            return 0
        
        now = time.time()
        oldest_request = min(self.requests)
        return max(0, self.time_window - (now - oldest_request))

# ============================================
# DATA SOURCE IMPLEMENTATIONS
# ============================================

class BaseDataSource:
    """Base class for all data sources"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.api_key = os.getenv(config.api_key_env, '')
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)
        self.session = self._create_session()
        self.cache_dir = Path('./data/cache') / config.name.lower().replace(' ', '_')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _wait_for_rate_limit(self) -> None:
        """Wait if rate limit is exceeded"""
        while not self.rate_limiter.acquire():
            wait_time = self.rate_limiter.wait_time()
            if wait_time > 0:
                logger.info(f"Rate limit reached for {self.config.name}. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time + 1)
    
    def fetch_data(self, symbol: str, start_date: datetime, 
                  end_date: datetime, interval: str = '1day') -> List[MarketDataRecord]:
        """Fetch data for given symbol and date range"""
        raise NotImplementedError("Subclasses must implement fetch_data method")

class AlphaVantageSource(BaseDataSource):
    """Alpha Vantage data source implementation"""
    
    def fetch_data(self, symbol: str, start_date: datetime, 
                  end_date: datetime, interval: str = '1day') -> List[MarketDataRecord]:
        """Fetch data from Alpha Vantage API"""
        
        if not self.api_key:
            logger.error("Alpha Vantage API key not found in environment variables")
            return []
        
        self._wait_for_rate_limit()
        
        # Map interval to Alpha Vantage function
        function_map = {
            '1day': 'TIME_SERIES_DAILY_ADJUSTED',
            '1min': 'TIME_SERIES_INTRADAY',
            '5min': 'TIME_SERIES_INTRADAY',
            '15min': 'TIME_SERIES_INTRADAY',
            '30min': 'TIME_SERIES_INTRADAY',
            '60min': 'TIME_SERIES_INTRADAY'
        }
        
        function = function_map.get(interval, 'TIME_SERIES_DAILY_ADJUSTED')
        
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full',
            'datatype': 'json'
        }
        
        # Add interval parameter for intraday data
        if interval != '1day':
            params['interval'] = interval
        
        try:
            response = self.session.get(self.config.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return []
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage API note: {data['Note']}")
                return []
            
            return self._parse_alpha_vantage_response(data, symbol, start_date, end_date, interval)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data from Alpha Vantage: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing Alpha Vantage response: {e}")
            return []
    
    def _parse_alpha_vantage_response(self, data: Dict, symbol: str, 
                                    start_date: datetime, end_date: datetime, 
                                    interval: str) -> List[MarketDataRecord]:
        """Parse Alpha Vantage API response"""
        records = []
        
        # Get the time series data key
        if interval == '1day':
            time_series_key = 'Time Series (Daily)'
        else:
            time_series_key = f'Time Series ({interval})'
        
        time_series = data.get(time_series_key, {})
        
        if not time_series:
            logger.warning(f"No time series data found in Alpha Vantage response for {symbol}")
            return []
        
        for date_str, day_data in time_series.items():
            try:
                # Parse timestamp
                if interval == '1day':
                    timestamp = datetime.strptime(date_str, '%Y-%m-%d')
                else:
                    timestamp = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                
                # Filter by date range
                if timestamp < start_date or timestamp > end_date:
                    continue
                
                # Create record
                record = MarketDataRecord(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(day_data['1. open']),
                    high=float(day_data['2. high']),
                    low=float(day_data['3. low']),
                    close=float(day_data['4. close']),
                    volume=int(day_data['5. volume']),
                    adjusted_close=float(day_data.get('5. adjusted close', day_data['4. close'])),
                    split_coefficient=float(day_data.get('8. split coefficient', 1.0)),
                    dividend_amount=float(day_data.get('7. dividend amount', 0.0))
                )
                
                records.append(record)
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parsing Alpha Vantage record for {date_str}: {e}")
                continue
        
        logger.info(f"Parsed {len(records)} records from Alpha Vantage for {symbol}")
        return records

class PolygonSource(BaseDataSource):
    """Polygon.io data source implementation"""
    
    def fetch_data(self, symbol: str, start_date: datetime, 
                  end_date: datetime, interval: str = '1day') -> List[MarketDataRecord]:
        """Fetch data from Polygon.io API"""
        
        if not self.api_key:
            logger.error("Polygon.io API key not found in environment variables")
            return []
        
        self._wait_for_rate_limit()
        
        # Map interval to Polygon format
        interval_map = {
            '1min': '1/minute',
            '5min': '5/minute',
            '15min': '15/minute',
            '30min': '30/minute',
            '1hour': '1/hour',
            '1day': '1/day'
        }
        
        polygon_interval = interval_map.get(interval, '1/day')
        
        # Build URL
        url = f"{self.config.base_url}/aggs/ticker/{symbol}/range/{polygon_interval}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        params = {
            'apiKey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if data.get('status') != 'OK':
                logger.error(f"Polygon.io API error: {data.get('error', 'Unknown error')}")
                return []
            
            return self._parse_polygon_response(data, symbol)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data from Polygon.io: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing Polygon.io response: {e}")
            return []
    
    def _parse_polygon_response(self, data: Dict, symbol: str) -> List[MarketDataRecord]:
        """Parse Polygon.io API response"""
        records = []
        
        results = data.get('results', [])
        
        if not results:
            logger.warning(f"No results found in Polygon.io response for {symbol}")
            return []
        
        for item in results:
            try:
                # Parse timestamp (Polygon returns Unix timestamp in milliseconds)
                timestamp = datetime.fromtimestamp(item['t'] / 1000)
                
                # Create record
                record = MarketDataRecord(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(item['o']),
                    high=float(item['h']),
                    low=float(item['l']),
                    close=float(item['c']),
                    volume=int(item['v']),
                    adjusted_close=float(item.get('c', item['c']))  # Polygon returns adjusted by default
                )
                
                records.append(record)
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parsing Polygon.io record: {e}")
                continue
        
        logger.info(f"Parsed {len(records)} records from Polygon.io for {symbol}")
        return records

class YahooFinanceSource(BaseDataSource):
    """Yahoo Finance data source implementation (free, no API key required)"""
    
    def fetch_data(self, symbol: str, start_date: datetime, 
                  end_date: datetime, interval: str = '1day') -> List[MarketDataRecord]:
        """Fetch data from Yahoo Finance API"""
        
        self._wait_for_rate_limit()
        
        # Convert datetime to Unix timestamp
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # Map interval to Yahoo format
        interval_map = {
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m',
            '1hour': '1h',
            '1day': '1d'
        }
        
        yahoo_interval = interval_map.get(interval, '1d')
        
        # Build URL
        url = f"{self.config.base_url}/{symbol}"
        
        params = {
            'period1': start_timestamp,
            'period2': end_timestamp,
            'interval': yahoo_interval,
            'includePrePost': 'false',
            'events': 'div,splits'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            return self._parse_yahoo_response(data, symbol)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data from Yahoo Finance: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing Yahoo Finance response: {e}")
            return []
    
    def _parse_yahoo_response(self, data: Dict, symbol: str) -> List[MarketDataRecord]:
        """Parse Yahoo Finance API response"""
        records = []
        
        try:
            chart_data = data['chart']['result'][0]
            timestamps = chart_data['timestamp']
            quotes = chart_data['indicators']['quote'][0]
            
            # Get adjusted close if available
            adj_close = None
            if 'adjclose' in chart_data['indicators']:
                adj_close = chart_data['indicators']['adjclose'][0]['adjclose']
            
            for i, timestamp in enumerate(timestamps):
                try:
                    # Create record
                    record = MarketDataRecord(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(timestamp),
                        open=float(quotes['open'][i]) if quotes['open'][i] is not None else 0.0,
                        high=float(quotes['high'][i]) if quotes['high'][i] is not None else 0.0,
                        low=float(quotes['low'][i]) if quotes['low'][i] is not None else 0.0,
                        close=float(quotes['close'][i]) if quotes['close'][i] is not None else 0.0,
                        volume=int(quotes['volume'][i]) if quotes['volume'][i] is not None else 0,
                        adjusted_close=float(adj_close[i]) if adj_close and adj_close[i] is not None else None
                    )
                    
                    # Skip records with zero values
                    if record.open > 0 and record.high > 0 and record.low > 0 and record.close > 0:
                        records.append(record)
                        
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Skipping invalid Yahoo Finance record at index {i}: {e}")
                    continue
                    
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing Yahoo Finance response structure: {e}")
            return []
        
        logger.info(f"Parsed {len(records)} records from Yahoo Finance for {symbol}")
        return records

# ============================================
# MAIN DATA DOWNLOADER CLASS
# ============================================

class MarketDataDownloader:
    """Main market data downloader with multi-source support"""
    
    def __init__(self, data_dir: Path = None, cache_enabled: bool = True):
        self.data_dir = data_dir or Path('./data/raw')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_enabled = cache_enabled
        self.cache_dir = Path('./data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data sources
        self.sources = {
            'alpha_vantage': AlphaVantageSource(DATA_SOURCES['alpha_vantage']),
            'polygon': PolygonSource(DATA_SOURCES['polygon']),
            'yahoo_finance': YahooFinanceSource(DATA_SOURCES['yahoo_finance'])
        }
        
        self.download_stats = {
            'total_symbols': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None
        }
    
    def download_symbol_data(self, symbol: str, start_date: datetime, 
                           end_date: datetime, source: str = 'alpha_vantage',
                           interval: str = '1day', save_to_file: bool = True) -> Optional[pd.DataFrame]:
        """Download data for a single symbol"""
        
        if source not in self.sources:
            logger.error(f"Unsupported data source: {source}")
            return None
        
        logger.info(f"📊 Downloading {symbol} data from {source} ({start_date.date()} to {end_date.date()})")
        
        try:
            # Check cache first
            if self.cache_enabled:
                cached_data = self._load_from_cache(symbol, start_date, end_date, source, interval)
                if cached_data is not None:
                    logger.info(f"✅ Loaded {symbol} from cache ({len(cached_data)} records)")
                    if save_to_file:
                        self._save_to_file(cached_data, symbol, source)
                    return cached_data
            
            # Fetch from source
            data_source = self.sources[source]
            records = data_source.fetch_data(symbol, start_date, end_date, interval)
            
            if not records:
                logger.warning(f"⚠️ No data retrieved for {symbol} from {source}")
                return None
            
            # Convert to DataFrame
            df = self._records_to_dataframe(records)
            
            # Cache the data
            if self.cache_enabled:
                self._save_to_cache(df, symbol, start_date, end_date, source, interval)
            
            # Save to file
            if save_to_file:
                self._save_to_file(df, symbol, source)
            
            self.download_stats['successful_downloads'] += 1
            self.download_stats['total_records'] += len(df)
            
            logger.info(f"✅ Successfully downloaded {symbol}: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to download {symbol} from {source}: {e}")
            self.download_stats['failed_downloads'] += 1
            return None
    
    def download_multiple_symbols(self, symbols: List[str], start_date: datetime,
                                 end_date: datetime, source: str = 'alpha_vantage',
                                 interval: str = '1day', max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        """Download data for multiple symbols with parallel processing"""
        
        logger.info(f"🚀 Starting batch download of {len(symbols)} symbols using {source}")
        
        self.download_stats['total_symbols'] = len(symbols)
        self.download_stats['start_time'] = datetime.now()
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_symbol = {
                executor.submit(self.download_symbol_data, symbol, start_date, end_date, source, interval): symbol
                for symbol in symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[symbol] = result
                except Exception as e:
                    logger.error(f"❌ Exception downloading {symbol}: {e}")
                    self.download_stats['failed_downloads'] += 1
        
        self.download_stats['end_time'] = datetime.now()
        self._print_download_summary()
        
        return results
    
    def _records_to_dataframe(self, records: List[MarketDataRecord]) -> pd.DataFrame:
        """Convert MarketDataRecord list to pandas DataFrame"""
        data = [asdict(record) for record in records]
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def _save_to_file(self, df: pd.DataFrame, symbol: str, source: str) -> None:
        """Save DataFrame to CSV file"""
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%SZ')
        filename = f"{symbol}_{source}_{timestamp}.csv"
        filepath = self.data_dir / filename
        
        df.to_csv(filepath)
        logger.info(f"💾 Saved {symbol} data to {filepath}")
    
    def _load_from_cache(self, symbol: str, start_date: datetime, end_date: datetime,
                        source: str, interval: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and valid"""
        cache_key = f"{symbol}_{source}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
        cache_file = self.cache_dir / cache_key
        
        if cache_file.exists():
            try:
                df = pd.read_pickle(cache_file)
                return df
            except Exception as e:
                logger.debug(f"Failed to load cache for {symbol}: {e}")
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, start_date: datetime,
                      end_date: datetime, source: str, interval: str) -> None:
        """Save DataFrame to cache"""
        cache_key = f"{symbol}_{source}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
        cache_file = self.cache_dir / cache_key
        
        try:
            df.to_pickle(cache_file)
        except Exception as e:
            logger.debug(f"Failed to cache data for {symbol}: {e}")
    
    def _print_download_summary(self) -> None:
        """Print download statistics summary"""
        stats = self.download_stats
        duration = (stats['end_time'] - stats['start_time']).total_seconds() if stats['start_time'] and stats['end_time'] else 0
        
        logger.info("=" * 50)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total symbols: {stats['total_symbols']}")
        logger.info(f"✅ Successful: {stats['successful_downloads']}")
        logger.info(f"❌ Failed: {stats['failed_downloads']}")
        logger.info(f"📊 Total records: {stats['total_records']:,}")
        logger.info(f"⏱️ Duration: {duration:.1f} seconds")
        
        if stats['successful_downloads'] > 0:
            avg_records = stats['total_records'] / stats['successful_downloads']
            logger.info(f"📈 Average records per symbol: {avg_records:.0f}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download market data for StockPredictionPro')
    parser.add_argument('--symbols', '-s', nargs='+', default=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
                       help='Stock symbols to download')
    parser.add_argument('--source', choices=['alpha_vantage', 'polygon', 'yahoo_finance'],
                       default='alpha_vantage', help='Data source')
    parser.add_argument('--start-date', type=str, default='2022-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', choices=['1min', '5min', '15min', '30min', '1hour', '1day'],
                       default='1day', help='Data interval')
    parser.add_argument('--output-dir', type=Path, default='./data/raw',
                       help='Output directory')
    parser.add_argument('--max-workers', type=int, default=3,
                       help='Maximum parallel workers')
    parser.add_argument('--disable-cache', action='store_true',
                       help='Disable caching')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Initialize downloader
    downloader = MarketDataDownloader(
        data_dir=args.output_dir,
        cache_enabled=not args.disable_cache
    )
    
    # Download data
    results = downloader.download_multiple_symbols(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
        source=args.source,
        interval=args.interval,
        max_workers=args.max_workers
    )
    
    logger.info(f"🎉 Download completed! Retrieved data for {len(results)} symbols.")

if __name__ == '__main__':
    main()
