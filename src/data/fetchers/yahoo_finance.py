# ============================================
# StockPredictionPro - src/data/fetchers/yahoo_finance.py
# Yahoo Finance data fetcher with yfinance library integration
# ============================================

import time
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

from ...utils.exceptions import (
    DataFetchError, ExternalAPIError, DataValidationError,
    InsufficientDataError
)
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ...utils.config_loader import get
from ...utils.helpers import retry, validate_symbols
from ...utils.validators import validate_data_for_analysis
from .base_fetcher import (
    BaseFetcher, DataRequest, DataResponse, FetcherCapabilities,
    DataType, DataFrequency, standardize_dataframe
)

logger = get_logger('data.fetchers.yahoo_finance')

class YahooFinanceFetcher(BaseFetcher):
    """
    Yahoo Finance data fetcher using yfinance library
    
    Features:
    - No API key required (free service)
    - Multiple asset classes (stocks, ETFs, indices, crypto, forex)
    - Historical and real-time data
    - Robust error handling with retries
    - Data validation and cleaning
    - Support for international markets
    """
    
    def __init__(self):
        """Initialize Yahoo Finance fetcher"""
        super().__init__(name="yahoo_finance")
        
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance library not available. Install with: pip install yfinance"
            )
        
        # Configuration from app config
        self.timeout = get('app_config', 'data_sources.yahoo_finance.timeout', 30)
        self.max_retries = get('app_config', 'data_sources.yahoo_finance.retries', 3)
        self.cache_ttl = get('app_config', 'data_sources.yahoo_finance.cache_ttl', 3600)
        
        # Yahoo Finance specific settings
        self.max_symbols_per_batch = 10  # Yahoo can handle multiple symbols
        self.rate_limit_delay = 0.1      # Small delay between requests
        
        # Data validation thresholds
        self.min_data_quality_score = 0.7
        self.max_missing_data_ratio = 0.15  # 15% maximum missing data
        
        # Supported periods and intervals
        self.valid_periods = [
            '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
        ]
        self.valid_intervals = [
            '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        ]
        
        logger.info("Yahoo Finance fetcher initialized successfully")
    
    def get_capabilities(self) -> FetcherCapabilities:
        """Get Yahoo Finance fetcher capabilities"""
        return FetcherCapabilities(
            name=self.name,
            supported_data_types=[
                DataType.STOCK_DATA.value,
                DataType.FUNDAMENTALS.value,
                DataType.OPTIONS_DATA.value,
                DataType.CRYPTO_DATA.value,
                DataType.FOREX_DATA.value
            ],
            supported_frequencies=[
                DataFrequency.MINUTE_1.value,
                DataFrequency.MINUTE_5.value,
                DataFrequency.MINUTE_15.value,
                DataFrequency.MINUTE_30.value,
                DataFrequency.HOUR_1.value,
                DataFrequency.DAILY.value,
                DataFrequency.WEEKLY.value,
                DataFrequency.MONTHLY.value
            ],
            supported_markets=[
                "US", "IN", "UK", "CA", "AU", "DE", "FR", "JP", "HK", "SG"
            ],
            supports_intraday=True,
            supports_historical=True,
            supports_real_time=False,  # Yahoo doesn't provide true real-time
            rate_limit_calls=None,      # No strict rate limits
            rate_limit_period_seconds=None,
            max_symbols_per_request=10,
            max_historical_days=None,   # No strict limit
            requires_api_key=False,
            data_quality_score=0.85,   # Generally good quality
            reliability_score=0.9      # Very reliable service
        )
    
    def can_fetch(self, request: DataRequest) -> bool:
        """Check if this fetcher can handle the request"""
        capabilities = self.get_capabilities()
        
        # Basic capability check
        if not capabilities.can_handle_request(request):
            return False
        
        # Yahoo Finance specific checks
        if request.data_type == DataType.STOCK_DATA.value:
            # Can handle most stock symbols
            return True
        elif request.data_type == DataType.FUNDAMENTALS.value:
            # Limited fundamental data available
            return True
        elif request.data_type in [DataType.CRYPTO_DATA.value, DataType.FOREX_DATA.value]:
            # Check if symbols have appropriate suffixes
            for symbol in request.symbols:
                if request.data_type == DataType.CRYPTO_DATA.value and not symbol.endswith('-USD'):
                    continue  # Crypto symbols should end with -USD
                if request.data_type == DataType.FOREX_DATA.value and '=' not in symbol:
                    continue  # Forex symbols should have = in them
            return True
        
        return False
    
    @time_it("yahoo_finance_fetch", include_args=True)
    def fetch_data(self, request: DataRequest) -> DataResponse:
        """
        Fetch data from Yahoo Finance
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse with fetched data
        """
        logger.info(f"Fetching {request.data_type} data for {len(request.symbols)} symbols from Yahoo Finance")
        
        try:
            # Validate request
            validation = self.validate_request(request)
            if not validation.is_valid:
                raise DataValidationError(
                    f"Request validation failed: {'; '.join(validation.errors)}"
                )
            
            # Pre-fetch hook
            request = self.pre_fetch_hook(request)
            
            # Route to appropriate fetch method based on data type
            if request.data_type == DataType.STOCK_DATA.value:
                response = self._fetch_stock_data(request)
            elif request.data_type == DataType.FUNDAMENTALS.value:
                response = self._fetch_fundamentals_data(request)
            elif request.data_type == DataType.OPTIONS_DATA.value:
                response = self._fetch_options_data(request)
            elif request.data_type in [DataType.CRYPTO_DATA.value, DataType.FOREX_DATA.value]:
                response = self._fetch_alternative_data(request)
            else:
                raise DataValidationError(f"Unsupported data type: {request.data_type}")
            
            # Post-fetch hook
            response = self.post_fetch_hook(request, response)
            
            logger.info(f"Successfully fetched data for {len(response.data)} symbols")
            return response
            
        except Exception as e:
            self.handle_error(e, request)
            raise DataFetchError(
                f"Yahoo Finance fetch failed: {str(e)}",
                source="yahoo_finance",
                context={'request': request.to_dict()}
            ) from e
    
    def _fetch_stock_data(self, request: DataRequest) -> DataResponse:
        """Fetch stock price data"""
        all_data = {}
        errors = {}
        warnings = {}
        
        # Process symbols in batches
        symbol_batches = [
            request.symbols[i:i + self.max_symbols_per_batch]
            for i in range(0, len(request.symbols), self.max_symbols_per_batch)
        ]
        
        for batch in symbol_batches:
            try:
                batch_data = self._fetch_stock_batch(batch, request)
                all_data.update(batch_data['data'])
                if batch_data['errors']:
                    errors.update(batch_data['errors'])
                if batch_data['warnings']:
                    warnings.update(batch_data['warnings'])
                    
                # Small delay between batches to be respectful
                if len(symbol_batches) > 1:
                    time.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Failed to fetch batch {batch}: {e}")
                for symbol in batch:
                    errors[symbol] = str(e)
        
        # Create response
        response = DataResponse(
            data=all_data,
            metadata={
                'source': self.name,
                'request_time': datetime.now().isoformat(),
                'symbols_requested': request.symbols,
                'symbols_fetched': list(all_data.keys()),
                'symbols_failed': list(errors.keys()),
                'data_type': request.data_type,
                'batches_processed': len(symbol_batches)
            },
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )
        
        if not all_data:
            raise InsufficientDataError(
                "No data could be fetched for any symbols",
                context={'symbols': request.symbols, 'errors': errors}
            )
        
        return response
    
    @retry(max_attempts=3, delay=1.0, exponential_backoff=True)
    def _fetch_stock_batch(self, symbols: List[str], request: DataRequest) -> Dict[str, Any]:
        """Fetch a batch of stock symbols"""
        batch_data = {'data': {}, 'errors': {}, 'warnings': {}}
        
        try:
            # Determine period and interval for yfinance
            period, start, end = self._convert_date_params(request)
            interval = self._convert_interval(request.frequency or request.interval or '1d')
            
            logger.debug(f"Fetching batch {symbols} with period={period}, interval={interval}")
            
            with Timer(f"yfinance_download_{len(symbols)}_symbols") as timer:
                # Use yfinance to download data
                if len(symbols) == 1:
                    ticker = yf.Ticker(symbols[0])
                    if start and end:
                        data = ticker.history(start=start, end=end, interval=interval, 
                                            auto_adjust=request.adjusted, prepost=False)
                    else:
                        data = ticker.history(period=period, interval=interval,
                                            auto_adjust=request.adjusted, prepost=False)
                    
                    if not data.empty:
                        batch_data['data'][symbols[0]] = self._process_dataframe(data, symbols[0])
                    else:
                        batch_data['errors'][symbols[0]] = "No data returned"
                
                else:
                    # Multiple symbols
                    if start and end:
                        data = yf.download(symbols, start=start, end=end, interval=interval,
                                         auto_adjust=request.adjusted, group_by='ticker',
                                         threads=True, progress=False)
                    else:
                        data = yf.download(symbols, period=period, interval=interval,
                                         auto_adjust=request.adjusted, group_by='ticker',
                                         threads=True, progress=False)
                    
                    # Process multi-symbol data
                    if not data.empty:
                        batch_data.update(self._process_multi_symbol_data(data, symbols))
                    else:
                        for symbol in symbols:
                            batch_data['errors'][symbol] = "No data returned"
            
            logger.debug(f"Batch fetch completed in {timer.result.duration_str}")
            
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            for symbol in symbols:
                batch_data['errors'][symbol] = str(e)
        
        return batch_data
    
    def _process_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process and validate a single symbol's DataFrame"""
        if df is None or df.empty:
            return None
        
        # Standardize the DataFrame
        df = standardize_dataframe(df)
        
        # Yahoo Finance specific cleaning
        df = self._clean_yahoo_data(df, symbol)
        
        # Validate data quality
        validation = validate_data_for_analysis(df, symbol)
        if not validation.is_valid:
            logger.warning(f"Data quality issues for {symbol}: {validation.errors}")
        
        return df
    
    def _process_multi_symbol_data(self, data: pd.DataFrame, symbols: List[str]) -> Dict[str, Any]:
        """Process multi-symbol data from yfinance"""
        result = {'data': {}, 'errors': {}, 'warnings': {}}
        
        try:
            # Handle case where only one symbol was fetched (no MultiIndex)
            if not isinstance(data.columns, pd.MultiIndex):
                if len(symbols) == 1:
                    processed_df = self._process_dataframe(data, symbols[0])
                    if processed_df is not None:
                        result['data'][symbols[0]] = processed_df
                    else:
                        result['errors'][symbols[0]] = "Failed to process data"
                return result
            
            # Handle multi-symbol MultiIndex columns
            for symbol in symbols:
                try:
                    if symbol in data.columns.get_level_values(0):
                        symbol_data = data[symbol]
                        
                        if not symbol_data.empty:
                            processed_df = self._process_dataframe(symbol_data, symbol)
                            if processed_df is not None:
                                result['data'][symbol] = processed_df
                            else:
                                result['errors'][symbol] = "Failed to process data"
                        else:
                            result['errors'][symbol] = "No data available"
                    else:
                        result['errors'][symbol] = "Symbol not found in response"
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    result['errors'][symbol] = str(e)
        
        except Exception as e:
            logger.error(f"Error processing multi-symbol data: {e}")
            for symbol in symbols:
                result['errors'][symbol] = str(e)
        
        return result
    
    def _clean_yahoo_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean Yahoo Finance specific data issues"""
        if df is None or df.empty:
            return df
        
        # Remove timezone info if present (Yahoo sometimes includes it)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Handle zero volume (common issue with Yahoo data)
        if 'Volume' in df.columns:
            # Replace zero volume with NaN for weekends/holidays
            df.loc[df['Volume'] == 0, 'Volume'] = np.nan
        
        # Handle missing price data
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                # Forward fill small gaps (max 3 days for daily data)
                df[col] = df[col].fillna(method='ffill', limit=3)
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Ensure proper sorting
        df = df.sort_index()
        
        # Remove duplicate dates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def _fetch_fundamentals_data(self, request: DataRequest) -> DataResponse:
        """Fetch fundamental data (limited from Yahoo Finance)"""
        all_data = {}
        errors = {}
        
        for symbol in request.symbols:
            try:
                with Timer(f"fetch_fundamentals_{symbol}"):
                    ticker = yf.Ticker(symbol)
                    
                    # Get basic info
                    info = ticker.info
                    
                    if info and len(info) > 1:  # Basic check for valid data
                        # Convert to DataFrame
                        df = pd.DataFrame([info])
                        df['Symbol'] = symbol
                        df['fetch_date'] = datetime.now().strftime('%Y-%m-%d')
                        
                        all_data[symbol] = df
                    else:
                        errors[symbol] = "No fundamental data available"
                
                # Small delay between requests
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to fetch fundamentals for {symbol}: {e}")
                errors[symbol] = str(e)
        
        return DataResponse(
            data=all_data,
            metadata={
                'source': self.name,
                'request_time': datetime.now().isoformat(),
                'data_type': request.data_type,
                'symbols_requested': request.symbols,
                'symbols_fetched': list(all_data.keys()),
                'symbols_failed': list(errors.keys())
            },
            errors=errors if errors else None
        )
    
    def _fetch_options_data(self, request: DataRequest) -> DataResponse:
        """Fetch options data"""
        # Placeholder for options data fetching
        # Yahoo Finance has options data but requires more complex handling
        raise DataValidationError(
            "Options data fetching not yet implemented",
            suggestions=["Use stock_data type instead", "Wait for options implementation"]
        )
    
    def _fetch_alternative_data(self, request: DataRequest) -> DataResponse:
        """Fetch crypto or forex data"""
        # Use the same stock data logic but with different symbol formats
        return self._fetch_stock_data(request)
    
    def _convert_date_params(self, request: DataRequest) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Convert DataRequest dates to yfinance parameters"""
        period = None
        start = None
        end = None
        
        if request.start_date and request.end_date:
            # Use specific date range
            start = request.start_date.strftime('%Y-%m-%d')
            end = request.end_date.strftime('%Y-%m-%d')
        elif request.start_date:
            # From start_date to now
            start = request.start_date.strftime('%Y-%m-%d')
            end = datetime.now().strftime('%Y-%m-%d')
        elif request.end_date:
            # Use period up to end_date (default to 1 year back)
            end = request.end_date.strftime('%Y-%m-%d')
            start = (request.end_date - timedelta(days=365)).strftime('%Y-%m-%d')
        else:
            # Use default period
            period = '1y'
        
        return period, start, end
    
    def _convert_interval(self, frequency: str) -> str:
        """Convert frequency to yfinance interval"""
        interval_mapping = {
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m',
            '1h': '1h',
            '1hour': '1h',
            'daily': '1d',
            '1d': '1d',
            'weekly': '1wk',
            '1w': '1wk',
            'monthly': '1mo',
            '1mo': '1mo'
        }
        
        return interval_mapping.get(frequency, '1d')
    
    def test_connection(self) -> bool:
        """Test connection to Yahoo Finance"""
        try:
            logger.info("Testing Yahoo Finance connection...")
            
            # Try to fetch a single day of data for a reliable stock
            ticker = yf.Ticker('AAPL')
            data = ticker.history(period='1d', interval='1d')
            
            if not data.empty:
                logger.info("Yahoo Finance connection test successful")
                return True
            else:
                logger.warning("Yahoo Finance connection test returned empty data")
                return False
                
        except Exception as e:
            logger.error(f"Yahoo Finance connection test failed: {e}")
            return False
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of commonly supported symbols"""
        return [
            # Major US stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'PYPL', 'UBER', 'SPOT',
            
            # Major US indices
            '^GSPC', '^IXIC', '^DJI', '^VIX',
            
            # Major US ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO',
            
            # Indian stocks (NSE)
            'INFY.NS', 'TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'WIPRO.NS',
            'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS',
            
            # Indian index
            '^NSEI',  # NIFTY 50
            
            # Major cryptocurrencies
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD',
            
            # Major forex pairs
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X'
        ]
    
    def get_market_hours(self, symbol: str) -> Dict[str, Any]:
        """Get market hours for a symbol (if available)"""
        try:
            ticker = yf.Ticker(symbol)
            
            # This is basic - Yahoo doesn't provide detailed market hours
            # We'll provide common market hours based on symbol pattern
            if symbol.endswith('.NS') or symbol.endswith('.BO'):
                # Indian markets
                return {
                    'market': 'India',
                    'timezone': 'Asia/Kolkata',
                    'open_time': '09:15',
                    'close_time': '15:30',
                    'trading_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                }
            elif any(symbol.startswith(prefix) for prefix in ['^', 'BTC-', 'ETH-']):
                # Indices or crypto (24/7 for crypto)
                return {
                    'market': 'Global/Crypto',
                    'timezone': 'UTC',
                    'trading': '24/7' if '-USD' in symbol else 'Market Hours'
                }
            else:
                # Default to US market hours
                return {
                    'market': 'United States',
                    'timezone': 'America/New_York',
                    'open_time': '09:30',
                    'close_time': '16:00',
                    'trading_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                }
                
        except Exception as e:
            logger.warning(f"Could not determine market hours for {symbol}: {e}")
            return {}
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about a specific symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if info and len(info) > 1:
                return {
                    'symbol': symbol,
                    'name': info.get('longName', info.get('shortName', 'Unknown')),
                    'exchange': info.get('exchange', 'Unknown'),
                    'currency': info.get('currency', 'Unknown'),
                    'market_cap': info.get('marketCap'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'country': info.get('country'),
                    'website': info.get('website'),
                    'description': info.get('longBusinessSummary', '')[:200] + '...' if info.get('longBusinessSummary') else ''
                }
            else:
                return {'symbol': symbol, 'error': 'No information available'}
                
        except Exception as e:
            return {'symbol': symbol, 'error': str(e)}
