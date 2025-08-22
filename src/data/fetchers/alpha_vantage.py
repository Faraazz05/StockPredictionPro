# ============================================
# StockPredictionPro - src/data/fetchers/alpha_vantage.py
# Alpha Vantage API data fetcher with rate limiting and error handling
# ============================================

import time
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import json

from ...utils.exceptions import (
    DataFetchError, ExternalAPIError, RateLimitError, 
    AuthenticationError, DataValidationError
)
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ...utils.config_loader import get
from ...utils.helpers import retry, validate_symbols
from .base_fetcher import BaseFetcher, DataRequest, DataResponse

logger = get_logger('data.fetchers.alpha_vantage')

class AlphaVantageFetcher(BaseFetcher):
    """
    Alpha Vantage API data fetcher
    
    Features:
    - Multiple data types (daily, intraday, fundamentals)
    - Intelligent rate limiting (5 calls/minute for free tier)
    - Comprehensive error handling and retries
    - Data validation and cleaning
    - Fallback and caching support
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage fetcher
        
        Args:
            api_key: Alpha Vantage API key (will use from config if None)
        """
        super().__init__(name="alpha_vantage")
        
        # Get API key from config or parameter
        self.api_key = api_key or get('app_config', 'data_sources.alpha_vantage.api_key', '')
        
        if not self.api_key:
            raise AuthenticationError(
                "Alpha Vantage API key not found",
                suggestions=[
                    "Set ALPHA_VANTAGE_KEY in .env file",
                    "Configure app_config.yaml data_sources.alpha_vantage.api_key",
                    "Pass api_key parameter to constructor"
                ]
            )
        
        # API configuration
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_calls = 5  # Free tier: 5 calls per minute
        self.rate_limit_period = 60  # seconds
        self.timeout = 30
        self.max_retries = 3
        
        # Rate limiting tracking
        self.call_timestamps: List[float] = []
        
        # Data type mappings
        self.functions = {
            'daily': 'TIME_SERIES_DAILY_ADJUSTED',
            'weekly': 'TIME_SERIES_WEEKLY_ADJUSTED', 
            'monthly': 'TIME_SERIES_MONTHLY_ADJUSTED',
            'intraday_1min': 'TIME_SERIES_INTRADAY',
            'intraday_5min': 'TIME_SERIES_INTRADAY',
            'intraday_15min': 'TIME_SERIES_INTRADAY',
            'intraday_30min': 'TIME_SERIES_INTRADAY',
            'intraday_60min': 'TIME_SERIES_INTRADAY',
            'fundamentals': 'OVERVIEW',
            'earnings': 'EARNINGS',
            'income_statement': 'INCOME_STATEMENT',
            'balance_sheet': 'BALANCE_SHEET',
            'cash_flow': 'CASH_FLOW'
        }
        
        logger.info(f"Alpha Vantage fetcher initialized with API key: {self.api_key[:8]}...")
    
    def can_fetch(self, request: DataRequest) -> bool:
        """Check if this fetcher can handle the request"""
        # Alpha Vantage primarily supports US stocks and some international
        if request.data_type in ['stock_data', 'fundamentals']:
            return True
        
        # Check if symbol format is supported
        symbol = request.symbols[0] if request.symbols else ""
        
        # US stocks (simple symbols)
        if symbol and len(symbol) <= 5 and symbol.isalpha():
            return True
        
        # Some international symbols are supported
        # (This would need expansion based on Alpha Vantage coverage)
        
        return False
    
    @time_it("alpha_vantage_fetch", include_args=True)
    def fetch_data(self, request: DataRequest) -> DataResponse:
        """
        Fetch data from Alpha Vantage API
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse with fetched data
        """
        logger.info(f"Fetching {request.data_type} data for {request.symbols} from Alpha Vantage")
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Process symbols (Alpha Vantage handles one symbol at a time)
            all_data = {}
            errors = {}
            
            for symbol in request.symbols:
                try:
                    with Timer(f"fetch_symbol_{symbol}") as timer:
                        symbol_data = self._fetch_symbol_data(symbol, request)
                        
                    if symbol_data is not None:
                        all_data[symbol] = symbol_data
                        logger.debug(f"Fetched {len(symbol_data)} records for {symbol} in {timer.result.duration_str}")
                    else:
                        errors[symbol] = "No data returned"
                        
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
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
                    'api_calls_made': len(request.symbols)
                },
                errors=errors if errors else None
            )
            
            if not all_data:
                raise DataFetchError(
                    "No data could be fetched for any symbols",
                    context={'symbols': request.symbols, 'errors': errors}
                )
            
            logger.info(f"Successfully fetched data for {len(all_data)} symbols")
            return response
            
        except Exception as e:
            logger.error(f"Alpha Vantage fetch failed: {e}")
            raise DataFetchError(
                f"Alpha Vantage API error: {str(e)}",
                source="alpha_vantage",
                context={'request': request.__dict__}
            ) from e
    
    def _validate_request(self, request: DataRequest):
        """Validate the data request"""
        if not request.symbols:
            raise DataValidationError("No symbols provided")
        
        # Validate symbols
        try:
            validated_symbols = validate_symbols(request.symbols)
            request.symbols = validated_symbols
        except Exception as e:
            raise DataValidationError(f"Invalid symbols: {e}")
        
        # Check data type support
        if request.data_type not in ['stock_data', 'fundamentals']:
            raise DataValidationError(f"Unsupported data type: {request.data_type}")
        
        # Check date range for historical data
        if request.data_type == 'stock_data':
            if request.start_date and request.end_date:
                if request.start_date >= request.end_date:
                    raise DataValidationError("Start date must be before end date")
    
    def _fetch_symbol_data(self, symbol: str, request: DataRequest) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol"""
        # Apply rate limiting
        self._enforce_rate_limit()
        
        if request.data_type == 'stock_data':
            return self._fetch_stock_data(symbol, request)
        elif request.data_type == 'fundamentals':
            return self._fetch_fundamentals(symbol, request)
        else:
            raise DataValidationError(f"Unsupported data type: {request.data_type}")
    
    @retry(max_attempts=3, delay=2.0, exponential_backoff=True)
    def _fetch_stock_data(self, symbol: str, request: DataRequest) -> Optional[pd.DataFrame]:
        """Fetch stock price data"""
        # Determine function and interval
        if hasattr(request, 'interval') and request.interval in ['1min', '5min', '15min', '30min', '60min']:
            function = self.functions['intraday_1min']  # Will be adjusted based on interval
            interval = request.interval
        else:
            function = self.functions['daily']
            interval = None
        
        # Build parameters
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full',  # Get full historical data
            'datatype': 'json'
        }
        
        # Add interval for intraday data
        if interval:
            params['interval'] = interval
        
        # Make API call
        response_data = self._make_api_call(params)
        
        # Parse response
        df = self._parse_stock_response(response_data, symbol, function)
        
        # Filter by date range if specified
        if df is not None and request.start_date and request.end_date:
            df = self._filter_by_date_range(df, request.start_date, request.end_date)
        
        return df
    
    def _fetch_fundamentals(self, symbol: str, request: DataRequest) -> Optional[pd.DataFrame]:
        """Fetch fundamental data"""
        params = {
            'function': self.functions['fundamentals'],
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        # Make API call
        response_data = self._make_api_call(params)
        
        # Parse response
        if not response_data or 'Symbol' not in response_data:
            logger.warning(f"No fundamental data found for {symbol}")
            return None
        
        # Convert to DataFrame (single row)
        df = pd.DataFrame([response_data])
        df['Symbol'] = symbol
        df['fetch_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return df
    
    def _make_api_call(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make API call with error handling"""
        try:
            logger.debug(f"Making Alpha Vantage API call: {params.get('function')} for {params.get('symbol')}")
            
            # Record call timestamp for rate limiting
            self.call_timestamps.append(time.time())
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout,
                headers={'User-Agent': 'StockPredictionPro/1.0'}
            )
            
            # Check HTTP status
            if response.status_code == 429:
                raise RateLimitError(
                    "Alpha Vantage rate limit exceeded",
                    retry_after=60
                )
            elif response.status_code == 401:
                raise AuthenticationError(
                    "Invalid Alpha Vantage API key",
                    suggestions=["Check your API key", "Verify API key permissions"]
                )
            elif response.status_code != 200:
                raise ExternalAPIError(
                    f"Alpha Vantage API error: HTTP {response.status_code}",
                    api_name="alpha_vantage",
                    status_code=response.status_code
                )
            
            # Parse JSON response
            data = response.json()
            
            # Check for API errors in response
            if 'Error Message' in data:
                raise ExternalAPIError(
                    f"Alpha Vantage API error: {data['Error Message']}",
                    api_name="alpha_vantage"
                )
            
            if 'Note' in data:
                # Rate limit message in response
                if 'call frequency' in data['Note'].lower():
                    raise RateLimitError(
                        f"Alpha Vantage rate limit: {data['Note']}",
                        retry_after=60
                    )
                else:
                    logger.warning(f"Alpha Vantage note: {data['Note']}")
            
            if 'Information' in data:
                # Often contains rate limit info
                if 'call frequency' in data['Information'].lower():
                    raise RateLimitError(
                        f"Alpha Vantage rate limit: {data['Information']}",
                        retry_after=60
                    )
                else:
                    logger.info(f"Alpha Vantage info: {data['Information']}")
            
            return data
            
        except requests.exceptions.Timeout:
            raise ExternalAPIError(
                "Alpha Vantage API timeout",
                api_name="alpha_vantage"
            )
        except requests.exceptions.ConnectionError:
            raise ExternalAPIError(
                "Failed to connect to Alpha Vantage API",
                api_name="alpha_vantage"
            )
        except json.JSONDecodeError:
            raise ExternalAPIError(
                "Invalid JSON response from Alpha Vantage API",
                api_name="alpha_vantage"
            )
    
    def _parse_stock_response(self, data: Dict[str, Any], symbol: str, function: str) -> Optional[pd.DataFrame]:
        """Parse stock data response into DataFrame"""
        try:
            # Find the time series data key
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key or time_series_key not in data:
                logger.warning(f"No time series data found for {symbol}")
                return None
            
            time_series_data = data[time_series_key]
            
            if not time_series_data:
                logger.warning(f"Empty time series data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series_data, orient='index')
            
            # Clean column names
            column_mapping = {}
            for col in df.columns:
                if 'open' in col.lower():
                    column_mapping[col] = 'Open'
                elif 'high' in col.lower():
                    column_mapping[col] = 'High'
                elif 'low' in col.lower():
                    column_mapping[col] = 'Low'
                elif 'close' in col.lower() and 'adjusted' not in col.lower():
                    column_mapping[col] = 'Close'
                elif 'adjusted' in col.lower() and 'close' in col.lower():
                    column_mapping[col] = 'Adj Close'
                elif 'volume' in col.lower():
                    column_mapping[col] = 'Volume'
                elif 'dividend' in col.lower():
                    column_mapping[col] = 'Dividend'
                elif 'split' in col.lower():
                    column_mapping[col] = 'Split'
            
            df = df.rename(columns=column_mapping)
            
            # Convert data types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Set index to datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            
            # Sort by date (oldest first)
            df = df.sort_index()
            
            # Use Adj Close as Close if available
            if 'Adj Close' in df.columns and 'Close' in df.columns:
                df['Close'] = df['Adj Close']
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns for {symbol}: {missing_columns}")
                return None
            
            # Remove rows with missing data
            df = df.dropna(subset=required_columns)
            
            if df.empty:
                logger.warning(f"No valid data after cleaning for {symbol}")
                return None
            
            logger.debug(f"Parsed {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse response for {symbol}: {e}")
            return None
    
    def _filter_by_date_range(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Filter DataFrame by date range"""
        try:
            # Ensure start_date and end_date are timezone-naive
            if start_date.tzinfo is not None:
                start_date = start_date.replace(tzinfo=None)
            if end_date.tzinfo is not None:
                end_date = end_date.replace(tzinfo=None)
            
            # Filter data
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_df = df.loc[mask]
            
            logger.debug(f"Filtered from {len(df)} to {len(filtered_df)} records")
            return filtered_df
            
        except Exception as e:
            logger.warning(f"Failed to filter by date range: {e}")
            return df
    
    def _enforce_rate_limit(self):
        """Enforce API rate limiting"""
        current_time = time.time()
        
        # Remove timestamps older than rate limit period
        self.call_timestamps = [
            timestamp for timestamp in self.call_timestamps 
            if current_time - timestamp < self.rate_limit_period
        ]
        
        # Check if we've exceeded the rate limit
        if len(self.call_timestamps) >= self.rate_limit_calls:
            # Calculate wait time
            oldest_call = min(self.call_timestamps)
            wait_time = self.rate_limit_period - (current_time - oldest_call)
            
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time + 1)  # Add 1 second buffer
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols (placeholder)"""
        # Alpha Vantage supports thousands of symbols
        # This would typically be loaded from a file or API
        return [
            # Major US stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'PYPL', 'UBER', 'SPOT',
            # Major indices (if supported)
            'SPY', 'QQQ', 'IWM', 'VTI'
        ]
    
    def get_available_data_types(self) -> List[str]:
        """Get available data types"""
        return ['stock_data', 'fundamentals']
    
    def test_connection(self) -> bool:
        """Test API connection and key validity"""
        try:
            # Make a simple API call
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': 'AAPL',
                'outputsize': 'compact',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for valid response
                if 'Time Series (Daily)' in data:
                    logger.info("Alpha Vantage connection test successful")
                    return True
                elif 'Error Message' in data:
                    logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                    return False
                elif 'Note' in data and 'call frequency' in data['Note'].lower():
                    logger.warning("Alpha Vantage rate limit reached during test")
                    return True  # Connection works, just rate limited
                else:
                    logger.error("Unexpected Alpha Vantage response format")
                    return False
            else:
                logger.error(f"Alpha Vantage connection test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Alpha Vantage connection test failed: {e}")
            return False
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status and usage information"""
        return {
            'name': self.name,
            'api_key_configured': bool(self.api_key),
            'api_key_preview': f"{self.api_key[:8]}..." if self.api_key else None,
            'rate_limit': f"{self.rate_limit_calls} calls per {self.rate_limit_period} seconds",
            'recent_calls': len(self.call_timestamps),
            'connection_status': 'untested',  # Would be updated by test_connection()
            'base_url': self.base_url,
            'supported_data_types': self.get_available_data_types()
        }
