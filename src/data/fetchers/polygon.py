# ============================================
# StockPredictionPro - src/data/fetchers/polygon.py
# Polygon.io API data fetcher with comprehensive market data
# ============================================

import time
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from pathlib import Path
import json

from ...utils.exceptions import (
    DataFetchError, ExternalAPIError, RateLimitError, 
    AuthenticationError, DataValidationError
)
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ...utils.config_loader import get
from ...utils.helpers import retry, validate_symbols, format_duration
from .base_fetcher import (
    BaseFetcher, DataRequest, DataResponse, FetcherCapabilities,
    DataType, DataFrequency, standardize_dataframe
)

logger = get_logger('data.fetchers.polygon')

class PolygonFetcher(BaseFetcher):
    """
    Polygon.io API data fetcher
    
    Features:
    - High-quality market data with tick-level precision
    - Real-time and historical data
    - Multiple asset classes (stocks, options, forex, crypto)
    - Advanced rate limiting and quota management
    - Comprehensive error handling
    - Data validation and cleaning
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon fetcher
        
        Args:
            api_key: Polygon.io API key (will use from config if None)
        """
        super().__init__(name="polygon")
        
        # Get API key from config or parameter
        self.api_key = api_key or get('app_config', 'data_sources.polygon.api_key', '')
        
        if not self.api_key:
            raise AuthenticationError(
                "Polygon.io API key not found",
                suggestions=[
                    "Set POLYGON_API_KEY in .env file",
                    "Configure app_config.yaml data_sources.polygon.api_key",
                    "Pass api_key parameter to constructor"
                ]
            )
        
        # API configuration
        self.base_url = "https://api.polygon.io"
        self.timeout = 30
        self.max_retries = 3
        
        # Rate limiting (varies by subscription tier)
        self.rate_limit_calls = 5     # Free tier: 5 calls per minute
        self.rate_limit_period = 60   # seconds
        self.call_timestamps: List[float] = []
        
        # API endpoints
        self.endpoints = {
            'aggregates': '/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}',
            'daily_bars': '/v1/open-close/{ticker}/{date}',
            'grouped_daily': '/v2/aggs/grouped/locale/us/market/stocks/{date}',
            'previous_close': '/v2/aggs/ticker/{ticker}/prev',
            'real_time_quote': '/v1/last_quote/stocks/{ticker}',
            'real_time_trade': '/v1/last/stocks/{ticker}',
            'ticker_details': '/v3/reference/tickers/{ticker}',
            'ticker_news': '/v2/reference/news',
            'market_holidays': '/v1/marketstatus/upcoming',
            'market_status': '/v1/marketstatus/now',
            'splits': '/v3/reference/splits',
            'dividends': '/v3/reference/dividends'
        }
        
        # Timespan mapping for Polygon API
        self.timespan_mapping = {
            DataFrequency.MINUTE_1.value: ('minute', 1),
            DataFrequency.MINUTE_5.value: ('minute', 5),
            DataFrequency.MINUTE_15.value: ('minute', 15),
            DataFrequency.MINUTE_30.value: ('minute', 30),
            DataFrequency.HOUR_1.value: ('hour', 1),
            DataFrequency.DAILY.value: ('day', 1),
            DataFrequency.WEEKLY.value: ('week', 1),
            DataFrequency.MONTHLY.value: ('month', 1),
            '1min': ('minute', 1),
            '5min': ('minute', 5),
            '15min': ('minute', 15),
            '30min': ('minute', 30),
            '1h': ('hour', 1),
            'daily': ('day', 1),
            'weekly': ('week', 1),
            'monthly': ('month', 1)
        }
        
        logger.info(f"Polygon.io fetcher initialized with API key: {self.api_key[:8]}...")
    
    def get_capabilities(self) -> FetcherCapabilities:
        """Get Polygon.io fetcher capabilities"""
        return FetcherCapabilities(
            name=self.name,
            supported_data_types=[
                DataType.STOCK_DATA.value,
                DataType.OPTIONS_DATA.value,
                DataType.FOREX_DATA.value,
                DataType.CRYPTO_DATA.value,
                DataType.FUNDAMENTALS.value
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
            supported_markets=["US"],  # Polygon primarily focuses on US markets
            supports_intraday=True,
            supports_historical=True,
            supports_real_time=True,
            rate_limit_calls=5,  # Free tier
            rate_limit_period_seconds=60,
            max_symbols_per_request=1,  # Polygon handles one symbol at a time
            max_historical_days=None,   # No strict limit with paid tiers
            requires_api_key=True,
            data_quality_score=0.95,   # Very high quality data
            reliability_score=0.92     # Generally reliable
        )
    
    def can_fetch(self, request: DataRequest) -> bool:
        """Check if this fetcher can handle the request"""
        capabilities = self.get_capabilities()
        
        # Basic capability check
        if not capabilities.can_handle_request(request):
            return False
        
        # Polygon-specific checks
        if request.data_type == DataType.STOCK_DATA.value:
            # Check if symbols are US stocks
            for symbol in request.symbols:
                # US stocks are typically 1-5 characters, no special suffixes
                if len(symbol) > 5 or '.' in symbol or '=' in symbol:
                    return False
            return True
        
        return True
    
    @time_it("polygon_fetch", include_args=True)
    def fetch_data(self, request: DataRequest) -> DataResponse:
        """
        Fetch data from Polygon.io API
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse with fetched data
        """
        logger.info(f"Fetching {request.data_type} data for {len(request.symbols)} symbols from Polygon.io")
        
        try:
            # Validate request
            validation = self.validate_request(request)
            if not validation.is_valid:
                raise DataValidationError(
                    f"Request validation failed: {'; '.join(validation.errors)}"
                )
            
            # Pre-fetch hook
            request = self.pre_fetch_hook(request)
            
            # Route to appropriate fetch method
            if request.data_type == DataType.STOCK_DATA.value:
                response = self._fetch_stock_data(request)
            elif request.data_type == DataType.FUNDAMENTALS.value:
                response = self._fetch_fundamentals_data(request)
            elif request.data_type == DataType.OPTIONS_DATA.value:
                response = self._fetch_options_data(request)
            else:
                raise DataValidationError(f"Unsupported data type: {request.data_type}")
            
            # Post-fetch hook
            response = self.post_fetch_hook(request, response)
            
            logger.info(f"Successfully fetched data for {len(response.data)} symbols")
            return response
            
        except Exception as e:
            self.handle_error(e, request)
            raise DataFetchError(
                f"Polygon.io fetch failed: {str(e)}",
                source="polygon",
                context={'request': request.to_dict()}
            ) from e
    
    def _fetch_stock_data(self, request: DataRequest) -> DataResponse:
        """Fetch stock price data"""
        all_data = {}
        errors = {}
        warnings = {}
        
        # Process symbols one at a time (Polygon limitation)
        for symbol in request.symbols:
            try:
                with Timer(f"fetch_symbol_{symbol}") as timer:
                    symbol_data = self._fetch_symbol_aggregates(symbol, request)
                
                if symbol_data is not None and not symbol_data.empty:
                    all_data[symbol] = symbol_data
                    logger.debug(f"Fetched {len(symbol_data)} records for {symbol} in {timer.result.duration_str}")
                else:
                    errors[symbol] = "No data returned"
                
                # Apply rate limiting
                self._enforce_rate_limit()
                
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
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )
        
        if not all_data:
            raise DataFetchError(
                "No data could be fetched for any symbols",
                context={'symbols': request.symbols, 'errors': errors}
            )
        
        return response
    
    @retry(max_attempts=3, delay=2.0, exponential_backoff=True)
    def _fetch_symbol_aggregates(self, symbol: str, request: DataRequest) -> Optional[pd.DataFrame]:
        """Fetch aggregated data for a single symbol"""
        try:
            # Determine timespan and multiplier
            timespan, multiplier = self._get_timespan_params(request.frequency or request.interval or 'daily')
            
            # Format dates for API
            if request.start_date and request.end_date:
                from_date = request.start_date.strftime('%Y-%m-%d')
                to_date = request.end_date.strftime('%Y-%m-%d')
            else:
                # Default to last 1 year
                to_date = datetime.now().strftime('%Y-%m-%d')
                from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Build URL
            url = f"{self.base_url}{self.endpoints['aggregates']}".format(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_date=from_date,
                to_date=to_date
            )
            
            # Add parameters
            params = {
                'adjusted': 'true' if request.adjusted else 'false',
                'sort': 'asc',
                'limit': 50000,  # Maximum results per request
                'apikey': self.api_key
            }
            
            logger.debug(f"Fetching aggregates for {symbol}: {timespan} {multiplier} from {from_date} to {to_date}")
            
            # Make API call
            response_data = self._make_api_call(url, params)
            
            # Parse response
            df = self._parse_aggregates_response(response_data, symbol)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch aggregates for {symbol}: {e}")
            raise
    
    def _parse_aggregates_response(self, data: Dict[str, Any], symbol: str) -> Optional[pd.DataFrame]:
        """Parse aggregates API response"""
        try:
            if not data or data.get('status') != 'OK':
                logger.warning(f"API returned non-OK status for {symbol}: {data.get('status', 'Unknown')}")
                return None
            
            results = data.get('results', [])
            if not results:
                logger.warning(f"No results returned for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Map Polygon columns to standard format
            column_mapping = {
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                't': 'timestamp',
                'vw': 'VWAP',
                'n': 'transactions'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Convert timestamp to datetime index
            if 'timestamp' in df.columns:
                # Polygon timestamps are in milliseconds
                df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('Date')
                df = df.drop('timestamp', axis=1)
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns for {symbol}: {missing_columns}")
                return None
            
            # Convert data types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'transactions']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date
            df = df.sort_index()
            
            # Remove any rows with missing data
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            if df.empty:
                logger.warning(f"No valid data after cleaning for {symbol}")
                return None
            
            logger.debug(f"Parsed {len(df)} records for {symbol}")
            return standardize_dataframe(df)
            
        except Exception as e:
            logger.error(f"Failed to parse response for {symbol}: {e}")
            return None
    
    def _fetch_fundamentals_data(self, request: DataRequest) -> DataResponse:
        """Fetch fundamental data"""
        all_data = {}
        errors = {}
        
        for symbol in request.symbols:
            try:
                with Timer(f"fetch_fundamentals_{symbol}"):
                    # Get ticker details
                    url = f"{self.base_url}{self.endpoints['ticker_details']}".format(ticker=symbol)
                    params = {'apikey': self.api_key}
                    
                    response_data = self._make_api_call(url, params)
                    
                    if response_data and response_data.get('status') == 'OK':
                        results = response_data.get('results', {})
                        
                        if results:
                            # Convert to DataFrame
                            df = pd.DataFrame([results])
                            df['Symbol'] = symbol
                            df['fetch_date'] = datetime.now().strftime('%Y-%m-%d')
                            
                            all_data[symbol] = df
                        else:
                            errors[symbol] = "No fundamental data available"
                    else:
                        errors[symbol] = f"API error: {response_data.get('status', 'Unknown')}"
                
                # Apply rate limiting
                self._enforce_rate_limit()
                
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
        # Polygon has extensive options data but requires more complex handling
        raise DataValidationError(
            "Options data fetching not yet implemented for Polygon.io",
            suggestions=[
                "Use stock_data type instead",
                "Contact support for options data implementation"
            ]
        )
    
    def _make_api_call(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call with error handling"""
        try:
            logger.debug(f"Making Polygon API call: {url}")
            
            # Record call timestamp for rate limiting
            self.call_timestamps.append(time.time())
            
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout,
                headers={'User-Agent': 'StockPredictionPro/1.0'}
            )
            
            # Check HTTP status
            if response.status_code == 429:
                raise RateLimitError(
                    "Polygon.io rate limit exceeded",
                    retry_after=60
                )
            elif response.status_code == 401:
                raise AuthenticationError(
                    "Invalid Polygon.io API key",
                    suggestions=["Check your API key", "Verify API key permissions"]
                )
            elif response.status_code == 403:
                raise AuthenticationError(
                    "Polygon.io API access denied",
                    suggestions=[
                        "Check your subscription plan",
                        "Verify API key permissions",
                        "Contact Polygon.io support"
                    ]
                )
            elif response.status_code != 200:
                raise ExternalAPIError(
                    f"Polygon.io API error: HTTP {response.status_code}",
                    api_name="polygon",
                    status_code=response.status_code
                )
            
            # Parse JSON response
            data = response.json()
            
            # Check for API-level errors
            if data.get('status') == 'ERROR':
                error_msg = data.get('error', 'Unknown API error')
                raise ExternalAPIError(
                    f"Polygon.io API error: {error_msg}",
                    api_name="polygon"
                )
            
            return data
            
        except requests.exceptions.Timeout:
            raise ExternalAPIError(
                "Polygon.io API timeout",
                api_name="polygon"
            )
        except requests.exceptions.ConnectionError:
            raise ExternalAPIError(
                "Failed to connect to Polygon.io API",
                api_name="polygon"
            )
        except json.JSONDecodeError:
            raise ExternalAPIError(
                "Invalid JSON response from Polygon.io API",
                api_name="polygon"
            )
    
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
    
    def _get_timespan_params(self, frequency: str) -> Tuple[str, int]:
        """Get timespan and multiplier for Polygon API"""
        if frequency in self.timespan_mapping:
            return self.timespan_mapping[frequency]
        else:
            # Default to daily
            logger.warning(f"Unknown frequency '{frequency}', defaulting to daily")
            return 'day', 1
    
    def test_connection(self) -> bool:
        """Test API connection and key validity"""
        try:
            logger.info("Testing Polygon.io connection...")
            
            # Get market status (simple API call)
            url = f"{self.base_url}{self.endpoints['market_status']}"
            params = {'apikey': self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK':
                    logger.info("Polygon.io connection test successful")
                    return True
                else:
                    logger.error(f"Polygon.io API error: {data}")
                    return False
            elif response.status_code == 401:
                logger.error("Polygon.io authentication failed - check API key")
                return False
            elif response.status_code == 403:
                logger.error("Polygon.io access denied - check subscription plan")
                return False
            else:
                logger.error(f"Polygon.io connection test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Polygon.io connection test failed: {e}")
            return False
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        try:
            url = f"{self.base_url}{self.endpoints['market_status']}"
            params = {'apikey': self.api_key}
            
            data = self._make_api_call(url, params)
            
            if data and data.get('status') == 'OK':
                return data.get('results', {})
            else:
                return {'error': 'Failed to get market status'}
                
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return {'error': str(e)}
    
    def get_market_holidays(self) -> List[Dict[str, Any]]:
        """Get upcoming market holidays"""
        try:
            url = f"{self.base_url}{self.endpoints['market_holidays']}"
            params = {'apikey': self.api_key}
            
            data = self._make_api_call(url, params)
            
            if data and data.get('status') == 'OK':
                return data.get('results', [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to get market holidays: {e}")
            return []
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of commonly supported symbols"""
        return [
            # Major US stocks (Polygon's main focus)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'PYPL', 'UBER', 'SPOT',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
            'JNJ', 'PFE', 'ABT', 'MRK', 'TMO', 'UNH', 'CVS', 'ABBV',
            'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT'
        ]
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status and usage information"""
        return {
            'name': self.name,
            'api_key_configured': bool(self.api_key),
            'api_key_preview': f"{self.api_key[:8]}..." if self.api_key else None,
            'rate_limit': f"{self.rate_limit_calls} calls per {self.rate_limit_period} seconds",
            'recent_calls': len(self.call_timestamps),
            'connection_status': 'untested',
            'base_url': self.base_url,
            'supported_data_types': self.get_capabilities().supported_data_types,
            'market_status': self.get_market_status()
        }
    
    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol (if subscription supports it)"""
        try:
            url = f"{self.base_url}{self.endpoints['real_time_quote']}".format(ticker=symbol)
            params = {'apikey': self.api_key}
            
            data = self._make_api_call(url, params)
            
            if data and data.get('status') == 'OK':
                return data.get('results', {})
            else:
                return {'error': f"Failed to get quote for {symbol}"}
                
        except Exception as e:
            logger.error(f"Failed to get real-time quote for {symbol}: {e}")
            return {'error': str(e)}
    
    def get_previous_close(self, symbol: str) -> Dict[str, Any]:
        """Get previous close data for a symbol"""
        try:
            url = f"{self.base_url}{self.endpoints['previous_close']}".format(ticker=symbol)
            params = {'apikey': self.api_key}
            
            data = self._make_api_call(url, params)
            
            if data and data.get('status') == 'OK':
                results = data.get('results', [])
                return results[0] if results else {}
            else:
                return {'error': f"Failed to get previous close for {symbol}"}
                
        except Exception as e:
            logger.error(f"Failed to get previous close for {symbol}: {e}")
            return {'error': str(e)}
