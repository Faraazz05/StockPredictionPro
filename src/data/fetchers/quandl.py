# ============================================
# StockPredictionPro - src/data/fetchers/quandl.py
# Quandl/Nasdaq Data Link API fetcher
# ============================================

import time
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
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

logger = get_logger('data.fetchers.quandl')

class QuandlFetcher(BaseFetcher):
    """
    Quandl (Nasdaq Data Link) API fetcher
    
    Features:
    - Access to alternative and traditional financial datasets
    - Core financial data, economic data, and alternative datasets
    - High-quality curated data from premium providers
    - Comprehensive global coverage
    - Both free and premium datasets
    - Historical and real-time data
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Quandl fetcher
        
        Args:
            api_key: Quandl API key (will use from config if None)
        """
        super().__init__(name="quandl")
        
        # Get API key from config or parameter
        self.api_key = api_key or get('app_config', 'data_sources.quandl.api_key', '')
        
        if not self.api_key:
            raise AuthenticationError(
                "Quandl API key not found",
                suggestions=[
                    "Get API key from https://data.nasdaq.com/",
                    "Set QUANDL_API_KEY in .env file", 
                    "Configure app_config.yaml data_sources.quandl.api_key",
                    "Pass api_key parameter to constructor"
                ]
            )
        
        # API configuration
        self.base_url = "https://data.nasdaq.com/api/v3"
        self.timeout = 30
        self.max_retries = 3
        
        # Rate limiting (varies by subscription)
        self.rate_limit_calls = 300  # Premium: 300/10sec, Free: 50/day
        self.rate_limit_period = 10  # seconds for premium
        self.call_timestamps: List[float] = []
        
        # Common Quandl datasets and codes
        self.dataset_mappings = {
            # Stock Markets
            'wiki': 'WIKI',           # Wiki EOD Stock Prices (discontinued but good example)
            'eod': 'EOD',             # End of Day Stock Prices
            'sf1': 'SF1',             # Sharadar Fundamentals
            'sf2': 'SF2',             # Sharadar Insiders
            'sep': 'SEP',             # Sharadar Equity Prices
            
            # Economic Data
            'fed': 'FED',             # Federal Reserve Economic Data
            'oecd': 'OECD',           # OECD Statistics
            'worldbank': 'WWDI',      # World Bank World Development Indicators
            'imf': 'ODA',             # IMF Data
            
            # Commodities
            'lme': 'LME',             # London Metal Exchange
            'chris': 'CHRIS',         # Continuous Futures
            'ofdp': 'OFDP',           # Open Financial Data Project
            
            # Alternative Data
            'zacks': 'ZACKS',         # Zacks Investment Research
            'aaii': 'AAII',           # American Association of Individual Investors
            'yale': 'YALE',           # Yale Department of Economics
            
            # International Markets
            'nse': 'NSE',             # National Stock Exchange of India
            'bse': 'BSE',             # Bombay Stock Exchange
            'tse': 'TSE',             # Tokyo Stock Exchange
            'lse': 'LSE',             # London Stock Exchange
            
            # Crypto (historical, now limited)
            'bchain': 'BCHAIN',       # Bitcoin Blockchain data
            'bitfinex': 'BITFINEX',   # Bitfinex Exchange data
        }
        
        # Common stock databases for different markets
        self.stock_databases = {
            'US': ['EOD', 'SF1', 'SEP'],
            'IN': ['NSE', 'BSE'],
            'JP': ['TSE'],
            'UK': ['LSE'],
            'Global': ['EOD']
        }
        
        # Frequency mapping for Quandl
        self.frequency_mapping = {
            DataFrequency.DAILY.value: 'daily',
            DataFrequency.WEEKLY.value: 'weekly',
            DataFrequency.MONTHLY.value: 'monthly',
            DataFrequency.QUARTERLY.value: 'quarterly',
            DataFrequency.ANNUALLY.value: 'annual',
            'daily': 'daily',
            'weekly': 'weekly',
            'monthly': 'monthly',
            'quarterly': 'quarterly',
            'annual': 'annual'
        }
        
        # Common indicators for different datasets
        self.common_indicators = {
            # Sharadar Fundamentals (SF1)
            'revenue': ('SF1', 'REVENUE'),
            'gross_profit': ('SF1', 'GP'),
            'operating_income': ('SF1', 'OPINC'),
            'net_income': ('SF1', 'NETINC'),
            'total_assets': ('SF1', 'ASSETS'),
            'total_debt': ('SF1', 'DEBT'),
            'market_cap': ('SF1', 'MARKETCAP'),
            'pe_ratio': ('SF1', 'PE'),
            'price_to_book': ('SF1', 'PB'),
            'roe': ('SF1', 'ROE'),
            'roa': ('SF1', 'ROA'),
            
            # Economic indicators via different databases
            'gdp_us': ('FED', 'GDP'),
            'inflation_us': ('FED', 'CPIAUCSL'),
            'unemployment_us': ('FED', 'UNRATE'),
            
            # Alternative data examples
            'investor_sentiment': ('AAII', 'AAII_SENTIMENT'),
            'insider_trading': ('SF2', 'TRADES'),
        }
        
        logger.info(f"Quandl fetcher initialized with API key: {self.api_key[:8]}...")
    
    def get_capabilities(self) -> FetcherCapabilities:
        """Get Quandl fetcher capabilities"""
        return FetcherCapabilities(
            name=self.name,
            supported_data_types=[
                DataType.STOCK_DATA.value,
                DataType.FUNDAMENTALS.value,
                DataType.ECONOMIC_DATA.value,
                "alternative_data",
                "commodities",
                "international_data"
            ],
            supported_frequencies=[
                DataFrequency.DAILY.value,
                DataFrequency.WEEKLY.value,
                DataFrequency.MONTHLY.value,
                DataFrequency.QUARTERLY.value,
                DataFrequency.ANNUALLY.value
            ],
            supported_markets=["US", "IN", "UK", "JP", "Global"],
            supports_intraday=False,    # Most Quandl data is daily or lower frequency
            supports_historical=True,
            supports_real_time=False,   # Mostly historical/delayed data
            rate_limit_calls=300,       # Premium tier
            rate_limit_period_seconds=10,
            max_symbols_per_request=1,  # Handle one series at a time
            max_historical_days=None,   # No strict limit
            requires_api_key=True,
            data_quality_score=0.9,    # High quality curated data
            reliability_score=0.85     # Generally reliable but depends on data provider
        )
    
    def can_fetch(self, request: DataRequest) -> bool:
        """Check if this fetcher can handle the request"""
        capabilities = self.get_capabilities()
        
        # Basic capability check
        if not capabilities.can_handle_request(request):
            return False
        
        # Quandl can handle various data types
        supported_types = [
            DataType.STOCK_DATA.value,
            DataType.FUNDAMENTALS.value, 
            DataType.ECONOMIC_DATA.value,
            "alternative_data",
            "commodities"
        ]
        
        if request.data_type in supported_types:
            return True
        
        return False
    
    @time_it("quandl_fetch", include_args=True)
    def fetch_data(self, request: DataRequest) -> DataResponse:
        """
        Fetch data from Quandl API
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse with fetched data
        """
        logger.info(f"Fetching {request.data_type} data for {len(request.symbols)} symbols from Quandl")
        
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
            elif request.data_type == DataType.ECONOMIC_DATA.value:
                response = self._fetch_economic_data(request)
            elif request.data_type in ["alternative_data", "commodities"]:
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
                f"Quandl fetch failed: {str(e)}",
                source="quandl",
                context={'request': request.to_dict()}
            ) from e
    
    def _fetch_stock_data(self, request: DataRequest) -> DataResponse:
        """Fetch stock price data"""
        all_data = {}
        errors = {}
        warnings = {}
        
        # Determine appropriate database based on symbols
        database = self._determine_stock_database(request.symbols)
        
        for symbol in request.symbols:
            try:
                with Timer(f"fetch_stock_{symbol}") as timer:
                    # Convert symbol to Quandl format
                    quandl_code = self._build_stock_code(database, symbol)
                    symbol_data = self._fetch_time_series(quandl_code, request)
                
                if symbol_data is not None and not symbol_data.empty:
                    # Process and standardize data
                    processed_data = self._process_stock_data(symbol_data, symbol, database)
                    if processed_data is not None:
                        all_data[symbol] = processed_data
                        logger.debug(f"Fetched {len(processed_data)} records for {symbol} in {timer.result.duration_str}")
                    else:
                        errors[symbol] = "Failed to process data"
                else:
                    errors[symbol] = "No data returned"
                
                # Apply rate limiting
                self._enforce_rate_limit()
                
            except Exception as e:
                logger.error(f"Failed to fetch stock data for {symbol}: {e}")
                errors[symbol] = str(e)
        
        return DataResponse(
            data=all_data,
            metadata={
                'source': self.name,
                'request_time': datetime.now().isoformat(),
                'symbols_requested': request.symbols,
                'symbols_fetched': list(all_data.keys()),
                'symbols_failed': list(errors.keys()),
                'data_type': request.data_type,
                'database_used': database,
                'api_calls_made': len(request.symbols)
            },
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )
    
    def _fetch_fundamentals_data(self, request: DataRequest) -> DataResponse:
        """Fetch fundamental data using Sharadar dataset"""
        all_data = {}
        errors = {}
        
        # Use Sharadar Fundamentals (SF1) database
        database = 'SF1'
        
        for symbol in request.symbols:
            try:
                with Timer(f"fetch_fundamentals_{symbol}"):
                    # Fetch multiple fundamental metrics for the symbol
                    fundamental_data = self._fetch_fundamentals_for_symbol(symbol, request)
                
                if fundamental_data is not None and not fundamental_data.empty:
                    all_data[symbol] = fundamental_data
                else:
                    errors[symbol] = "No fundamental data available"
                
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
                'database_used': database,
                'symbols_requested': request.symbols,
                'symbols_fetched': list(all_data.keys()),
                'symbols_failed': list(errors.keys())
            },
            errors=errors if errors else None
        )
    
    def _fetch_economic_data(self, request: DataRequest) -> DataResponse:
        """Fetch economic data"""
        all_data = {}
        errors = {}
        
        for symbol in request.symbols:
            try:
                with Timer(f"fetch_economic_{symbol}"):
                    # Resolve economic indicator to Quandl code
                    quandl_code = self._resolve_economic_indicator(symbol)
                    economic_data = self._fetch_time_series(quandl_code, request)
                
                if economic_data is not None and not economic_data.empty:
                    all_data[symbol] = economic_data
                else:
                    errors[symbol] = "No economic data available"
                
                # Apply rate limiting
                self._enforce_rate_limit()
                
            except Exception as e:
                logger.error(f"Failed to fetch economic data for {symbol}: {e}")
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
    
    def _fetch_alternative_data(self, request: DataRequest) -> DataResponse:
        """Fetch alternative datasets"""
        # This would be similar to other fetch methods but for alternative data
        # Implementation depends on specific alternative datasets requested
        raise DataValidationError(
            "Alternative data fetching requires specific dataset configuration",
            suggestions=[
                "Specify Quandl dataset code directly",
                "Use supported alternative data indicators",
                "Contact support for custom alternative data setup"
            ]
        )
    
    @retry(max_attempts=3, delay=2.0, exponential_backoff=True)
    def _fetch_time_series(self, quandl_code: str, request: DataRequest) -> Optional[pd.DataFrame]:
        """Fetch time series data for a Quandl code"""
        try:
            # Build URL
            url = f"{self.base_url}/datasets/{quandl_code}/data.json"
            
            # Build parameters
            params = {
                'api_key': self.api_key,
                'order': 'asc'  # Oldest first
            }
            
            # Add date range if specified
            if request.start_date:
                params['start_date'] = request.start_date.strftime('%Y-%m-%d')
            
            if request.end_date:
                params['end_date'] = request.end_date.strftime('%Y-%m-%d')
            
            # Add frequency transformation if supported
            frequency = request.frequency or request.interval
            if frequency and frequency in self.frequency_mapping:
                params['collapse'] = self.frequency_mapping[frequency]
            
            logger.debug(f"Fetching Quandl time series {quandl_code}")
            
            # Make API call
            response_data = self._make_api_call(url, params)
            
            # Parse response
            df = self._parse_time_series_response(response_data, quandl_code)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch time series {quandl_code}: {e}")
            raise
    
    def _parse_time_series_response(self, data: Dict[str, Any], quandl_code: str) -> Optional[pd.DataFrame]:
        """Parse Quandl time series response"""
        try:
            if not data or 'dataset_data' not in data:
                logger.warning(f"No dataset_data in response for {quandl_code}")
                return None
            
            dataset_data = data['dataset_data']
            
            if 'data' not in dataset_data or not dataset_data['data']:
                logger.warning(f"No data in response for {quandl_code}")
                return None
            
            # Get column names and data
            column_names = dataset_data.get('column_names', [])
            data_rows = dataset_data['data']
            
            if not column_names or not data_rows:
                logger.warning(f"Empty data or column names for {quandl_code}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=column_names)
            
            # Parse date column (usually first column)
            date_column = column_names[0]
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
            
            # Convert numeric columns
            for col in df.columns:
                if col != date_column:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            # Sort by date
            df = df.sort_index()
            
            if df.empty:
                logger.warning(f"No valid data after cleaning for {quandl_code}")
                return None
            
            logger.debug(f"Parsed {len(df)} observations for {quandl_code}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse response for {quandl_code}: {e}")
            return None
    
    def _make_api_call(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call with error handling"""
        try:
            logger.debug(f"Making Quandl API call: {url}")
            
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
                    "Quandl API rate limit exceeded",
                    retry_after=60
                )
            elif response.status_code == 401:
                raise AuthenticationError(
                    "Invalid Quandl API key",
                    suggestions=["Check your API key", "Verify API key permissions"]
                )
            elif response.status_code == 403:
                raise AuthenticationError(
                    "Quandl API access denied",
                    suggestions=[
                        "Check your subscription plan",
                        "Verify dataset permissions",
                        "Contact Quandl support"
                    ]
                )
            elif response.status_code == 404:
                raise DataValidationError(
                    "Quandl dataset not found",
                    suggestions=["Check dataset code", "Verify symbol format"]
                )
            elif response.status_code != 200:
                raise ExternalAPIError(
                    f"Quandl API error: HTTP {response.status_code}",
                    api_name="quandl",
                    status_code=response.status_code
                )
            
            # Parse JSON response
            data = response.json()
            
            # Check for API errors
            if 'quandl_error' in data:
                error_info = data['quandl_error']
                error_code = error_info.get('code', 'Unknown')
                error_message = error_info.get('message', 'Unknown error')
                
                raise ExternalAPIError(
                    f"Quandl API error {error_code}: {error_message}",
                    api_name="quandl"
                )
            
            return data
            
        except requests.exceptions.Timeout:
            raise ExternalAPIError("Quandl API timeout", api_name="quandl")
        except requests.exceptions.ConnectionError:
            raise ExternalAPIError("Failed to connect to Quandl API", api_name="quandl")
        except json.JSONDecodeError:
            raise ExternalAPIError("Invalid JSON response from Quandl API", api_name="quandl")
    
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
                time.sleep(wait_time + 1)
    
    def _determine_stock_database(self, symbols: List[str]) -> str:
        """Determine appropriate stock database based on symbols"""
        # Simple heuristic based on symbol format
        for symbol in symbols:
            if '.NS' in symbol or '.BO' in symbol:
                return 'NSE'  # Indian stocks
            elif len(symbol) <= 5 and symbol.isalpha():
                return 'EOD'  # US stocks
        
        # Default to EOD for US stocks
        return 'EOD'
    
    def _build_stock_code(self, database: str, symbol: str) -> str:
        """Build Quandl code for stock data"""
        # Clean symbol
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '').upper()
        
        if database == 'EOD':
            return f"EOD/{clean_symbol}"
        elif database == 'NSE':
            return f"NSE/{clean_symbol}"
        elif database == 'BSE':
            return f"BSE/{clean_symbol}"
        else:
            return f"{database}/{clean_symbol}"
    
    def _process_stock_data(self, df: pd.DataFrame, symbol: str, database: str) -> Optional[pd.DataFrame]:
        """Process and standardize stock data"""
        if df is None or df.empty:
            return None
        
        # Standard column mappings for different databases
        column_mappings = {
            'EOD': {
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume',
                'Adj_Open': 'Adj_Open',
                'Adj_High': 'Adj_High',
                'Adj_Low': 'Adj_Low',
                'Adj_Close': 'Adj_Close',
                'Adj_Volume': 'Adj_Volume'
            },
            'NSE': {
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Last': 'Last',
                'Total Trade Quantity': 'Volume',
                'Turnover (Lacs)': 'Turnover'
            }
        }
        
        # Apply column mapping if available
        if database in column_mappings:
            available_mappings = {
                old_col: new_col for old_col, new_col in column_mappings[database].items()
                if old_col in df.columns
            }
            df = df.rename(columns=available_mappings)
        
        # Standardize DataFrame
        df = standardize_dataframe(df)
        
        return df
    
    def _fetch_fundamentals_for_symbol(self, symbol: str, request: DataRequest) -> Optional[pd.DataFrame]:
        """Fetch fundamental data for a symbol using SF1 dataset"""
        try:
            # Common fundamental metrics
            metrics = ['REVENUE', 'NETINC', 'ASSETS', 'DEBT', 'MARKETCAP', 'PE', 'PB', 'ROE']
            
            fundamental_data = {}
            
            for metric in metrics:
                try:
                    quandl_code = f"SF1/{symbol}_{metric}_ARQ"  # Annual Reported Quarterly
                    metric_data = self._fetch_time_series(quandl_code, request)
                    
                    if metric_data is not None and not metric_data.empty:
                        # Take the value column (usually the second column)
                        value_col = metric_data.columns[0] if len(metric_data.columns) > 0 else 'Value'
                        fundamental_data[metric] = metric_data[value_col]
                    
                    # Small delay between metric requests
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.debug(f"Could not fetch {metric} for {symbol}: {e}")
                    continue
            
            if fundamental_data:
                # Combine all metrics into single DataFrame
                df = pd.DataFrame(fundamental_data)
                df.index.name = 'Date'
                return df
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch fundamentals for {symbol}: {e}")
            return None
    
    def _resolve_economic_indicator(self, indicator: str) -> str:
        """Resolve economic indicator name to Quandl code"""
        if indicator.lower() in self.common_indicators:
            database, code = self.common_indicators[indicator.lower()]
            return f"{database}/{code}"
        else:
            # Assume it's already a Quandl code
            return indicator.upper()
    
    def test_connection(self) -> bool:
        """Test API connection and key validity"""
        try:
            logger.info("Testing Quandl connection...")
            
            # Try to fetch metadata for a simple, reliable dataset
            url = f"{self.base_url}/datasets/FED/GDP.json"
            params = {'api_key': self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'dataset' in data:
                    logger.info("Quandl connection test successful")
                    return True
                else:
                    logger.warning("Quandl connection test returned unexpected format")
                    return False
            elif response.status_code == 401:
                logger.error("Quandl authentication failed - check API key")
                return False
            elif response.status_code == 403:
                logger.error("Quandl access denied - check subscription")
                return False
            else:
                logger.error(f"Quandl connection test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Quandl connection test failed: {e}")
            return False
    
    def search_datasets(self, query: str, per_page: int = 10) -> List[Dict[str, Any]]:
        """Search for datasets on Quandl"""
        try:
            url = f"{self.base_url}/datasets.json"
            params = {
                'query': query,
                'per_page': per_page,
                'api_key': self.api_key
            }
            
            data = self._make_api_call(url, params)
            
            if data and 'datasets' in data:
                results = []
                for dataset in data['datasets']:
                    results.append({
                        'database_code': dataset.get('database_code'),
                        'dataset_code': dataset.get('dataset_code'),
                        'name': dataset.get('name'),
                        'description': dataset.get('description', '')[:200] + '...' if dataset.get('description') else '',
                        'frequency': dataset.get('frequency'),
                        'newest_available_date': dataset.get('newest_available_date'),
                        'oldest_available_date': dataset.get('oldest_available_date'),
                        'premium': dataset.get('premium', False)
                    })
                return results
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to search datasets for '{query}': {e}")
            return []
    
    def get_dataset_info(self, dataset_code: str) -> Dict[str, Any]:
        """Get information about a specific dataset"""
        try:
            url = f"{self.base_url}/datasets/{dataset_code}.json"
            params = {'api_key': self.api_key}
            
            data = self._make_api_call(url, params)
            
            if data and 'dataset' in data:
                dataset = data['dataset']
                return {
                    'database_code': dataset.get('database_code'),
                    'dataset_code': dataset.get('dataset_code'),
                    'name': dataset.get('name'),
                    'description': dataset.get('description'),
                    'frequency': dataset.get('frequency'),
                    'column_names': dataset.get('column_names', []),
                    'newest_available_date': dataset.get('newest_available_date'),
                    'oldest_available_date': dataset.get('oldest_available_date'),
                    'premium': dataset.get('premium', False),
                    'refreshed_at': dataset.get('refreshed_at')
                }
            else:
                return {'error': f'Dataset {dataset_code} not found'}
                
        except Exception as e:
            logger.error(f"Failed to get dataset info for {dataset_code}: {e}")
            return {'error': str(e)}
    
    def get_supported_datasets(self) -> Dict[str, str]:
        """Get list of supported dataset mappings"""
        return self.dataset_mappings.copy()
    
    def get_supported_indicators(self) -> Dict[str, Tuple[str, str]]:
        """Get list of supported economic indicators"""
        return self.common_indicators.copy()
    
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
            'supported_datasets': len(self.dataset_mappings),
            'supported_indicators': len(self.common_indicators)
        }
