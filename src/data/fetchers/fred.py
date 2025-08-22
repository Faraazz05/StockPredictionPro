# ============================================
# StockPredictionPro - src/data/fetchers/fred.py
# FRED (Federal Reserve Economic Data) API fetcher
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
from ...utils.helpers import retry, format_duration
from .base_fetcher import (
    BaseFetcher, DataRequest, DataResponse, FetcherCapabilities,
    DataType, DataFrequency, standardize_dataframe
)

logger = get_logger('data.fetchers.fred')

class FREDFetcher(BaseFetcher):
    """
    FRED (Federal Reserve Economic Data) API fetcher
    
    Features:
    - Access to 800,000+ economic time series
    - Federal Reserve and other government economic data
    - GDP, inflation, unemployment, interest rates, etc.
    - High-quality, official economic indicators
    - No cost API with reasonable rate limits
    - Historical data going back decades
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED fetcher
        
        Args:
            api_key: FRED API key (will use from config if None)
        """
        super().__init__(name="fred")
        
        # Get API key from config or parameter
        self.api_key = api_key or get('app_config', 'data_sources.fred.api_key', '')
        
        if not self.api_key:
            raise AuthenticationError(
                "FRED API key not found",
                suggestions=[
                    "Get free API key from https://fred.stlouisfed.org/docs/api/api_key.html",
                    "Set FRED_API_KEY in .env file",
                    "Configure app_config.yaml data_sources.fred.api_key",
                    "Pass api_key parameter to constructor"
                ]
            )
        
        # API configuration
        self.base_url = "https://api.stlouisfed.org/fred"
        self.timeout = 30
        self.max_retries = 3
        
        # Rate limiting (FRED is generous with limits)
        self.rate_limit_calls = 120  # 120 calls per minute
        self.rate_limit_period = 60  # seconds
        self.call_timestamps: List[float] = []
        
        # Common FRED series IDs for economic indicators
        self.economic_indicators = {
            # GDP and Growth
            'gdp': 'GDP',                          # Gross Domestic Product
            'gdp_real': 'GDPC1',                   # Real GDP
            'gdp_growth': 'A191RL1Q225SBEA',       # GDP Growth Rate
            'gdp_per_capita': 'A939RX0Q048SBEA',   # Real GDP Per Capita
            
            # Inflation
            'cpi': 'CPIAUCSL',                     # Consumer Price Index
            'cpi_core': 'CPILFESL',                # Core CPI (ex food & energy)
            'pce': 'PCEPI',                        # PCE Price Index
            'pce_core': 'PCEPILFE',                # Core PCE Price Index
            'inflation_rate': 'T10YIE',            # 10-Year Breakeven Inflation Rate
            
            # Employment
            'unemployment': 'UNRATE',               # Unemployment Rate
            'employment': 'PAYEMS',                 # Total Nonfarm Payrolls
            'labor_participation': 'CIVPART',      # Labor Force Participation Rate
            'jobless_claims': 'ICSA',              # Initial Jobless Claims
            'continuing_claims': 'CCSA',           # Continuing Claims
            
            # Interest Rates
            'fed_funds': 'FEDFUNDS',               # Federal Funds Rate
            'treasury_10y': 'GS10',                # 10-Year Treasury Rate
            'treasury_2y': 'GS2',                  # 2-Year Treasury Rate
            'treasury_3m': 'GS3M',                 # 3-Month Treasury Rate
            'yield_curve': 'T10Y2Y',               # 10-Year Treasury - 2-Year Treasury
            
            # Money Supply
            'm1': 'M1SL',                          # M1 Money Stock
            'm2': 'M2SL',                          # M2 Money Stock
            'money_velocity': 'M2V',               # Velocity of M2 Money Stock
            
            # Housing
            'housing_starts': 'HOUST',             # Housing Starts
            'home_prices': 'CSUSHPISA',            # S&P/Case-Shiller Home Price Index
            'mortgage_rates': 'MORTGAGE30US',      # 30-Year Fixed Rate Mortgage
            'building_permits': 'PERMIT',          # New Private Housing Permits
            
            # Consumer and Business
            'consumer_sentiment': 'UMCSENT',       # Consumer Sentiment Index
            'retail_sales': 'RSAFS',               # Retail Sales
            'industrial_production': 'INDPRO',     # Industrial Production Index
            'capacity_utilization': 'TCU',         # Capacity Utilization
            'business_inventories': 'BUSINV',      # Total Business Inventories
            
            # International Trade
            'trade_balance': 'BOPGSTB',            # Trade Balance: Goods and Services
            'exports': 'EXPGS',                    # Exports of Goods and Services
            'imports': 'IMPGS',                    # Imports of Goods and Services
            'dollar_index': 'DTWEXBGS',            # Trade Weighted U.S. Dollar Index
            
            # Financial Markets
            'corporate_aaa': 'AAA',                # Moody's AAA Corporate Bond Yield
            'corporate_baa': 'BAA',                # Moody's BAA Corporate Bond Yield
            'credit_spread': 'BAA10Y',             # BAA-10 Year Treasury Spread
            'vix': 'VIXCLS',                       # CBOE Volatility Index
            'sp500': 'SP500',                      # S&P 500
            
            # Commodities
            'oil_wti': 'DCOILWTICO',               # WTI Crude Oil Price
            'gold_price': 'GOLDAMGBD228NLBM',      # Gold Fixing Price
            'copper_price': 'PCOPPUSDM',           # Copper Price
            
            # Regional Indicators
            'philly_fed': 'PHIL',                  # Philadelphia Fed Business Outlook
            'empire_state': 'GACDISA066MSFRBNY',   # Empire State Manufacturing Survey
            'chicago_pmi': 'NAPM',                 # Chicago PMI
        }
        
        # Frequency mapping
        self.frequency_mapping = {
            DataFrequency.DAILY.value: 'd',
            DataFrequency.WEEKLY.value: 'w',
            DataFrequency.MONTHLY.value: 'm',
            DataFrequency.QUARTERLY.value: 'q',
            DataFrequency.ANNUALLY.value: 'a',
            'daily': 'd',
            'weekly': 'w',
            'monthly': 'm',
            'quarterly': 'q',
            'annual': 'a',
            'yearly': 'a'
        }
        
        logger.info(f"FRED fetcher initialized with API key: {self.api_key[:8]}...")
    
    def get_capabilities(self) -> FetcherCapabilities:
        """Get FRED fetcher capabilities"""
        return FetcherCapabilities(
            name=self.name,
            supported_data_types=[
                DataType.ECONOMIC_DATA.value,
                "economic_indicators",
                "macro_data"
            ],
            supported_frequencies=[
                DataFrequency.DAILY.value,
                DataFrequency.WEEKLY.value,
                DataFrequency.MONTHLY.value,
                DataFrequency.QUARTERLY.value,
                DataFrequency.ANNUALLY.value
            ],
            supported_markets=["US"],  # FRED primarily focuses on US data
            supports_intraday=False,
            supports_historical=True,
            supports_real_time=False,  # FRED data has reporting lags
            rate_limit_calls=120,      # 120 calls per minute
            rate_limit_period_seconds=60,
            max_symbols_per_request=1,  # FRED handles one series at a time
            max_historical_days=None,   # No limit on historical data
            requires_api_key=True,
            data_quality_score=0.99,   # Official government data - highest quality
            reliability_score=0.98     # Very reliable government service
        )
    
    def can_fetch(self, request: DataRequest) -> bool:
        """Check if this fetcher can handle the request"""
        # FRED only handles economic data
        if request.data_type not in [DataType.ECONOMIC_DATA.value, "economic_indicators", "macro_data"]:
            return False
        
        # Check if symbols are FRED series IDs or known indicators
        for symbol in request.symbols:
            # Check if it's a known indicator or looks like a FRED series ID
            if (symbol.lower() in self.economic_indicators or 
                len(symbol) > 2 and symbol.isalnum()):
                continue
            else:
                return False
        
        return True
    
    @time_it("fred_fetch", include_args=True)
    def fetch_data(self, request: DataRequest) -> DataResponse:
        """
        Fetch data from FRED API
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse with fetched data
        """
        logger.info(f"Fetching {request.data_type} data for {len(request.symbols)} series from FRED")
        
        try:
            # Validate request
            validation = self.validate_request(request)
            if not validation.is_valid:
                raise DataValidationError(
                    f"Request validation failed: {'; '.join(validation.errors)}"
                )
            
            # Pre-fetch hook
            request = self.pre_fetch_hook(request)
            
            # Convert symbol names to FRED series IDs
            fred_symbols = self._resolve_series_ids(request.symbols)
            
            # Fetch data for each series
            all_data = {}
            errors = {}
            warnings = {}
            
            for original_symbol, fred_id in fred_symbols.items():
                try:
                    with Timer(f"fetch_series_{fred_id}") as timer:
                        series_data = self._fetch_series_data(fred_id, request)
                    
                    if series_data is not None and not series_data.empty:
                        all_data[original_symbol] = series_data
                        logger.debug(f"Fetched {len(series_data)} records for {original_symbol} ({fred_id}) in {timer.result.duration_str}")
                    else:
                        errors[original_symbol] = "No data returned"
                    
                    # Apply rate limiting
                    self._enforce_rate_limit()
                    
                except Exception as e:
                    logger.error(f"Failed to fetch data for {original_symbol} ({fred_id}): {e}")
                    errors[original_symbol] = str(e)
            
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
                    'fred_series_mapping': fred_symbols,
                    'api_calls_made': len(fred_symbols)
                },
                errors=errors if errors else None,
                warnings=warnings if warnings else None
            )
            
            # Post-fetch hook
            response = self.post_fetch_hook(request, response)
            
            if not all_data:
                raise DataFetchError(
                    "No data could be fetched for any series",
                    context={'symbols': request.symbols, 'errors': errors}
                )
            
            logger.info(f"Successfully fetched data for {len(all_data)} series")
            return response
            
        except Exception as e:
            self.handle_error(e, request)
            raise DataFetchError(
                f"FRED fetch failed: {str(e)}",
                source="fred",
                context={'request': request.to_dict()}
            ) from e
    
    def _resolve_series_ids(self, symbols: List[str]) -> Dict[str, str]:
        """Convert symbol names to FRED series IDs"""
        fred_symbols = {}
        
        for symbol in symbols:
            # Check if it's a known indicator name
            if symbol.lower() in self.economic_indicators:
                fred_symbols[symbol] = self.economic_indicators[symbol.lower()]
            else:
                # Assume it's already a FRED series ID
                fred_symbols[symbol] = symbol.upper()
        
        return fred_symbols
    
    @retry(max_attempts=3, delay=1.0, exponential_backoff=True)
    def _fetch_series_data(self, series_id: str, request: DataRequest) -> Optional[pd.DataFrame]:
        """Fetch data for a single FRED series"""
        try:
            # Build URL for series observations
            url = f"{self.base_url}/series/observations"
            
            # Build parameters
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'sort_order': 'asc'
            }
            
            # Add date range if specified
            if request.start_date:
                params['observation_start'] = request.start_date.strftime('%Y-%m-%d')
            
            if request.end_date:
                params['observation_end'] = request.end_date.strftime('%Y-%m-%d')
            
            # Add frequency if supported
            frequency = request.frequency or request.interval
            if frequency and frequency in self.frequency_mapping:
                params['frequency'] = self.frequency_mapping[frequency]
            
            logger.debug(f"Fetching FRED series {series_id} with params: {params}")
            
            # Make API call
            response_data = self._make_api_call(url, params)
            
            # Parse response
            df = self._parse_series_response(response_data, series_id)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch FRED series {series_id}: {e}")
            raise
    
    def _parse_series_response(self, data: Dict[str, Any], series_id: str) -> Optional[pd.DataFrame]:
        """Parse FRED series response"""
        try:
            if not data or 'observations' not in data:
                logger.warning(f"No observations returned for series {series_id}")
                return None
            
            observations = data['observations']
            if not observations:
                logger.warning(f"Empty observations for series {series_id}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(observations)
            
            # Parse dates
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Handle missing values (FRED uses '.' for missing values)
            df['value'] = df['value'].replace('.', np.nan)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Create standardized column name
            df = df.rename(columns={'value': series_id})
            
            # Remove rows with missing values
            df = df.dropna()
            
            # Sort by date
            df = df.sort_index()
            
            if df.empty:
                logger.warning(f"No valid data after cleaning for series {series_id}")
                return None
            
            logger.debug(f"Parsed {len(df)} observations for series {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse response for series {series_id}: {e}")
            return None
    
    def _make_api_call(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call with error handling"""
        try:
            logger.debug(f"Making FRED API call: {url}")
            
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
                    "FRED API rate limit exceeded",
                    retry_after=60
                )
            elif response.status_code == 400:
                raise DataValidationError(
                    f"FRED API bad request: {response.text}",
                    suggestions=["Check series ID", "Verify date format", "Check API parameters"]
                )
            elif response.status_code == 401:
                raise AuthenticationError(
                    "Invalid FRED API key",
                    suggestions=["Check your API key", "Get free key from FRED website"]
                )
            elif response.status_code != 200:
                raise ExternalAPIError(
                    f"FRED API error: HTTP {response.status_code}",
                    api_name="fred",
                    status_code=response.status_code
                )
            
            # Parse JSON response
            data = response.json()
            
            # Check for API errors
            if 'error_code' in data:
                error_code = data.get('error_code')
                error_message = data.get('error_message', 'Unknown API error')
                
                if error_code == 400:
                    raise DataValidationError(f"FRED API error: {error_message}")
                elif error_code == 429:
                    raise RateLimitError(f"FRED API rate limit: {error_message}")
                else:
                    raise ExternalAPIError(f"FRED API error {error_code}: {error_message}", api_name="fred")
            
            return data
            
        except requests.exceptions.Timeout:
            raise ExternalAPIError("FRED API timeout", api_name="fred")
        except requests.exceptions.ConnectionError:
            raise ExternalAPIError("Failed to connect to FRED API", api_name="fred")
        except json.JSONDecodeError:
            raise ExternalAPIError("Invalid JSON response from FRED API", api_name="fred")
    
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
    
    def test_connection(self) -> bool:
        """Test API connection and key validity"""
        try:
            logger.info("Testing FRED connection...")
            
            # Try to fetch a simple, reliable series (GDP)
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': 'GDP',
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': 1  # Just get one observation
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'observations' in data and data['observations']:
                    logger.info("FRED connection test successful")
                    return True
                else:
                    logger.warning("FRED connection test returned no data")
                    return False
            elif response.status_code == 401:
                logger.error("FRED authentication failed - check API key")
                return False
            else:
                logger.error(f"FRED connection test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"FRED connection test failed: {e}")
            return False
    
    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """Get information about a FRED series"""
        try:
            url = f"{self.base_url}/series"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            data = self._make_api_call(url, params)
            
            if data and 'seriess' in data and data['seriess']:
                series_info = data['seriess'][0]
                return {
                    'id': series_info.get('id'),
                    'title': series_info.get('title'),
                    'observation_start': series_info.get('observation_start'),
                    'observation_end': series_info.get('observation_end'),
                    'frequency': series_info.get('frequency'),
                    'frequency_short': series_info.get('frequency_short'),
                    'units': series_info.get('units'),
                    'seasonal_adjustment': series_info.get('seasonal_adjustment'),
                    'last_updated': series_info.get('last_updated'),
                    'notes': series_info.get('notes', '')[:200] + '...' if series_info.get('notes') else ''
                }
            else:
                return {'error': f'Series {series_id} not found'}
                
        except Exception as e:
            logger.error(f"Failed to get series info for {series_id}: {e}")
            return {'error': str(e)}
    
    def search_series(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for FRED series"""
        try:
            url = f"{self.base_url}/series/search"
            params = {
                'search_text': search_text,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'popularity'
            }
            
            data = self._make_api_call(url, params)
            
            if data and 'seriess' in data:
                results = []
                for series in data['seriess']:
                    results.append({
                        'id': series.get('id'),
                        'title': series.get('title'),
                        'frequency': series.get('frequency_short'),
                        'units': series.get('units'),
                        'last_updated': series.get('last_updated'),
                        'popularity': series.get('popularity'),
                        'observation_start': series.get('observation_start'),
                        'observation_end': series.get('observation_end')
                    })
                return results
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to search series for '{search_text}': {e}")
            return []
    
    def get_supported_indicators(self) -> Dict[str, str]:
        """Get list of supported economic indicators"""
        return self.economic_indicators.copy()
    
    def get_categories(self) -> List[Dict[str, Any]]:
        """Get FRED data categories"""
        try:
            url = f"{self.base_url}/category"
            params = {
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            data = self._make_api_call(url, params)
            
            if data and 'categories' in data:
                return data['categories']
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []
    
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
            'supported_indicators': len(self.economic_indicators),
            'data_quality': 'Official government data - highest quality'
        }
    
    def get_economic_dashboard(self) -> Dict[str, Any]:
        """Get key economic indicators for dashboard"""
        key_indicators = [
            'gdp_growth', 'unemployment', 'cpi', 'fed_funds', 
            'treasury_10y', 'sp500', 'consumer_sentiment'
        ]
        
        dashboard_data = {}
        
        for indicator in key_indicators:
            try:
                # Get latest value
                series_id = self.economic_indicators.get(indicator)
                if series_id:
                    url = f"{self.base_url}/series/observations"
                    params = {
                        'series_id': series_id,
                        'api_key': self.api_key,
                        'file_type': 'json',
                        'limit': 1,
                        'sort_order': 'desc'
                    }
                    
                    data = self._make_api_call(url, params)
                    
                    if data and 'observations' in data and data['observations']:
                        obs = data['observations'][0]
                        if obs['value'] != '.':
                            dashboard_data[indicator] = {
                                'value': float(obs['value']),
                                'date': obs['date'],
                                'series_id': series_id
                            }
                    
                    # Small delay to respect rate limits
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.warning(f"Failed to get {indicator}: {e}")
                continue
        
        return dashboard_data
