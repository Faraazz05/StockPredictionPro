# ============================================
# StockPredictionPro - src/data/fetchers/base_fetcher.py
# Base classes and interfaces for data fetchers
# ============================================

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Protocol, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

from ...utils.exceptions import (
    DataFetchError, InvalidParameterError, DataValidationError,
    ExternalAPIError
)
from ...utils.logger import get_logger
from ...utils.validators import ValidationResult
from ...utils.helpers import validate_symbols

logger = get_logger('data.fetchers.base')

# ============================================
# Enums and Constants
# ============================================

class DataType(Enum):
    """Supported data types"""
    STOCK_DATA = "stock_data"
    OPTIONS_DATA = "options_data"
    FUNDAMENTALS = "fundamentals"
    ECONOMIC_DATA = "economic_data"
    NEWS_DATA = "news_data"
    CRYPTO_DATA = "crypto_data"
    FOREX_DATA = "forex_data"
    COMMODITIES = "commodities"

class DataFrequency(Enum):
    """Data frequency options"""
    TICK = "tick"
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1h"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class FetcherStatus(Enum):
    """Fetcher status states"""
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    UNAUTHORIZED = "unauthorized"

# ============================================
# Data Structures
# ============================================

@dataclass
class DataRequest:
    """
    Standardized data request specification
    
    This class defines what data should be fetched and how it should be processed
    """
    # Required fields
    symbols: List[str]
    data_type: str  # Can be DataType enum or string
    
    # Optional time range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Data frequency/interval
    frequency: Optional[str] = DataFrequency.DAILY.value
    interval: Optional[str] = None  # For backwards compatibility
    
    # Data options
    adjusted: bool = True  # Use adjusted prices
    include_dividends: bool = False
    include_splits: bool = False
    
    # Fetcher preferences
    preferred_fetchers: List[str] = field(default_factory=list)
    exclude_fetchers: List[str] = field(default_factory=list)
    
    # Quality requirements
    min_data_points: int = 50
    max_missing_ratio: float = 0.1  # 10% maximum missing data
    
    # Metadata
    request_id: Optional[str] = None
    priority: int = 1  # 1=low, 2=normal, 3=high
    timeout_seconds: int = 300  # 5 minutes default
    
    # Custom parameters for specific fetchers
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and clean request after initialization"""
        # Validate and clean symbols
        if isinstance(self.symbols, str):
            self.symbols = [s.strip() for s in self.symbols.split(',') if s.strip()]
        
        self.symbols = [s.strip().upper() for s in self.symbols if s.strip()]
        
        if not self.symbols:
            raise InvalidParameterError(
                "At least one symbol must be provided",
                parameter_name="symbols",
                provided_value=self.symbols
            )
        
        # Convert data_type to string if enum
        if hasattr(self.data_type, 'value'):
            self.data_type = self.data_type.value
        
        # Set interval from frequency if not set
        if not self.interval and self.frequency:
            self.interval = self.frequency
        
        # Validate date range
        if self.start_date and self.end_date:
            if self.start_date >= self.end_date:
                raise InvalidParameterError(
                    "Start date must be before end date",
                    parameter_name="date_range",
                    provided_value=f"{self.start_date} to {self.end_date}"
                )
        
        # Set default end date to today if start date is set
        if self.start_date and not self.end_date:
            self.end_date = datetime.now()
    
    def get_date_range_days(self) -> Optional[int]:
        """Get the number of days in the date range"""
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days
        return None
    
    def is_intraday(self) -> bool:
        """Check if request is for intraday data"""
        intraday_frequencies = [
            DataFrequency.TICK.value, DataFrequency.MINUTE_1.value,
            DataFrequency.MINUTE_5.value, DataFrequency.MINUTE_15.value,
            DataFrequency.MINUTE_30.value, DataFrequency.HOUR_1.value
        ]
        return self.frequency in intraday_frequencies or (
            self.interval and any(freq in self.interval for freq in ['min', 'tick', 'h'])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbols': self.symbols,
            'data_type': self.data_type,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'frequency': self.frequency,
            'interval': self.interval,
            'adjusted': self.adjusted,
            'include_dividends': self.include_dividends,
            'include_splits': self.include_splits,
            'preferred_fetchers': self.preferred_fetchers,
            'exclude_fetchers': self.exclude_fetchers,
            'min_data_points': self.min_data_points,
            'max_missing_ratio': self.max_missing_ratio,
            'request_id': self.request_id,
            'priority': self.priority,
            'timeout_seconds': self.timeout_seconds,
            'custom_params': self.custom_params
        }

@dataclass
class DataResponse:
    """
    Standardized data response from fetchers
    
    This class contains the fetched data and metadata about the operation
    """
    # Main data (dict of symbol -> DataFrame)
    data: Dict[str, pd.DataFrame]
    
    # Response metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Errors encountered during fetching
    errors: Optional[Dict[str, str]] = None
    
    # Warnings (non-fatal issues)
    warnings: Optional[Dict[str, str]] = None
    
    # Fetcher information
    fetcher_name: Optional[str] = None
    fetch_time: Optional[datetime] = None
    
    # Data quality metrics
    quality_score: Optional[float] = None
    validation_results: Optional[Dict[str, ValidationResult]] = None
    
    def __post_init__(self):
        """Set default values after initialization"""
        if self.fetch_time is None:
            self.fetch_time = datetime.now()
        
        # Ensure data is not None
        if self.data is None:
            self.data = {}
    
    def get_symbols(self) -> List[str]:
        """Get list of symbols in the response"""
        return list(self.data.keys())
    
    def get_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get data for a specific symbol"""
        return self.data.get(symbol)
    
    def has_data(self) -> bool:
        """Check if response contains any data"""
        return bool(self.data) and any(
            df is not None and not df.empty 
            for df in self.data.values()
        )
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of data in response"""
        summary = {}
        
        for symbol, df in self.data.items():
            if df is not None and not df.empty:
                summary[symbol] = {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'date_range': {
                        'start': df.index[0].isoformat() if hasattr(df.index[0], 'isoformat') else str(df.index[0]),
                        'end': df.index[-1].isoformat() if hasattr(df.index[-1], 'isoformat') else str(df.index[-1])
                    } if len(df) > 0 else None,
                    'missing_data': df.isnull().sum().to_dict() if hasattr(df, 'isnull') else {}
                }
            else:
                summary[symbol] = {'rows': 0, 'columns': [], 'date_range': None}
        
        return summary
    
    def merge_responses(self, other: 'DataResponse') -> 'DataResponse':
        """Merge with another DataResponse"""
        merged_data = {**self.data, **other.data}
        merged_errors = {}
        merged_warnings = {}
        
        if self.errors:
            merged_errors.update(self.errors)
        if other.errors:
            merged_errors.update(other.errors)
        
        if self.warnings:
            merged_warnings.update(self.warnings)
        if other.warnings:
            merged_warnings.update(other.warnings)
        
        return DataResponse(
            data=merged_data,
            metadata={**self.metadata, **other.metadata},
            errors=merged_errors if merged_errors else None,
            warnings=merged_warnings if merged_warnings else None,
            fetcher_name=f"{self.fetcher_name},{other.fetcher_name}",
            fetch_time=max(self.fetch_time or datetime.min, other.fetch_time or datetime.min)
        )

@dataclass
class FetcherCapabilities:
    """
    Describes what a fetcher can do
    """
    name: str
    supported_data_types: List[str]
    supported_frequencies: List[str]
    supported_markets: List[str]  # US, IN, UK, etc.
    
    # Operational capabilities
    supports_intraday: bool = False
    supports_historical: bool = True
    supports_real_time: bool = False
    
    # Rate limiting info
    rate_limit_calls: Optional[int] = None
    rate_limit_period_seconds: Optional[int] = None
    
    # Data limitations
    max_symbols_per_request: int = 1
    max_historical_days: Optional[int] = None
    requires_api_key: bool = False
    
    # Quality indicators
    data_quality_score: float = 0.8  # 0.0 to 1.0
    reliability_score: float = 0.8   # 0.0 to 1.0
    
    def can_handle_request(self, request: DataRequest) -> bool:
        """Check if this fetcher can handle the request"""
        # Check data type
        if request.data_type not in self.supported_data_types:
            return False
        
        # Check frequency
        if request.frequency and request.frequency not in self.supported_frequencies:
            return False
        
        # Check intraday support
        if request.is_intraday() and not self.supports_intraday:
            return False
        
        # Check symbol count
        if len(request.symbols) > self.max_symbols_per_request:
            return False
        
        # Check historical range
        if (self.max_historical_days and 
            request.get_date_range_days() and 
            request.get_date_range_days() > self.max_historical_days):
            return False
        
        return True

# ============================================
# Base Fetcher Interface
# ============================================

class BaseFetcher(ABC):
    """
    Abstract base class for all data fetchers
    
    This class defines the interface that all data fetchers must implement.
    It provides common functionality and ensures consistency across different
    data sources.
    """
    
    def __init__(self, name: str):
        """
        Initialize base fetcher
        
        Args:
            name: Unique name for this fetcher
        """
        self.name = name
        self.logger = get_logger(f'data.fetchers.{name}')
        self.status = FetcherStatus.UNKNOWN
        self.last_error: Optional[str] = None
        self.last_success_time: Optional[datetime] = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
    
    @abstractmethod
    def can_fetch(self, request: DataRequest) -> bool:
        """
        Check if this fetcher can handle the request
        
        Args:
            request: Data request to evaluate
            
        Returns:
            True if fetcher can handle the request
        """
        pass
    
    @abstractmethod
    def fetch_data(self, request: DataRequest) -> DataResponse:
        """
        Fetch data according to the request
        
        Args:
            request: Data request specification
            
        Returns:
            DataResponse with fetched data
            
        Raises:
            DataFetchError: If data fetching fails
            ExternalAPIError: If external API fails
            DataValidationError: If data validation fails
        """
        pass
    
    def get_capabilities(self) -> FetcherCapabilities:
        """
        Get fetcher capabilities
        
        Returns:
            FetcherCapabilities describing what this fetcher can do
        """
        # Default implementation - override in subclasses
        return FetcherCapabilities(
            name=self.name,
            supported_data_types=[DataType.STOCK_DATA.value],
            supported_frequencies=[DataFrequency.DAILY.value],
            supported_markets=["US"]
        )
    
    def validate_request(self, request: DataRequest) -> ValidationResult:
        """
        Validate a data request
        
        Args:
            request: Request to validate
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()
        
        # Basic validation
        if not request.symbols:
            result.add_error("No symbols provided")
        
        # Check if we can handle this request
        if not self.can_fetch(request):
            capabilities = self.get_capabilities()
            
            if request.data_type not in capabilities.supported_data_types:
                result.add_error(f"Data type '{request.data_type}' not supported")
            
            if request.frequency and request.frequency not in capabilities.supported_frequencies:
                result.add_error(f"Frequency '{request.frequency}' not supported")
            
            if len(request.symbols) > capabilities.max_symbols_per_request:
                result.add_error(f"Too many symbols: {len(request.symbols)} > {capabilities.max_symbols_per_request}")
        
        return result
    
    def pre_fetch_hook(self, request: DataRequest) -> DataRequest:
        """
        Hook called before fetching data
        
        Args:
            request: Original request
            
        Returns:
            Modified request (can be the same object)
        """
        self.total_requests += 1
        return request
    
    def post_fetch_hook(self, request: DataRequest, response: DataResponse) -> DataResponse:
        """
        Hook called after fetching data
        
        Args:
            request: Original request
            response: Fetch response
            
        Returns:
            Modified response (can be the same object)
        """
        if response.has_data():
            self.successful_requests += 1
            self.last_success_time = datetime.now()
            self.status = FetcherStatus.AVAILABLE
            self.last_error = None
        else:
            self.failed_requests += 1
            self.status = FetcherStatus.ERROR
            self.last_error = "No data returned"
        
        # Set fetcher name in response
        response.fetcher_name = self.name
        
        return response
    
    def handle_error(self, error: Exception, request: DataRequest) -> None:
        """
        Handle errors that occur during fetching
        
        Args:
            error: Exception that occurred
            request: Request that caused the error
        """
        self.failed_requests += 1
        self.last_error = str(error)
        
        # Update status based on error type
        if isinstance(error, ExternalAPIError):
            if "rate limit" in str(error).lower():
                self.status = FetcherStatus.RATE_LIMITED
            elif "unauthorized" in str(error).lower() or "authentication" in str(error).lower():
                self.status = FetcherStatus.UNAUTHORIZED
            else:
                self.status = FetcherStatus.ERROR
        else:
            self.status = FetcherStatus.ERROR
        
        self.logger.error(f"Fetch error in {self.name}: {error}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the fetcher
        
        Returns:
            Dictionary with status information
        """
        return {
            'name': self.name,
            'status': self.status.value,
            'last_error': self.last_error,
            'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.successful_requests / max(self.total_requests, 1) * 100,
            'capabilities': self.get_capabilities().__dict__
        }
    
    def test_connection(self) -> bool:
        """
        Test if the fetcher can connect to its data source
        
        Returns:
            True if connection is successful
        """
        # Default implementation - override in subclasses
        return True
    
    def reset_statistics(self):
        """Reset request statistics"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.last_error = None

# ============================================
# Fetcher Protocol for Type Checking
# ============================================

class FetcherProtocol(Protocol):
    """Protocol for type checking fetcher implementations"""
    
    name: str
    
    def can_fetch(self, request: DataRequest) -> bool: ...
    def fetch_data(self, request: DataRequest) -> DataResponse: ...
    def get_capabilities(self) -> FetcherCapabilities: ...
    def get_status(self) -> Dict[str, Any]: ...
    def test_connection(self) -> bool: ...

# ============================================
# Utility Functions
# ============================================

def create_standard_columns_mapping() -> Dict[str, str]:
    """
    Create mapping for standardizing column names across different fetchers
    
    Returns:
        Dictionary mapping various column name formats to standard names
    """
    return {
        # Price columns
        'Open': 'Open', 'open': 'Open', 'OPEN': 'Open',
        'High': 'High', 'high': 'High', 'HIGH': 'High',
        'Low': 'Low', 'low': 'Low', 'LOW': 'Low',
        'Close': 'Close', 'close': 'Close', 'CLOSE': 'Close',
        'Adj Close': 'Adj Close', 'adj_close': 'Adj Close', 'adjusted_close': 'Adj Close',
        'adjusted close': 'Adj Close', 'Adjusted Close': 'Adj Close',
        
        # Volume columns
        'Volume': 'Volume', 'volume': 'Volume', 'VOLUME': 'Volume',
        'vol': 'Volume', 'Vol': 'Volume',
        
        # Dividend columns
        'Dividends': 'Dividends', 'dividends': 'Dividends', 'dividend': 'Dividends',
        'Dividend': 'Dividends', 'div': 'Dividends',
        
        # Stock split columns
        'Stock Splits': 'Stock Splits', 'stock_splits': 'Stock Splits',
        'splits': 'Stock Splits', 'Split': 'Stock Splits', 'split': 'Stock Splits'
    }

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names and structure
    
    Args:
        df: DataFrame to standardize
        
    Returns:
        Standardized DataFrame
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Standardize column names
    column_mapping = create_standard_columns_mapping()
    df = df.rename(columns=column_mapping)
    
    # Ensure numeric columns are proper types
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            logger.warning("Could not convert index to datetime")
    
    # Sort by date
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    
    return df

def validate_dataframe_structure(df: pd.DataFrame, data_type: str = "stock_data") -> ValidationResult:
    """
    Validate DataFrame structure for different data types
    
    Args:
        df: DataFrame to validate
        data_type: Type of data expected
        
    Returns:
        ValidationResult with validation status
    """
    result = ValidationResult()
    
    if df is None:
        result.add_error("DataFrame is None")
        return result
    
    if df.empty:
        result.add_error("DataFrame is empty")
        return result
    
    if data_type == "stock_data":
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            result.add_error(f"Missing required columns: {missing_columns}")
        
        # Check for reasonable data
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                if (df[col] <= 0).any():
                    result.add_warning(f"Non-positive values found in {col}")
        
        # Check OHLC relationships
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_high = (df['High'] < df[['Open', 'Low', 'Close']].max(axis=1)).any()
            invalid_low = (df['Low'] > df[['Open', 'High', 'Close']].min(axis=1)).any()
            
            if invalid_high:
                result.add_error("High prices are lower than other OHLC values")
            if invalid_low:
                result.add_error("Low prices are higher than other OHLC values")
    
    # Check index
    if not isinstance(df.index, pd.DatetimeIndex):
        result.add_warning("Index is not a DatetimeIndex")
    
    return result

# ============================================
# Type Variables for Generic Fetchers
# ============================================

T_Request = TypeVar('T_Request', bound=DataRequest)
T_Response = TypeVar('T_Response', bound=DataResponse)

class GenericFetcher(BaseFetcher, Generic[T_Request, T_Response]):
    """Generic base fetcher for type-safe implementations"""
    pass
