# ============================================
# StockPredictionPro - src/api/schemas/data_schemas.py
# Comprehensive Pydantic schemas for data-related API endpoints with validation and examples
# ============================================

from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Union, Literal, Annotated
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# ============================================
# Type Aliases (Fixed for Python 3.9+ compatibility)
# ============================================

# Use Annotated instead of constr/conint/confloat
SymbolStr = Annotated[str, Field(min_length=1, max_length=12)]
CompanyNameStr = Annotated[str, Field(max_length=200)]
CurrencyStr = Annotated[str, Field(min_length=3, max_length=3)]
PositiveFloat = Annotated[float, Field(gt=0)]
PositiveInt = Annotated[int, Field(gt=0)]
LimitedInt = Annotated[int, Field(ge=1, le=100)]
QualityFloat = Annotated[float, Field(ge=0.0, le=1.0)]

# ============================================
# Enums and Constants
# ============================================

class MarketType(str, Enum):
    """Market types supported"""
    US = "US"
    NSE = "NSE"
    BSE = "BSE"
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"

class DataSource(str, Enum):
    """Data source providers"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    QUANDL = "quandl"
    FRED = "fred"

class TimeInterval(str, Enum):
    """Time intervals for data fetching"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1wk"
    MONTH_1 = "1mo"
    QUARTER_1 = "3mo"

class DataQuality(str, Enum):
    """Data quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class DataStatus(str, Enum):
    """Data status indicators"""
    LIVE = "live"
    DELAYED = "delayed"
    END_OF_DAY = "end_of_day"
    HISTORICAL = "historical"
    CACHED = "cached"

# ============================================
# Base Data Schemas
# ============================================

class BaseDataModel(BaseModel):
    """Base model for all data schemas with common configuration"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        populate_by_name=True,
        json_schema_extra={
            "examples": []
        }
    )

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp",
        examples=["2023-01-01T12:00:00Z"]
    )
    
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp",
        examples=["2023-01-01T12:30:00Z"]
    )

# ============================================
# Symbol and Market Data Schemas
# ============================================

class SymbolInfo(BaseDataModel):
    """Stock symbol information"""
    
    symbol: SymbolStr = Field(
        description="Stock symbol (ticker)",
        examples=["AAPL", "MSFT", "RELIANCE.NS"]
    )
    
    name: CompanyNameStr = Field(
        description="Company name",
        examples=["Apple Inc.", "Microsoft Corporation", "Reliance Industries Limited"]
    )
    
    sector: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Industry sector",
        examples=["Technology", "Energy", "Healthcare"]
    )
    
    industry: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Specific industry",
        examples=["Consumer Electronics", "Oil & Gas", "Software"]
    )
    
    market: MarketType = Field(
        description="Market exchange",
        examples=["US", "NSE"]
    )
    
    currency: CurrencyStr = Field(
        description="Currency code",
        examples=["USD", "INR"]
    )
    
    country: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Country of incorporation",
        examples=["United States", "India"]
    )
    
    market_cap: Optional[PositiveFloat] = Field(
        default=None,
        description="Market capitalization in billions",
        examples=[2800.5, 150.2]
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Company description",
        max_length=1000
    )
    
    is_active: bool = Field(
        default=True,
        description="Whether the symbol is actively traded"
    )
    
    @field_validator('symbol', mode='before')
    @classmethod
    def symbol_to_upper(cls, v: str) -> str:
        """Convert symbol to uppercase"""
        return v.upper().strip() if v else v
    
    @field_validator('currency', mode='before')
    @classmethod
    def currency_to_upper(cls, v: str) -> str:
        """Convert currency to uppercase"""
        return v.upper().strip() if v else v

class OHLCVData(BaseDataModel):
    """OHLCV (Open, High, Low, Close, Volume) data point"""
    
    timestamp: datetime = Field(
        description="Data timestamp",
        examples=["2023-01-01T09:30:00Z"]
    )
    
    open: PositiveFloat = Field(
        description="Opening price",
        examples=[150.25]
    )
    
    high: PositiveFloat = Field(
        description="Highest price",
        examples=[152.75]
    )
    
    low: PositiveFloat = Field(
        description="Lowest price",
        examples=[149.80]
    )
    
    close: PositiveFloat = Field(
        description="Closing price",
        examples=[151.50]
    )
    
    volume: PositiveInt = Field(
        description="Trading volume",
        examples=[1234567]
    )
    
    adjusted_close: Optional[PositiveFloat] = Field(
        default=None,
        description="Adjusted closing price for splits/dividends",
        examples=[151.45]
    )
    
    @model_validator(mode='after')
    def validate_ohlcv_consistency(self):
        """Validate OHLCV data consistency"""
        
        # High must be the highest price
        prices = [self.open, self.close]
        if self.adjusted_close:
            prices.append(self.adjusted_close)
        
        if self.high < max(prices):
            raise ValueError('High must be >= all other prices')
        
        if self.high < self.low:
            raise ValueError('High must be >= low price')
        
        # Low must be the lowest price
        if self.low > min(prices):
            raise ValueError('Low must be <= all other prices')
        
        return self

class MarketData(BaseDataModel, TimestampMixin):
    """Complete market data for a symbol"""
    
    symbol: SymbolStr = Field(
        description="Stock symbol",
        examples=["AAPL"]
    )
    
    data: List[OHLCVData] = Field(
        description="OHLCV data points",
        min_length=1
    )
    
    interval: TimeInterval = Field(
        description="Data time interval",
        examples=["1d"]
    )
    
    source: DataSource = Field(
        description="Data source provider",
        examples=["yahoo_finance"]
    )
    
    quality: DataQuality = Field(
        default=DataQuality.UNKNOWN,
        description="Data quality assessment"
    )
    
    status: DataStatus = Field(
        default=DataStatus.HISTORICAL,
        description="Data status"
    )
    
    delay_minutes: Optional[Annotated[int, Field(ge=0)]] = Field(
        default=None,
        description="Data delay in minutes for delayed feeds"
    )
    
    @field_validator('symbol', mode='before')
    @classmethod
    def symbol_to_upper(cls, v: str) -> str:
        """Convert symbol to uppercase"""
        return v.upper().strip() if v else v
    
    @field_validator('data')
    @classmethod
    def data_must_be_chronological(cls, v: List[OHLCVData]) -> List[OHLCVData]:
        """Validate that data points are in chronological order"""
        if len(v) < 2:
            return v
        
        for i in range(1, len(v)):
            if v[i].timestamp <= v[i-1].timestamp:
                raise ValueError('Data points must be in chronological order')
        
        return v

# ============================================
# Request Schemas
# ============================================

class DataRequest(BaseDataModel):
    """Base data request schema"""
    
    symbol: SymbolStr = Field(
        description="Stock symbol to fetch",
        examples=["AAPL", "MSFT", "RELIANCE.NS"]
    )
    
    start_date: Optional[date] = Field(
        default=None,
        description="Start date for data (ISO format)",
        examples=["2023-01-01"]
    )
    
    end_date: Optional[date] = Field(
        default=None,
        description="End date for data (ISO format)",
        examples=["2023-12-31"]
    )
    
    interval: TimeInterval = Field(
        default=TimeInterval.DAY_1,
        description="Data time interval",
        examples=["1d"]
    )
    
    source: Optional[DataSource] = Field(
        default=None,
        description="Preferred data source (auto-select if not specified)"
    )
    
    @field_validator('symbol', mode='before')
    @classmethod
    def symbol_to_upper(cls, v: str) -> str:
        """Convert symbol to uppercase"""
        return v.upper().strip() if v else v
    
    @model_validator(mode='after')
    def validate_date_range(self):
        """Validate date range"""
        
        if self.start_date and self.end_date:
            if self.start_date >= self.end_date:
                raise ValueError('Start date must be before end date')
            
            # Limit maximum date range to 5 years
            max_range = timedelta(days=365 * 5)
            if (self.end_date - self.start_date) > max_range:
                raise ValueError('Date range cannot exceed 5 years')
        
        # Don't allow future dates
        today = date.today()
        if self.start_date and self.start_date > today:
            raise ValueError('Start date cannot be in the future')
        if self.end_date and self.end_date > today:
            raise ValueError('End date cannot be in the future')
        
        return self
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "symbol": "AAPL",
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "interval": "1d",
                    "source": "yahoo_finance"
                },
                {
                    "symbol": "RELIANCE.NS",
                    "start_date": "2023-06-01",
                    "end_date": "2023-06-30",
                    "interval": "1h"
                }
            ]
        }
    )

class MultiSymbolDataRequest(BaseDataModel):
    """Request schema for fetching multiple symbols"""
    
    symbols: List[SymbolStr] = Field(
        description="List of stock symbols",
        min_length=1,
        max_length=50,
        examples=[["AAPL", "MSFT", "GOOGL"]]
    )
    
    start_date: Optional[date] = Field(
        default=None,
        description="Start date for data"
    )
    
    end_date: Optional[date] = Field(
        default=None,
        description="End date for data"
    )
    
    interval: TimeInterval = Field(
        default=TimeInterval.DAY_1,
        description="Data time interval"
    )
    
    source: Optional[DataSource] = Field(
        default=None,
        description="Preferred data source"
    )
    
    @field_validator('symbols', mode='before')
    @classmethod
    def symbols_to_upper_and_unique(cls, v: List[str]) -> List[str]:
        """Convert symbols to uppercase and ensure uniqueness"""
        if not v:
            return v
        
        upper_symbols = [s.upper().strip() for s in v]
        unique_symbols = list(dict.fromkeys(upper_symbols))  # Preserve order while removing duplicates
        
        if len(unique_symbols) != len(upper_symbols):
            raise ValueError('Symbols must be unique')
        
        return unique_symbols

class RealTimeDataRequest(BaseDataModel):
    """Request schema for real-time data"""
    
    symbols: List[SymbolStr] = Field(
        description="List of symbols for real-time data",
        min_length=1,
        max_length=100
    )
    
    fields: Optional[List[str]] = Field(
        default=None,
        description="Specific fields to retrieve",
        examples=[["price", "volume", "change", "change_percent"]]
    )
    
    source: Optional[DataSource] = Field(
        default=None,
        description="Real-time data source"
    )
    
    @field_validator('symbols', mode='before')
    @classmethod
    def symbols_to_upper(cls, v: List[str]) -> List[str]:
        """Convert symbols to uppercase"""
        return [s.upper().strip() for s in v] if v else v

class DataValidationRequest(BaseDataModel):
    """Request schema for data validation"""
    
    symbol: SymbolStr = Field(
        description="Symbol to validate"
    )
    
    data: List[OHLCVData] = Field(
        description="Data to validate",
        min_length=1
    )
    
    validation_rules: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom validation rules"
    )
    
    @field_validator('symbol', mode='before')
    @classmethod
    def symbol_to_upper(cls, v: str) -> str:
        """Convert symbol to uppercase"""
        return v.upper().strip() if v else v

# ============================================
# Response Schemas
# ============================================

class DataResponse(BaseDataModel, TimestampMixin):
    """Base data response schema"""
    
    success: bool = Field(
        description="Whether the request was successful",
        examples=[True]
    )
    
    message: Optional[str] = Field(
        default=None,
        description="Response message",
        examples=["Data retrieved successfully"]
    )
    
    data_count: int = Field(
        description="Number of data points returned",
        examples=[252]
    )

class MarketDataResponse(DataResponse):
    """Market data response schema"""
    
    symbol_info: SymbolInfo = Field(
        description="Symbol information"
    )
    
    market_data: MarketData = Field(
        description="Market data"
    )
    
    statistics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Statistical summary of the data",
        examples=[{
            "mean_price": 150.25,
            "volatility": 0.25,
            "total_volume": 125000000,
            "price_change": 2.5,
            "price_change_percent": 1.65
        }]
    )

class MultiSymbolDataResponse(DataResponse):
    """Multi-symbol data response schema"""
    
    data: Dict[str, MarketData] = Field(
        description="Market data by symbol"
    )
    
    failed_symbols: Optional[List[str]] = Field(
        default=None,
        description="Symbols that failed to fetch"
    )
    
    summary: Dict[str, Any] = Field(
        description="Summary statistics",
        examples=[{
            "total_symbols": 3,
            "successful_symbols": 3,
            "failed_symbols": 0,
            "total_data_points": 756
        }]
    )

class RealTimeDataResponse(DataResponse):
    """Real-time data response schema"""
    
    data: Dict[str, Dict[str, Union[float, int, str]]] = Field(
        description="Real-time data by symbol",
        examples=[{
            "AAPL": {
                "price": 150.25,
                "volume": 1234567,
                "change": 2.5,
                "change_percent": 1.65,
                "last_updated": "2023-01-01T15:30:00Z"
            }
        }]
    )
    
    market_status: str = Field(
        description="Market status",
        examples=["OPEN", "CLOSED", "PRE_MARKET", "AFTER_HOURS"]
    )
    
    delay_info: Dict[str, int] = Field(
        description="Data delay information by symbol",
        examples=[{"AAPL": 0, "MSFT": 15}]
    )

class DataValidationResponse(DataResponse):
    """Data validation response schema"""
    
    validation_results: Dict[str, Any] = Field(
        description="Validation results",
        examples=[{
            "is_valid": True,
            "errors": [],
            "warnings": ["Volume unusually high on 2023-01-15"],
            "quality_score": 0.95,
            "completeness": 1.0,
            "outliers_detected": 2
        }]
    )
    
    corrected_data: Optional[List[OHLCVData]] = Field(
        default=None,
        description="Data with corrections applied"
    )

# ============================================
# Symbol Search and Discovery Schemas
# ============================================

class SymbolSearchRequest(BaseDataModel):
    """Symbol search request schema"""
    
    query: str = Field(
        min_length=1,
        max_length=100,
        description="Search query (symbol or company name)",
        examples=["AAPL", "Apple", "tech"]
    )
    
    market: Optional[MarketType] = Field(
        default=None,
        description="Filter by market"
    )
    
    sector: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Filter by sector"
    )
    
    limit: LimitedInt = Field(
        default=10,
        description="Maximum number of results"
    )
    
    include_inactive: bool = Field(
        default=False,
        description="Include inactive symbols"
    )

class SymbolSearchResponse(DataResponse):
    """Symbol search response schema"""
    
    symbols: List[SymbolInfo] = Field(
        description="Matching symbols"
    )
    
    search_metadata: Dict[str, Any] = Field(
        description="Search metadata",
        examples=[{
            "query": "apple",
            "total_matches": 15,
            "returned_count": 10,
            "search_time_ms": 45
        }]
    )

# ============================================
# Data Quality and Monitoring Schemas
# ============================================

class DataQualityMetrics(BaseDataModel):
    """Data quality metrics"""
    
    completeness: QualityFloat = Field(
        description="Data completeness ratio",
        examples=[0.98]
    )
    
    accuracy: QualityFloat = Field(
        description="Data accuracy score",
        examples=[0.95]
    )
    
    timeliness: QualityFloat = Field(
        description="Data timeliness score",
        examples=[0.90]
    )
    
    consistency: QualityFloat = Field(
        description="Data consistency score",
        examples=[0.97]
    )
    
    outliers_count: int = Field(
        ge=0,
        description="Number of outliers detected",
        examples=[3]
    )
    
    missing_data_points: int = Field(
        ge=0,
        description="Number of missing data points",
        examples=[5]
    )
    
    quality_issues: List[str] = Field(
        description="List of quality issues found",
        examples=[["Gap in data on 2023-01-15", "Unusual volume spike on 2023-01-20"]]
    )

class DataMonitoringRequest(BaseDataModel):
    """Data monitoring request schema"""
    
    symbols: List[str] = Field(
        min_length=1,
        max_length=1000,
        description="Symbols to monitor"
    )
    
    start_date: date = Field(
        description="Monitoring start date"
    )
    
    end_date: date = Field(
        description="Monitoring end date"
    )
    
    quality_threshold: QualityFloat = Field(
        default=0.90,
        description="Minimum quality threshold"
    )
    
    @model_validator(mode='after')
    def validate_date_range(self):
        """Validate date range"""
        if self.start_date >= self.end_date:
            raise ValueError('Start date must be before end date')
        return self

class DataMonitoringResponse(DataResponse):
    """Data monitoring response schema"""
    
    monitoring_results: Dict[str, DataQualityMetrics] = Field(
        description="Quality metrics by symbol"
    )
    
    alerts: List[Dict[str, Any]] = Field(
        description="Quality alerts",
        examples=[[{
            "symbol": "AAPL",
            "alert_type": "LOW_QUALITY",
            "message": "Data quality below threshold",
            "quality_score": 0.85,
            "timestamp": "2023-01-01T10:00:00Z"
        }]]
    )
    
    summary: Dict[str, Any] = Field(
        description="Monitoring summary",
        examples=[{
            "symbols_monitored": 100,
            "avg_quality_score": 0.94,
            "symbols_below_threshold": 5,
            "total_alerts": 12
        }]
    )

# ============================================
# Bulk Operations Schemas
# ============================================

class BulkDataRequest(BaseDataModel):
    """Bulk data operation request"""
    
    operation: Literal["fetch", "validate", "update", "delete"] = Field(
        description="Bulk operation type"
    )
    
    symbols: List[str] = Field(
        min_length=1,
        max_length=1000,
        description="Symbols for bulk operation"
    )
    
    parameters: Dict[str, Any] = Field(
        description="Operation-specific parameters",
        examples=[{
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "interval": "1d"
        }]
    )
    
    batch_size: Annotated[int, Field(ge=1, le=100)] = Field(
        default=10,
        description="Number of symbols to process per batch"
    )
    
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing"
    )

class BulkDataResponse(DataResponse):
    """Bulk data operation response"""
    
    job_id: str = Field(
        description="Bulk operation job ID",
        examples=["bulk_job_123456"]
    )
    
    status: Literal["pending", "processing", "completed", "failed", "cancelled"] = Field(
        description="Job status"
    )
    
    progress: Dict[str, int] = Field(
        description="Processing progress",
        examples=[{
            "total": 100,
            "completed": 75,
            "failed": 5,
            "pending": 20
        }]
    )
    
    results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Operation results (available when completed)"
    )
    
    errors: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Errors encountered during processing"
    )
    
    estimated_completion: Optional[datetime] = Field(
        default=None,
        description="Estimated completion time"
    )

# ============================================
# Export All Schemas
# ============================================

__all__ = [
    # Type aliases
    "SymbolStr",
    "CompanyNameStr", 
    "CurrencyStr",
    "PositiveFloat",
    "PositiveInt",
    "LimitedInt",
    "QualityFloat",
    
    # Enums
    "MarketType",
    "DataSource",
    "TimeInterval",
    "DataQuality", 
    "DataStatus",
    
    # Base models
    "BaseDataModel",
    "TimestampMixin",
    
    # Core data models
    "SymbolInfo",
    "OHLCVData",
    "MarketData",
    
    # Request schemas
    "DataRequest",
    "MultiSymbolDataRequest", 
    "RealTimeDataRequest",
    "DataValidationRequest",
    "SymbolSearchRequest",
    "DataMonitoringRequest",
    "BulkDataRequest",
    
    # Response schemas
    "DataResponse",
    "MarketDataResponse",
    "MultiSymbolDataResponse",
    "RealTimeDataResponse", 
    "DataValidationResponse",
    "SymbolSearchResponse",
    "DataMonitoringResponse",
    "BulkDataResponse",
    
    # Quality and monitoring
    "DataQualityMetrics",
]
