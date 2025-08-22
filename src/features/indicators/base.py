# ============================================
# StockPredictionPro - src/features/indicators/base.py
# Foundation classes and utilities for all technical indicators with advanced financial domain support
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import warnings

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it

logger = get_logger('features.indicators.base')

# ============================================
# Core Data Types and Enums
# ============================================

class IndicatorType(Enum):
    """Type of technical indicator"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    CUSTOM = "custom"
    COMPOSITE = "composite"

class TimeFrame(Enum):
    """Time frame for indicator calculations"""
    TICK = "tick"
    MINUTE = "1min"
    FIVE_MINUTE = "5min"
    FIFTEEN_MINUTE = "15min"
    THIRTY_MINUTE = "30min"
    HOURLY = "1h"
    FOUR_HOURLY = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"

class PriceField(Enum):
    """Standard price fields for OHLCV data"""
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    TYPICAL_PRICE = "typical"  # (H+L+C)/3
    WEIGHTED_CLOSE = "weighted"  # (H+L+2*C)/4
    MEDIAN_PRICE = "median"  # (H+L)/2

@dataclass
class IndicatorConfig:
    """Configuration for indicator calculation"""
    period: int = 14
    timeframe: TimeFrame = TimeFrame.DAILY
    price_field: PriceField = PriceField.CLOSE
    smoothing_method: str = "sma"  # sma, ema, wma, etc.
    fill_na: bool = True
    min_periods: Optional[int] = None
    dropna: bool = False
    
    def __post_init__(self):
        if self.min_periods is None:
            self.min_periods = max(1, self.period // 2)

@dataclass
class IndicatorResult:
    """Result container for indicator calculations"""
    name: str
    values: Union[pd.Series, pd.DataFrame]
    config: IndicatorConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_series(self) -> bool:
        """Check if result is a pandas Series"""
        return isinstance(self.values, pd.Series)
    
    @property
    def is_dataframe(self) -> bool:
        """Check if result is a pandas DataFrame"""
        return isinstance(self.values, pd.DataFrame)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'name': self.name,
            'values': self.values.to_dict() if hasattr(self.values, 'to_dict') else self.values,
            'config': self.config.__dict__,
            'metadata': self.metadata
        }

# ============================================
# Data Validation and Preprocessing
# ============================================

class DataValidator:
    """Validate and preprocess market data for indicator calculations"""
    
    @staticmethod
    def validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data format and consistency"""
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
        
        # Check for non-numeric data
        numeric_columns = ['open', 'high', 'low', 'close']
        if 'volume' in data.columns:
            numeric_columns.append('volume')
        
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except:
                    raise ValidationError(f"Column {col} contains non-numeric data")
        
        # Validate price relationships (High >= Low, etc.)
        invalid_hlc = (data['high'] < data['low']) | (data['high'] < data['close']) | (data['low'] > data['close'])
        if invalid_hlc.any():
            logger.warning(f"Found {invalid_hlc.sum()} rows with invalid price relationships")
            
        # Check for negative prices
        negative_prices = (data[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        if negative_prices.any():
            logger.warning(f"Found {negative_prices.sum()} rows with non-positive prices")
        
        # Check for missing values
        missing_values = data[numeric_columns].isnull().sum().sum()
        if missing_values > 0:
            logger.info(f"Found {missing_values} missing values in price data")
        
        return data
    
    @staticmethod
    def validate_series(data: pd.Series, name: str = "data") -> pd.Series:
        """Validate a single price series"""
        
        if not isinstance(data, pd.Series):
            raise ValidationError(f"{name} must be a pandas Series")
        
        if data.empty:
            raise ValidationError(f"{name} is empty")
        
        if not pd.api.types.is_numeric_dtype(data):
            try:
                data = pd.to_numeric(data, errors='coerce')
            except:
                raise ValidationError(f"{name} contains non-numeric data")
        
        return data
    
    @staticmethod
    def get_price_series(data: Union[pd.DataFrame, pd.Series], 
                        price_field: PriceField) -> pd.Series:
        """Extract specific price series from OHLCV data"""
        
        if isinstance(data, pd.Series):
            return data
        
        if price_field == PriceField.OPEN:
            return data['open']
        elif price_field == PriceField.HIGH:
            return data['high']
        elif price_field == PriceField.LOW:
            return data['low']
        elif price_field == PriceField.CLOSE:
            return data['close']
        elif price_field == PriceField.VOLUME:
            if 'volume' not in data.columns:
                raise ValidationError("Volume data not available")
            return data['volume']
        elif price_field == PriceField.TYPICAL_PRICE:
            return (data['high'] + data['low'] + data['close']) / 3
        elif price_field == PriceField.WEIGHTED_CLOSE:
            return (data['high'] + data['low'] + 2 * data['close']) / 4
        elif price_field == PriceField.MEDIAN_PRICE:
            return (data['high'] + data['low']) / 2
        else:
            raise ValidationError(f"Unknown price field: {price_field}")

# ============================================
# Base Indicator Classes
# ============================================

class BaseIndicator(ABC):
    """Abstract base class for all technical indicators"""
    
    def __init__(self, 
                 name: str,
                 indicator_type: IndicatorType,
                 config: Optional[IndicatorConfig] = None):
        self.name = name
        self.indicator_type = indicator_type
        self.config = config or IndicatorConfig()
        self.validator = DataValidator()
        
        # Calculation cache
        self._cache = {}
        self._cache_keys = set()
        
        logger.debug(f"Initialized {self.name} indicator")
    
    @abstractmethod
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> IndicatorResult:
        """Calculate the indicator values"""
        pass
    
    def __call__(self, data: Union[pd.DataFrame, pd.Series]) -> IndicatorResult:
        """Allow indicator to be called like a function"""
        return self.calculate(data)
    
    def _get_cache_key(self, data: Union[pd.DataFrame, pd.Series]) -> str:
        """Generate cache key for data"""
        if isinstance(data, pd.DataFrame):
            # Use first few and last few values plus shape for cache key
            key_data = str(data.iloc[[0, -1]].values.flatten()) + str(data.shape)
        else:
            key_data = str([data.iloc[0], data.iloc[-1]]) + str(len(data))
        
        config_str = str(self.config.__dict__)
        return f"{self.name}_{hash(key_data + config_str)}"
    
    def _get_cached_result(self, data: Union[pd.DataFrame, pd.Series]) -> Optional[IndicatorResult]:
        """Get cached calculation result"""
        cache_key = self._get_cache_key(data)
        return self._cache.get(cache_key)
    
    def _cache_result(self, data: Union[pd.DataFrame, pd.Series], result: IndicatorResult):
        """Cache calculation result"""
        cache_key = self._get_cache_key(data)
        
        # Limit cache size
        if len(self._cache) >= 100:
            # Remove oldest cache entry
            oldest_key = next(iter(self._cache_keys))
            del self._cache[oldest_key]
            self._cache_keys.remove(oldest_key)
        
        self._cache[cache_key] = result
        self._cache_keys.add(cache_key)
    
    def clear_cache(self):
        """Clear calculation cache"""
        self._cache.clear()
        self._cache_keys.clear()
    
    def validate_data(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Validate input data"""
        if isinstance(data, pd.DataFrame):
            return self.validator.validate_ohlcv(data)
        else:
            return self.validator.validate_series(data, self.name)
    
    def get_price_series(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Get price series based on config"""
        return self.validator.get_price_series(data, self.config.price_field)
    
    def _handle_insufficient_data(self, data_length: int, required_length: int) -> pd.Series:
        """Handle cases where there's insufficient data"""
        if data_length < required_length:
            logger.warning(f"{self.name}: Insufficient data. Required: {required_length}, Got: {data_length}")
            
        # Return NaN series of appropriate length
        return pd.Series([np.nan] * data_length)
    
    def _apply_fill_method(self, series: pd.Series) -> pd.Series:
        """Apply fill method for NaN values"""
        if not self.config.fill_na:
            return series
        
        # Forward fill then backward fill
        filled = series.fillna(method='ffill').fillna(method='bfill')
        return filled
    
    def create_result(self, values: Union[pd.Series, pd.DataFrame], 
                     metadata: Optional[Dict[str, Any]] = None) -> IndicatorResult:
        """Create standardized indicator result"""
        
        # Apply fill method if needed
        if isinstance(values, pd.Series):
            values = self._apply_fill_method(values)
        elif isinstance(values, pd.DataFrame):
            for col in values.columns:
                values[col] = self._apply_fill_method(values[col])
        
        # Drop NaN values if requested
        if self.config.dropna:
            values = values.dropna()
        
        metadata = metadata or {}
        metadata.update({
            'indicator_type': self.indicator_type.value,
            'config': self.config.__dict__,
            'calculated_at': datetime.now().isoformat()
        })
        
        return IndicatorResult(
            name=self.name,
            values=values,
            config=self.config,
            metadata=metadata
        )

class SingleValueIndicator(BaseIndicator):
    """Base class for indicators that return a single series"""
    
    @abstractmethod
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Calculate indicator values - to be implemented by subclasses"""
        pass
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> IndicatorResult:
        """Calculate the indicator with caching and validation"""
        
        # Check cache first
        cached_result = self._get_cached_result(data)
        if cached_result is not None:
            return cached_result
        
        # Validate data
        validated_data = self.validate_data(data)
        
        try:
            # Calculate values
            with Timer() as timer:
                values = self._calculate_values(validated_data)
            
            # Create result with timing metadata
            metadata = {
                'calculation_time': timer.elapsed,
                'data_length': len(validated_data),
                'non_null_values': values.notna().sum()
            }
            
            result = self.create_result(values, metadata)
            
            # Cache result
            self._cache_result(data, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating {self.name}: {e}")
            raise CalculationError(f"Failed to calculate {self.name}: {e}")

class MultiValueIndicator(BaseIndicator):
    """Base class for indicators that return multiple series (e.g., Bollinger Bands)"""
    
    @abstractmethod
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Calculate indicator values - to be implemented by subclasses"""
        pass
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> IndicatorResult:
        """Calculate the indicator with caching and validation"""
        
        # Check cache first
        cached_result = self._get_cached_result(data)
        if cached_result is not None:
            return cached_result
        
        # Validate data
        validated_data = self.validate_data(data)
        
        try:
            # Calculate values
            with Timer() as timer:
                values = self._calculate_values(validated_data)
            
            # Create result with timing metadata
            metadata = {
                'calculation_time': timer.elapsed,
                'data_length': len(validated_data),
                'columns': list(values.columns),
                'non_null_values': {col: values[col].notna().sum() for col in values.columns}
            }
            
            result = self.create_result(values, metadata)
            
            # Cache result
            self._cache_result(data, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating {self.name}: {e}")
            raise CalculationError(f"Failed to calculate {self.name}: {e}")

# ============================================
# Common Mathematical Functions
# ============================================

class MathUtils:
    """Mathematical utility functions for indicator calculations"""
    
    @staticmethod
    def simple_moving_average(series: pd.Series, period: int, min_periods: Optional[int] = None) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=period, min_periods=min_periods).mean()
    
    @staticmethod
    def exponential_moving_average(series: pd.Series, period: int, alpha: Optional[float] = None) -> pd.Series:
        """Calculate Exponential Moving Average"""
        if alpha is None:
            alpha = 2.0 / (period + 1)
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    @staticmethod
    def weighted_moving_average(series: pd.Series, period: int) -> pd.Series:
        """Calculate Weighted Moving Average"""
        def wma(values):
            if len(values) < period:
                return np.nan
            weights = np.arange(1, period + 1)
            return np.average(values[-period:], weights=weights)
        
        return series.rolling(window=period).apply(wma, raw=True)
    
    @staticmethod
    def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate True Range"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    @staticmethod
    def standard_deviation(series: pd.Series, period: int, ddof: int = 1) -> pd.Series:
        """Calculate rolling standard deviation"""
        return series.rolling(window=period).std(ddof=ddof)
    
    @staticmethod
    def correlation(series1: pd.Series, series2: pd.Series, period: int) -> pd.Series:
        """Calculate rolling correlation between two series"""
        return series1.rolling(window=period).corr(series2)
    
    @staticmethod
    def rate_of_change(series: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change"""
        return (series / series.shift(period) - 1) * 100
    
    @staticmethod
    def money_flow_multiplier(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Money Flow Multiplier for volume-based indicators"""
        return ((close - low) - (high - close)) / (high - low)
    
    @staticmethod
    def typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Typical Price (HLC/3)"""
        return (high + low + close) / 3
    
    @staticmethod
    def weighted_close(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Weighted Close Price (HLCC/4)"""
        return (high + low + 2 * close) / 4
    
    @staticmethod
    def median_price(high: pd.Series, low: pd.Series) -> pd.Series:
        """Calculate Median Price (HL/2)"""
        return (high + low) / 2

# ============================================
# Smoothing Functions
# ============================================

class SmoothingMethods:
    """Collection of smoothing methods for indicators"""
    
    METHODS = {
        'sma': MathUtils.simple_moving_average,
        'ema': MathUtils.exponential_moving_average,
        'wma': MathUtils.weighted_moving_average
    }
    
    @classmethod
    def smooth(cls, series: pd.Series, method: str, period: int, **kwargs) -> pd.Series:
        """Apply smoothing method to series"""
        if method not in cls.METHODS:
            raise ValueError(f"Unknown smoothing method: {method}. Available: {list(cls.METHODS.keys())}")
        
        smoothing_func = cls.METHODS[method]
        
        # Handle different function signatures
        if method == 'ema':
            return smoothing_func(series, period, kwargs.get('alpha'))
        else:
            return smoothing_func(series, period, kwargs.get('min_periods'))
    
    @classmethod
    def available_methods(cls) -> List[str]:
        """Get list of available smoothing methods"""
        return list(cls.METHODS.keys())

# ============================================
# Indicator Registry and Factory
# ============================================

class IndicatorRegistry:
    """Registry for all available indicators"""
    
    def __init__(self):
        self._indicators = {}
        self._indicator_types = {t.value: [] for t in IndicatorType}
        
    def register(self, indicator_class: type, name: str, indicator_type: IndicatorType):
        """Register an indicator class"""
        self._indicators[name] = indicator_class
        self._indicator_types[indicator_type.value].append(name)
        logger.debug(f"Registered indicator: {name}")
    
    def create(self, name: str, config: Optional[IndicatorConfig] = None, **kwargs) -> BaseIndicator:
        """Create indicator instance"""
        if name not in self._indicators:
            raise ValueError(f"Unknown indicator: {name}. Available: {list(self._indicators.keys())}")
        
        indicator_class = self._indicators[name]
        return indicator_class(config=config, **kwargs)
    
    def list_indicators(self, indicator_type: Optional[IndicatorType] = None) -> List[str]:
        """List available indicators"""
        if indicator_type is None:
            return list(self._indicators.keys())
        return self._indicator_types[indicator_type.value]
    
    def get_indicator_info(self, name: str) -> Dict[str, Any]:
        """Get information about an indicator"""
        if name not in self._indicators:
            raise ValueError(f"Unknown indicator: {name}")
        
        indicator_class = self._indicators[name]
        return {
            'name': name,
            'class': indicator_class.__name__,
            'docstring': indicator_class.__doc__,
            'module': indicator_class.__module__
        }

# Global indicator registry
indicator_registry = IndicatorRegistry()

# ============================================
# Utility Functions
# ============================================

def create_indicator_config(period: int = 14,
                          timeframe: TimeFrame = TimeFrame.DAILY,
                          price_field: PriceField = PriceField.CLOSE,
                          **kwargs) -> IndicatorConfig:
    """Create indicator configuration"""
    return IndicatorConfig(
        period=period,
        timeframe=timeframe,
        price_field=price_field,
        **kwargs
    )

def validate_indicator_data(data: Union[pd.DataFrame, pd.Series], 
                          required_columns: Optional[List[str]] = None) -> bool:
    """Validate data for indicator calculation"""
    try:
        if isinstance(data, pd.DataFrame):
            validator = DataValidator()
            validator.validate_ohlcv(data)
            
            if required_columns:
                missing = [col for col in required_columns if col not in data.columns]
                if missing:
                    raise ValidationError(f"Missing required columns: {missing}")
        else:
            validator = DataValidator()
            validator.validate_series(data)
        
        return True
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False

@time_it("bulk_indicator_calculation")
def calculate_multiple_indicators(data: Union[pd.DataFrame, pd.Series],
                                indicators: List[Tuple[str, IndicatorConfig]]) -> Dict[str, IndicatorResult]:
    """Calculate multiple indicators efficiently"""
    
    results = {}
    
    for indicator_name, config in indicators:
        try:
            indicator = indicator_registry.create(indicator_name, config)
            result = indicator.calculate(data)
            results[indicator_name] = result
            
        except Exception as e:
            logger.warning(f"Failed to calculate {indicator_name}: {e}")
    
    logger.info(f"Calculated {len(results)}/{len(indicators)} indicators successfully")
    return results

def combine_indicator_results(results: List[IndicatorResult], 
                            prefix: str = "indicator") -> pd.DataFrame:
    """Combine multiple indicator results into a single DataFrame"""
    
    combined_data = {}
    
    for result in results:
        if result.is_series:
            combined_data[f"{prefix}_{result.name}"] = result.values
        elif result.is_dataframe:
            for col in result.values.columns:
                combined_data[f"{prefix}_{result.name}_{col}"] = result.values[col]
    
    if not combined_data:
        return pd.DataFrame()
    
    # Align all series to the same index
    combined_df = pd.DataFrame(combined_data)
    return combined_df

# ============================================
# Indicator Performance Profiler
# ============================================

class IndicatorProfiler:
    """Profile indicator calculation performance"""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_indicator(self, 
                         indicator: BaseIndicator,
                         data: Union[pd.DataFrame, pd.Series],
                         iterations: int = 10) -> Dict[str, Any]:
        """Profile indicator calculation performance"""
        
        import time
        
        times = []
        
        for _ in range(iterations):
            # Clear cache to ensure fresh calculation
            indicator.clear_cache()
            
            start_time = time.time()
            result = indicator.calculate(data)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        profile_data = {
            'indicator_name': indicator.name,
            'data_length': len(data),
            'iterations': iterations,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_time': np.sum(times)
        }
        
        self.profiles[indicator.name] = profile_data
        return profile_data
    
    def compare_indicators(self, indicators: List[BaseIndicator],
                          data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Compare performance of multiple indicators"""
        
        profiles = []
        
        for indicator in indicators:
            profile = self.profile_indicator(indicator, data)
            profiles.append(profile)
        
        df = pd.DataFrame(profiles)
        df = df.sort_values('mean_time')
        
        return df

# ============================================
# Example Usage and Testing
# ============================================

def create_sample_data(length: int = 100, 
                      start_price: float = 100.0,
                      volatility: float = 0.02) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    
    # Generate price data using random walk
    np.random.seed(42)
    returns = np.random.normal(0, volatility, length)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    
    # Generate realistic OHLC from close prices
    daily_range = np.random.uniform(0.005, 0.03, length)  # Daily range as % of price
    data['high'] = data['close'] * (1 + daily_range * np.random.uniform(0.3, 1.0, length))
    data['low'] = data['close'] * (1 - daily_range * np.random.uniform(0.3, 1.0, length))
    data['open'] = data['low'] + (data['high'] - data['low']) * np.random.uniform(0.1, 0.9, length)
    
    # Generate volume
    data['volume'] = np.random.lognormal(mean=10, sigma=0.5, size=length).astype(int)
    
    return data

if __name__ == "__main__":
    # Example usage
    print("Testing Base Indicator Framework")
    
    # Create sample data
    sample_data = create_sample_data(100)
    print(f"Created sample data: {sample_data.shape}")
    print(sample_data.head())
    
    # Test data validation
    validator = DataValidator()
    validated = validator.validate_ohlcv(sample_data)
    print(f"Data validation passed: {validated.shape}")
    
    # Test price field extraction
    close_prices = validator.get_price_series(sample_data, PriceField.CLOSE)
    typical_prices = validator.get_price_series(sample_data, PriceField.TYPICAL_PRICE)
    
    print(f"Close prices: {close_prices.iloc[:5].values}")
    print(f"Typical prices: {typical_prices.iloc[:5].values}")
    
    # Test math utilities
    sma_20 = MathUtils.simple_moving_average(close_prices, 20)
    ema_20 = MathUtils.exponential_moving_average(close_prices, 20)
    
    print(f"SMA(20) last 5 values: {sma_20.iloc[-5:].values}")
    print(f"EMA(20) last 5 values: {ema_20.iloc[-5:].values}")
    
    print("Base indicator framework tests completed successfully!")
