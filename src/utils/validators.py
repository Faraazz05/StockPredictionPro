# ============================================
# StockPredictionPro - src/utils/validators.py
# Comprehensive data and parameter validation system
# ============================================

import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import warnings

from .exceptions import (
    DataValidationError, 
    InvalidParameterError, 
    InsufficientDataError,
    BusinessLogicError
)
from .logger import get_logger

logger = get_logger('validators')

# ============================================
# Base Validation Framework
# ============================================

class ValidationResult:
    """Container for validation results"""
    
    def __init__(self, is_valid: bool = True, errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        
    def add_error(self, error: str):
        """Add an error message"""
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)
        
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False
            
    def raise_if_invalid(self):
        """Raise exception if validation failed"""
        if not self.is_valid:
            error_msg = "; ".join(self.errors)
            raise DataValidationError(
                f"Validation failed: {error_msg}",
                validation_errors=self.errors
            )
    
    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"Validation: {status}"]
        
        if self.errors:
            parts.append(f"Errors: {', '.join(self.errors)}")
        if self.warnings:
            parts.append(f"Warnings: {', '.join(self.warnings)}")
            
        return " | ".join(parts)

class BaseValidator:
    """Base class for all validators"""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        
    def validate(self, data: Any) -> ValidationResult:
        """Override in subclasses"""
        raise NotImplementedError
        
    def _create_result(self, is_valid: bool = True) -> ValidationResult:
        """Create new validation result"""
        return ValidationResult(is_valid=is_valid)

# ============================================
# Symbol Validators
# ============================================

class SymbolValidator(BaseValidator):
    """Validate stock symbols"""
    
    VALID_EXCHANGES = {
        'US': ['NYSE', 'NASDAQ', 'AMEX'],
        'IN': ['NSE', 'BSE'],
        'UK': ['LSE'],
        'JP': ['TSE'],
        'HK': ['HKEX']
    }
    
    SYMBOL_PATTERNS = {
        'US': re.compile(r'^[A-Z]{1,5}$'),
        'IN': re.compile(r'^[A-Z0-9]{1,12}\.NS$|^[A-Z0-9]{1,12}\.BO$'),
        'INDEX': re.compile(r'^\^[A-Z0-9]{1,10}$'),
        'ETF': re.compile(r'^[A-Z]{2,5}$'),
        'FOREX': re.compile(r'^[A-Z]{6}=X$')
    }
    
    def validate_symbol(self, symbol: str) -> ValidationResult:
        """
        Validate individual stock symbol
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            ValidationResult with validation status
        """
        result = self._create_result()
        
        if not symbol or not isinstance(symbol, str):
            result.add_error("Symbol must be a non-empty string")
            return result
        
        symbol = symbol.strip().upper()
        
        # Check length
        if len(symbol) < 1 or len(symbol) > 20:
            result.add_error(f"Symbol length must be 1-20 characters: {symbol}")
        
        # Check for invalid characters
        if not re.match(r'^[A-Z0-9.\^=-]+$', symbol):
            result.add_error(f"Symbol contains invalid characters: {symbol}")
        
        # Pattern-based validation
        symbol_type = self._detect_symbol_type(symbol)
        if symbol_type == 'UNKNOWN':
            result.add_warning(f"Unknown symbol format: {symbol}")
        
        # Common symbol checks
        if symbol in ['', 'NULL', 'NONE', 'N/A']:
            result.add_error(f"Invalid symbol value: {symbol}")
        
        # Known problematic symbols
        problematic_symbols = ['TEST', 'SAMPLE', 'DUMMY']
        if symbol in problematic_symbols:
            result.add_warning(f"Potentially test symbol: {symbol}")
        
        return result
    
    def validate_symbols(self, symbols: Union[str, List[str]]) -> ValidationResult:
        """
        Validate list of symbols
        
        Args:
            symbols: Single symbol or list of symbols
            
        Returns:
            ValidationResult with validation status
        """
        result = self._create_result()
        
        # Convert to list if string
        if isinstance(symbols, str):
            symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
        elif isinstance(symbols, list):
            symbol_list = [str(s).strip() for s in symbols if s]
        else:
            result.add_error("Symbols must be string or list")
            return result
        
        if not symbol_list:
            result.add_error("At least one symbol must be provided")
            return result
        
        # Check for duplicates
        unique_symbols = set(symbol_list)
        if len(unique_symbols) != len(symbol_list):
            result.add_warning("Duplicate symbols found")
        
        # Validate each symbol
        valid_symbols = []
        for symbol in unique_symbols:
            symbol_result = self.validate_symbol(symbol)
            result.merge(symbol_result)
            
            if symbol_result.is_valid:
                valid_symbols.append(symbol)
        
        if not valid_symbols and self.strict_mode:
            result.add_error("No valid symbols found")
        
        return result
    
    def _detect_symbol_type(self, symbol: str) -> str:
        """Detect symbol type based on pattern"""
        for symbol_type, pattern in self.SYMBOL_PATTERNS.items():
            if pattern.match(symbol):
                return symbol_type
        return 'UNKNOWN'

# ============================================
# Date Validators
# ============================================

class DateValidator(BaseValidator):
    """Validate dates and date ranges"""
    
    def __init__(self, strict_mode: bool = False):
        super().__init__(strict_mode)
        self.min_date = datetime(1900, 1, 1)
        self.max_date = datetime.now() + timedelta(days=365)  # Allow 1 year future
    
    def validate_date(self, date_input: Any, param_name: str = "date") -> ValidationResult:
        """
        Validate single date
        
        Args:
            date_input: Date to validate
            param_name: Parameter name for error messages
            
        Returns:
            ValidationResult with validation status
        """
        result = self._create_result()
        
        try:
            if isinstance(date_input, str):
                # Try common date formats
                formats = [
                    "%Y-%m-%d",
                    "%Y/%m/%d", 
                    "%d-%m-%Y",
                    "%d/%m/%Y",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S"
                ]
                
                parsed_date = None
                for fmt in formats:
                    try:
                        parsed_date = datetime.strptime(date_input, fmt)
                        break
                    except ValueError:
                        continue
                
                if parsed_date is None:
                    try:
                        parsed_date = pd.to_datetime(date_input).to_pydatetime()
                    except Exception:
                        result.add_error(f"Unable to parse {param_name}: {date_input}")
                        return result
                        
            elif isinstance(date_input, (datetime, pd.Timestamp)):
                parsed_date = date_input.to_pydatetime() if isinstance(date_input, pd.Timestamp) else date_input
            else:
                result.add_error(f"{param_name} must be string, datetime, or pandas Timestamp")
                return result
            
            # Validate date range
            if parsed_date < self.min_date:
                result.add_error(f"{param_name} too early (before {self.min_date.strftime('%Y-%m-%d')})")
            
            if parsed_date > self.max_date:
                result.add_error(f"{param_name} too late (after {self.max_date.strftime('%Y-%m-%d')})")
            
        except Exception as e:
            result.add_error(f"Error validating {param_name}: {str(e)}")
        
        return result
    
    def validate_date_range(self, start_date: Any, end_date: Any) -> ValidationResult:
        """
        Validate date range
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            ValidationResult with validation status
        """
        result = self._create_result()
        
        # Validate individual dates
        start_result = self.validate_date(start_date, "start_date")
        end_result = self.validate_date(end_date, "end_date")
        
        result.merge(start_result)
        result.merge(end_result)
        
        if not (start_result.is_valid and end_result.is_valid):
            return result
        
        # Parse dates for comparison
        try:
            if isinstance(start_date, str):
                start_dt = pd.to_datetime(start_date).to_pydatetime()
            else:
                start_dt = start_date.to_pydatetime() if isinstance(start_date, pd.Timestamp) else start_date
                
            if isinstance(end_date, str):
                end_dt = pd.to_datetime(end_date).to_pydatetime()
            else:
                end_dt = end_date.to_pydatetime() if isinstance(end_date, pd.Timestamp) else end_date
            
            # Validate range logic
            if start_dt >= end_dt:
                result.add_error("Start date must be before end date")
            
            # Check range duration
            duration = end_dt - start_dt
            
            if duration.days < 1:
                result.add_warning("Date range is very short (less than 1 day)")
            elif duration.days > 365 * 10:  # 10 years
                result.add_warning("Date range is very long (more than 10 years)")
            
            # Check for weekends only (basic check)
            if duration.days <= 7:
                weekdays = [start_dt + timedelta(days=i) for i in range(duration.days + 1)]
                trading_days = [d for d in weekdays if d.weekday() < 5]  # Monday=0, Sunday=6
                
                if len(trading_days) < 2:
                    result.add_warning("Date range contains very few trading days")
                    
        except Exception as e:
            result.add_error(f"Error validating date range: {str(e)}")
        
        return result

# ============================================
# Data Quality Validators
# ============================================

class DataQualityValidator(BaseValidator):
    """Validate data quality for financial data"""
    
    def __init__(self, strict_mode: bool = False):
        super().__init__(strict_mode)
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.price_columns = ['Open', 'High', 'Low', 'Close']
        
    def validate_dataframe(self, df: pd.DataFrame, symbol: str = "Unknown") -> ValidationResult:
        """
        Validate pandas DataFrame for financial data
        
        Args:
            df: DataFrame to validate
            symbol: Symbol for context in error messages
            
        Returns:
            ValidationResult with validation status
        """
        result = self._create_result()
        
        # Basic DataFrame checks
        if df is None:
            result.add_error("DataFrame is None")
            return result
        
        if not isinstance(df, pd.DataFrame):
            result.add_error("Input must be a pandas DataFrame")
            return result
        
        if df.empty:
            result.add_error("DataFrame is empty")
            return result
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            result.add_error(f"Missing required columns: {missing_columns}")
        
        # Validate data types and values
        if result.is_valid:  # Only proceed if basic structure is valid
            self._validate_price_data(df, result, symbol)
            self._validate_volume_data(df, result, symbol)
            self._validate_data_consistency(df, result, symbol)
            self._validate_missing_data(df, result, symbol)
            self._validate_data_completeness(df, result, symbol)
        
        return result
    
    def _validate_price_data(self, df: pd.DataFrame, result: ValidationResult, symbol: str):
        """Validate price data columns"""
        for col in self.price_columns:
            if col not in df.columns:
                continue
                
            series = df[col]
            
            # Check for negative prices
            negative_count = (series < 0).sum()
            if negative_count > 0:
                result.add_error(f"{col} contains {negative_count} negative values")
            
            # Check for zero prices
            zero_count = (series == 0).sum()
            if zero_count > 0:
                result.add_warning(f"{col} contains {zero_count} zero values")
            
            # Check for extreme values
            if len(series.dropna()) > 0:
                median_price = series.median()
                
                # Values more than 100x or less than 1/100 of median
                extreme_high = (series > median_price * 100).sum()
                extreme_low = (series < median_price / 100).sum()
                
                if extreme_high > 0:
                    result.add_warning(f"{col} contains {extreme_high} extremely high values")
                if extreme_low > 0:
                    result.add_warning(f"{col} contains {extreme_low} extremely low values")
        
        # Validate OHLC relationships
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            self._validate_ohlc_relationships(df, result)
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame, result: ValidationResult):
        """Validate Open-High-Low-Close relationships"""
        # High should be >= Open, Low, Close
        high_violations = (
            (df['High'] < df['Open']) | 
            (df['High'] < df['Low']) | 
            (df['High'] < df['Close'])
        ).sum()
        
        if high_violations > 0:
            result.add_error(f"High price violations: {high_violations} cases where High < Open/Low/Close")
        
        # Low should be <= Open, High, Close
        low_violations = (
            (df['Low'] > df['Open']) | 
            (df['Low'] > df['High']) | 
            (df['Low'] > df['Close'])
        ).sum()
        
        if low_violations > 0:
            result.add_error(f"Low price violations: {low_violations} cases where Low > Open/High/Close")
    
    def _validate_volume_data(self, df: pd.DataFrame, result: ValidationResult, symbol: str):
        """Validate volume data"""
        if 'Volume' not in df.columns:
            return
        
        volume = df['Volume']
        
        # Check for negative volume
        negative_volume = (volume < 0).sum()
        if negative_volume > 0:
            result.add_error(f"Volume contains {negative_volume} negative values")
        
        # Check for zero volume (might be normal for some periods)
        zero_volume = (volume == 0).sum()
        zero_volume_pct = zero_volume / len(volume) * 100
        
        if zero_volume_pct > 10:  # More than 10% zero volume
            result.add_warning(f"High zero volume: {zero_volume_pct:.1f}% of records")
        
        # Check for extremely high volume spikes
        if len(volume.dropna()) > 0:
            median_volume = volume.median()
            volume_spikes = (volume > median_volume * 50).sum()  # 50x median
            
            if volume_spikes > 0:
                result.add_warning(f"Volume contains {volume_spikes} extreme spikes")
    
    def _validate_data_consistency(self, df: pd.DataFrame, result: ValidationResult, symbol: str):
        """Validate data consistency and detect anomalies"""
        # Check for duplicate dates
        if hasattr(df.index, 'duplicated'):
            duplicate_dates = df.index.duplicated().sum()
            if duplicate_dates > 0:
                result.add_error(f"Duplicate dates found: {duplicate_dates}")
        
        # Check for large price gaps (potential stock splits or errors)
        if 'Close' in df.columns and len(df) > 1:
            price_changes = df['Close'].pct_change().abs()
            large_gaps = (price_changes > 0.5).sum()  # 50% change
            
            if large_gaps > 0:
                result.add_warning(f"Large price gaps detected: {large_gaps} (possible splits/errors)")
        
        # Check date sequence (if index is datetime)
        if isinstance(df.index, pd.DatetimeIndex):
            # Check for proper chronological order
            if not df.index.is_monotonic_increasing:
                result.add_warning("Data is not in chronological order")
            
            # Check for reasonable date gaps
            date_diffs = df.index.to_series().diff().dt.days
            max_gap = date_diffs.max()
            
            if max_gap > 30:  # More than 30 days gap
                result.add_warning(f"Large date gap detected: {max_gap} days")
    
    def _validate_missing_data(self, df: pd.DataFrame, result: ValidationResult, symbol: str):
        """Validate missing data patterns"""
        total_rows = len(df)
        
        for col in self.required_columns:
            if col not in df.columns:
                continue
            
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / total_rows * 100
            
            if missing_pct > 0:
                if missing_pct > 20:  # More than 20% missing
                    result.add_error(f"{col} has {missing_pct:.1f}% missing values")
                elif missing_pct > 5:  # More than 5% missing
                    result.add_warning(f"{col} has {missing_pct:.1f}% missing values")
        
        # Check for complete missing rows
        complete_missing_rows = df.isnull().all(axis=1).sum()
        if complete_missing_rows > 0:
            result.add_error(f"Complete missing rows: {complete_missing_rows}")
    
    def _validate_data_completeness(self, df: pd.DataFrame, result: ValidationResult, symbol: str):
        """Validate data completeness for analysis"""
        row_count = len(df)
        
        # Minimum data requirements
        if row_count < 30:
            result.add_error(f"Insufficient data: {row_count} rows (minimum 30 required)")
        elif row_count < 100:
            result.add_warning(f"Limited data: {row_count} rows (100+ recommended)")
        
        # Check data recency (if dates are available)
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            latest_date = df.index.max()
            days_old = (datetime.now() - latest_date.to_pydatetime()).days
            
            if days_old > 7:
                result.add_warning(f"Data may be stale: latest date is {days_old} days old")

# ============================================
# Parameter Validators
# ============================================

class ParameterValidator(BaseValidator):
    """Validate model and analysis parameters"""
    
    def validate_model_params(self, params: Dict[str, Any], model_type: str) -> ValidationResult:
        """
        Validate model parameters
        
        Args:
            params: Parameter dictionary
            model_type: Type of model (e.g., 'linear', 'polynomial', 'svm')
            
        Returns:
            ValidationResult with validation status
        """
        result = self._create_result()
        
        if not isinstance(params, dict):
            result.add_error("Parameters must be a dictionary")
            return result
        
        # Common parameter validations
        if 'random_state' in params:
            self._validate_random_state(params['random_state'], result)
        
        # Model-specific validations
        if model_type == 'polynomial':
            self._validate_polynomial_params(params, result)
        elif model_type == 'svm':
            self._validate_svm_params(params, result)
        elif model_type == 'random_forest':
            self._validate_random_forest_params(params, result)
        
        return result
    
    def _validate_random_state(self, random_state: Any, result: ValidationResult):
        """Validate random state parameter"""
        if random_state is not None:
            if not isinstance(random_state, (int, np.integer)):
                result.add_error("random_state must be an integer or None")
            elif random_state < 0:
                result.add_error("random_state must be non-negative")
    
    def _validate_polynomial_params(self, params: Dict[str, Any], result: ValidationResult):
        """Validate polynomial regression parameters"""
        if 'degree' in params:
            degree = params['degree']
            if not isinstance(degree, (int, np.integer)):
                result.add_error("Polynomial degree must be an integer")
            elif degree < 1 or degree > 5:
                result.add_error("Polynomial degree must be between 1 and 5")
        
        if 'alpha' in params:
            alpha = params['alpha']
            if not isinstance(alpha, (int, float, np.number)):
                result.add_error("Alpha must be numeric")
            elif alpha < 0:
                result.add_error("Alpha must be non-negative")
    
    def _validate_svm_params(self, params: Dict[str, Any], result: ValidationResult):
        """Validate SVM parameters"""
        valid_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        
        if 'kernel' in params:
            if params['kernel'] not in valid_kernels:
                result.add_error(f"Invalid kernel: {params['kernel']}. Must be one of {valid_kernels}")
        
        if 'C' in params:
            C = params['C']
            if not isinstance(C, (int, float, np.number)):
                result.add_error("C parameter must be numeric")
            elif C <= 0:
                result.add_error("C parameter must be positive")
        
        if 'gamma' in params:
            gamma = params['gamma']
            if gamma not in ['scale', 'auto'] and not isinstance(gamma, (int, float, np.number)):
                result.add_error("Gamma must be 'scale', 'auto', or numeric")
            elif isinstance(gamma, (int, float, np.number)) and gamma <= 0:
                result.add_error("Numeric gamma must be positive")
    
    def _validate_random_forest_params(self, params: Dict[str, Any], result: ValidationResult):
        """Validate Random Forest parameters"""
        if 'n_estimators' in params:
            n_est = params['n_estimators']
            if not isinstance(n_est, (int, np.integer)):
                result.add_error("n_estimators must be an integer")
            elif n_est < 1:
                result.add_error("n_estimators must be positive")
            elif n_est > 1000:
                result.add_warning("n_estimators > 1000 may be slow")
        
        if 'max_depth' in params and params['max_depth'] is not None:
            max_depth = params['max_depth']
            if not isinstance(max_depth, (int, np.integer)):
                result.add_error("max_depth must be an integer or None")
            elif max_depth < 1:
                result.add_error("max_depth must be positive")

# ============================================
# Configuration Validators
# ============================================

class ConfigValidator(BaseValidator):
    """Validate configuration files and settings"""
    
    def validate_config_structure(self, config: Dict[str, Any], required_sections: List[str]) -> ValidationResult:
        """
        Validate configuration structure
        
        Args:
            config: Configuration dictionary
            required_sections: List of required top-level sections
            
        Returns:
            ValidationResult with validation status
        """
        result = self._create_result()
        
        if not isinstance(config, dict):
            result.add_error("Configuration must be a dictionary")
            return result
        
        # Check for required sections
        missing_sections = [section for section in required_sections if section not in config]
        if missing_sections:
            result.add_error(f"Missing required configuration sections: {missing_sections}")
        
        # Check for empty sections
        empty_sections = [section for section in required_sections 
                         if section in config and not config[section]]
        if empty_sections:
            result.add_warning(f"Empty configuration sections: {empty_sections}")
        
        return result

# ============================================
# Composite Validators
# ============================================

class DataPipelineValidator:
    """Comprehensive validator for data pipeline"""
    
    def __init__(self, strict_mode: bool = False):
        self.symbol_validator = SymbolValidator(strict_mode)
        self.date_validator = DateValidator(strict_mode)
        self.data_validator = DataQualityValidator(strict_mode)
        self.param_validator = ParameterValidator(strict_mode)
        
    def validate_data_request(self, symbols: Union[str, List[str]], 
                            start_date: Any, end_date: Any) -> ValidationResult:
        """
        Validate complete data request
        
        Args:
            symbols: Stock symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Combined validation result
        """
        result = ValidationResult()
        
        # Validate symbols
        symbol_result = self.symbol_validator.validate_symbols(symbols)
        result.merge(symbol_result)
        
        # Validate date range
        date_result = self.date_validator.validate_date_range(start_date, end_date)
        result.merge(date_result)
        
        return result
    
    def validate_training_data(self, df: pd.DataFrame, symbol: str, 
                             min_samples: int = 100) -> ValidationResult:
        """
        Validate data for model training
        
        Args:
            df: Training data DataFrame
            symbol: Stock symbol
            min_samples: Minimum required samples
            
        Returns:
            Validation result
        """
        result = ValidationResult()
        
        # Basic data quality validation
        data_result = self.data_validator.validate_dataframe(df, symbol)
        result.merge(data_result)
        
        # Training-specific checks
        if len(df) < min_samples:
            result.add_error(f"Insufficient training data: {len(df)} samples (minimum {min_samples})")
        
        # Check for sufficient variance
        if 'Close' in df.columns:
            price_std = df['Close'].std()
            price_mean = df['Close'].mean()
            
            if price_mean > 0 and (price_std / price_mean) < 0.01:  # Less than 1% coefficient of variation
                result.add_warning("Very low price variance - may not be suitable for training")
        
        return result

# ============================================
# Utility Functions
# ============================================

def validate_and_clean_symbols(symbols: Union[str, List[str]], strict: bool = False) -> List[str]:
    """
    Validate and return clean symbols, raising exception if invalid
    
    Args:
        symbols: Symbols to validate
        strict: Whether to use strict validation
        
    Returns:
        List of valid symbols
        
    Raises:
        InvalidParameterError: If validation fails
    """
    validator = SymbolValidator(strict_mode=strict)
    result = validator.validate_symbols(symbols)
    
    if not result.is_valid:
        raise InvalidParameterError(
            f"Symbol validation failed: {'; '.join(result.errors)}",
            parameter_name="symbols",
            provided_value=symbols
        )
    
    # Return cleaned symbols
    if isinstance(symbols, str):
        return [s.strip().upper() for s in symbols.split(',') if s.strip()]
    else:
        return [str(s).strip().upper() for s in symbols if s]

def validate_data_for_analysis(df: pd.DataFrame, symbol: str, 
                             analysis_type: str = "general") -> ValidationResult:
    """
    Validate data for specific type of analysis
    
    Args:
        df: Data to validate
        symbol: Stock symbol
        analysis_type: Type of analysis ('general', 'modeling', 'backtesting')
        
    Returns:
        Validation result
    """
    validator = DataQualityValidator(strict_mode=(analysis_type == 'modeling'))
    result = validator.validate_dataframe(df, symbol)
    
    # Analysis-specific requirements
    if analysis_type == 'modeling':
        if len(df) < 252:  # Less than 1 year of daily data
            result.add_warning("Limited data for modeling (less than 1 year)")
    elif analysis_type == 'backtesting':
        if len(df) < 504:  # Less than 2 years of daily data
            result.add_warning("Limited data for backtesting (less than 2 years)")
    
    return result

def create_validation_report(results: List[ValidationResult], title: str = "Validation Report") -> str:
    """
    Create formatted validation report
    
    Args:
        results: List of validation results
        title: Report title
        
    Returns:
        Formatted report string
    """
    lines = [
        f"\n{title}",
        "=" * len(title),
        ""
    ]
    
    total_errors = sum(len(r.errors) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)
    valid_count = sum(1 for r in results if r.is_valid)
    
    lines.append(f"Summary: {valid_count}/{len(results)} validations passed")
    lines.append(f"Total Errors: {total_errors}")
    lines.append(f"Total Warnings: {total_warnings}")
    lines.append("")
    
    for i, result in enumerate(results, 1):
        status = "✓ PASS" if result.is_valid else "✗ FAIL"
        lines.append(f"{i}. {status}")
        
        if result.errors:
            for error in result.errors:
                lines.append(f"   ERROR: {error}")
        
        if result.warnings:
            for warning in result.warnings:
                lines.append(f"   WARNING: {warning}")
        
        lines.append("")
    
    return "\n".join(lines)

# Export commonly used validators
symbol_validator = SymbolValidator()
date_validator = DateValidator()
data_validator = DataQualityValidator()
param_validator = ParameterValidator()
pipeline_validator = DataPipelineValidator()
