# ============================================
# StockPredictionPro - src/data/validators.py
# Comprehensive data validation system for financial time series
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ..utils.exceptions import DataValidationError, BusinessLogicError
from ..utils.logger import get_logger
from ..utils.timing import Timer, time_it
from ..utils.config_loader import get
from ..utils.helpers import safe_divide, validate_numeric_range
from ..utils.validators import ValidationResult

logger = get_logger('data.validators')

# ============================================
# Validation Enums and Types
# ============================================

class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DataQualityLevel(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"      # 95-100%
    GOOD = "good"               # 85-94%
    ACCEPTABLE = "acceptable"   # 70-84%
    POOR = "poor"              # 50-69%
    UNACCEPTABLE = "unacceptable"  # <50%

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    issue_type: str
    severity: ValidationSeverity
    message: str
    column: Optional[str] = None
    row_indices: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'issue_type': self.issue_type,
            'severity': self.severity.value,
            'message': self.message,
            'column': self.column,
            'row_indices': self.row_indices,
            'metadata': self.metadata
        }

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    data_shape: Tuple[int, int]
    validation_time: datetime
    quality_score: float
    quality_level: DataQualityLevel
    issues: List[ValidationIssue] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if data passes validation"""
        critical_issues = [issue for issue in self.issues if issue.severity == ValidationSeverity.CRITICAL]
        error_issues = [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
        return len(critical_issues) == 0 and len(error_issues) == 0
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues by severity level"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'data_shape': self.data_shape,
            'validation_time': self.validation_time.isoformat(),
            'quality_score': self.quality_score,
            'quality_level': self.quality_level.value,
            'is_valid': self.is_valid(),
            'issues_count': len(self.issues),
            'issues_by_severity': {
                severity.value: len(self.get_issues_by_severity(severity))
                for severity in ValidationSeverity
            },
            'issues': [issue.to_dict() for issue in self.issues],
            'summary_stats': self.summary_stats,
            'recommendations': self.recommendations
        }

# ============================================
# Base Validator Classes
# ============================================

class BaseValidator:
    """Base class for all data validators"""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.WARNING):
        """
        Initialize base validator
        
        Args:
            name: Validator name
            severity: Default severity level
        """
        self.name = name
        self.severity = severity
        self.logger = get_logger(f'data.validators.{name}')
    
    def validate(self, df: pd.DataFrame, symbol: Optional[str] = None) -> List[ValidationIssue]:
        """
        Validate data and return issues
        
        Args:
            df: DataFrame to validate
            symbol: Stock symbol for context
            
        Returns:
            List of validation issues
        """
        raise NotImplementedError("Subclasses must implement validate method")
    
    def _create_issue(self, issue_type: str, message: str, 
                     severity: Optional[ValidationSeverity] = None,
                     column: Optional[str] = None,
                     row_indices: Optional[List[int]] = None,
                     **metadata) -> ValidationIssue:
        """Create a validation issue"""
        return ValidationIssue(
            issue_type=issue_type,
            severity=severity or self.severity,
            message=message,
            column=column,
            row_indices=row_indices,
            metadata=metadata
        )

# ============================================
# Specific Validators
# ============================================

class DataTypeValidator(BaseValidator):
    """Validate data types and structure"""
    
    def __init__(self):
        super().__init__("data_type", ValidationSeverity.ERROR)
    
    def validate(self, df: pd.DataFrame, symbol: Optional[str] = None) -> List[ValidationIssue]:
        """Validate data types"""
        issues = []
        
        # Check if DataFrame is valid
        if df is None:
            issues.append(self._create_issue(
                "null_dataframe", "DataFrame is None", ValidationSeverity.CRITICAL
            ))
            return issues
        
        if df.empty:
            issues.append(self._create_issue(
                "empty_dataframe", "DataFrame is empty", ValidationSeverity.CRITICAL
            ))
            return issues
        
        # Check index type
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append(self._create_issue(
                "invalid_index", "Index must be DatetimeIndex for time series data",
                ValidationSeverity.ERROR
            ))
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issues.append(self._create_issue(
                "missing_columns", f"Missing required OHLCV columns: {missing_columns}",
                ValidationSeverity.ERROR, metadata={'missing_columns': missing_columns}
            ))
        
        # Validate numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        expected_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in expected_numeric:
            if col in df.columns and col not in numeric_columns:
                issues.append(self._create_issue(
                    "invalid_data_type", f"Column '{col}' should be numeric",
                    ValidationSeverity.ERROR, column=col
                ))
        
        return issues

class MissingDataValidator(BaseValidator):
    """Validate missing data patterns"""
    
    def __init__(self, max_missing_pct: float = 10.0):
        super().__init__("missing_data", ValidationSeverity.WARNING)
        self.max_missing_pct = max_missing_pct
    
    def validate(self, df: pd.DataFrame, symbol: Optional[str] = None) -> List[ValidationIssue]:
        """Validate missing data"""
        issues = []
        
        if df.empty:
            return issues
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                severity = ValidationSeverity.WARNING
                if missing_pct > self.max_missing_pct:
                    severity = ValidationSeverity.ERROR
                elif missing_pct > 50:
                    severity = ValidationSeverity.CRITICAL
                
                # Find missing data patterns
                missing_mask = df[column].isnull()
                consecutive_missing = self._find_consecutive_missing(missing_mask)
                
                issues.append(self._create_issue(
                    "missing_data", 
                    f"Column '{column}' has {missing_count} missing values ({missing_pct:.1f}%)",
                    severity, column=column,
                    metadata={
                        'missing_count': missing_count,
                        'missing_percentage': missing_pct,
                        'max_consecutive': max(consecutive_missing) if consecutive_missing else 0,
                        'missing_periods': len(consecutive_missing)
                    }
                ))
        
        return issues
    
    def _find_consecutive_missing(self, mask: pd.Series) -> List[int]:
        """Find consecutive missing periods"""
        consecutive_periods = []
        current_period = 0
        
        for is_missing in mask:
            if is_missing:
                current_period += 1
            else:
                if current_period > 0:
                    consecutive_periods.append(current_period)
                current_period = 0
        
        # Add final period if it ends with missing values
        if current_period > 0:
            consecutive_periods.append(current_period)
        
        return consecutive_periods

class OHLCVValidator(BaseValidator):
    """Validate OHLCV relationships and constraints"""
    
    def __init__(self):
        super().__init__("ohlcv", ValidationSeverity.ERROR)
    
    def validate(self, df: pd.DataFrame, symbol: Optional[str] = None) -> List[ValidationIssue]:
        """Validate OHLCV relationships"""
        issues = []
        
        if df.empty:
            return issues
        
        # Check if we have OHLCV columns
        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in ohlcv_columns if col in df.columns]
        
        if len(available_columns) < 4:  # Need at least OHLC
            return issues
        
        # Validate price relationships
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            issues.extend(self._validate_price_relationships(df))
        
        # Validate price values
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                issues.extend(self._validate_price_values(df, col))
        
        # Validate volume
        if 'Volume' in df.columns:
            issues.extend(self._validate_volume(df))
        
        return issues
    
    def _validate_price_relationships(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate OHLC price relationships"""
        issues = []
        
        # High should be >= max(Open, Low, Close)
        max_olc = df[['Open', 'Low', 'Close']].max(axis=1)
        invalid_high = df['High'] < max_olc
        invalid_high_indices = df[invalid_high].index.tolist()
        
        if len(invalid_high_indices) > 0:
            issues.append(self._create_issue(
                "invalid_high_price",
                f"High price is lower than Open/Low/Close in {len(invalid_high_indices)} records",
                ValidationSeverity.ERROR,
                column="High",
                row_indices=[df.index.get_loc(idx) for idx in invalid_high_indices[:10]],  # Limit to first 10
                metadata={'violation_count': len(invalid_high_indices)}
            ))
        
        # Low should be <= min(Open, High, Close)
        min_ohc = df[['Open', 'High', 'Close']].min(axis=1)
        invalid_low = df['Low'] > min_ohc
        invalid_low_indices = df[invalid_low].index.tolist()
        
        if len(invalid_low_indices) > 0:
            issues.append(self._create_issue(
                "invalid_low_price",
                f"Low price is higher than Open/High/Close in {len(invalid_low_indices)} records",
                ValidationSeverity.ERROR,
                column="Low",
                row_indices=[df.index.get_loc(idx) for idx in invalid_low_indices[:10]],
                metadata={'violation_count': len(invalid_low_indices)}
            ))
        
        return issues
    
    def _validate_price_values(self, df: pd.DataFrame, column: str) -> List[ValidationIssue]:
        """Validate individual price column values"""
        issues = []
        
        # Check for negative prices
        negative_prices = df[column] < 0
        negative_indices = df[negative_prices].index.tolist()
        
        if len(negative_indices) > 0:
            issues.append(self._create_issue(
                "negative_price",
                f"Column '{column}' has {len(negative_indices)} negative values",
                ValidationSeverity.ERROR,
                column=column,
                row_indices=[df.index.get_loc(idx) for idx in negative_indices[:10]],
                metadata={'violation_count': len(negative_indices)}
            ))
        
        # Check for zero prices
        zero_prices = df[column] == 0
        zero_indices = df[zero_prices].index.tolist()
        
        if len(zero_indices) > 0:
            issues.append(self._create_issue(
                "zero_price",
                f"Column '{column}' has {len(zero_indices)} zero values",
                ValidationSeverity.WARNING,
                column=column,
                metadata={'violation_count': len(zero_indices)}
            ))
        
        # Check for extreme values (outliers)
        if not df[column].empty:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            extreme_values = (df[column] < lower_bound) | (df[column] > upper_bound)
            extreme_indices = df[extreme_values].index.tolist()
            
            if len(extreme_indices) > len(df) * 0.05:  # More than 5% outliers
                issues.append(self._create_issue(
                    "extreme_price_values",
                    f"Column '{column}' has {len(extreme_indices)} extreme outlier values ({len(extreme_indices)/len(df)*100:.1f}%)",
                    ValidationSeverity.WARNING,
                    column=column,
                    metadata={'outlier_count': len(extreme_indices), 'outlier_percentage': len(extreme_indices)/len(df)*100}
                ))
        
        return issues
    
    def _validate_volume(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate volume data"""
        issues = []
        
        # Check for negative volume
        negative_volume = df['Volume'] < 0
        negative_indices = df[negative_volume].index.tolist()
        
        if len(negative_indices) > 0:
            issues.append(self._create_issue(
                "negative_volume",
                f"Volume has {len(negative_indices)} negative values",
                ValidationSeverity.ERROR,
                column="Volume",
                row_indices=[df.index.get_loc(idx) for idx in negative_indices[:10]],
                metadata={'violation_count': len(negative_indices)}
            ))
        
        # Check for zero volume (might be acceptable for weekends/holidays)
        zero_volume = df['Volume'] == 0
        zero_volume_count = zero_volume.sum()
        zero_volume_pct = (zero_volume_count / len(df)) * 100
        
        if zero_volume_pct > 10:  # More than 10% zero volume days
            issues.append(self._create_issue(
                "excessive_zero_volume",
                f"Volume has {zero_volume_count} zero values ({zero_volume_pct:.1f}% of data)",
                ValidationSeverity.WARNING,
                column="Volume",
                metadata={'zero_volume_count': zero_volume_count, 'zero_volume_percentage': zero_volume_pct}
            ))
        
        return issues

class TemporalValidator(BaseValidator):
    """Validate temporal aspects of time series data"""
    
    def __init__(self, max_gap_days: int = 7):
        super().__init__("temporal", ValidationSeverity.WARNING)
        self.max_gap_days = max_gap_days
    
    def validate(self, df: pd.DataFrame, symbol: Optional[str] = None) -> List[ValidationIssue]:
        """Validate temporal data aspects"""
        issues = []
        
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return issues
        
        # Check chronological order
        if not df.index.is_monotonic_increasing:
            issues.append(self._create_issue(
                "non_chronological",
                "Data is not in chronological order",
                ValidationSeverity.ERROR
            ))
        
        # Check for duplicate dates
        duplicate_dates = df.index.duplicated().sum()
        if duplicate_dates > 0:
            issues.append(self._create_issue(
                "duplicate_dates",
                f"Found {duplicate_dates} duplicate date entries",
                ValidationSeverity.ERROR,
                metadata={'duplicate_count': duplicate_dates}
            ))
        
        # Check for large gaps in data
        if len(df) > 1:
            date_diffs = df.index.to_series().diff().dt.days.dropna()
            large_gaps = date_diffs[date_diffs > self.max_gap_days]
            
            if len(large_gaps) > 0:
                max_gap = large_gaps.max()
                issues.append(self._create_issue(
                    "large_data_gaps",
                    f"Found {len(large_gaps)} gaps larger than {self.max_gap_days} days (max gap: {max_gap} days)",
                    ValidationSeverity.WARNING,
                    metadata={'gap_count': len(large_gaps), 'max_gap_days': max_gap}
                ))
        
        # Check for weekend data (might be unexpected for stocks)
        weekend_data = df.index.weekday >= 5  # Saturday=5, Sunday=6
        weekend_count = weekend_data.sum()
        
        if weekend_count > 0:
            weekend_pct = (weekend_count / len(df)) * 100
            severity = ValidationSeverity.INFO if weekend_pct < 5 else ValidationSeverity.WARNING
            
            issues.append(self._create_issue(
                "weekend_data",
                f"Found {weekend_count} records on weekends ({weekend_pct:.1f}% of data)",
                severity,
                metadata={'weekend_count': weekend_count, 'weekend_percentage': weekend_pct}
            ))
        
        return issues

class StatisticalValidator(BaseValidator):
    """Validate statistical properties of the data"""
    
    def __init__(self):
        super().__init__("statistical", ValidationSeverity.WARNING)
    
    def validate(self, df: pd.DataFrame, symbol: Optional[str] = None) -> List[ValidationIssue]:
        """Validate statistical properties"""
        issues = []
        
        if df.empty:
            return issues
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if df[column].isnull().all():
                continue
                
            issues.extend(self._validate_distribution(df, column))
            issues.extend(self._validate_variance(df, column))
            issues.extend(self._validate_autocorrelation(df, column))
        
        return issues
    
    def _validate_distribution(self, df: pd.DataFrame, column: str) -> List[ValidationIssue]:
        """Validate data distribution"""
        issues = []
        
        data = df[column].dropna()
        if len(data) < 30:  # Need sufficient data for statistical tests
            return issues
        
        # Check for constant values
        if data.nunique() == 1:
            issues.append(self._create_issue(
                "constant_values",
                f"Column '{column}' has constant values",
                ValidationSeverity.WARNING,
                column=column,
                metadata={'unique_values': 1}
            ))
            return issues
        
        # Check variance
        variance = data.var()
        if variance < 1e-10:  # Very low variance
            issues.append(self._create_issue(
                "low_variance",
                f"Column '{column}' has very low variance ({variance:.2e})",
                ValidationSeverity.WARNING,
                column=column,
                metadata={'variance': variance}
            ))
        
        # Check for extreme skewness
        try:
            skewness = stats.skew(data)
            if abs(skewness) > 5:
                issues.append(self._create_issue(
                    "extreme_skewness",
                    f"Column '{column}' has extreme skewness ({skewness:.2f})",
                    ValidationSeverity.INFO,
                    column=column,
                    metadata={'skewness': skewness}
                ))
        except Exception:
            pass
        
        # Check for extreme kurtosis
        try:
            kurt = stats.kurtosis(data)
            if abs(kurt) > 10:
                issues.append(self._create_issue(
                    "extreme_kurtosis",
                    f"Column '{column}' has extreme kurtosis ({kurt:.2f})",
                    ValidationSeverity.INFO,
                    column=column,
                    metadata={'kurtosis': kurt}
                ))
        except Exception:
            pass
        
        return issues
    
    def _validate_variance(self, df: pd.DataFrame, column: str) -> List[ValidationIssue]:
        """Validate variance properties"""
        issues = []
        
        if column in ['Open', 'High', 'Low', 'Close']:
            # Calculate returns for price columns
            returns = df[column].pct_change().dropna()
            
            if len(returns) < 50:
                return issues
            
            # Check for volatility clustering (ARCH effects)
            try:
                squared_returns = returns ** 2
                
                # Simple test for volatility clustering
                # High autocorrelation in squared returns suggests ARCH effects
                if len(squared_returns) > 10:
                    autocorr = squared_returns.autocorr(lag=1)
                    
                    if autocorr > 0.3:  # Strong autocorrelation in squared returns
                        issues.append(self._create_issue(
                            "volatility_clustering",
                            f"Column '{column}' shows evidence of volatility clustering (autocorr: {autocorr:.3f})",
                            ValidationSeverity.INFO,
                            column=column,
                            metadata={'volatility_autocorr': autocorr}
                        ))
            except Exception:
                pass
        
        return issues
    
    def _validate_autocorrelation(self, df: pd.DataFrame, column: str) -> List[ValidationIssue]:
        """Validate autocorrelation properties"""
        issues = []
        
        data = df[column].dropna()
        if len(data) < 50:
            return issues
        
        try:
            # Check for high autocorrelation (might indicate non-stationarity)
            autocorr_1 = data.autocorr(lag=1)
            
            if autocorr_1 > 0.95:  # Very high autocorrelation
                issues.append(self._create_issue(
                    "high_autocorrelation",
                    f"Column '{column}' has very high autocorrelation ({autocorr_1:.3f}), possibly non-stationary",
                    ValidationSeverity.INFO,
                    column=column,
                    metadata={'lag_1_autocorr': autocorr_1}
                ))
        except Exception:
            pass
        
        return issues

class BusinessLogicValidator(BaseValidator):
    """Validate business logic specific to financial data"""
    
    def __init__(self):
        super().__init__("business_logic", ValidationSeverity.WARNING)
    
    def validate(self, df: pd.DataFrame, symbol: Optional[str] = None) -> List[ValidationIssue]:
        """Validate business logic"""
        issues = []
        
        if df.empty:
            return issues
        
        # Validate price ranges
        issues.extend(self._validate_price_ranges(df, symbol))
        
        # Validate volume patterns
        issues.extend(self._validate_volume_patterns(df))
        
        # Validate price movements
        issues.extend(self._validate_price_movements(df))
        
        return issues
    
    def _validate_price_ranges(self, df: pd.DataFrame, symbol: Optional[str]) -> List[ValidationIssue]:
        """Validate reasonable price ranges"""
        issues = []
        
        price_columns = ['Open', 'High', 'Low', 'Close']
        
        for column in price_columns:
            if column not in df.columns:
                continue
                
            prices = df[column].dropna()
            if prices.empty:
                continue
            
            min_price = prices.min()
            max_price = prices.max()
            
            # Check for unreasonably low prices
            if min_price < 0.01:  # Less than 1 cent
                issues.append(self._create_issue(
                    "unreasonable_low_price",
                    f"Column '{column}' has unreasonably low prices (min: ${min_price:.4f})",
                    ValidationSeverity.WARNING,
                    column=column,
                    metadata={'min_price': min_price}
                ))
            
            # Check for unreasonably high prices (basic sanity check)
            if max_price > 100000:  # More than $100,000 per share
                issues.append(self._create_issue(
                    "unreasonable_high_price",
                    f"Column '{column}' has unreasonably high prices (max: ${max_price:.2f})",
                    ValidationSeverity.WARNING,
                    column=column,
                    metadata={'max_price': max_price}
                ))
        
        return issues
    
    def _validate_volume_patterns(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate volume patterns"""
        issues = []
        
        if 'Volume' not in df.columns:
            return issues
        
        volume = df['Volume'].dropna()
        if volume.empty:
            return issues
        
        # Check for suspiciously round volumes (might indicate manipulation)
        round_volumes = volume % 1000 == 0  # Exact thousands
        round_volume_pct = (round_volumes.sum() / len(volume)) * 100
        
        if round_volume_pct > 30:  # More than 30% round volumes
            issues.append(self._create_issue(
                "suspicious_round_volumes",
                f"Volume has {round_volume_pct:.1f}% of values as exact thousands (potentially suspicious)",
                ValidationSeverity.INFO,
                column="Volume",
                metadata={'round_volume_percentage': round_volume_pct}
            ))
        
        return issues
    
    def _validate_price_movements(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate price movement patterns"""
        issues = []
        
        if 'Close' not in df.columns:
            return issues
        
        close_prices = df['Close'].dropna()
        if len(close_prices) < 2:
            return issues
        
        # Calculate daily returns
        returns = close_prices.pct_change().dropna()
        
        if returns.empty:
            return issues
        
        # Check for extreme single-day movements
        extreme_returns = returns.abs() > 0.5  # More than 50% in a single day
        extreme_count = extreme_returns.sum()
        
        if extreme_count > 0:
            max_return = returns.abs().max()
            issues.append(self._create_issue(
                "extreme_price_movements",
                f"Found {extreme_count} extreme single-day price movements (max: {max_return*100:.1f}%)",
                ValidationSeverity.WARNING,
                column="Close",
                metadata={
                    'extreme_movement_count': extreme_count,
                    'max_single_day_return': max_return
                }
            ))
        
        # Check for too many consecutive identical closes
        consecutive_identical = (close_prices == close_prices.shift(1)).sum()
        consecutive_pct = (consecutive_identical / len(close_prices)) * 100
        
        if consecutive_pct > 10:  # More than 10% identical consecutive closes
            issues.append(self._create_issue(
                "consecutive_identical_prices",
                f"Found {consecutive_identical} consecutive identical closing prices ({consecutive_pct:.1f}% of data)",
                ValidationSeverity.WARNING,
                column="Close",
                metadata={
                    'consecutive_identical_count': consecutive_identical,
                    'consecutive_identical_percentage': consecutive_pct
                }
            ))
        
        return issues

# ============================================
# Main Data Quality Validator
# ============================================

class DataQualityValidator:
    """
    Comprehensive data quality validation system
    
    Features:
    - Multiple validation categories
    - Configurable severity levels
    - Quality scoring
    - Detailed reporting
    - Recommendations generation
    """
    
    def __init__(self, 
                 validators: Optional[List[BaseValidator]] = None,
                 quality_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize data quality validator
        
        Args:
            validators: List of validators to use (None for default)
            quality_thresholds: Custom quality thresholds
        """
        self.validators = validators or self._create_default_validators()
        
        self.quality_thresholds = quality_thresholds or {
            DataQualityLevel.EXCELLENT.value: 95.0,
            DataQualityLevel.GOOD.value: 85.0,
            DataQualityLevel.ACCEPTABLE.value: 70.0,
            DataQualityLevel.POOR.value: 50.0
        }
        
        logger.info(f"Data quality validator initialized with {len(self.validators)} validators")
    
    def _create_default_validators(self) -> List[BaseValidator]:
        """Create default set of validators"""
        return [
            DataTypeValidator(),
            MissingDataValidator(max_missing_pct=10.0),
            OHLCVValidator(),
            TemporalValidator(max_gap_days=7),
            StatisticalValidator(),
            BusinessLogicValidator()
        ]
    
    @time_it("data_quality_validation")
    def validate_data(self, df: pd.DataFrame, symbol: Optional[str] = None) -> ValidationReport:
        """
        Perform comprehensive data quality validation
        
        Args:
            df: DataFrame to validate
            symbol: Stock symbol for context
            
        Returns:
            Comprehensive validation report
        """
        logger.info(f"Starting data quality validation for {symbol or 'unknown symbol'}")
        
        # Collect all issues
        all_issues = []
        
        for validator in self.validators:
            try:
                with Timer(f"validator_{validator.name}") as timer:
                    issues = validator.validate(df, symbol)
                    all_issues.extend(issues)
                
                logger.debug(f"Validator '{validator.name}' found {len(issues)} issues in {timer.result.duration_str}")
                
            except Exception as e:
                logger.error(f"Validator '{validator.name}' failed: {e}")
                all_issues.append(ValidationIssue(
                    issue_type="validator_error",
                    severity=ValidationSeverity.ERROR,
                    message=f"Validator '{validator.name}' failed: {str(e)}"
                ))
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(df, all_issues)
        quality_level = self._determine_quality_level(quality_score)
        
        # Generate summary statistics
        summary_stats = self._generate_summary_stats(df, all_issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues, quality_score)
        
        # Create validation report
        report = ValidationReport(
            data_shape=df.shape,
            validation_time=datetime.now(),
            quality_score=quality_score,
            quality_level=quality_level,
            issues=all_issues,
            summary_stats=summary_stats,
            recommendations=recommendations
        )
        
        logger.info(f"Data quality validation complete: Score {quality_score:.1f}/100 ({quality_level.value})")
        
        return report
    
    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[ValidationIssue]) -> float:
        """Calculate overall data quality score (0-100)"""
        
        if df.empty:
            return 0.0
        
        base_score = 100.0
        
        # Deduct points based on severity and frequency
        severity_weights = {
            ValidationSeverity.CRITICAL: 25.0,
            ValidationSeverity.ERROR: 10.0,
            ValidationSeverity.WARNING: 3.0,
            ValidationSeverity.INFO: 1.0
        }
        
        for issue in issues:
            deduction = severity_weights.get(issue.severity, 1.0)
            
            # Adjust deduction based on issue metadata
            if 'violation_count' in issue.metadata:
                # Scale deduction based on percentage of affected data
                violation_pct = (issue.metadata['violation_count'] / len(df)) * 100
                deduction *= min(violation_pct / 10, 3.0)  # Cap at 3x multiplier
            
            base_score -= deduction
        
        # Ensure score doesn't go below 0
        return max(0.0, base_score)
    
    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """Determine quality level based on score"""
        
        if score >= self.quality_thresholds[DataQualityLevel.EXCELLENT.value]:
            return DataQualityLevel.EXCELLENT
        elif score >= self.quality_thresholds[DataQualityLevel.GOOD.value]:
            return DataQualityLevel.GOOD
        elif score >= self.quality_thresholds[DataQualityLevel.ACCEPTABLE.value]:
            return DataQualityLevel.ACCEPTABLE
        elif score >= self.quality_thresholds[DataQualityLevel.POOR.value]:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.UNACCEPTABLE
    
    def _generate_summary_stats(self, df: pd.DataFrame, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        if df.empty:
            return {}
        
        # Basic data stats
        stats = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'date_range': None,
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Date range if datetime index
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            stats['date_range'] = {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat(),
                'total_days': (df.index.max() - df.index.min()).days
            }
        
        # Issue statistics
        issue_counts = {}
        for severity in ValidationSeverity:
            issue_counts[severity.value] = len([i for i in issues if i.severity == severity])
        
        stats['issue_counts'] = issue_counts
        
        # Column-specific stats
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            stats['numeric_columns_stats'] = {}
            for col in numeric_columns:
                if not df[col].empty and not df[col].isnull().all():
                    stats['numeric_columns_stats'][col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'missing_count': int(df[col].isnull().sum())
                    }
        
        return stats
    
    def _generate_recommendations(self, issues: List[ValidationIssue], quality_score: float) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Group issues by type
        issue_types = {}
        for issue in issues:
            if issue.issue_type not in issue_types:
                issue_types[issue.issue_type] = []
            issue_types[issue.issue_type].append(issue)
        
        # Generate recommendations based on issue types
        if 'missing_data' in issue_types:
            missing_issues = issue_types['missing_data']
            high_missing = [i for i in missing_issues if i.metadata.get('missing_percentage', 0) > 10]
            if high_missing:
                recommendations.append("Consider imputation or removal of columns with >10% missing data")
        
        if 'invalid_high_price' in issue_types or 'invalid_low_price' in issue_types:
            recommendations.append("Fix OHLC relationship violations before using data for modeling")
        
        if 'negative_price' in issue_types or 'negative_volume' in issue_types:
            recommendations.append("Remove or correct negative price/volume values")
        
        if 'extreme_price_movements' in issue_types:
            recommendations.append("Review extreme price movements - may indicate stock splits or data errors")
        
        if 'large_data_gaps' in issue_types:
            recommendations.append("Consider filling large data gaps or adjusting analysis to account for missing periods")
        
        if 'non_chronological' in issue_types:
            recommendations.append("Sort data chronologically before analysis")
        
        if 'duplicate_dates' in issue_types:
            recommendations.append("Remove duplicate date entries")
        
        # Quality-based recommendations
        if quality_score < 70:
            recommendations.append("Data quality is below acceptable threshold - comprehensive cleaning recommended")
        elif quality_score < 85:
            recommendations.append("Data quality is acceptable but could be improved with additional cleaning")
        
        # Default recommendation if no specific issues
        if not recommendations:
            recommendations.append("Data quality is good - ready for analysis")
        
        return recommendations
    
    def validate_for_modeling(self, df: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[bool, ValidationReport]:
        """
        Validate data specifically for machine learning modeling
        
        Args:
            df: DataFrame to validate
            symbol: Stock symbol
            
        Returns:
            Tuple of (is_ready_for_modeling, validation_report)
        """
        
        report = self.validate_data(df, symbol)
        
        # Determine if data is ready for modeling
        critical_issues = report.get_issues_by_severity(ValidationSeverity.CRITICAL)
        error_issues = report.get_issues_by_severity(ValidationSeverity.ERROR)
        
        # Specific modeling readiness criteria
        modeling_ready = (
            len(critical_issues) == 0 and
            len(error_issues) == 0 and
            report.quality_score >= 70 and
            df.shape[0] >= 100 and  # Minimum 100 records
            not df.empty
        )
        
        if not modeling_ready:
            if len(critical_issues) > 0:
                report.recommendations.insert(0, "Critical data issues must be resolved before modeling")
            elif len(error_issues) > 0:
                report.recommendations.insert(0, "Data errors must be fixed before modeling")
            elif report.quality_score < 70:
                report.recommendations.insert(0, "Data quality must be improved before modeling")
            elif df.shape[0] < 100:
                report.recommendations.insert(0, "Insufficient data for reliable modeling (need >100 records)")
        
        logger.info(f"Modeling readiness assessment: {'✅ Ready' if modeling_ready else '❌ Not ready'}")
        
        return modeling_ready, report

# ============================================
# Factory Functions and Utilities
# ============================================

def create_validator(validator_type: str = 'comprehensive', **kwargs) -> DataQualityValidator:
    """
    Create pre-configured validator
    
    Args:
        validator_type: Type of validator ('basic', 'comprehensive', 'strict')
        **kwargs: Additional configuration
        
    Returns:
        Configured DataQualityValidator
    """
    
    if validator_type == 'basic':
        validators = [
            DataTypeValidator(),
            MissingDataValidator(max_missing_pct=15.0),
            OHLCVValidator()
        ]
    elif validator_type == 'comprehensive':
        validators = [
            DataTypeValidator(),
            MissingDataValidator(max_missing_pct=10.0),
            OHLCVValidator(),
            TemporalValidator(max_gap_days=7),
            StatisticalValidator(),
            BusinessLogicValidator()
        ]
    elif validator_type == 'strict':
        validators = [
            DataTypeValidator(),
            MissingDataValidator(max_missing_pct=5.0),
            OHLCVValidator(),
            TemporalValidator(max_gap_days=3),
            StatisticalValidator(),
            BusinessLogicValidator()
        ]
        
        # Stricter quality thresholds
        kwargs['quality_thresholds'] = {
            DataQualityLevel.EXCELLENT.value: 98.0,
            DataQualityLevel.GOOD.value: 90.0,
            DataQualityLevel.ACCEPTABLE.value: 80.0,
            DataQualityLevel.POOR.value: 60.0
        }
    else:
        raise ValueError(f"Unknown validator type: {validator_type}")
    
    return DataQualityValidator(validators=validators, **kwargs)

def quick_validate(df: pd.DataFrame, symbol: Optional[str] = None) -> ValidationReport:
    """Quick validation using comprehensive validator"""
    validator = create_validator('comprehensive')
    return validator.validate_data(df, symbol)

def validate_for_research(df: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[bool, ValidationReport]:
    """Validate data for research purposes using strict criteria"""
    validator = create_validator('strict')
    return validator.validate_for_modeling(df, symbol)

# Global validator instance
_default_validator: Optional[DataQualityValidator] = None

def get_default_validator() -> DataQualityValidator:
    """Get default validator instance"""
    global _default_validator
    
    if _default_validator is None:
        _default_validator = create_validator('comprehensive')
    
    return _default_validator
