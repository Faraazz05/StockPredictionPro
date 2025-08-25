"""
data/validate_data.py

Advanced data validation and quality assurance for StockPredictionPro.
Comprehensive checks for market data integrity, completeness, and consistency.
Provides detailed reporting and automated quality scoring.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
import warnings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockPredictionPro.DataValidator')

# ============================================
# VALIDATION CONFIGURATION AND ENUMS
# ============================================

class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

class ValidationCategory(Enum):
    """Categories of validation checks"""
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    INTEGRITY = "integrity"
    BUSINESS_RULES = "business_rules"

@dataclass
class ValidationResult:
    """Result of a single validation check"""
    check_name: str
    category: ValidationCategory
    level: ValidationLevel
    passed: bool
    message: str
    details: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ValidationConfig:
    """Configuration for validation rules"""
    # Missing data thresholds
    max_missing_ratio: float = 0.1  # 10% max missing per column
    max_missing_consecutive: int = 5  # Max consecutive missing values
    
    # Statistical anomaly detection
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    
    # Business rules for financial data
    min_price: float = 0.01  # Minimum valid stock price
    max_price: float = 10000.0  # Maximum reasonable stock price
    min_volume: int = 0  # Minimum valid volume
    max_volume: int = 1_000_000_000  # Maximum reasonable daily volume
    
    # Time series validation
    allow_weekends: bool = False
    allow_holidays: bool = False
    max_time_gap_days: int = 5
    
    # Data freshness
    max_data_age_days: int = 7

# ============================================
# CORE VALIDATION ENGINE
# ============================================

class MarketDataValidator:
    """Comprehensive market data validation system"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.results: List[ValidationResult] = []
        self.validation_score = 0.0
        self.start_time = None
        self.end_time = None
        
        # Validation statistics
        self.stats = {
            'total_checks': 0,
            'passed_checks': 0,
            'warnings': 0,
            'errors': 0,
            'critical_errors': 0
        }
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          expected_columns: List[str] = None,
                          symbol_column: str = 'symbol',
                          timestamp_column: str = 'timestamp') -> Dict[str, Any]:
        """
        Main validation method for pandas DataFrame
        
        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            symbol_column: Name of symbol column
            timestamp_column: Name of timestamp column
            
        Returns:
            Validation report dictionary
        """
        logger.info("🔍 Starting comprehensive data validation...")
        self.start_time = datetime.now()
        self.results = []
        
        # Reset statistics
        self.stats = {k: 0 for k in self.stats.keys()}
        
        # Run validation checks
        self._validate_schema(df, expected_columns)
        self._validate_completeness(df)
        self._validate_data_types(df)
        self._validate_time_series(df, timestamp_column)
        self._validate_business_rules(df)
        self._validate_statistical_properties(df)
        self._validate_consistency(df, symbol_column)
        self._validate_freshness(df, timestamp_column)
        
        self.end_time = datetime.now()
        
        # Calculate validation score
        self._calculate_validation_score()
        
        # Generate report
        report = self._generate_validation_report(df)
        
        logger.info(f"✅ Validation completed. Score: {self.validation_score:.1f}/100")
        return report
    
    def _validate_schema(self, df: pd.DataFrame, expected_columns: List[str] = None) -> None:
        """Validate DataFrame schema and structure"""
        logger.info("📋 Validating schema and structure...")
        
        # Check if DataFrame is empty
        if df.empty:
            self._add_result(
                "empty_dataframe",
                ValidationCategory.INTEGRITY,
                ValidationLevel.CRITICAL,
                False,
                "DataFrame is empty",
                {"row_count": 0, "column_count": 0}
            )
            return
        
        # Check expected columns
        if expected_columns:
            missing_columns = set(expected_columns) - set(df.columns)
            if missing_columns:
                self._add_result(
                    "missing_columns",
                    ValidationCategory.INTEGRITY,
                    ValidationLevel.ERROR,
                    False,
                    f"Missing required columns: {missing_columns}",
                    {"missing_columns": list(missing_columns)}
                )
        
        # Check for duplicate column names
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            self._add_result(
                "duplicate_columns",
                ValidationCategory.INTEGRITY,
                ValidationLevel.ERROR,
                False,
                f"Duplicate column names found: {duplicate_columns}",
                {"duplicate_columns": duplicate_columns}
            )
        
        # Schema validation passed
        self._add_result(
            "schema_structure",
            ValidationCategory.INTEGRITY,
            ValidationLevel.INFO,
            True,
            f"Schema validation passed: {len(df.columns)} columns, {len(df)} rows"
        )
    
    def _validate_completeness(self, df: pd.DataFrame) -> None:
        """Validate data completeness and missing values"""
        logger.info("📊 Validating data completeness...")
        
        # Overall missing data ratio
        overall_missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        if overall_missing_ratio > self.config.max_missing_ratio:
            self._add_result(
                "overall_missing_data",
                ValidationCategory.COMPLETENESS,
                ValidationLevel.ERROR,
                False,
                f"Overall missing data ratio {overall_missing_ratio:.2%} exceeds threshold {self.config.max_missing_ratio:.2%}",
                {"missing_ratio": overall_missing_ratio}
            )
        
        # Per-column missing data analysis
        column_missing_ratios = df.isnull().mean()
        problematic_columns = column_missing_ratios[column_missing_ratios > self.config.max_missing_ratio]
        
        for column, ratio in problematic_columns.items():
            self._add_result(
                f"missing_data_{column}",
                ValidationCategory.COMPLETENESS,
                ValidationLevel.WARNING if ratio < 0.5 else ValidationLevel.ERROR,
                False,
                f"Column '{column}' has {ratio:.2%} missing values",
                {"column": column, "missing_ratio": ratio}
            )
        
        # Check for consecutive missing values
        for column in df.select_dtypes(include=[np.number]).columns:
            consecutive_missing = self._find_consecutive_missing(df[column])
            if consecutive_missing > self.config.max_missing_consecutive:
                self._add_result(
                    f"consecutive_missing_{column}",
                    ValidationCategory.COMPLETENESS,
                    ValidationLevel.WARNING,
                    False,
                    f"Column '{column}' has {consecutive_missing} consecutive missing values",
                    {"column": column, "consecutive_missing": consecutive_missing}
                )
        
        # Completeness validation summary
        if not problematic_columns.empty:
            self._add_result(
                "completeness_summary",
                ValidationCategory.COMPLETENESS,
                ValidationLevel.INFO,
                True,
                f"Completeness validation completed. {len(problematic_columns)} columns need attention"
            )
        else:
            self._add_result(
                "completeness_summary",
                ValidationCategory.COMPLETENESS,
                ValidationLevel.INFO,
                True,
                "All columns meet completeness requirements"
            )
    
    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate data types and format consistency"""
        logger.info("🔧 Validating data types...")
        
        # Expected data types for financial data
        expected_numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'adjusted_close']
        expected_date_columns = ['timestamp', 'date']
        expected_string_columns = ['symbol']
        
        # Check numeric columns
        for column in expected_numeric_columns:
            if column in df.columns:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    self._add_result(
                        f"data_type_{column}",
                        ValidationCategory.ACCURACY,
                        ValidationLevel.ERROR,
                        False,
                        f"Column '{column}' should be numeric but is {df[column].dtype}",
                        {"column": column, "actual_type": str(df[column].dtype)}
                    )
                else:
                    # Check for invalid numeric values
                    invalid_count = df[column].isna().sum()
                    if invalid_count > 0:
                        self._add_result(
                            f"invalid_numeric_{column}",
                            ValidationCategory.ACCURACY,
                            ValidationLevel.WARNING,
                            False,
                            f"Column '{column}' contains {invalid_count} invalid numeric values"
                        )
        
        # Check date columns
        for column in expected_date_columns:
            if column in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[column]):
                    try:
                        # Try to convert to datetime
                        pd.to_datetime(df[column])
                        self._add_result(
                            f"convertible_datetime_{column}",
                            ValidationCategory.ACCURACY,
                            ValidationLevel.WARNING,
                            True,
                            f"Column '{column}' is convertible to datetime"
                        )
                    except:
                        self._add_result(
                            f"data_type_{column}",
                            ValidationCategory.ACCURACY,
                            ValidationLevel.ERROR,
                            False,
                            f"Column '{column}' cannot be converted to datetime",
                            {"column": column, "actual_type": str(df[column].dtype)}
                        )
        
        self._add_result(
            "data_types_summary",
            ValidationCategory.ACCURACY,
            ValidationLevel.INFO,
            True,
            "Data type validation completed"
        )
    
    def _validate_time_series(self, df: pd.DataFrame, timestamp_column: str) -> None:
        """Validate time series properties"""
        logger.info("📅 Validating time series properties...")
        
        if timestamp_column not in df.columns:
            self._add_result(
                "missing_timestamp_column",
                ValidationCategory.INTEGRITY,
                ValidationLevel.ERROR,
                False,
                f"Timestamp column '{timestamp_column}' not found in DataFrame"
            )
            return
        
        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            try:
                timestamps = pd.to_datetime(df[timestamp_column])
            except:
                self._add_result(
                    "timestamp_conversion_failed",
                    ValidationCategory.INTEGRITY,
                    ValidationLevel.ERROR,
                    False,
                    f"Cannot convert '{timestamp_column}' to datetime"
                )
                return
        else:
            timestamps = df[timestamp_column]
        
        # Check for monotonic increasing (sorted)
        if not timestamps.is_monotonic_increasing:
            self._add_result(
                "time_series_order",
                ValidationCategory.CONSISTENCY,
                ValidationLevel.WARNING,
                False,
                "Time series is not in chronological order"
            )
        
        # Check for duplicate timestamps
        duplicate_timestamps = timestamps.duplicated().sum()
        if duplicate_timestamps > 0:
            self._add_result(
                "duplicate_timestamps",
                ValidationCategory.CONSISTENCY,
                ValidationLevel.WARNING,
                False,
                f"Found {duplicate_timestamps} duplicate timestamps",
                {"duplicate_count": duplicate_timestamps}
            )
        
        # Check time gaps
        if len(timestamps) > 1:
            time_diffs = timestamps.diff().dropna()
            max_gap = time_diffs.max()
            max_gap_days = max_gap.total_seconds() / (24 * 3600)
            
            if max_gap_days > self.config.max_time_gap_days:
                self._add_result(
                    "large_time_gaps",
                    ValidationCategory.CONSISTENCY,
                    ValidationLevel.WARNING,
                    False,
                    f"Maximum time gap of {max_gap_days:.1f} days exceeds threshold",
                    {"max_gap_days": max_gap_days}
                )
        
        self._add_result(
            "time_series_summary",
            ValidationCategory.CONSISTENCY,
            ValidationLevel.INFO,
            True,
            "Time series validation completed"
        )
    
    def _validate_business_rules(self, df: pd.DataFrame) -> None:
        """Validate financial market business rules"""
        logger.info("💼 Validating business rules...")
        
        # Price validation
        price_columns = ['open', 'high', 'low', 'close', 'adjusted_close']
        for column in price_columns:
            if column in df.columns:
                # Check minimum price
                below_min = (df[column] < self.config.min_price) & df[column].notna()
                if below_min.any():
                    self._add_result(
                        f"price_too_low_{column}",
                        ValidationCategory.BUSINESS_RULES,
                        ValidationLevel.ERROR,
                        False,
                        f"Column '{column}' has {below_min.sum()} values below minimum price {self.config.min_price}",
                        {"column": column, "violation_count": below_min.sum()}
                    )
                
                # Check maximum price
                above_max = (df[column] > self.config.max_price) & df[column].notna()
                if above_max.any():
                    self._add_result(
                        f"price_too_high_{column}",
                        ValidationCategory.BUSINESS_RULES,
                        ValidationLevel.WARNING,
                        False,
                        f"Column '{column}' has {above_max.sum()} values above maximum price {self.config.max_price}",
                        {"column": column, "violation_count": above_max.sum()}
                    )
        
        # Volume validation
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0) & df['volume'].notna()
            if negative_volume.any():
                self._add_result(
                    "negative_volume",
                    ValidationCategory.BUSINESS_RULES,
                    ValidationLevel.ERROR,
                    False,
                    f"Found {negative_volume.sum()} negative volume values",
                    {"violation_count": negative_volume.sum()}
                )
            
            excessive_volume = (df['volume'] > self.config.max_volume) & df['volume'].notna()
            if excessive_volume.any():
                self._add_result(
                    "excessive_volume",
                    ValidationCategory.BUSINESS_RULES,
                    ValidationLevel.WARNING,
                    False,
                    f"Found {excessive_volume.sum()} excessive volume values",
                    {"violation_count": excessive_volume.sum()}
                )
        
        # OHLC relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= Open, Low, Close
            invalid_high = ((df['high'] < df['open']) | 
                           (df['high'] < df['low']) | 
                           (df['high'] < df['close'])) & df[['high', 'open', 'low', 'close']].notna().all(axis=1)
            
            if invalid_high.any():
                self._add_result(
                    "invalid_ohlc_high",
                    ValidationCategory.BUSINESS_RULES,
                    ValidationLevel.ERROR,
                    False,
                    f"Found {invalid_high.sum()} invalid high prices (not highest of OHLC)",
                    {"violation_count": invalid_high.sum()}
                )
            
            # Low should be <= Open, High, Close  
            invalid_low = ((df['low'] > df['open']) | 
                          (df['low'] > df['high']) | 
                          (df['low'] > df['close'])) & df[['high', 'open', 'low', 'close']].notna().all(axis=1)
            
            if invalid_low.any():
                self._add_result(
                    "invalid_ohlc_low",
                    ValidationCategory.BUSINESS_RULES,
                    ValidationLevel.ERROR,
                    False,
                    f"Found {invalid_low.sum()} invalid low prices (not lowest of OHLC)",
                    {"violation_count": invalid_low.sum()}
                )
        
        self._add_result(
            "business_rules_summary",
            ValidationCategory.BUSINESS_RULES,
            ValidationLevel.INFO,
            True,
            "Business rules validation completed"
        )
    
    def _validate_statistical_properties(self, df: pd.DataFrame) -> None:
        """Validate statistical properties and detect anomalies"""
        logger.info("📈 Validating statistical properties...")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = df[column].dropna()
            
            if len(series) == 0:
                continue
            
            # Z-score anomaly detection
            if len(series) > 10:  # Need sufficient data
                z_scores = np.abs(stats.zscore(series))
                z_anomalies = (z_scores > self.config.z_score_threshold).sum()
                
                if z_anomalies > 0:
                    anomaly_ratio = z_anomalies / len(series)
                    level = ValidationLevel.WARNING if anomaly_ratio < 0.05 else ValidationLevel.ERROR
                    
                    self._add_result(
                        f"z_score_anomalies_{column}",
                        ValidationCategory.ACCURACY,
                        level,
                        anomaly_ratio < 0.1,
                        f"Column '{column}' has {z_anomalies} Z-score anomalies ({anomaly_ratio:.2%})",
                        {"column": column, "anomaly_count": z_anomalies, "anomaly_ratio": anomaly_ratio}
                    )
            
            # IQR-based anomaly detection
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config.iqr_multiplier * IQR
            upper_bound = Q3 + self.config.iqr_multiplier * IQR
            
            iqr_anomalies = ((series < lower_bound) | (series > upper_bound)).sum()
            
            if iqr_anomalies > 0:
                anomaly_ratio = iqr_anomalies / len(series)
                level = ValidationLevel.WARNING if anomaly_ratio < 0.05 else ValidationLevel.ERROR
                
                self._add_result(
                    f"iqr_anomalies_{column}",
                    ValidationCategory.ACCURACY,
                    level,
                    anomaly_ratio < 0.1,
                    f"Column '{column}' has {iqr_anomalies} IQR anomalies ({anomaly_ratio:.2%})",
                    {"column": column, "anomaly_count": iqr_anomalies, "anomaly_ratio": anomaly_ratio}
                )
        
        self._add_result(
            "statistical_properties_summary",
            ValidationCategory.ACCURACY,
            ValidationLevel.INFO,
            True,
            "Statistical properties validation completed"
        )
    
    def _validate_consistency(self, df: pd.DataFrame, symbol_column: str) -> None:
        """Validate data consistency"""
        logger.info("🔗 Validating data consistency...")
        
        if symbol_column in df.columns:
            # Check symbol consistency
            unique_symbols = df[symbol_column].unique()
            null_symbols = df[symbol_column].isnull().sum()
            
            if null_symbols > 0:
                self._add_result(
                    "null_symbols",
                    ValidationCategory.CONSISTENCY,
                    ValidationLevel.ERROR,
                    False,
                    f"Found {null_symbols} null symbols",
                    {"null_count": null_symbols}
                )
            
            # Check for symbol format consistency (assuming standard format)
            invalid_symbol_format = 0
            for symbol in unique_symbols:
                if pd.isna(symbol):
                    continue
                if not isinstance(symbol, str) or len(symbol) < 1 or len(symbol) > 10:
                    invalid_symbol_format += 1
            
            if invalid_symbol_format > 0:
                self._add_result(
                    "invalid_symbol_format",
                    ValidationCategory.CONSISTENCY,
                    ValidationLevel.WARNING,
                    False,
                    f"Found {invalid_symbol_format} symbols with invalid format",
                    {"invalid_format_count": invalid_symbol_format}
                )
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            self._add_result(
                "duplicate_rows",
                ValidationCategory.CONSISTENCY,
                ValidationLevel.WARNING,
                False,
                f"Found {duplicate_rows} duplicate rows",
                {"duplicate_count": duplicate_rows}
            )
        
        self._add_result(
            "consistency_summary",
            ValidationCategory.CONSISTENCY,
            ValidationLevel.INFO,
            True,
            "Consistency validation completed"
        )
    
    def _validate_freshness(self, df: pd.DataFrame, timestamp_column: str) -> None:
        """Validate data freshness"""
        logger.info("🕐 Validating data freshness...")
        
        if timestamp_column not in df.columns:
            return
        
        try:
            latest_timestamp = pd.to_datetime(df[timestamp_column]).max()
            current_time = datetime.now()
            data_age_days = (current_time - latest_timestamp).total_seconds() / (24 * 3600)
            
            if data_age_days > self.config.max_data_age_days:
                self._add_result(
                    "stale_data",
                    ValidationCategory.COMPLETENESS,
                    ValidationLevel.WARNING,
                    False,
                    f"Data is {data_age_days:.1f} days old, exceeds freshness threshold",
                    {"data_age_days": data_age_days, "latest_timestamp": latest_timestamp.isoformat()}
                )
            else:
                self._add_result(
                    "data_freshness",
                    ValidationCategory.COMPLETENESS,
                    ValidationLevel.INFO,
                    True,
                    f"Data is fresh ({data_age_days:.1f} days old)",
                    {"data_age_days": data_age_days}
                )
        except Exception as e:
            self._add_result(
                "freshness_check_failed",
                ValidationCategory.COMPLETENESS,
                ValidationLevel.WARNING,
                False,
                f"Could not check data freshness: {e}"
            )
    
    def _find_consecutive_missing(self, series: pd.Series) -> int:
        """Find maximum consecutive missing values in a series"""
        is_null = series.isnull()
        consecutive_groups = is_null.ne(is_null.shift()).cumsum()
        consecutive_counts = is_null.groupby(consecutive_groups).sum()
        return consecutive_counts.max() if not consecutive_counts.empty else 0
    
    def _add_result(self, check_name: str, category: ValidationCategory, 
                   level: ValidationLevel, passed: bool, message: str, 
                   details: Dict[str, Any] = None) -> None:
        """Add validation result"""
        result = ValidationResult(
            check_name=check_name,
            category=category,
            level=level,
            passed=passed,
            message=message,
            details=details or {}
        )
        
        self.results.append(result)
        self.stats['total_checks'] += 1
        
        if passed:
            self.stats['passed_checks'] += 1
        else:
            if level == ValidationLevel.WARNING:
                self.stats['warnings'] += 1
            elif level == ValidationLevel.ERROR:
                self.stats['errors'] += 1
            elif level == ValidationLevel.CRITICAL:
                self.stats['critical_errors'] += 1
    
    def _calculate_validation_score(self) -> None:
        """Calculate overall validation score (0-100)"""
        if self.stats['total_checks'] == 0:
            self.validation_score = 0.0
            return
        
        # Weight different validation levels
        weights = {
            'passed': 1.0,
            'warnings': -0.3,
            'errors': -0.7,
            'critical_errors': -1.0
        }
        
        weighted_score = (
            self.stats['passed_checks'] * weights['passed'] +
            self.stats['warnings'] * weights['warnings'] +
            self.stats['errors'] * weights['errors'] +
            self.stats['critical_errors'] * weights['critical_errors']
        )
        
        # Normalize to 0-100 scale
        max_possible_score = self.stats['total_checks']
        self.validation_score = max(0, (weighted_score / max_possible_score) * 100)
    
    def _generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0
        
        # Group results by category and level
        results_by_category = {}
        results_by_level = {}
        
        for result in self.results:
            category = result.category.value
            level = result.level.value
            
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(asdict(result))
            
            if level not in results_by_level:
                results_by_level[level] = []
            results_by_level[level].append(asdict(result))
        
        # Data summary
        data_summary = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'column_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        
        report = {
            'validation_metadata': {
                'timestamp': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': duration,
                'validator_version': '1.0.0'
            },
            'validation_score': {
                'overall_score': round(self.validation_score, 1),
                'grade': self._score_to_grade(self.validation_score),
                'statistics': self.stats
            },
            'data_summary': data_summary,
            'validation_results': {
                'by_category': results_by_category,
                'by_level': results_by_level,
                'all_results': [asdict(result) for result in self.results]
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _score_to_grade(self, score: float) -> str:
        """Convert validation score to letter grade"""
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 85:
            return 'B+'
        elif score >= 80:
            return 'B'
        elif score >= 75:
            return 'C+'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if self.stats['critical_errors'] > 0:
            recommendations.append("🚨 CRITICAL: Address critical errors before using this data")
        
        if self.stats['errors'] > 0:
            recommendations.append("❌ Fix data errors to improve data quality")
        
        if self.stats['warnings'] > 5:
            recommendations.append("⚠️ Consider investigating data warnings")
        
        # Specific recommendations based on failed checks
        failed_checks = [result for result in self.results if not result.passed]
        
        missing_data_issues = [r for r in failed_checks if 'missing' in r.check_name]
        if missing_data_issues:
            recommendations.append("📊 Implement data imputation for missing values")
        
        anomaly_issues = [r for r in failed_checks if 'anomalies' in r.check_name]
        if anomaly_issues:
            recommendations.append("📈 Review statistical anomalies for data quality issues")
        
        business_rule_violations = [r for r in failed_checks if r.category == ValidationCategory.BUSINESS_RULES]
        if business_rule_violations:
            recommendations.append("💼 Address business rule violations to ensure data integrity")
        
        if not recommendations:
            recommendations.append("✅ Data quality is excellent! No immediate actions needed")
        
        return recommendations

def validate_csv_file(file_path: str, config: ValidationConfig = None) -> Dict[str, Any]:
    """
    Validate a CSV file containing market data
    
    Args:
        file_path: Path to CSV file
        config: Validation configuration
        
    Returns:
        Validation report dictionary
    """
    try:
        df = pd.read_csv(file_path)
        
        # Try to parse timestamp column
        timestamp_columns = ['timestamp', 'date', 'datetime']
        for col in timestamp_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    break
                except:
                    continue
        
        validator = MarketDataValidator(config)
        return validator.validate_dataframe(df)
        
    except Exception as e:
        logger.error(f"Failed to validate CSV file {file_path}: {e}")
        return {
            'validation_score': {'overall_score': 0, 'grade': 'F'},
            'error': str(e)
        }

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate market data for StockPredictionPro')
    parser.add_argument('file', help='Path to CSV file to validate')
    parser.add_argument('--output', '-o', help='Output path for validation report (JSON)')
    parser.add_argument('--expected-columns', nargs='+', help='List of expected column names')
    parser.add_argument('--min-score', type=float, default=70.0, help='Minimum acceptable validation score')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    logger.info(f"🚀 Starting validation of {args.file}")
    
    config = ValidationConfig()
    report = validate_csv_file(args.file, config)
    
    # Print summary
    score = report.get('validation_score', {}).get('overall_score', 0)
    grade = report.get('validation_score', {}).get('grade', 'F')
    
    print("\n" + "="*60)
    print("DATA VALIDATION REPORT")
    print("="*60)
    print(f"File: {args.file}")
    print(f"Validation Score: {score}/100 (Grade: {grade})")
    
    if 'validation_results' in report:
        stats = report['validation_score']['statistics']
        print(f"Total Checks: {stats['total_checks']}")
        print(f"✅ Passed: {stats['passed_checks']}")
        print(f"⚠️ Warnings: {stats['warnings']}")
        print(f"❌ Errors: {stats['errors']}")
        print(f"🚨 Critical: {stats['critical_errors']}")
    
    # Print recommendations
    if 'recommendations' in report:
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Detailed report saved to: {args.output}")
    
    # Exit with appropriate code
    success = score >= args.min_score and report.get('validation_score', {}).get('statistics', {}).get('critical_errors', 1) == 0
    
    if success:
        print(f"\n✅ Validation PASSED (Score: {score} >= {args.min_score})")
        exit(0)
    else:
        print(f"\n❌ Validation FAILED (Score: {score} < {args.min_score})")
        exit(1)

if __name__ == '__main__':
    main()
