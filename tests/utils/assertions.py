"""
tests/utils/assertions.py

Custom assertion utilities for StockPredictionPro testing framework.
Provides specialized assertions for financial data validation, API responses,
and machine learning model validation.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import sys
import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# ============================================
# CORE ASSERTION UTILITIES
# ============================================

class StockDataAssertions:
    """Specialized assertions for stock market data validation"""
    
    @staticmethod
    def assert_valid_ohlcv_schema(df: pd.DataFrame, message: str = "") -> None:
        """Assert DataFrame has valid OHLCV schema"""
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(df.columns)
        
        assert not missing_columns, (
            f"Missing OHLCV columns: {missing_columns}. {message}"
        )
        
        assert not df.empty, f"OHLCV DataFrame cannot be empty. {message}"
    
    @staticmethod
    def assert_valid_price_relationships(df: pd.DataFrame, message: str = "") -> None:
        """
        Assert OHLC price relationships are logically valid:
        - High >= max(Open, Low, Close) 
        - Low <= min(Open, High, Close)
        """
        required_cols = ['open', 'high', 'low', 'close']
        StockDataAssertions.assert_columns_present(df, required_cols)
        
        # Check High >= max(Open, Low, Close)
        max_prices = df[['open', 'low', 'close']].max(axis=1)
        invalid_high_mask = df['high'] < max_prices
        invalid_high_count = invalid_high_mask.sum()
        
        assert invalid_high_count == 0, (
            f"Found {invalid_high_count} records where High < max(Open, Low, Close). {message}"
        )
        
        # Check Low <= min(Open, High, Close)
        min_prices = df[['open', 'high', 'close']].min(axis=1)
        invalid_low_mask = df['low'] > min_prices
        invalid_low_count = invalid_low_mask.sum()
        
        assert invalid_low_count == 0, (
            f"Found {invalid_low_count} records where Low > min(Open, High, Close). {message}"
        )
    
    @staticmethod
    def assert_positive_prices(df: pd.DataFrame, columns: List[str] = None, message: str = "") -> None:
        """Assert all price columns contain positive values"""
        if columns is None:
            columns = ['open', 'high', 'low', 'close']
        
        for col in columns:
            if col in df.columns:
                non_positive_count = (df[col] <= 0).sum()
                assert non_positive_count == 0, (
                    f"Column '{col}' has {non_positive_count} non-positive values. {message}"
                )
    
    @staticmethod
    def assert_valid_volume(df: pd.DataFrame, min_volume: int = 0, message: str = "") -> None:
        """Assert volume data is valid (non-negative integers)"""
        assert 'volume' in df.columns, f"Volume column missing. {message}"
        
        # Check for negative volumes
        negative_volume_count = (df['volume'] < min_volume).sum()
        assert negative_volume_count == 0, (
            f"Found {negative_volume_count} records with volume < {min_volume}. {message}"
        )
        
        # Check for NaN volumes
        nan_volume_count = df['volume'].isna().sum()
        assert nan_volume_count == 0, (
            f"Found {nan_volume_count} records with NaN volume values. {message}"
        )
    
    @staticmethod
    def assert_chronological_order(df: pd.DataFrame, date_column: str = 'date', message: str = "") -> None:
        """Assert dates are in chronological order"""
        assert date_column in df.columns, f"Date column '{date_column}' missing. {message}"
        
        # Convert to datetime if not already
        dates = pd.to_datetime(df[date_column])
        
        is_monotonic = dates.is_monotonic_increasing
        assert is_monotonic, f"Dates are not in chronological order. {message}"
    
    @staticmethod
    def assert_no_duplicate_dates(df: pd.DataFrame, date_column: str = 'date', 
                                 symbol_column: str = 'Symbol', message: str = "") -> None:
        """Assert no duplicate dates for the same symbol"""
        if symbol_column in df.columns:
            # Check duplicates within each symbol
            duplicates = df.duplicated(subset=[date_column, symbol_column])
        else:
            # Check duplicates in date column only
            duplicates = df.duplicated(subset=[date_column])
        
        duplicate_count = duplicates.sum()
        assert duplicate_count == 0, (
            f"Found {duplicate_count} duplicate date records. {message}"
        )

class DataFrameAssertions:
    """General DataFrame validation assertions"""
    
    @staticmethod
    def assert_not_empty(df: pd.DataFrame, message: str = "") -> None:
        """Assert DataFrame is not empty"""
        assert not df.empty, f"DataFrame should not be empty. {message}"
        assert len(df) > 0, f"DataFrame should have at least one row. {message}"
    
    @staticmethod
    def assert_columns_present(df: pd.DataFrame, columns: List[str], message: str = "") -> None:
        """Assert all required columns are present"""
        missing_columns = set(columns) - set(df.columns)
        assert not missing_columns, (
            f"Missing required columns: {missing_columns}. Available: {list(df.columns)}. {message}"
        )
    
    @staticmethod
    def assert_no_null_values(df: pd.DataFrame, columns: List[str] = None, message: str = "") -> None:
        """Assert no null values in specified columns"""
        check_columns = columns if columns is not None else df.columns.tolist()
        
        null_columns = []
        for col in check_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    null_columns.append(f"{col}({null_count})")
        
        assert not null_columns, (
            f"Null values found in columns: {null_columns}. {message}"
        )
    
    @staticmethod
    def assert_data_types(df: pd.DataFrame, expected_types: Dict[str, str], message: str = "") -> None:
        """Assert columns have expected data types"""
        type_errors = []
        
        for column, expected_type in expected_types.items():
            if column not in df.columns:
                type_errors.append(f"Column '{column}' missing")
                continue
            
            actual_type = str(df[column].dtype)
            
            # Handle common type mappings
            if expected_type in ['int', 'integer'] and not pd.api.types.is_integer_dtype(df[column]):
                type_errors.append(f"'{column}': expected integer, got {actual_type}")
            elif expected_type in ['float', 'numeric'] and not pd.api.types.is_numeric_dtype(df[column]):
                type_errors.append(f"'{column}': expected numeric, got {actual_type}")
            elif expected_type in ['object', 'string'] and not pd.api.types.is_object_dtype(df[column]):
                type_errors.append(f"'{column}': expected object/string, got {actual_type}")
            elif expected_type in ['datetime'] and not pd.api.types.is_datetime64_any_dtype(df[column]):
                type_errors.append(f"'{column}': expected datetime, got {actual_type}")
        
        assert not type_errors, f"Data type mismatches: {type_errors}. {message}"
    
    @staticmethod
    def assert_value_ranges(df: pd.DataFrame, value_ranges: Dict[str, Tuple[float, float]], 
                           message: str = "") -> None:
        """Assert column values are within expected ranges"""
        range_errors = []
        
        for column, (min_val, max_val) in value_ranges.items():
            if column not in df.columns:
                range_errors.append(f"Column '{column}' missing")
                continue
            
            below_min = (df[column] < min_val).sum()
            above_max = (df[column] > max_val).sum()
            
            if below_min > 0:
                range_errors.append(f"'{column}': {below_min} values below {min_val}")
            if above_max > 0:
                range_errors.append(f"'{column}': {above_max} values above {max_val}")
        
        assert not range_errors, f"Value range violations: {range_errors}. {message}"

class APIResponseAssertions:
    """Assertions for API response validation"""
    
    @staticmethod
    def assert_valid_json_response(response_data: Any, message: str = "") -> None:
        """Assert response is valid JSON dictionary"""
        assert isinstance(response_data, dict), (
            f"API response should be a dictionary, got {type(response_data)}. {message}"
        )
        assert response_data, f"API response should not be empty. {message}"
    
    @staticmethod
    def assert_required_keys(response_data: Dict[str, Any], required_keys: List[str], 
                            message: str = "") -> None:
        """Assert required keys are present in API response"""
        missing_keys = set(required_keys) - set(response_data.keys())
        assert not missing_keys, (
            f"Missing required keys in API response: {missing_keys}. "
            f"Available keys: {list(response_data.keys())}. {message}"
        )
    
    @staticmethod
    def assert_alpha_vantage_daily_response(response_data: Dict[str, Any], message: str = "") -> None:
        """Assert Alpha Vantage daily price response is valid"""
        APIResponseAssertions.assert_valid_json_response(response_data, message)
        
        # Check for error message
        if 'Error Message' in response_data:
            assert False, f"Alpha Vantage API error: {response_data['Error Message']}. {message}"
        
        # Check for rate limit message
        if 'Note' in response_data:
            assert False, f"Alpha Vantage rate limited: {response_data['Note']}. {message}"
        
        # Check required structure
        required_keys = ['Meta Data', 'Time Series (Daily)']
        APIResponseAssertions.assert_required_keys(response_data, required_keys, message)
        
        # Validate time series data
        time_series = response_data['Time Series (Daily)']
        assert isinstance(time_series, dict), f"Time series should be dictionary. {message}"
        assert time_series, f"Time series should not be empty. {message}"
        
        # Check first entry structure
        first_entry = next(iter(time_series.values()))
        required_price_keys = ['1. open', '2. high', '3. low', '4. close', '5. volume']
        APIResponseAssertions.assert_required_keys(first_entry, required_price_keys, message)
    
    @staticmethod
    def assert_finnhub_profile_response(response_data: Dict[str, Any], message: str = "") -> None:
        """Assert Finnhub company profile response is valid"""
        APIResponseAssertions.assert_valid_json_response(response_data, message)
        
        # Check for empty response (often indicates invalid symbol)
        if not response_data or response_data == {}:
            assert False, f"Empty Finnhub profile response - possibly invalid symbol. {message}"
        
        # Check essential keys
        essential_keys = ['name', 'ticker']
        APIResponseAssertions.assert_required_keys(response_data, essential_keys, message)
    
    @staticmethod
    def assert_finnhub_sentiment_response(response_data: Dict[str, Any], message: str = "") -> None:
        """Assert Finnhub news sentiment response is valid"""
        APIResponseAssertions.assert_valid_json_response(response_data, message)
        
        # Should have buzz or sentiment data
        has_buzz = 'buzz' in response_data and response_data['buzz']
        has_sentiment = 'sentiment' in response_data and response_data['sentiment']
        
        assert has_buzz or has_sentiment, (
            f"Finnhub sentiment response missing both 'buzz' and 'sentiment' data. {message}"
        )

class ModelAssertions:
    """Assertions for machine learning model validation"""
    
    @staticmethod
    def assert_valid_predictions(predictions: np.ndarray, expected_length: int = None, 
                               message: str = "") -> None:
        """Assert model predictions are valid"""
        assert isinstance(predictions, (np.ndarray, list)), (
            f"Predictions should be numpy array or list, got {type(predictions)}. {message}"
        )
        
        predictions = np.array(predictions)
        
        assert predictions.size > 0, f"Predictions should not be empty. {message}"
        
        # Check for NaN or infinite values
        nan_count = np.isnan(predictions).sum()
        inf_count = np.isinf(predictions).sum()
        
        assert nan_count == 0, f"Found {nan_count} NaN values in predictions. {message}"
        assert inf_count == 0, f"Found {inf_count} infinite values in predictions. {message}"
        
        if expected_length is not None:
            assert len(predictions) == expected_length, (
                f"Expected {expected_length} predictions, got {len(predictions)}. {message}"
            )
    
    @staticmethod
    def assert_probability_predictions(probabilities: np.ndarray, message: str = "") -> None:
        """Assert probability predictions are valid (between 0 and 1, sum to 1 for multiclass)"""
        ModelAssertions.assert_valid_predictions(probabilities, message=message)
        
        probabilities = np.array(probabilities)
        
        # Check range [0, 1]
        below_zero = (probabilities < 0).sum()
        above_one = (probabilities > 1).sum()
        
        assert below_zero == 0, f"Found {below_zero} probability values < 0. {message}"
        assert above_one == 0, f"Found {above_one} probability values > 1. {message}"
        
        # For 2D arrays (multiclass), check if rows sum to ~1
        if probabilities.ndim == 2:
            row_sums = probabilities.sum(axis=1)
            sum_tolerance = 0.01  # Allow small floating point errors
            
            invalid_sums = np.abs(row_sums - 1.0) > sum_tolerance
            invalid_count = invalid_sums.sum()
            
            assert invalid_count == 0, (
                f"Found {invalid_count} probability rows that don't sum to 1. {message}"
            )
    
    @staticmethod
    def assert_classification_metrics(metrics: Dict[str, float], message: str = "") -> None:
        """Assert classification metrics are valid"""
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in expected_metrics:
            if metric in metrics:
                value = metrics[metric]
                assert 0 <= value <= 1, (
                    f"Metric '{metric}' should be between 0 and 1, got {value}. {message}"
                )
    
    @staticmethod
    def assert_regression_metrics(metrics: Dict[str, float], message: str = "") -> None:
        """Assert regression metrics are valid"""
        # MSE, RMSE, MAE should be non-negative
        non_negative_metrics = ['mse', 'rmse', 'mae']
        for metric in non_negative_metrics:
            if metric in metrics:
                value = metrics[metric]
                assert value >= 0, (
                    f"Metric '{metric}' should be non-negative, got {value}. {message}"
                )
        
        # R² should be between -inf and 1 (but typically between 0 and 1 for good models)
        if 'r2_score' in metrics:
            r2 = metrics['r2_score']
            assert r2 <= 1, f"R² score should be ≤ 1, got {r2}. {message}"

class PerformanceAssertions:
    """Assertions for performance and timing validation"""
    
    @staticmethod
    def assert_execution_time(execution_time: float, max_time: float, 
                            operation_name: str = "Operation", message: str = "") -> None:
        """Assert operation completed within time limit"""
        assert execution_time >= 0, f"Execution time cannot be negative: {execution_time}. {message}"
        assert execution_time <= max_time, (
            f"{operation_name} took {execution_time:.3f}s, "
            f"expected ≤ {max_time:.3f}s. {message}"
        )
    
    @staticmethod
    def assert_memory_usage(memory_mb: float, max_memory_mb: float, 
                          operation_name: str = "Operation", message: str = "") -> None:
        """Assert operation used acceptable amount of memory"""
        assert memory_mb >= 0, f"Memory usage cannot be negative: {memory_mb}. {message}"
        assert memory_mb <= max_memory_mb, (
            f"{operation_name} used {memory_mb:.1f}MB memory, "
            f"expected ≤ {max_memory_mb:.1f}MB. {message}"
        )
    
    @staticmethod
    def assert_throughput(items_per_second: float, min_throughput: float, 
                         operation_name: str = "Operation", message: str = "") -> None:
        """Assert operation achieved minimum throughput"""
        assert items_per_second >= 0, (
            f"Throughput cannot be negative: {items_per_second}. {message}"
        )
        assert items_per_second >= min_throughput, (
            f"{operation_name} achieved {items_per_second:.1f} items/sec, "
            f"expected ≥ {min_throughput:.1f} items/sec. {message}"
        )

# ============================================
# CONVENIENCE ASSERTION FUNCTIONS
# ============================================

def assert_valid_stock_dataframe(df: pd.DataFrame, message: str = "") -> None:
    """Comprehensive validation for stock market DataFrame"""
    StockDataAssertions.assert_valid_ohlcv_schema(df, message)
    StockDataAssertions.assert_valid_price_relationships(df, message) 
    StockDataAssertions.assert_positive_prices(df, message=message)
    StockDataAssertions.assert_valid_volume(df, message=message)
    StockDataAssertions.assert_chronological_order(df, message=message)
    StockDataAssertions.assert_no_duplicate_dates(df, message=message)

def assert_api_response_valid(response_data: Dict[str, Any], api_type: str, message: str = "") -> None:
    """Validate API response based on API type"""
    if api_type == 'alpha_vantage_daily':
        APIResponseAssertions.assert_alpha_vantage_daily_response(response_data, message)
    elif api_type == 'finnhub_profile':
        APIResponseAssertions.assert_finnhub_profile_response(response_data, message)
    elif api_type == 'finnhub_sentiment':
        APIResponseAssertions.assert_finnhub_sentiment_response(response_data, message)
    else:
        APIResponseAssertions.assert_valid_json_response(response_data, message)

def assert_model_output_valid(predictions: np.ndarray, model_type: str = 'regression', 
                             expected_length: int = None, message: str = "") -> None:
    """Validate model output based on model type"""
    ModelAssertions.assert_valid_predictions(predictions, expected_length, message)
    
    if model_type == 'classification_proba':
        ModelAssertions.assert_probability_predictions(predictions, message)

# ============================================
# TEST FUNCTIONS FOR ASSERTIONS
# ============================================

def test_stock_data_assertions():
    """Test StockDataAssertions with valid and invalid data"""
    # Create valid test data
    valid_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5),
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [98, 99, 100, 101, 102],
        'close': [104, 105, 106, 107, 108],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'Symbol': ['AAPL'] * 5
    })
    
    # These should not raise
    assert_valid_stock_dataframe(valid_data)
    
    # Test invalid OHLC relationships
    invalid_data = valid_data.copy()
    invalid_data.loc[0, 'high'] = 95  # High < Open, Low, Close
    
    with pytest.raises(AssertionError, match="High.*less than"):
        StockDataAssertions.assert_valid_price_relationships(invalid_data)

def test_api_response_assertions():
    """Test API response assertions"""
    # Valid Alpha Vantage response
    valid_av_response = {
        'Meta Data': {
            '1. Information': 'Daily Prices',
            '2. Symbol': 'AAPL'
        },
        'Time Series (Daily)': {
            '2023-01-01': {
                '1. open': '150.00',
                '2. high': '155.00',
                '3. low': '149.00',
                '4. close': '154.00',
                '5. volume': '1000000'
            }
        }
    }
    
    # Should not raise
    assert_api_response_valid(valid_av_response, 'alpha_vantage_daily')
    
    # Test error response
    error_response = {'Error Message': 'Invalid API call'}
    
    with pytest.raises(AssertionError, match="Alpha Vantage API error"):
        APIResponseAssertions.assert_alpha_vantage_daily_response(error_response)

def test_model_assertions():
    """Test model validation assertions"""
    # Valid predictions
    valid_predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert_model_output_valid(valid_predictions, 'regression')
    
    # Valid probabilities
    valid_probabilities = np.array([[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]])
    assert_model_output_valid(valid_probabilities, 'classification_proba')
    
    # Invalid probabilities (don't sum to 1)
    invalid_probabilities = np.array([[0.7, 0.4], [0.5, 0.3]])
    
    with pytest.raises(AssertionError, match="don't sum to 1"):
        ModelAssertions.assert_probability_predictions(invalid_probabilities)

if __name__ == "__main__":
    """Run tests when module is executed directly"""
    print("Running assertion validation tests...")
    
    try:
        test_stock_data_assertions()
        print("✅ Stock data assertions tests passed")
        
        test_api_response_assertions()
        print("✅ API response assertions tests passed")
        
        test_model_assertions()
        print("✅ Model assertions tests passed")
        
        print("\n🎉 All assertion utilities validated successfully!")
        print("✅ Ready to use in comprehensive test suite")
        
    except Exception as e:
        print(f"❌ Assertion validation failed: {e}")
        import traceback
        traceback.print_exc()
