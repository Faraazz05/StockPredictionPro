"""
tests/utils/test_helpers.py

Test helper utilities for StockPredictionPro
Provides validation functions, mock data generators, and testing utilities
for API responses, data quality, and system components.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from unittest.mock import Mock, MagicMock, patch
import tempfile
import requests
from decimal import Decimal

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Test configuration
TEST_DATA_DIR = Path(__file__).parent.parent / 'fixtures'
MOCK_RESPONSES_DIR = TEST_DATA_DIR / 'mock_responses'
TEST_CONFIGS_DIR = TEST_DATA_DIR / 'test_configs'

# Ensure test directories exist
for dir_path in [TEST_DATA_DIR, MOCK_RESPONSES_DIR, TEST_CONFIGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# DATA VALIDATION HELPERS
# ============================================

class DataValidationHelper:
    """Helper class for validating stock market data"""
    
    @staticmethod
    def validate_stock_data_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame has required stock data columns
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'Symbol']
        errors = []
        
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Validate data types
        if 'date' in df.columns:
            try:
                pd.to_datetime(df['date'])
            except Exception:
                errors.append("Date column is not valid datetime format")
        
        # Validate numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' is not numeric")
                
                # Check for negative values where inappropriate
                if col in ['open', 'high', 'low', 'close'] and (df[col] < 0).any():
                    errors.append(f"Column '{col}' contains negative values")
                
                if col == 'volume' and (df[col] < 0).any():
                    errors.append(f"Volume column contains negative values")
        
        # Validate OHLC logic
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= Open, Low, Close
            invalid_high = (df['high'] < df[['open', 'low', 'close']].max(axis=1)).any()
            if invalid_high:
                errors.append("High price is less than open, low, or close in some records")
            
            # Low should be <= Open, High, Close  
            invalid_low = (df['low'] > df[['open', 'high', 'close']].min(axis=1)).any()
            if invalid_low:
                errors.append("Low price is greater than open, high, or close in some records")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_api_response(response_data: Dict[str, Any], response_type: str) -> Tuple[bool, List[str]]:
        """
        Validate API response structure based on response type
        
        Args:
            response_data: API response dictionary
            response_type: Type of response ('daily_prices', 'fundamentals', 'news_sentiment', etc.)
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(response_data, dict):
            errors.append("Response data is not a dictionary")
            return False, errors
        
        if response_type == 'daily_prices':
            # Alpha Vantage daily prices format
            if 'Time Series (Daily)' in response_data:
                time_series = response_data['Time Series (Daily)']
                if not isinstance(time_series, dict) or not time_series:
                    errors.append("Time Series data is empty or invalid")
                else:
                    # Validate first entry structure
                    first_entry = next(iter(time_series.values()))
                    required_keys = ['1. open', '2. high', '3. low', '4. close', '5. volume']
                    missing_keys = set(required_keys) - set(first_entry.keys())
                    if missing_keys:
                        errors.append(f"Missing keys in time series data: {missing_keys}")
            
        elif response_type == 'fundamentals':
            # Finnhub fundamentals format
            required_keys = ['name', 'country', 'currency', 'exchange']
            missing_keys = set(required_keys) - set(response_data.keys())
            if missing_keys:
                errors.append(f"Missing fundamental data keys: {missing_keys}")
                
        elif response_type == 'news_sentiment':
            # Finnhub news sentiment format
            if 'buzz' not in response_data and 'sentiment' not in response_data:
                errors.append("Missing buzz or sentiment data in news response")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_date_range(df: pd.DataFrame, expected_days: int = None, 
                          start_date: datetime = None, end_date: datetime = None) -> Tuple[bool, List[str]]:
        """
        Validate date range in DataFrame
        
        Args:
            df: DataFrame with date column
            expected_days: Expected number of trading days
            start_date: Expected start date
            end_date: Expected end date
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        if 'date' not in df.columns:
            errors.append("DataFrame missing date column")
            return False, errors
        
        try:
            dates = pd.to_datetime(df['date'])
        except Exception as e:
            errors.append(f"Invalid date format: {e}")
            return False, errors
        
        if dates.empty:
            errors.append("No dates found in DataFrame")
            return False, errors
        
        # Check chronological order
        if not dates.is_monotonic_increasing:
            errors.append("Dates are not in chronological order")
        
        # Check for duplicates
        if dates.duplicated().any():
            errors.append("Duplicate dates found")
        
        # Validate date range
        if start_date and dates.min() > start_date:
            errors.append(f"Data starts later than expected: {dates.min()} > {start_date}")
        
        if end_date and dates.max() < end_date:
            errors.append(f"Data ends earlier than expected: {dates.max()} < {end_date}")
        
        # Check expected number of days (accounting for weekends/holidays)
        if expected_days:
            actual_days = len(dates.unique())
            if abs(actual_days - expected_days) > expected_days * 0.1:  # 10% tolerance
                errors.append(f"Unexpected number of trading days: {actual_days}, expected ~{expected_days}")
        
        return len(errors) == 0, errors

# ============================================
# MOCK DATA GENERATORS
# ============================================

class MockDataGenerator:
    """Generate realistic mock data for testing"""
    
    @staticmethod
    def generate_stock_data(symbol: str = 'AAPL', days: int = 30, 
                          start_price: float = 100.0) -> pd.DataFrame:
        """
        Generate realistic mock stock data
        
        Args:
            symbol: Stock symbol
            days: Number of days to generate
            start_price: Starting price
        
        Returns:
            DataFrame with OHLCV data
        """
        np.random.seed(42)  # For reproducible tests
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        # Generate realistic price movement
        returns = np.random.normal(0.001, 0.02, len(dates))  # ~0.1% daily return, 2% volatility
        prices = [start_price]
        
        for i in range(1, len(dates)):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        data = []
        for i, date in enumerate(dates):
            if i == 0:
                continue
                
            close = prices[i]
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
            
            # Generate high/low with realistic constraints
            daily_volatility = abs(np.random.normal(0, 0.015))
            high = max(open_price, close) * (1 + daily_volatility)
            low = min(open_price, close) * (1 - daily_volatility)
            
            # Ensure OHLC logic
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = int(np.random.normal(1000000, 200000))  # Realistic volume
            volume = max(volume, 100000)  # Minimum volume
            
            data.append({
                'date': date.date(),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume,
                'Symbol': symbol,
                'Source': 'mock'
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_fundamentals_data(symbol: str = 'AAPL') -> Dict[str, Any]:
        """Generate mock fundamental data"""
        return {
            'symbol': symbol,
            'company_name': f'{symbol} Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'market_cap': np.random.randint(1000000000, 3000000000000),  # 1B to 3T
            'pe_ratio': round(np.random.uniform(15, 35), 2),
            'forward_pe': round(np.random.uniform(12, 30), 2),
            'price_to_book': round(np.random.uniform(1, 10), 2),
            'debt_to_equity': round(np.random.uniform(0.1, 2.0), 2),
            'roe': round(np.random.uniform(0.05, 0.35), 4),
            'profit_margin': round(np.random.uniform(0.05, 0.30), 4),
            'beta': round(np.random.uniform(0.8, 1.5), 2),
            'dividend_yield': round(np.random.uniform(0, 0.04), 4),
            'last_updated': datetime.now().isoformat()
        }
    
    @staticmethod
    def generate_news_sentiment_data(symbol: str = 'AAPL') -> Dict[str, Any]:
        """Generate mock news sentiment data"""
        return {
            'symbol': symbol,
            'buzz_articles_week': np.random.randint(10, 100),
            'buzz_score': round(np.random.uniform(0.5, 2.0), 2),
            'buzz_weekly_avg': round(np.random.uniform(0.8, 1.5), 2),
            'company_news_score': round(np.random.uniform(0.3, 0.8), 2),
            'sector_avg_news_score': round(np.random.uniform(0.4, 0.7), 2),
            'bearish_percent': round(np.random.uniform(0.2, 0.4), 2),
            'bullish_percent': round(np.random.uniform(0.6, 0.8), 2),
            'last_updated': datetime.now().isoformat()
        }
    
    @staticmethod
    def generate_api_response_mock(response_type: str, symbol: str = 'AAPL') -> Dict[str, Any]:
        """Generate mock API responses for different endpoints"""
        
        if response_type == 'alpha_vantage_daily':
            # Mock Alpha Vantage daily prices response
            time_series = {}
            for i in range(10):  # 10 days of data
                date_str = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                price = 150 + np.random.uniform(-5, 5)
                time_series[date_str] = {
                    '1. open': f"{price:.2f}",
                    '2. high': f"{price + np.random.uniform(0, 3):.2f}",
                    '3. low': f"{price - np.random.uniform(0, 3):.2f}",
                    '4. close': f"{price + np.random.uniform(-2, 2):.2f}",
                    '5. volume': str(np.random.randint(1000000, 5000000))
                }
            
            return {
                'Meta Data': {
                    '1. Information': 'Daily Prices (open, high, low, close) and Volumes',
                    '2. Symbol': symbol,
                    '3. Last Refreshed': datetime.now().strftime('%Y-%m-%d'),
                    '4. Output Size': 'Compact'
                },
                'Time Series (Daily)': time_series
            }
        
        elif response_type == 'finnhub_profile':
            # Mock Finnhub company profile
            return {
                'country': 'US',
                'currency': 'USD',
                'exchange': 'NASDAQ',
                'ipo': '1980-12-12',
                'marketCapitalization': np.random.randint(1000000, 3000000),
                'name': f'{symbol} Inc.',
                'shareOutstanding': np.random.randint(1000, 20000),
                'ticker': symbol,
                'weburl': f'https://www.{symbol.lower()}.com',
                'logo': f'https://logo.clearbit.com/{symbol.lower()}.com',
                'finnhubIndustry': 'Technology'
            }
        
        elif response_type == 'finnhub_sentiment':
            # Mock Finnhub news sentiment
            return {
                'buzz': {
                    'articlesInLastWeek': np.random.randint(20, 100),
                    'buzz': round(np.random.uniform(0.5, 2.0), 2),
                    'weeklyAverage': round(np.random.uniform(0.8, 1.5), 2)
                },
                'companyNewsScore': round(np.random.uniform(0.3, 0.8), 2),
                'sectorAverageNewsScore': round(np.random.uniform(0.4, 0.7), 2),
                'sentiment': {
                    'bearishPercent': round(np.random.uniform(0.2, 0.4), 2),
                    'bullishPercent': round(np.random.uniform(0.6, 0.8), 2)
                }
            }
        
        else:
            return {'error': f'Unknown response type: {response_type}'}

# ============================================
# API MOCKING UTILITIES
# ============================================

class APITestHelper:
    """Helper for testing API integrations"""
    
    @staticmethod
    def mock_successful_response(data: Dict[str, Any], status_code: int = 200) -> Mock:
        """Create mock successful API response"""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = data
        mock_response.text = json.dumps(data)
        mock_response.raise_for_status.return_value = None
        return mock_response
    
    @staticmethod
    def mock_failed_response(status_code: int = 500, error_message: str = "Internal Server Error") -> Mock:
        """Create mock failed API response"""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.side_effect = ValueError("No JSON object could be decoded")
        mock_response.text = error_message
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(error_message)
        return mock_response
    
    @staticmethod
    def mock_timeout_response() -> Mock:
        """Create mock timeout response"""
        mock_response = Mock()
        mock_response.side_effect = requests.exceptions.Timeout("Request timed out")
        return mock_response

# ============================================
# FILE SYSTEM TEST HELPERS
# ============================================

class FileSystemTestHelper:
    """Helper for file system operations in tests"""
    
    @staticmethod
    def create_temp_csv(data: pd.DataFrame, filename: str = None) -> str:
        """Create temporary CSV file for testing"""
        if filename is None:
            filename = f"test_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        
        data.to_csv(file_path, index=False)
        return file_path
    
    @staticmethod
    def create_temp_json(data: Dict[str, Any], filename: str = None) -> str:
        """Create temporary JSON file for testing"""
        if filename is None:
            filename = f"test_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return file_path
    
    @staticmethod
    def cleanup_temp_files(file_paths: List[str]) -> None:
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove temp file {file_path}: {e}")

# ============================================
# ASSERTION HELPERS
# ============================================

class AssertionHelper:
    """Custom assertions for stock market data testing"""
    
    @staticmethod
    def assert_dataframe_not_empty(df: pd.DataFrame, message: str = "") -> None:
        """Assert DataFrame is not empty"""
        assert not df.empty, f"DataFrame should not be empty. {message}"
    
    @staticmethod
    def assert_columns_exist(df: pd.DataFrame, columns: List[str], message: str = "") -> None:
        """Assert all required columns exist"""
        missing = set(columns) - set(df.columns)
        assert not missing, f"Missing columns: {missing}. {message}"
    
    @staticmethod
    def assert_no_null_values(df: pd.DataFrame, columns: List[str] = None, message: str = "") -> None:
        """Assert no null values in specified columns"""
        check_columns = columns if columns else df.columns
        null_columns = [col for col in check_columns if df[col].isnull().any()]
        assert not null_columns, f"Null values found in columns: {null_columns}. {message}"
    
    @staticmethod
    def assert_positive_values(df: pd.DataFrame, columns: List[str], message: str = "") -> None:
        """Assert all values in columns are positive"""
        negative_columns = []
        for col in columns:
            if col in df.columns and (df[col] <= 0).any():
                negative_columns.append(col)
        
        assert not negative_columns, f"Non-positive values found in columns: {negative_columns}. {message}"
    
    @staticmethod
    def assert_valid_ohlc(df: pd.DataFrame, message: str = "") -> None:
        """Assert OHLC data follows logical constraints"""
        required_cols = ['open', 'high', 'low', 'close']
        AssertionHelper.assert_columns_exist(df, required_cols)
        
        # High >= max(Open, Low, Close)
        invalid_high = df['high'] < df[['open', 'low', 'close']].max(axis=1)
        assert not invalid_high.any(), f"Invalid high prices found. {message}"
        
        # Low <= min(Open, High, Close)
        invalid_low = df['low'] > df[['open', 'high', 'close']].min(axis=1)
        assert not invalid_low.any(), f"Invalid low prices found. {message}"
    
    @staticmethod
    def assert_date_range(df: pd.DataFrame, start_date: datetime = None, 
                         end_date: datetime = None, message: str = "") -> None:
        """Assert DataFrame covers expected date range"""
        assert 'date' in df.columns, f"Date column missing. {message}"
        
        dates = pd.to_datetime(df['date'])
        
        if start_date:
            assert dates.min() >= start_date, f"Data starts too late: {dates.min()} < {start_date}. {message}"
        
        if end_date:
            assert dates.max() <= end_date, f"Data ends too late: {dates.max()} > {end_date}. {message}"

# ============================================
# PERFORMANCE TEST HELPERS
# ============================================

class PerformanceTestHelper:
    """Helper for performance testing"""
    
    @staticmethod
    def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
        """Time function execution and return result with elapsed time"""
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        return result, elapsed_time
    
    @staticmethod
    def assert_execution_time(func, max_time: float, *args, **kwargs) -> Any:
        """Assert function executes within time limit"""
        result, elapsed_time = PerformanceTestHelper.time_function(func, *args, **kwargs)
        assert elapsed_time <= max_time, f"Function took {elapsed_time:.3f}s, expected <= {max_time}s"
        return result
    
    @staticmethod
    def measure_memory_usage(func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure memory usage of function execution"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        return result, memory_used

# ============================================
# ENVIRONMENT TEST HELPERS
# ============================================

class EnvironmentTestHelper:
    """Helper for testing environment setup"""
    
    @staticmethod
    def check_api_keys() -> Dict[str, bool]:
        """Check if API keys are available"""
        api_keys = {
            'ALPHA_VANTAGE_API_KEY': bool(os.getenv('ALPHA_VANTAGE_API_KEY')),
            'FINNHUB_API_KEY': bool(os.getenv('FINNHUB_API_KEY')),
            'POLYGON_API_KEY': bool(os.getenv('POLYGON_API_KEY')),
            'TWELVEDATA_API_KEY': bool(os.getenv('TWELVEDATA_API_KEY'))
        }
        return api_keys
    
    @staticmethod
    def skip_if_no_api_key(api_key_name: str):
        """Pytest decorator to skip test if API key is missing"""
        return pytest.mark.skipif(
            not os.getenv(api_key_name),
            reason=f"{api_key_name} not available in environment"
        )
    
    @staticmethod
    def require_network_connection():
        """Pytest decorator to skip test if no network connection"""
        try:
            requests.get('https://httpbin.org/status/200', timeout=5)
            return pytest.mark.skipif(False, reason="")
        except:
            return pytest.mark.skipif(True, reason="No network connection available")

# ============================================
# MAIN TEST FUNCTIONS
# ============================================

def test_data_validation_helper():
    """Test the DataValidationHelper class"""
    # Test with valid data
    valid_data = MockDataGenerator.generate_stock_data('AAPL', 10)
    is_valid, errors = DataValidationHelper.validate_stock_data_schema(valid_data)
    assert is_valid, f"Valid data failed validation: {errors}"
    
    # Test with invalid data
    invalid_data = valid_data.drop('close', axis=1)
    is_valid, errors = DataValidationHelper.validate_stock_data_schema(invalid_data)
    assert not is_valid, "Invalid data passed validation"
    assert 'close' in str(errors), "Missing column error not detected"

def test_mock_data_generator():
    """Test the MockDataGenerator class"""
    # Test stock data generation
    stock_data = MockDataGenerator.generate_stock_data('TSLA', 20, 200.0)
    
    assert len(stock_data) == 20, "Incorrect number of records generated"
    assert stock_data['Symbol'].iloc[0] == 'TSLA', "Incorrect symbol"
    assert all(stock_data['high'] >= stock_data[['open', 'low', 'close']].max(axis=1)), "OHLC logic violated"
    
    # Test fundamentals generation
    fundamentals = MockDataGenerator.generate_fundamentals_data('NVDA')
    assert fundamentals['symbol'] == 'NVDA', "Incorrect symbol in fundamentals"
    assert 'market_cap' in fundamentals, "Missing market cap in fundamentals"

def test_api_test_helper():
    """Test the APITestHelper class"""
    # Test successful response mock
    test_data = {'symbol': 'AAPL', 'price': 150.0}
    mock_response = APITestHelper.mock_successful_response(test_data)
    
    assert mock_response.status_code == 200
    assert mock_response.json() == test_data
    
    # Test failed response mock
    failed_response = APITestHelper.mock_failed_response(404, "Not Found")
    assert failed_response.status_code == 404

def test_assertion_helper():
    """Test the AssertionHelper class"""
    valid_data = MockDataGenerator.generate_stock_data('GOOG', 5)
    
    # Test assertions that should pass
    AssertionHelper.assert_dataframe_not_empty(valid_data)
    AssertionHelper.assert_columns_exist(valid_data, ['date', 'open', 'high', 'low', 'close'])
    AssertionHelper.assert_no_null_values(valid_data)
    AssertionHelper.assert_positive_values(valid_data, ['open', 'high', 'low', 'close', 'volume'])
    AssertionHelper.assert_valid_ohlc(valid_data)

def test_file_system_helper():
    """Test the FileSystemTestHelper class"""
    test_data = MockDataGenerator.generate_stock_data('MSFT', 5)
    temp_files = []
    
    try:
        # Test CSV creation
        csv_path = FileSystemTestHelper.create_temp_csv(test_data)
        temp_files.append(csv_path)
        assert os.path.exists(csv_path), "CSV file was not created"
        
        # Test JSON creation
        json_data = {'test': 'data', 'number': 123}
        json_path = FileSystemTestHelper.create_temp_json(json_data)
        temp_files.append(json_path)
        assert os.path.exists(json_path), "JSON file was not created"
        
    finally:
        # Clean up
        FileSystemTestHelper.cleanup_temp_files(temp_files)

def test_performance_helper():
    """Test the PerformanceTestHelper class"""
    def slow_function():
        import time
        time.sleep(0.1)
        return "done"
    
    def fast_function():
        return sum(range(1000))
    
    # Test timing
    result, elapsed = PerformanceTestHelper.time_function(fast_function)
    assert result == 499500, "Function result incorrect"
    assert elapsed >= 0, "Elapsed time should be non-negative"
    
    # Test assertion (this should pass)
    result = PerformanceTestHelper.assert_execution_time(fast_function, 1.0)
    assert result == 499500, "Performance assertion result incorrect"

if __name__ == "__main__":
    """Run basic tests when module is executed directly"""
    print("Running basic test helper validation...")
    
    try:
        test_data_validation_helper()
        print("✅ DataValidationHelper tests passed")
        
        test_mock_data_generator()
        print("✅ MockDataGenerator tests passed")
        
        test_api_test_helper()
        print("✅ APITestHelper tests passed")
        
        test_assertion_helper()
        print("✅ AssertionHelper tests passed")
        
        test_file_system_helper()
        print("✅ FileSystemTestHelper tests passed")
        
        test_performance_helper()
        print("✅ PerformanceTestHelper tests passed")
        
        print("\n🎉 All test helpers are working correctly!")
        print("✅ Ready to support comprehensive testing framework")
        
    except Exception as e:
        print(f"❌ Test helper validation failed: {e}")
        import traceback
        traceback.print_exc()
