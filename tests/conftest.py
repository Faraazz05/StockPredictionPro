"""
conftest.py

Pytest configuration and fixtures for StockPredictionPro tests.
Provides shared test data, mocks, API keys, and helper fixtures for clean,
reproducible testing across all test modules.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import test utilities
from tests.utils.mock_factories import (
    StockDataFactory, FundamentalsFactory, APIResponseFactory,
    MLModelFactory, SystemFactory, MockObjectFactory
)
from tests.utils.test_helpers import (
    DataValidationHelper, MockDataGenerator, APITestHelper,
    FileSystemTestHelper, AssertionHelper
)

# ============================================
# PYTEST CONFIGURATION
# ============================================

def pytest_configure(config):
    """Configure pytest settings and markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "api: marks tests that require API keys")
    config.addinivalue_line("markers", "network: marks tests that require network connectivity")
    config.addinivalue_line("markers", "performance: marks tests that measure performance")

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location"""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Mark tests that need API keys
        if any(marker in item.name.lower() for marker in ['api', 'fetch', 'extract']):
            item.add_marker(pytest.mark.api)
        
        # Mark tests that need network
        if any(marker in item.name.lower() for marker in ['network', 'requests', 'http']):
            item.add_marker(pytest.mark.network)

# ============================================
# SESSION-SCOPED FIXTURES
# ============================================

@pytest.fixture(scope="session")
def api_keys():
    """Provide API keys from environment or mock values for testing"""
    return {
        'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY', 'mock_alpha_vantage_key'),
        'FINNHUB_API_KEY': os.getenv('FINNHUB_API_KEY', 'mock_finnhub_key'),
        'POLYGON_API_KEY': os.getenv('POLYGON_API_KEY', 'mock_polygon_key'),
        'TWELVEDATA_API_KEY': os.getenv('TWELVEDATA_API_KEY', 'mock_twelvedata_key'),
        'QUANDL_API_KEY': os.getenv('QUANDL_API_KEY', 'mock_quandl_key')
    }

@pytest.fixture(scope="session")
def test_symbols():
    """Provide consistent test stock symbols"""
    return ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

@pytest.fixture(scope="session")
def temp_data_dir():
    """Create temporary directory for test data files"""
    temp_dir = tempfile.mkdtemp(prefix="stockpro_test_")
    yield Path(temp_dir)
    
    # Cleanup after all tests
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not cleanup temp directory {temp_dir}: {e}")

# ============================================
# MOCK DATA FIXTURES
# ============================================

@pytest.fixture
def sample_stock_data():
    """Generate sample OHLCV stock data"""
    return StockDataFactory.create_ohlcv_data(
        symbol='TEST', 
        days=30, 
        start_price=150.0,
        seed=42
    )

@pytest.fixture
def sample_multi_stock_data():
    """Generate multi-stock correlated data"""
    symbols = ['TEST1', 'TEST2', 'TEST3']
    return StockDataFactory.create_multi_stock_data(
        symbols=symbols,
        days=20,
        correlations={'TEST2': 0.7, 'TEST3': 0.5},
        seed=42
    )

@pytest.fixture
def sample_fundamentals():
    """Generate sample company fundamental data"""
    return FundamentalsFactory.create_company_fundamentals(
        symbol='TEST',
        company_type='tech'
    )

@pytest.fixture
def sample_alpha_vantage_response():
    """Generate sample Alpha Vantage API response"""
    return APIResponseFactory.create_alpha_vantage_response(
        symbol='TEST',
        days=10,
        response_type='success'
    )

@pytest.fixture
def sample_finnhub_response():
    """Generate sample Finnhub API response"""
    return APIResponseFactory.create_finnhub_profile_response(
        symbol='TEST',
        response_type='success',
        company_type='tech'
    )

@pytest.fixture
def sample_predictions():
    """Generate sample ML model predictions"""
    return MLModelFactory.create_predictions(
        n_samples=100,
        model_type='regression',
        seed=42
    )

@pytest.fixture
def sample_model_metrics():
    """Generate sample ML model performance metrics"""
    return MLModelFactory.create_model_metrics(
        model_type='regression',
        performance_level='good'
    )

# ============================================
# TEMPORARY FILE FIXTURES
# ============================================

@pytest.fixture
def temp_csv_file():
    """Create temporary CSV file with test data"""
    def _create_csv(data: pd.DataFrame = None, filename: str = None):
        if data is None:
            data = StockDataFactory.create_ohlcv_data('TEMP', 10)
        
        if filename is None:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.csv', delete=False
            )
            filename = temp_file.name
            temp_file.close()
        
        data.to_csv(filename, index=False)
        return filename
    
    created_files = []
    
    def wrapper(*args, **kwargs):
        filename = _create_csv(*args, **kwargs)
        created_files.append(filename)
        return filename
    
    yield wrapper
    
    # Cleanup
    FileSystemTestHelper.cleanup_temp_files(created_files)

@pytest.fixture
def temp_json_file():
    """Create temporary JSON file with test data"""
    def _create_json(data: Dict[str, Any] = None, filename: str = None):
        if data is None:
            data = {'test': True, 'timestamp': datetime.now().isoformat()}
        
        if filename is None:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False
            )
            filename = temp_file.name
            temp_file.close()
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filename
    
    created_files = []
    
    def wrapper(*args, **kwargs):
        filename = _create_json(*args, **kwargs)
        created_files.append(filename)
        return filename
    
    yield wrapper
    
    # Cleanup
    FileSystemTestHelper.cleanup_temp_files(created_files)

# ============================================
# API MOCKING FIXTURES
# ============================================

@pytest.fixture
def mock_requests_get():
    """Mock requests.get with configurable responses"""
    def _mock_get(url: str, *args, **kwargs):
        """Default mock implementation"""
        if 'alpha' in url.lower():
            mock_data = APIResponseFactory.create_alpha_vantage_response('MOCK', 5)
        elif 'finnhub' in url.lower():
            mock_data = APIResponseFactory.create_finnhub_profile_response('MOCK')
        elif 'polygon' in url.lower():
            mock_data = APIResponseFactory.create_polygon_intraday_response('MOCK', 3)
        else:
            mock_data = {'status': 'success', 'data': 'mock'}
        
        return APITestHelper.mock_successful_response(mock_data)
    
    with patch('requests.get', side_effect=_mock_get) as mock:
        yield mock

@pytest.fixture
def mock_requests_error():
    """Mock requests.get to simulate API errors"""
    def _mock_error(url: str, *args, **kwargs):
        return APITestHelper.mock_failed_response(500, "Internal Server Error")
    
    with patch('requests.get', side_effect=_mock_error) as mock:
        yield mock

@pytest.fixture
def mock_requests_timeout():
    """Mock requests.get to simulate timeouts"""
    def _mock_timeout(url: str, *args, **kwargs):
        return APITestHelper.mock_timeout_response()
    
    with patch('requests.get', side_effect=_mock_timeout) as mock:
        yield mock

@pytest.fixture
def mock_yfinance():
    """Mock yfinance Ticker for testing"""
    with patch('yfinance.Ticker') as mock_ticker:
        # Create mock ticker instance
        mock_instance = Mock()
        mock_ticker.return_value = mock_instance
        
        # Mock history method
        sample_data = StockDataFactory.create_ohlcv_data('MOCK', 30)
        sample_data = sample_data.set_index('date')
        sample_data.index = pd.to_datetime(sample_data.index)
        
        mock_instance.history.return_value = sample_data[['open', 'high', 'low', 'close', 'volume']]
        
        # Mock info property
        mock_instance.info = {
            'longName': 'Mock Company Inc.',
            'sector': 'Technology',
            'industry': 'Software',
            'marketCap': 1000000000,
            'trailingPE': 25.5,
            'beta': 1.2
        }
        
        yield mock_ticker

# ============================================
# ENVIRONMENT FIXTURES
# ============================================

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    def _set_env(**kwargs):
        original_values = {}
        
        for key, value in kwargs.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value
        
        return original_values
    
    def _restore_env(original_values):
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    original_values = {}
    
    def wrapper(**kwargs):
        nonlocal original_values
        original_values.update(_set_env(**kwargs))
    
    yield wrapper
    
    # Restore original environment
    _restore_env(original_values)

@pytest.fixture
def no_api_keys(mock_env_vars):
    """Remove API keys from environment for testing"""
    mock_env_vars(
        ALPHA_VANTAGE_API_KEY='',
        FINNHUB_API_KEY='',
        POLYGON_API_KEY='',
        TWELVEDATA_API_KEY=''
    )

# ============================================
# DATABASE/STORAGE FIXTURES
# ============================================

@pytest.fixture
def mock_database():
    """Mock database connection and cursor"""
    mock_conn = Mock()
    mock_cursor = Mock()
    
    # Default empty results
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.rowcount = 0
    
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.commit.return_value = None
    mock_conn.rollback.return_value = None
    
    return {
        'connection': mock_conn,
        'cursor': mock_cursor
    }

@pytest.fixture
def mock_cache():
    """Mock cache implementation for testing"""
    cache_data = {}
    
    class MockCache:
        @staticmethod
        def get(key: str):
            return cache_data.get(key)
        
        @staticmethod
        def set(key: str, value: Any, ttl: int = 3600):
            cache_data[key] = value
        
        @staticmethod
        def delete(key: str):
            cache_data.pop(key, None)
        
        @staticmethod
        def clear():
            cache_data.clear()
        
        @staticmethod
        def keys():
            return list(cache_data.keys())
    
    return MockCache()

# ============================================
# PERFORMANCE FIXTURES
# ============================================

@pytest.fixture
def performance_timer():
    """Time test execution for performance testing"""
    start_times = {}
    
    def start_timer(name: str = 'default'):
        start_times[name] = datetime.now()
    
    def end_timer(name: str = 'default'):
        if name not in start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = datetime.now() - start_times[name]
        return elapsed.total_seconds()
    
    def get_elapsed(name: str = 'default'):
        if name not in start_times:
            return 0
        
        elapsed = datetime.now() - start_times[name]
        return elapsed.total_seconds()
    
    timer = Mock()
    timer.start = start_timer
    timer.end = end_timer
    timer.elapsed = get_elapsed
    
    return timer

# ============================================
# VALIDATION FIXTURES
# ============================================

@pytest.fixture
def assert_helpers():
    """Provide assertion helpers for tests"""
    return AssertionHelper

@pytest.fixture
def data_validator():
    """Provide data validation helper"""
    return DataValidationHelper

# ============================================
# SKIP CONDITIONS
# ============================================

def pytest_runtest_setup(item):
    """Skip tests based on markers and environment"""
    # Skip API tests if no API keys available
    if item.get_closest_marker("api"):
        api_keys = [
            'ALPHA_VANTAGE_API_KEY',
            'FINNHUB_API_KEY', 
            'POLYGON_API_KEY',
            'TWELVEDATA_API_KEY'
        ]
        
        if not any(os.getenv(key) for key in api_keys):
            pytest.skip("API keys not available - skipping API test")
    
    # Skip network tests if explicitly requested
    if item.get_closest_marker("network"):
        if os.getenv("SKIP_NETWORK_TESTS", "").lower() in ("true", "1", "yes"):
            pytest.skip("Network tests disabled via SKIP_NETWORK_TESTS")

# ============================================
# CLEANUP FIXTURES
# ============================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test"""
    # Setup (runs before test)
    yield
    
    # Teardown (runs after test)
    # Reset random seeds for reproducibility
    np.random.seed(42)
    
    # Clear any matplotlib figures
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass

# ============================================
# REPORTING FIXTURES
# ============================================

@pytest.fixture
def test_results():
    """Collect test results for reporting"""
    results = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'errors': []
    }
    
    def add_result(status: str, error: str = None):
        if status in results:
            results[status] += 1
        if error:
            results['errors'].append(error)
    
    results['add'] = add_result
    return results

# ============================================
# CONFIGURATION FOR INTEGRATION TESTS
# ============================================

@pytest.fixture(scope="session")
def integration_config():
    """Configuration for integration tests"""
    return {
        'api_timeout': 30,
        'retry_attempts': 3,
        'test_symbols': ['AAPL', 'MSFT'],
        'max_test_duration': 300,  # 5 minutes
        'enable_live_api_tests': os.getenv('ENABLE_LIVE_API_TESTS', 'false').lower() == 'true'
    }

# ============================================
# CUSTOM PYTEST HOOKS
# ============================================

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Custom test summary reporting"""
    if hasattr(terminalreporter, 'stats'):
        total_tests = len(terminalreporter.stats.get('passed', [])) + \
                     len(terminalreporter.stats.get('failed', [])) + \
                     len(terminalreporter.stats.get('skipped', []))
        
        if total_tests > 0:
            print(f"\n🧪 StockPredictionPro Test Summary:")
            print(f"   Total Tests: {total_tests}")
            print(f"   ✅ Passed: {len(terminalreporter.stats.get('passed', []))}")
            print(f"   ❌ Failed: {len(terminalreporter.stats.get('failed', []))}")
            print(f"   ⏭️ Skipped: {len(terminalreporter.stats.get('skipped', []))}")
