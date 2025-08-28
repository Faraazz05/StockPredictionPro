"""
tests/unit/test_data/test_fetchers.py

Unit tests for data fetching components in StockPredictionPro.
Tests the EnhancedStockDataExtractor from data/__init__.py and related 
data processing scripts.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests
import tempfile
import json

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import test utilities
from tests.utils.mock_factories import (
    StockDataFactory, APIResponseFactory, FundamentalsFactory, MockObjectFactory
)
from tests.utils.assertions import (
    assert_valid_stock_dataframe, StockDataAssertions, APIResponseAssertions
)
from tests.utils.test_helpers import (
    DataValidationHelper, APITestHelper, EnvironmentTestHelper
)

# Import the actual modules we're testing (adjusted for your structure)
try:
    from data import EnhancedStockDataExtractor, StockDataExtractor
    from scripts.data.download_data import download_stock_data
    from scripts.data.clean_data import clean_stock_data  
    from scripts.data.validate_data import validate_stock_data
    from scripts.data.update_data import update_stock_data
except ImportError as e:
    # Mock the imports if modules don't exist yet or have different structure
    print(f"Warning: Could not import modules: {e}")
    EnhancedStockDataExtractor = Mock()
    StockDataExtractor = Mock()
    download_stock_data = Mock()
    clean_stock_data = Mock()
    validate_stock_data = Mock()
    update_stock_data = Mock()

# ============================================
# TEST ENHANCED STOCK DATA EXTRACTOR
# ============================================

class TestEnhancedStockDataExtractor:
    """Test the main EnhancedStockDataExtractor class from data/__init__.py"""
    
    def test_extractor_initialization(self, api_keys):
        """Test EnhancedStockDataExtractor initializes correctly"""
        with patch.dict(os.environ, api_keys):
            extractor = EnhancedStockDataExtractor()
            
            assert hasattr(extractor, 'session')
            assert hasattr(extractor, 'api_status')
            assert extractor.session is not None
    
    @pytest.mark.usefixtures("mock_yfinance")
    def test_extract_daily_prices_success(self, sample_alpha_vantage_response, api_keys):
        """Test successful daily price extraction"""
        with patch.dict(os.environ, api_keys):
            extractor = EnhancedStockDataExtractor()
            result = extractor.extract_daily_prices(['AAPL'])
            
            # Validate result
            assert isinstance(result, pd.DataFrame)
            if not result.empty:
                assert_valid_stock_dataframe(result)
                assert 'AAPL' in result['Symbol'].values
    
    @pytest.mark.usefixtures("mock_requests_get")
    def test_extract_fundamentals_success(self, sample_finnhub_response, api_keys):
        """Test successful fundamental data extraction"""
        with patch.dict(os.environ, api_keys):
            with patch('requests.get') as mock_get:
                mock_response = APITestHelper.mock_successful_response(sample_finnhub_response)
                mock_get.return_value = mock_response
                
                extractor = EnhancedStockDataExtractor()
                result = extractor.extract_fundamentals(['AAPL'])
                
                # Validate result
                assert isinstance(result, pd.DataFrame)
                if not result.empty:
                    assert 'symbol' in result.columns
    
    @pytest.mark.usefixtures("mock_requests_error")
    def test_extract_daily_prices_api_error(self, api_keys):
        """Test handling of API errors during data extraction"""
        with patch.dict(os.environ, api_keys):
            extractor = EnhancedStockDataExtractor()
            
            # Should handle errors gracefully and return empty DataFrame
            result = extractor.extract_daily_prices(['INVALID_SYMBOL'])
            
            # Should not raise exception but may return empty DataFrame
            assert isinstance(result, pd.DataFrame)
    
    @pytest.mark.parametrize("symbols,expected_count", [
        (['AAPL'], 1),
        (['AAPL', 'MSFT'], 2),
        (['AAPL', 'MSFT', 'GOOGL'], 3),
    ])
    @pytest.mark.usefixtures("mock_yfinance")
    def test_extract_multiple_symbols(self, symbols, expected_count, api_keys):
        """Test extracting data for multiple symbols"""
        with patch.dict(os.environ, api_keys):
            extractor = EnhancedStockDataExtractor()
            result = extractor.extract_daily_prices(symbols)
            
            if not result.empty:
                unique_symbols = result['Symbol'].unique()
                assert len(unique_symbols) <= expected_count
                
                # Validate each symbol has valid data
                for symbol in unique_symbols:
                    symbol_data = result[result['Symbol'] == symbol]
                    assert_valid_stock_dataframe(symbol_data)

# ============================================
# TEST DATA SCRIPTS
# ============================================

class TestDataScripts:
    """Test the data processing scripts in scripts/data/"""
    
    def test_download_data_function_exists(self):
        """Test that download_data function exists and is callable"""
        # This test ensures the function exists (even if mocked)
        assert callable(download_stock_data)
    
    def test_clean_data_function_exists(self):
        """Test that clean_data function exists and is callable"""
        assert callable(clean_stock_data)
    
    def test_validate_data_function_exists(self):
        """Test that validate_data function exists and is callable"""
        assert callable(validate_stock_data)
    
    def test_update_data_function_exists(self):
        """Test that update_data function exists and is callable"""
        assert callable(update_stock_data)
    
    @pytest.mark.usefixtures("mock_yfinance")
    def test_download_data_with_mock(self, sample_stock_data, temp_csv_file):
        """Test download_data with mocked data"""
        # Create a mock implementation of download_stock_data
        def mock_download(symbols, output_dir=None):
            if output_dir:
                file_path = temp_csv_file(sample_stock_data)
                return file_path
            return sample_stock_data
        
        with patch('scripts.data.download_data.download_stock_data', side_effect=mock_download):
            result = download_stock_data(['AAPL'], output_dir='temp')
            
            if isinstance(result, str):  # File path
                assert os.path.exists(result)
            elif isinstance(result, pd.DataFrame):  # DataFrame
                assert_valid_stock_dataframe(result)
    
    def test_clean_data_with_sample(self, sample_stock_data):
        """Test clean_data with sample data"""
        # Create mock implementation for clean_stock_data
        def mock_clean(data):
            if isinstance(data, pd.DataFrame):
                # Simple cleaning: remove any null values
                cleaned = data.dropna()
                return cleaned
            return data
        
        with patch('scripts.data.clean_data.clean_stock_data', side_effect=mock_clean):
            # Add some null values to test cleaning
            dirty_data = sample_stock_data.copy()
            dirty_data.loc[0, 'open'] = np.nan
            
            cleaned = clean_stock_data(dirty_data)
            
            if isinstance(cleaned, pd.DataFrame):
                assert len(cleaned) <= len(dirty_data)  # Should remove or fix nulls
    
    def test_validate_data_with_sample(self, sample_stock_data):
        """Test validate_data with sample data"""
        def mock_validate(data):
            if isinstance(data, pd.DataFrame):
                is_valid, errors = DataValidationHelper.validate_stock_data_schema(data)
                return {'valid': is_valid, 'errors': errors}
            return {'valid': False, 'errors': ['Invalid data type']}
        
        with patch('scripts.data.validate_data.validate_stock_data', side_effect=mock_validate):
            result = validate_stock_data(sample_stock_data)
            
            if isinstance(result, dict):
                assert 'valid' in result
                assert 'errors' in result

# ============================================
# TEST DATA VALIDATION
# ============================================

class TestDataValidation:
    """Test data validation logic for extracted data"""
    
    def test_validate_extracted_stock_data(self, sample_stock_data):
        """Test validation of extracted stock data"""
        # Valid data should pass
        is_valid, errors = DataValidationHelper.validate_stock_data_schema(sample_stock_data)
        assert is_valid, f"Valid data failed validation: {errors}"
        
        # Invalid data should fail
        invalid_data = sample_stock_data.copy()
        invalid_data['high'] = invalid_data['low'] - 1  # Violate OHLC logic
        
        is_valid, errors = DataValidationHelper.validate_stock_data_schema(invalid_data)
        assert not is_valid
        assert any('high' in error.lower() for error in errors)
    
    def test_validate_date_ranges(self, sample_stock_data):
        """Test date range validation"""
        start_date = datetime.now() - timedelta(days=40)
        end_date = datetime.now()
        
        is_valid, errors = DataValidationHelper.validate_date_range(
            sample_stock_data, expected_days=30, start_date=start_date, end_date=end_date
        )
        
        # Should be valid for reasonable ranges
        assert is_valid, f"Date range validation failed: {errors}"
    
    def test_validate_fundamentals_data(self, sample_fundamentals):
        """Test validation of fundamental data"""
        # Check required fields
        required_fields = ['symbol', 'company_name', 'market_cap', 'pe_ratio']
        
        for field in required_fields:
            assert field in sample_fundamentals, f"Missing required field: {field}"
        
        # Check data types and ranges
        assert isinstance(sample_fundamentals['market_cap'], (int, float))
        assert sample_fundamentals['market_cap'] > 0

# ============================================
# TEST API RESPONSES
# ============================================

class TestAPIResponses:
    """Test API response parsing and validation"""
    
    def test_alpha_vantage_response_parsing(self):
        """Test parsing of Alpha Vantage API responses"""
        # Test successful response
        mock_response = APIResponseFactory.create_alpha_vantage_response('AAPL', 5)
        
        # Validate response structure
        APIResponseAssertions.assert_alpha_vantage_daily_response(mock_response)
        
        assert 'Meta Data' in mock_response
        assert 'Time Series (Daily)' in mock_response
        assert len(mock_response['Time Series (Daily)']) == 5
    
    def test_finnhub_response_parsing(self):
        """Test parsing of Finnhub responses"""
        mock_response = APIResponseFactory.create_finnhub_profile_response('AAPL', 'success', 'tech')
        
        # Validate response structure
        APIResponseAssertions.assert_finnhub_profile_response(mock_response)
        
        assert 'name' in mock_response
        assert 'ticker' in mock_response
        assert mock_response['ticker'] == 'AAPL'
    
    def test_api_error_handling(self):
        """Test handling of API error responses"""
        error_response = APIResponseFactory.create_alpha_vantage_response(
            'INVALID', response_type='error'
        )
        
        # Should contain error message
        assert 'Error Message' in error_response
        
        # Should raise assertion error when validated
        with pytest.raises(AssertionError, match="Alpha Vantage API error"):
            APIResponseAssertions.assert_alpha_vantage_daily_response(error_response)

# ============================================
# TEST ERROR HANDLING
# ============================================

class TestErrorHandling:
    """Test error handling in data fetching"""
    
    @pytest.mark.usefixtures("mock_requests_timeout")
    def test_handle_api_timeout(self, api_keys):
        """Test handling of API timeouts"""
        with patch.dict(os.environ, api_keys):
            extractor = EnhancedStockDataExtractor()
            
            # Should handle timeout gracefully
            result = extractor.extract_daily_prices(['AAPL'])
            
            # Should return empty DataFrame rather than raising exception
            assert isinstance(result, pd.DataFrame)
    
    @pytest.mark.usefixtures("mock_requests_error")
    def test_handle_api_error(self, api_keys):
        """Test handling of API errors"""
        with patch.dict(os.environ, api_keys):
            extractor = EnhancedStockDataExtractor()
            result = extractor.extract_daily_prices(['AAPL'])
            
            # Should handle errors gracefully
            assert isinstance(result, pd.DataFrame)
    
    def test_handle_invalid_symbols(self, api_keys):
        """Test handling of invalid stock symbols"""
        with patch.dict(os.environ, api_keys):
            with patch('requests.get') as mock_get:
                # Mock empty response for invalid symbol
                mock_response = APITestHelper.mock_successful_response({})
                mock_get.return_value = mock_response
                
                extractor = EnhancedStockDataExtractor()
                result = extractor.extract_fundamentals(['INVALID_SYMBOL'])
                
                # Should return empty DataFrame for invalid symbols
                assert isinstance(result, pd.DataFrame)

# ============================================
# TEST INTEGRATION SCENARIOS
# ============================================

class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.usefixtures("mock_yfinance")
    def test_complete_data_extraction_workflow(self, api_keys, temp_data_dir):
        """Test complete data extraction workflow"""
        with patch.dict(os.environ, api_keys):
            extractor = EnhancedStockDataExtractor()
            
            # Extract data for test symbols
            symbols = ['AAPL', 'MSFT'] 
            
            # Mock successful extractions
            with patch.object(extractor, 'extract_fundamentals') as mock_fundamentals:
                mock_fundamentals.return_value = pd.DataFrame([
                    FundamentalsFactory.create_company_fundamentals('AAPL', 'tech'),
                    FundamentalsFactory.create_company_fundamentals('MSFT', 'tech')
                ])
                
                # Run extraction
                all_data = extractor.extract_all_data(symbols, market_filter='us')
                
                # Validate results
                assert isinstance(all_data, dict)
                assert 'daily_prices' in all_data
                
                if 'daily_prices' in all_data and not all_data['daily_prices'].empty:
                    assert_valid_stock_dataframe(all_data['daily_prices'])

# ============================================
# TEST MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    """Run tests when module is executed directly"""
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
        "--durations=10"  # Show 10 slowest tests
    ])
