"""
tests/unit/test_data/test_cache.py

Unit tests for caching components in StockPredictionPro.
Tests cache operations, expiration, persistence, and integration
with data fetching workflows.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import time
import json
import pickle
import tempfile
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import test utilities
from tests.utils.mock_factories import (
    StockDataFactory, APIResponseFactory, SystemFactory
)
from tests.utils.assertions import assert_valid_stock_dataframe
from tests.utils.test_helpers import FileSystemTestHelper

# Import cache modules (with fallback mocks)
try:
    from src.data.cache import CacheManager, DataCache
    from scripts.utils.cache_utils import CacheHelper
except ImportError:
    # Mock the imports if modules don't exist yet
    CacheManager = Mock()
    DataCache = Mock()
    CacheHelper = Mock()

# ============================================
# TEST CACHE MANAGER CORE FUNCTIONALITY
# ============================================

class TestCacheManager:
    """Test core caching functionality"""
    
    @pytest.fixture(autouse=True)
    def setup_cache(self, tmp_path):
        """Setup cache manager for each test"""
        self.cache_dir = tmp_path / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize cache manager
        self.cache = CacheManager(cache_dir=str(self.cache_dir), ttl_seconds=3600)
        yield
        
        # Cleanup
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
    
    def test_cache_initialization(self):
        """Test cache manager initializes correctly"""
        assert hasattr(self.cache, 'cache_dir')
        assert hasattr(self.cache, 'ttl_seconds')
        
        # Cache directory should exist
        assert Path(self.cache.cache_dir).exists()
    
    def test_cache_set_and_get_dataframe(self, sample_stock_data):
        """Test caching DataFrame data"""
        cache_key = "test_stock_data_AAPL"
        
        # Set data in cache
        success = self.cache.set(cache_key, sample_stock_data)
        assert success, "Cache set operation should succeed"
        
        # Get data from cache
        cached_data = self.cache.get(cache_key)
        
        # Validate cached data
        assert cached_data is not None, "Cache should return data"
        assert isinstance(cached_data, pd.DataFrame), "Cached data should be DataFrame"
        pd.testing.assert_frame_equal(cached_data, sample_stock_data)
    
    def test_cache_set_and_get_dict(self, sample_fundamentals):
        """Test caching dictionary data"""
        cache_key = "test_fundamentals_AAPL"
        
        # Set dict in cache
        success = self.cache.set(cache_key, sample_fundamentals)
        assert success
        
        # Get dict from cache
        cached_data = self.cache.get(cache_key)
        
        # Validate cached data
        assert cached_data is not None
        assert isinstance(cached_data, dict)
        assert cached_data == sample_fundamentals
    
    def test_cache_exists(self, sample_stock_data):
        """Test cache existence checking"""
        cache_key = "test_exists_AAPL"
        
        # Key should not exist initially
        assert not self.cache.exists(cache_key)
        
        # Set data
        self.cache.set(cache_key, sample_stock_data)
        
        # Key should exist now
        assert self.cache.exists(cache_key)
    
    def test_cache_delete(self, sample_stock_data):
        """Test cache deletion"""
        cache_key = "test_delete_AAPL"
        
        # Set and verify data exists
        self.cache.set(cache_key, sample_stock_data)
        assert self.cache.exists(cache_key)
        
        # Delete data
        success = self.cache.delete(cache_key)
        assert success
        
        # Verify data is gone
        assert not self.cache.exists(cache_key)
        assert self.cache.get(cache_key) is None
    
    def test_cache_clear(self, sample_stock_data):
        """Test clearing entire cache"""
        # Set multiple cache entries
        keys = ["test_clear_AAPL", "test_clear_MSFT", "test_clear_GOOGL"]
        
        for key in keys:
            self.cache.set(key, sample_stock_data)
            assert self.cache.exists(key)
        
        # Clear cache
        self.cache.clear()
        
        # Verify all entries are gone
        for key in keys:
            assert not self.cache.exists(key)
    
    @pytest.mark.slow
    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        cache_key = "test_expiry_AAPL"
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        
        # Create cache with 1 second TTL
        short_cache = CacheManager(cache_dir=str(self.cache_dir), ttl_seconds=1)
        
        # Set data
        short_cache.set(cache_key, test_data)
        assert short_cache.exists(cache_key)
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Data should be expired
        assert not short_cache.exists(cache_key)
        assert short_cache.get(cache_key) is None
    
    def test_cache_key_generation(self):
        """Test cache key generation and normalization"""
        # Test different key formats
        test_cases = [
            ("AAPL", "daily", "2023-01-01"),
            ("MSFT", "fundamentals", "latest"),
            ("GOOGL", "news", "2023-12-31")
        ]
        
        for symbol, data_type, date in test_cases:
            key = self.cache._generate_key(symbol, data_type, date)
            
            # Key should be string
            assert isinstance(key, str)
            
            # Key should contain relevant components
            assert symbol in key
            assert data_type in key
            
            # Key should be consistent
            key2 = self.cache._generate_key(symbol, data_type, date)
            assert key == key2

# ============================================
# TEST DATA-SPECIFIC CACHING
# ============================================

class TestDataCaching:
    """Test caching for specific data types"""
    
    @pytest.fixture(autouse=True)
    def setup_data_cache(self, tmp_path):
        """Setup data-specific cache"""
        self.cache_dir = tmp_path / "data_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.data_cache = DataCache(cache_dir=str(self.cache_dir))
        yield
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
    
    def test_cache_stock_prices(self, sample_stock_data):
        """Test caching stock price data"""
        symbol = "AAPL"
        
        # Cache stock data
        success = self.data_cache.cache_stock_prices(symbol, sample_stock_data)
        assert success
        
        # Retrieve cached data
        cached_data = self.data_cache.get_stock_prices(symbol)
        assert cached_data is not None
        assert_valid_stock_dataframe(cached_data)
        
        # Verify data integrity
        pd.testing.assert_frame_equal(cached_data, sample_stock_data)
    
    def test_cache_fundamentals(self, sample_fundamentals):
        """Test caching fundamental data"""
        symbol = "AAPL"
        
        # Cache fundamentals
        success = self.data_cache.cache_fundamentals(symbol, sample_fundamentals)
        assert success
        
        # Retrieve cached data
        cached_data = self.data_cache.get_fundamentals(symbol)
        assert cached_data is not None
        assert isinstance(cached_data, dict)
        assert cached_data['symbol'] == symbol
    
    def test_cache_technical_indicators(self):
        """Test caching technical indicators"""
        symbol = "AAPL"
        indicators = {
            'rsi_14': 65.5,
            'macd': 1.25,
            'macd_signal': 1.10,
            'sma_20': 150.25,
            'sma_50': 148.75,
            'bollinger_upper': 155.0,
            'bollinger_lower': 145.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache indicators
        success = self.data_cache.cache_technical_indicators(symbol, indicators)
        assert success
        
        # Retrieve cached data
        cached_data = self.data_cache.get_technical_indicators(symbol)
        assert cached_data is not None
        assert cached_data['rsi_14'] == 65.5
        assert cached_data['macd'] == 1.25
    
    def test_cache_news_sentiment(self):
        """Test caching news sentiment data"""
        symbol = "AAPL"
        sentiment = {
            'bullish_percent': 0.75,
            'bearish_percent': 0.25,
            'news_score': 0.68,
            'articles_count': 25,
            'last_updated': datetime.now().isoformat()
        }
        
        # Cache sentiment
        success = self.data_cache.cache_news_sentiment(symbol, sentiment)
        assert success
        
        # Retrieve cached data
        cached_data = self.data_cache.get_news_sentiment(symbol)
        assert cached_data is not None
        assert cached_data['bullish_percent'] == 0.75

# ============================================
# TEST CACHE PERSISTENCE
# ============================================

class TestCachePersistence:
    """Test cache persistence across sessions"""
    
    def test_cache_survives_restart(self, tmp_path, sample_stock_data):
        """Test that cache persists across cache manager restarts"""
        cache_dir = tmp_path / "persistence_test"
        cache_dir.mkdir(exist_ok=True)
        
        cache_key = "persistence_test_AAPL"
        
        # First cache manager - set data
        cache1 = CacheManager(cache_dir=str(cache_dir))
        cache1.set(cache_key, sample_stock_data)
        
        # Verify data exists
        assert cache1.exists(cache_key)
        
        # Create new cache manager (simulate restart)
        cache2 = CacheManager(cache_dir=str(cache_dir))
        
        # Data should still exist
        assert cache2.exists(cache_key)
        cached_data = cache2.get(cache_key)
        
        assert cached_data is not None
        pd.testing.assert_frame_equal(cached_data, sample_stock_data)
    
    def test_cache_file_format(self, tmp_path, sample_stock_data):
        """Test cache file format and structure"""
        cache_dir = tmp_path / "format_test"
        cache_dir.mkdir(exist_ok=True)
        
        cache = CacheManager(cache_dir=str(cache_dir))
        cache_key = "format_test_AAPL"
        
        # Set data
        cache.set(cache_key, sample_stock_data)
        
        # Check that cache files are created
        cache_files = list(cache_dir.glob("*"))
        assert len(cache_files) > 0
        
        # Check file extension (should be .pkl or .json)
        cache_file = cache_files[0]
        assert cache_file.suffix in ['.pkl', '.json', '.cache']

# ============================================
# TEST CACHE INTEGRATION
# ============================================

class TestCacheIntegration:
    """Test cache integration with data fetching"""
    
    @pytest.fixture(autouse=True)
    def setup_integration(self, tmp_path):
        """Setup integration test environment"""
        self.cache_dir = tmp_path / "integration_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = CacheManager(cache_dir=str(self.cache_dir))
        yield
    
    def test_fetch_with_cache_hit(self, sample_stock_data, mock_yfinance):
        """Test data fetching with cache hit"""
        symbol = "AAPL"
        cache_key = f"stock_prices_{symbol}"
        
        # Pre-populate cache
        self.cache.set(cache_key, sample_stock_data)
        
        # Mock data fetcher that should not be called
        with patch('data.EnhancedStockDataExtractor.extract_daily_prices') as mock_fetch:
            # Simulate fetch function that uses cache
            def fetch_with_cache(sym):
                cached = self.cache.get(f"stock_prices_{sym}")
                if cached is not None:
                    return cached
                
                # This should not be reached due to cache hit
                mock_fetch.return_value = StockDataFactory.create_ohlcv_data(sym, 10)
                return mock_fetch.return_value
            
            result = fetch_with_cache(symbol)
            
            # Should return cached data
            pd.testing.assert_frame_equal(result, sample_stock_data)
            
            # Mock fetch should not have been called
            mock_fetch.assert_not_called()
    
    def test_fetch_with_cache_miss(self, sample_stock_data):
        """Test data fetching with cache miss"""
        symbol = "MSFT"
        cache_key = f"stock_prices_{symbol}"
        
        # Ensure cache is empty
        assert not self.cache.exists(cache_key)
        
        # Mock data fetcher
        with patch('data.EnhancedStockDataExtractor.extract_daily_prices') as mock_fetch:
            mock_fetch.return_value = sample_stock_data
            
            # Simulate fetch function that uses cache
            def fetch_with_cache(sym):
                cached = self.cache.get(f"stock_prices_{sym}")
                if cached is not None:
                    return cached
                
                # Cache miss - fetch new data
                new_data = mock_fetch(sym)
                
                # Cache the new data
                self.cache.set(f"stock_prices_{sym}", new_data)
                return new_data
            
            result = fetch_with_cache(symbol)
            
            # Should return fetched data
            pd.testing.assert_frame_equal(result, sample_stock_data)
            
            # Mock fetch should have been called once
            mock_fetch.assert_called_once_with(symbol)
            
            # Data should now be cached
            assert self.cache.exists(cache_key)
    
    def test_cache_invalidation_on_error(self):
        """Test cache invalidation when data fetch fails"""
        symbol = "INVALID"
        cache_key = f"stock_prices_{symbol}"
        
        # Pre-populate cache with old data
        old_data = StockDataFactory.create_ohlcv_data(symbol, 5, seed=1)
        self.cache.set(cache_key, old_data)
        
        # Mock fetch that fails
        with patch('data.EnhancedStockDataExtractor.extract_daily_prices') as mock_fetch:
            mock_fetch.side_effect = Exception("API Error")
            
            def fetch_with_cache_invalidation(sym):
                try:
                    # Try to fetch new data
                    new_data = mock_fetch(sym)
                    self.cache.set(f"stock_prices_{sym}", new_data)
                    return new_data
                except Exception:
                    # On error, invalidate cache and return None
                    self.cache.delete(f"stock_prices_{sym}")
                    return None
            
            result = fetch_with_cache_invalidation(symbol)
            
            # Should return None due to fetch error
            assert result is None
            
            # Cache should be invalidated
            assert not self.cache.exists(cache_key)

# ============================================
# TEST CACHE PERFORMANCE
# ============================================

class TestCachePerformance:
    """Test cache performance characteristics"""
    
    @pytest.fixture(autouse=True)
    def setup_performance(self, tmp_path):
        """Setup performance test environment"""
        self.cache_dir = tmp_path / "performance_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = CacheManager(cache_dir=str(self.cache_dir))
        yield
    
    @pytest.mark.slow
    def test_cache_write_performance(self, performance_timer):
        """Test cache write performance"""
        # Generate large dataset
        large_data = StockDataFactory.create_ohlcv_data('PERF', 1000, seed=42)
        cache_key = "performance_test_write"
        
        # Measure write time
        performance_timer.start('cache_write')
        success = self.cache.set(cache_key, large_data)
        write_time = performance_timer.end('cache_write')
        
        assert success
        assert write_time < 2.0, f"Cache write took {write_time}s, expected < 2s"
    
    @pytest.mark.slow
    def test_cache_read_performance(self, performance_timer):
        """Test cache read performance"""
        # Set up data
        large_data = StockDataFactory.create_ohlcv_data('PERF', 1000, seed=42)
        cache_key = "performance_test_read"
        self.cache.set(cache_key, large_data)
        
        # Measure read time
        performance_timer.start('cache_read')
        cached_data = self.cache.get(cache_key)
        read_time = performance_timer.end('cache_read')
        
        assert cached_data is not None
        assert read_time < 1.0, f"Cache read took {read_time}s, expected < 1s"
    
    def test_cache_memory_usage(self):
        """Test cache memory usage"""
        import psutil
        process = psutil.Process()
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Cache multiple datasets
        for i in range(10):
            data = StockDataFactory.create_ohlcv_data(f'MEM{i}', 100, seed=i)
            self.cache.set(f"memory_test_{i}", data)
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 100, f"Memory growth {memory_growth}MB seems excessive"
    
    def test_concurrent_cache_access(self):
        """Test concurrent cache access"""
        import threading
        
        cache_key = "concurrent_test"
        test_data = StockDataFactory.create_ohlcv_data('CONC', 50)
        results = []
        
        def cache_operation(operation_id):
            try:
                if operation_id % 2 == 0:
                    # Write operation
                    success = self.cache.set(f"{cache_key}_{operation_id}", test_data)
                    results.append(('write', operation_id, success))
                else:
                    # Read operation
                    data = self.cache.get(f"{cache_key}_{operation_id-1}")
                    results.append(('read', operation_id, data is not None))
            except Exception as e:
                results.append(('error', operation_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=cache_operation, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 10
        errors = [r for r in results if r[0] == 'error']
        assert len(errors) == 0, f"Concurrent access errors: {errors}"

# ============================================
# TEST CACHE UTILITIES
# ============================================

class TestCacheUtilities:
    """Test cache utility functions"""
    
    def test_cache_key_hashing(self):
        """Test cache key hashing for consistency"""
        # Test data
        test_inputs = [
            ('AAPL', 'daily', '2023-01-01'),
            ('MSFT', 'fundamentals', 'latest'),
            ('GOOGL', 'news', '2023-12-31')
        ]
        
        for symbol, data_type, date in test_inputs:
            # Generate hash multiple times
            hash1 = CacheHelper.generate_cache_key(symbol, data_type, date)
            hash2 = CacheHelper.generate_cache_key(symbol, data_type, date)
            
            # Should be consistent
            assert hash1 == hash2
            
            # Should be string
            assert isinstance(hash1, str)
            
            # Should be reasonable length
            assert 10 <= len(hash1) <= 100
    
    def test_cache_size_calculation(self, tmp_path):
        """Test cache size calculation utilities"""
        cache_dir = tmp_path / "size_test"
        cache_dir.mkdir(exist_ok=True)
        
        cache = CacheManager(cache_dir=str(cache_dir))
        
        # Add some data
        for i in range(5):
            data = StockDataFactory.create_ohlcv_data(f'SIZE{i}', 100)
            cache.set(f"size_test_{i}", data)
        
        # Calculate cache size
        cache_size = CacheHelper.get_cache_size(str(cache_dir))
        
        # Should be positive
        assert cache_size > 0
        
        # Should be reasonable (not too large)
        assert cache_size < 50 * 1024 * 1024  # Less than 50MB
    
    def test_cache_cleanup_old_entries(self, tmp_path):
        """Test cleaning up old cache entries"""
        cache_dir = tmp_path / "cleanup_test"
        cache_dir.mkdir(exist_ok=True)
        
        # Create cache with short TTL
        cache = CacheManager(cache_dir=str(cache_dir), ttl_seconds=1)
        
        # Add entries
        for i in range(5):
            data = pd.DataFrame({'test': [i]})
            cache.set(f"cleanup_test_{i}", data)
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Run cleanup
        cleaned_count = CacheHelper.cleanup_expired_entries(str(cache_dir))
        
        # Should have cleaned some entries
        assert cleaned_count > 0

# ============================================
# INTEGRATION WITH ACTUAL DATA PIPELINE
# ============================================

class TestDataPipelineCacheIntegration:
    """Test cache integration with actual data pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_with_cache(self, api_keys, tmp_path):
        """Test full data pipeline with caching enabled"""
        cache_dir = tmp_path / "pipeline_cache"
        cache_dir.mkdir(exist_ok=True)
        
        with patch.dict(os.environ, api_keys):
            # Mock the data extractor
            with patch('data.EnhancedStockDataExtractor') as MockExtractor:
                mock_instance = MockExtractor.return_value
                
                # Setup mock data
                mock_daily_data = StockDataFactory.create_ohlcv_data('AAPL', 30)
                mock_instance.extract_daily_prices.return_value = mock_daily_data
                
                # Setup cache-enabled pipeline
                cache = CacheManager(cache_dir=str(cache_dir))
                
                def cached_data_pipeline(symbols, use_cache=True):
                    results = {}
                    
                    for symbol in symbols:
                        cache_key = f"daily_prices_{symbol}"
                        
                        if use_cache:
                            # Check cache first
                            cached_data = cache.get(cache_key)
                            if cached_data is not None:
                                results[symbol] = cached_data
                                continue
                        
                        # Extract new data
                        extractor = MockExtractor()
                        new_data = extractor.extract_daily_prices([symbol])
                        
                        # Cache the new data
                        if use_cache and new_data is not None and not new_data.empty:
                            cache.set(cache_key, new_data)
                        
                        results[symbol] = new_data
                    
                    return results
                
                # Test pipeline
                symbols = ['AAPL', 'MSFT']
                
                # First run - should fetch data
                results1 = cached_data_pipeline(symbols, use_cache=True)
                
                # Verify results
                assert len(results1) == 2
                for symbol in symbols:
                    assert symbol in results1
                    assert not results1[symbol].empty
                
                # Second run - should use cache
                results2 = cached_data_pipeline(symbols, use_cache=True)
                
                # Results should be identical
                for symbol in symbols:
                    pd.testing.assert_frame_equal(results1[symbol], results2[symbol])

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
