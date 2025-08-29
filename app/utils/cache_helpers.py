"""
app/utils/cache_helpers.py

Caching utilities for StockPredictionPro Streamlit application.
Provides efficient caching support with TTL, key generation, and cleanup.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import hashlib
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import streamlit as st
import pandas as pd
import logging
from datetime import datetime, timedelta

# Setup logging
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_VERSION = "v1.0"
DEFAULT_TTL = 3600  # 1 hour in seconds
MAX_CACHE_SIZE = 1000  # Maximum number of cache files

# ============================================
# CACHE KEY GENERATION
# ============================================

def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate consistent cache key from arguments
    
    Args:
        *args: Positional arguments for key
        **kwargs: Keyword arguments for key
    
    Returns:
        str: SHA256 hashed cache key with version prefix
    """
    # Combine args and kwargs into single string
    key_components = [str(arg) for arg in args]
    
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        key_components.extend([f"{k}={v}" for k, v in sorted_kwargs])
    
    key_string = "|".join(key_components)
    key_hash = hashlib.sha256(key_string.encode('utf-8')).hexdigest()[:16]
    
    return f"{CACHE_VERSION}_{key_hash}"


def generate_data_cache_key(symbol: str, data_type: str, date_range: str = None) -> str:
    """
    Generate cache key for stock data
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        data_type: Type of data ('daily', 'intraday', 'fundamentals')
        date_range: Optional date range string
    
    Returns:
        str: Cache key for stock data
    """
    components = [symbol.upper(), data_type.lower()]
    if date_range:
        components.append(date_range)
    
    return generate_cache_key("stock_data", *components)


def generate_model_cache_key(symbol: str, model_type: str, features: List[str] = None) -> str:
    """
    Generate cache key for model results
    
    Args:
        symbol: Stock symbol
        model_type: Type of model ('regression', 'classification')
        features: List of feature names used
    
    Returns:
        str: Cache key for model results
    """
    components = [symbol.upper(), model_type.lower()]
    if features:
        features_str = ",".join(sorted(features))
        components.append(features_str)
    
    return generate_cache_key("model_results", *components)

# ============================================
# FILE-BASED CACHING
# ============================================

class CacheHelper:
    """File-based cache helper with TTL support"""
    
    def __init__(self, cache_dir: str = None, default_ttl: int = DEFAULT_TTL):
        """
        Initialize cache helper
        
        Args:
            cache_dir: Directory for cache files (default: ~/.stockpred_cache)
            default_ttl: Default time-to-live in seconds
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.stockpred_cache")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get full path for cache file"""
        return self.cache_dir / f"{cache_key}.json"
    
    def is_expired(self, cache_key: str, ttl: int = None) -> bool:
        """
        Check if cache entry is expired
        
        Args:
            cache_key: Cache key to check
            ttl: Time-to-live in seconds (uses default if None)
        
        Returns:
            bool: True if expired or doesn't exist
        """
        cache_path = self.get_cache_path(cache_key)
        
        if not cache_path.exists():
            return True
        
        if ttl is None:
            ttl = self.default_ttl
        
        if ttl <= 0:  # No expiration
            return False
        
        file_age = time.time() - cache_path.stat().st_mtime
        return file_age > ttl
    
    def save(self, cache_key: str, data: Any, ttl: int = None) -> bool:
        """
        Save data to cache
        
        Args:
            cache_key: Cache key
            data: Data to cache (must be JSON serializable)
            ttl: Time-to-live in seconds
        
        Returns:
            bool: True if saved successfully
        """
        try:
            cache_path = self.get_cache_path(cache_key)
            
            # Prepare cache entry with metadata
            cache_entry = {
                "data": data,
                "timestamp": time.time(),
                "ttl": ttl or self.default_ttl,
                "version": CACHE_VERSION
            }
            
            # Special handling for pandas DataFrames
            if isinstance(data, pd.DataFrame):
                cache_entry["data"] = data.to_dict(orient='records')
                cache_entry["data_type"] = "pandas_dataframe"
                cache_entry["columns"] = list(data.columns)
                cache_entry["index_name"] = data.index.name
            
            # Save to JSON file
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=2, default=str)
            
            logger.debug(f"Cache saved: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cache {cache_key}: {e}")
            return False
    
    def load(self, cache_key: str, ttl: int = None) -> Optional[Any]:
        """
        Load data from cache if not expired
        
        Args:
            cache_key: Cache key
            ttl: Time-to-live in seconds
        
        Returns:
            Cached data or None if expired/missing
        """
        try:
            if self.is_expired(cache_key, ttl):
                return None
            
            cache_path = self.get_cache_path(cache_key)
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_entry = json.load(f)
            
            # Extract data
            data = cache_entry.get("data")
            data_type = cache_entry.get("data_type", "")
            
            # Reconstruct pandas DataFrame if needed
            if data_type == "pandas_dataframe":
                df = pd.DataFrame(data)
                if "columns" in cache_entry:
                    df.columns = cache_entry["columns"]
                if cache_entry.get("index_name"):
                    df.index.name = cache_entry["index_name"]
                return df
            
            logger.debug(f"Cache loaded: {cache_key}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load cache {cache_key}: {e}")
            return None
    
    def delete(self, cache_key: str) -> bool:
        """
        Delete cache entry
        
        Args:
            cache_key: Cache key to delete
        
        Returns:
            bool: True if deleted successfully
        """
        try:
            cache_path = self.get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"Cache deleted: {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache {cache_key}: {e}")
            return False
    
    def cleanup(self, max_age_days: int = 30) -> int:
        """
        Clean up old cache files
        
        Args:
            max_age_days: Delete files older than this many days
        
        Returns:
            int: Number of files deleted
        """
        deleted_count = 0
        max_age_seconds = max_age_days * 24 * 3600
        current_time = time.time()
        
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    if current_time - cache_file.stat().st_mtime > max_age_seconds:
                        cache_file.unlink()
                        deleted_count += 1
                        logger.debug(f"Cleaned up old cache: {cache_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean cache file {cache_file}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
        
        logger.info(f"Cache cleanup completed: {deleted_count} files deleted")
        return deleted_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            dict: Cache statistics
        """
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_files = len(cache_files)
            
            if total_files == 0:
                return {
                    "total_files": 0,
                    "total_size_mb": 0.0,
                    "avg_file_size_kb": 0.0,
                    "oldest_file": None,
                    "newest_file": None
                }
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in cache_files)
            total_size_mb = total_size / (1024 * 1024)
            avg_size_kb = (total_size / total_files) / 1024
            
            # Find oldest and newest files
            file_times = [(f.stat().st_mtime, f.name) for f in cache_files]
            file_times.sort()
            
            oldest_file = file_times[0][1]
            newest_file = file_times[-1][1]
            
            return {
                "total_files": total_files,
                "total_size_mb": round(total_size_mb, 2),
                "avg_file_size_kb": round(avg_size_kb, 2),
                "oldest_file": oldest_file,
                "newest_file": newest_file,
                "cache_dir": str(self.cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

# ============================================
# STREAMLIT INTEGRATION
# ============================================

def cached_data_loader(cache_key: str, loader_func: Callable, 
                      ttl: int = DEFAULT_TTL, 
                      show_spinner: bool = True) -> Any:
    """
    Load data with caching support for Streamlit
    
    Args:
        cache_key: Unique cache key
        loader_func: Function to load data if not cached
        ttl: Time-to-live in seconds
        show_spinner: Show loading spinner
    
    Returns:
        Loaded data (from cache or fresh)
    """
    cache_helper = CacheHelper()
    
    # Try to load from cache first
    cached_data = cache_helper.load(cache_key, ttl)
    
    if cached_data is not None:
        st.info(f"📦 Loaded from cache (key: {cache_key[:16]}...)")
        return cached_data
    
    # Load fresh data
    if show_spinner:
        with st.spinner("Loading data..."):
            fresh_data = loader_func()
    else:
        fresh_data = loader_func()
    
    # Cache the fresh data
    if fresh_data is not None:
        cache_helper.save(cache_key, fresh_data, ttl)
        st.success(f"💾 Data cached (TTL: {ttl}s)")
    
    return fresh_data


@st.cache_data(ttl=DEFAULT_TTL)
def cached_csv_loader(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load CSV with Streamlit caching
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame or None
    """
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None
    except Exception as e:
        logger.error(f"Failed to load CSV {file_path}: {e}")
        return None


def clear_streamlit_cache():
    """Clear Streamlit cache and file cache"""
    st.cache_data.clear()
    
    cache_helper = CacheHelper()
    deleted_count = cache_helper.cleanup(max_age_days=0)  # Delete all
    
    st.success(f"🗑️ Cache cleared! Deleted {deleted_count} files")


# ============================================
# INITIALIZATION
# ============================================

# Global cache helper instance
_cache_helper = None

def get_cache_helper() -> CacheHelper:
    """Get global cache helper instance"""
    global _cache_helper
    if _cache_helper is None:
        _cache_helper = CacheHelper()
    return _cache_helper

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    """
    Example usage patterns:
    
    # Basic caching
    cache = CacheHelper()
    cache.save("test_key", {"data": "value"}, ttl=3600)
    data = cache.load("test_key")
    
    # Stock data caching
    key = generate_data_cache_key("AAPL", "daily", "2025-01")
    cache.save(key, stock_dataframe)
    
    # Streamlit integration
    def load_stock_data():
        return pd.read_csv("AAPL_daily.csv")
    
    data = cached_data_loader("aapl_daily", load_stock_data, ttl=1800)
    
    # Cache management
    stats = cache.get_cache_stats()
    cleaned = cache.cleanup(max_age_days=7)
    """
    
    # Demo
    cache = CacheHelper()
    print("Cache directory:", cache.cache_dir)
    print("Cache stats:", cache.get_cache_stats())
    
    # Test caching
    test_data = {"symbol": "AAPL", "price": 150.0, "volume": 1000000}
    key = generate_cache_key("test", "demo")
    
    cache.save(key, test_data, ttl=60)
    loaded = cache.load(key)
    print("Cached data:", loaded)
