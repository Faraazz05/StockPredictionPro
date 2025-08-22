# ============================================
# StockPredictionPro - src/data/cache.py
# Advanced caching system for financial data with intelligent invalidation
# ============================================

import os
import time
import hashlib
import pickle
import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import numpy as np

from ..utils.exceptions import DataValidationError, BusinessLogicError
from ..utils.logger import get_logger
from ..utils.timing import Timer, time_it
from ..utils.config_loader import get
from ..utils.helpers import ensure_directory, format_duration, safe_divide
from ..utils.file_io import save_data, load_data, calculate_file_hash

logger = get_logger('data.cache')

# ============================================
# Cache Entry Management
# ============================================

@dataclass
class CacheEntry:
    """
    Cache entry with metadata and validation
    
    Features:
    - Expiration time tracking
    - Data integrity validation
    - Usage statistics
    - Dependency tracking
    """
    key: str
    data: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    data_hash: Optional[str] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize computed fields"""
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        
        if self.data_hash is None:
            self.data_hash = self._calculate_data_hash()
        
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
    
    def _calculate_data_hash(self) -> str:
        """Calculate hash of cached data"""
        try:
            if isinstance(self.data, pd.DataFrame):
                # Hash DataFrame content
                data_str = str(self.data.values.tobytes()) + str(self.data.columns.tolist())
            elif isinstance(self.data, (dict, list)):
                # Hash JSON-serializable data
                data_str = json.dumps(self.data, sort_keys=True, default=str)
            else:
                # Hash string representation
                data_str = str(self.data)
            
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(self.data).encode()).hexdigest()
    
    def _calculate_size(self) -> int:
        """Calculate approximate size of cached data"""
        try:
            if isinstance(self.data, pd.DataFrame):
                return self.data.memory_usage(deep=True).sum()
            elif isinstance(self.data, (np.ndarray)):
                return self.data.nbytes
            else:
                # Use pickle to estimate size
                return len(pickle.dumps(self.data))
        except Exception:
            return len(str(self.data))
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if cache entry is valid"""
        return not self.is_expired() and self.data is not None
    
    def touch(self):
        """Update access time and count"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'key': self.key,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'access_count': self.access_count,
            'data_hash': self.data_hash,
            'size_bytes': self.size_bytes,
            'metadata': self.metadata,
            'dependencies': self.dependencies
        }

# ============================================
# Cache Storage Backends
# ============================================

class MemoryCache:
    """In-memory cache with size limits and LRU eviction"""
    
    def __init__(self, max_size_bytes: int = 1024 * 1024 * 1024):  # 1GB default
        """
        Initialize memory cache
        
        Args:
            max_size_bytes: Maximum cache size in bytes
        """
        self.max_size_bytes = max_size_bytes
        self.entries: Dict[str, CacheEntry] = {}
        self.current_size_bytes = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry"""
        with self.lock:
            entry = self.entries.get(key)
            if entry and entry.is_valid():
                entry.touch()
                return entry
            elif entry:
                # Remove expired entry
                self._remove_entry(key)
            return None
    
    def put(self, entry: CacheEntry):
        """Put cache entry"""
        with self.lock:
            # Remove existing entry if present
            if entry.key in self.entries:
                self._remove_entry(entry.key)
            
            # Ensure we have space
            self._ensure_space(entry.size_bytes)
            
            # Add new entry
            self.entries[entry.key] = entry
            self.current_size_bytes += entry.size_bytes
            
            logger.debug(f"Cached {entry.key} ({entry.size_bytes} bytes)")
    
    def remove(self, key: str) -> bool:
        """Remove cache entry"""
        with self.lock:
            return self._remove_entry(key)
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry (internal)"""
        if key in self.entries:
            entry = self.entries.pop(key)
            self.current_size_bytes -= entry.size_bytes
            return True
        return False
    
    def _ensure_space(self, required_bytes: int):
        """Ensure sufficient space using LRU eviction"""
        while (self.current_size_bytes + required_bytes > self.max_size_bytes 
               and self.entries):
            
            # Find least recently used entry
            lru_key = min(
                self.entries.keys(),
                key=lambda k: self.entries[k].last_accessed or datetime.min
            )
            
            logger.debug(f"Evicting LRU entry: {lru_key}")
            self._remove_entry(lru_key)
    
    def clear(self):
        """Clear all entries"""
        with self.lock:
            self.entries.clear()
            self.current_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'entry_count': len(self.entries),
                'current_size_bytes': self.current_size_bytes,
                'max_size_bytes': self.max_size_bytes,
                'utilization_pct': (self.current_size_bytes / self.max_size_bytes) * 100,
                'total_accesses': sum(entry.access_count for entry in self.entries.values())
            }

class DiskCache:
    """Persistent disk cache with compression"""
    
    def __init__(self, 
                 cache_dir: Union[str, Path],
                 max_size_bytes: int = 5 * 1024 * 1024 * 1024,  # 5GB default
                 compression: bool = True):
        """
        Initialize disk cache
        
        Args:
            cache_dir: Cache directory
            max_size_bytes: Maximum cache size in bytes
            compression: Whether to use compression
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_bytes
        self.compression = compression
        
        # Ensure cache directory exists
        ensure_directory(self.cache_dir)
        
        # Metadata file for tracking entries
        self.metadata_file = self.cache_dir / '_cache_metadata.json'
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Load existing metadata
        self._load_metadata()
        
        # Cleanup on init
        self._cleanup_expired_entries()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry from disk"""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            # Remove from metadata if file is missing
            self.metadata.pop(key, None)
            self._save_metadata()
            return None
        
        try:
            # Load data
            if self.compression and file_path.suffix == '.gz':
                data = load_data(file_path)
            else:
                data = load_data(file_path)
            
            # Get metadata
            entry_meta = self.metadata.get(key, {})
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=datetime.fromisoformat(entry_meta['created_at']),
                expires_at=datetime.fromisoformat(entry_meta['expires_at']) if entry_meta.get('expires_at') else None,
                last_accessed=datetime.fromisoformat(entry_meta['last_accessed']) if entry_meta.get('last_accessed') else None,
                access_count=entry_meta.get('access_count', 0),
                data_hash=entry_meta.get('data_hash'),
                size_bytes=entry_meta.get('size_bytes', 0),
                metadata=entry_meta.get('metadata', {}),
                dependencies=entry_meta.get('dependencies', [])
            )
            
            if entry.is_valid():
                entry.touch()
                
                # Update metadata
                self.metadata[key] = entry.to_dict()
                self._save_metadata()
                
                return entry
            else:
                # Remove expired entry
                self.remove(key)
                return None
                
        except Exception as e:
            logger.error(f"Failed to load cache entry {key}: {e}")
            self.remove(key)
            return None
    
    def put(self, entry: CacheEntry):
        """Put cache entry to disk"""
        try:
            # Ensure space
            self._ensure_space(entry.size_bytes)
            
            # Save data
            file_path = self._get_file_path(entry.key)
            
            if self.compression:
                file_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            success = save_data(entry.data, file_path, compression=self.compression)
            
            if success:
                # Update metadata
                self.metadata[entry.key] = entry.to_dict()
                self._save_metadata()
                
                logger.debug(f"Disk cached {entry.key} ({entry.size_bytes} bytes)")
            else:
                logger.error(f"Failed to save cache entry {entry.key}")
                
        except Exception as e:
            logger.error(f"Failed to cache entry {entry.key}: {e}")
    
    def remove(self, key: str) -> bool:
        """Remove cache entry from disk"""
        try:
            file_path = self._get_file_path(key)
            
            # Try both compressed and uncompressed
            for path in [file_path, file_path.with_suffix(file_path.suffix + '.gz')]:
                if path.exists():
                    path.unlink()
            
            # Remove from metadata
            self.metadata.pop(key, None)
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove cache entry {key}: {e}")
            return False
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        # Create safe filename from key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"
    
    def _load_metadata(self):
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _cleanup_expired_entries(self):
        """Remove expired entries"""
        expired_keys = []
        
        for key, meta in self.metadata.items():
            if meta.get('expires_at'):
                expires_at = datetime.fromisoformat(meta['expires_at'])
                if expires_at < datetime.now():
                    expired_keys.append(key)
        
        for key in expired_keys:
            self.remove(key)
            
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _ensure_space(self, required_bytes: int):
        """Ensure sufficient disk space"""
        current_size = self._calculate_current_size()
        
        if current_size + required_bytes > self.max_size_bytes:
            # Sort by last access time (LRU)
            entries_by_access = sorted(
                self.metadata.items(),
                key=lambda x: x[1].get('last_accessed', '1970-01-01')
            )
            
            # Remove entries until we have space
            for key, meta in entries_by_access:
                if current_size + required_bytes <= self.max_size_bytes:
                    break
                
                entry_size = meta.get('size_bytes', 0)
                self.remove(key)
                current_size -= entry_size
                
                logger.debug(f"Evicted disk cache entry: {key}")
    
    def _calculate_current_size(self) -> int:
        """Calculate current cache size"""
        return sum(meta.get('size_bytes', 0) for meta in self.metadata.values())
    
    def clear(self):
        """Clear all entries"""
        for key in list(self.metadata.keys()):
            self.remove(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_size = self._calculate_current_size()
        
        return {
            'entry_count': len(self.metadata),
            'current_size_bytes': current_size,
            'max_size_bytes': self.max_size_bytes,
            'utilization_pct': (current_size / self.max_size_bytes) * 100,
            'total_accesses': sum(meta.get('access_count', 0) for meta in self.metadata.values())
        }

# ============================================
# Main Cache Manager
# ============================================

class FinancialDataCache:
    """
    Intelligent caching system for financial data
    
    Features:
    - Multi-level caching (memory + disk)
    - Symbol and date-based invalidation
    - Automatic cache warming
    - Dependency tracking
    - Cache analytics
    """
    
    def __init__(self, 
                 cache_dir: Optional[Union[str, Path]] = None,
                 memory_cache_size: int = 512 * 1024 * 1024,    # 512MB
                 disk_cache_size: int = 5 * 1024 * 1024 * 1024, # 5GB
                 default_ttl: int = 3600):  # 1 hour
        """
        Initialize financial data cache
        
        Args:
            cache_dir: Directory for disk cache
            memory_cache_size: Memory cache size in bytes
            disk_cache_size: Disk cache size in bytes
            default_ttl: Default time-to-live in seconds
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.stockpred_cache'
        
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        
        # Initialize cache backends
        self.memory_cache = MemoryCache(max_size_bytes=memory_cache_size)
        self.disk_cache = DiskCache(
            cache_dir=self.cache_dir,
            max_size_bytes=disk_cache_size,
            compression=True
        )
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'puts': 0,
            'evictions': 0,
            'start_time': datetime.now()
        }
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Financial data cache initialized at {self.cache_dir}")
    
    def _create_cache_key(self, symbol: str, data_type: str, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         **kwargs) -> str:
        """Create cache key for financial data"""
        
        key_parts = [symbol.upper(), data_type]
        
        if start_date:
            key_parts.append(start_date.strftime('%Y%m%d'))
        
        if end_date:
            key_parts.append(end_date.strftime('%Y%m%d'))
        
        # Add other parameters
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}:{v}")
        
        return "_".join(key_parts)
    
    @time_it("cache_get")
    def get(self, symbol: str, data_type: str,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            **kwargs) -> Optional[Any]:
        """
        Get cached data
        
        Args:
            symbol: Stock symbol
            data_type: Type of data (e.g., 'stock_data', 'fundamentals')
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional parameters
            
        Returns:
            Cached data or None
        """
        cache_key = self._create_cache_key(symbol, data_type, start_date, end_date, **kwargs)
        
        with self.lock:
            # Try memory cache first
            entry = self.memory_cache.get(cache_key)
            
            if entry:
                self.stats['hits'] += 1
                logger.debug(f"Memory cache hit: {cache_key}")
                return entry.data
            
            # Try disk cache
            entry = self.disk_cache.get(cache_key)
            
            if entry:
                self.stats['hits'] += 1
                logger.debug(f"Disk cache hit: {cache_key}")
                
                # Promote to memory cache
                self.memory_cache.put(entry)
                
                return entry.data
            
            # Cache miss
            self.stats['misses'] += 1
            logger.debug(f"Cache miss: {cache_key}")
            return None
    
    @time_it("cache_put")
    def put(self, data: Any, symbol: str, data_type: str,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            ttl: Optional[int] = None,
            dependencies: Optional[List[str]] = None,
            **kwargs):
        """
        Put data in cache
        
        Args:
            data: Data to cache
            symbol: Stock symbol
            data_type: Type of data
            start_date: Start date for data
            end_date: End date for data
            ttl: Time-to-live in seconds
            dependencies: List of dependencies
            **kwargs: Additional parameters
        """
        if data is None:
            return
        
        cache_key = self._create_cache_key(symbol, data_type, start_date, end_date, **kwargs)
        ttl = ttl or self.default_ttl
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            data=data,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=ttl),
            metadata={
                'symbol': symbol,
                'data_type': data_type,
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
                'parameters': kwargs
            },
            dependencies=dependencies or []
        )
        
        with self.lock:
            # Put in both caches
            self.memory_cache.put(entry)
            self.disk_cache.put(entry)
            
            self.stats['puts'] += 1
            logger.debug(f"Cached: {cache_key} (TTL: {ttl}s, Size: {entry.size_bytes} bytes)")
    
    def invalidate(self, symbol: Optional[str] = None, 
                  data_type: Optional[str] = None,
                  pattern: Optional[str] = None):
        """
        Invalidate cached data
        
        Args:
            symbol: Symbol to invalidate (None for all)
            data_type: Data type to invalidate (None for all)
            pattern: Key pattern to match
        """
        with self.lock:
            keys_to_remove = []
            
            # Find keys to remove
            all_keys = set(self.memory_cache.entries.keys()) | set(self.disk_cache.metadata.keys())
            
            for key in all_keys:
                should_remove = False
                
                if pattern and pattern in key:
                    should_remove = True
                elif symbol and symbol.upper() in key.upper():
                    if data_type is None or data_type in key:
                        should_remove = True
                elif symbol is None and data_type and data_type in key:
                    should_remove = True
                
                if should_remove:
                    keys_to_remove.append(key)
            
            # Remove keys
            for key in keys_to_remove:
                self.memory_cache.remove(key)
                self.disk_cache.remove(key)
            
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
    
    def warm_cache(self, symbols: List[str], data_types: List[str],
                   date_range: Tuple[datetime, datetime],
                   fetcher_func: Callable,
                   max_workers: int = 4):
        """
        Warm cache with data
        
        Args:
            symbols: List of symbols to cache
            data_types: List of data types to cache
            date_range: Date range for data
            fetcher_func: Function to fetch data
            max_workers: Number of worker threads
        """
        logger.info(f"Warming cache for {len(symbols)} symbols and {len(data_types)} data types")
        
        start_date, end_date = date_range
        
        def fetch_and_cache(symbol_datatype_pair):
            symbol, data_type = symbol_datatype_pair
            
            try:
                # Check if already cached
                cached_data = self.get(symbol, data_type, start_date, end_date)
                if cached_data is not None:
                    return f"{symbol}:{data_type} (cached)"
                
                # Fetch data
                data = fetcher_func(symbol, data_type, start_date, end_date)
                
                if data is not None:
                    # Cache the data
                    self.put(data, symbol, data_type, start_date, end_date)
                    return f"{symbol}:{data_type} (fetched)"
                else:
                    return f"{symbol}:{data_type} (no data)"
                    
            except Exception as e:
                logger.error(f"Failed to warm cache for {symbol}:{data_type}: {e}")
                return f"{symbol}:{data_type} (error)"
        
        # Create all combinations
        combinations = [(s, dt) for s in symbols for dt in data_types]
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_and_cache, combo) for combo in combinations]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        logger.info(f"Cache warming complete: {len(results)} operations")
        return results
    
    def cleanup_expired(self):
        """Clean up expired cache entries"""
        with self.lock:
            # Memory cache cleanup (automatic via is_valid check)
            memory_removed = 0
            for key in list(self.memory_cache.entries.keys()):
                entry = self.memory_cache.entries[key]
                if entry.is_expired():
                    self.memory_cache.remove(key)
                    memory_removed += 1
            
            # Disk cache cleanup
            self.disk_cache._cleanup_expired_entries()
            
            logger.info(f"Cleaned up {memory_removed} expired memory cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self.lock:
            memory_stats = self.memory_cache.get_stats()
            disk_stats = self.disk_cache.get_stats()
            
            total_hits = self.stats['hits']
            total_requests = total_hits + self.stats['misses']
            hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'hit_rate_pct': hit_rate,
                'total_hits': total_hits,
                'total_misses': self.stats['misses'],
                'total_puts': self.stats['puts'],
                'uptime_seconds': (datetime.now() - self.stats['start_time']).total_seconds(),
                'memory_cache': memory_stats,
                'disk_cache': disk_stats,
                'total_entries': memory_stats['entry_count'] + disk_stats['entry_count'],
                'total_size_bytes': memory_stats['current_size_bytes'] + disk_stats['current_size_bytes']
            }
    
    def create_performance_report(self) -> str:
        """Create detailed performance report"""
        stats = self.get_stats()
        
        lines = [
            "\nFinancial Data Cache Performance Report",
            "=" * 45,
            "",
            f"Hit Rate: {stats['hit_rate_pct']:.1f}%",
            f"Total Requests: {stats['total_hits'] + stats['total_misses']:,}",
            f"Cache Hits: {stats['total_hits']:,}",
            f"Cache Misses: {stats['total_misses']:,}",
            f"Total Entries: {stats['total_entries']:,}",
            f"Total Size: {stats['total_size_bytes'] / (1024*1024):.1f} MB",
            f"Uptime: {format_duration(stats['uptime_seconds'])}",
            "",
            "Memory Cache:",
            f"  Entries: {stats['memory_cache']['entry_count']:,}",
            f"  Size: {stats['memory_cache']['current_size_bytes'] / (1024*1024):.1f} MB",
            f"  Utilization: {stats['memory_cache']['utilization_pct']:.1f}%",
            "",
            "Disk Cache:",
            f"  Entries: {stats['disk_cache']['entry_count']:,}",
            f"  Size: {stats['disk_cache']['current_size_bytes'] / (1024*1024):.1f} MB",
            f"  Utilization: {stats['disk_cache']['utilization_pct']:.1f}%",
            ""
        ]
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear all caches"""
        with self.lock:
            self.memory_cache.clear()
            self.disk_cache.clear()
            
            # Reset stats
            self.stats = {
                'hits': 0,
                'misses': 0,
                'puts': 0,
                'evictions': 0,
                'start_time': datetime.now()
            }
            
            logger.info("All caches cleared")

# ============================================
# Cache Decorators
# ============================================

class cached_data:
    """
    Decorator for caching function results
    
    Usage:
        @cached_data(ttl=3600, cache_instance=my_cache)
        def fetch_stock_data(symbol, start_date, end_date):
            # Expensive data fetching operation
            return data
    """
    
    def __init__(self, 
                 ttl: int = 3600,
                 cache_instance: Optional[FinancialDataCache] = None,
                 key_func: Optional[Callable] = None):
        """
        Initialize cache decorator
        
        Args:
            ttl: Time-to-live in seconds
            cache_instance: Cache instance to use
            key_func: Function to generate cache key
        """
        self.ttl = ttl
        self.cache = cache_instance or default_cache
        self.key_func = key_func
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if self.key_func:
                cache_key = self.key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = None
            if hasattr(self.cache, 'memory_cache'):
                # It's a FinancialDataCache
                entry = self.cache.memory_cache.get(cache_key)
                if entry:
                    cached_result = entry.data
            
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                if hasattr(self.cache, 'memory_cache'):
                    # Create cache entry
                    entry = CacheEntry(
                        key=cache_key,
                        data=result,
                        created_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(seconds=self.ttl)
                    )
                    self.cache.memory_cache.put(entry)
                
                logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper

# ============================================
# Global Cache Instance
# ============================================

# Create default cache instance
default_cache = FinancialDataCache()

def get_cache() -> FinancialDataCache:
    """Get default cache instance"""
    return default_cache

def clear_cache():
    """Clear default cache"""
    default_cache.clear()

def cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return default_cache.get_stats()

def cache_report() -> str:
    """Get cache performance report"""
    return default_cache.create_performance_report()

# ============================================
# Utility Functions
# ============================================

def create_symbol_cache_key(symbol: str, data_type: str, 
                           params: Dict[str, Any]) -> str:
    """Create standardized cache key for symbol data"""
    key_parts = [symbol.upper(), data_type.lower()]
    
    # Add sorted parameters
    for key, value in sorted(params.items()):
        if value is not None:
            if isinstance(value, datetime):
                key_parts.append(f"{key}:{value.strftime('%Y%m%d')}")
            else:
                key_parts.append(f"{key}:{value}")
    
    return "_".join(key_parts)

def cache_data_with_deps(cache: FinancialDataCache, data: Any,
                        symbol: str, data_type: str,
                        dependencies: List[str],
                        ttl: int = 3600):
    """Cache data with dependency tracking"""
    cache.put(
        data=data,
        symbol=symbol,
        data_type=data_type,
        ttl=ttl,
        dependencies=dependencies
    )

def invalidate_symbol_cache(cache: FinancialDataCache, symbol: str):
    """Invalidate all cache entries for a symbol"""
    cache.invalidate(symbol=symbol)
    logger.info(f"Invalidated all cache entries for {symbol}")

def optimize_cache_performance(cache: FinancialDataCache):
    """Optimize cache performance by cleaning up expired entries"""
    cache.cleanup_expired()
    
    stats = cache.get_stats()
    logger.info(f"Cache optimization complete. Hit rate: {stats['hit_rate_pct']:.1f}%")
