# ============================================
# StockPredictionPro - src/utils/helpers.py
# Common utility functions and helper methods
# ============================================

import os
import re
import hashlib
import functools
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .exceptions import InvalidParameterError, BusinessLogicError
from .logger import get_logger

logger = get_logger('helpers')

# ============================================
# String and Text Utilities
# ============================================

def clean_symbol(symbol: str) -> str:
    """
    Clean and standardize stock symbol
    
    Args:
        symbol: Raw stock symbol
        
    Returns:
        Cleaned and standardized symbol
    """
    if not symbol or not isinstance(symbol, str):
        raise InvalidParameterError(
            "Symbol must be a non-empty string",
            parameter_name="symbol",
            provided_value=symbol
        )
    
    # Remove whitespace and convert to uppercase
    symbol = symbol.strip().upper()
    
    # Remove invalid characters
    symbol = re.sub(r'[^A-Z0-9.-]', '', symbol)
    
    # Handle common symbol formats
    symbol_mappings = {
        'GOOGL': 'GOOGL',
        'GOOGLE': 'GOOGL',
        'ALPHABET': 'GOOGL',
        'TSLA': 'TSLA',
        'TESLA': 'TSLA',
        'MSFT': 'MSFT',
        'MICROSOFT': 'MSFT',
        'AAPL': 'AAPL',
        'APPLE': 'AAPL'
    }
    
    return symbol_mappings.get(symbol, symbol)

def validate_symbols(symbols: Union[str, List[str]]) -> List[str]:
    """
    Validate and clean list of stock symbols
    
    Args:
        symbols: Single symbol or list of symbols
        
    Returns:
        List of validated symbols
    """
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(',') if s.strip()]
    
    if not symbols:
        raise InvalidParameterError(
            "At least one symbol must be provided",
            parameter_name="symbols",
            provided_value=symbols
        )
    
    cleaned_symbols = []
    for symbol in symbols:
        try:
            cleaned = clean_symbol(symbol)
            if cleaned and cleaned not in cleaned_symbols:
                cleaned_symbols.append(cleaned)
        except Exception as e:
            logger.warning(f"Invalid symbol '{symbol}': {e}")
    
    if not cleaned_symbols:
        raise InvalidParameterError(
            "No valid symbols found",
            parameter_name="symbols",
            provided_value=symbols
        )
    
    return cleaned_symbols

def generate_run_id(prefix: str = "run") -> str:
    """
    Generate unique run ID for tracking operations
    
    Args:
        prefix: Prefix for the run ID
        
    Returns:
        Unique run ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_id}"

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system operations
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    return filename.strip('.-')

# ============================================
# Date and Time Utilities
# ============================================

def parse_date(date_input: Union[str, datetime, pd.Timestamp]) -> datetime:
    """
    Parse various date formats into datetime object
    
    Args:
        date_input: Date in various formats
        
    Returns:
        Parsed datetime object
    """
    if isinstance(date_input, datetime):
        return date_input
    elif isinstance(date_input, pd.Timestamp):
        return date_input.to_pydatetime()
    elif isinstance(date_input, str):
        # Common date formats
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_input, fmt)
            except ValueError:
                continue
        
        # Try pandas parsing as fallback
        try:
            return pd.to_datetime(date_input).to_pydatetime()
        except Exception:
            pass
    
    raise InvalidParameterError(
        f"Unable to parse date: {date_input}",
        parameter_name="date_input",
        provided_value=date_input
    )

def validate_date_range(start_date: Any, end_date: Any) -> Tuple[datetime, datetime]:
    """
    Validate and parse date range
    
    Args:
        start_date: Start date in various formats
        end_date: End date in various formats
        
    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    
    if start_dt >= end_dt:
        raise InvalidParameterError(
            "Start date must be before end date",
            parameter_name="date_range",
            provided_value=f"{start_date} to {end_date}"
        )
    
    # Check if date range is reasonable
    max_range = timedelta(days=365 * 10)  # 10 years max
    if end_dt - start_dt > max_range:
        raise InvalidParameterError(
            "Date range is too large (maximum 10 years)",
            parameter_name="date_range",
            provided_value=f"{start_date} to {end_date}"
        )
    
    return start_dt, end_dt

def get_trading_days(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate number of trading days between dates (approximate)
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Approximate number of trading days
    """
    total_days = (end_date - start_date).days
    # Approximate: remove weekends (roughly 2/7 of days)
    trading_days = int(total_days * 5/7)
    return max(1, trading_days)

def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

# ============================================
# Financial Utilities
# ============================================

def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """
    Calculate returns from price series
    
    Args:
        prices: Price series
        method: Return calculation method ('simple', 'log')
        
    Returns:
        Returns series
    """
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise InvalidParameterError(
            f"Invalid return method: {method}",
            parameter_name="method",
            provided_value=method
        )
    
    return returns.fillna(0)

def annualize_returns(returns: Union[float, pd.Series], periods_per_year: int = 252) -> Union[float, pd.Series]:
    """
    Annualize returns
    
    Args:
        returns: Returns (single value or series)
        periods_per_year: Number of periods per year (252 for daily)
        
    Returns:
        Annualized returns
    """
    if isinstance(returns, pd.Series):
        return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
    else:
        return (1 + returns) ** periods_per_year - 1

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    
    if excess_returns.std() == 0:
        return 0.0
    
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        returns: Returns series
        
    Returns:
        Maximum drawdown as positive value
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return abs(drawdown.min())

def format_currency(value: float, currency: str = "INR") -> str:
    """
    Format currency values
    
    Args:
        value: Numeric value
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if currency == "INR":
        if abs(value) >= 10000000:  # 1 crore
            return f"₹{value/10000000:.1f}Cr"
        elif abs(value) >= 100000:  # 1 lakh
            return f"₹{value/100000:.1f}L"
        elif abs(value) >= 1000:
            return f"₹{value/1000:.1f}K"
        else:
            return f"₹{value:.0f}"
    elif currency == "USD":
        if abs(value) >= 1000000:
            return f"${value/1000000:.1f}M"
        elif abs(value) >= 1000:
            return f"${value/1000:.1f}K"
        else:
            return f"${value:.2f}"
    else:
        return f"{value:.2f} {currency}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage values
    
    Args:
        value: Decimal value (0.05 = 5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value*100:.{decimals}f}%"

# ============================================
# Data Utilities
# ============================================

def safe_divide(numerator: Union[float, pd.Series], denominator: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    Safe division that handles division by zero
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        
    Returns:
        Division result with zeros where denominator is zero
    """
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        # Handle pandas Series
        result = numerator / denominator
        return result.fillna(0)
    else:
        # Handle scalar values
        if denominator == 0:
            return 0.0
        return numerator / denominator

def remove_outliers(data: pd.Series, method: str = "iqr", threshold: float = 1.5) -> pd.Series:
    """
    Remove outliers from data series
    
    Args:
        data: Data series
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Data series with outliers removed
    """
    if method == "iqr":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return data[(data >= lower_bound) & (data <= upper_bound)]
    
    elif method == "zscore":
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[z_scores <= threshold]
    
    else:
        raise InvalidParameterError(
            f"Invalid outlier detection method: {method}",
            parameter_name="method",
            provided_value=method
        )

def interpolate_missing_data(data: pd.Series, method: str = "linear", limit: int = 5) -> pd.Series:
    """
    Interpolate missing data
    
    Args:
        data: Data series with missing values
        method: Interpolation method ('linear', 'forward', 'backward')
        limit: Maximum number of consecutive missing values to interpolate
        
    Returns:
        Series with interpolated values
    """
    if method == "linear":
        return data.interpolate(method='linear', limit=limit)
    elif method == "forward":
        return data.fillna(method='ffill', limit=limit)
    elif method == "backward":
        return data.fillna(method='bfill', limit=limit)
    else:
        raise InvalidParameterError(
            f"Invalid interpolation method: {method}",
            parameter_name="method",
            provided_value=method
        )

def normalize_data(data: pd.Series, method: str = "minmax") -> pd.Series:
    """
    Normalize data series
    
    Args:
        data: Data series to normalize
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized data series
    """
    if method == "minmax":
        return (data - data.min()) / (data.max() - data.min())
    elif method == "zscore":
        return (data - data.mean()) / data.std()
    elif method == "robust":
        median = data.median()
        mad = (data - median).abs().median()
        return (data - median) / mad
    else:
        raise InvalidParameterError(
            f"Invalid normalization method: {method}",
            parameter_name="method",
            provided_value=method
        )

# ============================================
# File and Path Utilities
# ============================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Calculate MD5 hash of file
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_file_size(file_path: Union[str, Path]) -> str:
    """
    Get human-readable file size
    
    Args:
        file_path: Path to file
        
    Returns:
        Formatted file size string
    """
    size_bytes = Path(file_path).stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f}PB"

def backup_file(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create backup of file with timestamp
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory to store backup (default: same directory)
        
    Returns:
        Path to backup file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if backup_dir is None:
        backup_dir = file_path.parent
    else:
        backup_dir = ensure_directory(backup_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    import shutil
    shutil.copy2(file_path, backup_path)
    
    return backup_path

# ============================================
# Performance Utilities
# ============================================

def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.debug(f"{func.__name__} executed in {format_duration(duration)}")
        
        return result
    
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0, exponential_backoff: bool = True):
    """
    Decorator to retry function on failure
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        exponential_backoff: Use exponential backoff
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2 ** attempt) if exponential_backoff else delay
                        logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            
            raise last_exception
        
        return wrapper
    
    return decorator

def parallel_execute(func: Callable, items: List[Any], max_workers: int = 4) -> List[Any]:
    """
    Execute function in parallel for list of items
    
    Args:
        func: Function to execute
        items: List of items to process
        max_workers: Maximum number of worker threads
        
    Returns:
        List of results
    """
    results = [None] * len(items)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(func, item): i 
            for i, item in enumerate(items)
        }
        
        # Collect results
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                logger.error(f"Parallel execution failed for item {index}: {e}")
                results[index] = None
    
    return results

# ============================================
# Caching Utilities
# ============================================

def memory_cache(maxsize: int = 128, ttl: int = 3600):
    """
    Simple memory cache decorator with TTL
    
    Args:
        maxsize: Maximum cache size
        ttl: Time to live in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if result is in cache and not expired
            if key in cache and current_time - cache_times[key] < ttl:
                return cache[key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Manage cache size
            if len(cache) >= maxsize:
                # Remove oldest entry
                oldest_key = min(cache_times.keys(), key=cache_times.get)
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            cache[key] = result
            cache_times[key] = current_time
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear() or cache_times.clear()
        wrapper.cache_info = lambda: {
            'size': len(cache),
            'maxsize': maxsize,
            'ttl': ttl
        }
        
        return wrapper
    
    return decorator

# ============================================
# Validation Utilities
# ============================================

def validate_numeric_range(value: Union[int, float], min_val: float, max_val: float, param_name: str) -> Union[int, float]:
    """
    Validate numeric value is within range
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        param_name: Parameter name for error messages
        
    Returns:
        Validated value
    """
    if not isinstance(value, (int, float)):
        raise InvalidParameterError(
            f"{param_name} must be numeric",
            parameter_name=param_name,
            provided_value=value
        )
    
    if value < min_val or value > max_val:
        raise InvalidParameterError(
            f"{param_name} must be between {min_val} and {max_val}",
            parameter_name=param_name,
            provided_value=value
        )
    
    return value

def validate_choice(value: Any, choices: List[Any], param_name: str) -> Any:
    """
    Validate value is in allowed choices
    
    Args:
        value: Value to validate
        choices: List of allowed choices
        param_name: Parameter name for error messages
        
    Returns:
        Validated value
    """
    if value not in choices:
        raise InvalidParameterError(
            f"{param_name} must be one of {choices}",
            parameter_name=param_name,
            provided_value=value
        )
    
    return value

# ============================================
# Utility Functions
# ============================================

def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (overwrites dict1)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'disk_free_gb': psutil.disk_usage('.').free / (1024**3),
        'current_time': datetime.now().isoformat(),
        'timezone': str(datetime.now().astimezone().tzinfo)
    }

def create_summary_table(data: Dict[str, Any], title: str = "Summary") -> str:
    """
    Create formatted summary table for display
    
    Args:
        data: Data dictionary
        title: Table title
        
    Returns:
        Formatted table string
    """
    lines = [
        f"\n{title}",
        "=" * len(title),
        ""
    ]
    
    max_key_len = max(len(str(k)) for k in data.keys()) if data else 0
    
    for key, value in data.items():
        if isinstance(value, float):
            if abs(value) < 0.01:
                value_str = f"{value:.4f}"
            else:
                value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        
        lines.append(f"{str(key):<{max_key_len}} : {value_str}")
    
    lines.append("")
    return "\n".join(lines)
