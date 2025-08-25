"""
notebook-utils.py

Comprehensive utility functions for Jupyter notebooks in StockPredictionPro.
Provides logging, file handling, timing, data processing, model helpers, and more.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import time
import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple

import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging for notebooks
def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration for notebooks"""
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True
    )
    return logging.getLogger('StockPredictionPro')

# Initialize logger
logger = setup_logging()

# ============================================
# FILE AND DIRECTORY UTILITIES
# ============================================

def ensure_dir_exists(dir_path: str) -> bool:
    """
    Ensure directory exists, create if missing
    
    Args:
        dir_path: Path to directory
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Directory ready: {dir_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create directory {dir_path}: {e}")
        return False

def get_project_root() -> Path:
    """Get the project root directory"""
    current_path = Path.cwd()
    while current_path != current_path.parent:
        if (current_path / 'notebooks').exists():
            return current_path
        current_path = current_path.parent
    return Path.cwd()

def safe_file_exists(filepath: str) -> bool:
    """Safely check if file exists with logging"""
    exists = os.path.exists(filepath)
    if exists:
        logger.debug(f"📁 File exists: {filepath}")
    else:
        logger.warning(f"📁 File not found: {filepath}")
    return exists

# ============================================
# DATA PERSISTENCE UTILITIES
# ============================================

def save_pickle(obj: Any, filepath: str) -> bool:
    """
    Save Python object to pickle file safely
    
    Args:
        obj: Python object to save
        filepath: Path to save file
        
    Returns:
        bool: True if saved successfully
    """
    try:
        ensure_dir_exists(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"💾 Saved pickle: {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save pickle {filepath}: {e}")
        return False

def load_pickle(filepath: str) -> Optional[Any]:
    """
    Load Python object from pickle file safely
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object or None if failed
    """
    try:
        if not safe_file_exists(filepath):
            return None
        
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"📂 Loaded pickle: {filepath}")
        return obj
    except Exception as e:
        logger.error(f"❌ Failed to load pickle {filepath}: {e}")
        return None

def save_csv(df: pd.DataFrame, filepath: str, **kwargs) -> bool:
    """
    Save pandas DataFrame to CSV safely
    
    Args:
        df: DataFrame to save
        filepath: Path to save file
        **kwargs: Additional arguments for to_csv
        
    Returns:
        bool: True if saved successfully
    """
    try:
        ensure_dir_exists(os.path.dirname(filepath))
        df.to_csv(filepath, **kwargs)
        logger.info(f"📊 Saved CSV: {filepath} (shape: {df.shape})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save CSV {filepath}: {e}")
        return False

def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load CSV file with comprehensive error handling
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments for read_csv
        
    Returns:
        DataFrame (empty if failed)
    """
    try:
        if not safe_file_exists(filepath):
            return pd.DataFrame()
        
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"📈 Loaded CSV: {filepath} (shape: {df.shape})")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load CSV {filepath}: {e}")
        return pd.DataFrame()

def save_json(data: Dict, filepath: str, indent: int = 4) -> bool:
    """
    Save dictionary to JSON file safely
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
        indent: JSON indentation
        
    Returns:
        bool: True if saved successfully
    """
    try:
        ensure_dir_exists(os.path.dirname(filepath))
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        logger.info(f"🔧 Saved JSON: {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save JSON {filepath}: {e}")
        return False

def load_json(filepath: str) -> Optional[Dict]:
    """
    Load JSON file safely
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary or None if failed
    """
    try:
        if not safe_file_exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"⚙️ Loaded JSON: {filepath}")
        return data
    except Exception as e:
        logger.error(f"❌ Failed to load JSON {filepath}: {e}")
        return None

# ============================================
# TIMING AND PERFORMANCE UTILITIES
# ============================================

class Timer:
    """Context manager and manual timer for performance measurement"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = 0

    def start(self) -> None:
        """Start timing"""
        self.start_time = time.perf_counter()
        logger.debug(f"⏱️ Started timer: {self.name}")

    def stop(self) -> float:
        """Stop timing and return elapsed time"""
        if self.start_time is None:
            logger.warning(f"⚠️ Timer '{self.name}' stopped before started")
            return 0
        
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        logger.info(f"⏱️ {self.name} completed in {self.elapsed:.4f} seconds")
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

def benchmark_function(func, *args, iterations: int = 1, **kwargs) -> Dict:
    """
    Benchmark a function with multiple iterations
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        iterations: Number of iterations
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with timing statistics
    """
    times = []
    results = []
    
    for i in range(iterations):
        with Timer() as timer:
            result = func(*args, **kwargs)
            results.append(result)
        times.append(timer.elapsed)
    
    stats = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times),
        'iterations': iterations
    }
    
    logger.info(f"🎯 Benchmark results: {stats}")
    return stats

# ============================================
# DATA PROCESSING UTILITIES
# ============================================

def safe_apply(df: pd.DataFrame, func, axis: int = 0, *args, **kwargs) -> pd.DataFrame:
    """
    Apply function safely with error handling
    
    Args:
        df: DataFrame to process
        func: Function to apply
        axis: Axis to apply function along
        *args, **kwargs: Additional arguments
        
    Returns:
        Processed DataFrame or original if failed
    """
    try:
        result = df.apply(func, axis=axis, *args, **kwargs)
        logger.debug(f"✅ Applied function {func.__name__} successfully")
        return result
    except Exception as e:
        logger.error(f"❌ Error applying function {func.__name__}: {e}")
        return df

def memory_usage_mb() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 ** 2)
        return memory_mb
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        return 0.0

def log_memory_usage(context: str = "Current") -> None:
    """Log current memory usage"""
    memory_mb = memory_usage_mb()
    logger.info(f"💾 {context} memory usage: {memory_mb:.2f} MB")

def dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> Dict:
    """
    Get comprehensive DataFrame information
    
    Args:
        df: DataFrame to analyze
        name: Name for logging
        
    Returns:
        Dictionary with DataFrame statistics
    """
    info = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 ** 2),
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    
    logger.info(f"📊 {name} Info: Shape {info['shape']}, Memory {info['memory_usage_mb']:.2f} MB")
    return info

def clean_dataframe(df: pd.DataFrame, 
                   drop_duplicates: bool = True,
                   fill_numeric_na: Union[str, float] = 'median',
                   fill_categorical_na: str = 'mode') -> pd.DataFrame:
    """
    Clean DataFrame with common preprocessing steps
    
    Args:
        df: DataFrame to clean
        drop_duplicates: Whether to drop duplicate rows
        fill_numeric_na: How to fill numeric NAs ('median', 'mean', or value)
        fill_categorical_na: How to fill categorical NAs ('mode' or value)
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    original_shape = df_clean.shape
    
    # Drop duplicates
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
        logger.info(f"🧹 Removed {original_shape[0] - df_clean.shape[0]} duplicate rows")
    
    # Handle numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            if fill_numeric_na == 'median':
                fill_value = df_clean[col].median()
            elif fill_numeric_na == 'mean':
                fill_value = df_clean[col].mean()
            else:
                fill_value = fill_numeric_na
            
            df_clean[col].fillna(fill_value, inplace=True)
    
    # Handle categorical columns
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            if fill_categorical_na == 'mode':
                fill_value = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
            else:
                fill_value = fill_categorical_na
            
            df_clean[col].fillna(fill_value, inplace=True)
    
    logger.info(f"🧹 DataFrame cleaned: {original_shape} → {df_clean.shape}")
    return df_clean

# ============================================
# MODEL UTILITIES
# ============================================

def count_parameters(params: Dict) -> int:
    """
    Count total parameters in JAX/Flax model
    
    Args:
        params: Parameters dictionary
        
    Returns:
        Total parameter count
    """
    total = 0
    try:
        def _count_recursive(param_dict):
            count = 0
            for key, value in param_dict.items():
                if isinstance(value, dict):
                    count += _count_recursive(value)
                elif hasattr(value, 'shape'):
                    count += np.prod(value.shape)
            return count
        
        total = _count_recursive(params)
        logger.info(f"🧠 Total parameters: {total:,}")
    except Exception as e:
        logger.error(f"❌ Error counting parameters: {e}")
    
    return total

def model_summary(model_name: str, params: Dict, metrics: Dict = None) -> Dict:
    """
    Create comprehensive model summary
    
    Args:
        model_name: Name of the model
        params: Model parameters
        metrics: Performance metrics dictionary
        
    Returns:
        Model summary dictionary
    """
    summary = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'parameter_count': count_parameters(params),
        'metrics': metrics or {}
    }
    
    logger.info(f"📋 Model summary created for {model_name}")
    return summary

# ============================================
# DISPLAY AND FORMATTING UTILITIES
# ============================================

def print_section(title: str, level: int = 1, char: str = '=') -> None:
    """
    Print formatted section header
    
    Args:
        title: Section title
        level: Header level (1-6)
        char: Character for decoration
    """
    width = min(80, len(title) + 20)
    if level == 1:
        print(f"\n{char * width}")
        print(f"{title.upper().center(width)}")
        print(f"{char * width}\n")
    elif level == 2:
        print(f"\n{char * (len(title) + 10)}")
        print(f"  {title.upper()}  ")
        print(f"{char * (len(title) + 10)}\n")
    else:
        prefix = '#' * level
        print(f"\n{prefix} {title}\n")

def format_number(num: float, precision: int = 2, as_percentage: bool = False) -> str:
    """
    Format number for display
    
    Args:
        num: Number to format
        precision: Decimal places
        as_percentage: Whether to format as percentage
        
    Returns:
        Formatted string
    """
    if as_percentage:
        return f"{num * 100:.{precision}f}%"
    
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.{precision}f}M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

def display_dataframe_sample(df: pd.DataFrame, name: str = "DataFrame", 
                           n_rows: int = 5) -> None:
    """
    Display formatted DataFrame sample with info
    
    Args:
        df: DataFrame to display
        name: Name for display
        n_rows: Number of rows to show
    """
    print_section(f"{name} Sample", level=3)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if not df.empty:
        print(f"\nFirst {n_rows} rows:")
        print(df.head(n_rows).to_string())
        
        print(f"\nColumn info:")
        print(f"Numeric: {len(df.select_dtypes(include=[np.number]).columns)}")
        print(f"Categorical: {len(df.select_dtypes(include=['object', 'category']).columns)}")
        print(f"Null values: {df.isnull().sum().sum()}")
    else:
        print("⚠️ DataFrame is empty")

# ============================================
# ERROR HANDLING AND VALIDATION
# ============================================

def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, 
                      required_columns: List[str] = None) -> bool:
    """
    Validate DataFrame meets requirements
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum required rows
        required_columns: List of required column names
        
    Returns:
        bool: True if valid
    """
    if df.empty:
        logger.error("❌ DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        logger.error(f"❌ DataFrame has {len(df)} rows, minimum {min_rows} required")
        return False
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return False
    
    logger.info("✅ DataFrame validation passed")
    return True

class NotebookError(Exception):
    """Custom exception for notebook operations"""
    pass

def handle_exception(func):
    """Decorator for graceful exception handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"❌ Exception in {func.__name__}: {e}")
            raise NotebookError(f"Error in {func.__name__}: {e}") from e
    return wrapper

# ============================================
# CONFIGURATION MANAGEMENT
# ============================================

class NotebookConfig:
    """Configuration management for notebooks"""
    
    def __init__(self, config_path: str = './config/notebook_config.json'):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if safe_file_exists(self.config_path):
            return load_json(self.config_path) or {}
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """Create default configuration"""
        default_config = {
            'data': {
                'raw_data_path': './data/raw/',
                'processed_data_path': './data/processed/',
                'output_path': './outputs/'
            },
            'models': {
                'random_seed': 42,
                'test_size': 0.2,
                'cv_folds': 5
            },
            'plotting': {
                'figure_size': [14, 8],
                'dpi': 100,
                'style': 'darkgrid'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }
        
        save_json(default_config, self.config_path)
        logger.info(f"📝 Created default config: {self.config_path}")
        return default_config
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value) -> None:
        """Update configuration value"""
        keys = key.split('.')
        config_section = self.config
        
        for k in keys[:-1]:
            if k not in config_section:
                config_section[k] = {}
            config_section = config_section[k]
        
        config_section[keys[-1]] = value
        save_json(self.config, self.config_path)
        logger.info(f"🔧 Updated config {key} = {value}")

# Initialize global configuration
config = NotebookConfig()

# ============================================
# INITIALIZATION AND CLEANUP
# ============================================

def setup_notebook_environment(suppress_warnings: bool = True) -> None:
    """
    Setup optimal notebook environment
    
    Args:
        suppress_warnings: Whether to suppress warnings
    """
    if suppress_warnings:
        warnings.filterwarnings('ignore')
    
    # Set pandas options
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 120)
    pd.set_option('display.precision', 4)
    
    # Set numpy options
    np.set_printoptions(precision=4, suppress=True)
    
    logger.info("🚀 Notebook environment setup completed")

def cleanup_memory() -> None:
    """Clean up memory and garbage collection"""
    import gc
    
    before_mb = memory_usage_mb()
    gc.collect()
    after_mb = memory_usage_mb()
    
    freed_mb = before_mb - after_mb
    logger.info(f"🧹 Memory cleanup: {freed_mb:.2f} MB freed (Now: {after_mb:.2f} MB)")

# ============================================
# MAIN UTILITIES SUMMARY
# ============================================

def get_available_functions() -> List[str]:
    """Get list of all available utility functions"""
    current_module = sys.modules[__name__]
    functions = [name for name, obj in vars(current_module).items() 
                if callable(obj) and not name.startswith('_')]
    return sorted(functions)

def print_utils_help() -> None:
    """Print help information about available utilities"""
    print_section("StockPredictionPro Notebook Utils", level=1)
    
    categories = {
        'File Operations': ['ensure_dir_exists', 'save_csv', 'load_csv', 'save_json', 'load_json', 'save_pickle', 'load_pickle'],
        'Data Processing': ['safe_apply', 'dataframe_info', 'clean_dataframe', 'validate_dataframe'],
        'Performance': ['Timer', 'benchmark_function', 'memory_usage_mb', 'log_memory_usage'],
        'Model Helpers': ['count_parameters', 'model_summary'],
        'Display': ['print_section', 'format_number', 'display_dataframe_sample'],
        'Configuration': ['NotebookConfig', 'setup_notebook_environment']
    }
    
    for category, functions in categories.items():
        print(f"\n{category}:")
        for func in functions:
            print(f"  • {func}")
    
    print(f"\n📚 Total functions available: {len(get_available_functions())}")
    print("Use help(function_name) for detailed documentation")

# Initialize environment on import
setup_notebook_environment()

# Success message
logger.info("✅ StockPredictionPro Notebook Utils loaded successfully")
print("📦 notebook-utils.py ready - Use print_utils_help() for available functions")
