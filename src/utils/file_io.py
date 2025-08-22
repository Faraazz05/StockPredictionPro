# ============================================
# StockPredictionPro - src/utils/file_io.py
# Advanced file I/O operations with data management
# ============================================

import os
import json
import pickle
import joblib
import gzip
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import contextlib

from .exceptions import DataValidationError, BusinessLogicError
from .logger import get_logger
from .helpers import ensure_directory, sanitize_filename, format_duration
from .timing import Timer

logger = get_logger('file_io')

# ============================================
# File Format Handlers
# ============================================

class BaseFileHandler:
    """Base class for file format handlers"""
    
    def __init__(self, compression: bool = False):
        self.compression = compression
        self.supported_extensions: List[str] = []
    
    def can_handle(self, file_path: Union[str, Path]) -> bool:
        """Check if handler can handle the file format"""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions
    
    def save(self, data: Any, file_path: Union[str, Path], **kwargs) -> bool:
        """Save data to file"""
        raise NotImplementedError
    
    def load(self, file_path: Union[str, Path], **kwargs) -> Any:
        """Load data from file"""
        raise NotImplementedError

class CSVHandler(BaseFileHandler):
    """Handler for CSV files"""
    
    def __init__(self, compression: bool = False):
        super().__init__(compression)
        self.supported_extensions = ['.csv']
    
    def save(self, data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> bool:
        """Save DataFrame to CSV"""
        try:
            file_path = Path(file_path)
            ensure_directory(file_path.parent)
            
            # Default CSV options
            csv_options = {
                'index': True,
                'float_format': '%.6f',
                **kwargs
            }
            
            if self.compression:
                csv_options['compression'] = 'gzip'
                if not file_path.suffix.endswith('.gz'):
                    file_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            data.to_csv(file_path, **csv_options)
            logger.debug(f"Saved CSV: {file_path} ({len(data)} rows)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save CSV {file_path}: {e}")
            return False
    
    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load DataFrame from CSV"""
        try:
            file_path = Path(file_path)
            
            # Default CSV options
            csv_options = {
                'index_col': 0,
                'parse_dates': True,
                **kwargs
            }
            
            # Auto-detect compression
            if file_path.suffix.lower().endswith('.gz'):
                csv_options['compression'] = 'gzip'
            
            data = pd.read_csv(file_path, **csv_options)
            logger.debug(f"Loaded CSV: {file_path} ({len(data)} rows)")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load CSV {file_path}: {e}")
            raise DataValidationError(f"Cannot load CSV file: {e}", context={'file_path': str(file_path)})

class ParquetHandler(BaseFileHandler):
    """Handler for Parquet files"""
    
    def __init__(self, compression: bool = True):
        super().__init__(compression)
        self.supported_extensions = ['.parquet', '.pq']
    
    def save(self, data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> bool:
        """Save DataFrame to Parquet"""
        try:
            file_path = Path(file_path)
            ensure_directory(file_path.parent)
            
            # Default Parquet options
            parquet_options = {
                'compression': 'snappy' if self.compression else None,
                'index': True,
                **kwargs
            }
            
            data.to_parquet(file_path, **parquet_options)
            logger.debug(f"Saved Parquet: {file_path} ({len(data)} rows)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save Parquet {file_path}: {e}")
            return False
    
    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load DataFrame from Parquet"""
        try:
            data = pd.read_parquet(file_path, **kwargs)
            logger.debug(f"Loaded Parquet: {file_path} ({len(data)} rows)")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load Parquet {file_path}: {e}")
            raise DataValidationError(f"Cannot load Parquet file: {e}", context={'file_path': str(file_path)})

class JSONHandler(BaseFileHandler):
    """Handler for JSON files"""
    
    def __init__(self, compression: bool = False):
        super().__init__(compression)
        self.supported_extensions = ['.json']
    
    def save(self, data: Any, file_path: Union[str, Path], **kwargs) -> bool:
        """Save data to JSON"""
        try:
            file_path = Path(file_path)
            ensure_directory(file_path.parent)
            
            # Default JSON options
            json_options = {
                'indent': 2,
                'default': self._json_serializer,
                **kwargs
            }
            
            if self.compression:
                file_path = file_path.with_suffix(file_path.suffix + '.gz')
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    json.dump(data, f, **json_options)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, **json_options)
            
            logger.debug(f"Saved JSON: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save JSON {file_path}: {e}")
            return False
    
    def load(self, file_path: Union[str, Path], **kwargs) -> Any:
        """Load data from JSON"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower().endswith('.gz'):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f, **kwargs)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f, **kwargs)
            
            logger.debug(f"Loaded JSON: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSON {file_path}: {e}")
            raise DataValidationError(f"Cannot load JSON file: {e}", context={'file_path': str(file_path)})
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        else:
            return str(obj)

class YAMLHandler(BaseFileHandler):
    """Handler for YAML files"""
    
    def __init__(self, compression: bool = False):
        super().__init__(compression)
        self.supported_extensions = ['.yaml', '.yml']
    
    def save(self, data: Any, file_path: Union[str, Path], **kwargs) -> bool:
        """Save data to YAML"""
        try:
            file_path = Path(file_path)
            ensure_directory(file_path.parent)
            
            # Default YAML options
            yaml_options = {
                'default_flow_style': False,
                'indent': 2,
                'sort_keys': False,
                **kwargs
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, **yaml_options)
            
            logger.debug(f"Saved YAML: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save YAML {file_path}: {e}")
            return False
    
    def load(self, file_path: Union[str, Path], **kwargs) -> Any:
        """Load data from YAML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            logger.debug(f"Loaded YAML: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load YAML {file_path}: {e}")
            raise DataValidationError(f"Cannot load YAML file: {e}", context={'file_path': str(file_path)})

class PickleHandler(BaseFileHandler):
    """Handler for Pickle files (for ML models)"""
    
    def __init__(self, compression: bool = True):
        super().__init__(compression)
        self.supported_extensions = ['.pkl', '.pickle']
    
    def save(self, data: Any, file_path: Union[str, Path], **kwargs) -> bool:
        """Save data to Pickle"""
        try:
            file_path = Path(file_path)
            ensure_directory(file_path.parent)
            
            if self.compression:
                # Use joblib for better compression and speed
                joblib.dump(data, file_path, compress=3)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.debug(f"Saved Pickle: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save Pickle {file_path}: {e}")
            return False
    
    def load(self, file_path: Union[str, Path], **kwargs) -> Any:
        """Load data from Pickle"""
        try:
            # Try joblib first (handles compression automatically)
            try:
                data = joblib.load(file_path)
            except:
                # Fallback to standard pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            logger.debug(f"Loaded Pickle: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load Pickle {file_path}: {e}")
            raise DataValidationError(f"Cannot load Pickle file: {e}", context={'file_path': str(file_path)})

# ============================================
# File Manager
# ============================================

class FileManager:
    """
    Advanced file I/O manager with format detection and data integrity
    
    Features:
    - Automatic format detection
    - Data integrity verification
    - Atomic operations
    - Backup and versioning
    - Batch operations
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None, 
                 enable_compression: bool = True, enable_backup: bool = True):
        """
        Initialize FileManager
        
        Args:
            base_path: Base directory for file operations
            enable_compression: Enable compression for supported formats
            enable_backup: Enable automatic backup of overwritten files
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.enable_compression = enable_compression
        self.enable_backup = enable_backup
        
        # Initialize handlers
        self.handlers = {
            'csv': CSVHandler(compression=enable_compression),
            'parquet': ParquetHandler(compression=enable_compression),
            'json': JSONHandler(compression=enable_compression),
            'yaml': YAMLHandler(compression=enable_compression),
            'pickle': PickleHandler(compression=enable_compression)
        }
        
        # File operation statistics
        self.stats = {
            'files_read': 0,
            'files_written': 0,
            'bytes_read': 0,
            'bytes_written': 0,
            'errors': 0
        }
    
    def get_handler(self, file_path: Union[str, Path]) -> Optional[BaseFileHandler]:
        """Get appropriate handler for file format"""
        file_path = Path(file_path)
        
        for handler in self.handlers.values():
            if handler.can_handle(file_path):
                return handler
        
        return None
    
    def save_data(self, data: Any, file_path: Union[str, Path], 
                  format_hint: Optional[str] = None, 
                  create_backup: Optional[bool] = None,
                  atomic: bool = True, **kwargs) -> bool:
        """
        Save data to file with automatic format detection
        
        Args:
            data: Data to save
            file_path: Target file path
            format_hint: Format hint ('csv', 'parquet', 'json', 'yaml', 'pickle')
            create_backup: Whether to create backup (overrides default)
            atomic: Use atomic write operation
            **kwargs: Additional arguments for format handler
            
        Returns:
            True if successful, False otherwise
        """
        file_path = self._resolve_path(file_path)
        
        # Get handler
        if format_hint:
            handler = self.handlers.get(format_hint)
            if not handler:
                logger.error(f"Unknown format hint: {format_hint}")
                return False
        else:
            handler = self.get_handler(file_path)
            if not handler:
                logger.error(f"Cannot determine format for: {file_path}")
                return False
        
        # Create backup if file exists
        backup_path = None
        if (create_backup if create_backup is not None else self.enable_backup):
            if file_path.exists():
                backup_path = self._create_backup(file_path)
        
        # Atomic write operation
        if atomic:
            return self._atomic_save(handler, data, file_path, **kwargs)
        else:
            return self._direct_save(handler, data, file_path, **kwargs)
    
    def load_data(self, file_path: Union[str, Path], 
                  format_hint: Optional[str] = None,
                  validate_integrity: bool = True, **kwargs) -> Any:
        """
        Load data from file with automatic format detection
        
        Args:
            file_path: Source file path
            format_hint: Format hint ('csv', 'parquet', 'json', 'yaml', 'pickle')
            validate_integrity: Validate file integrity before loading
            **kwargs: Additional arguments for format handler
            
        Returns:
            Loaded data
        """
        file_path = self._resolve_path(file_path)
        
        if not file_path.exists():
            raise DataValidationError(f"File not found: {file_path}")
        
        # Validate file integrity
        if validate_integrity:
            if not self._validate_file_integrity(file_path):
                raise DataValidationError(f"File integrity check failed: {file_path}")
        
        # Get handler
        if format_hint:
            handler = self.handlers.get(format_hint)
            if not handler:
                raise DataValidationError(f"Unknown format hint: {format_hint}")
        else:
            handler = self.get_handler(file_path)
            if not handler:
                raise DataValidationError(f"Cannot determine format for: {file_path}")
        
        # Load data
        with Timer(f"load_file_{file_path.name}", auto_log=False) as timer:
            data = handler.load(file_path, **kwargs)
        
        # Update statistics
        self.stats['files_read'] += 1
        self.stats['bytes_read'] += file_path.stat().st_size
        
        logger.debug(f"Loaded file {file_path} in {timer.result.duration_str}")
        return data
    
    def _resolve_path(self, file_path: Union[str, Path]) -> Path:
        """Resolve file path relative to base path"""
        file_path = Path(file_path)
        
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        return file_path
    
    def _atomic_save(self, handler: BaseFileHandler, data: Any, 
                    file_path: Path, **kwargs) -> bool:
        """Perform atomic save operation"""
        try:
            # Write to temporary file first
            temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
            
            with Timer(f"save_file_{file_path.name}", auto_log=False) as timer:
                success = handler.save(data, temp_path, **kwargs)
            
            if success:
                # Atomic move
                shutil.move(temp_path, file_path)
                
                # Update statistics
                self.stats['files_written'] += 1
                self.stats['bytes_written'] += file_path.stat().st_size
                
                logger.debug(f"Saved file {file_path} in {timer.result.duration_str}")
                return True
            else:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()
                return False
                
        except Exception as e:
            logger.error(f"Atomic save failed for {file_path}: {e}")
            self.stats['errors'] += 1
            return False
    
    def _direct_save(self, handler: BaseFileHandler, data: Any, 
                    file_path: Path, **kwargs) -> bool:
        """Perform direct save operation"""
        try:
            with Timer(f"save_file_{file_path.name}", auto_log=False) as timer:
                success = handler.save(data, file_path, **kwargs)
            
            if success:
                # Update statistics
                self.stats['files_written'] += 1
                self.stats['bytes_written'] += file_path.stat().st_size
                
                logger.debug(f"Saved file {file_path} in {timer.result.duration_str}")
            
            return success
            
        except Exception as e:
            logger.error(f"Direct save failed for {file_path}: {e}")
            self.stats['errors'] += 1
            return False
    
    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """Create backup of existing file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            backup_path = file_path.parent / "backups" / backup_name
            
            ensure_directory(backup_path.parent)
            shutil.copy2(file_path, backup_path)
            
            logger.debug(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.warning(f"Failed to create backup for {file_path}: {e}")
            return None
    
    def _validate_file_integrity(self, file_path: Path) -> bool:
        """Validate file integrity"""
        try:
            # Basic checks
            if not file_path.exists():
                return False
            
            if file_path.stat().st_size == 0:
                logger.warning(f"File is empty: {file_path}")
                return False
            
            # Format-specific validation
            handler = self.get_handler(file_path)
            if handler:
                # Try to read file headers
                try:
                    if isinstance(handler, (CSVHandler, ParquetHandler)):
                        # For data files, try to read first few rows
                        if isinstance(handler, CSVHandler):
                            pd.read_csv(file_path, nrows=1)
                        else:
                            pd.read_parquet(file_path).head(1)
                    elif isinstance(handler, JSONHandler):
                        # Try to parse JSON
                        with open(file_path, 'r') as f:
                            json.loads(f.read(100))  # Read first 100 chars
                    elif isinstance(handler, YAMLHandler):
                        # Try to parse YAML
                        with open(file_path, 'r') as f:
                            yaml.safe_load(f.read(100))
                except:
                    logger.warning(f"File format validation failed: {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"File integrity check failed for {file_path}: {e}")
            return False
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive file information"""
        file_path = self._resolve_path(file_path)
        
        if not file_path.exists():
            raise DataValidationError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        info = {
            'path': str(file_path),
            'name': file_path.name,
            'size_bytes': stat.st_size,
            'size_readable': self._format_file_size(stat.st_size),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'extension': file_path.suffix.lower(),
            'handler_available': self.get_handler(file_path) is not None
        }
        
        # Add format-specific info
        handler = self.get_handler(file_path)
        if handler:
            info['format'] = type(handler).__name__.replace('Handler', '').lower()
            
            # Try to get data info for structured files
            try:
                if isinstance(handler, (CSVHandler, ParquetHandler)):
                    data = handler.load(file_path)
                    if isinstance(data, pd.DataFrame):
                        info['rows'] = len(data)
                        info['columns'] = len(data.columns)
                        info['memory_usage'] = data.memory_usage(deep=True).sum()
            except:
                pass  # Skip if cannot read
        
        return info
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}PB"
    
    def list_files(self, directory: Optional[Union[str, Path]] = None,
                   pattern: str = "*", recursive: bool = False,
                   include_info: bool = False) -> List[Union[Path, Dict[str, Any]]]:
        """
        List files in directory
        
        Args:
            directory: Directory to search (default: base_path)
            pattern: File pattern to match
            recursive: Search recursively
            include_info: Include file information
            
        Returns:
            List of file paths or file info dictionaries
        """
        if directory is None:
            directory = self.base_path
        else:
            directory = self._resolve_path(directory)
        
        if not directory.exists():
            return []
        
        # Get file list
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        # Filter to actual files
        files = [f for f in files if f.is_file()]
        
        if include_info:
            return [self.get_file_info(f) for f in files]
        else:
            return files
    
    def batch_save(self, data_dict: Dict[str, Any], base_directory: Optional[Union[str, Path]] = None,
                   format_hint: Optional[str] = None, max_workers: int = 4) -> Dict[str, bool]:
        """
        Save multiple files in batch
        
        Args:
            data_dict: Dictionary mapping file paths to data
            base_directory: Base directory for relative paths
            format_hint: Format hint for all files
            max_workers: Maximum number of worker threads
            
        Returns:
            Dictionary mapping file paths to success status
        """
        if base_directory:
            base_directory = self._resolve_path(base_directory)
        
        results = {}
        
        def save_single(item):
            file_path, data = item
            if base_directory:
                full_path = base_directory / file_path
            else:
                full_path = file_path
            
            success = self.save_data(data, full_path, format_hint=format_hint)
            return file_path, success
        
        # Use thread pool for parallel saving
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(save_single, item) for item in data_dict.items()]
            
            for future in as_completed(futures):
                try:
                    file_path, success = future.result()
                    results[file_path] = success
                except Exception as e:
                    logger.error(f"Batch save error: {e}")
                    # Try to identify which file failed
                    for file_path in data_dict.keys():
                        if file_path not in results:
                            results[file_path] = False
                            break
        
        return results
    
    def batch_load(self, file_paths: List[Union[str, Path]], 
                   format_hint: Optional[str] = None,
                   max_workers: int = 4) -> Dict[str, Any]:
        """
        Load multiple files in batch
        
        Args:
            file_paths: List of file paths to load
            format_hint: Format hint for all files
            max_workers: Maximum number of worker threads
            
        Returns:
            Dictionary mapping file paths to loaded data
        """
        results = {}
        
        def load_single(file_path):
            try:
                data = self.load_data(file_path, format_hint=format_hint)
                return file_path, data, None
            except Exception as e:
                return file_path, None, e
        
        # Use thread pool for parallel loading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_single, fp) for fp in file_paths]
            
            for future in as_completed(futures):
                file_path, data, error = future.result()
                if error:
                    logger.error(f"Failed to load {file_path}: {error}")
                    results[str(file_path)] = None
                else:
                    results[str(file_path)] = data
        
        return results
    
    def cleanup_old_files(self, directory: Optional[Union[str, Path]] = None,
                         max_age_days: int = 30, pattern: str = "*",
                         dry_run: bool = False) -> List[Path]:
        """
        Clean up old files
        
        Args:
            directory: Directory to clean (default: base_path)
            max_age_days: Maximum age in days
            pattern: File pattern to match
            dry_run: If True, only report what would be deleted
            
        Returns:
            List of files that were (or would be) deleted
        """
        if directory is None:
            directory = self.base_path
        else:
            directory = self._resolve_path(directory)
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        cutoff_timestamp = cutoff_time.timestamp()
        
        files = self.list_files(directory, pattern, recursive=True)
        old_files = []
        
        for file_path in files:
            try:
                stat = file_path.stat()
                if stat.st_mtime < cutoff_timestamp:
                    old_files.append(file_path)
                    
                    if not dry_run:
                        file_path.unlink()
                        logger.debug(f"Deleted old file: {file_path}")
                    
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        if dry_run and old_files:
            logger.info(f"Would delete {len(old_files)} old files")
        elif old_files:
            logger.info(f"Deleted {len(old_files)} old files")
        
        return old_files
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get file operation statistics"""
        return dict(self.stats)
    
    def reset_statistics(self):
        """Reset file operation statistics"""
        for key in self.stats:
            self.stats[key] = 0

# ============================================
# Utility Functions
# ============================================

def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calculate hash of file contents
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hex digest of file hash
    """
    hash_algo = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_algo.update(chunk)
    
    return hash_algo.hexdigest()

def compare_files(file1: Union[str, Path], file2: Union[str, Path]) -> bool:
    """
    Compare two files for equality
    
    Args:
        file1: First file path
        file2: Second file path
        
    Returns:
        True if files are identical
    """
    try:
        return calculate_file_hash(file1) == calculate_file_hash(file2)
    except Exception:
        return False

@contextlib.contextmanager
def temporary_file(suffix: str = '', prefix: str = 'stockpred_', 
                  directory: Optional[Union[str, Path]] = None):
    """
    Context manager for temporary files
    
    Args:
        suffix: File suffix
        prefix: File prefix
        directory: Directory for temporary file
        
    Yields:
        Path to temporary file
    """
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=directory)
    temp_path = Path(temp_path)
    
    try:
        os.close(temp_fd)  # Close file descriptor
        yield temp_path
    finally:
        # Clean up
        if temp_path.exists():
            temp_path.unlink()

def archive_directory(source_dir: Union[str, Path], 
                     archive_path: Union[str, Path],
                     format: str = 'zip') -> bool:
    """
    Archive directory contents
    
    Args:
        source_dir: Source directory to archive
        archive_path: Target archive file path
        format: Archive format ('zip', 'tar', 'gztar')
        
    Returns:
        True if successful
    """
    try:
        source_dir = Path(source_dir)
        archive_path = Path(archive_path)
        
        ensure_directory(archive_path.parent)
        
        # Remove extension from archive_path for shutil.make_archive
        archive_base = archive_path.with_suffix('')
        
        shutil.make_archive(str(archive_base), format, str(source_dir))
        
        logger.info(f"Created archive: {archive_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create archive: {e}")
        return False

# ============================================
# Global File Manager Instance
# ============================================

# Create global file manager instance
default_file_manager = FileManager()

# Convenience functions using global instance
def save_data(data: Any, file_path: Union[str, Path], **kwargs) -> bool:
    """Save data using global file manager"""
    return default_file_manager.save_data(data, file_path, **kwargs)

def load_data(file_path: Union[str, Path], **kwargs) -> Any:
    """Load data using global file manager"""
    return default_file_manager.load_data(file_path, **kwargs)

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get file info using global file manager"""
    return default_file_manager.get_file_info(file_path)

def list_files(directory: Optional[Union[str, Path]] = None, **kwargs) -> List[Path]:
    """List files using global file manager"""
    return default_file_manager.list_files(directory, **kwargs)
