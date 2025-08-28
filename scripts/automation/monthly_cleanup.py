"""
scripts/automation/monthly_cleanup.py

Comprehensive monthly maintenance and cleanup for StockPredictionPro.
Manages disk space, archives old data, cleans logs, optimizes databases,
and performs system maintenance with detailed reporting and safety checks.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import logging
import shutil
import gzip
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import tempfile

# System utilities
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Database utilities
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'monthly_cleanup_{datetime.now().strftime("%Y%m")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StockPredictionPro.MonthlyCleanup')

# Directory configuration
PROJECT_ROOT = Path('.')
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
CACHE_DIR = DATA_DIR / 'cache'
TEMP_DIR = DATA_DIR / 'temp'
BACKUP_DIR = PROJECT_ROOT / 'backups'
ARCHIVE_DIR = PROJECT_ROOT / 'archive'

# Ensure directories exist
for dir_path in [BACKUP_DIR, ARCHIVE_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class CleanupConfig:
    """Configuration for monthly cleanup operations"""
    # Data retention settings
    raw_data_retention_days: int = 365  # Keep raw data for 1 year
    processed_data_retention_days: int = 180  # Keep processed data for 6 months
    cache_retention_days: int = 30  # Keep cache for 1 month
    temp_file_retention_days: int = 7  # Keep temp files for 1 week
    
    # Log management
    log_retention_days: int = 90  # Keep logs for 3 months
    compress_old_logs: bool = True  # Compress logs older than 30 days
    max_log_size_mb: int = 100  # Archive logs larger than 100MB
    
    # Model management
    model_retention_count: int = 5  # Keep last 5 versions of each model
    backup_models_before_cleanup: bool = True
    experimental_model_retention_days: int = 30
    
    # Output management
    report_retention_days: int = 180  # Keep reports for 6 months
    prediction_retention_days: int = 90  # Keep predictions for 3 months
    visualization_retention_days: int = 60  # Keep visualizations for 2 months
    
    # System maintenance
    optimize_databases: bool = True
    defragment_data: bool = False  # Only for Windows
    clear_system_cache: bool = True
    
    # Safety settings
    min_free_space_gb: float = 5.0  # Minimum free space to maintain
    backup_before_deletion: bool = True
    dry_run_mode: bool = False  # If True, only simulate cleanup
    
    # Notification settings
    send_cleanup_report: bool = True
    alert_on_cleanup_issues: bool = True

@dataclass
class CleanupStats:
    """Statistics from cleanup operations"""
    files_deleted: int = 0
    directories_cleaned: int = 0
    space_freed_mb: float = 0.0
    files_compressed: int = 0
    files_archived: int = 0
    files_backed_up: int = 0
    errors_encountered: int = 0

@dataclass
class CleanupResult:
    """Results from a single cleanup operation"""
    operation_name: str
    status: str  # success, failed, skipped
    message: str
    stats: CleanupStats
    execution_time: float
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

@dataclass
class MonthlyCleanupReport:
    """Comprehensive monthly cleanup report"""
    execution_date: str
    total_execution_time: float
    
    # Overall statistics
    total_files_processed: int
    total_space_freed_mb: float
    total_errors: int
    
    # Operation results
    cleanup_results: List[CleanupResult]
    
    # System health
    disk_space_before: Dict[str, float]
    disk_space_after: Dict[str, float]
    
    # Summary
    overall_status: str  # success, partial_success, failed
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

# ============================================
# CLEANUP COMPONENTS
# ============================================

class DataCleaner:
    """Clean and archive old data files"""
    
    def __init__(self, config: CleanupConfig):
        self.config = config
        
    def cleanup_data_directories(self) -> List[CleanupResult]:
        """Clean all data directories"""
        results = []
        
        # Clean raw data
        result = self._cleanup_data_directory(
            DATA_DIR / 'raw',
            self.config.raw_data_retention_days,
            'raw_data_cleanup'
        )
        results.append(result)
        
        # Clean processed data
        result = self._cleanup_data_directory(
            DATA_DIR / 'processed',
            self.config.processed_data_retention_days,
            'processed_data_cleanup'
        )
        results.append(result)
        
        # Clean cache
        result = self._cleanup_data_directory(
            CACHE_DIR,
            self.config.cache_retention_days,
            'cache_cleanup'
        )
        results.append(result)
        
        # Clean temp files
        result = self._cleanup_temp_files()
        results.append(result)
        
        return results
    
    def _cleanup_data_directory(self, directory: Path, retention_days: int, 
                              operation_name: str) -> CleanupResult:
        """Clean a specific data directory"""
        start_time = time.time()
        stats = CleanupStats()
        errors = []
        
        try:
            if not directory.exists():
                return CleanupResult(
                    operation_name=operation_name,
                    status='skipped',
                    message=f"Directory {directory} does not exist",
                    stats=stats,
                    execution_time=time.time() - start_time
                )
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            logger.info(f"🧹 Cleaning {directory} (files older than {cutoff_date.date()})")
            
            for file_path in directory.rglob('*'):
                if not file_path.is_file():
                    continue
                
                try:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_mtime < cutoff_date:
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        
                        # Backup if configured
                        if self.config.backup_before_deletion:
                            self._backup_file(file_path)
                            stats.files_backed_up += 1
                        
                        # Delete or simulate
                        if self.config.dry_run_mode:
                            logger.info(f"Would delete: {file_path}")
                        else:
                            file_path.unlink()
                            logger.debug(f"Deleted: {file_path}")
                        
                        stats.files_deleted += 1
                        stats.space_freed_mb += file_size_mb
                        
                except Exception as e:
                    error_msg = f"Failed to process {file_path}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    stats.errors_encountered += 1
            
            # Clean empty directories
            self._clean_empty_directories(directory, stats)
            
            status = 'success' if stats.errors_encountered == 0 else 'partial_success'
            message = f"Cleaned {stats.files_deleted} files, freed {stats.space_freed_mb:.1f} MB"
            
        except Exception as e:
            status = 'failed'
            message = f"Directory cleanup failed: {e}"
            errors.append(str(e))
            logger.error(f"❌ {operation_name} failed: {e}")
        
        return CleanupResult(
            operation_name=operation_name,
            status=status,
            message=message,
            stats=stats,
            execution_time=time.time() - start_time,
            errors=errors
        )
    
    def _cleanup_temp_files(self) -> CleanupResult:
        """Clean temporary files and system temp"""
        start_time = time.time()
        stats = CleanupStats()
        errors = []
        
        try:
            logger.info("🧹 Cleaning temporary files...")
            
            # Clean project temp directory
            if TEMP_DIR.exists():
                for file_path in TEMP_DIR.rglob('*'):
                    if not file_path.is_file():
                        continue
                    
                    try:
                        file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
                        
                        if file_age.days > self.config.temp_file_retention_days:
                            file_size_mb = file_path.stat().st_size / (1024 * 1024)
                            
                            if self.config.dry_run_mode:
                                logger.info(f"Would delete temp file: {file_path}")
                            else:
                                file_path.unlink()
                            
                            stats.files_deleted += 1
                            stats.space_freed_mb += file_size_mb
                            
                    except Exception as e:
                        errors.append(f"Failed to clean temp file {file_path}: {e}")
                        stats.errors_encountered += 1
            
            # Clean Python temp files
            self._clean_python_temp_files(stats, errors)
            
            # Clean system temp (with caution)
            self._clean_system_temp_files(stats, errors)
            
            status = 'success' if stats.errors_encountered == 0 else 'partial_success'
            message = f"Cleaned {stats.files_deleted} temp files, freed {stats.space_freed_mb:.1f} MB"
            
        except Exception as e:
            status = 'failed'
            message = f"Temp cleanup failed: {e}"
            errors.append(str(e))
        
        return CleanupResult(
            operation_name='temp_files_cleanup',
            status=status,
            message=message,
            stats=stats,
            execution_time=time.time() - start_time,
            errors=errors
        )
    
    def _clean_python_temp_files(self, stats: CleanupStats, errors: List[str]) -> None:
        """Clean Python-specific temporary files"""
        try:
            # Clean __pycache__ directories
            for pycache_dir in PROJECT_ROOT.rglob('__pycache__'):
                if pycache_dir.is_dir():
                    try:
                        if self.config.dry_run_mode:
                            logger.info(f"Would remove __pycache__: {pycache_dir}")
                        else:
                            shutil.rmtree(pycache_dir)
                            logger.debug(f"Removed __pycache__: {pycache_dir}")
                        
                        stats.directories_cleaned += 1
                        
                    except Exception as e:
                        errors.append(f"Failed to remove {pycache_dir}: {e}")
                        stats.errors_encountered += 1
            
            # Clean .pyc files
            for pyc_file in PROJECT_ROOT.rglob('*.pyc'):
                try:
                    file_size_mb = pyc_file.stat().st_size / (1024 * 1024)
                    
                    if self.config.dry_run_mode:
                        logger.info(f"Would delete .pyc file: {pyc_file}")
                    else:
                        pyc_file.unlink()
                    
                    stats.files_deleted += 1
                    stats.space_freed_mb += file_size_mb
                    
                except Exception as e:
                    errors.append(f"Failed to delete {pyc_file}: {e}")
                    stats.errors_encountered += 1
                    
        except Exception as e:
            errors.append(f"Python temp cleanup failed: {e}")
            stats.errors_encountered += 1
    
    def _clean_system_temp_files(self, stats: CleanupStats, errors: List[str]) -> None:
        """Carefully clean system temporary files"""
        try:
            # Only clean files that are clearly safe to delete
            system_temp = Path(tempfile.gettempdir())
            
            # Look for obviously temporary files
            temp_patterns = ['*.tmp', 'tmp*', '*.temp', '*.log.old']
            
            for pattern in temp_patterns:
                for temp_file in system_temp.glob(pattern):
                    try:
                        # Only delete files older than 7 days to be safe
                        file_age = datetime.now() - datetime.fromtimestamp(temp_file.stat().st_mtime)
                        
                        if file_age.days > 7 and temp_file.is_file():
                            file_size_mb = temp_file.stat().st_size / (1024 * 1024)
                            
                            if self.config.dry_run_mode:
                                logger.info(f"Would delete system temp file: {temp_file}")
                            else:
                                temp_file.unlink()
                            
                            stats.files_deleted += 1
                            stats.space_freed_mb += file_size_mb
                            
                    except (PermissionError, FileNotFoundError):
                        # Ignore permission errors for system files
                        continue
                    except Exception as e:
                        errors.append(f"System temp cleanup error {temp_file}: {e}")
                        stats.errors_encountered += 1
                        
        except Exception as e:
            errors.append(f"System temp directory cleanup failed: {e}")
            stats.errors_encountered += 1
    
    def _backup_file(self, file_path: Path) -> None:
        """Create backup of file before deletion"""
        try:
            # Create monthly backup directory
            backup_month_dir = BACKUP_DIR / datetime.now().strftime('%Y%m')
            backup_month_dir.mkdir(exist_ok=True)
            
            # Create backup file path
            relative_path = file_path.relative_to(PROJECT_ROOT)
            backup_file_path = backup_month_dir / relative_path
            
            # Ensure backup directory structure exists
            backup_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to backup
            if not self.config.dry_run_mode:
                shutil.copy2(file_path, backup_file_path)
                
        except Exception as e:
            logger.warning(f"Failed to backup {file_path}: {e}")
    
    def _clean_empty_directories(self, root_dir: Path, stats: CleanupStats) -> None:
        """Remove empty directories recursively"""
        try:
            for dir_path in sorted(root_dir.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if dir_path.is_dir() and dir_path != root_dir:
                    try:
                        if not any(dir_path.iterdir()):  # Directory is empty
                            if self.config.dry_run_mode:
                                logger.info(f"Would remove empty directory: {dir_path}")
                            else:
                                dir_path.rmdir()
                                logger.debug(f"Removed empty directory: {dir_path}")
                            
                            stats.directories_cleaned += 1
                            
                    except OSError:
                        # Directory not empty or permission error
                        continue
                        
        except Exception as e:
            logger.warning(f"Empty directory cleanup failed: {e}")

class LogCleaner:
    """Clean and archive log files"""
    
    def __init__(self, config: CleanupConfig):
        self.config = config
    
    def cleanup_logs(self) -> CleanupResult:
        """Clean and manage log files"""
        start_time = time.time()
        stats = CleanupStats()
        errors = []
        
        try:
            logger.info("📄 Cleaning and archiving log files...")
            
            if not LOGS_DIR.exists():
                return CleanupResult(
                    operation_name='log_cleanup',
                    status='skipped',
                    message="Logs directory does not exist",
                    stats=stats,
                    execution_time=time.time() - start_time
                )
            
            # Process each log file
            for log_file in LOGS_DIR.glob('*.log'):
                try:
                    self._process_log_file(log_file, stats, errors)
                except Exception as e:
                    errors.append(f"Failed to process log file {log_file}: {e}")
                    stats.errors_encountered += 1
            
            # Clean old rotated logs
            self._clean_rotated_logs(stats, errors)
            
            status = 'success' if stats.errors_encountered == 0 else 'partial_success'
            message = f"Processed logs: {stats.files_deleted} deleted, {stats.files_compressed} compressed"
            
        except Exception as e:
            status = 'failed'
            message = f"Log cleanup failed: {e}"
            errors.append(str(e))
        
        return CleanupResult(
            operation_name='log_cleanup',
            status=status,
            message=message,
            stats=stats,
            execution_time=time.time() - start_time,
            errors=errors
        )
    
    def _process_log_file(self, log_file: Path, stats: CleanupStats, errors: List[str]) -> None:
        """Process individual log file"""
        try:
            file_stat = log_file.stat()
            file_age = datetime.now() - datetime.fromtimestamp(file_stat.st_mtime)
            file_size_mb = file_stat.st_size / (1024 * 1024)
            
            # Delete old logs
            if file_age.days > self.config.log_retention_days:
                if self.config.backup_before_deletion:
                    self._archive_log_file(log_file)
                    stats.files_archived += 1
                
                if self.config.dry_run_mode:
                    logger.info(f"Would delete old log: {log_file}")
                else:
                    log_file.unlink()
                    logger.debug(f"Deleted old log: {log_file}")
                
                stats.files_deleted += 1
                stats.space_freed_mb += file_size_mb
                
            # Compress large or old logs
            elif (file_size_mb > self.config.max_log_size_mb or 
                  (self.config.compress_old_logs and file_age.days > 30)):
                
                compressed_size = self._compress_log_file(log_file)
                if compressed_size > 0:
                    stats.files_compressed += 1
                    stats.space_freed_mb += file_size_mb - compressed_size
                    
        except Exception as e:
            errors.append(f"Log file processing error {log_file}: {e}")
    
    def _compress_log_file(self, log_file: Path) -> float:
        """Compress log file and return new size in MB"""
        try:
            compressed_file = log_file.with_suffix(log_file.suffix + '.gz')
            
            if compressed_file.exists():
                return compressed_file.stat().st_size / (1024 * 1024)
            
            if self.config.dry_run_mode:
                logger.info(f"Would compress log: {log_file}")
                return log_file.stat().st_size / (1024 * 1024) * 0.1  # Estimate 90% compression
            
            with open(log_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Replace original with compressed version
            log_file.unlink()
            
            compressed_size_mb = compressed_file.stat().st_size / (1024 * 1024)
            logger.debug(f"Compressed log: {log_file} -> {compressed_file}")
            
            return compressed_size_mb
            
        except Exception as e:
            logger.warning(f"Log compression failed for {log_file}: {e}")
            return 0.0
    
    def _archive_log_file(self, log_file: Path) -> None:
        """Archive log file to monthly archive"""
        try:
            archive_month_dir = ARCHIVE_DIR / 'logs' / datetime.now().strftime('%Y%m')
            archive_month_dir.mkdir(parents=True, exist_ok=True)
            
            archive_path = archive_month_dir / log_file.name
            
            if not self.config.dry_run_mode:
                shutil.copy2(log_file, archive_path)
                logger.debug(f"Archived log: {log_file} -> {archive_path}")
                
        except Exception as e:
            logger.warning(f"Log archiving failed for {log_file}: {e}")
    
    def _clean_rotated_logs(self, stats: CleanupStats, errors: List[str]) -> None:
        """Clean rotated log files (.log.1, .log.2, etc.)"""
        try:
            rotated_patterns = ['*.log.*', '*.log.old', '*.log.bak']
            
            for pattern in rotated_patterns:
                for rotated_log in LOGS_DIR.glob(pattern):
                    try:
                        file_age = datetime.now() - datetime.fromtimestamp(rotated_log.stat().st_mtime)
                        
                        if file_age.days > self.config.log_retention_days:
                            file_size_mb = rotated_log.stat().st_size / (1024 * 1024)
                            
                            if self.config.dry_run_mode:
                                logger.info(f"Would delete rotated log: {rotated_log}")
                            else:
                                rotated_log.unlink()
                            
                            stats.files_deleted += 1
                            stats.space_freed_mb += file_size_mb
                            
                    except Exception as e:
                        errors.append(f"Rotated log cleanup error {rotated_log}: {e}")
                        stats.errors_encountered += 1
                        
        except Exception as e:
            errors.append(f"Rotated logs cleanup failed: {e}")
            stats.errors_encountered += 1

class ModelCleaner:
    """Clean and manage model files"""
    
    def __init__(self, config: CleanupConfig):
        self.config = config
    
    def cleanup_models(self) -> CleanupResult:
        """Clean old model files and maintain version limits"""
        start_time = time.time()
        stats = CleanupStats()
        errors = []
        
        try:
            logger.info("🤖 Cleaning model files...")
            
            # Clean experimental models
            self._clean_experimental_models(stats, errors)
            
            # Maintain version limits for production models
            self._maintain_model_versions(stats, errors)
            
            # Clean backup models
            self._clean_backup_models(stats, errors)
            
            status = 'success' if stats.errors_encountered == 0 else 'partial_success'
            message = f"Model cleanup: {stats.files_deleted} deleted, {stats.files_backed_up} backed up"
            
        except Exception as e:
            status = 'failed'
            message = f"Model cleanup failed: {e}"
            errors.append(str(e))
        
        return CleanupResult(
            operation_name='model_cleanup',
            status=status,
            message=message,
            stats=stats,
            execution_time=time.time() - start_time,
            errors=errors
        )
    
    def _clean_experimental_models(self, stats: CleanupStats, errors: List[str]) -> None:
        """Clean old experimental models"""
        try:
            experiments_dir = MODELS_DIR / 'experiments'
            
            if not experiments_dir.exists():
                return
            
            cutoff_date = datetime.now() - timedelta(days=self.config.experimental_model_retention_days)
            
            for model_file in experiments_dir.rglob('*.pkl'):
                try:
                    file_mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
                    
                    if file_mtime < cutoff_date:
                        file_size_mb = model_file.stat().st_size / (1024 * 1024)
                        
                        if self.config.backup_models_before_cleanup:
                            self._backup_model_file(model_file)
                            stats.files_backed_up += 1
                        
                        if self.config.dry_run_mode:
                            logger.info(f"Would delete experimental model: {model_file}")
                        else:
                            model_file.unlink()
                            logger.debug(f"Deleted experimental model: {model_file}")
                        
                        stats.files_deleted += 1
                        stats.space_freed_mb += file_size_mb
                        
                except Exception as e:
                    errors.append(f"Experimental model cleanup error {model_file}: {e}")
                    stats.errors_encountered += 1
                    
        except Exception as e:
            errors.append(f"Experimental models cleanup failed: {e}")
            stats.errors_encountered += 1
    
    def _maintain_model_versions(self, stats: CleanupStats, errors: List[str]) -> None:
        """Maintain version limits for each model type"""
        try:
            model_dirs = [MODELS_DIR / 'trained', MODELS_DIR / 'production']
            
            for model_dir in model_dirs:
                if not model_dir.exists():
                    continue
                
                # Group models by base name (symbol + model type)
                model_groups = {}
                
                for model_file in model_dir.glob('*.pkl'):
                    # Extract base name (remove timestamp/version info)
                    base_name = self._extract_model_base_name(model_file.name)
                    
                    if base_name not in model_groups:
                        model_groups[base_name] = []
                    
                    model_groups[base_name].append(model_file)
                
                # Clean each group
                for base_name, model_files in model_groups.items():
                    try:
                        self._clean_model_group(base_name, model_files, stats, errors)
                    except Exception as e:
                        errors.append(f"Model group cleanup error {base_name}: {e}")
                        stats.errors_encountered += 1
                        
        except Exception as e:
            errors.append(f"Model version maintenance failed: {e}")
            stats.errors_encountered += 1
    
    def _clean_model_group(self, base_name: str, model_files: List[Path], 
                          stats: CleanupStats, errors: List[str]) -> None:
        """Clean a group of model versions"""
        if len(model_files) <= self.config.model_retention_count:
            return  # No cleanup needed
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Keep the most recent models, delete the rest
        files_to_delete = model_files[self.config.model_retention_count:]
        
        for model_file in files_to_delete:
            try:
                file_size_mb = model_file.stat().st_size / (1024 * 1024)
                
                if self.config.backup_models_before_cleanup:
                    self._backup_model_file(model_file)
                    stats.files_backed_up += 1
                
                if self.config.dry_run_mode:
                    logger.info(f"Would delete old model version: {model_file}")
                else:
                    model_file.unlink()
                    logger.debug(f"Deleted old model version: {model_file}")
                
                stats.files_deleted += 1
                stats.space_freed_mb += file_size_mb
                
            except Exception as e:
                errors.append(f"Model deletion error {model_file}: {e}")
                stats.errors_encountered += 1
    
    def _clean_backup_models(self, stats: CleanupStats, errors: List[str]) -> None:
        """Clean old backup models"""
        try:
            backup_dir = MODELS_DIR / 'backup'
            
            if not backup_dir.exists():
                return
            
            # Keep backups for 90 days
            cutoff_date = datetime.now() - timedelta(days=90)
            
            for backup_file in backup_dir.rglob('*.pkl'):
                try:
                    file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    
                    if file_mtime < cutoff_date:
                        file_size_mb = backup_file.stat().st_size / (1024 * 1024)
                        
                        if self.config.dry_run_mode:
                            logger.info(f"Would delete old backup: {backup_file}")
                        else:
                            backup_file.unlink()
                            logger.debug(f"Deleted old backup: {backup_file}")
                        
                        stats.files_deleted += 1
                        stats.space_freed_mb += file_size_mb
                        
                except Exception as e:
                    errors.append(f"Backup model cleanup error {backup_file}: {e}")
                    stats.errors_encountered += 1
                    
        except Exception as e:
            errors.append(f"Backup models cleanup failed: {e}")
            stats.errors_encountered += 1
    
    def _extract_model_base_name(self, filename: str) -> str:
        """Extract base model name without timestamps"""
        # Remove common timestamp patterns and extensions
        base_name = filename.replace('.pkl', '')
        
        # Remove timestamp patterns
        import re
        # Remove patterns like _20231201_143052, _v_20231201, etc.
        base_name = re.sub(r'_\d{8}_\d{6}', '', base_name)
        base_name = re.sub(r'_v_\d{8}', '', base_name)
        base_name = re.sub(r'_\d{8}T\d{6}Z', '', base_name)
        base_name = re.sub(r'_retrained_\d+', '', base_name)
        base_name = re.sub(r'_optimized_\d+', '', base_name)
        
        return base_name
    
    def _backup_model_file(self, model_file: Path) -> None:
        """Backup model file to archive"""
        try:
            archive_month_dir = ARCHIVE_DIR / 'models' / datetime.now().strftime('%Y%m')
            archive_month_dir.mkdir(parents=True, exist_ok=True)
            
            archive_path = archive_month_dir / model_file.name
            
            if not self.config.dry_run_mode:
                shutil.copy2(model_file, archive_path)
                logger.debug(f"Backed up model: {model_file} -> {archive_path}")
                
        except Exception as e:
            logger.warning(f"Model backup failed for {model_file}: {e}")

class OutputCleaner:
    """Clean output files (reports, predictions, visualizations)"""
    
    def __init__(self, config: CleanupConfig):
        self.config = config
    
    def cleanup_outputs(self) -> CleanupResult:
        """Clean output directories"""
        start_time = time.time()
        stats = CleanupStats()
        errors = []
        
        try:
            logger.info("📊 Cleaning output files...")
            
            # Clean reports
            self._clean_output_directory(
                OUTPUTS_DIR / 'reports',
                self.config.report_retention_days,
                'reports',
                stats, errors
            )
            
            # Clean predictions
            self._clean_output_directory(
                OUTPUTS_DIR / 'predictions',
                self.config.prediction_retention_days,
                'predictions',
                stats, errors
            )
            
            # Clean visualizations
            self._clean_output_directory(
                OUTPUTS_DIR / 'visualizations',
                self.config.visualization_retention_days,
                'visualizations',
                stats, errors
            )
            
            status = 'success' if stats.errors_encountered == 0 else 'partial_success'
            message = f"Output cleanup: {stats.files_deleted} files, {stats.space_freed_mb:.1f} MB freed"
            
        except Exception as e:
            status = 'failed'
            message = f"Output cleanup failed: {e}"
            errors.append(str(e))
        
        return CleanupResult(
            operation_name='output_cleanup',
            status=status,
            message=message,
            stats=stats,
            execution_time=time.time() - start_time,
            errors=errors
        )
    
    def _clean_output_directory(self, directory: Path, retention_days: int, 
                               dir_type: str, stats: CleanupStats, errors: List[str]) -> None:
        """Clean specific output directory"""
        try:
            if not directory.exists():
                return
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for output_file in directory.rglob('*'):
                if not output_file.is_file():
                    continue
                
                try:
                    file_mtime = datetime.fromtimestamp(output_file.stat().st_mtime)
                    
                    if file_mtime < cutoff_date:
                        file_size_mb = output_file.stat().st_size / (1024 * 1024)
                        
                        # Archive important reports before deletion
                        if dir_type == 'reports' and self.config.backup_before_deletion:
                            self._archive_output_file(output_file, dir_type)
                            stats.files_archived += 1
                        
                        if self.config.dry_run_mode:
                            logger.info(f"Would delete {dir_type} file: {output_file}")
                        else:
                            output_file.unlink()
                            logger.debug(f"Deleted {dir_type} file: {output_file}")
                        
                        stats.files_deleted += 1
                        stats.space_freed_mb += file_size_mb
                        
                except Exception as e:
                    errors.append(f"Output file cleanup error {output_file}: {e}")
                    stats.errors_encountered += 1
                    
        except Exception as e:
            errors.append(f"Output directory cleanup failed for {directory}: {e}")
            stats.errors_encountered += 1
    
    def _archive_output_file(self, output_file: Path, dir_type: str) -> None:
        """Archive important output file"""
        try:
            archive_month_dir = ARCHIVE_DIR / dir_type / datetime.now().strftime('%Y%m')
            archive_month_dir.mkdir(parents=True, exist_ok=True)
            
            archive_path = archive_month_dir / output_file.name
            
            if not self.config.dry_run_mode:
                shutil.copy2(output_file, archive_path)
                logger.debug(f"Archived {dir_type} file: {output_file} -> {archive_path}")
                
        except Exception as e:
            logger.warning(f"Output file archiving failed for {output_file}: {e}")

class SystemOptimizer:
    """Perform system optimization tasks"""
    
    def __init__(self, config: CleanupConfig):
        self.config = config
    
    def optimize_system(self) -> CleanupResult:
        """Perform system optimization tasks"""
        start_time = time.time()
        stats = CleanupStats()
        errors = []
        
        try:
            logger.info("⚡ Performing system optimization...")
            
            # Database optimization
            if self.config.optimize_databases:
                self._optimize_databases(stats, errors)
            
            # Clear system cache
            if self.config.clear_system_cache:
                self._clear_system_cache(stats, errors)
            
            # Defragmentation (Windows only)
            if self.config.defragment_data and os.name == 'nt':
                self._defragment_data(stats, errors)
            
            status = 'success' if stats.errors_encountered == 0 else 'partial_success'
            message = f"System optimization completed with {stats.errors_encountered} errors"
            
        except Exception as e:
            status = 'failed'
            message = f"System optimization failed: {e}"
            errors.append(str(e))
        
        return CleanupResult(
            operation_name='system_optimization',
            status=status,
            message=message,
            stats=stats,
            execution_time=time.time() - start_time,
            errors=errors
        )
    
    def _optimize_databases(self, stats: CleanupStats, errors: List[str]) -> None:
        """Optimize database files"""
        try:
            if not HAS_SQLALCHEMY:
                return
            
            # Find SQLite databases
            sqlite_files = list(PROJECT_ROOT.rglob('*.db'))
            
            for db_file in sqlite_files:
                try:
                    if self.config.dry_run_mode:
                        logger.info(f"Would optimize database: {db_file}")
                        continue
                    
                    # Connect and optimize
                    engine = create_engine(f'sqlite:///{db_file}')
                    with engine.connect() as conn:
                        # Vacuum database
                        conn.execute(text('VACUUM'))
                        # Analyze for query optimization
                        conn.execute(text('ANALYZE'))
                    
                    logger.debug(f"Optimized database: {db_file}")
                    
                except Exception as e:
                    errors.append(f"Database optimization failed for {db_file}: {e}")
                    stats.errors_encountered += 1
                    
        except Exception as e:
            errors.append(f"Database optimization failed: {e}")
            stats.errors_encountered += 1
    
    def _clear_system_cache(self, stats: CleanupStats, errors: List[str]) -> None:
        """Clear system cache files"""
        try:
            if self.config.dry_run_mode:
                logger.info("Would clear system cache")
                return
            
            # Clear pip cache
            try:
                import subprocess
                result = subprocess.run(['pip', 'cache', 'purge'], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    logger.debug("Cleared pip cache")
                else:
                    errors.append(f"Pip cache clear failed: {result.stderr}")
                    stats.errors_encountered += 1
            except Exception as e:
                errors.append(f"Pip cache clear failed: {e}")
                stats.errors_encountered += 1
            
            # Clear other Python caches would be added here
            
        except Exception as e:
            errors.append(f"System cache clear failed: {e}")
            stats.errors_encountered += 1
    
    def _defragment_data(self, stats: CleanupStats, errors: List[str]) -> None:
        """Defragment data files (Windows only)"""
        try:
            # This would implement defragmentation logic for Windows
            # For now, just log that it would be done
            if self.config.dry_run_mode:
                logger.info("Would perform data defragmentation")
            else:
                logger.info("Defragmentation not implemented")
                
        except Exception as e:
            errors.append(f"Defragmentation failed: {e}")
            stats.errors_encountered += 1

# ============================================
# MAIN CLEANUP ORCHESTRATOR
# ============================================

class MonthlyCleanupOrchestrator:
    """Main orchestrator for monthly cleanup process"""
    
    def __init__(self, config: CleanupConfig = None):
        self.config = config or CleanupConfig()
        self.start_time = None
        
        # Initialize cleaners
        self.data_cleaner = DataCleaner(self.config)
        self.log_cleaner = LogCleaner(self.config)
        self.model_cleaner = ModelCleaner(self.config)
        self.output_cleaner = OutputCleaner(self.config)
        self.system_optimizer = SystemOptimizer(self.config)
    
    def run_monthly_cleanup(self) -> MonthlyCleanupReport:
        """Run complete monthly cleanup process"""
        logger.info("🧹 Starting monthly cleanup and maintenance...")
        self.start_time = time.time()
        
        try:
            # Check disk space before cleanup
            disk_space_before = self._get_disk_space_info()
            
            # Check minimum free space
            if not self._check_minimum_free_space(disk_space_before):
                logger.error("❌ Insufficient disk space to perform safe cleanup")
                return self._create_failure_report("Insufficient disk space")
            
            # Run all cleanup operations
            all_results = []
            
            # Data cleanup
            logger.info("🗂️ Step 1: Data cleanup")
            data_results = self.data_cleaner.cleanup_data_directories()
            all_results.extend(data_results)
            
            # Log cleanup
            logger.info("📄 Step 2: Log cleanup")
            log_result = self.log_cleaner.cleanup_logs()
            all_results.append(log_result)
            
            # Model cleanup
            logger.info("🤖 Step 3: Model cleanup")
            model_result = self.model_cleaner.cleanup_models()
            all_results.append(model_result)
            
            # Output cleanup
            logger.info("📊 Step 4: Output cleanup")
            output_result = self.output_cleaner.cleanup_outputs()
            all_results.append(output_result)
            
            # System optimization
            logger.info("⚡ Step 5: System optimization")
            system_result = self.system_optimizer.optimize_system()
            all_results.append(system_result)
            
            # Check disk space after cleanup
            disk_space_after = self._get_disk_space_info()
            
            # Generate final report
            report = self._generate_final_report(
                all_results, disk_space_before, disk_space_after
            )
            
            # Save results
            self._save_cleanup_results(report)
            
            # Print summary
            self._print_summary(report)
            
            logger.info("✅ Monthly cleanup completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"❌ Monthly cleanup failed: {e}")
            logger.error(traceback.format_exc())
            return self._create_failure_report(str(e))
    
    def _get_disk_space_info(self) -> Dict[str, float]:
        """Get disk space information"""
        try:
            if HAS_PSUTIL:
                disk_usage = psutil.disk_usage('.')
                return {
                    'total_gb': disk_usage.total / (1024**3),
                    'used_gb': disk_usage.used / (1024**3),
                    'free_gb': disk_usage.free / (1024**3),
                    'percent_used': (disk_usage.used / disk_usage.total) * 100
                }
            else:
                # Fallback using shutil
                disk_usage = shutil.disk_usage('.')
                return {
                    'total_gb': disk_usage.total / (1024**3),
                    'used_gb': (disk_usage.total - disk_usage.free) / (1024**3),
                    'free_gb': disk_usage.free / (1024**3),
                    'percent_used': ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100
                }
        except Exception as e:
            logger.warning(f"Could not get disk space info: {e}")
            return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0, 'percent_used': 0}
    
    def _check_minimum_free_space(self, disk_info: Dict[str, float]) -> bool:
        """Check if we have minimum required free space"""
        free_space_gb = disk_info.get('free_gb', 0)
        return free_space_gb >= self.config.min_free_space_gb
    
    def _generate_final_report(self, cleanup_results: List[CleanupResult],
                             disk_before: Dict[str, float], 
                             disk_after: Dict[str, float]) -> MonthlyCleanupReport:
        """Generate comprehensive final report"""
        execution_time = time.time() - self.start_time
        
        # Aggregate statistics
        total_files_processed = sum(r.stats.files_deleted + r.stats.files_compressed + 
                                  r.stats.files_archived for r in cleanup_results)
        total_space_freed = sum(r.stats.space_freed_mb for r in cleanup_results)
        total_errors = sum(r.stats.errors_encountered for r in cleanup_results)
        
        # Determine overall status
        failed_operations = [r for r in cleanup_results if r.status == 'failed']
        partial_operations = [r for r in cleanup_results if r.status == 'partial_success']
        
        if len(failed_operations) > len(cleanup_results) // 2:
            overall_status = 'failed'
        elif failed_operations or partial_operations:
            overall_status = 'partial_success'
        else:
            overall_status = 'success'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(cleanup_results, disk_after)
        
        return MonthlyCleanupReport(
            execution_date=datetime.now().strftime('%Y-%m-%d'),
            total_execution_time=execution_time,
            total_files_processed=total_files_processed,
            total_space_freed_mb=total_space_freed,
            total_errors=total_errors,
            cleanup_results=cleanup_results,
            disk_space_before=disk_before,
            disk_space_after=disk_after,
            overall_status=overall_status,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, results: List[CleanupResult], 
                                disk_info: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Disk space recommendations
        if disk_info.get('percent_used', 0) > 80:
            recommendations.append("Disk usage still high (>80%) - consider additional cleanup or storage expansion")
        
        # Error-based recommendations
        high_error_operations = [r for r in results if r.stats.errors_encountered > 5]
        if high_error_operations:
            recommendations.append(f"Review operations with high error rates: {[r.operation_name for r in high_error_operations]}")
        
        # Space savings recommendations
        total_space_freed = sum(r.stats.space_freed_mb for r in results)
        if total_space_freed < 100:  # Less than 100MB freed
            recommendations.append("Low space savings - consider reviewing retention policies")
        
        # Failed operations
        failed_operations = [r for r in results if r.status == 'failed']
        if failed_operations:
            recommendations.append(f"Investigate failed operations: {[r.operation_name for r in failed_operations]}")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Cleanup completed successfully - continue regular maintenance schedule")
        
        return recommendations
    
    def _save_cleanup_results(self, report: MonthlyCleanupReport) -> None:
        """Save cleanup results to files"""
        try:
            # Save detailed report
            report_file = OUTPUTS_DIR / 'reports' / f"monthly_cleanup_{report.execution_date.replace('-', '')}.json"
            report_file.parent.mkdir(exist_ok=True)
            report.save(report_file)
            
            # Save summary for trending
            summary_file = OUTPUTS_DIR / 'reports' / "monthly_cleanup_summary.json"
            self._update_summary_log(summary_file, report)
            
            logger.info(f"💾 Saved cleanup report: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save cleanup results: {e}")
    
    def _update_summary_log(self, summary_file: Path, report: MonthlyCleanupReport) -> None:
        """Update summary log with latest results"""
        try:
            # Load existing summary
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
            else:
                summary_data = {'monthly_cleanups': []}
            
            # Add current report summary
            summary_data['monthly_cleanups'].append({
                'date': report.execution_date,
                'status': report.overall_status,
                'execution_time': report.total_execution_time,
                'files_processed': report.total_files_processed,
                'space_freed_mb': report.total_space_freed_mb,
                'errors': report.total_errors
            })
            
            # Keep only last 12 months
            summary_data['monthly_cleanups'] = summary_data['monthly_cleanups'][-12:]
            
            # Save updated summary
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to update summary log: {e}")
    
    def _print_summary(self, report: MonthlyCleanupReport) -> None:
        """Print cleanup summary to console"""
        print("\n" + "="*60)
        print("MONTHLY CLEANUP SUMMARY")
        print("="*60)
        print(f"Date: {report.execution_date}")
        print(f"Status: {report.overall_status.upper()}")
        print(f"Execution Time: {report.total_execution_time/60:.1f} minutes")
        
        print(f"\nCleanup Results:")
        print(f"  📁 Files Processed: {report.total_files_processed}")
        print(f"  💾 Space Freed: {report.total_space_freed_mb/1024:.2f} GB")
        print(f"  ❌ Errors: {report.total_errors}")
        
        print(f"\nDisk Space:")
        before = report.disk_space_before
        after = report.disk_space_after
        print(f"  Before: {before.get('free_gb', 0):.1f} GB free ({before.get('percent_used', 0):.1f}% used)")
        print(f"  After:  {after.get('free_gb', 0):.1f} GB free ({after.get('percent_used', 0):.1f}% used)")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Show operation details
        print(f"\nOperation Details:")
        print("-" * 40)
        for result in report.cleanup_results:
            status_emoji = "✅" if result.status == 'success' else "⚠️" if result.status == 'partial_success' else "❌"
            print(f"{status_emoji} {result.operation_name:20}: {result.message}")
    
    def _create_failure_report(self, error_message: str) -> MonthlyCleanupReport:
        """Create failure report"""
        execution_time = time.time() - self.start_time if self.start_time else 0
        
        return MonthlyCleanupReport(
            execution_date=datetime.now().strftime('%Y-%m-%d'),
            total_execution_time=execution_time,
            total_files_processed=0,
            total_space_freed_mb=0.0,
            total_errors=1,
            cleanup_results=[],
            disk_space_before={},
            disk_space_after={},
            overall_status='failed',
            recommendations=[f"Address critical error: {error_message}"]
        )

def load_config_from_file(config_path: str) -> CleanupConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return CleanupConfig(**config_dict)
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        return CleanupConfig()

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='StockPredictionPro Monthly Cleanup')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--dry-run', action='store_true', help='Simulate cleanup without making changes')
    parser.add_argument('--operation', choices=['data', 'logs', 'models', 'outputs', 'system'],
                       help='Run specific cleanup operation only')
    parser.add_argument('--no-backup', action='store_true', help='Skip backing up files before deletion')
    parser.add_argument('--aggressive', action='store_true', help='Use more aggressive cleanup settings')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = CleanupConfig()
    
    # Override config with command line arguments
    if args.dry_run:
        config.dry_run_mode = True
        logger.info("🔍 DRY RUN MODE - No files will be deleted")
    
    if args.no_backup:
        config.backup_before_deletion = False
        config.backup_models_before_cleanup = False
    
    if args.aggressive:
        # More aggressive cleanup settings
        config.raw_data_retention_days = 180  # 6 months instead of 1 year
        config.processed_data_retention_days = 90  # 3 months instead of 6
        config.log_retention_days = 30  # 1 month instead of 3
        config.model_retention_count = 3  # 3 versions instead of 5
        logger.info("🔥 Using aggressive cleanup settings")
    
    # Run cleanup
    orchestrator = MonthlyCleanupOrchestrator(config)
    
    if args.operation:
        # Run specific operation only
        logger.info(f"Running {args.operation} cleanup only")
        
        if args.operation == 'data':
            results = orchestrator.data_cleaner.cleanup_data_directories()
        elif args.operation == 'logs':
            results = [orchestrator.log_cleaner.cleanup_logs()]
        elif args.operation == 'models':
            results = [orchestrator.model_cleaner.cleanup_models()]
        elif args.operation == 'outputs':
            results = [orchestrator.output_cleaner.cleanup_outputs()]
        elif args.operation == 'system':
            results = [orchestrator.system_optimizer.optimize_system()]
        
        # Print results
        for result in results:
            status_emoji = "✅" if result.status == 'success' else "⚠️" if result.status == 'partial_success' else "❌"
            print(f"{status_emoji} {result.operation_name}: {result.message}")
        
    else:
        # Run complete monthly cleanup
        report = orchestrator.run_monthly_cleanup()
        
        # Exit with appropriate code
        if report.overall_status == 'success':
            sys.exit(0)
        elif report.overall_status == 'partial_success':
            sys.exit(1)
        else:
            sys.exit(2)

if __name__ == '__main__':
    main()
