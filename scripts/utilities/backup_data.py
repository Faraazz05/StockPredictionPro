"""
scripts/utilities/backup_data.py

Comprehensive data backup system for StockPredictionPro.
Handles database backups, file system backups, model versioning,
and cloud storage integration with encryption and compression.

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
import hashlib
import tarfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import subprocess
import tempfile

# Cloud storage libraries (optional)
try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_AWS = True
except ImportError:
    HAS_AWS = False

try:
    from google.cloud import storage as gcs
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

try:
    from azure.storage.blob import BlobServiceClient
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False

# Database libraries
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# Encryption library
try:
    from cryptography.fernet import Fernet
    HAS_ENCRYPTION = True
except ImportError:
    HAS_ENCRYPTION = False

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'backup_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StockPredictionPro.Backup')

# Directory configuration
PROJECT_ROOT = Path('.')
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'
CONFIG_DIR = PROJECT_ROOT / 'config'
BACKUP_DIR = PROJECT_ROOT / 'backups'
TEMP_DIR = PROJECT_ROOT / 'temp'

# Ensure directories exist
for dir_path in [BACKUP_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class BackupConfig:
    """Configuration for backup operations"""
    # Backup settings
    backup_name: str = 'stockpro_backup'
    backup_type: str = 'full'  # full, incremental, differential
    compression: str = 'gzip'  # gzip, zip, tar, none
    encryption: bool = False
    
    # Data sources
    include_database: bool = True
    include_files: bool = True
    include_models: bool = True
    include_logs: bool = False
    include_config: bool = True
    
    # File system backup
    data_directories: List[str] = None
    exclude_patterns: List[str] = None
    max_file_size_mb: int = 1000  # Skip files larger than 1GB
    
    # Database backup
    database_url: Optional[str] = None
    database_type: str = 'postgresql'  # postgresql, mysql, sqlite
    backup_format: str = 'sql'  # sql, binary
    
    # Cloud storage
    cloud_provider: Optional[str] = None  # aws, gcs, azure
    cloud_bucket: Optional[str] = None
    cloud_region: str = 'us-east-1'
    
    # Retention settings
    local_retention_days: int = 30
    cloud_retention_days: int = 90
    max_local_backups: int = 10
    max_cloud_backups: int = 50
    
    # Performance settings
    parallel_uploads: int = 4
    chunk_size_mb: int = 100
    verify_backups: bool = True
    
    def __post_init__(self):
        if self.data_directories is None:
            self.data_directories = ['data', 'models', 'config']
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                '*.tmp', '*.log', '__pycache__', '.git',
                '*.pyc', '.pytest_cache', 'node_modules'
            ]

@dataclass
class BackupItem:
    """Individual backup item"""
    name: str
    path: str
    size_bytes: int
    checksum: str
    backup_time: datetime
    compressed: bool = False
    encrypted: bool = False

@dataclass
class BackupResult:
    """Result of backup operation"""
    backup_id: str
    backup_name: str
    backup_type: str
    start_time: datetime
    end_time: datetime
    duration: float
    
    # Backup content
    items: List[BackupItem]
    total_size_bytes: int
    compressed_size_bytes: int
    
    # Storage locations
    local_path: Optional[str] = None
    cloud_path: Optional[str] = None
    
    # Status
    status: str = 'success'  # success, failed, partial
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    # Verification
    verified: bool = False
    verification_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save_manifest(self, path: Path) -> None:
        """Save backup manifest"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

# ============================================
# BACKUP COMPONENTS
# ============================================

class FileSystemBackup:
    """Handle file system backups"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
    
    def create_backup(self, backup_id: str) -> Tuple[List[BackupItem], str]:
        """Create file system backup"""
        logger.info("Creating file system backup...")
        
        backup_items = []
        backup_path = BACKUP_DIR / f"{backup_id}_filesystem.tar.gz"
        
        try:
            with tarfile.open(backup_path, 'w:gz' if self.config.compression == 'gzip' else 'w') as tar:
                for directory in self.config.data_directories:
                    dir_path = PROJECT_ROOT / directory
                    
                    if not dir_path.exists():
                        logger.warning(f"Directory not found: {dir_path}")
                        continue
                    
                    logger.info(f"Backing up directory: {dir_path}")
                    
                    for file_path in self._get_files_to_backup(dir_path):
                        try:
                            # Check file size
                            file_size = file_path.stat().st_size
                            if file_size > self.config.max_file_size_mb * 1024 * 1024:
                                logger.warning(f"Skipping large file: {file_path} ({file_size/1024/1024:.1f} MB)")
                                continue
                            
                            # Add to archive
                            arcname = str(file_path.relative_to(PROJECT_ROOT))
                            tar.add(file_path, arcname=arcname)
                            
                            # Create backup item
                            checksum = self._calculate_checksum(file_path)
                            
                            backup_item = BackupItem(
                                name=arcname,
                                path=str(file_path),
                                size_bytes=file_size,
                                checksum=checksum,
                                backup_time=datetime.now(),
                                compressed=True
                            )
                            backup_items.append(backup_item)
                            
                        except Exception as e:
                            logger.error(f"Failed to backup file {file_path}: {e}")
                            continue
            
            logger.info(f"File system backup created: {backup_path}")
            return backup_items, str(backup_path)
            
        except Exception as e:
            logger.error(f"File system backup failed: {e}")
            raise
    
    def _get_files_to_backup(self, directory: Path) -> List[Path]:
        """Get list of files to backup"""
        files_to_backup = []
        
        for file_path in directory.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Check exclude patterns
            if self._should_exclude_file(file_path):
                continue
            
            files_to_backup.append(file_path)
        
        return files_to_backup
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded"""
        file_str = str(file_path)
        
        for pattern in self.config.exclude_patterns:
            if pattern.startswith('*'):
                # Wildcard pattern
                suffix = pattern[1:]
                if file_str.endswith(suffix):
                    return True
            elif pattern in file_str:
                return True
        
        return False
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ''

class DatabaseBackup:
    """Handle database backups"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
    
    def create_backup(self, backup_id: str) -> Tuple[List[BackupItem], Optional[str]]:
        """Create database backup"""
        if not self.config.include_database or not self.config.database_url:
            logger.info("Database backup skipped")
            return [], None
        
        logger.info("Creating database backup...")
        
        try:
            if self.config.database_type == 'postgresql':
                return self._backup_postgresql(backup_id)
            elif self.config.database_type == 'mysql':
                return self._backup_mysql(backup_id)
            elif self.config.database_type == 'sqlite':
                return self._backup_sqlite(backup_id)
            else:
                logger.warning(f"Unsupported database type: {self.config.database_type}")
                return [], None
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise
    
    def _backup_postgresql(self, backup_id: str) -> Tuple[List[BackupItem], str]:
        """Backup PostgreSQL database"""
        backup_path = BACKUP_DIR / f"{backup_id}_database.sql"
        
        # Parse database URL
        db_url = self.config.database_url
        
        # Use pg_dump
        cmd = [
            'pg_dump',
            db_url,
            '--file', str(backup_path),
            '--verbose',
            '--no-password'
        ]
        
        if self.config.backup_format == 'binary':
            cmd.extend(['--format=custom'])
            backup_path = backup_path.with_suffix('.dump')
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"pg_dump failed: {result.stderr}")
        
        # Create backup item
        file_size = backup_path.stat().st_size
        checksum = self._calculate_file_checksum(backup_path)
        
        backup_item = BackupItem(
            name=backup_path.name,
            path=str(backup_path),
            size_bytes=file_size,
            checksum=checksum,
            backup_time=datetime.now()
        )
        
        logger.info(f"PostgreSQL backup created: {backup_path}")
        return [backup_item], str(backup_path)
    
    def _backup_mysql(self, backup_id: str) -> Tuple[List[BackupItem], str]:
        """Backup MySQL database"""
        backup_path = BACKUP_DIR / f"{backup_id}_database.sql"
        
        # Parse connection details from URL
        # This is a simplified implementation
        cmd = [
            'mysqldump',
            '--single-transaction',
            '--routines',
            '--triggers',
            '--all-databases',
            '--result-file', str(backup_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"mysqldump failed: {result.stderr}")
        
        file_size = backup_path.stat().st_size
        checksum = self._calculate_file_checksum(backup_path)
        
        backup_item = BackupItem(
            name=backup_path.name,
            path=str(backup_path),
            size_bytes=file_size,
            checksum=checksum,
            backup_time=datetime.now()
        )
        
        logger.info(f"MySQL backup created: {backup_path}")
        return [backup_item], str(backup_path)
    
    def _backup_sqlite(self, backup_id: str) -> Tuple[List[BackupItem], str]:
        """Backup SQLite database"""
        # Extract database file path from URL
        if self.config.database_url.startswith('sqlite:///'):
            db_file = Path(self.config.database_url[10:])
        else:
            raise ValueError("Invalid SQLite database URL")
        
        if not db_file.exists():
            raise FileNotFoundError(f"SQLite database file not found: {db_file}")
        
        backup_path = BACKUP_DIR / f"{backup_id}_database.sqlite"
        
        # Copy SQLite file
        shutil.copy2(db_file, backup_path)
        
        file_size = backup_path.stat().st_size
        checksum = self._calculate_file_checksum(backup_path)
        
        backup_item = BackupItem(
            name=backup_path.name,
            path=str(backup_path),
            size_bytes=file_size,
            checksum=checksum,
            backup_time=datetime.now()
        )
        
        logger.info(f"SQLite backup created: {backup_path}")
        return [backup_item], str(backup_path)
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

class ModelBackup:
    """Handle model file backups"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
    
    def create_backup(self, backup_id: str) -> Tuple[List[BackupItem], Optional[str]]:
        """Create model backup"""
        if not self.config.include_models:
            logger.info("Model backup skipped")
            return [], None
        
        logger.info("Creating model backup...")
        
        backup_items = []
        backup_path = BACKUP_DIR / f"{backup_id}_models.tar.gz"
        
        try:
            with tarfile.open(backup_path, 'w:gz') as tar:
                models_dir = MODELS_DIR
                
                if not models_dir.exists():
                    logger.warning(f"Models directory not found: {models_dir}")
                    return [], None
                
                for model_file in models_dir.rglob('*.pkl'):
                    try:
                        # Add model file
                        arcname = str(model_file.relative_to(PROJECT_ROOT))
                        tar.add(model_file, arcname=arcname)
                        
                        # Create backup item
                        file_size = model_file.stat().st_size
                        checksum = self._calculate_checksum(model_file)
                        
                        backup_item = BackupItem(
                            name=arcname,
                            path=str(model_file),
                            size_bytes=file_size,
                            checksum=checksum,
                            backup_time=datetime.now(),
                            compressed=True
                        )
                        backup_items.append(backup_item)
                        
                    except Exception as e:
                        logger.error(f"Failed to backup model {model_file}: {e}")
                
                # Include model registry if exists
                registry_file = models_dir / 'model_registry.json'
                if registry_file.exists():
                    arcname = str(registry_file.relative_to(PROJECT_ROOT))
                    tar.add(registry_file, arcname=arcname)
                    
                    file_size = registry_file.stat().st_size
                    checksum = self._calculate_checksum(registry_file)
                    
                    backup_item = BackupItem(
                        name=arcname,
                        path=str(registry_file),
                        size_bytes=file_size,
                        checksum=checksum,
                        backup_time=datetime.now(),
                        compressed=True
                    )
                    backup_items.append(backup_item)
            
            logger.info(f"Model backup created: {backup_path}")
            return backup_items, str(backup_path)
            
        except Exception as e:
            logger.error(f"Model backup failed: {e}")
            raise
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

class CloudStorage:
    """Handle cloud storage operations"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
    
    def upload_backup(self, local_path: str, backup_id: str) -> Optional[str]:
        """Upload backup to cloud storage"""
        if not self.config.cloud_provider:
            return None
        
        try:
            if self.config.cloud_provider == 'aws':
                return self._upload_to_s3(local_path, backup_id)
            elif self.config.cloud_provider == 'gcs':
                return self._upload_to_gcs(local_path, backup_id)
            elif self.config.cloud_provider == 'azure':
                return self._upload_to_azure(local_path, backup_id)
            else:
                logger.warning(f"Unsupported cloud provider: {self.config.cloud_provider}")
                return None
                
        except Exception as e:
            logger.error(f"Cloud upload failed: {e}")
            return None
    
    def _upload_to_s3(self, local_path: str, backup_id: str) -> str:
        """Upload to AWS S3"""
        if not HAS_AWS:
            raise ImportError("boto3 required for AWS S3 uploads")
        
        s3_client = boto3.client('s3', region_name=self.config.cloud_region)
        
        key = f"backups/{datetime.now().strftime('%Y/%m/%d')}/{backup_id}/{Path(local_path).name}"
        
        # Upload file
        s3_client.upload_file(local_path, self.config.cloud_bucket, key)
        
        cloud_path = f"s3://{self.config.cloud_bucket}/{key}"
        logger.info(f"Uploaded to S3: {cloud_path}")
        
        return cloud_path
    
    def _upload_to_gcs(self, local_path: str, backup_id: str) -> str:
        """Upload to Google Cloud Storage"""
        if not HAS_GCS:
            raise ImportError("google-cloud-storage required for GCS uploads")
        
        client = gcs.Client()
        bucket = client.bucket(self.config.cloud_bucket)
        
        blob_name = f"backups/{datetime.now().strftime('%Y/%m/%d')}/{backup_id}/{Path(local_path).name}"
        blob = bucket.blob(blob_name)
        
        # Upload file
        blob.upload_from_filename(local_path)
        
        cloud_path = f"gs://{self.config.cloud_bucket}/{blob_name}"
        logger.info(f"Uploaded to GCS: {cloud_path}")
        
        return cloud_path
    
    def _upload_to_azure(self, local_path: str, backup_id: str) -> str:
        """Upload to Azure Blob Storage"""
        if not HAS_AZURE:
            raise ImportError("azure-storage-blob required for Azure uploads")
        
        blob_service_client = BlobServiceClient.from_connection_string(
            os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        )
        
        blob_name = f"backups/{datetime.now().strftime('%Y/%m/%d')}/{backup_id}/{Path(local_path).name}"
        
        # Upload file
        with open(local_path, 'rb') as data:
            blob_service_client.get_blob_client(
                container=self.config.cloud_bucket,
                blob=blob_name
            ).upload_blob(data, overwrite=True)
        
        cloud_path = f"azure://{self.config.cloud_bucket}/{blob_name}"
        logger.info(f"Uploaded to Azure: {cloud_path}")
        
        return cloud_path

class BackupEncryption:
    """Handle backup encryption"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.key = None
        
        if config.encryption and HAS_ENCRYPTION:
            self.key = self._get_encryption_key()
    
    def encrypt_file(self, file_path: str) -> str:
        """Encrypt backup file"""
        if not self.config.encryption or not HAS_ENCRYPTION:
            return file_path
        
        try:
            fernet = Fernet(self.key)
            encrypted_path = f"{file_path}.encrypted"
            
            with open(file_path, 'rb') as f:
                data = f.read()
            
            encrypted_data = fernet.encrypt(data)
            
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Remove original file
            os.remove(file_path)
            
            logger.info(f"File encrypted: {encrypted_path}")
            return encrypted_path
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return file_path
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key"""
        key_file = CONFIG_DIR / 'backup_key.key'
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            
            # Save key securely
            key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
            logger.info(f"Generated new encryption key: {key_file}")
            return key

# ============================================
# MAIN BACKUP ORCHESTRATOR
# ============================================

class BackupOrchestrator:
    """Main orchestrator for backup operations"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.filesystem_backup = FileSystemBackup(config)
        self.database_backup = DatabaseBackup(config)
        self.model_backup = ModelBackup(config)
        self.cloud_storage = CloudStorage(config)
        self.encryption = BackupEncryption(config)
    
    def create_backup(self) -> BackupResult:
        """Create comprehensive backup"""
        backup_id = f"{self.config.backup_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"🔄 Starting backup: {backup_id}")
        
        all_items = []
        backup_files = []
        warnings = []
        
        try:
            # File system backup
            if self.config.include_files:
                try:
                    fs_items, fs_path = self.filesystem_backup.create_backup(backup_id)
                    all_items.extend(fs_items)
                    if fs_path:
                        backup_files.append(fs_path)
                except Exception as e:
                    logger.error(f"File system backup failed: {e}")
                    warnings.append(f"File system backup failed: {e}")
            
            # Database backup
            if self.config.include_database:
                try:
                    db_items, db_path = self.database_backup.create_backup(backup_id)
                    all_items.extend(db_items)
                    if db_path:
                        backup_files.append(db_path)
                except Exception as e:
                    logger.error(f"Database backup failed: {e}")
                    warnings.append(f"Database backup failed: {e}")
            
            # Model backup
            if self.config.include_models:
                try:
                    model_items, model_path = self.model_backup.create_backup(backup_id)
                    all_items.extend(model_items)
                    if model_path:
                        backup_files.append(model_path)
                except Exception as e:
                    logger.error(f"Model backup failed: {e}")
                    warnings.append(f"Model backup failed: {e}")
            
            # Calculate sizes
            total_size = sum(item.size_bytes for item in all_items)
            compressed_size = sum(Path(f).stat().st_size for f in backup_files if Path(f).exists())
            
            # Create consolidated backup
            consolidated_path = None
            if len(backup_files) > 1:
                consolidated_path = self._create_consolidated_backup(backup_id, backup_files)
                backup_files = [consolidated_path]
            elif backup_files:
                consolidated_path = backup_files[0]
            
            # Encrypt if configured
            if consolidated_path and self.config.encryption:
                encrypted_path = self.encryption.encrypt_file(consolidated_path)
                backup_files = [encrypted_path]
                consolidated_path = encrypted_path
            
            # Upload to cloud
            cloud_path = None
            if consolidated_path and self.config.cloud_provider:
                cloud_path = self.cloud_storage.upload_backup(consolidated_path, backup_id)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Create result
            result = BackupResult(
                backup_id=backup_id,
                backup_name=self.config.backup_name,
                backup_type=self.config.backup_type,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                items=all_items,
                total_size_bytes=total_size,
                compressed_size_bytes=compressed_size,
                local_path=consolidated_path,
                cloud_path=cloud_path,
                status='success' if not warnings else 'partial',
                warnings=warnings
            )
            
            # Save backup manifest
            manifest_path = BACKUP_DIR / f"{backup_id}_manifest.json"
            result.save_manifest(manifest_path)
            
            # Verify backup if configured
            if self.config.verify_backups and consolidated_path:
                result.verified = self._verify_backup(consolidated_path)
                result.verification_time = datetime.now()
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            logger.info(f"✅ Backup completed: {backup_id}")
            logger.info(f"   Duration: {duration:.1f}s")
            logger.info(f"   Size: {total_size/1024/1024:.1f} MB -> {compressed_size/1024/1024:.1f} MB")
            logger.info(f"   Items: {len(all_items)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            
            return BackupResult(
                backup_id=backup_id,
                backup_name=self.config.backup_name,
                backup_type=self.config.backup_type,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                items=[],
                total_size_bytes=0,
                compressed_size_bytes=0,
                status='failed',
                error_message=str(e)
            )
    
    def _create_consolidated_backup(self, backup_id: str, backup_files: List[str]) -> str:
        """Create consolidated backup from multiple files"""
        consolidated_path = BACKUP_DIR / f"{backup_id}_consolidated.tar.gz"
        
        with tarfile.open(consolidated_path, 'w:gz') as tar:
            for backup_file in backup_files:
                if Path(backup_file).exists():
                    tar.add(backup_file, arcname=Path(backup_file).name)
        
        # Remove individual backup files
        for backup_file in backup_files:
            try:
                os.remove(backup_file)
            except OSError:
                pass
        
        logger.info(f"Created consolidated backup: {consolidated_path}")
        return str(consolidated_path)
    
    def _verify_backup(self, backup_path: str) -> bool:
        """Verify backup integrity"""
        try:
            logger.info("Verifying backup integrity...")
            
            # For tar.gz files, try to list contents
            if backup_path.endswith('.tar.gz'):
                with tarfile.open(backup_path, 'r:gz') as tar:
                    tar.getnames()  # This will raise exception if corrupted
            elif backup_path.endswith('.zip'):
                with zipfile.ZipFile(backup_path, 'r') as zip_file:
                    zip_file.testzip()  # Returns None if valid
            else:
                # For other files, just try to read
                with open(backup_path, 'rb') as f:
                    f.read(1024)  # Read first 1KB
            
            logger.info("✅ Backup verification passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup verification failed: {e}")
            return False
    
    def _cleanup_old_backups(self) -> None:
        """Clean up old backup files"""
        try:
            logger.info("Cleaning up old backups...")
            
            # Get all backup files
            backup_files = list(BACKUP_DIR.glob(f"{self.config.backup_name}_*"))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Remove old local backups
            if len(backup_files) > self.config.max_local_backups:
                files_to_remove = backup_files[self.config.max_local_backups:]
                
                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        logger.info(f"Removed old backup: {file_path}")
                    except OSError as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
            
            # Remove backups older than retention period
            cutoff_date = datetime.now() - timedelta(days=self.config.local_retention_days)
            
            for file_path in backup_files:
                try:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        file_path.unlink()
                        logger.info(f"Removed expired backup: {file_path}")
                except OSError as e:
                    logger.warning(f"Failed to remove expired backup {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        try:
            # Find manifest files
            manifest_files = list(BACKUP_DIR.glob("*_manifest.json"))
            
            for manifest_file in sorted(manifest_files, reverse=True):
                try:
                    with open(manifest_file, 'r') as f:
                        backup_data = json.load(f)
                    
                    backups.append({
                        'backup_id': backup_data.get('backup_id'),
                        'backup_name': backup_data.get('backup_name'),
                        'start_time': backup_data.get('start_time'),
                        'duration': backup_data.get('duration'),
                        'status': backup_data.get('status'),
                        'total_size_mb': backup_data.get('total_size_bytes', 0) / 1024 / 1024,
                        'items_count': len(backup_data.get('items', [])),
                        'local_path': backup_data.get('local_path'),
                        'cloud_path': backup_data.get('cloud_path')
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to read manifest {manifest_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
        
        return backups

def load_config_from_file(config_path: str) -> BackupConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return BackupConfig(**config_dict)
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        return BackupConfig()

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backup StockPredictionPro data')
    parser.add_argument('--config', help='Path to backup configuration JSON file')
    parser.add_argument('--name', default='stockpro_backup', help='Backup name')
    parser.add_argument('--type', choices=['full', 'incremental', 'differential'], 
                       default='full', help='Backup type')
    parser.add_argument('--compression', choices=['gzip', 'zip', 'tar', 'none'],
                       default='gzip', help='Compression type')
    parser.add_argument('--encrypt', action='store_true', help='Enable encryption')
    parser.add_argument('--no-database', action='store_true', help='Skip database backup')
    parser.add_argument('--no-files', action='store_true', help='Skip file backup')
    parser.add_argument('--no-models', action='store_true', help='Skip model backup')
    parser.add_argument('--database-url', help='Database URL for backup')
    parser.add_argument('--cloud-provider', choices=['aws', 'gcs', 'azure'], 
                       help='Cloud storage provider')
    parser.add_argument('--cloud-bucket', help='Cloud storage bucket')
    parser.add_argument('--list', action='store_true', help='List available backups')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = BackupConfig()
    
    # Override config with command line arguments
    config.backup_name = args.name
    config.backup_type = args.type
    config.compression = args.compression
    config.encryption = args.encrypt
    
    if args.no_database:
        config.include_database = False
    if args.no_files:
        config.include_files = False
    if args.no_models:
        config.include_models = False
    
    if args.database_url:
        config.database_url = args.database_url
    if args.cloud_provider:
        config.cloud_provider = args.cloud_provider
    if args.cloud_bucket:
        config.cloud_bucket = args.cloud_bucket
    
    try:
        orchestrator = BackupOrchestrator(config)
        
        if args.list:
            # List available backups
            backups = orchestrator.list_backups()
            
            print("\n" + "="*60)
            print("AVAILABLE BACKUPS")
            print("="*60)
            
            if not backups:
                print("No backups found")
            else:
                for backup in backups:
                    print(f"\n{backup['backup_id']}")
                    print(f"  Status: {backup['status']}")
                    print(f"  Date: {backup['start_time']}")
                    print(f"  Duration: {backup['duration']:.1f}s")
                    print(f"  Size: {backup['total_size_mb']:.1f} MB")
                    print(f"  Items: {backup['items_count']}")
                    if backup['local_path']:
                        print(f"  Local: {backup['local_path']}")
                    if backup['cloud_path']:
                        print(f"  Cloud: {backup['cloud_path']}")
        else:
            # Create backup
            result = orchestrator.create_backup()
            
            if result.status == 'success':
                print(f"\n✅ Backup completed successfully!")
                print(f"Backup ID: {result.backup_id}")
                print(f"Duration: {result.duration:.1f} seconds")
                print(f"Items backed up: {len(result.items)}")
                print(f"Size: {result.total_size_bytes/1024/1024:.1f} MB")
                
                if result.local_path:
                    print(f"Local path: {result.local_path}")
                if result.cloud_path:
                    print(f"Cloud path: {result.cloud_path}")
                
                sys.exit(0)
            else:
                print(f"\n❌ Backup failed or completed with warnings")
                if result.error_message:
                    print(f"Error: {result.error_message}")
                if result.warnings:
                    print("Warnings:")
                    for warning in result.warnings:
                        print(f"  - {warning}")
                
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n❌ Backup interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        print(f"❌ Backup failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
