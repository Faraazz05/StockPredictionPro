# ============================================
# StockPredictionPro - src/features/feature_store.py
# Advanced feature storage and retrieval system for financial machine learning
# ============================================

import numpy as np
import pandas as pd
import pickle
import joblib
import json
import sqlite3
import h5py
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import shutil
from abc import ABC, abstractmethod

from ..utils.exceptions import ValidationError, CalculationError
from ..utils.logger import get_logger
from ..utils.timing import time_it

logger = get_logger('features.feature_store')

# ============================================
# Configuration and Data Classes
# ============================================

@dataclass
class FeatureMetadata:
    """Metadata for stored features"""
    feature_name: str
    feature_type: str  # 'indicator', 'target', 'transformed'
    data_type: str     # 'numerical', 'categorical', 'binary'
    creation_date: datetime
    last_updated: datetime
    version: str = "1.0"
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    data_shape: Optional[Tuple[int, ...]] = None
    data_range: Optional[Tuple[float, float]] = None
    missing_count: int = 0
    checksum: str = ""
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['creation_date'] = self.creation_date.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary"""
        data['creation_date'] = datetime.fromisoformat(data['creation_date'])
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

@dataclass
class FeatureStoreConfig:
    """Configuration for feature store"""
    base_path: str = "./feature_store"
    storage_format: str = "hdf5"  # 'hdf5', 'parquet', 'pickle', 'sqlite'
    compression: str = "gzip"
    max_memory_usage: int = 1024  # MB
    enable_versioning: bool = True
    enable_caching: bool = True
    cache_size: int = 100  # Number of features to cache
    backup_enabled: bool = True
    backup_frequency: str = "daily"  # 'hourly', 'daily', 'weekly'
    parallel_io: bool = True
    max_workers: int = 4
    auto_cleanup: bool = True
    cleanup_days: int = 30
    enable_compression: bool = True
    validate_checksums: bool = True

# ============================================
# Abstract Base Storage Backend
# ============================================

class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def save_feature(self, feature_name: str, data: np.ndarray, metadata: FeatureMetadata) -> bool:
        """Save feature data and metadata"""
        pass
    
    @abstractmethod
    def load_feature(self, feature_name: str, version: Optional[str] = None) -> Tuple[np.ndarray, FeatureMetadata]:
        """Load feature data and metadata"""
        pass
    
    @abstractmethod
    def delete_feature(self, feature_name: str, version: Optional[str] = None) -> bool:
        """Delete feature"""
        pass
    
    @abstractmethod
    def list_features(self) -> List[str]:
        """List all available features"""
        pass
    
    @abstractmethod
    def feature_exists(self, feature_name: str, version: Optional[str] = None) -> bool:
        """Check if feature exists"""
        pass
    
    @abstractmethod
    def get_feature_metadata(self, feature_name: str, version: Optional[str] = None) -> FeatureMetadata:
        """Get feature metadata"""
        pass

# ============================================
# HDF5 Storage Backend
# ============================================

class HDF5Backend(StorageBackend):
    """HDF5-based storage backend for efficient numerical data storage"""
    
    def __init__(self, config: FeatureStoreConfig):
        super().__init__(config)
        self.data_file = self.base_path / "features.h5"
        self.metadata_file = self.base_path / "metadata.json"
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Ensure storage files exist"""
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)
    
    def _load_all_metadata(self) -> Dict[str, FeatureMetadata]:
        """Load all metadata from file"""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            return {
                name: FeatureMetadata.from_dict(meta_data)
                for name, meta_data in metadata_dict.items()
            }
        except Exception as e:
            logger.warning(f"Error loading metadata: {e}")
            return {}
    
    def _save_all_metadata(self, metadata_dict: Dict[str, FeatureMetadata]):
        """Save all metadata to file"""
        try:
            serializable_dict = {
                name: metadata.to_dict()
                for name, metadata in metadata_dict.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(serializable_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _calculate_checksum(self, data: np.ndarray) -> str:
        """Calculate checksum for data integrity"""
        return hashlib.md5(data.tobytes()).hexdigest()
    
    def save_feature(self, feature_name: str, data: np.ndarray, metadata: FeatureMetadata) -> bool:
        """Save feature data to HDF5 file"""
        try:
            # Calculate checksum
            metadata.checksum = self._calculate_checksum(data)
            metadata.data_shape = data.shape
            metadata.last_updated = datetime.now()
            
            # Calculate data range for numerical data
            if np.issubdtype(data.dtype, np.number):
                finite_data = data[np.isfinite(data)]
                if len(finite_data) > 0:
                    metadata.data_range = (float(np.min(finite_data)), float(np.max(finite_data)))
                metadata.missing_count = int(np.sum(~np.isfinite(data)))
            
            # Save data to HDF5
            with h5py.File(self.data_file, 'a') as f:
                # Delete existing dataset if it exists
                if feature_name in f:
                    del f[feature_name]
                
                # Create dataset with compression if enabled
                if self.config.enable_compression:
                    f.create_dataset(
                        feature_name, 
                        data=data, 
                        compression=self.config.compression,
                        shuffle=True,
                        fletcher32=True
                    )
                else:
                    f.create_dataset(feature_name, data=data)
            
            # Update metadata
            all_metadata = self._load_all_metadata()
            all_metadata[feature_name] = metadata
            self._save_all_metadata(all_metadata)
            
            logger.info(f"Successfully saved feature '{feature_name}' with shape {data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving feature '{feature_name}': {e}")
            return False
    
    def load_feature(self, feature_name: str, version: Optional[str] = None) -> Tuple[np.ndarray, FeatureMetadata]:
        """Load feature data from HDF5 file"""
        try:
            # Load metadata
            all_metadata = self._load_all_metadata()
            
            if feature_name not in all_metadata:
                raise ValueError(f"Feature '{feature_name}' not found")
            
            metadata = all_metadata[feature_name]
            
            # Load data from HDF5
            with h5py.File(self.data_file, 'r') as f:
                if feature_name not in f:
                    raise ValueError(f"Feature data '{feature_name}' not found in storage")
                
                data = f[feature_name][:]
            
            # Validate checksum if enabled
            if self.config.validate_checksums and metadata.checksum:
                actual_checksum = self._calculate_checksum(data)
                if actual_checksum != metadata.checksum:
                    logger.warning(f"Checksum mismatch for feature '{feature_name}'")
            
            logger.info(f"Successfully loaded feature '{feature_name}' with shape {data.shape}")
            return data, metadata
            
        except Exception as e:
            logger.error(f"Error loading feature '{feature_name}': {e}")
            raise
    
    def delete_feature(self, feature_name: str, version: Optional[str] = None) -> bool:
        """Delete feature from storage"""
        try:
            # Remove from HDF5
            with h5py.File(self.data_file, 'a') as f:
                if feature_name in f:
                    del f[feature_name]
            
            # Remove from metadata
            all_metadata = self._load_all_metadata()
            if feature_name in all_metadata:
                del all_metadata[feature_name]
                self._save_all_metadata(all_metadata)
            
            logger.info(f"Successfully deleted feature '{feature_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting feature '{feature_name}': {e}")
            return False
    
    def list_features(self) -> List[str]:
        """List all available features"""
        all_metadata = self._load_all_metadata()
        return list(all_metadata.keys())
    
    def feature_exists(self, feature_name: str, version: Optional[str] = None) -> bool:
        """Check if feature exists"""
        all_metadata = self._load_all_metadata()
        return feature_name in all_metadata
    
    def get_feature_metadata(self, feature_name: str, version: Optional[str] = None) -> FeatureMetadata:
        """Get feature metadata"""
        all_metadata = self._load_all_metadata()
        
        if feature_name not in all_metadata:
            raise ValueError(f"Feature '{feature_name}' not found")
        
        return all_metadata[feature_name]

# ============================================
# Parquet Storage Backend
# ============================================

class ParquetBackend(StorageBackend):
    """Parquet-based storage backend for tabular data"""
    
    def __init__(self, config: FeatureStoreConfig):
        super().__init__(config)
        self.data_dir = self.base_path / "parquet_data"
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_file = self.base_path / "parquet_metadata.json"
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Ensure storage files exist"""
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)
    
    def _get_feature_path(self, feature_name: str) -> Path:
        """Get file path for feature"""
        return self.data_dir / f"{feature_name}.parquet"
    
    def save_feature(self, feature_name: str, data: np.ndarray, metadata: FeatureMetadata) -> bool:
        """Save feature data to Parquet file"""
        try:
            # Convert to DataFrame
            if data.ndim == 1:
                df = pd.DataFrame({feature_name: data})
            else:
                # Multi-dimensional array
                columns = [f"{feature_name}_{i}" for i in range(data.shape[1])]
                df = pd.DataFrame(data, columns=columns)
            
            # Save to Parquet
            feature_path = self._get_feature_path(feature_name)
            df.to_parquet(
                feature_path, 
                compression=self.config.compression,
                index=False
            )
            
            # Update metadata
            metadata.data_shape = data.shape
            metadata.last_updated = datetime.now()
            
            # Save metadata
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            all_metadata[feature_name] = metadata.to_dict()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(all_metadata, f, indent=2)
            
            logger.info(f"Successfully saved feature '{feature_name}' to Parquet")
            return True
            
        except Exception as e:
            logger.error(f"Error saving feature '{feature_name}' to Parquet: {e}")
            return False
    
    def load_feature(self, feature_name: str, version: Optional[str] = None) -> Tuple[np.ndarray, FeatureMetadata]:
        """Load feature data from Parquet file"""
        try:
            feature_path = self._get_feature_path(feature_name)
            
            if not feature_path.exists():
                raise ValueError(f"Feature '{feature_name}' not found")
            
            # Load data
            df = pd.read_parquet(feature_path)
            data = df.values
            
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            if feature_name not in all_metadata:
                raise ValueError(f"Metadata for feature '{feature_name}' not found")
            
            metadata = FeatureMetadata.from_dict(all_metadata[feature_name])
            
            logger.info(f"Successfully loaded feature '{feature_name}' from Parquet")
            return data, metadata
            
        except Exception as e:
            logger.error(f"Error loading feature '{feature_name}' from Parquet: {e}")
            raise
    
    def delete_feature(self, feature_name: str, version: Optional[str] = None) -> bool:
        """Delete feature from Parquet storage"""
        try:
            feature_path = self._get_feature_path(feature_name)
            
            # Delete file
            if feature_path.exists():
                feature_path.unlink()
            
            # Remove from metadata
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            if feature_name in all_metadata:
                del all_metadata[feature_name]
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(all_metadata, f, indent=2)
            
            logger.info(f"Successfully deleted feature '{feature_name}' from Parquet")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting feature '{feature_name}' from Parquet: {e}")
            return False
    
    def list_features(self) -> List[str]:
        """List all available features"""
        try:
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
            return list(all_metadata.keys())
        except:
            return []
    
    def feature_exists(self, feature_name: str, version: Optional[str] = None) -> bool:
        """Check if feature exists"""
        feature_path = self._get_feature_path(feature_name)
        return feature_path.exists()
    
    def get_feature_metadata(self, feature_name: str, version: Optional[str] = None) -> FeatureMetadata:
        """Get feature metadata"""
        try:
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            if feature_name not in all_metadata:
                raise ValueError(f"Feature '{feature_name}' not found")
            
            return FeatureMetadata.from_dict(all_metadata[feature_name])
            
        except Exception as e:
            logger.error(f"Error getting metadata for feature '{feature_name}': {e}")
            raise

# ============================================
# Main Feature Store Class
# ============================================

class FeatureStore:
    """
    Advanced feature store for financial machine learning features.
    Provides efficient storage, retrieval, and management of features.
    """
    
    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        self.config = config or FeatureStoreConfig()
        self.cache = {} if self.config.enable_caching else None
        self.cache_order = [] if self.config.enable_caching else None
        
        # Initialize storage backend
        self.backend = self._create_backend()
        
        # Initialize backup system if enabled
        if self.config.backup_enabled:
            self.backup_dir = Path(self.config.base_path) / "backups"
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FeatureStore with {self.config.storage_format} backend")
    
    def _create_backend(self) -> StorageBackend:
        """Create appropriate storage backend"""
        if self.config.storage_format == 'hdf5':
            return HDF5Backend(self.config)
        elif self.config.storage_format == 'parquet':
            return ParquetBackend(self.config)
        else:
            raise ValueError(f"Unsupported storage format: {self.config.storage_format}")
    
    def _manage_cache(self, feature_name: str, data: np.ndarray):
        """Manage feature cache with LRU eviction"""
        if not self.config.enable_caching:
            return
        
        # Remove from cache if exists (for reordering)
        if feature_name in self.cache:
            self.cache_order.remove(feature_name)
        
        # Add to cache
        self.cache[feature_name] = data
        self.cache_order.append(feature_name)
        
        # Evict if cache is full
        while len(self.cache) > self.config.cache_size:
            oldest_feature = self.cache_order.pop(0)
            del self.cache[oldest_feature]
    
    def _get_from_cache(self, feature_name: str) -> Optional[np.ndarray]:
        """Get feature from cache"""
        if not self.config.enable_caching or feature_name not in self.cache:
            return None
        
        # Move to end (most recently used)
        self.cache_order.remove(feature_name)
        self.cache_order.append(feature_name)
        
        return self.cache[feature_name]
    
    @time_it("feature_save")
    def save_feature(self, 
                    feature_name: str, 
                    data: Union[np.ndarray, pd.Series, pd.DataFrame],
                    feature_type: str = 'indicator',
                    description: str = "",
                    parameters: Optional[Dict[str, Any]] = None,
                    dependencies: Optional[List[str]] = None,
                    tags: Optional[List[str]] = None,
                    overwrite: bool = False) -> bool:
        """
        Save feature to the feature store
        
        Args:
            feature_name: Name of the feature
            data: Feature data
            feature_type: Type of feature ('indicator', 'target', 'transformed')
            description: Feature description
            parameters: Parameters used to create the feature
            dependencies: List of features this depends on
            tags: Tags for categorization
            overwrite: Whether to overwrite existing feature
            
        Returns:
            Success status
        """
        
        # Check if feature exists
        if not overwrite and self.backend.feature_exists(feature_name):
            logger.warning(f"Feature '{feature_name}' already exists. Use overwrite=True to replace.")
            return False
        
        # Convert data to numpy array
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data_array = data.values
        else:
            data_array = np.asarray(data)
        
        # Validate data
        if data_array.size == 0:
            raise ValueError("Feature data cannot be empty")
        
        # Determine data type
        if np.issubdtype(data_array.dtype, np.integer):
            data_type = 'numerical'
        elif np.issubdtype(data_array.dtype, np.floating):
            data_type = 'numerical'
        elif data_array.dtype == bool:
            data_type = 'binary'
        else:
            data_type = 'categorical'
        
        # Create metadata
        metadata = FeatureMetadata(
            feature_name=feature_name,
            feature_type=feature_type,
            data_type=data_type,
            creation_date=datetime.now(),
            last_updated=datetime.now(),
            description=description,
            parameters=parameters or {},
            dependencies=dependencies or [],
            tags=tags or []
        )
        
        # Save to backend
        success = self.backend.save_feature(feature_name, data_array, metadata)
        
        if success:
            # Update cache
            self._manage_cache(feature_name, data_array)
            
            # Create backup if enabled
            if self.config.backup_enabled:
                self._create_backup(feature_name, data_array, metadata)
        
        return success
    
    @time_it("feature_load")
    def load_feature(self, 
                    feature_name: str, 
                    version: Optional[str] = None,
                    use_cache: bool = True) -> Tuple[np.ndarray, FeatureMetadata]:
        """
        Load feature from the feature store
        
        Args:
            feature_name: Name of the feature to load
            version: Specific version to load (if versioning enabled)
            use_cache: Whether to use cached data
            
        Returns:
            Tuple of (feature_data, metadata)
        """
        
        # Check cache first
        if use_cache and self.config.enable_caching:
            cached_data = self._get_from_cache(feature_name)
            if cached_data is not None:
                # Get metadata from backend
                metadata = self.backend.get_feature_metadata(feature_name, version)
                logger.debug(f"Loaded feature '{feature_name}' from cache")
                return cached_data, metadata
        
        # Load from backend
        data, metadata = self.backend.load_feature(feature_name, version)
        
        # Update cache
        if use_cache:
            self._manage_cache(feature_name, data)
        
        return data, metadata
    
    def load_multiple_features(self, 
                              feature_names: List[str],
                              use_cache: bool = True,
                              parallel: bool = None) -> Dict[str, Tuple[np.ndarray, FeatureMetadata]]:
        """
        Load multiple features efficiently
        
        Args:
            feature_names: List of feature names to load
            use_cache: Whether to use cached data
            parallel: Whether to load in parallel (None=auto)
            
        Returns:
            Dictionary mapping feature names to (data, metadata) tuples
        """
        
        results = {}
        
        # Determine if parallel loading should be used
        if parallel is None:
            parallel = self.config.parallel_io and len(feature_names) > 3
        
        if parallel:
            # Parallel loading
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_feature = {
                    executor.submit(self.load_feature, name, None, use_cache): name
                    for name in feature_names
                }
                
                for future in as_completed(future_to_feature):
                    feature_name = future_to_feature[future]
                    try:
                        results[feature_name] = future.result()
                    except Exception as exc:
                        logger.error(f"Feature '{feature_name}' generated an exception: {exc}")
        else:
            # Sequential loading
            for feature_name in feature_names:
                try:
                    results[feature_name] = self.load_feature(feature_name, None, use_cache)
                except Exception as e:
                    logger.error(f"Error loading feature '{feature_name}': {e}")
        
        logger.info(f"Successfully loaded {len(results)}/{len(feature_names)} features")
        return results
    
    def delete_feature(self, feature_name: str, version: Optional[str] = None) -> bool:
        """Delete feature from the store"""
        success = self.backend.delete_feature(feature_name, version)
        
        if success and self.config.enable_caching:
            # Remove from cache
            if feature_name in self.cache:
                del self.cache[feature_name]
                self.cache_order.remove(feature_name)
        
        return success
    
    def list_features(self, 
                     feature_type: Optional[str] = None,
                     tags: Optional[List[str]] = None) -> List[str]:
        """
        List available features with optional filtering
        
        Args:
            feature_type: Filter by feature type
            tags: Filter by tags (features must have all specified tags)
            
        Returns:
            List of feature names
        """
        all_features = self.backend.list_features()
        
        if not feature_type and not tags:
            return all_features
        
        # Filter features
        filtered_features = []
        
        for feature_name in all_features:
            try:
                metadata = self.backend.get_feature_metadata(feature_name)
                
                # Filter by feature type
                if feature_type and metadata.feature_type != feature_type:
                    continue
                
                # Filter by tags
                if tags and not all(tag in metadata.tags for tag in tags):
                    continue
                
                filtered_features.append(feature_name)
                
            except Exception as e:
                logger.warning(f"Error checking metadata for feature '{feature_name}': {e}")
        
        return filtered_features
    
    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a feature"""
        try:
            metadata = self.backend.get_feature_metadata(feature_name)
            
            info = {
                'name': feature_name,
                'type': metadata.feature_type,
                'data_type': metadata.data_type,
                'shape': metadata.data_shape,
                'creation_date': metadata.creation_date,
                'last_updated': metadata.last_updated,
                'version': metadata.version,
                'description': metadata.description,
                'parameters': metadata.parameters,
                'dependencies': metadata.dependencies,
                'tags': metadata.tags,
                'data_range': metadata.data_range,
                'missing_count': metadata.missing_count,
                'in_cache': feature_name in self.cache if self.config.enable_caching else False
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting info for feature '{feature_name}': {e}")
            return {}
    
    def search_features(self, 
                       query: str,
                       search_in: List[str] = ['name', 'description', 'tags']) -> List[str]:
        """
        Search for features by name, description, or tags
        
        Args:
            query: Search query
            search_in: Fields to search in
            
        Returns:
            List of matching feature names
        """
        query_lower = query.lower()
        matching_features = []
        
        for feature_name in self.backend.list_features():
            try:
                metadata = self.backend.get_feature_metadata(feature_name)
                
                # Check if query matches
                match_found = False
                
                if 'name' in search_in and query_lower in feature_name.lower():
                    match_found = True
                
                if 'description' in search_in and query_lower in metadata.description.lower():
                    match_found = True
                
                if 'tags' in search_in:
                    for tag in metadata.tags:
                        if query_lower in tag.lower():
                            match_found = True
                            break
                
                if match_found:
                    matching_features.append(feature_name)
                    
            except Exception as e:
                logger.warning(f"Error searching feature '{feature_name}': {e}")
        
        return matching_features
    
    def get_feature_dependencies(self, feature_name: str) -> Dict[str, List[str]]:
        """Get dependency tree for a feature"""
        try:
            metadata = self.backend.get_feature_metadata(feature_name)
            
            dependencies = {
                'direct': metadata.dependencies,
                'all': []
            }
            
            # Recursively find all dependencies
            def find_all_deps(name, visited=None):
                if visited is None:
                    visited = set()
                
                if name in visited:
                    return []
                
                visited.add(name)
                all_deps = []
                
                try:
                    feat_metadata = self.backend.get_feature_metadata(name)
                    for dep in feat_metadata.dependencies:
                        all_deps.append(dep)
                        all_deps.extend(find_all_deps(dep, visited))
                except:
                    pass
                
                return list(set(all_deps))
            
            dependencies['all'] = find_all_deps(feature_name)
            return dependencies
            
        except Exception as e:
            logger.error(f"Error getting dependencies for feature '{feature_name}': {e}")
            return {'direct': [], 'all': []}
    
    def _create_backup(self, feature_name: str, data: np.ndarray, metadata: FeatureMetadata):
        """Create backup of feature"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"{feature_name}_{timestamp}.pkl"
            
            backup_data = {
                'data': data,
                'metadata': metadata.to_dict()
            }
            
            with open(backup_file, 'wb') as f:
                pickle.dump(backup_data, f)
            
            logger.debug(f"Created backup for feature '{feature_name}'")
            
        except Exception as e:
            logger.warning(f"Error creating backup for feature '{feature_name}': {e}")
    
    def cleanup_old_backups(self):
        """Clean up old backup files"""
        if not self.config.backup_enabled or not self.config.auto_cleanup:
            return
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.cleanup_days)
            
            for backup_file in self.backup_dir.glob("*.pkl"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    backup_file.unlink()
                    logger.debug(f"Removed old backup: {backup_file}")
                    
        except Exception as e:
            logger.warning(f"Error cleaning up backups: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            all_features = self.backend.list_features()
            
            total_features = len(all_features)
            total_size = 0
            feature_types = {}
            data_types = {}
            
            for feature_name in all_features:
                try:
                    metadata = self.backend.get_feature_metadata(feature_name)
                    
                    # Count feature types
                    feature_types[metadata.feature_type] = feature_types.get(metadata.feature_type, 0) + 1
                    
                    # Count data types
                    data_types[metadata.data_type] = data_types.get(metadata.data_type, 0) + 1
                    
                    # Estimate size (rough approximation)
                    if metadata.data_shape:
                        size = np.prod(metadata.data_shape) * 8  # Assume 8 bytes per element
                        total_size += size
                        
                except Exception as e:
                    logger.warning(f"Error getting stats for feature '{feature_name}': {e}")
            
            stats = {
                'total_features': total_features,
                'estimated_total_size_mb': total_size / (1024 * 1024),
                'feature_types': feature_types,
                'data_types': data_types,
                'cache_size': len(self.cache) if self.config.enable_caching else 0,
                'backend_type': self.config.storage_format
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the feature cache"""
        if self.config.enable_caching:
            self.cache.clear()
            self.cache_order.clear()
            logger.info("Feature cache cleared")

# ============================================
# Utility Functions
# ============================================

def create_feature_store(storage_format: str = "hdf5",
                        base_path: str = "./feature_store",
                        **kwargs) -> FeatureStore:
    """
    Create a feature store with specified configuration
    
    Args:
        storage_format: Storage backend format
        base_path: Base directory for feature store
        **kwargs: Additional configuration options
        
    Returns:
        Configured FeatureStore instance
    """
    
    config = FeatureStoreConfig(
        storage_format=storage_format,
        base_path=base_path,
        **kwargs
    )
    
    return FeatureStore(config)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Feature Store")
    
    # Create sample data
    np.random.seed(42)
    
    # Sample features
    price_data = np.random.randn(1000) * 10 + 100
    volume_data = np.random.exponential(1000, 1000)
    rsi_data = np.random.uniform(20, 80, 1000)
    
    # Create feature store
    store = create_feature_store(
        storage_format="hdf5",
        base_path="./test_feature_store",
        enable_caching=True,
        cache_size=10
    )
    
    print(f"Created feature store with {store.config.storage_format} backend")
    
    # Test saving features
    print("\n1. Testing Feature Saving")
    
    success1 = store.save_feature(
        "stock_prices",
        price_data,
        feature_type="indicator",
        description="Daily stock prices",
        parameters={"source": "yahoo", "symbol": "AAPL"},
        tags=["price", "daily", "stock"]
    )
    print(f"Saved stock_prices: {success1}")
    
    success2 = store.save_feature(
        "trading_volume",
        volume_data,
        feature_type="indicator", 
        description="Daily trading volume",
        dependencies=["stock_prices"],
        tags=["volume", "daily", "stock"]
    )
    print(f"Saved trading_volume: {success2}")
    
    success3 = store.save_feature(
        "rsi_14",
        rsi_data,
        feature_type="indicator",
        description="14-day RSI indicator",
        parameters={"period": 14},
        dependencies=["stock_prices"],
        tags=["momentum", "rsi", "technical"]
    )
    print(f"Saved rsi_14: {success3}")
    
    # Test loading features
    print("\n2. Testing Feature Loading")
    
    loaded_prices, prices_metadata = store.load_feature("stock_prices")
    print(f"Loaded stock_prices: shape={loaded_prices.shape}, type={prices_metadata.feature_type}")
    
    loaded_volume, volume_metadata = store.load_feature("trading_volume")
    print(f"Loaded trading_volume: shape={loaded_volume.shape}, dependencies={volume_metadata.dependencies}")
    
    # Test multiple feature loading
    print("\n3. Testing Multiple Feature Loading")
    
    multiple_features = store.load_multiple_features(
        ["stock_prices", "trading_volume", "rsi_14"],
        parallel=True
    )
    print(f"Loaded {len(multiple_features)} features in parallel")
    
    # Test listing features
    print("\n4. Testing Feature Listing")
    
    all_features = store.list_features()
    print(f"All features: {all_features}")
    
    indicator_features = store.list_features(feature_type="indicator")
    print(f"Indicator features: {indicator_features}")
    
    stock_features = store.list_features(tags=["stock"])
    print(f"Stock features: {stock_features}")
    
    # Test feature search
    print("\n5. Testing Feature Search")
    
    price_features = store.search_features("price")
    print(f"Features matching 'price': {price_features}")
    
    rsi_features = store.search_features("rsi")
    print(f"Features matching 'rsi': {rsi_features}")
    
    # Test feature info
    print("\n6. Testing Feature Info")
    
    prices_info = store.get_feature_info("stock_prices")
    print("Stock prices info:")
    for key, value in prices_info.items():
        print(f"  {key}: {value}")
    
    # Test dependencies
    print("\n7. Testing Feature Dependencies")
    
    rsi_deps = store.get_feature_dependencies("rsi_14")
    print(f"RSI dependencies: {rsi_deps}")
    
    # Test storage stats
    print("\n8. Testing Storage Statistics")
    
    stats = store.get_storage_stats()
    print("Storage statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test cache
    print("\n9. Testing Cache")
    
    print(f"Cache size before: {len(store.cache)}")
    
    # Load same feature again (should come from cache)
    cached_prices, _ = store.load_feature("stock_prices")
    print(f"Loaded from cache: {np.array_equal(cached_prices, loaded_prices)}")
    
    # Clear cache
    store.clear_cache()
    print(f"Cache size after clear: {len(store.cache)}")
    
    # Cleanup
    print("\n10. Cleanup")
    
    store.delete_feature("stock_prices")
    store.delete_feature("trading_volume")
    store.delete_feature("rsi_14")
    
    remaining_features = store.list_features()
    print(f"Remaining features: {remaining_features}")
    
    print("\nFeature store testing completed successfully!")
