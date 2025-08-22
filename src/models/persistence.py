# ============================================
# StockPredictionPro - src/models/persistence.py
# Advanced model persistence with versioning, metadata tracking, and production deployment features
# ============================================

import numpy as np
import pandas as pd
import pickle
import joblib
import json
import os
import shutil
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import sqlite3
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

# Core ML imports
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Import our modules
from .factory import model_factory
from ..utils.exceptions import ModelValidationError, BusinessLogicError
from ..utils.logger import get_logger
from ..utils.timing import Timer, time_it

logger = get_logger('models.persistence')

# ============================================
# Model Metadata and Versioning
# ============================================

class ModelStatus(Enum):
    """Model status enumeration"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    model_name: str
    model_type: str
    task_type: str
    version: str
    created_at: datetime
    updated_at: datetime
    status: ModelStatus
    
    # Model information
    performance_metrics: Dict[str, float]
    training_duration: float
    model_size_bytes: int
    feature_count: int
    parameter_count: Optional[int]
    
    # Training information
    training_data_hash: str
    training_samples: int
    validation_samples: int
    hyperparameters: Dict[str, Any]
    
    # Performance tracking
    cross_validation_scores: List[float]
    feature_importance: Optional[Dict[str, float]]
    
    # Deployment information
    deployment_environment: Optional[str] = None
    deployment_url: Optional[str] = None
    prediction_count: int = 0
    last_prediction_time: Optional[datetime] = None
    
    # Additional metadata
    tags: List[str] = None
    description: str = ""
    creator: str = "system"
    model_family: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.last_prediction_time:
            data['last_prediction_time'] = self.last_prediction_time.isoformat()
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary"""
        # Convert ISO strings back to datetime
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('last_prediction_time'):
            data['last_prediction_time'] = datetime.fromisoformat(data['last_prediction_time'])
        data['status'] = ModelStatus(data['status'])
        return cls(**data)

# ============================================
# Model Storage Backend
# ============================================

class ModelStorageBackend:
    """Abstract base class for model storage backends"""
    
    def save_model(self, model: BaseEstimator, model_id: str, metadata: ModelMetadata) -> bool:
        """Save model with metadata"""
        raise NotImplementedError
    
    def load_model(self, model_id: str) -> Tuple[BaseEstimator, ModelMetadata]:
        """Load model with metadata"""
        raise NotImplementedError
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model"""
        raise NotImplementedError
    
    def list_models(self, filters: Optional[Dict] = None) -> List[ModelMetadata]:
        """List models with optional filters"""
        raise NotImplementedError

class FileSystemStorageBackend(ModelStorageBackend):
    """File system based storage backend"""
    
    def __init__(self, base_path: str = "models", use_compression: bool = True):
        self.base_path = Path(base_path)
        self.use_compression = use_compression
        self.models_dir = self.base_path / "models"
        self.metadata_dir = self.base_path / "metadata"
        self.index_file = self.base_path / "model_index.json"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize index
        self._initialize_index()
        
        logger.info(f"Initialized filesystem storage backend at {self.base_path}")
    
    def _initialize_index(self):
        """Initialize model index"""
        if not self.index_file.exists():
            self._save_index({})
    
    def _load_index(self) -> Dict[str, Dict]:
        """Load model index"""
        try:
            with open(self.index_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_index(self, index: Dict[str, Dict]):
        """Save model index"""
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2, default=str)
    
    def _get_model_path(self, model_id: str) -> Path:
        """Get model file path"""
        extension = ".pkl" if not self.use_compression else ".joblib"
        return self.models_dir / f"{model_id}{extension}"
    
    def _get_metadata_path(self, model_id: str) -> Path:
        """Get metadata file path"""
        return self.metadata_dir / f"{model_id}_metadata.json"
    
    def save_model(self, model: BaseEstimator, model_id: str, metadata: ModelMetadata) -> bool:
        """Save model with metadata to filesystem"""
        try:
            # Save model
            model_path = self._get_model_path(model_id)
            if self.use_compression:
                joblib.dump(model, model_path, compress=3)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Calculate model size
            metadata.model_size_bytes = model_path.stat().st_size
            
            # Save metadata
            metadata_path = self._get_metadata_path(model_id)
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)
            
            # Update index
            index = self._load_index()
            index[model_id] = {
                'model_path': str(model_path),
                'metadata_path': str(metadata_path),
                'created_at': metadata.created_at.isoformat(),
                'model_name': metadata.model_name,
                'status': metadata.status.value
            }
            self._save_index(index)
            
            logger.info(f"Successfully saved model {model_id} to filesystem")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            return False
    
    def load_model(self, model_id: str) -> Tuple[BaseEstimator, ModelMetadata]:
        """Load model with metadata from filesystem"""
        try:
            # Check index
            index = self._load_index()
            if model_id not in index:
                raise ValueError(f"Model {model_id} not found")
            
            # Load model
            model_path = self._get_model_path(model_id)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            if self.use_compression:
                model = joblib.load(model_path)
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Load metadata
            metadata_path = self._get_metadata_path(model_id)
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            metadata = ModelMetadata.from_dict(metadata_dict)
            
            logger.info(f"Successfully loaded model {model_id}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model from filesystem"""
        try:
            # Remove files
            model_path = self._get_model_path(model_id)
            metadata_path = self._get_metadata_path(model_id)
            
            if model_path.exists():
                model_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Update index
            index = self._load_index()
            if model_id in index:
                del index[model_id]
                self._save_index(index)
            
            logger.info(f"Successfully deleted model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def list_models(self, filters: Optional[Dict] = None) -> List[ModelMetadata]:
        """List models with optional filters"""
        try:
            models = []
            index = self._load_index()
            
            for model_id in index.keys():
                try:
                    metadata_path = self._get_metadata_path(model_id)
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata_dict = json.load(f)
                        metadata = ModelMetadata.from_dict(metadata_dict)
                        
                        # Apply filters
                        if self._matches_filters(metadata, filters):
                            models.append(metadata)
                except Exception as e:
                    logger.warning(f"Error loading metadata for {model_id}: {e}")
            
            # Sort by creation time (newest first)
            models.sort(key=lambda x: x.created_at, reverse=True)
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def _matches_filters(self, metadata: ModelMetadata, filters: Optional[Dict]) -> bool:
        """Check if metadata matches filters"""
        if not filters:
            return True
        
        for key, value in filters.items():
            if hasattr(metadata, key):
                attr_value = getattr(metadata, key)
                if isinstance(value, list):
                    if attr_value not in value:
                        return False
                else:
                    if attr_value != value:
                        return False
        
        return True

class DatabaseStorageBackend(ModelStorageBackend):
    """Database-backed storage with SQLite"""
    
    def __init__(self, db_path: str = "models.db", models_dir: str = "models"):
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"Initialized database storage backend at {db_path}")
    
    def _initialize_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    status TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_name ON models(model_name)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_status ON models(status)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at ON models(created_at)
            ''')
            
            conn.commit()
    
    def save_model(self, model: BaseEstimator, model_id: str, metadata: ModelMetadata) -> bool:
        """Save model to database"""
        try:
            # Save model file
            model_path = self.models_dir / f"{model_id}.joblib"
            joblib.dump(model, model_path, compress=3)
            
            # Update metadata with file size
            metadata.model_size_bytes = model_path.stat().st_size
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO models 
                    (model_id, model_name, model_type, task_type, version, 
                     created_at, updated_at, status, file_path, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_id, metadata.model_name, metadata.model_type, metadata.task_type,
                    metadata.version, metadata.created_at, metadata.updated_at, 
                    metadata.status.value, str(model_path), json.dumps(metadata.to_dict())
                ))
                conn.commit()
            
            logger.info(f"Successfully saved model {model_id} to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            return False
    
    def load_model(self, model_id: str) -> Tuple[BaseEstimator, ModelMetadata]:
        """Load model from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT file_path, metadata_json FROM models WHERE model_id = ?',
                    (model_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    raise ValueError(f"Model {model_id} not found")
                
                file_path, metadata_json = result
                
                # Load model
                model = joblib.load(file_path)
                
                # Load metadata
                metadata_dict = json.loads(metadata_json)
                metadata = ModelMetadata.from_dict(metadata_dict)
                
                logger.info(f"Successfully loaded model {model_id}")
                return model, metadata
                
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get file path
                cursor = conn.execute(
                    'SELECT file_path FROM models WHERE model_id = ?',
                    (model_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    file_path = Path(result[0])
                    if file_path.exists():
                        file_path.unlink()
                
                # Delete from database
                conn.execute('DELETE FROM models WHERE model_id = ?', (model_id,))
                conn.commit()
            
            logger.info(f"Successfully deleted model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def list_models(self, filters: Optional[Dict] = None) -> List[ModelMetadata]:
        """List models from database with optional filters"""
        try:
            query = 'SELECT metadata_json FROM models'
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    if key in ['model_name', 'model_type', 'task_type', 'status']:
                        conditions.append(f'{key} = ?')
                        params.append(value if key != 'status' else value.value if hasattr(value, 'value') else value)
                
                if conditions:
                    query += ' WHERE ' + ' AND '.join(conditions)
            
            query += ' ORDER BY created_at DESC'
            
            models = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                for (metadata_json,) in cursor.fetchall():
                    metadata_dict = json.loads(metadata_json)
                    metadata = ModelMetadata.from_dict(metadata_dict)
                    models.append(metadata)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

# ============================================
# Model Registry and Manager
# ============================================

class ModelRegistry:
    """Advanced model registry with versioning and lifecycle management"""
    
    def __init__(self, 
                 storage_backend: Optional[ModelStorageBackend] = None,
                 enable_auto_versioning: bool = True,
                 max_versions_per_model: int = 10):
        
        self.storage_backend = storage_backend or FileSystemStorageBackend()
        self.enable_auto_versioning = enable_auto_versioning
        self.max_versions_per_model = max_versions_per_model
        
        # Cache for frequently accessed models
        self._model_cache = {}
        self._cache_max_size = 5
        
        logger.info("Initialized ModelRegistry")
    
    def _generate_model_id(self, model_name: str, version: str) -> str:
        """Generate unique model ID"""
        return f"{model_name}_v{version}_{uuid.uuid4().hex[:8]}"
    
    def _generate_version(self, model_name: str) -> str:
        """Generate next version number for a model"""
        if not self.enable_auto_versioning:
            return "1.0.0"
        
        # Get existing versions
        existing_models = self.list_models(filters={'model_name': model_name})
        
        if not existing_models:
            return "1.0.0"
        
        # Parse versions and find the highest
        versions = []
        for metadata in existing_models:
            try:
                version_parts = metadata.version.split('.')
                if len(version_parts) == 3:
                    major, minor, patch = map(int, version_parts)
                    versions.append((major, minor, patch))
            except:
                continue
        
        if not versions:
            return "1.0.0"
        
        # Increment minor version
        latest_version = max(versions)
        return f"{latest_version[0]}.{latest_version[1] + 1}.0"
    
    def _calculate_data_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        """Calculate hash of training data"""
        combined_data = np.concatenate([X.flatten(), y.flatten()])
        return hashlib.md5(combined_data.tobytes()).hexdigest()
    
    def _extract_model_info(self, model: BaseEstimator) -> Dict[str, Any]:
        """Extract information from model"""
        info = {
            'model_type': type(model).__name__,
            'model_family': type(model).__module__.split('.')[-1] if hasattr(type(model), '__module__') else 'unknown'
        }
        
        # Try to get parameter count
        try:
            if hasattr(model, 'get_params'):
                params = model.get_params()
                info['parameter_count'] = len(params)
            else:
                info['parameter_count'] = None
        except:
            info['parameter_count'] = None
        
        return info
    
    @time_it("model_registration", include_args=True)
    def register_model(self,
                      model: BaseEstimator,
                      model_name: str,
                      task_type: str,
                      training_data: Tuple[np.ndarray, np.ndarray],
                      performance_metrics: Dict[str, float],
                      hyperparameters: Dict[str, Any],
                      version: Optional[str] = None,
                      description: str = "",
                      tags: Optional[List[str]] = None,
                      cross_validation_scores: Optional[List[float]] = None,
                      feature_importance: Optional[Dict[str, float]] = None,
                      training_duration: float = 0.0) -> str:
        """Register a new model"""
        
        X, y = training_data
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version(model_name)
        
        # Generate model ID
        model_id = self._generate_model_id(model_name, version)
        
        # Extract model information
        model_info = self._extract_model_info(model)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_info['model_type'],
            task_type=task_type,
            version=version,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=ModelStatus.TRAINED,
            
            performance_metrics=performance_metrics,
            training_duration=training_duration,
            model_size_bytes=0,  # Will be set by storage backend
            feature_count=X.shape[1] if len(X.shape) > 1 else 1,
            parameter_count=model_info.get('parameter_count'),
            
            training_data_hash=self._calculate_data_hash(X, y),
            training_samples=len(X),
            validation_samples=0,  # Could be updated later
            hyperparameters=hyperparameters,
            
            cross_validation_scores=cross_validation_scores or [],
            feature_importance=feature_importance,
            
            tags=tags or [],
            description=description,
            model_family=model_info.get('model_family', '')
        )
        
        # Save model
        if self.storage_backend.save_model(model, model_id, metadata):
            logger.info(f"Successfully registered model {model_name} v{version} with ID {model_id}")
            
            # Cleanup old versions if needed
            self._cleanup_old_versions(model_name)
            
            return model_id
        else:
            raise ModelValidationError(f"Failed to register model {model_name}")
    
    def _cleanup_old_versions(self, model_name: str):
        """Cleanup old versions of a model"""
        try:
            existing_models = self.list_models(filters={'model_name': model_name})
            
            if len(existing_models) > self.max_versions_per_model:
                # Sort by creation time and keep only the latest versions
                existing_models.sort(key=lambda x: x.created_at, reverse=True)
                models_to_delete = existing_models[self.max_versions_per_model:]
                
                for metadata in models_to_delete:
                    if metadata.status != ModelStatus.DEPLOYED:  # Don't delete deployed models
                        self.delete_model(metadata.model_id)
                        logger.info(f"Cleaned up old version {metadata.model_id}")
                        
        except Exception as e:
            logger.warning(f"Failed to cleanup old versions for {model_name}: {e}")
    
    def load_model(self, model_id: str, use_cache: bool = True) -> Tuple[BaseEstimator, ModelMetadata]:
        """Load a model by ID"""
        
        # Check cache first
        if use_cache and model_id in self._model_cache:
            logger.debug(f"Loading model {model_id} from cache")
            return self._model_cache[model_id]
        
        # Load from storage
        model, metadata = self.storage_backend.load_model(model_id)
        
        # Update cache
        if use_cache:
            self._update_cache(model_id, (model, metadata))
        
        return model, metadata
    
    def load_latest_model(self, model_name: str) -> Tuple[BaseEstimator, ModelMetadata]:
        """Load the latest version of a model"""
        models = self.list_models(filters={'model_name': model_name})
        
        if not models:
            raise ValueError(f"No models found with name {model_name}")
        
        # Get the latest model
        latest_model = max(models, key=lambda x: x.created_at)
        return self.load_model(latest_model.model_id)
    
    def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model status"""
        try:
            model, metadata = self.load_model(model_id)
            metadata.status = status
            metadata.updated_at = datetime.now()
            
            return self.storage_backend.save_model(model, model_id, metadata)
            
        except Exception as e:
            logger.error(f"Failed to update model status: {e}")
            return False
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model"""
        # Remove from cache
        if model_id in self._model_cache:
            del self._model_cache[model_id]
        
        return self.storage_backend.delete_model(model_id)
    
    def list_models(self, filters: Optional[Dict] = None) -> List[ModelMetadata]:
        """List models with optional filters"""
        return self.storage_backend.list_models(filters)
    
    def _update_cache(self, model_id: str, model_data: Tuple[BaseEstimator, ModelMetadata]):
        """Update model cache with LRU eviction"""
        if len(self._model_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._model_cache))
            del self._model_cache[oldest_key]
        
        self._model_cache[model_id] = model_data
    
    def get_model_performance_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get performance history for a model"""
        models = self.list_models(filters={'model_name': model_name})
        
        history = []
        for metadata in models:
            history.append({
                'version': metadata.version,
                'created_at': metadata.created_at,
                'performance_metrics': metadata.performance_metrics,
                'cross_validation_scores': metadata.cross_validation_scores,
                'model_id': metadata.model_id
            })
        
        # Sort by creation time
        history.sort(key=lambda x: x['created_at'])
        return history
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple models"""
        comparison = {
            'models': [],
            'performance_comparison': {},
            'metadata_comparison': {}
        }
        
        for model_id in model_ids:
            try:
                _, metadata = self.load_model(model_id)
                comparison['models'].append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load model {model_id} for comparison: {e}")
        
        if not comparison['models']:
            return comparison
        
        # Performance comparison
        metrics_keys = set()
        for metadata in comparison['models']:
            metrics_keys.update(metadata.performance_metrics.keys())
        
        for metric in metrics_keys:
            comparison['performance_comparison'][metric] = []
            for metadata in comparison['models']:
                value = metadata.performance_metrics.get(metric)
                comparison['performance_comparison'][metric].append({
                    'model_id': metadata.model_id,
                    'value': value
                })
        
        # Metadata comparison
        comparison['metadata_comparison'] = {
            'training_times': [(m.model_id, m.training_duration) for m in comparison['models']],
            'model_sizes': [(m.model_id, m.model_size_bytes) for m in comparison['models']],
            'feature_counts': [(m.model_id, m.feature_count) for m in comparison['models']],
            'training_samples': [(m.model_id, m.training_samples) for m in comparison['models']]
        }
        
        return comparison

# ============================================
# Model Deployment Manager
# ============================================

class ModelDeploymentManager:
    """Manage model deployments and serving"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.deployed_models = {}
        self.prediction_logs = []
        
        logger.info("Initialized ModelDeploymentManager")
    
    def deploy_model(self, 
                    model_id: str, 
                    deployment_name: str,
                    environment: str = "production",
                    auto_load: bool = True) -> bool:
        """Deploy a model for serving"""
        try:
            # Load model
            model, metadata = self.registry.load_model(model_id)
            
            # Update metadata
            metadata.status = ModelStatus.DEPLOYED
            metadata.deployment_environment = environment
            metadata.updated_at = datetime.now()
            
            # Save updated metadata
            self.registry.storage_backend.save_model(model, model_id, metadata)
            
            # Store in deployed models
            if auto_load:
                self.deployed_models[deployment_name] = {
                    'model': model,
                    'metadata': metadata,
                    'deployed_at': datetime.now()
                }
            
            logger.info(f"Successfully deployed model {model_id} as {deployment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model {model_id}: {e}")
            return False
    
    def undeploy_model(self, deployment_name: str) -> bool:
        """Undeploy a model"""
        try:
            if deployment_name in self.deployed_models:
                model_info = self.deployed_models[deployment_name]
                model_id = model_info['metadata'].model_id
                
                # Update status
                self.registry.update_model_status(model_id, ModelStatus.TRAINED)
                
                # Remove from deployed models
                del self.deployed_models[deployment_name]
                
                logger.info(f"Successfully undeployed model {deployment_name}")
                return True
            else:
                logger.warning(f"Deployment {deployment_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to undeploy model {deployment_name}: {e}")
            return False
    
    def predict(self, 
               deployment_name: str, 
               X: np.ndarray,
               log_predictions: bool = True) -> np.ndarray:
        """Make predictions using a deployed model"""
        
        if deployment_name not in self.deployed_models:
            raise ValueError(f"Model {deployment_name} not deployed")
        
        model_info = self.deployed_models[deployment_name]
        model = model_info['model']
        metadata = model_info['metadata']
        
        # Make predictions
        predictions = model.predict(X)
        
        # Log prediction
        if log_predictions:
            log_entry = {
                'deployment_name': deployment_name,
                'model_id': metadata.model_id,
                'timestamp': datetime.now(),
                'input_shape': X.shape,
                'prediction_count': len(predictions)
            }
            self.prediction_logs.append(log_entry)
            
            # Update metadata
            metadata.prediction_count += len(predictions)
            metadata.last_prediction_time = datetime.now()
        
        return predictions
    
    def predict_proba(self, deployment_name: str, X: np.ndarray) -> np.ndarray:
        """Make probability predictions using a deployed model"""
        
        if deployment_name not in self.deployed_models:
            raise ValueError(f"Model {deployment_name} not deployed")
        
        model = self.deployed_models[deployment_name]['model']
        
        if not hasattr(model, 'predict_proba'):
            raise AttributeError(f"Model {deployment_name} does not support probability predictions")
        
        return model.predict_proba(X)
    
    def get_deployment_info(self, deployment_name: str) -> Dict[str, Any]:
        """Get information about a deployment"""
        if deployment_name not in self.deployed_models:
            raise ValueError(f"Deployment {deployment_name} not found")
        
        model_info = self.deployed_models[deployment_name]
        metadata = model_info['metadata']
        
        return {
            'deployment_name': deployment_name,
            'model_id': metadata.model_id,
            'model_name': metadata.model_name,
            'version': metadata.version,
            'deployed_at': model_info['deployed_at'],
            'environment': metadata.deployment_environment,
            'prediction_count': metadata.prediction_count,
            'last_prediction_time': metadata.last_prediction_time,
            'status': metadata.status.value
        }
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all current deployments"""
        deployments = []
        for deployment_name in self.deployed_models.keys():
            try:
                info = self.get_deployment_info(deployment_name)
                deployments.append(info)
            except Exception as e:
                logger.warning(f"Error getting info for deployment {deployment_name}: {e}")
        
        return deployments
    
    def get_prediction_logs(self, 
                           deployment_name: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Get prediction logs"""
        logs = self.prediction_logs
        
        if deployment_name:
            logs = [log for log in logs if log['deployment_name'] == deployment_name]
        
        # Return most recent logs first
        logs = sorted(logs, key=lambda x: x['timestamp'], reverse=True)
        return logs[:limit]

# ============================================
# Global Instances and Convenience Functions
# ============================================

# Global registry instance
default_registry = ModelRegistry()
default_deployment_manager = ModelDeploymentManager(default_registry)

# Convenience functions
def register_model(model: BaseEstimator,
                  model_name: str,
                  task_type: str,
                  training_data: Tuple[np.ndarray, np.ndarray],
                  performance_metrics: Dict[str, float],
                  hyperparameters: Dict[str, Any],
                  **kwargs) -> str:
    """Register a model using the default registry"""
    return default_registry.register_model(
        model, model_name, task_type, training_data, 
        performance_metrics, hyperparameters, **kwargs
    )

def load_model(model_id: str) -> Tuple[BaseEstimator, ModelMetadata]:
    """Load a model using the default registry"""
    return default_registry.load_model(model_id)

def load_latest_model(model_name: str) -> Tuple[BaseEstimator, ModelMetadata]:
    """Load the latest version of a model"""
    return default_registry.load_latest_model(model_name)

def list_models(filters: Optional[Dict] = None) -> List[ModelMetadata]:
    """List models using the default registry"""
    return default_registry.list_models(filters)

def deploy_model(model_id: str, deployment_name: str, **kwargs) -> bool:
    """Deploy a model using the default deployment manager"""
    return default_deployment_manager.deploy_model(model_id, deployment_name, **kwargs)

def predict(deployment_name: str, X: np.ndarray) -> np.ndarray:
    """Make predictions using a deployed model"""
    return default_deployment_manager.predict(deployment_name, X)

def get_model_performance_history(model_name: str) -> List[Dict[str, Any]]:
    """Get performance history for a model"""
    return default_registry.get_model_performance_history(model_name)

def compare_models(model_ids: List[str]) -> Dict[str, Any]:
    """Compare multiple models"""
    return default_registry.compare_models(model_ids)

# ============================================
# Model Export and Import
# ============================================

class ModelExporter:
    """Export models in various formats"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def export_to_pickle(self, model_id: str, filepath: str) -> bool:
        """Export model to pickle format"""
        try:
            model, metadata = self.registry.load_model(model_id)
            
            export_data = {
                'model': model,
                'metadata': metadata.to_dict(),
                'exported_at': datetime.now().isoformat(),
                'export_format': 'pickle'
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(export_data, f)
            
            logger.info(f"Successfully exported model {model_id} to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export model {model_id}: {e}")
            return False
    
    def export_to_joblib(self, model_id: str, filepath: str) -> bool:
        """Export model to joblib format"""
        try:
            model, metadata = self.registry.load_model(model_id)
            
            export_data = {
                'model': model,
                'metadata': metadata.to_dict(),
                'exported_at': datetime.now().isoformat(),
                'export_format': 'joblib'
            }
            
            joblib.dump(export_data, filepath, compress=3)
            
            logger.info(f"Successfully exported model {model_id} to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export model {model_id}: {e}")
            return False

class ModelImporter:
    """Import models from various formats"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def import_from_pickle(self, filepath: str, new_model_name: Optional[str] = None) -> Optional[str]:
        """Import model from pickle format"""
        try:
            with open(filepath, 'rb') as f:
                export_data = pickle.load(f)
            
            model = export_data['model']
            metadata_dict = export_data['metadata']
            
            # Create new metadata
            if new_model_name:
                metadata_dict['model_name'] = new_model_name
            
            # Register imported model
            # This is a simplified version - you might want to extract more info
            model_id = self.registry.register_model(
                model=model,
                model_name=metadata_dict['model_name'],
                task_type=metadata_dict['task_type'],
                training_data=(np.array([[0]]), np.array([0])),  # Dummy data
                performance_metrics=metadata_dict.get('performance_metrics', {}),
                hyperparameters=metadata_dict.get('hyperparameters', {}),
                description=f"Imported from {filepath}"
            )
            
            logger.info(f"Successfully imported model from {filepath} with ID {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to import model from {filepath}: {e}")
            return None
    
    def import_from_joblib(self, filepath: str, new_model_name: Optional[str] = None) -> Optional[str]:
        """Import model from joblib format"""
        try:
            export_data = joblib.load(filepath)
            
            model = export_data['model']
            metadata_dict = export_data['metadata']
            
            # Create new metadata
            if new_model_name:
                metadata_dict['model_name'] = new_model_name
            
            # Register imported model
            model_id = self.registry.register_model(
                model=model,
                model_name=metadata_dict['model_name'],
                task_type=metadata_dict['task_type'],
                training_data=(np.array([[0]]), np.array([0])),  # Dummy data
                performance_metrics=metadata_dict.get('performance_metrics', {}),
                hyperparameters=metadata_dict.get('hyperparameters', {}),
                description=f"Imported from {filepath}"
            )
            
            logger.info(f"Successfully imported model from {filepath} with ID {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to import model from {filepath}: {e}")
            return None
