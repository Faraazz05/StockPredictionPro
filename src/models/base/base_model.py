# ============================================
# StockPredictionPro - src/models/base/base_model.py
# Base model interface for all machine learning models
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import pickle
import joblib
import json
from pathlib import Path
import warnings

from ...utils.exceptions import (
    ModelValidationError, BusinessLogicError, InvalidParameterError
)
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ...utils.config_loader import get
from ...utils.governance import governance, log_governance_event
from ...utils.validators import ValidationResult
from ...utils.helpers import ensure_directory

logger = get_logger('models.base.model')

# ============================================
# Model Enums and Types
# ============================================

class ModelStatus(Enum):
    """Model lifecycle status"""
    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ERROR = "error"
    DEPRECATED = "deprecated"

class ModelType(Enum):
    """Model type categories"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"
    TIME_SERIES = "time_series"

class PredictionType(Enum):
    """Prediction output types"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    PROBABILITY = "probability"
    RANKING = "ranking"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    name: str
    model_type: str
    prediction_type: str
    version: str
    created_at: datetime
    updated_at: datetime
    
    # Model specifications
    algorithm: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    
    # Training information
    training_data_hash: Optional[str] = None
    training_samples: Optional[int] = None
    training_features: Optional[int] = None
    training_duration: Optional[float] = None
    
    # Performance metrics
    training_score: Optional[float] = None
    validation_score: Optional[float] = None
    test_score: Optional[float] = None
    cross_validation_mean: Optional[float] = None
    cross_validation_std: Optional[float] = None
    
    # Model artifacts
    model_size_bytes: Optional[int] = None
    model_file_path: Optional[str] = None
    
    # Business context
    business_purpose: Optional[str] = None
    expected_performance: Optional[float] = None
    performance_threshold: Optional[float] = None
    
    # Governance and compliance
    data_sources: List[str] = field(default_factory=list)
    feature_engineering_steps: List[str] = field(default_factory=list)
    validation_approach: Optional[str] = None
    
    # Deployment information
    deployment_environment: Optional[str] = None
    deployment_date: Optional[datetime] = None
    last_prediction_date: Optional[datetime] = None
    prediction_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            'model_id': self.model_id,
            'name': self.name,
            'model_type': self.model_type,
            'prediction_type': self.prediction_type,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'algorithm': self.algorithm,
            'hyperparameters': self.hyperparameters,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'training_data_hash': self.training_data_hash,
            'training_samples': self.training_samples,
            'training_features': self.training_features,
            'training_duration': self.training_duration,
            'training_score': self.training_score,
            'validation_score': self.validation_score,
            'test_score': self.test_score,
            'cross_validation_mean': self.cross_validation_mean,
            'cross_validation_std': self.cross_validation_std,
            'model_size_bytes': self.model_size_bytes,
            'model_file_path': self.model_file_path,
            'business_purpose': self.business_purpose,
            'expected_performance': self.expected_performance,
            'performance_threshold': self.performance_threshold,
            'data_sources': self.data_sources,
            'feature_engineering_steps': self.feature_engineering_steps,
            'validation_approach': self.validation_approach,
            'deployment_environment': self.deployment_environment,
            'deployment_date': self.deployment_date.isoformat() if self.deployment_date else None,
            'last_prediction_date': self.last_prediction_date.isoformat() if self.last_prediction_date else None,
            'prediction_count': self.prediction_count
        }

@dataclass
class ModelPerformance:
    """Model performance tracking"""
    accuracy_score: Optional[float] = None
    precision_score: Optional[float] = None
    recall_score: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc_score: Optional[float] = None
    
    # Regression metrics
    mse_score: Optional[float] = None
    rmse_score: Optional[float] = None
    mae_score: Optional[float] = None
    r2_score: Optional[float] = None
    
    # Financial metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_return: Optional[float] = None
    win_rate: Optional[float] = None
    
    # Additional metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert performance to dictionary"""
        return {
            'accuracy_score': self.accuracy_score,
            'precision_score': self.precision_score,
            'recall_score': self.recall_score,
            'f1_score': self.f1_score,
            'roc_auc_score': self.roc_auc_score,
            'mse_score': self.mse_score,
            'rmse_score': self.rmse_score,
            'mae_score': self.mae_score,
            'r2_score': self.r2_score,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_return': self.total_return,
            'win_rate': self.win_rate,
            'custom_metrics': self.custom_metrics
        }

# ============================================
# Base Model Class
# ============================================

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models
    
    Features:
    - Comprehensive model lifecycle management
    - Standardized training and prediction interface
    - Performance tracking and monitoring
    - Model persistence and versioning
    - Governance and compliance support
    - Financial domain integration
    """
    
    def __init__(self,
                 name: str,
                 model_type: str,
                 prediction_type: str = "regression",
                 version: str = "1.0.0",
                 model_params: Optional[Dict[str, Any]] = None,
                 business_purpose: Optional[str] = None,
                 expected_performance: Optional[float] = None,
                 performance_threshold: Optional[float] = None,
                 **kwargs):
        """
        Initialize base model
        
        Args:
            name: Model name
            model_type: Type of model (e.g., 'random_forest', 'xgboost')
            prediction_type: Type of prediction ('regression', 'classification')
            version: Model version
            model_params: Model hyperparameters
            business_purpose: Business purpose description
            expected_performance: Expected performance score
            performance_threshold: Minimum acceptable performance
            **kwargs: Additional model configuration
        """
        # Core model properties
        self.name = name
        self.model_type = model_type
        self.prediction_type = prediction_type
        self.version = version
        
        # Model parameters
        self.model_params = model_params or {}
        self.model_params.update(kwargs)
        
        # Model instance (will be created by subclasses)
        self.model: Optional[Any] = None
        
        # Model state
        self.status = ModelStatus.CREATED
        self.is_fitted = False
        
        # Training information
        self.feature_names: Optional[List[str]] = None
        self.target_name: Optional[str] = None
        self.training_score: Optional[float] = None
        self.validation_score: Optional[float] = None
        self.test_score: Optional[float] = None
        
        # Performance tracking
        self.performance: ModelPerformance = ModelPerformance()
        
        # Timing information
        self.last_training_time: Optional[datetime] = None
        self.last_prediction_time: Optional[datetime] = None
        self.training_duration: Optional[float] = None
        
        # Error tracking
        self.last_error: Optional[str] = None
        self.error_count: int = 0
        
        # Metadata
        self.metadata = ModelMetadata(
            model_id=f"{name}_{model_type}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=name,
            model_type=model_type,
            prediction_type=prediction_type,
            version=version,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            algorithm=model_type,
            hyperparameters=self.model_params,
            business_purpose=business_purpose,
            expected_performance=expected_performance,
            performance_threshold=performance_threshold
        )
        
        # Governance tracking
        self.governance_enabled = governance.config.get('enabled', False)
        
        logger.info(f"Initialized model {self.name} ({self.model_type}) version {self.version}")
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance. Must be implemented by subclasses."""
        pass
    
    def _validate_input_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> ValidationResult:
        """
        Validate input data for training or prediction
        
        Args:
            X: Feature matrix
            y: Target variable (optional, for training)
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()
        
        # Basic data validation
        if X is None or X.empty:
            result.add_error("Feature matrix X is empty or None")
            return result
        
        # Check for missing values
        if X.isnull().any().any():
            missing_cols = X.columns[X.isnull().any()].tolist()
            result.add_warning(f"Missing values found in columns: {missing_cols}")
        
        # Check data types
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            result.add_warning(f"Non-numeric columns found: {non_numeric_cols}")
        
        # Validate target variable if provided
        if y is not None:
            if y.isnull().any():
                result.add_warning(f"Missing values found in target variable ({y.isnull().sum()} missing)")
            
            if len(X) != len(y):
                result.add_error(f"Feature matrix and target variable length mismatch: {len(X)} vs {len(y)}")
        
        # Check for sufficient data
        if len(X) < 10:
            result.add_error("Insufficient data: need at least 10 samples")
        elif len(X) < 100:
            result.add_warning("Limited data: recommend at least 100 samples for reliable training")
        
        # Check feature matrix properties
        if X.shape[1] == 0:
            result.add_error("No features provided")
        elif X.shape[1] > len(X):
            result.add_warning("More features than samples - risk of overfitting")
        
        return result
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess feature matrix for model consumption
        
        Args:
            X: Feature matrix
            
        Returns:
            Preprocessed feature array
        """
        # Basic preprocessing - subclasses can override for specific needs
        
        # Handle missing values (simple forward fill)
        if X.isnull().any().any():
            X = X.fillna(method='ffill').fillna(0)
        
        # Convert to numpy array
        return X.values
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        """
        Fit the model to training data
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional fitting parameters
            
        Returns:
            Fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions using the fitted model
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions array
        """
        pass
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate model score on given data
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Model score
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before scoring")
        
        predictions = self.predict(X)
        
        # Default scoring based on prediction type
        if self.prediction_type == "classification":
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, predictions)
        else:
            from sklearn.metrics import r2_score
            return r2_score(y, predictions)
    
    def update_metadata(self, updates: Dict[str, Any]):
        """
        Update model metadata
        
        Args:
            updates: Dictionary of metadata updates
        """
        for key, value in updates.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
            else:
                logger.warning(f"Unknown metadata key: {key}")
        
        self.metadata.updated_at = datetime.now()
    
    def update_performance(self, metrics: Dict[str, float]):
        """
        Update model performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
        """
        for metric, value in metrics.items():
            if hasattr(self.performance, metric):
                setattr(self.performance, metric, value)
            else:
                self.performance.custom_metrics[metric] = value
        
        logger.debug(f"Updated performance metrics for {self.name}")
    
    def log_prediction(self):
        """Log prediction for tracking"""
        self.last_prediction_time = datetime.now()
        self.metadata.prediction_count += 1
        self.metadata.last_prediction_date = self.last_prediction_time
        
        if self.governance_enabled:
            log_governance_event(
                event_type='model_prediction',
                action='predict',
                resource=f"model:{self.metadata.model_id}",
                details={
                    'model_name': self.name,
                    'model_type': self.model_type,
                    'prediction_count': self.metadata.prediction_count
                }
            )
    
    def validate_model_health(self) -> ValidationResult:
        """
        Validate model health and readiness
        
        Returns:
            ValidationResult with health status
        """
        result = ValidationResult()
        
        # Check if model is fitted
        if not self.is_fitted:
            result.add_error("Model is not fitted")
        
        # Check model status
        if self.status == ModelStatus.ERROR:
            result.add_error(f"Model is in error state: {self.last_error}")
        
        # Check performance thresholds
        if (self.metadata.performance_threshold and 
            self.validation_score and 
            self.validation_score < self.metadata.performance_threshold):
            result.add_warning(f"Model performance ({self.validation_score:.3f}) below threshold ({self.metadata.performance_threshold:.3f})")
        
        # Check for recent activity
        if self.last_prediction_time:
            days_since_prediction = (datetime.now() - self.last_prediction_time).days
            if days_since_prediction > 30:
                result.add_warning(f"Model hasn't been used for {days_since_prediction} days")
        
        # Check error rate
        if self.error_count > 10:
            result.add_warning(f"High error count: {self.error_count}")
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary with model details
        """
        health = self.validate_model_health()
        
        return {
            'metadata': self.metadata.to_dict(),
            'performance': self.performance.to_dict(),
            'status': self.status.value,
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'training_duration': self.training_duration,
            'error_count': self.error_count,
            'health_status': health.is_valid,
            'health_issues': health.errors + health.warnings
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get concise model summary
        
        Returns:
            Dictionary with key model information
        """
        return {
            'name': self.name,
            'model_type': self.model_type,
            'prediction_type': self.prediction_type,
            'version': self.version,
            'status': self.status.value,
            'is_fitted': self.is_fitted,
            'training_score': self.training_score,
            'validation_score': self.validation_score,
            'test_score': self.test_score,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'prediction_count': self.metadata.prediction_count,
            'created_at': self.metadata.created_at.isoformat(),
            'last_updated': self.metadata.updated_at.isoformat()
        }
    
    def save_model(self, 
                   file_path: Optional[Union[str, Path]] = None,
                   save_format: str = 'joblib',
                   include_metadata: bool = True) -> str:
        """
        Save model to disk
        
        Args:
            file_path: Path to save model (auto-generated if None)
            save_format: Save format ('joblib', 'pickle')
            include_metadata: Whether to save metadata separately
            
        Returns:
            Path where model was saved
        """
        if not self.is_fitted:
            raise BusinessLogicError("Cannot save unfitted model")
        
        # Generate file path if not provided
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.name}_{self.model_type}_{timestamp}"
            file_path = Path(f"models/{filename}")
        else:
            file_path = Path(file_path)
        
        # Ensure directory exists
        ensure_directory(file_path.parent)
        
        try:
            # Prepare model data for saving
            model_data = {
                'model': self.model,
                'name': self.name,
                'model_type': self.model_type,
                'prediction_type': self.prediction_type,
                'version': self.version,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'model_params': self.model_params,
                'is_fitted': self.is_fitted,
                'training_score': self.training_score,
                'validation_score': self.validation_score,
                'test_score': self.test_score
            }
            
            # Save model
            if save_format == 'joblib':
                model_path = file_path.with_suffix('.joblib')
                joblib.dump(model_data, model_path)
            elif save_format == 'pickle':
                model_path = file_path.with_suffix('.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
            
            # Save metadata separately if requested
            if include_metadata:
                metadata_path = file_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(self.metadata.to_dict(), f, indent=2)
            
            # Update metadata
            self.metadata.model_file_path = str(model_path)
            self.metadata.model_size_bytes = model_path.stat().st_size
            
            # Log governance event
            if self.governance_enabled:
                log_governance_event(
                    event_type='model_save',
                    action='save_model',
                    resource=f"model:{self.metadata.model_id}",
                    details={
                        'model_name': self.name,
                        'file_path': str(model_path),
                        'save_format': save_format,
                        'model_size_bytes': self.metadata.model_size_bytes
                    }
                )
            
            logger.info(f"Model {self.name} saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save model {self.name}: {e}")
            raise
    
    @classmethod
    def load_model(cls, 
                   file_path: Union[str, Path],
                   load_metadata: bool = True) -> 'BaseModel':
        """
        Load model from disk
        
        Args:
            file_path: Path to model file
            load_metadata: Whether to load metadata
            
        Returns:
            Loaded model instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        try:
            # Load model data
            if file_path.suffix == '.joblib':
                model_data = joblib.load(file_path)
            elif file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    model_data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Create model instance
            # Note: This is a simplified loading - specific model classes should override
            model_instance = cls(
                name=model_data['name'],
                model_type=model_data['model_type'],
                prediction_type=model_data['prediction_type'],
                version=model_data['version'],
                model_params=model_data['model_params']
            )
            
            # Restore model state
            model_instance.model = model_data['model']
            model_instance.feature_names = model_data['feature_names']
            model_instance.target_name = model_data['target_name']
            model_instance.is_fitted = model_data['is_fitted']
            model_instance.training_score = model_data['training_score']
            model_instance.validation_score = model_data['validation_score']
            model_instance.test_score = model_data['test_score']
            
            # Load metadata if available
            if load_metadata:
                metadata_path = file_path.with_suffix('.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata_dict = json.load(f)
                    
                    # Update metadata (simplified - full restoration would need more work)
                    for key, value in metadata_dict.items():
                        if hasattr(model_instance.metadata, key):
                            setattr(model_instance.metadata, key, value)
            
            # Update status
            model_instance.status = ModelStatus.TRAINED if model_instance.is_fitted else ModelStatus.CREATED
            
            logger.info(f"Model {model_instance.name} loaded from {file_path}")
            return model_instance
            
        except Exception as e:
            logger.error(f"Failed to load model from {file_path}: {e}")
            raise
    
    def clone(self, name_suffix: str = "_clone") -> 'BaseModel':
        """
        Create a clone of the model
        
        Args:
            name_suffix: Suffix to add to cloned model name
            
        Returns:
            Cloned model instance
        """
        # Create new instance with same parameters
        cloned_model = self.__class__(
            name=self.name + name_suffix,
            model_type=self.model_type,
            prediction_type=self.prediction_type,
            version=self.version,
            model_params=self.model_params.copy(),
            business_purpose=self.metadata.business_purpose,
            expected_performance=self.metadata.expected_performance,
            performance_threshold=self.metadata.performance_threshold
        )
        
        # Copy fitted state if applicable
        if self.is_fitted and self.model is not None:
            try:
                # Try to clone the underlying model
                if hasattr(self.model, 'copy'):
                    cloned_model.model = self.model.copy()
                elif hasattr(self.model, 'clone'):
                    cloned_model.model = self.model.clone()
                else:
                    # Fallback: create new model with same parameters
                    cloned_model.model = cloned_model._create_model()
                
                # Copy other fitted attributes
                cloned_model.feature_names = self.feature_names.copy() if self.feature_names else None
                cloned_model.target_name = self.target_name
                cloned_model.is_fitted = False  # Clone needs to be retrained
                
            except Exception as e:
                logger.warning(f"Could not clone fitted model state: {e}")
        
        logger.info(f"Created clone of model {self.name}")
        return cloned_model
    
    def __repr__(self) -> str:
        """String representation of the model"""
        status_str = f"fitted" if self.is_fitted else "unfitted"
        score_str = f", score={self.validation_score:.3f}" if self.validation_score else ""
        
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.model_type}', {status_str}{score_str})"
    
    def __str__(self) -> str:
        """Human-readable string representation"""
        return f"{self.name} ({self.model_type}) - {self.status.value}"

# ============================================
# Model Registry and Management
# ============================================

class ModelRegistry:
    """
    Registry for managing multiple models
    
    Features:
    - Model registration and discovery
    - Version management
    - Performance tracking
    - Model comparison
    """
    
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.model_history: Dict[str, List[BaseModel]] = {}
        self.logger = get_logger('models.registry')
    
    def register_model(self, model: BaseModel) -> str:
        """
        Register a model in the registry
        
        Args:
            model: Model to register
            
        Returns:
            Model ID
        """
        model_id = model.metadata.model_id
        
        # Add to registry
        self.models[model_id] = model
        
        # Add to history
        if model.name not in self.model_history:
            self.model_history[model.name] = []
        self.model_history[model.name].append(model)
        
        self.logger.info(f"Registered model {model.name} with ID {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[BaseModel]:
        """Get model by ID"""
        return self.models.get(model_id)
    
    def get_models_by_name(self, name: str) -> List[BaseModel]:
        """Get all versions of a model by name"""
        return self.model_history.get(name, [])
    
    def get_best_model(self, name: str, metric: str = 'validation_score') -> Optional[BaseModel]:
        """
        Get best performing version of a model
        
        Args:
            name: Model name
            metric: Metric to use for comparison
            
        Returns:
            Best model or None
        """
        models = self.get_models_by_name(name)
        if not models:
            return None
        
        # Filter fitted models with scores
        fitted_models = [m for m in models if m.is_fitted and getattr(m, metric, None) is not None]
        
        if not fitted_models:
            return None
        
        # Return model with highest score
        return max(fitted_models, key=lambda m: getattr(m, metric))
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        return [model.get_model_summary() for model in self.models.values()]
    
    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """
        Compare models by performance metrics
        
        Args:
            model_names: List of model names to compare
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for name in model_names:
            best_model = self.get_best_model(name)
            if best_model:
                comparison_data.append({
                    'model_name': name,
                    'model_type': best_model.model_type,
                    'training_score': best_model.training_score,
                    'validation_score': best_model.validation_score,
                    'test_score': best_model.test_score,
                    'prediction_count': best_model.metadata.prediction_count,
                    'created_at': best_model.metadata.created_at,
                    'status': best_model.status.value
                })
        
        return pd.DataFrame(comparison_data)

# Global model registry
model_registry = ModelRegistry()

# ============================================
# Utility Functions
# ============================================

def create_model_id(name: str, model_type: str, version: str) -> str:
    """Create standardized model ID"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{name}_{model_type}_{version}_{timestamp}"

def validate_model_parameters(params: Dict[str, Any], 
                            required_params: List[str] = None,
                            param_types: Dict[str, type] = None) -> ValidationResult:
    """
    Validate model parameters
    
    Args:
        params: Parameters to validate
        required_params: List of required parameter names
        param_types: Expected types for parameters
        
    Returns:
        ValidationResult
    """
    result = ValidationResult()
    
    # Check required parameters
    if required_params:
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            result.add_error(f"Missing required parameters: {missing_params}")
    
    # Check parameter types
    if param_types:
        for param, expected_type in param_types.items():
            if param in params and not isinstance(params[param], expected_type):
                result.add_error(f"Parameter '{param}' must be of type {expected_type.__name__}, got {type(params[param]).__name__}")
    
    return result

def get_model_registry() -> ModelRegistry:
    """Get global model registry"""
    return model_registry
