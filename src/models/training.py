# ============================================
# StockPredictionPro - src/models/training.py
# Advanced model training with automated pipelines, monitoring, and production-ready workflows
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
import warnings
import json
import pickle
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import queue

# Core ML imports
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    train_test_split, cross_val_score, validation_curve, learning_curve,
    StratifiedKFold, KFold, TimeSeriesSplit
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# Import our modules
from .factory import model_factory, get_model_recommendations
from .persistence import default_registry, ModelStatus, register_model
from .selection import default_selector, run_model_selection_pipeline
from .optimization.bayesian_opt import BayesianOptimizer
from .optimization.grid_search import GridSearchOptimizer
from .optimization.optuna_opt import OptunaOptimizer
from .optimization.random_search import RandomSearchOptimizer

from ..utils.exceptions import ModelValidationError, BusinessLogicError
from ..utils.logger import get_logger
from ..utils.timing import Timer, time_it

logger = get_logger('models.training')

# ============================================
# Training Configuration and Status
# ============================================

class TrainingStatus(Enum):
    """Training status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class ValidationStrategy(Enum):
    """Validation strategy options"""
    TRAIN_TEST_SPLIT = "train_test_split"
    CROSS_VALIDATION = "cross_validation"
    TIME_SERIES_SPLIT = "time_series_split"
    HOLDOUT = "holdout"
    NONE = "none"

@dataclass
class TrainingConfig:
    """Comprehensive training configuration"""
    # Model configuration
    model_name: str = "gradient_boosting"
    model_params: Dict[str, Any] = None
    task_type: str = "classification"
    performance_preset: str = "balanced"
    
    # Data configuration
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    shuffle: bool = True
    
    # Validation configuration
    validation_strategy: ValidationStrategy = ValidationStrategy.CROSS_VALIDATION
    cv_folds: int = 5
    time_aware_cv: bool = True
    
    # Training configuration
    auto_optimize: bool = False
    optimization_method: str = "random_search"
    optimization_trials: int = 50
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = "accuracy"
    early_stopping_threshold: float = 0.001
    
    # Resource configuration
    n_jobs: int = -1
    memory_limit_gb: Optional[float] = None
    timeout_minutes: Optional[int] = None
    
    # Monitoring configuration
    enable_monitoring: bool = True
    log_level: str = "INFO"
    save_intermediate: bool = False
    checkpoint_frequency: int = 10
    
    # Output configuration
    auto_register: bool = True
    model_tags: List[str] = None
    model_description: str = ""
    save_training_data: bool = False
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {}
        if self.model_tags is None:
            self.model_tags = []

@dataclass
class TrainingResult:
    """Comprehensive training results"""
    # Model information
    model: BaseEstimator
    model_name: str
    model_id: Optional[str]
    
    # Performance metrics
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    cv_scores: Dict[str, List[float]]
    
    # Training information
    training_duration: float
    best_parameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    
    # Configuration and metadata
    config: TrainingConfig
    training_status: TrainingStatus
    error_message: Optional[str] = None
    
    # Training history
    training_history: List[Dict[str, Any]] = None
    optimization_history: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.training_history is None:
            self.training_history = []
    
    @property
    def primary_score(self) -> float:
        """Get primary validation score"""
        return self.validation_metrics.get('accuracy', 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result_dict = asdict(self)
        # Remove non-serializable model object
        result_dict.pop('model', None)
        result_dict['training_status'] = self.training_status.value
        result_dict['config']['validation_strategy'] = self.config.validation_strategy.value
        return result_dict

# ============================================
# Training Monitor
# ============================================

class TrainingMonitor:
    """Monitor training progress and resource usage"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.start_time = None
        self.metrics_history = []
        self.resource_usage = []
        self.is_monitoring = False
        self.stop_monitoring = threading.Event()
        
    def start_monitoring(self):
        """Start monitoring training"""
        if not self.config.enable_monitoring:
            return
        
        self.start_time = datetime.now()
        self.is_monitoring = True
        self.stop_monitoring.clear()
        
        # Start resource monitoring thread
        if self.config.memory_limit_gb or self.config.timeout_minutes:
            monitoring_thread = threading.Thread(target=self._monitor_resources)
            monitoring_thread.daemon = True
            monitoring_thread.start()
        
        logger.info("Training monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring training"""
        self.is_monitoring = False
        self.stop_monitoring.set()
        logger.info("Training monitoring stopped")
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """Log training metric"""
        if not self.is_monitoring:
            return
        
        metric_entry = {
            'timestamp': datetime.now(),
            'metric_name': metric_name,
            'value': value,
            'step': step,
            'elapsed_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }
        
        self.metrics_history.append(metric_entry)
        
        if self.config.log_level == "DEBUG":
            logger.debug(f"Metric {metric_name}: {value:.4f} at step {step}")
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics"""
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value, epoch)
    
    def _monitor_resources(self):
        """Monitor system resources during training"""
        try:
            import psutil
            process = psutil.Process()
            
            while not self.stop_monitoring.is_set():
                try:
                    # Memory usage
                    memory_info = process.memory_info()
                    memory_gb = memory_info.rss / 1024 / 1024 / 1024
                    
                    # CPU usage
                    cpu_percent = process.cpu_percent()
                    
                    # Log resource usage
                    resource_entry = {
                        'timestamp': datetime.now(),
                        'memory_gb': memory_gb,
                        'cpu_percent': cpu_percent,
                        'elapsed_time': (datetime.now() - self.start_time).total_seconds()
                    }
                    self.resource_usage.append(resource_entry)
                    
                    # Check memory limit
                    if self.config.memory_limit_gb and memory_gb > self.config.memory_limit_gb:
                        logger.warning(f"Memory usage ({memory_gb:.2f} GB) exceeded limit ({self.config.memory_limit_gb} GB)")
                        raise MemoryError(f"Memory limit exceeded: {memory_gb:.2f} GB > {self.config.memory_limit_gb} GB")
                    
                    # Check timeout
                    if self.config.timeout_minutes:
                        elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
                        if elapsed_minutes > self.config.timeout_minutes:
                            logger.warning(f"Training timeout ({elapsed_minutes:.1f} min) exceeded limit ({self.config.timeout_minutes} min)")
                            raise TimeoutError(f"Training timeout: {elapsed_minutes:.1f} min > {self.config.timeout_minutes} min")
                    
                    time.sleep(1)  # Check every second
                    
                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    logger.error(f"Error monitoring resources: {e}")
                    break
                    
        except ImportError:
            logger.warning("psutil not available for resource monitoring")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training monitoring summary"""
        if not self.metrics_history:
            return {}
        
        summary = {
            'training_duration': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'total_metrics_logged': len(self.metrics_history),
            'metrics_tracked': list(set(m['metric_name'] for m in self.metrics_history))
        }
        
        if self.resource_usage:
            memory_usage = [r['memory_gb'] for r in self.resource_usage]
            cpu_usage = [r['cpu_percent'] for r in self.resource_usage]
            
            summary.update({
                'peak_memory_gb': max(memory_usage),
                'avg_memory_gb': np.mean(memory_usage),
                'peak_cpu_percent': max(cpu_usage),
                'avg_cpu_percent': np.mean(cpu_usage)
            })
        
        return summary

# ============================================
# Early Stopping Implementation
# ============================================

class EarlyStopping:
    """Early stopping callback for training"""
    
    def __init__(self, 
                 patience: int = 10,
                 min_improvement: float = 0.001,
                 metric: str = "accuracy",
                 mode: str = "max"):
        self.patience = patience
        self.min_improvement = min_improvement
        self.metric = metric
        self.mode = mode
        
        self.best_score = None
        self.best_epoch = 0
        self.wait = 0
        self.should_stop = False
        self.history = []
    
    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Check if training should stop early"""
        
        current_score = metrics.get(self.metric)
        if current_score is None:
            return False
        
        self.history.append({
            'epoch': epoch,
            'score': current_score,
            'metrics': metrics.copy()
        })
        
        # Determine if current score is better
        is_better = False
        if self.best_score is None:
            is_better = True
        elif self.mode == "max":
            is_better = current_score > self.best_score + self.min_improvement
        elif self.mode == "min":
            is_better = current_score < self.best_score - self.min_improvement
        
        if is_better:
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
        
        # Check if should stop
        if self.wait >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered after {epoch} epochs. "
                       f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
        
        return self.should_stop
    
    def get_best_state(self) -> Dict[str, Any]:
        """Get best training state"""
        if not self.history:
            return {}
        
        best_entry = self.history[self.best_epoch] if self.best_epoch < len(self.history) else self.history[-1]
        return {
            'best_epoch': self.best_epoch,
            'best_score': self.best_score,
            'best_metrics': best_entry['metrics']
        }

# ============================================
# Model Trainer
# ============================================

class ModelTrainer:
    """Advanced model trainer with monitoring, optimization, and production features"""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.monitor = TrainingMonitor(self.config)
        self.early_stopping = None
        
        if self.config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_improvement=self.config.early_stopping_threshold,
                metric=self.config.early_stopping_metric
            )
        
        logger.info("Initialized ModelTrainer")
    
    def _prepare_data(self, 
                     X: Union[pd.DataFrame, np.ndarray], 
                     y: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Prepare and validate training data"""
        
        # Convert to arrays
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Validate data
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
        
        if len(X) == 0:
            raise ValueError("Empty dataset provided")
        
        # Check for missing values
        if np.isnan(X).any():
            logger.warning("Missing values detected in features")
        
        if np.isnan(y).any():
            logger.warning("Missing values detected in target")
        
        data_info = {
            'n_samples': len(X),
            'n_features': X.shape[1] if len(X.shape) > 1 else 1,
            'feature_names': feature_names,
            'has_missing_X': np.isnan(X).any(),
            'has_missing_y': np.isnan(y).any(),
            'target_classes': np.unique(y).tolist() if self.config.task_type == 'classification' else None
        }
        
        logger.info(f"Data prepared: {data_info['n_samples']} samples, {data_info['n_features']} features")
        
        return X, y, data_info
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation/test sets"""
        
        if self.config.validation_strategy == ValidationStrategy.NONE:
            return X, None, None, y, None, None
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if self.config.stratify and self.config.task_type == 'classification' else None,
            shuffle=self.config.shuffle
        )
        
        # Second split: separate train and validation
        if self.config.validation_strategy == ValidationStrategy.TRAIN_TEST_SPLIT:
            return X_temp, None, X_test, y_temp, None, y_test
        
        elif self.config.validation_strategy in [ValidationStrategy.HOLDOUT, ValidationStrategy.CROSS_VALIDATION]:
            # Create validation split
            val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.config.random_state,
                stratify=y_temp if self.config.stratify and self.config.task_type == 'classification' else None,
                shuffle=self.config.shuffle
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        elif self.config.validation_strategy == ValidationStrategy.TIME_SERIES_SPLIT:
            # For time series, use temporal splits
            split_idx = int(len(X_temp) * (1 - self.config.validation_size))
            X_train, X_val = X_temp[:split_idx], X_temp[split_idx:]
            y_train, y_val = y_temp[:split_idx], y_temp[split_idx:]
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        else:
            return X_temp, None, X_test, y_temp, None, y_test
    
    def _create_cv_splitter(self, X: np.ndarray, y: np.ndarray):
        """Create cross-validation splitter"""
        if self.config.time_aware_cv:
            return TimeSeriesSplit(n_splits=self.config.cv_folds)
        elif self.config.task_type == 'classification':
            return StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        else:
            return KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
    
    def _create_model(self) -> BaseEstimator:
        """Create model instance"""
        try:
            return model_factory.create_model(
                model_name=self.config.model_name,
                task_type=self.config.task_type,
                performance_preset=self.config.performance_preset,
                custom_params=self.config.model_params
            )
        except Exception as e:
            logger.error(f"Failed to create model {self.config.model_name}: {e}")
            raise ModelValidationError(f"Model creation failed: {e}")
    
    def _optimize_hyperparameters(self, 
                                 X: np.ndarray, 
                                 y: np.ndarray) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Optimize model hyperparameters"""
        
        logger.info(f"Starting hyperparameter optimization using {self.config.optimization_method}")
        
        # Create model factory function
        def model_factory_func(**params):
            combined_params = {**self.config.model_params, **params}
            return model_factory.create_model(
                model_name=self.config.model_name,
                task_type=self.config.task_type,
                performance_preset=self.config.performance_preset,
                custom_params=combined_params
            )
        
        try:
            # Select optimization method
            if self.config.optimization_method == "bayesian":
                from .optimization.bayesian_opt import create_financial_parameter_space
                param_space = create_financial_parameter_space(self.config.model_name)
                optimizer = BayesianOptimizer(
                    model_factory=model_factory_func,
                    parameter_space=param_space,
                    n_calls=self.config.optimization_trials,
                    time_aware_cv=self.config.time_aware_cv,
                    cv_folds=self.config.cv_folds,
                    verbose=False
                )
                
            elif self.config.optimization_method == "optuna":
                from .optimization.optuna_opt import create_financial_optuna_spaces
                param_space = create_financial_optuna_spaces(self.config.model_name)
                optimizer = OptunaOptimizer(
                    model_factory=model_factory_func,
                    parameter_space=param_space,
                    n_trials=self.config.optimization_trials,
                    time_aware_cv=self.config.time_aware_cv,
                    cv_folds=self.config.cv_folds,
                    verbose=False
                )
                
            elif self.config.optimization_method == "grid_search":
                from .optimization.grid_search import create_financial_grid_spaces
                param_grid = create_financial_grid_spaces(self.config.model_name, resolution='coarse')
                optimizer = GridSearchOptimizer(
                    model_factory=model_factory_func,
                    parameter_grid=param_grid,
                    early_stopping=True,
                    time_aware_cv=self.config.time_aware_cv,
                    cv_folds=self.config.cv_folds,
                    verbose=0
                )
                
            else:  # random_search
                from .optimization.random_search import create_financial_random_spaces
                param_space = create_financial_random_spaces(self.config.model_name)
                optimizer = RandomSearchOptimizer(
                    model_factory=model_factory_func,
                    parameter_space=param_space,
                    n_iter=self.config.optimization_trials,
                    time_aware_cv=self.config.time_aware_cv,
                    cv_folds=self.config.cv_folds,
                    verbose=False
                )
            
            # Run optimization
            results = optimizer.optimize(X, y)
            
            optimized_model = results['best_model']
            optimization_history = results
            
            logger.info(f"Hyperparameter optimization completed. Best score: {results['best_score']:.4f}")
            
            return optimized_model, optimization_history
            
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {e}. Using default model.")
            return self._create_model(), {}
    
    def _evaluate_model(self, 
                       model: BaseEstimator,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                       X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """Comprehensive model evaluation"""
        
        evaluation_results = {
            'training_metrics': {},
            'validation_metrics': {},
            'test_metrics': {},
            'cv_scores': {}
        }
        
        # Define metrics based on task type
        if self.config.task_type == 'classification':
            metric_functions = {
                'accuracy': accuracy_score,
                'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # Add ROC AUC for binary classification
            if len(np.unique(y_train)) == 2:
                metric_functions['roc_auc'] = roc_auc_score
        else:
            metric_functions = {
                'r2_score': r2_score,
                'mae': mean_absolute_error,
                'mse': mean_squared_error,
                'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
            }
        
        # Training metrics
        train_predictions = model.predict(X_train)
        for metric_name, metric_func in metric_functions.items():
            try:
                if metric_name == 'roc_auc' and hasattr(model, 'predict_proba'):
                    train_proba = model.predict_proba(X_train)[:, 1]
                    score = metric_func(y_train, train_proba)
                else:
                    score = metric_func(y_train, train_predictions)
                evaluation_results['training_metrics'][metric_name] = float(score)
            except Exception as e:
                logger.warning(f"Error calculating training {metric_name}: {e}")
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            val_predictions = model.predict(X_val)
            for metric_name, metric_func in metric_functions.items():
                try:
                    if metric_name == 'roc_auc' and hasattr(model, 'predict_proba'):
                        val_proba = model.predict_proba(X_val)[:, 1]
                        score = metric_func(y_val, val_proba)
                    else:
                        score = metric_func(y_val, val_predictions)
                    evaluation_results['validation_metrics'][metric_name] = float(score)
                except Exception as e:
                    logger.warning(f"Error calculating validation {metric_name}: {e}")
        
        # Test metrics
        if X_test is not None and y_test is not None:
            test_predictions = model.predict(X_test)
            for metric_name, metric_func in metric_functions.items():
                try:
                    if metric_name == 'roc_auc' and hasattr(model, 'predict_proba'):
                        test_proba = model.predict_proba(X_test)[:, 1]
                        score = metric_func(y_test, test_proba)
                    else:
                        score = metric_func(y_test, test_predictions)
                    evaluation_results['test_metrics'][metric_name] = float(score)
                except Exception as e:
                    logger.warning(f"Error calculating test {metric_name}: {e}")
        
        # Cross-validation scores
        if self.config.validation_strategy == ValidationStrategy.CROSS_VALIDATION:
            cv_splitter = self._create_cv_splitter(X_train, y_train)
            
            for metric_name in metric_functions.keys():
                try:
                    if metric_name == 'roc_auc':
                        scorer = 'roc_auc'
                    elif metric_name == 'mae':
                        scorer = 'neg_mean_absolute_error'
                    elif metric_name == 'mse':
                        scorer = 'neg_mean_squared_error'
                    elif metric_name == 'rmse':
                        scorer = 'neg_root_mean_squared_error'
                    else:
                        scorer = metric_name
                    
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=cv_splitter,
                        scoring=scorer,
                        n_jobs=1
                    )
                    
                    # Convert negative scores back to positive
                    if metric_name in ['mae', 'mse', 'rmse']:
                        cv_scores = -cv_scores
                    
                    evaluation_results['cv_scores'][metric_name] = cv_scores.tolist()
                    
                except Exception as e:
                    logger.warning(f"Error calculating CV {metric_name}: {e}")
        
        return evaluation_results
    
    def _extract_feature_importance(self, model: BaseEstimator, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                return {name: float(imp) for name, imp in zip(feature_names, importances)}
            
            elif hasattr(model, 'coef_'):
                # Linear models
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef[0]  # Take first class for multiclass
                return {name: float(abs(c)) for name, c in zip(feature_names, coef)}
            
            elif hasattr(model, 'named_steps'):
                # Pipeline - try to extract from the final estimator
                final_estimator = model.named_steps.get('model') or model.steps[-1][1]
                return self._extract_feature_importance(final_estimator, feature_names)
            
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error extracting feature importance: {e}")
            return None
    
    @time_it("model_training", include_args=True)
    def train(self, 
              X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray]) -> TrainingResult:
        """Train model with comprehensive monitoring and evaluation"""
        
        logger.info(f"Starting training: {self.config.model_name} ({self.config.task_type})")
        
        # Start monitoring
        self.monitor.start_monitoring()
        training_start = datetime.now()
        
        try:
            # Prepare data
            X, y, data_info = self._prepare_data(X, y)
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)
            
            logger.info(f"Data splits - Train: {len(X_train)}, "
                       f"Val: {len(X_val) if X_val is not None else 0}, "
                       f"Test: {len(X_test) if X_test is not None else 0}")
            
            # Create or optimize model
            optimization_history = None
            if self.config.auto_optimize:
                model, optimization_history = self._optimize_hyperparameters(X_train, y_train)
            else:
                model = self._create_model()
            
            # Train model
            logger.info("Training model...")
            fit_start = datetime.now()
            
            # Handle pipeline or regular model
            if isinstance(model, Pipeline):
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            fit_duration = (datetime.now() - fit_start).total_seconds()
            
            # Log training completion
            self.monitor.log_metric('training_complete', 1.0)
            
            # Evaluate model
            logger.info("Evaluating model...")
            evaluation_results = self._evaluate_model(
                model, X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Extract feature importance
            feature_importance = self._extract_feature_importance(model, data_info['feature_names'])
            
            # Calculate total training duration
            total_duration = (datetime.now() - training_start).total_seconds()
            
            # Create training result
            result = TrainingResult(
                model=model,
                model_name=self.config.model_name,
                model_id=None,  # Will be set if registered
                training_metrics=evaluation_results['training_metrics'],
                validation_metrics=evaluation_results['validation_metrics'],
                test_metrics=evaluation_results['test_metrics'],
                cv_scores=evaluation_results['cv_scores'],
                training_duration=total_duration,
                best_parameters=model.get_params(),
                feature_importance=feature_importance,
                config=self.config,
                training_status=TrainingStatus.COMPLETED,
                optimization_history=optimization_history
            )
            
            # Auto-register model if requested
            if self.config.auto_register:
                try:
                    model_id = register_model(
                        model=model,
                        model_name=f"{self.config.model_name}_trained",
                        task_type=self.config.task_type,
                        training_data=(X_train, y_train) if self.config.save_training_data else (np.array([[0]]), np.array([0])),
                        performance_metrics=result.validation_metrics or result.training_metrics,
                        hyperparameters=result.best_parameters,
                        description=self.config.model_description or f"Trained {self.config.model_name} model",
                        tags=self.config.model_tags + ['auto_trained'],
                        cross_validation_scores=result.cv_scores.get('accuracy', []),
                        feature_importance=result.feature_importance,
                        training_duration=result.training_duration
                    )
                    result.model_id = model_id
                    logger.info(f"Model registered with ID: {model_id}")
                except Exception as e:
                    logger.warning(f"Failed to register model: {e}")
            
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Add monitoring summary to result
            result.training_history = [{
                'monitoring_summary': self.monitor.get_training_summary(),
                'data_info': data_info
            }]
            
            logger.info(f"Training completed successfully. "
                       f"Primary score: {result.primary_score:.4f}, "
                       f"Duration: {total_duration:.1f}s")
            
            return result
            
        except Exception as e:
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Create failed result
            logger.error(f"Training failed: {e}")
            
            result = TrainingResult(
                model=None,
                model_name=self.config.model_name,
                model_id=None,
                training_metrics={},
                validation_metrics={},
                test_metrics={},
                cv_scores={},
                training_duration=(datetime.now() - training_start).total_seconds(),
                best_parameters={},
                feature_importance=None,
                config=self.config,
                training_status=TrainingStatus.FAILED,
                error_message=str(e)
            )
            
            return result

# ============================================
# Batch Training Manager
# ============================================

class BatchTrainingManager:
    """Manage batch training of multiple models"""
    
    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self.training_queue = queue.Queue()
        self.active_trainings = {}
        self.completed_trainings = {}
        
    def add_training_job(self, 
                        job_id: str,
                        config: TrainingConfig,
                        X: Union[pd.DataFrame, np.ndarray],
                        y: Union[pd.Series, np.ndarray]) -> str:
        """Add training job to queue"""
        
        job = {
            'job_id': job_id,
            'config': config,
            'X': X,
            'y': y,
            'submitted_at': datetime.now(),
            'status': TrainingStatus.PENDING
        }
        
        self.training_queue.put(job)
        logger.info(f"Added training job {job_id} to queue")
        
        return job_id
    
    def start_batch_training(self) -> Dict[str, TrainingResult]:
        """Start batch training process"""
        
        logger.info(f"Starting batch training with max {self.max_concurrent} concurrent jobs")
        
        threads = []
        
        while not self.training_queue.empty() or threads:
            # Start new threads if under limit
            while len(threads) < self.max_concurrent and not self.training_queue.empty():
                job = self.training_queue.get()
                thread = threading.Thread(target=self._train_job, args=(job,))
                thread.start()
                threads.append((thread, job['job_id']))
                logger.info(f"Started training job {job['job_id']}")
            
            # Check for completed threads
            completed_threads = []
            for thread, job_id in threads:
                if not thread.is_alive():
                    thread.join()
                    completed_threads.append((thread, job_id))
            
            # Remove completed threads
            for completed in completed_threads:
                threads.remove(completed)
            
            time.sleep(1)  # Brief pause
        
        logger.info(f"Batch training completed. {len(self.completed_trainings)} jobs finished")
        return self.completed_trainings.copy()
    
    def _train_job(self, job: Dict[str, Any]):
        """Train a single job"""
        job_id = job['job_id']
        
        try:
            self.active_trainings[job_id] = {
                'job': job,
                'started_at': datetime.now(),
                'status': TrainingStatus.RUNNING
            }
            
            # Create trainer and train
            trainer = ModelTrainer(job['config'])
            result = trainer.train(job['X'], job['y'])
            
            # Store result
            self.completed_trainings[job_id] = result
            
            # Clean up
            if job_id in self.active_trainings:
                del self.active_trainings[job_id]
            
            logger.info(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            
            # Store failed result
            result = TrainingResult(
                model=None,
                model_name=job['config'].model_name,
                model_id=None,
                training_metrics={},
                validation_metrics={},
                test_metrics={},
                cv_scores={},
                training_duration=0,
                best_parameters={},
                feature_importance=None,
                config=job['config'],
                training_status=TrainingStatus.FAILED,
                error_message=str(e)
            )
            
            self.completed_trainings[job_id] = result
            
            # Clean up
            if job_id in self.active_trainings:
                del self.active_trainings[job_id]
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'queued': self.training_queue.qsize(),
            'active': len(self.active_trainings),
            'completed': len(self.completed_trainings),
            'active_jobs': list(self.active_trainings.keys()),
            'completed_jobs': list(self.completed_trainings.keys())
        }

# ============================================
# Training Pipeline
# ============================================

class TrainingPipeline:
    """Complete training pipeline with model selection and optimization"""
    
    def __init__(self):
        self.pipeline_history = []
    
    @time_it("training_pipeline", include_args=True)
    def run_automated_training_pipeline(self,
                                       X: Union[pd.DataFrame, np.ndarray],
                                       y: Union[pd.Series, np.ndarray],
                                       task_type: str = 'classification',
                                       use_model_selection: bool = True,
                                       use_hyperparameter_optimization: bool = True,
                                       register_all_models: bool = False) -> Dict[str, Any]:
        """Run complete automated training pipeline"""
        
        logger.info("Starting automated training pipeline")
        pipeline_start = datetime.now()
        
        pipeline_result = {
            'pipeline_start': pipeline_start,
            'pipeline_config': {
                'task_type': task_type,
                'use_model_selection': use_model_selection,
                'use_hyperparameter_optimization': use_hyperparameter_optimization,
                'register_all_models': register_all_models
            },
            'model_selection_result': None,
            'training_results': {},
            'best_model_result': None,
            'pipeline_duration': 0
        }
        
        try:
            # Step 1: Model Selection (if requested)
            if use_model_selection:
                logger.info("Step 1: Running model selection")
                selection_result = run_model_selection_pipeline(
                    X, y, task_type,
                    include_ensemble_optimization=True,
                    register_best_model=False  # We'll handle registration manually
                )
                pipeline_result['model_selection_result'] = selection_result
                
                # Get selected model configuration
                selected_model_name = selection_result['final_result'].model_name
                selected_hyperparams = selection_result['final_result'].hyperparameters
            else:
                # Use default model
                data_characteristics = {
                    'n_samples': len(X),
                    'n_features': X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else 1,
                    'has_missing': False,
                    'has_categorical': False
                }
                
                recommendations = get_model_recommendations(data_characteristics, task_type)
                selected_model_name = recommendations[0]['model_name'] if recommendations else 'gradient_boosting'
                selected_hyperparams = {}
            
            # Step 2: Train selected model with optimization
            logger.info(f"Step 2: Training selected model: {selected_model_name}")
            
            training_config = TrainingConfig(
                model_name=selected_model_name,
                model_params=selected_hyperparams,
                task_type=task_type,
                auto_optimize=use_hyperparameter_optimization,
                optimization_method='optuna',
                optimization_trials=50,
                auto_register=True,
                model_description=f"Model trained through automated pipeline on {datetime.now().strftime('%Y-%m-%d')}",
                model_tags=['auto_pipeline', task_type, 'optimized' if use_hyperparameter_optimization else 'default']
            )
            
            trainer = ModelTrainer(training_config)
            training_result = trainer.train(X, y)
            
            pipeline_result['training_results'][selected_model_name] = training_result
            pipeline_result['best_model_result'] = training_result
            
            # Step 3: Train additional models for comparison (if requested)
            if register_all_models and use_model_selection:
                logger.info("Step 3: Training additional models for comparison")
                
                selection_results = pipeline_result['model_selection_result']['individual_selection']['all_results']
                top_models = selection_results[:3]  # Train top 3 models
                
                for model_result in top_models[1:]:  # Skip the first one (already trained)
                    model_name = model_result.model_name
                    
                    try:
                        additional_config = TrainingConfig(
                            model_name=model_name,
                            task_type=task_type,
                            auto_optimize=False,  # Use default parameters for speed
                            auto_register=True,
                            model_description=f"Additional model from pipeline comparison",
                            model_tags=['auto_pipeline', 'comparison', task_type]
                        )
                        
                        additional_trainer = ModelTrainer(additional_config)
                        additional_result = trainer.train(X, y)
                        
                        pipeline_result['training_results'][model_name] = additional_result
                        
                    except Exception as e:
                        logger.warning(f"Failed to train additional model {model_name}: {e}")
            
            # Calculate pipeline duration
            pipeline_result['pipeline_duration'] = (datetime.now() - pipeline_start).total_seconds()
            
            # Store in history
            self.pipeline_history.append(pipeline_result)
            
            logger.info(f"Automated training pipeline completed successfully in {pipeline_result['pipeline_duration']:.1f}s")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Automated training pipeline failed: {e}")
            pipeline_result['pipeline_duration'] = (datetime.now() - pipeline_start).total_seconds()
            pipeline_result['error'] = str(e)
            return pipeline_result

# ============================================
# Global Instances and Convenience Functions
# ============================================

# Global instances
default_trainer = ModelTrainer()
default_batch_manager = BatchTrainingManager()
default_pipeline = TrainingPipeline()

# Convenience functions
def train_model(X: Union[pd.DataFrame, np.ndarray],
               y: Union[pd.Series, np.ndarray],
               model_name: str = "gradient_boosting",
               task_type: str = "classification",
               **config_kwargs) -> TrainingResult:
    """Train a model with default configuration"""
    config = TrainingConfig(
        model_name=model_name,
        task_type=task_type,
        **config_kwargs
    )
    trainer = ModelTrainer(config)
    return trainer.train(X, y)

def run_automated_pipeline(X: Union[pd.DataFrame, np.ndarray],
                          y: Union[pd.Series, np.ndarray],
                          task_type: str = "classification",
                          **kwargs) -> Dict[str, Any]:
    """Run automated training pipeline"""
    return default_pipeline.run_automated_training_pipeline(X, y, task_type, **kwargs)

def create_training_config(model_name: str = "gradient_boosting",
                          task_type: str = "classification",
                          **kwargs) -> TrainingConfig:
    """Create training configuration"""
    return TrainingConfig(
        model_name=model_name,
        task_type=task_type,
        **kwargs
    )
