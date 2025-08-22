# ============================================
# StockPredictionPro - src/models/factory.py
# Advanced model factory with intelligent model creation, configuration management, and auto-tuning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Type
from datetime import datetime
import warnings
import json
import inspect

# Core ML imports
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import all our model creation functions
# Classification models
from .classification.gradient_boosting import create_gradient_boosting_classifier
from .classification.random_forest import create_random_forest_classifier
from .classification.svm import create_svm_classifier
from .classification.logistic import create_logistic_classifier
from .classification.naive_bayes import create_naive_bayes_classifier
from .classification.knn import create_knn_classifier
from .classification.neural_network import create_neural_network_classifier
from .classification.ensemble import create_ensemble_classifier  # From classification ensemble

# Regression models
try:
    from .regression.gradient_boosting import create_gradient_boosting_regressor
    from .regression.random_forest import create_random_forest_regressor
    from .regression.svr import create_svr_regressor  # SVR not SVM
    from .regression.linear import create_linear_regressor
    from .regression.ridge import create_ridge_regressor
    from .regression.lasso import create_lasso_regressor
    from .regression.elastic_net import create_elastic_net_regressor
    from .regression.polynomial import create_polynomial_regressor
    from .regression.multiple import create_multiple_regressor
    from .regression.neural_network import create_neural_network_regressor
    REGRESSION_AVAILABLE = True
except ImportError:
    REGRESSION_AVAILABLE = False
    warnings.warn("Some regression models not available")

# Ensemble models
from .ensemble.voting import create_voting_classifier, create_voting_regressor
from .ensemble.stacking import create_stacking_classifier, create_stacking_regressor
from .ensemble.bagging import create_bagging_classifier, create_bagging_regressor
from .ensemble.blending import create_blending_classifier, create_blending_regressor

# Optimization modules
from .optimization.bayesian_opt import BayesianOptimizer, create_financial_parameter_space
from .optimization.grid_search import GridSearchOptimizer, create_financial_grid_spaces
from .optimization.optuna_opt import OptunaOptimizer, create_financial_optuna_spaces
from .optimization.random_search import RandomSearchOptimizer, create_financial_random_spaces

from ..utils.exceptions import ModelValidationError, BusinessLogicError
from ..utils.logger import get_logger
from ..utils.timing import Timer, time_it

logger = get_logger('models.factory')

# ============================================
# Model Registry and Configuration
# ============================================

class ModelRegistry:
    """Central registry for all available models and their configurations"""
    
    def __init__(self):
        self._classification_models = {}
        self._regression_models = {}
        self._ensemble_models = {}
        self._model_metadata = {}
        
        # Initialize registry
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize the model registry with all available models"""
        
        # Classification models
        self._classification_models = {
            'gradient_boosting': {
                'factory_func': create_gradient_boosting_classifier,
                'category': 'tree_based',
                'complexity': 'high',
                'training_time': 'medium',
                'interpretability': 'medium',
                'handles_missing': True,
                'handles_categorical': True,
                'description': 'Gradient Boosting Classifier with advanced regularization'
            },
            'random_forest': {
                'factory_func': create_random_forest_classifier,
                'category': 'tree_based',
                'complexity': 'medium',
                'training_time': 'fast',
                'interpretability': 'medium',
                'handles_missing': False,
                'handles_categorical': True,
                'description': 'Random Forest with optimized parameters'
            },
            'svm': {
                'factory_func': create_svm_classifier,
                'category': 'kernel_based',
                'complexity': 'high',
                'training_time': 'slow',
                'interpretability': 'low',
                'handles_missing': False,
                'handles_categorical': False,
                'description': 'Support Vector Machine with multiple kernels'
            },
            'logistic': {
                'factory_func': create_logistic_classifier,
                'category': 'linear',
                'complexity': 'low',
                'training_time': 'fast',
                'interpretability': 'high',
                'handles_missing': False,
                'handles_categorical': False,
                'description': 'Logistic Regression with regularization'
            },
            'naive_bayes': {
                'factory_func': create_naive_bayes_classifier,
                'category': 'probabilistic',
                'complexity': 'low',
                'training_time': 'very_fast',
                'interpretability': 'medium',
                'handles_missing': False,
                'handles_categorical': True,
                'description': 'Naive Bayes with multiple distributions'
            },
            'knn': {
                'factory_func': create_knn_classifier,
                'category': 'distance_based',
                'complexity': 'low',
                'training_time': 'very_fast',
                'interpretability': 'low',
                'handles_missing': False,
                'handles_categorical': False,
                'description': 'K-Nearest Neighbors with adaptive parameters'
            },
            'neural_network': {
                'factory_func': create_neural_network_classifier,
                'category': 'neural',
                'complexity': 'high',
                'training_time': 'slow',
                'interpretability': 'low',
                'handles_missing': False,
                'handles_categorical': False,
                'description': 'Multi-layer Perceptron with adaptive architecture'
            },
            'ensemble': {
                'factory_func': create_ensemble_classifier,
                'category': 'ensemble',
                'complexity': 'high',
                'training_time': 'slow',
                'interpretability': 'low',
                'handles_missing': True,
                'handles_categorical': True,
                'description': 'Ensemble classifier from classification module'
            }
        }
        
        # Regression models (if available)
        if REGRESSION_AVAILABLE:
            self._regression_models = {
                'gradient_boosting': {
                    'factory_func': create_gradient_boosting_regressor,
                    'category': 'tree_based',
                    'complexity': 'high',
                    'training_time': 'medium',
                    'interpretability': 'medium',
                    'handles_missing': True,
                    'handles_categorical': True,
                    'description': 'Gradient Boosting Regressor with advanced regularization'
                },
                'random_forest': {
                    'factory_func': create_random_forest_regressor,
                    'category': 'tree_based',
                    'complexity': 'medium',
                    'training_time': 'fast',
                    'interpretability': 'medium',
                    'handles_missing': False,
                    'handles_categorical': True,
                    'description': 'Random Forest Regressor'
                },
                'svr': {
                    'factory_func': create_svr_regressor,
                    'category': 'kernel_based',
                    'complexity': 'high',
                    'training_time': 'slow',
                    'interpretability': 'low',
                    'handles_missing': False,
                    'handles_categorical': False,
                    'description': 'Support Vector Regression'
                },
                'linear': {
                    'factory_func': create_linear_regressor,
                    'category': 'linear',
                    'complexity': 'low',
                    'training_time': 'very_fast',
                    'interpretability': 'high',
                    'handles_missing': False,
                    'handles_categorical': False,
                    'description': 'Linear Regression'
                },
                'ridge': {
                    'factory_func': create_ridge_regressor,
                    'category': 'linear',
                    'complexity': 'low',
                    'training_time': 'very_fast',
                    'interpretability': 'high',
                    'handles_missing': False,
                    'handles_categorical': False,
                    'description': 'Ridge Regression with L2 regularization'
                },
                'lasso': {
                    'factory_func': create_lasso_regressor,
                    'category': 'linear',
                    'complexity': 'low',
                    'training_time': 'fast',
                    'interpretability': 'high',
                    'handles_missing': False,
                    'handles_categorical': False,
                    'description': 'Lasso Regression with L1 regularization'
                },
                'elastic_net': {
                    'factory_func': create_elastic_net_regressor,
                    'category': 'linear',
                    'complexity': 'low',
                    'training_time': 'fast',
                    'interpretability': 'high',
                    'handles_missing': False,
                    'handles_categorical': False,
                    'description': 'Elastic Net with L1 + L2 regularization'
                },
                'polynomial': {
                    'factory_func': create_polynomial_regressor,
                    'category': 'polynomial',
                    'complexity': 'medium',
                    'training_time': 'fast',
                    'interpretability': 'medium',
                    'handles_missing': False,
                    'handles_categorical': False,
                    'description': 'Polynomial Regression with feature transformation'
                },
                'multiple': {
                    'factory_func': create_multiple_regressor,
                    'category': 'linear',
                    'complexity': 'low',
                    'training_time': 'very_fast',
                    'interpretability': 'high',
                    'handles_missing': False,
                    'handles_categorical': False,
                    'description': 'Multiple Linear Regression'
                },
                'neural_network': {
                    'factory_func': create_neural_network_regressor,
                    'category': 'neural',
                    'complexity': 'high',
                    'training_time': 'slow',
                    'interpretability': 'low',
                    'handles_missing': False,
                    'handles_categorical': False,
                    'description': 'Neural Network Regressor'
                }
            }
        
        # Ensemble models
        self._ensemble_models = {
            'voting_classifier': {
                'factory_func': create_voting_classifier,
                'task_type': 'classification',
                'category': 'ensemble',
                'complexity': 'high',
                'training_time': 'slow',
                'interpretability': 'low',
                'description': 'Voting ensemble of multiple classifiers'
            },
            'voting_regressor': {
                'factory_func': create_voting_regressor,
                'task_type': 'regression',
                'category': 'ensemble',
                'complexity': 'high',
                'training_time': 'slow',
                'interpretability': 'low',
                'description': 'Voting ensemble of multiple regressors'
            },
            'stacking_classifier': {
                'factory_func': create_stacking_classifier,
                'task_type': 'classification',
                'category': 'ensemble',
                'complexity': 'very_high',
                'training_time': 'slow',
                'interpretability': 'very_low',
                'description': 'Stacking ensemble with meta-learning'
            },
            'stacking_regressor': {
                'factory_func': create_stacking_regressor,
                'task_type': 'regression',
                'category': 'ensemble',
                'complexity': 'very_high',
                'training_time': 'slow',
                'interpretability': 'very_low',
                'description': 'Stacking ensemble with meta-learning'
            },
            'bagging_classifier': {
                'factory_func': create_bagging_classifier,
                'task_type': 'classification',
                'category': 'ensemble',
                'complexity': 'medium',
                'training_time': 'medium',
                'interpretability': 'low',
                'description': 'Bagging ensemble with bootstrap sampling'
            },
            'bagging_regressor': {
                'factory_func': create_bagging_regressor,
                'task_type': 'regression',
                'category': 'ensemble',
                'complexity': 'medium',
                'training_time': 'medium',
                'interpretability': 'low',
                'description': 'Bagging ensemble with bootstrap sampling'
            },
            'blending_classifier': {
                'factory_func': create_blending_classifier,
                'task_type': 'classification',
                'category': 'ensemble',
                'complexity': 'high',
                'training_time': 'medium',
                'interpretability': 'low',
                'description': 'Blending ensemble with holdout validation'
            },
            'blending_regressor': {
                'factory_func': create_blending_regressor,
                'task_type': 'regression',
                'category': 'ensemble',
                'complexity': 'high',
                'training_time': 'medium',
                'interpretability': 'low',
                'description': 'Blending ensemble with holdout validation'
            }
        }
    
    def get_available_models(self, task_type: str = None) -> Dict[str, Dict]:
        """Get all available models, optionally filtered by task type"""
        if task_type == 'classification':
            return self._classification_models.copy()
        elif task_type == 'regression':
            return self._regression_models.copy() if REGRESSION_AVAILABLE else {}
        elif task_type == 'ensemble':
            return self._ensemble_models.copy()
        else:
            # Return all models
            all_models = {}
            all_models.update(self._classification_models)
            if REGRESSION_AVAILABLE:
                all_models.update(self._regression_models)
            all_models.update(self._ensemble_models)
            return all_models
    
    def get_model_info(self, model_name: str, task_type: str = None) -> Optional[Dict]:
        """Get information about a specific model"""
        if task_type == 'classification':
            return self._classification_models.get(model_name)
        elif task_type == 'regression':
            return self._regression_models.get(model_name) if REGRESSION_AVAILABLE else None
        elif task_type == 'ensemble':
            return self._ensemble_models.get(model_name)
        else:
            # Search in all categories
            all_models = self.get_available_models()
            return all_models.get(model_name)
    
    def get_models_by_category(self, category: str, task_type: str = None) -> Dict[str, Dict]:
        """Get models filtered by category"""
        available_models = self.get_available_models(task_type)
        return {
            name: info for name, info in available_models.items()
            if info.get('category') == category
        }
    
    def get_models_by_criteria(self, 
                              complexity: Optional[str] = None,
                              training_time: Optional[str] = None,
                              interpretability: Optional[str] = None,
                              handles_missing: Optional[bool] = None,
                              task_type: Optional[str] = None) -> Dict[str, Dict]:
        """Get models filtered by multiple criteria"""
        available_models = self.get_available_models(task_type)
        filtered_models = {}
        
        for name, info in available_models.items():
            # Check all criteria
            if complexity and info.get('complexity') != complexity:
                continue
            if training_time and info.get('training_time') != training_time:
                continue
            if interpretability and info.get('interpretability') != interpretability:
                continue
            if handles_missing is not None and info.get('handles_missing') != handles_missing:
                continue
            
            filtered_models[name] = info
        
        return filtered_models

# Global model registry instance
model_registry = ModelRegistry()

# ============================================
# Advanced Model Factory
# ============================================

class ModelFactory:
    """Advanced model factory with intelligent model creation and configuration management"""
    
    def __init__(self):
        self.registry = model_registry
        self.created_models = {}
        self.model_cache = {}
        
        logger.info("Initialized ModelFactory with comprehensive model registry")
    
    @time_it("model_creation", include_args=True)
    def create_model(self, 
                     model_name: str,
                     task_type: str = 'classification',
                     performance_preset: str = 'balanced',
                     custom_params: Optional[Dict[str, Any]] = None,
                     auto_tune: bool = False,
                     tune_method: str = 'random_search',
                     tune_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                     **kwargs) -> BaseEstimator:
        """
        Create a model with advanced configuration options
        
        Args:
            model_name: Name of the model to create
            task_type: Type of task ('classification', 'regression', 'ensemble')
            performance_preset: Performance preset ('fast', 'balanced', 'accurate')
            custom_params: Custom parameters to override defaults
            auto_tune: Whether to automatically tune hyperparameters
            tune_method: Tuning method ('random_search', 'grid_search', 'bayesian', 'optuna')
            tune_data: Data for hyperparameter tuning (X, y)
        
        Returns:
            Configured model instance
        """
        logger.info(f"Creating model: {model_name} for {task_type} task")
        
        # Get model info
        model_info = self.registry.get_model_info(model_name, task_type)
        
        if not model_info:
            # Try to find in any category
            all_models = self.registry.get_available_models()
            if model_name not in all_models:
                available_models = list(self.registry.get_available_models(task_type).keys())
                raise ValueError(f"Model '{model_name}' not available for task '{task_type}'. "
                               f"Available models: {available_models}")
            else:
                model_info = all_models[model_name]
        
        factory_func = model_info['factory_func']
        
        try:
            # Prepare parameters
            params = {}
            
            # Add performance preset
            if 'performance_preset' in inspect.signature(factory_func).parameters:
                params['performance_preset'] = performance_preset
            
            # Add custom parameters
            if custom_params:
                params.update(custom_params)
            
            # Add any additional kwargs
            params.update(kwargs)
            
            # Create base model
            model = factory_func(**params)
            
            # Auto-tune if requested
            if auto_tune and tune_data is not None:
                X_tune, y_tune = tune_data
                model = self._auto_tune_model(model, model_name, X_tune, y_tune, tune_method)
            
            # Store created model info
            model_id = f"{model_name}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.created_models[model_id] = {
                'model': model,
                'model_name': model_name,
                'task_type': task_type,
                'performance_preset': performance_preset,
                'custom_params': custom_params,
                'auto_tuned': auto_tune,
                'creation_time': datetime.now(),
                'model_info': model_info
            }
            
            logger.info(f"Successfully created {model_name} model with ID: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            raise ModelValidationError(f"Model creation failed: {e}")
    
    def _auto_tune_model(self, model: BaseEstimator, model_name: str,
                        X: np.ndarray, y: np.ndarray, tune_method: str) -> BaseEstimator:
        """Auto-tune model hyperparameters"""
        
        logger.info(f"Auto-tuning {model_name} using {tune_method}")
        
        try:
            # Create model factory function for tuning
            def model_factory(**params):
                # Get base model info
                model_info = self.registry.get_model_info(model_name)
                if not model_info:
                    raise ValueError(f"Model {model_name} not found in registry")
                
                base_factory = model_info['factory_func']
                return base_factory(**params)
            
            # Get parameter space based on tuning method - using only available models
            available_param_space_models = ['gradient_boosting', 'random_forest', 'svm', 'logistic']
            
            if model_name not in available_param_space_models:
                logger.warning(f"No predefined parameter space for {model_name}. Using default tuning.")
                return model
            
            # Get parameter space based on tuning method
            if tune_method in ['bayesian', 'optuna']:
                if tune_method == 'bayesian':
                    param_space = create_financial_parameter_space(model_name)
                    optimizer = BayesianOptimizer(
                        model_factory=model_factory,
                        parameter_space=param_space,
                        n_calls=50,
                        verbose=False
                    )
                else:  # optuna
                    param_space = create_financial_optuna_spaces(model_name)
                    optimizer = OptunaOptimizer(
                        model_factory=model_factory,
                        parameter_space=param_space,
                        n_trials=50,
                        verbose=False
                    )
            
            elif tune_method == 'grid_search':
                param_space = create_financial_grid_spaces(model_name, resolution='coarse')
                optimizer = GridSearchOptimizer(
                    model_factory=model_factory,
                    parameter_grid=param_space,
                    early_stopping=True,
                    verbose=0
                )
            
            elif tune_method == 'random_search':
                param_space = create_financial_random_spaces(model_name)
                optimizer = RandomSearchOptimizer(
                    model_factory=model_factory,
                    parameter_space=param_space,
                    n_iter=50,
                    verbose=False
                )
            
            else:
                raise ValueError(f"Unknown tuning method: {tune_method}")
            
            # Run optimization
            results = optimizer.optimize(X, y)
            tuned_model = results['best_model']
            
            logger.info(f"Auto-tuning completed. Best score: {results['best_score']:.4f}")
            return tuned_model
            
        except Exception as e:
            logger.warning(f"Auto-tuning failed: {e}. Returning original model.")
            return model
    
    def create_model_pipeline(self,
                             model_name: str,
                             task_type: str = 'classification',
                             preprocessing_steps: Optional[List[Tuple[str, Any]]] = None,
                             **model_params) -> Pipeline:
        """Create a complete preprocessing + model pipeline"""
        
        logger.info(f"Creating model pipeline for {model_name}")
        
        # Default preprocessing steps
        if preprocessing_steps is None:
            preprocessing_steps = [
                ('scaler', StandardScaler()),
            ]
            
            # Add label encoder for classification
            if task_type == 'classification':
                preprocessing_steps.insert(0, ('label_encoder', LabelEncoder()))
        
        # Create model
        model = self.create_model(model_name, task_type, **model_params)
        
        # Create pipeline
        pipeline_steps = preprocessing_steps + [('model', model)]
        pipeline = Pipeline(pipeline_steps)
        
        return pipeline
    
    def create_ensemble_with_base_models(self,
                                       ensemble_type: str,
                                       base_model_names: List[str],
                                       task_type: str = 'classification',
                                       **ensemble_params) -> BaseEstimator:
        """Create ensemble with specified base models"""
        
        logger.info(f"Creating {ensemble_type} ensemble with base models: {base_model_names}")
        
        # Create base models
        base_models = []
        for model_name in base_model_names:
            try:
                model = self.create_model(model_name, task_type)
                base_models.append((model_name, model))
            except Exception as e:
                logger.warning(f"Failed to create base model {model_name}: {e}")
        
        if not base_models:
            raise ValueError("No valid base models created for ensemble")
        
        # Create ensemble
        if ensemble_type == 'voting':
            if task_type == 'classification':
                ensemble = create_voting_classifier(base_models=base_models, **ensemble_params)
            else:
                ensemble = create_voting_regressor(base_models=base_models, **ensemble_params)
        
        elif ensemble_type == 'stacking':
            # For stacking, we need to pass the base models differently
            base_model_instances = [model for name, model in base_models]
            if task_type == 'classification':
                ensemble = create_stacking_classifier(base_models=base_model_instances, **ensemble_params)
            else:
                ensemble = create_stacking_regressor(base_models=base_model_instances, **ensemble_params)
        
        elif ensemble_type == 'bagging':
            # For bagging, typically use one base model type
            if base_models:
                base_estimator = base_models[0][1]  # Use first model as base
                if task_type == 'classification':
                    ensemble = create_bagging_classifier(base_estimator=base_estimator, **ensemble_params)
                else:
                    ensemble = create_bagging_regressor(base_estimator=base_estimator, **ensemble_params)
            else:
                raise ValueError("Need at least one base model for bagging")
        
        elif ensemble_type == 'blending':
            base_model_instances = [model for name, model in base_models]
            if task_type == 'classification':
                ensemble = create_blending_classifier(base_models=base_model_instances, **ensemble_params)
            else:
                ensemble = create_blending_regressor(base_models=base_model_instances, **ensemble_params)
        
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        return ensemble
    
    def get_model_recommendations(self,
                                 data_characteristics: Dict[str, Any],
                                 task_type: str = 'classification',
                                 priority: str = 'accuracy') -> List[Dict[str, Any]]:
        """Get model recommendations based on data characteristics"""
        
        logger.info(f"Getting model recommendations for {task_type} task with priority: {priority}")
        
        # Extract data characteristics
        n_samples = data_characteristics.get('n_samples', 1000)
        n_features = data_characteristics.get('n_features', 10)
        has_missing = data_characteristics.get('has_missing', False)
        has_categorical = data_characteristics.get('has_categorical', False)
        is_high_dim = n_features > 100
        is_large_dataset = n_samples > 10000
        
        # Get available models
        available_models = self.registry.get_available_models(task_type)
        
        recommendations = []
        
        for model_name, model_info in available_models.items():
            score = 0
            reasons = []
            
            # Data compatibility checks
            if has_missing and not model_info.get('handles_missing', False):
                score -= 2
                reasons.append("Doesn't handle missing values well")
            else:
                score += 1
                
            if has_categorical and not model_info.get('handles_categorical', False):
                score -= 1
                reasons.append("May need categorical encoding")
            else:
                score += 1
            
            # Performance priority adjustments
            if priority == 'accuracy':
                if model_info.get('complexity') in ['high', 'very_high']:
                    score += 2
                    reasons.append("High complexity for better accuracy")
            elif priority == 'speed':
                if model_info.get('training_time') in ['very_fast', 'fast']:
                    score += 2
                    reasons.append("Fast training time")
                if model_info.get('complexity') == 'low':
                    score += 1
            elif priority == 'interpretability':
                if model_info.get('interpretability') in ['high', 'medium']:
                    score += 2
                    reasons.append("Good interpretability")
            
            # Dataset size considerations
            if is_large_dataset:
                if model_info.get('training_time') in ['very_fast', 'fast']:
                    score += 1
                    reasons.append("Efficient for large datasets")
                elif model_info.get('training_time') == 'slow':
                    score -= 1
                    reasons.append("May be slow on large datasets")
            
            if is_high_dim:
                if model_info.get('category') in ['linear', 'neural']:
                    score += 1
                    reasons.append("Handles high-dimensional data well")
            
            # Specific model bonuses
            if model_name in ['random_forest', 'gradient_boosting']:
                score += 1
                reasons.append("Generally robust performer")
            
            recommendation = {
                'model_name': model_name,
                'score': score,
                'model_info': model_info,
                'reasons': reasons,
                'recommended_preset': self._get_recommended_preset(model_info, priority)
            }
            
            recommendations.append(recommendation)
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Generated {len(recommendations)} model recommendations")
        return recommendations
    
    def _get_recommended_preset(self, model_info: Dict, priority: str) -> str:
        """Get recommended preset based on model info and priority"""
        if priority == 'speed':
            return 'fast'
        elif priority == 'accuracy':
            if model_info.get('complexity') in ['high', 'very_high']:
                return 'accurate'
            else:
                return 'balanced'
        else:
            return 'balanced'
    
    def create_financial_model_suite(self,
                                   task_type: str = 'classification',
                                   include_ensembles: bool = True) -> Dict[str, BaseEstimator]:
        """Create a complete suite of models optimized for financial data"""
        
        logger.info(f"Creating financial model suite for {task_type}")
        
        financial_models = {}
        
        # Core individual models
        if task_type == 'classification':
            core_models = ['gradient_boosting', 'random_forest', 'svm', 'logistic']
        else:
            core_models = ['gradient_boosting', 'random_forest', 'svr', 'linear', 'ridge']
        
        for model_name in core_models:
            try:
                model = self.create_model(
                    model_name=model_name,
                    task_type=task_type,
                    performance_preset='balanced'
                )
                financial_models[f'{model_name}_financial'] = model
            except Exception as e:
                logger.warning(f"Failed to create {model_name} for financial suite: {e}")
        
        # Add ensemble models if requested
        if include_ensembles and len(financial_models) >= 2:
            try:
                # Voting ensemble
                base_model_names = list(core_models[:3])  # Use first 3 models
                voting_ensemble = self.create_ensemble_with_base_models(
                    ensemble_type='voting',
                    base_model_names=base_model_names,
                    task_type=task_type,
                    voting='soft' if task_type == 'classification' else None
                )
                financial_models['voting_ensemble_financial'] = voting_ensemble
                
                # Stacking ensemble
                stacking_ensemble = self.create_ensemble_with_base_models(
                    ensemble_type='stacking',
                    base_model_names=base_model_names,
                    task_type=task_type
                )
                financial_models['stacking_ensemble_financial'] = stacking_ensemble
                
            except Exception as e:
                logger.warning(f"Failed to create ensemble models: {e}")
        
        logger.info(f"Created financial model suite with {len(financial_models)} models")
        return financial_models
    
    def get_creation_history(self) -> Dict[str, Dict]:
        """Get history of all created models"""
        return self.created_models.copy()
    
    def clear_cache(self):
        """Clear model cache"""
        self.model_cache.clear()
        logger.info("Cleared model cache")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of factory state and capabilities"""
        available_models = self.registry.get_available_models()
        
        summary = {
            'total_models_available': len(available_models),
            'classification_models': len(self.registry.get_available_models('classification')),
            'regression_models': len(self.registry.get_available_models('regression')) if REGRESSION_AVAILABLE else 0,
            'ensemble_models': len(self.registry.get_available_models('ensemble')),
            'models_created': len(self.created_models),
            'available_categories': list(set(info.get('category') for info in available_models.values())),
            'tuning_methods': ['random_search', 'grid_search', 'bayesian', 'optuna'],
            'regression_available': REGRESSION_AVAILABLE
        }
        
        return summary

# Global model factory instance
model_factory = ModelFactory()

# ============================================
# Convenience Functions
# ============================================

def create_model(model_name: str, task_type: str = 'classification', **kwargs) -> BaseEstimator:
    """Convenience function to create a model"""
    return model_factory.create_model(model_name, task_type, **kwargs)

def get_available_models(task_type: str = None) -> Dict[str, Dict]:
    """Get all available models"""
    return model_registry.get_available_models(task_type)

def get_model_recommendations(data_characteristics: Dict[str, Any],
                            task_type: str = 'classification',
                            priority: str = 'accuracy') -> List[Dict[str, Any]]:
    """Get model recommendations based on data characteristics"""
    return model_factory.get_model_recommendations(data_characteristics, task_type, priority)

def create_financial_suite(task_type: str = 'classification') -> Dict[str, BaseEstimator]:
    """Create a complete suite of financial models"""
    return model_factory.create_financial_model_suite(task_type)

# ============================================
# Configuration Management
# ============================================

class ModelConfiguration:
    """Manage model configurations and presets"""
    
    def __init__(self):
        self.configurations = {}
        self._load_default_configurations()
    
    def _load_default_configurations(self):
        """Load default model configurations"""
        
        # Performance presets for different use cases
        self.configurations = {
            'financial_trading': {
                'models': ['gradient_boosting', 'random_forest', 'svm'],
                'performance_preset': 'accurate',
                'ensemble_type': 'stacking',
                'auto_tune': True,
                'tune_method': 'bayesian'
            },
            'financial_risk': {
                'models': ['logistic', 'random_forest', 'gradient_boosting'],
                'performance_preset': 'balanced',
                'ensemble_type': 'voting',
                'auto_tune': True,
                'tune_method': 'optuna'
            },
            'high_frequency': {
                'models': ['logistic', 'naive_bayes', 'knn'],
                'performance_preset': 'fast',
                'ensemble_type': None,
                'auto_tune': False
            },
            'research_analysis': {
                'models': ['gradient_boosting', 'random_forest', 'svm', 'neural_network'],
                'performance_preset': 'accurate',
                'ensemble_type': 'stacking',
                'auto_tune': True,
                'tune_method': 'optuna'
            }
        }
    
    def get_configuration(self, config_name: str) -> Dict[str, Any]:
        """Get a specific configuration"""
        return self.configurations.get(config_name, {})
    
    def create_models_from_config(self, config_name: str, 
                                task_type: str = 'classification') -> Dict[str, BaseEstimator]:
        """Create models based on a configuration"""
        
        config = self.get_configuration(config_name)
        if not config:
            raise ValueError(f"Configuration '{config_name}' not found")
        
        models = {}
        model_names = config.get('models', [])
        performance_preset = config.get('performance_preset', 'balanced')
        
        # Create individual models
        for model_name in model_names:
            try:
                model = model_factory.create_model(
                    model_name=model_name,
                    task_type=task_type,
                    performance_preset=performance_preset
                )
                models[f'{model_name}_{config_name}'] = model
            except Exception as e:
                logger.warning(f"Failed to create {model_name} for config {config_name}: {e}")
        
        # Create ensemble if specified
        ensemble_type = config.get('ensemble_type')
        if ensemble_type and len(models) >= 2:
            try:
                ensemble = model_factory.create_ensemble_with_base_models(
                    ensemble_type=ensemble_type,
                    base_model_names=model_names[:3],  # Use first 3 models
                    task_type=task_type
                )
                models[f'{ensemble_type}_ensemble_{config_name}'] = ensemble
            except Exception as e:
                logger.warning(f"Failed to create ensemble for config {config_name}: {e}")
        
        return models

# Global configuration manager
model_configuration = ModelConfiguration()

def create_models_from_config(config_name: str, task_type: str = 'classification') -> Dict[str, BaseEstimator]:
    """Create models from a predefined configuration"""
    return model_configuration.create_models_from_config(config_name, task_type)

def get_available_configurations() -> List[str]:
    """Get list of available configurations"""
    return list(model_configuration.configurations.keys())
