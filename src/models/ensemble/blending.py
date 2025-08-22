# ============================================
# StockPredictionPro - src/models/ensemble/blending.py
# Advanced model blending techniques for financial prediction with holdout-based meta-learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
from collections import defaultdict
import warnings

# Core ML imports
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, roc_curve, log_loss
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV

# Import our model factory functions
from ..classification.gradient_boosting import create_gradient_boosting_classifier
from ..classification.random_forest import create_random_forest_classifier
from ..classification.svm import create_svm_classifier
from ..classification.logistic import create_logistic_classifier
from ..classification.neural_network import create_neural_network_classifier

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it

logger = get_logger('models.ensemble.blending')

# ============================================
# Advanced Blending Meta-Learners
# ============================================

class OptimalWeightBlender(BaseEstimator):
    """Find optimal linear combination weights using various optimization methods"""
    
    def __init__(self, method: str = 'ridge', regularization: float = 1.0, 
                 non_negative: bool = True, normalize_weights: bool = True):
        self.method = method
        self.regularization = regularization
        self.non_negative = non_negative
        self.normalize_weights = normalize_weights
        self.weights_ = None
        self.intercept_ = None
        self.meta_model_ = None
        
    def fit(self, blend_features: np.ndarray, targets: np.ndarray):
        """Learn optimal blending weights"""
        
        if self.method == 'ridge':
            self.meta_model_ = Ridge(alpha=self.regularization, positive=self.non_negative)
        elif self.method == 'lasso':
            self.meta_model_ = Lasso(alpha=self.regularization, positive=self.non_negative)
        elif self.method == 'elastic_net':
            self.meta_model_ = ElasticNet(alpha=self.regularization, positive=self.non_negative)
        elif self.method == 'ols':
            self.meta_model_ = LinearRegression(positive=self.non_negative)
        elif self.method == 'constrained_ols':
            # Custom implementation for sum-to-one constraint
            self._fit_constrained_ols(blend_features, targets)
            return self
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.meta_model_.fit(blend_features, targets)
        self.weights_ = self.meta_model_.coef_
        self.intercept_ = self.meta_model_.intercept_
        
        # Normalize weights if requested
        if self.normalize_weights and np.sum(np.abs(self.weights_)) > 0:
            self.weights_ = self.weights_ / np.sum(np.abs(self.weights_))
            
        return self
    
    def _fit_constrained_ols(self, X: np.ndarray, y: np.ndarray):
        """Fit OLS with sum-to-one constraint using scipy optimization"""
        try:
            from scipy.optimize import minimize
            
            def objective(weights):
                predictions = X @ weights
                return np.mean((y - predictions) ** 2)
            
            def constraint_sum_to_one(weights):
                return np.sum(weights) - 1.0
            
            # Initial guess - equal weights
            n_models = X.shape[1]
            x0 = np.ones(n_models) / n_models
            
            # Constraints
            constraints = [{'type': 'eq', 'fun': constraint_sum_to_one}]
            
            # Bounds (non-negative if requested)
            if self.non_negative:
                bounds = [(0, None) for _ in range(n_models)]
            else:
                bounds = None
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', 
                            constraints=constraints, bounds=bounds)
            
            if result.success:
                self.weights_ = result.x
                self.intercept_ = 0.0  # No intercept with sum-to-one constraint
            else:
                logger.warning("Constrained optimization failed, using equal weights")
                self.weights_ = np.ones(n_models) / n_models
                self.intercept_ = 0.0
                
        except ImportError:
            logger.warning("scipy not available, using equal weights")
            n_models = X.shape[1]
            self.weights_ = np.ones(n_models) / n_models
            self.intercept_ = 0.0
    
    def predict(self, blend_features: np.ndarray) -> np.ndarray:
        """Make predictions using learned weights"""
        if self.weights_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = blend_features @ self.weights_
        if self.intercept_ is not None:
            predictions += self.intercept_
            
        return predictions

class AdaptiveBlender(BaseEstimator):
    """Adaptive blending with time-varying and context-aware weights"""
    
    def __init__(self, window_size: int = 100, adaptation_rate: float = 0.1,
                 context_features: Optional[List[str]] = None):
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.context_features = context_features or []
        self.weights_history_ = []
        self.performance_history_ = []
        self.current_weights_ = None
        self.base_blender_ = None
        
    def fit(self, blend_features: np.ndarray, targets: np.ndarray, 
            context_data: Optional[np.ndarray] = None):
        """Initialize adaptive blending"""
        
        # Start with optimal weights as baseline
        self.base_blender_ = OptimalWeightBlender(method='ridge')
        self.base_blender_.fit(blend_features, targets)
        self.current_weights_ = self.base_blender_.weights_.copy()
        
        # Initialize history
        self.weights_history_ = [self.current_weights_.copy()]
        
        return self
    
    def predict(self, blend_features: np.ndarray, 
                context_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Make adaptive predictions"""
        if self.current_weights_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return blend_features @ self.current_weights_
    
    def update_weights(self, blend_features: np.ndarray, targets: np.ndarray, 
                      predictions: np.ndarray):
        """Update weights based on recent performance"""
        
        # Calculate individual model errors
        individual_errors = []
        for i in range(blend_features.shape[1]):
            model_predictions = blend_features[:, i]
            error = np.mean((targets - model_predictions) ** 2)
            individual_errors.append(error)
        
        individual_errors = np.array(individual_errors)
        
        # Calculate inverse error weights (lower error = higher weight)
        inverse_errors = 1.0 / (individual_errors + 1e-8)
        new_weights = inverse_errors / np.sum(inverse_errors)
        
        # Adapt current weights using exponential moving average
        self.current_weights_ = (1 - self.adaptation_rate) * self.current_weights_ + \
                               self.adaptation_rate * new_weights
        
        # Store in history
        self.weights_history_.append(self.current_weights_.copy())
        
        # Maintain window size
        if len(self.weights_history_) > self.window_size:
            self.weights_history_ = self.weights_history_[-self.window_size:]

class NonLinearBlender(BaseEstimator):
    """Non-linear blending using neural networks or other non-linear methods"""
    
    def __init__(self, method: str = 'neural_network', hidden_layers: List[int] = [32, 16],
                 regularization: float = 0.001):
        self.method = method
        self.hidden_layers = hidden_layers
        self.regularization = regularization
        self.meta_model_ = None
        
    def fit(self, blend_features: np.ndarray, targets: np.ndarray):
        """Train non-linear meta-model"""
        
        if self.method == 'neural_network':
            try:
                from sklearn.neural_network import MLPRegressor
                self.meta_model_ = MLPRegressor(
                    hidden_layer_sizes=tuple(self.hidden_layers),
                    alpha=self.regularization,
                    max_iter=1000,
                    random_state=42
                )
            except ImportError:
                logger.warning("Neural network not available, using Random Forest")
                self.method = 'random_forest'
        
        if self.method == 'random_forest':
            self.meta_model_ = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        
        self.meta_model_.fit(blend_features, targets)
        return self
    
    def predict(self, blend_features: np.ndarray) -> np.ndarray:
        """Make non-linear predictions"""
        if self.meta_model_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.meta_model_.predict(blend_features)

class HierarchicalBlender(BaseEstimator):
    """Hierarchical blending with multiple blending levels"""
    
    def __init__(self, level_configs: List[Dict[str, Any]], 
                 final_blender: str = 'ridge'):
        self.level_configs = level_configs
        self.final_blender = final_blender
        self.level_blenders_ = []
        self.final_blender_ = None
        
    def fit(self, blend_features: np.ndarray, targets: np.ndarray):
        """Train hierarchical blending structure"""
        
        current_features = blend_features.copy()
        
        # Train each level
        for level_config in self.level_configs:
            method = level_config.get('method', 'ridge')
            groups = level_config.get('groups', [[i] for i in range(current_features.shape[1])])
            
            level_outputs = []
            level_blenders = []
            
            for group in groups:
                group_features = current_features[:, group]
                
                # Create blender for this group
                if method == 'ridge':
                    blender = OptimalWeightBlender(method='ridge')
                elif method == 'neural_network':
                    blender = NonLinearBlender(method='neural_network')
                else:
                    blender = OptimalWeightBlender(method=method)
                
                blender.fit(group_features, targets)
                group_predictions = blender.predict(group_features)
                
                level_outputs.append(group_predictions)
                level_blenders.append(blender)
            
            self.level_blenders_.append(level_blenders)
            current_features = np.column_stack(level_outputs)
        
        # Train final blender
        self.final_blender_ = OptimalWeightBlender(method=self.final_blender)
        self.final_blender_.fit(current_features, targets)
        
        return self
    
    def predict(self, blend_features: np.ndarray) -> np.ndarray:
        """Make hierarchical predictions"""
        current_features = blend_features.copy()
        
        # Process through each level
        for level_idx, (level_blenders, level_config) in enumerate(
            zip(self.level_blenders_, self.level_configs)):
            
            groups = level_config.get('groups', [[i] for i in range(current_features.shape[1])])
            level_outputs = []
            
            for group_idx, (blender, group) in enumerate(zip(level_blenders, groups)):
                group_features = current_features[:, group]
                group_predictions = blender.predict(group_features)
                level_outputs.append(group_predictions)
            
            current_features = np.column_stack(level_outputs)
        
        # Final prediction
        return self.final_blender_.predict(current_features)

# ============================================
# Main Blending Classifier/Regressor
# ============================================

class FinancialBlendingModel(BaseEstimator):
    """
    Advanced model blending framework for financial prediction
    
    Features:
    - Multiple blending strategies: linear, non-linear, hierarchical, adaptive
    - Holdout-based meta-feature generation (no cross-validation data leakage)
    - Time-aware blending for financial time series
    - Comprehensive blending analysis and diagnostics
    - Financial domain optimizations (volatility weighting, regime-aware blending)
    """
    
    def __init__(self,
                 name: str = "blending_model",
                 base_models: Optional[List[Any]] = None,
                 blending_method: str = 'ridge',
                 holdout_size: float = 0.2,
                 blend_holdout_size: float = 0.2,
                 time_aware: bool = True,
                 adaptive_blending: bool = False,
                 hierarchical_config: Optional[List[Dict]] = None,
                 regularization: float = 1.0,
                 calibrate_probabilities: bool = True,
                 random_state: int = 42,
                 task_type: str = 'classification',
                 **kwargs):
        """
        Initialize Financial Blending Model
        
        Args:
            name: Model name
            base_models: List of base models to blend
            blending_method: Blending method ('ridge', 'lasso', 'neural_network', 'adaptive', 'hierarchical')
            holdout_size: Size of holdout set for meta-feature generation
            blend_holdout_size: Additional holdout for final evaluation
            time_aware: Whether to use time-aware splitting
            adaptive_blending: Whether to use adaptive weight updates
            hierarchical_config: Configuration for hierarchical blending
            regularization: Regularization strength
            calibrate_probabilities: Whether to calibrate probabilities (classification)
            random_state: Random seed
            task_type: Type of task ('classification' or 'regression')
        """
        self.name = name
        self.base_models = base_models or self._create_default_models()
        self.blending_method = blending_method
        self.holdout_size = holdout_size
        self.blend_holdout_size = blend_holdout_size
        self.time_aware = time_aware
        self.adaptive_blending = adaptive_blending
        self.hierarchical_config = hierarchical_config
        self.regularization = regularization
        self.calibrate_probabilities = calibrate_probabilities
        self.random_state = random_state
        self.task_type = task_type
        
        # Fitted components
        self.base_models_ = []
        self.blender_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.calibrated_blender_ = None
        self.blending_weights_ = None
        self.meta_features_ = None
        self.blending_analysis_ = None
        self.is_fitted_ = False
        
        # Performance tracking
        self.holdout_scores_ = {}
        self.base_model_scores_ = {}
        self.blend_improvement_ = None
        
        logger.info(f"Initialized {blending_method} blending model: {self.name}")
    
    def _create_default_models(self) -> List[Any]:
        """Create default set of diverse base models"""
        try:
            models = [
                create_gradient_boosting_classifier(performance_preset='balanced'),
                create_random_forest_classifier(performance_preset='balanced'),
                create_svm_classifier(performance_preset='balanced'),
                create_logistic_classifier(performance_preset='balanced')
            ]
            
            # Add neural network if possible
            try:
                nn_model = create_neural_network_classifier(
                    architecture='balanced',
                    epochs=100,
                    verbose=0
                )
                models.append(nn_model)
            except:
                logger.info("Neural network not available, using other models")
            
            return models
            
        except Exception as e:
            logger.error(f"Error creating default models: {e}")
            return []
    
    def _preprocess_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Preprocess features with scaling"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.scaler_ is None:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = self.scaler_.transform(X)
        
        return X_scaled
    
    def _preprocess_targets(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Preprocess targets with encoding for classification"""
        if isinstance(y, pd.Series):
            y = y.values
        
        if self.task_type == 'classification':
            if self.label_encoder_ is None:
                self.label_encoder_ = LabelEncoder()
                y_encoded = self.label_encoder_.fit_transform(y)
                self.classes_ = self.label_encoder_.classes_
            else:
                y_encoded = self.label_encoder_.transform(y)
            return y_encoded
        else:
            return y.astype(float)
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data for blending with time-awareness if requested"""
        
        if self.time_aware:
            # Time-aware split - use later data for holdout
            split_point = int(len(X) * (1 - self.holdout_size))
            X_train, X_holdout = X[:split_point], X[split_point:]
            y_train, y_holdout = y[:split_point], y[split_point:]
        else:
            # Random split
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X, y, test_size=self.holdout_size, 
                random_state=self.random_state,
                stratify=y if self.task_type == 'classification' else None
            )
        
        return X_train, X_holdout, y_train, y_holdout
    
    def _generate_meta_features(self, X_train: np.ndarray, X_holdout: np.ndarray,
                           y_train: np.ndarray, y_holdout: np.ndarray) -> np.ndarray:

        """Generate meta-features using holdout predictions"""
        
        meta_features = []
        self.base_model_scores_ = {}
        
        for i, base_model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {type(base_model).__name__}")
            
            try:
                # Clone and fit model
                model = clone(base_model)
                model.fit(X_train, y_train)
                self.base_models_.append(model)
                
                # Generate predictions on holdout set
                if self.task_type == 'classification':
                    if hasattr(model, 'predict_proba'):
                        # Use probabilities as meta-features (more informative)
                        holdout_pred = model.predict_proba(X_holdout)
                        if holdout_pred.shape[1] == 2:
                            # For binary classification, use positive class probability
                            meta_features.append(holdout_pred[:, 1])
                        else:
                            # For multiclass, use all probabilities
                            for j in range(holdout_pred.shape[1]):
                                meta_features.append(holdout_pred[:, j])
                    else:
                        # Use predictions as meta-features
                        holdout_pred = model.predict(X_holdout)
                        meta_features.append(holdout_pred.astype(float))
                else:
                    # Regression
                    holdout_pred = model.predict(X_holdout)
                    meta_features.append(holdout_pred)
                
                # Evaluate base model performance
                train_pred = model.predict(X_train)
                if self.task_type == 'classification':
                    train_score = accuracy_score(y_train, train_pred)
                    holdout_score = accuracy_score(y_holdout, model.predict(X_holdout))
                else:
                    train_score = r2_score(y_train, train_pred)
                    holdout_score = r2_score(y_holdout, model.predict(X_holdout))
                
                self.base_model_scores_[f'model_{i}'] = {
                    'train_score': train_score,
                    'holdout_score': holdout_score,
                    'model_type': type(base_model).__name__
                }
                
                logger.debug(f"Model {i} - Train: {train_score:.4f}, Holdout: {holdout_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to train base model {i}: {e}")
        
        if not meta_features:
            raise ValueError("No base models were successfully trained")
        
        # Stack meta-features
        meta_features = np.column_stack(meta_features)
        logger.info(f"Generated meta-features shape: {meta_features.shape}")
        
        return meta_features
    
    def _create_blender(self) -> BaseEstimator:
        """Create the meta-learner blender"""
        
        if self.blending_method == 'ridge':
            return OptimalWeightBlender(method='ridge', regularization=self.regularization)
        elif self.blending_method == 'lasso':
            return OptimalWeightBlender(method='lasso', regularization=self.regularization)
        elif self.blending_method == 'elastic_net':
            return OptimalWeightBlender(method='elastic_net', regularization=self.regularization)
        elif self.blending_method == 'ols':
            return OptimalWeightBlender(method='ols')
        elif self.blending_method == 'constrained_ols':
            return OptimalWeightBlender(method='constrained_ols')
        elif self.blending_method == 'neural_network':
            return NonLinearBlender(method='neural_network')
        elif self.blending_method == 'adaptive':
            return AdaptiveBlender()
        elif self.blending_method == 'hierarchical':
            config = self.hierarchical_config or [
                {'method': 'ridge', 'groups': [[0, 1], [2, 3]]},
                {'method': 'ridge', 'groups': [[0], [1]]}
            ]
            return HierarchicalBlender(level_configs=config)
        else:
            raise ValueError(f"Unknown blending method: {self.blending_method}")
    
    def _analyze_blending(self, meta_features: np.ndarray, y_holdout: np.ndarray):
        """Analyze blending weights and performance"""
        
        analysis = {
            'blending_method': self.blending_method,
            'n_base_models': len(self.base_models_),
            'meta_features_shape': meta_features.shape,
            'base_model_performance': self.base_model_scores_.copy()
        }
        
        # Analyze blending weights
        if hasattr(self.blender_, 'weights_') and self.blender_.weights_ is not None:
            weights = self.blender_.weights_
            analysis['blending_weights'] = {
                'weights': weights.tolist(),
                'weight_distribution': {
                    'mean': float(np.mean(weights)),
                    'std': float(np.std(weights)),
                    'min': float(np.min(weights)),
                    'max': float(np.max(weights)),
                    'entropy': float(-np.sum(weights * np.log(weights + 1e-8)))
                }
            }
            
            # Weight interpretation
            weight_interpretation = []
            for i, weight in enumerate(weights):
                model_name = self.base_model_scores_[f'model_{i}']['model_type']
                importance = 'High' if weight > np.mean(weights) + np.std(weights) else \
                           'Low' if weight < np.mean(weights) - np.std(weights) else 'Medium'
                weight_interpretation.append({
                    'model': model_name,
                    'weight': float(weight),
                    'importance': importance
                })
            
            analysis['weight_interpretation'] = weight_interpretation
        
        # Performance analysis
        blend_pred = self.blender_.predict(meta_features)
        
        if self.task_type == 'classification':
            # Convert predictions for classification
            if hasattr(self, 'classes_') and len(self.classes_) == 2:
                # Binary classification - threshold at 0.5
                blend_pred_class = (blend_pred > 0.5).astype(int)
            else:
                # Multiclass - round to nearest integer
                blend_pred_class = np.round(blend_pred).astype(int)
                blend_pred_class = np.clip(blend_pred_class, 0, len(self.classes_) - 1)
            
            blend_score = accuracy_score(y_holdout, blend_pred_class)
            
            # Compare with base models
            base_scores = [scores['holdout_score'] for scores in self.base_model_scores_.values()]
            best_base_score = max(base_scores)
            avg_base_score = np.mean(base_scores)
            
            analysis['performance_comparison'] = {
                'blend_score': float(blend_score),
                'best_base_score': float(best_base_score),
                'avg_base_score': float(avg_base_score),
                'improvement_vs_best': float(blend_score - best_base_score),
                'improvement_vs_avg': float(blend_score - avg_base_score)
            }
            
        else:
            # Regression
            blend_score = r2_score(y_holdout, blend_pred)
            blend_mse = mean_squared_error(y_holdout, blend_pred)
            
            base_scores = [scores['holdout_score'] for scores in self.base_model_scores_.values()]
            best_base_score = max(base_scores)
            avg_base_score = np.mean(base_scores)
            
            analysis['performance_comparison'] = {
                'blend_r2': float(blend_score),
                'blend_mse': float(blend_mse),
                'best_base_r2': float(best_base_score),
                'avg_base_r2': float(avg_base_score),
                'improvement_vs_best': float(blend_score - best_base_score),
                'improvement_vs_avg': float(blend_score - avg_base_score)
            }
        
        self.blending_analysis_ = analysis
        
        # Store blend improvement
        if self.task_type == 'classification':
            self.blend_improvement_ = blend_score - best_base_score
        else:
            self.blend_improvement_ = blend_score - best_base_score
    
    def _calibrate_probabilities(self, meta_features: np.ndarray, y_holdout: np.ndarray):
        """Calibrate blended probabilities for classification"""
        if self.task_type != 'classification' or not self.calibrate_probabilities:
            return
        
        try:
            # Create calibrated version of blender
            self.calibrated_blender_ = CalibratedClassifierCV(
                base_estimator=self.blender_,
                method='isotonic',
                cv=3
            )
            self.calibrated_blender_.fit(meta_features, y_holdout)
            logger.debug("Calibrated blending probabilities")
        except Exception as e:
            logger.warning(f"Could not calibrate probabilities: {e}")
            self.calibrated_blender_ = None
    
    @time_it("blending_fit", include_args=True)
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], **kwargs):
        """Fit the blending model"""
        
        logger.info(f"Fitting blending model on {len(X)} samples")
        
        try:
            # Preprocess data
            X_processed = self._preprocess_features(X)
            y_processed = self._preprocess_targets(y)
            
            # Split data for holdout-based blending
            X_train, X_holdout, y_train, y_holdout = self._split_data(X_processed, y_processed)
            
            logger.info(f"Train set: {len(X_train)}, Holdout set: {len(X_holdout)}")
            
            # Generate meta-features using base models
            meta_features = self._generate_meta_features(X_train, X_holdout,y_train, y_holdout)
            self.meta_features_ = meta_features
            
            # Train blender on meta-features
            self.blender_ = self._create_blender()
            
            # For classification, we need to handle the target properly
            if self.task_type == 'classification':
                if len(self.classes_) == 2:
                    # Binary classification - convert to probabilities for blending
                    y_holdout_blend = y_holdout.astype(float)
                else:
                    # Multiclass - use encoded labels
                    y_holdout_blend = y_holdout.astype(float)
            else:
                y_holdout_blend = y_holdout
            
            self.blender_.fit(meta_features, y_holdout_blend)
            
            # Store blending weights if available
            if hasattr(self.blender_, 'weights_'):
                self.blending_weights_ = self.blender_.weights_
            
            # Analyze blending performance
            self._analyze_blending(meta_features, y_holdout)
            
            # Calibrate probabilities for classification
            if self.task_type == 'classification':
                self._calibrate_probabilities(meta_features, y_holdout)
            
            self.is_fitted_ = True
            logger.info(f"Blending model fitted successfully. Improvement: {self.blend_improvement_:.4f}")
            
            return self
            
        except Exception as e:
            logger.error(f"Blending model fitting failed: {e}")
            raise
    
    @time_it("blending_predict", include_args=True)
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make blended predictions"""
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            # Generate meta-features from base models
            meta_features = []
            for model in self.base_models_:
                if self.task_type == 'classification':
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X_processed)
                        if pred_proba.shape[1] == 2:
                            meta_features.append(pred_proba[:, 1])
                        else:
                            for j in range(pred_proba.shape[1]):
                                meta_features.append(pred_proba[:, j])
                    else:
                        pred = model.predict(X_processed)
                        meta_features.append(pred.astype(float))
                else:
                    pred = model.predict(X_processed)
                    meta_features.append(pred)
            
            meta_features = np.column_stack(meta_features)
            
            # Make blended prediction
            blend_pred = self.blender_.predict(meta_features)
            
            if self.task_type == 'classification':
                if len(self.classes_) == 2:
                    # Binary classification
                    pred_class = (blend_pred > 0.5).astype(int)
                else:
                    # Multiclass
                    pred_class = np.round(blend_pred).astype(int)
                    pred_class = np.clip(pred_class, 0, len(self.classes_) - 1)
                
                # Decode predictions
                return self.label_encoder_.inverse_transform(pred_class)
            else:
                return blend_pred
                
        except Exception as e:
            logger.error(f"Blending prediction failed: {e}")
            raise
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make blended probability predictions (classification only)"""
        
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            # Generate meta-features from base models
            meta_features = []
            for model in self.base_models_:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X_processed)
                    if pred_proba.shape[1] == 2:
                        meta_features.append(pred_proba[:, 1])
                    else:
                        for j in range(pred_proba.shape[1]):
                            meta_features.append(pred_proba[:, j])
                else:
                    pred = model.predict(X_processed)
                    meta_features.append(pred.astype(float))
            
            meta_features = np.column_stack(meta_features)
            
            # Get blended probabilities
            if self.calibrated_blender_ is not None:
                probabilities = self.calibrated_blender_.predict_proba(meta_features)
            else:
                # Convert blender predictions to probabilities
                blend_pred = self.blender_.predict(meta_features)
                
                if len(self.classes_) == 2:
                    # Binary classification
                    prob_positive = np.clip(blend_pred, 0, 1)
                    prob_negative = 1 - prob_positive
                    probabilities = np.column_stack([prob_negative, prob_positive])
                else:
                    # Multiclass - use softmax
                    exp_pred = np.exp(blend_pred - np.max(blend_pred))
                    probabilities = exp_pred / np.sum(exp_pred)
                    probabilities = probabilities.reshape(1, -1) if probabilities.ndim == 1 else probabilities
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Blending probability prediction failed: {e}")
            raise
    
    def get_blending_analysis(self) -> Dict[str, Any]:
        """Get comprehensive blending analysis"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to get blending analysis")
        
        return self.blending_analysis_.copy() if self.blending_analysis_ else {}
    
    def get_blending_weights(self) -> Optional[np.ndarray]:
        """Get blending weights"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to get blending weights")
        
        return self.blending_weights_.copy() if self.blending_weights_ is not None else None
    
    def plot_blending_analysis(self) -> Any:
        """Plot blending analysis results"""
        if not self.blending_analysis_:
            logger.warning("Blending analysis not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            analysis = self.blending_analysis_
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Base model performance comparison
            base_perf = analysis['base_model_performance']
            model_names = [info['model_type'] for info in base_perf.values()]
            holdout_scores = [info['holdout_score'] for info in base_perf.values()]
            
            # Add blend performance
            perf_comp = analysis.get('performance_comparison', {})
            if self.task_type == 'classification':
                blend_score = perf_comp.get('blend_score', 0)
                metric_name = 'Accuracy'
            else:
                blend_score = perf_comp.get('blend_r2', 0)
                metric_name = 'R² Score'
            
            model_names.append('Blended Model')
            holdout_scores.append(blend_score)
            colors = ['steelblue'] * (len(model_names) - 1) + ['red']
            
            bars = axes[0, 0].bar(range(len(model_names)), holdout_scores, color=colors, alpha=0.7)
            axes[0, 0].set_title(f'Model Performance Comparison ({metric_name})')
            axes[0, 0].set_xticks(range(len(model_names)))
            axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
            axes[0, 0].set_ylabel(metric_name)
            
            # Add values on bars
            for bar, score in zip(bars, holdout_scores):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, score + 0.005,
                               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Blending weights
            if 'blending_weights' in analysis:
                weights_info = analysis['blending_weights']
                weights = weights_info['weights']
                base_model_names = [info['model_type'] for info in base_perf.values()]
                
                bars = axes[0, 1].bar(range(len(weights)), weights, alpha=0.7, color='orange')
                axes[0, 1].set_title('Blending Weights')
                axes[0, 1].set_xticks(range(len(weights)))
                axes[0, 1].set_xticklabels(base_model_names, rotation=45, ha='right')
                axes[0, 1].set_ylabel('Weight')
                
                # Add values on bars
                for bar, weight in zip(bars, weights):
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2, weight + 0.01,
                                   f'{weight:.3f}', ha='center', va='bottom')
            else:
                axes[0, 1].text(0.5, 0.5, 'Weights not available\nfor this blending method',
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Blending Weights')
            
            # Weight distribution analysis
            if 'blending_weights' in analysis:
                weight_dist = weights_info['weight_distribution']
                dist_metrics = ['Mean', 'Std', 'Min', 'Max', 'Entropy']
                dist_values = [
                    weight_dist['mean'], weight_dist['std'], 
                    weight_dist['min'], weight_dist['max'], 
                    weight_dist['entropy']
                ]
                
                bars = axes[0, 2].bar(dist_metrics, dist_values, alpha=0.7, color='green')
                axes[0, 2].set_title('Weight Distribution Statistics')
                axes[0, 2].set_ylabel('Value')
                axes[0, 2].tick_params(axis='x', rotation=45)
                
                # Add values on bars
                for bar, value in zip(bars, dist_values):
                    axes[0, 2].text(bar.get_x() + bar.get_width()/2, value + max(dist_values) * 0.01,
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            else:
                axes[0, 2].axis('off')
            
            # Performance improvement analysis
            if 'performance_comparison' in analysis:
                perf_comp = analysis['performance_comparison']
                
                if self.task_type == 'classification':
                    categories = ['Blend Score', 'Best Base', 'Average Base']
                    values = [perf_comp['blend_score'], perf_comp['best_base_score'], perf_comp['avg_base_score']]
                    improvement_vs_best = perf_comp['improvement_vs_best']
                    improvement_vs_avg = perf_comp['improvement_vs_avg']
                else:
                    categories = ['Blend R²', 'Best Base R²', 'Average Base R²']
                    values = [perf_comp['blend_r2'], perf_comp['best_base_r2'], perf_comp['avg_base_r2']]
                    improvement_vs_best = perf_comp['improvement_vs_best']
                    improvement_vs_avg = perf_comp['improvement_vs_avg']
                
                colors = ['red', 'blue', 'gray']
                bars = axes[1, 0].bar(categories, values, color=colors, alpha=0.7)
                axes[1, 0].set_title(f'Performance Comparison\nImprovement vs Best: {improvement_vs_best:+.3f}')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Add values on bars
                for bar, value in zip(bars, values):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, value + max(values) * 0.01,
                                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Model importance based on weights
            if 'weight_interpretation' in analysis:
                weight_interp = analysis['weight_interpretation']
                model_types = [item['model'] for item in weight_interp]
                model_weights = [item['weight'] for item in weight_interp]
                importances = [item['importance'] for item in weight_interp]
                
                # Color by importance
                importance_colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
                colors = [importance_colors[imp] for imp in importances]
                
                bars = axes[1, 1].barh(range(len(model_types)), model_weights, color=colors, alpha=0.7)
                axes[1, 1].set_title('Model Importance Analysis')
                axes[1, 1].set_yticks(range(len(model_types)))
                axes[1, 1].set_yticklabels(model_types)
                axes[1, 1].set_xlabel('Blending Weight')
                
                # Add values on bars
                for bar, weight, importance in zip(bars, model_weights, importances):
                    axes[1, 1].text(weight + max(model_weights) * 0.01, bar.get_y() + bar.get_height()/2,
                                   f'{weight:.3f} ({importance})', va='center', fontsize=9)
            else:
                axes[1, 1].axis('off')
            
            # Summary statistics
            summary_text = f"Blending Method: {analysis['blending_method'].title()}\n"
            summary_text += f"Base Models: {analysis['n_base_models']}\n"
            summary_text += f"Meta Features: {analysis['meta_features_shape']}\n\n"
            
            if 'performance_comparison' in analysis:
                perf = analysis['performance_comparison']
                if self.task_type == 'classification':
                    summary_text += f"Blend Accuracy: {perf['blend_score']:.4f}\n"
                    summary_text += f"Best Base Accuracy: {perf['best_base_score']:.4f}\n"
                    summary_text += f"Improvement: {perf['improvement_vs_best']:+.4f}\n"
                else:
                    summary_text += f"Blend R²: {perf['blend_r2']:.4f}\n"
                    summary_text += f"Best Base R²: {perf['best_base_r2']:.4f}\n"
                    summary_text += f"Improvement: {perf['improvement_vs_best']:+.4f}\n"
            
            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 2].set_title('Blending Summary')
            axes[1, 2].axis('off')
            
            plt.suptitle(f'Blending Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = {
            'model_name': self.name,
            'model_family': 'Blending Ensemble',
            'task_type': self.task_type,
            'blending_method': self.blending_method,
            'n_base_models': len(self.base_models_),
            'base_model_types': [type(model).__name__ for model in self.base_models],
            'holdout_size': self.holdout_size,
            'time_aware_splitting': self.time_aware,
            'adaptive_blending': self.adaptive_blending,
            'probability_calibration': self.calibrate_probabilities,
            'is_fitted': self.is_fitted_
        }
        
        if self.is_fitted_:
            summary.update({
                'blend_improvement': self.blend_improvement_,
                'blending_analysis': self.blending_analysis_
            })
            
            if self.blending_weights_ is not None:
                summary['blending_weights'] = self.blending_weights_.tolist()
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_blending_classifier(blending_method: str = 'ridge',
                             base_models: Optional[List[str]] = None,
                             complexity: str = 'balanced',
                             **kwargs) -> FinancialBlendingModel:
    """Create blending classifier with different complexity levels"""
    
    # Default base models by complexity
    complexity_models = {
        'simple': ['gradient_boosting', 'random_forest', 'logistic'],
        'balanced': ['gradient_boosting', 'random_forest', 'svm', 'logistic', 'neural_network'],
        'comprehensive': ['gradient_boosting', 'random_forest', 'svm', 'logistic', 
                         'neural_network', 'naive_bayes', 'knn']
    }
    
    if base_models is None:
        base_models = complexity_models.get(complexity, complexity_models['balanced'])
    
    # Create base model instances
    model_instances = []
    for model_name in base_models:
        try:
            if model_name == 'gradient_boosting':
                model = create_gradient_boosting_classifier(performance_preset='balanced')
            elif model_name == 'random_forest':
                model = create_random_forest_classifier(performance_preset='balanced')
            elif model_name == 'svm':
                model = create_svm_classifier(performance_preset='balanced')
            elif model_name == 'logistic':
                model = create_logistic_classifier(performance_preset='balanced')
            elif model_name == 'neural_network':
                model = create_neural_network_classifier(
                    architecture='balanced', epochs=100, verbose=0
                )
            else:
                continue
            
            model_instances.append(model)
        except Exception as e:
            logger.warning(f"Could not create {model_name}: {e}")
    
    config = {
        'name': f'{blending_method}_blending_{complexity}',
        'base_models': model_instances,
        'blending_method': blending_method,
        'task_type': 'classification',
        'calibrate_probabilities': True,
        'random_state': 42
    }
    
    config.update(kwargs)
    
    return FinancialBlendingModel(**config)

def create_blending_regressor(blending_method: str = 'ridge',
                            base_models: Optional[List[str]] = None,
                            complexity: str = 'balanced',
                            **kwargs) -> FinancialBlendingModel:
    """Create blending regressor"""
    
    classifier = create_blending_classifier(
        blending_method=blending_method,
        base_models=base_models,
        complexity=complexity,
        **kwargs
    )
    
    # Convert to regression
    classifier.task_type = 'regression'
    classifier.calibrate_probabilities = False
    classifier.name = classifier.name.replace('classification', 'regression')
    
    return classifier

def create_ridge_blending(**kwargs) -> FinancialBlendingModel:
    """Create Ridge regression blending"""
    return create_blending_classifier(
        blending_method='ridge',
        name='ridge_blending_classifier',
        **kwargs
    )

def create_neural_blending(**kwargs) -> FinancialBlendingModel:
    """Create neural network blending"""
    return create_blending_classifier(
        blending_method='neural_network',
        name='neural_blending_classifier',
        **kwargs
    )

def create_adaptive_blending(**kwargs) -> FinancialBlendingModel:
    """Create adaptive blending with time-varying weights"""
    return create_blending_classifier(
        blending_method='adaptive',
        adaptive_blending=True,
        time_aware=True,
        name='adaptive_blending_classifier',
        **kwargs
    )

def create_hierarchical_blending(**kwargs) -> FinancialBlendingModel:
    """Create hierarchical blending"""
    return create_blending_classifier(
        blending_method='hierarchical',
        name='hierarchical_blending_classifier',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def compare_blending_methods(X: Union[pd.DataFrame, np.ndarray],
                           y: Union[pd.Series, np.ndarray],
                           methods: List[str] = ['ridge', 'lasso', 'neural_network'],
                           task_type: str = 'classification') -> Dict[str, Any]:
    """Compare different blending methods"""
    
    logger.info(f"Comparing blending methods: {methods}")
    
    results = {}
    
    for method in methods:
        logger.info(f"Evaluating {method} blending")
        
        try:
            if task_type == 'classification':
                blender = create_blending_classifier(blending_method=method)
            else:
                blender = create_blending_regressor(blending_method=method)
            
            # Fit and evaluate
            blender.fit(X, y)
            
            # Get blending analysis
            analysis = blender.get_blending_analysis()
            
            if task_type == 'classification':
                score = analysis['performance_comparison']['blend_score']
                improvement = analysis['performance_comparison']['improvement_vs_best']
            else:
                score = analysis['performance_comparison']['blend_r2']
                improvement = analysis['performance_comparison']['improvement_vs_best']
            
            results[method] = {
                'score': score,
                'improvement': improvement,
                'blending_analysis': analysis,
                'model': blender
            }
            
        except Exception as e:
            logger.warning(f"Error with {method} blending: {e}")
            results[method] = {'error': str(e)}
    
    # Add comparison summary
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_method = max(valid_results.keys(), key=lambda k: valid_results[k]['score'])
        best_improvement = max(valid_results.keys(), key=lambda k: valid_results[k]['improvement'])
        
        results['comparison'] = {
            'best_score': best_method,
            'best_improvement': best_improvement,
            'method_rankings': sorted(valid_results.keys(), 
                                   key=lambda k: valid_results[k]['score'], reverse=True)
        }
    
    logger.info(f"Blending comparison complete. Best method: {results['comparison']['best_score']}")
    
    return results

def optimize_blending_weights(predictions: np.ndarray, targets: np.ndarray,
                            method: str = 'ridge') -> Dict[str, Any]:
    """Find optimal blending weights for given predictions"""
    
    logger.info(f"Optimizing blending weights using {method}")
    
    # Create weight optimizer
    optimizer = OptimalWeightBlender(method=method)
    optimizer.fit(predictions, targets)
    
    weights = optimizer.weights_
    
    # Evaluate blended predictions
    blended_pred = optimizer.predict(predictions)
    
    if len(np.unique(targets)) <= 10:  # Assume classification
        score = accuracy_score(targets, np.round(blended_pred))
        individual_scores = [accuracy_score(targets, pred) for pred in predictions.T]
    else:  # Regression
        score = r2_score(targets, blended_pred)
        individual_scores = [r2_score(targets, pred) for pred in predictions.T]
    
    best_individual = max(individual_scores)
    improvement = score - best_individual
    
    results = {
        'optimal_weights': weights.tolist(),
        'blended_score': float(score),
        'individual_scores': individual_scores,
        'best_individual_score': float(best_individual),
        'improvement': float(improvement),
        'weight_stats': {
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'entropy': float(-np.sum(weights * np.log(weights + 1e-8)))
        }
    }
    
    logger.info(f"Weight optimization complete. Improvement: {improvement:.4f}")
    
    return results
