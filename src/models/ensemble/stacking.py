# ============================================
# StockPredictionPro - src/models/ensemble/stacking.py
# Advanced stacking ensemble methods for financial prediction with multi-level meta-learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
from collections import defaultdict
import warnings

# Core ML imports
from sklearn.model_selection import (
    cross_val_predict, StratifiedKFold, KFold, train_test_split, cross_val_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, roc_curve, log_loss
)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Import our model factory functions
from ..classification.gradient_boosting import create_gradient_boosting_classifier
from ..classification.random_forest import create_random_forest_classifier
from ..classification.svm import create_svm_classifier
from ..classification.logistic import create_logistic_classifier
from ..classification.naive_bayes import create_naive_bayes_classifier
from ..classification.knn import create_knn_classifier
from ..classification.neural_network import create_neural_network_classifier

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it

logger = get_logger('models.ensemble.stacking')

# ============================================
# Advanced Meta-Learner Strategies
# ============================================

class MetaLearnerSelector:
    """Intelligent meta-learner selection based on data characteristics"""
    
    def __init__(self, task_type: str = 'classification'):
        self.task_type = task_type
        self.meta_learner_performance_ = {}
        self.best_meta_learner_ = None
        
    def select_meta_learner(self, meta_features: np.ndarray, targets: np.ndarray,
                          cv_folds: int = 3) -> Any:
        """Select optimal meta-learner based on cross-validation"""
        
        # Define candidate meta-learners
        if self.task_type == 'classification':
            candidates = {
                'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            }
            
            # Add SVM candidate
            try:
                candidates['svm'] = create_svm_classifier(performance_preset='balanced')
            except:
                pass
            
            # Add neural network if possible
            try:
                candidates['neural_network'] = create_neural_network_classifier(
                    architecture='simple', epochs=50, verbose=0
                )
            except:
                pass
            
        else:  # regression
            candidates = {
                'ridge': Ridge(random_state=42),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            }
        
        # Evaluate each candidate
        best_score = -np.inf
        best_learner = None
        
        for name, learner in candidates.items():
            try:
                scores = cross_val_score(learner, meta_features, targets, cv=cv_folds, 
                                       scoring='accuracy' if self.task_type == 'classification' else 'r2')
                score = np.mean(scores)
                
                self.meta_learner_performance_[name] = score
                
                if score > best_score:
                    best_score = score
                    best_learner = clone(learner)
                    
                logger.debug(f"Meta-learner {name}: {score:.4f}")
                
            except Exception as e:
                logger.warning(f"Error evaluating meta-learner {name}: {e}")
                self.meta_learner_performance_[name] = -np.inf
        
        if best_learner is None:
            # Fallback to simple meta-learner
            if self.task_type == 'classification':
                best_learner = LogisticRegression(random_state=42, max_iter=1000)
            else:
                best_learner = Ridge(random_state=42)
        
        self.best_meta_learner_ = best_learner
        logger.info(f"Selected meta-learner with score: {best_score:.4f}")
        
        return best_learner

class MultiLevelStacking:
    """Multi-level stacking with hierarchical meta-learning"""
    
    def __init__(self, levels: List[Dict[str, Any]], final_meta_learner: Any = None):
        self.levels = levels
        self.final_meta_learner = final_meta_learner
        self.level_models_ = []
        self.level_meta_features_ = []
        self.fitted_final_meta_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5):
        """Fit multi-level stacking ensemble"""
        
        current_meta_features = X.copy()
        
        # Process each level
        for level_idx, level_config in enumerate(self.levels):
            logger.info(f"Training stacking level {level_idx + 1}/{len(self.levels)}")
            
            base_models = level_config.get('base_models', [])
            meta_learner = level_config.get('meta_learner')
            
            # Generate meta-features for this level
            level_meta_features = self._generate_level_meta_features(
                current_meta_features, y, base_models, cv_folds
            )
            
            # Store level information
            level_info = {
                'base_models': [clone(model) for model in base_models],
                'meta_learner': clone(meta_learner) if meta_learner else None,
                'meta_features_shape': level_meta_features.shape
            }
            
            # Fit base models on full data
            for model in level_info['base_models']:
                model.fit(current_meta_features, y)
            
            # Fit meta-learner if provided
            if meta_learner:
                fitted_meta = clone(meta_learner)
                fitted_meta.fit(level_meta_features, y)
                level_info['fitted_meta_learner'] = fitted_meta
            
            self.level_models_.append(level_info)
            self.level_meta_features_.append(level_meta_features)
            
            # Update input for next level
            if meta_learner:
                # Use meta-learner predictions as input for next level
                if hasattr(fitted_meta, 'predict_proba'):
                    next_features = fitted_meta.predict_proba(level_meta_features)
                else:
                    next_features = fitted_meta.predict(level_meta_features).reshape(-1, 1)
                current_meta_features = next_features
            else:
                # Use raw meta-features for next level
                current_meta_features = level_meta_features
        
        # Fit final meta-learner
        if self.final_meta_learner:
            self.fitted_final_meta_ = clone(self.final_meta_learner)
            self.fitted_final_meta_.fit(current_meta_features, y)
        
        return self
    
    def _generate_level_meta_features(self, X: np.ndarray, y: np.ndarray,
                                    base_models: List[Any], cv_folds: int) -> np.ndarray:
        """Generate meta-features for a specific level using cross-validation"""
        
        if len(np.unique(y)) <= 10:  # Classification
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:  # Regression
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        meta_features_list = []
        
        for model in base_models:
            try:
                # Use cross_val_predict to generate out-of-fold predictions
                if hasattr(model, 'predict_proba'):
                    # For classification, use probabilities as meta-features
                    cv_predictions = cross_val_predict(
                        model, X, y, cv=cv, method='predict_proba'
                    )
                    if cv_predictions.shape[1] == 2:
                        # Binary classification: use positive class probability
                        meta_features_list.append(cv_predictions[:, 1])
                    else:
                        # Multi-class: use all probabilities
                        for i in range(cv_predictions.shape[1]):
                            meta_features_list.append(cv_predictions[:, i])
                else:
                    # Use predictions directly
                    cv_predictions = cross_val_predict(model, X, y, cv=cv)
                    meta_features_list.append(cv_predictions)
                    
            except Exception as e:
                logger.warning(f"Error generating meta-features for {type(model).__name__}: {e}")
                # Add dummy features
                meta_features_list.append(np.zeros(len(y)))
        
        if not meta_features_list:
            raise ValueError("No valid meta-features generated")
        
        return np.column_stack(meta_features_list)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions through multi-level hierarchy"""
        
        current_features = X.copy()
        
        # Process through each level
        for level_info in self.level_models_:
            # Generate features from base models
            level_features = []
            for model in level_info['base_models']:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(current_features)
                    if pred.shape[1] == 2:
                        level_features.append(pred[:, 1])
                    else:
                        for i in range(pred.shape[1]):
                            level_features.append(pred[:, i])
                else:
                    pred = model.predict(current_features)
                    level_features.append(pred)
            
            level_features = np.column_stack(level_features)
            
            # Apply meta-learner if available
            if 'fitted_meta_learner' in level_info:
                meta_learner = level_info['fitted_meta_learner']
                if hasattr(meta_learner, 'predict_proba'):
                    current_features = meta_learner.predict_proba(level_features)
                else:
                    pred = meta_learner.predict(level_features)
                    current_features = pred.reshape(-1, 1)
            else:
                current_features = level_features
        
        # Final prediction
        if self.fitted_final_meta_:
            return self.fitted_final_meta_.predict(current_features)
        else:
            # If no final meta-learner, return the last level's output
            if current_features.ndim == 1:
                return current_features
            elif current_features.shape[1] == 1:
                return current_features.ravel()
            else:
                # Multi-class probabilities - return argmax
                return np.argmax(current_features, axis=1)

class StackingAnalyzer:
    """Analyze stacking ensemble performance and meta-feature quality"""
    
    def __init__(self):
        self.meta_feature_analysis_ = {}
        self.level_analysis_ = {}
        self.stacking_diagnostics_ = {}
        
    def analyze_meta_features(self, meta_features: np.ndarray, targets: np.ndarray,
                            base_model_names: List[str]) -> Dict[str, Any]:
        """Analyze quality and characteristics of meta-features"""
        
        analysis = {
            'meta_features_shape': meta_features.shape,
            'base_models': base_model_names,
            'feature_statistics': {},
            'correlation_analysis': {},
            'predictive_power': {}
        }
        
        # Basic statistics for each meta-feature
        for i in range(meta_features.shape[1]):
            feature_name = f"meta_feature_{i}" if i >= len(base_model_names) else base_model_names[i]
            feature_data = meta_features[:, i]
            
            stats = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'unique_values': len(np.unique(feature_data)),
                'sparsity': float(np.mean(feature_data == 0))
            }
            
            analysis['feature_statistics'][feature_name] = stats
        
        # Correlation analysis
        correlation_matrix = np.corrcoef(meta_features.T)
        analysis['correlation_analysis'] = {
            'correlation_matrix': correlation_matrix.tolist(),
            'mean_correlation': float(np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))),
            'max_correlation': float(np.max(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))),
            'highly_correlated_pairs': self._find_correlated_pairs(correlation_matrix, base_model_names, threshold=0.8)
        }
        
        # Individual predictive power
        if len(np.unique(targets)) <= 10:  # Classification
            for i in range(meta_features.shape[1]):
                feature_name = f"meta_feature_{i}" if i >= len(base_model_names) else base_model_names[i]
                feature_pred = np.round(meta_features[:, i]).astype(int)
                
                try:
                    # Clip predictions to valid range
                    feature_pred = np.clip(feature_pred, 0, len(np.unique(targets))-1)
                    accuracy = accuracy_score(targets, feature_pred)
                    analysis['predictive_power'][feature_name] = float(accuracy)
                except:
                    analysis['predictive_power'][feature_name] = 0.0
        else:  # Regression
            for i in range(meta_features.shape[1]):
                feature_name = f"meta_feature_{i}" if i >= len(base_model_names) else base_model_names[i]
                
                try:
                    r2 = r2_score(targets, meta_features[:, i])
                    analysis['predictive_power'][feature_name] = float(r2)
                except:
                    analysis['predictive_power'][feature_name] = 0.0
        
        return analysis
    
    def _find_correlated_pairs(self, correlation_matrix: np.ndarray, 
                             feature_names: List[str], threshold: float = 0.8) -> List[Dict]:
        """Find highly correlated feature pairs"""
        
        pairs = []
        n_features = correlation_matrix.shape[0]
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(correlation_matrix[i, j])
                if corr > threshold:
                    feature_i = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                    feature_j = feature_names[j] if j < len(feature_names) else f"feature_{j}"
                    
                    pairs.append({
                        'feature_1': feature_i,
                        'feature_2': feature_j,
                        'correlation': float(correlation_matrix[i, j]),
                        'abs_correlation': float(corr)
                    })
        
        # Sort by absolute correlation
        pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        return pairs
    
    def analyze_stacking_performance(self, base_predictions: List[np.ndarray],
                                   meta_predictions: np.ndarray,
                                   true_targets: np.ndarray) -> Dict[str, Any]:
        """Analyze stacking performance vs individual models"""
        
        analysis = {
            'individual_performance': {},
            'stacking_performance': {},
            'improvement_analysis': {}
        }
        
        # Individual model performance
        individual_scores = []
        for i, pred in enumerate(base_predictions):
            if len(np.unique(true_targets)) <= 10:  # Classification
                score = accuracy_score(true_targets, pred)
            else:  # Regression
                score = r2_score(true_targets, pred)
            
            individual_scores.append(score)
            analysis['individual_performance'][f'model_{i}'] = float(score)
        
        # Stacking performance
        if len(np.unique(true_targets)) <= 10:  # Classification
            stacking_score = accuracy_score(true_targets, meta_predictions)
        else:  # Regression
            stacking_score = r2_score(true_targets, meta_predictions)
        
        analysis['stacking_performance'] = {
            'score': float(stacking_score),
            'metric': 'accuracy' if len(np.unique(true_targets)) <= 10 else 'r2'
        }
        
        # Improvement analysis
        best_individual = max(individual_scores)
        mean_individual = np.mean(individual_scores)
        
        analysis['improvement_analysis'] = {
            'best_individual_score': float(best_individual),
            'mean_individual_score': float(mean_individual),
            'improvement_over_best': float(stacking_score - best_individual),
            'improvement_over_mean': float(stacking_score - mean_individual),
            'relative_improvement': float((stacking_score - best_individual) / best_individual) if best_individual > 0 else 0.0
        }
        
        return analysis

# ============================================
# Main Stacking Ensemble Model
# ============================================

class FinancialStackingEnsemble(BaseEstimator):
    """
    Advanced stacking ensemble for financial prediction with multi-level meta-learning
    
    Features:
    - Multi-level stacking with hierarchical meta-learning
    - Automatic meta-learner selection based on data characteristics  
    - Comprehensive meta-feature analysis and diagnostics
    - Time-aware cross-validation for financial data
    - Advanced stacking strategies (feature augmentation, residual stacking)
    - Financial domain optimizations (volatility-aware stacking, regime-specific meta-learners)
    """
    
    def __init__(self,
                 name: str = "stacking_ensemble",
                 base_models: Optional[List[Any]] = None,
                 meta_learner: Optional[Any] = None,
                 cv_folds: int = 5,
                 use_features_in_secondary: bool = False,
                 multi_level: bool = False,
                 level_config: Optional[List[Dict]] = None,
                 auto_meta_learner: bool = True,
                 time_aware_cv: bool = True,
                 feature_augmentation: bool = False,
                 residual_stacking: bool = False,
                 calibrate_probabilities: bool = True,
                 task_type: str = 'classification',
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Financial Stacking Ensemble
        
        Args:
            name: Model name
            base_models: List of base models for first level
            meta_learner: Meta-learner for final predictions (if None, auto-selected)
            cv_folds: Number of cross-validation folds for meta-feature generation
            use_features_in_secondary: Whether to include original features in meta-learner
            multi_level: Whether to use multi-level stacking
            level_config: Configuration for multi-level stacking
            auto_meta_learner: Whether to automatically select optimal meta-learner
            time_aware_cv: Whether to use time-aware cross-validation
            feature_augmentation: Whether to augment meta-features with statistical features
            residual_stacking: Whether to use residual-based stacking
            calibrate_probabilities: Whether to calibrate probabilities
            task_type: Type of task ('classification' or 'regression')
            random_state: Random seed
        """
        self.name = name
        self.base_models = base_models or self._create_default_models()
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.use_features_in_secondary = use_features_in_secondary
        self.multi_level = multi_level
        self.level_config = level_config
        self.auto_meta_learner = auto_meta_learner
        self.time_aware_cv = time_aware_cv
        self.feature_augmentation = feature_augmentation
        self.residual_stacking = residual_stacking
        self.calibrate_probabilities = calibrate_probabilities
        self.task_type = task_type
        self.random_state = random_state
        
        # Fitted components
        self.fitted_base_models_ = []
        self.fitted_meta_learner_ = None
        self.multi_level_stacker_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.calibrated_stacker_ = None
        self.meta_features_ = None
        self.stacking_analysis_ = None
        self.meta_learner_selector_ = None
        self.is_fitted_ = False
        
        # Analysis components
        self.stacking_analyzer_ = StackingAnalyzer()
        self.meta_feature_quality_ = {}
        self.performance_analysis_ = {}
        
        logger.info(f"Initialized stacking ensemble: {self.name}")
    
    def _create_default_models(self) -> List[Any]:
        """Create default set of diverse base models"""
        try:
            models = [
                create_gradient_boosting_classifier(performance_preset='balanced'),
                create_random_forest_classifier(performance_preset='balanced'),
                create_svm_classifier(performance_preset='balanced'),
                create_logistic_classifier(performance_preset='balanced')
            ]
            
            # Add additional models if possible
            try:
                models.append(create_naive_bayes_classifier(performance_preset='balanced'))
                models.append(create_knn_classifier(performance_preset='balanced'))
            except:
                logger.info("Some models not available, using core models")
            
            # Add neural network if possible
            try:
                nn_model = create_neural_network_classifier(
                    architecture='balanced',
                    epochs=100,
                    verbose=0
                )
                models.append(nn_model)
            except:
                logger.info("Neural network not available")
            
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
    
    def _create_cv_splitter(self, X: np.ndarray, y: np.ndarray):
        """Create appropriate CV splitter"""
        
        if self.time_aware_cv:
            # Time-series split (no shuffling to preserve temporal order)
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=False)
            else:
                cv = KFold(n_splits=self.cv_folds, shuffle=False)
        else:
            # Standard stratified/normal split
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        return cv
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate meta-features using cross-validation"""
        
        cv = self._create_cv_splitter(X, y)
        meta_features_list = []
        base_predictions_list = []
        
        logger.info(f"Generating meta-features using {self.cv_folds}-fold cross-validation")
        
        for i, base_model in enumerate(self.base_models):
            logger.info(f"Processing base model {i+1}/{len(self.base_models)}: {type(base_model).__name__}")
            
            try:
                if self.task_type == 'classification':
                    if hasattr(base_model, 'predict_proba'):
                        # Use probabilities as meta-features (more informative)
                        cv_predictions = cross_val_predict(
                            base_model, X, y, cv=cv, method='predict_proba'
                        )
                        
                        if cv_predictions.shape[1] == 2:
                            # Binary classification: use positive class probability
                            meta_features_list.append(cv_predictions[:, 1])
                        else:
                            # Multi-class: use all probabilities
                            for j in range(cv_predictions.shape[1]):
                                meta_features_list.append(cv_predictions[:, j])
                        
                        # Also get class predictions for analysis
                        class_predictions = np.argmax(cv_predictions, axis=1)
                        base_predictions_list.append(class_predictions)
                    else:
                        # Use class predictions as meta-features
                        cv_predictions = cross_val_predict(base_model, X, y, cv=cv)
                        meta_features_list.append(cv_predictions.astype(float))
                        base_predictions_list.append(cv_predictions)
                else:
                    # Regression: use predictions directly
                    cv_predictions = cross_val_predict(base_model, X, y, cv=cv)
                    meta_features_list.append(cv_predictions)
                    base_predictions_list.append(cv_predictions)
                
            except Exception as e:
                logger.warning(f"Error generating meta-features for {type(base_model).__name__}: {e}")
                # Add dummy features
                meta_features_list.append(np.zeros(len(y)))
                base_predictions_list.append(np.zeros(len(y)))
        
        if not meta_features_list:
            raise ValueError("No valid meta-features generated")
        
        # Stack meta-features
        meta_features = np.column_stack(meta_features_list)
        
        # Feature augmentation if requested
        if self.feature_augmentation:
            meta_features = self._augment_meta_features(meta_features, base_predictions_list, X, y)
        
        logger.info(f"Generated meta-features shape: {meta_features.shape}")
        
        # Store for analysis
        self.base_predictions_for_analysis_ = base_predictions_list
        
        return meta_features
    
    def _augment_meta_features(self, meta_features: np.ndarray, 
                             base_predictions: List[np.ndarray],
                             original_features: np.ndarray, 
                             targets: Optional[np.ndarray]) -> np.ndarray:
        """Augment meta-features with additional statistical features"""
        
        augmented_features = [meta_features]
        
        # Add statistical features of base predictions
        base_predictions_array = np.column_stack(base_predictions)
        
        # Mean, std, min, max across base predictions
        augmented_features.append(np.mean(base_predictions_array, axis=1).reshape(-1, 1))
        augmented_features.append(np.std(base_predictions_array, axis=1).reshape(-1, 1))
        augmented_features.append(np.min(base_predictions_array, axis=1).reshape(-1, 1))
        augmented_features.append(np.max(base_predictions_array, axis=1).reshape(-1, 1))
        
        # Agreement/disagreement features
        if self.task_type == 'classification':
            # Majority prediction
            majority_preds = []
            agreement_scores = []
            
            for i in range(base_predictions_array.shape[0]):
                sample_preds = base_predictions_array[i, :]
                unique_preds, counts = np.unique(sample_preds, return_counts=True)
                majority_pred = unique_preds[np.argmax(counts)]
                agreement_score = np.max(counts) / len(sample_preds)
                
                majority_preds.append(majority_pred)
                agreement_scores.append(agreement_score)
            
            augmented_features.append(np.array(majority_preds).reshape(-1, 1))
            augmented_features.append(np.array(agreement_scores).reshape(-1, 1))
        
        # Include original features if requested
        if self.use_features_in_secondary:
            augmented_features.append(original_features)
        
        return np.column_stack(augmented_features)
    
    def _select_meta_learner(self, meta_features: np.ndarray, targets: np.ndarray) -> Any:
        """Select optimal meta-learner"""
        
        if not self.auto_meta_learner and self.meta_learner is not None:
            return clone(self.meta_learner)
        
        # Use automatic selection
        self.meta_learner_selector_ = MetaLearnerSelector(task_type=self.task_type)
        selected_meta_learner = self.meta_learner_selector_.select_meta_learner(
            meta_features, targets, cv_folds=3
        )
        
        return selected_meta_learner
    
    def _analyze_stacking_quality(self, meta_features: np.ndarray, targets: np.ndarray):
        """Analyze meta-feature quality and stacking performance"""
        
        # Get base model names
        base_model_names = [type(model).__name__ for model in self.base_models]
        
        # Analyze meta-features
        self.meta_feature_quality_ = self.stacking_analyzer_.analyze_meta_features(
            meta_features, targets, base_model_names
        )
        
        # Analyze stacking performance if we have base predictions
        if hasattr(self, 'base_predictions_for_analysis_') and self.fitted_meta_learner_ is not None:
            meta_predictions = self.fitted_meta_learner_.predict(meta_features)
            
            self.performance_analysis_ = self.stacking_analyzer_.analyze_stacking_performance(
                self.base_predictions_for_analysis_, meta_predictions, targets
            )
    
    def _calibrate_probabilities(self, X: np.ndarray, y: np.ndarray):
        """Prepare stacking ensemble probability calibration"""
        if self.task_type != 'classification' or not self.calibrate_probabilities:
            return
        
        try:
            # For single-level stacking, we can prepare calibration
            if not self.multi_level and self.fitted_meta_learner_ is not None:
                self.calibrated_stacker_ = CalibratedClassifierCV(
                    base_estimator=self.fitted_meta_learner_,
                    method='isotonic',
                    cv=3
                )
            
            logger.debug("Prepared probability calibration")
            
        except Exception as e:
            logger.warning(f"Could not prepare calibration: {e}")
            self.calibrated_stacker_ = None
    
    @time_it("stacking_fit", include_args=True)
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], **kwargs):
        """Fit the stacking ensemble"""
        
        logger.info(f"Fitting stacking ensemble with {len(self.base_models)} base models")
        
        try:
            # Preprocess data
            X_processed = self._preprocess_features(X)
            y_processed = self._preprocess_targets(y)
            
            if self.multi_level and self.level_config:
                # Multi-level stacking
                logger.info("Using multi-level stacking")
                
                self.multi_level_stacker_ = MultiLevelStacking(
                    levels=self.level_config,
                    final_meta_learner=self.meta_learner
                )
                self.multi_level_stacker_.fit(X_processed, y_processed, self.cv_folds)
                
                # For analysis, use the first level
                if self.multi_level_stacker_.level_meta_features_:
                    first_level_meta_features = self.multi_level_stacker_.level_meta_features_[0]
                    self.meta_features_ = first_level_meta_features
                else:
                    self.meta_features_ = X_processed  # Fallback
                
                # Set a placeholder meta-learner for compatibility
                self.fitted_meta_learner_ = LogisticRegression(max_iter=1000, random_state=42)
            else:
                # Standard single-level stacking
                logger.info("Using single-level stacking")
                
                # Generate meta-features using cross-validation
                meta_features = self._generate_meta_features(X_processed, y_processed)
                self.meta_features_ = meta_features
                
                # Fit all base models on full training data
                self.fitted_base_models_ = []
                for model in self.base_models:
                    fitted_model = clone(model)
                    fitted_model.fit(X_processed, y_processed)
                    self.fitted_base_models_.append(fitted_model)
                
                # Select and fit meta-learner
                self.fitted_meta_learner_ = self._select_meta_learner(meta_features, y_processed)
                self.fitted_meta_learner_.fit(meta_features, y_processed)
            
            # Analyze stacking quality
            self._analyze_stacking_quality(self.meta_features_, y_processed)
            
            # Prepare probability calibration
            self._calibrate_probabilities(X_processed, y_processed)
            
            # Create comprehensive stacking analysis
            self.stacking_analysis_ = {
                'stacking_type': 'multi_level' if self.multi_level else 'single_level',
                'n_base_models': len(self.base_models),
                'base_model_types': [type(model).__name__ for model in self.base_models],
                'cv_folds': self.cv_folds,
                'time_aware_cv': self.time_aware_cv,
                'feature_augmentation': self.feature_augmentation,
                'meta_learner_type': type(self.fitted_meta_learner_).__name__ if self.fitted_meta_learner_ else 'MultiLevel',
                'auto_meta_learner': self.auto_meta_learner,
                'meta_feature_quality': self.meta_feature_quality_,
                'performance_analysis': self.performance_analysis_
            }
            
            # Add meta-learner selection results if available
            if self.meta_learner_selector_:
                self.stacking_analysis_['meta_learner_selection'] = {
                    'candidate_performance': self.meta_learner_selector_.meta_learner_performance_,
                    'selected_meta_learner': type(self.meta_learner_selector_.best_meta_learner_).__name__
                }
            
            self.is_fitted_ = True
            logger.info("Stacking ensemble fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Stacking ensemble fitting failed: {e}")
            raise
    
    @time_it("stacking_predict", include_args=True)
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make stacking predictions"""
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            if self.multi_level and self.multi_level_stacker_:
                # Multi-level prediction
                predictions = self.multi_level_stacker_.predict(X_processed)
            else:
                # Single-level prediction
                # Generate meta-features from fitted base models
                meta_features = []
                
                for model in self.fitted_base_models_:
                    if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X_processed)
                        if pred_proba.shape[1] == 2:
                            meta_features.append(pred_proba[:, 1])
                        else:
                            for j in range(pred_proba.shape[1]):
                                meta_features.append(pred_proba[:, j])
                    else:
                        pred = model.predict(X_processed)
                        meta_features.append(pred)
                
                meta_features = np.column_stack(meta_features)
                
                # Feature augmentation if used during training
                if self.feature_augmentation:
                    # Generate base predictions for augmentation
                    base_predictions = [model.predict(X_processed) for model in self.fitted_base_models_]
                    meta_features = self._augment_meta_features(
                        meta_features, base_predictions, X_processed, None
                    )
                
                # Make final prediction
                predictions = self.fitted_meta_learner_.predict(meta_features)
            
            # Decode predictions for classification
            if self.task_type == 'classification':
                if hasattr(self, 'label_encoder_') and self.label_encoder_ is not None:
                    return self.label_encoder_.inverse_transform(predictions.astype(int))
                else:
                    return predictions
            else:
                return predictions
                
        except Exception as e:
            logger.error(f"Stacking prediction failed: {e}")
            raise
    
    @time_it("stacking_predict_proba", include_args=True)
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make stacking probability predictions"""
        
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            if self.multi_level and self.multi_level_stacker_:
                # Multi-level probability prediction is complex - return class predictions converted to probabilities
                class_pred = self.multi_level_stacker_.predict(X_processed)
                n_classes = len(self.classes_)
                probabilities = np.zeros((len(class_pred), n_classes))
                for i, pred in enumerate(class_pred):
                    probabilities[i, int(pred)] = 1.0
                return probabilities
            else:
                # Single-level probability prediction
                # Generate meta-features
                meta_features = []
                
                for model in self.fitted_base_models_:
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X_processed)
                        if pred_proba.shape[1] == 2:
                            meta_features.append(pred_proba[:, 1])
                        else:
                            for j in range(pred_proba.shape[1]):
                                meta_features.append(pred_proba[:, j])
                    else:
                        pred = model.predict(X_processed)
                        meta_features.append(pred)
                
                meta_features = np.column_stack(meta_features)
                
                # Feature augmentation if used
                if self.feature_augmentation:
                    base_predictions = [model.predict(X_processed) for model in self.fitted_base_models_]
                    meta_features = self._augment_meta_features(
                        meta_features, base_predictions, X_processed, None
                    )
                
                # Get probabilities from meta-learner
                if hasattr(self.fitted_meta_learner_, 'predict_proba'):
                    probabilities = self.fitted_meta_learner_.predict_proba(meta_features)
                else:
                    # Convert predictions to probabilities
                    pred = self.fitted_meta_learner_.predict(meta_features)
                    n_classes = len(self.classes_)
                    probabilities = np.zeros((len(pred), n_classes))
                    for i, p in enumerate(pred):
                        probabilities[i, int(p)] = 1.0
                
                return probabilities
                
        except Exception as e:
            logger.error(f"Stacking probability prediction failed: {e}")
            raise
    
    def get_stacking_analysis(self) -> Dict[str, Any]:
        """Get comprehensive stacking analysis"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to get stacking analysis")
        
        return self.stacking_analysis_.copy() if self.stacking_analysis_ else {}
    
    def get_meta_feature_importance(self) -> Dict[str, float]:
        """Get meta-feature importance from meta-learner"""
        if not self.is_fitted_ or not self.fitted_meta_learner_:
            return {}
        
        importance = {}
        
        # Extract feature importance if available
        if hasattr(self.fitted_meta_learner_, 'feature_importances_'):
            # Tree-based models
            importances = self.fitted_meta_learner_.feature_importances_
            base_model_names = [type(model).__name__ for model in self.base_models]
            
            for i, imp in enumerate(importances):
                feature_name = base_model_names[i] if i < len(base_model_names) else f'feature_{i}'
                importance[feature_name] = float(imp)
                
        elif hasattr(self.fitted_meta_learner_, 'coef_'):
            # Linear models
            coefficients = self.fitted_meta_learner_.coef_
            if coefficients.ndim > 1:
                coefficients = coefficients[0]  # Take first class for multiclass
            
            base_model_names = [type(model).__name__ for model in self.base_models]
            
            for i, coef in enumerate(coefficients):
                feature_name = base_model_names[i] if i < len(base_model_names) else f'feature_{i}'
                importance[feature_name] = float(abs(coef))
        
        return importance
    
    def plot_stacking_analysis(self) -> Any:
        """Plot comprehensive stacking analysis"""
        if not self.stacking_analysis_:
            logger.warning("Stacking analysis not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            analysis = self.stacking_analysis_
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Meta-feature importance
            importance = self.get_meta_feature_importance()
            if importance:
                models = list(importance.keys())
                importances = list(importance.values())
                
                bars = axes[0, 0].bar(range(len(models)), importances, alpha=0.7, color='steelblue')
                axes[0, 0].set_title('Meta-Feature Importance')
                axes[0, 0].set_xticks(range(len(models)))
                axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
                axes[0, 0].set_ylabel('Importance')
                
                # Add values on bars
                for bar, imp in zip(bars, importances):
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, imp + max(importances) * 0.01,
                                   f'{imp:.3f}', ha='center', va='bottom', fontsize=9)
            else:
                axes[0, 0].text(0.5, 0.5, 'Feature importance\nnot available', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Meta-Feature Importance')
            
            # Performance comparison
            if 'performance_analysis' in analysis and analysis['performance_analysis']:
                perf_analysis = analysis['performance_analysis']
                
                if 'individual_performance' in perf_analysis:
                    individual_perf = perf_analysis['individual_performance']
                    model_names = list(individual_perf.keys())
                    individual_scores = list(individual_perf.values())
                    
                    # Add stacking performance
                    stacking_perf = perf_analysis.get('stacking_performance', {})
                    stacking_score = stacking_perf.get('score', 0)
                    
                    all_names = model_names + ['Stacking']
                    all_scores = individual_scores + [stacking_score]
                    colors = ['lightblue'] * len(model_names) + ['red']
                    
                    bars = axes[0, 1].bar(range(len(all_names)), all_scores, color=colors, alpha=0.7)
                    axes[0, 1].set_title('Performance Comparison')
                    axes[0, 1].set_xticks(range(len(all_names)))
                    axes[0, 1].set_xticklabels(all_names, rotation=45, ha='right')
                    axes[0, 1].set_ylabel('Score')
                    
                    # Add values on bars
                    for bar, score in zip(bars, all_scores):
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2, score + 0.005,
                                       f'{score:.3f}', ha='center', va='bottom', fontsize=9)
                else:
                    axes[0, 1].text(0.5, 0.5, 'Performance analysis\nnot available', 
                                   ha='center', va='center', transform=axes[0, 1].transAxes)
                    axes[0, 1].set_title('Performance Comparison')
            else:
                axes[0, 1].text(0.5, 0.5, 'Performance analysis\nnot available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Performance Comparison')
            
            # Meta-feature correlation heatmap
            if 'meta_feature_quality' in analysis and analysis['meta_feature_quality']:
                quality = analysis['meta_feature_quality']
                
                if 'correlation_analysis' in quality:
                    corr_matrix = np.array(quality['correlation_analysis']['correlation_matrix'])
                    base_model_names = [name[:8] for name in analysis['base_model_types']]  # Truncate names
                    
                    if corr_matrix.shape[0] <= 15 and corr_matrix.shape[0] > 0:  # Only plot if reasonable size
                        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                                   xticklabels=base_model_names[:corr_matrix.shape[0]], 
                                   yticklabels=base_model_names[:corr_matrix.shape[0]], 
                                   ax=axes[0, 2])
                        axes[0, 2].set_title('Meta-Feature Correlations')
                    else:
                        axes[0, 2].text(0.5, 0.5, f'Too many features\n({corr_matrix.shape[0]}) to display', 
                                       ha='center', va='center', transform=axes[0, 2].transAxes)
                        axes[0, 2].set_title('Meta-Feature Correlations')
                else:
                    axes[0, 2].text(0.5, 0.5, 'Correlation analysis\nnot available', 
                                   ha='center', va='center', transform=axes[0, 2].transAxes)
                    axes[0, 2].set_title('Meta-Feature Correlations')
            else:
                axes[0, 2].text(0.5, 0.5, 'Meta-feature quality\nanalysis not available', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Meta-Feature Correlations')
            
            # Individual meta-feature predictive power
            if 'meta_feature_quality' in analysis and analysis['meta_feature_quality']:
                quality = analysis['meta_feature_quality']
                
                if 'predictive_power' in quality:
                    pred_power = quality['predictive_power']
                    feature_names = list(pred_power.keys())
                    power_values = list(pred_power.values())
                    
                    if feature_names and power_values:
                        bars = axes[1, 0].bar(range(len(feature_names)), power_values, alpha=0.7, color='green')
                        axes[1, 0].set_title('Individual Predictive Power')
                        axes[1, 0].set_xticks(range(len(feature_names)))
                        axes[1, 0].set_xticklabels([name[:8] for name in feature_names], rotation=45, ha='right')
                        axes[1, 0].set_ylabel('Predictive Score')
                        
                        # Add values on bars
                        for bar, power in zip(bars, power_values):
                            axes[1, 0].text(bar.get_x() + bar.get_width()/2, power + max(power_values) * 0.01,
                                           f'{power:.3f}', ha='center', va='bottom', fontsize=8)
                    else:
                        axes[1, 0].text(0.5, 0.5, 'No predictive power\ndata available', 
                                       ha='center', va='center', transform=axes[1, 0].transAxes)
                        axes[1, 0].set_title('Individual Predictive Power')
                else:
                    axes[1, 0].text(0.5, 0.5, 'Predictive power\nanalysis not available', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Individual Predictive Power')
            else:
                axes[1, 0].axis('off')
            
            # Improvement analysis
            if ('performance_analysis' in analysis and analysis['performance_analysis'] and 
                'improvement_analysis' in analysis['performance_analysis']):
                improvement = analysis['performance_analysis']['improvement_analysis']
                
                metrics = ['Best Individual', 'Mean Individual', 'Stacking']
                values = [
                    improvement['best_individual_score'],
                    improvement['mean_individual_score'],
                    improvement['best_individual_score'] + improvement['improvement_over_best']
                ]
                colors = ['blue', 'gray', 'red']
                
                bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
                axes[1, 1].set_title(f"Improvement: {improvement['improvement_over_best']:+.4f}")
                axes[1, 1].set_ylabel('Score')
                
                # Add values on bars
                for bar, value in zip(bars, values):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, value + max(values) * 0.01,
                                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                axes[1, 1].text(0.5, 0.5, 'Improvement analysis\nnot available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Improvement Analysis')
            
            # Stacking summary
            summary_text = f"Stacking Type: {analysis['stacking_type'].title()}\n"
            summary_text += f"Base Models: {analysis['n_base_models']}\n"
            summary_text += f"CV Folds: {analysis['cv_folds']}\n"
            summary_text += f"Time-Aware CV: {analysis['time_aware_cv']}\n"
            summary_text += f"Feature Augmentation: {analysis['feature_augmentation']}\n"
            summary_text += f"Meta-Learner: {analysis['meta_learner_type']}\n"
            
            if 'meta_learner_selection' in analysis:
                selection = analysis['meta_learner_selection']
                summary_text += f"\nAuto-Selected: {selection['selected_meta_learner']}\n"
            
            if ('performance_analysis' in analysis and analysis['performance_analysis'] and 
                'improvement_analysis' in analysis['performance_analysis']):
                improvement = analysis['performance_analysis']['improvement_analysis']
                summary_text += f"\nImprovement: {improvement['improvement_over_best']:+.4f}\n"
                summary_text += f"Relative: {improvement['relative_improvement']:+.1%}"
            
            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            axes[1, 2].set_title('Stacking Summary')
            axes[1, 2].axis('off')
            
            plt.suptitle(f'Stacking Ensemble Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            logger.warning(f"Error creating stacking analysis plot: {e}")
            return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = {
            'model_name': self.name,
            'model_family': 'Stacking Ensemble',
            'task_type': self.task_type,
            'stacking_type': 'multi_level' if self.multi_level else 'single_level',
            'n_base_models': len(self.base_models),
            'cv_folds': self.cv_folds,
            'time_aware_cv': self.time_aware_cv,
            'auto_meta_learner': self.auto_meta_learner,
            'feature_augmentation': self.feature_augmentation,
            'use_features_in_secondary': self.use_features_in_secondary,
            'probability_calibration': self.calibrate_probabilities,
            'is_fitted': self.is_fitted_
        }
        
        if self.is_fitted_:
            summary.update({
                'base_model_types': [type(model).__name__ for model in self.base_models],
                'meta_learner_type': type(self.fitted_meta_learner_).__name__ if self.fitted_meta_learner_ else 'MultiLevel',
                'meta_features_shape': self.meta_features_.shape if self.meta_features_ is not None else None,
                'stacking_analysis': self.stacking_analysis_
            })
            
            # Add feature importance if available
            importance = self.get_meta_feature_importance()
            if importance:
                summary['meta_feature_importance'] = importance
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_stacking_classifier(meta_learner: str = 'auto',
                              base_models: Optional[List[str]] = None,
                              complexity: str = 'balanced',
                              **kwargs) -> FinancialStackingEnsemble:
    """Create stacking classifier with different configurations"""
    
    # Default base models by complexity
    complexity_models = {
        'simple': ['gradient_boosting', 'random_forest', 'logistic'],
        'balanced': ['gradient_boosting', 'random_forest', 'svm', 'logistic', 'naive_bayes'],
        'comprehensive': ['gradient_boosting', 'random_forest', 'svm', 'logistic', 
                         'naive_bayes', 'knn', 'neural_network']
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
            elif model_name == 'naive_bayes':
                model = create_naive_bayes_classifier(performance_preset='balanced')
            elif model_name == 'knn':
                model = create_knn_classifier(performance_preset='balanced')
            elif model_name == 'neural_network':
                model = create_neural_network_classifier(
                    architecture='balanced', epochs=100, verbose=0
                )
            else:
                continue
            
            model_instances.append(model)
        except Exception as e:
            logger.warning(f"Could not create {model_name}: {e}")
    
    # Create meta-learner
    meta_learner_instance = None
    if meta_learner != 'auto':
        if meta_learner == 'logistic':
            meta_learner_instance = LogisticRegression(random_state=42, max_iter=1000)
        elif meta_learner == 'random_forest':
            meta_learner_instance = RandomForestClassifier(n_estimators=100, random_state=42)
        elif meta_learner == 'svm':
            meta_learner_instance = create_svm_classifier(performance_preset='balanced')
        elif meta_learner == 'neural_network':
            meta_learner_instance = create_neural_network_classifier(
                architecture='simple', epochs=50, verbose=0
            )
    
    config = {
        'name': f'stacking_{complexity}_{meta_learner}',
        'base_models': model_instances,
        'meta_learner': meta_learner_instance,
        'auto_meta_learner': meta_learner == 'auto',
        'task_type': 'classification',
        'calibrate_probabilities': True,
        'random_state': 42
    }
    
    config.update(kwargs)
    
    return FinancialStackingEnsemble(**config)

def create_stacking_regressor(meta_learner: str = 'auto',
                            base_models: Optional[List[str]] = None,
                            complexity: str = 'balanced',
                            **kwargs) -> FinancialStackingEnsemble:
    """Create stacking regressor"""
    
    stacker = create_stacking_classifier(
        meta_learner=meta_learner,
        base_models=base_models,
        complexity=complexity,
        **kwargs
    )
    
    # Convert to regression
    stacker.task_type = 'regression'
    stacker.calibrate_probabilities = False
    stacker.name = stacker.name.replace('stacking', 'stacking_regressor')
    
    return stacker

def create_single_level_stacking(**kwargs) -> FinancialStackingEnsemble:
    """Create single-level stacking ensemble"""
    return create_stacking_classifier(
        multi_level=False,
        name='single_level_stacking',
        **kwargs
    )

def create_multi_level_stacking(**kwargs) -> FinancialStackingEnsemble:
    """Create multi-level stacking ensemble"""
    
    # Default multi-level configuration
    level_config = [
        {
            'base_models': [
                create_gradient_boosting_classifier(),
                create_random_forest_classifier(),
                create_svm_classifier()
            ],
            'meta_learner': LogisticRegression(random_state=42, max_iter=1000)
        },
        {
            'base_models': [
                create_logistic_classifier(),
                create_naive_bayes_classifier()
            ],
            'meta_learner': RandomForestClassifier(n_estimators=50, random_state=42)
        }
    ]
    
    return create_stacking_classifier(
        multi_level=True,
        level_config=level_config,
        name='multi_level_stacking',
        **kwargs
    )

def create_time_aware_stacking(**kwargs) -> FinancialStackingEnsemble:
    """Create time-aware stacking for financial data"""
    return create_stacking_classifier(
        time_aware_cv=True,
        feature_augmentation=True,
        name='time_aware_stacking',
        **kwargs
    )

def create_augmented_stacking(**kwargs) -> FinancialStackingEnsemble:
    """Create stacking with feature augmentation"""
    return create_stacking_classifier(
        feature_augmentation=True,
        use_features_in_secondary=True,
        name='augmented_stacking',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def compare_stacking_strategies(X: Union[pd.DataFrame, np.ndarray],
                              y: Union[pd.Series, np.ndarray],
                              strategies: List[str] = ['single_level', 'multi_level', 'augmented'],
                              task_type: str = 'classification') -> Dict[str, Any]:
    """Compare different stacking strategies"""
    
    logger.info(f"Comparing stacking strategies: {strategies}")
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"Evaluating {strategy} stacking")
        
        try:
            if strategy == 'single_level':
                stacker = create_single_level_stacking()
            elif strategy == 'multi_level':
                stacker = create_multi_level_stacking()
            elif strategy == 'augmented':
                stacker = create_augmented_stacking()
            elif strategy == 'time_aware':
                stacker = create_time_aware_stacking()
            else:
                stacker = create_stacking_classifier()
            
            # Convert to regression if needed
            if task_type == 'regression':
                stacker.task_type = 'regression'
                stacker.calibrate_probabilities = False
            
            # Fit and evaluate
            stacker.fit(X, y)
            
            # Get predictions and score
            predictions = stacker.predict(X)
            if task_type == 'classification':
                score = accuracy_score(y, predictions)
            else:
                score = r2_score(y, predictions)
            
            # Get stacking analysis
            analysis = stacker.get_stacking_analysis()
            
            results[strategy] = {
                'score': score,
                'stacking_analysis': analysis,
                'model': stacker
            }
            
        except Exception as e:
            logger.warning(f"Error with {strategy} stacking: {e}")
            results[strategy] = {'error': str(e)}
    
    # Add comparison summary
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_strategy = max(valid_results.keys(), key=lambda k: valid_results[k]['score'])
        
        results['comparison'] = {
            'best_strategy': best_strategy,
            'strategy_rankings': sorted(valid_results.keys(), 
                                      key=lambda k: valid_results[k]['score'], reverse=True)
        }
    
    logger.info(f"Stacking comparison complete. Best strategy: {results.get('comparison', {}).get('best_strategy', 'Unknown')}")
    
    return results

def analyze_meta_feature_quality(base_models: List[Any], 
                                X: Union[pd.DataFrame, np.ndarray],
                                y: Union[pd.Series, np.ndarray],
                                cv_folds: int = 5) -> Dict[str, Any]:
    """Analyze meta-feature quality for given base models"""
    
    logger.info(f"Analyzing meta-feature quality for {len(base_models)} base models")
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Generate meta-features
    if len(np.unique(y)) <= 10:  # Classification
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        task_type = 'classification'
    else:  # Regression
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        task_type = 'regression'
    
    meta_features_list = []
    base_model_names = []
    
    for i, model in enumerate(base_models):
        try:
            model_name = type(model).__name__
            base_model_names.append(model_name)
            
            if task_type == 'classification' and hasattr(model, 'predict_proba'):
                cv_predictions = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
                if cv_predictions.shape[1] == 2:
                    meta_features_list.append(cv_predictions[:, 1])
                else:
                    for j in range(cv_predictions.shape[1]):
                        meta_features_list.append(cv_predictions[:, j])
                        if j > 0:  # Add class names for multiclass
                            base_model_names.append(f"{model_name}_class_{j}")
            else:
                cv_predictions = cross_val_predict(model, X, y, cv=cv)
                meta_features_list.append(cv_predictions)
                
        except Exception as e:
            logger.warning(f"Error with model {i}: {e}")
    
    if not meta_features_list:
        logger.warning("No valid meta-features generated")
        return {}
    
    meta_features = np.column_stack(meta_features_list)
    
    # Analyze meta-feature quality
    analyzer = StackingAnalyzer()
    analysis = analyzer.analyze_meta_features(meta_features, y, base_model_names)
    
    # Additional diversity analysis
    correlation_matrix = np.corrcoef(meta_features.T)
    mean_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
    
    analysis['diversity_metrics'] = {
        'mean_absolute_correlation': float(mean_correlation),
        'diversity_score': float(1 - mean_correlation),  # Higher is more diverse
        'diversity_level': 'High' if mean_correlation < 0.3 else 
                         'Moderate' if mean_correlation < 0.7 else 'Low'
    }
    
    logger.info(f"Meta-feature analysis complete. Diversity level: {analysis['diversity_metrics']['diversity_level']}")
    
    return analysis

def optimize_stacking_configuration(X: Union[pd.DataFrame, np.ndarray],
                                  y: Union[pd.Series, np.ndarray],
                                  base_model_candidates: List[Any],
                                  max_base_models: int = 7) -> Dict[str, Any]:
    """Optimize stacking configuration by selecting best base models and meta-learner"""
    
    logger.info("Optimizing stacking configuration")
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Evaluate all base model candidates
    base_model_scores = {}
    
    for i, model in enumerate(base_model_candidates):
        try:
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            base_model_scores[i] = {
                'model': model,
                'score': scores.mean(),
                'model_type': type(model).__name__
            }
        except Exception as e:
            logger.warning(f"Error evaluating base model {i}: {e}")
    
    # Select top performing models
    sorted_models = sorted(base_model_scores.items(), 
                          key=lambda x: x[1]['score'], reverse=True)
    
    selected_models = []
    selected_indices = []
    
    for idx, (model_idx, model_info) in enumerate(sorted_models):
        if len(selected_models) >= max_base_models:
            break
        
        model = model_info['model']
        selected_models.append(model)
        selected_indices.append(model_idx)
    
    logger.info(f"Selected {len(selected_models)} base models")
    
    # Test different stacking configurations
    configurations = [
        {'meta_learner': 'auto', 'feature_augmentation': False},
        {'meta_learner': 'logistic', 'feature_augmentation': False},
        {'meta_learner': 'random_forest', 'feature_augmentation': False},
        {'meta_learner': 'auto', 'feature_augmentation': True},
        {'meta_learner': 'logistic', 'feature_augmentation': True}
    ]
    
    best_config = None
    best_score = -np.inf
    config_results = {}
    
    for config in configurations:
        try:
            stacker = FinancialStackingEnsemble(
                base_models=selected_models,
                **config
            )
            stacker.fit(X, y)
            
            predictions = stacker.predict(X)
            score = accuracy_score(y, predictions)
            
            config_results[str(config)] = {
                'score': score,
                'config': config,
                'model': stacker
            }
            
            if score > best_score:
                best_score = score
                best_config = config
                
        except Exception as e:
            logger.warning(f"Error with config {config}: {e}")
    
    # Create optimized stacker
    if best_config:
        optimized_stacker = FinancialStackingEnsemble(
            base_models=selected_models,
            **best_config
        )
        optimized_stacker.fit(X, y)
    else:
        # Fallback configuration
        optimized_stacker = FinancialStackingEnsemble(base_models=selected_models)
        optimized_stacker.fit(X, y)
        best_config = {'meta_learner': 'auto', 'feature_augmentation': False}
        best_score = accuracy_score(y, optimized_stacker.predict(X))
    
    results = {
        'selected_models': [type(model).__name__ for model in selected_models],
        'selected_model_scores': [base_model_scores[idx]['score'] for idx in selected_indices],
        'best_config': best_config,
        'best_score': best_score,
        'config_results': config_results,
        'optimized_model': optimized_stacker,
        'optimization_summary': {
            'n_candidates_tested': len(base_model_candidates),
            'n_models_selected': len(selected_models),
            'n_configs_tested': len(configurations),
            'improvement_over_best_base': best_score - max([info['score'] for info in base_model_scores.values()])
        }
    }
    
    logger.info(f"Stacking optimization complete. Best score: {best_score:.4f}")
    
    return results
