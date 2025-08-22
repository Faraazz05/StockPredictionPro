# ============================================
# StockPredictionPro - src/models/ensemble/bagging.py
# Advanced bagging ensemble methods for financial prediction with bootstrap aggregation
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
from collections import defaultdict
import warnings

# Core ML imports
from sklearn.ensemble import (
    BaggingClassifier, BaggingRegressor, RandomForestClassifier, 
    RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, roc_curve, log_loss
)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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

logger = get_logger('models.ensemble.bagging')

# ============================================
# Advanced Bagging Strategies
# ============================================

class AdvancedBootstrapSampler:
    """Advanced bootstrap sampling with financial domain optimizations"""
    
    def __init__(self, strategy: str = 'standard', 
                 block_size: Optional[int] = None,
                 overlap_ratio: float = 0.0,
                 time_aware: bool = True):
        self.strategy = strategy
        self.block_size = block_size
        self.overlap_ratio = overlap_ratio
        self.time_aware = time_aware
        
    def generate_bootstrap_samples(self, X: np.ndarray, y: np.ndarray, 
                                 n_samples: int, random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate bootstrap samples using specified strategy"""
        
        np.random.seed(random_state)
        samples = []
        
        for i in range(n_samples):
            if self.strategy == 'standard':
                sample_X, sample_y = self._standard_bootstrap(X, y)
            elif self.strategy == 'balanced':
                sample_X, sample_y = self._balanced_bootstrap(X, y)
            elif self.strategy == 'block':
                sample_X, sample_y = self._block_bootstrap(X, y)
            elif self.strategy == 'stratified':
                sample_X, sample_y = self._stratified_bootstrap(X, y)
            elif self.strategy == 'time_series':
                sample_X, sample_y = self._time_series_bootstrap(X, y)
            else:
                sample_X, sample_y = self._standard_bootstrap(X, y)
            
            samples.append((sample_X, sample_y))
            
            # Update random state for next iteration
            np.random.seed(random_state + i + 1)
        
        return samples
    
    def _standard_bootstrap(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Standard bootstrap sampling with replacement"""
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def _balanced_bootstrap(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balanced bootstrap ensuring class distribution"""
        if len(np.unique(y)) > 10:  # Regression
            return self._standard_bootstrap(X, y)
        
        # Classification - maintain class balance
        n_samples = len(X)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        balanced_indices = []
        samples_per_class = n_samples // len(unique_classes)
        
        for cls in unique_classes:
            class_indices = np.where(y == cls)[0]
            if len(class_indices) == 0:
                continue
            
            # Sample with replacement from this class
            sampled_indices = np.random.choice(
                class_indices, 
                size=min(samples_per_class, len(class_indices)), 
                replace=True
            )
            balanced_indices.extend(sampled_indices)
        
        # Fill remaining samples randomly
        remaining_samples = n_samples - len(balanced_indices)
        if remaining_samples > 0:
            additional_indices = np.random.choice(
                len(X), size=remaining_samples, replace=True
            )
            balanced_indices.extend(additional_indices)
        
        balanced_indices = np.array(balanced_indices)
        return X[balanced_indices], y[balanced_indices]
    
    def _block_bootstrap(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Block bootstrap for time series data"""
        n_samples = len(X)
        
        if self.block_size is None:
            # Auto-determine block size
            self.block_size = max(1, int(np.sqrt(n_samples)))
        
        n_blocks = int(np.ceil(n_samples / self.block_size))
        sampled_indices = []
        
        for _ in range(n_blocks):
            # Random starting point
            start_idx = np.random.randint(0, max(1, n_samples - self.block_size + 1))
            end_idx = min(start_idx + self.block_size, n_samples)
            
            block_indices = list(range(start_idx, end_idx))
            sampled_indices.extend(block_indices)
        
        # Trim to desired length
        sampled_indices = sampled_indices[:n_samples]
        sampled_indices = np.array(sampled_indices)
        
        return X[sampled_indices], y[sampled_indices]
    
    def _stratified_bootstrap(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Stratified bootstrap maintaining proportions"""
        if len(np.unique(y)) > 10:  # Regression - use quantile stratification
            return self._quantile_stratified_bootstrap(X, y)
        
        # Classification stratification
        n_samples = len(X)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_proportions = class_counts / n_samples
        
        stratified_indices = []
        
        for cls, proportion in zip(unique_classes, class_proportions):
            class_indices = np.where(y == cls)[0]
            n_class_samples = int(n_samples * proportion)
            
            if len(class_indices) > 0:
                sampled_indices = np.random.choice(
                    class_indices, size=n_class_samples, replace=True
                )
                stratified_indices.extend(sampled_indices)
        
        stratified_indices = np.array(stratified_indices)
        return X[stratified_indices], y[stratified_indices]
    
    def _quantile_stratified_bootstrap(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantile-based stratification for regression"""
        n_samples = len(X)
        n_quantiles = 5  # Quintiles
        
        quantiles = np.percentile(y, np.linspace(0, 100, n_quantiles + 1))
        stratified_indices = []
        
        for i in range(n_quantiles):
            if i == 0:
                mask = y <= quantiles[i + 1]
            elif i == n_quantiles - 1:
                mask = y >= quantiles[i]
            else:
                mask = (y > quantiles[i]) & (y <= quantiles[i + 1])
            
            quantile_indices = np.where(mask)[0]
            
            if len(quantile_indices) > 0:
                n_quantile_samples = len(quantile_indices)
                sampled_indices = np.random.choice(
                    quantile_indices, size=n_quantile_samples, replace=True
                )
                stratified_indices.extend(sampled_indices)
        
        stratified_indices = np.array(stratified_indices)
        return X[stratified_indices], y[stratified_indices]
    
    def _time_series_bootstrap(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Time-series aware bootstrap preserving temporal order"""
        n_samples = len(X)
        
        if self.block_size is None:
            self.block_size = max(10, int(n_samples * 0.1))  # 10% of data
        
        # Generate overlapping blocks
        max_start = n_samples - self.block_size
        if max_start <= 0:
            return self._standard_bootstrap(X, y)
        
        # Calculate overlap
        overlap = int(self.block_size * self.overlap_ratio)
        step_size = max(1, self.block_size - overlap)
        
        # Generate possible starting points
        start_points = list(range(0, max_start + 1, step_size))
        
        # Sample blocks
        n_blocks_needed = int(np.ceil(n_samples / self.block_size))
        sampled_starts = np.random.choice(start_points, size=n_blocks_needed, replace=True)
        
        sampled_indices = []
        for start in sampled_starts:
            end = min(start + self.block_size, n_samples)
            block_indices = list(range(start, end))
            sampled_indices.extend(block_indices)
        
        # Trim to desired length
        sampled_indices = sampled_indices[:n_samples]
        sampled_indices = np.array(sampled_indices)
        
        return X[sampled_indices], y[sampled_indices]

class DiversityAnalyzer:
    """Analyze diversity and performance of bagged ensembles"""
    
    def __init__(self):
        self.diversity_metrics_ = {}
        self.performance_analysis_ = {}
        
    def analyze_ensemble_diversity(self, predictions: List[np.ndarray], 
                                 true_targets: np.ndarray) -> Dict[str, Any]:
        """Comprehensive diversity analysis of ensemble members"""
        
        predictions_array = np.array(predictions)  # Shape: (n_models, n_samples)
        n_models, n_samples = predictions_array.shape
        
        analysis = {
            'basic_metrics': {},
            'pairwise_diversity': {},
            'ensemble_diversity': {},
            'performance_diversity': {}
        }
        
        # Basic metrics
        analysis['basic_metrics'] = {
            'n_models': n_models,
            'n_samples': n_samples,
            'prediction_variance': float(np.var(predictions_array, axis=0).mean()),
            'prediction_std': float(np.std(predictions_array, axis=0).mean())
        }
        
        # Pairwise diversity measures
        pairwise_disagreements = []
        pairwise_correlations = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Disagreement rate
                disagreement = np.mean(predictions_array[i] != predictions_array[j])
                pairwise_disagreements.append(disagreement)
                
                # Correlation
                try:
                    correlation = np.corrcoef(predictions_array[i], predictions_array[j])[0, 1]
                    if not np.isnan(correlation):
                        pairwise_correlations.append(correlation)
                except:
                    pass
        
        analysis['pairwise_diversity'] = {
            'mean_disagreement': float(np.mean(pairwise_disagreements)),
            'std_disagreement': float(np.std(pairwise_disagreements)),
            'max_disagreement': float(np.max(pairwise_disagreements)),
            'min_disagreement': float(np.min(pairwise_disagreements))
        }
        
        if pairwise_correlations:
            analysis['pairwise_diversity'].update({
                'mean_correlation': float(np.mean(pairwise_correlations)),
                'std_correlation': float(np.std(pairwise_correlations)),
                'max_correlation': float(np.max(pairwise_correlations)),
                'min_correlation': float(np.min(pairwise_correlations))
            })
        
        # Ensemble-level diversity
        # Q-statistic and other measures
        q_statistics = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                q_stat = self._calculate_q_statistic(
                    predictions_array[i], predictions_array[j], true_targets
                )
                if not np.isnan(q_stat):
                    q_statistics.append(q_stat)
        
        if q_statistics:
            analysis['ensemble_diversity'] = {
                'mean_q_statistic': float(np.mean(q_statistics)),
                'std_q_statistic': float(np.std(q_statistics)),
                'diversity_score': float(1 - np.mean(np.abs(q_statistics)))  # Higher is more diverse
            }
        
        # Performance diversity
        individual_accuracies = []
        for i in range(n_models):
            if len(np.unique(true_targets)) <= 10:  # Classification
                accuracy = accuracy_score(true_targets, predictions_array[i])
            else:  # Regression
                accuracy = r2_score(true_targets, predictions_array[i])
            individual_accuracies.append(accuracy)
        
        analysis['performance_diversity'] = {
            'individual_accuracies': individual_accuracies,
            'mean_accuracy': float(np.mean(individual_accuracies)),
            'std_accuracy': float(np.std(individual_accuracies)),
            'accuracy_range': float(np.max(individual_accuracies) - np.min(individual_accuracies)),
            'performance_diversity_ratio': float(np.std(individual_accuracies) / np.mean(individual_accuracies))
        }
        
        return analysis
    
    def _calculate_q_statistic(self, pred1: np.ndarray, pred2: np.ndarray, 
                              targets: np.ndarray) -> float:
        """Calculate Q-statistic for measuring classifier diversity"""
        
        # Binary correctness indicators
        correct1 = (pred1 == targets).astype(int)
        correct2 = (pred2 == targets).astype(int)
        
        # Contingency table
        n11 = np.sum((correct1 == 1) & (correct2 == 1))  # Both correct
        n10 = np.sum((correct1 == 1) & (correct2 == 0))  # Only 1 correct
        n01 = np.sum((correct1 == 0) & (correct2 == 1))  # Only 2 correct
        n00 = np.sum((correct1 == 0) & (correct2 == 0))  # Both wrong
        
        # Q-statistic
        numerator = n11 * n00 - n01 * n10
        denominator = n11 * n00 + n01 * n10
        
        if denominator == 0:
            return np.nan
        
        return numerator / denominator
    
    def calculate_ensemble_performance(self, individual_predictions: List[np.ndarray],
                                     ensemble_prediction: np.ndarray,
                                     true_targets: np.ndarray) -> Dict[str, Any]:
        """Calculate ensemble vs individual performance metrics"""
        
        performance = {
            'individual_performance': {},
            'ensemble_performance': {},
            'improvement_analysis': {}
        }
        
        # Individual performance
        individual_scores = []
        for i, pred in enumerate(individual_predictions):
            if len(np.unique(true_targets)) <= 10:  # Classification
                score = accuracy_score(true_targets, pred)
            else:  # Regression
                score = r2_score(true_targets, pred)
            
            individual_scores.append(score)
            performance['individual_performance'][f'model_{i}'] = float(score)
        
        # Ensemble performance
        if len(np.unique(true_targets)) <= 10:  # Classification
            ensemble_score = accuracy_score(true_targets, ensemble_prediction)
            metric_name = 'accuracy'
        else:  # Regression
            ensemble_score = r2_score(true_targets, ensemble_prediction)
            metric_name = 'r2'
        
        performance['ensemble_performance'] = {
            'score': float(ensemble_score),
            'metric': metric_name
        }
        
        # Improvement analysis
        best_individual = max(individual_scores)
        mean_individual = np.mean(individual_scores)
        worst_individual = min(individual_scores)
        
        performance['improvement_analysis'] = {
            'best_individual': float(best_individual),
            'mean_individual': float(mean_individual),
            'worst_individual': float(worst_individual),
            'improvement_over_best': float(ensemble_score - best_individual),
            'improvement_over_mean': float(ensemble_score - mean_individual),
            'improvement_over_worst': float(ensemble_score - worst_individual),
            'relative_improvement': float((ensemble_score - best_individual) / best_individual) if best_individual > 0 else 0.0
        }
        
        return performance

# ============================================
# Main Bagging Ensemble Model
# ============================================

class FinancialBaggingEnsemble(BaseEstimator):
    """
    Advanced bagging ensemble for financial prediction with bootstrap aggregation
    
    Features:
    - Multiple bootstrap strategies (standard, balanced, block, stratified, time-series)
    - Advanced diversity analysis and ensemble diagnostics
    - Financial domain optimizations (time-aware sampling, volatility-based weighting)
    - Out-of-bag error estimation and feature importance aggregation
    - Comprehensive performance analysis and visualization
    """
    
    def __init__(self,
                 name: str = "bagging_ensemble",
                 base_estimator: Optional[Any] = None,
                 n_estimators: int = 100,
                 max_samples: Union[int, float] = 1.0,
                 max_features: Union[int, float] = 1.0,
                 bootstrap: bool = True,
                 bootstrap_features: bool = False,
                 bootstrap_strategy: str = 'standard',
                 block_size: Optional[int] = None,
                 time_aware: bool = True,
                 oob_score: bool = True,
                 warm_start: bool = False,
                 n_jobs: Optional[int] = None,
                 random_state: int = 42,
                 verbose: int = 0,
                 task_type: str = 'classification',
                 **kwargs):
        """
        Initialize Financial Bagging Ensemble
        
        Args:
            name: Model name
            base_estimator: Base estimator to use (if None, uses DecisionTree)
            n_estimators: Number of base estimators in ensemble
            max_samples: Number/fraction of samples to draw for each base estimator
            max_features: Number/fraction of features to draw for each base estimator
            bootstrap: Whether to bootstrap samples
            bootstrap_features: Whether to bootstrap features
            bootstrap_strategy: Bootstrap sampling strategy
            block_size: Block size for block/time-series bootstrap
            time_aware: Whether to use time-aware sampling
            oob_score: Whether to use out-of-bag samples to estimate performance
            warm_start: Whether to reuse solution of previous call to fit
            n_jobs: Number of jobs to run in parallel
            random_state: Random seed
            verbose: Verbosity level
            task_type: Type of task ('classification' or 'regression')
        """
        self.name = name
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.bootstrap_strategy = bootstrap_strategy
        self.block_size = block_size
        self.time_aware = time_aware
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.task_type = task_type
        
        # Fitted components
        self.bagging_ensemble_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.bootstrap_sampler_ = None
        self.diversity_analyzer_ = None
        self.is_fitted_ = False
        
        # Analysis results
        self.oob_score_ = None
        self.feature_importances_ = None
        self.diversity_analysis_ = None
        self.performance_analysis_ = None
        self.bagging_analysis_ = None
        
        logger.info(f"Initialized bagging ensemble: {self.name}")
    
    def _create_base_estimator(self) -> Any:
        """Create default base estimator if none provided"""
        if self.base_estimator is not None:
            return self.base_estimator
        
        # Default to decision tree
        if self.task_type == 'classification':
            return DecisionTreeClassifier(random_state=self.random_state)
        else:
            return DecisionTreeRegressor(random_state=self.random_state)
    
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
    
    def _create_bagging_ensemble(self) -> Union[BaggingClassifier, BaggingRegressor]:
        """Create the bagging ensemble with appropriate configuration"""
        
        base_estimator = self._create_base_estimator()
        
        common_params = {
            'base_estimator': base_estimator,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'bootstrap_features': self.bootstrap_features,
            'oob_score': self.oob_score,
            'warm_start': self.warm_start,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
        
        if self.task_type == 'classification':
            return BaggingClassifier(**common_params)
        else:
            return BaggingRegressor(**common_params)
    
    def _setup_bootstrap_sampler(self):
        """Setup advanced bootstrap sampler if needed"""
        if self.bootstrap_strategy != 'standard':
            self.bootstrap_sampler_ = AdvancedBootstrapSampler(
                strategy=self.bootstrap_strategy,
                block_size=self.block_size,
                time_aware=self.time_aware
            )
    
    def _analyze_bagging_performance(self, X: np.ndarray, y: np.ndarray):
        """Analyze bagging ensemble performance and diversity"""
        
        if not hasattr(self.bagging_ensemble_, 'estimators_'):
            return
        
        # Get individual predictions
        individual_predictions = []
        for estimator in self.bagging_ensemble_.estimators_:
            pred = estimator.predict(X)
            individual_predictions.append(pred)
        
        # Get ensemble prediction
        ensemble_prediction = self.bagging_ensemble_.predict(X)
        
        # Initialize diversity analyzer
        self.diversity_analyzer_ = DiversityAnalyzer()
        
        # Analyze diversity
        self.diversity_analysis_ = self.diversity_analyzer_.analyze_ensemble_diversity(
            individual_predictions, y
        )
        
        # Analyze performance
        self.performance_analysis_ = self.diversity_analyzer_.calculate_ensemble_performance(
            individual_predictions, ensemble_prediction, y
        )
        
        # Aggregate feature importances if available
        self._aggregate_feature_importances()
    
    def _aggregate_feature_importances(self):
        """Aggregate feature importances from base estimators"""
        if not hasattr(self.bagging_ensemble_, 'estimators_'):
            return
        
        importances_list = []
        
        for estimator in self.bagging_ensemble_.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                importances_list.append(estimator.feature_importances_)
        
        if importances_list:
            # Average feature importances
            self.feature_importances_ = np.mean(importances_list, axis=0)
            
            # Calculate standard deviation
            self.feature_importances_std_ = np.std(importances_list, axis=0)
    
    @time_it("bagging_fit", include_args=True)
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], **kwargs):
        """Fit the bagging ensemble"""
        
        logger.info(f"Fitting bagging ensemble with {self.n_estimators} estimators")
        
        try:
            # Preprocess data
            X_processed = self._preprocess_features(X)
            y_processed = self._preprocess_targets(y)
            
            # Setup bootstrap sampler
            self._setup_bootstrap_sampler()
            
            # Create and fit bagging ensemble
            self.bagging_ensemble_ = self._create_bagging_ensemble()
            
            # Use custom bootstrap strategy if specified
            if self.bootstrap_strategy != 'standard' and self.bootstrap_sampler_:
                # Custom fitting with advanced bootstrap
                self._fit_with_custom_bootstrap(X_processed, y_processed)
            else:
                # Standard scikit-learn bagging
                self.bagging_ensemble_.fit(X_processed, y_processed)
            
            # Store OOB score if available
            if self.oob_score and hasattr(self.bagging_ensemble_, 'oob_score_'):
                self.oob_score_ = self.bagging_ensemble_.oob_score_
            
            # Analyze ensemble performance
            self._analyze_bagging_performance(X_processed, y_processed)
            
            # Create comprehensive analysis
            self.bagging_analysis_ = {
                'ensemble_type': 'bagging',
                'n_estimators': self.n_estimators,
                'base_estimator_type': type(self._create_base_estimator()).__name__,
                'bootstrap_strategy': self.bootstrap_strategy,
                'max_samples': self.max_samples,
                'max_features': self.max_features,
                'oob_score': self.oob_score_,
                'diversity_analysis': self.diversity_analysis_,
                'performance_analysis': self.performance_analysis_
            }
            
            self.is_fitted_ = True
            logger.info("Bagging ensemble fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Bagging ensemble fitting failed: {e}")
            raise
    
    def _fit_with_custom_bootstrap(self, X: np.ndarray, y: np.ndarray):
        """Fit ensemble with custom bootstrap strategy"""
        
        # Generate bootstrap samples
        bootstrap_samples = self.bootstrap_sampler_.generate_bootstrap_samples(
            X, y, self.n_estimators, self.random_state
        )
        
        # Manually fit each estimator
        estimators = []
        for i, (sample_X, sample_y) in enumerate(bootstrap_samples):
            estimator = clone(self._create_base_estimator())
            estimator.set_params(random_state=self.random_state + i)
            estimator.fit(sample_X, sample_y)
            estimators.append(estimator)
        
        # Store fitted estimators
        self.bagging_ensemble_.estimators_ = estimators
        self.bagging_ensemble_.n_features_ = X.shape[1]
        
        if self.task_type == 'classification':
            self.bagging_ensemble_.classes_ = self.classes_
            self.bagging_ensemble_.n_classes_ = len(self.classes_)
    
    @time_it("bagging_predict", include_args=True)
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make bagging predictions"""
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            # Make prediction
            predictions = self.bagging_ensemble_.predict(X_processed)
            
            # Decode predictions for classification
            if self.task_type == 'classification':
                if hasattr(self, 'label_encoder_') and self.label_encoder_ is not None:
                    return self.label_encoder_.inverse_transform(predictions.astype(int))
                else:
                    return predictions
            else:
                return predictions
                
        except Exception as e:
            logger.error(f"Bagging prediction failed: {e}")
            raise
    
    @time_it("bagging_predict_proba", include_args=True)
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make bagging probability predictions"""
        
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            # Get probabilities
            probabilities = self.bagging_ensemble_.predict_proba(X_processed)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Bagging probability prediction failed: {e}")
            raise
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get aggregated feature importance"""
        return self.feature_importances_
    
    def get_oob_score(self) -> Optional[float]:
        """Get out-of-bag score"""
        return self.oob_score_
    
    def get_bagging_analysis(self) -> Dict[str, Any]:
        """Get comprehensive bagging analysis"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to get bagging analysis")
        
        return self.bagging_analysis_.copy() if self.bagging_analysis_ else {}
    
    def plot_bagging_analysis(self) -> Any:
        """Plot comprehensive bagging analysis"""
        if not self.bagging_analysis_:
            logger.warning("Bagging analysis not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            analysis = self.bagging_analysis_
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Feature importance plot
            if self.feature_importances_ is not None:
                n_features = len(self.feature_importances_)
                feature_names = [f'Feature_{i}' for i in range(n_features)][:15]  # Top 15
                importances = self.feature_importances_[:15]
                
                bars = axes[0, 0].bar(range(len(importances)), importances, alpha=0.7, color='forestgreen')
                axes[0, 0].set_title('Feature Importance (Top 15)')
                axes[0, 0].set_xticks(range(len(importances)))
                axes[0, 0].set_xticklabels(feature_names, rotation=45, ha='right')
                axes[0, 0].set_ylabel('Importance')
                
                # Add values on bars
                for bar, imp in zip(bars, importances):
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, imp + max(importances) * 0.01,
                                   f'{imp:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                axes[0, 0].text(0.5, 0.5, 'Feature importance\nnot available', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Feature Importance')
            
            # Performance comparison
            if 'performance_analysis' in analysis and analysis['performance_analysis']:
                perf_analysis = analysis['performance_analysis']
                
                if 'individual_performance' in perf_analysis:
                    individual_perf = perf_analysis['individual_performance']
                    n_models = len(individual_perf)
                    individual_scores = list(individual_perf.values())
                    
                    # Show distribution of individual performances
                    axes[0, 1].hist(individual_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    
                    # Add ensemble performance line
                    ensemble_score = perf_analysis['ensemble_performance']['score']
                    axes[0, 1].axvline(ensemble_score, color='red', linestyle='--', linewidth=2,
                                      label=f'Ensemble: {ensemble_score:.3f}')
                    
                    # Add mean individual performance line
                    mean_individual = np.mean(individual_scores)
                    axes[0, 1].axvline(mean_individual, color='blue', linestyle='--', linewidth=2,
                                      label=f'Mean Individual: {mean_individual:.3f}')
                    
                    axes[0, 1].set_title('Performance Distribution')
                    axes[0, 1].set_xlabel('Performance Score')
                    axes[0, 1].set_ylabel('Count')
                    axes[0, 1].legend()
                else:
                    axes[0, 1].text(0.5, 0.5, 'Performance analysis\nnot available', 
                                   ha='center', va='center', transform=axes[0, 1].transAxes)
                    axes[0, 1].set_title('Performance Distribution')
            else:
                axes[0, 1].axis('off')
            
            # Diversity metrics
            if 'diversity_analysis' in analysis and analysis['diversity_analysis']:
                diversity = analysis['diversity_analysis']
                
                if 'pairwise_diversity' in diversity:
                    pairwise = diversity['pairwise_diversity']
                    
                    metrics = ['Mean Disagreement', 'Std Disagreement', 'Max Disagreement']
                    values = [
                        pairwise.get('mean_disagreement', 0),
                        pairwise.get('std_disagreement', 0),
                        pairwise.get('max_disagreement', 0)
                    ]
                    
                    bars = axes[0, 2].bar(metrics, values, alpha=0.7, color='coral')
                    axes[0, 2].set_title('Diversity Metrics')
                    axes[0, 2].set_ylabel('Value')
                    axes[0, 2].tick_params(axis='x', rotation=45)
                    
                    # Add values on bars
                    for bar, value in zip(bars, values):
                        axes[0, 2].text(bar.get_x() + bar.get_width()/2, value + max(values) * 0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                else:
                    axes[0, 2].text(0.5, 0.5, 'Diversity metrics\nnot available', 
                                   ha='center', va='center', transform=axes[0, 2].transAxes)
                    axes[0, 2].set_title('Diversity Metrics')
            else:
                axes[0, 2].axis('off')
            
            # OOB Score evolution (if available)
            if hasattr(self.bagging_ensemble_, 'oob_score_') and self.bagging_ensemble_.oob_score_:
                # For now, just show the final OOB score
                axes[1, 0].bar(['OOB Score'], [self.bagging_ensemble_.oob_score_], 
                              alpha=0.7, color='gold')
                axes[1, 0].set_title('Out-of-Bag Score')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].text(0, self.bagging_ensemble_.oob_score_ + 0.01,
                               f'{self.bagging_ensemble_.oob_score_:.3f}', 
                               ha='center', va='bottom', fontweight='bold')
            else:
                axes[1, 0].text(0.5, 0.5, 'OOB score\nnot available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Out-of-Bag Score')
            
            # Improvement analysis
            if ('performance_analysis' in analysis and analysis['performance_analysis'] and
                'improvement_analysis' in analysis['performance_analysis']):
                improvement = analysis['performance_analysis']['improvement_analysis']
                
                metrics = ['Best Individual', 'Mean Individual', 'Ensemble']
                values = [
                    improvement['best_individual'],
                    improvement['mean_individual'],
                    improvement['best_individual'] + improvement['improvement_over_best']
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
            
            # Bagging summary
            summary_text = f"Ensemble Type: {analysis['ensemble_type'].title()}\n"
            summary_text += f"N Estimators: {analysis['n_estimators']}\n"
            summary_text += f"Base Estimator: {analysis['base_estimator_type']}\n"
            summary_text += f"Bootstrap Strategy: {analysis['bootstrap_strategy'].title()}\n"
            summary_text += f"Max Samples: {analysis['max_samples']}\n"
            summary_text += f"Max Features: {analysis['max_features']}\n"
            
            if analysis.get('oob_score'):
                summary_text += f"\nOOB Score: {analysis['oob_score']:.4f}\n"
            
            if ('performance_analysis' in analysis and analysis['performance_analysis'] and
                'improvement_analysis' in analysis['performance_analysis']):
                improvement = analysis['performance_analysis']['improvement_analysis']
                summary_text += f"\nImprovement: {improvement['improvement_over_best']:+.4f}\n"
                summary_text += f"Relative: {improvement['relative_improvement']:+.1%}"
            
            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 2].set_title('Bagging Summary')
            axes[1, 2].axis('off')
            
            plt.suptitle(f'Bagging Ensemble Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            logger.warning(f"Error creating bagging analysis plot: {e}")
            return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = {
            'model_name': self.name,
            'model_family': 'Bagging Ensemble',
            'task_type': self.task_type,
            'n_estimators': self.n_estimators,
            'base_estimator_type': type(self._create_base_estimator()).__name__,
            'bootstrap_strategy': self.bootstrap_strategy,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'bootstrap_features': self.bootstrap_features,
            'oob_score_enabled': self.oob_score,
            'time_aware': self.time_aware,
            'is_fitted': self.is_fitted_
        }
        
        if self.is_fitted_:
            summary.update({
                'oob_score': self.oob_score_,
                'feature_importances_available': self.feature_importances_ is not None,
                'bagging_analysis': self.bagging_analysis_
            })
            
            if self.feature_importances_ is not None:
                summary['n_features'] = len(self.feature_importances_)
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_bagging_classifier(base_estimator: str = 'decision_tree',
                            n_estimators: int = 100,
                            bootstrap_strategy: str = 'standard',
                            **kwargs) -> FinancialBaggingEnsemble:
    """Create bagging classifier with specified base estimator"""
    
    # Create base estimator
    if base_estimator == 'decision_tree':
        base_est = DecisionTreeClassifier(random_state=42)
    elif base_estimator == 'logistic':
        base_est = create_logistic_classifier(performance_preset='balanced')
    elif base_estimator == 'svm':
        base_est = create_svm_classifier(performance_preset='balanced')
    elif base_estimator == 'naive_bayes':
        base_est = create_naive_bayes_classifier(performance_preset='balanced')
    else:
        base_est = DecisionTreeClassifier(random_state=42)
    
    config = {
        'name': f'bagging_{base_estimator}_{bootstrap_strategy}',
        'base_estimator': base_est,
        'n_estimators': n_estimators,
        'bootstrap_strategy': bootstrap_strategy,
        'task_type': 'classification',
        'random_state': 42
    }
    
    config.update(kwargs)
    
    return FinancialBaggingEnsemble(**config)

def create_bagging_regressor(base_estimator: str = 'decision_tree',
                           n_estimators: int = 100,
                           bootstrap_strategy: str = 'standard',
                           **kwargs) -> FinancialBaggingEnsemble:
    """Create bagging regressor"""
    
    bagging = create_bagging_classifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        bootstrap_strategy=bootstrap_strategy,
        **kwargs
    )
    
    # Convert to regression
    bagging.task_type = 'regression'
    bagging.name = bagging.name.replace('bagging', 'bagging_regressor')
    
    return bagging

def create_random_forest_bagging(**kwargs) -> FinancialBaggingEnsemble:
    """Create Random Forest using bagging framework"""
    return create_bagging_classifier(
        base_estimator='decision_tree',
        bootstrap_features=True,
        name='random_forest_bagging',
        **kwargs
    )

def create_extra_trees_bagging(**kwargs) -> FinancialBaggingEnsemble:
    """Create Extra Trees using bagging framework"""
    return create_bagging_classifier(
        base_estimator='decision_tree',
        bootstrap=False,  # Extra Trees don't bootstrap samples
        bootstrap_features=True,
        name='extra_trees_bagging',
        **kwargs
    )

def create_time_series_bagging(**kwargs) -> FinancialBaggingEnsemble:
    """Create time-series aware bagging for financial data"""
    return create_bagging_classifier(
        bootstrap_strategy='time_series',
        time_aware=True,
        name='time_series_bagging',
        **kwargs
    )

def create_balanced_bagging(**kwargs) -> FinancialBaggingEnsemble:
    """Create balanced bagging for imbalanced datasets"""
    return create_bagging_classifier(
        bootstrap_strategy='balanced',
        name='balanced_bagging',
        **kwargs
    )

def create_block_bagging(block_size: int = 50, **kwargs) -> FinancialBaggingEnsemble:
    """Create block bagging for time series"""
    return create_bagging_classifier(
        bootstrap_strategy='block',
        block_size=block_size,
        name='block_bagging',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def compare_bagging_strategies(X: Union[pd.DataFrame, np.ndarray],
                             y: Union[pd.Series, np.ndarray],
                             strategies: List[str] = ['standard', 'balanced', 'block', 'time_series'],
                             task_type: str = 'classification') -> Dict[str, Any]:
    """Compare different bagging strategies"""
    
    logger.info(f"Comparing bagging strategies: {strategies}")
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"Evaluating {strategy} bagging")
        
        try:
            if task_type == 'classification':
                bagger = create_bagging_classifier(bootstrap_strategy=strategy)
            else:
                bagger = create_bagging_regressor(bootstrap_strategy=strategy)
            
            # Fit and evaluate
            bagger.fit(X, y)
            
            # Get predictions and score
            predictions = bagger.predict(X)
            if task_type == 'classification':
                score = accuracy_score(y, predictions)
            else:
                score = r2_score(y, predictions)
            
            # Get bagging analysis
            analysis = bagger.get_bagging_analysis()
            
            results[strategy] = {
                'score': score,
                'oob_score': bagger.get_oob_score(),
                'bagging_analysis': analysis,
                'model': bagger
            }
            
        except Exception as e:
            logger.warning(f"Error with {strategy} bagging: {e}")
            results[strategy] = {'error': str(e)}
    
    # Add comparison summary
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_strategy = max(valid_results.keys(), key=lambda k: valid_results[k]['score'])
        
        # Best OOB score
        oob_strategies = {k: v for k, v in valid_results.items() if v['oob_score'] is not None}
        best_oob = None
        if oob_strategies:
            best_oob = max(oob_strategies.keys(), key=lambda k: oob_strategies[k]['oob_score'])
        
        results['comparison'] = {
            'best_score': best_strategy,
            'best_oob': best_oob,
            'strategy_rankings': sorted(valid_results.keys(), 
                                      key=lambda k: valid_results[k]['score'], reverse=True)
        }
    
    logger.info(f"Bagging comparison complete. Best strategy: {results.get('comparison', {}).get('best_score', 'Unknown')}")
    
    return results

def analyze_ensemble_diversity(models: List[Any], X: Union[pd.DataFrame, np.ndarray],
                             y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """Analyze diversity among a set of models"""
    
    logger.info(f"Analyzing ensemble diversity for {len(models)} models")
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Fit all models and get predictions
    predictions = []
    model_names = []
    
    for i, model in enumerate(models):
        try:
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            pred = fitted_model.predict(X)
            predictions.append(pred)
            model_names.append(f"{type(model).__name__}_{i}")
        except Exception as e:
            logger.warning(f"Error with model {i}: {e}")
    
    if len(predictions) < 2:
        logger.warning("Need at least 2 models for diversity analysis")
        return {}
    
    # Create diversity analyzer
    analyzer = DiversityAnalyzer()
    analysis = analyzer.analyze_ensemble_diversity(predictions, y)
    
    # Add model information
    analysis['model_names'] = model_names
    analysis['n_models_analyzed'] = len(predictions)
    
    logger.info(f"Diversity analysis complete. Mean disagreement: {analysis['pairwise_diversity']['mean_disagreement']:.4f}")
    
    return analysis

def optimize_bagging_parameters(X: Union[pd.DataFrame, np.ndarray],
                               y: Union[pd.Series, np.ndarray],
                               param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
    """Optimize bagging parameters using grid search"""
    
    logger.info("Optimizing bagging parameters")
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_samples': [0.5, 0.8, 1.0],
            'max_features': [0.5, 0.8, 1.0],
            'bootstrap_strategy': ['standard', 'balanced']
        }
    
    best_score = -np.inf
    best_params = None
    results = {}
    
    # Grid search
    from itertools import product
    
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    for i, param_values in enumerate(param_combinations):
        params = dict(zip(param_names, param_values))
        
        try:
            # Create bagging model
            bagger = FinancialBaggingEnsemble(**params)
            bagger.fit(X, y)
            
            # Evaluate
            predictions = bagger.predict(X)
            if len(np.unique(y)) <= 10:  # Classification
                score = accuracy_score(y, predictions)
            else:  # Regression
                score = r2_score(y, predictions)
            
            results[str(params)] = {
                'params': params,
                'score': score,
                'oob_score': bagger.get_oob_score(),
                'model': bagger
            }
            
            if score > best_score:
                best_score = score
                best_params = params
                
            logger.debug(f"Params {params}: score={score:.4f}")
            
        except Exception as e:
            logger.warning(f"Error with params {params}: {e}")
    
    # Create optimized model
    if best_params:
        optimized_model = FinancialBaggingEnsemble(**best_params)
        optimized_model.fit(X, y)
    else:
        # Fallback
        optimized_model = FinancialBaggingEnsemble()
        optimized_model.fit(X, y)
        best_params = {}
        best_score = accuracy_score(y, optimized_model.predict(X))
    
    optimization_results = {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results,
        'optimized_model': optimized_model,
        'optimization_summary': {
            'n_combinations_tested': len(param_combinations),
            'best_combination': str(best_params),
            'improvement_range': best_score - min([r['score'] for r in results.values() if 'score' in r])
        }
    }
    
    logger.info(f"Parameter optimization complete. Best score: {best_score:.4f}")
    
    return optimization_results
