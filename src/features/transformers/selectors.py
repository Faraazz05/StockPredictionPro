# ============================================
# StockPredictionPro - src/features/transformers/selectors.py
# Advanced feature selection transformers for financial machine learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set
from dataclasses import dataclass, field
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel, RFE, RFECV,
    mutual_info_regression, mutual_info_classif, f_regression, f_classif,
    chi2, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.utils.validation import check_array, check_is_fitted
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('features.transformers.selectors')

# ============================================
# Configuration and Base Classes
# ============================================

@dataclass
class SelectorConfig:
    """Configuration for feature selection transformers"""
    k: Optional[int] = None
    percentile: Optional[float] = None
    threshold: Optional[float] = None
    alpha: float = 0.05
    feature_names: Optional[List[str]] = None
    scoring: str = 'mutual_info'  # 'mutual_info', 'f_test', 'correlation', 'chi2'
    cv: int = 5
    random_state: int = 42
    n_jobs: int = -1
    
    def __post_init__(self):
        if self.k is not None and self.k <= 0:
            raise ValueError("k must be positive")
        if self.percentile is not None and not (0 < self.percentile <= 100):
            raise ValueError("percentile must be between 0 and 100")

class BaseSelector(BaseEstimator, TransformerMixin):
    """Base class for all feature selection transformers"""
    
    def __init__(self, config: Optional[SelectorConfig] = None):
        self.config = config or SelectorConfig()
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.selected_features_ = None
        self.feature_scores_ = None
        self.is_fitted_ = False
    
    def _validate_input(self, X, y=None):
        """Validate input data"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            if not self.is_fitted_ and self.feature_names_in_ is None:
                self.feature_names_in_ = X.columns.tolist()
        else:
            X_array = check_array(X, accept_sparse=False, dtype=np.float64)
        
        if not self.is_fitted_:
            self.n_features_in_ = X_array.shape[1]
            if self.config.feature_names:
                if len(self.config.feature_names) != X_array.shape[1]:
                    raise ValueError("Length of feature_names must match number of features")
                self.feature_names_in_ = self.config.feature_names
        else:
            if X_array.shape[1] != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} features, got {X_array.shape[1]}")
        
        if y is not None:
            y = np.asarray(y)
            if len(y) != X_array.shape[0]:
                raise ValueError("X and y must have the same number of samples")
        
        return X_array, y
    
    def _get_feature_names(self):
        """Get feature names for input features"""
        if self.feature_names_in_:
            return self.feature_names_in_
        else:
            return [f'feature_{i}' for i in range(self.n_features_in_)]
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        check_is_fitted(self, 'is_fitted_')
        
        if input_features is None:
            input_features = self._get_feature_names()
        
        if self.selected_features_ is not None:
            return [input_features[i] for i in self.selected_features_]
        else:
            return input_features
    
    def get_support(self, indices=False):
        """Get mask or indices of selected features"""
        check_is_fitted(self, 'is_fitted_')
        
        if self.selected_features_ is None:
            mask = np.ones(self.n_features_in_, dtype=bool)
        else:
            mask = np.zeros(self.n_features_in_, dtype=bool)
            mask[self.selected_features_] = True
        
        return self.selected_features_ if indices else mask

# ============================================
# Statistical Feature Selection
# ============================================

class StatisticalSelector(BaseSelector):
    """
    Statistical feature selection using various statistical tests.
    Supports mutual information, F-test, correlation, and chi-square tests.
    """
    
    def __init__(self, config: Optional[SelectorConfig] = None):
        super().__init__(config)
        self.selector_ = None
    
    def fit(self, X, y):
        """Fit the statistical selector"""
        X, y = self._validate_input(X, y)
        
        if y is None:
            raise ValueError("Statistical selection requires target variable y")
        
        # Determine problem type
        is_classification = len(np.unique(y)) < 20  # Heuristic
        
        # Select appropriate scoring function
        if self.config.scoring == 'mutual_info':
            if is_classification:
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
        elif self.config.scoring == 'f_test':
            if is_classification:
                score_func = f_classif
            else:
                score_func = f_regression
        elif self.config.scoring == 'chi2':
            if not is_classification:
                logger.warning("Chi-square test typically used for classification, but target appears continuous")
            score_func = chi2
            # Ensure non-negative features for chi2
            X = np.abs(X)
        elif self.config.scoring == 'correlation':
            score_func = self._correlation_score
        else:
            raise ValueError(f"Unknown scoring method: {self.config.scoring}")
        
        # Create selector
        if self.config.k is not None:
            self.selector_ = SelectKBest(score_func=score_func, k=self.config.k)
        elif self.config.percentile is not None:
            self.selector_ = SelectPercentile(score_func=score_func, percentile=self.config.percentile)
        else:
            # Default to top 50% of features
            self.selector_ = SelectPercentile(score_func=score_func, percentile=50)
        
        # Fit selector
        self.selector_.fit(X, y)
        
        # Store results
        self.selected_features_ = self.selector_.get_support(indices=True)
        self.feature_scores_ = self.selector_.scores_
        
        self.is_fitted_ = True
        return self
    
    def _correlation_score(self, X, y):
        """Calculate correlation scores for feature selection"""
        scores = np.zeros(X.shape[1])
        p_values = np.ones(X.shape[1])
        
        for i in range(X.shape[1]):
            try:
                # Use Pearson correlation by default
                corr, p_val = pearsonr(X[:, i], y)
                scores[i] = abs(corr)  # Use absolute correlation
                p_values[i] = p_val
            except:
                scores[i] = 0
                p_values[i] = 1
        
        return scores, p_values
    
    def transform(self, X):
        """Transform data by selecting features"""
        check_is_fitted(self, 'is_fitted_')
        X, _ = self._validate_input(X)
        
        return self.selector_.transform(X)

class VarianceSelector(BaseSelector):
    """
    Variance-based feature selection.
    Removes features with low variance (quasi-constant features).
    """
    
    def __init__(self, 
                 threshold: float = 0.0,
                 config: Optional[SelectorConfig] = None):
        super().__init__(config)
        self.threshold = threshold
        self.selector_ = None
    
    def fit(self, X, y=None):
        """Fit the variance selector"""
        X, _ = self._validate_input(X, y)
        
        self.selector_ = VarianceThreshold(threshold=self.threshold)
        self.selector_.fit(X)
        
        # Store results
        self.selected_features_ = self.selector_.get_support(indices=True)
        self.feature_scores_ = self.selector_.variances_
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data by removing low variance features"""
        check_is_fitted(self, 'is_fitted_')
        X, _ = self._validate_input(X)
        
        return self.selector_.transform(X)

# ============================================
# Model-Based Feature Selection
# ============================================

class ModelBasedSelector(BaseSelector):
    """
    Model-based feature selection using feature importance from ML models.
    Supports various models like Random Forest, Lasso, etc.
    """
    
    def __init__(self, 
                 estimator: Optional[Any] = None,
                 threshold: Optional[Union[str, float]] = None,
                 config: Optional[SelectorConfig] = None):
        super().__init__(config)
        self.estimator = estimator
        self.threshold = threshold or 'mean'
        self.selector_ = None
    
    def fit(self, X, y):
        """Fit the model-based selector"""
        X, y = self._validate_input(X, y)
        
        if y is None:
            raise ValueError("Model-based selection requires target variable y")
        
        # Use default estimator if none provided
        if self.estimator is None:
            # Determine problem type
            is_classification = len(np.unique(y)) < 20
            
            if is_classification:
                self.estimator = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
            else:
                self.estimator = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
        
        # Create selector
        self.selector_ = SelectFromModel(
            estimator=self.estimator,
            threshold=self.threshold
        )
        
        # Fit selector
        self.selector_.fit(X, y)
        
        # Store results
        self.selected_features_ = self.selector_.get_support(indices=True)
        
        # Get feature importances
        if hasattr(self.selector_.estimator_, 'feature_importances_'):
            self.feature_scores_ = self.selector_.estimator_.feature_importances_
        elif hasattr(self.selector_.estimator_, 'coef_'):
            self.feature_scores_ = np.abs(self.selector_.estimator_.coef_)
        else:
            self.feature_scores_ = None
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data by selecting important features"""
        check_is_fitted(self, 'is_fitted_')
        X, _ = self._validate_input(X)
        
        return self.selector_.transform(X)

class RecursiveFeatureElimination(BaseSelector):
    """
    Recursive Feature Elimination (RFE) selector.
    Recursively eliminates features based on model performance.
    """
    
    def __init__(self, 
                 estimator: Optional[Any] = None,
                 n_features_to_select: Optional[int] = None,
                 step: Union[int, float] = 1,
                 with_cv: bool = True,
                 config: Optional[SelectorConfig] = None):
        super().__init__(config)
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.with_cv = with_cv
        self.selector_ = None
    
    def fit(self, X, y):
        """Fit the RFE selector"""
        X, y = self._validate_input(X, y)
        
        if y is None:
            raise ValueError("RFE requires target variable y")
        
        # Use default estimator if none provided
        if self.estimator is None:
            # Determine problem type
            is_classification = len(np.unique(y)) < 20
            
            if is_classification:
                self.estimator = RandomForestClassifier(
                    n_estimators=50,  # Fewer trees for speed in RFE
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
            else:
                self.estimator = RandomForestRegressor(
                    n_estimators=50,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
        
        # Create RFE selector
        if self.with_cv:
            self.selector_ = RFECV(
                estimator=self.estimator,
                step=self.step,
                cv=self.config.cv,
                scoring=None,  # Use default scoring
                n_jobs=self.config.n_jobs
            )
        else:
            self.selector_ = RFE(
                estimator=self.estimator,
                n_features_to_select=self.n_features_to_select,
                step=self.step
            )
        
        # Fit selector
        self.selector_.fit(X, y)
        
        # Store results
        self.selected_features_ = self.selector_.get_support(indices=True)
        
        # Feature rankings (1 = selected, higher = eliminated earlier)
        self.feature_scores_ = 1.0 / self.selector_.ranking_  # Convert ranking to importance-like score
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data by selecting features via RFE"""
        check_is_fitted(self, 'is_fitted_')
        X, _ = self._validate_input(X)
        
        return self.selector_.transform(X)

# ============================================
# Financial-Specific Selectors
# ============================================

class FinancialCorrelationSelector(BaseSelector):
    """
    Financial correlation-based feature selector.
    Removes highly correlated features while preserving diversity.
    """
    
    def __init__(self, 
                 correlation_threshold: float = 0.95,
                 method: str = 'pearson',  # 'pearson', 'spearman', 'kendall'
                 keep_policy: str = 'first',  # 'first', 'last', 'target_corr'
                 config: Optional[SelectorConfig] = None):
        super().__init__(config)
        self.correlation_threshold = correlation_threshold
        self.method = method
        self.keep_policy = keep_policy
        self.correlation_matrix_ = None
        self.removed_features_ = None
    
    def fit(self, X, y=None):
        """Fit the correlation selector"""
        X, y = self._validate_input(X, y)
        
        # Calculate correlation matrix
        if self.method == 'pearson':
            self.correlation_matrix_ = np.corrcoef(X.T)
        elif self.method == 'spearman':
            self.correlation_matrix_ = np.array([[spearmanr(X[:, i], X[:, j])[0] 
                                                for j in range(X.shape[1])] 
                                               for i in range(X.shape[1])])
        else:
            raise ValueError(f"Unknown correlation method: {self.method}")
        
        # Handle NaN values in correlation matrix
        self.correlation_matrix_ = np.nan_to_num(self.correlation_matrix_, nan=0.0)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        n_features = X.shape[1]
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(self.correlation_matrix_[i, j]) > self.correlation_threshold:
                    high_corr_pairs.append((i, j, abs(self.correlation_matrix_[i, j])))
        
        # Determine which features to remove
        features_to_remove = set()
        
        for i, j, corr in high_corr_pairs:
            if i in features_to_remove or j in features_to_remove:
                continue  # One of the pair already removed
            
            if self.keep_policy == 'first':
                features_to_remove.add(j)  # Remove the later feature
            elif self.keep_policy == 'last':
                features_to_remove.add(i)  # Remove the earlier feature
            elif self.keep_policy == 'target_corr' and y is not None:
                # Keep the feature more correlated with target
                target_corr_i = abs(np.corrcoef(X[:, i], y)[0, 1])
                target_corr_j = abs(np.corrcoef(X[:, j], y)[0, 1])
                
                if np.isnan(target_corr_i):
                    target_corr_i = 0
                if np.isnan(target_corr_j):
                    target_corr_j = 0
                
                if target_corr_i >= target_corr_j:
                    features_to_remove.add(j)
                else:
                    features_to_remove.add(i)
            else:
                features_to_remove.add(j)  # Default to removing later feature
        
        # Store results
        self.removed_features_ = sorted(list(features_to_remove))
        self.selected_features_ = [i for i in range(n_features) if i not in features_to_remove]
        
        # Feature scores (1 - max correlation with other features)
        self.feature_scores_ = np.zeros(n_features)
        for i in range(n_features):
            max_corr = 0
            for j in range(n_features):
                if i != j:
                    max_corr = max(max_corr, abs(self.correlation_matrix_[i, j]))
            self.feature_scores_[i] = 1 - max_corr
        
        logger.info(f"Removed {len(features_to_remove)} highly correlated features")
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data by removing correlated features"""
        check_is_fitted(self, 'is_fitted_')
        X, _ = self._validate_input(X)
        
        return X[:, self.selected_features_]

class FinancialStabilitySelector(BaseSelector):
    """
    Financial stability-based feature selector.
    Selects features based on stability over time (rolling correlations with target).
    """
    
    def __init__(self, 
                 window_size: int = 252,  # 1 year
                 min_stability: float = 0.1,
                 stability_metric: str = 'correlation_consistency',
                 config: Optional[SelectorConfig] = None):
        super().__init__(config)
        self.window_size = window_size
        self.min_stability = min_stability
        self.stability_metric = stability_metric
        self.stability_scores_ = None
    
    def fit(self, X, y):
        """Fit the stability selector"""
        X, y = self._validate_input(X, y)
        
        if y is None:
            raise ValueError("Stability selection requires target variable y")
        
        n_samples, n_features = X.shape
        
        if n_samples < self.window_size * 2:
            logger.warning(f"Insufficient data for stability analysis. Need at least {self.window_size * 2} samples")
            # Fall back to correlation-based selection
            correlations = [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(n_features)]
            self.stability_scores_ = np.array(correlations)
        else:
            self.stability_scores_ = self._calculate_stability_scores(X, y)
        
        # Select features above stability threshold
        stable_features = np.where(self.stability_scores_ >= self.min_stability)[0]
        
        if len(stable_features) == 0:
            logger.warning("No features meet stability criteria, selecting top 50%")
            n_select = max(1, n_features // 2)
            stable_features = np.argsort(self.stability_scores_)[-n_select:]
        
        self.selected_features_ = stable_features
        self.feature_scores_ = self.stability_scores_
        
        self.is_fitted_ = True
        return self
    
    def _calculate_stability_scores(self, X, y):
        """Calculate stability scores for features"""
        n_samples, n_features = X.shape
        stability_scores = np.zeros(n_features)
        
        for feature_idx in range(n_features):
            rolling_correlations = []
            
            # Calculate rolling correlations
            for start_idx in range(0, n_samples - self.window_size + 1, self.window_size // 4):
                end_idx = start_idx + self.window_size
                
                X_window = X[start_idx:end_idx, feature_idx]
                y_window = y[start_idx:end_idx]
                
                try:
                    corr, _ = pearsonr(X_window, y_window)
                    if not np.isnan(corr):
                        rolling_correlations.append(corr)
                except:
                    continue
            
            if len(rolling_correlations) < 2:
                stability_scores[feature_idx] = 0
                continue
            
            # Calculate stability metric
            if self.stability_metric == 'correlation_consistency':
                # Consistency of correlation sign and magnitude
                correlations = np.array(rolling_correlations)
                
                # Penalize sign changes
                sign_consistency = np.mean(np.sign(correlations) == np.sign(correlations[0]))
                
                # Reward consistent magnitude
                magnitude_consistency = 1 - np.std(np.abs(correlations)) / (np.mean(np.abs(correlations)) + 1e-8)
                
                stability_scores[feature_idx] = sign_consistency * magnitude_consistency * np.mean(np.abs(correlations))
            
            elif self.stability_metric == 'correlation_std':
                # Lower standard deviation = more stable
                stability_scores[feature_idx] = 1 / (1 + np.std(rolling_correlations))
            
            else:
                raise ValueError(f"Unknown stability metric: {self.stability_metric}")
        
        return stability_scores
    
    def transform(self, X):
        """Transform data by selecting stable features"""
        check_is_fitted(self, 'is_fitted_')
        X, _ = self._validate_input(X)
        
        return X[:, self.selected_features_]

# ============================================
# Ensemble Feature Selection
# ============================================

class EnsembleSelector(BaseSelector):
    """
    Ensemble feature selector that combines multiple selection methods.
    Uses voting or ranking aggregation to make final selections.
    """
    
    def __init__(self, 
                 selectors: List[Tuple[str, BaseSelector]],
                 voting_strategy: str = 'majority',  # 'majority', 'unanimous', 'weighted'
                 weights: Optional[List[float]] = None,
                 config: Optional[SelectorConfig] = None):
        super().__init__(config)
        self.selectors = selectors
        self.voting_strategy = voting_strategy
        self.weights = weights
        self.fitted_selectors_ = []
        self.selection_votes_ = None
    
    def fit(self, X, y=None):
        """Fit the ensemble selector"""
        X, y = self._validate_input(X, y)
        
        n_features = X.shape[1]
        self.fitted_selectors_ = []
        all_selections = []
        
        # Fit each selector
        for name, selector in self.selectors:
            try:
                fitted_selector = selector.fit(X, y)
                self.fitted_selectors_.append((name, fitted_selector))
                
                # Get selected features
                selected = fitted_selector.get_support(indices=True)
                selection_mask = np.zeros(n_features, dtype=bool)
                selection_mask[selected] = True
                all_selections.append(selection_mask)
                
            except Exception as e:
                logger.warning(f"Selector {name} failed: {e}")
                continue
        
        if not all_selections:
            raise RuntimeError("No selectors were successfully fitted")
        
        # Aggregate selections
        all_selections = np.array(all_selections)
        self.selection_votes_ = np.sum(all_selections, axis=0)
        
        # Apply voting strategy
        if self.voting_strategy == 'majority':
            # Select features chosen by majority of selectors
            threshold = len(all_selections) / 2
            selected_mask = self.selection_votes_ > threshold
        
        elif self.voting_strategy == 'unanimous':
            # Select features chosen by all selectors
            selected_mask = self.selection_votes_ == len(all_selections)
        
        elif self.voting_strategy == 'weighted':
            # Weighted voting
            if self.weights is None:
                self.weights = [1.0] * len(all_selections)
            
            weighted_votes = np.sum(all_selections * np.array(self.weights)[:, np.newaxis], axis=0)
            threshold = sum(self.weights) / 2
            selected_mask = weighted_votes > threshold
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        
        # Ensure at least one feature is selected
        if not np.any(selected_mask):
            logger.warning("No features selected by ensemble, selecting most voted features")
            n_select = max(1, min(10, n_features // 4))  # Select up to 25% or 10 features
            top_voted = np.argsort(self.selection_votes_)[-n_select:]
            selected_mask = np.zeros(n_features, dtype=bool)
            selected_mask[top_voted] = True
        
        self.selected_features_ = np.where(selected_mask)[0]
        self.feature_scores_ = self.selection_votes_.astype(float) / len(all_selections)
        
        logger.info(f"Ensemble selected {len(self.selected_features_)} features from {n_features}")
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data using ensemble selection"""
        check_is_fitted(self, 'is_fitted_')
        X, _ = self._validate_input(X)
        
        return X[:, self.selected_features_]
    
    def get_selector_results(self):
        """Get results from individual selectors"""
        check_is_fitted(self, 'is_fitted_')
        
        results = {}
        for name, selector in self.fitted_selectors_:
            results[name] = {
                'selected_features': selector.get_support(indices=True),
                'n_selected': len(selector.get_support(indices=True)),
                'feature_scores': getattr(selector, 'feature_scores_', None)
            }
        
        return results

# ============================================
# Composite and Adaptive Selectors
# ============================================

class AdaptiveSelector(BaseSelector):
    """
    Adaptive feature selector that chooses selection method based on data characteristics.
    """
    
    def __init__(self, 
                 auto_select_method: bool = True,
                 fallback_method: str = 'statistical',
                 config: Optional[SelectorConfig] = None):
        super().__init__(config)
        self.auto_select_method = auto_select_method
        self.fallback_method = fallback_method
        self.selected_method_ = None
        self.fitted_selector_ = None
    
    def fit(self, X, y=None):
        """Fit the adaptive selector"""
        X, y = self._validate_input(X, y)
        
        n_samples, n_features = X.shape
        
        # Choose selection method based on data characteristics
        if self.auto_select_method:
            self.selected_method_ = self._choose_selection_method(X, y, n_samples, n_features)
        else:
            self.selected_method_ = self.fallback_method
        
        # Create and fit the selected selector
        selector = self._create_selector(self.selected_method_)
        
        try:
            self.fitted_selector_ = selector.fit(X, y)
        except Exception as e:
            logger.warning(f"Selected method {self.selected_method_} failed: {e}, falling back to {self.fallback_method}")
            selector = self._create_selector(self.fallback_method)
            self.fitted_selector_ = selector.fit(X, y)
            self.selected_method_ = self.fallback_method
        
        # Store results
        self.selected_features_ = self.fitted_selector_.get_support(indices=True)
        self.feature_scores_ = getattr(self.fitted_selector_, 'feature_scores_', None)
        
        self.is_fitted_ = True
        return self
    
    def _choose_selection_method(self, X, y, n_samples, n_features):
        """Choose appropriate selection method based on data characteristics"""
        
        # High-dimensional data
        if n_features > n_samples:
            return 'variance'  # Start with variance filtering
        
        # Many features relative to samples
        elif n_features > n_samples * 0.5:
            return 'model_based'  # Use model-based selection
        
        # Time series data (if y is provided and looks time-dependent)
        elif y is not None and n_samples > 100:
            # Simple check for time dependence
            autocorr = np.corrcoef(y[:-1], y[1:])[0, 1]
            if not np.isnan(autocorr) and abs(autocorr) > 0.3:
                return 'stability'  # Use stability-based selection
        
        # Default to statistical selection
        return 'statistical'
    
    def _create_selector(self, method):
        """Create selector instance based on method"""
        if method == 'statistical':
            return StatisticalSelector(self.config)
        elif method == 'model_based':
            return ModelBasedSelector(config=self.config)
        elif method == 'variance':
            return VarianceSelector(config=self.config)
        elif method == 'correlation':
            return FinancialCorrelationSelector(config=self.config)
        elif method == 'stability':
            return FinancialStabilitySelector(config=self.config)
        elif method == 'rfe':
            return RecursiveFeatureElimination(config=self.config)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def transform(self, X):
        """Transform data using adaptive selection"""
        check_is_fitted(self, 'is_fitted_')
        X, _ = self._validate_input(X)
        
        return self.fitted_selector_.transform(X)
    
    def get_selected_method(self):
        """Get the selected method"""
        check_is_fitted(self, 'is_fitted_')
        return self.selected_method_

# ============================================
# Utility Functions
# ============================================

@time_it("feature_selection")
def create_financial_selector(X: Union[pd.DataFrame, np.ndarray],
                             y: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             method: str = 'adaptive',
                             target_features: Optional[int] = None,
                             **kwargs) -> BaseSelector:
    """
    Create appropriate feature selector for financial data
    
    Args:
        X: Input feature matrix
        y: Target variable
        feature_names: Names of features
        method: Selection method ('statistical', 'model_based', 'correlation', 'adaptive', etc.)
        target_features: Target number of features to select
        **kwargs: Additional arguments for selector
        
    Returns:
        Fitted selector instance
    """
    
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or X.columns.tolist()
        X = X.values
    
    # Determine k or percentile based on target_features
    config_args = kwargs.copy()
    if target_features is not None:
        if target_features < 1:
            # Interpret as percentile
            config_args['percentile'] = target_features * 100
        else:
            # Interpret as number of features
            config_args['k'] = min(target_features, X.shape[1])
    
    config = SelectorConfig(
        feature_names=feature_names,
        **{k: v for k, v in config_args.items() if k in SelectorConfig.__dataclass_fields__}
    )
    
    if method == 'statistical':
        selector = StatisticalSelector(config=config)
    elif method == 'model_based':
        selector = ModelBasedSelector(config=config, **kwargs)
    elif method == 'variance':
        selector = VarianceSelector(config=config, **kwargs)
    elif method == 'correlation':
        selector = FinancialCorrelationSelector(config=config, **kwargs)
    elif method == 'stability':
        selector = FinancialStabilitySelector(config=config, **kwargs)
    elif method == 'rfe':
        selector = RecursiveFeatureElimination(config=config, **kwargs)
    elif method == 'ensemble':
        # Create ensemble with multiple methods
        methods = [
            ('statistical', StatisticalSelector(config)),
            ('model_based', ModelBasedSelector(config=config)),
            ('correlation', FinancialCorrelationSelector(config=config))
        ]
        selector = EnsembleSelector(methods, config=config, **kwargs)
    elif method == 'adaptive':
        selector = AdaptiveSelector(config=config, **kwargs)
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    selector.fit(X, y)
    logger.info(f"Created and fitted {method} selector, selected {len(selector.selected_features_)} features")
    return selector

def analyze_feature_importance(X: Union[pd.DataFrame, np.ndarray],
                              y: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              methods: List[str] = None) -> pd.DataFrame:
    """Analyze feature importance using multiple methods"""
    
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or X.columns.tolist()
        X = X.values
    elif feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    methods = methods or ['statistical', 'model_based', 'correlation']
    importance_results = {'feature': feature_names}
    
    for method in methods:
        try:
            selector = create_financial_selector(X, y, feature_names, method=method)
            
            if hasattr(selector, 'feature_scores_') and selector.feature_scores_ is not None:
                importance_results[f'{method}_score'] = selector.feature_scores_
            
            # Add selection indicator
            selected_mask = selector.get_support()
            importance_results[f'{method}_selected'] = selected_mask
            
        except Exception as e:
            logger.warning(f"Error analyzing with {method}: {e}")
            importance_results[f'{method}_score'] = [0] * len(feature_names)
            importance_results[f'{method}_selected'] = [False] * len(feature_names)
    
    # Calculate aggregate scores
    score_columns = [col for col in importance_results.keys() if 'score' in col]
    if score_columns:
        scores_df = pd.DataFrame({col: importance_results[col] for col in score_columns})
        importance_results['avg_score'] = scores_df.mean(axis=1)
        importance_results['score_std'] = scores_df.std(axis=1)
    
    # Calculate selection frequency
    selected_columns = [col for col in importance_results.keys() if 'selected' in col]
    if selected_columns:
        selected_df = pd.DataFrame({col: importance_results[col] for col in selected_columns})
        importance_results['selection_frequency'] = selected_df.sum(axis=1) / len(selected_columns)
    
    return pd.DataFrame(importance_results)

def select_diverse_features(X: Union[pd.DataFrame, np.ndarray],
                           y: np.ndarray,
                           n_features: int,
                           diversity_weight: float = 0.3) -> Tuple[np.ndarray, List[int]]:
    """
    Select diverse features balancing importance and diversity
    
    Args:
        X: Feature matrix
        y: Target variable
        n_features: Number of features to select
        diversity_weight: Weight for diversity vs importance (0-1)
        
    Returns:
        Selected features and their indices
    """
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Get feature importance
    importance_analysis = analyze_feature_importance(X, y)
    
    if 'avg_score' not in importance_analysis.columns:
        # Fallback to simple correlation
        correlations = [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
        importance_scores = np.array(correlations)
    else:
        importance_scores = importance_analysis['avg_score'].values
    
    # Calculate correlation matrix for diversity
    corr_matrix = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(corr_matrix, 0)  # Remove self-correlations
    
    # Greedy selection balancing importance and diversity
    selected_features = []
    remaining_features = list(range(X.shape[1]))
    
    # Start with most important feature
    best_feature = np.argmax(importance_scores)
    selected_features.append(best_feature)
    remaining_features.remove(best_feature)
    
    # Select remaining features
    for _ in range(min(n_features - 1, len(remaining_features))):
        best_score = -np.inf
        best_feature = None
        
        for feature_idx in remaining_features:
            # Importance component
            importance_component = importance_scores[feature_idx]
            
            # Diversity component (negative correlation with selected features)
            if selected_features:
                max_corr_with_selected = np.max([corr_matrix[feature_idx, sel_idx] 
                                               for sel_idx in selected_features])
                diversity_component = 1 - max_corr_with_selected
            else:
                diversity_component = 1
            
            # Combined score
            combined_score = ((1 - diversity_weight) * importance_component + 
                            diversity_weight * diversity_component)
            
            if combined_score > best_score:
                best_score = combined_score
                best_feature = feature_idx
        
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
    
    return X[:, selected_features], selected_features

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Feature Selection Transformers")
    
    # Create sample financial data
    np.random.seed(42)
    n_samples, n_features = 1000, 50
    
    # Generate features with different characteristics
    # Important features
    X_important = np.random.normal(0, 1, (n_samples, 5))
    
    # Noise features
    X_noise = np.random.normal(0, 1, (n_samples, 20))
    
    # Correlated features
    X_corr_base = np.random.normal(0, 1, (n_samples, 5))
    X_correlated = np.column_stack([X_corr_base, 
                                   X_corr_base + np.random.normal(0, 0.1, (n_samples, 5)),
                                   X_corr_base * 2 + np.random.normal(0, 0.2, (n_samples, 5))])
    
    # Low variance features
    X_low_var = np.random.normal(0, 0.01, (n_samples, 10))
    
    # Combine all features
    X = np.column_stack([X_important, X_noise, X_correlated, X_low_var])
    
    # Create target that depends on important features
    y = (X_important[:, 0] * 2 + X_important[:, 1] * 1.5 - X_important[:, 2] * 1 + 
         np.random.normal(0, 0.5, n_samples))
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    print(f"Original data shape: {X.shape}")
    print(f"Target correlation with first 5 features: {[np.corrcoef(X[:, i], y)[0, 1] for i in range(5)]}")
    
    # Test statistical selector
    print("\n1. Testing Statistical Selector")
    config = SelectorConfig(k=15, scoring='mutual_info')
    stat_selector = StatisticalSelector(config)
    X_stat = stat_selector.fit_transform(X, y)
    stat_names = stat_selector.get_feature_names_out(feature_names)
    print(f"Statistical selection: {X.shape[1]} → {X_stat.shape[1]}")
    print(f"Selected features: {stat_names[:10]}...")
    
    # Test model-based selector
    print("\n2. Testing Model-Based Selector")
    model_selector = ModelBasedSelector(threshold='median', config=config)
    X_model = model_selector.fit_transform(X, y)
    model_names = model_selector.get_feature_names_out(feature_names)
    print(f"Model-based selection: {X.shape[1]} → {X_model.shape[1]}")
    print(f"Feature importances (top 5): {model_selector.feature_scores_[:5]}")
    
    # Test correlation selector
    print("\n3. Testing Correlation Selector")
    corr_selector = FinancialCorrelationSelector(
        correlation_threshold=0.8,
        keep_policy='target_corr'
    )
    X_corr = corr_selector.fit_transform(X, y)
    corr_names = corr_selector.get_feature_names_out(feature_names)
    print(f"Correlation filtering: {X.shape[1]} → {X_corr.shape[1]}")
    print(f"Removed {len(corr_selector.removed_features_)} highly correlated features")
    
    # Test ensemble selector
    print("\n4. Testing Ensemble Selector")
    ensemble_selectors = [
        ('statistical', StatisticalSelector(SelectorConfig(percentile=30))),
        ('model_based', ModelBasedSelector(config=SelectorConfig())),
        ('correlation', FinancialCorrelationSelector())
    ]
    
    ensemble_selector = EnsembleSelector(
        ensemble_selectors,
        voting_strategy='majority'
    )
    X_ensemble = ensemble_selector.fit_transform(X, y)
    ensemble_names = ensemble_selector.get_feature_names_out(feature_names)
    print(f"Ensemble selection: {X.shape[1]} → {X_ensemble.shape[1]}")
    
    # Show individual selector results
    selector_results = ensemble_selector.get_selector_results()
    for name, results in selector_results.items():
        print(f"  {name}: {results['n_selected']} features")
    
    # Test adaptive selector
    print("\n5. Testing Adaptive Selector")
    adaptive_selector = AdaptiveSelector(auto_select_method=True)
    X_adaptive = adaptive_selector.fit_transform(X, y)
    adaptive_names = adaptive_selector.get_feature_names_out(feature_names)
    print(f"Adaptive selection: {X.shape[1]} → {X_adaptive.shape[1]}")
    print(f"Selected method: {adaptive_selector.get_selected_method()}")
    
    # Test utility functions
    print("\n6. Testing Utility Functions")
    
    # Feature importance analysis
    importance_df = analyze_feature_importance(X, y, feature_names, 
                                             methods=['statistical', 'model_based'])
    print("Top 10 most important features:")
    top_features = importance_df.nlargest(10, 'avg_score')[['feature', 'avg_score', 'selection_frequency']]
    print(top_features)
    
    # Diverse feature selection
    X_diverse, diverse_indices = select_diverse_features(X, y, n_features=20, diversity_weight=0.3)
    diverse_names = [feature_names[i] for i in diverse_indices]
    print(f"\nDiverse selection: {X.shape[1]} → {X_diverse.shape[1]}")
    print(f"Selected diverse features: {diverse_names[:10]}...")
    
    # Auto selector creation
    print("\n7. Testing Auto Selector Creation")
    auto_selector = create_financial_selector(X, y, feature_names, method='ensemble', target_features=25)
    X_auto = auto_selector.transform(X)
    print(f"Auto selector: {X.shape[1]} → {X_auto.shape[1]}")
    
    print("\nFeature selection transformers testing completed successfully!")
