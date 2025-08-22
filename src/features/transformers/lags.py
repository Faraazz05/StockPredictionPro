# ============================================
# StockPredictionPro - src/features/transformers/lags.py
# Advanced lag feature transformations for time series financial machine learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('features.transformers.lags')

# ============================================
# Configuration and Base Classes
# ============================================

@dataclass
class LagConfig:
    """Configuration for lag transformers"""
    max_lag: int = 5
    min_lag: int = 1
    lag_step: int = 1
    include_original: bool = True
    feature_names: Optional[List[str]] = None
    fill_method: str = 'forward'  # 'forward', 'backward', 'zero', 'drop'
    seasonal_lags: Optional[List[int]] = None
    auto_detect_seasonality: bool = False
    max_features: Optional[int] = None
    correlation_threshold: Optional[float] = None
    
    def __post_init__(self):
        if self.max_lag < self.min_lag:
            raise ValueError("max_lag must be >= min_lag")
        if self.lag_step < 1:
            raise ValueError("lag_step must be at least 1")
        if self.seasonal_lags is None:
            self.seasonal_lags = []

class BaseLagTransformer(BaseEstimator, TransformerMixin):
    """Base class for all lag transformers"""
    
    def __init__(self, config: Optional[LagConfig] = None):
        self.config = config or LagConfig()
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.lag_names_ = None
        self.selected_lags_ = None
        self.is_fitted_ = False
    
    def _validate_input(self, X):
        """Validate input data"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = check_array(X, accept_sparse=False, dtype=np.float64)
        
        if not self.is_fitted_:
            self.n_features_in_ = X_array.shape[1]
            if isinstance(X, pd.DataFrame):
                self.feature_names_in_ = X.columns.tolist()
            elif self.config.feature_names:
                if len(self.config.feature_names) != X_array.shape[1]:
                    raise ValueError("Length of feature_names must match number of features")
                self.feature_names_in_ = self.config.feature_names
        else:
            if X_array.shape[1] != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} features, got {X_array.shape[1]}")
        
        return X_array
    
    def _get_feature_names(self):
        """Get feature names for input features"""
        if self.feature_names_in_:
            return self.feature_names_in_
        else:
            return [f'feature_{i}' for i in range(self.n_features_in_)]
    
    def _handle_missing_values(self, X_lagged):
        """Handle missing values created by lagging"""
        if self.config.fill_method == 'forward':
            # Forward fill
            for i in range(X_lagged.shape[1]):
                mask = ~np.isnan(X_lagged[:, i])
                if mask.any():
                    first_valid = np.where(mask)[0][0]
                    X_lagged[:first_valid, i] = X_lagged[first_valid, i]
        
        elif self.config.fill_method == 'backward':
            # Backward fill
            for i in range(X_lagged.shape[1]):
                mask = ~np.isnan(X_lagged[:, i])
                if mask.any():
                    last_valid = np.where(mask)[0][-1]
                    X_lagged[last_valid+1:, i] = X_lagged[last_valid, i]
        
        elif self.config.fill_method == 'zero':
            # Fill with zeros
            X_lagged = np.nan_to_num(X_lagged, nan=0.0)
        
        elif self.config.fill_method == 'drop':
            # This will be handled by caller (drop rows with NaN)
            pass
        
        return X_lagged
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        check_is_fitted(self, 'is_fitted_')
        
        if input_features is None:
            input_features = self._get_feature_names()
        
        output_names = []
        if self.config.include_original:
            output_names.extend(input_features)
        
        if hasattr(self, 'lag_names_') and self.lag_names_:
            output_names.extend(self.lag_names_)
        
        return output_names

# ============================================
# Simple Lag Transformers
# ============================================

class SimpleLagTransformer(BaseLagTransformer):
    """
    Creates simple lagged features for time series data.
    Generates features like X(t-1), X(t-2), etc.
    """
    
    def __init__(self, config: Optional[LagConfig] = None):
        super().__init__(config)
        self.lag_indices_ = None
    
    def fit(self, X, y=None):
        """Fit the simple lag transformer"""
        X = self._validate_input(X)
        
        # Generate lag indices
        self.lag_indices_ = list(range(
            self.config.min_lag, 
            self.config.max_lag + 1, 
            self.config.lag_step
        ))
        
        # Add seasonal lags if specified
        if self.config.seasonal_lags:
            self.lag_indices_.extend(self.config.seasonal_lags)
        
        # Auto-detect seasonal lags if requested
        if self.config.auto_detect_seasonality and y is not None:
            seasonal_lags = self._detect_seasonal_lags(X, y)
            self.lag_indices_.extend(seasonal_lags)
        
        # Remove duplicates and sort
        self.lag_indices_ = sorted(list(set(self.lag_indices_)))
        
        # Generate lag feature names
        feature_names = self._get_feature_names()
        self.lag_names_ = []
        
        for feature_idx in range(self.n_features_in_):
            for lag in self.lag_indices_:
                lag_name = f"{feature_names[feature_idx]}_lag_{lag}"
                self.lag_names_.append(lag_name)
        
        # Feature selection if specified
        if y is not None and (self.config.max_features or self.config.correlation_threshold):
            self._select_lag_features(X, y)
        
        self.is_fitted_ = True
        return self
    
    def _detect_seasonal_lags(self, X, y, max_seasonal_lag=252):
        """Auto-detect seasonal lags using autocorrelation"""
        seasonal_lags = []
        
        # Common financial seasonal patterns
        common_periods = [5, 21, 63, 126, 252]  # Weekly, monthly, quarterly, semi-annual, annual
        
        for period in common_periods:
            if period <= max_seasonal_lag and period <= len(X) // 3:
                # Check if this period shows significant autocorrelation
                if len(y) > period:
                    autocorr = np.corrcoef(y[:-period], y[period:])[0, 1]
                    if not np.isnan(autocorr) and abs(autocorr) > 0.1:
                        seasonal_lags.append(period)
        
        logger.info(f"Auto-detected seasonal lags: {seasonal_lags}")
        return seasonal_lags
    
    def _select_lag_features(self, X, y):
        """Select most important lag features"""
        # Generate all lag features first
        X_full_lags = self._create_lag_features(X)
        
        # Calculate feature importance (correlation with target)
        importances = []
        n_samples = min(len(y), X_full_lags.shape[0])
        
        for i in range(X_full_lags.shape[1]):
            feature_data = X_full_lags[:n_samples, i]
            target_data = y[:n_samples]
            
            # Remove NaN values for correlation calculation
            mask = ~(np.isnan(feature_data) | np.isnan(target_data))
            if mask.sum() > 10:  # Need at least 10 samples
                try:
                    corr = np.corrcoef(feature_data[mask], target_data[mask])[0, 1]
                    importances.append(abs(corr) if not np.isnan(corr) else 0)
                except:
                    importances.append(0)
            else:
                importances.append(0)
        
        # Select features based on criteria
        selected_indices = []
        
        if self.config.correlation_threshold:
            # Select features above correlation threshold
            selected_indices = [i for i, imp in enumerate(importances) 
                              if imp >= self.config.correlation_threshold]
        
        if self.config.max_features and (not selected_indices or len(selected_indices) > self.config.max_features):
            # Select top features by importance
            n_select = min(self.config.max_features, len(importances))
            selected_indices = np.argsort(importances)[-n_select:].tolist()
        
        if selected_indices:
            self.selected_lags_ = selected_indices
            self.lag_names_ = [self.lag_names_[i] for i in selected_indices]
            logger.info(f"Selected {len(selected_indices)} lag features out of {len(importances)}")
    
    def _create_lag_features(self, X):
        """Create lagged features from input data"""
        n_samples, n_features = X.shape
        max_lag = max(self.lag_indices_)
        
        # Initialize output array
        n_lag_features = len(self.lag_indices_) * n_features
        X_lagged = np.full((n_samples, n_lag_features), np.nan)
        
        feature_idx = 0
        for orig_feature_idx in range(n_features):
            for lag in self.lag_indices_:
                if lag < n_samples:
                    X_lagged[lag:, feature_idx] = X[:-lag, orig_feature_idx]
                feature_idx += 1
        
        return X_lagged
    
    def transform(self, X):
        """Transform data with lag features"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        # Create lag features
        X_lagged = self._create_lag_features(X)
        
        # Apply feature selection if performed
        if self.selected_lags_ is not None:
            X_lagged = X_lagged[:, self.selected_lags_]
        
        # Handle missing values
        X_lagged = self._handle_missing_values(X_lagged)
        
        # Combine with original features if requested
        if self.config.include_original:
            return np.hstack([X, X_lagged])
        else:
            return X_lagged

# ============================================
# Rolling Window Lag Features
# ============================================

class RollingLagTransformer(BaseLagTransformer):
    """
    Creates rolling window statistics of lagged features.
    Generates features like rolling mean, std, min, max of past N values.
    """
    
    def __init__(self, 
                 window_sizes: List[int] = [3, 5, 10],
                 statistics: List[str] = ['mean', 'std', 'min', 'max'],
                 config: Optional[LagConfig] = None):
        super().__init__(config)
        self.window_sizes = window_sizes
        self.statistics = statistics
        self.stat_functions = {
            'mean': np.nanmean,
            'std': np.nanstd,
            'min': np.nanmin,
            'max': np.nanmax,
            'median': np.nanmedian,
            'sum': np.nansum,
            'skew': lambda x: stats.skew(x, nan_policy='omit'),
            'kurtosis': lambda x: stats.kurtosis(x, nan_policy='omit')
        }
    
    def fit(self, X, y=None):
        """Fit the rolling lag transformer"""
        X = self._validate_input(X)
        
        # Generate rolling lag feature names
        feature_names = self._get_feature_names()
        self.lag_names_ = []
        
        for feature_idx in range(self.n_features_in_):
            for window_size in self.window_sizes:
                for stat_name in self.statistics:
                    if stat_name in self.stat_functions:
                        lag_name = f"{feature_names[feature_idx]}_rolling_{stat_name}_{window_size}"
                        self.lag_names_.append(lag_name)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with rolling lag features"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        n_samples, n_features = X.shape
        rolling_features = []
        
        for feature_idx in range(n_features):
            feature_data = X[:, feature_idx]
            
            for window_size in self.window_sizes:
                for stat_name in self.statistics:
                    if stat_name in self.stat_functions:
                        stat_func = self.stat_functions[stat_name]
                        
                        # Calculate rolling statistic
                        rolling_stat = np.full(n_samples, np.nan)
                        
                        for i in range(window_size - 1, n_samples):
                            window_data = feature_data[i - window_size + 1:i + 1]
                            rolling_stat[i] = stat_func(window_data)
                        
                        rolling_features.append(rolling_stat)
        
        X_rolling = np.column_stack(rolling_features) if rolling_features else np.array([]).reshape(n_samples, 0)
        
        # Handle missing values
        X_rolling = self._handle_missing_values(X_rolling)
        
        # Combine with original features if requested
        if self.config.include_original:
            return np.hstack([X, X_rolling])
        else:
            return X_rolling

# ============================================
# Difference and Change Lag Features
# ============================================

class DifferenceLagTransformer(BaseLagTransformer):
    """
    Creates difference-based lag features.
    Generates features like first differences, percentage changes, and momentum indicators.
    """
    
    def __init__(self, 
                 difference_orders: List[int] = [1, 2],
                 include_percentage_change: bool = True,
                 include_log_returns: bool = True,
                 config: Optional[LagConfig] = None):
        super().__init__(config)
        self.difference_orders = difference_orders
        self.include_percentage_change = include_percentage_change
        self.include_log_returns = include_log_returns
    
    def fit(self, X, y=None):
        """Fit the difference lag transformer"""
        X = self._validate_input(X)
        
        # Generate difference lag feature names
        feature_names = self._get_feature_names()
        self.lag_names_ = []
        
        for feature_idx in range(self.n_features_in_):
            feature_name = feature_names[feature_idx]
            
            # Difference features
            for order in self.difference_orders:
                lag_name = f"{feature_name}_diff_{order}"
                self.lag_names_.append(lag_name)
            
            # Percentage change features
            if self.include_percentage_change:
                for lag in range(self.config.min_lag, self.config.max_lag + 1):
                    lag_name = f"{feature_name}_pct_change_{lag}"
                    self.lag_names_.append(lag_name)
            
            # Log return features
            if self.include_log_returns:
                for lag in range(self.config.min_lag, self.config.max_lag + 1):
                    lag_name = f"{feature_name}_log_return_{lag}"
                    self.lag_names_.append(lag_name)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with difference lag features"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        n_samples, n_features = X.shape
        difference_features = []
        
        for feature_idx in range(n_features):
            feature_data = X[:, feature_idx]
            
            # Difference features
            for order in self.difference_orders:
                diff_feature = np.full(n_samples, np.nan)
                if order < n_samples:
                    diff_feature[order:] = np.diff(feature_data, n=order)
                difference_features.append(diff_feature)
            
            # Percentage change features
            if self.include_percentage_change:
                for lag in range(self.config.min_lag, self.config.max_lag + 1):
                    pct_change = np.full(n_samples, np.nan)
                    if lag < n_samples:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pct_change[lag:] = (feature_data[lag:] - feature_data[:-lag]) / feature_data[:-lag]
                        # Handle infinite values
                        pct_change = np.where(np.isinf(pct_change), np.nan, pct_change)
                    difference_features.append(pct_change)
            
            # Log return features
            if self.include_log_returns:
                for lag in range(self.config.min_lag, self.config.max_lag + 1):
                    log_return = np.full(n_samples, np.nan)
                    if lag < n_samples:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            log_return[lag:] = np.log(feature_data[lag:] / feature_data[:-lag])
                        # Handle infinite and NaN values
                        log_return = np.where(np.isinf(log_return), np.nan, log_return)
                    difference_features.append(log_return)
        
        X_diff = np.column_stack(difference_features) if difference_features else np.array([]).reshape(n_samples, 0)
        
        # Handle missing values
        X_diff = self._handle_missing_values(X_diff)
        
        # Combine with original features if requested
        if self.config.include_original:
            return np.hstack([X, X_diff])
        else:
            return X_diff

# ============================================
# Seasonal and Cyclical Lag Features
# ============================================

class SeasonalLagTransformer(BaseLagTransformer):
    """
    Creates seasonal lag features for financial time series.
    Handles common financial seasonality patterns like day-of-week, month-of-year effects.
    """
    
    def __init__(self, 
                 seasonal_periods: Dict[str, int] = None,
                 include_harmonics: bool = True,
                 n_harmonics: int = 3,
                 config: Optional[LagConfig] = None):
        super().__init__(config)
        self.seasonal_periods = seasonal_periods or {
            'weekly': 5,      # 5 trading days
            'monthly': 21,    # ~21 trading days per month
            'quarterly': 63,  # ~63 trading days per quarter
            'yearly': 252     # ~252 trading days per year
        }
        self.include_harmonics = include_harmonics
        self.n_harmonics = n_harmonics
    
    def fit(self, X, y=None):
        """Fit the seasonal lag transformer"""
        X = self._validate_input(X)
        
        # Generate seasonal lag feature names
        feature_names = self._get_feature_names()
        self.lag_names_ = []
        
        for feature_idx in range(self.n_features_in_):
            feature_name = feature_names[feature_idx]
            
            # Seasonal lag features
            for season_name, period in self.seasonal_periods.items():
                if period <= len(X):
                    lag_name = f"{feature_name}_seasonal_{season_name}"
                    self.lag_names_.append(lag_name)
            
            # Harmonic features
            if self.include_harmonics:
                for season_name, period in self.seasonal_periods.items():
                    if period <= len(X):
                        for harmonic in range(1, self.n_harmonics + 1):
                            cos_name = f"{feature_name}_cos_{season_name}_h{harmonic}"
                            sin_name = f"{feature_name}_sin_{season_name}_h{harmonic}"
                            self.lag_names_.extend([cos_name, sin_name])
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with seasonal lag features"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        n_samples, n_features = X.shape
        seasonal_features = []
        
        for feature_idx in range(n_features):
            feature_data = X[:, feature_idx]
            
            # Seasonal lag features
            for season_name, period in self.seasonal_periods.items():
                if period <= n_samples:
                    seasonal_lag = np.full(n_samples, np.nan)
                    seasonal_lag[period:] = feature_data[:-period]
                    seasonal_features.append(seasonal_lag)
            
            # Harmonic features
            if self.include_harmonics:
                for season_name, period in self.seasonal_periods.items():
                    if period <= n_samples:
                        for harmonic in range(1, self.n_harmonics + 1):
                            # Create time index
                            t = np.arange(n_samples)
                            
                            # Cosine and sine components
                            cos_component = feature_data * np.cos(2 * np.pi * harmonic * t / period)
                            sin_component = feature_data * np.sin(2 * np.pi * harmonic * t / period)
                            
                            seasonal_features.extend([cos_component, sin_component])
        
        X_seasonal = np.column_stack(seasonal_features) if seasonal_features else np.array([]).reshape(n_samples, 0)
        
        # Handle missing values
        X_seasonal = self._handle_missing_values(X_seasonal)
        
        # Combine with original features if requested
        if self.config.include_original:
            return np.hstack([X, X_seasonal])
        else:
            return X_seasonal

# ============================================
# Composite Lag Transformer
# ============================================

class CompositeLagTransformer(BaseLagTransformer):
    """
    Combines multiple lag transformers for comprehensive lag feature engineering.
    """
    
    def __init__(self, 
                 transformers: List[Tuple[str, BaseLagTransformer]],
                 config: Optional[LagConfig] = None):
        super().__init__(config)
        self.transformers = transformers
        self.fitted_transformers_ = []
    
    def fit(self, X, y=None):
        """Fit all component transformers"""
        X = self._validate_input(X)
        
        self.fitted_transformers_ = []
        all_lag_names = []
        
        for name, transformer in self.transformers:
            # Set transformer config to exclude original features (we'll handle that)
            transformer.config.include_original = False
            
            # Fit transformer
            fitted_transformer = transformer.fit(X, y)
            self.fitted_transformers_.append((name, fitted_transformer))
            
            # Collect lag names
            if hasattr(fitted_transformer, 'lag_names_'):
                transformer_names = [f"{name}_{lname}" for lname in fitted_transformer.lag_names_]
                all_lag_names.extend(transformer_names)
        
        self.lag_names_ = all_lag_names
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data using all component transformers"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        all_lag_features = []
        
        for name, transformer in self.fitted_transformers_:
            # Get lag features from this transformer
            transformer_output = transformer.transform(X)
            all_lag_features.append(transformer_output)
        
        if not all_lag_features:
            X_lags = np.array([]).reshape(X.shape[0], 0)
        else:
            X_lags = np.hstack(all_lag_features)
        
        # Handle missing values for combined features
        X_lags = self._handle_missing_values(X_lags)
        
        if self.config.include_original:
            return np.hstack([X, X_lags])
        else:
            return X_lags

# ============================================
# Utility Functions
# ============================================

@time_it("lag_generation")
def create_comprehensive_lags(X: Union[pd.DataFrame, np.ndarray],
                            feature_names: Optional[List[str]] = None,
                            max_lag: int = 5,
                            include_rolling: bool = True,
                            include_differences: bool = True,
                            include_seasonal: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Create comprehensive lag features for financial time series
    
    Args:
        X: Input feature matrix
        feature_names: Names of input features
        max_lag: Maximum lag to create
        include_rolling: Whether to include rolling window statistics
        include_differences: Whether to include difference-based lags
        include_seasonal: Whether to include seasonal lags
        
    Returns:
        Tuple of (transformed_features, feature_names)
    """
    
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or X.columns.tolist()
        X = X.values
    
    config = LagConfig(
        max_lag=max_lag,
        feature_names=feature_names,
        include_original=True,
        fill_method='forward'
    )
    
    transformers = []
    
    # Simple lags
    simple_transformer = SimpleLagTransformer(config)
    transformers.append(('simple', simple_transformer))
    
    # Rolling window features
    if include_rolling:
        rolling_transformer = RollingLagTransformer(
            window_sizes=[3, 5, 10],
            statistics=['mean', 'std', 'min', 'max'],
            config=config
        )
        transformers.append(('rolling', rolling_transformer))
    
    # Difference features
    if include_differences:
        diff_transformer = DifferenceLagTransformer(
            difference_orders=[1],
            include_percentage_change=True,
            include_log_returns=True,
            config=config
        )
        transformers.append(('diff', diff_transformer))
    
    # Seasonal features
    if include_seasonal:
        seasonal_transformer = SeasonalLagTransformer(
            seasonal_periods={'weekly': 5, 'monthly': 21},
            include_harmonics=False,  # Keep it simple for comprehensive function
            config=config
        )
        transformers.append(('seasonal', seasonal_transformer))
    
    # Create composite transformer
    composite = CompositeLagTransformer(transformers, config)
    X_transformed = composite.fit_transform(X)
    output_names = composite.get_feature_names_out(feature_names)
    
    logger.info(f"Created {X_transformed.shape[1] - X.shape[1]} lag features")
    return X_transformed, output_names

def analyze_lag_importance(X: np.ndarray, 
                         y: np.ndarray,
                         lag_names: List[str],
                         max_lag: int = 10) -> pd.DataFrame:
    """Analyze importance of different lag periods"""
    
    # Separate features by lag period
    lag_importance = {}
    
    for lag in range(1, max_lag + 1):
        lag_features = [i for i, name in enumerate(lag_names) if f'_lag_{lag}' in name or f'_{lag}' in name]
        
        if lag_features:
            X_lag = X[:, lag_features]
            
            # Calculate mutual information
            try:
                if len(np.unique(y)) < 20:  # Classification
                    from sklearn.feature_selection import mutual_info_classif
                    importance = mutual_info_classif(X_lag, y, random_state=42).mean()
                else:  # Regression
                    importance = mutual_info_regression(X_lag, y, random_state=42).mean()
                
                lag_importance[lag] = importance
            except:
                lag_importance[lag] = 0.0
    
    # Create DataFrame
    importance_df = pd.DataFrame(list(lag_importance.items()), columns=['lag', 'importance'])
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df

def detect_optimal_lags(X: Union[pd.DataFrame, np.ndarray],
                       y: np.ndarray,
                       max_lag: int = 20,
                       significance_threshold: float = 0.05) -> List[int]:
    """
    Detect optimal lag periods using statistical tests
    
    Args:
        X: Input features (should be single feature for lag detection)
        y: Target variable
        max_lag: Maximum lag to test
        significance_threshold: P-value threshold for significance
        
    Returns:
        List of significant lag periods
    """
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    if X.shape[1] > 1:
        # Use first feature if multiple features provided
        X = X[:, 0:1]
        logger.warning("Multiple features provided, using first feature for lag detection")
    
    feature_data = X.flatten()
    significant_lags = []
    
    for lag in range(1, min(max_lag + 1, len(feature_data) // 2)):
        if lag >= len(feature_data):
            break
        
        # Create lag feature
        lagged_feature = np.full(len(feature_data), np.nan)
        lagged_feature[lag:] = feature_data[:-lag]
        
        # Remove NaN values
        mask = ~(np.isnan(lagged_feature) | np.isnan(y))
        if mask.sum() < 10:  # Need minimum samples
            continue
        
        lagged_clean = lagged_feature[mask]
        y_clean = y[mask]
        
        # Test correlation significance
        try:
            corr, p_value = stats.pearsonr(lagged_clean, y_clean)
            if p_value < significance_threshold and abs(corr) > 0.05:
                significant_lags.append(lag)
        except:
            continue
    
    logger.info(f"Detected {len(significant_lags)} significant lags: {significant_lags}")
    return significant_lags

def optimize_lag_features(X: np.ndarray,
                         y: np.ndarray,
                         lag_names: List[str],
                         max_features: int = 50) -> Tuple[np.ndarray, List[str]]:
    """
    Optimize lag features by selecting the most important ones
    
    Args:
        X: Feature matrix with lag features
        y: Target variable
        lag_names: Names of lag features
        max_features: Maximum number of features to keep
        
    Returns:
        Optimized features and names
    """
    
    # Calculate feature importance
    try:
        if len(np.unique(y)) < 20:  # Classification
            from sklearn.feature_selection import mutual_info_classif
            importances = mutual_info_classif(X, y, random_state=42)
        else:  # Regression
            importances = mutual_info_regression(X, y, random_state=42)
    except:
        # Fallback to correlation
        importances = []
        for i in range(X.shape[1]):
            try:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                importances.append(abs(corr) if not np.isnan(corr) else 0)
            except:
                importances.append(0)
        importances = np.array(importances)
    
    # Select top features
    if len(importances) > max_features:
        top_indices = np.argsort(importances)[-max_features:]
        X_optimized = X[:, top_indices]
        optimized_names = [lag_names[i] for i in top_indices]
    else:
        X_optimized = X
        optimized_names = lag_names
    
    logger.info(f"Optimized from {len(lag_names)} to {len(optimized_names)} lag features")
    return X_optimized, optimized_names

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Lag Transformers")
    
    # Create sample time series data
    np.random.seed(42)
    n_samples = 500
    
    # Generate realistic financial time series
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Price series with some autocorrelation
    returns = np.random.normal(0, 0.02, n_samples)
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]  # Add some momentum
    
    prices = 100 * np.cumprod(1 + returns)
    volume = np.random.lognormal(10, 0.5, n_samples)
    volatility = np.abs(np.random.normal(0.2, 0.05, n_samples))
    
    X = np.column_stack([prices, returns, volume])
    feature_names = ['price', 'returns', 'volume']
    y = returns[1:] * 100  # Next period return as target
    
    print(f"Original data shape: {X.shape}")
    
    # Test simple lag transformer
    print("\n1. Testing Simple Lag Transformer")
    config = LagConfig(max_lag=5, feature_names=feature_names, fill_method='forward')
    simple_transformer = SimpleLagTransformer(config)
    X_simple = simple_transformer.fit_transform(X[:-1], y)  # Adjust for target
    simple_names = simple_transformer.get_feature_names_out()
    print(f"Simple lags shape: {X_simple.shape}")
    print(f"Created {len(simple_transformer.lag_names_)} lag features")
    
    # Test rolling lag transformer
    print("\n2. Testing Rolling Lag Transformer")
    rolling_transformer = RollingLagTransformer(
        window_sizes=[3, 5, 10],
        statistics=['mean', 'std', 'min', 'max'],
        config=config
    )
    X_rolling = rolling_transformer.fit_transform(X[:-1])
    rolling_names = rolling_transformer.get_feature_names_out()
    print(f"Rolling lags shape: {X_rolling.shape}")
    print(f"Created {len(rolling_transformer.lag_names_)} rolling features")
    
    # Test difference lag transformer
    print("\n3. Testing Difference Lag Transformer")
    diff_transformer = DifferenceLagTransformer(
        difference_orders=[1],
        include_percentage_change=True,
        include_log_returns=True,
        config=config
    )
    X_diff = diff_transformer.fit_transform(X[:-1])
    diff_names = diff_transformer.get_feature_names_out()
    print(f"Difference lags shape: {X_diff.shape}")
    print(f"Created {len(diff_transformer.lag_names_)} difference features")
    
    # Test comprehensive lag creation
    print("\n4. Testing Comprehensive Lag Creation")
    X_comprehensive, comprehensive_names = create_comprehensive_lags(
        X[:-1], feature_names,
        max_lag=5,
        include_rolling=True,
        include_differences=True,
        include_seasonal=False  # Skip seasonal for short series
    )
    print(f"Comprehensive lags shape: {X_comprehensive.shape}")
    print(f"Total features: {len(comprehensive_names)}")
    
    # Test lag importance analysis
    print("\n5. Testing Lag Importance Analysis")
    lag_importance = analyze_lag_importance(X_comprehensive, y, comprehensive_names, max_lag=5)
    print("Lag importance by period:")
    print(lag_importance)
    
    # Test optimal lag detection
    print("\n6. Testing Optimal Lag Detection")
    optimal_lags = detect_optimal_lags(X[:-1, :1], y, max_lag=10)  # Use price only
    print(f"Optimal lags detected: {optimal_lags}")
    
    # Test feature optimization
    print("\n7. Testing Feature Optimization")
    X_optimized, optimized_names = optimize_lag_features(
        X_comprehensive, y, comprehensive_names, max_features=20
    )
    print(f"Optimized lags shape: {X_optimized.shape}")
    print(f"Selected features: {optimized_names[:10]}...")  # Show first 10
    
    print("\nLag transformers testing completed successfully!")
