# ============================================
# StockPredictionPro - src/features/transformers/scalers.py
# Advanced scaling and normalization transformers for financial machine learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.utils.validation import check_array, check_is_fitted
from scipy import stats
from scipy.special import boxcox

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('features.transformers.scalers')

# ============================================
# Configuration and Base Classes
# ============================================

@dataclass
class ScalerConfig:
    """Configuration for scaling transformers"""
    feature_range: Tuple[float, float] = (0, 1)
    quantile_range: Tuple[float, float] = (25.0, 75.0)
    n_quantiles: int = 1000
    output_distribution: str = 'uniform'  # 'uniform' or 'normal'
    feature_names: Optional[List[str]] = None
    with_centering: bool = True
    with_scaling: bool = True
    robust: bool = False
    clip_outliers: bool = False
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'percentile'
    outlier_threshold: float = 3.0
    
    def __post_init__(self):
        if self.feature_range[0] >= self.feature_range[1]:
            raise ValueError("feature_range[0] must be < feature_range[1]")
        if self.quantile_range[0] >= self.quantile_range[1]:
            raise ValueError("quantile_range[0] must be < quantile_range[1]")

class BaseScaler(BaseEstimator, TransformerMixin):
    """Base class for all scaling transformers"""
    
    def __init__(self, config: Optional[ScalerConfig] = None):
        self.config = config or ScalerConfig()
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False
        self.outlier_bounds_ = None
    
    def _validate_input(self, X):
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
        
        return X_array
    
    def _clip_outliers(self, X, fit_mode=True):
        """Clip outliers based on specified method"""
        if not self.config.clip_outliers:
            return X
        
        if fit_mode:
            self.outlier_bounds_ = {}
        
        X_clipped = X.copy()
        
        for feature_idx in range(X.shape[1]):
            feature_data = X[:, feature_idx]
            
            if fit_mode:
                if self.config.outlier_method == 'iqr':
                    Q1 = np.percentile(feature_data, 25)
                    Q3 = np.percentile(feature_data, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.config.outlier_threshold * IQR
                    upper_bound = Q3 + self.config.outlier_threshold * IQR
                
                elif self.config.outlier_method == 'zscore':
                    mean_val = np.mean(feature_data)
                    std_val = np.std(feature_data)
                    lower_bound = mean_val - self.config.outlier_threshold * std_val
                    upper_bound = mean_val + self.config.outlier_threshold * std_val
                
                elif self.config.outlier_method == 'percentile':
                    lower_bound = np.percentile(feature_data, self.config.outlier_threshold)
                    upper_bound = np.percentile(feature_data, 100 - self.config.outlier_threshold)
                
                else:
                    raise ValueError(f"Unknown outlier method: {self.config.outlier_method}")
                
                self.outlier_bounds_[feature_idx] = (lower_bound, upper_bound)
            
            # Apply clipping
            if self.outlier_bounds_ and feature_idx in self.outlier_bounds_:
                lower_bound, upper_bound = self.outlier_bounds_[feature_idx]
                X_clipped[:, feature_idx] = np.clip(feature_data, lower_bound, upper_bound)
        
        return X_clipped
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        check_is_fitted(self, 'is_fitted_')
        
        if input_features is None:
            if self.feature_names_in_:
                return self.feature_names_in_.copy()
            else:
                return [f'feature_{i}' for i in range(self.n_features_in_)]
        else:
            return list(input_features)

# ============================================
# Financial-Specific Scalers
# ============================================

class FinancialStandardScaler(BaseScaler):
    """
    Financial-aware standard scaler that handles financial time series characteristics.
    Includes options for rolling normalization and handling of financial returns.
    """
    
    def __init__(self, 
                 rolling_window: Optional[int] = None,
                 handle_returns: bool = True,
                 return_features: Optional[List[int]] = None,
                 config: Optional[ScalerConfig] = None):
        super().__init__(config)
        self.rolling_window = rolling_window
        self.handle_returns = handle_returns
        self.return_features = return_features or []
        self.scaler_ = None
        self.rolling_stats_ = None
    
    def fit(self, X, y=None):
        """Fit the financial standard scaler"""
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=True)
        
        if self.rolling_window:
            # Rolling standardization
            self._fit_rolling_stats(X)
        else:
            # Standard sklearn StandardScaler
            self.scaler_ = StandardScaler(
                with_mean=self.config.with_centering,
                with_std=self.config.with_scaling
            )
            self.scaler_.fit(X)
        
        self.is_fitted_ = True
        return self
    
    def _fit_rolling_stats(self, X):
        """Fit rolling statistics for financial time series"""
        n_samples, n_features = X.shape
        
        if self.rolling_window >= n_samples:
            logger.warning(f"Rolling window ({self.rolling_window}) >= n_samples ({n_samples}), using standard scaler")
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X)
            return
        
        # Calculate rolling means and stds for the last rolling_window samples
        # This will be used as the normalization parameters
        recent_data = X[-self.rolling_window:]
        self.rolling_stats_ = {
            'mean': np.mean(recent_data, axis=0),
            'std': np.std(recent_data, axis=0)
        }
        
        # Handle zero standard deviations
        self.rolling_stats_['std'][self.rolling_stats_['std'] == 0] = 1.0
    
    def transform(self, X):
        """Transform data with financial standard scaling"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=False)
        
        if self.rolling_window and self.rolling_stats_:
            # Apply rolling standardization
            if self.config.with_centering and self.config.with_scaling:
                X_scaled = (X - self.rolling_stats_['mean']) / self.rolling_stats_['std']
            elif self.config.with_centering:
                X_scaled = X - self.rolling_stats_['mean']
            elif self.config.with_scaling:
                X_scaled = X / self.rolling_stats_['std']
            else:
                X_scaled = X.copy()
        else:
            # Use sklearn scaler
            X_scaled = self.scaler_.transform(X)
        
        # Special handling for return features
        if self.handle_returns and self.return_features:
            for feature_idx in self.return_features:
                if feature_idx < X_scaled.shape[1]:
                    # Returns are already somewhat normalized, just ensure reasonable range
                    X_scaled[:, feature_idx] = np.clip(X_scaled[:, feature_idx], -10, 10)
        
        return X_scaled

class FinancialRobustScaler(BaseScaler):
    """
    Robust scaler designed for financial data with outliers.
    Uses median and IQR for scaling, less sensitive to extreme values.
    """
    
    def __init__(self, 
                 quantile_range: Tuple[float, float] = (25.0, 75.0),
                 config: Optional[ScalerConfig] = None):
        super().__init__(config)
        self.quantile_range = quantile_range
        self.scaler_ = None
    
    def fit(self, X, y=None):
        """Fit the financial robust scaler"""
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=True)
        
        self.scaler_ = RobustScaler(
            quantile_range=self.quantile_range,
            with_centering=self.config.with_centering,
            with_scaling=self.config.with_scaling
        )
        self.scaler_.fit(X)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with robust scaling"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=False)
        
        return self.scaler_.transform(X)

class FinancialMinMaxScaler(BaseScaler):
    """
    MinMax scaler with financial-specific enhancements.
    Includes options for feature-specific ranges and return handling.
    """
    
    def __init__(self, 
                 feature_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
                 config: Optional[ScalerConfig] = None):
        super().__init__(config)
        self.feature_ranges = feature_ranges or {}
        self.scalers_ = {}
    
    def fit(self, X, y=None):
        """Fit the financial MinMax scaler"""
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=True)
        
        for feature_idx in range(X.shape[1]):
            # Use feature-specific range if provided
            feature_range = self.feature_ranges.get(feature_idx, self.config.feature_range)
            
            scaler = MinMaxScaler(feature_range=feature_range)
            scaler.fit(X[:, [feature_idx]])
            self.scalers_[feature_idx] = scaler
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with MinMax scaling"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=False)
        
        X_scaled = np.zeros_like(X)
        
        for feature_idx in range(X.shape[1]):
            scaler = self.scalers_[feature_idx]
            X_scaled[:, [feature_idx]] = scaler.transform(X[:, [feature_idx]])
        
        return X_scaled

# ============================================
# Distribution-Based Scalers
# ============================================

class FinancialQuantileScaler(BaseScaler):
    """
    Quantile-based scaling for financial data.
    Maps features to uniform or normal distribution.
    """
    
    def __init__(self, 
                 output_distribution: str = 'uniform',
                 n_quantiles: int = 1000,
                 subsample: int = 100000,
                 config: Optional[ScalerConfig] = None):
        super().__init__(config)
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.scaler_ = None
    
    def fit(self, X, y=None):
        """Fit the financial quantile scaler"""
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=True)
        
        self.scaler_ = QuantileTransformer(
            n_quantiles=min(self.n_quantiles, X.shape[0]),
            output_distribution=self.output_distribution,
            subsample=min(self.subsample, X.shape[0]),
            random_state=42
        )
        self.scaler_.fit(X)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with quantile scaling"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=False)
        
        return self.scaler_.transform(X)

class LogScaler(BaseScaler):
    """
    Logarithmic scaling for financial data.
    Useful for features with exponential distributions like prices, volumes.
    """
    
    def __init__(self, 
                 base: float = np.e,
                 handle_negative: str = 'clip',  # 'clip', 'shift', 'error'
                 handle_zero: str = 'small_value',  # 'small_value', 'error'
                 config: Optional[ScalerConfig] = None):
        super().__init__(config)
        self.base = base
        self.handle_negative = handle_negative
        self.handle_zero = handle_zero
        self.shift_values_ = None
        self.min_values_ = None
    
    def fit(self, X, y=None):
        """Fit the log scaler"""
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=True)
        
        self.min_values_ = np.min(X, axis=0)
        
        if self.handle_negative == 'shift':
            # Shift negative values to make them positive
            self.shift_values_ = np.where(self.min_values_ <= 0, -self.min_values_ + 1e-8, 0)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with logarithmic scaling"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=False)
        
        X_transformed = X.copy()
        
        for feature_idx in range(X.shape[1]):
            feature_data = X_transformed[:, feature_idx]
            
            # Handle negative values
            if self.handle_negative == 'clip':
                feature_data = np.clip(feature_data, 1e-8, None)
            elif self.handle_negative == 'shift':
                feature_data = feature_data + self.shift_values_[feature_idx]
            elif self.handle_negative == 'error':
                if np.any(feature_data <= 0):
                    raise ValueError(f"Feature {feature_idx} contains non-positive values")
            
            # Handle zero values
            if self.handle_zero == 'small_value':
                feature_data = np.where(feature_data == 0, 1e-8, feature_data)
            elif self.handle_zero == 'error':
                if np.any(feature_data == 0):
                    raise ValueError(f"Feature {feature_idx} contains zero values")
            
            # Apply logarithm
            if self.base == np.e:
                X_transformed[:, feature_idx] = np.log(feature_data)
            elif self.base == 10:
                X_transformed[:, feature_idx] = np.log10(feature_data)
            elif self.base == 2:
                X_transformed[:, feature_idx] = np.log2(feature_data)
            else:
                X_transformed[:, feature_idx] = np.log(feature_data) / np.log(self.base)
        
        return X_transformed
    
    def inverse_transform(self, X):
        """Inverse transform logarithmic scaling"""
        check_is_fitted(self, 'is_fitted_')
        
        X_original = X.copy()
        
        for feature_idx in range(X.shape[1]):
            feature_data = X_original[:, feature_idx]
            
            # Apply inverse logarithm
            if self.base == np.e:
                feature_data = np.exp(feature_data)
            else:
                feature_data = np.power(self.base, feature_data)
            
            # Reverse shift if applied
            if self.handle_negative == 'shift' and self.shift_values_ is not None:
                feature_data = feature_data - self.shift_values_[feature_idx]
            
            X_original[:, feature_idx] = feature_data
        
        return X_original

class BoxCoxScaler(BaseScaler):
    """
    Box-Cox transformation scaler for financial data.
    Automatically finds optimal lambda parameter for normalization.
    """
    
    def __init__(self, 
                 lambda_range: Tuple[float, float] = (-2, 2),
                 config: Optional[ScalerConfig] = None):
        super().__init__(config)
        self.lambda_range = lambda_range
        self.lambdas_ = None
        self.shift_values_ = None
    
    def fit(self, X, y=None):
        """Fit the Box-Cox scaler"""
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=True)
        
        self.lambdas_ = np.zeros(X.shape[1])
        self.shift_values_ = np.zeros(X.shape[1])
        
        for feature_idx in range(X.shape[1]):
            feature_data = X[:, feature_idx]
            
            # Box-Cox requires positive values
            min_val = np.min(feature_data)
            if min_val <= 0:
                shift = -min_val + 1e-8
                feature_data = feature_data + shift
                self.shift_values_[feature_idx] = shift
            
            try:
                # Find optimal lambda
                _, fitted_lambda = boxcox(feature_data)
                
                # Constrain lambda to reasonable range
                fitted_lambda = np.clip(fitted_lambda, self.lambda_range[0], self.lambda_range[1])
                self.lambdas_[feature_idx] = fitted_lambda
                
            except Exception as e:
                logger.warning(f"Box-Cox fitting failed for feature {feature_idx}: {e}")
                self.lambdas_[feature_idx] = 1.0  # No transformation
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with Box-Cox scaling"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=False)
        
        X_transformed = X.copy()
        
        for feature_idx in range(X.shape[1]):
            feature_data = X_transformed[:, feature_idx]
            
            # Apply shift if needed
            if self.shift_values_[feature_idx] != 0:
                feature_data = feature_data + self.shift_values_[feature_idx]
            
            # Apply Box-Cox transformation
            lambda_val = self.lambdas_[feature_idx]
            
            if abs(lambda_val) < 1e-8:  # lambda ≈ 0
                X_transformed[:, feature_idx] = np.log(feature_data)
            else:
                X_transformed[:, feature_idx] = (np.power(feature_data, lambda_val) - 1) / lambda_val
        
        return X_transformed

# ============================================
# Time Series Aware Scalers
# ============================================

class TimeSeriesScaler(BaseScaler):
    """
    Time series aware scaler that handles temporal dependencies.
    Supports expanding window and rolling window normalization.
    """
    
    def __init__(self, 
                 window_type: str = 'expanding',  # 'expanding', 'rolling'
                 window_size: Optional[int] = None,
                 min_periods: int = 30,
                 scaling_method: str = 'standard',  # 'standard', 'minmax', 'robust'
                 config: Optional[ScalerConfig] = None):
        super().__init__(config)
        self.window_type = window_type
        self.window_size = window_size
        self.min_periods = min_periods
        self.scaling_method = scaling_method
        self.base_scaler_ = None
        self.is_time_series_fitted_ = False
    
    def fit(self, X, y=None):
        """Fit the time series scaler"""
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=True)
        
        # For time series, we don't pre-fit statistics like traditional scalers
        # Instead, we'll compute them dynamically during transform
        self.is_time_series_fitted_ = True
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with time-series aware scaling"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=False)
        
        n_samples, n_features = X.shape
        X_scaled = np.zeros_like(X)
        
        for i in range(n_samples):
            if self.window_type == 'expanding':
                # Use all data from start to current point
                if i + 1 >= self.min_periods:
                    window_data = X[:i+1]
                else:
                    # Not enough data, use available data or no scaling
                    window_data = X[:i+1] if i > 0 else X[0:1]
            
            elif self.window_type == 'rolling':
                # Use fixed window size
                if self.window_size is None:
                    raise ValueError("window_size must be specified for rolling window")
                
                start_idx = max(0, i - self.window_size + 1)
                window_data = X[start_idx:i+1]
                
                if len(window_data) < self.min_periods:
                    # Not enough data for scaling
                    X_scaled[i] = X[i]
                    continue
            
            # Compute scaling parameters for this window
            if self.scaling_method == 'standard':
                mean_vals = np.mean(window_data, axis=0)
                std_vals = np.std(window_data, axis=0)
                std_vals[std_vals == 0] = 1.0  # Avoid division by zero
                X_scaled[i] = (X[i] - mean_vals) / std_vals
            
            elif self.scaling_method == 'minmax':
                min_vals = np.min(window_data, axis=0)
                max_vals = np.max(window_data, axis=0)
                range_vals = max_vals - min_vals
                range_vals[range_vals == 0] = 1.0  # Avoid division by zero
                X_scaled[i] = (X[i] - min_vals) / range_vals
            
            elif self.scaling_method == 'robust':
                median_vals = np.median(window_data, axis=0)
                q75 = np.percentile(window_data, 75, axis=0)
                q25 = np.percentile(window_data, 25, axis=0)
                iqr_vals = q75 - q25
                iqr_vals[iqr_vals == 0] = 1.0  # Avoid division by zero
                X_scaled[i] = (X[i] - median_vals) / iqr_vals
            
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        return X_scaled

# ============================================
# Composite and Adaptive Scalers
# ============================================

class AdaptiveScaler(BaseScaler):
    """
    Adaptive scaler that chooses the best scaling method for each feature
    based on data distribution characteristics.
    """
    
    def __init__(self, 
                 auto_select: bool = True,
                 feature_scalers: Optional[Dict[int, str]] = None,
                 config: Optional[ScalerConfig] = None):
        super().__init__(config)
        self.auto_select = auto_select
        self.feature_scalers = feature_scalers or {}
        self.selected_scalers_ = {}
        self.fitted_scalers_ = {}
    
    def fit(self, X, y=None):
        """Fit the adaptive scaler"""
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=True)
        
        for feature_idx in range(X.shape[1]):
            feature_data = X[:, feature_idx]
            
            # Use manually specified scaler if provided
            if feature_idx in self.feature_scalers:
                scaler_type = self.feature_scalers[feature_idx]
            elif self.auto_select:
                scaler_type = self._select_best_scaler(feature_data)
            else:
                scaler_type = 'standard'  # Default
            
            self.selected_scalers_[feature_idx] = scaler_type
            
            # Create and fit the selected scaler
            scaler = self._create_scaler(scaler_type)
            scaler.fit(feature_data.reshape(-1, 1))
            self.fitted_scalers_[feature_idx] = scaler
        
        self.is_fitted_ = True
        return self
    
    def _select_best_scaler(self, feature_data):
        """Automatically select the best scaler for a feature"""
        
        # Test for normality
        _, normality_p = stats.normaltest(feature_data)
        
        # Test for skewness
        skewness = stats.skew(feature_data)
        
        # Check for outliers
        q75, q25 = np.percentile(feature_data, [75, 25])
        iqr = q75 - q25
        outlier_count = np.sum((feature_data < (q25 - 1.5 * iqr)) | 
                              (feature_data > (q75 + 1.5 * iqr)))
        outlier_ratio = outlier_count / len(feature_data)
        
        # Check for positive values (for log scaling)
        all_positive = np.all(feature_data > 0)
        
        # Selection logic
        if all_positive and abs(skewness) > 2:  # Highly skewed positive data
            return 'log'
        elif outlier_ratio > 0.1:  # Many outliers
            return 'robust'
        elif normality_p < 0.05:  # Non-normal distribution
            return 'quantile'
        else:  # Approximately normal
            return 'standard'
    
    def _create_scaler(self, scaler_type):
        """Create scaler instance based on type"""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler(feature_range=self.config.feature_range)
        elif scaler_type == 'robust':
            return RobustScaler(quantile_range=self.config.quantile_range)
        elif scaler_type == 'quantile':
            return QuantileTransformer(
                n_quantiles=min(self.config.n_quantiles, 1000),
                output_distribution=self.config.output_distribution
            )
        elif scaler_type == 'log':
            return LogScaler(config=self.config)
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def transform(self, X):
        """Transform data with adaptive scaling"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=False)
        
        X_scaled = np.zeros_like(X)
        
        for feature_idx in range(X.shape[1]):
            scaler = self.fitted_scalers_[feature_idx]
            X_scaled[:, [feature_idx]] = scaler.transform(X[:, [feature_idx]])
        
        return X_scaled
    
    def get_selected_scalers(self):
        """Get the selected scaler for each feature"""
        check_is_fitted(self, 'is_fitted_')
        return self.selected_scalers_.copy()

class CompositeScaler(BaseScaler):
    """
    Composite scaler that applies different scalers to different feature groups.
    """
    
    def __init__(self, 
                 scaler_groups: Dict[str, Dict],
                 config: Optional[ScalerConfig] = None):
        """
        Args:
            scaler_groups: Dict with format:
                {
                    'group_name': {
                        'scaler': scaler_instance,
                        'features': [feature_indices]
                    }
                }
        """
        super().__init__(config)
        self.scaler_groups = scaler_groups
        self.fitted_groups_ = {}
    
    def fit(self, X, y=None):
        """Fit the composite scaler"""
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=True)
        
        for group_name, group_config in self.scaler_groups.items():
            scaler = group_config['scaler']
            feature_indices = group_config['features']
            
            # Validate feature indices
            max_idx = max(feature_indices) if feature_indices else -1
            if max_idx >= X.shape[1]:
                raise ValueError(f"Feature index {max_idx} is out of bounds for {X.shape[1]} features")
            
            # Fit scaler on selected features
            X_group = X[:, feature_indices]
            fitted_scaler = scaler.fit(X_group)
            
            self.fitted_groups_[group_name] = {
                'scaler': fitted_scaler,
                'features': feature_indices
            }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with composite scaling"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        X = self._clip_outliers(X, fit_mode=False)
        
        X_scaled = X.copy()
        
        for group_name, group_config in self.fitted_groups_.items():
            scaler = group_config['scaler']
            feature_indices = group_config['features']
            
            # Transform selected features
            X_group = X_scaled[:, feature_indices]
            X_scaled[:, feature_indices] = scaler.transform(X_group)
        
        return X_scaled

# ============================================
# Utility Functions
# ============================================

@time_it("scaling_transformation")
def create_financial_scaler(X: Union[pd.DataFrame, np.ndarray],
                           feature_names: Optional[List[str]] = None,
                           scaler_type: str = 'adaptive',
                           **kwargs) -> BaseScaler:
    """
    Create appropriate scaler for financial data
    
    Args:
        X: Input feature matrix
        feature_names: Names of features
        scaler_type: Type of scaler ('standard', 'robust', 'minmax', 'quantile', 'adaptive')
        **kwargs: Additional arguments for scaler
        
    Returns:
        Fitted scaler instance
    """
    
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or X.columns.tolist()
        X = X.values
    
    config = ScalerConfig(
        feature_names=feature_names,
        **{k: v for k, v in kwargs.items() if k in ScalerConfig.__dataclass_fields__}
    )
    
    if scaler_type == 'standard':
        scaler = FinancialStandardScaler(config=config, **kwargs)
    elif scaler_type == 'robust':
        scaler = FinancialRobustScaler(config=config, **kwargs)
    elif scaler_type == 'minmax':
        scaler = FinancialMinMaxScaler(config=config, **kwargs)
    elif scaler_type == 'quantile':
        scaler = FinancialQuantileScaler(config=config, **kwargs)
    elif scaler_type == 'log':
        scaler = LogScaler(config=config, **kwargs)
    elif scaler_type == 'boxcox':
        scaler = BoxCoxScaler(config=config, **kwargs)
    elif scaler_type == 'timeseries':
        scaler = TimeSeriesScaler(config=config, **kwargs)
    elif scaler_type == 'adaptive':
        scaler = AdaptiveScaler(config=config, **kwargs)
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    scaler.fit(X)
    logger.info(f"Created and fitted {scaler_type} scaler for {X.shape[1]} features")
    return scaler

def analyze_feature_distributions(X: Union[pd.DataFrame, np.ndarray],
                                feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Analyze feature distributions to recommend scaling methods"""
    
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or X.columns.tolist()
        X = X.values
    elif feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    analysis_results = []
    
    for i, feature_name in enumerate(feature_names):
        feature_data = X[:, i]
        
        # Basic statistics
        mean_val = np.mean(feature_data)
        std_val = np.std(feature_data)
        median_val = np.median(feature_data)
        min_val = np.min(feature_data)
        max_val = np.max(feature_data)
        
        # Distribution tests
        skewness = stats.skew(feature_data)
        kurtosis = stats.kurtosis(feature_data)
        
        try:
            _, normality_p = stats.normaltest(feature_data)
        except:
            normality_p = np.nan
        
        # Outlier detection
        q75, q25 = np.percentile(feature_data, [75, 25])
        iqr = q75 - q25
        outlier_count = np.sum((feature_data < (q25 - 1.5 * iqr)) | 
                              (feature_data > (q75 + 1.5 * iqr)))
        outlier_ratio = outlier_count / len(feature_data)
        
        # Recommend scaler
        all_positive = np.all(feature_data > 0)
        
        if all_positive and abs(skewness) > 2:
            recommended_scaler = 'log'
        elif outlier_ratio > 0.1:
            recommended_scaler = 'robust'
        elif normality_p < 0.05:
            recommended_scaler = 'quantile'
        else:
            recommended_scaler = 'standard'
        
        analysis_results.append({
            'feature': feature_name,
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'min': min_val,
            'max': max_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normality_p': normality_p,
            'outlier_ratio': outlier_ratio,
            'all_positive': all_positive,
            'recommended_scaler': recommended_scaler
        })
    
    return pd.DataFrame(analysis_results)

def compare_scaling_methods(X: Union[pd.DataFrame, np.ndarray],
                           y: Optional[np.ndarray] = None,
                           methods: List[str] = None) -> Dict[str, Dict[str, float]]:
    """Compare different scaling methods on data quality metrics"""
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    methods = methods or ['standard', 'robust', 'minmax', 'quantile']
    results = {}
    
    for method in methods:
        try:
            scaler = create_financial_scaler(X, scaler_type=method)
            X_scaled = scaler.transform(X)
            
            # Calculate metrics
            metrics = {
                'mean_abs_skewness': np.mean(np.abs([stats.skew(X_scaled[:, i]) 
                                                   for i in range(X_scaled.shape[1])])),
                'mean_abs_kurtosis': np.mean(np.abs([stats.kurtosis(X_scaled[:, i]) 
                                                   for i in range(X_scaled.shape[1])])),
                'mean_std': np.mean([np.std(X_scaled[:, i]) for i in range(X_scaled.shape[1])]),
                'max_range': np.max([np.ptp(X_scaled[:, i]) for i in range(X_scaled.shape[1])]),
                'condition_number': np.linalg.cond(X_scaled) if X_scaled.shape[0] >= X_scaled.shape[1] else np.inf
            }
            
            results[method] = metrics
            
        except Exception as e:
            logger.warning(f"Error evaluating {method} scaler: {e}")
            results[method] = {'error': str(e)}
    
    return results

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Financial Scalers")
    
    # Create sample financial data with different characteristics
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic financial features
    prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, n_samples))  # Price series
    returns = np.diff(np.log(prices))  # Log returns
    volume = np.random.lognormal(10, 1, n_samples)  # Log-normal volume
    volatility = np.abs(np.random.normal(0.2, 0.05, n_samples))  # Volatility
    rsi = 50 + 30 * np.tanh(np.cumsum(returns[:n_samples-1]) * 10)  # RSI-like indicator
    
    X = np.column_stack([prices[1:], returns, volume[1:], volatility[1:], rsi])
    feature_names = ['price', 'returns', 'volume', 'volatility', 'rsi']
    
    print(f"Original data shape: {X.shape}")
    print(f"Original data ranges: {[(name, X[:, i].min(), X[:, i].max()) for i, name in enumerate(feature_names)]}")
    
    # Test distribution analysis
    print("\n1. Testing Feature Distribution Analysis")
    dist_analysis = analyze_feature_distributions(X, feature_names)
    print("Distribution analysis results:")
    print(dist_analysis[['feature', 'skewness', 'outlier_ratio', 'recommended_scaler']])
    
    # Test adaptive scaler
    print("\n2. Testing Adaptive Scaler")
    adaptive_scaler = AdaptiveScaler(auto_select=True)
    X_adaptive = adaptive_scaler.fit_transform(X)
    selected_scalers = adaptive_scaler.get_selected_scalers()
    print("Selected scalers for each feature:")
    for i, (name, scaler_type) in enumerate(zip(feature_names, selected_scalers.values())):
        print(f"  {name}: {scaler_type}")
    
    # Test financial-specific scalers
    print("\n3. Testing Financial Standard Scaler")
    financial_scaler = FinancialStandardScaler(
        rolling_window=252,  # 1 year rolling window
        handle_returns=True,
        return_features=[1]  # Returns column
    )
    X_financial = financial_scaler.fit_transform(X)
    print(f"Financial scaled ranges: {[(name, X_financial[:, i].min(), X_financial[:, i].max()) for i, name in enumerate(feature_names)]}")
    
    # Test time series scaler
    print("\n4. Testing Time Series Scaler")
    ts_scaler = TimeSeriesScaler(
        window_type='rolling',
        window_size=60,
        scaling_method='standard'
    )
    X_timeseries = ts_scaler.fit_transform(X)
    print(f"Time series scaled shape: {X_timeseries.shape}")
    
    # Test composite scaler
    print("\n5. Testing Composite Scaler")
    scaler_groups = {
        'prices': {
            'scaler': LogScaler(),
            'features': [0, 2]  # price and volume
        },
        'indicators': {
            'scaler': FinancialStandardScaler(),
            'features': [1, 3, 4]  # returns, volatility, rsi
        }
    }
    
    composite_scaler = CompositeScaler(scaler_groups)
    X_composite = composite_scaler.fit_transform(X)
    print(f"Composite scaled shape: {X_composite.shape}")
    
    # Test Box-Cox scaler
    print("\n6. Testing Box-Cox Scaler")
    # Use only positive features for Box-Cox
    X_positive = X[:, [0, 2, 3]]  # price, volume, volatility
    boxcox_scaler = BoxCoxScaler()
    X_boxcox = boxcox_scaler.fit_transform(X_positive)
    print(f"Box-Cox lambdas: {boxcox_scaler.lambdas_}")
    
    # Compare scaling methods
    print("\n7. Comparing Scaling Methods")
    comparison = compare_scaling_methods(X, methods=['standard', 'robust', 'minmax', 'quantile', 'adaptive'])
    print("Scaling method comparison:")
    for method, metrics in comparison.items():
        if 'error' not in metrics:
            print(f"  {method}: skewness={metrics['mean_abs_skewness']:.3f}, "
                  f"std={metrics['mean_std']:.3f}, condition_number={metrics['condition_number']:.2f}")
    
    # Test utility function
    print("\n8. Testing Utility Function")
    auto_scaler = create_financial_scaler(X, feature_names, scaler_type='adaptive', 
                                         clip_outliers=True, outlier_method='iqr')
    X_auto = auto_scaler.transform(X)
    print(f"Auto scaler created and applied: {X_auto.shape}")
    
    print("\nFinancial scalers testing completed successfully!")
