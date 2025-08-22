# ============================================
# StockPredictionPro - src/data/processors/transformer.py
# Advanced data transformation pipeline for financial time series
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    PowerTransformer, KBinsDiscretizer
)
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_regression, f_classif, mutual_info_regression, mutual_info_classif
)
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from scipy.signal import savgol_filter
import warnings

from ...utils.exceptions import DataValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ...utils.config_loader import get
from ...utils.helpers import safe_divide
from ...utils.validators import ValidationResult

logger = get_logger('data.processors.transformer')

# ============================================
# Custom Transformers for Financial Data
# ============================================

class FinancialReturnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transform price data to returns with various methods
    
    Features:
    - Simple and log returns
    - Volatility scaling
    - Outlier handling
    - Missing value treatment
    """
    
    def __init__(self, 
                 method: str = 'simple',
                 periods: int = 1,
                 fillna_method: str = 'forward',
                 clip_outliers: bool = True,
                 outlier_threshold: float = 5.0):
        """
        Initialize returns transformer
        
        Args:
            method: 'simple', 'log', 'pct_change'
            periods: Number of periods for return calculation
            fillna_method: Method to handle missing values
            clip_outliers: Whether to clip extreme outliers
            outlier_threshold: Standard deviations for outlier threshold
        """
        self.method = method
        self.periods = periods
        self.fillna_method = fillna_method
        self.clip_outliers = clip_outliers
        self.outlier_threshold = outlier_threshold
        
        # Store statistics for inverse transform
        self.statistics_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer and calculate statistics"""
        X = self._validate_input(X)
        
        # Calculate base statistics for each column
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.statistics_[col] = {
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'median': X[col].median(),
                    'q95': X[col].quantile(0.95),
                    'q05': X[col].quantile(0.05)
                }
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform prices to returns"""
        X = self._validate_input(X)
        X_transformed = X.copy()
        
        # Apply returns transformation to price columns
        price_columns = self._identify_price_columns(X)
        
        for col in price_columns:
            if self.method == 'simple':
                X_transformed[col] = X[col].pct_change(periods=self.periods)
            elif self.method == 'log':
                X_transformed[col] = np.log(X[col] / X[col].shift(self.periods))
            elif self.method == 'pct_change':
                X_transformed[col] = X[col].pct_change(periods=self.periods)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        # Handle missing values
        if self.fillna_method == 'forward':
            X_transformed = X_transformed.fillna(method='ffill')
        elif self.fillna_method == 'backward':
            X_transformed = X_transformed.fillna(method='bfill')
        elif self.fillna_method == 'zero':
            X_transformed = X_transformed.fillna(0)
        elif self.fillna_method == 'drop':
            X_transformed = X_transformed.dropna()
        
        # Clip outliers if requested
        if self.clip_outliers:
            for col in price_columns:
                if col in self.statistics_:
                    stats = self.statistics_[col]
                    lower_bound = stats['mean'] - self.outlier_threshold * stats['std']
                    upper_bound = stats['mean'] + self.outlier_threshold * stats['std']
                    X_transformed[col] = X_transformed[col].clip(lower_bound, upper_bound)
        
        return X_transformed
    
    def _identify_price_columns(self, X: pd.DataFrame) -> List[str]:
        """Identify price columns for transformation"""
        price_keywords = ['open', 'high', 'low', 'close', 'price', 'adj', 'volume']
        price_columns = []
        
        for col in X.columns:
            if any(keyword in col.lower() for keyword in price_keywords):
                if pd.api.types.is_numeric_dtype(X[col]) and (X[col] > 0).all():
                    price_columns.append(col)
        
        return price_columns
    
    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input data"""
        if not isinstance(X, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise DataValidationError("Input DataFrame is empty")
        
        return X

class TechnicalIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Generate technical indicators from OHLCV data
    
    Features:
    - 20+ common technical indicators
    - Configurable parameters
    - Automatic feature naming
    - Missing value handling
    """
    
    def __init__(self, 
                 indicators: Optional[List[str]] = None,
                 periods: Optional[Dict[str, List[int]]] = None,
                 fillna_method: str = 'forward'):
        """
        Initialize technical indicator transformer
        
        Args:
            indicators: List of indicators to calculate
            periods: Dictionary of periods for each indicator
            fillna_method: Method to handle missing values
        """
        self.indicators = indicators or [
            'sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'stochastic'
        ]
        self.periods = periods or {
            'sma': [10, 20, 50],
            'ema': [12, 26],
            'rsi': [14],
            'macd': [(12, 26, 9)],
            'bollinger': [20],
            'atr': [14],
            'stochastic': [14]
        }
        self.fillna_method = fillna_method
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no parameters to learn)"""
        self._validate_ohlcv_data(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators"""
        X = X.copy()
        self._validate_ohlcv_data(X)
        
        # Generate indicators
        for indicator in self.indicators:
            if indicator == 'sma':
                X = self._add_sma(X)
            elif indicator == 'ema':
                X = self._add_ema(X)
            elif indicator == 'rsi':
                X = self._add_rsi(X)
            elif indicator == 'macd':
                X = self._add_macd(X)
            elif indicator == 'bollinger':
                X = self._add_bollinger_bands(X)
            elif indicator == 'atr':
                X = self._add_atr(X)
            elif indicator == 'stochastic':
                X = self._add_stochastic(X)
            elif indicator == 'williams_r':
                X = self._add_williams_r(X)
            elif indicator == 'cci':
                X = self._add_cci(X)
            elif indicator == 'momentum':
                X = self._add_momentum(X)
        
        # Handle missing values
        if self.fillna_method == 'forward':
            X = X.fillna(method='ffill')
        elif self.fillna_method == 'backward':
            X = X.fillna(method='bfill')
        elif self.fillna_method == 'drop':
            X = X.dropna()
        
        return X
    
    def _add_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple Moving Averages"""
        for period in self.periods.get('sma', [20]):
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        return df
    
    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Exponential Moving Averages"""
        for period in self.periods.get('ema', [12, 26]):
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        return df
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Relative Strength Index"""
        for period in self.periods.get('rsi', [14]):
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = safe_divide(avg_gain, avg_loss)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator"""
        for params in self.periods.get('macd', [(12, 26, 9)]):
            fast, slow, signal = params
            
            ema_fast = df['Close'].ewm(span=fast).mean()
            ema_slow = df['Close'].ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            df[f'MACD_{fast}_{slow}'] = macd_line
            df[f'MACD_Signal_{fast}_{slow}_{signal}'] = signal_line
            df[f'MACD_Histogram_{fast}_{slow}_{signal}'] = histogram
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands"""
        for period in self.periods.get('bollinger', [20]):
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            
            df[f'BB_Upper_{period}'] = sma + (2 * std)
            df[f'BB_Lower_{period}'] = sma - (2 * std)
            df[f'BB_Middle_{period}'] = sma
            df[f'BB_Width_{period}'] = df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']
            df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / df[f'BB_Width_{period}']
        return df
    
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        for period in self.periods.get('atr', [14]):
            df[f'ATR_{period}'] = true_range.rolling(window=period).mean()
        return df
    
    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        for period in self.periods.get('stochastic', [14]):
            lowest_low = df['Low'].rolling(window=period).min()
            highest_high = df['High'].rolling(window=period).max()
            
            k_percent = 100 * safe_divide(
                (df['Close'] - lowest_low),
                (highest_high - lowest_low)
            )
            
            df[f'Stoch_K_{period}'] = k_percent
            df[f'Stoch_D_{period}'] = k_percent.rolling(window=3).mean()
        return df
    
    def _add_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Williams %R"""
        for period in self.periods.get('williams_r', [14]):
            highest_high = df['High'].rolling(window=period).max()
            lowest_low = df['Low'].rolling(window=period).min()
            
            williams_r = -100 * safe_divide(
                (highest_high - df['Close']),
                (highest_high - lowest_low)
            )
            
            df[f'Williams_R_{period}'] = williams_r
        return df
    
    def _add_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Commodity Channel Index"""
        for period in self.periods.get('cci', [20]):
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )
            
            df[f'CCI_{period}'] = safe_divide(
                (typical_price - sma_tp),
                (0.015 * mean_deviation)
            )
        return df
    
    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Momentum indicators"""
        for period in self.periods.get('momentum', [10]):
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'ROC_{period}'] = df['Close'].pct_change(periods=period)
        return df
    
    def _validate_ohlcv_data(self, df: pd.DataFrame):
        """Validate OHLCV data format"""
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise DataValidationError(f"Missing required OHLCV columns: {missing_columns}")

class LaggedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Create lagged features for time series data
    
    Features:
    - Multiple lag periods
    - Rolling window statistics
    - Difference features
    - Custom aggregation functions
    """
    
    def __init__(self,
                 lag_periods: List[int] = [1, 2, 3, 5, 10],
                 rolling_windows: List[int] = [5, 10, 20],
                 difference_periods: List[int] = [1, 5],
                 feature_columns: Optional[List[str]] = None,
                 rolling_functions: List[str] = ['mean', 'std', 'min', 'max']):
        """
        Initialize lagged features transformer
        
        Args:
            lag_periods: Periods for lag features
            rolling_windows: Windows for rolling statistics
            difference_periods: Periods for difference features
            feature_columns: Columns to create lagged features for
            rolling_functions: Statistical functions for rolling windows
        """
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.difference_periods = difference_periods
        self.feature_columns = feature_columns
        self.rolling_functions = rolling_functions
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer"""
        if self.feature_columns is None:
            # Auto-detect numeric columns
            self.feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features"""
        X_transformed = X.copy()
        
        for col in self.feature_columns:
            if col not in X.columns:
                continue
                
            # Lag features
            for lag in self.lag_periods:
                X_transformed[f'{col}_lag_{lag}'] = X[col].shift(lag)
            
            # Rolling window features
            for window in self.rolling_windows:
                for func in self.rolling_functions:
                    if func == 'mean':
                        X_transformed[f'{col}_rolling_{window}_mean'] = X[col].rolling(window).mean()
                    elif func == 'std':
                        X_transformed[f'{col}_rolling_{window}_std'] = X[col].rolling(window).std()
                    elif func == 'min':
                        X_transformed[f'{col}_rolling_{window}_min'] = X[col].rolling(window).min()
                    elif func == 'max':
                        X_transformed[f'{col}_rolling_{window}_max'] = X[col].rolling(window).max()
                    elif func == 'median':
                        X_transformed[f'{col}_rolling_{window}_median'] = X[col].rolling(window).median()
                    elif func == 'skew':
                        X_transformed[f'{col}_rolling_{window}_skew'] = X[col].rolling(window).skew()
                    elif func == 'kurt':
                        X_transformed[f'{col}_rolling_{window}_kurt'] = X[col].rolling(window).kurt()
            
            # Difference features
            for period in self.difference_periods:
                X_transformed[f'{col}_diff_{period}'] = X[col].diff(periods=period)
                X_transformed[f'{col}_pct_change_{period}'] = X[col].pct_change(periods=period)
        
        return X_transformed

class VolatilityTransformer(BaseEstimator, TransformerMixin):
    """
    Transform data based on volatility characteristics
    
    Features:
    - GARCH-style volatility modeling
    - Volatility regime detection
    - Risk-adjusted features
    - Volatility clustering detection
    """
    
    def __init__(self,
                 volatility_window: int = 20,
                 volatility_method: str = 'rolling_std',
                 regime_threshold: float = 2.0,
                 adjust_for_volatility: bool = True):
        """
        Initialize volatility transformer
        
        Args:
            volatility_window: Window for volatility calculation
            volatility_method: Method for volatility estimation
            regime_threshold: Threshold for regime detection
            adjust_for_volatility: Whether to create volatility-adjusted features
        """
        self.volatility_window = volatility_window
        self.volatility_method = volatility_method
        self.regime_threshold = regime_threshold
        self.adjust_for_volatility = adjust_for_volatility
        
        self.volatility_stats_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit volatility statistics"""
        # Calculate volatility for return columns
        return_columns = self._identify_return_columns(X)
        
        for col in return_columns:
            volatility = self._calculate_volatility(X[col])
            self.volatility_stats_[col] = {
                'mean_volatility': volatility.mean(),
                'std_volatility': volatility.std(),
                'high_vol_threshold': volatility.quantile(0.75),
                'low_vol_threshold': volatility.quantile(0.25)
            }
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features"""
        X_transformed = X.copy()
        
        return_columns = self._identify_return_columns(X)
        
        for col in return_columns:
            # Calculate volatility
            volatility = self._calculate_volatility(X[col])
            X_transformed[f'{col}_volatility'] = volatility
            
            # Volatility regime indicators
            if col in self.volatility_stats_:
                stats = self.volatility_stats_[col]
                X_transformed[f'{col}_high_vol_regime'] = (
                    volatility > stats['high_vol_threshold']
                ).astype(int)
                X_transformed[f'{col}_low_vol_regime'] = (
                    volatility < stats['low_vol_threshold']
                ).astype(int)
            
            # Volatility-adjusted returns
            if self.adjust_for_volatility:
                X_transformed[f'{col}_vol_adjusted'] = safe_divide(X[col], volatility)
            
            # Volatility clustering (persistence)
            vol_change = volatility.pct_change()
            X_transformed[f'{col}_vol_clustering'] = (
                vol_change.rolling(5).std()
            )
        
        return X_transformed
    
    def _calculate_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculate volatility using specified method"""
        if self.volatility_method == 'rolling_std':
            return returns.rolling(self.volatility_window).std()
        elif self.volatility_method == 'ewm_std':
            return returns.ewm(span=self.volatility_window).std()
        elif self.volatility_method == 'parkinson':
            # Parkinson estimator (requires High/Low data)
            # This is a placeholder - would need OHLC data
            return returns.rolling(self.volatility_window).std()
        else:
            return returns.rolling(self.volatility_window).std()
    
    def _identify_return_columns(self, X: pd.DataFrame) -> List[str]:
        """Identify return columns for volatility calculation"""
        return_keywords = ['return', 'pct_change', 'log_return']
        return_columns = []
        
        for col in X.columns:
            if any(keyword in col.lower() for keyword in return_keywords):
                return_columns.append(col)
            elif 'close' in col.lower() and X[col].dtype in [np.float64, np.float32]:
                # Assume close price columns might be returns
                if X[col].abs().max() < 1:  # Likely returns if values are small
                    return_columns.append(col)
        
        return return_columns

class ScalingTransformer(BaseEstimator, TransformerMixin):
    """
    Advanced scaling transformer with multiple methods
    
    Features:
    - Multiple scaling methods
    - Robust scaling for outliers
    - Time-aware scaling
    - Feature-specific scaling
    """
    
    def __init__(self,
                 method: str = 'robust',
                 feature_range: Tuple[float, float] = (0, 1),
                 quantile_range: Tuple[float, float] = (25.0, 75.0),
                 per_feature_scaling: bool = True):
        """
        Initialize scaling transformer
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust', 'quantile', 'power')
            feature_range: Range for MinMax scaling
            quantile_range: Quantile range for robust scaling
            per_feature_scaling: Whether to scale each feature separately
        """
        self.method = method
        self.feature_range = feature_range
        self.quantile_range = quantile_range
        self.per_feature_scaling = per_feature_scaling
        
        self.scalers_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit scalers for each feature or globally"""
        if self.per_feature_scaling:
            for col in X.select_dtypes(include=[np.number]).columns:
                self.scalers_[col] = self._create_scaler()
                # Reshape for sklearn compatibility
                data_reshaped = X[col].values.reshape(-1, 1)
                self.scalers_[col].fit(data_reshaped)
        else:
            self.scalers_['global'] = self._create_scaler()
            numeric_data = X.select_dtypes(include=[np.number])
            self.scalers_['global'].fit(numeric_data)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scalers"""
        X_transformed = X.copy()
        
        if self.per_feature_scaling:
            for col in X.select_dtypes(include=[np.number]).columns:
                if col in self.scalers_:
                    # Reshape for sklearn compatibility
                    data_reshaped = X[col].values.reshape(-1, 1)
                    scaled_data = self.scalers_[col].transform(data_reshaped)
                    X_transformed[col] = scaled_data.flatten()
        else:
            if 'global' in self.scalers_:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                scaled_data = self.scalers_['global'].transform(X[numeric_cols])
                X_transformed[numeric_cols] = scaled_data
        
        return X_transformed
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled features"""
        X_inverse = X.copy()
        
        if self.per_feature_scaling:
            for col in X.select_dtypes(include=[np.number]).columns:
                if col in self.scalers_:
                    data_reshaped = X[col].values.reshape(-1, 1)
                    inverse_data = self.scalers_[col].inverse_transform(data_reshaped)
                    X_inverse[col] = inverse_data.flatten()
        else:
            if 'global' in self.scalers_:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                inverse_data = self.scalers_['global'].inverse_transform(X[numeric_cols])
                X_inverse[numeric_cols] = inverse_data
        
        return X_inverse
    
    def _create_scaler(self):
        """Create scaler based on method"""
        if self.method == 'standard':
            return StandardScaler()
        elif self.method == 'minmax':
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.method == 'robust':
            return RobustScaler(quantile_range=self.quantile_range)
        elif self.method == 'quantile':
            return QuantileTransformer(output_distribution='uniform')
        elif self.method == 'power':
            return PowerTransformer(method='yeo-johnson')
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

# ============================================
# Feature Selection Transformers
# ============================================

class FinancialFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Financial-specific feature selection
    
    Features:
    - Statistical feature selection
    - Correlation-based filtering
    - Information-theoretic selection
    - Domain knowledge filtering
    """
    
    def __init__(self,
                 method: str = 'mutual_info',
                 k_features: int = 20,
                 correlation_threshold: float = 0.95,
                 variance_threshold: float = 0.01,
                 remove_constant: bool = True):
        """
        Initialize feature selector
        
        Args:
            method: Selection method ('mutual_info', 'f_score', 'rfe', 'correlation')
            k_features: Number of features to select
            correlation_threshold: Threshold for correlation filtering
            variance_threshold: Minimum variance threshold
            remove_constant: Whether to remove constant features
        """
        self.method = method
        self.k_features = k_features
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.remove_constant = remove_constant
        
        self.selector_ = None
        self.selected_features_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit feature selector"""
        # Remove constant and low-variance features
        X_filtered = self._remove_low_variance_features(X)
        
        # Remove highly correlated features
        X_filtered = self._remove_correlated_features(X_filtered)
        
        # Apply statistical feature selection
        if self.method == 'mutual_info':
            # Determine if classification or regression
            if y.dtype == 'object' or len(y.unique()) < 10:
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
            
            self.selector_ = SelectKBest(score_func=score_func, k=min(self.k_features, X_filtered.shape[1]))
        
        elif self.method == 'f_score':
            if y.dtype == 'object' or len(y.unique()) < 10:
                score_func = f_classif
            else:
                score_func = f_regression
            
            self.selector_ = SelectKBest(score_func=score_func, k=min(self.k_features, X_filtered.shape[1]))
        
        elif self.method == 'percentile':
            if y.dtype == 'object' or len(y.unique()) < 10:
                score_func = f_classif
            else:
                score_func = f_regression
            
            percentile = min(100, (self.k_features / X_filtered.shape[1]) * 100)
            self.selector_ = SelectPercentile(score_func=score_func, percentile=percentile)
        
        # Fit selector
        if self.selector_ is not None:
            self.selector_.fit(X_filtered, y)
            
            # Get selected feature names
            selected_mask = self.selector_.get_support()
            self.selected_features_ = X_filtered.columns[selected_mask].tolist()
        else:
            self.selected_features_ = X_filtered.columns.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using selected features"""
        if self.selected_features_ is None:
            return X
        
        # Return only selected features
        available_features = [col for col in self.selected_features_ if col in X.columns]
        return X[available_features]
    
    def _remove_low_variance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with low variance"""
        if not self.remove_constant:
            return X
        
        # Calculate variance for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        variances = X[numeric_cols].var()
        
        # Remove low-variance features
        high_variance_cols = variances[variances > self.variance_threshold].index
        
        # Keep non-numeric columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        selected_cols = list(high_variance_cols) + list(non_numeric_cols)
        
        return X[selected_cols]
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        numeric_data = X.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return X
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns 
                   if any(upper_tri[column] > self.correlation_threshold)]
        
        # Keep non-numeric columns and uncorrelated numeric columns
        cols_to_keep = [col for col in X.columns if col not in to_drop]
        
        return X[cols_to_keep]

# ============================================
# Pipeline Transformer
# ============================================

class FinancialTransformationPipeline(BaseEstimator, TransformerMixin):
    """
    Complete transformation pipeline for financial data
    
    Features:
    - Configurable transformation steps
    - Automatic pipeline construction
    - Error handling and logging
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize transformation pipeline
        
        Args:
            config: Configuration dictionary for transformations
        """
        self.config = config or self._get_default_config()
        self.transformers_ = []
        self.feature_names_ = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default transformation configuration"""
        return {
            'returns_transform': {
                'enabled': True,
                'method': 'simple',
                'periods': 1
            },
            'technical_indicators': {
                'enabled': True,
                'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger']
            },
            'lagged_features': {
                'enabled': True,
                'lag_periods': [1, 2, 3, 5],
                'rolling_windows': [5, 10, 20]
            },
            'volatility_features': {
                'enabled': True,
                'volatility_window': 20
            },
            'scaling': {
                'enabled': True,
                'method': 'robust'
            },
            'feature_selection': {
                'enabled': True,
                'method': 'mutual_info',
                'k_features': 30
            }
        }
    
    @time_it("financial_pipeline_fit")
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the transformation pipeline"""
        logger.info("Fitting financial transformation pipeline...")
        
        # Build pipeline based on configuration
        self._build_pipeline()
        
        # Fit each transformer
        X_current = X.copy()
        
        for name, transformer in self.transformers_:
            try:
                with Timer(f"fit_{name}") as timer:
                    if name == 'feature_selection' and y is not None:
                        transformer.fit(X_current, y)
                    else:
                        transformer.fit(X_current)
                
                logger.debug(f"Fitted {name} in {timer.result.duration_str}")
                
                # Transform for next step (except for scaling which should be last)
                if name != 'scaling' and name != 'feature_selection':
                    X_current = transformer.transform(X_current)
                    
            except Exception as e:
                logger.error(f"Error fitting {name}: {e}")
                raise
        
        # Store final feature names
        if hasattr(X_current, 'columns'):
            self.feature_names_ = X_current.columns.tolist()
        
        logger.info(f"Pipeline fitted successfully with {len(self.transformers_)} transformers")
        return self
    
    @time_it("financial_pipeline_transform")
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline"""
        X_transformed = X.copy()
        
        # Apply each transformer
        for name, transformer in self.transformers_:
            try:
                with Timer(f"transform_{name}") as timer:
                    X_transformed = transformer.transform(X_transformed)
                
                logger.debug(f"Applied {name} transformation in {timer.result.duration_str}")
                
            except Exception as e:
                logger.error(f"Error applying {name}: {e}")
                raise
        
        return X_transformed
    
    def _build_pipeline(self):
        """Build transformation pipeline based on configuration"""
        self.transformers_ = []
        
        # 1. Returns transformation
        if self.config.get('returns_transform', {}).get('enabled', False):
            returns_config = self.config['returns_transform']
            transformer = FinancialReturnsTransformer(
                method=returns_config.get('method', 'simple'),
                periods=returns_config.get('periods', 1)
            )
            self.transformers_.append(('returns_transform', transformer))
        
        # 2. Technical indicators
        if self.config.get('technical_indicators', {}).get('enabled', False):
            indicators_config = self.config['technical_indicators']
            transformer = TechnicalIndicatorTransformer(
                indicators=indicators_config.get('indicators', ['sma', 'ema', 'rsi'])
            )
            self.transformers_.append(('technical_indicators', transformer))
        
        # 3. Lagged features
        if self.config.get('lagged_features', {}).get('enabled', False):
            lagged_config = self.config['lagged_features']
            transformer = LaggedFeaturesTransformer(
                lag_periods=lagged_config.get('lag_periods', [1, 2, 3]),
                rolling_windows=lagged_config.get('rolling_windows', [5, 10])
            )
            self.transformers_.append(('lagged_features', transformer))
        
        # 4. Volatility features
        if self.config.get('volatility_features', {}).get('enabled', False):
            vol_config = self.config['volatility_features']
            transformer = VolatilityTransformer(
                volatility_window=vol_config.get('volatility_window', 20)
            )
            self.transformers_.append(('volatility_features', transformer))
        
        # 5. Feature selection (before scaling)
        if self.config.get('feature_selection', {}).get('enabled', False):
            selection_config = self.config['feature_selection']
            transformer = FinancialFeatureSelector(
                method=selection_config.get('method', 'mutual_info'),
                k_features=selection_config.get('k_features', 30)
            )
            self.transformers_.append(('feature_selection', transformer))
        
        # 6. Scaling (should be last)
        if self.config.get('scaling', {}).get('enabled', False):
            scaling_config = self.config['scaling']
            transformer = ScalingTransformer(
                method=scaling_config.get('method', 'robust')
            )
            self.transformers_.append(('scaling', transformer))
    
    def get_feature_names(self) -> List[str]:
        """Get names of output features"""
        return self.feature_names_ or []
    
    def get_transformer_info(self) -> Dict[str, Any]:
        """Get information about fitted transformers"""
        info = {}
        
        for name, transformer in self.transformers_:
            info[name] = {
                'type': type(transformer).__name__,
                'fitted': hasattr(transformer, 'statistics_') or hasattr(transformer, 'scalers_'),
                'parameters': transformer.get_params() if hasattr(transformer, 'get_params') else {}
            }
        
        return info

# ============================================
# Factory Functions
# ============================================

def create_returns_transformer(method: str = 'simple', **kwargs) -> FinancialReturnsTransformer:
    """Create returns transformer with common configurations"""
    return FinancialReturnsTransformer(method=method, **kwargs)

def create_technical_indicators_transformer(indicators: List[str] = None, **kwargs) -> TechnicalIndicatorTransformer:
    """Create technical indicators transformer with common configurations"""
    if indicators is None:
        indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr']
    
    return TechnicalIndicatorTransformer(indicators=indicators, **kwargs)

def create_lagged_features_transformer(max_lag: int = 5, **kwargs) -> LaggedFeaturesTransformer:
    """Create lagged features transformer with common configurations"""
    lag_periods = list(range(1, max_lag + 1))
    return LaggedFeaturesTransformer(lag_periods=lag_periods, **kwargs)

def create_financial_pipeline(pipeline_type: str = 'comprehensive', **kwargs) -> FinancialTransformationPipeline:
    """
    Create pre-configured financial transformation pipeline
    
    Args:
        pipeline_type: Type of pipeline ('basic', 'comprehensive', 'minimal')
        
    Returns:
        Configured FinancialTransformationPipeline
    """
    if pipeline_type == 'basic':
        config = {
            'technical_indicators': {'enabled': True, 'indicators': ['sma', 'ema', 'rsi']},
            'scaling': {'enabled': True, 'method': 'robust'}
        }
    elif pipeline_type == 'comprehensive':
        config = {
            'returns_transform': {'enabled': True, 'method': 'simple'},
            'technical_indicators': {'enabled': True, 'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr']},
            'lagged_features': {'enabled': True, 'lag_periods': [1, 2, 3, 5], 'rolling_windows': [5, 10, 20]},
            'volatility_features': {'enabled': True, 'volatility_window': 20},
            'feature_selection': {'enabled': True, 'method': 'mutual_info', 'k_features': 30},
            'scaling': {'enabled': True, 'method': 'robust'}
        }
    elif pipeline_type == 'minimal':
        config = {
            'technical_indicators': {'enabled': True, 'indicators': ['sma', 'rsi']},
            'scaling': {'enabled': True, 'method': 'standard'}
        }
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    # Override with any provided kwargs
    config.update(kwargs)
    
    return FinancialTransformationPipeline(config=config)
