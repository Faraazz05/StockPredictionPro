# ============================================
# StockPredictionPro - src/features/targets/regression.py
# Advanced target engineering for financial regression tasks
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('features.targets.regression')

# ============================================
# Configuration and Enums
# ============================================

class RegressionTarget(Enum):
    """Types of financial regression targets"""
    RETURNS = "returns"                        # Price returns (absolute or log)
    VOLATILITY = "volatility"                  # Realized volatility
    PRICE = "price"                           # Future price levels
    SPREAD = "spread"                         # Bid-ask spreads
    VOLUME = "volume"                         # Trading volume
    DRAWDOWN = "drawdown"                     # Maximum drawdown
    SHARPE_RATIO = "sharpe_ratio"            # Risk-adjusted returns
    VAR = "var"                              # Value at Risk
    CORRELATION = "correlation"               # Rolling correlations
    MOMENTUM = "momentum"                     # Momentum indicators
    MEAN_REVERSION = "mean_reversion"        # Mean reversion strength
    VOLATILITY_SURFACE = "volatility_surface" # Implied volatility surfaces

@dataclass
class RegressionConfig:
    """Configuration for regression target engineering"""
    target_type: RegressionTarget = RegressionTarget.RETURNS
    lookahead_periods: int = 1
    return_type: str = 'simple'  # 'simple', 'log', 'excess'
    scaling_method: str = 'none'  # 'none', 'standard', 'robust', 'minmax', 'rank'
    outlier_treatment: str = 'none'  # 'none', 'winsorize', 'clip', 'remove'
    outlier_threshold: float = 3.0
    smoothing_window: Optional[int] = None  # Rolling average smoothing
    handle_missing: str = 'drop'  # 'drop', 'interpolate', 'forward_fill'
    transformation: str = 'none'  # 'none', 'log', 'sqrt', 'box_cox', 'yeo_johnson'
    volatility_window: int = 20
    risk_free_rate: float = 0.02  # Annual risk-free rate
    confidence_level: float = 0.05  # For VaR calculations
    
    def __post_init__(self):
        if self.lookahead_periods < 1:
            raise ValueError("lookahead_periods must be at least 1")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

# ============================================
# Base Regression Target Creator
# ============================================

class BaseRegressionTarget:
    """Base class for all regression target creators"""
    
    def __init__(self, config: Optional[RegressionConfig] = None):
        self.config = config or RegressionConfig()
        self.scaler_ = None
        self.target_stats_ = None
        self.outlier_bounds_ = None
        self.is_fitted_ = False
    
    def _validate_input(self, data):
        """Validate input data"""
        if isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError("DataFrame must have exactly one column for target creation")
            return data.iloc[:, 0].values
        elif isinstance(data, np.ndarray):
            if data.ndim > 1 and data.shape[1] != 1:
                raise ValueError("Array must be 1-dimensional or have shape (n, 1)")
            return data.flatten()
        else:
            return np.asarray(data).flatten()
    
    def _handle_missing_values(self, data):
        """Handle missing values in target data"""
        if self.config.handle_missing == 'drop':
            return data[~pd.isna(data)]
        elif self.config.handle_missing == 'interpolate':
            if isinstance(data, pd.Series):
                return data.interpolate().fillna(method='bfill').fillna(method='ffill')
            else:
                df = pd.Series(data)
                return df.interpolate().fillna(method='bfill').fillna(method='ffill').values
        elif self.config.handle_missing == 'forward_fill':
            if isinstance(data, pd.Series):
                return data.fillna(method='ffill')
            else:
                df = pd.Series(data)
                return df.fillna(method='ffill').values
        else:
            raise ValueError(f"Unknown handle_missing method: {self.config.handle_missing}")
    
    def _apply_transformation(self, data, fit_mode=True):
        """Apply mathematical transformations to data"""
        if self.config.transformation == 'none':
            return data
        
        elif self.config.transformation == 'log':
            # Ensure positive values for log transformation
            if np.any(data <= 0):
                logger.warning("Log transformation requires positive values, adding constant")
                min_val = np.min(data)
                data = data - min_val + 1e-8
            
            return np.log(data)
        
        elif self.config.transformation == 'sqrt':
            # Ensure non-negative values for sqrt transformation
            if np.any(data < 0):
                logger.warning("Sqrt transformation requires non-negative values, taking absolute value")
                data = np.abs(data)
            
            return np.sqrt(data)
        
        elif self.config.transformation == 'box_cox':
            from scipy.stats import boxcox
            if fit_mode:
                # Ensure positive values
                if np.any(data <= 0):
                    min_val = np.min(data)
                    data = data - min_val + 1e-8
                
                transformed_data, self.lambda_ = boxcox(data)
                return transformed_data
            else:
                # Use fitted lambda
                if hasattr(self, 'lambda_'):
                    if np.any(data <= 0):
                        min_val = np.min(data)
                        data = data - min_val + 1e-8
                    
                    if abs(self.lambda_) < 1e-6:
                        return np.log(data)
                    else:
                        return (np.power(data, self.lambda_) - 1) / self.lambda_
                else:
                    raise ValueError("Must fit transformation before applying")
        
        elif self.config.transformation == 'yeo_johnson':
            from scipy.stats import yeojohnson
            if fit_mode:
                transformed_data, self.lambda_ = yeojohnson(data)
                return transformed_data
            else:
                if hasattr(self, 'lambda_'):
                    return yeojohnson(data, lmbda=self.lambda_)
                else:
                    raise ValueError("Must fit transformation before applying")
        
        else:
            raise ValueError(f"Unknown transformation: {self.config.transformation}")
    
    def _handle_outliers(self, data, fit_mode=True):
        """Handle outliers in target data"""
        if self.config.outlier_treatment == 'none':
            return data
        
        if fit_mode:
            # Calculate outlier bounds
            if self.config.outlier_treatment in ['winsorize', 'clip']:
                mean_val = np.mean(data)
                std_val = np.std(data)
                lower_bound = mean_val - self.config.outlier_threshold * std_val
                upper_bound = mean_val + self.config.outlier_threshold * std_val
                self.outlier_bounds_ = (lower_bound, upper_bound)
            
            elif self.config.outlier_treatment == 'remove':
                # Use IQR method for removal
                q75, q25 = np.percentile(data, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - self.config.outlier_threshold * iqr
                upper_bound = q75 + self.config.outlier_threshold * iqr
                self.outlier_bounds_ = (lower_bound, upper_bound)
        
        # Apply outlier treatment
        if self.outlier_bounds_ is None:
            return data
        
        lower_bound, upper_bound = self.outlier_bounds_
        
        if self.config.outlier_treatment == 'winsorize':
            # Cap outliers at bounds
            return np.clip(data, lower_bound, upper_bound)
        
        elif self.config.outlier_treatment == 'clip':
            # Same as winsorize
            return np.clip(data, lower_bound, upper_bound)
        
        elif self.config.outlier_treatment == 'remove':
            # Remove outliers (only during fitting)
            if fit_mode:
                mask = (data >= lower_bound) & (data <= upper_bound)
                return data[mask]
            else:
                # During transform, clip instead of remove
                return np.clip(data, lower_bound, upper_bound)
        
        return data
    
    def _apply_smoothing(self, data):
        """Apply smoothing to reduce noise"""
        if self.config.smoothing_window is None or self.config.smoothing_window <= 1:
            return data
        
        # Simple moving average smoothing
        if isinstance(data, pd.Series):
            return data.rolling(window=self.config.smoothing_window, center=True).mean()
        else:
            df = pd.Series(data)
            smoothed = df.rolling(window=self.config.smoothing_window, center=True).mean()
            return smoothed.fillna(method='bfill').fillna(method='ffill').values
    
    def _apply_scaling(self, data, fit_mode=True):
        """Apply scaling to target data"""
        if self.config.scaling_method == 'none':
            return data
        
        data_2d = data.reshape(-1, 1)
        
        if fit_mode:
            # Fit scaler
            if self.config.scaling_method == 'standard':
                self.scaler_ = StandardScaler()
            elif self.config.scaling_method == 'robust':
                self.scaler_ = RobustScaler()
            elif self.config.scaling_method == 'minmax':
                self.scaler_ = MinMaxScaler()
            elif self.config.scaling_method == 'rank':
                # Rank-based scaling (to uniform distribution)
                self.scaler_ = None  # Handle separately
                ranks = stats.rankdata(data)
                return (ranks - 1) / (len(ranks) - 1)  # Scale to [0, 1]
            else:
                raise ValueError(f"Unknown scaling method: {self.config.scaling_method}")
            
            if self.scaler_ is not None:
                scaled_data = self.scaler_.fit_transform(data_2d)
                return scaled_data.flatten()
        
        else:
            # Transform using fitted scaler
            if self.config.scaling_method == 'rank':
                # For rank scaling, use empirical CDF from training data
                if hasattr(self, 'training_data_'):
                    # Approximate rank transformation
                    ranks = np.searchsorted(np.sort(self.training_data_), data, side='left')
                    return ranks / len(self.training_data_)
                else:
                    logger.warning("No training data available for rank scaling, returning original data")
                    return data
            
            elif self.scaler_ is not None:
                scaled_data = self.scaler_.transform(data_2d)
                return scaled_data.flatten()
        
        return data
    
    def _calculate_target_statistics(self, data):
        """Calculate and store target statistics"""
        self.target_stats_ = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'n_samples': len(data)
        }
    
    def fit(self, data, **kwargs):
        """Fit the regression target creator"""
        data = self._validate_input(data)
        data = self._handle_missing_values(data)
        
        # Store original data for rank scaling
        if self.config.scaling_method == 'rank':
            self.training_data_ = data.copy()
        
        # Apply transformations in order
        transformed_data = self._apply_transformation(data, fit_mode=True)
        handled_data = self._handle_outliers(transformed_data, fit_mode=True)
        smoothed_data = self._apply_smoothing(handled_data)
        scaled_data = self._apply_scaling(smoothed_data, fit_mode=True)
        
        # Calculate statistics
        self._calculate_target_statistics(scaled_data)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data):
        """Transform data to regression targets"""
        if not self.is_fitted_:
            raise ValueError("Must fit the target creator before transforming")
        
        data = self._validate_input(data)
        data = self._handle_missing_values(data)
        
        # Apply same transformations as during fitting
        transformed_data = self._apply_transformation(data, fit_mode=False)
        handled_data = self._handle_outliers(transformed_data, fit_mode=False)
        smoothed_data = self._apply_smoothing(handled_data)
        scaled_data = self._apply_scaling(smoothed_data, fit_mode=False)
        
        return scaled_data
    
    def fit_transform(self, data, **kwargs):
        """Fit and transform in one step"""
        return self.fit(data, **kwargs).transform(data)
    
    def inverse_transform(self, data):
        """Inverse transform to get original scale"""
        if not self.is_fitted_:
            raise ValueError("Must fit the target creator before inverse transforming")
        
        # Reverse scaling
        if self.config.scaling_method != 'none' and self.scaler_ is not None:
            data_2d = data.reshape(-1, 1)
            unscaled_data = self.scaler_.inverse_transform(data_2d).flatten()
        else:
            unscaled_data = data
        
        # Reverse transformation (simplified - some transformations may not be perfectly reversible)
        if self.config.transformation == 'log':
            return np.exp(unscaled_data)
        elif self.config.transformation == 'sqrt':
            return unscaled_data ** 2
        else:
            return unscaled_data
    
    def get_target_statistics(self):
        """Get target statistics"""
        if not self.is_fitted_:
            raise ValueError("Must fit the target creator before getting statistics")
        return self.target_stats_.copy()

# ============================================
# Specific Regression Targets
# ============================================

class ReturnsTarget(BaseRegressionTarget):
    """
    Creates return-based regression targets.
    Supports simple, log, and excess returns.
    """
    
    def __init__(self, 
                 return_type: str = 'simple',
                 benchmark_returns: Optional[np.ndarray] = None,
                 config: Optional[RegressionConfig] = None):
        
        if config is None:
            config = RegressionConfig()
        
        config.target_type = RegressionTarget.RETURNS
        config.return_type = return_type
        
        super().__init__(config)
        self.benchmark_returns = benchmark_returns
    
    def _calculate_returns(self, prices):
        """Calculate returns from prices"""
        if self.config.return_type == 'simple':
            returns = np.diff(prices) / prices[:-1]
        elif self.config.return_type == 'log':
            returns = np.diff(np.log(prices))
        elif self.config.return_type == 'excess':
            if self.benchmark_returns is None:
                raise ValueError("Benchmark returns required for excess returns")
            simple_returns = np.diff(prices) / prices[:-1]
            # Align lengths
            min_len = min(len(simple_returns), len(self.benchmark_returns))
            returns = simple_returns[:min_len] - self.benchmark_returns[:min_len]
        else:
            raise ValueError(f"Unknown return type: {self.config.return_type}")
        
        return returns
    
    def fit(self, price_data, **kwargs):
        """Fit returns target creator"""
        price_data = self._validate_input(price_data)
        
        # Calculate returns
        returns = self._calculate_returns(price_data)
        
        # Shift returns for lookahead
        if self.config.lookahead_periods > 1:
            # Multi-period returns
            target_returns = np.zeros(len(returns) - self.config.lookahead_periods + 1)
            for i in range(len(target_returns)):
                if self.config.return_type == 'log':
                    # Sum log returns for multi-period
                    target_returns[i] = np.sum(returns[i:i + self.config.lookahead_periods])
                else:
                    # Compound simple returns
                    compound_return = 1.0
                    for j in range(self.config.lookahead_periods):
                        compound_return *= (1 + returns[i + j])
                    target_returns[i] = compound_return - 1
            
            returns = target_returns
        
        # Use parent's fit method
        return super().fit(returns, **kwargs)

class VolatilityTarget(BaseRegressionTarget):
    """
    Creates volatility-based regression targets.
    Predicts future realized volatility.
    """
    
    def __init__(self, 
                 volatility_type: str = 'realized',  # 'realized', 'garch', 'range'
                 annualize: bool = True,
                 config: Optional[RegressionConfig] = None):
        
        if config is None:
            config = RegressionConfig()
        
        config.target_type = RegressionTarget.VOLATILITY
        
        super().__init__(config)
        self.volatility_type = volatility_type
        self.annualize = annualize
    
    def _calculate_volatility(self, prices):
        """Calculate volatility from prices"""
        if self.volatility_type == 'realized':
            # Standard realized volatility
            returns = np.diff(np.log(prices))
            volatility = pd.Series(returns).rolling(window=self.config.volatility_window).std()
            
            if self.annualize:
                volatility = volatility * np.sqrt(252)  # Annualize assuming 252 trading days
        
        elif self.volatility_type == 'range':
            # Range-based volatility (requires OHLC data)
            # Simplified version using price data
            rolling_max = pd.Series(prices).rolling(window=self.config.volatility_window).max()
            rolling_min = pd.Series(prices).rolling(window=self.config.volatility_window).min()
            range_vol = (rolling_max - rolling_min) / rolling_min
            volatility = range_vol
        
        elif self.volatility_type == 'garch':
            # Simplified GARCH-like volatility
            returns = pd.Series(np.diff(np.log(prices)))
            # Use EWMA as GARCH approximation
            volatility = returns.ewm(span=self.config.volatility_window).std()
            
            if self.annualize:
                volatility = volatility * np.sqrt(252)
        
        else:
            raise ValueError(f"Unknown volatility type: {self.volatility_type}")
        
        return volatility.values
    
    def fit(self, price_data, **kwargs):
        """Fit volatility target creator"""
        price_data = self._validate_input(price_data)
        
        # Calculate volatility
        volatility = self._calculate_volatility(price_data)
        
        # Create future volatility targets
        if self.config.lookahead_periods > 0:
            target_vol = volatility[self.config.lookahead_periods:]
        else:
            target_vol = volatility
        
        # Use parent's fit method
        return super().fit(target_vol, **kwargs)

class ValueAtRiskTarget(BaseRegressionTarget):
    """
    Creates Value at Risk (VaR) regression targets.
    Predicts future portfolio risk measures.
    """
    
    def __init__(self, 
                 var_method: str = 'parametric',  # 'parametric', 'historical', 'monte_carlo'
                 confidence_level: float = 0.05,
                 config: Optional[RegressionConfig] = None):
        
        if config is None:
            config = RegressionConfig()
        
        config.target_type = RegressionTarget.VAR
        config.confidence_level = confidence_level
        
        super().__init__(config)
        self.var_method = var_method
    
    def _calculate_var(self, returns):
        """Calculate Value at Risk"""
        returns_series = pd.Series(returns)
        
        if self.var_method == 'parametric':
            # Parametric VaR assuming normal distribution
            rolling_mean = returns_series.rolling(window=self.config.volatility_window).mean()
            rolling_std = returns_series.rolling(window=self.config.volatility_window).std()
            
            # VaR at confidence level
            z_score = stats.norm.ppf(self.config.confidence_level)
            var = rolling_mean + z_score * rolling_std
        
        elif self.var_method == 'historical':
            # Historical VaR
            def rolling_var(window_data):
                if len(window_data) < 10:
                    return np.nan
                return np.percentile(window_data, self.config.confidence_level * 100)
            
            var = returns_series.rolling(window=self.config.volatility_window).apply(
                rolling_var, raw=True
            )
        
        elif self.var_method == 'monte_carlo':
            # Simplified Monte Carlo VaR
            # In practice, this would involve more sophisticated simulation
            rolling_mean = returns_series.rolling(window=self.config.volatility_window).mean()
            rolling_std = returns_series.rolling(window=self.config.volatility_window).std()
            
            # Simulate many scenarios and take percentile
            n_simulations = 1000
            var_values = []
            
            for i in range(len(rolling_mean)):
                if pd.isna(rolling_mean.iloc[i]) or pd.isna(rolling_std.iloc[i]):
                    var_values.append(np.nan)
                    continue
                
                simulated_returns = np.random.normal(
                    rolling_mean.iloc[i], 
                    rolling_std.iloc[i], 
                    n_simulations
                )
                var_value = np.percentile(simulated_returns, self.config.confidence_level * 100)
                var_values.append(var_value)
            
            var = pd.Series(var_values, index=returns_series.index)
        
        else:
            raise ValueError(f"Unknown VaR method: {self.var_method}")
        
        return var.values
    
    def fit(self, price_data, **kwargs):
        """Fit VaR target creator"""
        price_data = self._validate_input(price_data)
        
        # Calculate returns
        returns = np.diff(np.log(price_data))
        
        # Calculate VaR
        var_values = self._calculate_var(returns)
        
        # Create future VaR targets
        if self.config.lookahead_periods > 0:
            target_var = var_values[self.config.lookahead_periods:]
        else:
            target_var = var_values
        
        # Use parent's fit method
        return super().fit(target_var, **kwargs)

class SharpeRatioTarget(BaseRegressionTarget):
    """
    Creates Sharpe ratio regression targets.
    Predicts future risk-adjusted returns.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 config: Optional[RegressionConfig] = None):
        
        if config is None:
            config = RegressionConfig()
        
        config.target_type = RegressionTarget.SHARPE_RATIO
        config.risk_free_rate = risk_free_rate
        
        super().__init__(config)
    
    def _calculate_sharpe_ratio(self, returns):
        """Calculate rolling Sharpe ratio"""
        returns_series = pd.Series(returns)
        
        # Convert annual risk-free rate to period rate
        period_rf_rate = self.config.risk_free_rate / 252
        
        # Calculate rolling mean and std
        rolling_mean = returns_series.rolling(window=self.config.volatility_window).mean()
        rolling_std = returns_series.rolling(window=self.config.volatility_window).std()
        
        # Calculate Sharpe ratio
        excess_return = rolling_mean - period_rf_rate
        sharpe_ratio = excess_return / rolling_std
        
        # Annualize
        sharpe_ratio = sharpe_ratio * np.sqrt(252)
        
        return sharpe_ratio.values
    
    def fit(self, price_data, **kwargs):
        """Fit Sharpe ratio target creator"""
        price_data = self._validate_input(price_data)
        
        # Calculate returns
        returns = np.diff(np.log(price_data))
        
        # Calculate Sharpe ratio
        sharpe_values = self._calculate_sharpe_ratio(returns)
        
        # Create future Sharpe ratio targets
        if self.config.lookahead_periods > 0:
            target_sharpe = sharpe_values[self.config.lookahead_periods:]
        else:
            target_sharpe = sharpe_values
        
        # Use parent's fit method
        return super().fit(target_sharpe, **kwargs)

class MomentumTarget(BaseRegressionTarget):
    """
    Creates momentum-based regression targets.
    Predicts future momentum strength.
    """
    
    def __init__(self, 
                 momentum_window: int = 12,
                 momentum_type: str = 'price',  # 'price', 'return', 'risk_adjusted'
                 config: Optional[RegressionConfig] = None):
        
        if config is None:
            config = RegressionConfig()
        
        config.target_type = RegressionTarget.MOMENTUM
        
        super().__init__(config)
        self.momentum_window = momentum_window
        self.momentum_type = momentum_type
    
    def _calculate_momentum(self, data):
        """Calculate momentum values"""
        data_series = pd.Series(data)
        
        if self.momentum_type == 'price':
            # Price momentum (rate of change)
            momentum = data_series.pct_change(periods=self.momentum_window)
        
        elif self.momentum_type == 'return':
            # Return momentum (moving average of returns)
            returns = data_series.pct_change()
            momentum = returns.rolling(window=self.momentum_window).mean()
        
        elif self.momentum_type == 'risk_adjusted':
            # Risk-adjusted momentum
            returns = data_series.pct_change()
            rolling_mean = returns.rolling(window=self.momentum_window).mean()
            rolling_std = returns.rolling(window=self.momentum_window).std()
            momentum = rolling_mean / rolling_std
        
        else:
            raise ValueError(f"Unknown momentum type: {self.momentum_type}")
        
        return momentum.values
    
    def fit(self, price_data, **kwargs):
        """Fit momentum target creator"""
        price_data = self._validate_input(price_data)
        
        # Calculate momentum
        momentum_values = self._calculate_momentum(price_data)
        
        # Create future momentum targets
        if self.config.lookahead_periods > 0:
            target_momentum = momentum_values[self.config.lookahead_periods:]
        else:
            target_momentum = momentum_values
        
        # Use parent's fit method
        return super().fit(target_momentum, **kwargs)

class MeanReversionTarget(BaseRegressionTarget):
    """
    Creates mean reversion regression targets.
    Predicts mean reversion strength and speed.
    """
    
    def __init__(self, 
                 lookback_window: int = 20,
                 reversion_measure: str = 'z_score',  # 'z_score', 'deviation', 'half_life'
                 config: Optional[RegressionConfig] = None):
        
        if config is None:
            config = RegressionConfig()
        
        config.target_type = RegressionTarget.MEAN_REVERSION
        
        super().__init__(config)
        self.lookback_window = lookback_window
        self.reversion_measure = reversion_measure
    
    def _calculate_mean_reversion(self, data):
        """Calculate mean reversion measures"""
        data_series = pd.Series(data)
        
        if self.reversion_measure == 'z_score':
            # Z-score from rolling mean
            rolling_mean = data_series.rolling(window=self.lookback_window).mean()
            rolling_std = data_series.rolling(window=self.lookback_window).std()
            reversion = (data_series - rolling_mean) / rolling_std
        
        elif self.reversion_measure == 'deviation':
            # Percentage deviation from rolling mean
            rolling_mean = data_series.rolling(window=self.lookback_window).mean()
            reversion = (data_series - rolling_mean) / rolling_mean
        
        elif self.reversion_measure == 'half_life':
            # Estimate mean reversion half-life (simplified)
            returns = data_series.pct_change()
            rolling_autocorr = returns.rolling(window=self.lookback_window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
            )
            # Half-life estimation: -log(2)/log(rho)
            reversion = -np.log(2) / np.log(np.abs(rolling_autocorr))
            reversion = np.where(np.isinf(reversion), np.nan, reversion)
        
        else:
            raise ValueError(f"Unknown reversion measure: {self.reversion_measure}")
        
        return reversion.values
    
    def fit(self, price_data, **kwargs):
        """Fit mean reversion target creator"""
        price_data = self._validate_input(price_data)
        
        # Calculate mean reversion
        reversion_values = self._calculate_mean_reversion(price_data)
        
        # Create future mean reversion targets
        if self.config.lookahead_periods > 0:
            target_reversion = reversion_values[self.config.lookahead_periods:]
        else:
            target_reversion = reversion_values
        
        # Use parent's fit method
        return super().fit(target_reversion, **kwargs)

# ============================================
# Utility Functions
# ============================================

@time_it("regression_target_creation")
def create_regression_target(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                           target_type: str = 'returns',
                           **kwargs) -> Tuple[np.ndarray, BaseRegressionTarget]:
    """
    Create regression targets for financial data
    
    Args:
        data: Input price/return data
        target_type: Type of regression target
        **kwargs: Additional arguments for specific targets
        
    Returns:
        Tuple of (target_array, fitted_target_creator)
    """
    
    if target_type == 'returns':
        target_creator = ReturnsTarget(**kwargs)
    elif target_type == 'volatility':
        target_creator = VolatilityTarget(**kwargs)
    elif target_type == 'var':
        target_creator = ValueAtRiskTarget(**kwargs)
    elif target_type == 'sharpe_ratio':
        target_creator = SharpeRatioTarget(**kwargs)
    elif target_type == 'momentum':
        target_creator = MomentumTarget(**kwargs)
    elif target_type == 'mean_reversion':
        target_creator = MeanReversionTarget(**kwargs)
    else:
        # Generic target
        target_creator = BaseRegressionTarget(**kwargs)
    
    targets = target_creator.fit_transform(data)
    
    logger.info(f"Created {target_type} regression targets: {len(targets)} samples")
    return targets, target_creator

def analyze_target_properties(targets: np.ndarray,
                             target_creator: BaseRegressionTarget) -> Dict[str, Any]:
    """Analyze properties of regression targets"""
    
    # Get target statistics
    stats_dict = target_creator.get_target_statistics()
    
    # Additional analysis
    analysis = {
        'basic_statistics': stats_dict,
        'distribution_analysis': {
            'is_normal': _test_normality(targets),
            'has_outliers': _detect_outliers(targets),
            'autocorrelation': _calculate_autocorrelation(targets),
            'stationarity': _test_stationarity(targets)
        },
        'target_quality': {
            'signal_to_noise': stats_dict['std'] / (np.abs(stats_dict['mean']) + 1e-8),
            'dynamic_range': stats_dict['max'] - stats_dict['min'],
            'effective_samples': len(targets) - np.sum(pd.isna(targets))
        }
    }
    
    return analysis

def _test_normality(data, alpha=0.05):
    """Test if data follows normal distribution"""
    try:
        _, p_value = stats.normaltest(data[~pd.isna(data)])
        return p_value > alpha
    except:
        return False

def _detect_outliers(data, threshold=3.0):
    """Detect outliers using z-score method"""
    clean_data = data[~pd.isna(data)]
    if len(clean_data) < 10:
        return False
    
    z_scores = np.abs(stats.zscore(clean_data))
    outlier_count = np.sum(z_scores > threshold)
    return outlier_count > len(clean_data) * 0.05  # More than 5% outliers

def _calculate_autocorrelation(data, max_lags=10):
    """Calculate autocorrelation for multiple lags"""
    clean_data = data[~pd.isna(data)]
    if len(clean_data) < 20:
        return [0] * max_lags
    
    autocorrs = []
    for lag in range(1, max_lags + 1):
        if len(clean_data) > lag:
            try:
                corr = np.corrcoef(clean_data[:-lag], clean_data[lag:])[0, 1]
                autocorrs.append(corr if not np.isnan(corr) else 0)
            except:
                autocorrs.append(0)
        else:
            autocorrs.append(0)
    
    return autocorrs

def _test_stationarity(data):
    """Simple test for stationarity (using variance)"""
    clean_data = data[~pd.isna(data)]
    if len(clean_data) < 50:
        return True  # Assume stationary for small samples
    
    # Split data in half and compare variances
    mid_point = len(clean_data) // 2
    var1 = np.var(clean_data[:mid_point])
    var2 = np.var(clean_data[mid_point:])
    
    # If variance ratio is close to 1, likely stationary
    ratio = max(var1, var2) / (min(var1, var2) + 1e-8)
    return ratio < 2.0

def create_multi_target_regression(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                                 target_types: List[str],
                                 **kwargs) -> Dict[str, Tuple[np.ndarray, BaseRegressionTarget]]:
    """Create multiple regression targets from the same data"""
    
    results = {}
    
    for target_type in target_types:
        try:
            targets, target_creator = create_regression_target(data, target_type, **kwargs)
            results[target_type] = (targets, target_creator)
            
            # Log target info
            analysis = analyze_target_properties(targets, target_creator)
            logger.info(f"{target_type}: {analysis['target_quality']['effective_samples']} samples, "
                       f"S/N ratio: {analysis['target_quality']['signal_to_noise']:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to create {target_type} target: {e}")
    
    return results

def validate_regression_targets(targets: np.ndarray,
                              min_samples: int = 100) -> Dict[str, Any]:
    """Validate regression targets for ML training"""
    
    clean_targets = targets[~pd.isna(targets)]
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'recommendations': []
    }
    
    # Check minimum sample size
    if len(clean_targets) < min_samples:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Only {len(clean_targets)} valid samples, need at least {min_samples}")
        validation_results['recommendations'].append("Collect more data or reduce target complexity")
    
    # Check for constant targets
    if np.std(clean_targets) < 1e-8:
        validation_results['is_valid'] = False
        validation_results['issues'].append("Target values are nearly constant")
        validation_results['recommendations'].append("Check data preprocessing or target calculation")
    
    # Check for extreme values
    if _detect_outliers(clean_targets):
        validation_results['issues'].append("Extreme outliers detected")
        validation_results['recommendations'].append("Consider outlier treatment or robust scaling")
    
    # Check missing value ratio
    missing_ratio = np.sum(pd.isna(targets)) / len(targets)
    if missing_ratio > 0.1:
        validation_results['issues'].append(f"High missing value ratio: {missing_ratio:.1%}")
        validation_results['recommendations'].append("Improve missing value handling strategy")
    
    validation_results['target_statistics'] = {
        'n_total': len(targets),
        'n_valid': len(clean_targets),
        'missing_ratio': missing_ratio,
        'mean': np.mean(clean_targets),
        'std': np.std(clean_targets),
        'min': np.min(clean_targets),
        'max': np.max(clean_targets)
    }
    
    return validation_results

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Financial Regression Targets")
    
    # Create sample financial data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic price series with trend and volatility
    returns = np.random.normal(0.001, 0.02, n_samples)  # Daily returns with slight positive drift
    # Add volatility clustering
    volatility = np.abs(np.random.normal(0.02, 0.005, n_samples))
    for i in range(1, len(returns)):
        returns[i] = np.random.normal(0.001, volatility[i])
        volatility[i] = 0.9 * volatility[i-1] + 0.1 * abs(returns[i])
    
    prices = 100 * np.cumprod(1 + returns)
    
    print(f"Generated {n_samples} price observations")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"Return statistics: mean={np.mean(returns):.4f}, std={np.std(returns):.4f}")
    
    # Test returns target
    print("\n1. Testing Returns Target")
    return_targets, return_creator = create_regression_target(
        prices, 'returns',
        return_type='log',
        lookahead_periods=1,
        outlier_treatment='winsorize',
        scaling_method='standard'
    )
    
    return_analysis = analyze_target_properties(return_targets, return_creator)
    print(f"Returns target: {len(return_targets)} samples")
    print(f"Mean: {return_analysis['basic_statistics']['mean']:.4f}")
    print(f"Std: {return_analysis['basic_statistics']['std']:.4f}")
    print(f"Normal distribution: {return_analysis['distribution_analysis']['is_normal']}")
    
    # Test volatility target
    print("\n2. Testing Volatility Target")
    vol_targets, vol_creator = create_regression_target(
        prices, 'volatility',
        volatility_type='realized',
        volatility_window=20,
        annualize=True,
        transformation='log',
        scaling_method='robust'
    )
    
    vol_analysis = analyze_target_properties(vol_targets, vol_creator)
    print(f"Volatility target: {len(vol_targets)} samples")
    print(f"Signal-to-noise ratio: {vol_analysis['target_quality']['signal_to_noise']:.3f}")
    print(f"Has outliers: {vol_analysis['distribution_analysis']['has_outliers']}")
    
    # Test VaR target
    print("\n3. Testing Value at Risk Target")
    var_targets, var_creator = create_regression_target(
        prices, 'var',
        var_method='historical',
        confidence_level=0.05,
        volatility_window=60,
        scaling_method='minmax'
    )
    
    var_analysis = analyze_target_properties(var_targets, var_creator)
    print(f"VaR target: {len(var_targets)} samples")
    print(f"Mean VaR: {var_analysis['basic_statistics']['mean']:.4f}")
    
    # Test Sharpe ratio target
    print("\n4. Testing Sharpe Ratio Target")
    sharpe_targets, sharpe_creator = create_regression_target(
        prices, 'sharpe_ratio',
        risk_free_rate=0.02,
        volatility_window=252,  # Annual window
        outlier_treatment='clip',
        scaling_method='standard'
    )
    
    sharpe_analysis = analyze_target_properties(sharpe_targets, sharpe_creator)
    print(f"Sharpe ratio target: {len(sharpe_targets)} samples")
    print(f"Mean Sharpe: {sharpe_analysis['basic_statistics']['mean']:.3f}")
    
    # Test momentum target
    print("\n5. Testing Momentum Target")
    momentum_targets, momentum_creator = create_regression_target(
        prices, 'momentum',
        momentum_window=12,
        momentum_type='risk_adjusted',
        smoothing_window=3,
        scaling_method='rank'
    )
    
    momentum_analysis = analyze_target_properties(momentum_targets, momentum_creator)
    print(f"Momentum target: {len(momentum_targets)} samples")
    print(f"Autocorrelation (lag 1): {momentum_analysis['distribution_analysis']['autocorrelation'][0]:.3f}")
    
    # Test mean reversion target
    print("\n6. Testing Mean Reversion Target")
    reversion_targets, reversion_creator = create_regression_target(
        prices, 'mean_reversion',
        lookback_window=20,
        reversion_measure='z_score',
        scaling_method='standard'
    )
    
    reversion_analysis = analyze_target_properties(reversion_targets, reversion_creator)
    print(f"Mean reversion target: {len(reversion_targets)} samples")
    print(f"Stationary: {reversion_analysis['distribution_analysis']['stationarity']}")
    
    # Test multi-target creation
    print("\n7. Testing Multi-Target Creation")
    multi_targets = create_multi_target_regression(
        prices,
        ['returns', 'volatility', 'momentum'],
        lookahead_periods=1,
        scaling_method='standard'
    )
    
    print(f"Created {len(multi_targets)} different target types")
    for target_type, (targets, creator) in multi_targets.items():
        analysis = analyze_target_properties(targets, creator)
        print(f"  {target_type}: {analysis['target_quality']['effective_samples']} samples, "
              f"range: [{analysis['basic_statistics']['min']:.3f}, {analysis['basic_statistics']['max']:.3f}]")
    
    # Test target validation
    print("\n8. Testing Target Validation")
    for target_type, (targets, creator) in multi_targets.items():
        validation = validate_regression_targets(targets, min_samples=100)
        print(f"{target_type} validation:")
        print(f"  Valid: {validation['is_valid']}")
        if validation['issues']:
            print(f"  Issues: {', '.join(validation['issues'])}")
        if validation['recommendations']:
            print(f"  Recommendations: {', '.join(validation['recommendations'])}")
    
    # Test inverse transformation
    print("\n9. Testing Inverse Transformation")
    # Test with returns target
    original_scale_returns = return_creator.inverse_transform(return_targets)
    print(f"Original returns range: [{np.min(returns[1:]):.4f}, {np.max(returns[1:]):.4f}]")
    print(f"Inverse transformed range: [{np.min(original_scale_returns):.4f}, {np.max(original_scale_returns):.4f}]")
    
    print("\nFinancial regression targets testing completed successfully!")
