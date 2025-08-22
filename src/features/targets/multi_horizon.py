# ============================================
# StockPredictionPro - src/features/targets/multi_horizon.py
# Advanced multi-horizon target engineering for financial machine learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from sklearn.preprocessing import StandardScaler

from .classification import BaseClassificationTarget, ClassificationConfig, DirectionClassifier
from .regression import BaseRegressionTarget, RegressionConfig, ReturnsTarget, VolatilityTarget
from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('features.targets.multi_horizon')

# ============================================
# Configuration and Enums
# ============================================

class HorizonType(Enum):
    """Types of prediction horizons"""
    INTRADAY = "intraday"           # Minutes/hours
    DAILY = "daily"                 # Days
    WEEKLY = "weekly"               # Weeks
    MONTHLY = "monthly"             # Months
    QUARTERLY = "quarterly"         # Quarters
    YEARLY = "yearly"               # Years
    CUSTOM = "custom"               # Custom periods

class AggregationMethod(Enum):
    """Methods for aggregating multi-horizon targets"""
    LAST = "last"                   # Last value in horizon
    MEAN = "mean"                   # Average over horizon
    SUM = "sum"                     # Sum over horizon
    MAX = "max"                     # Maximum in horizon
    MIN = "min"                     # Minimum in horizon
    VOLATILITY = "volatility"       # Volatility over horizon
    SHARPE = "sharpe"              # Sharpe ratio over horizon
    DRAWDOWN = "drawdown"          # Maximum drawdown in horizon

@dataclass
class MultiHorizonConfig:
    """Configuration for multi-horizon target engineering"""
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    horizon_type: HorizonType = HorizonType.DAILY
    horizon_names: Optional[List[str]] = None
    aggregation_method: AggregationMethod = AggregationMethod.LAST
    target_type: str = 'returns'  # 'returns', 'classification', 'volatility', etc.
    include_cumulative: bool = True
    include_relative_horizons: bool = True
    scaling_per_horizon: bool = True
    alignment_method: str = 'pad'  # 'pad', 'drop', 'interpolate'
    risk_free_rate: float = 0.02
    parallel_processing: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        if not self.horizons:
            raise ValueError("At least one horizon must be specified")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("All horizons must be positive")

# ============================================
# Base Multi-Horizon Target Creator
# ============================================

class BaseMultiHorizonTarget:
    """Base class for multi-horizon target creation"""
    
    def __init__(self, config: Optional[MultiHorizonConfig] = None):
        self.config = config or MultiHorizonConfig()
        self.horizon_creators_ = {}
        self.scalers_ = {}
        self.target_shapes_ = {}
        self.alignment_info_ = {}
        self.is_fitted_ = False
    
    def _validate_input(self, data):
        """Validate input data"""
        if isinstance(data, pd.Series):
            return data.values, data.index
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError("DataFrame must have exactly one column")
            return data.iloc[:, 0].values, data.index
        elif isinstance(data, np.ndarray):
            if data.ndim > 1 and data.shape[1] != 1:
                raise ValueError("Array must be 1-dimensional or have shape (n, 1)")
            return data.flatten(), None
        else:
            data_array = np.asarray(data).flatten()
            return data_array, None
    
    def _get_horizon_names(self):
        """Get names for different horizons"""
        if self.config.horizon_names:
            if len(self.config.horizon_names) != len(self.config.horizons):
                raise ValueError("Number of horizon names must match number of horizons")
            return self.config.horizon_names
        
        # Generate default names based on horizon type
        names = []
        for horizon in self.config.horizons:
            if self.config.horizon_type == HorizonType.INTRADAY:
                if horizon < 60:
                    names.append(f"{horizon}min")
                else:
                    names.append(f"{horizon//60}h")
            elif self.config.horizon_type == HorizonType.DAILY:
                if horizon == 1:
                    names.append("1d")
                elif horizon < 7:
                    names.append(f"{horizon}d")
                elif horizon % 7 == 0:
                    names.append(f"{horizon//7}w")
                else:
                    names.append(f"{horizon}d")
            elif self.config.horizon_type == HorizonType.WEEKLY:
                names.append(f"{horizon}w")
            elif self.config.horizon_type == HorizonType.MONTHLY:
                names.append(f"{horizon}m")
            elif self.config.horizon_type == HorizonType.QUARTERLY:
                names.append(f"{horizon}q")
            elif self.config.horizon_type == HorizonType.YEARLY:
                names.append(f"{horizon}y")
            else:
                names.append(f"h{horizon}")
        
        return names
    
    def _create_single_horizon_target(self, data, horizon, index=None):
        """Create target for a single horizon"""
        if self.config.target_type == 'returns':
            return self._create_returns_target(data, horizon, index)
        elif self.config.target_type == 'classification':
            return self._create_classification_target(data, horizon, index)
        elif self.config.target_type == 'volatility':
            return self._create_volatility_target(data, horizon, index)
        elif self.config.target_type == 'momentum':
            return self._create_momentum_target(data, horizon, index)
        elif self.config.target_type == 'mean_reversion':
            return self._create_mean_reversion_target(data, horizon, index)
        else:
            raise ValueError(f"Unknown target type: {self.config.target_type}")
    
    def _create_returns_target(self, data, horizon, index=None):
        """Create returns target for specific horizon"""
        if len(data) <= horizon:
            return np.array([])
        
        if self.config.aggregation_method == AggregationMethod.LAST:
            # Simple forward returns
            if isinstance(data, pd.Series) or index is not None:
                price_series = pd.Series(data, index=index) if index is not None else pd.Series(data)
                future_prices = price_series.shift(-horizon)
                returns = (future_prices - price_series) / price_series
                return returns.values[:-horizon]
            else:
                future_prices = data[horizon:]
                current_prices = data[:-horizon]
                returns = (future_prices - current_prices) / current_prices
                return returns
        
        elif self.config.aggregation_method == AggregationMethod.SUM:
            # Cumulative returns over horizon
            returns = []
            for i in range(len(data) - horizon):
                period_return = (data[i + horizon] - data[i]) / data[i]
                returns.append(period_return)
            return np.array(returns)
        
        elif self.config.aggregation_method == AggregationMethod.MEAN:
            # Average returns over horizon
            returns = []
            for i in range(len(data) - horizon):
                period_returns = np.diff(data[i:i + horizon + 1]) / data[i:i + horizon]
                avg_return = np.mean(period_returns)
                returns.append(avg_return)
            return np.array(returns)
        
        elif self.config.aggregation_method == AggregationMethod.VOLATILITY:
            # Volatility over horizon
            returns = []
            for i in range(len(data) - horizon):
                period_data = data[i:i + horizon + 1]
                period_returns = np.diff(period_data) / period_data[:-1]
                volatility = np.std(period_returns) * np.sqrt(252/horizon)  # Annualized
                returns.append(volatility)
            return np.array(returns)
        
        elif self.config.aggregation_method == AggregationMethod.SHARPE:
            # Sharpe ratio over horizon
            returns = []
            daily_rf = self.config.risk_free_rate / 252
            
            for i in range(len(data) - horizon):
                period_data = data[i:i + horizon + 1]
                period_returns = np.diff(period_data) / period_data[:-1]
                excess_returns = period_returns - daily_rf
                
                if len(excess_returns) > 1 and np.std(excess_returns) > 0:
                    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                else:
                    sharpe = 0
                
                returns.append(sharpe)
            return np.array(returns)
        
        elif self.config.aggregation_method == AggregationMethod.DRAWDOWN:
            # Maximum drawdown over horizon
            returns = []
            for i in range(len(data) - horizon):
                period_data = data[i:i + horizon + 1]
                cummax = np.maximum.accumulate(period_data)
                drawdown = (period_data - cummax) / cummax
                max_drawdown = np.min(drawdown)
                returns.append(max_drawdown)
            return np.array(returns)
        
        else:
            raise ValueError(f"Aggregation method {self.config.aggregation_method} not supported for returns")
    
    def _create_classification_target(self, data, horizon, index=None):
        """Create classification target for specific horizon"""
        # First create returns target
        returns = self._create_returns_target(data, horizon, index)
        
        if len(returns) == 0:
            return np.array([])
        
        # Convert to classification using direction classifier
        direction_classifier = DirectionClassifier(
            return_threshold=0.0,
            include_sideways=True,
            sideways_threshold=0.005
        )
        
        # Reshape returns for classifier
        classification_targets = direction_classifier.fit_transform(returns)
        return classification_targets
    
    def _create_volatility_target(self, data, horizon, index=None):
        """Create volatility target for specific horizon"""
        if len(data) <= horizon:
            return np.array([])
        
        volatilities = []
        for i in range(len(data) - horizon):
            period_data = data[i:i + horizon + 1]
            period_returns = np.diff(period_data) / period_data[:-1]
            
            if len(period_returns) > 1:
                vol = np.std(period_returns) * np.sqrt(252/horizon)  # Annualized
            else:
                vol = 0
            
            volatilities.append(vol)
        
        return np.array(volatilities)
    
    def _create_momentum_target(self, data, horizon, index=None):
        """Create momentum target for specific horizon"""
        if len(data) <= horizon:
            return np.array([])
        
        momentum = []
        for i in range(len(data) - horizon):
            # Price momentum over horizon
            current_price = data[i]
            future_price = data[i + horizon]
            momentum_val = (future_price - current_price) / current_price
            momentum.append(momentum_val)
        
        return np.array(momentum)
    
    def _create_mean_reversion_target(self, data, horizon, index=None):
        """Create mean reversion target for specific horizon"""
        if len(data) <= horizon * 2:
            return np.array([])
        
        reversion = []
        for i in range(horizon, len(data) - horizon):
            # Calculate deviation from mean
            historical_data = data[i-horizon:i]
            historical_mean = np.mean(historical_data)
            current_price = data[i]
            
            # Deviation from historical mean
            deviation = (current_price - historical_mean) / historical_mean
            
            # Future reversion (negative correlation with current deviation)
            future_data = data[i:i + horizon + 1]
            future_returns = np.diff(future_data) / future_data[:-1]
            avg_future_return = np.mean(future_returns)
            
            # Mean reversion signal (negative deviation should predict positive returns)
            reversion_signal = -deviation * avg_future_return
            reversion.append(reversion_signal)
        
        return np.array(reversion)
    
    def _align_targets(self, targets_dict):
        """Align targets across different horizons"""
        if not targets_dict:
            return {}
        
        # Find the common length (shortest target)
        min_length = min(len(targets) for targets in targets_dict.values() if len(targets) > 0)
        
        if min_length == 0:
            logger.warning("No valid targets created for any horizon")
            return {}
        
        aligned_targets = {}
        
        if self.config.alignment_method == 'drop':
            # Simply truncate all to minimum length
            for horizon_name, targets in targets_dict.items():
                if len(targets) >= min_length:
                    aligned_targets[horizon_name] = targets[:min_length]
                else:
                    aligned_targets[horizon_name] = np.array([])
        
        elif self.config.alignment_method == 'pad':
            # Pad shorter sequences with NaN
            max_length = max(len(targets) for targets in targets_dict.values())
            
            for horizon_name, targets in targets_dict.items():
                if len(targets) < max_length:
                    padding = np.full(max_length - len(targets), np.nan)
                    aligned_targets[horizon_name] = np.concatenate([targets, padding])
                else:
                    aligned_targets[horizon_name] = targets
        
        elif self.config.alignment_method == 'interpolate':
            # Use interpolation for missing values (simplified)
            max_length = max(len(targets) for targets in targets_dict.values())
            
            for horizon_name, targets in targets_dict.items():
                if len(targets) < max_length:
                    # Simple forward fill for now
                    aligned = np.full(max_length, np.nan)
                    aligned[:len(targets)] = targets
                    
                    # Forward fill
                    last_valid = None
                    for i in range(max_length):
                        if not np.isnan(aligned[i]):
                            last_valid = aligned[i]
                        elif last_valid is not None:
                            aligned[i] = last_valid
                    
                    aligned_targets[horizon_name] = aligned
                else:
                    aligned_targets[horizon_name] = targets
        
        else:
            raise ValueError(f"Unknown alignment method: {self.config.alignment_method}")
        
        return aligned_targets
    
    def _apply_scaling_per_horizon(self, targets_dict):
        """Apply scaling separately for each horizon"""
        if not self.config.scaling_per_horizon:
            return targets_dict
        
        scaled_targets = {}
        
        for horizon_name, targets in targets_dict.items():
            if len(targets) == 0:
                scaled_targets[horizon_name] = targets
                continue
            
            # Remove NaN values for scaling
            valid_mask = ~np.isnan(targets)
            if not np.any(valid_mask):
                scaled_targets[horizon_name] = targets
                continue
            
            valid_targets = targets[valid_mask]
            
            # Fit scaler
            scaler = StandardScaler()
            scaled_valid = scaler.fit_transform(valid_targets.reshape(-1, 1)).flatten()
            
            # Store scaler
            self.scalers_[horizon_name] = scaler
            
            # Apply scaling to all data (preserving NaN)
            scaled_all = targets.copy()
            scaled_all[valid_mask] = scaled_valid
            scaled_targets[horizon_name] = scaled_all
        
        return scaled_targets
    
    def _create_relative_horizon_features(self, targets_dict):
        """Create relative features between horizons"""
        if not self.config.include_relative_horizons or len(targets_dict) < 2:
            return {}
        
        relative_features = {}
        horizon_names = list(targets_dict.keys())
        
        # Create pairwise relative features
        for i, horizon1 in enumerate(horizon_names):
            for j, horizon2 in enumerate(horizon_names[i+1:], i+1):
                targets1 = targets_dict[horizon1]
                targets2 = targets_dict[horizon2]
                
                if len(targets1) == 0 or len(targets2) == 0:
                    continue
                
                # Ensure same length
                min_len = min(len(targets1), len(targets2))
                t1 = targets1[:min_len]
                t2 = targets2[:min_len]
                
                # Calculate relative features
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Ratio
                    ratio = np.divide(t1, t2, out=np.full_like(t1, np.nan), where=t2!=0)
                    relative_features[f"{horizon1}_over_{horizon2}_ratio"] = ratio
                    
                    # Difference
                    diff = t1 - t2
                    relative_features[f"{horizon1}_minus_{horizon2}_diff"] = diff
                    
                    # Correlation (rolling)
                    if len(t1) > 20:
                        corr = pd.Series(t1).rolling(20).corr(pd.Series(t2))
                        relative_features[f"{horizon1}_{horizon2}_rolling_corr"] = corr.values
        
        return relative_features
    
    def _create_cumulative_features(self, targets_dict):
        """Create cumulative features across horizons"""
        if not self.config.include_cumulative:
            return {}
        
        cumulative_features = {}
        
        # Sort horizons by value
        sorted_horizons = sorted(targets_dict.items(), 
                               key=lambda x: self.config.horizons[self.config.horizons.index(int(x[0].split('h')[-1] if 'h' in x[0] else x[0].rstrip('dw')))])
        
        # Create cumulative sums and means
        cumulative_sum = None
        cumulative_count = 0
        
        for horizon_name, targets in sorted_horizons:
            if len(targets) == 0:
                continue
            
            if cumulative_sum is None:
                cumulative_sum = targets.copy()
                cumulative_count = 1
            else:
                # Ensure same length
                min_len = min(len(cumulative_sum), len(targets))
                cumulative_sum = cumulative_sum[:min_len] + targets[:min_len]
                cumulative_count += 1
            
            # Store cumulative features
            cumulative_features[f"cumulative_sum_{horizon_name}"] = cumulative_sum.copy()
            cumulative_features[f"cumulative_mean_{horizon_name}"] = cumulative_sum / cumulative_count
        
        return cumulative_features
    
    def fit(self, data, **kwargs):
        """Fit multi-horizon target creator"""
        data_array, index = self._validate_input(data)
        
        if len(data_array) < max(self.config.horizons) * 2:
            raise ValueError(f"Need at least {max(self.config.horizons) * 2} data points")
        
        horizon_names = self._get_horizon_names()
        
        # Create targets for each horizon
        targets_dict = {}
        
        if self.config.parallel_processing and len(self.config.horizons) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_horizon = {
                    executor.submit(self._create_single_horizon_target, data_array, horizon, index): (horizon, name)
                    for horizon, name in zip(self.config.horizons, horizon_names)
                }
                
                for future in as_completed(future_to_horizon):
                    horizon, name = future_to_horizon[future]
                    try:
                        targets = future.result()
                        targets_dict[name] = targets
                        self.target_shapes_[name] = len(targets)
                    except Exception as exc:
                        logger.error(f"Horizon {horizon} generated an exception: {exc}")
                        targets_dict[name] = np.array([])
        else:
            # Sequential processing
            for horizon, name in zip(self.config.horizons, horizon_names):
                try:
                    targets = self._create_single_horizon_target(data_array, horizon, index)
                    targets_dict[name] = targets
                    self.target_shapes_[name] = len(targets)
                except Exception as exc:
                    logger.error(f"Error creating targets for horizon {horizon}: {exc}")
                    targets_dict[name] = np.array([])
        
        # Align targets across horizons
        targets_dict = self._align_targets(targets_dict)
        
        # Apply scaling per horizon
        targets_dict = self._apply_scaling_per_horizon(targets_dict)
        
        # Store alignment info
        self.alignment_info_ = {
            'original_shapes': self.target_shapes_.copy(),
            'aligned_length': len(next(iter(targets_dict.values()))) if targets_dict else 0
        }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data):
        """Transform data to multi-horizon targets"""
        if not self.is_fitted_:
            raise ValueError("Must fit before transforming")
        
        data_array, index = self._validate_input(data)
        horizon_names = self._get_horizon_names()
        
        # Create targets for each horizon
        targets_dict = {}
        
        for horizon, name in zip(self.config.horizons, horizon_names):
            try:
                targets = self._create_single_horizon_target(data_array, horizon, index)
                
                # Apply scaling if fitted
                if name in self.scalers_ and len(targets) > 0:
                    valid_mask = ~np.isnan(targets)
                    if np.any(valid_mask):
                        scaled_targets = targets.copy()
                        valid_targets = targets[valid_mask]
                        scaled_valid = self.scalers_[name].transform(valid_targets.reshape(-1, 1)).flatten()
                        scaled_targets[valid_mask] = scaled_valid
                        targets = scaled_targets
                
                targets_dict[name] = targets
            except Exception as exc:
                logger.error(f"Error transforming data for horizon {name}: {exc}")
                targets_dict[name] = np.array([])
        
        # Align targets
        targets_dict = self._align_targets(targets_dict)
        
        return targets_dict
    
    def fit_transform(self, data, **kwargs):
        """Fit and transform in one step"""
        self.fit(data, **kwargs)
        targets_dict = self.transform(data)
        
        # Add relative and cumulative features if requested
        if self.config.include_relative_horizons:
            relative_features = self._create_relative_horizon_features(targets_dict)
            targets_dict.update(relative_features)
        
        if self.config.include_cumulative:
            cumulative_features = self._create_cumulative_features(targets_dict)
            targets_dict.update(cumulative_features)
        
        return targets_dict
    
    def get_target_info(self):
        """Get information about created targets"""
        if not self.is_fitted_:
            raise ValueError("Must fit before getting target info")
        
        info = {
            'horizons': self.config.horizons,
            'horizon_names': self._get_horizon_names(),
            'target_type': self.config.target_type,
            'aggregation_method': self.config.aggregation_method.value,
            'original_shapes': self.target_shapes_,
            'alignment_info': self.alignment_info_,
            'scaling_applied': self.config.scaling_per_horizon,
            'n_scalers': len(self.scalers_)
        }
        
        return info

# ============================================
# Specialized Multi-Horizon Targets
# ============================================

class MultiHorizonReturnsTarget(BaseMultiHorizonTarget):
    """
    Multi-horizon returns prediction targets.
    Creates return targets for multiple time horizons.
    """
    
    def __init__(self, 
                 horizons: List[int] = [1, 5, 10, 20],
                 return_type: str = 'simple',
                 aggregation_method: AggregationMethod = AggregationMethod.LAST,
                 config: Optional[MultiHorizonConfig] = None):
        
        if config is None:
            config = MultiHorizonConfig()
        
        config.horizons = horizons
        config.target_type = 'returns'
        config.aggregation_method = aggregation_method
        
        super().__init__(config)
        self.return_type = return_type

class MultiHorizonVolatilityTarget(BaseMultiHorizonTarget):
    """
    Multi-horizon volatility prediction targets.
    Creates volatility targets for multiple time horizons.
    """
    
    def __init__(self, 
                 horizons: List[int] = [5, 10, 20, 60],
                 volatility_type: str = 'realized',
                 config: Optional[MultiHorizonConfig] = None):
        
        if config is None:
            config = MultiHorizonConfig()
        
        config.horizons = horizons
        config.target_type = 'volatility'
        config.aggregation_method = AggregationMethod.VOLATILITY
        
        super().__init__(config)
        self.volatility_type = volatility_type

class MultiHorizonClassificationTarget(BaseMultiHorizonTarget):
    """
    Multi-horizon classification targets.
    Creates directional classification targets for multiple horizons.
    """
    
    def __init__(self, 
                 horizons: List[int] = [1, 3, 5, 10],
                 classification_type: str = 'direction',
                 config: Optional[MultiHorizonConfig] = None):
        
        if config is None:
            config = MultiHorizonConfig()
        
        config.horizons = horizons
        config.target_type = 'classification'
        config.scaling_per_horizon = False  # Classification doesn't need scaling
        
        super().__init__(config)
        self.classification_type = classification_type

class MultiHorizonRiskTarget(BaseMultiHorizonTarget):
    """
    Multi-horizon risk targets.
    Creates risk-based targets (VaR, drawdown, etc.) for multiple horizons.
    """
    
    def __init__(self, 
                 horizons: List[int] = [5, 10, 20, 60],
                 risk_measures: List[str] = ['volatility', 'drawdown', 'sharpe'],
                 config: Optional[MultiHorizonConfig] = None):
        
        if config is None:
            config = MultiHorizonConfig()
        
        config.horizons = horizons
        config.target_type = 'returns'  # Base on returns but create risk measures
        
        super().__init__(config)
        self.risk_measures = risk_measures
    
    def fit_transform(self, data, **kwargs):
        """Create multiple risk targets for each horizon"""
        data_array, index = self._validate_input(data)
        horizon_names = self._get_horizon_names()
        
        all_targets = {}
        
        for horizon, horizon_name in zip(self.config.horizons, horizon_names):
            for risk_measure in self.risk_measures:
                # Temporarily set aggregation method
                original_aggregation = self.config.aggregation_method
                
                if risk_measure == 'volatility':
                    self.config.aggregation_method = AggregationMethod.VOLATILITY
                elif risk_measure == 'drawdown':
                    self.config.aggregation_method = AggregationMethod.DRAWDOWN
                elif risk_measure == 'sharpe':
                    self.config.aggregation_method = AggregationMethod.SHARPE
                
                # Create target
                try:
                    targets = self._create_single_horizon_target(data_array, horizon, index)
                    target_name = f"{horizon_name}_{risk_measure}"
                    all_targets[target_name] = targets
                except Exception as e:
                    logger.error(f"Error creating {risk_measure} target for horizon {horizon}: {e}")
                
                # Restore original aggregation
                self.config.aggregation_method = original_aggregation
        
        # Align all targets
        all_targets = self._align_targets(all_targets)
        
        self.is_fitted_ = True
        return all_targets

# ============================================
# Utility Functions
# ============================================

@time_it("multi_horizon_target_creation")
def create_multi_horizon_targets(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                                horizons: List[int] = [1, 5, 10, 20],
                                target_type: str = 'returns',
                                **kwargs) -> Tuple[Dict[str, np.ndarray], BaseMultiHorizonTarget]:
    """
    Create multi-horizon targets for financial data
    
    Args:
        data: Input price/return data
        horizons: List of prediction horizons
        target_type: Type of targets ('returns', 'volatility', 'classification', 'risk')
        **kwargs: Additional arguments for specific target creators
        
    Returns:
        Tuple of (targets_dict, fitted_creator)
    """
    
    if target_type == 'returns':
        creator = MultiHorizonReturnsTarget(horizons=horizons, **kwargs)
    elif target_type == 'volatility':
        creator = MultiHorizonVolatilityTarget(horizons=horizons, **kwargs)
    elif target_type == 'classification':
        creator = MultiHorizonClassificationTarget(horizons=horizons, **kwargs)
    elif target_type == 'risk':
        creator = MultiHorizonRiskTarget(horizons=horizons, **kwargs)
    else:
        # Generic multi-horizon
        config = MultiHorizonConfig(horizons=horizons, target_type=target_type, **kwargs)
        creator = BaseMultiHorizonTarget(config)
    
    targets_dict = creator.fit_transform(data)
    
    logger.info(f"Created {target_type} targets for {len(horizons)} horizons: "
               f"{list(targets_dict.keys())}")
    
    return targets_dict, creator

def analyze_multi_horizon_targets(targets_dict: Dict[str, np.ndarray],
                                 creator: BaseMultiHorizonTarget) -> Dict[str, Any]:
    """Analyze multi-horizon target properties"""
    
    analysis = {
        'target_info': creator.get_target_info(),
        'horizon_analysis': {},
        'cross_horizon_analysis': {}
    }
    
    # Analyze each horizon
    for horizon_name, targets in targets_dict.items():
        if len(targets) == 0:
            continue
        
        valid_targets = targets[~pd.isna(targets)]
        
        if len(valid_targets) > 0:
            horizon_stats = {
                'n_samples': len(valid_targets),
                'mean': np.mean(valid_targets),
                'std': np.std(valid_targets),
                'min': np.min(valid_targets),
                'max': np.max(valid_targets),
                'missing_ratio': np.sum(pd.isna(targets)) / len(targets),
                'autocorrelation': _calculate_autocorr(valid_targets, lag=1)
            }
            
            analysis['horizon_analysis'][horizon_name] = horizon_stats
    
    # Cross-horizon analysis
    horizon_names = list(targets_dict.keys())
    if len(horizon_names) > 1:
        # Calculate correlations between horizons
        correlations = {}
        for i, h1 in enumerate(horizon_names):
            for h2 in horizon_names[i+1:]:
                targets1 = targets_dict[h1]
                targets2 = targets_dict[h2]
                
                # Align lengths and remove NaN
                min_len = min(len(targets1), len(targets2))
                t1 = targets1[:min_len]
                t2 = targets2[:min_len]
                
                valid_mask = ~(pd.isna(t1) | pd.isna(t2))
                if np.sum(valid_mask) > 10:
                    corr = np.corrcoef(t1[valid_mask], t2[valid_mask])[0, 1]
                    correlations[f"{h1}_vs_{h2}"] = corr if not np.isnan(corr) else 0
        
        analysis['cross_horizon_analysis']['correlations'] = correlations
        
        # Calculate horizon consistency (decreasing correlation with distance)
        if len(correlations) > 0:
            avg_correlation = np.mean(list(correlations.values()))
            analysis['cross_horizon_analysis']['avg_correlation'] = avg_correlation
            analysis['cross_horizon_analysis']['consistency'] = avg_correlation > 0.3
    
    return analysis

def _calculate_autocorr(data, lag=1):
    """Calculate autocorrelation for given lag"""
    if len(data) <= lag:
        return 0
    
    try:
        return np.corrcoef(data[:-lag], data[lag:])[0, 1]
    except:
        return 0

def create_trading_horizons(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                           strategy_type: str = 'momentum') -> Dict[str, np.ndarray]:
    """Create targets for common trading strategy horizons"""
    
    if strategy_type == 'momentum':
        # Momentum strategy horizons
        horizons = [1, 3, 5, 10, 20]  # 1 day to 1 month
        targets_dict, _ = create_multi_horizon_targets(
            data, horizons=horizons, target_type='returns',
            aggregation_method=AggregationMethod.LAST
        )
    
    elif strategy_type == 'mean_reversion':
        # Mean reversion horizons (shorter term)
        horizons = [1, 2, 3, 5]  # 1 to 5 days
        targets_dict, _ = create_multi_horizon_targets(
            data, horizons=horizons, target_type='mean_reversion'
        )
    
    elif strategy_type == 'volatility':
        # Volatility trading horizons
        horizons = [5, 10, 20, 60]  # 1 week to 3 months
        targets_dict, _ = create_multi_horizon_targets(
            data, horizons=horizons, target_type='volatility'
        )
    
    elif strategy_type == 'trend_following':
        # Trend following horizons (longer term)
        horizons = [10, 20, 60, 120]  # 2 weeks to 6 months
        targets_dict, _ = create_multi_horizon_targets(
            data, horizons=horizons, target_type='returns',
            aggregation_method=AggregationMethod.SUM
        )
    
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    return targets_dict

def validate_multi_horizon_targets(targets_dict: Dict[str, np.ndarray],
                                  min_samples_per_horizon: int = 100) -> Dict[str, Any]:
    """Validate multi-horizon targets for ML training"""
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'recommendations': [],
        'horizon_validation': {}
    }
    
    valid_horizons = 0
    
    for horizon_name, targets in targets_dict.items():
        horizon_validation = {
            'is_valid': True,
            'issues': [],
            'n_samples': len(targets),
            'n_valid_samples': np.sum(~pd.isna(targets)) if len(targets) > 0 else 0
        }
        
        # Check minimum samples
        if horizon_validation['n_valid_samples'] < min_samples_per_horizon:
            horizon_validation['is_valid'] = False
            horizon_validation['issues'].append(
                f"Only {horizon_validation['n_valid_samples']} valid samples, need {min_samples_per_horizon}"
            )
        
        # Check missing value ratio
        if len(targets) > 0:
            missing_ratio = np.sum(pd.isna(targets)) / len(targets)
            if missing_ratio > 0.2:
                horizon_validation['issues'].append(f"High missing value ratio: {missing_ratio:.1%}")
        
        # Check for constant values
        valid_targets = targets[~pd.isna(targets)]
        if len(valid_targets) > 1 and np.std(valid_targets) < 1e-8:
            horizon_validation['is_valid'] = False
            horizon_validation['issues'].append("Target values are nearly constant")
        
        validation_results['horizon_validation'][horizon_name] = horizon_validation
        
        if horizon_validation['is_valid']:
            valid_horizons += 1
    
    # Overall validation
    if valid_horizons == 0:
        validation_results['is_valid'] = False
        validation_results['issues'].append("No valid horizons found")
        validation_results['recommendations'].append("Check data quality and horizon configuration")
    elif valid_horizons < len(targets_dict) / 2:
        validation_results['issues'].append(f"Only {valid_horizons}/{len(targets_dict)} horizons are valid")
        validation_results['recommendations'].append("Consider reducing number of horizons or improving data")
    
    validation_results['n_valid_horizons'] = valid_horizons
    validation_results['n_total_horizons'] = len(targets_dict)
    
    return validation_results

def create_ensemble_targets(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                           target_types: List[str] = ['returns', 'volatility', 'classification'],
                           horizons: List[int] = [1, 5, 10, 20]) -> Dict[str, Dict[str, np.ndarray]]:
    """Create ensemble of different target types across multiple horizons"""
    
    ensemble_targets = {}
    
    for target_type in target_types:
        try:
            targets_dict, _ = create_multi_horizon_targets(
                data, horizons=horizons, target_type=target_type
            )
            ensemble_targets[target_type] = targets_dict
            
            logger.info(f"Created {target_type} targets for {len(horizons)} horizons")
            
        except Exception as e:
            logger.error(f"Failed to create {target_type} targets: {e}")
    
    return ensemble_targets

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Multi-Horizon Financial Targets")
    
    # Create sample financial data
    np.random.seed(42)
    n_samples = 2000
    
    # Generate realistic price series
    returns = np.random.normal(0.0005, 0.02, n_samples)
    # Add trend and volatility clustering
    for i in range(1, len(returns)):
        returns[i] += 0.02 * returns[i-1]  # Momentum
        if i > 20:
            returns[i] += 0.001 * np.mean(returns[i-20:i])  # Longer trend
    
    prices = 100 * np.cumprod(1 + returns)
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    price_series = pd.Series(prices, index=dates)
    
    print(f"Generated {n_samples} price observations")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Test multi-horizon returns
    print("\n1. Testing Multi-Horizon Returns Targets")
    returns_targets, returns_creator = create_multi_horizon_targets(
        price_series, 
        horizons=[1, 5, 10, 20, 60],
        target_type='returns',
        aggregation_method=AggregationMethod.LAST,
        scaling_per_horizon=True,
        include_relative_horizons=True
    )
    
    print(f"Created returns targets for {len(returns_targets)} features:")
    for name, targets in list(returns_targets.items())[:8]:  # Show first 8
        valid_count = np.sum(~pd.isna(targets))
        print(f"  {name}: {valid_count} valid samples")
    
    # Test multi-horizon volatility
    print("\n2. Testing Multi-Horizon Volatility Targets")
    vol_targets, vol_creator = create_multi_horizon_targets(
        price_series,
        horizons=[5, 10, 20, 60],
        target_type='volatility',
        scaling_per_horizon=True
    )
    
    print(f"Created volatility targets: {list(vol_targets.keys())}")
    
    # Test multi-horizon classification
    print("\n3. Testing Multi-Horizon Classification Targets")
    class_targets, class_creator = create_multi_horizon_targets(
        price_series,
        horizons=[1, 3, 5, 10],
        target_type='classification'
    )
    
    print(f"Created classification targets: {list(class_targets.keys())}")
    for name, targets in class_targets.items():
        if len(targets) > 0:
            unique_classes = np.unique(targets[~pd.isna(targets)])
            print(f"  {name}: {len(unique_classes)} classes")
    
    # Test multi-horizon risk targets
    print("\n4. Testing Multi-Horizon Risk Targets")
    risk_targets, risk_creator = create_multi_horizon_targets(
        price_series,
        horizons=[5, 10, 20],
        target_type='risk',
        risk_measures=['volatility', 'sharpe', 'drawdown']
    )
    
    print(f"Created risk targets: {list(risk_targets.keys())[:6]}...")  # Show first 6
    
    # Test target analysis
    print("\n5. Testing Target Analysis")
    returns_analysis = analyze_multi_horizon_targets(returns_targets, returns_creator)
    
    print("Target info:")
    print(f"  Horizons: {returns_analysis['target_info']['horizons']}")
    print(f"  Aligned length: {returns_analysis['target_info']['alignment_info']['aligned_length']}")
    
    print("Horizon analysis (first 3):")
    for i, (horizon, stats) in enumerate(list(returns_analysis['horizon_analysis'].items())[:3]):
        print(f"  {horizon}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"missing={stats['missing_ratio']:.1%}")
    
    if 'correlations' in returns_analysis['cross_horizon_analysis']:
        print("Cross-horizon correlations (first 3):")
        correlations = returns_analysis['cross_horizon_analysis']['correlations']
        for i, (pair, corr) in enumerate(list(correlations.items())[:3]):
            print(f"  {pair}: {corr:.3f}")
    
    # Test trading strategy horizons
    print("\n6. Testing Trading Strategy Horizons")
    
    momentum_targets = create_trading_horizons(price_series, 'momentum')
    print(f"Momentum strategy targets: {list(momentum_targets.keys())}")
    
    vol_trading_targets = create_trading_horizons(price_series, 'volatility')
    print(f"Volatility trading targets: {list(vol_trading_targets.keys())}")
    
    # Test validation
    print("\n7. Testing Multi-Horizon Validation")
    validation = validate_multi_horizon_targets(returns_targets, min_samples_per_horizon=100)
    
    print(f"Overall validation: {validation['is_valid']}")
    print(f"Valid horizons: {validation['n_valid_horizons']}/{validation['n_total_horizons']}")
    
    if validation['issues']:
        print(f"Issues: {validation['issues']}")
    if validation['recommendations']:
        print(f"Recommendations: {validation['recommendations']}")
    
    # Test ensemble targets
    print("\n8. Testing Ensemble Targets")
    ensemble_targets = create_ensemble_targets(
        price_series,
        target_types=['returns', 'volatility'],
        horizons=[1, 5, 10, 20]
    )
    
    print(f"Created ensemble with {len(ensemble_targets)} target types:")
    for target_type, targets_dict in ensemble_targets.items():
        print(f"  {target_type}: {len(targets_dict)} horizons")
    
    # Test target info
    print("\n9. Testing Target Info")
    target_info = returns_creator.get_target_info()
    print(f"Target type: {target_info['target_type']}")
    print(f"Aggregation method: {target_info['aggregation_method']}")
    print(f"Scaling applied: {target_info['scaling_applied']}")
    print(f"Number of scalers: {target_info['n_scalers']}")
    
    print("\nMulti-horizon financial targets testing completed successfully!")
