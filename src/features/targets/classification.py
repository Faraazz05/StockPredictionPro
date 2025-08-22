# ============================================
# StockPredictionPro - src/features/targets/classification.py
# Advanced target engineering for financial classification tasks
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('features.targets.classification')

# ============================================
# Configuration and Enums
# ============================================

class ClassificationTarget(Enum):
    """Types of financial classification targets"""
    DIRECTION = "direction"                    # Price direction (up/down)
    REGIME = "regime"                         # Market regime classification
    VOLATILITY_REGIME = "volatility_regime"   # Volatility regime classification
    TREND_STRENGTH = "trend_strength"         # Trend strength categories
    RISK_LEVEL = "risk_level"                # Risk level classification
    BREAKOUT = "breakout"                     # Breakout detection
    REVERSAL = "reversal"                     # Reversal pattern detection
    SECTOR_ROTATION = "sector_rotation"       # Sector rotation signals
    EARNINGS_SURPRISE = "earnings_surprise"   # Earnings surprise direction
    CREDIT_RATING = "credit_rating"           # Credit rating changes

@dataclass
class ClassificationConfig:
    """Configuration for classification target engineering"""
    target_type: ClassificationTarget = ClassificationTarget.DIRECTION
    n_classes: int = 2
    class_names: Optional[List[str]] = None
    threshold_method: str = 'quantile'  # 'quantile', 'fixed', 'adaptive', 'zscore'
    thresholds: Optional[List[float]] = None
    balance_classes: bool = True
    min_class_size: int = 10
    lookahead_periods: int = 1
    smoothing_window: Optional[int] = None
    handle_missing: str = 'drop'  # 'drop', 'neutral', 'interpolate'
    
    def __post_init__(self):
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if self.lookahead_periods < 1:
            raise ValueError("lookahead_periods must be at least 1")

# ============================================
# Base Classification Target Creator
# ============================================

class BaseClassificationTarget:
    """Base class for all classification target creators"""
    
    def __init__(self, config: Optional[ClassificationConfig] = None):
        self.config = config or ClassificationConfig()
        self.label_encoder_ = None
        self.class_weights_ = None
        self.threshold_values_ = None
        self.class_distribution_ = None
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
        """Handle missing values in input data"""
        if self.config.handle_missing == 'drop':
            return data[~pd.isna(data)]
        elif self.config.handle_missing == 'neutral':
            # Replace with median for neutral class assignment
            median_val = np.nanmedian(data)
            return np.where(pd.isna(data), median_val, data)
        elif self.config.handle_missing == 'interpolate':
            if isinstance(data, pd.Series):
                return data.interpolate().fillna(method='bfill').fillna(method='ffill')
            else:
                df = pd.Series(data)
                return df.interpolate().fillna(method='bfill').fillna(method='ffill').values
        else:
            raise ValueError(f"Unknown handle_missing method: {self.config.handle_missing}")
    
    def _calculate_thresholds(self, data):
        """Calculate thresholds for class boundaries"""
        if self.config.thresholds is not None:
            # Use provided thresholds
            return self.config.thresholds
        
        if self.config.threshold_method == 'quantile':
            # Quantile-based thresholds
            if self.config.n_classes == 2:
                return [np.median(data)]
            else:
                quantiles = np.linspace(0, 1, self.config.n_classes + 1)[1:-1]
                return np.quantile(data, quantiles)
        
        elif self.config.threshold_method == 'fixed':
            # Fixed thresholds (e.g., 0 for returns)
            if self.config.n_classes == 2:
                return [0.0]
            else:
                # Create symmetric thresholds around 0
                step = 2.0 / self.config.n_classes
                return [step * (i + 1) - 1.0 for i in range(self.config.n_classes - 1)]
        
        elif self.config.threshold_method == 'zscore':
            # Z-score based thresholds
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if self.config.n_classes == 2:
                return [mean_val]
            elif self.config.n_classes == 3:
                return [mean_val - 0.5 * std_val, mean_val + 0.5 * std_val]
            else:
                # Multiple z-score thresholds
                z_scores = np.linspace(-1, 1, self.config.n_classes + 1)[1:-1]
                return [mean_val + z * std_val for z in z_scores]
        
        elif self.config.threshold_method == 'adaptive':
            # Adaptive thresholds based on data distribution
            return self._calculate_adaptive_thresholds(data)
        
        else:
            raise ValueError(f"Unknown threshold method: {self.config.threshold_method}")
    
    def _calculate_adaptive_thresholds(self, data):
        """Calculate adaptive thresholds based on data characteristics"""
        # Use a combination of quantiles and volatility
        volatility = np.std(data)
        
        if self.config.n_classes == 2:
            # For binary classification, use median with volatility adjustment
            median_val = np.median(data)
            return [median_val]
        
        elif self.config.n_classes == 3:
            # For ternary classification, use volatility-adjusted thresholds
            median_val = np.median(data)
            threshold_width = 0.5 * volatility
            return [median_val - threshold_width, median_val + threshold_width]
        
        else:
            # For multi-class, use adjusted quantiles
            base_quantiles = np.linspace(0, 1, self.config.n_classes + 1)[1:-1]
            # Adjust quantiles based on volatility
            adjustment = min(0.1, volatility / (2 * np.std(data)))
            adjusted_quantiles = base_quantiles + adjustment * (base_quantiles - 0.5)
            return np.quantile(data, np.clip(adjusted_quantiles, 0.01, 0.99))
    
    def _assign_classes(self, data, thresholds):
        """Assign class labels based on thresholds"""
        classes = np.zeros(len(data), dtype=int)
        
        for i, threshold in enumerate(thresholds):
            classes[data > threshold] = i + 1
        
        return classes
    
    def _apply_smoothing(self, classes):
        """Apply smoothing to reduce noise in class assignments"""
        if self.config.smoothing_window is None or self.config.smoothing_window <= 1:
            return classes
        
        # Mode-based smoothing
        smoothed_classes = classes.copy()
        window = self.config.smoothing_window
        
        for i in range(window // 2, len(classes) - window // 2):
            window_classes = classes[i - window // 2:i + window // 2 + 1]
            # Use most frequent class in window
            unique, counts = np.unique(window_classes, return_counts=True)
            most_frequent = unique[np.argmax(counts)]
            smoothed_classes[i] = most_frequent
        
        return smoothed_classes
    
    def _balance_classes(self, classes):
        """Balance classes if requested"""
        if not self.config.balance_classes:
            return classes
        
        unique_classes, counts = np.unique(classes, return_counts=True)
        
        # Check minimum class size
        min_count = np.min(counts)
        if min_count < self.config.min_class_size:
            logger.warning(f"Smallest class has only {min_count} samples, less than minimum {self.config.min_class_size}")
        
        # For now, just log the imbalance - actual balancing would be done during training
        max_imbalance = np.max(counts) / np.min(counts)
        if max_imbalance > 3:
            logger.warning(f"Class imbalance detected: max/min ratio = {max_imbalance:.2f}")
        
        return classes
    
    def fit(self, data, **kwargs):
        """Fit the classification target creator"""
        data = self._validate_input(data)
        data = self._handle_missing_values(data)
        
        # Calculate thresholds
        self.threshold_values_ = self._calculate_thresholds(data)
        
        # Create class labels
        classes = self._assign_classes(data, self.threshold_values_)
        
        # Apply smoothing if requested
        classes = self._apply_smoothing(classes)
        
        # Balance classes
        classes = self._balance_classes(classes)
        
        # Fit label encoder
        self.label_encoder_ = LabelEncoder()
        encoded_classes = self.label_encoder_.fit_transform(classes)
        
        # Calculate class distribution
        unique_classes, counts = np.unique(encoded_classes, return_counts=True)
        self.class_distribution_ = dict(zip(unique_classes, counts))
        
        # Calculate class weights
        if self.config.balance_classes:
            try:
                self.class_weights_ = compute_class_weight(
                    'balanced', 
                    classes=unique_classes, 
                    y=encoded_classes
                )
                self.class_weights_ = dict(zip(unique_classes, self.class_weights_))
            except:
                self.class_weights_ = {cls: 1.0 for cls in unique_classes}
        
        self.is_fitted_ = True
        return self
    
    def transform(self, data):
        """Transform data to classification targets"""
        if not self.is_fitted_:
            raise ValueError("Must fit the target creator before transforming")
        
        data = self._validate_input(data)
        data = self._handle_missing_values(data)
        
        # Assign classes
        classes = self._assign_classes(data, self.threshold_values_)
        
        # Apply smoothing if requested
        classes = self._apply_smoothing(classes)
        
        # Encode classes
        encoded_classes = self.label_encoder_.transform(classes)
        
        return encoded_classes
    
    def fit_transform(self, data, **kwargs):
        """Fit and transform in one step"""
        return self.fit(data, **kwargs).transform(data)
    
    def get_class_names(self):
        """Get class names"""
        if self.config.class_names:
            return self.config.class_names
        else:
            if hasattr(self.label_encoder_, 'classes_'):
                return [f"class_{i}" for i in self.label_encoder_.classes_]
            else:
                return [f"class_{i}" for i in range(self.config.n_classes)]
    
    def get_class_distribution(self):
        """Get class distribution"""
        if not self.is_fitted_:
            raise ValueError("Must fit the target creator before getting distribution")
        return self.class_distribution_.copy()
    
    def get_class_weights(self):
        """Get class weights for balanced training"""
        if not self.is_fitted_:
            raise ValueError("Must fit the target creator before getting weights")
        return self.class_weights_.copy() if self.class_weights_ else None

# ============================================
# Specific Classification Targets
# ============================================

class DirectionClassifier(BaseClassificationTarget):
    """
    Creates directional classification targets (up/down/sideways).
    Most common classification task in finance.
    """
    
    def __init__(self, 
                 return_threshold: float = 0.0,
                 include_sideways: bool = False,
                 sideways_threshold: Optional[float] = None,
                 config: Optional[ClassificationConfig] = None):
        
        if config is None:
            config = ClassificationConfig()
        
        # Override config for direction classification
        config.target_type = ClassificationTarget.DIRECTION
        config.n_classes = 3 if include_sideways else 2
        config.threshold_method = 'fixed'
        
        if include_sideways and sideways_threshold is not None:
            config.thresholds = [-sideways_threshold, sideways_threshold]
        else:
            config.thresholds = [return_threshold]
        
        if include_sideways:
            config.class_names = ['down', 'sideways', 'up']
        else:
            config.class_names = ['down', 'up']
        
        super().__init__(config)
        self.return_threshold = return_threshold
        self.include_sideways = include_sideways
        self.sideways_threshold = sideways_threshold

class RegimeClassifier(BaseClassificationTarget):
    """
    Creates market regime classification targets.
    Identifies bull/bear/sideways market conditions.
    """
    
    def __init__(self, 
                 lookback_window: int = 20,
                 volatility_threshold: float = 0.02,
                 trend_threshold: float = 0.001,
                 config: Optional[ClassificationConfig] = None):
        
        if config is None:
            config = ClassificationConfig()
        
        config.target_type = ClassificationTarget.REGIME
        config.n_classes = 4
        config.class_names = ['bear_volatile', 'bear_calm', 'bull_calm', 'bull_volatile']
        
        super().__init__(config)
        self.lookback_window = lookback_window
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
    
    def _create_regime_features(self, data):
        """Create regime features from price data"""
        if len(data) < self.lookback_window:
            raise ValueError(f"Need at least {self.lookback_window} data points")
        
        # Calculate rolling returns and volatility
        returns = pd.Series(data).pct_change()
        rolling_return = returns.rolling(self.lookback_window).mean()
        rolling_volatility = returns.rolling(self.lookback_window).std()
        
        return rolling_return.values, rolling_volatility.values
    
    def fit(self, data, **kwargs):
        """Fit regime classifier"""
        data = self._validate_input(data)
        
        # Create regime features
        avg_returns, volatilities = self._create_regime_features(data)
        
        # Create regime classes
        regimes = np.zeros(len(data))
        
        for i in range(len(data)):
            if pd.isna(avg_returns[i]) or pd.isna(volatilities[i]):
                regimes[i] = 1  # Default to bear_calm
                continue
            
            is_bull = avg_returns[i] > self.trend_threshold
            is_volatile = volatilities[i] > self.volatility_threshold
            
            if is_bull and is_volatile:
                regimes[i] = 3  # bull_volatile
            elif is_bull and not is_volatile:
                regimes[i] = 2  # bull_calm
            elif not is_bull and is_volatile:
                regimes[i] = 0  # bear_volatile
            else:
                regimes[i] = 1  # bear_calm
        
        # Use parent's fit method with regime data
        return super().fit(regimes, **kwargs)

class VolatilityRegimeClassifier(BaseClassificationTarget):
    """
    Creates volatility regime classification targets.
    Classifies periods as low, medium, or high volatility.
    """
    
    def __init__(self, 
                 volatility_window: int = 20,
                 regime_window: int = 252,
                 n_regimes: int = 3,
                 config: Optional[ClassificationConfig] = None):
        
        if config is None:
            config = ClassificationConfig()
        
        config.target_type = ClassificationTarget.VOLATILITY_REGIME
        config.n_classes = n_regimes
        config.threshold_method = 'quantile'
        
        if n_regimes == 3:
            config.class_names = ['low_vol', 'medium_vol', 'high_vol']
        else:
            config.class_names = [f'vol_regime_{i}' for i in range(n_regimes)]
        
        super().__init__(config)
        self.volatility_window = volatility_window
        self.regime_window = regime_window
    
    def _calculate_volatility(self, data):
        """Calculate rolling volatility"""
        returns = pd.Series(data).pct_change()
        volatility = returns.rolling(self.volatility_window).std() * np.sqrt(252)  # Annualized
        return volatility.values
    
    def fit(self, data, **kwargs):
        """Fit volatility regime classifier"""
        data = self._validate_input(data)
        
        # Calculate volatility
        volatility = self._calculate_volatility(data)
        
        # Use parent's fit method with volatility data
        return super().fit(volatility, **kwargs)

class TrendStrengthClassifier(BaseClassificationTarget):
    """
    Creates trend strength classification targets.
    Classifies periods based on trend strength (weak, moderate, strong).
    """
    
    def __init__(self, 
                 trend_window: int = 20,
                 strength_method: str = 'slope',  # 'slope', 'adx', 'r_squared'
                 config: Optional[ClassificationConfig] = None):
        
        if config is None:
            config = ClassificationConfig()
        
        config.target_type = ClassificationTarget.TREND_STRENGTH
        config.n_classes = 3
        config.class_names = ['weak_trend', 'moderate_trend', 'strong_trend']
        config.threshold_method = 'quantile'
        
        super().__init__(config)
        self.trend_window = trend_window
        self.strength_method = strength_method
    
    def _calculate_trend_strength(self, data):
        """Calculate trend strength using specified method"""
        if self.strength_method == 'slope':
            return self._calculate_slope_strength(data)
        elif self.strength_method == 'adx':
            return self._calculate_adx_strength(data)
        elif self.strength_method == 'r_squared':
            return self._calculate_r_squared_strength(data)
        else:
            raise ValueError(f"Unknown strength method: {self.strength_method}")
    
    def _calculate_slope_strength(self, data):
        """Calculate trend strength using slope of linear regression"""
        prices = pd.Series(data)
        strength = np.zeros(len(data))
        
        for i in range(self.trend_window - 1, len(data)):
            window_data = prices.iloc[i - self.trend_window + 1:i + 1]
            x = np.arange(len(window_data))
            
            # Linear regression slope
            slope = np.polyfit(x, window_data, 1)[0]
            # Normalize by price level
            strength[i] = abs(slope) / window_data.mean()
        
        return strength
    
    def _calculate_r_squared_strength(self, data):
        """Calculate trend strength using R-squared of linear regression"""
        prices = pd.Series(data)
        strength = np.zeros(len(data))
        
        for i in range(self.trend_window - 1, len(data)):
            window_data = prices.iloc[i - self.trend_window + 1:i + 1]
            x = np.arange(len(window_data))
            
            # Linear regression R-squared
            try:
                correlation = np.corrcoef(x, window_data)[0, 1]
                r_squared = correlation ** 2 if not np.isnan(correlation) else 0
                strength[i] = r_squared
            except:
                strength[i] = 0
        
        return strength
    
    def _calculate_adx_strength(self, data):
        """Calculate trend strength using simplified ADX-like measure"""
        # Simplified ADX calculation - in practice, would need OHLC data
        returns = pd.Series(data).pct_change()
        abs_returns = returns.abs()
        
        # Rolling average of absolute returns as trend strength proxy
        strength = abs_returns.rolling(self.trend_window).mean().fillna(0).values
        
        return strength
    
    def fit(self, data, **kwargs):
        """Fit trend strength classifier"""
        data = self._validate_input(data)
        
        # Calculate trend strength
        strength = self._calculate_trend_strength(data)
        
        # Use parent's fit method with strength data
        return super().fit(strength, **kwargs)

class BreakoutClassifier(BaseClassificationTarget):
    """
    Creates breakout classification targets.
    Identifies breakout patterns (upward breakout, downward breakout, no breakout).
    """
    
    def __init__(self, 
                 lookback_window: int = 20,
                 breakout_threshold: float = 2.0,  # Number of standard deviations
                 confirmation_periods: int = 3,
                 config: Optional[ClassificationConfig] = None):
        
        if config is None:
            config = ClassificationConfig()
        
        config.target_type = ClassificationTarget.BREAKOUT
        config.n_classes = 3
        config.class_names = ['breakout_down', 'no_breakout', 'breakout_up']
        
        super().__init__(config)
        self.lookback_window = lookback_window
        self.breakout_threshold = breakout_threshold
        self.confirmation_periods = confirmation_periods
    
    def _detect_breakouts(self, data):
        """Detect breakout patterns"""
        prices = pd.Series(data)
        breakouts = np.ones(len(data))  # Default to no_breakout (class 1)
        
        for i in range(self.lookback_window, len(data) - self.confirmation_periods):
            # Calculate support/resistance levels
            window_data = prices.iloc[i - self.lookback_window:i]
            resistance = window_data.max()
            support = window_data.min()
            
            current_price = prices.iloc[i]
            
            # Check for breakout
            range_size = resistance - support
            if range_size == 0:
                continue
            
            # Upward breakout
            if current_price > resistance + self.breakout_threshold * range_size * 0.1:
                # Confirm with next few periods
                future_prices = prices.iloc[i:i + self.confirmation_periods]
                if (future_prices > current_price * 0.98).sum() >= self.confirmation_periods // 2:
                    breakouts[i] = 2  # breakout_up
            
            # Downward breakout
            elif current_price < support - self.breakout_threshold * range_size * 0.1:
                # Confirm with next few periods
                future_prices = prices.iloc[i:i + self.confirmation_periods]
                if (future_prices < current_price * 1.02).sum() >= self.confirmation_periods // 2:
                    breakouts[i] = 0  # breakout_down
        
        return breakouts
    
    def fit(self, data, **kwargs):
        """Fit breakout classifier"""
        data = self._validate_input(data)
        
        # Detect breakouts
        breakouts = self._detect_breakouts(data)
        
        # Use parent's fit method with breakout data
        return super().fit(breakouts, **kwargs)

# ============================================
# Utility Functions
# ============================================

@time_it("classification_target_creation")
def create_classification_target(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                                target_type: str = 'direction',
                                **kwargs) -> Tuple[np.ndarray, BaseClassificationTarget]:
    """
    Create classification targets for financial data
    
    Args:
        data: Input price/return data
        target_type: Type of classification target
        **kwargs: Additional arguments for specific classifiers
        
    Returns:
        Tuple of (target_array, fitted_classifier)
    """
    
    if target_type == 'direction':
        classifier = DirectionClassifier(**kwargs)
    elif target_type == 'regime':
        classifier = RegimeClassifier(**kwargs)
    elif target_type == 'volatility_regime':
        classifier = VolatilityRegimeClassifier(**kwargs)
    elif target_type == 'trend_strength':
        classifier = TrendStrengthClassifier(**kwargs)
    elif target_type == 'breakout':
        classifier = BreakoutClassifier(**kwargs)
    else:
        # Generic classifier
        classifier = BaseClassificationTarget(**kwargs)
    
    targets = classifier.fit_transform(data)
    
    logger.info(f"Created {target_type} classification targets: {len(np.unique(targets))} classes")
    return targets, classifier

def analyze_target_distribution(targets: np.ndarray,
                               classifier: BaseClassificationTarget) -> Dict[str, Any]:
    """Analyze distribution of classification targets"""
    
    unique_targets, counts = np.unique(targets, return_counts=True)
    class_names = classifier.get_class_names()
    
    distribution = {}
    for target, count in zip(unique_targets, counts):
        class_name = class_names[target] if target < len(class_names) else f'class_{target}'
        distribution[class_name] = {
            'count': int(count),
            'percentage': float(count / len(targets) * 100)
        }
    
    # Calculate balance metrics
    max_count = np.max(counts)
    min_count = np.min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else np.inf
    
    analysis = {
        'n_samples': len(targets),
        'n_classes': len(unique_targets),
        'class_distribution': distribution,
        'imbalance_ratio': imbalance_ratio,
        'is_balanced': imbalance_ratio <= 2.0,
        'class_weights': classifier.get_class_weights()
    }
    
    return analysis

def create_multi_target_classification(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                                     target_types: List[str],
                                     **kwargs) -> Dict[str, Tuple[np.ndarray, BaseClassificationTarget]]:
    """Create multiple classification targets from the same data"""
    
    results = {}
    
    for target_type in target_types:
        try:
            targets, classifier = create_classification_target(data, target_type, **kwargs)
            results[target_type] = (targets, classifier)
            
            # Log target info
            analysis = analyze_target_distribution(targets, classifier)
            logger.info(f"{target_type}: {analysis['n_classes']} classes, "
                       f"imbalance ratio: {analysis['imbalance_ratio']:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to create {target_type} target: {e}")
    
    return results

def validate_classification_targets(targets: np.ndarray,
                                  min_class_size: int = 10) -> Dict[str, Any]:
    """Validate classification targets for ML training"""
    
    unique_targets, counts = np.unique(targets, return_counts=True)
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'recommendations': []
    }
    
    # Check minimum class size
    if np.min(counts) < min_class_size:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Smallest class has only {np.min(counts)} samples")
        validation_results['recommendations'].append("Consider combining rare classes or collecting more data")
    
    # Check class imbalance
    imbalance_ratio = np.max(counts) / np.min(counts)
    if imbalance_ratio > 10:
        validation_results['issues'].append(f"Severe class imbalance: {imbalance_ratio:.1f}:1")
        validation_results['recommendations'].append("Consider using class weights or resampling techniques")
    elif imbalance_ratio > 3:
        validation_results['issues'].append(f"Moderate class imbalance: {imbalance_ratio:.1f}:1")
        validation_results['recommendations'].append("Consider using class weights")
    
    # Check for missing classes
    expected_classes = np.arange(len(unique_targets))
    if not np.array_equal(unique_targets, expected_classes):
        validation_results['issues'].append("Missing or non-sequential class labels")
        validation_results['recommendations'].append("Ensure all classes are present in training data")
    
    validation_results['class_counts'] = dict(zip(unique_targets, counts))
    validation_results['imbalance_ratio'] = imbalance_ratio
    
    return validation_results

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Financial Classification Targets")
    
    # Create sample financial data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic price series
    returns = np.random.normal(0, 0.02, n_samples)
    # Add some trend and momentum
    for i in range(1, len(returns)):
        returns[i] += 0.05 * returns[i-1]  # Momentum
    
    prices = 100 * np.cumprod(1 + returns)
    
    print(f"Generated {n_samples} price observations")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Test direction classifier
    print("\n1. Testing Direction Classifier")
    direction_targets, direction_classifier = create_classification_target(
        returns, 'direction', 
        include_sideways=True, 
        sideways_threshold=0.01
    )
    
    direction_analysis = analyze_target_distribution(direction_targets, direction_classifier)
    print(f"Direction classes: {direction_analysis['n_classes']}")
    print("Class distribution:")
    for class_name, info in direction_analysis['class_distribution'].items():
        print(f"  {class_name}: {info['count']} ({info['percentage']:.1f}%)")
    
    # Test regime classifier
    print("\n2. Testing Regime Classifier")
    regime_targets, regime_classifier = create_classification_target(
        prices, 'regime',
        lookback_window=20,
        volatility_threshold=0.02
    )
    
    regime_analysis = analyze_target_distribution(regime_targets, regime_classifier)
    print(f"Regime classes: {regime_analysis['n_classes']}")
    print("Regime distribution:")
    for class_name, info in regime_analysis['class_distribution'].items():
        print(f"  {class_name}: {info['count']} ({info['percentage']:.1f}%)")
    
    # Test volatility regime classifier
    print("\n3. Testing Volatility Regime Classifier")
    vol_targets, vol_classifier = create_classification_target(
        prices, 'volatility_regime',
        volatility_window=20,
        n_regimes=3
    )
    
    vol_analysis = analyze_target_distribution(vol_targets, vol_classifier)
    print(f"Volatility regime imbalance ratio: {vol_analysis['imbalance_ratio']:.2f}")
    
    # Test trend strength classifier
    print("\n4. Testing Trend Strength Classifier")
    trend_targets, trend_classifier = create_classification_target(
        prices, 'trend_strength',
        trend_window=20,
        strength_method='slope'
    )
    
    trend_analysis = analyze_target_distribution(trend_targets, trend_classifier)
    print(f"Trend strength classes: {trend_analysis['n_classes']}")
    
    # Test breakout classifier
    print("\n5. Testing Breakout Classifier")
    breakout_targets, breakout_classifier = create_classification_target(
        prices, 'breakout',
        lookback_window=20,
        breakout_threshold=2.0
    )
    
    breakout_analysis = analyze_target_distribution(breakout_targets, breakout_classifier)
    print(f"Breakout detection classes: {breakout_analysis['n_classes']}")
    
    # Test multi-target creation
    print("\n6. Testing Multi-Target Creation")
    multi_targets = create_multi_target_classification(
        prices, 
        ['direction', 'volatility_regime', 'trend_strength']
    )
    
    print(f"Created {len(multi_targets)} different target types")
    for target_type, (targets, classifier) in multi_targets.items():
        analysis = analyze_target_distribution(targets, classifier)
        print(f"  {target_type}: {analysis['n_classes']} classes, "
              f"balance: {'Good' if analysis['is_balanced'] else 'Poor'}")
    
    # Test target validation
    print("\n7. Testing Target Validation")
    for target_type, (targets, classifier) in multi_targets.items():
        validation = validate_classification_targets(targets, min_class_size=20)
        print(f"{target_type} validation:")
        print(f"  Valid: {validation['is_valid']}")
        if validation['issues']:
            print(f"  Issues: {', '.join(validation['issues'])}")
        if validation['recommendations']:
            print(f"  Recommendations: {', '.join(validation['recommendations'])}")
    
    print("\nFinancial classification targets testing completed successfully!")
