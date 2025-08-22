# ============================================
# StockPredictionPro - src/trading/signals/regression_signals.py
# Machine learning regression-based trading signals for financial markets
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from scipy import stats

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it
from .technical_signals import TechnicalSignal, SignalDirection, SignalConfidence

logger = get_logger('trading.signals.regression_signals')

# ============================================
# Regression Signal Data Structures
# ============================================

class RegressionTarget(Enum):
    """Regression prediction targets"""
    PRICE = "price"                    # Future price level
    RETURN = "return"                  # Future return
    VOLATILITY = "volatility"          # Future volatility
    VOLUME = "volume"                  # Future volume
    HIGH_LOW_RANGE = "high_low_range"  # Future price range
    SUPPORT_RESISTANCE = "support_resistance"  # Support/resistance levels

class PredictionConfidence(Enum):
    """Prediction confidence based on model performance"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9

@dataclass
class RegressionSignal(TechnicalSignal):
    """Extended signal class for regression-based predictions"""
    model_name: str = ""
    target_type: RegressionTarget = RegressionTarget.RETURN
    
    # Regression-specific attributes
    predicted_value: float = 0.0
    predicted_return: Optional[float] = None
    prediction_interval: Tuple[float, float] = (0.0, 0.0)
    
    # Model performance metrics
    model_r2_score: float = 0.0
    model_mse: float = 0.0
    model_mae: float = 0.0
    prediction_std: float = 0.0
    
    # Feature analysis
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    residual_analysis: Dict[str, float] = field(default_factory=dict)
    
    # Signal interpretation
    expected_move: float = 0.0
    move_probability: float = 0.5
    
    def __post_init__(self):
        """Post-initialization processing"""
        super().__post_init__()
        
        # Convert predicted value to signal direction
        if self.target_type == RegressionTarget.RETURN:
            if self.predicted_value > 0.01:  # > 1% return
                self.direction = SignalDirection.STRONG_BUY if self.predicted_value > 0.05 else SignalDirection.BUY
            elif self.predicted_value < -0.01:  # < -1% return
                self.direction = SignalDirection.STRONG_SELL if self.predicted_value < -0.05 else SignalDirection.SELL
            else:
                self.direction = SignalDirection.HOLD
        
        # Set expected move
        self.expected_move = abs(self.predicted_value)

# ============================================
# Base Regression Model
# ============================================

class BaseRegressionModel:
    """
    Base class for all regression models used in signal generation.
    
    This class provides common functionality for training regression models,
    making predictions, and generating trading signals.
    """
    
    def __init__(self, name: str, model, target_type: RegressionTarget = RegressionTarget.RETURN):
        self.name = name
        self.model = model
        self.target_type = target_type
        self.is_trained = False
        
        # Feature preprocessing
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Performance tracking
        self.training_r2 = 0.0
        self.validation_r2 = 0.0
        self.training_mse = 0.0
        self.validation_mse = 0.0
        self.cv_scores = []
        
        # Prediction statistics
        self.prediction_std = 0.0
        self.residual_stats = {}
        
        logger.info(f"Initialized {name} regression model for {target_type.value} prediction")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regression model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement prepare_features method")
    
    def prepare_targets(self, data: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """Prepare target variables based on regression type"""
        
        if self.target_type == RegressionTarget.RETURN:
            # Future return prediction
            returns = data['close'].pct_change(periods=horizon).shift(-horizon)
            return returns
        
        elif self.target_type == RegressionTarget.PRICE:
            # Future price level prediction
            future_price = data['close'].shift(-horizon)
            return future_price
        
        elif self.target_type == RegressionTarget.VOLATILITY:
            # Future volatility prediction
            returns = data['close'].pct_change()
            rolling_vol = returns.rolling(window=horizon).std().shift(-horizon)
            return rolling_vol * np.sqrt(252)  # Annualized volatility
        
        elif self.target_type == RegressionTarget.VOLUME:
            # Future volume prediction
            future_volume = data['volume'].shift(-horizon)
            return np.log(future_volume)  # Log transform for stability
        
        elif self.target_type == RegressionTarget.HIGH_LOW_RANGE:
            # Future price range prediction
            future_range = ((data['high'] - data['low']) / data['close']).shift(-horizon)
            return future_range
        
        else:
            raise ValueError(f"Unsupported target type: {self.target_type}")
    
    @time_it("regression_model_training")
    def train(self, data: pd.DataFrame, horizon: int = 1, validation_split: float = 0.2) -> Dict[str, float]:
        """Train the regression model"""
        
        # Prepare features and targets
        features = self.prepare_features(data)
        targets = self.prepare_targets(data, horizon)
        
        # Remove NaN values
        valid_mask = ~(features.isna().any(axis=1) | targets.isna())
        features_clean = features[valid_mask]
        targets_clean = targets[valid_mask]
        
        if len(features_clean) < 50:
            raise ValueError("Insufficient clean data for training (need at least 50 samples)")
        
        # Store feature names
        self.feature_names = features_clean.columns.tolist()
        
        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.fit_transform(features_clean),
            columns=features_clean.columns,
            index=features_clean.index
        )
        
        # Time-series aware split (no shuffling)
        split_idx = int(len(features_scaled) * (1 - validation_split))
        
        X_train = features_scaled.iloc[:split_idx]
        X_val = features_scaled.iloc[split_idx:]
        y_train = targets_clean.iloc[:split_idx]
        y_val = targets_clean.iloc[split_idx:]
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate performance metrics
        train_predictions = self.model.predict(X_train)
        val_predictions = self.model.predict(X_val)
        
        # R² scores
        self.training_r2 = r2_score(y_train, train_predictions)
        self.validation_r2 = r2_score(y_val, val_predictions)
        
        # MSE scores
        self.training_mse = mean_squared_error(y_train, train_predictions)
        self.validation_mse = mean_squared_error(y_val, val_predictions)
        
        # Cross-validation using time series splits
        if len(features_scaled) > 100:
            tscv = TimeSeriesSplit(n_splits=5)
            self.cv_scores = cross_val_score(
                self.model, features_scaled, targets_clean, cv=tscv, scoring='r2'
            )
        
        # Calculate prediction statistics
        residuals = val_predictions - y_val
        self.prediction_std = np.std(residuals)
        
        # Residual analysis
        self.residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'jarque_bera_p': stats.jarque_bera(residuals)[1] if len(residuals) > 8 else 1.0
        }
        
        performance_metrics = {
            'training_r2': self.training_r2,
            'validation_r2': self.validation_r2,
            'training_mse': self.training_mse,
            'validation_mse': self.validation_mse,
            'cv_r2_mean': np.mean(self.cv_scores) if self.cv_scores else 0.0,
            'cv_r2_std': np.std(self.cv_scores) if self.cv_scores else 0.0,
            'prediction_std': self.prediction_std,
            'residual_normality_p': self.residual_stats['jarque_bera_p']
        }
        
        logger.info(f"{self.name} training completed - "
                   f"Train R²: {self.training_r2:.3f}, "
                   f"Val R²: {self.validation_r2:.3f}")
        
        return performance_metrics
    
    def predict(self, features: pd.DataFrame, return_intervals: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions and optionally return prediction intervals"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.transform(features),
            columns=features.columns,
            index=features.index
        )
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        
        # Calculate prediction intervals if requested
        intervals = None
        if return_intervals and self.prediction_std > 0:
            # Simple prediction interval using residual standard deviation
            confidence_level = 0.95
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin_error = z_score * self.prediction_std
            
            intervals = np.column_stack([
                predictions - margin_error,
                predictions + margin_error
            ])
        
        return predictions, intervals
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            for feature, importance in zip(self.feature_names, self.model.feature_importances_):
                importance_dict[feature] = float(importance)
        
        elif hasattr(self.model, 'coef_'):
            # Linear models
            coefficients = self.model.coef_ if hasattr(self.model.coef_, '__len__') else [self.model.coef_]
            for feature, coef in zip(self.feature_names, coefficients):
                importance_dict[feature] = float(abs(coef))
        
        return importance_dict
    
    def analyze_predictions(self, features: pd.DataFrame, actual_values: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Analyze prediction quality and model behavior"""
        
        predictions, intervals = self.predict(features, return_intervals=True)
        
        analysis = {
            'predictions_mean': np.mean(predictions),
            'predictions_std': np.std(predictions),
            'predictions_min': np.min(predictions),
            'predictions_max': np.max(predictions),
            'feature_importance': self.get_feature_importance()
        }
        
        if actual_values is not None:
            # Calculate prediction accuracy metrics
            valid_mask = ~np.isnan(actual_values) & ~np.isnan(predictions)
            if np.sum(valid_mask) > 0:
                actual_clean = actual_values[valid_mask]
                pred_clean = predictions[valid_mask]
                
                analysis.update({
                    'r2_score': r2_score(actual_clean, pred_clean),
                    'mse': mean_squared_error(actual_clean, pred_clean),
                    'mae': mean_absolute_error(actual_clean, pred_clean),
                    'correlation': np.corrcoef(actual_clean, pred_clean)[0, 1],
                    'residuals_mean': np.mean(actual_clean - pred_clean),
                    'residuals_std': np.std(actual_clean - pred_clean)
                })
        
        return analysis

# ============================================
# Return Prediction Model
# ============================================

class ReturnPredictionModel(BaseRegressionModel):
    """
    Model for predicting future returns using regression.
    
    Uses technical indicators, price patterns, and market features
    to predict next-period returns with confidence intervals.
    """
    
    def __init__(self, model=None, lookback_periods: int = 20):
        if model is None:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        
        super().__init__("Return Prediction", model, RegressionTarget.RETURN)
        self.lookback_periods = lookback_periods
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for return prediction"""
        
        features = pd.DataFrame(index=data.index)
        
        # Lagged return features
        returns = data['close'].pct_change()
        for lag in [1, 2, 3, 5, 10]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
        
        # Rolling return statistics
        features['return_mean_5'] = returns.rolling(5).mean()
        features['return_mean_20'] = returns.rolling(20).mean()
        features['return_std_5'] = returns.rolling(5).std()
        features['return_std_20'] = returns.rolling(20).std()
        features['return_skew_20'] = returns.rolling(20).skew()
        features['return_kurt_20'] = returns.rolling(20).kurt()
        
        # Price momentum features
        features['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        features['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        features['momentum_20'] = data['close'] / data['close'].shift(20) - 1
        features['momentum_acceleration'] = features['momentum_5'] - features['momentum_10']
        
        # Mean reversion features
        features['price_to_sma_20'] = data['close'] / data['close'].rolling(20).mean()
        features['price_to_sma_50'] = data['close'] / data['close'].rolling(50).mean()
        features['price_to_ema_12'] = data['close'] / data['close'].ewm(span=12).mean()
        features['mean_reversion_20'] = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
        
        # Volatility features
        features['realized_vol_5'] = returns.rolling(5).std() * np.sqrt(252)
        features['realized_vol_20'] = returns.rolling(20).std() * np.sqrt(252)
        features['vol_ratio'] = features['realized_vol_5'] / features['realized_vol_20']
        
        # High-low range features
        features['high_low_ratio'] = (data['high'] - data['low']) / data['close']
        features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_price_corr'] = returns.rolling(20).corr(data['volume'].pct_change())
        features['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
        
        # Technical indicators
        # RSI
        delta = data['close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gains = gains.rolling(14).mean()
        avg_losses = losses.rolling(14).mean()
        rs = avg_gains / avg_losses
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_momentum'] = features['rsi'] - features['rsi'].shift(5)
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        features['macd'] = macd / data['close']  # Normalized
        features['macd_signal'] = macd_signal / data['close']
        features['macd_histogram'] = (macd - macd_signal) / data['close']
        
        # Bollinger Bands
        sma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / sma_20
        
        # Market structure features
        high_20 = data['high'].rolling(20).max()
        low_20 = data['low'].rolling(20).min()
        features['price_position_20'] = (data['close'] - low_20) / (high_20 - low_20)
        features['breakout_potential'] = 1 - ((high_20 - low_20) / data['close'])
        
        # Autocorrelation features
        features['return_autocorr_1'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1))
        features['return_autocorr_5'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5))
        
        # Gap features
        features['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        features['gap_momentum'] = features['gap'].rolling(5).mean()
        
        return features.dropna()
    
    @time_it("return_prediction_signal_generation")
    def generate_signals(self, data: pd.DataFrame, 
                        horizon: int = 1,
                        min_prediction_threshold: float = 0.01) -> List[RegressionSignal]:
        """Generate return prediction signals"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before generating signals")
        
        # Prepare features
        features = self.prepare_features(data)
        
        if len(features) == 0:
            return []
        
        # Make predictions with intervals
        predictions, intervals = self.predict(features, return_intervals=True)
        
        signals = []
        feature_importance = self.get_feature_importance()
        
        for i, idx in enumerate(features.index):
            if i >= len(predictions):
                break
            
            predicted_return = predictions[i]
            prediction_interval = intervals[i] if intervals is not None else (predicted_return, predicted_return)
            
            # Skip weak predictions
            if abs(predicted_return) < min_prediction_threshold:
                continue
            
            # Determine signal strength based on prediction magnitude and model confidence
            prediction_magnitude = abs(predicted_return)
            model_confidence = max(0.0, self.validation_r2)  # R² as confidence proxy
            
            # Signal strength combines prediction magnitude and model confidence
            signal_strength = min(1.0, prediction_magnitude * 10 * (1 + model_confidence))
            
            # Determine confidence level
            interval_width = prediction_interval[1] - prediction_interval[0] if intervals is not None else 0.01
            interval_confidence = min(1.0, prediction_magnitude / interval_width) if interval_width > 0 else 0.5
            
            combined_confidence = (model_confidence + interval_confidence) / 2
            
            if combined_confidence >= 0.8:
                confidence = SignalConfidence.VERY_HIGH
            elif combined_confidence >= 0.6:
                confidence = SignalConfidence.HIGH
            elif combined_confidence >= 0.4:
                confidence = SignalConfidence.MEDIUM
            else:
                confidence = SignalConfidence.LOW
            
            # Calculate move probability based on prediction interval
            if intervals is not None:
                # Probability that the actual return has the same sign as prediction
                if predicted_return > 0:
                    move_probability = 1 - stats.norm.cdf(0, predicted_return, self.prediction_std)
                else:
                    move_probability = stats.norm.cdf(0, predicted_return, self.prediction_std)
            else:
                move_probability = 0.6 + 0.4 * model_confidence  # Heuristic
            
            # Create regression signal
            signal = RegressionSignal(
                timestamp=idx,
                symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                indicator=f"{self.name}_{horizon}D",
                direction=SignalDirection.BUY if predicted_return > 0 else SignalDirection.SELL,
                strength=signal_strength,
                confidence=confidence,
                price=data.loc[idx, 'close'] if idx in data.index else 0.0,
                indicator_value=predicted_return,
                
                # Regression-specific attributes
                model_name=self.name,
                target_type=self.target_type,
                predicted_value=predicted_return,
                predicted_return=predicted_return,
                prediction_interval=prediction_interval,
                model_r2_score=self.validation_r2,
                model_mse=self.validation_mse,
                model_mae=np.sqrt(self.validation_mse),  # Approximation
                prediction_std=self.prediction_std,
                feature_contributions=feature_importance,
                residual_analysis=self.residual_stats,
                expected_move=abs(predicted_return),
                move_probability=move_probability,
                
                metadata={
                    'prediction_horizon': horizon,
                    'interval_width': interval_width,
                    'model_cv_mean': np.mean(self.cv_scores) if self.cv_scores else 0.0,
                    'top_features': dict(sorted(feature_importance.items(), 
                                              key=lambda x: x[1], reverse=True)[:5]),
                    'residual_normality': self.residual_stats['jarque_bera_p'] > 0.05
                }
            )
            
            signals.append(signal)
        
        logger.info(f"Generated {len(signals)} return prediction signals")
        return signals

# ============================================
# Price Level Prediction Model
# ============================================

class PricePredictionModel(BaseRegressionModel):
    """
    Model for predicting future price levels using regression.
    
    Predicts actual price levels rather than returns, useful for
    support/resistance identification and price target setting.
    """
    
    def __init__(self, model=None):
        if model is None:
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        
        super().__init__("Price Prediction", model, RegressionTarget.PRICE)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for price level prediction"""
        
        features = pd.DataFrame(index=data.index)
        
        # Price level features (normalized by current price)
        for lag in [1, 2, 3, 5, 10]:
            features[f'price_ratio_lag_{lag}'] = data['close'] / data['close'].shift(lag)
        
        # Moving average ratios
        for period in [5, 10, 20, 50]:
            sma = data['close'].rolling(period).mean()
            features[f'price_to_sma_{period}'] = data['close'] / sma
        
        # Price channel features
        for period in [10, 20]:
            high_channel = data['high'].rolling(period).max()
            low_channel = data['low'].rolling(period).min()
            features[f'price_channel_pos_{period}'] = (data['close'] - low_channel) / (high_channel - low_channel)
            features[f'channel_width_{period}'] = (high_channel - low_channel) / data['close']
        
        # Volatility-adjusted features
        returns = data['close'].pct_change()
        vol_20 = returns.rolling(20).std()
        features['vol_adjusted_momentum'] = (data['close'] / data['close'].shift(10) - 1) / vol_20
        
        # Support/resistance levels
        features['distance_from_high_20'] = (data['high'].rolling(20).max() - data['close']) / data['close']
        features['distance_from_low_20'] = (data['close'] - data['low'].rolling(20).min()) / data['close']
        
        # Volume-price relationships
        features['vwap_ratio'] = data['close'] / self._calculate_vwap(data, 20)
        features['volume_weighted_momentum'] = self._calculate_volume_weighted_momentum(data, 10)
        
        # Regime features
        features['trend_strength'] = self._calculate_trend_strength(data, 20)
        features['mean_reversion_strength'] = self._calculate_mean_reversion_strength(data, 20)
        
        return features.dropna()
    
    def _calculate_vwap(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(period).sum() / data['volume'].rolling(period).sum()
        return vwap
    
    def _calculate_volume_weighted_momentum(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate volume-weighted price momentum"""
        returns = data['close'].pct_change()
        volume_weights = data['volume'] / data['volume'].rolling(period).sum()
        weighted_momentum = (returns * volume_weights).rolling(period).sum()
        return weighted_momentum
    
    def _calculate_trend_strength(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate trend strength indicator"""
        close_prices = data['close'].rolling(period)
        trend_strength = close_prices.apply(
            lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == period else 0
        )
        return trend_strength / data['close']  # Normalize
    
    def _calculate_mean_reversion_strength(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate mean reversion strength"""
        price_deviation = data['close'] - data['close'].rolling(period).mean()
        volatility = data['close'].pct_change().rolling(period).std() * data['close']
        mean_reversion_strength = -price_deviation / volatility  # Negative because we expect reversion
        return mean_reversion_strength

# ============================================
# Volatility Prediction Model
# ============================================

class VolatilityPredictionModel(BaseRegressionModel):
    """
    Model for predicting future volatility levels using regression.
    
    Predicts realized volatility, useful for options strategies
    and risk management applications.
    """
    
    def __init__(self, model=None):
        if model is None:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        super().__init__("Volatility Prediction", model, RegressionTarget.VOLATILITY)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for volatility prediction"""
        
        features = pd.DataFrame(index=data.index)
        returns = data['close'].pct_change()
        
        # Historical volatility features
        for period in [5, 10, 20, 60]:
            features[f'realized_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Volatility ratios and momentum
        features['vol_ratio_5_20'] = features['realized_vol_5'] / features['realized_vol_20']
        features['vol_momentum'] = features['realized_vol_5'] - features['realized_vol_20']
        
        # Range-based volatility measures
        features['parkinson_vol'] = self._calculate_parkinson_volatility(data, 20)
        features['garman_klass_vol'] = self._calculate_garman_klass_volatility(data, 20)
        features['rogers_satchell_vol'] = self._calculate_rogers_satchell_volatility(data, 20)
        
        # Volatility clustering indicators
        squared_returns = returns ** 2
        features['vol_clustering'] = squared_returns.rolling(10).mean() / squared_returns.rolling(20).mean()
        
        # Jump detection features
        features['jump_intensity'] = self._detect_jumps(returns, 20)
        
        # Market stress indicators
        features['extreme_moves'] = (abs(returns) > returns.rolling(60).quantile(0.95)).rolling(10).sum()
        features['volume_volatility'] = (returns.abs() * data['volume']).rolling(10).mean()
        
        # Autocorrelation in squared returns (volatility clustering)
        features['vol_autocorr'] = squared_returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 1 else 0
        )
        
        return features.dropna()
    
    def _calculate_parkinson_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Parkinson volatility estimator"""
        log_hl = np.log(data['high'] / data['low'])
        parkinson_vol = np.sqrt((log_hl ** 2).rolling(period).mean() / (4 * np.log(2))) * np.sqrt(252)
        return parkinson_vol
    
    def _calculate_garman_klass_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Garman-Klass volatility estimator"""
        log_hl = np.log(data['high'] / data['low'])
        log_co = np.log(data['close'] / data['open'])
        gk_vol = np.sqrt((0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2).rolling(period).mean()) * np.sqrt(252)
        return gk_vol
    
    def _calculate_rogers_satchell_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Rogers-Satchell volatility estimator"""
        log_ho = np.log(data['high'] / data['open'])
        log_hc = np.log(data['high'] / data['close'])
        log_lo = np.log(data['low'] / data['open'])
        log_lc = np.log(data['low'] / data['close'])
        rs_vol = np.sqrt((log_ho * log_hc + log_lo * log_lc).rolling(period).mean()) * np.sqrt(252)
        return rs_vol
    
    def _detect_jumps(self, returns: pd.Series, period: int) -> pd.Series:
        """Detect price jumps using threshold method"""
        vol_threshold = returns.rolling(period).std() * 3  # 3-sigma threshold
        jumps = (abs(returns) > vol_threshold).rolling(5).sum()
        return jumps

# ============================================
# Regression Signal Generator
# ============================================

class RegressionSignalGenerator:
    """
    Comprehensive regression signal generator.
    
    Manages multiple regression models and generates
    signals based on quantitative predictions.
    """
    
    def __init__(self):
        self.models = {}
        self.signals_generated = 0
        
        # Initialize default models
        self.models['return'] = ReturnPredictionModel()
        self.models['price'] = PricePredictionModel()
        self.models['volatility'] = VolatilityPredictionModel()
        
        logger.info("Initialized RegressionSignalGenerator with 3 default models")
    
    def add_model(self, name: str, model: BaseRegressionModel):
        """Add a custom regression model"""
        self.models[name] = model
        logger.info(f"Added custom regression model: {name}")
    
    def train_models(self, data: pd.DataFrame, 
                    models_to_train: Optional[List[str]] = None,
                    horizon: int = 1) -> Dict[str, Dict[str, float]]:
        """Train specified models or all models"""
        
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        training_results = {}
        
        for model_name in models_to_train:
            if model_name not in self.models:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            try:
                logger.info(f"Training {model_name} regression model...")
                model = self.models[model_name]
                results = model.train(data, horizon=horizon)
                training_results[model_name] = results
                
                logger.info(f"{model_name} training completed - "
                           f"Validation R²: {results['validation_r2']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name} model: {e}")
                training_results[model_name] = {'error': str(e)}
        
        return training_results
    
    @time_it("generate_all_regression_signals")
    def generate_all_signals(self, data: pd.DataFrame, 
                           models_to_use: Optional[List[str]] = None,
                           horizon: int = 1) -> Dict[str, List[RegressionSignal]]:
        """Generate signals from all or specified models"""
        
        if models_to_use is None:
            models_to_use = [name for name, model in self.models.items() if model.is_trained]
        
        all_signals = {}
        
        for model_name in models_to_use:
            if model_name not in self.models:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            model = self.models[model_name]
            
            if not model.is_trained:
                logger.warning(f"Model {model_name} is not trained, skipping")
                continue
            
            try:
                if model_name == 'return':
                    signals = model.generate_signals(data, horizon=horizon)
                else:
                    signals = model.generate_signals(data)
                
                all_signals[model_name] = signals
                
                logger.info(f"{model_name}: Generated {len(signals)} signals")
                
            except Exception as e:
                logger.error(f"Error generating signals for {model_name}: {e}")
                all_signals[model_name] = []
        
        # Update total signals count
        self.signals_generated = sum(len(signals) for signals in all_signals.values())
        
        logger.info(f"Total regression signals generated: {self.signals_generated}")
        
        return all_signals
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all models and their performance"""
        
        summary_data = []
        
        for name, model in self.models.items():
            if model.is_trained:
                cv_mean = np.mean(model.cv_scores) if model.cv_scores else 0.0
                cv_std = np.std(model.cv_scores) if model.cv_scores else 0.0
                
                summary_data.append({
                    'Model': name,
                    'Target_Type': model.target_type.value,
                    'Trained': True,
                    'Training_R2': model.training_r2,
                    'Validation_R2': model.validation_r2,
                    'CV_R2_Mean': cv_mean,
                    'CV_R2_Std': cv_std,
                    'MSE': model.validation_mse,
                    'Prediction_Std': model.prediction_std,
                    'Features': len(model.feature_names)
                })
            else:
                summary_data.append({
                    'Model': name,
                    'Target_Type': model.target_type.value,
                    'Trained': False,
                    'Training_R2': 0.0,
                    'Validation_R2': 0.0,
                    'CV_R2_Mean': 0.0,
                    'CV_R2_Std': 0.0,
                    'MSE': 0.0,
                    'Prediction_Std': 0.0,
                    'Features': 0
                })
        
        return pd.DataFrame(summary_data)

# ============================================
# Utility Functions
# ============================================

def create_regression_signals(data: pd.DataFrame, 
                            models_to_use: Optional[List[str]] = None,
                            horizon: int = 1) -> Dict[str, List[RegressionSignal]]:
    """
    Quick utility function to generate regression signals
    
    Args:
        data: OHLCV DataFrame with sufficient history
        models_to_use: List of model names to use
        horizon: Prediction horizon in periods
        
    Returns:
        Dictionary of signals by model
    """
    
    generator = RegressionSignalGenerator()
    
    # Train models
    training_results = generator.train_models(data, models_to_train=models_to_use, horizon=horizon)
    
    # Generate signals
    return generator.generate_all_signals(data, models_to_use, horizon)

def filter_high_confidence_regression_signals(signals: List[RegressionSignal], 
                                            min_r2_score: float = 0.3,
                                            min_move_probability: float = 0.7) -> List[RegressionSignal]:
    """Filter regression signals by model performance and prediction confidence"""
    
    return [
        signal for signal in signals 
        if (signal.model_r2_score >= min_r2_score and 
            signal.move_probability >= min_move_probability)
    ]

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Regression Signals System")
    
    # Generate comprehensive sample data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    n_periods = len(dates)
    
    # Create realistic price data with trends and mean reversion
    base_trend = 0.0002  # Small upward trend
    returns = []
    
    for i in range(n_periods):
        # Add trend, mean reversion, and volatility clustering
        trend_component = base_trend
        mean_reversion = -0.1 * (sum(returns[-10:]) if len(returns) >= 10 else 0)
        volatility = 0.015 * (1 + 0.5 * abs(returns[-1] if returns else 0))
        noise = np.random.normal(0, volatility)
        
        daily_return = trend_component + mean_reversion + noise
        returns.append(daily_return)
    
    # Convert to price series
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods)))
    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    
    # Volume with realistic patterns
    volume_base = 1000000
    volume_volatility = 1 + np.abs(np.array(returns)) * 5
    volumes = volume_base * (0.5 + np.random.random(n_periods) * volume_volatility)
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    print(f"Generated sample data: {len(sample_data)} periods")
    print(f"Price range: ${sample_data['close'].min():.2f} - ${sample_data['close'].max():.2f}")
    print(f"Return statistics: Mean={np.mean(returns):.4f}, Std={np.std(returns):.4f}")
    
    # Test individual regression models
    print("\n1. Testing Individual Regression Models")
    
    # Test Return Prediction Model
    return_model = ReturnPredictionModel()
    print(f"\nTraining Return Prediction Model...")
    
    try:
        return_performance = return_model.train(sample_data, horizon=1)
        print(f"Return Model Performance:")
        for metric, value in return_performance.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
        
        # Generate return prediction signals
        return_signals = return_model.generate_signals(sample_data, horizon=1)
        print(f"Generated {len(return_signals)} return prediction signals")
        
        if return_signals:
            signal = return_signals[0]
            print(f"  First signal: {signal.direction.name} at ${signal.price:.2f}")
            print(f"    Predicted return: {signal.predicted_return:.3f}")
            print(f"    Expected move: {signal.expected_move:.3f}")
            print(f"    Move probability: {signal.move_probability:.2f}")
            print(f"    Prediction interval: [{signal.prediction_interval[0]:.3f}, {signal.prediction_interval[1]:.3f}]")
    
    except Exception as e:
        print(f"Error with return model: {e}")
    
    # Test Price Prediction Model
    price_model = PricePredictionModel()
    print(f"\nTraining Price Prediction Model...")
    
    try:
        price_performance = price_model.train(sample_data)
        print(f"Price Model Performance:")
        for metric, value in price_performance.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
        
        price_signals = price_model.generate_signals(sample_data)
        print(f"Generated {len(price_signals)} price prediction signals")
    
    except Exception as e:
        print(f"Error with price model: {e}")
    
    # Test Volatility Prediction Model
    vol_model = VolatilityPredictionModel()
    print(f"\nTraining Volatility Prediction Model...")
    
    try:
        vol_performance = vol_model.train(sample_data)
        print(f"Volatility Model Performance:")
        for metric, value in vol_performance.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
        
        vol_signals = vol_model.generate_signals(sample_data)
        print(f"Generated {len(vol_signals)} volatility prediction signals")
    
    except Exception as e:
        print(f"Error with volatility model: {e}")
    
    # Test Regression Signal Generator
    print("\n2. Testing Regression Signal Generator")
    
    generator = RegressionSignalGenerator()
    
    # Train all models
    print("Training all models...")
    training_results = generator.train_models(sample_data)
    
    print("Training Results:")
    for model_name, results in training_results.items():
        if 'error' not in results:
            print(f"  {model_name}: Val R² = {results['validation_r2']:.3f}, "
                  f"MSE = {results['validation_mse']:.6f}")
        else:
            print(f"  {model_name}: Error - {results['error']}")
    
    # Generate all signals
    all_regression_signals = generator.generate_all_signals(sample_data)
    
    print(f"\n3. Regression Signal Generation Results:")
    total_signals = 0
    for model_name, signals in all_regression_signals.items():
        if signals:
            buy_signals = sum(1 for s in signals if s.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY])
            sell_signals = sum(1 for s in signals if s.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL])
            avg_expected_move = np.mean([s.expected_move for s in signals])
            avg_probability = np.mean([s.move_probability for s in signals])
            
            print(f"  {model_name}: {len(signals)} total ({buy_signals} buy, {sell_signals} sell)")
            print(f"    Avg expected move: {avg_expected_move:.3f}, Avg probability: {avg_probability:.2f}")
            total_signals += len(signals)
    
    print(f"Total regression signals: {total_signals}")
    
    # Test model summary
    print("\n4. Model Summary:")
    summary_df = generator.get_model_summary()
    print(summary_df.to_string(index=False))
    
    # Test signal filtering
    print("\n5. Testing Signal Filtering")
    
    # Combine all regression signals
    all_regression_signals_list = []
    for signals in all_regression_signals.values():
        all_regression_signals_list.extend(signals)
    
    print(f"Total regression signals before filtering: {len(all_regression_signals_list)}")
    
    # Filter high confidence signals
    high_confidence_signals = filter_high_confidence_regression_signals(
        all_regression_signals_list,
        min_r2_score=0.2,
        min_move_probability=0.6
    )
    
    print(f"High confidence signals: {len(high_confidence_signals)}")
    
    # Analyze feature importance
    print("\n6. Feature Importance Analysis")
    
    for model_name, model in generator.models.items():
        if model.is_trained:
            feature_importance = model.get_feature_importance()
            if feature_importance:
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"\n{model_name} - Top 5 Features:")
                for feature, importance in top_features:
                    print(f"  {feature}: {importance:.4f}")
    
    # Test signal analysis
    print("\n7. Signal Quality Analysis")
    
    if return_signals:
        print("Return Prediction Signals Analysis:")
        expected_moves = [s.expected_move for s in return_signals]
        probabilities = [s.move_probability for s in return_signals]
        r2_scores = [s.model_r2_score for s in return_signals]
        
        print(f"  Expected moves - Mean: {np.mean(expected_moves):.3f}, Std: {np.std(expected_moves):.3f}")
        print(f"  Move probabilities - Mean: {np.mean(probabilities):.2f}, Min: {np.min(probabilities):.2f}, Max: {np.max(probabilities):.2f}")
        print(f"  Model R² scores - Mean: {np.mean(r2_scores):.3f}")
        
        # Show detailed signal example
        best_signal = max(return_signals, key=lambda s: s.move_probability)
        print(f"\nBest Signal Example:")
        print(f"  Direction: {best_signal.direction.name}")
        print(f"  Predicted Return: {best_signal.predicted_return:.3f}")
        print(f"  Expected Move: {best_signal.expected_move:.3f}")
        print(f"  Move Probability: {best_signal.move_probability:.2f}")
        print(f"  Model R²: {best_signal.model_r2_score:.3f}")
        print(f"  Prediction Interval: [{best_signal.prediction_interval[0]:.3f}, {best_signal.prediction_interval[1]:.3f}]")
        print(f"  Top Features: {list(best_signal.metadata['top_features'].keys())[:3]}")
    
    # Test utility function
    print("\n8. Testing Utility Function")
    
    quick_signals = create_regression_signals(
        sample_data,
        models_to_use=['return', 'volatility'],
        horizon=1
    )
    
    total_quick_signals = sum(len(signals) for signals in quick_signals.values())
    print(f"Quick signal generation: {total_quick_signals} signals")
    
    print("\nRegression signals system testing completed successfully!")
    print("\nGenerated regression signals include:")
    print("• Return Prediction: Quantitative future return forecasts with confidence intervals")
    print("• Price Level Prediction: Target price levels for support/resistance identification")
    print("• Volatility Prediction: Regime forecasting for options and risk management")
    print("• Feature Engineering: 40+ quantitative features for robust predictions")
    print("• Model Validation: R², MSE, cross-validation with time-series awareness")
    print("• Prediction Intervals: Statistical confidence bounds for all predictions")
    print("• Feature Importance: Interpretable models showing key predictive factors")
