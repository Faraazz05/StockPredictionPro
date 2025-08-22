# ============================================
# StockPredictionPro - src/trading/signals/classification_signals.py
# Machine learning classification-based trading signals for financial markets
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it
from .technical_signals import TechnicalSignal, SignalDirection, SignalConfidence

logger = get_logger('trading.signals.classification_signals')

# ============================================
# Classification Signal Data Structures
# ============================================

class PredictionHorizon(Enum):
    """Prediction time horizons"""
    INTRADAY = "1H"
    DAILY = "1D"
    WEEKLY = "1W"
    MONTHLY = "1M"

class ClassificationTarget(Enum):
    """Classification target types"""
    DIRECTION = "direction"          # Up/Down/Sideways
    VOLATILITY = "volatility"        # High/Low volatility
    REGIME = "regime"                # Bull/Bear/Sideways market
    BREAKOUT = "breakout"            # Breakout/Breakdown/Consolidation
    REVERSAL = "reversal"            # Reversal/Continuation

@dataclass
class ClassificationSignal(TechnicalSignal):
    """Extended signal class for ML classification signals"""
    model_name: str = ""
    prediction_horizon: PredictionHorizon = PredictionHorizon.DAILY
    target_type: ClassificationTarget = ClassificationTarget.DIRECTION
    
    # ML-specific attributes
    prediction_probability: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_confidence: float = 0.0
    cross_validation_score: float = 0.0
    
    # Prediction details
    predicted_class: str = ""
    predicted_return: Optional[float] = None
    prediction_range: Optional[Tuple[float, float]] = None

# ============================================
# Base Classification Model
# ============================================

class BaseClassificationModel:
    """
    Base class for all classification models used in signal generation.
    
    This class provides common functionality for training models,
    making predictions, and generating trading signals.
    """
    
    def __init__(self, name: str, model, target_type: ClassificationTarget = ClassificationTarget.DIRECTION):
        self.name = name
        self.model = model
        self.target_type = target_type
        self.is_trained = False
        
        # Feature engineering
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
        # Performance tracking
        self.training_score = 0.0
        self.validation_score = 0.0
        self.cv_scores = []
        
        logger.info(f"Initialized {name} classification model for {target_type.value}")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement prepare_features method")
    
    def prepare_targets(self, data: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """Prepare target variables based on classification type"""
        
        if self.target_type == ClassificationTarget.DIRECTION:
            # Next period return direction
            returns = data['close'].pct_change(periods=horizon).shift(-horizon)
            
            # Convert to categorical: -1 (down), 0 (flat), 1 (up)
            targets = pd.cut(
                returns,
                bins=[-np.inf, -0.005, 0.005, np.inf],
                labels=[-1, 0, 1]
            ).astype(int)
            
            return targets
        
        elif self.target_type == ClassificationTarget.VOLATILITY:
            # Next period volatility level
            returns = data['close'].pct_change()
            rolling_vol = returns.rolling(window=horizon).std().shift(-horizon)
            vol_threshold = rolling_vol.quantile(0.7)  # Top 30% as high volatility
            
            targets = (rolling_vol > vol_threshold).astype(int)
            return targets
        
        elif self.target_type == ClassificationTarget.BREAKOUT:
            # Breakout from consolidation patterns
            high_roll = data['high'].rolling(window=20).max()
            low_roll = data['low'].rolling(window=20).min()
            price_range = (high_roll - low_roll) / data['close']
            
            # Breakout conditions
            future_high = data['high'].shift(-horizon)
            future_low = data['low'].shift(-horizon)
            
            breakout_up = (future_high > high_roll) & (price_range < 0.05)
            breakout_down = (future_low < low_roll) & (price_range < 0.05)
            
            targets = pd.Series(0, index=data.index)  # No breakout
            targets[breakout_up] = 1   # Upward breakout
            targets[breakout_down] = -1 # Downward breakout
            
            return targets
        
        else:
            raise ValueError(f"Unsupported target type: {self.target_type}")
    
    @time_it("model_training")
    def train(self, data: pd.DataFrame, horizon: int = 1, validation_split: float = 0.2) -> Dict[str, float]:
        """Train the classification model"""
        
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
        
        # Encode labels
        targets_encoded = self.label_encoder.fit_transform(targets_clean)
        
        # Split data for validation
        split_idx = int(len(features_scaled) * (1 - validation_split))
        
        X_train = features_scaled.iloc[:split_idx]
        X_val = features_scaled.iloc[split_idx:]
        y_train = targets_encoded[:split_idx]
        y_val = targets_encoded[split_idx:]
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate performance metrics
        train_predictions = self.model.predict(X_train)
        val_predictions = self.model.predict(X_val)
        
        self.training_score = accuracy_score(y_train, train_predictions)
        self.validation_score = accuracy_score(y_val, val_predictions)
        
        # Cross-validation
        if len(features_scaled) > 100:
            self.cv_scores = cross_val_score(
                self.model, features_scaled, targets_encoded, cv=5
            )
        
        performance_metrics = {
            'training_accuracy': self.training_score,
            'validation_accuracy': self.validation_score,
            'cv_mean': np.mean(self.cv_scores) if self.cv_scores else 0.0,
            'cv_std': np.std(self.cv_scores) if self.cv_scores else 0.0,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
        
        logger.info(f"{self.name} training completed - "
                   f"Train Acc: {self.training_score:.3f}, "
                   f"Val Acc: {self.validation_score:.3f}")
        
        return performance_metrics
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions and return probabilities"""
        
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
        probabilities = None
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)
        elif hasattr(self.model, 'decision_function'):
            # Convert decision function to probabilities for SVM
            decision_scores = self.model.decision_function(features_scaled)
            probabilities = 1 / (1 + np.exp(-decision_scores))  # Sigmoid
        
        return predictions, probabilities
    
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
            coefficients = np.abs(self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_)
            for feature, coef in zip(self.feature_names, coefficients):
                importance_dict[feature] = float(coef)
        
        return importance_dict

# ============================================
# Direction Prediction Model
# ============================================

class DirectionPredictionModel(BaseClassificationModel):
    """
    Model for predicting price direction (Up/Down/Sideways).
    
    Uses technical indicators, price patterns, and market features
    to predict next-period price movement direction.
    """
    
    def __init__(self, model=None, lookback_periods: int = 20):
        if model is None:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        
        super().__init__("Direction Prediction", model, ClassificationTarget.DIRECTION)
        self.lookback_periods = lookback_periods
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for direction prediction"""
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns_1'] = data['close'].pct_change(1)
        features['returns_3'] = data['close'].pct_change(3)
        features['returns_5'] = data['close'].pct_change(5)
        features['returns_10'] = data['close'].pct_change(10)
        
        # Volatility features
        features['volatility_5'] = features['returns_1'].rolling(5).std()
        features['volatility_20'] = features['returns_1'].rolling(20).std()
        features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
        
        # Price position features
        features['price_to_sma_20'] = data['close'] / data['close'].rolling(20).mean()
        features['price_to_sma_50'] = data['close'] / data['close'].rolling(50).mean()
        features['price_to_ema_12'] = data['close'] / data['close'].ewm(span=12).mean()
        
        # High/Low relative position
        high_20 = data['high'].rolling(20).max()
        low_20 = data['low'].rolling(20).min()
        features['price_position'] = (data['close'] - low_20) / (high_20 - low_20)
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_price_trend'] = (features['returns_1'] * features['volume_ratio']).rolling(5).mean()
        
        # Technical indicators
        # RSI
        delta = data['close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gains = gains.rolling(14).mean()
        avg_losses = losses.rolling(14).mean()
        rs = avg_gains / avg_losses
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd - macd_signal
        
        # Bollinger Bands
        sma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / sma_20
        
        # Momentum indicators
        features['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        features['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        
        # Support/Resistance levels
        features['distance_from_high_20'] = (high_20 - data['close']) / data['close']
        features['distance_from_low_20'] = (data['close'] - low_20) / data['close']
        
        # Market microstructure features
        features['high_low_ratio'] = (data['high'] - data['low']) / data['close']
        features['open_close_ratio'] = (data['close'] - data['open']) / data['open']
        features['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        
        return features.dropna()
    
    @time_it("direction_signal_generation")
    def generate_signals(self, data: pd.DataFrame, 
                        prediction_horizon: PredictionHorizon = PredictionHorizon.DAILY) -> List[ClassificationSignal]:
        """Generate direction prediction signals"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before generating signals")
        
        # Prepare features
        features = self.prepare_features(data)
        
        if len(features) == 0:
            return []
        
        # Make predictions
        predictions, probabilities = self.predict(features)
        
        # Decode predictions
        predicted_directions = self.label_encoder.inverse_transform(predictions)
        
        signals = []
        
        # Get feature importance for context
        feature_importance = self.get_feature_importance()
        
        for i, idx in enumerate(features.index):
            if i >= len(predictions):
                break
            
            pred_class = predicted_directions[i]
            pred_prob = probabilities[i] if probabilities is not None else np.array([0.5])
            
            # Convert prediction to signal direction
            if pred_class == 1:  # Up
                signal_direction = SignalDirection.BUY
                signal_strength = float(pred_prob[self.label_encoder.transform([1])[0]])
            elif pred_class == -1:  # Down
                signal_direction = SignalDirection.SELL  
                signal_strength = float(pred_prob[self.label_encoder.transform([-1])[0]])
            else:  # Sideways
                signal_direction = SignalDirection.HOLD
                signal_strength = float(pred_prob[self.label_encoder.transform([0])[0]])
            
            # Skip weak signals
            if signal_direction == SignalDirection.HOLD or signal_strength < 0.6:
                continue
            
            # Determine confidence based on probability and model performance
            model_confidence = self.validation_score
            combined_confidence = signal_strength * model_confidence
            
            if combined_confidence >= 0.8:
                confidence = SignalConfidence.VERY_HIGH
            elif combined_confidence >= 0.7:
                confidence = SignalConfidence.HIGH
            elif combined_confidence >= 0.6:
                confidence = SignalConfidence.MEDIUM
            else:
                confidence = SignalConfidence.LOW
            
            # Create classification signal
            signal = ClassificationSignal(
                timestamp=idx,
                symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                indicator=f"{self.name}_Direction",
                direction=signal_direction,
                strength=signal_strength,
                confidence=confidence,
                price=data.loc[idx, 'close'],
                indicator_value=float(pred_class),
                
                # Classification-specific attributes
                model_name=self.name,
                prediction_horizon=prediction_horizon,
                target_type=self.target_type,
                prediction_probability=signal_strength,
                feature_importance=feature_importance,
                model_confidence=model_confidence,
                cross_validation_score=np.mean(self.cv_scores) if self.cv_scores else 0.0,
                predicted_class=str(pred_class),
                
                metadata={
                    'prediction_probabilities': pred_prob.tolist() if probabilities is not None else [],
                    'top_features': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]),
                    'model_accuracy': self.validation_score
                }
            )
            
            signals.append(signal)
        
        logger.info(f"Generated {len(signals)} direction prediction signals")
        return signals

# ============================================
# Volatility Prediction Model
# ============================================

class VolatilityPredictionModel(BaseClassificationModel):
    """
    Model for predicting volatility regimes (High/Low volatility periods).
    
    Helps identify when markets are likely to experience high or low volatility,
    useful for options trading and risk management.
    """
    
    def __init__(self, model=None):
        if model is None:
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        
        super().__init__("Volatility Prediction", model, ClassificationTarget.VOLATILITY)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for volatility prediction"""
        
        features = pd.DataFrame(index=data.index)
        
        # Historical volatility features
        returns = data['close'].pct_change()
        features['realized_vol_5'] = returns.rolling(5).std() * np.sqrt(252)
        features['realized_vol_20'] = returns.rolling(20).std() * np.sqrt(252)
        features['realized_vol_60'] = returns.rolling(60).std() * np.sqrt(252)
        
        # Volatility ratios
        features['vol_ratio_5_20'] = features['realized_vol_5'] / features['realized_vol_20']
        features['vol_ratio_20_60'] = features['realized_vol_20'] / features['realized_vol_60']
        
        # Price range features
        features['high_low_range'] = (data['high'] - data['low']) / data['close']
        features['avg_range_5'] = features['high_low_range'].rolling(5).mean()
        features['avg_range_20'] = features['high_low_range'].rolling(20).mean()
        
        # Volume-volatility relationship
        volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_volatility'] = (returns.abs() * volume_ratio).rolling(10).mean()
        
        # VIX-like features (implied volatility proxies)
        # Garman-Klass volatility estimator
        ln_hl = np.log(data['high'] / data['low'])
        ln_co = np.log(data['close'] / data['open'])
        gk_vol = 0.5 * ln_hl**2 - (2*np.log(2) - 1) * ln_co**2
        features['gk_volatility'] = gk_vol.rolling(20).mean()
        
        # Volatility clustering features
        vol_shocks = (returns.abs() - returns.abs().rolling(20).mean()) / returns.abs().rolling(20).std()
        features['vol_shock'] = vol_shocks
        features['vol_persistence'] = vol_shocks.rolling(5).sum()
        
        # Market stress indicators
        features['extreme_moves'] = (returns.abs() > returns.rolling(60).std() * 2).rolling(10).sum()
        features['gap_volatility'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)).rolling(10).std()
        
        return features.dropna()

# ============================================
# Breakout Prediction Model
# ============================================

class BreakoutPredictionModel(BaseClassificationModel):
    """
    Model for predicting breakout events from consolidation patterns.
    
    Identifies when stocks are likely to break out of trading ranges,
    useful for momentum and trend-following strategies.
    """
    
    def __init__(self, model=None):
        if model is None:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        super().__init__("Breakout Prediction", model, ClassificationTarget.BREAKOUT)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for breakout prediction"""
        
        features = pd.DataFrame(index=data.index)
        
        # Consolidation pattern features
        high_20 = data['high'].rolling(20).max()
        low_20 = data['low'].rolling(20).min()
        range_20 = high_20 - low_20
        features['consolidation_range'] = range_20 / data['close']
        features['price_position_in_range'] = (data['close'] - low_20) / range_20
        
        # Volatility compression
        vol_current = data['close'].pct_change().rolling(10).std()
        vol_long_term = data['close'].pct_change().rolling(60).std()
        features['volatility_compression'] = vol_current / vol_long_term
        
        # Volume patterns
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_trend'] = data['volume'].rolling(10).mean() / data['volume'].rolling(30).mean()
        
        # Support/Resistance strength
        touches_high = (data['high'] >= high_20 * 0.99).rolling(20).sum()
        touches_low = (data['low'] <= low_20 * 1.01).rolling(20).sum()
        features['resistance_strength'] = touches_high
        features['support_strength'] = touches_low
        
        # Momentum building
        returns = data['close'].pct_change()
        features['momentum_5'] = returns.rolling(5).mean()
        features['momentum_acceleration'] = returns.rolling(5).mean() - returns.rolling(10).mean()
        
        # Technical indicator patterns
        # RSI approaching mid-line
        delta = data['close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gains = gains.rolling(14).mean()
        avg_losses = losses.rolling(14).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        features['rsi_mid_proximity'] = 1 - abs(rsi - 50) / 50
        
        # MACD convergence
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        features['macd_convergence'] = abs(macd - macd_signal) / data['close']
        
        return features.dropna()

# ============================================
# Classification Signal Generator
# ============================================

class ClassificationSignalGenerator:
    """
    Comprehensive ML classification signal generator.
    
    Manages multiple classification models and generates
    signals based on machine learning predictions.
    """
    
    def __init__(self):
        self.models = {}
        self.signals_generated = 0
        
        # Initialize default models
        self.models['direction'] = DirectionPredictionModel()
        self.models['volatility'] = VolatilityPredictionModel()
        self.models['breakout'] = BreakoutPredictionModel()
        
        logger.info("Initialized ClassificationSignalGenerator with 3 default models")
    
    def add_model(self, name: str, model: BaseClassificationModel):
        """Add a custom classification model"""
        self.models[name] = model
        logger.info(f"Added custom model: {name}")
    
    def train_models(self, data: pd.DataFrame, models_to_train: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Train specified models or all models"""
        
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        training_results = {}
        
        for model_name in models_to_train:
            if model_name not in self.models:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            try:
                logger.info(f"Training {model_name} model...")
                model = self.models[model_name]
                results = model.train(data)
                training_results[model_name] = results
                
                logger.info(f"{model_name} training completed - "
                           f"Validation accuracy: {results['validation_accuracy']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name} model: {e}")
                training_results[model_name] = {'error': str(e)}
        
        return training_results
    
    @time_it("generate_all_classification_signals")
    def generate_all_signals(self, data: pd.DataFrame, 
                           models_to_use: Optional[List[str]] = None,
                           prediction_horizon: PredictionHorizon = PredictionHorizon.DAILY) -> Dict[str, List[ClassificationSignal]]:
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
                signals = model.generate_signals(data, prediction_horizon)
                all_signals[model_name] = signals
                
                logger.info(f"{model_name}: Generated {len(signals)} signals")
                
            except Exception as e:
                logger.error(f"Error generating signals for {model_name}: {e}")
                all_signals[model_name] = []
        
        # Update total signals count
        self.signals_generated = sum(len(signals) for signals in all_signals.values())
        
        logger.info(f"Total classification signals generated: {self.signals_generated}")
        
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
                    'Training_Accuracy': model.training_score,
                    'Validation_Accuracy': model.validation_score,
                    'CV_Mean': cv_mean,
                    'CV_Std': cv_std,
                    'Features': len(model.feature_names)
                })
            else:
                summary_data.append({
                    'Model': name,
                    'Target_Type': model.target_type.value,
                    'Trained': False,
                    'Training_Accuracy': 0.0,
                    'Validation_Accuracy': 0.0,
                    'CV_Mean': 0.0,
                    'CV_Std': 0.0,
                    'Features': 0
                })
        
        return pd.DataFrame(summary_data)
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        
        models_to_save = {
            name: {
                'model': model.model if model.is_trained else None,
                'scaler': model.scaler if model.is_trained else None,
                'label_encoder': model.label_encoder if model.is_trained else None,
                'feature_names': model.feature_names,
                'performance': {
                    'training_score': model.training_score,
                    'validation_score': model.validation_score,
                    'cv_scores': model.cv_scores
                }
            }
            for name, model in self.models.items()
        }
        
        joblib.dump(models_to_save, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        
        try:
            saved_models = joblib.load(filepath)
            
            for name, saved_data in saved_models.items():
                if name in self.models and saved_data['model'] is not None:
                    model = self.models[name]
                    model.model = saved_data['model']
                    model.scaler = saved_data['scaler']
                    model.label_encoder = saved_data['label_encoder']
                    model.feature_names = saved_data['feature_names']
                    model.training_score = saved_data['performance']['training_score']
                    model.validation_score = saved_data['performance']['validation_score']
                    model.cv_scores = saved_data['performance']['cv_scores']
                    model.is_trained = True
            
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# ============================================
# Utility Functions
# ============================================

def create_classification_signals(data: pd.DataFrame, 
                                models_to_use: Optional[List[str]] = None,
                                prediction_horizon: PredictionHorizon = PredictionHorizon.DAILY) -> Dict[str, List[ClassificationSignal]]:
    """
    Quick utility function to generate classification signals
    
    Args:
        data: OHLCV DataFrame with sufficient history
        models_to_use: List of model names to use
        prediction_horizon: Prediction time horizon
        
    Returns:
        Dictionary of signals by model
    """
    
    generator = ClassificationSignalGenerator()
    
    # Train models if needed
    training_results = generator.train_models(data)
    
    # Generate signals
    return generator.generate_all_signals(data, models_to_use, prediction_horizon)

def filter_high_confidence_ml_signals(signals: List[ClassificationSignal], 
                                     min_model_confidence: float = 0.7,
                                     min_prediction_probability: float = 0.8) -> List[ClassificationSignal]:
    """Filter ML signals by model confidence and prediction probability"""
    
    return [
        signal for signal in signals 
        if (signal.model_confidence >= min_model_confidence and 
            signal.prediction_probability >= min_prediction_probability)
    ]

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Classification Signals System")
    
    # Generate comprehensive sample data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    n_periods = len(dates)
    
    # Create realistic price data with trends and patterns
    base_trend = np.linspace(0, 0.3, n_periods)  # 30% upward trend over period
    noise = np.random.normal(0, 0.02, n_periods)
    regime_changes = np.sin(np.linspace(0, 4*np.pi, n_periods)) * 0.1
    
    log_returns = base_trend/n_periods + noise + regime_changes/n_periods
    prices = 100 * np.exp(np.cumsum(log_returns))
    
    # Generate OHLC from close prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    
    # Volume with realistic patterns
    volume_base = 1000000
    volume_trend = 1 + 0.5 * np.abs(log_returns)  # Higher volume on big moves
    volumes = volume_base * (0.5 + np.random.random(n_periods) * volume_trend)
    
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
    
    # Test individual models
    print("\n1. Testing Individual Classification Models")
    
    # Test Direction Prediction Model
    direction_model = DirectionPredictionModel()
    print(f"\nTraining Direction Prediction Model...")
    
    try:
        direction_performance = direction_model.train(sample_data)
        print(f"Direction Model Performance:")
        for metric, value in direction_performance.items():
            print(f"  {metric}: {value:.3f}" if isinstance(value, float) else f"  {metric}: {value}")
        
        # Generate direction signals
        direction_signals = direction_model.generate_signals(sample_data)
        print(f"Generated {len(direction_signals)} direction signals")
        
        if direction_signals:
            print(f"  First signal: {direction_signals[0].direction.name} at ${direction_signals[0].price:.2f} "
                  f"(prob: {direction_signals[0].prediction_probability:.2f})")
    
    except Exception as e:
        print(f"Error with direction model: {e}")
    
    # Test Volatility Prediction Model
    volatility_model = VolatilityPredictionModel()
    print(f"\nTraining Volatility Prediction Model...")
    
    try:
        vol_performance = volatility_model.train(sample_data)
        print(f"Volatility Model Performance:")
        for metric, value in vol_performance.items():
            print(f"  {metric}: {value:.3f}" if isinstance(value, float) else f"  {metric}: {value}")
        
        # Generate volatility signals  
        vol_signals = volatility_model.generate_signals(sample_data)
        print(f"Generated {len(vol_signals)} volatility signals")
    
    except Exception as e:
        print(f"Error with volatility model: {e}")
    
    # Test Breakout Prediction Model
    breakout_model = BreakoutPredictionModel()
    print(f"\nTraining Breakout Prediction Model...")
    
    try:
        breakout_performance = breakout_model.train(sample_data)
        print(f"Breakout Model Performance:")
        for metric, value in breakout_performance.items():
            print(f"  {metric}: {value:.3f}" if isinstance(value, float) else f"  {metric}: {value}")
        
        # Generate breakout signals
        breakout_signals = breakout_model.generate_signals(sample_data)
        print(f"Generated {len(breakout_signals)} breakout signals")
    
    except Exception as e:
        print(f"Error with breakout model: {e}")
    
    # Test Classification Signal Generator
    print("\n2. Testing Classification Signal Generator")
    
    generator = ClassificationSignalGenerator()
    
    # Train all models
    print("Training all models...")
    training_results = generator.train_models(sample_data)
    
    print("Training Results:")
    for model_name, results in training_results.items():
        if 'error' not in results:
            print(f"  {model_name}: Val Acc = {results['validation_accuracy']:.3f}, "
                  f"CV = {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        else:
            print(f"  {model_name}: Error - {results['error']}")
    
    # Generate all signals
    all_ml_signals = generator.generate_all_signals(sample_data)
    
    print(f"\n3. Classification Signal Generation Results:")
    total_signals = 0
    for model_name, signals in all_ml_signals.items():
        if signals:
            buy_signals = sum(1 for s in signals if s.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY])
            sell_signals = sum(1 for s in signals if s.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL])
            avg_prob = np.mean([s.prediction_probability for s in signals])
            avg_confidence = np.mean([s.model_confidence for s in signals])
            
            print(f"  {model_name}: {len(signals)} total ({buy_signals} buy, {sell_signals} sell)")
            print(f"    Avg prediction prob: {avg_prob:.2f}, Avg model confidence: {avg_confidence:.2f}")
            total_signals += len(signals)
    
    print(f"Total ML signals: {total_signals}")
    
    # Test model summary
    print("\n4. Model Summary:")
    summary_df = generator.get_model_summary()
    print(summary_df.to_string(index=False))
    
    # Test signal filtering
    print("\n5. Testing Signal Filtering")
    
    # Combine all ML signals
    all_ml_signals_list = []
    for signals in all_ml_signals.values():
        all_ml_signals_list.extend(signals)
    
    print(f"Total ML signals before filtering: {len(all_ml_signals_list)}")
    
    # Filter high confidence signals
    high_confidence_signals = filter_high_confidence_ml_signals(
        all_ml_signals_list,
        min_model_confidence=0.6,
        min_prediction_probability=0.7
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
    
    # Test model persistence
    print("\n7. Testing Model Save/Load")
    
    try:
        # Save models
        generator.save_models("test_ml_models.joblib")
        print("✓ Models saved successfully")
        
        # Create new generator and load models
        new_generator = ClassificationSignalGenerator()
        new_generator.load_models("test_ml_models.joblib")
        print("✓ Models loaded successfully")
        
        # Test loaded models
        loaded_signals = new_generator.generate_all_signals(sample_data.tail(50))
        total_loaded_signals = sum(len(signals) for signals in loaded_signals.values())
        print(f"✓ Generated {total_loaded_signals} signals from loaded models")
    
    except Exception as e:
        print(f"Model save/load error: {e}")
    
    # Test utility function
    print("\n8. Testing Utility Function")
    
    quick_signals = create_classification_signals(
        sample_data,
        models_to_use=['direction', 'volatility'],
        prediction_horizon=PredictionHorizon.DAILY
    )
    
    total_quick_signals = sum(len(signals) for signals in quick_signals.values())
    print(f"Quick signal generation: {total_quick_signals} signals")
    
    print("\nClassification signals system testing completed successfully!")
    print("\nGenerated ML signals include:")
    print("• Direction Prediction: Multi-class classification for price direction")
    print("• Volatility Prediction: Regime classification for volatility levels")
    print("• Breakout Prediction: Pattern recognition for consolidation breakouts")
    print("• Feature Engineering: Comprehensive technical and statistical features")
    print("• Model Validation: Cross-validation and performance tracking")
    print("• Signal Quality: Probability-based confidence and strength scoring")
    print("• Model Persistence: Save/load functionality for production deployment")
