# ============================================
# StockPredictionPro - src/trading/signals/composite_signals.py
# Composite signal aggregation and ensemble methods for enhanced trading decisions
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import networkx as nx

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it
from .technical_signals import TechnicalSignal, SignalDirection, SignalConfidence
from .classification_signals import ClassificationSignal
from .regression_signals import RegressionSignal

logger = get_logger('trading.signals.composite_signals')

# ============================================
# Composite Signal Data Structures
# ============================================

class AggregationMethod(Enum):
    """Methods for aggregating multiple signals"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    STRENGTH_WEIGHTED = "strength_weighted"
    BAYESIAN_ENSEMBLE = "bayesian_ensemble"
    MACHINE_LEARNING = "machine_learning"
    CONSENSUS_SCORING = "consensus_scoring"
    HIERARCHICAL = "hierarchical"

class ConflictResolution(Enum):
    """Methods for resolving conflicting signals"""
    STRONGEST_SIGNAL = "strongest_signal"
    HIGHEST_CONFIDENCE = "highest_confidence"
    LATEST_SIGNAL = "latest_signal"
    CONSERVATIVE = "conservative"  # Default to HOLD
    AGGRESSIVE = "aggressive"      # Take stronger directional signal
    WEIGHTED_CONSENSUS = "weighted_consensus"

@dataclass
class CompositeSignal(TechnicalSignal):
    """Extended signal class for composite signals"""
    component_signals: List[TechnicalSignal] = field(default_factory=list)
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    conflict_resolution: ConflictResolution = ConflictResolution.WEIGHTED_CONSENSUS
    
    # Composite-specific attributes
    consensus_score: float = 0.0
    signal_diversity: float = 0.0
    component_count: int = 0
    technical_weight: float = 0.0
    ml_weight: float = 0.0
    
    # Quality metrics
    agreement_ratio: float = 0.0
    confidence_spread: float = 0.0
    strength_spread: float = 0.0
    
    # Signal breakdown
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    
    def __post_init__(self):
        """Post-initialization processing"""
        super().__post_init__()
        self.component_count = len(self.component_signals)
        self._analyze_components()
    
    def _analyze_components(self):
        """Analyze component signals for quality metrics"""
        if not self.component_signals:
            return
        
        # Count signal directions
        directions = [s.direction for s in self.component_signals]
        self.buy_signals = sum(1 for d in directions if d in [SignalDirection.BUY, SignalDirection.STRONG_BUY])
        self.sell_signals = sum(1 for d in directions if d in [SignalDirection.SELL, SignalDirection.STRONG_SELL])
        self.hold_signals = sum(1 for d in directions if d == SignalDirection.HOLD)
        
        # Calculate agreement ratio
        total_signals = len(self.component_signals)
        max_agreement = max(self.buy_signals, self.sell_signals, self.hold_signals)
        self.agreement_ratio = max_agreement / total_signals if total_signals > 0 else 0
        
        # Calculate signal diversity (entropy-based)
        if total_signals > 0:
            probs = [self.buy_signals/total_signals, self.sell_signals/total_signals, self.hold_signals/total_signals]
            probs = [p for p in probs if p > 0]  # Remove zero probabilities
            self.signal_diversity = -sum(p * np.log2(p) for p in probs) if probs else 0
        
        # Calculate confidence and strength spreads
        confidences = [s.confidence.value for s in self.component_signals]
        strengths = [s.strength for s in self.component_signals]
        
        self.confidence_spread = np.std(confidences) if len(confidences) > 1 else 0
        self.strength_spread = np.std(strengths) if len(strengths) > 1 else 0
        
        # Calculate technical vs ML weights
        technical_signals = [s for s in self.component_signals 
                           if not isinstance(s, (ClassificationSignal, RegressionSignal))]
        ml_signals = [s for s in self.component_signals 
                     if isinstance(s, (ClassificationSignal, RegressionSignal))]
        
        self.technical_weight = len(technical_signals) / total_signals
        self.ml_weight = len(ml_signals) / total_signals

# ============================================
# Signal Aggregator Classes
# ============================================

class BaseSignalAggregator:
    """
    Base class for signal aggregation methods.
    
    Provides common functionality for combining multiple signals
    into a single composite signal.
    """
    
    def __init__(self, name: str, method: AggregationMethod):
        self.name = name
        self.method = method
        self.signals_processed = 0
        
        logger.debug(f"Initialized {name} aggregator using {method.value}")
    
    def aggregate(self, signals: List[TechnicalSignal], 
                 weights: Optional[Dict[str, float]] = None) -> CompositeSignal:
        """Aggregate multiple signals - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement aggregate method")
    
    def _calculate_base_metrics(self, signals: List[TechnicalSignal]) -> Dict[str, float]:
        """Calculate base metrics from component signals"""
        if not signals:
            return {}
        
        # Direction analysis
        directions = [s.direction for s in signals]
        buy_count = sum(1 for d in directions if d in [SignalDirection.BUY, SignalDirection.STRONG_BUY])
        sell_count = sum(1 for d in directions if d in [SignalDirection.SELL, SignalDirection.STRONG_SELL])
        hold_count = sum(1 for d in directions if d == SignalDirection.HOLD)
        
        # Strength and confidence statistics
        strengths = [s.strength for s in signals]
        confidences = [s.confidence.value for s in signals]
        
        return {
            'avg_strength': np.mean(strengths),
            'std_strength': np.std(strengths),
            'avg_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'buy_ratio': buy_count / len(signals),
            'sell_ratio': sell_count / len(signals),
            'hold_ratio': hold_count / len(signals),
            'consensus_score': max(buy_count, sell_count, hold_count) / len(signals)
        }
    
    def _resolve_conflicts(self, signals: List[TechnicalSignal], 
                          resolution: ConflictResolution) -> Tuple[SignalDirection, float, SignalConfidence]:
        """Resolve conflicting signals based on resolution method"""
        
        if not signals:
            return SignalDirection.HOLD, 0.0, SignalConfidence.LOW
        
        if resolution == ConflictResolution.STRONGEST_SIGNAL:
            strongest = max(signals, key=lambda s: s.strength)
            return strongest.direction, strongest.strength, strongest.confidence
        
        elif resolution == ConflictResolution.HIGHEST_CONFIDENCE:
            highest_conf = max(signals, key=lambda s: s.confidence.value)
            return highest_conf.direction, highest_conf.strength, highest_conf.confidence
        
        elif resolution == ConflictResolution.LATEST_SIGNAL:
            latest = max(signals, key=lambda s: s.timestamp)
            return latest.direction, latest.strength, latest.confidence
        
        elif resolution == ConflictResolution.CONSERVATIVE:
            # Only act if strong consensus
            directions = [s.direction for s in signals]
            buy_count = sum(1 for d in directions if d in [SignalDirection.BUY, SignalDirection.STRONG_BUY])
            sell_count = sum(1 for d in directions if d in [SignalDirection.SELL, SignalDirection.STRONG_SELL])
            
            total = len(signals)
            if buy_count / total > 0.7:  # 70% consensus for buy
                avg_strength = np.mean([s.strength for s in signals if s.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]])
                return SignalDirection.BUY, avg_strength, SignalConfidence.MEDIUM
            elif sell_count / total > 0.7:  # 70% consensus for sell
                avg_strength = np.mean([s.strength for s in signals if s.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]])
                return SignalDirection.SELL, avg_strength, SignalConfidence.MEDIUM
            else:
                return SignalDirection.HOLD, 0.5, SignalConfidence.LOW
        
        elif resolution == ConflictResolution.AGGRESSIVE:
            # Take the stronger directional signal
            directional_signals = [s for s in signals if s.direction != SignalDirection.HOLD]
            if directional_signals:
                strongest = max(directional_signals, key=lambda s: s.strength * s.confidence.value)
                return strongest.direction, strongest.strength, strongest.confidence
            else:
                return SignalDirection.HOLD, 0.5, SignalConfidence.LOW
        
        else:  # WEIGHTED_CONSENSUS
            return self._weighted_consensus(signals)
    
    def _weighted_consensus(self, signals: List[TechnicalSignal]) -> Tuple[SignalDirection, float, SignalConfidence]:
        """Calculate weighted consensus from multiple signals"""
        
        if not signals:
            return SignalDirection.HOLD, 0.0, SignalConfidence.LOW
        
        # Calculate weighted votes
        buy_weight = 0.0
        sell_weight = 0.0
        hold_weight = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = signal.strength * signal.confidence.value
            total_weight += weight
            
            if signal.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                multiplier = 2.0 if signal.direction == SignalDirection.STRONG_BUY else 1.0
                buy_weight += weight * multiplier
            elif signal.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
                multiplier = 2.0 if signal.direction == SignalDirection.STRONG_SELL else 1.0
                sell_weight += weight * multiplier
            else:
                hold_weight += weight
        
        # Determine consensus direction
        if total_weight == 0:
            return SignalDirection.HOLD, 0.0, SignalConfidence.LOW
        
        buy_ratio = buy_weight / total_weight
        sell_ratio = sell_weight / total_weight
        hold_ratio = hold_weight / total_weight
        
        # Choose direction with highest weighted ratio
        if buy_ratio > sell_ratio and buy_ratio > hold_ratio:
            direction = SignalDirection.BUY
            strength = buy_ratio
        elif sell_ratio > hold_ratio:
            direction = SignalDirection.SELL
            strength = sell_ratio
        else:
            direction = SignalDirection.HOLD
            strength = hold_ratio
        
        # Determine confidence based on consensus strength
        if strength >= 0.8:
            confidence = SignalConfidence.VERY_HIGH
        elif strength >= 0.6:
            confidence = SignalConfidence.HIGH
        elif strength >= 0.4:
            confidence = SignalConfidence.MEDIUM
        else:
            confidence = SignalConfidence.LOW
        
        return direction, strength, confidence

class WeightedAverageAggregator(BaseSignalAggregator):
    """
    Weighted average signal aggregator.
    
    Combines signals using weighted averages of strength and confidence,
    with customizable weights for different signal types.
    """
    
    def __init__(self, 
                 technical_weight: float = 0.4,
                 ml_classification_weight: float = 0.35,
                 ml_regression_weight: float = 0.25):
        super().__init__("Weighted Average", AggregationMethod.WEIGHTED_AVERAGE)
        
        # Normalize weights
        total = technical_weight + ml_classification_weight + ml_regression_weight
        self.technical_weight = technical_weight / total
        self.ml_classification_weight = ml_classification_weight / total
        self.ml_regression_weight = ml_regression_weight / total
        
        logger.info(f"Weighted aggregator: Technical={self.technical_weight:.2f}, "
                   f"Classification={self.ml_classification_weight:.2f}, "
                   f"Regression={self.ml_regression_weight:.2f}")
    
    @time_it("weighted_average_aggregation")
    def aggregate(self, signals: List[TechnicalSignal], 
                 weights: Optional[Dict[str, float]] = None) -> CompositeSignal:
        """Aggregate signals using weighted average method"""
        
        if not signals:
            raise ValueError("Cannot aggregate empty signal list")
        
        # Categorize signals by type
        technical_signals = []
        classification_signals = []
        regression_signals = []
        
        for signal in signals:
            if isinstance(signal, ClassificationSignal):
                classification_signals.append(signal)
            elif isinstance(signal, RegressionSignal):
                regression_signals.append(signal)
            else:
                technical_signals.append(signal)
        
        # Calculate weighted contributions
        weighted_directions = []
        weighted_strengths = []
        weighted_confidences = []
        
        # Process technical signals
        if technical_signals:
            tech_contribution = self._process_signal_group(technical_signals, self.technical_weight)
            weighted_directions.extend(tech_contribution['directions'])
            weighted_strengths.extend(tech_contribution['strengths'])
            weighted_confidences.extend(tech_contribution['confidences'])
        
        # Process classification signals
        if classification_signals:
            class_contribution = self._process_signal_group(classification_signals, self.ml_classification_weight)
            weighted_directions.extend(class_contribution['directions'])
            weighted_strengths.extend(class_contribution['strengths'])
            weighted_confidences.extend(class_contribution['confidences'])
        
        # Process regression signals
        if regression_signals:
            reg_contribution = self._process_signal_group(regression_signals, self.ml_regression_weight)
            weighted_directions.extend(reg_contribution['directions'])
            weighted_strengths.extend(reg_contribution['strengths'])
            weighted_confidences.extend(reg_contribution['confidences'])
        
        # Calculate final composite values
        if not weighted_directions:
            final_direction = SignalDirection.HOLD
            final_strength = 0.0
            final_confidence = SignalConfidence.LOW
        else:
            # Use weighted consensus for final decision
            temp_signals = []
            for i, direction in enumerate(weighted_directions):
                temp_signal = TechnicalSignal(
                    timestamp=signals[0].timestamp,
                    symbol=signals[0].symbol,
                    indicator="temp",
                    direction=direction,
                    strength=weighted_strengths[i],
                    confidence=weighted_confidences[i],
                    price=signals[0].price
                )
                temp_signals.append(temp_signal)
            
            final_direction, final_strength, final_confidence = self._weighted_consensus(temp_signals)
        
        # Create composite signal
        composite = CompositeSignal(
            timestamp=signals[0].timestamp,
            symbol=signals[0].symbol,
            indicator="Weighted_Average_Composite",
            direction=final_direction,
            strength=final_strength,
            confidence=final_confidence,
            price=signals[0].price,
            indicator_value=final_strength,
            
            # Composite-specific attributes
            component_signals=signals,
            aggregation_method=self.method,
            conflict_resolution=ConflictResolution.WEIGHTED_CONSENSUS,
            consensus_score=self._calculate_consensus_score(signals),
            
            metadata={
                'technical_signals': len(technical_signals),
                'classification_signals': len(classification_signals),
                'regression_signals': len(regression_signals),
                'aggregation_weights': {
                    'technical': self.technical_weight,
                    'classification': self.ml_classification_weight,
                    'regression': self.ml_regression_weight
                }
            }
        )
        
        self.signals_processed += 1
        return composite
    
    def _process_signal_group(self, signals: List[TechnicalSignal], group_weight: float) -> Dict[str, List]:
        """Process a group of similar signals"""
        
        if not signals:
            return {'directions': [], 'strengths': [], 'confidences': []}
        
        # Calculate individual signal weights within the group
        individual_weight = group_weight / len(signals)
        
        directions = []
        strengths = []
        confidences = []
        
        for signal in signals:
            # Weight the signal by its strength and confidence
            signal_weight = individual_weight * signal.strength * signal.confidence.value
            
            directions.append(signal.direction)
            strengths.append(signal_weight)
            confidences.append(signal.confidence)
        
        return {
            'directions': directions,
            'strengths': strengths,
            'confidences': confidences
        }
    
    def _calculate_consensus_score(self, signals: List[TechnicalSignal]) -> float:
        """Calculate consensus score for the signal group"""
        
        if not signals:
            return 0.0
        
        directions = [s.direction for s in signals]
        buy_count = sum(1 for d in directions if d in [SignalDirection.BUY, SignalDirection.STRONG_BUY])
        sell_count = sum(1 for d in directions if d in [SignalDirection.SELL, SignalDirection.STRONG_SELL])
        hold_count = sum(1 for d in directions if d == SignalDirection.HOLD)
        
        max_agreement = max(buy_count, sell_count, hold_count)
        return max_agreement / len(signals)

class BayesianEnsembleAggregator(BaseSignalAggregator):
    """
    Bayesian ensemble signal aggregator.
    
    Uses Bayesian inference to combine signals, taking into account
    the historical accuracy of each signal type.
    """
    
    def __init__(self, prior_beliefs: Optional[Dict[str, float]] = None):
        super().__init__("Bayesian Ensemble", AggregationMethod.BAYESIAN_ENSEMBLE)
        
        # Default prior beliefs about signal reliability
        self.prior_beliefs = prior_beliefs or {
            'technical': 0.55,      # 55% prior accuracy for technical signals
            'classification': 0.65,  # 65% prior accuracy for ML classification
            'regression': 0.60,      # 60% prior accuracy for ML regression
            'composite': 0.70        # 70% prior accuracy for composite signals
        }
        
        # Track signal performance for updating beliefs
        self.signal_history = {signal_type: {'correct': 0, 'total': 0} 
                              for signal_type in self.prior_beliefs.keys()}
    
    def aggregate(self, signals: List[TechnicalSignal], 
                 weights: Optional[Dict[str, float]] = None) -> CompositeSignal:
        """Aggregate signals using Bayesian ensemble method"""
        
        if not signals:
            raise ValueError("Cannot aggregate empty signal list")
        
        # Calculate Bayesian weights for each signal
        bayesian_weights = self._calculate_bayesian_weights(signals)
        
        # Apply Bayesian inference
        buy_probability = 0.0
        sell_probability = 0.0
        hold_probability = 0.0
        
        total_weight = sum(bayesian_weights)
        
        for i, signal in enumerate(signals):
            weight = bayesian_weights[i] / total_weight if total_weight > 0 else 0
            signal_prob = signal.strength * signal.confidence.value
            
            if signal.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                buy_probability += weight * signal_prob
            elif signal.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
                sell_probability += weight * signal_prob
            else:
                hold_probability += weight * signal_prob
        
        # Normalize probabilities
        total_prob = buy_probability + sell_probability + hold_probability
        if total_prob > 0:
            buy_probability /= total_prob
            sell_probability /= total_prob
            hold_probability /= total_prob
        
        # Determine final direction and confidence
        if buy_probability > sell_probability and buy_probability > hold_probability:
            final_direction = SignalDirection.BUY
            final_strength = buy_probability
        elif sell_probability > hold_probability:
            final_direction = SignalDirection.SELL
            final_strength = sell_probability
        else:
            final_direction = SignalDirection.HOLD
            final_strength = hold_probability
        
        # Convert probability to confidence level
        if final_strength >= 0.8:
            final_confidence = SignalConfidence.VERY_HIGH
        elif final_strength >= 0.6:
            final_confidence = SignalConfidence.HIGH
        elif final_strength >= 0.4:
            final_confidence = SignalConfidence.MEDIUM
        else:
            final_confidence = SignalConfidence.LOW
        
        # Create composite signal
        composite = CompositeSignal(
            timestamp=signals[0].timestamp,
            symbol=signals[0].symbol,
            indicator="Bayesian_Ensemble_Composite",
            direction=final_direction,
            strength=final_strength,
            confidence=final_confidence,
            price=signals[0].price,
            indicator_value=final_strength,
            
            component_signals=signals,
            aggregation_method=self.method,
            consensus_score=max(buy_probability, sell_probability, hold_probability),
            
            metadata={
                'bayesian_weights': bayesian_weights,
                'probabilities': {
                    'buy': buy_probability,
                    'sell': sell_probability,
                    'hold': hold_probability
                },
                'prior_beliefs': self.prior_beliefs.copy(),
                'signal_performance': self.signal_history.copy()
            }
        )
        
        return composite
    
    def _calculate_bayesian_weights(self, signals: List[TechnicalSignal]) -> List[float]:
        """Calculate Bayesian weights for each signal based on historical performance"""
        
        weights = []
        
        for signal in signals:
            # Determine signal type
            if isinstance(signal, ClassificationSignal):
                signal_type = 'classification'
            elif isinstance(signal, RegressionSignal):
                signal_type = 'regression'
            elif isinstance(signal, CompositeSignal):
                signal_type = 'composite'
            else:
                signal_type = 'technical'
            
            # Get current belief about this signal type's accuracy
            prior = self.prior_beliefs.get(signal_type, 0.5)
            
            # Update belief based on historical performance
            history = self.signal_history[signal_type]
            if history['total'] > 10:  # Only update if we have enough data
                observed_accuracy = history['correct'] / history['total']
                # Bayesian update: weighted average of prior and observed
                alpha = min(history['total'] / 100, 0.7)  # Confidence in observed data
                updated_belief = (1 - alpha) * prior + alpha * observed_accuracy
            else:
                updated_belief = prior
            
            # Weight signal by updated belief and intrinsic signal quality
            signal_quality = signal.strength * signal.confidence.value
            bayesian_weight = updated_belief * signal_quality
            
            weights.append(bayesian_weight)
        
        return weights
    
    def update_signal_performance(self, signal_type: str, was_correct: bool):
        """Update signal performance tracking for Bayesian learning"""
        
        if signal_type in self.signal_history:
            self.signal_history[signal_type]['total'] += 1
            if was_correct:
                self.signal_history[signal_type]['correct'] += 1
            
            logger.debug(f"Updated {signal_type} performance: "
                        f"{self.signal_history[signal_type]['correct']}/{self.signal_history[signal_type]['total']}")

class MLEnsembleAggregator(BaseSignalAggregator):
    """
    Machine Learning ensemble signal aggregator.
    
    Uses ML models to learn optimal signal combination patterns
    from historical data and market outcomes.
    """
    
    def __init__(self, model_type: str = 'voting'):
        super().__init__("ML Ensemble", AggregationMethod.MACHINE_LEARNING)
        
        self.model_type = model_type
        self.is_trained = False
        self.scaler = StandardScaler()
        
        # Initialize ensemble model
        if model_type == 'voting':
            self.model = VotingClassifier([
                ('lr', LogisticRegression(random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
            ])
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        logger.info(f"Initialized ML ensemble aggregator with {model_type} model")
    
    def train(self, historical_signals: List[List[TechnicalSignal]], 
             outcomes: List[int]) -> Dict[str, float]:
        """Train the ML ensemble on historical signal data"""
        
        # Extract features from historical signals
        features = []
        for signal_group in historical_signals:
            feature_vector = self._extract_features(signal_group)
            features.append(feature_vector)
        
        features_array = np.array(features)
        outcomes_array = np.array(outcomes)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_array)
        
        # Train model
        self.model.fit(features_scaled, outcomes_array)
        self.is_trained = True
        
        # Calculate training metrics
        train_score = self.model.score(features_scaled, outcomes_array)
        
        logger.info(f"ML ensemble trained with accuracy: {train_score:.3f}")
        
        return {
            'training_accuracy': train_score,
            'training_samples': len(features),
            'feature_count': features_array.shape[1]
        }
    
    def aggregate(self, signals: List[TechnicalSignal], 
                 weights: Optional[Dict[str, float]] = None) -> CompositeSignal:
        """Aggregate signals using trained ML ensemble"""
        
        if not self.is_trained:
            logger.warning("ML ensemble not trained, falling back to weighted consensus")
            return self._fallback_aggregation(signals)
        
        # Extract features from current signals
        features = self._extract_features(signals)
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        prediction_proba = None
        
        if hasattr(self.model, 'predict_proba'):
            prediction_proba = self.model.predict_proba(features_scaled)[0]
        
        # Convert ML prediction to signal direction
        if prediction == 1:  # Buy signal
            final_direction = SignalDirection.BUY
            final_strength = prediction_proba[1] if prediction_proba is not None else 0.7
        elif prediction == -1:  # Sell signal
            final_direction = SignalDirection.SELL
            final_strength = prediction_proba[0] if prediction_proba is not None else 0.7
        else:  # Hold signal
            final_direction = SignalDirection.HOLD
            final_strength = prediction_proba[1] if prediction_proba is not None else 0.5
        
        # Determine confidence based on prediction probability
        if final_strength >= 0.8:
            final_confidence = SignalConfidence.VERY_HIGH
        elif final_strength >= 0.6:
            final_confidence = SignalConfidence.HIGH
        elif final_strength >= 0.4:
            final_confidence = SignalConfidence.MEDIUM
        else:
            final_confidence = SignalConfidence.LOW
        
        # Create composite signal
        composite = CompositeSignal(
            timestamp=signals[0].timestamp,
            symbol=signals[0].symbol,
            indicator="ML_Ensemble_Composite",
            direction=final_direction,
            strength=final_strength,
            confidence=final_confidence,
            price=signals[0].price,
            indicator_value=final_strength,
            
            component_signals=signals,
            aggregation_method=self.method,
            consensus_score=final_strength,
            
            metadata={
                'ml_prediction': prediction,
                'prediction_probabilities': prediction_proba.tolist() if prediction_proba is not None else [],
                'features_used': len(features),
                'model_type': self.model_type
            }
        )
        
        return composite
    
    def _extract_features(self, signals: List[TechnicalSignal]) -> List[float]:
        """Extract feature vector from signals for ML model"""
        
        if not signals:
            return [0.0] * 20  # Return zero vector
        
        features = []
        
        # Signal count features
        total_signals = len(signals)
        technical_count = sum(1 for s in signals if not isinstance(s, (ClassificationSignal, RegressionSignal)))
        ml_count = total_signals - technical_count
        
        features.extend([
            total_signals,
            technical_count / total_signals if total_signals > 0 else 0,
            ml_count / total_signals if total_signals > 0 else 0
        ])
        
        # Direction distribution features
        directions = [s.direction for s in signals]
        buy_count = sum(1 for d in directions if d in [SignalDirection.BUY, SignalDirection.STRONG_BUY])
        sell_count = sum(1 for d in directions if d in [SignalDirection.SELL, SignalDirection.STRONG_SELL])
        hold_count = sum(1 for d in directions if d == SignalDirection.HOLD)
        
        features.extend([
            buy_count / total_signals if total_signals > 0 else 0,
            sell_count / total_signals if total_signals > 0 else 0,
            hold_count / total_signals if total_signals > 0 else 0
        ])
        
        # Strength and confidence statistics
        strengths = [s.strength for s in signals]
        confidences = [s.confidence.value for s in signals]
        
        features.extend([
            np.mean(strengths),
            np.std(strengths),
            np.max(strengths),
            np.min(strengths),
            np.mean(confidences),
            np.std(confidences),
            np.max(confidences),
            np.min(confidences)
        ])
        
        # Agreement metrics
        max_agreement = max(buy_count, sell_count, hold_count)
        agreement_ratio = max_agreement / total_signals if total_signals > 0 else 0
        
        # Signal diversity (entropy)
        if total_signals > 0:
            probs = [buy_count/total_signals, sell_count/total_signals, hold_count/total_signals]
            probs = [p for p in probs if p > 0]
            diversity = -sum(p * np.log2(p) for p in probs) if probs else 0
        else:
            diversity = 0
        
        features.extend([
            agreement_ratio,
            diversity
        ])
        
        # Pad or truncate to fixed size
        target_size = 20
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]
        
        return features
    
    def _fallback_aggregation(self, signals: List[TechnicalSignal]) -> CompositeSignal:
        """Fallback to simple aggregation when ML model is not trained"""
        
        fallback_aggregator = WeightedAverageAggregator()
        composite = fallback_aggregator.aggregate(signals)
        composite.indicator = "ML_Ensemble_Fallback"
        composite.aggregation_method = self.method
        
        return composite

# ============================================
# Composite Signal Generator
# ============================================

class CompositeSignalGenerator:
    """
    Comprehensive composite signal generator.
    
    Combines signals from multiple sources using various aggregation
    methods to create enhanced trading signals.
    """
    
    def __init__(self):
        self.aggregators = {}
        self.signals_generated = 0
        
        # Initialize default aggregators
        self.aggregators['weighted_average'] = WeightedAverageAggregator()
        self.aggregators['bayesian_ensemble'] = BayesianEnsembleAggregator()
        self.aggregators['ml_ensemble'] = MLEnsembleAggregator()
        
        # Signal filtering parameters
        self.min_component_signals = 2
        self.max_component_signals = 10
        self.min_consensus_threshold = 0.5
        
        logger.info("Initialized CompositeSignalGenerator with 3 default aggregators")
    
    def add_aggregator(self, name: str, aggregator: BaseSignalAggregator):
        """Add a custom aggregator"""
        self.aggregators[name] = aggregator
        logger.info(f"Added custom aggregator: {name}")
    
    @time_it("composite_signal_generation")
    def generate_composite_signals(self, 
                                 signal_groups: Dict[str, List[TechnicalSignal]],
                                 aggregation_methods: Optional[List[str]] = None,
                                 symbol_filter: Optional[List[str]] = None) -> Dict[str, List[CompositeSignal]]:
        """
        Generate composite signals from multiple signal groups
        
        Args:
            signal_groups: Dictionary of signal_type -> signals
            aggregation_methods: List of aggregation methods to use
            symbol_filter: Optional list of symbols to process
            
        Returns:
            Dictionary of aggregation_method -> composite_signals
        """
        
        if aggregation_methods is None:
            aggregation_methods = list(self.aggregators.keys())
        
        # Group signals by symbol and timestamp
        symbol_timestamp_groups = self._group_signals_by_symbol_timestamp(signal_groups, symbol_filter)
        
        composite_signals = {method: [] for method in aggregation_methods}
        
        for group_key, signals in symbol_timestamp_groups.items():
            symbol, timestamp = group_key
            
            # Filter signals by quality and quantity
            if not self._validate_signal_group(signals):
                continue
            
            # Generate composite signals using different methods
            for method in aggregation_methods:
                if method not in self.aggregators:
                    logger.warning(f"Unknown aggregation method: {method}")
                    continue
                
                try:
                    aggregator = self.aggregators[method]
                    composite_signal = aggregator.aggregate(signals)
                    composite_signals[method].append(composite_signal)
                    
                except Exception as e:
                    logger.error(f"Error generating composite signal with {method}: {e}")
        
        # Update signals generated count
        self.signals_generated = sum(len(signals) for signals in composite_signals.values())
        
        logger.info(f"Generated {self.signals_generated} composite signals across {len(aggregation_methods)} methods")
        
        return composite_signals
    
    def _group_signals_by_symbol_timestamp(self, 
                                         signal_groups: Dict[str, List[TechnicalSignal]],
                                         symbol_filter: Optional[List[str]] = None) -> Dict[Tuple[str, pd.Timestamp], List[TechnicalSignal]]:
        """Group signals by symbol and timestamp for aggregation"""
        
        grouped_signals = {}
        
        # Collect all signals
        all_signals = []
        for signal_type, signals in signal_groups.items():
            all_signals.extend(signals)
        
        # Group by symbol and timestamp
        for signal in all_signals:
            # Apply symbol filter if specified
            if symbol_filter and signal.symbol not in symbol_filter:
                continue
            
            group_key = (signal.symbol, signal.timestamp)
            
            if group_key not in grouped_signals:
                grouped_signals[group_key] = []
            
            grouped_signals[group_key].append(signal)
        
        return grouped_signals
    
    def _validate_signal_group(self, signals: List[TechnicalSignal]) -> bool:
        """Validate signal group for composite generation"""
        
        # Check signal count
        if len(signals) < self.min_component_signals:
            return False
        
        if len(signals) > self.max_component_signals:
            # Keep only the strongest signals
            signals.sort(key=lambda s: s.strength * s.confidence.value, reverse=True)
            signals = signals[:self.max_component_signals]
        
        # Check for minimum consensus
        if len(signals) > 1:
            directions = [s.direction for s in signals]
            buy_count = sum(1 for d in directions if d in [SignalDirection.BUY, SignalDirection.STRONG_BUY])
            sell_count = sum(1 for d in directions if d in [SignalDirection.SELL, SignalDirection.STRONG_SELL])
            hold_count = sum(1 for d in directions if d == SignalDirection.HOLD)
            
            max_agreement = max(buy_count, sell_count, hold_count)
            consensus_ratio = max_agreement / len(signals)
            
            if consensus_ratio < self.min_consensus_threshold:
                return False
        
        return True
    
    def analyze_composite_quality(self, composite_signals: List[CompositeSignal]) -> pd.DataFrame:
        """Analyze the quality of generated composite signals"""
        
        if not composite_signals:
            return pd.DataFrame()
        
        analysis_data = []
        
        for signal in composite_signals:
            analysis_data.append({
                'Symbol': signal.symbol,
                'Timestamp': signal.timestamp,
                'Direction': signal.direction.name,
                'Strength': signal.strength,
                'Confidence': signal.confidence.name,
                'Consensus_Score': signal.consensus_score,
                'Component_Count': signal.component_count,
                'Agreement_Ratio': signal.agreement_ratio,
                'Signal_Diversity': signal.signal_diversity,
                'Technical_Weight': signal.technical_weight,
                'ML_Weight': signal.ml_weight,
                'Buy_Signals': signal.buy_signals,
                'Sell_Signals': signal.sell_signals,
                'Hold_Signals': signal.hold_signals,
                'Aggregation_Method': signal.aggregation_method.value
            })
        
        return pd.DataFrame(analysis_data)
    
    def get_aggregator_summary(self) -> pd.DataFrame:
        """Get summary of all aggregators and their usage"""
        
        summary_data = []
        
        for name, aggregator in self.aggregators.items():
            summary_data.append({
                'Aggregator': name,
                'Method': aggregator.method.value,
                'Signals_Processed': aggregator.signals_processed,
                'Type': type(aggregator).__name__
            })
        
        return pd.DataFrame(summary_data)

# ============================================
# Utility Functions
# ============================================

def create_composite_signals(signal_groups: Dict[str, List[TechnicalSignal]], 
                            aggregation_methods: Optional[List[str]] = None) -> Dict[str, List[CompositeSignal]]:
    """
    Quick utility function to generate composite signals
    
    Args:
        signal_groups: Dictionary of signal_type -> signals
        aggregation_methods: List of aggregation methods to use
        
    Returns:
        Dictionary of aggregation_method -> composite_signals
    """
    
    generator = CompositeSignalGenerator()
    return generator.generate_composite_signals(signal_groups, aggregation_methods)

def filter_composite_signals_by_consensus(signals: List[CompositeSignal], 
                                        min_consensus: float = 0.7,
                                        min_component_count: int = 3) -> List[CompositeSignal]:
    """Filter composite signals by consensus quality"""
    
    return [
        signal for signal in signals 
        if (signal.consensus_score >= min_consensus and 
            signal.component_count >= min_component_count)
    ]

def rank_composite_signals(signals: List[CompositeSignal]) -> List[CompositeSignal]:
    """Rank composite signals by overall quality score"""
    
    def quality_score(signal: CompositeSignal) -> float:
        # Weighted quality score combining multiple factors
        score = (
            signal.strength * 0.3 +
            signal.confidence.value * 0.25 +
            signal.consensus_score * 0.25 +
            signal.agreement_ratio * 0.2
        )
        
        # Bonus for signal diversity and component count
        diversity_bonus = min(signal.signal_diversity / 2, 0.1)  # Max 10% bonus
        component_bonus = min((signal.component_count - 2) / 10, 0.1)  # Max 10% bonus
        
        return score + diversity_bonus + component_bonus
    
    return sorted(signals, key=quality_score, reverse=True)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Composite Signals System")
    
    # Generate sample signals for testing
    from datetime import datetime
    import pandas as pd
    
    sample_timestamp = pd.Timestamp('2023-06-15 09:30:00')
    sample_price = 150.0
    
    # Create mock technical signals
    technical_signals = [
        TechnicalSignal(
            timestamp=sample_timestamp,
            symbol='AAPL',
            indicator='SMA_Cross',
            direction=SignalDirection.BUY,
            strength=0.75,
            confidence=SignalConfidence.HIGH,
            price=sample_price
        ),
        TechnicalSignal(
            timestamp=sample_timestamp,
            symbol='AAPL',
            indicator='RSI',
            direction=SignalDirection.BUY,
            strength=0.68,
            confidence=SignalConfidence.MEDIUM,
            price=sample_price
        ),
        TechnicalSignal(
            timestamp=sample_timestamp,
            symbol='AAPL',
            indicator='MACD',
            direction=SignalDirection.SELL,
            strength=0.55,
            confidence=SignalConfidence.LOW,
            price=sample_price
        )
    ]
    
    # Create mock ML signals
    classification_signals = [
        ClassificationSignal(
            timestamp=sample_timestamp,
            symbol='AAPL',
            indicator='Direction_Classifier',
            direction=SignalDirection.BUY,
            strength=0.82,
            confidence=SignalConfidence.VERY_HIGH,
            price=sample_price,
            model_name='Random Forest',
            prediction_probability=0.82
        )
    ]
    
    regression_signals = [
        RegressionSignal(
            timestamp=sample_timestamp,
            symbol='AAPL',
            indicator='Return_Predictor',
            direction=SignalDirection.BUY,
            strength=0.71,
            confidence=SignalConfidence.HIGH,
            price=sample_price,
            model_name='XGBoost',
            predicted_value=0.025,
            move_probability=0.71
        )
    ]
    
    # Test individual aggregators
    print("\n1. Testing Individual Aggregators")
    
    all_signals = technical_signals + classification_signals + regression_signals
    
    # Test Weighted Average Aggregator
    weighted_aggregator = WeightedAverageAggregator()
    weighted_composite = weighted_aggregator.aggregate(all_signals)
    
    print(f"Weighted Average Composite:")
    print(f"  Direction: {weighted_composite.direction.name}")
    print(f"  Strength: {weighted_composite.strength:.3f}")
    print(f"  Confidence: {weighted_composite.confidence.name}")
    print(f"  Consensus Score: {weighted_composite.consensus_score:.3f}")
    print(f"  Component Count: {weighted_composite.component_count}")
    print(f"  Agreement Ratio: {weighted_composite.agreement_ratio:.3f}")
    
    # Test Bayesian Ensemble Aggregator
    bayesian_aggregator = BayesianEnsembleAggregator()
    bayesian_composite = bayesian_aggregator.aggregate(all_signals)
    
    print(f"\nBayesian Ensemble Composite:")
    print(f"  Direction: {bayesian_composite.direction.name}")
    print(f"  Strength: {bayesian_composite.strength:.3f}")
    print(f"  Confidence: {bayesian_composite.confidence.name}")
    print(f"  Consensus Score: {bayesian_composite.consensus_score:.3f}")
    
    # Show Bayesian probabilities
    if 'probabilities' in bayesian_composite.metadata:
        probs = bayesian_composite.metadata['probabilities']
        print(f"  Probabilities: Buy={probs['buy']:.3f}, Sell={probs['sell']:.3f}, Hold={probs['hold']:.3f}")
    
    # Test ML Ensemble Aggregator (without training)
    ml_aggregator = MLEnsembleAggregator()
    ml_composite = ml_aggregator.aggregate(all_signals)  # Will use fallback
    
    print(f"\nML Ensemble Composite (fallback):")
    print(f"  Direction: {ml_composite.direction.name}")
    print(f"  Strength: {ml_composite.strength:.3f}")
    print(f"  Confidence: {ml_composite.confidence.name}")
    
    print("\n2. Testing Composite Signal Generator")
    
    # Prepare signal groups
    signal_groups = {
        'technical': technical_signals,
        'classification': classification_signals,
        'regression': regression_signals
    }
    
    # Generate composite signals
    generator = CompositeSignalGenerator()
    composite_signals = generator.generate_composite_signals(signal_groups)
    
    print("Generated Composite Signals:")
    for method, signals in composite_signals.items():
        print(f"  {method}: {len(signals)} signals")
        
        if signals:
            signal = signals[0]
            print(f"    Direction: {signal.direction.name}")
            print(f"    Strength: {signal.strength:.3f}")
            print(f"    Consensus: {signal.consensus_score:.3f}")
            print(f"    Components: {signal.component_count}")
    
    # Test signal analysis
    print("\n3. Composite Signal Quality Analysis")
    
    all_composite_signals = []
    for signals in composite_signals.values():
        all_composite_signals.extend(signals)
    
    if all_composite_signals:
        analysis_df = generator.analyze_composite_quality(all_composite_signals)
        print("Composite Signal Analysis:")
        print(analysis_df.to_string(index=False))
    
    # Test signal filtering and ranking
    print("\n4. Testing Signal Filtering and Ranking")
    
    # Filter by consensus
    high_consensus_signals = filter_composite_signals_by_consensus(
        all_composite_signals,
        min_consensus=0.6,
        min_component_count=3
    )
    
    print(f"High consensus signals: {len(high_consensus_signals)}")
    
    # Rank signals by quality
    ranked_signals = rank_composite_signals(all_composite_signals)
    
    print("Top 3 Ranked Composite Signals:")
    for i, signal in enumerate(ranked_signals[:3]):
        print(f"  #{i+1}: {signal.aggregation_method.value}")
        print(f"      Direction: {signal.direction.name}")
        print(f"      Strength: {signal.strength:.3f}")
        print(f"      Consensus: {signal.consensus_score:.3f}")
        print(f"      Agreement: {signal.agreement_ratio:.3f}")
    
    # Test aggregator summary
    print("\n5. Aggregator Summary")
    
    summary_df = generator.get_aggregator_summary()
    print(summary_df.to_string(index=False))
    
    # Test utility functions
    print("\n6. Testing Utility Functions")
    
    quick_composite = create_composite_signals(
        signal_groups,
        aggregation_methods=['weighted_average', 'bayesian_ensemble']
    )
    
    total_quick_signals = sum(len(signals) for signals in quick_composite.values())
    print(f"Quick composite generation: {total_quick_signals} signals")
    
    # Test with multiple symbols and timestamps
    print("\n7. Testing Multi-Symbol Composite Generation")
    
    # Create signals for multiple symbols
    multi_symbol_signals = {
        'technical': [],
        'classification': [],
        'regression': []
    }
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    timestamps = [
        pd.Timestamp('2023-06-15 09:30:00'),
        pd.Timestamp('2023-06-15 10:00:00'),
        pd.Timestamp('2023-06-15 10:30:00')
    ]
    
    for symbol in symbols:
        for timestamp in timestamps:
            # Technical signal
            multi_symbol_signals['technical'].append(
                TechnicalSignal(
                    timestamp=timestamp,
                    symbol=symbol,
                    indicator='RSI',
                    direction=np.random.choice([SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]),
                    strength=np.random.uniform(0.5, 0.9),
                    confidence=np.random.choice(list(SignalConfidence)),
                    price=np.random.uniform(100, 200)
                )
            )
            
            # Classification signal
            multi_symbol_signals['classification'].append(
                ClassificationSignal(
                    timestamp=timestamp,
                    symbol=symbol,
                    indicator='Direction_Classifier',
                    direction=np.random.choice([SignalDirection.BUY, SignalDirection.SELL]),
                    strength=np.random.uniform(0.6, 0.95),
                    confidence=np.random.choice([SignalConfidence.MEDIUM, SignalConfidence.HIGH, SignalConfidence.VERY_HIGH]),
                    price=np.random.uniform(100, 200),
                    model_name='Random Forest',
                    prediction_probability=np.random.uniform(0.6, 0.95)
                )
            )
    
    # Generate multi-symbol composite signals
    multi_composite = generator.generate_composite_signals(multi_symbol_signals)
    
    print("Multi-Symbol Composite Results:")
    for method, signals in multi_composite.items():
        if signals:
            symbol_counts = {}
            for signal in signals:
                symbol_counts[signal.symbol] = symbol_counts.get(signal.symbol, 0) + 1
            
            print(f"  {method}: {len(signals)} total signals")
            print(f"    Symbol distribution: {symbol_counts}")
    
    print("\nComposite signals system testing completed successfully!")
    print("\nGenerated composite signals include:")
    print("• Weighted Average: Combines signals using customizable type-based weights")
    print("• Bayesian Ensemble: Uses Bayesian inference with historical performance learning")
    print("• ML Ensemble: Learns optimal signal combination patterns from data")
    print("• Signal Quality Analysis: Consensus scoring, diversity metrics, agreement ratios")
    print("• Conflict Resolution: Multiple methods for handling contradictory signals")
    print("• Multi-Symbol Support: Handles signals across multiple assets and timeframes")
    print("• Extensible Framework: Easy to add custom aggregation methods")
