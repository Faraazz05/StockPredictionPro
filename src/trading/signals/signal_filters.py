# ============================================
# StockPredictionPro - src/trading/signals/signal_filters.py
# Comprehensive signal filtering and validation system for enhanced trading decisions
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from sklearn.preprocessing import StandardScaler
import networkx as nx

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it
from .technical_signals import TechnicalSignal, SignalDirection, SignalConfidence
from .classification_signals import ClassificationSignal
from .regression_signals import RegressionSignal
from .composite_signals import CompositeSignal

logger = get_logger('trading.signals.signal_filters')

# ============================================
# Filter Data Structures and Enums
# ============================================

class FilterType(Enum):
    """Types of signal filters"""
    QUALITY_FILTER = "quality_filter"
    TIME_FILTER = "time_filter"
    MARKET_FILTER = "market_filter"
    VOLATILITY_FILTER = "volatility_filter"
    VOLUME_FILTER = "volume_filter"
    MOMENTUM_FILTER = "momentum_filter"
    CORRELATION_FILTER = "correlation_filter"
    REGIME_FILTER = "regime_filter"
    RISK_FILTER = "risk_filter"
    CONSENSUS_FILTER = "consensus_filter"

class FilterAction(Enum):
    """Actions filters can take"""
    ACCEPT = "accept"      # Signal passes filter
    REJECT = "reject"      # Signal is filtered out
    MODIFY = "modify"      # Signal is modified
    FLAG = "flag"          # Signal is flagged but not rejected

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    CONSOLIDATING = "consolidating"

@dataclass
class FilterResult:
    """Result of applying a filter to a signal"""
    signal: TechnicalSignal
    action: FilterAction
    filter_name: str
    filter_type: FilterType
    
    # Filter details
    original_strength: float = 0.0
    modified_strength: Optional[float] = None
    original_confidence: SignalConfidence = SignalConfidence.LOW
    modified_confidence: Optional[SignalConfidence] = None
    
    # Filter reasoning
    filter_score: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Store original values"""
        self.original_strength = self.signal.strength
        self.original_confidence = self.signal.confidence

@dataclass
class FilterConfiguration:
    """Configuration for signal filters"""
    enabled: bool = True
    strictness: float = 0.5  # 0.0 = very lenient, 1.0 = very strict
    
    # Quality thresholds
    min_strength: float = 0.3
    min_confidence: SignalConfidence = SignalConfidence.LOW
    max_age_hours: int = 24
    
    # Market conditions
    allowed_regimes: List[MarketRegime] = field(default_factory=lambda: list(MarketRegime))
    min_volume_ratio: float = 0.5  # Minimum volume vs average
    max_volatility: float = 0.5   # Maximum acceptable volatility
    
    # Risk parameters
    max_drawdown_period: int = 5  # Days
    correlation_threshold: float = 0.8
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

# ============================================
# Base Filter Class
# ============================================

class BaseSignalFilter:
    """
    Base class for all signal filters.
    
    This class provides common functionality for filtering and
    validating trading signals based on various criteria.
    """
    
    def __init__(self, name: str, filter_type: FilterType, config: Optional[FilterConfiguration] = None):
        self.name = name
        self.filter_type = filter_type
        self.config = config or FilterConfiguration()
        self.signals_processed = 0
        self.signals_rejected = 0
        self.signals_modified = 0
        
        logger.debug(f"Initialized {name} filter ({filter_type.value})")
    
    def apply_filter(self, signal: TechnicalSignal, 
                    market_data: Optional[pd.DataFrame] = None,
                    context: Optional[Dict[str, Any]] = None) -> FilterResult:
        """Apply filter to a signal - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement apply_filter method")
    
    def _create_result(self, signal: TechnicalSignal, action: FilterAction, 
                      filter_score: float = 0.0, reasoning: str = "",
                      **kwargs) -> FilterResult:
        """Create a filter result"""
        
        result = FilterResult(
            signal=signal,
            action=action,
            filter_name=self.name,
            filter_type=self.filter_type,
            filter_score=filter_score,
            reasoning=reasoning,
            metadata=kwargs
        )
        
        # Update processing statistics
        self.signals_processed += 1
        if action == FilterAction.REJECT:
            self.signals_rejected += 1
        elif action == FilterAction.MODIFY:
            self.signals_modified += 1
        
        return result
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get filter performance statistics"""
        
        acceptance_rate = (self.signals_processed - self.signals_rejected) / self.signals_processed if self.signals_processed > 0 else 0
        modification_rate = self.signals_modified / self.signals_processed if self.signals_processed > 0 else 0
        
        return {
            'filter_name': self.name,
            'filter_type': self.filter_type.value,
            'signals_processed': self.signals_processed,
            'signals_accepted': self.signals_processed - self.signals_rejected,
            'signals_rejected': self.signals_rejected,
            'signals_modified': self.signals_modified,
            'acceptance_rate': acceptance_rate,
            'rejection_rate': self.signals_rejected / self.signals_processed if self.signals_processed > 0 else 0,
            'modification_rate': modification_rate
        }

# ============================================
# Quality Filters
# ============================================

class QualityFilter(BaseSignalFilter):
    """
    Quality-based signal filter.
    
    Filters signals based on strength, confidence, and overall quality metrics.
    Rejects weak or low-confidence signals to improve signal quality.
    """
    
    def __init__(self, config: Optional[FilterConfiguration] = None):
        super().__init__("Quality Filter", FilterType.QUALITY_FILTER, config)
        
        # Quality thresholds
        self.min_strength = self.config.min_strength
        self.min_confidence = self.config.min_confidence
        self.strictness = self.config.strictness
    
    @time_it("quality_filter_apply")
    def apply_filter(self, signal: TechnicalSignal, 
                    market_data: Optional[pd.DataFrame] = None,
                    context: Optional[Dict[str, Any]] = None) -> FilterResult:
        """Apply quality-based filtering"""
        
        quality_score = self._calculate_quality_score(signal)
        
        # Determine quality threshold based on strictness
        quality_threshold = 0.3 + (self.strictness * 0.4)  # Range: 0.3 to 0.7
        
        # Check basic quality criteria
        strength_pass = signal.strength >= self.min_strength
        confidence_pass = signal.confidence.value >= self.min_confidence.value
        quality_pass = quality_score >= quality_threshold
        
        # Additional quality checks for ML signals
        if isinstance(signal, ClassificationSignal):
            ml_quality_pass = signal.model_confidence >= 0.5 and signal.prediction_probability >= 0.6
        elif isinstance(signal, RegressionSignal):
            ml_quality_pass = signal.model_r2_score >= 0.1 and signal.move_probability >= 0.55
        else:
            ml_quality_pass = True
        
        # Determine action
        if strength_pass and confidence_pass and quality_pass and ml_quality_pass:
            action = FilterAction.ACCEPT
            reasoning = f"High quality signal (score: {quality_score:.3f})"
        elif quality_score >= quality_threshold * 0.8:  # Marginal quality - modify
            action = FilterAction.MODIFY
            reasoning = f"Marginal quality - reducing strength (score: {quality_score:.3f})"
            
            # Reduce signal strength for marginal quality
            signal.strength *= 0.8
            signal.confidence = SignalConfidence.LOW if signal.confidence == SignalConfidence.MEDIUM else signal.confidence
        else:
            action = FilterAction.REJECT
            reasoning = f"Low quality signal rejected (score: {quality_score:.3f})"
        
        return self._create_result(
            signal=signal,
            action=action,
            filter_score=quality_score,
            reasoning=reasoning,
            quality_components={
                'strength_pass': strength_pass,
                'confidence_pass': confidence_pass,
                'quality_pass': quality_pass,
                'ml_quality_pass': ml_quality_pass
            }
        )
    
    def _calculate_quality_score(self, signal: TechnicalSignal) -> float:
        """Calculate overall quality score for signal"""
        
        # Base score from strength and confidence
        base_score = (signal.strength * 0.6) + (signal.confidence.value * 0.4)
        
        # Bonus for ML signals with good performance
        ml_bonus = 0.0
        if isinstance(signal, ClassificationSignal):
            if signal.model_confidence > 0.7 and signal.prediction_probability > 0.8:
                ml_bonus = 0.1
        elif isinstance(signal, RegressionSignal):
            if signal.model_r2_score > 0.3 and signal.move_probability > 0.7:
                ml_bonus = 0.1
        
        # Bonus for composite signals with high consensus
        consensus_bonus = 0.0
        if isinstance(signal, CompositeSignal):
            if signal.consensus_score > 0.8 and signal.agreement_ratio > 0.7:
                consensus_bonus = 0.15
        
        # Penalty for very recent signals (less reliable)
        recency_penalty = 0.0
        if hasattr(signal, 'timestamp'):
            age_hours = (datetime.now() - signal.timestamp).total_seconds() / 3600
            if age_hours < 0.5:  # Less than 30 minutes old
                recency_penalty = 0.05
        
        final_score = base_score + ml_bonus + consensus_bonus - recency_penalty
        return max(0.0, min(1.0, final_score))

class TimeFilter(BaseSignalFilter):
    """
    Time-based signal filter.
    
    Filters signals based on timing criteria such as market hours,
    signal age, and temporal clustering.
    """
    
    def __init__(self, config: Optional[FilterConfiguration] = None):
        super().__init__("Time Filter", FilterType.TIME_FILTER, config)
        
        self.max_age_hours = self.config.max_age_hours
        
        # Market hours (can be customized)
        self.market_open = 9.5   # 9:30 AM
        self.market_close = 16.0 # 4:00 PM
        self.allow_premarket = True
        self.allow_afterhours = False
    
    def apply_filter(self, signal: TechnicalSignal, 
                    market_data: Optional[pd.DataFrame] = None,
                    context: Optional[Dict[str, Any]] = None) -> FilterResult:
        """Apply time-based filtering"""
        
        current_time = datetime.now()
        signal_time = signal.timestamp if hasattr(signal, 'timestamp') else current_time
        
        # Calculate signal age
        age_hours = (current_time - signal_time).total_seconds() / 3600
        
        # Check if signal is too old
        if age_hours > self.max_age_hours:
            return self._create_result(
                signal=signal,
                action=FilterAction.REJECT,
                reasoning=f"Signal too old ({age_hours:.1f} hours)",
                age_hours=age_hours
            )
        
        # Check market hours
        signal_hour = signal_time.hour + signal_time.minute / 60.0
        is_market_hours = self.market_open <= signal_hour <= self.market_close
        
        if not is_market_hours and not self.allow_premarket and not self.allow_afterhours:
            return self._create_result(
                signal=signal,
                action=FilterAction.REJECT,
                reasoning=f"Signal outside market hours ({signal_hour:.1f})",
                signal_hour=signal_hour
            )
        
        # Adjust signal strength based on timing
        time_adjustment = 1.0
        
        # Reduce strength for very fresh signals (less reliable)
        if age_hours < 0.25:  # Less than 15 minutes
            time_adjustment *= 0.8
        
        # Reduce strength for after-hours signals
        if not is_market_hours:
            time_adjustment *= 0.7
        
        # Apply time adjustment if significant
        action = FilterAction.ACCEPT
        if time_adjustment < 0.95:
            action = FilterAction.MODIFY
            signal.strength *= time_adjustment
            reasoning = f"Strength adjusted for timing (factor: {time_adjustment:.2f})"
        else:
            reasoning = "Signal timing acceptable"
        
        return self._create_result(
            signal=signal,
            action=action,
            filter_score=time_adjustment,
            reasoning=reasoning,
            age_hours=age_hours,
            market_hours=is_market_hours
        )

# ============================================
# Market Condition Filters
# ============================================

class MarketRegimeFilter(BaseSignalFilter):
    """
    Market regime-based signal filter.
    
    Filters signals based on current market conditions and regimes.
    Different signal types may be more or less effective in different regimes.
    """
    
    def __init__(self, config: Optional[FilterConfiguration] = None):
        super().__init__("Market Regime Filter", FilterType.REGIME_FILTER, config)
        
        # Regime detection parameters
        self.trend_window = 20
        self.volatility_window = 20
        self.regime_cache = {}  # Cache regime calculations
        self.cache_expiry = timedelta(hours=1)
    
    def apply_filter(self, signal: TechnicalSignal, 
                    market_data: Optional[pd.DataFrame] = None,
                    context: Optional[Dict[str, Any]] = None) -> FilterResult:
        """Apply market regime filtering"""
        
        if market_data is None or len(market_data) < self.trend_window:
            return self._create_result(
                signal=signal,
                action=FilterAction.ACCEPT,
                reasoning="Insufficient market data for regime analysis"
            )
        
        # Detect current market regime
        current_regime = self._detect_market_regime(market_data)
        
        # Get signal compatibility with current regime
        compatibility_score = self._calculate_regime_compatibility(signal, current_regime)
        
        # Determine filter action based on compatibility
        if compatibility_score >= 0.7:
            action = FilterAction.ACCEPT
            reasoning = f"Signal compatible with {current_regime.value} regime"
        elif compatibility_score >= 0.5:
            action = FilterAction.MODIFY
            # Adjust signal strength based on regime compatibility
            signal.strength *= compatibility_score
            reasoning = f"Signal strength adjusted for {current_regime.value} regime"
        else:
            action = FilterAction.REJECT
            reasoning = f"Signal incompatible with {current_regime.value} regime"
        
        return self._create_result(
            signal=signal,
            action=action,
            filter_score=compatibility_score,
            reasoning=reasoning,
            market_regime=current_regime.value,
            regime_compatibility=compatibility_score
        )
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime from price data"""
        
        # Use cache if recent
        cache_key = f"regime_{len(market_data)}"
        if cache_key in self.regime_cache:
            cache_time, regime = self.regime_cache[cache_key]
            if datetime.now() - cache_time < self.cache_expiry:
                return regime
        
        # Calculate regime indicators
        returns = market_data['close'].pct_change().dropna()
        
        # Trend detection
        price_change = (market_data['close'].iloc[-1] - market_data['close'].iloc[-self.trend_window]) / market_data['close'].iloc[-self.trend_window]
        trend_strength = abs(price_change)
        
        # Volatility calculation
        volatility = returns.tail(self.volatility_window).std() * np.sqrt(252)
        vol_percentile = self._calculate_volatility_percentile(returns, volatility)
        
        # Regime classification logic
        if trend_strength > 0.05:  # Strong trend
            if price_change > 0:
                regime = MarketRegime.BULL_MARKET
            else:
                regime = MarketRegime.BEAR_MARKET
        elif vol_percentile > 0.8:
            regime = MarketRegime.HIGH_VOLATILITY
        elif vol_percentile < 0.2:
            regime = MarketRegime.LOW_VOLATILITY
        elif trend_strength < 0.02:
            regime = MarketRegime.CONSOLIDATING
        else:
            regime = MarketRegime.SIDEWAYS_MARKET
        
        # Cache result
        self.regime_cache[cache_key] = (datetime.now(), regime)
        
        return regime
    
    def _calculate_volatility_percentile(self, returns: pd.Series, current_vol: float) -> float:
        """Calculate volatility percentile over historical data"""
        
        if len(returns) < 60:  # Need enough history
            return 0.5
        
        # Calculate rolling volatilities
        rolling_vols = returns.rolling(window=20).std() * np.sqrt(252)
        rolling_vols = rolling_vols.dropna()
        
        if len(rolling_vols) == 0:
            return 0.5
        
        # Calculate percentile
        percentile = (rolling_vols < current_vol).mean()
        return percentile
    
    def _calculate_regime_compatibility(self, signal: TechnicalSignal, regime: MarketRegime) -> float:
        """Calculate how compatible a signal is with the current market regime"""
        
        # Base compatibility matrix
        compatibility_rules = {
            MarketRegime.BULL_MARKET: {
                'buy_signals': 1.0,
                'sell_signals': 0.3,
                'momentum_indicators': 0.9,
                'mean_reversion': 0.4,
                'ml_signals': 0.8
            },
            MarketRegime.BEAR_MARKET: {
                'buy_signals': 0.3,
                'sell_signals': 1.0,
                'momentum_indicators': 0.9,
                'mean_reversion': 0.4,
                'ml_signals': 0.8
            },
            MarketRegime.SIDEWAYS_MARKET: {
                'buy_signals': 0.6,
                'sell_signals': 0.6,
                'momentum_indicators': 0.4,
                'mean_reversion': 0.9,
                'ml_signals': 0.7
            },
            MarketRegime.HIGH_VOLATILITY: {
                'buy_signals': 0.5,
                'sell_signals': 0.5,
                'momentum_indicators': 0.7,
                'mean_reversion': 0.8,
                'ml_signals': 0.6
            },
            MarketRegime.LOW_VOLATILITY: {
                'buy_signals': 0.8,
                'sell_signals': 0.8,
                'momentum_indicators': 0.8,
                'mean_reversion': 0.5,
                'ml_signals': 0.9
            },
            MarketRegime.CONSOLIDATING: {
                'buy_signals': 0.5,
                'sell_signals': 0.5,
                'momentum_indicators': 0.3,
                'mean_reversion': 0.9,
                'ml_signals': 0.7
            }
        }
        
        rules = compatibility_rules.get(regime, {})
        
        # Determine signal characteristics
        signal_type = self._classify_signal_type(signal)
        direction_type = 'buy_signals' if signal.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY] else 'sell_signals'
        
        # Calculate compatibility score
        direction_compatibility = rules.get(direction_type, 0.5)
        type_compatibility = rules.get(signal_type, 0.5)
        
        # Weighted average
        compatibility = (direction_compatibility * 0.6) + (type_compatibility * 0.4)
        
        return compatibility
    
    def _classify_signal_type(self, signal: TechnicalSignal) -> str:
        """Classify signal type for regime compatibility"""
        
        if isinstance(signal, (ClassificationSignal, RegressionSignal)):
            return 'ml_signals'
        
        # Technical indicator classification
        momentum_indicators = ['MACD', 'RSI', 'Momentum', 'Stochastic']
        mean_reversion_indicators = ['BB', 'Bollinger', 'Mean_Reversion']
        
        indicator_name = signal.indicator.upper()
        
        if any(momentum_ind in indicator_name for momentum_ind in momentum_indicators):
            return 'momentum_indicators'
        elif any(mean_rev_ind in indicator_name for mean_rev_ind in mean_reversion_indicators):
            return 'mean_reversion'
        else:
            return 'momentum_indicators'  # Default

class VolatilityFilter(BaseSignalFilter):
    """
    Volatility-based signal filter.
    
    Filters signals based on market volatility conditions.
    High volatility periods may require different signal treatment.
    """
    
    def __init__(self, config: Optional[FilterConfiguration] = None):
        super().__init__("Volatility Filter", FilterType.VOLATILITY_FILTER, config)
        
        self.max_volatility = self.config.max_volatility
        self.volatility_window = 20
    
    def apply_filter(self, signal: TechnicalSignal, 
                    market_data: Optional[pd.DataFrame] = None,
                    context: Optional[Dict[str, Any]] = None) -> FilterResult:
        """Apply volatility-based filtering"""
        
        if market_data is None or len(market_data) < self.volatility_window:
            return self._create_result(
                signal=signal,
                action=FilterAction.ACCEPT,
                reasoning="Insufficient data for volatility analysis"
            )
        
        # Calculate current volatility
        returns = market_data['close'].pct_change().dropna()
        current_vol = returns.tail(self.volatility_window).std() * np.sqrt(252)
        
        # Calculate volatility score (0 = low vol, 1 = high vol)
        vol_percentile = self._calculate_vol_percentile(returns, current_vol)
        
        # Determine action based on volatility
        if current_vol > self.max_volatility:
            # High volatility - be more conservative
            if vol_percentile > 0.9:  # Extreme volatility
                action = FilterAction.REJECT
                reasoning = f"Extreme volatility rejected (vol: {current_vol:.2%})"
            else:
                action = FilterAction.MODIFY
                # Reduce signal strength in high volatility
                volatility_adjustment = max(0.5, 1.0 - (vol_percentile - 0.5))
                signal.strength *= volatility_adjustment
                reasoning = f"Signal strength reduced for high volatility (adj: {volatility_adjustment:.2f})"
        else:
            action = FilterAction.ACCEPT
            reasoning = f"Volatility acceptable ({current_vol:.2%})"
        
        return self._create_result(
            signal=signal,
            action=action,
            filter_score=1.0 - vol_percentile,  # Lower volatility = higher score
            reasoning=reasoning,
            current_volatility=current_vol,
            volatility_percentile=vol_percentile
        )
    
    def _calculate_vol_percentile(self, returns: pd.Series, current_vol: float) -> float:
        """Calculate volatility percentile"""
        
        if len(returns) < 60:
            return 0.5
        
        rolling_vols = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
        rolling_vols = rolling_vols.dropna()
        
        if len(rolling_vols) == 0:
            return 0.5
        
        percentile = (rolling_vols < current_vol).mean()
        return percentile

# ============================================
# Volume and Liquidity Filters
# ============================================

class VolumeFilter(BaseSignalFilter):
    """
    Volume-based signal filter.
    
    Filters signals based on trading volume conditions.
    Low volume signals may be less reliable.
    """
    
    def __init__(self, config: Optional[FilterConfiguration] = None):
        super().__init__("Volume Filter", FilterType.VOLUME_FILTER, config)
        
        self.min_volume_ratio = self.config.min_volume_ratio
        self.volume_window = 20
    
    def apply_filter(self, signal: TechnicalSignal, 
                    market_data: Optional[pd.DataFrame] = None,
                    context: Optional[Dict[str, Any]] = None) -> FilterResult:
        """Apply volume-based filtering"""
        
        if market_data is None or 'volume' not in market_data.columns or len(market_data) < self.volume_window:
            return self._create_result(
                signal=signal,
                action=FilterAction.ACCEPT,
                reasoning="Insufficient volume data for analysis"
            )
        
        # Calculate volume metrics
        current_volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['volume'].tail(self.volume_window).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume trend (increasing/decreasing)
        recent_avg = market_data['volume'].tail(5).mean()
        older_avg = market_data['volume'].tail(self.volume_window).head(5).mean()
        volume_trend = (recent_avg / older_avg - 1) if older_avg > 0 else 0
        
        # Determine action based on volume
        if volume_ratio < self.min_volume_ratio:
            action = FilterAction.REJECT
            reasoning = f"Low volume rejected (ratio: {volume_ratio:.2f})"
        elif volume_ratio < self.min_volume_ratio * 1.5:  # Marginal volume
            action = FilterAction.MODIFY
            # Reduce strength for low volume
            volume_adjustment = volume_ratio / (self.min_volume_ratio * 1.5)
            signal.strength *= volume_adjustment
            reasoning = f"Strength reduced for low volume (adj: {volume_adjustment:.2f})"
        else:
            # Normal or high volume
            action = FilterAction.ACCEPT
            
            # Boost strength for exceptionally high volume
            if volume_ratio > 2.0:
                volume_boost = min(1.2, 1.0 + (volume_ratio - 2.0) * 0.1)
                signal.strength = min(1.0, signal.strength * volume_boost)
                action = FilterAction.MODIFY
                reasoning = f"Strength boosted for high volume (boost: {volume_boost:.2f})"
            else:
                reasoning = f"Volume acceptable (ratio: {volume_ratio:.2f})"
        
        return self._create_result(
            signal=signal,
            action=action,
            filter_score=volume_ratio,
            reasoning=reasoning,
            volume_ratio=volume_ratio,
            volume_trend=volume_trend,
            current_volume=current_volume,
            average_volume=avg_volume
        )

# ============================================
# Correlation and Redundancy Filters
# ============================================

class CorrelationFilter(BaseSignalFilter):
    """
    Correlation-based signal filter.
    
    Filters redundant signals that are highly correlated with
    existing signals to avoid over-concentration.
    """
    
    def __init__(self, config: Optional[FilterConfiguration] = None):
        super().__init__("Correlation Filter", FilterType.CORRELATION_FILTER, config)
        
        self.correlation_threshold = self.config.correlation_threshold
        self.signal_history = []  # Track recent signals for correlation analysis
        self.max_history_size = 100
    
    def apply_filter(self, signal: TechnicalSignal, 
                    market_data: Optional[pd.DataFrame] = None,
                    context: Optional[Dict[str, Any]] = None) -> FilterResult:
        """Apply correlation-based filtering"""
        
        # Check for similar recent signals
        correlation_score = self._calculate_signal_correlation(signal)
        
        if correlation_score > self.correlation_threshold:
            action = FilterAction.REJECT
            reasoning = f"Highly correlated with recent signals (correlation: {correlation_score:.3f})"
        elif correlation_score > self.correlation_threshold * 0.7:
            action = FilterAction.MODIFY
            # Reduce strength for correlated signals
            correlation_adjustment = 1.0 - (correlation_score - self.correlation_threshold * 0.7) / (self.correlation_threshold * 0.3)
            signal.strength *= correlation_adjustment
            reasoning = f"Strength reduced for correlation (adj: {correlation_adjustment:.2f})"
        else:
            action = FilterAction.ACCEPT
            reasoning = f"Low correlation with recent signals (correlation: {correlation_score:.3f})"
        
        # Add signal to history (regardless of action)
        self._add_to_history(signal)
        
        return self._create_result(
            signal=signal,
            action=action,
            filter_score=1.0 - correlation_score,
            reasoning=reasoning,
            correlation_score=correlation_score,
            history_size=len(self.signal_history)
        )
    
    def _calculate_signal_correlation(self, signal: TechnicalSignal) -> float:
        """Calculate correlation with recent signals"""
        
        if not self.signal_history:
            return 0.0
        
        # Extract signal features for correlation analysis
        signal_features = self._extract_signal_features(signal)
        
        correlations = []
        for historical_signal in self.signal_history:
            # Only compare signals from same symbol
            if historical_signal.symbol == signal.symbol:
                hist_features = self._extract_signal_features(historical_signal)
                correlation = self._calculate_feature_correlation(signal_features, hist_features)
                correlations.append(correlation)
        
        # Return maximum correlation with recent signals
        return max(correlations) if correlations else 0.0
    
    def _extract_signal_features(self, signal: TechnicalSignal) -> np.ndarray:
        """Extract numerical features from signal for correlation analysis"""
        
        features = [
            float(signal.direction.value),  # Direction as numeric
            signal.strength,
            signal.confidence.value,
            hash(signal.indicator) % 1000 / 1000.0,  # Indicator type as numeric
        ]
        
        # Add ML-specific features
        if isinstance(signal, ClassificationSignal):
            features.extend([
                signal.prediction_probability,
                signal.model_confidence,
                1.0  # ML signal indicator
            ])
        elif isinstance(signal, RegressionSignal):
            features.extend([
                signal.predicted_value if signal.predicted_value else 0.0,
                signal.move_probability,
                2.0  # Regression signal indicator
            ])
        else:
            features.extend([0.0, 0.0, 0.0])  # Technical signal
        
        return np.array(features)
    
    def _calculate_feature_correlation(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate correlation between two feature vectors"""
        
        if len(features1) != len(features2):
            return 0.0
        
        # Normalize features
        features1_norm = (features1 - np.mean(features1)) / (np.std(features1) + 1e-8)
        features2_norm = (features2 - np.mean(features2)) / (np.std(features2) + 1e-8)
        
        # Calculate correlation
        correlation = np.corrcoef(features1_norm, features2_norm)[0, 1]
        
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _add_to_history(self, signal: TechnicalSignal):
        """Add signal to history for future correlation analysis"""
        
        self.signal_history.append(signal)
        
        # Maintain history size
        if len(self.signal_history) > self.max_history_size:
            self.signal_history.pop(0)

# ============================================
# Signal Filter Manager
# ============================================

class SignalFilterManager:
    """
    Comprehensive signal filter manager.
    
    Orchestrates multiple filters to validate and enhance trading signals.
    Provides configurable filtering pipeline with detailed analytics.
    """
    
    def __init__(self, config: Optional[FilterConfiguration] = None):
        self.config = config or FilterConfiguration()
        self.filters = []
        self.filter_results_history = []
        self.signals_processed = 0
        
        # Initialize default filters
        self._initialize_default_filters()
        
        logger.info(f"Initialized SignalFilterManager with {len(self.filters)} filters")
    
    def _initialize_default_filters(self):
        """Initialize default filter set"""
        
        self.filters = [
            QualityFilter(self.config),
            TimeFilter(self.config),
            MarketRegimeFilter(self.config),
            VolatilityFilter(self.config),
            VolumeFilter(self.config),
            CorrelationFilter(self.config)
        ]
    
    def add_filter(self, filter_instance: BaseSignalFilter):
        """Add a custom filter to the pipeline"""
        self.filters.append(filter_instance)
        logger.info(f"Added custom filter: {filter_instance.name}")
    
    def remove_filter(self, filter_name: str):
        """Remove a filter from the pipeline"""
        self.filters = [f for f in self.filters if f.name != filter_name]
        logger.info(f"Removed filter: {filter_name}")
    
    @time_it("signal_filtering_pipeline")
    def filter_signals(self, signals: List[TechnicalSignal],
                      market_data: Optional[pd.DataFrame] = None,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Filter a list of signals through the complete pipeline
        
        Args:
            signals: List of signals to filter
            market_data: Market data for context
            context: Additional context information
            
        Returns:
            Dictionary containing filtered signals and analysis
        """
        
        if not signals:
            return {
                'accepted_signals': [],
                'rejected_signals': [],
                'modified_signals': [],
                'filter_results': [],
                'summary': {}
            }
        
        accepted_signals = []
        rejected_signals = []
        modified_signals = []
        all_filter_results = []
        
        for signal in signals:
            signal_results = self._process_single_signal(signal, market_data, context)
            
            # Determine final action (most restrictive wins)
            final_action = self._determine_final_action(signal_results)
            
            # Categorize signal based on final action
            if final_action == FilterAction.ACCEPT:
                accepted_signals.append(signal)
            elif final_action == FilterAction.REJECT:
                rejected_signals.append(signal)
            elif final_action == FilterAction.MODIFY:
                modified_signals.append(signal)
            
            all_filter_results.extend(signal_results)
        
        # Generate summary statistics
        summary = self._generate_filter_summary(all_filter_results)
        
        self.signals_processed += len(signals)
        
        result = {
            'accepted_signals': accepted_signals,
            'rejected_signals': rejected_signals,
            'modified_signals': modified_signals,
            'filter_results': all_filter_results,
            'summary': summary
        }
        
        logger.info(f"Filtered {len(signals)} signals: "
                   f"{len(accepted_signals)} accepted, "
                   f"{len(rejected_signals)} rejected, "
                   f"{len(modified_signals)} modified")
        
        return result
    
    def _process_single_signal(self, signal: TechnicalSignal,
                             market_data: Optional[pd.DataFrame],
                             context: Optional[Dict[str, Any]]) -> List[FilterResult]:
        """Process a single signal through all filters"""
        
        results = []
        current_signal = signal  # Track signal modifications
        
        for filter_instance in self.filters:
            if not filter_instance.config.enabled:
                continue
            
            try:
                result = filter_instance.apply_filter(current_signal, market_data, context)
                results.append(result)
                
                # Update signal if modified
                if result.action == FilterAction.MODIFY:
                    current_signal = result.signal
                
                # Stop processing if rejected
                if result.action == FilterAction.REJECT:
                    break
                    
            except Exception as e:
                logger.error(f"Error applying {filter_instance.name}: {e}")
                # Continue with other filters
        
        return results
    
    def _determine_final_action(self, filter_results: List[FilterResult]) -> FilterAction:
        """Determine final action from all filter results"""
        
        if not filter_results:
            return FilterAction.ACCEPT
        
        # Any rejection overrides everything
        if any(result.action == FilterAction.REJECT for result in filter_results):
            return FilterAction.REJECT
        
        # Any modification means the signal was modified
        if any(result.action == FilterAction.MODIFY for result in filter_results):
            return FilterAction.MODIFY
        
        # All filters accepted
        return FilterAction.ACCEPT
    
    def _generate_filter_summary(self, filter_results: List[FilterResult]) -> Dict[str, Any]:
        """Generate summary statistics from filter results"""
        
        if not filter_results:
            return {}
        
        # Count actions by filter
        filter_stats = {}
        for result in filter_results:
            filter_name = result.filter_name
            if filter_name not in filter_stats:
                filter_stats[filter_name] = {
                    'accept': 0, 'reject': 0, 'modify': 0, 'flag': 0
                }
            filter_stats[filter_name][result.action.value] += 1
        
        # Overall statistics
        total_results = len(filter_results)
        actions = [result.action for result in filter_results]
        
        summary = {
            'total_filter_applications': total_results,
            'action_counts': {
                'accept': sum(1 for a in actions if a == FilterAction.ACCEPT),
                'reject': sum(1 for a in actions if a == FilterAction.REJECT),
                'modify': sum(1 for a in actions if a == FilterAction.MODIFY),
                'flag': sum(1 for a in actions if a == FilterAction.FLAG)
            },
            'filter_stats': filter_stats,
            'average_filter_score': np.mean([r.filter_score for r in filter_results]),
            'filters_active': len(set(r.filter_name for r in filter_results))
        }
        
        return summary
    
    def get_filter_performance(self) -> pd.DataFrame:
        """Get performance statistics for all filters"""
        
        performance_data = []
        
        for filter_instance in self.filters:
            stats = filter_instance.get_filter_stats()
            performance_data.append(stats)
        
        return pd.DataFrame(performance_data)
    
    def optimize_filter_parameters(self, historical_signals: List[TechnicalSignal],
                                 outcomes: List[bool]) -> Dict[str, Any]:
        """Optimize filter parameters based on historical performance"""
        
        # This is a placeholder for parameter optimization
        # In practice, this would use techniques like grid search or Bayesian optimization
        
        logger.info("Filter parameter optimization not implemented yet")
        
        return {
            'optimization_status': 'not_implemented',
            'current_parameters': {
                'quality_threshold': self.config.min_strength,
                'time_threshold': self.config.max_age_hours,
                'volume_threshold': self.config.min_volume_ratio
            }
        }

# ============================================
# Utility Functions
# ============================================

def filter_signals(signals: List[TechnicalSignal],
                  market_data: Optional[pd.DataFrame] = None,
                  filter_config: Optional[FilterConfiguration] = None) -> Dict[str, Any]:
    """
    Quick utility function to filter signals
    
    Args:
        signals: List of signals to filter
        market_data: Market data for context
        filter_config: Filter configuration
        
    Returns:
        Dictionary containing filtered signals and analysis
    """
    
    manager = SignalFilterManager(filter_config)
    return manager.filter_signals(signals, market_data)

def create_custom_filter_config(min_strength: float = 0.5,
                              min_confidence: SignalConfidence = SignalConfidence.MEDIUM,
                              max_age_hours: int = 12,
                              min_volume_ratio: float = 0.8,
                              strictness: float = 0.7) -> FilterConfiguration:
    """Create a custom filter configuration"""
    
    return FilterConfiguration(
        min_strength=min_strength,
        min_confidence=min_confidence,
        max_age_hours=max_age_hours,
        min_volume_ratio=min_volume_ratio,
        strictness=strictness
    )

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Signal Filters System")
    
    # Generate sample signals and market data for testing
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range('2023-06-01', periods=100, freq='D')
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
    volumes = 1000000 * (0.5 + np.random.random(100))
    
    market_data = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Create sample signals
    sample_timestamp = pd.Timestamp.now()
    sample_signals = [
        # High quality signal
        TechnicalSignal(
            timestamp=sample_timestamp,
            symbol='AAPL',
            indicator='RSI',
            direction=SignalDirection.BUY,
            strength=0.85,
            confidence=SignalConfidence.HIGH,
            price=150.0
        ),
        # Low quality signal
        TechnicalSignal(
            timestamp=sample_timestamp,
            symbol='AAPL',
            indicator='SMA',
            direction=SignalDirection.SELL,
            strength=0.25,
            confidence=SignalConfidence.LOW,
            price=150.0
        ),
        # Old signal
        TechnicalSignal(
            timestamp=sample_timestamp - timedelta(hours=25),
            symbol='MSFT',
            indicator='MACD',
            direction=SignalDirection.BUY,
            strength=0.75,
            confidence=SignalConfidence.MEDIUM,
            price=250.0
        ),
        # ML signal
        ClassificationSignal(
            timestamp=sample_timestamp,
            symbol='GOOGL',
            indicator='Direction_Classifier',
            direction=SignalDirection.BUY,
            strength=0.78,
            confidence=SignalConfidence.HIGH,
            price=2500.0,
            model_name='Random Forest',
            prediction_probability=0.85,
            model_confidence=0.75
        )
    ]
    
    print(f"Created {len(sample_signals)} test signals")
    
    # Test individual filters
    print("\n1. Testing Individual Filters")
    
    # Quality Filter
    quality_filter = QualityFilter()
    for i, signal in enumerate(sample_signals[:2]):
        result = quality_filter.apply_filter(signal)
        print(f"Signal {i+1} - Quality Filter: {result.action.value} - {result.reasoning}")
    
    # Time Filter
    time_filter = TimeFilter()
    for i, signal in enumerate(sample_signals[2:3]):
        result = time_filter.apply_filter(signal)
        print(f"Signal {i+3} - Time Filter: {result.action.value} - {result.reasoning}")
    
    # Volume Filter
    volume_filter = VolumeFilter()
    result = volume_filter.apply_filter(sample_signals[0], market_data)
    print(f"Volume Filter: {result.action.value} - {result.reasoning}")
    
    # Market Regime Filter
    regime_filter = MarketRegimeFilter()
    result = regime_filter.apply_filter(sample_signals[0], market_data)
    print(f"Regime Filter: {result.action.value} - {result.reasoning}")
    print(f"  Detected regime: {result.metadata.get('market_regime', 'unknown')}")
    
    print("\n2. Testing Signal Filter Manager")
    
    # Test complete filtering pipeline
    filter_manager = SignalFilterManager()
    
    filter_results = filter_manager.filter_signals(sample_signals, market_data)
    
    print("Filtering Results:")
    print(f"  Accepted signals: {len(filter_results['accepted_signals'])}")
    print(f"  Rejected signals: {len(filter_results['rejected_signals'])}")
    print(f"  Modified signals: {len(filter_results['modified_signals'])}")
    
    # Show detailed results
    print("\nDetailed Filter Results:")
    for result in filter_results['filter_results']:
        print(f"  {result.signal.symbol} - {result.filter_name}: {result.action.value}")
        print(f"    Reason: {result.reasoning}")
        if result.filter_score > 0:
            print(f"    Score: {result.filter_score:.3f}")
    
    # Summary statistics
    summary = filter_results['summary']
    print(f"\nSummary Statistics:")
    print(f"  Total filter applications: {summary['total_filter_applications']}")
    print(f"  Action counts: {summary['action_counts']}")
    print(f"  Average filter score: {summary['average_filter_score']:.3f}")
    print(f"  Active filters: {summary['filters_active']}")
    
    print("\n3. Testing Custom Filter Configuration")
    
    # Create strict filter configuration
    strict_config = create_custom_filter_config(
        min_strength=0.7,
        min_confidence=SignalConfidence.HIGH,
        max_age_hours=6,
        min_volume_ratio=1.0,
        strictness=0.9
    )
    
    strict_manager = SignalFilterManager(strict_config)
    strict_results = strict_manager.filter_signals(sample_signals, market_data)
    
    print("Strict Filtering Results:")
    print(f"  Accepted signals: {len(strict_results['accepted_signals'])}")
    print(f"  Rejected signals: {len(strict_results['rejected_signals'])}")
    print(f"  Modified signals: {len(strict_results['modified_signals'])}")
    
    print("\n4. Testing Filter Performance Analysis")
    
    # Get filter performance statistics
    performance_df = filter_manager.get_filter_performance()
    print("Filter Performance:")
    print(performance_df[['filter_name', 'signals_processed', 'acceptance_rate', 'rejection_rate']].to_string(index=False))
    
    print("\n5. Testing Correlation Filter")
    
    # Create similar signals to test correlation filtering
    similar_signals = [
        TechnicalSignal(
            timestamp=sample_timestamp,
            symbol='AAPL',
            indicator='RSI',
            direction=SignalDirection.BUY,
            strength=0.80,
            confidence=SignalConfidence.HIGH,
            price=150.0
        ),
        TechnicalSignal(
            timestamp=sample_timestamp,
            symbol='AAPL',
            indicator='Stochastic',  # Similar momentum indicator
            direction=SignalDirection.BUY,
            strength=0.82,
            confidence=SignalConfidence.HIGH,
            price=150.0
        )
    ]
    
    correlation_filter = CorrelationFilter()
    
    # Process first signal
    result1 = correlation_filter.apply_filter(similar_signals[0])
    print(f"First signal - Correlation Filter: {result1.action.value}")
    
    # Process similar signal
    result2 = correlation_filter.apply_filter(similar_signals[1])
    print(f"Similar signal - Correlation Filter: {result2.action.value} - {result2.reasoning}")
    
    print("\n6. Testing Utility Functions")
    
    # Test utility function
    utility_results = filter_signals(
        signals=sample_signals[:2],
        market_data=market_data,
        filter_config=strict_config
    )
    
    total_utility_signals = (len(utility_results['accepted_signals']) + 
                           len(utility_results['rejected_signals']) + 
                           len(utility_results['modified_signals']))
    
    print(f"Utility function processed {total_utility_signals} signals")
    
    print("\nSignal filters system testing completed successfully!")
    print("\nImplemented signal filters include:")
    print("• Quality Filter: Filters signals based on strength, confidence, and ML performance")
    print("• Time Filter: Validates signal timing, age, and market hours")
    print("• Market Regime Filter: Adapts signals based on bull/bear/sideways market conditions")
    print("• Volatility Filter: Adjusts signals based on market volatility levels")
    print("• Volume Filter: Validates signals based on trading volume conditions")
    print("• Correlation Filter: Removes redundant highly-correlated signals")
    print("• Extensible Framework: Easy to add custom filters with specific logic")
    print("• Performance Analytics: Detailed statistics on filter effectiveness")
