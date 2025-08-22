# ============================================
# StockPredictionPro - src/trading/execution/slippage.py
# Comprehensive slippage modeling and analysis system for trading execution
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum  
import uuid
from collections import defaultdict, deque
import threading
import math
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it
from .limit_orders import OrderSide, Fill
from .market_orders import MarketOrder

logger = get_logger('trading.execution.slippage')

# ============================================
# Slippage Data Structures and Enums
# ============================================

class SlippageType(Enum):
    """Types of slippage"""
    MARKET_IMPACT = "market_impact"        # Price impact from order size
    BID_ASK_SPREAD = "bid_ask_spread"      # Cost of crossing spread
    TIMING_SLIPPAGE = "timing_slippage"    # Delays in execution
    INFORMATION_SLIPPAGE = "information_slippage"  # Price moves due to new information
    LIQUIDITY_SLIPPAGE = "liquidity_slippage"      # Low liquidity periods

class SlippageModel(Enum):
    """Slippage estimation models"""
    FIXED_BASIS_POINTS = "fixed_basis_points"
    VOLUME_PERCENTAGE = "volume_percentage"
    SQUARE_ROOT = "square_root"
    LINEAR_IMPACT = "linear_impact"
    HISTORICAL = "historical"
    MACHINE_LEARNING = "machine_learning"
    ABDI_RANALDO = "abdi_ranaldo"

class LiquidityRegime(Enum):
    """Market liquidity regimes"""
    HIGH_LIQUIDITY = "high_liquidity"
    NORMAL_LIQUIDITY = "normal_liquidity" 
    LOW_LIQUIDITY = "low_liquidity"
    STRESSED_LIQUIDITY = "stressed_liquidity"

@dataclass
class SlippageEstimate:
    """Comprehensive slippage estimate"""
    
    # Basic estimate information
    symbol: str
    order_side: OrderSide
    quantity: int
    reference_price: float
    
    # Slippage components (all in basis points)
    market_impact_bps: float = 0.0
    spread_cost_bps: float = 0.0
    timing_cost_bps: float = 0.0
    liquidity_premium_bps: float = 0.0
    total_estimated_slippage_bps: float = 0.0
    
    # Confidence and risk metrics
    confidence_level: float = 0.95
    slippage_volatility: float = 0.0
    worst_case_slippage_bps: float = 0.0
    best_case_slippage_bps: float = 0.0
    
    # Market context
    liquidity_regime: LiquidityRegime = LiquidityRegime.NORMAL_LIQUIDITY
    participation_rate: float = 0.0
    market_volatility: float = 0.0
    
    # Model details
    model_used: SlippageModel = SlippageModel.FIXED_BASIS_POINTS
    estimation_timestamp: datetime = field(default_factory=datetime.now)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_estimated_cost_dollars(self) -> float:
        """Calculate total estimated cost in dollars"""
        return (self.total_estimated_slippage_bps / 10000) * self.reference_price * self.quantity
    
    @property
    def market_impact_dollars(self) -> float:
        """Calculate market impact cost in dollars"""
        return (self.market_impact_bps / 10000) * self.reference_price * self.quantity
    
    @property
    def spread_cost_dollars(self) -> float:
        """Calculate spread cost in dollars"""
        return (self.spread_cost_bps / 10000) * self.reference_price * self.quantity

@dataclass
class SlippageAnalysis:
    """Post-trade slippage analysis"""
    
    # Trade information
    symbol: str
    order_side: OrderSide
    quantity: int
    reference_price: float
    average_fill_price: float
    
    # Realized slippage
    realized_slippage_bps: float = 0.0
    realized_slippage_dollars: float = 0.0
    
    # Comparison with estimates
    estimated_slippage_bps: float = 0.0
    estimation_error_bps: float = 0.0
    estimation_accuracy: float = 0.0
    
    # Timing analysis
    execution_start_time: datetime = field(default_factory=datetime.now)
    execution_end_time: datetime = field(default_factory=datetime.now)
    execution_duration_seconds: float = 0.0
    
    # Market conditions during execution
    avg_volatility_during_execution: float = 0.0
    avg_volume_during_execution: float = 0.0
    liquidity_regime_during_execution: LiquidityRegime = LiquidityRegime.NORMAL_LIQUIDITY
    
    # Attribution analysis
    slippage_attribution: Dict[SlippageType, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.execution_duration_seconds = (self.execution_end_time - self.execution_start_time).total_seconds()
        
        # Calculate realized slippage
        if self.order_side == OrderSide.BUY:
            self.realized_slippage_bps = ((self.average_fill_price - self.reference_price) / self.reference_price) * 10000
        else:
            self.realized_slippage_bps = ((self.reference_price - self.average_fill_price) / self.reference_price) * 10000
        
        self.realized_slippage_dollars = (self.realized_slippage_bps / 10000) * self.reference_price * self.quantity
        
        # Calculate estimation error
        self.estimation_error_bps = abs(self.realized_slippage_bps - self.estimated_slippage_bps)
        
        if self.estimated_slippage_bps > 0:
            error_rate = self.estimation_error_bps / self.estimated_slippage_bps
            self.estimation_accuracy = max(0, 1 - error_rate)

# ============================================
# Base Slippage Estimator
# ============================================

class BaseSlippageEstimator:
    """
    Base class for slippage estimation models.
    
    Provides common functionality for estimating slippage
    based on order characteristics and market conditions.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calibration_data = defaultdict(list)
        self.model_parameters = {}
        
        logger.debug(f"Initialized {model_name} slippage estimator")
    
    def estimate_slippage(self, symbol: str, order_side: OrderSide, quantity: int,
                         reference_price: float, market_data: Dict[str, Any]) -> SlippageEstimate:
        """Estimate slippage - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement estimate_slippage method")
    
    def calibrate(self, historical_executions: List[SlippageAnalysis]):
        """Calibrate model parameters using historical execution data"""
        logger.info(f"Calibrating {self.model_name} with {len(historical_executions)} executions")
        
        # Store calibration data
        for execution in historical_executions:
            self.calibration_data[execution.symbol].append(execution)
        
        # Model-specific calibration logic in subclasses
        self._perform_calibration()
    
    def _perform_calibration(self):
        """Perform model-specific calibration - to be implemented by subclasses"""
        pass
    
    def _calculate_participation_rate(self, quantity: int, avg_daily_volume: float) -> float:
        """Calculate order participation rate"""
        if avg_daily_volume <= 0:
            return 0.1  # Default assumption
        return quantity / avg_daily_volume
    
    def _determine_liquidity_regime(self, market_data: Dict[str, Any]) -> LiquidityRegime:
        """Determine current liquidity regime"""
        
        volume_ratio = market_data.get('volume_ratio', 1.0)  # Current vs average volume
        spread_percentile = market_data.get('spread_percentile', 0.5)
        volatility_percentile = market_data.get('volatility_percentile', 0.5)
        
        # Liquidity regime classification logic
        if volume_ratio > 1.5 and spread_percentile < 0.3:
            return LiquidityRegime.HIGH_LIQUIDITY
        elif volume_ratio < 0.5 or spread_percentile > 0.8 or volatility_percentile > 0.9:
            return LiquidityRegime.LOW_LIQUIDITY
        elif volatility_percentile > 0.8 or spread_percentile > 0.7:
            return LiquidityRegime.STRESSED_LIQUIDITY
        else:
            return LiquidityRegime.NORMAL_LIQUIDITY

# ============================================
# Fixed Basis Points Model
# ============================================

class FixedBasisPointsEstimator(BaseSlippageEstimator):
    """
    Fixed basis points slippage model.
    
    Simple model that applies a fixed slippage cost based on
    order characteristics with some adjustments for market conditions.
    """
    
    def __init__(self, base_slippage_bps: float = 5.0, 
                 volume_impact_factor: float = 0.1):
        super().__init__("Fixed Basis Points")
        self.base_slippage_bps = base_slippage_bps
        self.volume_impact_factor = volume_impact_factor
    
    def estimate_slippage(self, symbol: str, order_side: OrderSide, quantity: int,
                         reference_price: float, market_data: Dict[str, Any]) -> SlippageEstimate:
        """Estimate slippage using fixed basis points model"""
        
        # Base slippage
        base_cost = self.base_slippage_bps
        
        # Volume impact adjustment
        avg_volume = market_data.get('average_volume', 1000000)
        participation_rate = self._calculate_participation_rate(quantity, avg_volume)
        volume_impact = participation_rate * self.volume_impact_factor * 10000  # Convert to bps
        
        # Spread cost
        spread = market_data.get('spread', 0.01)
        spread_cost_bps = (spread / reference_price) * 10000 / 2  # Half spread
        
        # Liquidity regime adjustment
        liquidity_regime = self._determine_liquidity_regime(market_data)
        liquidity_multiplier = {
            LiquidityRegime.HIGH_LIQUIDITY: 0.7,
            LiquidityRegime.NORMAL_LIQUIDITY: 1.0,
            LiquidityRegime.LOW_LIQUIDITY: 1.5,
            LiquidityRegime.STRESSED_LIQUIDITY: 2.0
        }[liquidity_regime]
        
        # Calculate components
        market_impact_bps = base_cost + volume_impact
        market_impact_bps *= liquidity_multiplier
        
        total_slippage_bps = market_impact_bps + spread_cost_bps
        
        # Confidence intervals (simple approximation)
        volatility = market_data.get('volatility', 0.02)
        slippage_volatility = total_slippage_bps * volatility * 2  # Rough approximation
        
        return SlippageEstimate(
            symbol=symbol,
            order_side=order_side,
            quantity=quantity,
            reference_price=reference_price,
            market_impact_bps=market_impact_bps,
            spread_cost_bps=spread_cost_bps,
            total_estimated_slippage_bps=total_slippage_bps,
            slippage_volatility=slippage_volatility,
            worst_case_slippage_bps=total_slippage_bps + 2 * slippage_volatility,
            best_case_slippage_bps=max(0, total_slippage_bps - slippage_volatility),
            liquidity_regime=liquidity_regime,
            participation_rate=participation_rate,
            market_volatility=volatility,
            model_used=SlippageModel.FIXED_BASIS_POINTS
        )

# ============================================
# Square Root Impact Model
# ============================================

class SquareRootImpactEstimator(BaseSlippageEstimator):
    """
    Square root market impact model.
    
    Based on the widely-used assumption that market impact
    scales with the square root of order size.
    """
    
    def __init__(self, impact_coefficient: float = 1.0):
        super().__init__("Square Root Impact")
        self.impact_coefficient = impact_coefficient
    
    def estimate_slippage(self, symbol: str, order_side: OrderSide, quantity: int,
                         reference_price: float, market_data: Dict[str, Any]) -> SlippageEstimate:
        """Estimate slippage using square root impact model"""
        
        # Market data
        avg_volume = market_data.get('average_volume', 1000000)
        volatility = market_data.get('volatility', 0.02)
        spread = market_data.get('spread', 0.01)
        
        # Calculate participation rate
        participation_rate = self._calculate_participation_rate(quantity, avg_volume)
        
        # Square root impact: σ * sqrt(participation_rate) * coefficient
        # Impact is proportional to volatility and square root of participation
        market_impact_bps = (self.impact_coefficient * volatility * 
                           np.sqrt(participation_rate) * 10000)
        
        # Spread cost
        spread_cost_bps = (spread / reference_price) * 10000 / 2
        
        # Liquidity adjustments
        liquidity_regime = self._determine_liquidity_regime(market_data)
        liquidity_adjustment = {
            LiquidityRegime.HIGH_LIQUIDITY: 0.8,
            LiquidityRegime.NORMAL_LIQUIDITY: 1.0,
            LiquidityRegime.LOW_LIQUIDITY: 1.4,
            LiquidityRegime.STRESSED_LIQUIDITY: 2.2
        }[liquidity_regime]
        
        market_impact_bps *= liquidity_adjustment
        
        # Total slippage
        total_slippage_bps = market_impact_bps + spread_cost_bps
        
        # Confidence intervals based on model uncertainty
        model_uncertainty = 0.3  # 30% model uncertainty
        slippage_volatility = total_slippage_bps * model_uncertainty
        
        return SlippageEstimate(
            symbol=symbol,
            order_side=order_side,
            quantity=quantity,
            reference_price=reference_price,
            market_impact_bps=market_impact_bps,
            spread_cost_bps=spread_cost_bps,
            total_estimated_slippage_bps=total_slippage_bps,
            slippage_volatility=slippage_volatility,
            worst_case_slippage_bps=total_slippage_bps + 2 * slippage_volatility,
            best_case_slippage_bps=max(0, total_slippage_bps - slippage_volatility),
            liquidity_regime=liquidity_regime,
            participation_rate=participation_rate,
            market_volatility=volatility,
            model_used=SlippageModel.SQUARE_ROOT
        )

# ============================================
# Machine Learning Slippage Estimator
# ============================================

class MLSlippageEstimator(BaseSlippageEstimator):
    """
    Machine learning-based slippage estimator.
    
    Uses historical execution data to train a model
    that predicts slippage based on multiple factors.
    """
    
    def __init__(self):
        super().__init__("Machine Learning")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def estimate_slippage(self, symbol: str, order_side: OrderSide, quantity: int,
                         reference_price: float, market_data: Dict[str, Any]) -> SlippageEstimate:
        """Estimate slippage using trained ML model"""
        
        if not self.is_trained:
            logger.warning("ML model not trained, falling back to square root model")
            fallback = SquareRootImpactEstimator()
            estimate = fallback.estimate_slippage(symbol, order_side, quantity, reference_price, market_data)
            estimate.model_used = SlippageModel.MACHINE_LEARNING
            return estimate
        
        # Prepare features
        features = self._prepare_features(symbol, order_side, quantity, reference_price, market_data)
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        predicted_slippage_bps = self.model.predict(features_scaled)[0]
        
        # Get prediction uncertainty (for Random Forest)
        if hasattr(self.model, 'estimators_'):
            tree_predictions = [tree.predict(features_scaled)[0] for tree in self.model.estimators_]
            slippage_volatility = np.std(tree_predictions)
        else:
            slippage_volatility = predicted_slippage_bps * 0.2  # 20% uncertainty
        
        # Decompose into components (simplified attribution)
        spread = market_data.get('spread', 0.01)
        spread_cost_bps = (spread / reference_price) * 10000 / 2
        market_impact_bps = max(0, predicted_slippage_bps - spread_cost_bps)
        
        # Calculate participation rate and liquidity regime
        avg_volume = market_data.get('average_volume', 1000000)
        participation_rate = self._calculate_participation_rate(quantity, avg_volume)
        liquidity_regime = self._determine_liquidity_regime(market_data)
        
        return SlippageEstimate(
            symbol=symbol,
            order_side=order_side,
            quantity=quantity,
            reference_price=reference_price,
            market_impact_bps=market_impact_bps,
            spread_cost_bps=spread_cost_bps,
            total_estimated_slippage_bps=predicted_slippage_bps,
            slippage_volatility=slippage_volatility,
            worst_case_slippage_bps=predicted_slippage_bps + 2 * slippage_volatility,
            best_case_slippage_bps=max(0, predicted_slippage_bps - slippage_volatility),
            liquidity_regime=liquidity_regime,
            participation_rate=participation_rate,
            market_volatility=market_data.get('volatility', 0.02),
            model_used=SlippageModel.MACHINE_LEARNING
        )
    
    def _prepare_features(self, symbol: str, order_side: OrderSide, quantity: int,
                         reference_price: float, market_data: Dict[str, Any]) -> List[float]:
        """Prepare features for ML model"""
        
        features = []
        
        # Order characteristics
        features.append(float(quantity))
        features.append(float(reference_price))
        features.append(1.0 if order_side == OrderSide.BUY else -1.0)
        
        # Market characteristics
        features.append(market_data.get('average_volume', 1000000))
        features.append(market_data.get('volatility', 0.02))
        features.append(market_data.get('spread', 0.01))
        features.append(market_data.get('volume_ratio', 1.0))
        
        # Derived features
        avg_volume = market_data.get('average_volume', 1000000)
        participation_rate = self._calculate_participation_rate(quantity, avg_volume)
        features.append(participation_rate)
        features.append(np.sqrt(participation_rate))
        features.append(np.log(quantity + 1))
        features.append(market_data.get('volatility', 0.02) * participation_rate)
        
        # Time-based features
        now = datetime.now()
        features.append(now.hour)  # Hour of day
        features.append(now.weekday())  # Day of week
        
        # Market regime indicators
        features.append(market_data.get('spread_percentile', 0.5))
        features.append(market_data.get('volatility_percentile', 0.5))
        features.append(market_data.get('volume_percentile', 0.5))
        
        return features
    
    def _perform_calibration(self):
        """Train ML model on historical execution data"""
        
        if not self.calibration_data:
            logger.warning("No calibration data available for ML model")
            return
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for symbol, executions in self.calibration_data.items():
            for execution in executions:
                # Reconstruct market data from execution (simplified)
                market_data = {
                    'average_volume': execution.avg_volume_during_execution,
                    'volatility': execution.avg_volatility_during_execution,
                    'spread': 0.01,  # Would need to be stored in execution data
                    'volume_ratio': 1.0,  # Would need to be stored
                    'spread_percentile': 0.5,
                    'volatility_percentile': 0.5,
                    'volume_percentile': 0.5
                }
                
                features = self._prepare_features(
                    execution.symbol, execution.order_side, execution.quantity,
                    execution.reference_price, market_data
                )
                
                X_train.append(features)
                y_train.append(execution.realized_slippage_bps)
        
        if len(X_train) < 10:
            logger.warning("Insufficient training data for ML model")
            return
        
        # Scale features and train model
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.model.predict(X_train_scaled)
        r2_score = self.model.score(X_train_scaled, y_train)
        
        logger.info(f"ML slippage model trained on {len(X_train)} samples, R² = {r2_score:.3f}")

# ============================================
# Abdi-Ranaldo Spread Estimator
# ============================================

class AbdiRanaldoEstimator(BaseSlippageEstimator):
    """
    Abdi-Ranaldo spread estimation model.
    
    Uses high-low-close data to estimate effective spreads
    and market impact without requiring bid-ask data.
    """
    
    def __init__(self):
        super().__init__("Abdi-Ranaldo")
    
    def estimate_slippage(self, symbol: str, order_side: OrderSide, quantity: int,
                         reference_price: float, market_data: Dict[str, Any]) -> SlippageEstimate:
        """Estimate slippage using Abdi-Ranaldo model"""
        
        # Extract OHLC data
        highs = market_data.get('highs', [reference_price])
        lows = market_data.get('lows', [reference_price])
        closes = market_data.get('closes', [reference_price])
        
        # Estimate effective spread
        estimated_spread = self._estimate_effective_spread(highs, lows, closes)
        
        # Market impact estimation
        avg_volume = market_data.get('average_volume', 1000000)
        participation_rate = self._calculate_participation_rate(quantity, avg_volume)
        
        # Base impact using estimated spread
        spread_cost_bps = (estimated_spread / reference_price) * 10000 / 2
        
        # Additional market impact for large orders
        if participation_rate > 0.01:  # > 1% of daily volume
            additional_impact = np.sqrt(participation_rate) * estimated_spread / reference_price * 10000
        else:
            additional_impact = 0
        
        market_impact_bps = additional_impact
        total_slippage_bps = spread_cost_bps + market_impact_bps
        
        # Uncertainty estimation
        volatility = market_data.get('volatility', 0.02)
        slippage_volatility = total_slippage_bps * volatility
        
        # Liquidity regime
        liquidity_regime = self._determine_liquidity_regime(market_data)
        
        return SlippageEstimate(
            symbol=symbol,
            order_side=order_side,
            quantity=quantity,
            reference_price=reference_price,
            market_impact_bps=market_impact_bps,
            spread_cost_bps=spread_cost_bps,
            total_estimated_slippage_bps=total_slippage_bps,
            slippage_volatility=slippage_volatility,
            worst_case_slippage_bps=total_slippage_bps + 2 * slippage_volatility,
            best_case_slippage_bps=max(0, total_slippage_bps - slippage_volatility),
            liquidity_regime=liquidity_regime,
            participation_rate=participation_rate,
            market_volatility=volatility,
            model_used=SlippageModel.ABDI_RANALDO,
            metadata={'estimated_spread': estimated_spread}
        )
    
    def _estimate_effective_spread(self, highs: List[float], lows: List[float], 
                                  closes: List[float]) -> float:
        """Estimate effective spread using Abdi-Ranaldo method"""
        
        if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
            return 0.01  # Default spread
        
        # Convert to numpy arrays
        highs = np.array(highs[-20:])  # Use last 20 observations
        lows = np.array(lows[-20:])
        closes = np.array(closes[-20:])
        
        # Abdi-Ranaldo estimator
        # S = 2 * sqrt(α * (H-C) * (C-L))
        # where α is estimated from price variance decomposition
        
        # Calculate price ranges
        high_close = highs - closes
        close_low = closes - lows
        
        # Estimate α parameter (simplified version)
        # In practice, this would involve more sophisticated estimation
        price_returns = np.diff(closes) / closes[:-1]
        alpha = min(0.3, np.var(price_returns) * 0.5)  # Rough approximation
        
        # Calculate effective spread estimates
        spread_estimates = 2 * np.sqrt(alpha * high_close * close_low)
        
        # Remove invalid estimates and take median
        valid_estimates = spread_estimates[spread_estimates > 0]
        
        if len(valid_estimates) > 0:
            return np.median(valid_estimates)
        else:
            return 0.01  # Fallback

# ============================================
# Slippage Manager
# ============================================

class SlippageManager:
    """
    Comprehensive slippage management system.
    
    Provides pre-trade slippage estimation, post-trade analysis,
    model calibration, and slippage monitoring capabilities.
    """
    
    def __init__(self):
        self.estimators = {}
        self.execution_history = []
        self.slippage_budgets = defaultdict(float)  # By strategy/portfolio
        self.model_performance = defaultdict(dict)
        
        # Initialize default estimators
        self.estimators['fixed_bp'] = FixedBasisPointsEstimator()
        self.estimators['sqrt_impact'] = SquareRootImpactEstimator()
        self.estimators['ml'] = MLSlippageEstimator()
        self.estimators['abdi_ranaldo'] = AbdiRanaldoEstimator()
        
        # Configuration
        self.default_model = 'sqrt_impact'
        self.confidence_level = 0.95
        
        # Threading
        self._lock = threading.RLock()
        
        logger.info("Initialized SlippageManager with 4 estimation models")
    
    @time_it("slippage_estimation")
    def estimate_slippage(self, symbol: str, order_side: OrderSide, quantity: int,
                         reference_price: float, market_data: Dict[str, Any],
                         model: Optional[str] = None) -> SlippageEstimate:
        """
        Estimate pre-trade slippage
        
        Args:
            symbol: Trading symbol
            order_side: Buy or sell
            quantity: Order quantity
            reference_price: Reference price for slippage calculation
            market_data: Current market data
            model: Model to use (None for default)
            
        Returns:
            SlippageEstimate object
        """
        
        model_name = model or self.default_model
        
        if model_name not in self.estimators:
            logger.warning(f"Unknown model {model_name}, using default")
            model_name = self.default_model
        
        estimator = self.estimators[model_name]
        
        try:
            estimate = estimator.estimate_slippage(symbol, order_side, quantity, reference_price, market_data)
            
            logger.debug(f"Slippage estimate for {symbol}: {estimate.total_estimated_slippage_bps:.1f} bps "
                        f"(${estimate.total_estimated_cost_dollars:.2f})")
            
            return estimate
            
        except Exception as e:
            logger.error(f"Error estimating slippage: {e}")
            # Return conservative estimate
            return SlippageEstimate(
                symbol=symbol,
                order_side=order_side,
                quantity=quantity,
                reference_price=reference_price,
                total_estimated_slippage_bps=20.0,  # Conservative 20 bps
                model_used=SlippageModel.FIXED_BASIS_POINTS
            )
    
    def analyze_execution(self, order: MarketOrder, fills: List[Fill], 
                         market_data: Optional[Dict[str, Any]] = None) -> SlippageAnalysis:
        """
        Analyze post-trade slippage
        
        Args:
            order: Executed market order
            fills: List of fills for the order
            market_data: Market data during execution
            
        Returns:
            SlippageAnalysis object
        """
        
        if not fills:
            logger.warning("No fills provided for slippage analysis")
            return SlippageAnalysis(
                symbol=order.symbol,
                order_side=order.side,
                quantity=order.quantity,
                reference_price=order.market_price_at_creation or 0.0,
                average_fill_price=0.0
            )
        
        # Calculate average fill price
        total_value = sum(fill.quantity * fill.price for fill in fills)
        total_quantity = sum(fill.quantity for fill in fills)
        average_fill_price = total_value / total_quantity if total_quantity > 0 else 0.0
        
        # Reference price (price at order creation)
        reference_price = order.market_price_at_creation or fills[0].price
        
        # Create analysis
        analysis = SlippageAnalysis(
            symbol=order.symbol,
            order_side=order.side,
            quantity=total_quantity,
            reference_price=reference_price,
            average_fill_price=average_fill_price,
            execution_start_time=order.start_execution_time or order.creation_time,
            execution_end_time=fills[-1].timestamp if fills else order.creation_time,
            estimated_slippage_bps=order.expected_shortfall or 0.0
        )
        
        # Add market data if available
        if market_data:
            analysis.avg_volatility_during_execution = market_data.get('volatility', 0.0)
            analysis.avg_volume_during_execution = market_data.get('average_volume', 0.0)
            analysis.liquidity_regime_during_execution = self.estimators[self.default_model]._determine_liquidity_regime(market_data)
        
        # Slippage attribution (simplified)
        self._attribute_slippage(analysis, market_data)
        
        # Store in history
        with self._lock:
            self.execution_history.append(analysis)
        
        logger.info(f"Slippage analysis for {order.order_id}: "
                   f"{analysis.realized_slippage_bps:.1f} bps "
                   f"(${analysis.realized_slippage_dollars:.2f})")
        
        return analysis
    
    def _attribute_slippage(self, analysis: SlippageAnalysis, market_data: Optional[Dict[str, Any]]):
        """Attribute realized slippage to different sources"""
        
        total_slippage = analysis.realized_slippage_bps
        
        if not market_data:
            # Simple attribution without market data
            analysis.slippage_attribution = {
                SlippageType.MARKET_IMPACT: total_slippage * 0.6,
                SlippageType.BID_ASK_SPREAD: total_slippage * 0.4
            }
            return
        
        # Estimate spread component
        spread = market_data.get('spread', 0.01)
        spread_cost_bps = (spread / analysis.reference_price) * 10000 / 2
        
        # Estimate timing component based on execution duration
        if analysis.execution_duration_seconds > 60:  # More than 1 minute
            volatility = market_data.get('volatility', 0.02)
            timing_cost_bps = volatility * np.sqrt(analysis.execution_duration_seconds / 3600) * 10000
        else:
            timing_cost_bps = 0
        
        # Remaining is market impact
        market_impact_bps = total_slippage - spread_cost_bps - timing_cost_bps
        
        analysis.slippage_attribution = {
            SlippageType.MARKET_IMPACT: market_impact_bps,
            SlippageType.BID_ASK_SPREAD: spread_cost_bps,
            SlippageType.TIMING_SLIPPAGE: timing_cost_bps
        }
    
    def calibrate_models(self, symbol: Optional[str] = None):
        """Calibrate slippage models using historical execution data"""
        
        with self._lock:
            if not self.execution_history:
                logger.warning("No execution history available for calibration")
                return
            
            # Filter data by symbol if specified
            if symbol:
                calibration_data = [ex for ex in self.execution_history if ex.symbol == symbol]
                logger.info(f"Calibrating models for {symbol} with {len(calibration_data)} executions")
            else:
                calibration_data = self.execution_history
                logger.info(f"Calibrating models with {len(calibration_data)} executions")
            
            # Calibrate each model
            for model_name, estimator in self.estimators.items():
                try:
                    estimator.calibrate(calibration_data)
                    
                    # Calculate model performance metrics
                    self._evaluate_model_performance(model_name, calibration_data)
                    
                except Exception as e:
                    logger.error(f"Error calibrating {model_name}: {e}")
    
    def _evaluate_model_performance(self, model_name: str, execution_data: List[SlippageAnalysis]):
        """Evaluate model performance on historical data"""
        
        if not execution_data:
            return
        
        estimator = self.estimators[model_name]
        errors = []
        
        for execution in execution_data:
            # Re-estimate using current model
            market_data = {
                'average_volume': execution.avg_volume_during_execution,
                'volatility': execution.avg_volatility_during_execution,
                'spread': 0.01,  # Would need to be stored
                'volume_ratio': 1.0
            }
            
            try:
                estimate = estimator.estimate_slippage(
                    execution.symbol, execution.order_side, execution.quantity,
                    execution.reference_price, market_data
                )
                
                error = abs(estimate.total_estimated_slippage_bps - execution.realized_slippage_bps)
                errors.append(error)
                
            except Exception:
                continue
        
        if errors:
            self.model_performance[model_name] = {
                'mean_absolute_error': np.mean(errors),
                'median_absolute_error': np.median(errors),
                'std_error': np.std(errors),
                'sample_size': len(errors)
            }
            
            logger.info(f"{model_name} performance: MAE = {np.mean(errors):.1f} bps")
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models"""
        return dict(self.model_performance)
    
    def set_slippage_budget(self, strategy_id: str, budget_bps: float):
        """Set slippage budget for a strategy"""
        self.slippage_budgets[strategy_id] = budget_bps
        logger.info(f"Set slippage budget for {strategy_id}: {budget_bps} bps")
    
    def check_slippage_budget(self, strategy_id: str, estimated_slippage_bps: float) -> bool:
        """Check if estimated slippage is within budget"""
        
        if strategy_id not in self.slippage_budgets:
            return True  # No budget set
        
        budget = self.slippage_budgets[strategy_id]
        within_budget = estimated_slippage_bps <= budget
        
        if not within_budget:
            logger.warning(f"Slippage estimate {estimated_slippage_bps:.1f} bps exceeds "
                          f"budget {budget:.1f} bps for {strategy_id}")
        
        return within_budget
    
    def get_slippage_statistics(self, symbol: Optional[str] = None, 
                              days: int = 30) -> Dict[str, Any]:
        """Get slippage statistics for recent executions"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self._lock:
            # Filter executions
            filtered_executions = [
                ex for ex in self.execution_history
                if ex.execution_start_time >= cutoff_date
                and (symbol is None or ex.symbol == symbol)
            ]
        
        if not filtered_executions:
            return {'error': 'No executions found for the specified criteria'}
        
        # Calculate statistics
        slippages = [ex.realized_slippage_bps for ex in filtered_executions]
        dollar_costs = [ex.realized_slippage_dollars for ex in filtered_executions]
        
        return {
            'symbol': symbol or 'All',
            'period_days': days,
            'execution_count': len(filtered_executions),
            'slippage_statistics': {
                'mean_bps': np.mean(slippages),
                'median_bps': np.median(slippages),
                'std_bps': np.std(slippages),
                'min_bps': np.min(slippages),
                'max_bps': np.max(slippages),
                'p95_bps': np.percentile(slippages, 95)
            },
            'cost_statistics': {
                'total_dollars': np.sum(dollar_costs),
                'mean_dollars': np.mean(dollar_costs),
                'median_dollars': np.median(dollar_costs)
            }
        }

# ============================================
# Utility Functions
# ============================================

def estimate_execution_slippage(symbol: str, side: str, quantity: int, 
                               reference_price: float, market_data: Dict[str, Any],
                               model: str = 'sqrt_impact') -> SlippageEstimate:
    """
    Quick utility function to estimate slippage
    
    Args:
        symbol: Trading symbol
        side: 'BUY' or 'SELL'
        quantity: Order quantity
        reference_price: Reference price
        market_data: Market data dictionary
        model: Model to use
        
    Returns:
        SlippageEstimate object
    """
    
    manager = SlippageManager()
    order_side = OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL
    
    return manager.estimate_slippage(symbol, order_side, quantity, reference_price, market_data, model)

def calculate_optimal_order_size(symbol: str, side: str, target_quantity: int,
                                max_slippage_bps: float, market_data: Dict[str, Any]) -> int:
    """
    Calculate optimal order size to stay within slippage budget
    
    Args:
        symbol: Trading symbol
        side: 'BUY' or 'SELL'
        target_quantity: Desired total quantity
        max_slippage_bps: Maximum acceptable slippage
        market_data: Market data dictionary
        
    Returns:
        Recommended order size
    """
    
    manager = SlippageManager()
    order_side = OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL
    
    # Binary search for optimal size
    low, high = 1, target_quantity
    optimal_size = target_quantity
    
    while low <= high:
        mid = (low + high) // 2
        
        estimate = manager.estimate_slippage(
            symbol, order_side, mid, 
            market_data.get('price', 100.0), market_data
        )
        
        if estimate.total_estimated_slippage_bps <= max_slippage_bps:
            optimal_size = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return optimal_size

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Slippage System")
    
    # Sample market data
    market_data = {
        'price': 150.00,
        'average_volume': 10000000,  # 10M daily volume
        'volatility': 0.30,          # 30% annualized volatility
        'spread': 0.02,              # $0.02 spread
        'volume_ratio': 0.8,         # 80% of normal volume
        'spread_percentile': 0.6,    # 60th percentile spread
        'volatility_percentile': 0.7, # 70th percentile volatility
        'volume_percentile': 0.4,    # 40th percentile volume
        'highs': [150.5, 151.0, 150.8, 151.2, 150.9],
        'lows': [149.5, 149.8, 150.0, 150.1, 150.2],
        'closes': [150.0, 150.5, 150.3, 150.8, 150.5]
    }
    
    # Initialize slippage manager
    manager = SlippageManager()
    
    print("\n1. Testing Different Slippage Models")
    
    test_cases = [
        ("Small Order", 1000),
        ("Medium Order", 10000),
        ("Large Order", 100000)
    ]
    
    models = ['fixed_bp', 'sqrt_impact', 'abdi_ranaldo']
    
    for case_name, quantity in test_cases:
        print(f"\n{case_name} ({quantity:,} shares):")
        
        for model in models:
            estimate = manager.estimate_slippage(
                symbol="AAPL",
                order_side=OrderSide.BUY,
                quantity=quantity,
                reference_price=150.00,
                market_data=market_data,
                model=model
            )
            
            print(f"  {model}: {estimate.total_estimated_slippage_bps:.1f} bps "
                  f"(${estimate.total_estimated_cost_dollars:.2f})")
            print(f"    Market Impact: {estimate.market_impact_bps:.1f} bps")
            print(f"    Spread Cost: {estimate.spread_cost_bps:.1f} bps")
            print(f"    Liquidity Regime: {estimate.liquidity_regime.value}")
    
    print("\n2. Testing Slippage Budget Management")
    
    # Set slippage budgets
    manager.set_slippage_budget("momentum_strategy", 15.0)  # 15 bps budget
    manager.set_slippage_budget("mean_reversion", 8.0)     # 8 bps budget
    
    # Test budget checks
    test_estimate = manager.estimate_slippage(
        symbol="AAPL",
        order_side=OrderSide.BUY,
        quantity=50000,
        reference_price=150.00,
        market_data=market_data
    )
    
    within_momentum_budget = manager.check_slippage_budget("momentum_strategy", test_estimate.total_estimated_slippage_bps)
    within_mean_rev_budget = manager.check_slippage_budget("mean_reversion", test_estimate.total_estimated_slippage_bps)
    
    print(f"Large order slippage estimate: {test_estimate.total_estimated_slippage_bps:.1f} bps")
    print(f"  Within momentum budget (15 bps): {'Yes' if within_momentum_budget else 'No'}")
    print(f"  Within mean reversion budget (8 bps): {'Yes' if within_mean_rev_budget else 'No'}")
    
    print("\n3. Testing Optimal Order Sizing")
    
    # Calculate optimal order size for slippage budget
    optimal_size = calculate_optimal_order_size(
        symbol="AAPL",
        side="BUY",
        target_quantity=100000,
        max_slippage_bps=10.0,
        market_data=market_data
    )
    
    print(f"Target quantity: 100,000 shares")
    print(f"Slippage budget: 10.0 bps")
    print(f"Optimal order size: {optimal_size:,} shares")
    
    # Verify the optimal size
    optimal_estimate = manager.estimate_slippage(
        symbol="AAPL",
        order_side=OrderSide.BUY,
        quantity=optimal_size,
        reference_price=150.00,
        market_data=market_data
    )
    
    print(f"Estimated slippage for optimal size: {optimal_estimate.total_estimated_slippage_bps:.1f} bps")
    
    print("\n4. Testing Post-Trade Analysis")
    
    # Create mock market order and fills
    from .market_orders import MarketOrder, ExecutionUrgency
    
    mock_order = MarketOrder(
        order_id="test_order_123",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=5000,
        urgency=ExecutionUrgency.NORMAL
    )
    mock_order.market_price_at_creation = 150.00
    mock_order.start_execution_time = datetime.now() - timedelta(minutes=2)
    
    # Create mock fills
    from .limit_orders import Fill
    
    mock_fills = [
        Fill(
            fill_id="fill_1",
            order_id="test_order_123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=2000,
            price=150.05,
            timestamp=datetime.now() - timedelta(minutes=1, seconds=30),
            liquidity_flag="Taker"
        ),
        Fill(
            fill_id="fill_2",
            order_id="test_order_123",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=3000,
            price=150.08,
            timestamp=datetime.now() - timedelta(seconds=30),
            liquidity_flag="Taker"
        )
    ]
    
    # Analyze execution
    analysis = manager.analyze_execution(mock_order, mock_fills, market_data)
    
    print(f"Post-Trade Slippage Analysis:")
    print(f"  Realized Slippage: {analysis.realized_slippage_bps:.1f} bps")
    print(f"  Dollar Cost: ${analysis.realized_slippage_dollars:.2f}")
    print(f"  Execution Time: {analysis.execution_duration_seconds:.1f} seconds")
    print(f"  Average Fill Price: ${analysis.average_fill_price:.2f}")
    
    if analysis.slippage_attribution:
        print(f"  Slippage Attribution:")
        for slip_type, amount in analysis.slippage_attribution.items():
            print(f"    {slip_type.value}: {amount:.1f} bps")
    
    print("\n5. Testing Utility Functions")
    
    # Quick slippage estimate
    quick_estimate = estimate_execution_slippage(
        symbol="TSLA",
        side="SELL",
        quantity=5000,
        reference_price=800.00,
        market_data={
            'average_volume': 20000000,
            'volatility': 0.45,
            'spread': 0.05,
            'volume_ratio': 1.2
        }
    )
    
    print(f"Quick estimate for TSLA:")
    print(f"  Slippage: {quick_estimate.total_estimated_slippage_bps:.1f} bps")
    print(f"  Cost: ${quick_estimate.total_estimated_cost_dollars:.2f}")
    print(f"  Confidence Interval: [{quick_estimate.best_case_slippage_bps:.1f}, {quick_estimate.worst_case_slippage_bps:.1f}] bps")
    
    print("\n6. Testing Model Performance")
    
    # Add some mock execution history for performance testing
    mock_executions = []
    for i in range(10):
        mock_analysis = SlippageAnalysis(
            symbol="AAPL",
            order_side=OrderSide.BUY,
            quantity=1000 + i * 1000,
            reference_price=150.0,
            average_fill_price=150.0 + np.random.normal(0.02, 0.01),
            avg_volatility_during_execution=0.25,
            avg_volume_during_execution=8000000
        )
        mock_executions.append(mock_analysis)
    
    manager.execution_history.extend(mock_executions)
    
    # Calibrate models
    manager.calibrate_models("AAPL")
    
    # Get performance metrics
    performance = manager.get_model_performance()
    
    print(f"Model Performance Metrics:")
    for model_name, metrics in performance.items():
        if metrics:
            print(f"  {model_name}:")
            print(f"    Mean Absolute Error: {metrics['mean_absolute_error']:.1f} bps")
            print(f"    Sample Size: {metrics['sample_size']}")
    
    print("\n7. Testing Slippage Statistics")
    
    # Get slippage statistics
    stats = manager.get_slippage_statistics(symbol="AAPL", days=30)
    
    if 'error' not in stats:
        print(f"Slippage Statistics for AAPL (30 days):")
        print(f"  Executions: {stats['execution_count']}")
        
        slippage_stats = stats['slippage_statistics']
        print(f"  Mean Slippage: {slippage_stats['mean_bps']:.1f} bps")
        print(f"  Median Slippage: {slippage_stats['median_bps']:.1f} bps")
        print(f"  95th Percentile: {slippage_stats['p95_bps']:.1f} bps")
        
        cost_stats = stats['cost_statistics']
        print(f"  Total Cost: ${cost_stats['total_dollars']:.2f}")
        print(f"  Average Cost: ${cost_stats['mean_dollars']:.2f}")
    
    print("\nSlippage system testing completed successfully!")
    print("\nImplemented features include:")
    print("• Multiple slippage estimation models (Fixed BP, Square Root, ML, Abdi-Ranaldo)")
    print("• Pre-trade slippage estimation with confidence intervals")
    print("• Post-trade slippage analysis and attribution")
    print("• Model calibration using historical execution data")
    print("• Slippage budget management and monitoring")
    print("• Optimal order sizing for slippage constraints")
    print("• Comprehensive slippage statistics and reporting")
    print("• Liquidity regime detection and adjustments")
    print("• Machine learning-based adaptive estimation")
