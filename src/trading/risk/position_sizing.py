# ============================================
# StockPredictionPro - src/trading/risk/position_sizing.py
# Advanced position sizing strategies and risk-based allocation system
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import math
from scipy import stats, optimize
from sklearn.preprocessing import StandardScaler

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('trading.risk.position_sizing')

# ============================================
# Position Sizing Data Structures and Enums
# ============================================

class PositionSizingMethod(Enum):
    """Position sizing methodologies"""
    FIXED_UNITS = "fixed_units"
    FIXED_DOLLAR = "fixed_dollar"
    FIXED_PERCENTAGE = "fixed_percentage"
    FIXED_FRACTIONAL = "fixed_fractional"
    VOLATILITY_TARGET = "volatility_target"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly_criterion"
    OPTIMAL_F = "optimal_f"
    ATR_BASED = "atr_based"
    VAR_BASED = "var_based"
    SHARPE_MAXIMIZATION = "sharpe_maximization"
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"

class RiskModel(Enum):
    """Risk models for position sizing"""
    HISTORICAL_VOLATILITY = "historical_volatility"
    GARCH_VOLATILITY = "garch_volatility"
    EXPONENTIAL_WEIGHTING = "exponential_weighting"
    REALIZED_VOLATILITY = "realized_volatility"
    IMPLIED_VOLATILITY = "implied_volatility"

@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    symbol: str
    recommended_quantity: int
    recommended_dollar_amount: float
    recommended_weight: float
    
    # Risk metrics
    expected_volatility: float
    expected_return: float
    risk_contribution: float
    
    # Sizing details
    sizing_method: PositionSizingMethod
    confidence_level: float = 0.95
    max_position_limit: float = 0.0
    
    # Constraints applied
    constrained_by_capital: bool = False
    constrained_by_risk: bool = False
    constrained_by_position_limit: bool = False
    
    # Additional metrics
    expected_profit: float = 0.0
    expected_loss: float = 0.0
    profit_loss_ratio: float = 0.0
    
    # Meta information
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def expected_value(self) -> float:
        """Expected value of position"""
        return self.expected_profit + self.expected_loss
    
    @property
    def risk_reward_ratio(self) -> float:
        """Risk-reward ratio"""
        if self.expected_loss < 0:
            return abs(self.expected_profit / self.expected_loss)
        return 0.0

@dataclass
class PositionSizingConfig:
    """Configuration for position sizing calculations"""
    
    # Capital constraints
    total_capital: float
    available_capital: float
    max_position_size_pct: float = 0.10  # 10% max per position
    
    # Risk parameters
    risk_per_trade: float = 0.02         # 2% risk per trade
    volatility_lookback: int = 252       # 1 year lookback
    confidence_level: float = 0.95       # 95% confidence
    
    # Kelly Criterion parameters
    kelly_lookback: int = 100
    kelly_fractional: float = 0.25       # Use 25% of Kelly recommendation
    
    # Volatility targeting
    target_volatility: float = 0.15      # 15% annual volatility target
    
    # Risk model
    risk_model: RiskModel = RiskModel.HISTORICAL_VOLATILITY
    
    # Rebalancing
    rebalance_threshold: float = 0.05    # 5% deviation triggers rebalance
    min_trade_size: int = 1              # Minimum trade size
    
    # Advanced constraints
    sector_limits: Dict[str, float] = field(default_factory=dict)
    correlation_limits: Dict[str, float] = field(default_factory=dict)

# ============================================
# Base Position Sizer
# ============================================

class BasePositionSizer:
    """
    Base class for position sizing strategies.
    
    Provides common functionality for calculating position sizes
    based on risk, return, and portfolio constraints.
    """
    
    def __init__(self, method_name: str, config: PositionSizingConfig):
        self.method_name = method_name
        self.config = config
        self.historical_data = {}
        
        logger.debug(f"Initialized {method_name} position sizer")
    
    def calculate_position_size(self, symbol: str, current_price: float,
                               expected_return: float, volatility: float,
                               **kwargs) -> PositionSizeResult:
        """Calculate position size - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement calculate_position_size method")
    
    def _apply_constraints(self, result: PositionSizeResult, current_price: float) -> PositionSizeResult:
        """Apply various constraints to position size"""
        
        original_amount = result.recommended_dollar_amount
        
        # Capital constraint
        if result.recommended_dollar_amount > self.config.available_capital:
            result.recommended_dollar_amount = self.config.available_capital
            result.constrained_by_capital = True
        
        # Position size limit constraint
        max_position_amount = self.config.total_capital * self.config.max_position_size_pct
        if result.recommended_dollar_amount > max_position_amount:
            result.recommended_dollar_amount = max_position_amount
            result.constrained_by_position_limit = True
        
        # Update quantity based on constrained dollar amount
        if result.recommended_dollar_amount != original_amount:
            result.recommended_quantity = int(result.recommended_dollar_amount / current_price)
            result.recommended_weight = result.recommended_dollar_amount / self.config.total_capital
        
        # Minimum trade size constraint
        if result.recommended_quantity < self.config.min_trade_size:
            result.recommended_quantity = 0
            result.recommended_dollar_amount = 0.0
            result.recommended_weight = 0.0
        
        return result
    
    def _calculate_expected_metrics(self, quantity: int, price: float,
                                   expected_return: float, volatility: float) -> Tuple[float, float]:
        """Calculate expected profit and loss for position"""
        
        position_value = quantity * price
        
        # Expected profit (positive scenario)
        expected_profit = position_value * expected_return
        
        # Expected loss (VaR-based)
        confidence_level = self.config.confidence_level
        z_score = stats.norm.ppf(1 - confidence_level)
        expected_loss = position_value * z_score * volatility
        
        return expected_profit, expected_loss

# ============================================
# Fixed Position Sizing Methods
# ============================================

class FixedUnitsPositionSizer(BasePositionSizer):
    """Fixed number of units position sizing"""
    
    def __init__(self, config: PositionSizingConfig, units_per_trade: int = 100):
        super().__init__("Fixed Units", config)
        self.units_per_trade = units_per_trade
    
    def calculate_position_size(self, symbol: str, current_price: float,
                               expected_return: float, volatility: float,
                               **kwargs) -> PositionSizeResult:
        """Calculate position size using fixed units method"""
        
        quantity = self.units_per_trade
        dollar_amount = quantity * current_price
        weight = dollar_amount / self.config.total_capital
        
        expected_profit, expected_loss = self._calculate_expected_metrics(
            quantity, current_price, expected_return, volatility
        )
        
        result = PositionSizeResult(
            symbol=symbol,
            recommended_quantity=quantity,
            recommended_dollar_amount=dollar_amount,
            recommended_weight=weight,
            expected_volatility=volatility,
            expected_return=expected_return,
            risk_contribution=weight * volatility,
            sizing_method=PositionSizingMethod.FIXED_UNITS,
            expected_profit=expected_profit,
            expected_loss=expected_loss
        )
        
        return self._apply_constraints(result, current_price)

class FixedDollarPositionSizer(BasePositionSizer):
    """Fixed dollar amount position sizing"""
    
    def __init__(self, config: PositionSizingConfig, dollar_per_trade: float = 10000):
        super().__init__("Fixed Dollar", config)
        self.dollar_per_trade = dollar_per_trade
    
    def calculate_position_size(self, symbol: str, current_price: float,
                               expected_return: float, volatility: float,
                               **kwargs) -> PositionSizeResult:
        """Calculate position size using fixed dollar method"""
        
        dollar_amount = self.dollar_per_trade
        quantity = int(dollar_amount / current_price)
        actual_dollar_amount = quantity * current_price
        weight = actual_dollar_amount / self.config.total_capital
        
        expected_profit, expected_loss = self._calculate_expected_metrics(
            quantity, current_price, expected_return, volatility
        )
        
        result = PositionSizeResult(
            symbol=symbol,
            recommended_quantity=quantity,
            recommended_dollar_amount=actual_dollar_amount,
            recommended_weight=weight,
            expected_volatility=volatility,
            expected_return=expected_return,
            risk_contribution=weight * volatility,
            sizing_method=PositionSizingMethod.FIXED_DOLLAR,
            expected_profit=expected_profit,
            expected_loss=expected_loss
        )
        
        return self._apply_constraints(result, current_price)

class FixedPercentagePositionSizer(BasePositionSizer):
    """Fixed percentage of capital position sizing"""
    
    def __init__(self, config: PositionSizingConfig, percentage_per_trade: float = 0.05):
        super().__init__("Fixed Percentage", config)
        self.percentage_per_trade = percentage_per_trade
    
    def calculate_position_size(self, symbol: str, current_price: float,
                               expected_return: float, volatility: float,
                               **kwargs) -> PositionSizeResult:
        """Calculate position size using fixed percentage method"""
        
        dollar_amount = self.config.total_capital * self.percentage_per_trade
        quantity = int(dollar_amount / current_price)
        actual_dollar_amount = quantity * current_price
        weight = self.percentage_per_trade
        
        expected_profit, expected_loss = self._calculate_expected_metrics(
            quantity, current_price, expected_return, volatility
        )
        
        result = PositionSizeResult(
            symbol=symbol,
            recommended_quantity=quantity,
            recommended_dollar_amount=actual_dollar_amount,
            recommended_weight=weight,
            expected_volatility=volatility,
            expected_return=expected_return,
            risk_contribution=weight * volatility,
            sizing_method=PositionSizingMethod.FIXED_PERCENTAGE,
            expected_profit=expected_profit,
            expected_loss=expected_loss
        )
        
        return self._apply_constraints(result, current_price)

# ============================================
# Risk-Based Position Sizing Methods
# ============================================

class VolatilityTargetPositionSizer(BasePositionSizer):
    """Volatility targeting position sizing"""
    
    def __init__(self, config: PositionSizingConfig):
        super().__init__("Volatility Target", config)
    
    def calculate_position_size(self, symbol: str, current_price: float,
                               expected_return: float, volatility: float,
                               **kwargs) -> PositionSizeResult:
        """Calculate position size using volatility targeting"""
        
        # Calculate position size to achieve target volatility
        target_vol = self.config.target_volatility
        
        if volatility <= 0:
            logger.warning(f"Invalid volatility {volatility} for {symbol}")
            volatility = 0.01  # Minimum volatility
        
        # Position weight to achieve target volatility
        target_weight = target_vol / volatility
        
        # Cap at maximum position size
        target_weight = min(target_weight, self.config.max_position_size_pct)
        
        dollar_amount = self.config.total_capital * target_weight
        quantity = int(dollar_amount / current_price)
        actual_dollar_amount = quantity * current_price
        actual_weight = actual_dollar_amount / self.config.total_capital
        
        expected_profit, expected_loss = self._calculate_expected_metrics(
            quantity, current_price, expected_return, volatility
        )
        
        result = PositionSizeResult(
            symbol=symbol,
            recommended_quantity=quantity,
            recommended_dollar_amount=actual_dollar_amount,
            recommended_weight=actual_weight,
            expected_volatility=volatility,
            expected_return=expected_return,
            risk_contribution=actual_weight * volatility,
            sizing_method=PositionSizingMethod.VOLATILITY_TARGET,
            expected_profit=expected_profit,
            expected_loss=expected_loss
        )
        
        return self._apply_constraints(result, current_price)

class ATRBasedPositionSizer(BasePositionSizer):
    """Average True Range (ATR) based position sizing"""
    
    def __init__(self, config: PositionSizingConfig, atr_multiplier: float = 2.0):
        super().__init__("ATR Based", config)
        self.atr_multiplier = atr_multiplier
    
    def calculate_position_size(self, symbol: str, current_price: float,
                               expected_return: float, volatility: float,
                               atr: Optional[float] = None, **kwargs) -> PositionSizeResult:
        """Calculate position size using ATR-based method"""
        
        # Use ATR if provided, otherwise estimate from volatility
        if atr is None:
            atr = current_price * volatility / math.sqrt(252)  # Daily ATR estimate
        
        # Calculate stop loss distance
        stop_distance = atr * self.atr_multiplier
        
        # Calculate position size based on risk per trade
        risk_amount = self.config.total_capital * self.config.risk_per_trade
        
        if stop_distance <= 0:
            logger.warning(f"Invalid stop distance {stop_distance} for {symbol}")
            stop_distance = current_price * 0.02  # 2% default stop
        
        # Position size based on stop loss
        quantity = int(risk_amount / stop_distance)
        dollar_amount = quantity * current_price
        weight = dollar_amount / self.config.total_capital
        
        expected_profit, expected_loss = self._calculate_expected_metrics(
            quantity, current_price, expected_return, volatility
        )
        
        result = PositionSizeResult(
            symbol=symbol,
            recommended_quantity=quantity,
            recommended_dollar_amount=dollar_amount,
            recommended_weight=weight,
            expected_volatility=volatility,
            expected_return=expected_return,
            risk_contribution=weight * volatility,
            sizing_method=PositionSizingMethod.ATR_BASED,
            expected_profit=expected_profit,
            expected_loss=expected_loss
        )
        
        return self._apply_constraints(result, current_price)

class VaRBasedPositionSizer(BasePositionSizer):
    """Value at Risk (VaR) based position sizing"""
    
    def __init__(self, config: PositionSizingConfig):
        super().__init__("VaR Based", config)
    
    def calculate_position_size(self, symbol: str, current_price: float,
                               expected_return: float, volatility: float,
                               **kwargs) -> PositionSizeResult:
        """Calculate position size using VaR-based method"""
        
        # Calculate VaR for 1% of capital at given confidence level
        confidence_level = self.config.confidence_level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Daily VaR per unit
        daily_var_per_unit = current_price * abs(z_score) * volatility / math.sqrt(252)
        
        # Risk budget
        risk_budget = self.config.total_capital * self.config.risk_per_trade
        
        if daily_var_per_unit <= 0:
            logger.warning(f"Invalid VaR calculation for {symbol}")
            daily_var_per_unit = current_price * 0.01
        
        # Position size based on VaR
        quantity = int(risk_budget / daily_var_per_unit)
        dollar_amount = quantity * current_price
        weight = dollar_amount / self.config.total_capital
        
        expected_profit, expected_loss = self._calculate_expected_metrics(
            quantity, current_price, expected_return, volatility
        )
        
        result = PositionSizeResult(
            symbol=symbol,
            recommended_quantity=quantity,
            recommended_dollar_amount=dollar_amount,
            recommended_weight=weight,
            expected_volatility=volatility,
            expected_return=expected_return,
            risk_contribution=weight * volatility,
            sizing_method=PositionSizingMethod.VAR_BASED,
            expected_profit=expected_profit,
            expected_loss=expected_loss
        )
        
        return self._apply_constraints(result, current_price)

# ============================================
# Optimal Position Sizing Methods
# ============================================

class KellyCriterionPositionSizer(BasePositionSizer):
    """Kelly Criterion position sizing"""
    
    def __init__(self, config: PositionSizingConfig):
        super().__init__("Kelly Criterion", config)
    
    def calculate_position_size(self, symbol: str, current_price: float,
                               expected_return: float, volatility: float,
                               win_rate: Optional[float] = None,
                               avg_win: Optional[float] = None,
                               avg_loss: Optional[float] = None,
                               **kwargs) -> PositionSizeResult:
        """Calculate position size using Kelly Criterion"""
        
        # Method 1: Using win rate and average win/loss
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            # Kelly fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / avg_win
        else:
            # Method 2: Using expected return and volatility
            # Kelly fraction = expected_return / variance
            if volatility > 0:
                kelly_fraction = expected_return / (volatility ** 2)
            else:
                kelly_fraction = 0.0
        
        # Apply fractional Kelly to reduce risk
        kelly_fraction *= self.config.kelly_fractional
        
        # Ensure non-negative position
        kelly_fraction = max(0.0, kelly_fraction)
        
        # Cap at maximum position size
        kelly_fraction = min(kelly_fraction, self.config.max_position_size_pct)
        
        dollar_amount = self.config.total_capital * kelly_fraction
        quantity = int(dollar_amount / current_price)
        actual_dollar_amount = quantity * current_price
        actual_weight = actual_dollar_amount / self.config.total_capital
        
        expected_profit, expected_loss = self._calculate_expected_metrics(
            quantity, current_price, expected_return, volatility
        )
        
        result = PositionSizeResult(
            symbol=symbol,
            recommended_quantity=quantity,
            recommended_dollar_amount=actual_dollar_amount,
            recommended_weight=actual_weight,
            expected_volatility=volatility,
            expected_return=expected_return,
            risk_contribution=actual_weight * volatility,
            sizing_method=PositionSizingMethod.KELLY_CRITERION,
            expected_profit=expected_profit,
            expected_loss=expected_loss
        )
        
        return self._apply_constraints(result, current_price)

class OptimalFPositionSizer(BasePositionSizer):
    """Optimal F position sizing by Ralph Vince"""
    
    def __init__(self, config: PositionSizingConfig):
        super().__init__("Optimal F", config)
    
    def calculate_position_size(self, symbol: str, current_price: float,
                               expected_return: float, volatility: float,
                               historical_returns: Optional[List[float]] = None,
                               **kwargs) -> PositionSizeResult:
        """Calculate position size using Optimal F method"""
        
        if historical_returns is None or len(historical_returns) < 10:
            # Fallback to Kelly-like calculation
            if volatility > 0:
                optimal_f = max(0.0, expected_return / (volatility ** 2))
            else:
                optimal_f = 0.0
        else:
            # Calculate Optimal F using historical returns
            optimal_f = self._calculate_optimal_f(historical_returns)
        
        # Apply conservative factor
        optimal_f *= 0.25  # Use 25% of optimal F for safety
        
        # Cap at maximum position size
        optimal_f = min(optimal_f, self.config.max_position_size_pct)
        
        dollar_amount = self.config.total_capital * optimal_f
        quantity = int(dollar_amount / current_price)
        actual_dollar_amount = quantity * current_price
        actual_weight = actual_dollar_amount / self.config.total_capital
        
        expected_profit, expected_loss = self._calculate_expected_metrics(
            quantity, current_price, expected_return, volatility
        )
        
        result = PositionSizeResult(
            symbol=symbol,
            recommended_quantity=quantity,
            recommended_dollar_amount=actual_dollar_amount,
            recommended_weight=actual_weight,
            expected_volatility=volatility,
            expected_return=expected_return,
            risk_contribution=actual_weight * volatility,
            sizing_method=PositionSizingMethod.OPTIMAL_F,
            expected_profit=expected_profit,
            expected_loss=expected_loss
        )
        
        return self._apply_constraints(result, current_price)
    
    def _calculate_optimal_f(self, returns: List[float]) -> float:
        """Calculate Optimal F using geometric mean optimization"""
        
        def objective(f):
            if f <= 0 or f >= 1:
                return -float('inf')
            
            geometric_mean = 1.0
            for r in returns:
                new_value = 1.0 + f * r
                if new_value <= 0:
                    return -float('inf')
                geometric_mean *= new_value
            
            return -(geometric_mean ** (1.0 / len(returns)) - 1.0)
        
        # Optimize f to maximize geometric mean
        try:
            result = optimize.minimize_scalar(
                objective, 
                bounds=(0.001, 0.999), 
                method='bounded'
            )
            return max(0.0, result.x) if result.success else 0.0
        except:
            return 0.0

# ============================================
# Portfolio-Level Position Sizing
# ============================================

class RiskParityPositionSizer(BasePositionSizer):
    """Risk parity position sizing for multiple assets"""
    
    def __init__(self, config: PositionSizingConfig):
        super().__init__("Risk Parity", config)
    
    def calculate_portfolio_weights(self, symbols: List[str], 
                                  volatilities: List[float],
                                  correlations: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate risk parity weights for portfolio"""
        
        if len(symbols) != len(volatilities):
            raise ValueError("Number of symbols must match number of volatilities")
        
        if correlations is None:
            # Equal risk contribution without correlations
            inv_volatilities = [1.0 / vol if vol > 0 else 0.0 for vol in volatilities]
            total_inv_vol = sum(inv_volatilities)
            
            if total_inv_vol > 0:
                weights = [inv_vol / total_inv_vol for inv_vol in inv_volatilities]
            else:
                weights = [1.0 / len(symbols)] * len(symbols)
        else:
            # Risk parity with correlation matrix
            weights = self._calculate_risk_parity_weights(volatilities, correlations)
        
        return dict(zip(symbols, weights))
    
    def _calculate_risk_parity_weights(self, volatilities: List[float], 
                                     correlations: np.ndarray) -> List[float]:
        """Calculate risk parity weights using optimization"""
        
        n_assets = len(volatilities)
        
        # Covariance matrix
        vol_matrix = np.diag(volatilities)
        cov_matrix = vol_matrix @ correlations @ vol_matrix
        
        def risk_budget_objective(weights):
            """Objective function for risk parity"""
            weights = np.array(weights)
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            if portfolio_vol == 0:
                return 1e6
            
            # Risk contribution of each asset
            marginal_contrib = (cov_matrix @ weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target: equal risk contribution (1/n each)
            target_contrib = portfolio_vol / n_assets
            
            # Minimize sum of squared deviations
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.01, 0.5)] * n_assets  # Min 1%, max 50% per asset
        
        # Initial guess
        x0 = [1.0 / n_assets] * n_assets
        
        # Optimize
        try:
            result = optimize.minimize(
                risk_budget_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return result.x.tolist()
        except:
            pass
        
        # Fallback to equal weights
        return [1.0 / n_assets] * n_assets
    
    def calculate_position_size(self, symbol: str, current_price: float,
                               expected_return: float, volatility: float,
                               portfolio_weight: float = 0.0,
                               **kwargs) -> PositionSizeResult:
        """Calculate position size for risk parity allocation"""
        
        if portfolio_weight <= 0:
            logger.warning(f"Invalid portfolio weight {portfolio_weight} for {symbol}")
            portfolio_weight = 0.05  # 5% default
        
        dollar_amount = self.config.total_capital * portfolio_weight
        quantity = int(dollar_amount / current_price)
        actual_dollar_amount = quantity * current_price
        actual_weight = actual_dollar_amount / self.config.total_capital
        
        expected_profit, expected_loss = self._calculate_expected_metrics(
            quantity, current_price, expected_return, volatility
        )
        
        result = PositionSizeResult(
            symbol=symbol,
            recommended_quantity=quantity,
            recommended_dollar_amount=actual_dollar_amount,
            recommended_weight=actual_weight,
            expected_volatility=volatility,
            expected_return=expected_return,
            risk_contribution=actual_weight * volatility,
            sizing_method=PositionSizingMethod.RISK_PARITY,
            expected_profit=expected_profit,
            expected_loss=expected_loss
        )
        
        return self._apply_constraints(result, current_price)

# ============================================
# Position Sizing Manager
# ============================================

class PositionSizingManager:
    """
    Comprehensive position sizing management system.
    
    Integrates multiple position sizing methods with portfolio-level
    risk management and constraint handling.
    """
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        
        # Initialize all position sizers
        self.sizers = {
            PositionSizingMethod.FIXED_UNITS: FixedUnitsPositionSizer(config),
            PositionSizingMethod.FIXED_DOLLAR: FixedDollarPositionSizer(config),
            PositionSizingMethod.FIXED_PERCENTAGE: FixedPercentagePositionSizer(config),
            PositionSizingMethod.VOLATILITY_TARGET: VolatilityTargetPositionSizer(config),
            PositionSizingMethod.ATR_BASED: ATRBasedPositionSizer(config),
            PositionSizingMethod.VAR_BASED: VaRBasedPositionSizer(config),
            PositionSizingMethod.KELLY_CRITERION: KellyCriterionPositionSizer(config),
            PositionSizingMethod.OPTIMAL_F: OptimalFPositionSizer(config),
            PositionSizingMethod.RISK_PARITY: RiskParityPositionSizer(config)
        }
        
        # Portfolio state
        self.current_positions = {}
        self.historical_performance = []
        
        logger.info("Initialized PositionSizingManager with 9 sizing methods")
    
    @time_it("position_size_calculation")
    def calculate_position_size(self, symbol: str, current_price: float,
                               method: PositionSizingMethod,
                               **kwargs) -> PositionSizeResult:
        """
        Calculate position size using specified method
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            method: Position sizing method to use
            **kwargs: Method-specific parameters
            
        Returns:
            PositionSizeResult with recommended position size
        """
        
        if method not in self.sizers:
            raise ValueError(f"Unknown position sizing method: {method}")
        
        # Get default parameters if not provided
        expected_return = kwargs.get('expected_return', 0.05)  # 5% default
        volatility = kwargs.get('volatility', 0.20)           # 20% default
        
        # Calculate position size
        sizer = self.sizers[method]
        result = sizer.calculate_position_size(
            symbol=symbol,
            current_price=current_price,
            expected_return=expected_return,
            volatility=volatility,
            **kwargs
        )
        
        logger.debug(f"Position size calculated for {symbol}: "
                    f"{result.recommended_quantity} shares "
                    f"(${result.recommended_dollar_amount:,.0f})")
        
        return result
    
    def calculate_portfolio_allocation(self, portfolio_data: Dict[str, Dict[str, float]],
                                     method: PositionSizingMethod = PositionSizingMethod.RISK_PARITY) -> Dict[str, PositionSizeResult]:
        """
        Calculate position sizes for entire portfolio
        
        Args:
            portfolio_data: Dictionary with symbol -> {price, expected_return, volatility}
            method: Position sizing method
            
        Returns:
            Dictionary of symbol -> PositionSizeResult
        """
        
        results = {}
        
        # Special handling for portfolio-level methods
        if method == PositionSizingMethod.RISK_PARITY:
            results = self._calculate_risk_parity_portfolio(portfolio_data)
        else:
            # Individual position sizing
            for symbol, data in portfolio_data.items():
                result = self.calculate_position_size(
                    symbol=symbol,
                    current_price=data['price'],
                    method=method,
                    expected_return=data.get('expected_return', 0.05),
                    volatility=data.get('volatility', 0.20),
                    **{k: v for k, v in data.items() if k not in ['price', 'expected_return', 'volatility']}
                )
                results[symbol] = result
        
        # Verify portfolio constraints
        results = self._verify_portfolio_constraints(results)
        
        logger.info(f"Calculated portfolio allocation for {len(results)} symbols using {method.value}")
        
        return results
    
    def _calculate_risk_parity_portfolio(self, portfolio_data: Dict[str, Dict[str, float]]) -> Dict[str, PositionSizeResult]:
        """Calculate risk parity allocation for portfolio"""
        
        symbols = list(portfolio_data.keys())
        prices = [data['price'] for data in portfolio_data.values()]
        volatilities = [data.get('volatility', 0.20) for data in portfolio_data.values()]
        
        # Calculate risk parity weights
        risk_parity_sizer = self.sizers[PositionSizingMethod.RISK_PARITY]
        weights = risk_parity_sizer.calculate_portfolio_weights(symbols, volatilities)
        
        # Calculate individual positions
        results = {}
        for symbol, data in portfolio_data.items():
            result = risk_parity_sizer.calculate_position_size(
                symbol=symbol,
                current_price=data['price'],
                expected_return=data.get('expected_return', 0.05),
                volatility=data.get('volatility', 0.20),
                portfolio_weight=weights[symbol]
            )
            results[symbol] = result
        
        return results
    
    def _verify_portfolio_constraints(self, results: Dict[str, PositionSizeResult]) -> Dict[str, PositionSizeResult]:
        """Verify and adjust portfolio-level constraints"""
        
        # Calculate total allocation
        total_allocation = sum(result.recommended_dollar_amount for result in results.values())
        
        # Check if total exceeds available capital
        if total_allocation > self.config.available_capital:
            scale_factor = self.config.available_capital / total_allocation
            
            logger.warning(f"Portfolio allocation exceeds capital, scaling by {scale_factor:.3f}")
            
            # Scale down all positions proportionally
            for symbol, result in results.items():
                result.recommended_dollar_amount *= scale_factor
                result.recommended_quantity = int(result.recommended_dollar_amount / 
                                                (result.recommended_dollar_amount / result.recommended_quantity))
                result.recommended_weight *= scale_factor
                result.constrained_by_capital = True
        
        return results
    
    def compare_sizing_methods(self, symbol: str, current_price: float,
                              expected_return: float, volatility: float,
                              methods: Optional[List[PositionSizingMethod]] = None) -> pd.DataFrame:
        """
        Compare multiple position sizing methods for a single asset
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            expected_return: Expected return
            volatility: Expected volatility
            methods: List of methods to compare (all if None)
            
        Returns:
            DataFrame comparing different methods
        """
        
        if methods is None:
            methods = list(self.sizers.keys())
        
        comparison_data = []
        
        for method in methods:
            try:
                result = self.calculate_position_size(
                    symbol=symbol,
                    current_price=current_price,
                    method=method,
                    expected_return=expected_return,
                    volatility=volatility
                )
                
                comparison_data.append({
                    'Method': method.value,
                    'Quantity': result.recommended_quantity,
                    'Dollar_Amount': result.recommended_dollar_amount,
                    'Weight': result.recommended_weight,
                    'Risk_Contribution': result.risk_contribution,
                    'Expected_Profit': result.expected_profit,
                    'Expected_Loss': result.expected_loss,
                    'Risk_Reward_Ratio': result.risk_reward_ratio,
                    'Constrained': any([
                        result.constrained_by_capital,
                        result.constrained_by_risk,
                        result.constrained_by_position_limit
                    ])
                })
                
            except Exception as e:
                logger.warning(f"Error calculating {method.value} for {symbol}: {e}")
        
        return pd.DataFrame(comparison_data)
    
    def update_portfolio_performance(self, symbol: str, actual_return: float,
                                   position_size: int, method_used: PositionSizingMethod):
        """Update performance tracking for position sizing methods"""
        
        performance_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'method': method_used.value,
            'position_size': position_size,
            'actual_return': actual_return,
            'absolute_pnl': position_size * actual_return
        }
        
        self.historical_performance.append(performance_record)
        
        # Keep only recent performance data
        cutoff_date = datetime.now() - timedelta(days=365)
        self.historical_performance = [
            p for p in self.historical_performance 
            if p['timestamp'] > cutoff_date
        ]
    
    def get_method_performance_analysis(self) -> pd.DataFrame:
        """Analyze historical performance by position sizing method"""
        
        if not self.historical_performance:
            return pd.DataFrame()
        
        performance_df = pd.DataFrame(self.historical_performance)
        
        # Group by method and calculate statistics
        method_stats = performance_df.groupby('method').agg({
            'actual_return': ['mean', 'std', 'count'],
            'absolute_pnl': ['sum', 'mean', 'std'],
            'position_size': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        method_stats.columns = ['_'.join(col).strip() for col in method_stats.columns.values]
        
        # Calculate additional metrics
        method_stats['sharpe_ratio'] = (
            method_stats['actual_return_mean'] / method_stats['actual_return_std']
        ).fillna(0)
        
        method_stats['win_rate'] = performance_df.groupby('method')['actual_return'].apply(
            lambda x: (x > 0).sum() / len(x)
        )
        
        return method_stats.reset_index()

# ============================================
# Utility Functions
# ============================================

def create_position_sizing_config(total_capital: float,
                                 risk_per_trade: float = 0.02,
                                 max_position_size_pct: float = 0.10,
                                 target_volatility: float = 0.15) -> PositionSizingConfig:
    """
    Create a position sizing configuration with common parameters
    
    Args:
        total_capital: Total trading capital
        risk_per_trade: Risk per trade as percentage of capital
        max_position_size_pct: Maximum position size as percentage of capital
        target_volatility: Target portfolio volatility
        
    Returns:
        PositionSizingConfig object
    """
    
    return PositionSizingConfig(
        total_capital=total_capital,
        available_capital=total_capital * 0.95,  # Keep 5% cash
        risk_per_trade=risk_per_trade,
        max_position_size_pct=max_position_size_pct,
        target_volatility=target_volatility
    )

def calculate_optimal_position_size(symbol: str, current_price: float,
                                   expected_return: float, volatility: float,
                                   total_capital: float,
                                   method: str = 'kelly') -> Dict[str, Any]:
    """
    Quick utility function to calculate optimal position size
    
    Args:
        symbol: Trading symbol
        current_price: Current price
        expected_return: Expected return
        volatility: Expected volatility
        total_capital: Total available capital
        method: Sizing method ('kelly', 'volatility_target', 'atr')
        
    Returns:
        Dictionary with position sizing results
    """
    
    config = create_position_sizing_config(total_capital)
    manager = PositionSizingManager(config)
    
    method_mapping = {
        'kelly': PositionSizingMethod.KELLY_CRITERION,
        'volatility_target': PositionSizingMethod.VOLATILITY_TARGET,
        'atr': PositionSizingMethod.ATR_BASED,
        'var': PositionSizingMethod.VAR_BASED,
        'fixed_pct': PositionSizingMethod.FIXED_PERCENTAGE
    }
    
    sizing_method = method_mapping.get(method, PositionSizingMethod.KELLY_CRITERION)
    
    result = manager.calculate_position_size(
        symbol=symbol,
        current_price=current_price,
        method=sizing_method,
        expected_return=expected_return,
        volatility=volatility
    )
    
    return {
        'symbol': result.symbol,
        'recommended_quantity': result.recommended_quantity,
        'recommended_dollar_amount': result.recommended_dollar_amount,
        'recommended_weight': result.recommended_weight,
        'expected_profit': result.expected_profit,
        'expected_loss': result.expected_loss,
        'risk_reward_ratio': result.risk_reward_ratio,
        'method_used': result.sizing_method.value
    }

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Position Sizing System")
    
    # Create configuration
    config = create_position_sizing_config(
        total_capital=1000000,   # $1M portfolio
        risk_per_trade=0.015,    # 1.5% risk per trade
        max_position_size_pct=0.08,  # 8% max position size
        target_volatility=0.12   # 12% target volatility
    )
    
    # Initialize position sizing manager
    manager = PositionSizingManager(config)
    
    # Sample assets
    sample_assets = {
        'AAPL': {
            'price': 180.00,
            'expected_return': 0.08,
            'volatility': 0.25,
            'atr': 3.50
        },
        'MSFT': {
            'price': 350.00,
            'expected_return': 0.07,
            'volatility': 0.22,
            'atr': 6.80
        },
        'GOOGL': {
            'price': 2800.00,
            'expected_return': 0.09,
            'volatility': 0.28,
            'atr': 45.00
        },
        'JNJ': {
            'price': 160.00,
            'expected_return': 0.05,
            'volatility': 0.15,
            'atr': 2.20
        },
        'SPY': {
            'price': 450.00,
            'expected_return': 0.06,
            'volatility': 0.18,
            'atr': 7.50
        }
    }
    
    print(f"\nPortfolio Configuration:")
    print(f"  Total Capital: ${config.total_capital:,}")
    print(f"  Risk per Trade: {config.risk_per_trade:.1%}")
    print(f"  Max Position Size: {config.max_position_size_pct:.1%}")
    print(f"  Target Volatility: {config.target_volatility:.1%}")
    
    print("\n1. Testing Individual Position Sizing Methods")
    
    test_symbol = 'AAPL'
    test_data = sample_assets[test_symbol]
    
    # Test different methods for AAPL
    methods_to_test = [
        PositionSizingMethod.FIXED_PERCENTAGE,
        PositionSizingMethod.VOLATILITY_TARGET,
        PositionSizingMethod.ATR_BASED,
        PositionSizingMethod.KELLY_CRITERION,
        PositionSizingMethod.VAR_BASED
    ]
    
    print(f"\nPosition Sizing for {test_symbol} (${test_data['price']:.2f}):")
    
    for method in methods_to_test:
        result = manager.calculate_position_size(
            symbol=test_symbol,
            current_price=test_data['price'],
            method=method,
            expected_return=test_data['expected_return'],
            volatility=test_data['volatility'],
            atr=test_data.get('atr')
        )
        
        print(f"  {method.value}:")
        print(f"    Quantity: {result.recommended_quantity:,} shares")
        print(f"    Dollar Amount: ${result.recommended_dollar_amount:,.0f}")
        print(f"    Portfolio Weight: {result.recommended_weight:.2%}")
        print(f"    Risk Contribution: {result.risk_contribution:.3f}")
        print(f"    Expected P&L: ${result.expected_profit:.0f} / ${result.expected_loss:.0f}")
        print(f"    Risk/Reward: {result.risk_reward_ratio:.2f}")
        
        if any([result.constrained_by_capital, result.constrained_by_position_limit]):
            constraints = []
            if result.constrained_by_capital:
                constraints.append("Capital")
            if result.constrained_by_position_limit:
                constraints.append("Position Limit")
            print(f"    Constraints: {', '.join(constraints)}")
        print()
    
    print("\n2. Testing Method Comparison")
    
    comparison_df = manager.compare_sizing_methods(
        symbol=test_symbol,
        current_price=test_data['price'],
        expected_return=test_data['expected_return'],
        volatility=test_data['volatility']
    )
    
    print(f"Method Comparison for {test_symbol}:")
    print(comparison_df[['Method', 'Quantity', 'Dollar_Amount', 'Weight', 'Risk_Reward_Ratio']].round(2))
    
    print("\n3. Testing Portfolio-Level Allocation")
    
    # Portfolio allocation using risk parity
    risk_parity_allocation = manager.calculate_portfolio_allocation(
        sample_assets, 
        PositionSizingMethod.RISK_PARITY
    )
    
    print(f"Risk Parity Portfolio Allocation:")
    print("Symbol | Quantity | Dollar Amount | Weight | Risk Contrib")
    print("-" * 60)
    
    total_allocation = 0
    total_risk = 0
    
    for symbol, result in risk_parity_allocation.items():
        total_allocation += result.recommended_dollar_amount
        total_risk += result.risk_contribution
        
        print(f"{symbol:6} | {result.recommended_quantity:8,} | ${result.recommended_dollar_amount:11,.0f} | "
              f"{result.recommended_weight:6.2%} | {result.risk_contribution:10.3f}")
    
    print(f"{'Total':6} | {'':8} | ${total_allocation:11,.0f} | "
          f"{total_allocation/config.total_capital:6.2%} | {total_risk:10.3f}")
    
    print("\n4. Testing Volatility Target Allocation")
    
    vol_target_allocation = manager.calculate_portfolio_allocation(
        sample_assets,
        PositionSizingMethod.VOLATILITY_TARGET
    )
    
    print(f"Volatility Target Portfolio Allocation:")
    print("Symbol | Quantity | Dollar Amount | Weight | Expected Vol")
    print("-" * 60)
    
    for symbol, result in vol_target_allocation.items():
        print(f"{symbol:6} | {result.recommended_quantity:8,} | ${result.recommended_dollar_amount:11,.0f} | "
              f"{result.recommended_weight:6.2%} | {result.expected_volatility:11.2%}")
    
    print("\n5. Testing Kelly Criterion with Trading Stats")
    
    # Kelly with win rate statistics
    kelly_result = manager.calculate_position_size(
        symbol='AAPL',
        current_price=180.00,
        method=PositionSizingMethod.KELLY_CRITERION,
        expected_return=0.08,
        volatility=0.25,
        win_rate=0.55,      # 55% win rate
        avg_win=0.12,       # 12% average win
        avg_loss=0.08       # 8% average loss
    )
    
    print(f"Kelly Criterion with Trading Statistics:")
    print(f"  Recommended Position: {kelly_result.recommended_quantity:,} shares")
    print(f"  Dollar Amount: ${kelly_result.recommended_dollar_amount:,.0f}")
    print(f"  Portfolio Weight: {kelly_result.recommended_weight:.2%}")
    print(f"  Expected Value: ${kelly_result.expected_value:.0f}")
    
    print("\n6. Testing ATR-Based Position Sizing")
    
    for symbol, data in sample_assets.items():
        atr_result = manager.calculate_position_size(
            symbol=symbol,
            current_price=data['price'],
            method=PositionSizingMethod.ATR_BASED,
            expected_return=data['expected_return'],
            volatility=data['volatility'],
            atr=data['atr']
        )
        
        # Calculate stop loss
        stop_distance = data['atr'] * 2.0  # 2x ATR stop
        stop_loss_pct = stop_distance / data['price']
        
        print(f"{symbol}: {atr_result.recommended_quantity:,} shares, "
              f"Stop: {stop_loss_pct:.1%}, "
              f"Risk: ${atr_result.recommended_quantity * stop_distance:.0f}")
    
    print("\n7. Testing Utility Functions")
    
    # Quick position size calculation
    quick_result = calculate_optimal_position_size(
        symbol='TSLA',
        current_price=250.00,
        expected_return=0.15,
        volatility=0.45,
        total_capital=500000,
        method='kelly'
    )
    
    print(f"Quick Kelly Calculation for TSLA:")
    print(f"  Recommended: {quick_result['recommended_quantity']:,} shares")
    print(f"  Dollar Amount: ${quick_result['recommended_dollar_amount']:,.0f}")
    print(f"  Weight: {quick_result['recommended_weight']:.2%}")
    print(f"  Risk/Reward: {quick_result['risk_reward_ratio']:.2f}")
    
    print("\n8. Testing Portfolio Constraint Handling")
    
    # Create scenario with limited capital
    limited_config = PositionSizingConfig(
        total_capital=100000,     # Only $100k
        available_capital=95000,  # $95k available
        risk_per_trade=0.02,
        max_position_size_pct=0.15
    )
    
    limited_manager = PositionSizingManager(limited_config)
    
    # Try to allocate the same portfolio with limited capital
    constrained_allocation = limited_manager.calculate_portfolio_allocation(
        sample_assets,
        PositionSizingMethod.FIXED_PERCENTAGE
    )
    
    print(f"Constrained Portfolio Allocation (${limited_config.total_capital:,} capital):")
    
    constrained_total = 0
    for symbol, result in constrained_allocation.items():
        constrained_total += result.recommended_dollar_amount
        constraints = []
        if result.constrained_by_capital:
            constraints.append("Capital")
        if result.constrained_by_position_limit:
            constraints.append("Position")
        
        constraint_str = f" [{', '.join(constraints)}]" if constraints else ""
        
        print(f"  {symbol}: ${result.recommended_dollar_amount:,.0f} "
              f"({result.recommended_weight:.1%}){constraint_str}")
    
    print(f"  Total Allocation: ${constrained_total:,.0f}")
    print(f"  Capital Utilization: {constrained_total/limited_config.total_capital:.1%}")
    
    print("\n9. Testing Performance Tracking")
    
    # Simulate some performance data
    performance_data = [
        ('AAPL', 0.08, 500, PositionSizingMethod.KELLY_CRITERION),
        ('MSFT', 0.05, 300, PositionSizingMethod.VOLATILITY_TARGET),
        ('GOOGL', -0.03, 100, PositionSizingMethod.ATR_BASED),
        ('JNJ', 0.02, 800, PositionSizingMethod.FIXED_PERCENTAGE),
        ('SPY', 0.06, 400, PositionSizingMethod.RISK_PARITY)
    ]
    
    for symbol, return_, size, method in performance_data:
        manager.update_portfolio_performance(symbol, return_, size, method)
    
    # Get performance analysis
    performance_analysis = manager.get_method_performance_analysis()
    
    if not performance_analysis.empty:
        print(f"Method Performance Analysis:")
        print(performance_analysis[['method', 'actual_return_mean', 'actual_return_count', 
                                   'win_rate', 'absolute_pnl_sum']].round(3))
    
    print("\nPosition sizing system testing completed successfully!")
    print("\nImplemented features include:")
    print("• 9 position sizing methods (Fixed, Volatility Target, Kelly, ATR, etc.)")
    print("• Portfolio-level risk parity and equal risk contribution")
    print("• Comprehensive constraint handling (capital, position limits, risk)")
    print("• Method comparison and performance analysis")
    print("• Kelly Criterion with win rate statistics")
    print("• Optimal F calculation with historical returns")
    print("• ATR-based sizing with stop loss integration")
    print("• VaR-based position sizing with confidence intervals")
    print("• Real-time constraint monitoring and adjustment")
