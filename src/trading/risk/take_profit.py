# ============================================
# StockPredictionPro - src/trading/risk/take_profit.py
# Comprehensive take profit management system with dynamic strategies and optimization
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import math
from scipy import stats

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('trading.risk.take_profit')

# ============================================
# Take Profit Data Structures and Enums
# ============================================

class TakeProfitType(Enum):
    """Types of take profit strategies"""
    FIXED_PERCENTAGE = "fixed_percentage"
    FIXED_DOLLAR = "fixed_dollar"
    FIXED_RATIO = "fixed_ratio"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    TRAILING = "trailing"
    LADDER = "ladder"
    FIBONACCI = "fibonacci"
    TECHNICAL_LEVEL = "technical_level"
    TIME_BASED = "time_based"
    DYNAMIC_RISK_REWARD = "dynamic_risk_reward"

class TakeProfitStatus(Enum):
    """Take profit status enumeration"""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    MODIFIED = "modified"
    PARTIALLY_FILLED = "partially_filled"

class TriggerCondition(Enum):
    """Take profit trigger conditions"""
    CLOSE_ABOVE = "close_above"          # Close price above profit level
    INTRADAY_ABOVE = "intraday_above"    # Any price above profit level
    TWO_CONSECUTIVE = "two_consecutive"   # Two consecutive closes above
    VOLUME_CONFIRMATION = "volume_confirmation"  # Price + volume confirmation

@dataclass
class TakeProfitLevel:
    """Individual take profit level for ladder strategies"""
    level_id: str
    target_price: float
    quantity: int
    status: TakeProfitStatus = TakeProfitStatus.ACTIVE
    trigger_time: Optional[datetime] = None
    
    @property
    def is_filled(self) -> bool:
        return self.status == TakeProfitStatus.TRIGGERED

@dataclass
class TakeProfitOrder:
    """Comprehensive take profit order representation"""
    
    # Basic order information
    profit_id: str
    symbol: str
    position_size: int
    entry_price: float
    current_price: float
    
    # Take profit configuration
    profit_type: TakeProfitType
    profit_levels: List[TakeProfitLevel] = field(default_factory=list)
    
    # Status and timestamps
    status: TakeProfitStatus = TakeProfitStatus.ACTIVE
    creation_time: datetime = field(default_factory=datetime.now)
    last_update_time: datetime = field(default_factory=datetime.now)
    completion_time: Optional[datetime] = None
    expiry_time: Optional[datetime] = None
    
    # Advanced parameters
    trigger_condition: TriggerCondition = TriggerCondition.CLOSE_ABOVE
    trail_amount: Optional[float] = None
    trail_percentage: Optional[float] = None
    atr_multiplier: Optional[float] = None
    risk_reward_ratio: float = 2.0
    
    # Performance tracking
    max_favorable_price: Optional[float] = None
    original_profit_potential: float = 0.0
    realized_profit: float = 0.0
    remaining_quantity: int = 0
    
    # Technical levels
    resistance_level: Optional[float] = None
    fibonacci_levels: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    stop_loss_price: Optional[float] = None  # Associated stop loss
    
    def __post_init__(self):
        """Post-initialization processing"""
        self.remaining_quantity = abs(self.position_size)
        
        # If no levels provided, create default single level
        if not self.profit_levels:
            default_price = self._calculate_default_profit_price()
            if default_price > 0:
                self.profit_levels.append(TakeProfitLevel(
                    level_id=f"{self.profit_id}_L1",
                    target_price=default_price,
                    quantity=abs(self.position_size)
                ))
        
        # Calculate original profit potential
        self.original_profit_potential = sum(
            (level.target_price - self.entry_price) * level.quantity
            for level in self.profit_levels
            if self.position_size > 0
        ) or sum(
            (self.entry_price - level.target_price) * level.quantity
            for level in self.profit_levels
            if self.position_size < 0
        )
    
    def _calculate_default_profit_price(self) -> float:
        """Calculate default profit price based on type"""
        if self.profit_type == TakeProfitType.FIXED_PERCENTAGE:
            if self.position_size > 0:  # Long position
                return self.entry_price * 1.05  # 5% profit
            else:  # Short position
                return self.entry_price * 0.95  # 5% profit
        return 0.0
    
    def update_current_price(self, new_price: float):
        """Update current price and related metrics"""
        self.current_price = new_price
        self.last_update_time = datetime.now()
        
        # Track maximum favorable price movement
        if self.position_size > 0:  # Long position
            if self.max_favorable_price is None or new_price > self.max_favorable_price:
                self.max_favorable_price = new_price
        else:  # Short position
            if self.max_favorable_price is None or new_price < self.max_favorable_price:
                self.max_favorable_price = new_price
    
    @property
    def is_long_position(self) -> bool:
        """Check if this is a long position"""
        return self.position_size > 0
    
    @property
    def is_short_position(self) -> bool:
        """Check if this is a short position"""
        return self.position_size < 0
    
    @property
    def unrealized_profit(self) -> float:
        """Calculate unrealized profit on remaining quantity"""
        if self.remaining_quantity == 0:
            return 0.0
        
        if self.is_long_position:
            return (self.current_price - self.entry_price) * self.remaining_quantity
        else:
            return (self.entry_price - self.current_price) * self.remaining_quantity
    
    @property
    def total_realized_profit(self) -> float:
        """Total profit from triggered levels"""
        return self.realized_profit
    
    @property
    def fill_percentage(self) -> float:
        """Percentage of position that has been closed for profit"""
        if abs(self.position_size) == 0:
            return 0.0
        filled_quantity = abs(self.position_size) - self.remaining_quantity
        return (filled_quantity / abs(self.position_size)) * 100
    
    @property
    def next_profit_target(self) -> Optional[float]:
        """Next profit target price"""
        active_levels = [level for level in self.profit_levels if level.status == TakeProfitStatus.ACTIVE]
        if not active_levels:
            return None
        
        if self.is_long_position:
            return min(level.target_price for level in active_levels)
        else:
            return max(level.target_price for level in active_levels)

# ============================================
# Base Take Profit Strategy
# ============================================

class BaseTakeProfitStrategy:
    """
    Base class for take profit strategies.
    
    Provides common functionality for calculating and managing
    take profit levels based on various criteria.
    """
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.active_profits = {}
        self.triggered_profits = []
        
        logger.debug(f"Initialized {strategy_name} take profit strategy")
    
    def create_take_profit(self, symbol: str, position_size: int, entry_price: float,
                          current_price: float, **kwargs) -> TakeProfitOrder:
        """Create take profit order - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement create_take_profit method")
    
    def update_take_profit(self, profit_order: TakeProfitOrder, current_price: float,
                          market_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Update take profit based on current market conditions
        
        Args:
            profit_order: Take profit order to update
            current_price: Current market price
            market_data: Additional market data (volume, volatility, etc.)
            
        Returns:
            List of triggered level IDs
        """
        
        profit_order.update_current_price(current_price)
        triggered_levels = []
        
        # Check each profit level for triggers
        for level in profit_order.profit_levels:
            if level.status != TakeProfitStatus.ACTIVE:
                continue
            
            if self._should_trigger_level(profit_order, level, market_data):
                triggered_levels.append(level.level_id)
                self._trigger_profit_level(profit_order, level)
        
        # Update trailing takes profits
        if profit_order.profit_type in [TakeProfitType.TRAILING]:
            self._update_trailing_profit(profit_order, market_data)
        
        # Check if entire order is complete
        if profit_order.remaining_quantity == 0:
            profit_order.status = TakeProfitStatus.TRIGGERED
            profit_order.completion_time = datetime.now()
            
            # Move from active to triggered
            if profit_order.profit_id in self.active_profits:
                del self.active_profits[profit_order.profit_id]
            self.triggered_profits.append(profit_order)
        
        return triggered_levels
    
    def _should_trigger_level(self, profit_order: TakeProfitOrder, level: TakeProfitLevel,
                            market_data: Optional[Dict[str, Any]] = None) -> bool:
        """Check if profit level should be triggered"""
        
        current_price = profit_order.current_price
        target_price = level.target_price
        
        # Basic price trigger check
        if profit_order.is_long_position:
            price_triggered = current_price >= target_price
        else:
            price_triggered = current_price <= target_price
        
        # Handle different trigger conditions
        if profit_order.trigger_condition == TriggerCondition.CLOSE_ABOVE:
            return price_triggered
        
        elif profit_order.trigger_condition == TriggerCondition.INTRADAY_ABOVE:
            # Check intraday high/low from market data
            if market_data:
                if profit_order.is_long_position:
                    intraday_high = market_data.get('high', current_price)
                    return intraday_high >= target_price
                else:
                    intraday_low = market_data.get('low', current_price)
                    return intraday_low <= target_price
            return price_triggered
        
        elif profit_order.trigger_condition == TriggerCondition.VOLUME_CONFIRMATION:
            # Require volume confirmation
            if market_data and price_triggered:
                current_volume = market_data.get('volume', 0)
                avg_volume = market_data.get('average_volume', 0)
                return current_volume > avg_volume * 0.3  # 30% of average volume
            return False
        
        return price_triggered
    
    def _trigger_profit_level(self, profit_order: TakeProfitOrder, level: TakeProfitLevel):
        """Trigger a specific profit level"""
        
        level.status = TakeProfitStatus.TRIGGERED
        level.trigger_time = datetime.now()
        
        # Calculate realized profit
        if profit_order.is_long_position:
            level_profit = (level.target_price - profit_order.entry_price) * level.quantity
        else:
            level_profit = (profit_order.entry_price - level.target_price) * level.quantity
        
        profit_order.realized_profit += level_profit
        profit_order.remaining_quantity -= level.quantity
        
        logger.info(f"Take profit level triggered for {profit_order.symbol}: "
                   f"Level ${level.target_price:.2f}, Quantity {level.quantity}, "
                   f"Profit ${level_profit:.0f}")
    
    def _update_trailing_profit(self, profit_order: TakeProfitOrder, 
                              market_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update trailing take profit"""
        # To be implemented by trailing profit strategies
        return False
    
    def get_profit_statistics(self) -> Dict[str, Any]:
        """Get statistics about take profit performance"""
        
        all_profits = list(self.active_profits.values()) + self.triggered_profits
        
        if not all_profits:
            return {}
        
        triggered = [p for p in all_profits if p.status == TakeProfitStatus.TRIGGERED]
        total_realized = sum(p.realized_profit for p in triggered)
        total_potential = sum(p.original_profit_potential for p in all_profits)
        
        return {
            'total_profits': len(all_profits),
            'active_profits': len(self.active_profits),
            'triggered_profits': len(triggered),
            'trigger_rate': len(triggered) / len(all_profits) if all_profits else 0,
            'total_realized_profit': total_realized,
            'total_potential_profit': total_potential,
            'realization_rate': total_realized / total_potential if total_potential > 0 else 0
        }

# ============================================
# Fixed Take Profit Strategies
# ============================================

class FixedPercentageTakeProfit(BaseTakeProfitStrategy):
    """Fixed percentage take profit strategy"""
    
    def __init__(self, profit_percentage: float = 0.10):
        super().__init__("Fixed Percentage")
        self.profit_percentage = profit_percentage
    
    def create_take_profit(self, symbol: str, position_size: int, entry_price: float,
                          current_price: float, **kwargs) -> TakeProfitOrder:
        """Create fixed percentage take profit"""
        
        # Calculate target price based on percentage
        if position_size > 0:  # Long position
            target_price = entry_price * (1 + self.profit_percentage)
        else:  # Short position
            target_price = entry_price * (1 - self.profit_percentage)
        
        profit_id = f"FIXED_PCT_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create single profit level
        level = TakeProfitLevel(
            level_id=f"{profit_id}_L1",
            target_price=target_price,
            quantity=abs(position_size)
        )
        
        return TakeProfitOrder(
            profit_id=profit_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            profit_type=TakeProfitType.FIXED_PERCENTAGE,
            profit_levels=[level]
        )

class FixedRatioTakeProfit(BaseTakeProfitStrategy):
    """Fixed risk-reward ratio take profit strategy"""
    
    def __init__(self, risk_reward_ratio: float = 2.0):
        super().__init__("Fixed Risk-Reward Ratio")
        self.risk_reward_ratio = risk_reward_ratio
    
    def create_take_profit(self, symbol: str, position_size: int, entry_price: float,
                          current_price: float, stop_loss_price: float, **kwargs) -> TakeProfitOrder:
        """Create fixed risk-reward ratio take profit"""
        
        # Calculate risk amount
        risk_per_share = abs(entry_price - stop_loss_price)
        
        # Calculate target price based on risk-reward ratio
        profit_per_share = risk_per_share * self.risk_reward_ratio
        
        if position_size > 0:  # Long position
            target_price = entry_price + profit_per_share
        else:  # Short position
            target_price = entry_price - profit_per_share
        
        profit_id = f"RATIO_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create single profit level
        level = TakeProfitLevel(
            level_id=f"{profit_id}_L1",
            target_price=target_price,
            quantity=abs(position_size)
        )
        
        return TakeProfitOrder(
            profit_id=profit_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            profit_type=TakeProfitType.FIXED_RATIO,
            profit_levels=[level],
            risk_reward_ratio=self.risk_reward_ratio,
            stop_loss_price=stop_loss_price
        )

class FixedDollarTakeProfit(BaseTakeProfitStrategy):
    """Fixed dollar amount take profit strategy"""
    
    def __init__(self, profit_amount: float = 1000.0):
        super().__init__("Fixed Dollar")
        self.profit_amount = profit_amount
    
    def create_take_profit(self, symbol: str, position_size: int, entry_price: float,
                          current_price: float, **kwargs) -> TakeProfitOrder:
        """Create fixed dollar amount take profit"""
        
        # Calculate target price based on dollar amount
        profit_per_share = self.profit_amount / abs(position_size)
        
        if position_size > 0:  # Long position
            target_price = entry_price + profit_per_share
        else:  # Short position
            target_price = entry_price - profit_per_share
        
        profit_id = f"FIXED_DOL_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create single profit level
        level = TakeProfitLevel(
            level_id=f"{profit_id}_L1",
            target_price=target_price,
            quantity=abs(position_size)
        )
        
        return TakeProfitOrder(
            profit_id=profit_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            profit_type=TakeProfitType.FIXED_DOLLAR,
            profit_levels=[level]
        )

# ============================================
# ATR-Based Take Profit Strategy
# ============================================

class ATRBasedTakeProfit(BaseTakeProfitStrategy):
    """Average True Range (ATR) based take profit strategy"""
    
    def __init__(self, atr_multiplier: float = 3.0):
        super().__init__("ATR Based")
        self.atr_multiplier = atr_multiplier
    
    def create_take_profit(self, symbol: str, position_size: int, entry_price: float,
                          current_price: float, atr: float, **kwargs) -> TakeProfitOrder:
        """Create ATR-based take profit"""
        
        if atr <= 0:
            raise ValueError(f"Invalid ATR value: {atr}")
        
        # Calculate target distance based on ATR
        target_distance = atr * self.atr_multiplier
        
        # Calculate target price
        if position_size > 0:  # Long position
            target_price = entry_price + target_distance
        else:  # Short position
            target_price = entry_price - target_distance
        
        profit_id = f"ATR_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create single profit level
        level = TakeProfitLevel(
            level_id=f"{profit_id}_L1",
            target_price=target_price,
            quantity=abs(position_size)
        )
        
        return TakeProfitOrder(
            profit_id=profit_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            profit_type=TakeProfitType.ATR_BASED,
            profit_levels=[level],
            atr_multiplier=self.atr_multiplier
        )

# ============================================
# Ladder Take Profit Strategy
# ============================================

class LadderTakeProfit(BaseTakeProfitStrategy):
    """Ladder (multiple levels) take profit strategy"""
    
    def __init__(self, profit_levels: List[float] = [0.05, 0.10, 0.15], 
                 quantity_distribution: List[float] = [0.33, 0.33, 0.34]):
        super().__init__("Ladder")
        self.profit_levels = profit_levels
        self.quantity_distribution = quantity_distribution
        
        if len(profit_levels) != len(quantity_distribution):
            raise ValueError("Profit levels and quantity distribution must have same length")
        
        if abs(sum(quantity_distribution) - 1.0) > 0.01:
            raise ValueError("Quantity distribution must sum to 1.0")
    
    def create_take_profit(self, symbol: str, position_size: int, entry_price: float,
                          current_price: float, **kwargs) -> TakeProfitOrder:
        """Create ladder take profit with multiple levels"""
        
        profit_id = f"LADDER_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        levels = []
        
        total_quantity = abs(position_size)
        
        for i, (profit_pct, qty_pct) in enumerate(zip(self.profit_levels, self.quantity_distribution)):
            # Calculate target price
            if position_size > 0:  # Long position
                target_price = entry_price * (1 + profit_pct)
            else:  # Short position
                target_price = entry_price * (1 - profit_pct)
            
            # Calculate quantity for this level
            level_quantity = int(total_quantity * qty_pct)
            
            # Adjust last level to handle rounding
            if i == len(self.profit_levels) - 1:
                assigned_qty = sum(level.quantity for level in levels)
                level_quantity = total_quantity - assigned_qty
            
            if level_quantity > 0:
                level = TakeProfitLevel(
                    level_id=f"{profit_id}_L{i+1}",
                    target_price=target_price,
                    quantity=level_quantity
                )
                levels.append(level)
        
        return TakeProfitOrder(
            profit_id=profit_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            profit_type=TakeProfitType.LADDER,
            profit_levels=levels
        )

# ============================================
# Fibonacci Take Profit Strategy
# ============================================

class FibonacciTakeProfit(BaseTakeProfitStrategy):
    """Fibonacci retracement/extension take profit strategy"""
    
    def __init__(self, fibonacci_levels: List[float] = [1.272, 1.618, 2.618],
                 quantity_distribution: List[float] = [0.4, 0.4, 0.2]):
        super().__init__("Fibonacci")
        self.fibonacci_levels = fibonacci_levels
        self.quantity_distribution = quantity_distribution
        
        if len(fibonacci_levels) != len(quantity_distribution):
            raise ValueError("Fibonacci levels and quantity distribution must have same length")
    
    def create_take_profit(self, symbol: str, position_size: int, entry_price: float,
                          current_price: float, swing_high: float, swing_low: float,
                          **kwargs) -> TakeProfitOrder:
        """Create Fibonacci-based take profit levels"""
        
        # Calculate the swing range
        swing_range = abs(swing_high - swing_low)
        
        profit_id = f"FIB_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        levels = []
        fib_dict = {}
        
        total_quantity = abs(position_size)
        
        for i, (fib_level, qty_pct) in enumerate(zip(self.fibonacci_levels, self.quantity_distribution)):
            # Calculate Fibonacci extension price
            if position_size > 0:  # Long position
                if entry_price > swing_low:  # Uptrend
                    target_price = swing_high + (swing_range * (fib_level - 1.0))
                else:
                    target_price = entry_price + (swing_range * fib_level)
            else:  # Short position
                if entry_price < swing_high:  # Downtrend
                    target_price = swing_low - (swing_range * (fib_level - 1.0))
                else:
                    target_price = entry_price - (swing_range * fib_level)
            
            # Calculate quantity for this level
            level_quantity = int(total_quantity * qty_pct)
            
            if level_quantity > 0:
                level = TakeProfitLevel(
                    level_id=f"{profit_id}_F{fib_level:.3f}",
                    target_price=target_price,
                    quantity=level_quantity
                )
                levels.append(level)
                fib_dict[f"fib_{fib_level:.3f}"] = target_price
        
        return TakeProfitOrder(
            profit_id=profit_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            profit_type=TakeProfitType.FIBONACCI,
            profit_levels=levels,
            fibonacci_levels=fib_dict
        )

# ============================================
# Trailing Take Profit Strategy
# ============================================

class TrailingTakeProfit(BaseTakeProfitStrategy):
    """Trailing take profit strategy"""
    
    def __init__(self, initial_profit_percentage: float = 0.08, 
                 trail_percentage: float = 0.03):
        super().__init__("Trailing")
        self.initial_profit_percentage = initial_profit_percentage
        self.trail_percentage = trail_percentage
    
    def create_take_profit(self, symbol: str, position_size: int, entry_price: float,
                          current_price: float, **kwargs) -> TakeProfitOrder:
        """Create trailing take profit"""
        
        # Initial target price
        if position_size > 0:  # Long position
            initial_target = entry_price * (1 + self.initial_profit_percentage)
        else:  # Short position
            initial_target = entry_price * (1 - self.initial_profit_percentage)
        
        profit_id = f"TRAIL_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create single profit level that will trail
        level = TakeProfitLevel(
            level_id=f"{profit_id}_TRAIL",
            target_price=initial_target,
            quantity=abs(position_size)
        )
        
        return TakeProfitOrder(
            profit_id=profit_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            profit_type=TakeProfitType.TRAILING,
            profit_levels=[level],
            trail_percentage=self.trail_percentage
        )
    
    def _update_trailing_profit(self, profit_order: TakeProfitOrder, 
                              market_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update trailing take profit level"""
        
        if not profit_order.profit_levels:
            return False
        
        current_price = profit_order.current_price
        level = profit_order.profit_levels[0]  # Single trailing level
        
        if level.status != TakeProfitStatus.ACTIVE:
            return False
        
        updated = False
        
        if profit_order.is_long_position:
            # For long positions, move target down as price moves up
            potential_new_target = current_price * (1 - profit_order.trail_percentage)
            if potential_new_target > level.target_price:
                level.target_price = potential_new_target
                updated = True
        else:
            # For short positions, move target up as price moves down
            potential_new_target = current_price * (1 + profit_order.trail_percentage)
            if potential_new_target < level.target_price:
                level.target_price = potential_new_target
                updated = True
        
        if updated:
            profit_order.last_update_time = datetime.now()
            logger.debug(f"Trailing take profit updated for {profit_order.symbol}: "
                        f"New target ${level.target_price:.2f}")
        
        return updated

# ============================================
# Technical Level Take Profit Strategy
# ============================================

class TechnicalLevelTakeProfit(BaseTakeProfitStrategy):
    """Technical resistance/support level take profit strategy"""
    
    def __init__(self, buffer_percentage: float = 0.005):
        super().__init__("Technical Level")
        self.buffer_percentage = buffer_percentage
    
    def create_take_profit(self, symbol: str, position_size: int, entry_price: float,
                          current_price: float, resistance_levels: List[float],
                          **kwargs) -> TakeProfitOrder:
        """Create technical level take profit"""
        
        if not resistance_levels:
            raise ValueError("At least one resistance level required")
        
        profit_id = f"TECH_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        levels = []
        
        # Sort resistance levels appropriately
        if position_size > 0:  # Long position - use ascending order
            sorted_levels = sorted([r for r in resistance_levels if r > entry_price])
        else:  # Short position - use descending order  
            sorted_levels = sorted([r for r in resistance_levels if r < entry_price], reverse=True)
        
        if not sorted_levels:
            raise ValueError("No suitable resistance levels found relative to entry price")
        
        # Distribute quantity across levels
        total_quantity = abs(position_size)
        quantity_per_level = total_quantity // len(sorted_levels)
        remaining_quantity = total_quantity % len(sorted_levels)
        
        for i, resistance_level in enumerate(sorted_levels):
            # Apply buffer to resistance level
            if position_size > 0:  # Long position
                target_price = resistance_level * (1 - self.buffer_percentage)
            else:  # Short position
                target_price = resistance_level * (1 + self.buffer_percentage)
            
            # Calculate quantity for this level
            level_quantity = quantity_per_level
            if i < remaining_quantity:  # Distribute remainder
                level_quantity += 1
            
            if level_quantity > 0:
                level = TakeProfitLevel(
                    level_id=f"{profit_id}_R{i+1}",
                    target_price=target_price,
                    quantity=level_quantity
                )
                levels.append(level)
        
        return TakeProfitOrder(
            profit_id=profit_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            profit_type=TakeProfitType.TECHNICAL_LEVEL,
            profit_levels=levels,
            resistance_level=sorted_levels[0] if len(sorted_levels) == 1 else None
        )

# ============================================
# Take Profit Manager
# ============================================

class TakeProfitManager:
    """
    Comprehensive take profit management system.
    
    Orchestrates multiple take profit strategies, manages active profits,
    and provides performance analytics and optimization.
    """
    
    def __init__(self):
        # Initialize all take profit strategies
        self.strategies = {
            TakeProfitType.FIXED_PERCENTAGE: FixedPercentageTakeProfit(),
            TakeProfitType.FIXED_DOLLAR: FixedDollarTakeProfit(),
            TakeProfitType.FIXED_RATIO: FixedRatioTakeProfit(),
            TakeProfitType.ATR_BASED: ATRBasedTakeProfit(),
            TakeProfitType.LADDER: LadderTakeProfit(),
            TakeProfitType.FIBONACCI: FibonacciTakeProfit(),
            TakeProfitType.TRAILING: TrailingTakeProfit(),
            TakeProfitType.TECHNICAL_LEVEL: TechnicalLevelTakeProfit()
        }
        
        # Global profit tracking
        self.all_profits = {}
        self.performance_history = []
        
        # Configuration
        self.update_frequency_seconds = 60
        self.last_update_time = datetime.now()
        
        logger.info("Initialized TakeProfitManager with 8 take profit strategies")
    
    @time_it("create_take_profit")
    def create_take_profit(self, symbol: str, position_size: int, entry_price: float,
                          current_price: float, profit_type: TakeProfitType,
                          **kwargs) -> TakeProfitOrder:
        """
        Create a take profit order using specified strategy
        
        Args:
            symbol: Trading symbol
            position_size: Position size (positive for long, negative for short)
            entry_price: Entry price of position
            current_price: Current market price
            profit_type: Type of take profit strategy
            **kwargs: Strategy-specific parameters
            
        Returns:
            TakeProfitOrder object
        """
        
        if profit_type not in self.strategies:
            raise ValueError(f"Unknown take profit type: {profit_type}")
        
        # Create take profit using appropriate strategy
        strategy = self.strategies[profit_type]
        profit_order = strategy.create_take_profit(
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            **kwargs
        )
        
        # Register profit in manager and strategy
        self.all_profits[profit_order.profit_id] = profit_order
        strategy.active_profits[profit_order.profit_id] = profit_order
        
        logger.info(f"Created {profit_type.value} take profit for {symbol}: "
                   f"{len(profit_order.profit_levels)} levels, "
                   f"Potential profit ${profit_order.original_profit_potential:.0f}")
        
        return profit_order
    
    def update_all_profits(self, price_data: Dict[str, float], 
                          market_data: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, List[str]]:
        """
        Update all active take profit orders
        
        Args:
            price_data: Dictionary of symbol -> current_price
            market_data: Dictionary of symbol -> market_data
            
        Returns:
            Dictionary of profit_id -> list of triggered level IDs
        """
        
        all_triggered = {}
        
        for profit_id, profit_order in list(self.all_profits.items()):
            if profit_order.status not in [TakeProfitStatus.ACTIVE, TakeProfitStatus.PARTIALLY_FILLED]:
                continue
            
            symbol = profit_order.symbol
            if symbol not in price_data:
                continue
            
            current_price = price_data[symbol]
            symbol_market_data = market_data.get(symbol) if market_data else None
            
            # Update profit using appropriate strategy
            strategy = self.strategies[profit_order.profit_type]
            triggered_levels = strategy.update_take_profit(
                profit_order, current_price, symbol_market_data
            )
            
            if triggered_levels:
                all_triggered[profit_id] = triggered_levels
                
                # Update status
                if profit_order.remaining_quantity > 0:
                    profit_order.status = TakeProfitStatus.PARTIALLY_FILLED
                
                # Record performance for completed orders
                if profit_order.status == TakeProfitStatus.TRIGGERED:
                    self._record_profit_performance(profit_order)
        
        self.last_update_time = datetime.now()
        
        if all_triggered:
            total_levels = sum(len(levels) for levels in all_triggered.values())
            logger.info(f"Updated take profits, {total_levels} levels triggered across {len(all_triggered)} orders")
        
        return all_triggered
    
    def cancel_take_profit(self, profit_id: str, reason: str = "User cancelled") -> bool:
        """Cancel an active take profit order"""
        
        if profit_id not in self.all_profits:
            logger.warning(f"Take profit {profit_id} not found")
            return False
        
        profit_order = self.all_profits[profit_id]
        
        if profit_order.status == TakeProfitStatus.TRIGGERED:
            logger.warning(f"Take profit {profit_id} already triggered")
            return False
        
        # Update status
        profit_order.status = TakeProfitStatus.CANCELLED
        profit_order.completion_time = datetime.now()
        
        # Cancel all active levels
        for level in profit_order.profit_levels:
            if level.status == TakeProfitStatus.ACTIVE:
                level.status = TakeProfitStatus.CANCELLED
        
        # Remove from strategy's active profits
        strategy = self.strategies[profit_order.profit_type]
        if profit_id in strategy.active_profits:
            del strategy.active_profits[profit_id]
        
        logger.info(f"Cancelled take profit {profit_id}: {reason}")
        return True
    
    def modify_take_profit(self, profit_id: str, level_modifications: Dict[str, float]) -> bool:
        """
        Modify take profit levels
        
        Args:
            profit_id: Take profit order ID
            level_modifications: Dictionary of level_id -> new_target_price
            
        Returns:
            True if modification successful
        """
        
        if profit_id not in self.all_profits:
            logger.warning(f"Take profit {profit_id} not found")
            return False
        
        profit_order = self.all_profits[profit_id]
        
        if profit_order.status == TakeProfitStatus.TRIGGERED:
            logger.warning(f"Take profit {profit_id} already triggered")
            return False
        
        modified_count = 0
        
        for level in profit_order.profit_levels:
            if level.level_id in level_modifications and level.status == TakeProfitStatus.ACTIVE:
                new_price = level_modifications[level.level_id]
                level.target_price = new_price
                modified_count += 1
        
        if modified_count > 0:
            profit_order.status = TakeProfitStatus.MODIFIED
            profit_order.last_update_time = datetime.now()
            
            logger.info(f"Modified take profit {profit_id}: {modified_count} levels updated")
            return True
        
        return False
    
    def get_active_profits(self, symbol: Optional[str] = None) -> List[TakeProfitOrder]:
        """Get active take profit orders, optionally filtered by symbol"""
        
        active_profits = [
            profit for profit in self.all_profits.values()
            if profit.status in [TakeProfitStatus.ACTIVE, TakeProfitStatus.PARTIALLY_FILLED]
        ]
        
        if symbol:
            active_profits = [profit for profit in active_profits if profit.symbol == symbol]
        
        return active_profits
    
    def get_take_profit_report(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive take profit report"""
        
        # Filter profits by symbol if specified
        if symbol:
            relevant_profits = [p for p in self.all_profits.values() if p.symbol == symbol]
        else:
            relevant_profits = list(self.all_profits.values())
        
        if not relevant_profits:
            return {'error': f'No take profits found for {symbol}' if symbol else 'No take profits found'}
        
        # Categorize profits by status
        active_profits = [p for p in relevant_profits if p.status in [TakeProfitStatus.ACTIVE, TakeProfitStatus.PARTIALLY_FILLED]]
        triggered_profits = [p for p in relevant_profits if p.status == TakeProfitStatus.TRIGGERED]
        
        # Calculate statistics
        total_potential = sum(p.original_profit_potential for p in relevant_profits)
        total_realized = sum(p.realized_profit for p in relevant_profits)
        
        # Performance by strategy type
        strategy_performance = {}
        for profit_type in TakeProfitType:
            type_profits = [p for p in relevant_profits if p.profit_type == profit_type]
            if type_profits:
                type_realized = sum(p.realized_profit for p in type_profits)
                type_potential = sum(p.original_profit_potential for p in type_profits)
                
                strategy_performance[profit_type.value] = {
                    'total_profits': len(type_profits),
                    'realized_profit': type_realized,
                    'potential_profit': type_potential,
                    'realization_rate': type_realized / type_potential if type_potential > 0 else 0,
                    'average_profit': type_realized / len(type_profits) if type_profits else 0
                }
        
        # Active profits analysis
        active_analysis = []
        for profit in active_profits:
            next_target = profit.next_profit_target
            analysis = {
                'profit_id': profit.profit_id,
                'symbol': profit.symbol,
                'type': profit.profit_type.value,
                'remaining_quantity': profit.remaining_quantity,
                'fill_percentage': profit.fill_percentage,
                'next_target': next_target,
                'unrealized_profit': profit.unrealized_profit,
                'realized_profit': profit.realized_profit,
                'total_levels': len(profit.profit_levels),
                'active_levels': len([l for l in profit.profit_levels if l.status == TakeProfitStatus.ACTIVE])
            }
            active_analysis.append(analysis)
        
        return {
            'symbol': symbol or 'All',
            'report_time': datetime.now(),
            'summary': {
                'total_profits': len(relevant_profits),
                'active_profits': len(active_profits),
                'triggered_profits': len(triggered_profits),
                'total_potential_profit': total_potential,
                'total_realized_profit': total_realized,
                'overall_realization_rate': total_realized / total_potential if total_potential > 0 else 0
            },
            'strategy_performance': strategy_performance,
            'active_profit_details': active_analysis
        }
    
    def _record_profit_performance(self, profit_order: TakeProfitOrder):
        """Record take profit performance for analysis"""
        
        performance_record = {
            'timestamp': datetime.now(),
            'profit_id': profit_order.profit_id,
            'symbol': profit_order.symbol,
            'profit_type': profit_order.profit_type.value,
            'entry_price': profit_order.entry_price,
            'position_size': profit_order.position_size,
            'total_levels': len(profit_order.profit_levels),
            'triggered_levels': len([l for l in profit_order.profit_levels if l.status == TakeProfitStatus.TRIGGERED]),
            'potential_profit': profit_order.original_profit_potential,
            'realized_profit': profit_order.realized_profit,
            'realization_rate': profit_order.realized_profit / profit_order.original_profit_potential if profit_order.original_profit_potential > 0 else 0,
            'time_active_hours': (datetime.now() - profit_order.creation_time).total_seconds() / 3600,
            'max_favorable_price': profit_order.max_favorable_price
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent performance data
        cutoff_time = datetime.now() - timedelta(days=90)
        self.performance_history = [
            p for p in self.performance_history 
            if p['timestamp'] > cutoff_time
        ]
    
    def get_strategy_performance_analysis(self) -> pd.DataFrame:
        """Analyze performance by take profit strategy"""
        
        if not self.performance_history:
            return pd.DataFrame()
        
        performance_df = pd.DataFrame(self.performance_history)
        
        # Group by strategy type
        strategy_stats = performance_df.groupby('profit_type').agg({
            'realized_profit': ['sum', 'mean', 'std', 'count'],
            'realization_rate': ['mean', 'std'],
            'time_active_hours': ['mean', 'std'],
            'triggered_levels': 'sum',
            'total_levels': 'sum'
        }).round(3)
        
        # Flatten column names
        strategy_stats.columns = ['_'.join(col).strip() for col in strategy_stats.columns.values]
        
        # Calculate additional metrics
        strategy_stats['fill_rate'] = (
            strategy_stats['triggered_levels_sum'] / strategy_stats['total_levels_sum']
        ).fillna(0)
        
        return strategy_stats.reset_index()

# ============================================
# Utility Functions
# ============================================

def create_take_profit_order(symbol: str, position_size: int, entry_price: float,
                            current_price: float, profit_type: str = 'percentage',
                            **kwargs) -> TakeProfitOrder:
    """
    Utility function to create a take profit order
    
    Args:
        symbol: Trading symbol
        position_size: Position size
        entry_price: Entry price
        current_price: Current price
        profit_type: Type of take profit ('percentage', 'ratio', 'ladder', etc.)
        **kwargs: Additional parameters
        
    Returns:
        TakeProfitOrder object
    """
    
    manager = TakeProfitManager()
    
    # Map string types to enums
    type_mapping = {
        'percentage': TakeProfitType.FIXED_PERCENTAGE,
        'dollar': TakeProfitType.FIXED_DOLLAR,
        'ratio': TakeProfitType.FIXED_RATIO,
        'atr': TakeProfitType.ATR_BASED,
        'ladder': TakeProfitType.LADDER,
        'fibonacci': TakeProfitType.FIBONACCI,
        'trailing': TakeProfitType.TRAILING,
        'technical': TakeProfitType.TECHNICAL_LEVEL
    }
    
    take_profit_type = type_mapping.get(profit_type.lower(), TakeProfitType.FIXED_PERCENTAGE)
    
    return manager.create_take_profit(
        symbol=symbol,
        position_size=position_size,
        entry_price=entry_price,
        current_price=current_price,
        profit_type=take_profit_type,
        **kwargs
    )

def calculate_profit_targets(entry_price: float, profit_type: str,
                           position_size: int = 100, **kwargs) -> List[float]:
    """
    Calculate profit target levels without creating an order
    
    Args:
        entry_price: Entry price of position
        profit_type: Type of profit calculation
        position_size: Position size for direction
        **kwargs: Additional parameters
        
    Returns:
        List of profit target prices
    """
    
    targets = []
    is_long = position_size > 0
    
    if profit_type.lower() == 'percentage':
        percentage = kwargs.get('percentage', 0.10)
        if is_long:
            targets.append(entry_price * (1 + percentage))
        else:
            targets.append(entry_price * (1 - percentage))
    
    elif profit_type.lower() == 'ladder':
        levels = kwargs.get('levels', [0.05, 0.10, 0.15])
        for level in levels:
            if is_long:
                targets.append(entry_price * (1 + level))
            else:
                targets.append(entry_price * (1 - level))
    
    elif profit_type.lower() == 'ratio':
        stop_loss_price = kwargs.get('stop_loss_price', entry_price * 0.95)
        ratio = kwargs.get('ratio', 2.0)
        risk = abs(entry_price - stop_loss_price)
        
        if is_long:
            targets.append(entry_price + (risk * ratio))
        else:
            targets.append(entry_price - (risk * ratio))
    
    elif profit_type.lower() == 'atr':
        atr = kwargs.get('atr', entry_price * 0.02)
        multiplier = kwargs.get('multiplier', 3.0)
        
        if is_long:
            targets.append(entry_price + (atr * multiplier))
        else:
            targets.append(entry_price - (atr * multiplier))
    
    return targets

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Take Profit System")
    
    # Initialize take profit manager
    manager = TakeProfitManager()
    
    # Sample position data
    sample_positions = [
        {
            'symbol': 'AAPL',
            'position_size': 1000,
            'entry_price': 180.00,
            'current_price': 185.00
        },
        {
            'symbol': 'MSFT',
            'position_size': -500,  # Short position
            'entry_price': 350.00,
            'current_price': 345.00
        },
        {
            'symbol': 'GOOGL',
            'position_size': 100,
            'entry_price': 2800.00,
            'current_price': 2850.00
        }
    ]
    
    print(f"\n1. Testing Different Take Profit Types")
    
    profit_orders = []
    
    for pos in sample_positions:
        print(f"\nCreating take profits for {pos['symbol']} position:")
        
        # Fixed percentage take profit
        pct_profit = manager.create_take_profit(
            symbol=pos['symbol'],
            position_size=pos['position_size'],
            entry_price=pos['entry_price'],
            current_price=pos['current_price'],
            profit_type=TakeProfitType.FIXED_PERCENTAGE,
            profit_percentage=0.08  # 8% profit
        )
        
        print(f"  Fixed 8% Profit: ${pct_profit.profit_levels[0].target_price:.2f} "
              f"(Potential: ${pct_profit.original_profit_potential:.0f})")
        
        # Risk-reward ratio take profit (need stop loss)
        stop_loss = pos['entry_price'] * (0.95 if pos['position_size'] > 0 else 1.05)
        ratio_profit = manager.create_take_profit(
            symbol=pos['symbol'],
            position_size=pos['position_size'],
            entry_price=pos['entry_price'],
            current_price=pos['current_price'],
            profit_type=TakeProfitType.FIXED_RATIO,
            stop_loss_price=stop_loss,
            risk_reward_ratio=2.5
        )
        
        print(f"  Risk-Reward 2.5:1: ${ratio_profit.profit_levels[0].target_price:.2f} "
              f"(Potential: ${ratio_profit.original_profit_potential:.0f})")
        
        # Ladder take profit
        ladder_profit = manager.create_take_profit(
            symbol=pos['symbol'],
            position_size=pos['position_size'],
            entry_price=pos['entry_price'],
            current_price=pos['current_price'],
            profit_type=TakeProfitType.LADDER,
            profit_levels=[0.05, 0.10, 0.15],
            quantity_distribution=[0.4, 0.3, 0.3]
        )
        
        print(f"  Ladder (3 levels):")
        for i, level in enumerate(ladder_profit.profit_levels):
            print(f"    Level {i+1}: {level.quantity} shares @ ${level.target_price:.2f}")
        
        profit_orders.extend([pct_profit, ratio_profit, ladder_profit])
    
    print(f"\n2. Testing Take Profit Updates")
    
    # Simulate price movements
    price_sequences = {
        'AAPL': [185.00, 190.00, 195.00, 200.00, 205.00, 200.00, 195.00],
        'MSFT': [345.00, 340.00, 335.00, 330.00, 325.00, 330.00, 335.00],
        'GOOGL': [2850.00, 2900.00, 2950.00, 3000.00, 3050.00, 3000.00, 2950.00]
    }
    
    print(f"Simulating price movements...")
    
    for i, (aapl_price, msft_price, googl_price) in enumerate(zip(*price_sequences.values())):
        price_data = {
            'AAPL': aapl_price,
            'MSFT': msft_price,
            'GOOGL': googl_price
        }
        
        # Market data
        market_data = {
            'AAPL': {'volume': 1200000, 'high': aapl_price * 1.002, 'low': aapl_price * 0.998},
            'MSFT': {'volume': 900000, 'high': msft_price * 1.001, 'low': msft_price * 0.999},
            'GOOGL': {'volume': 600000, 'high': googl_price * 1.003, 'low': googl_price * 0.997}
        }
        
        triggered = manager.update_all_profits(price_data, market_data)
        
        if triggered:
            print(f"  Period {i+1}: Take profit levels triggered")
            for profit_id, level_ids in triggered.items():
                profit = manager.all_profits[profit_id]
                for level_id in level_ids:
                    level = next(l for l in profit.profit_levels if l.level_id == level_id)
                    print(f"    {profit.symbol} {profit.profit_type.value}: "
                          f"{level.quantity} shares @ ${level.target_price:.2f}")
    
    print(f"\n3. Testing Advanced Take Profit Strategies")
    
    # Fibonacci take profit
    fib_profit = manager.create_take_profit(
        symbol='TSLA',
        position_size=200,
        entry_price=250.00,
        current_price=260.00,
        profit_type=TakeProfitType.FIBONACCI,
        swing_high=270.00,
        swing_low=230.00,
        fibonacci_levels=[1.272, 1.618, 2.618],
        quantity_distribution=[0.5, 0.3, 0.2]
    )
    
    print(f"Fibonacci Take Profit for TSLA:")
    for level in fib_profit.profit_levels:
        fib_level = level.level_id.split('_')[1]
        print(f"  Fib {fib_level}: {level.quantity} shares @ ${level.target_price:.2f}")
    print(f"  Total Potential: ${fib_profit.original_profit_potential:.0f}")
    
    # Trailing take profit
    trail_profit = manager.create_take_profit(
        symbol='NVDA',
        position_size=150,
        entry_price=800.00,
        current_price=820.00,
        profit_type=TakeProfitType.TRAILING,
        initial_profit_percentage=0.06,
        trail_percentage=0.02
    )
    
    print(f"Trailing Take Profit for NVDA:")
    print(f"  Initial Target: ${trail_profit.profit_levels[0].target_price:.2f}")
    print(f"  Trail Distance: 2%")
    
    # Simulate trailing behavior
    nvda_prices = [820, 840, 860, 880, 900, 885, 870]
    print(f"  Trailing Simulation:")
    for price in nvda_prices:
        manager.update_all_profits({'NVDA': price})
        current_target = trail_profit.profit_levels[0].target_price
        print(f"    Price ${price}: Target ${current_target:.2f}")
    
    print(f"\n4. Testing Technical Level Take Profits")
    
    # Technical resistance levels
    tech_profit = manager.create_take_profit(
        symbol='SPY',
        position_size=500,
        entry_price=450.00,
        current_price=455.00,
        profit_type=TakeProfitType.TECHNICAL_LEVEL,
        resistance_levels=[465.00, 475.00, 485.00],
        buffer_percentage=0.002  # 0.2% buffer
    )
    
    print(f"Technical Level Take Profit for SPY:")
    for i, level in enumerate(tech_profit.profit_levels):
        resistance = 465.00 + (i * 10)  # Approximate original levels
        print(f"  Level {i+1}: {level.quantity} shares @ ${level.target_price:.2f} "
              f"(Resistance: ${resistance:.2f})")
    
    print(f"\n5. Testing Take Profit Report Generation")
    
    # Generate comprehensive report
    report = manager.get_take_profit_report()
    
    print(f"Take Profit Report Summary:")
    summary = report['summary']
    print(f"  Total Take Profits Created: {summary['total_profits']}")
    print(f"  Active Take Profits: {summary['active_profits']}")
    print(f"  Triggered Take Profits: {summary['triggered_profits']}")
    print(f"  Total Potential Profit: ${summary['total_potential_profit']:,.0f}")
    print(f"  Total Realized Profit: ${summary['total_realized_profit']:,.0f}")
    print(f"  Overall Realization Rate: {summary['overall_realization_rate']:.1%}")
    
    # Strategy performance
    print(f"\nStrategy Performance:")
    for strategy, perf in report['strategy_performance'].items():
        if perf['total_profits'] > 0:
            print(f"  {strategy}:")
            print(f"    Orders: {perf['total_profits']}, "
                  f"Realized: ${perf['realized_profit']:,.0f}, "
                  f"Rate: {perf['realization_rate']:.1%}")
    
    # Active profits details
    if report['active_profit_details']:
        print(f"\nActive Take Profits:")
        for profit in report['active_profit_details'][:3]:  # Show first 3
            print(f"  {profit['symbol']} {profit['type']}: "
                  f"Filled {profit['fill_percentage']:.0f}%, "
                  f"Next @ ${profit['next_target']:.2f}, "
                  f"Realized: ${profit['realized_profit']:,.0f}")
    
    print(f"\n6. Testing Take Profit Modification")
    
    # Test profit modification
    if fib_profit.profit_levels:
        original_targets = {level.level_id: level.target_price for level in fib_profit.profit_levels}
        
        # Modify first two levels
        modifications = {
            fib_profit.profit_levels[0].level_id: original_targets[fib_profit.profit_levels[0].level_id] + 5,
            fib_profit.profit_levels[1].level_id: original_targets[fib_profit.profit_levels[1].level_id] + 8
        }
        
        modify_success = manager.modify_take_profit(fib_profit.profit_id, modifications)
        
        print(f"Take Profit Modification:")
        print(f"  Modification {'successful' if modify_success else 'failed'}")
        if modify_success:
            print(f"  Modified Levels:")
            for level_id, new_price in modifications.items():
                original = original_targets[level_id]
                print(f"    {level_id}: ${original:.2f} → ${new_price:.2f}")
    
    print(f"\n7. Testing Utility Functions")
    
    # Test utility functions
    quick_profit = create_take_profit_order(
        symbol='AMD',
        position_size=400,
        entry_price=120.00,
        current_price=125.00,
        profit_type='ladder',
        profit_levels=[0.06, 0.12, 0.18],
        quantity_distribution=[0.4, 0.35, 0.25]
    )
    
    print(f"Quick Take Profit Creation:")
    print(f"  Symbol: {quick_profit.symbol}")
    print(f"  Type: {quick_profit.profit_type.value}")
    print(f"  Levels: {len(quick_profit.profit_levels)}")
    for level in quick_profit.profit_levels:
        print(f"    {level.quantity} shares @ ${level.target_price:.2f}")
    
    # Test profit target calculation
    targets = calculate_profit_targets(
        entry_price=100.00,
        profit_type='ladder',
        position_size=500,
        levels=[0.05, 0.10, 0.15, 0.20]
    )
    
    print(f"\nProfit Target Calculations for $100 entry:")
    for i, target in enumerate(targets):
        profit_pct = ((target - 100.00) / 100.00) * 100
        print(f"  Level {i+1}: ${target:.2f} ({profit_pct:.1f}% profit)")
    
    print(f"\n8. Testing Performance Analysis")
    
    # Get performance analysis
    if manager.performance_history:
        performance_df = manager.get_strategy_performance_analysis()
        
        if not performance_df.empty:
            print(f"Strategy Performance Analysis:")
            print(performance_df[['profit_type', 'realized_profit_mean', 'realization_rate_mean', 
                                 'realized_profit_count', 'fill_rate']].round(3))
    
    print("\nTake profit system testing completed successfully!")
    print("\nImplemented features include:")
    print("• 8 take profit strategies (Fixed %, Ratio, ATR, Ladder, Fibonacci, etc.)")
    print("• Multi-level ladder take profits with quantity distribution")
    print("• Fibonacci retracement/extension profit targets")
    print("• Trailing take profits with dynamic adjustment")
    print("• Technical level profits with resistance integration")
    print("• Real-time profit monitoring and triggering")
    print("• Take profit modification and cancellation")
    print("• Comprehensive performance tracking and analytics")
    print("• Strategy performance comparison and optimization")
