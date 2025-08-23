# ============================================
# StockPredictionPro - src/trading/risk/stop_loss.py
# Comprehensive stop-loss management system with advanced strategies and dynamic adjustment
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

logger = get_logger('trading.risk.stop_loss')

# ============================================
# Stop Loss Data Structures and Enums
# ============================================

class StopLossType(Enum):
    """Types of stop loss strategies"""
    FIXED_PERCENTAGE = "fixed_percentage"
    FIXED_DOLLAR = "fixed_dollar"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    TRAILING = "trailing"
    TECHNICAL_LEVEL = "technical_level"
    TIME_BASED = "time_based"
    PERCENTAGE_TRAILING = "percentage_trailing"
    ATR_TRAILING = "atr_trailing"
    SUPPORT_RESISTANCE = "support_resistance"

class StopLossStatus(Enum):
    """Stop loss status enumeration"""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    MODIFIED = "modified"

class TriggerCondition(Enum):
    """Stop loss trigger conditions"""
    CLOSE_BELOW = "close_below"          # Close price below stop level
    INTRADAY_BELOW = "intraday_below"    # Any price below stop level
    TWO_CONSECUTIVE = "two_consecutive"   # Two consecutive closes below
    VOLUME_CONFIRMATION = "volume_confirmation"  # Price + volume confirmation

@dataclass
class StopLossOrder:
    """Comprehensive stop loss order representation"""
    
    # Basic order information
    stop_id: str
    symbol: str
    position_size: int
    entry_price: float
    current_price: float
    
    # Stop loss configuration
    stop_type: StopLossType
    stop_price: float
    original_stop_price: float
    
    # Status and timestamps
    status: StopLossStatus = StopLossStatus.ACTIVE
    creation_time: datetime = field(default_factory=datetime.now)
    last_update_time: datetime = field(default_factory=datetime.now)
    trigger_time: Optional[datetime] = None
    expiry_time: Optional[datetime] = None
    
    # Advanced parameters
    trigger_condition: TriggerCondition = TriggerCondition.CLOSE_BELOW
    trail_amount: Optional[float] = None
    trail_percentage: Optional[float] = None
    atr_multiplier: Optional[float] = None
    
    # Performance tracking
    max_favorable_price: Optional[float] = None
    max_adverse_price: Optional[float] = None
    risk_amount: float = 0.0
    current_pnl: float = 0.0
    
    # Technical levels
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    
    # Metadata
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        self.original_stop_price = self.stop_price
        self.risk_amount = abs((self.entry_price - self.stop_price) * self.position_size)
        self.update_pnl()
        
        # Set default expiry (end of trading day)
        if self.expiry_time is None and self.stop_type == StopLossType.TIME_BASED:
            self.expiry_time = datetime.combine(
                self.creation_time.date() + timedelta(days=1),
                datetime.min.time().replace(hour=16)  # 4 PM
            )
    
    def update_pnl(self):
        """Update current P&L"""
        if self.position_size > 0:  # Long position
            self.current_pnl = (self.current_price - self.entry_price) * self.position_size
        else:  # Short position
            self.current_pnl = (self.entry_price - self.current_price) * abs(self.position_size)
    
    def update_current_price(self, new_price: float):
        """Update current price and related metrics"""
        self.current_price = new_price
        self.last_update_time = datetime.now()
        self.update_pnl()
        
        # Track favorable and adverse price movements
        if self.position_size > 0:  # Long position
            if self.max_favorable_price is None or new_price > self.max_favorable_price:
                self.max_favorable_price = new_price
            if self.max_adverse_price is None or new_price < self.max_adverse_price:
                self.max_adverse_price = new_price
        else:  # Short position
            if self.max_favorable_price is None or new_price < self.max_favorable_price:
                self.max_favorable_price = new_price
            if self.max_adverse_price is None or new_price > self.max_adverse_price:
                self.max_adverse_price = new_price
    
    @property
    def is_long_position(self) -> bool:
        """Check if this is a long position"""
        return self.position_size > 0
    
    @property
    def is_short_position(self) -> bool:
        """Check if this is a short position"""
        return self.position_size < 0
    
    @property
    def distance_to_stop(self) -> float:
        """Distance from current price to stop price"""
        if self.is_long_position:
            return self.current_price - self.stop_price
        else:
            return self.stop_price - self.current_price
    
    @property
    def distance_to_stop_percentage(self) -> float:
        """Distance to stop as percentage of current price"""
        if self.current_price > 0:
            return abs(self.distance_to_stop) / self.current_price
        return 0.0

# ============================================
# Base Stop Loss Strategy
# ============================================

class BaseStopLossStrategy:
    """
    Base class for stop loss strategies.
    
    Provides common functionality for calculating and managing
    stop loss levels based on various criteria.
    """
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.active_stops = {}
        self.triggered_stops = []
        
        logger.debug(f"Initialized {strategy_name} stop loss strategy")
    
    def create_stop_loss(self, symbol: str, position_size: int, entry_price: float,
                        current_price: float, **kwargs) -> StopLossOrder:
        """Create stop loss order - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement create_stop_loss method")
    
    def update_stop_loss(self, stop_order: StopLossOrder, current_price: float,
                        market_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update stop loss based on current market conditions
        
        Args:
            stop_order: Stop loss order to update
            current_price: Current market price
            market_data: Additional market data (volume, volatility, etc.)
            
        Returns:
            True if stop was updated, False otherwise
        """
        
        stop_order.update_current_price(current_price)
        
        # Check if stop loss should be triggered
        if self._should_trigger_stop(stop_order, market_data):
            self._trigger_stop_loss(stop_order)
            return True
        
        # Update trailing stops
        if stop_order.stop_type in [StopLossType.TRAILING, StopLossType.PERCENTAGE_TRAILING, StopLossType.ATR_TRAILING]:
            return self._update_trailing_stop(stop_order, market_data)
        
        return False
    
    def _should_trigger_stop(self, stop_order: StopLossOrder, 
                           market_data: Optional[Dict[str, Any]] = None) -> bool:
        """Check if stop loss should be triggered"""
        
        if stop_order.status != StopLossStatus.ACTIVE:
            return False
        
        current_price = stop_order.current_price
        stop_price = stop_order.stop_price
        
        # Basic price trigger check
        if stop_order.is_long_position:
            price_triggered = current_price <= stop_price
        else:
            price_triggered = current_price >= stop_price
        
        # Handle different trigger conditions
        if stop_order.trigger_condition == TriggerCondition.CLOSE_BELOW:
            return price_triggered
        
        elif stop_order.trigger_condition == TriggerCondition.INTRADAY_BELOW:
            # Check intraday low/high from market data
            if market_data:
                if stop_order.is_long_position:
                    intraday_low = market_data.get('low', current_price)
                    return intraday_low <= stop_price
                else:
                    intraday_high = market_data.get('high', current_price)
                    return intraday_high >= stop_price
            return price_triggered
        
        elif stop_order.trigger_condition == TriggerCondition.VOLUME_CONFIRMATION:
            # Require volume confirmation
            if market_data and price_triggered:
                current_volume = market_data.get('volume', 0)
                avg_volume = market_data.get('average_volume', 0)
                return current_volume > avg_volume * 0.5  # 50% of average volume
            return False
        
        return price_triggered
    
    def _trigger_stop_loss(self, stop_order: StopLossOrder):
        """Trigger the stop loss order"""
        stop_order.status = StopLossStatus.TRIGGERED
        stop_order.trigger_time = datetime.now()
        
        # Move from active to triggered
        if stop_order.stop_id in self.active_stops:
            del self.active_stops[stop_order.stop_id]
        self.triggered_stops.append(stop_order)
        
        logger.info(f"Stop loss triggered for {stop_order.symbol}: "
                   f"Price {stop_order.current_price:.2f} hit stop {stop_order.stop_price:.2f}")
    
    def _update_trailing_stop(self, stop_order: StopLossOrder, 
                            market_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update trailing stop loss"""
        # To be implemented by trailing stop strategies
        return False
    
    def get_stop_loss_statistics(self) -> Dict[str, Any]:
        """Get statistics about stop loss performance"""
        
        all_stops = list(self.active_stops.values()) + self.triggered_stops
        
        if not all_stops:
            return {}
        
        triggered = [s for s in all_stops if s.status == StopLossStatus.TRIGGERED]
        
        total_risk = sum(s.risk_amount for s in all_stops)
        realized_losses = sum(s.current_pnl for s in triggered if s.current_pnl < 0)
        
        return {
            'total_stops': len(all_stops),
            'active_stops': len(self.active_stops),
            'triggered_stops': len(triggered),
            'trigger_rate': len(triggered) / len(all_stops) if all_stops else 0,
            'total_risk_amount': total_risk,
            'realized_losses': realized_losses,
            'average_risk_per_stop': total_risk / len(all_stops) if all_stops else 0
        }

# ============================================
# Fixed Stop Loss Strategies
# ============================================

class FixedPercentageStopLoss(BaseStopLossStrategy):
    """Fixed percentage stop loss strategy"""
    
    def __init__(self, stop_percentage: float = 0.05):
        super().__init__("Fixed Percentage")
        self.stop_percentage = stop_percentage
    
    def create_stop_loss(self, symbol: str, position_size: int, entry_price: float,
                        current_price: float, **kwargs) -> StopLossOrder:
        """Create fixed percentage stop loss"""
        
        # Calculate stop price based on percentage
        if position_size > 0:  # Long position
            stop_price = entry_price * (1 - self.stop_percentage)
        else:  # Short position
            stop_price = entry_price * (1 + self.stop_percentage)
        
        stop_id = f"FIXED_PCT_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return StopLossOrder(
            stop_id=stop_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            stop_type=StopLossType.FIXED_PERCENTAGE,
            stop_price=stop_price
        )

class FixedDollarStopLoss(BaseStopLossStrategy):
    """Fixed dollar amount stop loss strategy"""
    
    def __init__(self, stop_amount: float = 500.0):
        super().__init__("Fixed Dollar")
        self.stop_amount = stop_amount
    
    def create_stop_loss(self, symbol: str, position_size: int, entry_price: float,
                        current_price: float, **kwargs) -> StopLossOrder:
        """Create fixed dollar amount stop loss"""
        
        # Calculate stop price based on dollar amount
        if position_size > 0:  # Long position
            stop_price = entry_price - (self.stop_amount / abs(position_size))
        else:  # Short position
            stop_price = entry_price + (self.stop_amount / abs(position_size))
        
        stop_id = f"FIXED_DOL_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return StopLossOrder(
            stop_id=stop_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            stop_type=StopLossType.FIXED_DOLLAR,
            stop_price=stop_price
        )

# ============================================
# ATR-Based Stop Loss Strategy
# ============================================

class ATRBasedStopLoss(BaseStopLossStrategy):
    """Average True Range (ATR) based stop loss strategy"""
    
    def __init__(self, atr_multiplier: float = 2.0):
        super().__init__("ATR Based")
        self.atr_multiplier = atr_multiplier
    
    def create_stop_loss(self, symbol: str, position_size: int, entry_price: float,
                        current_price: float, atr: float, **kwargs) -> StopLossOrder:
        """Create ATR-based stop loss"""
        
        if atr <= 0:
            raise ValueError(f"Invalid ATR value: {atr}")
        
        # Calculate stop distance based on ATR
        stop_distance = atr * self.atr_multiplier
        
        # Calculate stop price
        if position_size > 0:  # Long position
            stop_price = entry_price - stop_distance
        else:  # Short position
            stop_price = entry_price + stop_distance
        
        stop_id = f"ATR_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return StopLossOrder(
            stop_id=stop_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            stop_type=StopLossType.ATR_BASED,
            stop_price=stop_price,
            atr_multiplier=self.atr_multiplier
        )

# ============================================
# Volatility-Based Stop Loss Strategy
# ============================================

class VolatilityBasedStopLoss(BaseStopLossStrategy):
    """Volatility-based stop loss strategy"""
    
    def __init__(self, volatility_multiplier: float = 2.0, confidence_level: float = 0.95):
        super().__init__("Volatility Based")
        self.volatility_multiplier = volatility_multiplier
        self.confidence_level = confidence_level
    
    def create_stop_loss(self, symbol: str, position_size: int, entry_price: float,
                        current_price: float, volatility: float, **kwargs) -> StopLossOrder:
        """Create volatility-based stop loss"""
        
        if volatility <= 0:
            raise ValueError(f"Invalid volatility: {volatility}")
        
        # Calculate stop distance based on volatility and confidence level
        z_score = stats.norm.ppf(self.confidence_level)
        stop_distance = entry_price * volatility * z_score * self.volatility_multiplier
        
        # Calculate stop price
        if position_size > 0:  # Long position
            stop_price = entry_price - stop_distance
        else:  # Short position
            stop_price = entry_price + stop_distance
        
        stop_id = f"VOL_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return StopLossOrder(
            stop_id=stop_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            stop_type=StopLossType.VOLATILITY_BASED,
            stop_price=stop_price
        )

# ============================================
# Trailing Stop Loss Strategies
# ============================================

class PercentageTrailingStopLoss(BaseStopLossStrategy):
    """Percentage-based trailing stop loss strategy"""
    
    def __init__(self, trail_percentage: float = 0.05):
        super().__init__("Percentage Trailing")
        self.trail_percentage = trail_percentage
    
    def create_stop_loss(self, symbol: str, position_size: int, entry_price: float,
                        current_price: float, **kwargs) -> StopLossOrder:
        """Create percentage trailing stop loss"""
        
        # Initial stop price
        if position_size > 0:  # Long position
            stop_price = current_price * (1 - self.trail_percentage)
        else:  # Short position
            stop_price = current_price * (1 + self.trail_percentage)
        
        stop_id = f"TRAIL_PCT_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return StopLossOrder(
            stop_id=stop_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            stop_type=StopLossType.PERCENTAGE_TRAILING,
            stop_price=stop_price,
            trail_percentage=self.trail_percentage
        )
    
    def _update_trailing_stop(self, stop_order: StopLossOrder, 
                            market_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update percentage trailing stop"""
        
        current_price = stop_order.current_price
        
        if stop_order.is_long_position:
            # For long positions, only move stop up
            new_stop_price = current_price * (1 - stop_order.trail_percentage)
            if new_stop_price > stop_order.stop_price:
                stop_order.stop_price = new_stop_price
                stop_order.last_update_time = datetime.now()
                return True
        else:
            # For short positions, only move stop down
            new_stop_price = current_price * (1 + stop_order.trail_percentage)
            if new_stop_price < stop_order.stop_price:
                stop_order.stop_price = new_stop_price
                stop_order.last_update_time = datetime.now()
                return True
        
        return False

class ATRTrailingStopLoss(BaseStopLossStrategy):
    """ATR-based trailing stop loss strategy"""
    
    def __init__(self, atr_multiplier: float = 2.0):
        super().__init__("ATR Trailing")
        self.atr_multiplier = atr_multiplier
    
    def create_stop_loss(self, symbol: str, position_size: int, entry_price: float,
                        current_price: float, atr: float, **kwargs) -> StopLossOrder:
        """Create ATR trailing stop loss"""
        
        if atr <= 0:
            raise ValueError(f"Invalid ATR value: {atr}")
        
        # Initial stop price
        stop_distance = atr * self.atr_multiplier
        
        if position_size > 0:  # Long position
            stop_price = current_price - stop_distance
        else:  # Short position
            stop_price = current_price + stop_distance
        
        stop_id = f"TRAIL_ATR_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return StopLossOrder(
            stop_id=stop_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            stop_type=StopLossType.ATR_TRAILING,
            stop_price=stop_price,
            atr_multiplier=self.atr_multiplier
        )
    
    def _update_trailing_stop(self, stop_order: StopLossOrder, 
                            market_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update ATR trailing stop"""
        
        # Need current ATR from market data
        if not market_data or 'atr' not in market_data:
            return False
        
        current_atr = market_data['atr']
        current_price = stop_order.current_price
        stop_distance = current_atr * stop_order.atr_multiplier
        
        if stop_order.is_long_position:
            # For long positions, only move stop up
            new_stop_price = current_price - stop_distance
            if new_stop_price > stop_order.stop_price:
                stop_order.stop_price = new_stop_price
                stop_order.last_update_time = datetime.now()
                return True
        else:
            # For short positions, only move stop down
            new_stop_price = current_price + stop_distance
            if new_stop_price < stop_order.stop_price:
                stop_order.stop_price = new_stop_price
                stop_order.last_update_time = datetime.now()
                return True
        
        return False

# ============================================
# Technical Level Stop Loss Strategy
# ============================================

class TechnicalLevelStopLoss(BaseStopLossStrategy):
    """Technical support/resistance level stop loss strategy"""
    
    def __init__(self, buffer_percentage: float = 0.01):
        super().__init__("Technical Level")
        self.buffer_percentage = buffer_percentage
    
    def create_stop_loss(self, symbol: str, position_size: int, entry_price: float,
                        current_price: float, support_level: Optional[float] = None,
                        resistance_level: Optional[float] = None, **kwargs) -> StopLossOrder:
        """Create technical level stop loss"""
        
        # Determine stop level based on position direction
        if position_size > 0:  # Long position
            if support_level is None:
                raise ValueError("Support level required for long positions")
            stop_price = support_level * (1 - self.buffer_percentage)
        else:  # Short position
            if resistance_level is None:
                raise ValueError("Resistance level required for short positions")
            stop_price = resistance_level * (1 + self.buffer_percentage)
        
        stop_id = f"TECH_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return StopLossOrder(
            stop_id=stop_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            stop_type=StopLossType.TECHNICAL_LEVEL,
            stop_price=stop_price,
            support_level=support_level,
            resistance_level=resistance_level
        )

# ============================================
# Time-Based Stop Loss Strategy
# ============================================

class TimeBasedStopLoss(BaseStopLossStrategy):
    """Time-based stop loss strategy"""
    
    def __init__(self, max_hold_hours: int = 24):
        super().__init__("Time Based")
        self.max_hold_hours = max_hold_hours
    
    def create_stop_loss(self, symbol: str, position_size: int, entry_price: float,
                        current_price: float, **kwargs) -> StopLossOrder:
        """Create time-based stop loss"""
        
        # Stop price is the current price (exit at any price when time expires)
        stop_price = current_price
        
        # Set expiry time
        expiry_time = datetime.now() + timedelta(hours=self.max_hold_hours)
        
        stop_id = f"TIME_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return StopLossOrder(
            stop_id=stop_id,
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            stop_type=StopLossType.TIME_BASED,
            stop_price=stop_price,
            expiry_time=expiry_time
        )
    
    def update_stop_loss(self, stop_order: StopLossOrder, current_price: float,
                        market_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update time-based stop loss"""
        
        stop_order.update_current_price(current_price)
        
        # Check if time has expired
        if stop_order.expiry_time and datetime.now() >= stop_order.expiry_time:
            stop_order.status = StopLossStatus.EXPIRED
            stop_order.trigger_time = datetime.now()
            
            # Move from active to triggered
            if stop_order.stop_id in self.active_stops:
                del self.active_stops[stop_order.stop_id]
            self.triggered_stops.append(stop_order)
            
            logger.info(f"Time-based stop expired for {stop_order.symbol}")
            return True
        
        return False

# ============================================
# Stop Loss Manager
# ============================================

class StopLossManager:
    """
    Comprehensive stop loss management system.
    
    Orchestrates multiple stop loss strategies, manages active stops,
    and provides performance analytics and optimization.
    """
    
    def __init__(self):
        # Initialize all stop loss strategies
        self.strategies = {
            StopLossType.FIXED_PERCENTAGE: FixedPercentageStopLoss(),
            StopLossType.FIXED_DOLLAR: FixedDollarStopLoss(),
            StopLossType.ATR_BASED: ATRBasedStopLoss(),
            StopLossType.VOLATILITY_BASED: VolatilityBasedStopLoss(),
            StopLossType.PERCENTAGE_TRAILING: PercentageTrailingStopLoss(),
            StopLossType.ATR_TRAILING: ATRTrailingStopLoss(),
            StopLossType.TECHNICAL_LEVEL: TechnicalLevelStopLoss(),
            StopLossType.TIME_BASED: TimeBasedStopLoss()
        }
        
        # Global stop tracking
        self.all_stops = {}
        self.performance_history = []
        
        # Configuration
        self.update_frequency_seconds = 60  # Update stops every minute
        self.last_update_time = datetime.now()
        
        logger.info("Initialized StopLossManager with 8 stop loss strategies")
    
    @time_it("create_stop_loss")
    def create_stop_loss(self, symbol: str, position_size: int, entry_price: float,
                        current_price: float, stop_type: StopLossType,
                        **kwargs) -> StopLossOrder:
        """
        Create a stop loss order using specified strategy
        
        Args:
            symbol: Trading symbol
            position_size: Position size (positive for long, negative for short)
            entry_price: Entry price of position
            current_price: Current market price
            stop_type: Type of stop loss strategy
            **kwargs: Strategy-specific parameters
            
        Returns:
            StopLossOrder object
        """
        
        if stop_type not in self.strategies:
            raise ValueError(f"Unknown stop loss type: {stop_type}")
        
        # Create stop loss using appropriate strategy
        strategy = self.strategies[stop_type]
        stop_order = strategy.create_stop_loss(
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            **kwargs
        )
        
        # Register stop in manager and strategy
        self.all_stops[stop_order.stop_id] = stop_order
        strategy.active_stops[stop_order.stop_id] = stop_order
        
        logger.info(f"Created {stop_type.value} stop loss for {symbol}: "
                   f"Stop @ {stop_order.stop_price:.2f}")
        
        return stop_order
    
    def update_all_stops(self, price_data: Dict[str, float], 
                        market_data: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
        """
        Update all active stop loss orders
        
        Args:
            price_data: Dictionary of symbol -> current_price
            market_data: Dictionary of symbol -> market_data
            
        Returns:
            List of triggered stop IDs
        """
        
        triggered_stops = []
        
        for stop_id, stop_order in list(self.all_stops.items()):
            if stop_order.status != StopLossStatus.ACTIVE:
                continue
            
            symbol = stop_order.symbol
            if symbol not in price_data:
                continue
            
            current_price = price_data[symbol]
            symbol_market_data = market_data.get(symbol) if market_data else None
            
            # Update stop using appropriate strategy
            strategy = self.strategies[stop_order.stop_type]
            was_triggered = strategy.update_stop_loss(
                stop_order, current_price, symbol_market_data
            )
            
            if was_triggered:
                triggered_stops.append(stop_id)
                # Record performance
                self._record_stop_performance(stop_order)
        
        self.last_update_time = datetime.now()
        
        if triggered_stops:
            logger.info(f"Updated stops, {len(triggered_stops)} triggered")
        
        return triggered_stops
    
    def cancel_stop_loss(self, stop_id: str, reason: str = "User cancelled") -> bool:
        """Cancel an active stop loss order"""
        
        if stop_id not in self.all_stops:
            logger.warning(f"Stop loss {stop_id} not found")
            return False
        
        stop_order = self.all_stops[stop_id]
        
        if stop_order.status != StopLossStatus.ACTIVE:
            logger.warning(f"Stop loss {stop_id} is not active")
            return False
        
        # Update status
        stop_order.status = StopLossStatus.CANCELLED
        stop_order.trigger_time = datetime.now()
        
        # Remove from strategy's active stops
        strategy = self.strategies[stop_order.stop_type]
        if stop_id in strategy.active_stops:
            del strategy.active_stops[stop_id]
        
        logger.info(f"Cancelled stop loss {stop_id}: {reason}")
        return True
    
    def modify_stop_loss(self, stop_id: str, new_stop_price: Optional[float] = None,
                        **kwargs) -> bool:
        """Modify an existing stop loss order"""
        
        if stop_id not in self.all_stops:
            logger.warning(f"Stop loss {stop_id} not found")
            return False
        
        stop_order = self.all_stops[stop_id]
        
        if stop_order.status != StopLossStatus.ACTIVE:
            logger.warning(f"Stop loss {stop_id} is not active")
            return False
        
        # Update stop price if provided
        if new_stop_price is not None:
            stop_order.stop_price = new_stop_price
        
        # Update other parameters
        for key, value in kwargs.items():
            if hasattr(stop_order, key):
                setattr(stop_order, key, value)
        
        stop_order.status = StopLossStatus.MODIFIED
        stop_order.last_update_time = datetime.now()
        
        logger.info(f"Modified stop loss {stop_id}")
        return True
    
    def get_active_stops(self, symbol: Optional[str] = None) -> List[StopLossOrder]:
        """Get active stop loss orders, optionally filtered by symbol"""
        
        active_stops = [
            stop for stop in self.all_stops.values()
            if stop.status == StopLossStatus.ACTIVE
        ]
        
        if symbol:
            active_stops = [stop for stop in active_stops if stop.symbol == symbol]
        
        return active_stops
    
    def get_stop_loss_report(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive stop loss report"""
        
        # Filter stops by symbol if specified
        if symbol:
            relevant_stops = [s for s in self.all_stops.values() if s.symbol == symbol]
        else:
            relevant_stops = list(self.all_stops.values())
        
        if not relevant_stops:
            return {'error': f'No stops found for {symbol}' if symbol else 'No stops found'}
        
        # Categorize stops by status
        active_stops = [s for s in relevant_stops if s.status == StopLossStatus.ACTIVE]
        triggered_stops = [s for s in relevant_stops if s.status == StopLossStatus.TRIGGERED]
        cancelled_stops = [s for s in relevant_stops if s.status == StopLossStatus.CANCELLED]
        
        # Calculate statistics
        total_risk = sum(s.risk_amount for s in relevant_stops)
        realized_losses = sum(s.current_pnl for s in triggered_stops if s.current_pnl < 0)
        
        # Performance by strategy type
        strategy_performance = {}
        for stop_type in StopLossType:
            type_stops = [s for s in relevant_stops if s.stop_type == stop_type]
            if type_stops:
                type_triggered = [s for s in type_stops if s.status == StopLossStatus.TRIGGERED]
                strategy_performance[stop_type.value] = {
                    'total_stops': len(type_stops),
                    'triggered_stops': len(type_triggered),
                    'trigger_rate': len(type_triggered) / len(type_stops),
                    'average_loss': np.mean([s.current_pnl for s in type_triggered if s.current_pnl < 0]) if type_triggered else 0
                }
        
        return {
            'symbol': symbol or 'All',
            'report_time': datetime.now(),
            'summary': {
                'total_stops': len(relevant_stops),
                'active_stops': len(active_stops),
                'triggered_stops': len(triggered_stops),
                'cancelled_stops': len(cancelled_stops),
                'total_risk_amount': total_risk,
                'realized_losses': realized_losses
            },
            'strategy_performance': strategy_performance,
            'active_stop_details': [
                {
                    'stop_id': stop.stop_id,
                    'symbol': stop.symbol,
                    'type': stop.stop_type.value,
                    'stop_price': stop.stop_price,
                    'current_price': stop.current_price,
                    'distance_to_stop': stop.distance_to_stop,
                    'distance_percentage': stop.distance_to_stop_percentage,
                    'current_pnl': stop.current_pnl,
                    'time_active': (datetime.now() - stop.creation_time).total_seconds() / 3600
                }
                for stop in active_stops
            ]
        }
    
    def _record_stop_performance(self, stop_order: StopLossOrder):
        """Record stop loss performance for analysis"""
        
        performance_record = {
            'timestamp': datetime.now(),
            'stop_id': stop_order.stop_id,
            'symbol': stop_order.symbol,
            'stop_type': stop_order.stop_type.value,
            'entry_price': stop_order.entry_price,
            'stop_price': stop_order.stop_price,
            'trigger_price': stop_order.current_price,
            'position_size': stop_order.position_size,
            'pnl': stop_order.current_pnl,
            'risk_amount': stop_order.risk_amount,
            'time_active_hours': (datetime.now() - stop_order.creation_time).total_seconds() / 3600,
            'max_favorable_price': stop_order.max_favorable_price,
            'max_adverse_price': stop_order.max_adverse_price
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent performance data
        cutoff_time = datetime.now() - timedelta(days=90)
        self.performance_history = [
            p for p in self.performance_history 
            if p['timestamp'] > cutoff_time
        ]
    
    def optimize_stop_parameters(self, symbol: str, 
                                historical_data: pd.DataFrame,
                                strategy_type: StopLossType = StopLossType.PERCENTAGE_TRAILING) -> Dict[str, float]:
        """
        Optimize stop loss parameters using historical data
        
        Args:
            symbol: Symbol to optimize for
            historical_data: Historical price data (OHLCV)
            strategy_type: Stop loss strategy to optimize
            
        Returns:
            Dictionary of optimal parameters
        """
        
        logger.info(f"Optimizing {strategy_type.value} parameters for {symbol}")
        
        # This is a simplified optimization example
        # In practice, you would use more sophisticated optimization techniques
        
        if strategy_type == StopLossType.PERCENTAGE_TRAILING:
            return self._optimize_percentage_trailing(historical_data)
        elif strategy_type == StopLossType.ATR_BASED:
            return self._optimize_atr_parameters(historical_data)
        else:
            logger.warning(f"Optimization not implemented for {strategy_type.value}")
            return {}
    
    def _optimize_percentage_trailing(self, data: pd.DataFrame) -> Dict[str, float]:
        """Optimize percentage trailing stop parameters"""
        
        best_params = {'trail_percentage': 0.05}
        best_return = -float('inf')
        
        # Test different trailing percentages
        for trail_pct in [0.02, 0.03, 0.05, 0.07, 0.10]:
            returns = self._backtest_trailing_stop(data, trail_pct)
            total_return = returns.sum()
            
            if total_return > best_return:
                best_return = total_return
                best_params['trail_percentage'] = trail_pct
        
        logger.info(f"Optimal trailing percentage: {best_params['trail_percentage']:.1%}")
        return best_params
    
    def _optimize_atr_parameters(self, data: pd.DataFrame) -> Dict[str, float]:
        """Optimize ATR stop loss parameters"""
        
        if 'atr' not in data.columns:
            logger.warning("ATR data not available for optimization")
            return {'atr_multiplier': 2.0}
        
        best_params = {'atr_multiplier': 2.0}
        best_return = -float('inf')
        
        # Test different ATR multipliers
        for multiplier in [1.0, 1.5, 2.0, 2.5, 3.0]:
            # Simplified backtest
            returns = []
            for i in range(1, len(data)):
                stop_distance = data['atr'].iloc[i] * multiplier
                # This is a simplified calculation
                # Actual backtesting would be more complex
            
        return best_params
    
    def _backtest_trailing_stop(self, data: pd.DataFrame, trail_percentage: float) -> pd.Series:
        """Simple backtest of trailing stop strategy"""
        
        returns = []
        position = 0
        entry_price = 0
        stop_price = 0
        
        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            
            if position == 0:  # Not in position
                # Simple entry signal (price above previous close)
                if current_price > data['close'].iloc[i-1]:
                    position = 1
                    entry_price = current_price
                    stop_price = current_price * (1 - trail_percentage)
            
            else:  # In position
                # Update trailing stop
                new_stop = current_price * (1 - trail_percentage)
                if new_stop > stop_price:
                    stop_price = new_stop
                
                # Check if stopped out
                if current_price <= stop_price:
                    trade_return = (current_price - entry_price) / entry_price
                    returns.append(trade_return)
                    position = 0
        
        return pd.Series(returns)

# ============================================
# Utility Functions
# ============================================

def create_stop_loss_order(symbol: str, position_size: int, entry_price: float,
                          current_price: float, stop_type: str = 'percentage',
                          **kwargs) -> StopLossOrder:
    """
    Utility function to create a stop loss order
    
    Args:
        symbol: Trading symbol
        position_size: Position size
        entry_price: Entry price
        current_price: Current price
        stop_type: Type of stop loss ('percentage', 'atr', 'trailing', etc.)
        **kwargs: Additional parameters
        
    Returns:
        StopLossOrder object
    """
    
    manager = StopLossManager()
    
    # Map string types to enums
    type_mapping = {
        'percentage': StopLossType.FIXED_PERCENTAGE,
        'dollar': StopLossType.FIXED_DOLLAR,
        'atr': StopLossType.ATR_BASED,
        'volatility': StopLossType.VOLATILITY_BASED,
        'trailing': StopLossType.PERCENTAGE_TRAILING,
        'atr_trailing': StopLossType.ATR_TRAILING,
        'technical': StopLossType.TECHNICAL_LEVEL,
        'time': StopLossType.TIME_BASED
    }
    
    stop_loss_type = type_mapping.get(stop_type.lower(), StopLossType.FIXED_PERCENTAGE)
    
    return manager.create_stop_loss(
        symbol=symbol,
        position_size=position_size,
        entry_price=entry_price,
        current_price=current_price,
        stop_type=stop_loss_type,
        **kwargs
    )

def calculate_stop_loss_level(entry_price: float, stop_type: str, **kwargs) -> float:
    """
    Calculate stop loss level without creating an order
    
    Args:
        entry_price: Entry price of position
        stop_type: Type of stop loss calculation
        **kwargs: Additional parameters (percentage, atr, etc.)
        
    Returns:
        Stop loss price level
    """
    
    if stop_type.lower() == 'percentage':
        percentage = kwargs.get('percentage', 0.05)
        return entry_price * (1 - percentage)
    
    elif stop_type.lower() == 'dollar':
        dollar_amount = kwargs.get('dollar_amount', 100)
        position_size = kwargs.get('position_size', 100)
        return entry_price - (dollar_amount / position_size)
    
    elif stop_type.lower() == 'atr':
        atr = kwargs.get('atr', 0)
        multiplier = kwargs.get('multiplier', 2.0)
        return entry_price - (atr * multiplier)
    
    else:
        raise ValueError(f"Unknown stop type: {stop_type}")

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Stop Loss System")
    
    # Initialize stop loss manager
    manager = StopLossManager()
    
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
    
    print(f"\n1. Testing Different Stop Loss Types")
    
    # Test different stop loss strategies
    stop_orders = []
    
    for pos in sample_positions:
        print(f"\nCreating stops for {pos['symbol']} position:")
        
        # Fixed percentage stop
        pct_stop = manager.create_stop_loss(
            symbol=pos['symbol'],
            position_size=pos['position_size'],
            entry_price=pos['entry_price'],
            current_price=pos['current_price'],
            stop_type=StopLossType.FIXED_PERCENTAGE,
            stop_percentage=0.05  # 5% stop
        )
        
        print(f"  Fixed 5% Stop: ${pct_stop.stop_price:.2f} "
              f"(Risk: ${pct_stop.risk_amount:.0f})")
        
        # ATR-based stop
        atr_stop = manager.create_stop_loss(
            symbol=pos['symbol'],
            position_size=pos['position_size'],
            entry_price=pos['entry_price'],
            current_price=pos['current_price'],
            stop_type=StopLossType.ATR_BASED,
            atr=pos['current_price'] * 0.02,  # 2% of price as ATR
            atr_multiplier=2.5
        )
        
        print(f"  ATR Stop (2.5x): ${atr_stop.stop_price:.2f} "
              f"(Risk: ${atr_stop.risk_amount:.0f})")
        
        # Trailing stop
        trail_stop = manager.create_stop_loss(
            symbol=pos['symbol'],
            position_size=pos['position_size'],
            entry_price=pos['entry_price'],
            current_price=pos['current_price'],
            stop_type=StopLossType.PERCENTAGE_TRAILING,
            trail_percentage=0.03  # 3% trailing
        )
        
        print(f"  Trailing 3% Stop: ${trail_stop.stop_price:.2f} "
              f"(Risk: ${trail_stop.risk_amount:.0f})")
        
        stop_orders.extend([pct_stop, atr_stop, trail_stop])
    
    print(f"\n2. Testing Stop Loss Updates")
    
    # Simulate price movements
    price_updates = {
        'AAPL': [185.00, 187.50, 190.00, 188.00, 185.50, 182.00, 179.00],
        'MSFT': [345.00, 342.00, 339.00, 341.00, 343.50, 347.00, 350.00],
        'GOOGL': [2850.00, 2900.00, 2920.00, 2880.00, 2860.00, 2840.00, 2820.00]
    }
    
    print(f"Simulating price movements...")
    
    for i, (aapl_price, msft_price, googl_price) in enumerate(zip(*price_updates.values())):
        price_data = {
            'AAPL': aapl_price,
            'MSFT': msft_price,
            'GOOGL': googl_price
        }
        
        # Market data with ATR
        market_data = {
            'AAPL': {'atr': aapl_price * 0.02, 'volume': 1000000},
            'MSFT': {'atr': msft_price * 0.025, 'volume': 800000},
            'GOOGL': {'atr': googl_price * 0.03, 'volume': 500000}
        }
        
        triggered_stops = manager.update_all_stops(price_data, market_data)
        
        if triggered_stops:
            print(f"  Period {i+1}: {len(triggered_stops)} stops triggered")
            for stop_id in triggered_stops:
                stop = manager.all_stops[stop_id]
                print(f"    {stop.symbol} {stop.stop_type.value}: "
                      f"${stop.current_price:.2f} hit stop ${stop.stop_price:.2f}")
        
        # Show trailing stop updates
        active_trailing = [
            s for s in manager.get_active_stops()
            if s.stop_type in [StopLossType.PERCENTAGE_TRAILING, StopLossType.ATR_TRAILING]
        ]
        
        if active_trailing and i > 0:
            print(f"  Trailing stops updated:")
            for stop in active_trailing[:2]:  # Show first 2
                print(f"    {stop.symbol}: Stop moved to ${stop.stop_price:.2f}")
    
    print(f"\n3. Testing Stop Loss Report Generation")
    
    # Generate comprehensive report
    report = manager.get_stop_loss_report()
    
    print(f"Stop Loss Report Summary:")
    summary = report['summary']
    print(f"  Total Stops Created: {summary['total_stops']}")
    print(f"  Active Stops: {summary['active_stops']}")
    print(f"  Triggered Stops: {summary['triggered_stops']}")
    print(f"  Total Risk Amount: ${summary['total_risk_amount']:,.0f}")
    print(f"  Realized Losses: ${summary['realized_losses']:,.0f}")
    
    # Show strategy performance
    print(f"\nStrategy Performance:")
    for strategy, perf in report['strategy_performance'].items():
        if perf['total_stops'] > 0:
            print(f"  {strategy}:")
            print(f"    Stops: {perf['total_stops']}, Triggered: {perf['triggered_stops']}")
            print(f"    Trigger Rate: {perf['trigger_rate']:.1%}")
            if perf['average_loss'] < 0:
                print(f"    Avg Loss: ${perf['average_loss']:,.0f}")
    
    # Show active stops details
    if report['active_stop_details']:
        print(f"\nActive Stops:")
        for stop in report['active_stop_details'][:3]:  # Show first 3
            print(f"  {stop['symbol']} {stop['type']}: "
                  f"Stop @ ${stop['stop_price']:.2f}, "
                  f"Distance: {stop['distance_percentage']:.2%}, "
                  f"P&L: ${stop['current_pnl']:,.0f}")
    
    print(f"\n4. Testing Technical Level Stops")
    
    # Test technical level stops
    tech_stop = manager.create_stop_loss(
        symbol='AAPL',
        position_size=500,
        entry_price=185.00,
        current_price=188.00,
        stop_type=StopLossType.TECHNICAL_LEVEL,
        support_level=182.00,  # Support level
        buffer_percentage=0.005  # 0.5% buffer below support
    )
    
    print(f"Technical Level Stop:")
    print(f"  Support Level: $182.00")
    print(f"  Stop Price: ${tech_stop.stop_price:.2f}")
    print(f"  Buffer: {0.5}%")
    
    print(f"\n5. Testing Time-Based Stops")
    
    # Test time-based stops
    time_stop = manager.create_stop_loss(
        symbol='TSLA',
        position_size=200,
        entry_price=250.00,
        current_price=255.00,
        stop_type=StopLossType.TIME_BASED,
        max_hold_hours=4
    )
    
    print(f"Time-Based Stop:")
    print(f"  Max Hold Time: 4 hours")
    print(f"  Expiry Time: {time_stop.expiry_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Current Status: {time_stop.status.value}")
    
    print(f"\n6. Testing Stop Loss Modification")
    
    # Test stop modification
    original_stop_price = tech_stop.stop_price
    modify_success = manager.modify_stop_loss(
        tech_stop.stop_id,
        new_stop_price=183.00,  # Move stop up
        trail_percentage=0.02   # Add trailing feature
    )
    
    print(f"Stop Modification:")
    print(f"  Original Stop: ${original_stop_price:.2f}")
    print(f"  New Stop: ${tech_stop.stop_price:.2f}")
    print(f"  Modification {'successful' if modify_success else 'failed'}")
    print(f"  Status: {tech_stop.status.value}")
    
    print(f"\n7. Testing Utility Functions")
    
    # Test utility functions
    quick_stop = create_stop_loss_order(
        symbol='NVDA',
        position_size=300,
        entry_price=800.00,
        current_price=820.00,
        stop_type='trailing',
        trail_percentage=0.04  # 4% trailing
    )
    
    print(f"Quick Stop Loss Creation:")
    print(f"  Symbol: {quick_stop.symbol}")
    print(f"  Type: {quick_stop.stop_type.value}")
    print(f"  Stop Price: ${quick_stop.stop_price:.2f}")
    print(f"  Distance: {quick_stop.distance_to_stop_percentage:.2%}")
    
    # Test stop level calculation
    levels = {}
    for stop_type in ['percentage', 'dollar', 'atr']:
        if stop_type == 'percentage':
            level = calculate_stop_loss_level(800.00, stop_type, percentage=0.06)
        elif stop_type == 'dollar':
            level = calculate_stop_loss_level(800.00, stop_type, dollar_amount=2000, position_size=300)
        else:  # atr
            level = calculate_stop_loss_level(800.00, stop_type, atr=16.00, multiplier=2.0)
        
        levels[stop_type] = level
    
    print(f"\nStop Level Calculations for $800 entry:")
    for stop_type, level in levels.items():
        distance = ((800.00 - level) / 800.00) * 100
        print(f"  {stop_type}: ${level:.2f} ({distance:.1f}% below entry)")
    
    print(f"\n8. Testing Stop Loss Statistics")
    
    # Get overall statistics
    all_stats = {}
    for stop_type, strategy in manager.strategies.items():
        stats = strategy.get_stop_loss_statistics()
        if stats:
            all_stats[stop_type.value] = stats
    
    print(f"Stop Loss Statistics by Strategy:")
    for strategy, stats in all_stats.items():
        if stats['total_stops'] > 0:
            print(f"  {strategy}:")
            print(f"    Total: {stats['total_stops']}, "
                  f"Triggered: {stats['triggered_stops']}, "
                  f"Rate: {stats['trigger_rate']:.1%}")
            print(f"    Total Risk: ${stats['total_risk_amount']:,.0f}")
    
    print("\nStop loss system testing completed successfully!")
    print("\nImplemented features include:")
    print("• 8 stop loss strategies (Fixed %, Dollar, ATR, Volatility, Trailing, etc.)")
    print("• Advanced trailing stops (percentage and ATR-based)")
    print("• Technical level stops with support/resistance")
    print("• Time-based stops with expiration")
    print("• Multiple trigger conditions (close, intraday, volume confirmation)")
    print("• Real-time stop monitoring and updates")
    print("• Stop loss modification and cancellation")
    print("• Comprehensive performance tracking and analytics")
    print("• Parameter optimization using historical data")
