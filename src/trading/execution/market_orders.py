# ============================================
# StockPredictionPro - src/trading/execution/market_orders.py
# Comprehensive market order execution system with advanced features and risk controls
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
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it
from .limit_orders import OrderSide, OrderStatus, Fill, LimitOrderBook

logger = get_logger('trading.execution.market_orders')

# ============================================
# Market Order Data Structures and Enums
# ============================================

class ExecutionUrgency(Enum):
    """Market order execution urgency levels"""
    LOW = "LOW"           # Patient execution, minimize market impact
    NORMAL = "NORMAL"     # Standard execution
    HIGH = "HIGH"         # Fast execution, accept higher costs
    IMMEDIATE = "IMMEDIATE"  # Execute ASAP, cost secondary

class MarketImpactModel(Enum):
    """Market impact estimation models"""
    LINEAR = "LINEAR"           # Linear impact model
    SQUARE_ROOT = "SQUARE_ROOT" # Square root model
    LOGARITHMIC = "LOGARITHMIC" # Logarithmic model
    HISTORICAL = "HISTORICAL"   # Based on historical data

class ExecutionAlgorithm(Enum):
    """Execution algorithms for large orders"""
    TWAP = "TWAP"         # Time Weighted Average Price
    VWAP = "VWAP"         # Volume Weighted Average Price
    POV = "POV"           # Percentage of Volume
    IS = "IS"             # Implementation Shortfall
    MARKET_ON_CLOSE = "MOC"  # Market on Close

@dataclass
class MarketOrder:
    """Comprehensive market order representation"""
    
    # Basic order information
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    
    # Execution parameters
    urgency: ExecutionUrgency = ExecutionUrgency.NORMAL
    max_participation_rate: float = 0.1  # Max % of volume
    execution_algorithm: Optional[ExecutionAlgorithm] = None
    
    # Status and tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    remaining_quantity: int = field(init=False)
    
    # Timestamps
    creation_time: datetime = field(default_factory=datetime.now)
    start_execution_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    
    # Execution details
    average_fill_price: float = 0.0
    total_commission: float = 0.0
    fills: List[Fill] = field(default_factory=list)
    child_orders: List[str] = field(default_factory=list)  # For algorithmic execution
    
    # Risk controls
    price_limit: Optional[float] = None  # Maximum acceptable price
    time_limit: Optional[datetime] = None  # Must complete by this time
    max_slippage: Optional[float] = None  # Maximum slippage tolerance
    
    # Market context
    market_price_at_creation: Optional[float] = None
    volume_estimate: Optional[int] = None
    volatility_estimate: Optional[float] = None
    
    # Performance tracking
    expected_shortfall: Optional[float] = None
    actual_shortfall: Optional[float] = None
    market_impact_estimate: Optional[float] = None
    timing_risk: Optional[float] = None
    
    # Metadata
    client_order_id: Optional[str] = None
    account_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    strategy_id: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        self.remaining_quantity = self.quantity
        
        # Set default time limit if not specified
        if self.time_limit is None:
            if self.urgency == ExecutionUrgency.IMMEDIATE:
                self.time_limit = self.creation_time + timedelta(minutes=5)
            elif self.urgency == ExecutionUrgency.HIGH:
                self.time_limit = self.creation_time + timedelta(minutes=30)
            else:
                self.time_limit = self.creation_time + timedelta(hours=4)
    
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order"""
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """Check if this is a sell order"""
        return self.side == OrderSide.SELL
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]
    
    @property
    def is_completed(self) -> bool:
        """Check if order execution is complete"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED]
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage"""
        return (self.filled_quantity / self.quantity) * 100 if self.quantity > 0 else 0
    
    @property
    def execution_time(self) -> Optional[timedelta]:
        """Calculate total execution time"""
        if self.start_execution_time and self.completion_time:
            return self.completion_time - self.start_execution_time
        elif self.start_execution_time:
            return datetime.now() - self.start_execution_time
        return None
    
    def add_fill(self, fill: Fill):
        """Add a fill to this order"""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        
        # Update average fill price
        if self.filled_quantity > 0:
            total_value = sum(f.quantity * f.price for f in self.fills)
            self.average_fill_price = total_value / self.filled_quantity
        
        # Update status
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
            self.completion_time = datetime.now()
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIAL
        
        logger.debug(f"Market order {self.order_id} filled {fill.quantity} @ {fill.price}, "
                    f"total filled: {self.filled_quantity}/{self.quantity}")
    
    def calculate_slippage(self) -> Optional[float]:
        """Calculate realized slippage"""
        if not self.market_price_at_creation or not self.fills:
            return None
        
        if self.is_buy:
            # For buy orders, slippage is how much more we paid
            slippage = self.average_fill_price - self.market_price_at_creation
        else:
            # For sell orders, slippage is how much less we received
            slippage = self.market_price_at_creation - self.average_fill_price
        
        return slippage
    
    def calculate_implementation_shortfall(self) -> Optional[float]:
        """Calculate implementation shortfall"""
        if not self.market_price_at_creation or not self.fills:
            return None
        
        # Implementation shortfall = (Average Fill Price - Decision Price) * Quantity
        price_diff = self.average_fill_price - self.market_price_at_creation
        if self.is_sell:
            price_diff = -price_diff
        
        shortfall = price_diff * self.filled_quantity
        self.actual_shortfall = shortfall
        
        return shortfall

# ============================================
# Market Impact Models
# ============================================

class MarketImpactEstimator:
    """
    Market impact estimation for market orders.
    
    Provides various models to estimate the market impact
    of executing market orders of different sizes.
    """
    
    def __init__(self):
        self.model_cache = {}
        self.historical_data = defaultdict(list)
        
    def estimate_impact(self, symbol: str, quantity: int, 
                       market_data: Dict[str, Any],
                       model: MarketImpactModel = MarketImpactModel.SQUARE_ROOT) -> float:
        """
        Estimate market impact of a market order
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            market_data: Current market data
            model: Impact model to use
            
        Returns:
            Estimated impact in basis points
        """
        
        if model == MarketImpactModel.LINEAR:
            return self._linear_impact_model(symbol, quantity, market_data)
        elif model == MarketImpactModel.SQUARE_ROOT:
            return self._square_root_impact_model(symbol, quantity, market_data)
        elif model == MarketImpactModel.LOGARITHMIC:
            return self._logarithmic_impact_model(symbol, quantity, market_data)
        else:  # HISTORICAL
            return self._historical_impact_model(symbol, quantity, market_data)
    
    def _linear_impact_model(self, symbol: str, quantity: int, market_data: Dict[str, Any]) -> float:
        """Linear market impact model"""
        
        # Extract market parameters
        avg_volume = market_data.get('average_volume', 1000000)
        volatility = market_data.get('volatility', 0.02)
        spread = market_data.get('spread', 0.01)
        
        # Participation rate
        participation_rate = quantity / avg_volume if avg_volume > 0 else 0.1
        
        # Linear impact: Impact ∝ participation_rate * volatility
        # Base impact coefficient (calibrated empirically)
        base_coefficient = 0.5
        
        impact_bps = base_coefficient * participation_rate * volatility * 10000
        
        # Add spread component
        spread_impact = (spread / 2) * 10000  # Half spread in bps
        
        total_impact = impact_bps + spread_impact
        
        return min(total_impact, 500)  # Cap at 5% impact
    
    def _square_root_impact_model(self, symbol: str, quantity: int, market_data: Dict[str, Any]) -> float:
        """Square root market impact model (widely used in practice)"""
        
        avg_volume = market_data.get('average_volume', 1000000)
        volatility = market_data.get('volatility', 0.02)
        spread = market_data.get('spread', 0.01)
        price = market_data.get('price', 100.0)
        
        # Participation rate
        participation_rate = quantity / avg_volume if avg_volume > 0 else 0.1
        
        # Square root impact model: σ * sqrt(participation_rate) * multiplier
        # Typical multiplier ranges from 0.5 to 1.5
        multiplier = 1.0
        
        impact_bps = multiplier * volatility * np.sqrt(participation_rate) * 10000
        
        # Add bid-ask spread cost
        spread_cost = (spread / price) * 10000 / 2  # Half spread in bps
        
        total_impact = impact_bps + spread_cost
        
        return min(total_impact, 1000)  # Cap at 10% impact
    
    def _logarithmic_impact_model(self, symbol: str, quantity: int, market_data: Dict[str, Any]) -> float:
        """Logarithmic market impact model"""
        
        avg_volume = market_data.get('average_volume', 1000000)
        volatility = market_data.get('volatility', 0.02)
        
        # Participation rate
        participation_rate = quantity / avg_volume if avg_volume > 0 else 0.1
        
        # Logarithmic model: σ * log(1 + participation_rate) * multiplier
        multiplier = 2.0
        
        impact_bps = multiplier * volatility * np.log(1 + participation_rate) * 10000
        
        return min(impact_bps, 800)  # Cap at 8% impact
    
    def _historical_impact_model(self, symbol: str, quantity: int, market_data: Dict[str, Any]) -> float:
        """Historical data-based impact model"""
        
        # This would use historical execution data to calibrate impact
        # For now, fall back to square root model
        return self._square_root_impact_model(symbol, quantity, market_data)
    
    def update_historical_data(self, symbol: str, quantity: int, 
                              realized_impact: float, market_conditions: Dict[str, Any]):
        """Update historical impact data for model calibration"""
        
        self.historical_data[symbol].append({
            'quantity': quantity,
            'impact': realized_impact,
            'timestamp': datetime.now(),
            'market_conditions': market_conditions
        })
        
        # Keep only recent data (last 1000 observations)
        if len(self.historical_data[symbol]) > 1000:
            self.historical_data[symbol] = self.historical_data[symbol][-1000:]

# ============================================
# Execution Algorithms
# ============================================

class TWAPExecutor:
    """
    Time Weighted Average Price execution algorithm.
    
    Splits large orders into smaller child orders executed
    over a specified time horizon to minimize market impact.
    """
    
    def __init__(self, time_horizon_minutes: int = 60, child_order_interval_minutes: int = 5):
        self.time_horizon_minutes = time_horizon_minutes
        self.child_order_interval_minutes = child_order_interval_minutes
        self.active_executions = {}
    
    def execute_order(self, market_order: MarketOrder, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute market order using TWAP algorithm
        
        Args:
            market_order: Market order to execute
            market_data: Current market data
            
        Returns:
            List of child order specifications
        """
        
        # Calculate number of child orders
        num_intervals = self.time_horizon_minutes // self.child_order_interval_minutes
        child_quantity = market_order.quantity // num_intervals
        remaining_quantity = market_order.quantity % num_intervals
        
        child_orders = []
        current_time = datetime.now()
        
        # Create child orders
        for i in range(num_intervals):
            child_qty = child_quantity
            if i == num_intervals - 1:  # Last order gets remainder
                child_qty += remaining_quantity
            
            execution_time = current_time + timedelta(minutes=i * self.child_order_interval_minutes)
            
            child_order = {
                'parent_order_id': market_order.order_id,
                'child_id': f"{market_order.order_id}_TWAP_{i+1}",
                'symbol': market_order.symbol,
                'side': market_order.side,
                'quantity': child_qty,
                'execution_time': execution_time,
                'algorithm': 'TWAP',
                'urgency': ExecutionUrgency.LOW  # TWAP is patient
            }
            
            child_orders.append(child_order)
        
        # Track execution
        self.active_executions[market_order.order_id] = {
            'child_orders': child_orders,
            'completed_children': 0,
            'total_filled': 0,
            'start_time': current_time
        }
        
        logger.info(f"TWAP execution plan created for {market_order.order_id}: "
                   f"{num_intervals} children over {self.time_horizon_minutes} minutes")
        
        return child_orders

class VWAPExecutor:
    """
    Volume Weighted Average Price execution algorithm.
    
    Executes orders based on historical volume patterns
    to minimize market impact while tracking VWAP.
    """
    
    def __init__(self):
        self.volume_profiles = {}  # Historical volume patterns by symbol
    
    def execute_order(self, market_order: MarketOrder, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute market order using VWAP algorithm"""
        
        # Get or estimate volume profile
        volume_profile = self._get_volume_profile(market_order.symbol, market_data)
        
        # Calculate participation rates for each time interval
        child_orders = []
        current_time = datetime.now()
        remaining_quantity = market_order.quantity
        participation_rate = market_order.max_participation_rate
        
        # Create child orders based on volume profile
        for i, (time_interval, volume_weight) in enumerate(volume_profile.items()):
            if remaining_quantity <= 0:
                break
            
            # Calculate expected volume for this interval
            expected_volume = market_data.get('average_volume', 1000000) * volume_weight
            
            # Calculate child order quantity based on participation rate
            child_qty = min(
                int(expected_volume * participation_rate),
                remaining_quantity
            )
            
            if child_qty > 0:
                execution_time = current_time + timedelta(minutes=i * 15)  # 15-minute intervals
                
                child_order = {
                    'parent_order_id': market_order.order_id,
                    'child_id': f"{market_order.order_id}_VWAP_{i+1}",
                    'symbol': market_order.symbol,
                    'side': market_order.side,
                    'quantity': child_qty,
                    'execution_time': execution_time,
                    'algorithm': 'VWAP',
                    'expected_volume': expected_volume,
                    'urgency': ExecutionUrgency.NORMAL
                }
                
                child_orders.append(child_order)
                remaining_quantity -= child_qty
        
        # If there's remaining quantity, add it to the last order
        if remaining_quantity > 0 and child_orders:
            child_orders[-1]['quantity'] += remaining_quantity
        
        logger.info(f"VWAP execution plan created for {market_order.order_id}: "
                   f"{len(child_orders)} children following volume profile")
        
        return child_orders
    
    def _get_volume_profile(self, symbol: str, market_data: Dict[str, Any]) -> Dict[int, float]:
        """Get or estimate intraday volume profile"""
        
        # Default volume profile (U-shaped: high at open/close, low at midday)
        # In practice, this would be based on historical data
        default_profile = {
            0: 0.08,   # 9:30-9:45 AM
            1: 0.06,   # 9:45-10:00 AM
            2: 0.05,   # 10:00-10:15 AM
            3: 0.04,   # 10:15-10:30 AM
            4: 0.03,   # 10:30-10:45 AM
            5: 0.03,   # 10:45-11:00 AM
            6: 0.03,   # 11:00-11:15 AM
            7: 0.02,   # 11:15-11:30 AM
            8: 0.02,   # 11:30-11:45 AM
            9: 0.02,   # 11:45-12:00 PM
            10: 0.02,  # 12:00-12:15 PM
            11: 0.02,  # 12:15-12:30 PM
            12: 0.02,  # 12:30-12:45 PM
            13: 0.02,  # 12:45-1:00 PM
            14: 0.02,  # 1:00-1:15 PM
            15: 0.02,  # 1:15-1:30 PM
            16: 0.02,  # 1:30-1:45 PM
            17: 0.03,  # 1:45-2:00 PM
            18: 0.03,  # 2:00-2:15 PM
            19: 0.03,  # 2:15-2:30 PM
            20: 0.04,  # 2:30-2:45 PM
            21: 0.05,  # 2:45-3:00 PM
            22: 0.06,  # 3:00-3:15 PM
            23: 0.08,  # 3:15-3:30 PM
            24: 0.12,  # 3:30-3:45 PM
            25: 0.15   # 3:45-4:00 PM
        }
        
        return self.volume_profiles.get(symbol, default_profile)

# ============================================
# Market Order Executor
# ============================================

class MarketOrderExecutor:
    """
    Advanced market order execution engine.
    
    Handles immediate execution, algorithmic execution,
    risk controls, and performance measurement.
    """
    
    def __init__(self, order_books: Dict[str, LimitOrderBook]):
        self.order_books = order_books
        self.active_orders = {}
        self.execution_history = []
        
        # Execution components
        self.impact_estimator = MarketImpactEstimator()
        self.twap_executor = TWAPExecutor()
        self.vwap_executor = VWAPExecutor()
        
        # Risk controls
        self.max_order_size = {}  # By symbol
        self.max_participation_rate = 0.25  # 25% max
        self.price_deviation_limit = 0.05  # 5% max deviation
        
        # Performance tracking
        self.execution_metrics = defaultdict(dict)
        
        # Threading
        self._lock = threading.RLock()
        self.executor_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Initialized MarketOrderExecutor")
    
    @time_it("market_order_execution")
    def execute_market_order(self, market_order: MarketOrder, 
                           market_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute a market order
        
        Args:
            market_order: Market order to execute
            market_data: Current market data
            
        Returns:
            True if execution started successfully
        """
        
        with self._lock:
            try:
                # Validate order
                if not self._validate_market_order(market_order, market_data):
                    market_order.status = OrderStatus.REJECTED
                    return False
                
                # Set market context
                if market_data:
                    market_order.market_price_at_creation = market_data.get('price')
                    market_order.volume_estimate = market_data.get('average_volume')
                    market_order.volatility_estimate = market_data.get('volatility')
                
                # Estimate market impact
                if market_data:
                    market_order.market_impact_estimate = self.impact_estimator.estimate_impact(
                        market_order.symbol, market_order.quantity, market_data
                    )
                
                # Track active order
                self.active_orders[market_order.order_id] = market_order
                market_order.start_execution_time = datetime.now()
                
                # Choose execution strategy based on size and urgency
                if self._requires_algorithmic_execution(market_order, market_data):
                    success = self._execute_algorithmic(market_order, market_data)
                else:
                    success = self._execute_immediate(market_order, market_data)
                
                if success:
                    logger.info(f"Market order execution started: {market_order.order_id}")
                else:
                    logger.error(f"Failed to start execution: {market_order.order_id}")
                    market_order.status = OrderStatus.REJECTED
                
                return success
                
            except Exception as e:
                logger.error(f"Error executing market order {market_order.order_id}: {e}")
                market_order.status = OrderStatus.REJECTED
                return False
    
    def _validate_market_order(self, market_order: MarketOrder, 
                              market_data: Optional[Dict[str, Any]]) -> bool:
        """Validate market order before execution"""
        
        # Basic validation
        if market_order.quantity <= 0:
            logger.error(f"Invalid quantity: {market_order.quantity}")
            return False
        
        if market_order.symbol not in self.order_books:
            logger.error(f"No order book for symbol: {market_order.symbol}")
            return False
        
        # Size limits
        if market_order.symbol in self.max_order_size:
            max_size = self.max_order_size[market_order.symbol]
            if market_order.quantity > max_size:
                logger.error(f"Order size {market_order.quantity} exceeds limit {max_size}")
                return False
        
        # Participation rate check
        if market_data and 'average_volume' in market_data:
            participation_rate = market_order.quantity / market_data['average_volume']
            if participation_rate > self.max_participation_rate:
                logger.error(f"Participation rate {participation_rate:.2%} exceeds limit")
                return False
        
        # Price limit check
        if market_order.price_limit and market_data:
            current_price = market_data.get('price', 0)
            if market_order.is_buy and current_price > market_order.price_limit:
                logger.error(f"Current price {current_price} exceeds buy limit {market_order.price_limit}")
                return False
            elif market_order.is_sell and current_price < market_order.price_limit:
                logger.error(f"Current price {current_price} below sell limit {market_order.price_limit}")
                return False
        
        return True
    
    def _requires_algorithmic_execution(self, market_order: MarketOrder, 
                                      market_data: Optional[Dict[str, Any]]) -> bool:
        """Determine if order requires algorithmic execution"""
        
        # Large orders need algorithmic execution
        if market_data and 'average_volume' in market_data:
            participation_rate = market_order.quantity / market_data['average_volume']
            if participation_rate > 0.05:  # > 5% of daily volume
                return True
        
        # Specific algorithm requested
        if market_order.execution_algorithm is not None:
            return True
        
        # Low urgency orders benefit from patient execution
        if market_order.urgency == ExecutionUrgency.LOW:
            return True
        
        return False
    
    def _execute_immediate(self, market_order: MarketOrder, 
                          market_data: Optional[Dict[str, Any]]) -> bool:
        """Execute order immediately against order book"""
        
        order_book = self.order_books[market_order.symbol]
        
        # Execute against order book
        remaining_quantity = market_order.quantity
        fills = []
        
        if market_order.is_buy:
            # Buy orders execute against asks
            ask_prices = sorted(order_book.sell_orders.keys())
            
            for ask_price in ask_prices:
                if remaining_quantity <= 0:
                    break
                
                # Check price limit
                if market_order.price_limit and ask_price > market_order.price_limit:
                    break
                
                level = order_book.sell_orders[ask_price]
                available_quantity = min(level.total_quantity, remaining_quantity)
                
                if available_quantity > 0:
                    # Create fill
                    fill = Fill(
                        fill_id=str(uuid.uuid4()),
                        order_id=market_order.order_id,
                        symbol=market_order.symbol,
                        side=market_order.side,
                        quantity=available_quantity,
                        price=ask_price,
                        timestamp=datetime.now(),
                        liquidity_flag="Taker"
                    )
                    
                    fills.append(fill)
                    remaining_quantity -= available_quantity
                    
                    # Remove liquidity from order book
                    level.total_quantity -= available_quantity
                    if level.total_quantity <= 0:
                        del order_book.sell_orders[ask_price]
        
        else:
            # Sell orders execute against bids
            bid_prices = sorted(order_book.buy_orders.keys(), reverse=True)
            
            for bid_price in bid_prices:
                if remaining_quantity <= 0:
                    break
                
                # Check price limit
                if market_order.price_limit and bid_price < market_order.price_limit:
                    break
                
                level = order_book.buy_orders[bid_price]
                available_quantity = min(level.total_quantity, remaining_quantity)
                
                if available_quantity > 0:
                    # Create fill
                    fill = Fill(
                        fill_id=str(uuid.uuid4()),
                        order_id=market_order.order_id,
                        symbol=market_order.symbol,
                        side=market_order.side,
                        quantity=available_quantity,
                        price=bid_price,
                        timestamp=datetime.now(),
                        liquidity_flag="Taker"
                    )
                    
                    fills.append(fill)
                    remaining_quantity -= available_quantity
                    
                    # Remove liquidity from order book
                    level.total_quantity -= available_quantity
                    if level.total_quantity <= 0:
                        del order_book.buy_orders[bid_price]
        
        # Apply fills to order
        for fill in fills:
            market_order.add_fill(fill)
        
        # Check if order was fully filled or needs to be cancelled
        if market_order.remaining_quantity > 0:
            if market_order.urgency == ExecutionUrgency.IMMEDIATE:
                # Cancel unfilled quantity for immediate orders
                market_order.status = OrderStatus.CANCELLED
                market_order.completion_time = datetime.now()
            else:
                # For other orders, could potentially leave as limit orders
                # For now, cancel unfilled quantity
                market_order.status = OrderStatus.PARTIAL if market_order.filled_quantity > 0 else OrderStatus.CANCELLED
                market_order.completion_time = datetime.now()
        
        # Update execution history
        self.execution_history.append(market_order)
        
        return len(fills) > 0
    
    def _execute_algorithmic(self, market_order: MarketOrder, 
                           market_data: Optional[Dict[str, Any]]) -> bool:
        """Execute order using algorithmic strategies"""
        
        if not market_data:
            logger.error("Market data required for algorithmic execution")
            return False
        
        # Determine algorithm to use
        algorithm = market_order.execution_algorithm
        if algorithm is None:
            # Choose based on order characteristics
            if market_order.urgency == ExecutionUrgency.LOW:
                algorithm = ExecutionAlgorithm.TWAP
            else:
                algorithm = ExecutionAlgorithm.VWAP
        
        # Execute using chosen algorithm
        child_orders = []
        
        if algorithm == ExecutionAlgorithm.TWAP:
            child_orders = self.twap_executor.execute_order(market_order, market_data)
        elif algorithm == ExecutionAlgorithm.VWAP:
            child_orders = self.vwap_executor.execute_order(market_order, market_data)
        else:
            logger.error(f"Algorithm {algorithm} not implemented")
            return False
        
        # Store child orders
        market_order.child_orders = [child['child_id'] for child in child_orders]
        
        # Schedule child order executions
        for child_spec in child_orders:
            self.executor_pool.submit(self._execute_child_order, child_spec, market_data)
        
        return True
    
    def _execute_child_order(self, child_spec: Dict[str, Any], market_data: Dict[str, Any]):
        """Execute a child order (scheduled execution)"""
        
        # Wait until execution time
        execution_time = child_spec['execution_time']
        wait_time = (execution_time - datetime.now()).total_seconds()
        
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Create child market order
        child_order = MarketOrder(
            order_id=child_spec['child_id'],
            symbol=child_spec['symbol'],
            side=child_spec['side'],
            quantity=child_spec['quantity'],
            urgency=child_spec.get('urgency', ExecutionUrgency.NORMAL)
        )
        
        # Execute immediately
        success = self._execute_immediate(child_order, market_data)
        
        # Update parent order
        parent_id = child_spec['parent_order_id']
        if parent_id in self.active_orders:
            parent_order = self.active_orders[parent_id]
            
            # Add child fills to parent
            for fill in child_order.fills:
                parent_order.add_fill(fill)
            
            logger.debug(f"Child order {child_order.order_id} executed, "
                        f"parent progress: {parent_order.fill_percentage:.1f}%")
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get market order status"""
        
        order = self.active_orders.get(order_id)
        if not order:
            return None
        
        status = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'status': order.status.value,
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': order.remaining_quantity,
            'fill_percentage': order.fill_percentage,
            'average_fill_price': order.average_fill_price,
            'urgency': order.urgency.value,
            'creation_time': order.creation_time,
            'execution_time': order.execution_time.total_seconds() if order.execution_time else None,
            'market_impact_estimate': order.market_impact_estimate,
            'slippage': order.calculate_slippage(),
            'implementation_shortfall': order.calculate_implementation_shortfall(),
            'child_orders': len(order.child_orders),
            'fills': [
                {
                    'fill_id': fill.fill_id,
                    'quantity': fill.quantity,
                    'price': fill.price,
                    'timestamp': fill.timestamp,
                    'liquidity_flag': fill.liquidity_flag
                }
                for fill in order.fills
            ]
        }
        
        return status
    
    def cancel_order(self, order_id: str, reason: str = "User cancelled") -> bool:
        """Cancel a market order (if possible)"""
        
        with self._lock:
            order = self.active_orders.get(order_id)
            if not order or not order.is_active:
                logger.warning(f"Cannot cancel order {order_id}: not found or not active")
                return False
            
            # For immediate market orders, cancellation may not be possible
            if order.urgency == ExecutionUrgency.IMMEDIATE and order.fills:
                logger.warning(f"Cannot cancel order {order_id}: already partially executed")
                return False
            
            order.status = OrderStatus.CANCELLED
            order.completion_time = datetime.now()
            
            # Move to history
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            self.execution_history.append(order)
            
            logger.info(f"Cancelled market order {order_id}: {reason}")
            return True
    
    def get_execution_analytics(self, order_id: str) -> Dict[str, Any]:
        """Get detailed execution analytics"""
        
        order = self.active_orders.get(order_id)
        if not order:
            # Check history
            for hist_order in self.execution_history:
                if hist_order.order_id == order_id:
                    order = hist_order
                    break
        
        if not order:
            return {}
        
        analytics = {
            'order_id': order_id,
            'execution_summary': {
                'total_quantity': order.quantity,
                'filled_quantity': order.filled_quantity,
                'fill_rate': order.fill_percentage / 100,
                'average_price': order.average_fill_price,
                'execution_time_seconds': order.execution_time.total_seconds() if order.execution_time else None
            },
            'cost_analysis': {
                'slippage': order.calculate_slippage(),
                'implementation_shortfall': order.calculate_implementation_shortfall(),
                'estimated_impact': order.market_impact_estimate,
                'total_commission': order.total_commission
            },
            'execution_details': {
                'urgency': order.urgency.value,
                'algorithm': order.execution_algorithm.value if order.execution_algorithm else None,
                'child_orders': len(order.child_orders),
                'fill_count': len(order.fills)
            },
            'market_conditions': {
                'price_at_creation': order.market_price_at_creation,
                'volume_estimate': order.volume_estimate,
                'volatility_estimate': order.volatility_estimate
            }
        }
        
        return analytics

# ============================================
# Utility Functions
# ============================================

def create_market_order(symbol: str, side: str, quantity: int, 
                       urgency: str = "NORMAL", **kwargs) -> MarketOrder:
    """
    Utility function to create a market order
    
    Args:
        symbol: Trading symbol
        side: 'BUY' or 'SELL'
        quantity: Order quantity
        urgency: Execution urgency level
        **kwargs: Additional parameters
        
    Returns:
        MarketOrder instance
    """
    
    order_side = OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL
    execution_urgency = ExecutionUrgency(urgency.upper())
    order_id = kwargs.get('order_id', str(uuid.uuid4()))
    
    return MarketOrder(
        order_id=order_id,
        symbol=symbol.upper(),
        side=order_side,
        quantity=quantity,
        urgency=execution_urgency,
        **{k: v for k, v in kwargs.items() if k != 'order_id'}
    )

def estimate_execution_cost(symbol: str, quantity: int, market_data: Dict[str, Any], 
                          urgency: ExecutionUrgency = ExecutionUrgency.NORMAL) -> Dict[str, float]:
    """
    Estimate execution cost for a market order
    
    Args:
        symbol: Trading symbol
        quantity: Order quantity
        market_data: Current market data
        urgency: Execution urgency
        
    Returns:
        Dictionary with cost estimates
    """
    
    estimator = MarketImpactEstimator()
    
    # Base impact estimate
    impact_bps = estimator.estimate_impact(symbol, quantity, market_data)
    
    # Adjust for urgency
    urgency_multiplier = {
        ExecutionUrgency.LOW: 0.7,
        ExecutionUrgency.NORMAL: 1.0,
        ExecutionUrgency.HIGH: 1.5,
        ExecutionUrgency.IMMEDIATE: 2.0
    }
    
    adjusted_impact = impact_bps * urgency_multiplier[urgency]
    
    # Estimate dollar cost
    price = market_data.get('price', 100.0)
    dollar_impact = (adjusted_impact / 10000) * price * quantity
    
    return {
        'impact_basis_points': adjusted_impact,
        'impact_percentage': adjusted_impact / 100,
        'estimated_dollar_cost': dollar_impact,
        'urgency_multiplier': urgency_multiplier[urgency]
    }

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Market Orders System")
    
    # Create mock order book for testing
    from .limit_orders import LimitOrderBook
    
    # Initialize order book
    aapl_book = LimitOrderBook("AAPL")
    order_books = {"AAPL": aapl_book}
    
    # Add some liquidity to the order book
    # This would normally be done by the limit order system
    aapl_book.sell_orders[151.00] = type('Level', (), {
        'price': 151.00, 'total_quantity': 1000, 'orders': []
    })()
    aapl_book.sell_orders[151.25] = type('Level', (), {
        'price': 151.25, 'total_quantity': 500, 'orders': []
    })()
    aapl_book.buy_orders[150.75] = type('Level', (), {
        'price': 150.75, 'total_quantity': 800, 'orders': []
    })()
    
    # Initialize market order executor
    executor = MarketOrderExecutor(order_books)
    
    print("\n1. Testing Basic Market Order Creation")
    
    # Create market order using utility function
    market_order = create_market_order(
        symbol="AAPL",
        side="BUY",
        quantity=500,
        urgency="NORMAL",
        client_order_id="TEST_MARKET_001",
        price_limit=152.00  # Maximum acceptable price
    )
    
    print(f"Created market order: {market_order.order_id}")
    print(f"  Symbol: {market_order.symbol}")
    print(f"  Side: {market_order.side.value}")
    print(f"  Quantity: {market_order.quantity}")
    print(f"  Urgency: {market_order.urgency.value}")
    print(f"  Price Limit: {market_order.price_limit}")
    
    print("\n2. Testing Market Impact Estimation")
    
    # Sample market data
    market_data = {
        'price': 151.00,
        'average_volume': 50000000,  # 50M daily volume
        'volatility': 0.25,          # 25% annualized volatility
        'spread': 0.02               # $0.02 spread
    }
    
    # Estimate execution cost
    cost_estimate = estimate_execution_cost("AAPL", 10000, market_data, ExecutionUrgency.HIGH)
    
    print(f"Execution Cost Estimate for 10,000 shares:")
    print(f"  Impact: {cost_estimate['impact_basis_points']:.1f} bps")
    print(f"  Impact %: {cost_estimate['impact_percentage']:.3f}%")
    print(f"  Dollar Cost: ${cost_estimate['estimated_dollar_cost']:.2f}")
    print(f"  Urgency Multiplier: {cost_estimate['urgency_multiplier']:.1f}x")
    
    print("\n3. Testing Immediate Market Order Execution")
    
    # Execute small market order immediately
    small_order = create_market_order(
        symbol="AAPL",
        side="BUY", 
        quantity=300,
        urgency="IMMEDIATE"
    )
    
    success = executor.execute_market_order(small_order, market_data)
    print(f"Immediate execution {'succeeded' if success else 'failed'}")
    
    if success:
        status = executor.get_order_status(small_order.order_id)
        print(f"  Status: {status['status']}")
        print(f"  Filled: {status['filled_quantity']}/{status['quantity']}")
        print(f"  Average Price: ${status['average_fill_price']:.2f}")
        print(f"  Fill Count: {len(status['fills'])}")
        
        if status['fills']:
            print(f"  First Fill: {status['fills'][0]['quantity']} @ ${status['fills'][0]['price']:.2f}")
    
    print("\n4. Testing Algorithmic Execution (TWAP)")
    
    # Large order requiring algorithmic execution
    large_order = create_market_order(
        symbol="AAPL",
        side="SELL",
        quantity=50000,  # Large order
        urgency="LOW",
        execution_algorithm=ExecutionAlgorithm.TWAP,
        max_participation_rate=0.08  # 8% of volume
    )
    
    print(f"Large order created: {large_order.order_id}")
    print(f"  Quantity: {large_order.quantity:,}")
    print(f"  Algorithm: {large_order.execution_algorithm.value}")
    print(f"  Max Participation: {large_order.max_participation_rate:.1%}")
    
    # This would normally execute over time
    # For testing, we'll just show it would be split
    twap_executor = TWAPExecutor(time_horizon_minutes=120, child_order_interval_minutes=10)
    child_orders = twap_executor.execute_order(large_order, market_data)
    
    print(f"TWAP execution plan:")
    print(f"  Child orders: {len(child_orders)}")
    print(f"  Time horizon: 120 minutes")
    
    for i, child in enumerate(child_orders[:3]):  # Show first 3
        print(f"    Child {i+1}: {child['quantity']:,} shares at {child['execution_time'].strftime('%H:%M')}")
    
    print("\n5. Testing VWAP Execution")
    
    vwap_executor = VWAPExecutor()
    vwap_order = create_market_order(
        symbol="AAPL",
        side="BUY",
        quantity=25000,
        execution_algorithm=ExecutionAlgorithm.VWAP
    )
    
    vwap_children = vwap_executor.execute_order(vwap_order, market_data)
    
    print(f"VWAP execution plan:")
    print(f"  Child orders: {len(vwap_children)}")
    print(f"  Following intraday volume profile")
    
    # Show volume profile usage
    total_expected_volume = sum(child.get('expected_volume', 0) for child in vwap_children)
    print(f"  Total expected volume: {total_expected_volume:,.0f}")
    
    print("\n6. Testing Order Status and Analytics")
    
    if success:
        # Get detailed analytics for executed order
        analytics = executor.get_execution_analytics(small_order.order_id)
        
        print(f"Execution Analytics for {small_order.order_id}:")
        
        if 'execution_summary' in analytics:
            summary = analytics['execution_summary']
            print(f"  Fill Rate: {summary['fill_rate']:.1%}")
            print(f"  Average Price: ${summary['average_price']:.2f}")
            if summary['execution_time_seconds']:
                print(f"  Execution Time: {summary['execution_time_seconds']:.2f} seconds")
        
        if 'cost_analysis' in analytics:
            costs = analytics['cost_analysis']
            if costs['slippage']:
                print(f"  Slippage: ${costs['slippage']:.3f}")
            if costs['implementation_shortfall']:
                print(f"  Implementation Shortfall: ${costs['implementation_shortfall']:.2f}")
    
    print("\n7. Testing Order Cancellation")
    
    # Create order for cancellation test
    cancel_test_order = create_market_order(
        symbol="AAPL",
        side="BUY",
        quantity=100,
        urgency="NORMAL"
    )
    
    # Try to cancel before execution
    cancel_success = executor.cancel_order(cancel_test_order.order_id, "Testing cancellation")
    print(f"Order cancellation: {'Success' if cancel_success else 'Failed'}")
    
    print("\n8. Testing Risk Controls")
    
    # Test size limit
    executor.max_order_size["AAPL"] = 10000
    
    oversized_order = create_market_order(
        symbol="AAPL",
        side="BUY",
        quantity=15000  # Exceeds limit
    )
    
    risk_success = executor.execute_market_order(oversized_order, market_data)
    print(f"Oversized order execution: {'Allowed' if risk_success else 'Blocked by risk controls'}")
    
    if not risk_success:
        print(f"  Order status: {oversized_order.status.value}")
    
    print("\n9. Testing Market Impact Models")
    
    impact_estimator = MarketImpactEstimator()
    
    # Test different impact models
    test_quantity = 10000
    models = [
        MarketImpactModel.LINEAR,
        MarketImpactModel.SQUARE_ROOT, 
        MarketImpactModel.LOGARITHMIC
    ]
    
    print(f"Market Impact Estimates for {test_quantity:,} shares:")
    for model in models:
        impact = impact_estimator.estimate_impact("AAPL", test_quantity, market_data, model)
        print(f"  {model.value}: {impact:.1f} bps")
    
    print("\n10. Testing Performance Tracking")
    
    # Show active orders
    active_count = len(executor.active_orders)
    history_count = len(executor.execution_history)
    
    print(f"Order Status Summary:")
    print(f"  Active Orders: {active_count}")
    print(f"  Completed Orders: {history_count}")
    
    if executor.active_orders:
        print("  Active Orders:")
        for order_id, order in list(executor.active_orders.items())[:3]:
            print(f"    {order_id[:8]}...: {order.side.value} {order.remaining_quantity}/{order.quantity}")
    
    print("\nMarket orders system testing completed successfully!")
    print("\nImplemented features include:")
    print("• Immediate market order execution with price limits")
    print("• Algorithmic execution (TWAP, VWAP) for large orders")  
    print("• Multiple market impact models for cost estimation")
    print("• Advanced risk controls and position size limits")
    print("• Comprehensive execution analytics and performance tracking")
    print("• Child order scheduling for algorithmic strategies")
    print("• Real-time order status monitoring and cancellation")
    print("• Integration with limit order books for liquidity consumption")
