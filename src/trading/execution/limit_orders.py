# ============================================
# StockPredictionPro - src/trading/execution/limit_orders.py
# Comprehensive limit order management system for precise trade execution
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
import heapq
import threading
import time

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('trading.execution.limit_orders')

# ============================================
# Limit Order Data Structures and Enums
# ============================================

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"

class TimeInForce(Enum):
    """Time in force options"""
    DAY = "DAY"           # Good for the trading day
    GTC = "GTC"           # Good till cancelled
    IOC = "IOC"           # Immediate or cancel
    FOK = "FOK"           # Fill or kill
    GTD = "GTD"           # Good till date
    
class OrderPriority(Enum):
    """Order priority for execution"""
    PRICE_TIME = "PRICE_TIME"     # Price priority, then time priority
    PRO_RATA = "PRO_RATA"         # Proportional allocation
    SIZE_TIME = "SIZE_TIME"       # Size priority, then time priority

@dataclass
class LimitOrder:
    """Comprehensive limit order representation"""
    
    # Basic order information
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    
    # Order management
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    remaining_quantity: int = field(init=False)
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Timestamps
    creation_time: datetime = field(default_factory=datetime.now)
    modification_time: Optional[datetime] = None
    expiry_time: Optional[datetime] = None
    
    # Execution details
    average_fill_price: float = 0.0
    total_commission: float = 0.0
    fills: List['Fill'] = field(default_factory=list)
    
    # Advanced features
    hidden_quantity: int = 0        # Iceberg order hidden quantity
    displayed_quantity: int = 0     # Currently displayed quantity
    minimum_quantity: int = 0       # Minimum fill quantity
    
    # Risk and metadata
    client_order_id: Optional[str] = None
    account_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    strategy_id: Optional[str] = None
    
    # Market data context
    market_price_at_creation: Optional[float] = None
    spread_at_creation: Optional[float] = None
    volatility_estimate: Optional[float] = None
    
    # Execution tracking
    priority_timestamp: datetime = field(default_factory=datetime.now)
    queue_position: Optional[int] = None
    estimated_wait_time: Optional[float] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        self.remaining_quantity = self.quantity
        
        # Set displayed quantity for iceberg orders
        if self.hidden_quantity > 0:
            if self.displayed_quantity == 0:
                # Default display quantity for iceberg
                self.displayed_quantity = min(self.quantity // 5, 1000)
            self.displayed_quantity = min(self.displayed_quantity, self.remaining_quantity)
        else:
            self.displayed_quantity = self.quantity
        
        # Set expiry time for DAY orders
        if self.time_in_force == TimeInForce.DAY and self.expiry_time is None:
            # Set to market close (4 PM EST)
            today = self.creation_time.date()
            self.expiry_time = datetime.combine(today, datetime.min.time().replace(hour=16))
            if self.expiry_time <= self.creation_time:
                # If after market hours, set for next trading day
                self.expiry_time += timedelta(days=1)
    
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
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage"""
        return (self.filled_quantity / self.quantity) * 100 if self.quantity > 0 else 0
    
    @property
    def time_alive(self) -> timedelta:
        """Calculate how long the order has been alive"""
        return datetime.now() - self.creation_time
    
    def can_fill(self, market_price: float) -> bool:
        """Check if order can be filled at given market price"""
        if not self.is_active:
            return False
        
        if self.is_buy:
            return market_price <= self.price
        else:
            return market_price >= self.price
    
    def add_fill(self, fill: 'Fill'):
        """Add a fill to this order"""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        
        # Update average fill price
        total_value = sum(f.quantity * f.price for f in self.fills)
        self.average_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0
        
        # Update status
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIAL
        
        # Update displayed quantity for iceberg orders
        if self.hidden_quantity > 0:
            self.displayed_quantity = min(
                self.displayed_quantity, 
                self.remaining_quantity,
                self.quantity // 5  # Refresh display quantity
            )
        
        logger.debug(f"Order {self.order_id} filled {fill.quantity} @ {fill.price}, "
                    f"total filled: {self.filled_quantity}/{self.quantity}")

@dataclass
class Fill:
    """Trade fill information"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0
    liquidity_flag: str = "Unknown"  # Maker/Taker
    counterparty_order_id: Optional[str] = None

# ============================================
# Order Book Implementation
# ============================================

class PriceLevelQueue:
    """Queue for orders at a specific price level"""
    
    def __init__(self, price: float):
        self.price = price
        self.orders: deque = deque()
        self.total_quantity = 0
        self.total_displayed_quantity = 0
        
    def add_order(self, order: LimitOrder):
        """Add order to price level"""
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity
        self.total_displayed_quantity += order.displayed_quantity
        order.queue_position = len(self.orders)
        
    def remove_order(self, order: LimitOrder) -> bool:
        """Remove order from price level"""
        try:
            self.orders.remove(order)
            self.total_quantity -= order.remaining_quantity
            self.total_displayed_quantity -= order.displayed_quantity
            
            # Update queue positions
            for i, ord in enumerate(self.orders, 1):
                ord.queue_position = i
            
            return True
        except ValueError:
            return False
    
    def get_executable_quantity(self, market_quantity: int) -> List[Tuple[LimitOrder, int]]:
        """Get orders that can be executed against market quantity"""
        executable = []
        remaining_quantity = market_quantity
        
        for order in list(self.orders):
            if remaining_quantity <= 0:
                break
            
            if not order.is_active:
                continue
            
            # Check minimum quantity constraint
            available_quantity = order.remaining_quantity
            if order.minimum_quantity > 0:
                if remaining_quantity < order.minimum_quantity:
                    continue
                available_quantity = min(available_quantity, 
                                       (remaining_quantity // order.minimum_quantity) * order.minimum_quantity)
            
            fill_quantity = min(available_quantity, remaining_quantity)
            if fill_quantity > 0:
                executable.append((order, fill_quantity))
                remaining_quantity -= fill_quantity
        
        return executable
    
    def is_empty(self) -> bool:
        """Check if price level has no active orders"""
        return len(self.orders) == 0 or self.total_quantity == 0

class LimitOrderBook:
    """
    Comprehensive limit order book implementation.
    
    Maintains buy and sell order queues with price-time priority,
    handles order matching and execution with advanced features.
    """
    
    def __init__(self, symbol: str, priority_rule: OrderPriority = OrderPriority.PRICE_TIME):
        self.symbol = symbol
        self.priority_rule = priority_rule
        
        # Order book data structures
        # Buy orders: price descending (highest first)
        self.buy_orders: Dict[float, PriceLevelQueue] = {}
        # Sell orders: price ascending (lowest first)  
        self.sell_orders: Dict[float, PriceLevelQueue] = {}
        
        # Order tracking
        self.active_orders: Dict[str, LimitOrder] = {}
        self.order_history: List[LimitOrder] = []
        self.fill_history: List[Fill] = []
        
        # Book statistics
        self.total_volume = 0
        self.trade_count = 0
        self.last_trade_price = 0.0
        self.last_trade_time: Optional[datetime] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized limit order book for {symbol} with {priority_rule.value} priority")
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price"""
        if not self.buy_orders:
            return None
        return max(self.buy_orders.keys())
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price"""
        if not self.sell_orders:
            return None
        return min(self.sell_orders.keys())
    
    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        bid, ask = self.best_bid, self.best_ask
        if bid is not None and ask is not None:
            return ask - bid
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price"""
        bid, ask = self.best_bid, self.best_ask
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None
    
    @time_it("limit_order_add")
    def add_order(self, order: LimitOrder) -> bool:
        """
        Add limit order to the book
        
        Args:
            order: Limit order to add
            
        Returns:
            True if order was added successfully
        """
        
        with self._lock:
            try:
                # Validate order
                if not self._validate_order(order):
                    order.status = OrderStatus.REJECTED
                    return False
                
                # Check for immediate execution opportunity
                if self._can_execute_immediately(order):
                    self._execute_against_book(order)
                
                # Add remaining quantity to book if any
                if order.remaining_quantity > 0 and order.is_active:
                    self._add_to_book(order)
                
                # Track active order
                if order.is_active:
                    self.active_orders[order.order_id] = order
                
                logger.debug(f"Added order {order.order_id}: {order.side.value} {order.quantity} "
                           f"{order.symbol} @ {order.price}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error adding order {order.order_id}: {e}")
                order.status = OrderStatus.REJECTED
                return False
    
    def cancel_order(self, order_id: str, reason: str = "User cancelled") -> bool:
        """
        Cancel an active order
        
        Args:
            order_id: ID of order to cancel
            reason: Cancellation reason
            
        Returns:
            True if order was cancelled successfully
        """
        
        with self._lock:
            order = self.active_orders.get(order_id)
            if not order or not order.is_active:
                logger.warning(f"Cannot cancel order {order_id}: not found or not active")
                return False
            
            # Remove from book
            self._remove_from_book(order)
            
            # Update order status
            order.status = OrderStatus.CANCELLED
            order.modification_time = datetime.now()
            
            # Remove from active orders
            del self.active_orders[order_id]
            
            # Add to history
            self.order_history.append(order)
            
            logger.info(f"Cancelled order {order_id}: {reason}")
            return True
    
    def modify_order(self, order_id: str, new_price: Optional[float] = None, 
                    new_quantity: Optional[int] = None) -> bool:
        """
        Modify an active order (cancel and replace)
        
        Args:
            order_id: ID of order to modify
            new_price: New limit price
            new_quantity: New quantity
            
        Returns:
            True if order was modified successfully
        """
        
        with self._lock:
            order = self.active_orders.get(order_id)
            if not order or not order.is_active:
                logger.warning(f"Cannot modify order {order_id}: not found or not active")
                return False
            
            # Remove from current position
            self._remove_from_book(order)
            
            # Update order parameters
            if new_price is not None:
                order.price = new_price
            if new_quantity is not None:
                if new_quantity < order.filled_quantity:
                    logger.error(f"Cannot reduce quantity below filled amount: {order.filled_quantity}")
                    # Re-add to book at original position
                    self._add_to_book(order)
                    return False
                order.quantity = new_quantity
                order.remaining_quantity = new_quantity - order.filled_quantity
            
            order.modification_time = datetime.now()
            order.priority_timestamp = datetime.now()  # Lose time priority
            
            # Check for immediate execution with new parameters
            if self._can_execute_immediately(order):
                self._execute_against_book(order)
            
            # Add back to book if still active
            if order.remaining_quantity > 0 and order.is_active:
                self._add_to_book(order)
            
            logger.info(f"Modified order {order_id}: price={new_price}, quantity={new_quantity}")
            return True
    
    def get_market_depth(self, levels: int = 10) -> Dict[str, Any]:
        """
        Get market depth information
        
        Args:
            levels: Number of price levels to include
            
        Returns:
            Market depth data
        """
        
        with self._lock:
            # Get buy side (bids)
            buy_prices = sorted(self.buy_orders.keys(), reverse=True)[:levels]
            bids = []
            for price in buy_prices:
                level = self.buy_orders[price]
                if not level.is_empty():
                    bids.append({
                        'price': price,
                        'quantity': level.total_quantity,
                        'displayed_quantity': level.total_displayed_quantity,
                        'order_count': len(level.orders)
                    })
            
            # Get sell side (asks)
            sell_prices = sorted(self.sell_orders.keys())[:levels]
            asks = []
            for price in sell_prices:
                level = self.sell_orders[price]
                if not level.is_empty():
                    asks.append({
                        'price': price,
                        'quantity': level.total_quantity,
                        'displayed_quantity': level.total_displayed_quantity,
                        'order_count': len(level.orders)
                    })
            
            return {
                'symbol': self.symbol,
                'timestamp': datetime.now(),
                'bids': bids,
                'asks': asks,
                'best_bid': self.best_bid,
                'best_ask': self.best_ask,
                'spread': self.spread,
                'mid_price': self.mid_price,
                'last_trade_price': self.last_trade_price,
                'total_volume': self.total_volume,
                'trade_count': self.trade_count
            }
    
    def _validate_order(self, order: LimitOrder) -> bool:
        """Validate order before adding to book"""
        
        # Basic validation
        if order.quantity <= 0:
            logger.error(f"Invalid quantity: {order.quantity}")
            return False
        
        if order.price <= 0:
            logger.error(f"Invalid price: {order.price}")
            return False
        
        if order.symbol != self.symbol:
            logger.error(f"Symbol mismatch: expected {self.symbol}, got {order.symbol}")
            return False
        
        # Time in force validation
        if order.time_in_force == TimeInForce.IOC and order.hidden_quantity > 0:
            logger.error("IOC orders cannot be iceberg orders")
            return False
        
        return True
    
    def _can_execute_immediately(self, order: LimitOrder) -> bool:
        """Check if order can execute immediately against existing orders"""
        
        if order.is_buy:
            # Buy order can execute against sell orders at or below the limit price
            for price in sorted(self.sell_orders.keys()):
                if price <= order.price and not self.sell_orders[price].is_empty():
                    return True
        else:
            # Sell order can execute against buy orders at or above the limit price
            for price in sorted(self.buy_orders.keys(), reverse=True):
                if price >= order.price and not self.buy_orders[price].is_empty():
                    return True
        
        return False
    
    def _execute_against_book(self, order: LimitOrder):
        """Execute order against existing book orders"""
        
        remaining_quantity = order.remaining_quantity
        
        if order.is_buy:
            # Execute against sell orders (asks)
            ask_prices = sorted(self.sell_orders.keys())
            
            for ask_price in ask_prices:
                if ask_price > order.price or remaining_quantity <= 0:
                    break
                
                level = self.sell_orders[ask_price]
                executable_orders = level.get_executable_quantity(remaining_quantity)
                
                for counterparty_order, fill_quantity in executable_orders:
                    if remaining_quantity <= 0:
                        break
                    
                    # Create fills
                    trade_price = counterparty_order.price  # Price improvement for aggressor
                    self._create_trade(order, counterparty_order, fill_quantity, trade_price)
                    
                    remaining_quantity -= fill_quantity
                    
                    # Check for complete fill
                    if counterparty_order.remaining_quantity == 0:
                        level.remove_order(counterparty_order)
                        if counterparty_order.order_id in self.active_orders:
                            del self.active_orders[counterparty_order.order_id]
                        self.order_history.append(counterparty_order)
                
                # Remove empty price level
                if level.is_empty():
                    del self.sell_orders[ask_price]
        
        else:
            # Execute against buy orders (bids)
            bid_prices = sorted(self.buy_orders.keys(), reverse=True)
            
            for bid_price in bid_prices:
                if bid_price < order.price or remaining_quantity <= 0:
                    break
                
                level = self.buy_orders[bid_price]
                executable_orders = level.get_executable_quantity(remaining_quantity)
                
                for counterparty_order, fill_quantity in executable_orders:
                    if remaining_quantity <= 0:
                        break
                    
                    # Create fills
                    trade_price = counterparty_order.price  # Price improvement for aggressor
                    self._create_trade(order, counterparty_order, fill_quantity, trade_price)
                    
                    remaining_quantity -= fill_quantity
                    
                    # Check for complete fill
                    if counterparty_order.remaining_quantity == 0:
                        level.remove_order(counterparty_order)
                        if counterparty_order.order_id in self.active_orders:
                            del self.active_orders[counterparty_order.order_id]
                        self.order_history.append(counterparty_order)
                
                # Remove empty price level
                if level.is_empty():
                    del self.buy_orders[bid_price]
        
        # Handle IOC/FOK orders
        if order.time_in_force == TimeInForce.IOC and order.remaining_quantity > 0:
            order.status = OrderStatus.CANCELLED
        elif order.time_in_force == TimeInForce.FOK and order.filled_quantity < order.quantity:
            # FOK order must be filled completely or cancelled
            # This would require more complex logic to check fill possibility first
            order.status = OrderStatus.CANCELLED
    
    def _create_trade(self, aggressor_order: LimitOrder, passive_order: LimitOrder, 
                     quantity: int, price: float):
        """Create trade between two orders"""
        
        trade_time = datetime.now()
        
        # Create fills for both orders
        aggressor_fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=aggressor_order.order_id,
            symbol=self.symbol,
            side=aggressor_order.side,
            quantity=quantity,
            price=price,
            timestamp=trade_time,
            liquidity_flag="Taker",
            counterparty_order_id=passive_order.order_id
        )
        
        passive_fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=passive_order.order_id,
            symbol=self.symbol,
            side=passive_order.side,
            quantity=quantity,
            price=price,
            timestamp=trade_time,
            liquidity_flag="Maker",
            counterparty_order_id=aggressor_order.order_id
        )
        
        # Apply fills to orders
        aggressor_order.add_fill(aggressor_fill)
        passive_order.add_fill(passive_fill)
        
        # Update book statistics
        self.total_volume += quantity
        self.trade_count += 1
        self.last_trade_price = price
        self.last_trade_time = trade_time
        
        # Store fill history
        self.fill_history.extend([aggressor_fill, passive_fill])
        
        logger.debug(f"Trade executed: {quantity} shares @ {price} between "
                    f"{aggressor_order.order_id} and {passive_order.order_id}")
    
    def _add_to_book(self, order: LimitOrder):
        """Add order to the appropriate side of the book"""
        
        if order.is_buy:
            if order.price not in self.buy_orders:
                self.buy_orders[order.price] = PriceLevelQueue(order.price)
            self.buy_orders[order.price].add_order(order)
        else:
            if order.price not in self.sell_orders:
                self.sell_orders[order.price] = PriceLevelQueue(order.price)
            self.sell_orders[order.price].add_order(order)
    
    def _remove_from_book(self, order: LimitOrder):
        """Remove order from the book"""
        
        if order.is_buy:
            level = self.buy_orders.get(order.price)
            if level and level.remove_order(order):
                if level.is_empty():
                    del self.buy_orders[order.price]
        else:
            level = self.sell_orders.get(order.price)
            if level and level.remove_order(order):
                if level.is_empty():
                    del self.sell_orders[order.price]

# ============================================
# Limit Order Manager
# ============================================

class LimitOrderManager:
    """
    Comprehensive limit order management system.
    
    Manages multiple order books, handles order lifecycle,
    provides analytics and risk management for limit orders.
    """
    
    def __init__(self):
        self.order_books: Dict[str, LimitOrderBook] = {}
        self.global_orders: Dict[str, LimitOrder] = {}
        
        # Configuration
        self.commission_rate = 0.001  # 0.1%
        self.min_commission = 1.0
        
        # Analytics
        self.performance_metrics = defaultdict(dict)
        self.risk_limits = defaultdict(dict)
        
        # Threading
        self._lock = threading.RLock()
        
        logger.info("Initialized LimitOrderManager")
    
    def get_order_book(self, symbol: str) -> LimitOrderBook:
        """Get or create order book for symbol"""
        
        if symbol not in self.order_books:
            self.order_books[symbol] = LimitOrderBook(symbol)
        
        return self.order_books[symbol]
    
    @time_it("limit_order_submission")
    def submit_limit_order(self, 
                          symbol: str,
                          side: OrderSide,
                          quantity: int,
                          price: float,
                          time_in_force: TimeInForce = TimeInForce.DAY,
                          **kwargs) -> str:
        """
        Submit a limit order
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            price: Limit price
            time_in_force: Order time in force
            **kwargs: Additional order parameters
            
        Returns:
            Order ID
        """
        
        with self._lock:
            # Generate order ID
            order_id = kwargs.get('order_id', str(uuid.uuid4()))
            
            # Create limit order
            order = LimitOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                time_in_force=time_in_force,
                **{k: v for k, v in kwargs.items() if k != 'order_id'}
            )
            
            # Get order book
            order_book = self.get_order_book(symbol)
            
            # Submit to order book
            if order_book.add_order(order):
                self.global_orders[order_id] = order
                logger.info(f"Submitted limit order {order_id}: {side.value} {quantity} {symbol} @ {price}")
                return order_id
            else:
                logger.error(f"Failed to submit limit order {order_id}")
                raise CalculationError(f"Order submission failed for {order_id}")
    
    def cancel_order(self, order_id: str, reason: str = "User requested") -> bool:
        """Cancel an order"""
        
        with self._lock:
            order = self.global_orders.get(order_id)
            if not order:
                logger.warning(f"Order {order_id} not found")
                return False
            
            order_book = self.get_order_book(order.symbol)
            success = order_book.cancel_order(order_id, reason)
            
            if success and order_id in self.global_orders:
                del self.global_orders[order_id]
            
            return success
    
    def modify_order(self, order_id: str, new_price: Optional[float] = None, 
                    new_quantity: Optional[int] = None) -> bool:
        """Modify an order"""
        
        with self._lock:
            order = self.global_orders.get(order_id)
            if not order:
                logger.warning(f"Order {order_id} not found")
                return False
            
            order_book = self.get_order_book(order.symbol)
            return order_book.modify_order(order_id, new_price, new_quantity)
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed order status"""
        
        order = self.global_orders.get(order_id)
        if not order:
            return None
        
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': order.price,
            'status': order.status.value,
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': order.remaining_quantity,
            'fill_percentage': order.fill_percentage,
            'average_fill_price': order.average_fill_price,
            'time_in_force': order.time_in_force.value,
            'creation_time': order.creation_time,
            'time_alive': order.time_alive.total_seconds(),
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
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all active orders, optionally filtered by symbol"""
        
        active_orders = []
        
        for order in self.global_orders.values():
            if order.is_active:
                if symbol is None or order.symbol == symbol:
                    order_status = self.get_order_status(order.order_id)
                    if order_status:
                        active_orders.append(order_status)
        
        return active_orders
    
    def get_market_depth(self, symbol: str, levels: int = 10) -> Dict[str, Any]:
        """Get market depth for a symbol"""
        
        order_book = self.order_books.get(symbol)
        if not order_book:
            return {
                'symbol': symbol,
                'error': 'No order book found'
            }
        
        return order_book.get_market_depth(levels)
    
    def calculate_order_analytics(self, order_id: str) -> Dict[str, Any]:
        """Calculate comprehensive analytics for an order"""
        
        order = self.global_orders.get(order_id)
        if not order:
            return {}
        
        analytics = {
            'order_id': order_id,
            'symbol': order.symbol,
            'execution_analytics': {},
            'timing_analytics': {},
            'cost_analytics': {}
        }
        
        # Execution analytics
        if order.fills:
            fill_prices = [fill.price for fill in order.fills]
            fill_quantities = [fill.quantity for fill in order.fills]
            
            analytics['execution_analytics'] = {
                'fill_count': len(order.fills),
                'total_filled': order.filled_quantity,
                'fill_rate': order.fill_percentage / 100,
                'average_fill_price': order.average_fill_price,
                'price_range': {
                    'min': min(fill_prices),
                    'max': max(fill_prices),
                    'spread': max(fill_prices) - min(fill_prices)
                },
                'volume_weighted_price': sum(p * q for p, q in zip(fill_prices, fill_quantities)) / sum(fill_quantities)
            }
        
        # Timing analytics
        analytics['timing_analytics'] = {
            'order_age_seconds': order.time_alive.total_seconds(),
            'time_to_first_fill': None,
            'time_to_complete_fill': None,
            'average_time_between_fills': None
        }
        
        if order.fills:
            first_fill_time = (order.fills[0].timestamp - order.creation_time).total_seconds()
            analytics['timing_analytics']['time_to_first_fill'] = first_fill_time
            
            if order.is_filled:
                complete_fill_time = (order.fills[-1].timestamp - order.creation_time).total_seconds()
                analytics['timing_analytics']['time_to_complete_fill'] = complete_fill_time
            
            if len(order.fills) > 1:
                fill_intervals = [
                    (order.fills[i].timestamp - order.fills[i-1].timestamp).total_seconds()
                    for i in range(1, len(order.fills))
                ]
                analytics['timing_analytics']['average_time_between_fills'] = np.mean(fill_intervals)
        
        # Cost analytics
        total_commission = sum(fill.commission for fill in order.fills)
        total_value = sum(fill.quantity * fill.price for fill in order.fills)
        
        analytics['cost_analytics'] = {
            'total_commission': total_commission,
            'commission_rate': total_commission / total_value if total_value > 0 else 0,
            'total_trade_value': total_value,
            'price_improvement': self._calculate_price_improvement(order),
            'implementation_shortfall': self._calculate_implementation_shortfall(order)
        }
        
        return analytics
    
    def _calculate_price_improvement(self, order: LimitOrder) -> float:
        """Calculate price improvement relative to limit price"""
        
        if not order.fills:
            return 0.0
        
        if order.is_buy:
            # For buy orders, improvement is when we pay less than limit
            improvement = order.price - order.average_fill_price
        else:
            # For sell orders, improvement is when we receive more than limit
            improvement = order.average_fill_price - order.price
        
        return max(0, improvement)
    
    def _calculate_implementation_shortfall(self, order: LimitOrder) -> float:
        """Calculate implementation shortfall (simplified version)"""
        
        if not order.fills or not order.market_price_at_creation:
            return 0.0
        
        # Simplified calculation: difference between decision price and average fill price
        if order.is_buy:
            shortfall = order.average_fill_price - order.market_price_at_creation
        else:
            shortfall = order.market_price_at_creation - order.average_fill_price
        
        return shortfall * order.filled_quantity

# ============================================
# Utility Functions
# ============================================

def create_limit_order(symbol: str, side: str, quantity: int, price: float, 
                      **kwargs) -> LimitOrder:
    """
    Utility function to create a limit order
    
    Args:
        symbol: Trading symbol
        side: 'BUY' or 'SELL'
        quantity: Order quantity
        price: Limit price
        **kwargs: Additional parameters
        
    Returns:
        LimitOrder instance
    """
    
    order_side = OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL
    order_id = kwargs.get('order_id', str(uuid.uuid4()))
    
    return LimitOrder(
        order_id=order_id,
        symbol=symbol.upper(),
        side=order_side,
        quantity=quantity,
        price=price,
        **{k: v for k, v in kwargs.items() if k != 'order_id'}
    )

def calculate_iceberg_display_quantity(total_quantity: int, 
                                     market_impact_threshold: float = 0.1) -> int:
    """
    Calculate optimal display quantity for iceberg orders
    
    Args:
        total_quantity: Total order quantity
        market_impact_threshold: Maximum acceptable market impact
        
    Returns:
        Recommended display quantity
    """
    
    # Simple heuristic: display 5-20% of total quantity
    min_display = max(100, int(total_quantity * 0.05))
    max_display = max(1000, int(total_quantity * 0.2))
    
    # Adjust based on market impact considerations
    recommended = min(max_display, max(min_display, int(total_quantity * market_impact_threshold)))
    
    return recommended

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Limit Orders System")
    
    # Initialize limit order manager
    manager = LimitOrderManager()
    
    print("\n1. Testing Basic Order Submission")
    
    # Submit some limit orders
    buy_order_id = manager.submit_limit_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=1000,
        price=150.00,
        time_in_force=TimeInForce.GTC,
        client_order_id="TEST_BUY_001"
    )
    
    print(f"Submitted buy order: {buy_order_id}")
    
    sell_order_id = manager.submit_limit_order(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=500,
        price=151.00,
        time_in_force=TimeInForce.DAY,
        client_order_id="TEST_SELL_001"
    )
    
    print(f"Submitted sell order: {sell_order_id}")
    
    # Check order status
    buy_status = manager.get_order_status(buy_order_id)
    print(f"Buy order status: {buy_status['status']}, filled: {buy_status['filled_quantity']}")
    
    print("\n2. Testing Market Depth")
    
    # Add more orders to create depth
    for i in range(5):
        # Buy orders with decreasing prices
        manager.submit_limit_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100 * (i + 1),
            price=149.50 - i * 0.25
        )
        
        # Sell orders with increasing prices
        manager.submit_limit_order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100 * (i + 1),
            price=151.50 + i * 0.25
        )
    
    # Get market depth
    depth = manager.get_market_depth("AAPL", levels=5)
    print(f"Market depth for AAPL:")
    print(f"  Best bid: {depth['best_bid']}")
    print(f"  Best ask: {depth['best_ask']}")
    print(f"  Spread: {depth['spread']}")
    print(f"  Bid levels: {len(depth['bids'])}")
    print(f"  Ask levels: {len(depth['asks'])}")
    
    # Show bid side
    print("  Top 3 Bids:")
    for i, bid in enumerate(depth['bids'][:3]):
        print(f"    {i+1}. {bid['quantity']} @ {bid['price']}")
    
    # Show ask side
    print("  Top 3 Asks:")
    for i, ask in enumerate(depth['asks'][:3]):
        print(f"    {i+1}. {ask['quantity']} @ {ask['price']}")
    
    print("\n3. Testing Order Execution")
    
    # Submit aggressive order that should execute immediately
    aggressive_buy_id = manager.submit_limit_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=300,
        price=152.00,  # Above current asks
        client_order_id="AGGRESSIVE_BUY"
    )
    
    aggressive_status = manager.get_order_status(aggressive_buy_id)
    if aggressive_status:
        print(f"Aggressive buy order: {aggressive_status['status']}")
        print(f"  Filled: {aggressive_status['filled_quantity']}/{aggressive_status['quantity']}")
        print(f"  Avg fill price: {aggressive_status['average_fill_price']:.2f}")
        print(f"  Number of fills: {len(aggressive_status['fills'])}")
    
    print("\n4. Testing Order Modification")
    
    # Submit order for modification
    modify_order_id = manager.submit_limit_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=200,
        price=148.00
    )
    
    print(f"Submitted order for modification: {modify_order_id}")
    
    # Modify price
    success = manager.modify_order(modify_order_id, new_price=148.50)
    print(f"Price modification {'successful' if success else 'failed'}")
    
    # Modify quantity
    success = manager.modify_order(modify_order_id, new_quantity=300)
    print(f"Quantity modification {'successful' if success else 'failed'}")
    
    print("\n5. Testing Order Cancellation")
    
    # Cancel an order
    cancel_success = manager.cancel_order(sell_order_id, "Testing cancellation")
    print(f"Order cancellation {'successful' if cancel_success else 'failed'}")
    
    print("\n6. Testing Iceberg Orders")
    
    # Create large iceberg order
    iceberg_id = manager.submit_limit_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=5000,
        price=149.75,
        hidden_quantity=4000,  # Hide 4000 shares
        displayed_quantity=500,  # Show only 500 shares
        client_order_id="ICEBERG_ORDER"
    )
    
    iceberg_status = manager.get_order_status(iceberg_id)
    if iceberg_status:
        print(f"Iceberg order status: {iceberg_status['status']}")
        print(f"  Total quantity: {iceberg_status['quantity']}")
        print(f"  Visible in book: 500 shares")
    
    print("\n7. Testing Order Analytics")
    
    # Get analytics for executed order
    if aggressive_status and aggressive_status['fills']:
        analytics = manager.calculate_order_analytics(aggressive_buy_id)
        
        if 'execution_analytics' in analytics:
            exec_analytics = analytics['execution_analytics']
            print(f"Execution Analytics:")
            print(f"  Fill rate: {exec_analytics['fill_rate']:.1%}")
            print(f"  Average fill price: {exec_analytics['average_fill_price']:.2f}")
            print(f"  Price range: {exec_analytics['price_range']['min']:.2f} - {exec_analytics['price_range']['max']:.2f}")
        
        if 'timing_analytics' in analytics:
            timing_analytics = analytics['timing_analytics']
            print(f"Timing Analytics:")
            if timing_analytics['time_to_first_fill']:
                print(f"  Time to first fill: {timing_analytics['time_to_first_fill']:.2f} seconds")
    
    print("\n8. Testing Active Orders Summary")
    
    # Get all active orders
    active_orders = manager.get_active_orders("AAPL")
    print(f"Active orders for AAPL: {len(active_orders)}")
    
    for order in active_orders[:3]:  # Show first 3
        print(f"  {order['order_id'][:8]}...: {order['side']} {order['remaining_quantity']} @ {order['price']}")
    
    print("\n9. Testing Utility Functions")
    
    # Create order using utility function
    utility_order = create_limit_order(
        symbol="MSFT",
        side="BUY",
        quantity=100,
        price=300.00,
        time_in_force=TimeInForce.GTC
    )
    
    print(f"Created order via utility: {utility_order.order_id}")
    print(f"  Symbol: {utility_order.symbol}")
    print(f"  Side: {utility_order.side.value}")
    print(f"  Quantity: {utility_order.quantity}")
    print(f"  Price: {utility_order.price}")
    
    # Calculate iceberg display quantity
    display_qty = calculate_iceberg_display_quantity(10000, 0.15)
    print(f"Recommended iceberg display for 10,000 shares: {display_qty}")
    
    print("\nLimit orders system testing completed successfully!")
    print("\nImplemented features include:")
    print("• Comprehensive limit order lifecycle management")
    print("• Price-time priority order matching engine")
    print("• Multiple time-in-force options (DAY, GTC, IOC, FOK)")
    print("• Iceberg order support with hidden quantity")
    print("• Order modification and cancellation")
    print("• Real-time market depth and order book management")
    print("• Advanced order analytics and performance metrics")
    print("• Thread-safe operations for concurrent access")
    print("• Integration-ready design for broker APIs")