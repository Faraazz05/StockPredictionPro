# ============================================
# StockPredictionPro - src/evaluation/backtesting/engine.py
# Advanced backtesting engine for financial machine learning strategies
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('evaluation.backtesting.engine')

# ============================================
# Core Data Structures and Enums
# ============================================

class OrderType(Enum):
    """Order types for trading"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionSide(Enum):
    """Position sides"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class Order:
    """Represents a trading order"""
    id: str
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_quantity: float = 0.0
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Trade:
    """Represents an executed trade"""
    id: str
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    order_id: str
    pnl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    side: PositionSide
    quantity: float
    avg_price: float
    market_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    entry_timestamp: Optional[pd.Timestamp] = None
    last_update: Optional[pd.Timestamp] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioState:
    """Represents portfolio state at a point in time"""
    timestamp: pd.Timestamp
    cash: float
    total_value: float
    positions: Dict[str, Position]
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    drawdown: float = 0.0
    leverage: float = 1.0
    margin_used: float = 0.0

# ============================================
# Commission Models
# ============================================

class CommissionModel:
    """Base class for commission models"""
    
    def calculate_commission(self, trade: Trade) -> float:
        """Calculate commission for a trade"""
        return 0.0

class FixedCommissionModel(CommissionModel):
    """Fixed commission per trade"""
    
    def __init__(self, commission: float):
        self.commission = commission
    
    def calculate_commission(self, trade: Trade) -> float:
        return self.commission

class PercentageCommissionModel(CommissionModel):
    """Percentage-based commission"""
    
    def __init__(self, rate: float, min_commission: float = 0.0):
        self.rate = rate
        self.min_commission = min_commission
    
    def calculate_commission(self, trade: Trade) -> float:
        commission = abs(trade.quantity * trade.price * self.rate)
        return max(commission, self.min_commission)

class TieredCommissionModel(CommissionModel):
    """Tiered commission based on trade value"""
    
    def __init__(self, tiers: List[Tuple[float, float]]):
        """
        Args:
            tiers: List of (threshold, rate) tuples, sorted by threshold
        """
        self.tiers = sorted(tiers, key=lambda x: x[0])
    
    def calculate_commission(self, trade: Trade) -> float:
        trade_value = abs(trade.quantity * trade.price)
        
        for threshold, rate in self.tiers:
            if trade_value <= threshold:
                return trade_value * rate
        
        # Use highest tier rate if trade_value exceeds all thresholds
        return trade_value * self.tiers[-1][1]

# ============================================
# Slippage Models
# ============================================

class SlippageModel:
    """Base class for slippage models"""
    
    def apply_slippage(self, order: Order, market_price: float) -> float:
        """Apply slippage to order execution price"""
        return market_price

class FixedSlippageModel(SlippageModel):
    """Fixed slippage per trade"""
    
    def __init__(self, slippage: float):
        self.slippage = slippage
    
    def apply_slippage(self, order: Order, market_price: float) -> float:
        if order.side == OrderSide.BUY:
            return market_price * (1 + self.slippage)
        else:
            return market_price * (1 - self.slippage)

class VolumeBasedSlippageModel(SlippageModel):
    """Volume-based slippage model"""
    
    def __init__(self, base_slippage: float, volume_impact: float):
        self.base_slippage = base_slippage
        self.volume_impact = volume_impact
    
    def apply_slippage(self, order: Order, market_price: float) -> float:
        # Slippage increases with order size (simplified model)
        volume_factor = order.quantity * self.volume_impact
        total_slippage = self.base_slippage + volume_factor
        
        if order.side == OrderSide.BUY:
            return market_price * (1 + total_slippage)
        else:
            return market_price * (1 - total_slippage)

# ============================================
# Backtesting Engine Configuration
# ============================================

@dataclass
class BacktestConfig:
    """Configuration for backtesting engine"""
    initial_cash: float = 1000000.0
    commission_model: CommissionModel = field(default_factory=lambda: PercentageCommissionModel(0.001))
    slippage_model: SlippageModel = field(default_factory=lambda: FixedSlippageModel(0.0005))
    
    # Risk management
    max_leverage: float = 1.0
    margin_requirement: float = 0.0
    position_size_limit: float = 0.1  # Max 10% of portfolio in single position
    
    # Execution settings
    fill_on_bar_close: bool = True  # Fill orders at bar close vs next bar open
    partial_fills: bool = False
    max_orders_per_bar: int = 100
    
    # Data settings
    adjust_for_splits: bool = True
    adjust_for_dividends: bool = True
    
    # Performance tracking
    benchmark_symbol: Optional[str] = None
    track_orders: bool = True
    track_trades: bool = True
    track_positions: bool = True
    
    # Logging
    log_level: str = 'INFO'
    save_intermediate_states: bool = False

# ============================================
# Main Backtesting Engine
# ============================================

class BacktestEngine:
    """
    Advanced backtesting engine for financial ML strategies.
    
    This engine provides comprehensive backtesting capabilities including:
    - Order management and execution
    - Position tracking
    - Portfolio management  
    - Performance analytics
    - Risk management
    - Multi-asset support
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        
        # Portfolio state
        self.cash = self.config.initial_cash
        self.initial_cash = self.config.initial_cash
        self.positions: Dict[str, Position] = {}
        self.portfolio_history: List[PortfolioState] = []
        
        # Order and trade tracking
        self.orders: List[Order] = []
        self.pending_orders: List[Order] = []
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.equity_curve: List[float] = []
        self.benchmark_curve: List[float] = []
        self.drawdowns: List[float] = []
        
        # State tracking
        self.current_timestamp: Optional[pd.Timestamp] = None
        self.current_data: Dict[str, Any] = {}
        self.is_running = False
        
        # Strategy and data
        self.strategy = None
        self.data_handler = None
        
        # Counters
        self._order_id_counter = 0
        self._trade_id_counter = 0
        
        logger.info(f"BacktestEngine initialized with ${self.config.initial_cash:,.2f} initial capital")
    
    def set_strategy(self, strategy):
        """Set the trading strategy"""
        self.strategy = strategy
        if hasattr(strategy, 'set_engine'):
            strategy.set_engine(self)
    
    def set_data_handler(self, data_handler):
        """Set the data handler"""
        self.data_handler = data_handler
    
    @time_it("backtest_execution")
    def run_backtest(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the backtest simulation
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            Dictionary containing backtest results
        """
        
        if self.strategy is None:
            raise ValueError("Strategy must be set before running backtest")
        
        if self.data_handler is None:
            raise ValueError("Data handler must be set before running backtest")
        
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        self.is_running = True
        
        try:
            # Initialize strategy
            if hasattr(self.strategy, 'initialize'):
                self.strategy.initialize()
            
            # Get data iterator
            data_iterator = self.data_handler.get_data_iterator(start_date, end_date)
            
            # Main backtest loop
            for timestamp, market_data in data_iterator:
                self.current_timestamp = timestamp
                self.current_data = market_data
                
                # Process pending orders
                self._process_pending_orders()
                
                # Update portfolio
                self._update_portfolio()
                
                # Generate strategy signals
                if hasattr(self.strategy, 'on_data'):
                    self.strategy.on_data(timestamp, market_data)
                
                # Record portfolio state
                self._record_portfolio_state()
                
                # Check risk limits
                self._check_risk_limits()
            
            # Finalize backtest
            self._finalize_backtest()
            
            # Generate results
            results = self._generate_results()
            
            logger.info(f"Backtest completed. Final portfolio value: ${self.get_portfolio_value():,.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
        finally:
            self.is_running = False
    
    def submit_order(self, symbol: str, side: OrderSide, quantity: float, 
                    order_type: OrderType = OrderType.MARKET, 
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    **kwargs) -> str:
        """
        Submit a trading order
        
        Args:
            symbol: Symbol to trade
            side: Buy or sell
            quantity: Number of shares/units
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            **kwargs: Additional order parameters
            
        Returns:
            Order ID
        """
        
        if not self.is_running:
            raise RuntimeError("Cannot submit orders when backtest is not running")
        
        # Generate order ID
        order_id = f"ORDER_{self._order_id_counter:06d}"
        self._order_id_counter += 1
        
        # Create order
        order = Order(
            id=order_id,
            timestamp=self.current_timestamp,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            metadata=kwargs
        )
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order {order_id} rejected: validation failed")
            return order_id
        
        # Add to pending orders
        self.pending_orders.append(order)
        self.orders.append(order)
        
        logger.debug(f"Order submitted: {order_id} - {side.value} {quantity} {symbol}")
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        
        for order in self.pending_orders:
            if order.id == order_id and order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                self.pending_orders.remove(order)
                logger.debug(f"Order cancelled: {order_id}")
                return True
        
        return False
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
        return self.positions.get(symbol)
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        return self.positions.copy()
    
    def get_portfolio_value(self) -> float:
        """Get current total portfolio value"""
        market_value = sum(pos.quantity * pos.market_price for pos in self.positions.values())
        return self.cash + market_value
    
    def get_cash(self) -> float:
        """Get current cash balance"""
        return self.cash
    
    def _process_pending_orders(self):
        """Process all pending orders"""
        
        orders_to_remove = []
        
        for order in self.pending_orders:
            if self._should_fill_order(order):
                self._fill_order(order)
                orders_to_remove.append(order)
        
        # Remove filled/cancelled orders
        for order in orders_to_remove:
            if order in self.pending_orders:
                self.pending_orders.remove(order)
    
    def _should_fill_order(self, order: Order) -> bool:
        """Determine if order should be filled"""
        
        symbol_data = self.current_data.get(order.symbol, {})
        
        if not symbol_data:
            return False
        
        current_price = symbol_data.get('close') if self.config.fill_on_bar_close else symbol_data.get('open')
        
        if current_price is None:
            return False
        
        if order.order_type == OrderType.MARKET:
            return True
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                return current_price <= order.price
            else:
                return current_price >= order.price
        
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                return current_price >= order.stop_price
            else:
                return current_price <= order.stop_price
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop triggered, now check limit
            if order.side == OrderSide.BUY:
                if current_price >= order.stop_price:
                    return current_price <= order.price
            else:
                if current_price <= order.stop_price:
                    return current_price >= order.price
        
        return False
    
    def _fill_order(self, order: Order):
        """Fill an order and create trade"""
        
        symbol_data = self.current_data.get(order.symbol, {})
        market_price = symbol_data.get('close') if self.config.fill_on_bar_close else symbol_data.get('open')
        
        # Apply slippage
        fill_price = self.config.slippage_model.apply_slippage(order, market_price)
        
        # For limit orders, ensure we don't get better than limit price
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                fill_price = min(fill_price, order.price)
            else:
                fill_price = max(fill_price, order.price)
        
        # Create trade
        trade_id = f"TRADE_{self._trade_id_counter:06d}"
        self._trade_id_counter += 1
        
        trade = Trade(
            id=trade_id,
            timestamp=self.current_timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=0.0,  # Will be calculated below
            order_id=order.id
        )
        
        # Calculate commission
        trade.commission = self.config.commission_model.calculate_commission(trade)
        
        # Update order
        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.fill_quantity = order.quantity
        order.commission = trade.commission
        
        # Execute trade
        self._execute_trade(trade)
        
        # Record trade
        self.trades.append(trade)
        
        logger.debug(f"Order filled: {order.id} -> {trade_id} at ${fill_price:.4f}")
    
    def _execute_trade(self, trade: Trade):
        """Execute a trade and update positions"""
        
        symbol = trade.symbol
        
        # Update cash
        trade_value = trade.quantity * trade.price
        if trade.side == OrderSide.BUY:
            self.cash -= trade_value + trade.commission
        else:
            self.cash += trade_value - trade.commission
        
        # Update position
        if symbol not in self.positions:
            # New position
            if trade.side == OrderSide.BUY:
                position = Position(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    quantity=trade.quantity,
                    avg_price=trade.price,
                    market_price=trade.price,
                    entry_timestamp=trade.timestamp,
                    last_update=trade.timestamp,
                    total_commission=trade.commission
                )
            else:
                position = Position(
                    symbol=symbol,
                    side=PositionSide.SHORT,
                    quantity=trade.quantity,
                    avg_price=trade.price,
                    market_price=trade.price,
                    entry_timestamp=trade.timestamp,
                    last_update=trade.timestamp,
                    total_commission=trade.commission
                )
            
            self.positions[symbol] = position
        
        else:
            # Existing position
            position = self.positions[symbol]
            
            if ((position.side == PositionSide.LONG and trade.side == OrderSide.BUY) or
                (position.side == PositionSide.SHORT and trade.side == OrderSide.SELL)):
                # Adding to position
                total_quantity = position.quantity + trade.quantity
                total_cost = (position.quantity * position.avg_price + 
                             trade.quantity * trade.price)
                position.avg_price = total_cost / total_quantity
                position.quantity = total_quantity
                position.total_commission += trade.commission
                position.last_update = trade.timestamp
            
            else:
                # Reducing or reversing position
                if trade.quantity >= position.quantity:
                    # Position reversal or closure
                    remaining_quantity = trade.quantity - position.quantity
                    
                    # Calculate realized PnL for closed portion
                    if position.side == PositionSide.LONG:
                        realized_pnl = position.quantity * (trade.price - position.avg_price)
                    else:
                        realized_pnl = position.quantity * (position.avg_price - trade.price)
                    
                    realized_pnl -= trade.commission
                    position.realized_pnl += realized_pnl
                    trade.pnl = realized_pnl
                    
                    if remaining_quantity > 0:
                        # Position reversal
                        position.side = PositionSide.LONG if trade.side == OrderSide.BUY else PositionSide.SHORT
                        position.quantity = remaining_quantity
                        position.avg_price = trade.price
                        position.entry_timestamp = trade.timestamp
                    else:
                        # Position closure
                        del self.positions[symbol]
                
                else:
                    # Partial position reduction
                    if position.side == PositionSide.LONG:
                        realized_pnl = trade.quantity * (trade.price - position.avg_price)
                    else:
                        realized_pnl = trade.quantity * (position.avg_price - trade.price)
                    
                    realized_pnl -= trade.commission
                    position.realized_pnl += realized_pnl
                    position.quantity -= trade.quantity
                    position.total_commission += trade.commission
                    position.last_update = trade.timestamp
                    trade.pnl = realized_pnl
    
    def _update_portfolio(self):
        """Update portfolio positions with current market prices"""
        
        for symbol, position in self.positions.items():
            symbol_data = self.current_data.get(symbol, {})
            
            if symbol_data and 'close' in symbol_data:
                position.market_price = symbol_data['close']
                
                # Calculate unrealized PnL
                if position.side == PositionSide.LONG:
                    position.unrealized_pnl = position.quantity * (position.market_price - position.avg_price)
                else:
                    position.unrealized_pnl = position.quantity * (position.avg_price - position.market_price)
                
                position.last_update = self.current_timestamp
    
    def _record_portfolio_state(self):
        """Record current portfolio state"""
        
        total_value = self.get_portfolio_value()
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_commission = sum(pos.total_commission for pos in self.positions.values())
        
        # Calculate drawdown
        if self.equity_curve:
            peak_value = max(self.equity_curve)
            drawdown = (peak_value - total_value) / peak_value if peak_value > 0 else 0.0
        else:
            drawdown = 0.0
        
        # Create portfolio state
        state = PortfolioState(
            timestamp=self.current_timestamp,
            cash=self.cash,
            total_value=total_value,
            positions=copy.deepcopy(self.positions),
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_commission=total_commission,
            drawdown=drawdown
        )
        
        # Record state
        self.portfolio_history.append(state)
        self.equity_curve.append(total_value)
        self.drawdowns.append(drawdown)
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order before submission"""
        
        # Check basic parameters
        if order.quantity <= 0:
            logger.warning(f"Invalid quantity: {order.quantity}")
            return False
        
        # Check position size limits
        current_position = self.positions.get(order.symbol)
        if current_position:
            current_size = abs(current_position.quantity)
        else:
            current_size = 0
        
        portfolio_value = self.get_portfolio_value()
        symbol_data = self.current_data.get(order.symbol, {})
        
        if symbol_data and 'close' in symbol_data:
            estimated_price = symbol_data['close']
            new_position_value = (current_size + order.quantity) * estimated_price
            position_ratio = new_position_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_ratio > self.config.position_size_limit:
                logger.warning(f"Position size limit exceeded: {position_ratio:.2%} > {self.config.position_size_limit:.2%}")
                return False
        
        # Check cash requirements for buy orders
        if order.side == OrderSide.BUY:
            if order.order_type == OrderType.MARKET and symbol_data and 'close' in symbol_data:
                required_cash = order.quantity * symbol_data['close'] * 1.1  # Add buffer for slippage
                if required_cash > self.cash:
                    logger.warning(f"Insufficient cash: required ${required_cash:.2f}, available ${self.cash:.2f}")
                    return False
            elif order.price:
                required_cash = order.quantity * order.price * 1.05  # Add buffer
                if required_cash > self.cash:
                    logger.warning(f"Insufficient cash for limit order: required ${required_cash:.2f}, available ${self.cash:.2f}")
                    return False
        
        return True
    
    def _check_risk_limits(self):
        """Check and enforce risk limits"""
        
        portfolio_value = self.get_portfolio_value()
        
        # Check leverage
        market_value = sum(abs(pos.quantity * pos.market_price) for pos in self.positions.values())
        current_leverage = market_value / portfolio_value if portfolio_value > 0 else 0
        
        if current_leverage > self.config.max_leverage:
            logger.warning(f"Leverage limit exceeded: {current_leverage:.2f}x > {self.config.max_leverage:.2f}x")
            # In a real implementation, you might force position closures here
    
    def _finalize_backtest(self):
        """Finalize backtest - close all positions"""
        
        # Close all open positions at market prices
        for symbol, position in list(self.positions.items()):
            symbol_data = self.current_data.get(symbol, {})
            
            if symbol_data and 'close' in symbol_data:
                # Submit market order to close position
                if position.side == PositionSide.LONG:
                    self.submit_order(symbol, OrderSide.SELL, position.quantity, OrderType.MARKET)
                else:
                    self.submit_order(symbol, OrderSide.BUY, position.quantity, OrderType.MARKET)
        
        # Process final orders
        self._process_pending_orders()
        self._update_portfolio()
        self._record_portfolio_state()
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive backtest results"""
        
        if not self.equity_curve:
            return {'error': 'No data available for results generation'}
        
        # Basic performance metrics
        initial_value = self.initial_cash
        final_value = self.equity_curve[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate performance metrics
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        results = {
            'summary': {
                'initial_capital': initial_value,
                'final_value': final_value,
                'total_return': total_return,
                'total_trades': len(self.trades),
                'total_orders': len(self.orders),
                'winning_trades': len([t for t in self.trades if t.pnl and t.pnl > 0]),
                'losing_trades': len([t for t in self.trades if t.pnl and t.pnl < 0]),
            },
            'equity_curve': self.equity_curve,
            'drawdowns': self.drawdowns,
            'portfolio_history': self.portfolio_history,
            'orders': self.orders if self.config.track_orders else [],
            'trades': self.trades if self.config.track_trades else [],
            'positions': self.positions if self.config.track_positions else {}
        }
        
        # Add detailed performance metrics if we have enough data
        if len(returns) > 1:
            results['performance'] = {
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'volatility': returns.std() * np.sqrt(252),
                'max_drawdown': max(self.drawdowns) if self.drawdowns else 0,
                'calmar_ratio': total_return / max(self.drawdowns) if self.drawdowns and max(self.drawdowns) > 0 else 0,
                'win_rate': results['summary']['winning_trades'] / len(self.trades) if self.trades else 0,
            }
        
        return results

# ============================================
# Data Handler Interface
# ============================================

class DataHandler:
    """Base class for data handlers"""
    
    def get_data_iterator(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Return iterator over market data"""
        raise NotImplementedError
    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols"""
        raise NotImplementedError

class PandasDataHandler(DataHandler):
    """Data handler for pandas DataFrame"""
    
    def __init__(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
        """
        Initialize with data
        
        Args:
            data: Either a single DataFrame with MultiIndex columns (symbol, field)
                  or a dictionary of {symbol: DataFrame}
        """
        
        if isinstance(data, pd.DataFrame):
            self.data = data
            self.multi_symbol = True
        else:
            # Convert dict to MultiIndex DataFrame
            combined_data = {}
            for symbol, df in data.items():
                for column in df.columns:
                    combined_data[(symbol, column)] = df[column]
            
            self.data = pd.DataFrame(combined_data)
            self.multi_symbol = len(data) > 1
        
        # Ensure datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
    
    def get_data_iterator(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Get iterator over market data"""
        
        # Filter by date range
        data_slice = self.data
        if start_date:
            data_slice = data_slice[data_slice.index >= start_date]
        if end_date:
            data_slice = data_slice[data_slice.index <= end_date]
        
        # Iterate over timestamps
        for timestamp in data_slice.index:
            row_data = data_slice.loc[timestamp]
            
            # Convert to dictionary format
            market_data = {}
            
            if self.multi_symbol:
                # MultiIndex columns
                for (symbol, field), value in row_data.items():
                    if symbol not in market_data:
                        market_data[symbol] = {}
                    market_data[symbol][field] = value
            else:
                # Single symbol
                symbol = 'DEFAULT'
                market_data[symbol] = row_data.to_dict()
            
            yield timestamp, market_data
    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols"""
        if self.multi_symbol:
            return list(set(col[0] for col in self.data.columns))
        else:
            return ['DEFAULT']

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Backtesting Engine")
    
    # Generate sample market data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Create sample price data for multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = {}
    
    for symbol in symbols:
        # Generate realistic price series
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        
        # Create OHLC data
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        data[symbol] = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    print(f"Generated sample data for {len(symbols)} symbols over {len(dates)} days")
    
    # Create simple buy-and-hold strategy for testing
    class SimpleStrategy:
        def __init__(self):
            self.engine = None
            self.initialized = False
        
        def set_engine(self, engine):
            self.engine = engine
        
        def initialize(self):
            self.initialized = True
            print("Strategy initialized")
        
        def on_data(self, timestamp, market_data):
            if not self.initialized:
                return
            
            # Simple strategy: buy on first day, hold until end
            if timestamp.month == 1 and timestamp.day == 3:  # First trading day
                for symbol in market_data.keys():
                    if symbol in ['AAPL', 'MSFT']:  # Only trade these symbols
                        # Buy $100,000 worth of each
                        price = market_data[symbol]['close']
                        quantity = 100000 // price
                        
                        self.engine.submit_order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        )
                        print(f"Submitted buy order for {quantity} shares of {symbol} at ${price:.2f}")
    
    # Create and configure backtest engine
    print("\n1. Testing Basic Backtesting Engine")
    
    config = BacktestConfig(
        initial_cash=1000000,
        commission_model=PercentageCommissionModel(0.001),  # 0.1% commission
        slippage_model=FixedSlippageModel(0.0005),          # 0.05% slippage
        position_size_limit=0.5                              # Max 50% per position
    )
    
    engine = BacktestEngine(config)
    
    # Set data handler and strategy
    data_handler = PandasDataHandler(data)
    strategy = SimpleStrategy()
    
    engine.set_data_handler(data_handler)
    engine.set_strategy(strategy)
    
    # Run backtest
    results = engine.run_backtest('2023-01-01', '2023-12-31')
    
    print("\n2. Backtest Results Summary:")
    print(f"Initial Capital: ${results['summary']['initial_capital']:,.2f}")
    print(f"Final Value: ${results['summary']['final_value']:,.2f}")
    print(f"Total Return: {results['summary']['total_return']:.2%}")
    print(f"Total Trades: {results['summary']['total_trades']}")
    print(f"Total Orders: {results['summary']['total_orders']}")
    
    if 'performance' in results:
        print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results['performance']['max_drawdown']:.2%}")
        print(f"Win Rate: {results['performance']['win_rate']:.1%}")
    
    # Test different commission models
    print("\n3. Testing Commission Models:")
    
    # Fixed commission
    fixed_comm = FixedCommissionModel(9.99)
    test_trade = Trade('TEST', pd.Timestamp.now(), 'AAPL', OrderSide.BUY, 100, 150.0, 0.0, 'ORDER_TEST')
    print(f"Fixed Commission ($9.99): ${fixed_comm.calculate_commission(test_trade):.2f}")
    
    # Percentage commission
    pct_comm = PercentageCommissionModel(0.005, min_commission=1.0)
    print(f"Percentage Commission (0.5%): ${pct_comm.calculate_commission(test_trade):.2f}")
    
    # Tiered commission
    tiers = [(10000, 0.01), (50000, 0.005), (float('inf'), 0.001)]
    tiered_comm = TieredCommissionModel(tiers)
    print(f"Tiered Commission: ${tiered_comm.calculate_commission(test_trade):.2f}")
    
    # Test slippage models
    print("\n4. Testing Slippage Models:")
    
    test_order = Order('TEST', pd.Timestamp.now(), 'AAPL', OrderSide.BUY, OrderType.MARKET, 100)
    market_price = 150.0
    
    fixed_slip = FixedSlippageModel(0.001)
    print(f"Fixed Slippage (0.1%): ${fixed_slip.apply_slippage(test_order, market_price):.4f}")
    
    volume_slip = VolumeBasedSlippageModel(0.0005, 0.000001)
    print(f"Volume-based Slippage: ${volume_slip.apply_slippage(test_order, market_price):.4f}")
    
    # Analyze trades
    print("\n5. Trade Analysis:")
    
    if results['trades']:
        profitable_trades = [t for t in results['trades'] if t.pnl and t.pnl > 0]
        losing_trades = [t for t in results['trades'] if t.pnl and t.pnl < 0]
        
        if profitable_trades:
            avg_profit = np.mean([t.pnl for t in profitable_trades])
            print(f"Average Profitable Trade: ${avg_profit:.2f}")
        
        if losing_trades:
            avg_loss = np.mean([t.pnl for t in losing_trades])
            print(f"Average Losing Trade: ${avg_loss:.2f}")
        
        total_commission = sum(t.commission for t in results['trades'])
        print(f"Total Commission Paid: ${total_commission:.2f}")
    
    # Show equity curve sample
    print("\n6. Equity Curve Sample (First 10 days):")
    for i, value in enumerate(results['equity_curve'][:10]):
        print(f"Day {i+1}: ${value:,.2f}")
    
    print("\nBacktesting engine testing completed successfully!")
