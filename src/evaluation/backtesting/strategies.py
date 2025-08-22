# ============================================
# StockPredictionPro - src/evaluation/backtesting/strategies.py
# Comprehensive trading strategy framework for backtesting
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import math

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it
from .engine import OrderSide, OrderType

logger = get_logger('evaluation.backtesting.strategies')

# ============================================
# Strategy Base Classes and Interfaces
# ============================================

class StrategyState(Enum):
    """Strategy execution states"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@dataclass
class TradingSignal:
    """Container for trading signals"""
    timestamp: pd.Timestamp
    symbol: str
    signal_type: SignalType
    strength: float  # Signal strength (0.0 to 1.0)
    price: Optional[float] = None
    quantity: Optional[float] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    trades_executed: int = 0
    win_rate: float = 0.0
    avg_signal_strength: float = 0.0
    signal_accuracy: float = 0.0
    execution_time_ms: float = 0.0

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    This abstract class defines the interface that all trading strategies
    must implement. It provides common functionality and enforces
    the strategy lifecycle methods.
    """
    
    def __init__(self, name: str, symbols: List[str], **kwargs):
        """
        Initialize base strategy
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            **kwargs: Strategy-specific parameters
        """
        
        self.name = name
        self.symbols = symbols
        self.state = StrategyState.INITIALIZED
        self.parameters = kwargs
        
        # Strategy components
        self.engine = None
        self.indicators = {}
        self.signals_history = []
        self.metrics = StrategyMetrics()
        
        # Risk management
        self.position_size = kwargs.get('position_size', 0.1)  # 10% of portfolio
        self.max_positions = kwargs.get('max_positions', 5)
        self.stop_loss_pct = kwargs.get('stop_loss_pct', 0.05)  # 5%
        self.take_profit_pct = kwargs.get('take_profit_pct', 0.15)  # 15%
        
        # Performance tracking
        self.start_time = None
        self.signals_generated = 0
        
        logger.info(f"Strategy '{name}' initialized with symbols: {symbols}")
    
    def set_engine(self, engine):
        """Set the backtesting engine"""
        self.engine = engine
    
    @abstractmethod
    def initialize(self):
        """
        Initialize strategy before backtesting starts.
        Override this method to set up indicators, load data, etc.
        """
        pass
    
    @abstractmethod
    def on_data(self, timestamp: pd.Timestamp, market_data: Dict[str, Any]):
        """
        Process new market data and generate trading signals.
        
        Args:
            timestamp: Current timestamp
            market_data: Dictionary of market data by symbol
        """
        pass
    
    def on_trade_executed(self, trade):
        """
        Called when a trade is executed.
        Override to implement custom trade handling logic.
        
        Args:
            trade: Executed trade object
        """
        self.metrics.trades_executed += 1
    
    def on_order_filled(self, order):
        """
        Called when an order is filled.
        Override to implement custom order handling logic.
        
        Args:
            order: Filled order object
        """
        pass
    
    def generate_signal(self, symbol: str, signal_type: SignalType, 
                       strength: float = 1.0, **kwargs) -> TradingSignal:
        """
        Generate a trading signal
        
        Args:
            symbol: Symbol to trade
            signal_type: Type of signal
            strength: Signal strength (0.0 to 1.0)
            **kwargs: Additional signal metadata
            
        Returns:
            TradingSignal object
        """
        
        signal = TradingSignal(
            timestamp=pd.Timestamp.now(),
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            **kwargs
        )
        
        self.signals_history.append(signal)
        self.signals_generated += 1
        
        # Update metrics
        self.metrics.total_signals += 1
        if signal_type == SignalType.BUY:
            self.metrics.buy_signals += 1
        elif signal_type == SignalType.SELL:
            self.metrics.sell_signals += 1
        
        return signal
    
    def execute_signal(self, signal: TradingSignal):
        """
        Execute a trading signal by placing orders
        
        Args:
            signal: Trading signal to execute
        """
        
        if not self.engine:
            logger.warning("No engine available for signal execution")
            return
        
        try:
            # Calculate position size
            portfolio_value = self.engine.get_portfolio_value()
            position_value = portfolio_value * self.position_size
            
            # Get current price
            current_position = self.engine.get_position(signal.symbol)
            
            if signal.signal_type == SignalType.BUY:
                # Calculate quantity based on position size
                price = signal.price or self._get_current_price(signal.symbol)
                if price and price > 0:
                    quantity = position_value / price
                    
                    # Submit buy order
                    order_id = self.engine.submit_order(
                        symbol=signal.symbol,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        order_type=OrderType.MARKET
                    )
                    
                    logger.debug(f"Buy signal executed: {signal.symbol} x {quantity:.2f} @ {price:.2f}")
            
            elif signal.signal_type == SignalType.SELL:
                if current_position and current_position.quantity > 0:
                    # Submit sell order for entire position
                    order_id = self.engine.submit_order(
                        symbol=signal.symbol,
                        side=OrderSide.SELL,
                        quantity=current_position.quantity,
                        order_type=OrderType.MARKET
                    )
                    
                    logger.debug(f"Sell signal executed: {signal.symbol} x {current_position.quantity:.2f}")
            
            elif signal.signal_type == SignalType.CLOSE:
                if current_position and current_position.quantity != 0:
                    # Close position
                    side = OrderSide.SELL if current_position.quantity > 0 else OrderSide.BUY
                    order_id = self.engine.submit_order(
                        symbol=signal.symbol,
                        side=side,
                        quantity=abs(current_position.quantity),
                        order_type=OrderType.MARKET
                    )
                    
                    logger.debug(f"Close signal executed: {signal.symbol}")
        
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        if hasattr(self.engine, 'current_data') and self.engine.current_data:
            symbol_data = self.engine.current_data.get(symbol, {})
            return symbol_data.get('close')
        return None
    
    def get_metrics(self) -> StrategyMetrics:
        """Get current strategy metrics"""
        if self.signals_history:
            total_strength = sum(s.strength for s in self.signals_history)
            self.metrics.avg_signal_strength = total_strength / len(self.signals_history)
        
        return self.metrics

# ============================================
# Technical Indicator Strategies
# ============================================

class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy
    
    Generates buy signals when short MA crosses above long MA,
    and sell signals when short MA crosses below long MA.
    """
    
    def __init__(self, symbols: List[str], short_window: int = 20, long_window: int = 50, **kwargs):
        super().__init__("MA Crossover", symbols, **kwargs)
        
        self.short_window = short_window
        self.long_window = long_window
        
        # Price history for MA calculation
        self.price_history = {symbol: [] for symbol in symbols}
        self.ma_short_history = {symbol: [] for symbol in symbols}
        self.ma_long_history = {symbol: [] for symbol in symbols}
        
        # Previous MA values for crossover detection
        self.prev_ma_short = {symbol: None for symbol in symbols}
        self.prev_ma_long = {symbol: None for symbol in symbols}
    
    def initialize(self):
        """Initialize strategy"""
        self.state = StrategyState.RUNNING
        logger.info(f"MA Crossover strategy initialized: {self.short_window}/{self.long_window}")
    
    def on_data(self, timestamp: pd.Timestamp, market_data: Dict[str, Any]):
        """Process market data and generate signals"""
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            symbol_data = market_data[symbol]
            close_price = symbol_data.get('close')
            
            if close_price is None:
                continue
            
            # Update price history
            self.price_history[symbol].append(close_price)
            
            # Keep only necessary history
            max_window = max(self.short_window, self.long_window)
            if len(self.price_history[symbol]) > max_window * 2:
                self.price_history[symbol] = self.price_history[symbol][-max_window * 2:]
            
            # Calculate moving averages if we have enough data
            if len(self.price_history[symbol]) >= self.long_window:
                prices = np.array(self.price_history[symbol])
                
                # Calculate MAs
                ma_short = np.mean(prices[-self.short_window:])
                ma_long = np.mean(prices[-self.long_window:])
                
                # Store MA history
                self.ma_short_history[symbol].append(ma_short)
                self.ma_long_history[symbol].append(ma_long)
                
                # Check for crossover signals
                if (self.prev_ma_short[symbol] is not None and 
                    self.prev_ma_long[symbol] is not None):
                    
                    # Bullish crossover (buy signal)
                    if (self.prev_ma_short[symbol] <= self.prev_ma_long[symbol] and
                        ma_short > ma_long):
                        
                        signal = self.generate_signal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            strength=0.8,
                            price=close_price,
                            ma_short=ma_short,
                            ma_long=ma_long
                        )
                        
                        self.execute_signal(signal)
                    
                    # Bearish crossover (sell signal)
                    elif (self.prev_ma_short[symbol] >= self.prev_ma_long[symbol] and
                          ma_short < ma_long):
                        
                        signal = self.generate_signal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            strength=0.8,
                            price=close_price,
                            ma_short=ma_short,
                            ma_long=ma_long
                        )
                        
                        self.execute_signal(signal)
                
                # Update previous values
                self.prev_ma_short[symbol] = ma_short
                self.prev_ma_long[symbol] = ma_long

class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) Strategy
    
    Generates buy signals when RSI is oversold (< 30),
    and sell signals when RSI is overbought (> 70).
    """
    
    def __init__(self, symbols: List[str], rsi_period: int = 14, 
                 oversold_threshold: float = 30, overbought_threshold: float = 70, **kwargs):
        super().__init__("RSI Strategy", symbols, **kwargs)
        
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        
        # Price change history for RSI calculation
        self.price_history = {symbol: [] for symbol in symbols}
        self.rsi_history = {symbol: [] for symbol in symbols}
        self.positions_held = {symbol: False for symbol in symbols}
    
    def initialize(self):
        """Initialize strategy"""
        self.state = StrategyState.RUNNING
        logger.info(f"RSI strategy initialized: period={self.rsi_period}, "
                   f"thresholds=({self.oversold_threshold}, {self.overbought_threshold})")
    
    def on_data(self, timestamp: pd.Timestamp, market_data: Dict[str, Any]):
        """Process market data and generate RSI signals"""
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            symbol_data = market_data[symbol]
            close_price = symbol_data.get('close')
            
            if close_price is None:
                continue
            
            # Update price history
            self.price_history[symbol].append(close_price)
            
            # Keep only necessary history
            if len(self.price_history[symbol]) > self.rsi_period * 2:
                self.price_history[symbol] = self.price_history[symbol][-self.rsi_period * 2:]
            
            # Calculate RSI if we have enough data
            if len(self.price_history[symbol]) > self.rsi_period:
                rsi = self._calculate_rsi(symbol)
                self.rsi_history[symbol].append(rsi)
                
                current_position = self.engine.get_position(symbol) if self.engine else None
                has_position = current_position and current_position.quantity > 0
                
                # Buy signal (RSI oversold and no current position)
                if rsi < self.oversold_threshold and not has_position:
                    signal = self.generate_signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=min(1.0, (self.oversold_threshold - rsi) / self.oversold_threshold),
                        price=close_price,
                        rsi=rsi
                    )
                    
                    self.execute_signal(signal)
                    self.positions_held[symbol] = True
                
                # Sell signal (RSI overbought and has position)
                elif rsi > self.overbought_threshold and has_position:
                    signal = self.generate_signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=min(1.0, (rsi - self.overbought_threshold) / (100 - self.overbought_threshold)),
                        price=close_price,
                        rsi=rsi
                    )
                    
                    self.execute_signal(signal)
                    self.positions_held[symbol] = False
    
    def _calculate_rsi(self, symbol: str) -> float:
        """Calculate RSI for symbol"""
        prices = np.array(self.price_history[symbol])
        
        if len(prices) < 2:
            return 50.0  # Neutral RSI
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        if len(gains) >= self.rsi_period:
            avg_gain = np.mean(gains[-self.rsi_period:])
            avg_loss = np.mean(losses[-self.rsi_period:])
        else:
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
        
        # Avoid division by zero
        if avg_loss == 0:
            return 100.0
        
        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

# ============================================
# Mean Reversion Strategies
# ============================================

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Bollinger Bands
    
    Generates buy signals when price touches lower band,
    and sell signals when price touches upper band.
    """
    
    def __init__(self, symbols: List[str], lookback_period: int = 20, 
                 num_std: float = 2.0, **kwargs):
        super().__init__("Mean Reversion", symbols, **kwargs)
        
        self.lookback_period = lookback_period
        self.num_std = num_std
        
        # Price history for calculations
        self.price_history = {symbol: [] for symbol in symbols}
        self.bollinger_history = {symbol: [] for symbol in symbols}
    
    def initialize(self):
        """Initialize strategy"""
        self.state = StrategyState.RUNNING
        logger.info(f"Mean Reversion strategy initialized: period={self.lookback_period}, "
                   f"std_dev={self.num_std}")
    
    def on_data(self, timestamp: pd.Timestamp, market_data: Dict[str, Any]):
        """Process market data and generate mean reversion signals"""
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            symbol_data = market_data[symbol]
            close_price = symbol_data.get('close')
            
            if close_price is None:
                continue
            
            # Update price history
            self.price_history[symbol].append(close_price)
            
            # Keep only necessary history
            if len(self.price_history[symbol]) > self.lookback_period * 2:
                self.price_history[symbol] = self.price_history[symbol][-self.lookback_period * 2:]
            
            # Calculate Bollinger Bands if we have enough data
            if len(self.price_history[symbol]) >= self.lookback_period:
                prices = np.array(self.price_history[symbol][-self.lookback_period:])
                
                # Calculate bands
                sma = np.mean(prices)
                std_dev = np.std(prices, ddof=1)
                upper_band = sma + (self.num_std * std_dev)
                lower_band = sma - (self.num_std * std_dev)
                
                # Store band values
                self.bollinger_history[symbol].append({
                    'sma': sma,
                    'upper': upper_band,
                    'lower': lower_band,
                    'price': close_price
                })
                
                current_position = self.engine.get_position(symbol) if self.engine else None
                has_position = current_position and current_position.quantity > 0
                
                # Buy signal (price at or below lower band)
                if close_price <= lower_band and not has_position:
                    # Signal strength based on how far below lower band
                    distance_ratio = max(0, (lower_band - close_price) / (upper_band - lower_band))
                    strength = min(1.0, 0.5 + distance_ratio)
                    
                    signal = self.generate_signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=strength,
                        price=close_price,
                        sma=sma,
                        upper_band=upper_band,
                        lower_band=lower_band
                    )
                    
                    self.execute_signal(signal)
                
                # Sell signal (price at or above upper band)
                elif close_price >= upper_band and has_position:
                    # Signal strength based on how far above upper band
                    distance_ratio = max(0, (close_price - upper_band) / (upper_band - lower_band))
                    strength = min(1.0, 0.5 + distance_ratio)
                    
                    signal = self.generate_signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=strength,
                        price=close_price,
                        sma=sma,
                        upper_band=upper_band,
                        lower_band=lower_band
                    )
                    
                    self.execute_signal(signal)

# ============================================
# Momentum Strategies
# ============================================

class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy based on price rate of change
    
    Generates buy signals for strong upward momentum,
    and sell signals for strong downward momentum.
    """
    
    def __init__(self, symbols: List[str], momentum_period: int = 10, 
                 momentum_threshold: float = 0.02, **kwargs):
        super().__init__("Momentum Strategy", symbols, **kwargs)
        
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        
        # Price history for momentum calculation
        self.price_history = {symbol: [] for symbol in symbols}
        self.momentum_history = {symbol: [] for symbol in symbols}
    
    def initialize(self):
        """Initialize strategy"""
        self.state = StrategyState.RUNNING
        logger.info(f"Momentum strategy initialized: period={self.momentum_period}, "
                   f"threshold={self.momentum_threshold:.2%}")
    
    def on_data(self, timestamp: pd.Timestamp, market_data: Dict[str, Any]):
        """Process market data and generate momentum signals"""
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            symbol_data = market_data[symbol]
            close_price = symbol_data.get('close')
            
            if close_price is None:
                continue
            
            # Update price history
            self.price_history[symbol].append(close_price)
            
            # Calculate momentum if we have enough data
            if len(self.price_history[symbol]) > self.momentum_period:
                current_price = self.price_history[symbol][-1]
                past_price = self.price_history[symbol][-(self.momentum_period + 1)]
                
                # Calculate rate of change
                momentum = (current_price - past_price) / past_price
                self.momentum_history[symbol].append(momentum)
                
                current_position = self.engine.get_position(symbol) if self.engine else None
                has_position = current_position and current_position.quantity > 0
                
                # Buy signal (strong positive momentum)
                if momentum > self.momentum_threshold and not has_position:
                    strength = min(1.0, momentum / (self.momentum_threshold * 2))
                    
                    signal = self.generate_signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=strength,
                        price=close_price,
                        momentum=momentum
                    )
                    
                    self.execute_signal(signal)
                
                # Sell signal (strong negative momentum or momentum weakening)
                elif has_position and (momentum < -self.momentum_threshold or 
                                     momentum < self.momentum_threshold * 0.5):
                    strength = min(1.0, abs(momentum) / self.momentum_threshold)
                    
                    signal = self.generate_signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=strength,
                        price=close_price,
                        momentum=momentum
                    )
                    
                    self.execute_signal(signal)

# ============================================
# Multi-Asset Portfolio Strategy
# ============================================

class PortfolioRebalancingStrategy(BaseStrategy):
    """
    Portfolio rebalancing strategy with target allocations
    
    Periodically rebalances portfolio to maintain target weights
    for each asset based on risk-return characteristics.
    """
    
    def __init__(self, symbols: List[str], target_weights: Dict[str, float],
                 rebalance_frequency: int = 21, tolerance: float = 0.05, **kwargs):
        super().__init__("Portfolio Rebalancing", symbols, **kwargs)
        
        self.target_weights = target_weights
        self.rebalance_frequency = rebalance_frequency  # Days
        self.tolerance = tolerance  # 5% tolerance
        
        # Validation
        if abs(sum(target_weights.values()) - 1.0) > 0.01:
            raise ValueError("Target weights must sum to 1.0")
        
        # Tracking
        self.days_since_rebalance = 0
        self.last_rebalance_date = None
        self.rebalance_history = []
    
    def initialize(self):
        """Initialize strategy"""
        self.state = StrategyState.RUNNING
        logger.info(f"Portfolio Rebalancing strategy initialized with targets: {self.target_weights}")
    
    def on_data(self, timestamp: pd.Timestamp, market_data: Dict[str, Any]):
        """Process market data and rebalance portfolio"""
        
        self.days_since_rebalance += 1
        
        # Check if it's time to rebalance
        should_rebalance = (
            self.days_since_rebalance >= self.rebalance_frequency or
            self._check_drift_threshold(market_data)
        )
        
        if should_rebalance:
            self._rebalance_portfolio(timestamp, market_data)
            self.days_since_rebalance = 0
            self.last_rebalance_date = timestamp
    
    def _check_drift_threshold(self, market_data: Dict[str, Any]) -> bool:
        """Check if portfolio has drifted beyond tolerance"""
        
        if not self.engine:
            return False
        
        portfolio_value = self.engine.get_portfolio_value()
        if portfolio_value <= 0:
            return False
        
        current_weights = self._get_current_weights(market_data)
        
        # Check if any weight has drifted beyond tolerance
        for symbol in self.symbols:
            target_weight = self.target_weights.get(symbol, 0)
            current_weight = current_weights.get(symbol, 0)
            
            if abs(current_weight - target_weight) > self.tolerance:
                return True
        
        return False
    
    def _get_current_weights(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        
        if not self.engine:
            return {}
        
        portfolio_value = self.engine.get_portfolio_value()
        if portfolio_value <= 0:
            return {}
        
        current_weights = {}
        
        for symbol in self.symbols:
            position = self.engine.get_position(symbol)
            if position:
                symbol_data = market_data.get(symbol, {})
                current_price = symbol_data.get('close', position.market_price)
                position_value = position.quantity * current_price
                current_weights[symbol] = position_value / portfolio_value
            else:
                current_weights[symbol] = 0.0
        
        return current_weights
    
    def _rebalance_portfolio(self, timestamp: pd.Timestamp, market_data: Dict[str, Any]):
        """Rebalance portfolio to target weights"""
        
        if not self.engine:
            return
        
        portfolio_value = self.engine.get_portfolio_value()
        if portfolio_value <= 0:
            return
        
        current_weights = self._get_current_weights(market_data)
        rebalance_trades = []
        
        logger.info(f"Rebalancing portfolio at {timestamp}")
        
        # Calculate required trades for each symbol
        for symbol in self.symbols:
            target_weight = self.target_weights.get(symbol, 0)
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.01:  # Only trade if difference > 1%
                target_value = portfolio_value * target_weight
                current_position = self.engine.get_position(symbol)
                current_value = current_position.quantity * current_position.market_price if current_position else 0
                
                trade_value = target_value - current_value
                
                symbol_data = market_data.get(symbol, {})
                current_price = symbol_data.get('close')
                
                if current_price and current_price > 0:
                    trade_quantity = trade_value / current_price
                    
                    if abs(trade_quantity) > 0.01:  # Minimum trade size
                        rebalance_trades.append({
                            'symbol': symbol,
                            'quantity': trade_quantity,
                            'price': current_price,
                            'weight_diff': weight_diff
                        })
        
        # Execute rebalancing trades
        for trade in rebalance_trades:
            side = OrderSide.BUY if trade['quantity'] > 0 else OrderSide.SELL
            quantity = abs(trade['quantity'])
            
            signal = self.generate_signal(
                symbol=trade['symbol'],
                signal_type=SignalType.BUY if side == OrderSide.BUY else SignalType.SELL,
                strength=1.0,  # Rebalancing is always full strength
                price=trade['price'],
                quantity=quantity,
                metadata={
                    'rebalance_trade': True,
                    'weight_diff': trade['weight_diff']
                }
            )
            
            # Execute rebalancing order
            order_id = self.engine.submit_order(
                symbol=trade['symbol'],
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            
            logger.debug(f"Rebalance trade: {side.value} {quantity:.2f} {trade['symbol']} "
                        f"@ {trade['price']:.2f}")
        
        # Record rebalancing event
        self.rebalance_history.append({
            'timestamp': timestamp,
            'trades_executed': len(rebalance_trades),
            'portfolio_value': portfolio_value,
            'weights_before': current_weights.copy(),
            'target_weights': self.target_weights.copy()
        })

# ============================================
# Machine Learning Strategy
# ============================================

class MLPredictionStrategy(BaseStrategy):
    """
    Machine Learning based prediction strategy
    
    Uses ML model predictions to generate trading signals.
    Model should output probability or confidence scores.
    """
    
    def __init__(self, symbols: List[str], ml_model, 
                 confidence_threshold: float = 0.6,
                 feature_columns: Optional[List[str]] = None,
                 prediction_horizon: int = 1, **kwargs):
        super().__init__("ML Prediction", symbols, **kwargs)
        
        self.ml_model = ml_model
        self.confidence_threshold = confidence_threshold
        self.feature_columns = feature_columns or ['open', 'high', 'low', 'close', 'volume']
        self.prediction_horizon = prediction_horizon
        
        # Feature history for model input
        self.feature_history = {symbol: [] for symbol in symbols}
        self.prediction_history = {symbol: [] for symbol in symbols}
    
    def initialize(self):
        """Initialize strategy"""
        self.state = StrategyState.RUNNING
        logger.info(f"ML Prediction strategy initialized with threshold: {self.confidence_threshold}")
    
    def on_data(self, timestamp: pd.Timestamp, market_data: Dict[str, Any]):
        """Process market data and generate ML-based signals"""
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            symbol_data = market_data[symbol]
            
            # Extract features
            features = []
            for col in self.feature_columns:
                value = symbol_data.get(col)
                if value is not None:
                    features.append(value)
            
            if len(features) != len(self.feature_columns):
                continue
            
            # Store feature history
            self.feature_history[symbol].append(features)
            
            # Keep reasonable history size
            if len(self.feature_history[symbol]) > 100:
                self.feature_history[symbol] = self.feature_history[symbol][-100:]
            
            # Generate prediction if we have enough history
            if len(self.feature_history[symbol]) >= 10:  # Minimum history for prediction
                try:
                    # Prepare model input (example for sklearn-like models)
                    model_input = np.array([self.feature_history[symbol][-1]])
                    
                    # Get model prediction
                    if hasattr(self.ml_model, 'predict_proba'):
                        # Classification model
                        probabilities = self.ml_model.predict_proba(model_input)[0]
                        prediction_confidence = max(probabilities)
                        predicted_class = np.argmax(probabilities)
                        
                        # Map classes to signals (0=sell, 1=hold, 2=buy)
                        if predicted_class == 2 and prediction_confidence > self.confidence_threshold:
                            signal_type = SignalType.BUY
                        elif predicted_class == 0 and prediction_confidence > self.confidence_threshold:
                            signal_type = SignalType.SELL
                        else:
                            signal_type = SignalType.HOLD
                    
                    elif hasattr(self.ml_model, 'predict'):
                        # Regression model
                        prediction = self.ml_model.predict(model_input)[0]
                        prediction_confidence = min(1.0, abs(prediction))
                        
                        if prediction > self.confidence_threshold:
                            signal_type = SignalType.BUY
                        elif prediction < -self.confidence_threshold:
                            signal_type = SignalType.SELL
                        else:
                            signal_type = SignalType.HOLD
                    
                    else:
                        continue
                    
                    # Store prediction
                    self.prediction_history[symbol].append({
                        'timestamp': timestamp,
                        'prediction': prediction_confidence,
                        'signal_type': signal_type
                    })
                    
                    # Generate and execute signal
                    if signal_type != SignalType.HOLD:
                        current_position = self.engine.get_position(symbol) if self.engine else None
                        has_position = current_position and current_position.quantity > 0
                        
                        # Only act if signal aligns with current position state
                        if ((signal_type == SignalType.BUY and not has_position) or
                            (signal_type == SignalType.SELL and has_position)):
                            
                            signal = self.generate_signal(
                                symbol=symbol,
                                signal_type=signal_type,
                                strength=prediction_confidence,
                                price=symbol_data.get('close'),
                                confidence=prediction_confidence,
                                metadata={
                                    'ml_prediction': True,
                                    'model_type': type(self.ml_model).__name__
                                }
                            )
                            
                            self.execute_signal(signal)
                
                except Exception as e:
                    logger.error(f"Error generating ML prediction for {symbol}: {e}")

# ============================================
# Strategy Factory
# ============================================

class StrategyFactory:
    """Factory for creating predefined strategies"""
    
    @staticmethod
    def create_strategy(strategy_type: str, symbols: List[str], **kwargs) -> BaseStrategy:
        """
        Create a strategy instance
        
        Args:
            strategy_type: Type of strategy to create
            symbols: List of symbols to trade
            **kwargs: Strategy-specific parameters
            
        Returns:
            Strategy instance
        """
        
        strategy_map = {
            'moving_average': MovingAverageCrossoverStrategy,
            'rsi': RSIStrategy,
            'mean_reversion': MeanReversionStrategy,
            'momentum': MomentumStrategy,
            'portfolio_rebalancing': PortfolioRebalancingStrategy,
            'ml_prediction': MLPredictionStrategy
        }
        
        if strategy_type not in strategy_map:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_class = strategy_map[strategy_type]
        return strategy_class(symbols, **kwargs)
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategy types"""
        return [
            'moving_average',
            'rsi', 
            'mean_reversion',
            'momentum',
            'portfolio_rebalancing',
            'ml_prediction'
        ]

# ============================================
# Strategy Utilities
# ============================================

def create_simple_ma_strategy(symbols: List[str], short_ma: int = 20, long_ma: int = 50) -> BaseStrategy:
    """Quick utility to create MA crossover strategy"""
    return MovingAverageCrossoverStrategy(symbols, short_window=short_ma, long_window=long_ma)

def create_rsi_strategy(symbols: List[str], rsi_period: int = 14) -> BaseStrategy:
    """Quick utility to create RSI strategy"""
    return RSIStrategy(symbols, rsi_period=rsi_period)

def create_balanced_portfolio_strategy(symbols: List[str]) -> BaseStrategy:
    """Quick utility to create equal-weight portfolio strategy"""
    target_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
    return PortfolioRebalancingStrategy(symbols, target_weights)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Trading Strategies")
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Test strategy creation
    print("\n1. Testing Strategy Factory")
    
    factory = StrategyFactory()
    available_strategies = factory.get_available_strategies()
    print(f"Available strategies: {available_strategies}")
    
    # Create different strategies
    ma_strategy = factory.create_strategy('moving_average', test_symbols, 
                                        short_window=10, long_window=30)
    print(f"Created: {ma_strategy.name}")
    
    rsi_strategy = factory.create_strategy('rsi', test_symbols, 
                                         rsi_period=14, oversold_threshold=25)
    print(f"Created: {rsi_strategy.name}")
    
    # Test portfolio strategy
    target_weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
    portfolio_strategy = factory.create_strategy('portfolio_rebalancing', test_symbols,
                                               target_weights=target_weights)
    print(f"Created: {portfolio_strategy.name}")
    
    # Test manual strategy creation
    print("\n2. Testing Manual Strategy Creation")
    
    momentum_strategy = MomentumStrategy(test_symbols, momentum_period=15, 
                                       momentum_threshold=0.03)
    print(f"Manually created: {momentum_strategy.name}")
    
    mean_reversion_strategy = MeanReversionStrategy(test_symbols, lookback_period=30, 
                                                  num_std=2.5)
    print(f"Manually created: {mean_reversion_strategy.name}")
    
    # Test strategy initialization
    print("\n3. Testing Strategy Initialization")
    
    strategies = [ma_strategy, rsi_strategy, momentum_strategy, mean_reversion_strategy]
    
    for strategy in strategies:
        try:
            strategy.initialize()
            print(f"✓ {strategy.name}: {strategy.state.value}")
        except Exception as e:
            print(f"✗ {strategy.name}: {e}")
    
    # Test signal generation
    print("\n4. Testing Signal Generation")
    
    # Generate sample market data
    np.random.seed(42)
    sample_data = {}
    base_price = 100
    
    for symbol in test_symbols:
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 50)  # 50 days of returns
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        sample_data[symbol] = {
            'open': prices[-2],
            'high': prices[-1] * 1.01,
            'low': prices[-1] * 0.99,
            'close': prices[-1],
            'volume': np.random.randint(1000000, 10000000)
        }
    
    # Test signal generation for each strategy
    timestamp = pd.Timestamp.now()
    
    for strategy in strategies:
        print(f"\nTesting {strategy.name}:")
        
        # Simulate multiple data points to build history
        for i in range(30):  # 30 data points
            # Create slightly varying data
            current_data = {}
            for symbol in test_symbols:
                price_change = np.random.normal(0, 0.01)
                current_data[symbol] = {
                    'open': sample_data[symbol]['close'] * (1 + price_change * 0.5),
                    'high': sample_data[symbol]['close'] * (1 + abs(price_change)),
                    'low': sample_data[symbol]['close'] * (1 - abs(price_change)),
                    'close': sample_data[symbol]['close'] * (1 + price_change),
                    'volume': sample_data[symbol]['volume']
                }
            
            # Process data
            strategy.on_data(timestamp + pd.Timedelta(days=i), current_data)
        
        # Show results
        metrics = strategy.get_metrics()
        print(f"  Total signals: {metrics.total_signals}")
        print(f"  Buy signals: {metrics.buy_signals}")
        print(f"  Sell signals: {metrics.sell_signals}")
        print(f"  Avg signal strength: {metrics.avg_signal_strength:.3f}")
        
        # Show recent signals
        recent_signals = strategy.signals_history[-3:] if strategy.signals_history else []
        for signal in recent_signals:
            print(f"  Signal: {signal.signal_type.value} {signal.symbol} "
                  f"(strength: {signal.strength:.2f})")
    
    # Test utility functions
    print("\n5. Testing Utility Functions")
    
    simple_ma = create_simple_ma_strategy(['AAPL'], short_ma=5, long_ma=15)
    print(f"Simple MA strategy: {simple_ma.name}")
    
    simple_rsi = create_rsi_strategy(['MSFT'])
    print(f"Simple RSI strategy: {simple_rsi.name}")
    
    balanced_portfolio = create_balanced_portfolio_strategy(['AAPL', 'MSFT', 'GOOGL'])
    print(f"Balanced portfolio strategy: {balanced_portfolio.name}")
    print(f"Target weights: {balanced_portfolio.target_weights}")
    
    # Test ML strategy (mock model)
    print("\n6. Testing ML Strategy (Mock Model)")
    
    class MockMLModel:
        def predict_proba(self, X):
            # Mock predictions: random probabilities for 3 classes (sell, hold, buy)
            np.random.seed(42)
            probs = np.random.dirichlet([1, 2, 1], size=len(X))  # Favor hold
            return probs
    
    mock_model = MockMLModel()
    ml_strategy = MLPredictionStrategy(['AAPL'], mock_model, confidence_threshold=0.5)
    ml_strategy.initialize()
    
    # Test ML strategy with data
    for i in range(15):
        test_data = {
            'AAPL': {
                'open': 100 + i,
                'high': 102 + i,
                'low': 98 + i,
                'close': 101 + i,
                'volume': 5000000 + i * 100000
            }
        }
        
        ml_strategy.on_data(timestamp + pd.Timedelta(days=i), test_data)
    
    ml_metrics = ml_strategy.get_metrics()
    print(f"ML Strategy signals: {ml_metrics.total_signals}")
    print(f"ML Strategy predictions: {len(ml_strategy.prediction_history['AAPL'])}")
    
    print("\nTrading strategies testing completed successfully!")
