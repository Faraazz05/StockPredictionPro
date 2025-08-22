# ============================================
# StockPredictionPro - src/trading/signals/technical_signals.py
# Comprehensive technical indicator-based trading signals for financial markets
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import talib
from scipy import stats

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('trading.signals.technical_signals')

# ============================================
# Signal Data Structures and Enums
# ============================================

class SignalDirection(Enum):
    """Signal direction enum"""
    BUY = 1
    SELL = -1
    HOLD = 0
    STRONG_BUY = 2
    STRONG_SELL = -2

class SignalConfidence(Enum):
    """Signal confidence levels"""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9

@dataclass
class TechnicalSignal:
    """Container for technical trading signals"""
    timestamp: pd.Timestamp
    symbol: str
    indicator: str
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    confidence: SignalConfidence
    price: float
    
    # Technical details
    indicator_value: float
    threshold: Optional[float] = None
    crossover_value: Optional[float] = None
    
    # Context information
    timeframe: str = "1D"
    lookback_period: int = 14
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal data"""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Signal strength must be between 0.0 and 1.0, got {self.strength}")

# ============================================
# Base Technical Indicator
# ============================================

class BaseTechnicalIndicator:
    """
    Base class for all technical indicators.
    
    This class provides common functionality for calculating technical
    indicators and generating trading signals.
    """
    
    def __init__(self, name: str, lookback_period: int = 14):
        self.name = name
        self.lookback_period = lookback_period
        self.signals_generated = 0
        
        # Parameters for signal generation
        self.overbought_threshold = 70
        self.oversold_threshold = 30
        self.signal_threshold = 0.5
        
        logger.debug(f"Initialized {name} indicator with {lookback_period} period")
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate indicator values - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement calculate method")
    
    def generate_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate trading signals - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_signals method")
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input data format"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            raise ValidationError(f"Data must contain columns: {required_columns}")
        
        if len(data) < self.lookback_period:
            raise ValidationError(f"Data length {len(data)} is less than lookback period {self.lookback_period}")
        
        return data.dropna()
    
    def _calculate_signal_strength(self, indicator_value: float, threshold: float, 
                                 current_price: float, price_change: float) -> float:
        """Calculate signal strength based on indicator value and price action"""
        
        # Base strength from indicator distance from threshold
        indicator_strength = min(1.0, abs(indicator_value - threshold) / threshold)
        
        # Price momentum factor (0.5 to 1.5)
        momentum_factor = 1.0 + (price_change / current_price) * 10
        momentum_factor = max(0.5, min(1.5, momentum_factor))
        
        # Combine factors
        strength = indicator_strength * momentum_factor * 0.5
        
        return max(0.0, min(1.0, strength))
    
    def _determine_confidence(self, strength: float, volume_factor: float = 1.0) -> SignalConfidence:
        """Determine signal confidence based on strength and volume"""
        
        adjusted_strength = strength * volume_factor
        
        if adjusted_strength >= 0.8:
            return SignalConfidence.VERY_HIGH
        elif adjusted_strength >= 0.6:
            return SignalConfidence.HIGH
        elif adjusted_strength >= 0.4:
            return SignalConfidence.MEDIUM
        else:
            return SignalConfidence.LOW

# ============================================
# Moving Average Indicators
# ============================================

class SimpleMovingAverage(BaseTechnicalIndicator):
    """
    Simple Moving Average (SMA) indicator and signals.
    
    Generates signals based on price crossovers with moving average
    and moving average crossovers (golden cross / death cross).
    """
    
    def __init__(self, short_period: int = 20, long_period: int = 50):
        super().__init__("SMA", max(short_period, long_period))
        self.short_period = short_period
        self.long_period = long_period
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate SMA values"""
        data = self._validate_data(data)
        
        sma_short = data['close'].rolling(window=self.short_period).mean()
        sma_long = data['close'].rolling(window=self.long_period).mean()
        
        return {
            f'SMA_{self.short_period}': sma_short,
            f'SMA_{self.long_period}': sma_long
        }
    
    @time_it("sma_signal_generation")
    def generate_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate SMA-based trading signals"""
        data = self._validate_data(data)
        smas = self.calculate(data)
        
        sma_short = smas[f'SMA_{self.short_period}']
        sma_long = smas[f'SMA_{self.long_period}']
        
        signals = []
        
        for i in range(1, len(data)):
            current_idx = data.index[i]
            prev_idx = data.index[i-1]
            
            current_price = data.loc[current_idx, 'close']
            current_volume = data.loc[current_idx, 'volume']
            
            # Skip if SMA values are NaN
            if pd.isna(sma_short.iloc[i]) or pd.isna(sma_long.iloc[i]):
                continue
            
            current_sma_short = sma_short.iloc[i]
            current_sma_long = sma_long.iloc[i]
            prev_sma_short = sma_short.iloc[i-1]
            prev_sma_long = sma_long.iloc[i-1]
            
            # Golden Cross: Short MA crosses above Long MA (BUY signal)
            if (prev_sma_short <= prev_sma_long and 
                current_sma_short > current_sma_long):
                
                crossover_strength = abs(current_sma_short - current_sma_long) / current_sma_long
                strength = min(1.0, crossover_strength * 10)
                
                # Volume confirmation factor
                avg_volume = data['volume'].rolling(window=20).mean().iloc[i]
                volume_factor = min(2.0, current_volume / avg_volume) if avg_volume > 0 else 1.0
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="SMA_Golden_Cross",
                    direction=SignalDirection.BUY,
                    strength=strength,
                    confidence=self._determine_confidence(strength, volume_factor),
                    price=current_price,
                    indicator_value=current_sma_short - current_sma_long,
                    crossover_value=current_sma_long,
                    lookback_period=self.long_period,
                    metadata={
                        'sma_short': current_sma_short,
                        'sma_long': current_sma_long,
                        'volume_factor': volume_factor
                    }
                )
                
                signals.append(signal)
                self.signals_generated += 1
            
            # Death Cross: Short MA crosses below Long MA (SELL signal)
            elif (prev_sma_short >= prev_sma_long and 
                  current_sma_short < current_sma_long):
                
                crossover_strength = abs(current_sma_long - current_sma_short) / current_sma_long
                strength = min(1.0, crossover_strength * 10)
                
                # Volume confirmation factor
                avg_volume = data['volume'].rolling(window=20).mean().iloc[i]
                volume_factor = min(2.0, current_volume / avg_volume) if avg_volume > 0 else 1.0
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="SMA_Death_Cross",
                    direction=SignalDirection.SELL,
                    strength=strength,
                    confidence=self._determine_confidence(strength, volume_factor),
                    price=current_price,
                    indicator_value=current_sma_short - current_sma_long,
                    crossover_value=current_sma_long,
                    lookback_period=self.long_period,
                    metadata={
                        'sma_short': current_sma_short,
                        'sma_long': current_sma_long,
                        'volume_factor': volume_factor
                    }
                )
                
                signals.append(signal)
                self.signals_generated += 1
        
        logger.info(f"Generated {len(signals)} SMA signals")
        return signals

class ExponentialMovingAverage(BaseTechnicalIndicator):
    """
    Exponential Moving Average (EMA) indicator and signals.
    
    More responsive to recent price changes than SMA.
    """
    
    def __init__(self, short_period: int = 12, long_period: int = 26):
        super().__init__("EMA", max(short_period, long_period))
        self.short_period = short_period
        self.long_period = long_period
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate EMA values"""
        data = self._validate_data(data)
        
        ema_short = data['close'].ewm(span=self.short_period).mean()
        ema_long = data['close'].ewm(span=self.long_period).mean()
        
        return {
            f'EMA_{self.short_period}': ema_short,
            f'EMA_{self.long_period}': ema_long
        }
    
    def generate_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate EMA-based trading signals"""
        # Similar to SMA but with EMA calculations
        data = self._validate_data(data)
        emas = self.calculate(data)
        
        ema_short = emas[f'EMA_{self.short_period}']
        ema_long = emas[f'EMA_{self.long_period}']
        
        signals = []
        
        for i in range(1, len(data)):
            current_idx = data.index[i]
            current_price = data.loc[current_idx, 'close']
            
            if pd.isna(ema_short.iloc[i]) or pd.isna(ema_long.iloc[i]):
                continue
            
            current_ema_short = ema_short.iloc[i]
            current_ema_long = ema_long.iloc[i]
            prev_ema_short = ema_short.iloc[i-1]
            prev_ema_long = ema_long.iloc[i-1]
            
            # EMA Golden Cross
            if (prev_ema_short <= prev_ema_long and 
                current_ema_short > current_ema_long):
                
                strength = min(1.0, abs(current_ema_short - current_ema_long) / current_ema_long * 10)
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="EMA_Golden_Cross",
                    direction=SignalDirection.BUY,
                    strength=strength,
                    confidence=self._determine_confidence(strength),
                    price=current_price,
                    indicator_value=current_ema_short - current_ema_long,
                    crossover_value=current_ema_long,
                    lookback_period=self.long_period
                )
                
                signals.append(signal)
            
            # EMA Death Cross
            elif (prev_ema_short >= prev_ema_long and 
                  current_ema_short < current_ema_long):
                
                strength = min(1.0, abs(current_ema_long - current_ema_short) / current_ema_long * 10)
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="EMA_Death_Cross",
                    direction=SignalDirection.SELL,
                    strength=strength,
                    confidence=self._determine_confidence(strength),
                    price=current_price,
                    indicator_value=current_ema_short - current_ema_long,
                    crossover_value=current_ema_long,
                    lookback_period=self.long_period
                )
                
                signals.append(signal)
        
        return signals

# ============================================
# Momentum Indicators
# ============================================

class RelativeStrengthIndex(BaseTechnicalIndicator):
    """
    Relative Strength Index (RSI) momentum oscillator.
    
    Identifies overbought/oversold conditions and generates
    reversal signals based on RSI thresholds and divergences.
    """
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__("RSI", period)
        self.period = period
        self.overbought_threshold = overbought
        self.oversold_threshold = oversold
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI values using Wilder's smoothing method"""
        data = self._validate_data(data)
        
        # Calculate price changes
        delta = data['close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate initial averages
        avg_gains = gains.rolling(window=self.period).mean()
        avg_losses = losses.rolling(window=self.period).mean()
        
        # Apply Wilder's smoothing (exponential with alpha = 1/period)
        alpha = 1.0 / self.period
        
        for i in range(self.period, len(gains)):
            avg_gains.iloc[i] = alpha * gains.iloc[i] + (1 - alpha) * avg_gains.iloc[i-1]
            avg_losses.iloc[i] = alpha * losses.iloc[i] + (1 - alpha) * avg_losses.iloc[i-1]
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @time_it("rsi_signal_generation")
    def generate_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate RSI-based trading signals"""
        data = self._validate_data(data)
        rsi = self.calculate(data)
        
        signals = []
        
        for i in range(1, len(data)):
            current_idx = data.index[i]
            current_price = data.loc[current_idx, 'close']
            current_rsi = rsi.iloc[i]
            prev_rsi = rsi.iloc[i-1]
            
            if pd.isna(current_rsi):
                continue
            
            # Oversold condition -> BUY signal
            if (prev_rsi <= self.oversold_threshold and 
                current_rsi > self.oversold_threshold):
                
                # Strength based on how oversold it was
                oversold_depth = max(0, self.oversold_threshold - prev_rsi)
                strength = min(1.0, oversold_depth / self.oversold_threshold + 0.5)
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="RSI_Oversold_Reversal",
                    direction=SignalDirection.BUY,
                    strength=strength,
                    confidence=self._determine_confidence(strength),
                    price=current_price,
                    indicator_value=current_rsi,
                    threshold=self.oversold_threshold,
                    lookback_period=self.period,
                    metadata={
                        'previous_rsi': prev_rsi,
                        'oversold_depth': oversold_depth
                    }
                )
                
                signals.append(signal)
                self.signals_generated += 1
            
            # Overbought condition -> SELL signal
            elif (prev_rsi >= self.overbought_threshold and 
                  current_rsi < self.overbought_threshold):
                
                # Strength based on how overbought it was
                overbought_excess = max(0, prev_rsi - self.overbought_threshold)
                strength = min(1.0, overbought_excess / (100 - self.overbought_threshold) + 0.5)
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="RSI_Overbought_Reversal",
                    direction=SignalDirection.SELL,
                    strength=strength,
                    confidence=self._determine_confidence(strength),
                    price=current_price,
                    indicator_value=current_rsi,
                    threshold=self.overbought_threshold,
                    lookback_period=self.period,
                    metadata={
                        'previous_rsi': prev_rsi,
                        'overbought_excess': overbought_excess
                    }
                )
                
                signals.append(signal)
                self.signals_generated += 1
            
            # Extreme RSI conditions
            elif current_rsi <= 20:  # Extremely oversold -> STRONG BUY
                strength = min(1.0, (20 - current_rsi) / 20 + 0.7)
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="RSI_Extreme_Oversold",
                    direction=SignalDirection.STRONG_BUY,
                    strength=strength,
                    confidence=SignalConfidence.VERY_HIGH,
                    price=current_price,
                    indicator_value=current_rsi,
                    threshold=20,
                    lookback_period=self.period
                )
                
                signals.append(signal)
                self.signals_generated += 1
            
            elif current_rsi >= 80:  # Extremely overbought -> STRONG SELL
                strength = min(1.0, (current_rsi - 80) / 20 + 0.7)
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="RSI_Extreme_Overbought",
                    direction=SignalDirection.STRONG_SELL,
                    strength=strength,
                    confidence=SignalConfidence.VERY_HIGH,
                    price=current_price,
                    indicator_value=current_rsi,
                    threshold=80,
                    lookback_period=self.period
                )
                
                signals.append(signal)
                self.signals_generated += 1
        
        logger.info(f"Generated {len(signals)} RSI signals")
        return signals

class MACD(BaseTechnicalIndicator):
    """
    Moving Average Convergence Divergence (MACD) indicator.
    
    Trend-following momentum indicator that shows the relationship
    between two moving averages of prices.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD", slow_period + signal_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD line, signal line, and histogram"""
        data = self._validate_data(data)
        
        # Calculate EMAs
        ema_fast = data['close'].ewm(span=self.fast_period).mean()
        ema_slow = data['close'].ewm(span=self.slow_period).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD line)
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        
        # MACD histogram
        histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }
    
    @time_it("macd_signal_generation")
    def generate_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate MACD-based trading signals"""
        data = self._validate_data(data)
        macd_data = self.calculate(data)
        
        macd_line = macd_data['MACD']
        signal_line = macd_data['Signal']
        histogram = macd_data['Histogram']
        
        signals = []
        
        for i in range(1, len(data)):
            current_idx = data.index[i]
            current_price = data.loc[current_idx, 'close']
            
            if (pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]) or 
                pd.isna(histogram.iloc[i])):
                continue
            
            current_macd = macd_line.iloc[i]
            current_signal = signal_line.iloc[i]
            current_histogram = histogram.iloc[i]
            
            prev_macd = macd_line.iloc[i-1]
            prev_signal = signal_line.iloc[i-1]
            prev_histogram = histogram.iloc[i-1]
            
            # MACD line crosses above signal line -> BUY
            if prev_macd <= prev_signal and current_macd > current_signal:
                
                # Strength based on histogram magnitude and crossover speed
                crossover_strength = abs(current_macd - current_signal)
                histogram_strength = abs(current_histogram)
                strength = min(1.0, (crossover_strength + histogram_strength) * 1000)
                
                # Additional strength if MACD is below zero (bullish reversal)
                if current_macd < 0:
                    strength *= 1.2
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="MACD_Bullish_Crossover",
                    direction=SignalDirection.BUY,
                    strength=min(1.0, strength),
                    confidence=self._determine_confidence(strength),
                    price=current_price,
                    indicator_value=current_macd,
                    crossover_value=current_signal,
                    lookback_period=self.slow_period,
                    metadata={
                        'macd': current_macd,
                        'signal': current_signal,
                        'histogram': current_histogram,
                        'crossover_below_zero': current_macd < 0
                    }
                )
                
                signals.append(signal)
                self.signals_generated += 1
            
            # MACD line crosses below signal line -> SELL
            elif prev_macd >= prev_signal and current_macd < current_signal:
                
                crossover_strength = abs(current_signal - current_macd)
                histogram_strength = abs(current_histogram)
                strength = min(1.0, (crossover_strength + histogram_strength) * 1000)
                
                # Additional strength if MACD is above zero (bearish reversal)
                if current_macd > 0:
                    strength *= 1.2
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="MACD_Bearish_Crossover",
                    direction=SignalDirection.SELL,
                    strength=min(1.0, strength),
                    confidence=self._determine_confidence(strength),
                    price=current_price,
                    indicator_value=current_macd,
                    crossover_value=current_signal,
                    lookback_period=self.slow_period,
                    metadata={
                        'macd': current_macd,
                        'signal': current_signal,
                        'histogram': current_histogram,
                        'crossover_above_zero': current_macd > 0
                    }
                )
                
                signals.append(signal)
                self.signals_generated += 1
            
            # Zero line crossovers (trend changes)
            elif prev_macd <= 0 and current_macd > 0:  # MACD crosses above zero
                strength = min(1.0, abs(current_macd) * 1000 + 0.6)
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="MACD_Zero_Line_Bullish",
                    direction=SignalDirection.BUY,
                    strength=strength,
                    confidence=self._determine_confidence(strength),
                    price=current_price,
                    indicator_value=current_macd,
                    threshold=0,
                    lookback_period=self.slow_period,
                    metadata={'zero_line_cross': 'bullish'}
                )
                
                signals.append(signal)
                self.signals_generated += 1
            
            elif prev_macd >= 0 and current_macd < 0:  # MACD crosses below zero
                strength = min(1.0, abs(current_macd) * 1000 + 0.6)
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="MACD_Zero_Line_Bearish",
                    direction=SignalDirection.SELL,
                    strength=strength,
                    confidence=self._determine_confidence(strength),
                    price=current_price,
                    indicator_value=current_macd,
                    threshold=0,
                    lookback_period=self.slow_period,
                    metadata={'zero_line_cross': 'bearish'}
                )
                
                signals.append(signal)
                self.signals_generated += 1
        
        logger.info(f"Generated {len(signals)} MACD signals")
        return signals

# ============================================
# Volume Indicators
# ============================================

class VolumeWeightedAveragePrice(BaseTechnicalIndicator):
    """
    Volume Weighted Average Price (VWAP) indicator.
    
    Shows the average price at which a stock has traded
    throughout the day, weighted by volume.
    """
    
    def __init__(self, period: int = 20):
        super().__init__("VWAP", period)
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
        data = self._validate_data(data)
        
        # Typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Volume weighted typical price
        vwtp = typical_price * data['volume']
        
        # VWAP calculation
        vwap = vwtp.rolling(window=self.period).sum() / data['volume'].rolling(window=self.period).sum()
        
        return vwap
    
    def generate_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate VWAP-based trading signals"""
        data = self._validate_data(data)
        vwap = self.calculate(data)
        
        signals = []
        
        for i in range(1, len(data)):
            current_idx = data.index[i]
            current_price = data.loc[current_idx, 'close']
            current_vwap = vwap.iloc[i]
            prev_price = data.loc[data.index[i-1], 'close']
            prev_vwap = vwap.iloc[i-1]
            
            if pd.isna(current_vwap):
                continue
            
            # Price crosses above VWAP -> BUY
            if prev_price <= prev_vwap and current_price > current_vwap:
                
                strength = min(1.0, abs(current_price - current_vwap) / current_vwap * 10)
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="VWAP_Bullish_Cross",
                    direction=SignalDirection.BUY,
                    strength=strength,
                    confidence=self._determine_confidence(strength),
                    price=current_price,
                    indicator_value=current_price - current_vwap,
                    crossover_value=current_vwap,
                    lookback_period=self.period
                )
                
                signals.append(signal)
            
            # Price crosses below VWAP -> SELL
            elif prev_price >= prev_vwap and current_price < current_vwap:
                
                strength = min(1.0, abs(current_vwap - current_price) / current_vwap * 10)
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="VWAP_Bearish_Cross",
                    direction=SignalDirection.SELL,
                    strength=strength,
                    confidence=self._determine_confidence(strength),
                    price=current_price,
                    indicator_value=current_price - current_vwap,
                    crossover_value=current_vwap,
                    lookback_period=self.period
                )
                
                signals.append(signal)
        
        return signals

# ============================================
# Volatility Indicators
# ============================================

class BollingerBands(BaseTechnicalIndicator):
    """
    Bollinger Bands volatility indicator.
    
    Uses standard deviation to create dynamic support and resistance levels.
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("Bollinger_Bands", period)
        self.period = period
        self.std_dev = std_dev
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        data = self._validate_data(data)
        
        # Middle band (SMA)
        middle_band = data['close'].rolling(window=self.period).mean()
        
        # Standard deviation
        std = data['close'].rolling(window=self.period).std()
        
        # Upper and lower bands
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        
        return {
            'Upper_Band': upper_band,
            'Middle_Band': middle_band,
            'Lower_Band': lower_band
        }
    
    @time_it("bollinger_signal_generation")
    def generate_signals(self, data: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate Bollinger Bands trading signals"""
        data = self._validate_data(data)
        bands = self.calculate(data)
        
        upper_band = bands['Upper_Band']
        middle_band = bands['Middle_Band']
        lower_band = bands['Lower_Band']
        
        signals = []
        
        for i in range(1, len(data)):
            current_idx = data.index[i]
            current_price = data.loc[current_idx, 'close']
            prev_price = data.loc[data.index[i-1], 'close']
            
            if (pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]) or 
                pd.isna(middle_band.iloc[i])):
                continue
            
            current_upper = upper_band.iloc[i]
            current_lower = lower_band.iloc[i]
            current_middle = middle_band.iloc[i]
            
            prev_upper = upper_band.iloc[i-1]
            prev_lower = lower_band.iloc[i-1]
            
            # Bollinger Band squeeze breakout
            band_width = (current_upper - current_lower) / current_middle
            prev_band_width = (prev_upper - prev_lower) / middle_band.iloc[i-1]
            
            # Price bounces off lower band -> BUY (mean reversion)
            if (prev_price <= prev_lower and current_price > current_lower):
                
                bounce_strength = (current_price - current_lower) / (current_middle - current_lower)
                strength = min(1.0, bounce_strength + 0.3)
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="BB_Lower_Band_Bounce",
                    direction=SignalDirection.BUY,
                    strength=strength,
                    confidence=self._determine_confidence(strength),
                    price=current_price,
                    indicator_value=current_price,
                    threshold=current_lower,
                    lookback_period=self.period,
                    metadata={
                        'upper_band': current_upper,
                        'middle_band': current_middle,
                        'lower_band': current_lower,
                        'band_width': band_width
                    }
                )
                
                signals.append(signal)
                self.signals_generated += 1
            
            # Price bounces off upper band -> SELL (mean reversion)
            elif (prev_price >= prev_upper and current_price < current_upper):
                
                bounce_strength = (current_upper - current_price) / (current_upper - current_middle)
                strength = min(1.0, bounce_strength + 0.3)
                
                signal = TechnicalSignal(
                    timestamp=current_idx,
                    symbol=data.get('symbol', 'UNKNOWN')[0] if 'symbol' in data else 'UNKNOWN',
                    indicator="BB_Upper_Band_Bounce",
                    direction=SignalDirection.SELL,
                    strength=strength,
                    confidence=self._determine_confidence(strength),
                    price=current_price,
                    indicator_value=current_price,
                    threshold=current_upper,
                    lookback_period=self.period,
                    metadata={
                        'upper_band': current_upper,
                        'middle_band': current_middle,
                        'lower_band': current_lower,
                        'band_width': band_width
                    }
                )
                
                signals.append(signal)
                self.signals_generated += 1
        
        logger.info(f"Generated {len(signals)} Bollinger Bands signals")
        return signals

# ============================================
# Technical Signal Generator
# ============================================

class TechnicalSignalGenerator:
    """
    Comprehensive technical signal generator that combines
    multiple technical indicators to produce trading signals.
    """
    
    def __init__(self):
        self.indicators = {
            'SMA': SimpleMovingAverage(),
            'EMA': ExponentialMovingAverage(),
            'RSI': RelativeStrengthIndex(),
            'MACD': MACD(),
            'VWAP': VolumeWeightedAveragePrice(),
            'BB': BollingerBands()
        }
        
        self.signals_generated = 0
        
        logger.info("Initialized TechnicalSignalGenerator with indicators: " + 
                   ", ".join(self.indicators.keys()))
    
    @time_it("generate_all_signals")
    def generate_all_signals(self, data: pd.DataFrame, 
                           indicators: Optional[List[str]] = None) -> Dict[str, List[TechnicalSignal]]:
        """
        Generate signals from all or specified technical indicators
        
        Args:
            data: OHLCV data
            indicators: List of indicator names to use (None for all)
            
        Returns:
            Dictionary of indicator_name -> signals
        """
        
        if indicators is None:
            indicators = list(self.indicators.keys())
        
        all_signals = {}
        
        for indicator_name in indicators:
            if indicator_name not in self.indicators:
                logger.warning(f"Unknown indicator: {indicator_name}")
                continue
            
            try:
                indicator = self.indicators[indicator_name]
                signals = indicator.generate_signals(data)
                all_signals[indicator_name] = signals
                
                logger.info(f"{indicator_name}: Generated {len(signals)} signals")
                
            except Exception as e:
                logger.error(f"Error generating signals for {indicator_name}: {e}")
                all_signals[indicator_name] = []
        
        # Update total signals count
        self.signals_generated = sum(len(signals) for signals in all_signals.values())
        
        logger.info(f"Total signals generated: {self.signals_generated}")
        
        return all_signals
    
    def get_signal_summary(self, signals_dict: Dict[str, List[TechnicalSignal]]) -> pd.DataFrame:
        """Generate summary of signals by indicator and direction"""
        
        summary_data = []
        
        for indicator, signals in signals_dict.items():
            if not signals:
                continue
            
            # Count by direction
            direction_counts = {}
            confidence_counts = {}
            
            for signal in signals:
                direction = signal.direction.name
                confidence = signal.confidence.name
                
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
                confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
            
            # Calculate average strength
            avg_strength = np.mean([s.strength for s in signals])
            
            summary_data.append({
                'Indicator': indicator,
                'Total_Signals': len(signals),
                'Buy_Signals': direction_counts.get('BUY', 0) + direction_counts.get('STRONG_BUY', 0),
                'Sell_Signals': direction_counts.get('SELL', 0) + direction_counts.get('STRONG_SELL', 0),
                'Avg_Strength': avg_strength,
                'High_Confidence': confidence_counts.get('HIGH', 0) + confidence_counts.get('VERY_HIGH', 0),
                'Low_Confidence': confidence_counts.get('LOW', 0)
            })
        
        return pd.DataFrame(summary_data)

# ============================================
# Utility Functions
# ============================================

def create_technical_signals(data: pd.DataFrame, 
                            indicators: Optional[List[str]] = None) -> Dict[str, List[TechnicalSignal]]:
    """
    Quick utility function to generate technical signals
    
    Args:
        data: OHLCV DataFrame
        indicators: List of indicators to use
        
    Returns:
        Dictionary of signals by indicator
    """
    
    generator = TechnicalSignalGenerator()
    return generator.generate_all_signals(data, indicators)

def filter_signals_by_strength(signals: List[TechnicalSignal], 
                             min_strength: float = 0.5) -> List[TechnicalSignal]:
    """Filter signals by minimum strength threshold"""
    
    return [signal for signal in signals if signal.strength >= min_strength]

def filter_signals_by_confidence(signals: List[TechnicalSignal], 
                                min_confidence: SignalConfidence = SignalConfidence.MEDIUM) -> List[TechnicalSignal]:
    """Filter signals by minimum confidence level"""
    
    return [signal for signal in signals if signal.confidence.value >= min_confidence.value]

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Technical Signals System")
    
    # Generate sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    n_periods = len(dates)
    
    # Create realistic price data
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = 100 * np.cumprod(1 + returns)
    
    # Generate OHLC from close prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    
    # Volume with some correlation to price changes
    volume_base = 1000000
    price_changes = np.diff(prices) / prices[:-1]
    volume_multiplier = 1 + np.abs(np.concatenate([[0], price_changes])) * 5
    volumes = volume_base * (0.5 + np.random.random(n_periods) * volume_multiplier)
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    print(f"Generated sample data: {len(sample_data)} periods")
    print(f"Price range: ${sample_data['close'].min():.2f} - ${sample_data['close'].max():.2f}")
    
    # Test individual indicators
    print("\n1. Testing Individual Indicators")
    
    # Test SMA
    sma_indicator = SimpleMovingAverage(short_period=10, long_period=20)
    sma_signals = sma_indicator.generate_signals(sample_data)
    print(f"SMA Signals: {len(sma_signals)}")
    
    if sma_signals:
        print(f"  First SMA signal: {sma_signals[0].direction.name} at {sma_signals[0].price:.2f} "
              f"(strength: {sma_signals[0].strength:.2f})")
    
    # Test RSI
    rsi_indicator = RelativeStrengthIndex(period=14)
    rsi_signals = rsi_indicator.generate_signals(sample_data)
    print(f"RSI Signals: {len(rsi_signals)}")
    
    if rsi_signals:
        print(f"  First RSI signal: {rsi_signals[0].direction.name} at {rsi_signals[0].price:.2f} "
              f"(RSI: {rsi_signals[0].indicator_value:.1f})")
    
    # Test MACD
    macd_indicator = MACD()
    macd_signals = macd_indicator.generate_signals(sample_data)
    print(f"MACD Signals: {len(macd_signals)}")
    
    if macd_signals:
        print(f"  First MACD signal: {macd_signals[0].direction.name} at {macd_signals[0].price:.2f} "
              f"(strength: {macd_signals[0].strength:.2f})")
    
    # Test Bollinger Bands
    bb_indicator = BollingerBands(period=20)
    bb_signals = bb_indicator.generate_signals(sample_data)
    print(f"Bollinger Bands Signals: {len(bb_signals)}")
    
    print("\n2. Testing Technical Signal Generator")
    
    # Test comprehensive signal generation
    signal_generator = TechnicalSignalGenerator()
    all_signals = signal_generator.generate_all_signals(sample_data)
    
    print("Signal Generation Results:")
    for indicator, signals in all_signals.items():
        if signals:
            buy_signals = sum(1 for s in signals if s.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY])
            sell_signals = sum(1 for s in signals if s.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL])
            avg_strength = np.mean([s.strength for s in signals])
            
            print(f"  {indicator}: {len(signals)} total ({buy_signals} buy, {sell_signals} sell), "
                  f"avg strength: {avg_strength:.2f}")
    
    # Test signal summary
    summary_df = signal_generator.get_signal_summary(all_signals)
    print(f"\n3. Signal Summary:")
    print(summary_df.to_string(index=False))
    
    # Test signal filtering
    print("\n4. Testing Signal Filtering")
    
    # Combine all signals
    all_signals_list = []
    for signals in all_signals.values():
        all_signals_list.extend(signals)
    
    print(f"Total signals before filtering: {len(all_signals_list)}")
    
    # Filter by strength
    high_strength_signals = filter_signals_by_strength(all_signals_list, min_strength=0.7)
    print(f"High strength signals (>= 0.7): {len(high_strength_signals)}")
    
    # Filter by confidence
    high_confidence_signals = filter_signals_by_confidence(all_signals_list, 
                                                          min_confidence=SignalConfidence.HIGH)
    print(f"High confidence signals: {len(high_confidence_signals)}")
    
    # Combined filter
    premium_signals = [s for s in all_signals_list 
                      if s.strength >= 0.6 and s.confidence.value >= SignalConfidence.MEDIUM.value]
    print(f"Premium signals (strength >= 0.6, confidence >= MEDIUM): {len(premium_signals)}")
    
    print("\n5. Signal Quality Analysis")
    
    if all_signals_list:
        # Analyze signal distribution
        strengths = [s.strength for s in all_signals_list]
        print(f"Signal strength distribution:")
        print(f"  Mean: {np.mean(strengths):.2f}")
        print(f"  Std:  {np.std(strengths):.2f}")
        print(f"  Min:  {np.min(strengths):.2f}")
        print(f"  Max:  {np.max(strengths):.2f}")
        
        # Direction distribution
        directions = [s.direction.name for s in all_signals_list]
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        print(f"Signal direction distribution:")
        for direction, count in direction_counts.items():
            print(f"  {direction}: {count} ({count/len(all_signals_list)*100:.1f}%)")
    
    print("\n6. Testing Utility Functions")
    
    # Test quick signal generation
    quick_signals = create_technical_signals(sample_data, indicators=['RSI', 'MACD'])
    total_quick_signals = sum(len(signals) for signals in quick_signals.values())
    print(f"Quick signal generation: {total_quick_signals} signals from RSI and MACD")
    
    # Test with specific time period
    recent_data = sample_data.tail(100)  # Last 100 periods
    recent_signals = create_technical_signals(recent_data)
    total_recent_signals = sum(len(signals) for signals in recent_signals.values())
    print(f"Recent period signals: {total_recent_signals} signals from last 100 periods")
    
    print("\nTechnical signals system testing completed successfully!")
    print("\nGenerated signals include:")
    print("• Moving Average Crossovers: Golden cross and death cross signals")
    print("• RSI Reversals: Overbought/oversold and extreme conditions")
    print("• MACD Signals: Bullish/bearish crossovers and zero line crosses")
    print("• Bollinger Bands: Mean reversion and volatility breakouts")
    print("• Volume Indicators: VWAP crossovers with volume confirmation")
    print("• Signal Quality Metrics: Strength, confidence, and filtering capabilities")
