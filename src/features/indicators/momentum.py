# ============================================
# StockPredictionPro - src/features/indicators/momentum.py
# Comprehensive momentum indicators for technical analysis with advanced financial domain support
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import warnings

from .base import (
    BaseIndicator, SingleValueIndicator, MultiValueIndicator,
    IndicatorConfig, IndicatorType, IndicatorResult,
    MathUtils, SmoothingMethods, indicator_registry,
    PriceField, TimeFrame
)

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('features.indicators.momentum')

# ============================================
# Core Momentum Indicators
# ============================================

class RelativeStrengthIndex(SingleValueIndicator):
    """
    Relative Strength Index (RSI)
    
    RSI measures the speed and change of price movements. It oscillates between 0-100.
    Values above 70 are typically considered overbought, below 30 oversold.
    
    Formula: RSI = 100 - (100 / (1 + RS))
    Where RS = Average Gain / Average Loss over period
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("RSI", IndicatorType.MOMENTUM, config or IndicatorConfig(period=14))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period + 1:
            return self._handle_insufficient_data(len(price_series), self.config.period + 1)
        
        # Calculate price changes
        delta = price_series.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # Calculate rolling averages using Wilder's smoothing
        alpha = 1.0 / self.config.period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Handle division by zero
        rsi = rsi.fillna(50.0)
        
        return rsi

class RateOfChange(SingleValueIndicator):
    """
    Rate of Change (ROC)
    
    ROC measures the percentage change in price from one period to another.
    It's a momentum oscillator that fluctuates above and below zero.
    
    Formula: ROC = ((Close - Close n periods ago) / Close n periods ago) * 100
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("ROC", IndicatorType.MOMENTUM, config or IndicatorConfig(period=12))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period + 1:
            return self._handle_insufficient_data(len(price_series), self.config.period + 1)
        
        # Calculate rate of change
        roc = price_series.pct_change(periods=self.config.period) * 100
        
        return roc

class MomentumIndicator(SingleValueIndicator):
    """
    Momentum Indicator
    
    Measures the amount that a security's price has changed over a given time span.
    Shows the difference between current price and price n periods ago.
    
    Formula: Momentum = Close - Close n periods ago
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("MOMENTUM", IndicatorType.MOMENTUM, config or IndicatorConfig(period=10))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period + 1:
            return self._handle_insufficient_data(len(price_series), self.config.period + 1)
        
        # Calculate momentum
        momentum = price_series - price_series.shift(self.config.period)
        
        return momentum

class StochasticOscillator(MultiValueIndicator):
    """
    Stochastic Oscillator
    
    Compares the closing price to the price range over a given period.
    Consists of %K (fast stochastic) and %D (slow stochastic).
    
    Formula:
    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA of %K over d_period
    """
    
    def __init__(self, 
                 k_period: int = 14,
                 d_period: int = 3,
                 smooth_k: int = 3,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("STOCHASTIC", IndicatorType.MOMENTUM, config or IndicatorConfig())
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Stochastic Oscillator requires OHLC data")
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        if len(data) < self.k_period:
            logger.warning(f"Stochastic: Insufficient data. Required: {self.k_period}, Got: {len(data)}")
        
        # Calculate highest high and lowest low over k_period
        highest_high = high.rolling(window=self.k_period).max()
        lowest_low = low.rolling(window=self.k_period).min()
        
        # Calculate raw %K
        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        raw_k = raw_k.fillna(50.0)  # Handle division by zero
        
        # Smooth %K if needed
        if self.smooth_k > 1:
            percent_k = raw_k.rolling(window=self.smooth_k).mean()
        else:
            percent_k = raw_k
        
        # Calculate %D (signal line)
        percent_d = percent_k.rolling(window=self.d_period).mean()
        
        result = pd.DataFrame({
            'stoch_k': percent_k,
            'stoch_d': percent_d,
            'stoch_raw': raw_k
        }, index=data.index)
        
        return result

class StochasticRSI(MultiValueIndicator):
    """
    Stochastic RSI (StochRSI)
    
    Applies the Stochastic oscillator formula to RSI values instead of price.
    More sensitive than either RSI or Stochastic alone.
    
    Formula: StochRSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 stoch_period: int = 14,
                 k_smooth: int = 3,
                 d_smooth: int = 3,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("STOCH_RSI", IndicatorType.MOMENTUM, config or IndicatorConfig())
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_smooth = k_smooth
        self.d_smooth = d_smooth
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        price_series = self.get_price_series(data)
        
        # Calculate RSI first
        rsi_indicator = RelativeStrengthIndex(IndicatorConfig(period=self.rsi_period))
        rsi_values = rsi_indicator._calculate_values(data)
        
        # Apply Stochastic formula to RSI
        rsi_high = rsi_values.rolling(window=self.stoch_period).max()
        rsi_low = rsi_values.rolling(window=self.stoch_period).min()
        
        stoch_rsi = (rsi_values - rsi_low) / (rsi_high - rsi_low)
        stoch_rsi = stoch_rsi.fillna(0.5)  # Handle division by zero
        
        # Smooth %K and calculate %D
        stoch_rsi_k = stoch_rsi.rolling(window=self.k_smooth).mean()
        stoch_rsi_d = stoch_rsi_k.rolling(window=self.d_smooth).mean()
        
        result = pd.DataFrame({
            'stochrsi_k': stoch_rsi_k * 100,  # Convert to 0-100 scale
            'stochrsi_d': stoch_rsi_d * 100,
            'stochrsi': stoch_rsi * 100
        }, index=rsi_values.index)
        
        return result

class CommodityChannelIndex(SingleValueIndicator):
    """
    Commodity Channel Index (CCI)
    
    CCI measures the difference between current price and average price.
    Oscillates above and below zero, with no upper or lower bounds.
    
    Formula: CCI = (Typical Price - SMA of TP) / (0.015 × Mean Deviation)
    Where Typical Price = (High + Low + Close) / 3
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("CCI", IndicatorType.MOMENTUM, config or IndicatorConfig(period=20))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("CCI requires OHLC data")
        
        if len(data) < self.config.period:
            return self._handle_insufficient_data(len(data), self.config.period)
        
        # Calculate typical price
        typical_price = MathUtils.typical_price(data['high'], data['low'], data['close'])
        
        # Calculate SMA of typical price
        tp_sma = MathUtils.simple_moving_average(typical_price, self.config.period)
        
        # Calculate mean deviation
        def mean_deviation(window_data):
            if len(window_data) < 1:
                return np.nan
            mean_val = window_data.mean()
            return np.mean(np.abs(window_data - mean_val))
        
        mean_dev = typical_price.rolling(
            window=self.config.period,
            min_periods=self.config.min_periods
        ).apply(mean_deviation, raw=True)
        
        # Calculate CCI
        cci = (typical_price - tp_sma) / (0.015 * mean_dev)
        
        return cci

class WilliamsPercentR(SingleValueIndicator):
    """
    Williams %R
    
    Momentum oscillator that moves between 0 and -100.
    Similar to Stochastic but inverted and on a different scale.
    
    Formula: %R = (Highest High - Close) / (Highest High - Lowest Low) × -100
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("WILLIAMS_R", IndicatorType.MOMENTUM, config or IndicatorConfig(period=14))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Williams %R requires OHLC data")
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        if len(data) < self.config.period:
            return self._handle_insufficient_data(len(data), self.config.period)
        
        # Calculate highest high and lowest low
        highest_high = high.rolling(window=self.config.period).max()
        lowest_low = low.rolling(window=self.config.period).min()
        
        # Calculate Williams %R
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        williams_r = williams_r.fillna(-50.0)  # Handle division by zero
        
        return williams_r

# ============================================
# Advanced Momentum Indicators
# ============================================

class AwesomeOscillator(SingleValueIndicator):
    """
    Awesome Oscillator (AO)
    
    Measures market momentum by comparing recent market momentum to 
    longer-term market momentum.
    
    Formula: AO = SMA(Median Price, 5) - SMA(Median Price, 34)
    Where Median Price = (High + Low) / 2
    """
    
    def __init__(self, 
                 fast_period: int = 5,
                 slow_period: int = 34,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("AO", IndicatorType.MOMENTUM, config or IndicatorConfig())
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Awesome Oscillator requires OHLC data")
        
        if len(data) < self.slow_period:
            return self._handle_insufficient_data(len(data), self.slow_period)
        
        # Calculate median price
        median_price = MathUtils.median_price(data['high'], data['low'])
        
        # Calculate fast and slow SMAs
        fast_sma = MathUtils.simple_moving_average(median_price, self.fast_period)
        slow_sma = MathUtils.simple_moving_average(median_price, self.slow_period)
        
        # Calculate Awesome Oscillator
        ao = fast_sma - slow_sma
        
        return ao

class TrueStrengthIndex(SingleValueIndicator):
    """
    True Strength Index (TSI)
    
    Uses moving averages of the underlying momentum to filter out price noise.
    Oscillates between -100 and +100.
    
    Formula: TSI = 100 × (Double Smoothed Momentum / Double Smoothed |Momentum|)
    """
    
    def __init__(self, 
                 fast_period: int = 13,
                 slow_period: int = 25,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("TSI", IndicatorType.MOMENTUM, config or IndicatorConfig())
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.slow_period * 2:
            return self._handle_insufficient_data(len(price_series), self.slow_period * 2)
        
        # Calculate momentum
        momentum = price_series.diff()
        abs_momentum = momentum.abs()
        
        # First smoothing
        momentum_smooth1 = MathUtils.exponential_moving_average(momentum, self.slow_period)
        abs_momentum_smooth1 = MathUtils.exponential_moving_average(abs_momentum, self.slow_period)
        
        # Second smoothing
        momentum_smooth2 = MathUtils.exponential_moving_average(momentum_smooth1, self.fast_period)
        abs_momentum_smooth2 = MathUtils.exponential_moving_average(abs_momentum_smooth1, self.fast_period)
        
        # Calculate TSI
        tsi = 100 * momentum_smooth2 / abs_momentum_smooth2
        tsi = tsi.fillna(0.0)
        
        return tsi

class UltimateOscillator(SingleValueIndicator):
    """
    Ultimate Oscillator
    
    Uses three different time periods to reduce false signals and 
    improve timing of entries and exits.
    
    Combines short, medium, and long-term price momentum.
    """
    
    def __init__(self, 
                 short_period: int = 7,
                 medium_period: int = 14,
                 long_period: int = 28,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("UO", IndicatorType.MOMENTUM, config or IndicatorConfig())
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Ultimate Oscillator requires OHLC data")
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        if len(data) < self.long_period + 1:
            return self._handle_insufficient_data(len(data), self.long_period + 1)
        
        # Calculate True Low
        prev_close = close.shift(1)
        true_low = pd.concat([low, prev_close], axis=1).min(axis=1)
        
        # Calculate Buying Pressure
        buying_pressure = close - true_low
        
        # Calculate True Range
        true_range = MathUtils.true_range(high, low, close)
        
        # Calculate averages for each period
        def calculate_uo_component(period):
            bp_sum = buying_pressure.rolling(window=period).sum()
            tr_sum = true_range.rolling(window=period).sum()
            return bp_sum / tr_sum
        
        short_avg = calculate_uo_component(self.short_period)
        medium_avg = calculate_uo_component(self.medium_period)
        long_avg = calculate_uo_component(self.long_period)
        
        # Calculate Ultimate Oscillator
        uo = 100 * (4 * short_avg + 2 * medium_avg + long_avg) / 7
        
        return uo

# ============================================
# Momentum Divergence Analyzer
# ============================================

class MomentumDivergenceAnalyzer(BaseIndicator):
    """
    Momentum Divergence Analyzer
    
    Detects bullish and bearish divergences between price and momentum indicators.
    Divergences often signal potential trend reversals.
    """
    
    def __init__(self, 
                 indicator_name: str = "RSI",
                 lookback_period: int = 5,
                 min_divergence_strength: float = 0.5):
        super().__init__("MOMENTUM_DIVERGENCE", IndicatorType.MOMENTUM)
        self.indicator_name = indicator_name
        self.lookback_period = lookback_period
        self.min_divergence_strength = min_divergence_strength
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> IndicatorResult:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Divergence analysis requires OHLC data")
        
        close_prices = data['close']
        
        # Calculate the momentum indicator
        if self.indicator_name == "RSI":
            momentum_indicator = RelativeStrengthIndex()
        elif self.indicator_name == "ROC":
            momentum_indicator = RateOfChange()
        elif self.indicator_name == "CCI":
            momentum_indicator = CommodityChannelIndex()
        else:
            raise ValueError(f"Unsupported indicator: {self.indicator_name}")
        
        momentum_values = momentum_indicator.calculate(data).values
        
        # Find local peaks and troughs
        price_peaks = self._find_peaks(close_prices, self.lookback_period)
        price_troughs = self._find_troughs(close_prices, self.lookback_period)
        momentum_peaks = self._find_peaks(momentum_values, self.lookback_period)
        momentum_troughs = self._find_troughs(momentum_values, self.lookback_period)
        
        # Analyze divergences
        bullish_div = self._detect_bullish_divergence(
            close_prices, momentum_values, price_troughs, momentum_troughs
        )
        bearish_div = self._detect_bearish_divergence(
            close_prices, momentum_values, price_peaks, momentum_peaks
        )
        
        # Create result DataFrame
        result = pd.DataFrame({
            'bullish_divergence': bullish_div,
            'bearish_divergence': bearish_div,
            'momentum_values': momentum_values,
            'price_peaks': price_peaks,
            'price_troughs': price_troughs,
            'momentum_peaks': momentum_peaks,
            'momentum_troughs': momentum_troughs
        }, index=data.index)
        
        return self.create_result(result, {
            'indicator_used': self.indicator_name,
            'divergences_detected': (bullish_div | bearish_div).sum()
        })
    
    def _find_peaks(self, series: pd.Series, window: int) -> pd.Series:
        """Find local peaks in a series"""
        peaks = pd.Series(False, index=series.index)
        
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                peaks.iloc[i] = True
        
        return peaks
    
    def _find_troughs(self, series: pd.Series, window: int) -> pd.Series:
        """Find local troughs in a series"""
        troughs = pd.Series(False, index=series.index)
        
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                troughs.iloc[i] = True
        
        return troughs
    
    def _detect_bullish_divergence(self, price: pd.Series, momentum: pd.Series,
                                 price_troughs: pd.Series, momentum_troughs: pd.Series) -> pd.Series:
        """Detect bullish divergence: lower price lows with higher momentum lows"""
        bullish_div = pd.Series(False, index=price.index)
        
        price_trough_indices = price_troughs[price_troughs].index
        momentum_trough_indices = momentum_troughs[momentum_troughs].index
        
        for i in range(1, len(price_trough_indices)):
            current_price_trough = price_trough_indices[i]
            prev_price_trough = price_trough_indices[i-1]
            
            # Find corresponding momentum troughs
            current_momentum_trough = self._find_nearest_trough(
                current_price_trough, momentum_trough_indices
            )
            prev_momentum_trough = self._find_nearest_trough(
                prev_price_trough, momentum_trough_indices
            )
            
            if current_momentum_trough and prev_momentum_trough:
                # Check for bullish divergence
                price_lower = price[current_price_trough] < price[prev_price_trough]
                momentum_higher = momentum[current_momentum_trough] > momentum[prev_momentum_trough]
                
                if price_lower and momentum_higher:
                    bullish_div[current_price_trough] = True
        
        return bullish_div
    
    def _detect_bearish_divergence(self, price: pd.Series, momentum: pd.Series,
                                 price_peaks: pd.Series, momentum_peaks: pd.Series) -> pd.Series:
        """Detect bearish divergence: higher price highs with lower momentum highs"""
        bearish_div = pd.Series(False, index=price.index)
        
        price_peak_indices = price_peaks[price_peaks].index
        momentum_peak_indices = momentum_peaks[momentum_peaks].index
        
        for i in range(1, len(price_peak_indices)):
            current_price_peak = price_peak_indices[i]
            prev_price_peak = price_peak_indices[i-1]
            
            # Find corresponding momentum peaks
            current_momentum_peak = self._find_nearest_peak(
                current_price_peak, momentum_peak_indices
            )
            prev_momentum_peak = self._find_nearest_peak(
                prev_price_peak, momentum_peak_indices
            )
            
            if current_momentum_peak and prev_momentum_peak:
                # Check for bearish divergence
                price_higher = price[current_price_peak] > price[prev_price_peak]
                momentum_lower = momentum[current_momentum_peak] < momentum[prev_momentum_peak]
                
                if price_higher and momentum_lower:
                    bearish_div[current_price_peak] = True
        
        return bearish_div
    
    def _find_nearest_trough(self, target_index, trough_indices):
        """Find nearest trough to target index"""
        if len(trough_indices) == 0:
            return None
        
        differences = abs(trough_indices - target_index)
        nearest_idx = differences.argmin()
        
        if differences.iloc[nearest_idx] <= pd.Timedelta(days=self.lookback_period * 2):
            return trough_indices[nearest_idx]
        
        return None
    
    def _find_nearest_peak(self, target_index, peak_indices):
        """Find nearest peak to target index"""
        if len(peak_indices) == 0:
            return None
        
        differences = abs(peak_indices - target_index)
        nearest_idx = differences.argmin()
        
        if differences.iloc[nearest_idx] <= pd.Timedelta(days=self.lookback_period * 2):
            return peak_indices[nearest_idx]
        
        return None

# ============================================
# Momentum Signal Generator
# ============================================

class MomentumSignalGenerator(BaseIndicator):
    """
    Comprehensive Momentum Signal Generator
    
    Combines multiple momentum indicators to generate trading signals
    with configurable thresholds and confirmation requirements.
    """
    
    def __init__(self, 
                 indicators: Optional[List[str]] = None,
                 overbought_threshold: float = 70,
                 oversold_threshold: float = 30,
                 signal_confirmation: bool = True):
        super().__init__("MOMENTUM_SIGNALS", IndicatorType.MOMENTUM)
        
        self.indicators = indicators or ["RSI", "STOCHASTIC", "CCI"]
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.signal_confirmation = signal_confirmation
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> IndicatorResult:
        signals = pd.DataFrame(index=data.index)
        indicator_values = {}
        
        # Calculate each indicator
        for indicator_name in self.indicators:
            if indicator_name == "RSI":
                indicator = RelativeStrengthIndex()
                result = indicator.calculate(data)
                values = result.values
                
                # RSI signals
                signals[f'{indicator_name}_overbought'] = values > self.overbought_threshold
                signals[f'{indicator_name}_oversold'] = values < self.oversold_threshold
                
            elif indicator_name == "STOCHASTIC":
                indicator = StochasticOscillator()
                result = indicator.calculate(data)
                values = result.values['stoch_k']
                
                # Stochastic signals
                signals[f'{indicator_name}_overbought'] = values > self.overbought_threshold
                signals[f'{indicator_name}_oversold'] = values < self.oversold_threshold
                
            elif indicator_name == "CCI":
                indicator = CommodityChannelIndex()
                result = indicator.calculate(data)
                values = result.values
                
                # CCI signals (different thresholds)
                signals[f'{indicator_name}_overbought'] = values > 100
                signals[f'{indicator_name}_oversold'] = values < -100
                
            elif indicator_name == "WILLIAMS_R":
                indicator = WilliamsPercentR()
                result = indicator.calculate(data)
                values = result.values
                
                # Williams %R signals
                signals[f'{indicator_name}_overbought'] = values > -20
                signals[f'{indicator_name}_oversold'] = values < -80
            
            indicator_values[indicator_name] = values
        
        # Generate composite signals
        overbought_cols = [col for col in signals.columns if 'overbought' in col]
        oversold_cols = [col for col in signals.columns if 'oversold' in col]
        
        if self.signal_confirmation:
            # Require confirmation from multiple indicators
            min_confirmations = max(1, len(self.indicators) // 2)
            signals['composite_overbought'] = signals[overbought_cols].sum(axis=1) >= min_confirmations
            signals['composite_oversold'] = signals[oversold_cols].sum(axis=1) >= min_confirmations
        else:
            # Any indicator can trigger signal
            signals['composite_overbought'] = signals[overbought_cols].any(axis=1)
            signals['composite_oversold'] = signals[oversold_cols].any(axis=1)
        
        # Generate trading signals
        signals['buy_signal'] = signals['composite_oversold'] & ~signals['composite_oversold'].shift(1)
        signals['sell_signal'] = signals['composite_overbought'] & ~signals['composite_overbought'].shift(1)
        
        # Add indicator values to result
        for name, values in indicator_values.items():
            if isinstance(values, pd.Series):
                signals[f'{name}_values'] = values
            else:
                # For multi-column indicators, add main column
                signals[f'{name}_values'] = values.iloc[:, 0] if hasattr(values, 'iloc') else values
        
        metadata = {
            'indicators_used': self.indicators,
            'overbought_threshold': self.overbought_threshold,
            'oversold_threshold': self.oversold_threshold,
            'total_buy_signals': signals['buy_signal'].sum(),
            'total_sell_signals': signals['sell_signal'].sum()
        }
        
        return self.create_result(signals, metadata)

# ============================================
# Register All Momentum Indicators
# ============================================

# Register core momentum indicators
indicator_registry.register(RelativeStrengthIndex, "RSI", IndicatorType.MOMENTUM)
indicator_registry.register(RateOfChange, "ROC", IndicatorType.MOMENTUM)
indicator_registry.register(MomentumIndicator, "MOMENTUM", IndicatorType.MOMENTUM)
indicator_registry.register(StochasticOscillator, "STOCHASTIC", IndicatorType.MOMENTUM)
indicator_registry.register(StochasticRSI, "STOCH_RSI", IndicatorType.MOMENTUM)
indicator_registry.register(CommodityChannelIndex, "CCI", IndicatorType.MOMENTUM)
indicator_registry.register(WilliamsPercentR, "WILLIAMS_R", IndicatorType.MOMENTUM)

# Register advanced momentum indicators
indicator_registry.register(AwesomeOscillator, "AO", IndicatorType.MOMENTUM)
indicator_registry.register(TrueStrengthIndex, "TSI", IndicatorType.MOMENTUM)
indicator_registry.register(UltimateOscillator, "UO", IndicatorType.MOMENTUM)

# Register analysis tools
indicator_registry.register(MomentumDivergenceAnalyzer, "MOMENTUM_DIVERGENCE", IndicatorType.MOMENTUM)
indicator_registry.register(MomentumSignalGenerator, "MOMENTUM_SIGNALS", IndicatorType.MOMENTUM)

# ============================================
# Utility Functions
# ============================================

def create_momentum_signals(data: Union[pd.DataFrame, pd.Series],
                           rsi_period: int = 14,
                           stoch_k: int = 14,
                           stoch_d: int = 3) -> pd.DataFrame:
    """Create basic momentum signals using RSI and Stochastic"""
    
    # Calculate indicators
    rsi = RelativeStrengthIndex(IndicatorConfig(period=rsi_period))
    stochastic = StochasticOscillator(k_period=stoch_k, d_period=stoch_d)
    
    rsi_values = rsi.calculate(data).values
    stoch_values = stochastic.calculate(data).values
    
    # Create signals
    signals = pd.DataFrame(index=data.index if isinstance(data, pd.DataFrame) else data.index)
    
    # RSI signals
    signals['rsi'] = rsi_values
    signals['rsi_overbought'] = rsi_values > 70
    signals['rsi_oversold'] = rsi_values < 30
    
    # Stochastic signals
    signals['stoch_k'] = stoch_values['stoch_k']
    signals['stoch_d'] = stoch_values['stoch_d']
    signals['stoch_overbought'] = stoch_values['stoch_k'] > 80
    signals['stoch_oversold'] = stoch_values['stoch_k'] < 20
    
    # Combined signals
    signals['buy_signal'] = signals['rsi_oversold'] & signals['stoch_oversold']
    signals['sell_signal'] = signals['rsi_overbought'] & signals['stoch_overbought']
    
    # Stochastic crossover signals
    signals['stoch_bullish_cross'] = (
        (stoch_values['stoch_k'] > stoch_values['stoch_d']) & 
        (stoch_values['stoch_k'].shift(1) <= stoch_values['stoch_d'].shift(1))
    )
    signals['stoch_bearish_cross'] = (
        (stoch_values['stoch_k'] < stoch_values['stoch_d']) & 
        (stoch_values['stoch_k'].shift(1) >= stoch_values['stoch_d'].shift(1))
    )
    
    return signals

@time_it("momentum_suite_calculation")
def calculate_momentum_suite(data: Union[pd.DataFrame, pd.Series],
                           include_advanced: bool = True) -> Dict[str, IndicatorResult]:
    """Calculate comprehensive momentum indicator suite"""
    
    results = {}
    
    # Core momentum indicators
    rsi = RelativeStrengthIndex()
    results['rsi'] = rsi.calculate(data)
    
    roc = RateOfChange()
    results['roc'] = roc.calculate(data)
    
    momentum = MomentumIndicator()
    results['momentum'] = momentum.calculate(data)
    
    # Multi-value indicators (require OHLC)
    if isinstance(data, pd.DataFrame):
        stochastic = StochasticOscillator()
        results['stochastic'] = stochastic.calculate(data)
        
        cci = CommodityChannelIndex()
        results['cci'] = cci.calculate(data)
        
        williams_r = WilliamsPercentR()
        results['williams_r'] = williams_r.calculate(data)
        
        # Advanced indicators
        if include_advanced:
            stoch_rsi = StochasticRSI()
            results['stoch_rsi'] = stoch_rsi.calculate(data)
            
            ao = AwesomeOscillator()
            results['ao'] = ao.calculate(data)
            
            tsi = TrueStrengthIndex()
            results['tsi'] = tsi.calculate(data)
            
            uo = UltimateOscillator()
            results['uo'] = uo.calculate(data)
    
    logger.info(f"Calculated {len(results)} momentum indicators")
    return results

def analyze_momentum_divergences(data: Union[pd.DataFrame, pd.Series],
                               indicators: List[str] = ["RSI", "CCI"]) -> Dict[str, Any]:
    """Analyze momentum divergences across multiple indicators"""
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Divergence analysis requires OHLC data")
    
    divergence_results = {}
    
    for indicator_name in indicators:
        try:
            analyzer = MomentumDivergenceAnalyzer(indicator_name=indicator_name)
            result = analyzer.calculate(data)
            
            bullish_count = result.values['bullish_divergence'].sum()
            bearish_count = result.values['bearish_divergence'].sum()
            
            divergence_results[indicator_name] = {
                'bullish_divergences': bullish_count,
                'bearish_divergences': bearish_count,
                'total_divergences': bullish_count + bearish_count,
                'divergence_data': result.values
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze divergences for {indicator_name}: {e}")
    
    return divergence_results

def create_momentum_dashboard(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Create momentum analysis dashboard"""
    
    dashboard = pd.DataFrame(index=data.index if isinstance(data, pd.DataFrame) else data.index)
    
    # Calculate key momentum indicators
    momentum_suite = calculate_momentum_suite(data, include_advanced=False)
    
    # Add indicator values
    dashboard['rsi'] = momentum_suite['rsi'].values
    dashboard['roc'] = momentum_suite['roc'].values
    dashboard['momentum'] = momentum_suite['momentum'].values
    
    if isinstance(data, pd.DataFrame):
        stoch_values = momentum_suite['stochastic'].values
        dashboard['stoch_k'] = stoch_values['stoch_k']
        dashboard['stoch_d'] = stoch_values['stoch_d']
        dashboard['cci'] = momentum_suite['cci'].values
        dashboard['williams_r'] = momentum_suite['williams_r'].values
    
    # Generate signals
    if isinstance(data, pd.DataFrame):
        signal_generator = MomentumSignalGenerator()
        signals = signal_generator.calculate(data)
        
        dashboard['buy_signal'] = signals.values['buy_signal']
        dashboard['sell_signal'] = signals.values['sell_signal']
        dashboard['overbought'] = signals.values['composite_overbought']
        dashboard['oversold'] = signals.values['composite_oversold']
    
    # Add momentum strength score
    momentum_cols = ['rsi', 'roc', 'momentum']
    if isinstance(data, pd.DataFrame):
        momentum_cols.extend(['stoch_k', 'cci', 'williams_r'])
    
    # Normalize indicators to 0-100 scale and calculate composite strength
    normalized_momentum = pd.DataFrame(index=dashboard.index)
    
    if 'rsi' in dashboard.columns:
        normalized_momentum['rsi_norm'] = dashboard['rsi']
    
    if 'stoch_k' in dashboard.columns:
        normalized_momentum['stoch_norm'] = dashboard['stoch_k']
    
    if 'cci' in dashboard.columns:
        # Normalize CCI to 0-100 scale
        cci_values = dashboard['cci']
        normalized_momentum['cci_norm'] = np.clip((cci_values + 200) / 4, 0, 100)
    
    if 'williams_r' in dashboard.columns:
        # Normalize Williams %R to 0-100 scale
        williams_values = dashboard['williams_r']
        normalized_momentum['williams_norm'] = 100 + williams_values  # Convert from -100,0 to 0,100
    
    # Calculate composite momentum strength
    dashboard['momentum_strength'] = normalized_momentum.mean(axis=1)
    
    return dashboard

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    from .base import create_sample_data
    
    print("Testing Momentum Indicators")
    
    # Create sample data
    sample_data = create_sample_data(200, start_price=100.0, volatility=0.02)
    print(f"Sample data shape: {sample_data.shape}")
    
    # Test individual indicators
    rsi = RelativeStrengthIndex()
    rsi_result = rsi.calculate(sample_data)
    print(f"RSI last 5 values: {rsi_result.values.tail()}")
    
    stochastic = StochasticOscillator()
    stoch_result = stochastic.calculate(sample_data)
    print(f"Stochastic last 5 values:\n{stoch_result.values.tail()}")
    
    cci = CommodityChannelIndex()
    cci_result = cci.calculate(sample_data)
    print(f"CCI last 5 values: {cci_result.values.tail()}")
    
    # Test momentum suite
    momentum_suite = calculate_momentum_suite(sample_data)
    print(f"Momentum suite calculated {len(momentum_suite)} indicators")
    
    # Test signal generation
    momentum_signals = create_momentum_signals(sample_data)
    buy_signals = momentum_signals['buy_signal'].sum()
    sell_signals = momentum_signals['sell_signal'].sum()
    print(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
    
    # Test dashboard
    dashboard = create_momentum_dashboard(sample_data)
    print(f"Dashboard created with {len(dashboard.columns)} columns")
    print(f"Average momentum strength: {dashboard['momentum_strength'].mean():.2f}")
    
    print("Momentum indicators testing completed successfully!")
