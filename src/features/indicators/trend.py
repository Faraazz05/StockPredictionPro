# ============================================
# StockPredictionPro - src/features/indicators/trend.py
# Comprehensive trend indicators for technical analysis with advanced financial domain support
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

logger = get_logger('features.indicators.trend')

# ============================================
# Moving Average Family
# ============================================

class SimpleMovingAverage(SingleValueIndicator):
    """
    Simple Moving Average (SMA)
    
    The SMA is the arithmetic mean of prices over a specified period.
    It's a lagging indicator that smooths price data to identify trend direction.
    
    Formula: SMA = (P1 + P2 + ... + Pn) / n
    Where P = Price, n = Period
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("SMA", IndicatorType.TREND, config)
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period:
            return self._handle_insufficient_data(len(price_series), self.config.period)
        
        sma = MathUtils.simple_moving_average(
            price_series, 
            self.config.period, 
            self.config.min_periods
        )
        
        return sma

class ExponentialMovingAverage(SingleValueIndicator):
    """
    Exponential Moving Average (EMA)
    
    The EMA gives more weight to recent prices, making it more responsive
    to new information than the SMA.
    
    Formula: EMA = (Price * (2/(n+1))) + (Previous EMA * (1-(2/(n+1))))
    Where n = Period
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None, alpha: Optional[float] = None):
        super().__init__("EMA", IndicatorType.TREND, config)
        self.alpha = alpha
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period:
            return self._handle_insufficient_data(len(price_series), self.config.period)
        
        ema = MathUtils.exponential_moving_average(
            price_series, 
            self.config.period,
            self.alpha
        )
        
        return ema

class WeightedMovingAverage(SingleValueIndicator):
    """
    Weighted Moving Average (WMA)
    
    The WMA assigns linearly decreasing weights to older prices,
    with the most recent price having the highest weight.
    
    Formula: WMA = (P1*n + P2*(n-1) + ... + Pn*1) / (n + (n-1) + ... + 1)
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("WMA", IndicatorType.TREND, config)
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period:
            return self._handle_insufficient_data(len(price_series), self.config.period)
        
        wma = MathUtils.weighted_moving_average(price_series, self.config.period)
        return wma

class DoubleExponentialMovingAverage(SingleValueIndicator):
    """
    Double Exponential Moving Average (DEMA)
    
    DEMA reduces lag by applying EMA twice with a smoothing factor.
    More responsive than EMA while reducing noise.
    
    Formula: DEMA = 2*EMA1 - EMA2
    Where EMA1 = EMA(Price), EMA2 = EMA(EMA1)
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("DEMA", IndicatorType.TREND, config)
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period * 2:
            return self._handle_insufficient_data(len(price_series), self.config.period * 2)
        
        ema1 = MathUtils.exponential_moving_average(price_series, self.config.period)
        ema2 = MathUtils.exponential_moving_average(ema1, self.config.period)
        
        dema = 2 * ema1 - ema2
        return dema

class TripleExponentialMovingAverage(SingleValueIndicator):
    """
    Triple Exponential Moving Average (TEMA)
    
    TEMA reduces lag even further than DEMA by applying EMA three times.
    Highly responsive to price changes while maintaining smoothness.
    
    Formula: TEMA = 3*EMA1 - 3*EMA2 + EMA3
    Where EMA1 = EMA(Price), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("TEMA", IndicatorType.TREND, config)
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period * 3:
            return self._handle_insufficient_data(len(price_series), self.config.period * 3)
        
        ema1 = MathUtils.exponential_moving_average(price_series, self.config.period)
        ema2 = MathUtils.exponential_moving_average(ema1, self.config.period)
        ema3 = MathUtils.exponential_moving_average(ema2, self.config.period)
        
        tema = 3 * ema1 - 3 * ema2 + ema3
        return tema

# ============================================
# Adaptive Moving Averages
# ============================================

class AdaptiveMovingAverage(SingleValueIndicator):
    """
    Adaptive Moving Average (AMA) / Kaufman's Adaptive Moving Average (KAMA)
    
    AMA adjusts its smoothing constant based on market volatility and trend efficiency.
    Fast during trending markets, slow during sideways markets.
    
    Formula: AMA = Previous AMA + SC * (Price - Previous AMA)
    Where SC = Smoothing Constant based on Efficiency Ratio
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None, 
                 fast_period: int = 2, slow_period: int = 30):
        super().__init__("AMA", IndicatorType.TREND, config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_sc = 2.0 / (fast_period + 1)  # Fast smoothing constant
        self.slow_sc = 2.0 / (slow_period + 1)  # Slow smoothing constant
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period + 1:
            return self._handle_insufficient_data(len(price_series), self.config.period + 1)
        
        # Calculate direction (change in price over period)
        direction = abs(price_series - price_series.shift(self.config.period))
        
        # Calculate volatility (sum of absolute changes)
        volatility = abs(price_series.diff()).rolling(window=self.config.period).sum()
        
        # Calculate efficiency ratio
        efficiency_ratio = direction / volatility
        efficiency_ratio = efficiency_ratio.fillna(0)
        
        # Calculate smoothing constant
        smoothing_constant = (efficiency_ratio * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2
        
        # Calculate AMA
        ama = pd.Series(index=price_series.index, dtype=float)
        ama.iloc[self.config.period] = price_series.iloc[self.config.period]
        
        for i in range(self.config.period + 1, len(price_series)):
            ama.iloc[i] = ama.iloc[i-1] + smoothing_constant.iloc[i] * (price_series.iloc[i] - ama.iloc[i-1])
        
        return ama

# ============================================
# MACD Family
# ============================================

class MACD(MultiValueIndicator):
    """
    Moving Average Convergence Divergence (MACD)
    
    MACD is a momentum oscillator that shows the relationship between 
    two moving averages of a price series.
    
    Components:
    - MACD Line: EMA(fast) - EMA(slow)
    - Signal Line: EMA of MACD Line
    - Histogram: MACD Line - Signal Line
    """
    
    def __init__(self, 
                 fast_period: int = 12, 
                 slow_period: int = 26, 
                 signal_period: int = 9,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("MACD", IndicatorType.TREND, config or IndicatorConfig())
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        price_series = self.get_price_series(data)
        
        required_length = max(self.slow_period, self.fast_period) + self.signal_period
        if len(price_series) < required_length:
            logger.warning(f"MACD: Insufficient data. Required: {required_length}, Got: {len(price_series)}")
            
        # Calculate EMAs
        ema_fast = MathUtils.exponential_moving_average(price_series, self.fast_period)
        ema_slow = MathUtils.exponential_moving_average(price_series, self.slow_period)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = MathUtils.exponential_moving_average(macd_line, self.signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Create result DataFrame
        result = pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }, index=price_series.index)
        
        return result

class MACDSignalAnalyzer(BaseIndicator):
    """
    MACD Signal Analyzer
    
    Analyzes MACD for trading signals:
    - Bullish: MACD line crosses above signal line
    - Bearish: MACD line crosses below signal line
    - Divergences: Price vs MACD divergence patterns
    """
    
    def __init__(self, macd_config: Optional[Dict[str, int]] = None):
        super().__init__("MACD_SIGNALS", IndicatorType.TREND)
        self.macd_config = macd_config or {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> IndicatorResult:
        # Calculate MACD first
        macd_indicator = MACD(**self.macd_config)
        macd_result = macd_indicator.calculate(data)
        macd_df = macd_result.values
        
        # Generate signals
        signals = pd.DataFrame(index=macd_df.index)
        
        # Crossover signals
        signals['bullish_crossover'] = (
            (macd_df['macd'] > macd_df['signal']) & 
            (macd_df['macd'].shift(1) <= macd_df['signal'].shift(1))
        )
        
        signals['bearish_crossover'] = (
            (macd_df['macd'] < macd_df['signal']) & 
            (macd_df['macd'].shift(1) >= macd_df['signal'].shift(1))
        )
        
        # Zero line crossings
        signals['macd_above_zero'] = macd_df['macd'] > 0
        signals['macd_below_zero'] = macd_df['macd'] < 0
        
        # Histogram analysis
        signals['histogram_increasing'] = macd_df['histogram'] > macd_df['histogram'].shift(1)
        signals['histogram_decreasing'] = macd_df['histogram'] < macd_df['histogram'].shift(1)
        
        # Signal strength (absolute histogram value)
        signals['signal_strength'] = abs(macd_df['histogram'])
        
        return self.create_result(signals, {'macd_values': macd_df})

# ============================================
# Trend Direction and Strength
# ============================================

class AverageDirectionalIndex(MultiValueIndicator):
    """
    Average Directional Index (ADX)
    
    ADX measures trend strength without regard to trend direction.
    Values above 25 typically indicate a strong trend.
    
    Components:
    - +DI: Positive Directional Indicator
    - -DI: Negative Directional Indicator  
    - ADX: Average Directional Index
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("ADX", IndicatorType.TREND, config or IndicatorConfig(period=14))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("ADX requires OHLC data")
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        if len(data) < self.config.period * 2:
            logger.warning(f"ADX: Insufficient data for reliable calculation")
        
        # Calculate True Range
        tr = MathUtils.true_range(high, low, close)
        
        # Calculate Directional Movement
        plus_dm = pd.Series(index=data.index, dtype=float)
        minus_dm = pd.Series(index=data.index, dtype=float)
        
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_dm = pd.Series(plus_dm, index=data.index)
        minus_dm = pd.Series(minus_dm, index=data.index)
        
        # Smooth the values
        tr_smooth = tr.rolling(window=self.config.period).mean()
        plus_dm_smooth = plus_dm.rolling(window=self.config.period).mean()
        minus_dm_smooth = minus_dm.rolling(window=self.config.period).mean()
        
        # Calculate DI+ and DI-
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)
        
        # Calculate ADX
        adx = dx.rolling(window=self.config.period).mean()
        
        result = pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'dx': dx
        }, index=data.index)
        
        return result

class ParabolicSAR(SingleValueIndicator):
    """
    Parabolic Stop and Reverse (SAR)
    
    Parabolic SAR is a trend-following indicator that provides potential
    reversal points. It appears as dots above or below price.
    
    Formula: SAR = SAR_prev + AF * (EP - SAR_prev)
    Where AF = Acceleration Factor, EP = Extreme Point
    """
    
    def __init__(self, 
                 step: float = 0.02, 
                 max_step: float = 0.2,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("PSAR", IndicatorType.TREND, config or IndicatorConfig())
        self.step = step
        self.max_step = max_step
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Parabolic SAR requires OHLC data")
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        if len(data) < 2:
            return pd.Series([np.nan] * len(data), index=data.index)
        
        # Initialize arrays
        psar = pd.Series(index=data.index, dtype=float)
        ep = pd.Series(index=data.index, dtype=float)
        af = pd.Series(index=data.index, dtype=float)
        trend = pd.Series(index=data.index, dtype=int)
        
        # Initialize first values
        psar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1  # 1 for uptrend, -1 for downtrend
        af.iloc[0] = self.step
        ep.iloc[0] = high.iloc[0]
        
        for i in range(1, len(data)):
            # Calculate SAR
            psar.iloc[i] = psar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - psar.iloc[i-1])
            
            # Determine trend
            if trend.iloc[i-1] == 1:  # Previous uptrend
                if low.iloc[i] <= psar.iloc[i]:  # Trend reversal
                    trend.iloc[i] = -1
                    psar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = self.step
                else:  # Continue uptrend
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + self.step, self.max_step)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            else:  # Previous downtrend
                if high.iloc[i] >= psar.iloc[i]:  # Trend reversal
                    trend.iloc[i] = 1
                    psar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = self.step
                else:  # Continue downtrend
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + self.step, self.max_step)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
            
            # Adjust SAR for next period
            if trend.iloc[i] == 1:  # Uptrend
                psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[i])
            else:  # Downtrend
                psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[i])
        
        return psar

# ============================================
# Trend Channel Indicators
# ============================================

class LinearRegressionLine(SingleValueIndicator):
    """
    Linear Regression Line (Trend Line)
    
    Fits a linear regression line through price data to identify
    the underlying trend direction and strength.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("LINEAR_REG", IndicatorType.TREND, config)
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period:
            return self._handle_insufficient_data(len(price_series), self.config.period)
        
        def linear_reg(window_data):
            if len(window_data) < 2:
                return np.nan
            
            x = np.arange(len(window_data))
            y = window_data.values
            
            # Handle NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return np.nan
            
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Linear regression
            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            return slope * (len(window_data) - 1) + intercept
        
        linear_reg_line = price_series.rolling(
            window=self.config.period,
            min_periods=self.config.min_periods
        ).apply(linear_reg, raw=False)
        
        return linear_reg_line

class LinearRegressionSlope(SingleValueIndicator):
    """
    Linear Regression Slope
    
    Measures the slope of the linear regression line to quantify
    trend strength and direction.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("LINEAR_REG_SLOPE", IndicatorType.TREND, config)
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period:
            return self._handle_insufficient_data(len(price_series), self.config.period)
        
        def calculate_slope(window_data):
            if len(window_data) < 2:
                return np.nan
            
            x = np.arange(len(window_data))
            y = window_data.values
            
            # Handle NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return np.nan
            
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Calculate slope
            slope = np.polyfit(x_clean, y_clean, 1)[0]
            return slope
        
        slope = price_series.rolling(
            window=self.config.period,
            min_periods=self.config.min_periods
        ).apply(calculate_slope, raw=False)
        
        return slope

# ============================================
# Composite Trend Indicators
# ============================================

class TrendStrengthIndex(SingleValueIndicator):
    """
    Custom Trend Strength Index
    
    Combines multiple trend indicators to provide a single
    trend strength measurement from 0 (no trend) to 100 (strong trend).
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("TSI", IndicatorType.TREND, config or IndicatorConfig(period=20))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Trend Strength Index requires OHLC data")
        
        # Calculate component indicators
        close_prices = data['close']
        
        # Moving average slope component
        sma = MathUtils.simple_moving_average(close_prices, self.config.period)
        ma_slope = sma.diff() / sma * 100
        ma_component = np.clip(abs(ma_slope), 0, 10) / 10 * 25  # 0-25 points
        
        # Price vs MA distance component
        ma_distance = abs(close_prices - sma) / sma * 100
        distance_component = np.clip(ma_distance, 0, 5) / 5 * 25  # 0-25 points
        
        # Directional movement component
        if len(data) >= self.config.period:
            adx_indicator = AverageDirectionalIndex(IndicatorConfig(period=min(14, self.config.period)))
            adx_result = adx_indicator.calculate(data)
            adx_values = adx_result.values['adx']
            adx_component = np.clip(adx_values, 0, 50) / 50 * 25  # 0-25 points
        else:
            adx_component = pd.Series([0] * len(data), index=data.index)
        
        # Volatility component (inverse - lower volatility = stronger trend)
        volatility = close_prices.pct_change().rolling(window=self.config.period).std() * np.sqrt(252) * 100
        volatility_component = np.clip(25 - volatility, 0, 25)  # 0-25 points
        
        # Combine components
        tsi = ma_component + distance_component + adx_component + volatility_component
        tsi = np.clip(tsi, 0, 100)
        
        return tsi

# ============================================
# Register All Indicators
# ============================================

# Register moving averages
indicator_registry.register(SimpleMovingAverage, "SMA", IndicatorType.TREND)
indicator_registry.register(ExponentialMovingAverage, "EMA", IndicatorType.TREND)
indicator_registry.register(WeightedMovingAverage, "WMA", IndicatorType.TREND)
indicator_registry.register(DoubleExponentialMovingAverage, "DEMA", IndicatorType.TREND)
indicator_registry.register(TripleExponentialMovingAverage, "TEMA", IndicatorType.TREND)
indicator_registry.register(AdaptiveMovingAverage, "AMA", IndicatorType.TREND)

# Register MACD family
indicator_registry.register(MACD, "MACD", IndicatorType.TREND)
indicator_registry.register(MACDSignalAnalyzer, "MACD_SIGNALS", IndicatorType.TREND)

# Register trend direction indicators
indicator_registry.register(AverageDirectionalIndex, "ADX", IndicatorType.TREND)
indicator_registry.register(ParabolicSAR, "PSAR", IndicatorType.TREND)

# Register trend channel indicators
indicator_registry.register(LinearRegressionLine, "LINEAR_REG", IndicatorType.TREND)
indicator_registry.register(LinearRegressionSlope, "LINEAR_REG_SLOPE", IndicatorType.TREND)

# Register composite indicators
indicator_registry.register(TrendStrengthIndex, "TSI", IndicatorType.TREND)

# ============================================
# Utility Functions
# ============================================

def create_ma_crossover_signals(data: Union[pd.DataFrame, pd.Series],
                               fast_period: int = 10,
                               slow_period: int = 20) -> pd.DataFrame:
    """Create moving average crossover signals"""
    
    sma_fast = SimpleMovingAverage(IndicatorConfig(period=fast_period))
    sma_slow = SimpleMovingAverage(IndicatorConfig(period=slow_period))
    
    fast_ma = sma_fast.calculate(data).values
    slow_ma = sma_slow.calculate(data).values
    
    signals = pd.DataFrame(index=fast_ma.index)
    signals['fast_ma'] = fast_ma
    signals['slow_ma'] = slow_ma
    signals['bullish_cross'] = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    signals['bearish_cross'] = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    signals['trend'] = np.where(fast_ma > slow_ma, 1, -1)
    
    return signals

def analyze_trend_strength(data: Union[pd.DataFrame, pd.Series],
                          periods: List[int] = [10, 20, 50]) -> pd.DataFrame:
    """Analyze trend strength across multiple timeframes"""
    
    results = pd.DataFrame(index=data.index if isinstance(data, pd.DataFrame) else data.index)
    
    for period in periods:
        # Moving average trend
        sma = SimpleMovingAverage(IndicatorConfig(period=period))
        ma_values = sma.calculate(data).values
        
        if isinstance(data, pd.DataFrame):
            price = data['close']
        else:
            price = data
        
        # Trend direction
        results[f'ma_{period}_trend'] = np.where(price > ma_values, 1, -1)
        
        # Trend strength (distance from MA)
        results[f'ma_{period}_strength'] = abs(price - ma_values) / ma_values * 100
        
        # Slope analysis
        slope_indicator = LinearRegressionSlope(IndicatorConfig(period=period))
        slope_values = slope_indicator.calculate(data).values
        results[f'ma_{period}_slope'] = slope_values
    
    # Overall trend consensus
    trend_cols = [col for col in results.columns if '_trend' in col]
    results['trend_consensus'] = results[trend_cols].sum(axis=1) / len(trend_cols)
    
    # Overall strength
    strength_cols = [col for col in results.columns if '_strength' in col]
    results['avg_strength'] = results[strength_cols].mean(axis=1)
    
    return results

@time_it("trend_indicator_suite")
def calculate_trend_suite(data: Union[pd.DataFrame, pd.Series],
                         include_advanced: bool = True) -> Dict[str, IndicatorResult]:
    """Calculate a comprehensive suite of trend indicators"""
    
    results = {}
    
    # Basic moving averages
    for period in [10, 20, 50]:
        sma = SimpleMovingAverage(IndicatorConfig(period=period))
        results[f'sma_{period}'] = sma.calculate(data)
        
        ema = ExponentialMovingAverage(IndicatorConfig(period=period))
        results[f'ema_{period}'] = ema.calculate(data)
    
    # MACD
    if isinstance(data, pd.DataFrame) or len(data) >= 26:
        macd = MACD()
        results['macd'] = macd.calculate(data)
    
    # Advanced indicators (if requested and data is OHLC)
    if include_advanced and isinstance(data, pd.DataFrame):
        # ADX
        adx = AverageDirectionalIndex()
        results['adx'] = adx.calculate(data)
        
        # Parabolic SAR
        psar = ParabolicSAR()
        results['psar'] = psar.calculate(data)
        
        # Trend Strength Index
        tsi = TrendStrengthIndex()
        results['tsi'] = tsi.calculate(data)
    
    logger.info(f"Calculated {len(results)} trend indicators")
    return results

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    from .base import create_sample_data
    
    print("Testing Trend Indicators")
    
    # Create sample data
    sample_data = create_sample_data(200, start_price=100.0, volatility=0.02)
    print(f"Sample data shape: {sample_data.shape}")
    
    # Test individual indicators
    sma = SimpleMovingAverage(IndicatorConfig(period=20))
    sma_result = sma.calculate(sample_data)
    print(f"SMA(20) last 5 values: {sma_result.values.tail()}")
    
    ema = ExponentialMovingAverage(IndicatorConfig(period=20))
    ema_result = ema.calculate(sample_data)
    print(f"EMA(20) last 5 values: {ema_result.values.tail()}")
    
    # Test MACD
    macd = MACD()
    macd_result = macd.calculate(sample_data)
    print(f"MACD components: {macd_result.values.columns.tolist()}")
    print(f"MACD last values:\n{macd_result.values.tail()}")
    
    # Test ADX
    adx = AverageDirectionalIndex()
    adx_result = adx.calculate(sample_data)
    print(f"ADX last values:\n{adx_result.values[['adx', 'plus_di', 'minus_di']].tail()}")
    
    # Test trend suite
    trend_suite = calculate_trend_suite(sample_data)
    print(f"Trend suite calculated {len(trend_suite)} indicators")
    
    # Test crossover signals
    ma_signals = create_ma_crossover_signals(sample_data, 10, 20)
    bullish_crosses = ma_signals['bullish_cross'].sum()
    bearish_crosses = ma_signals['bearish_cross'].sum()
    print(f"MA Crossover Signals - Bullish: {bullish_crosses}, Bearish: {bearish_crosses}")
    
    print("Trend indicators testing completed successfully!")
