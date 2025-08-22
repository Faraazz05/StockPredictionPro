# ============================================
# StockPredictionPro - src/features/indicators/volatility.py
# Comprehensive volatility indicators for technical analysis with advanced financial domain support
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import warnings
from scipy import stats

from .base import (
    BaseIndicator, SingleValueIndicator, MultiValueIndicator,
    IndicatorConfig, IndicatorType, IndicatorResult,
    MathUtils, SmoothingMethods, indicator_registry,
    PriceField, TimeFrame
)

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('features.indicators.volatility')

# ============================================
# Core Volatility Indicators
# ============================================

class BollingerBands(MultiValueIndicator):
    """
    Bollinger Bands
    
    Bollinger Bands consist of a moving average and two standard deviation bands.
    They expand and contract based on volatility, helping identify overbought/oversold conditions.
    
    Components:
    - Middle Band: Simple Moving Average
    - Upper Band: SMA + (Standard Deviation × multiplier)
    - Lower Band: SMA - (Standard Deviation × multiplier)
    """
    
    def __init__(self, 
                 std_multiplier: float = 2.0,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("BOLLINGER_BANDS", IndicatorType.VOLATILITY, config or IndicatorConfig(period=20))
        self.std_multiplier = std_multiplier
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period:
            return self._handle_insufficient_data(len(price_series), self.config.period)
        
        # Calculate moving average and standard deviation
        sma = MathUtils.simple_moving_average(price_series, self.config.period, self.config.min_periods)
        rolling_std = MathUtils.standard_deviation(price_series, self.config.period)
        
        # Calculate bands
        upper_band = sma + (self.std_multiplier * rolling_std)
        lower_band = sma - (self.std_multiplier * rolling_std)
        
        # Calculate additional metrics
        bandwidth = (upper_band - lower_band) / sma * 100  # Bandwidth as percentage
        percent_b = (price_series - lower_band) / (upper_band - lower_band)  # %B indicator
        
        result = pd.DataFrame({
            'bb_middle': sma,
            'bb_upper': upper_band,
            'bb_lower': lower_band,
            'bb_bandwidth': bandwidth,
            'bb_percent_b': percent_b
        }, index=price_series.index)
        
        return result

class AverageTrueRange(SingleValueIndicator):
    """
    Average True Range (ATR)
    
    ATR measures volatility by calculating the average of true ranges over a period.
    Higher ATR values indicate higher volatility.
    
    True Range = max(high - low, |high - previous_close|, |low - previous_close|)
    ATR = Average of True Range over period
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("ATR", IndicatorType.VOLATILITY, config or IndicatorConfig(period=14))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("ATR requires OHLC data")
        
        if len(data) < self.config.period:
            return self._handle_insufficient_data(len(data), self.config.period)
        
        # Calculate True Range
        true_range = MathUtils.true_range(data['high'], data['low'], data['close'])
        
        # Calculate ATR using smoothed moving average (Wilder's smoothing)
        alpha = 1.0 / self.config.period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()
        
        return atr

class HistoricalVolatility(SingleValueIndicator):
    """
    Historical Volatility (HV)
    
    Measures the standard deviation of logarithmic returns over a period.
    Typically annualized by multiplying by sqrt(252) for daily data.
    
    Formula: HV = σ(log(P(t)/P(t-1))) × √252
    Where σ is standard deviation, P is price
    """
    
    def __init__(self, 
                 annualized: bool = True,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("HIST_VOL", IndicatorType.VOLATILITY, config or IndicatorConfig(period=20))
        self.annualized = annualized
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period + 1:
            return self._handle_insufficient_data(len(price_series), self.config.period + 1)
        
        # Calculate logarithmic returns
        log_returns = np.log(price_series / price_series.shift(1))
        
        # Calculate rolling standard deviation
        volatility = log_returns.rolling(
            window=self.config.period,
            min_periods=self.config.min_periods
        ).std()
        
        # Annualize if requested
        if self.annualized:
            volatility = volatility * np.sqrt(252)  # Assuming 252 trading days per year
        
        return volatility

class ChaikinVolatility(SingleValueIndicator):
    """
    Chaikin Volatility
    
    Measures the rate of change of the trading range (high-low spread).
    Positive values indicate increasing volatility, negative values decreasing volatility.
    
    Formula: CV = ((EMA(H-L) - EMA(H-L)[n periods ago]) / EMA(H-L)[n periods ago]) × 100
    """
    
    def __init__(self, 
                 ema_period: int = 10,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("CHAIKIN_VOL", IndicatorType.VOLATILITY, config or IndicatorConfig(period=10))
        self.ema_period = ema_period
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Chaikin Volatility requires OHLC data")
        
        if len(data) < self.config.period + self.ema_period:
            return self._handle_insufficient_data(len(data), self.config.period + self.ema_period)
        
        # Calculate high-low range
        high_low_range = data['high'] - data['low']
        
        # Smooth the range with EMA
        ema_range = MathUtils.exponential_moving_average(high_low_range, self.ema_period)
        
        # Calculate rate of change
        chaikin_volatility = MathUtils.rate_of_change(ema_range, self.config.period)
        
        return chaikin_volatility

class RelativeVolatility(SingleValueIndicator):
    """
    Relative Volatility Index (RVI)
    
    Similar to RSI but uses standard deviation instead of price changes.
    Values above 50 indicate higher volatility, below 50 lower volatility.
    
    Formula: RVI = 100 × (U / (U + D))
    Where U = upward volatility, D = downward volatility
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("RVI", IndicatorType.VOLATILITY, config or IndicatorConfig(period=14))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period * 2:
            return self._handle_insufficient_data(len(price_series), self.config.period * 2)
        
        # Calculate standard deviation of returns
        returns = price_series.pct_change()
        rolling_std = returns.rolling(window=10).std()  # 10-period rolling std
        
        # Determine direction of price change
        price_direction = (price_series > price_series.shift(1)).astype(int)
        
        # Separate upward and downward volatility
        up_vol = rolling_std.where(price_direction == 1, 0)
        down_vol = rolling_std.where(price_direction == 0, 0)
        
        # Calculate smoothed averages
        avg_up = up_vol.rolling(window=self.config.period).mean()
        avg_down = down_vol.rolling(window=self.config.period).mean()
        
        # Calculate RVI
        rvi = 100 * avg_up / (avg_up + avg_down)
        rvi = rvi.fillna(50.0)  # Handle division by zero
        
        return rvi

# ============================================
# Advanced Volatility Indicators
# ============================================

class KeltnerChannels(MultiValueIndicator):
    """
    Keltner Channels
    
    Similar to Bollinger Bands but uses ATR instead of standard deviation.
    More responsive to volatility changes.
    
    Components:
    - Middle Line: EMA of price
    - Upper Channel: EMA + (ATR × multiplier)
    - Lower Channel: EMA - (ATR × multiplier)
    """
    
    def __init__(self, 
                 atr_multiplier: float = 2.0,
                 atr_period: int = 14,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("KELTNER_CHANNELS", IndicatorType.VOLATILITY, config or IndicatorConfig(period=20))
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Keltner Channels require OHLC data")
        
        price_series = self.get_price_series(data)
        
        if len(data) < max(self.config.period, self.atr_period):
            return self._handle_insufficient_data(len(data), max(self.config.period, self.atr_period))
        
        # Calculate EMA of price (middle line)
        middle_line = MathUtils.exponential_moving_average(price_series, self.config.period)
        
        # Calculate ATR
        atr_indicator = AverageTrueRange(IndicatorConfig(period=self.atr_period))
        atr_values = atr_indicator._calculate_values(data)
        
        # Calculate channels
        upper_channel = middle_line + (self.atr_multiplier * atr_values)
        lower_channel = middle_line - (self.atr_multiplier * atr_values)
        
        result = pd.DataFrame({
            'kc_middle': middle_line,
            'kc_upper': upper_channel,
            'kc_lower': lower_channel,
            'kc_width': upper_channel - lower_channel
        }, index=data.index)
        
        return result

class DonchianChannels(MultiValueIndicator):
    """
    Donchian Channels
    
    Uses the highest high and lowest low over a period to create channels.
    Measures volatility based on price extremes.
    
    Components:
    - Upper Channel: Highest high over period
    - Lower Channel: Lowest low over period
    - Middle Channel: (Upper + Lower) / 2
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("DONCHIAN_CHANNELS", IndicatorType.VOLATILITY, config or IndicatorConfig(period=20))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Donchian Channels require OHLC data")
        
        if len(data) < self.config.period:
            return self._handle_insufficient_data(len(data), self.config.period)
        
        # Calculate highest high and lowest low
        upper_channel = data['high'].rolling(window=self.config.period).max()
        lower_channel = data['low'].rolling(window=self.config.period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        # Calculate channel width as volatility measure
        channel_width = upper_channel - lower_channel
        
        result = pd.DataFrame({
            'dc_upper': upper_channel,
            'dc_lower': lower_channel,
            'dc_middle': middle_channel,
            'dc_width': channel_width
        }, index=data.index)
        
        return result

class VolatilityRatio(SingleValueIndicator):
    """
    Volatility Ratio (VR)
    
    Compares short-term volatility to long-term volatility.
    Values > 1 indicate increasing volatility, < 1 decreasing volatility.
    
    Formula: VR = Short-term volatility / Long-term volatility
    """
    
    def __init__(self, 
                 short_period: int = 10,
                 long_period: int = 30,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("VOL_RATIO", IndicatorType.VOLATILITY, config or IndicatorConfig())
        self.short_period = short_period
        self.long_period = long_period
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.long_period + 1:
            return self._handle_insufficient_data(len(price_series), self.long_period + 1)
        
        # Calculate returns
        returns = price_series.pct_change()
        
        # Calculate short and long-term volatilities
        short_vol = returns.rolling(window=self.short_period).std()
        long_vol = returns.rolling(window=self.long_period).std()
        
        # Calculate volatility ratio
        vol_ratio = short_vol / long_vol
        vol_ratio = vol_ratio.fillna(1.0)  # Handle division by zero
        
        return vol_ratio

class UlcerIndex(SingleValueIndicator):
    """
    Ulcer Index (UI)
    
    Measures downside volatility by focusing on drawdowns from recent highs.
    Higher values indicate higher downside risk.
    
    Formula: UI = sqrt(sum((% drawdown)²) / period)
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("ULCER_INDEX", IndicatorType.VOLATILITY, config or IndicatorConfig(period=14))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        price_series = self.get_price_series(data)
        
        if len(price_series) < self.config.period:
            return self._handle_insufficient_data(len(price_series), self.config.period)
        
        # Calculate rolling maximum (peak)
        rolling_max = price_series.rolling(window=self.config.period, min_periods=1).max()
        
        # Calculate percentage drawdown
        drawdown = (price_series - rolling_max) / rolling_max * 100
        
        # Calculate squared drawdowns
        drawdown_squared = drawdown ** 2
        
        # Calculate Ulcer Index
        ulcer_index = np.sqrt(
            drawdown_squared.rolling(window=self.config.period).mean()
        )
        
        return ulcer_index

# ============================================
# Volatility Breakout Indicators
# ============================================

class VolatilityBreakout(BaseIndicator):
    """
    Volatility Breakout Detector
    
    Detects when volatility breaks above or below significant levels.
    Useful for identifying periods of expanding or contracting volatility.
    """
    
    def __init__(self, 
                 volatility_indicator: str = "ATR",
                 breakout_threshold: float = 1.5,
                 lookback_period: int = 20):
        super().__init__("VOL_BREAKOUT", IndicatorType.VOLATILITY)
        self.volatility_indicator = volatility_indicator
        self.breakout_threshold = breakout_threshold
        self.lookback_period = lookback_period
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> IndicatorResult:
        # Calculate base volatility indicator
        if self.volatility_indicator == "ATR":
            vol_indicator = AverageTrueRange()
        elif self.volatility_indicator == "HIST_VOL":
            vol_indicator = HistoricalVolatility()
        else:
            raise ValueError(f"Unsupported volatility indicator: {self.volatility_indicator}")
        
        vol_values = vol_indicator.calculate(data).values
        
        # Calculate moving average and standard deviation of volatility
        vol_ma = vol_values.rolling(window=self.lookback_period).mean()
        vol_std = vol_values.rolling(window=self.lookback_period).std()
        
        # Define breakout thresholds
        upper_threshold = vol_ma + (self.breakout_threshold * vol_std)
        lower_threshold = vol_ma - (self.breakout_threshold * vol_std)
        
        # Detect breakouts
        volatility_breakout_up = vol_values > upper_threshold
        volatility_breakout_down = vol_values < lower_threshold
        
        # Create result DataFrame
        result = pd.DataFrame({
            'volatility': vol_values,
            'vol_ma': vol_ma,
            'vol_upper_threshold': upper_threshold,
            'vol_lower_threshold': lower_threshold,
            'breakout_up': volatility_breakout_up,
            'breakout_down': volatility_breakout_down,
            'breakout_signal': volatility_breakout_up.astype(int) - volatility_breakout_down.astype(int)
        }, index=data.index if isinstance(data, pd.DataFrame) else data.index)
        
        metadata = {
            'volatility_indicator': self.volatility_indicator,
            'breakout_threshold': self.breakout_threshold,
            'breakouts_up': volatility_breakout_up.sum(),
            'breakouts_down': volatility_breakout_down.sum()
        }
        
        return self.create_result(result, metadata)

class VolatilityRegimeDetector(BaseIndicator):
    """
    Volatility Regime Detector
    
    Classifies market conditions into low, medium, and high volatility regimes.
    Uses percentile-based classification for regime identification.
    """
    
    def __init__(self, 
                 volatility_indicator: str = "HIST_VOL",
                 regime_window: int = 252,  # 1 year for regime classification
                 low_percentile: float = 33.33,
                 high_percentile: float = 66.67):
        super().__init__("VOL_REGIME", IndicatorType.VOLATILITY)
        self.volatility_indicator = volatility_indicator
        self.regime_window = regime_window
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> IndicatorResult:
        # Calculate base volatility indicator
        if self.volatility_indicator == "ATR":
            vol_indicator = AverageTrueRange()
        elif self.volatility_indicator == "HIST_VOL":
            vol_indicator = HistoricalVolatility()
        else:
            raise ValueError(f"Unsupported volatility indicator: {self.volatility_indicator}")
        
        vol_values = vol_indicator.calculate(data).values
        
        # Calculate rolling percentiles
        def calculate_percentiles(window_data):
            if len(window_data) < 10:
                return [np.nan, np.nan]
            return [
                np.percentile(window_data, self.low_percentile),
                np.percentile(window_data, self.high_percentile)
            ]
        
        rolling_percentiles = vol_values.rolling(
            window=self.regime_window,
            min_periods=min(50, self.regime_window // 2)
        ).apply(lambda x: calculate_percentiles(x), raw=True, result_type='expand')
        
        if len(rolling_percentiles.columns) >= 2:
            low_threshold = rolling_percentiles.iloc[:, 0]
            high_threshold = rolling_percentiles.iloc[:, 1]
        else:
            low_threshold = pd.Series(index=vol_values.index, dtype=float)
            high_threshold = pd.Series(index=vol_values.index, dtype=float)
        
        # Classify regimes
        regime = pd.Series(index=vol_values.index, dtype=int)
        regime[vol_values <= low_threshold] = 1  # Low volatility
        regime[(vol_values > low_threshold) & (vol_values < high_threshold)] = 2  # Medium volatility
        regime[vol_values >= high_threshold] = 3  # High volatility
        
        # Create regime labels
        regime_labels = regime.map({1: 'Low', 2: 'Medium', 3: 'High'})
        
        result = pd.DataFrame({
            'volatility': vol_values,
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'regime': regime,
            'regime_label': regime_labels
        }, index=vol_values.index)
        
        # Calculate regime statistics
        regime_counts = regime.value_counts()
        metadata = {
            'volatility_indicator': self.volatility_indicator,
            'regime_window': self.regime_window,
            'low_regime_periods': regime_counts.get(1, 0),
            'medium_regime_periods': regime_counts.get(2, 0),
            'high_regime_periods': regime_counts.get(3, 0),
            'current_regime': regime_labels.iloc[-1] if not regime_labels.empty else 'Unknown'
        }
        
        return self.create_result(result, metadata)

# ============================================
# Volatility Clustering Analysis
# ============================================

class VolatilityClusteringAnalyzer(BaseIndicator):
    """
    Volatility Clustering Analyzer
    
    Analyzes volatility clustering patterns (periods of high volatility 
    tend to be followed by high volatility, and vice versa).
    
    Uses GARCH-like concepts to identify clustering effects.
    """
    
    def __init__(self, 
                 cluster_threshold: float = 1.5,
                 min_cluster_length: int = 3,
                 volatility_period: int = 20):
        super().__init__("VOL_CLUSTERING", IndicatorType.VOLATILITY)
        self.cluster_threshold = cluster_threshold
        self.min_cluster_length = min_cluster_length
        self.volatility_period = volatility_period
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> IndicatorResult:
        price_series = self.get_price_series(data) if hasattr(self, 'get_price_series') else data
        
        # Calculate returns and volatility
        returns = price_series.pct_change().dropna()
        rolling_vol = returns.rolling(window=self.volatility_period).std()
        
        # Calculate long-term average volatility
        long_term_vol = rolling_vol.rolling(window=252, min_periods=50).mean()  # 1-year rolling average
        
        # Identify high volatility periods
        high_vol_periods = rolling_vol > (long_term_vol * self.cluster_threshold)
        
        # Identify clusters (consecutive high volatility periods)
        clusters = pd.Series(index=high_vol_periods.index, dtype=int)
        cluster_id = 0
        in_cluster = False
        cluster_length = 0
        
        for i, is_high_vol in enumerate(high_vol_periods):
            if is_high_vol:
                if not in_cluster:
                    cluster_id += 1
                    in_cluster = True
                    cluster_length = 1
                else:
                    cluster_length += 1
                clusters.iloc[i] = cluster_id
            else:
                if in_cluster and cluster_length < self.min_cluster_length:
                    # Remove short clusters
                    clusters[clusters == cluster_id] = 0
                in_cluster = False
                cluster_length = 0
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id in clusters[clusters > 0].unique():
            cluster_data = rolling_vol[clusters == cluster_id]
            cluster_stats[cluster_id] = {
                'start_date': cluster_data.index[0],
                'end_date': cluster_data.index[-1],
                'length': len(cluster_data),
                'avg_volatility': cluster_data.mean(),
                'max_volatility': cluster_data.max()
            }
        
        result = pd.DataFrame({
            'volatility': rolling_vol,
            'long_term_vol': long_term_vol,
            'high_vol_threshold': long_term_vol * self.cluster_threshold,
            'is_high_vol': high_vol_periods,
            'cluster_id': clusters,
            'in_cluster': clusters > 0
        }, index=rolling_vol.index)
        
        metadata = {
            'cluster_threshold': self.cluster_threshold,
            'min_cluster_length': self.min_cluster_length,
            'total_clusters': len(cluster_stats),
            'cluster_statistics': cluster_stats,
            'current_in_cluster': bool(clusters.iloc[-1] > 0) if len(clusters) > 0 else False
        }
        
        return self.create_result(result, metadata)

# ============================================
# Register All Volatility Indicators
# ============================================

# Register core volatility indicators
indicator_registry.register(BollingerBands, "BOLLINGER_BANDS", IndicatorType.VOLATILITY)
indicator_registry.register(AverageTrueRange, "ATR", IndicatorType.VOLATILITY)
indicator_registry.register(HistoricalVolatility, "HIST_VOL", IndicatorType.VOLATILITY)
indicator_registry.register(ChaikinVolatility, "CHAIKIN_VOL", IndicatorType.VOLATILITY)
indicator_registry.register(RelativeVolatility, "RVI", IndicatorType.VOLATILITY)

# Register advanced volatility indicators
indicator_registry.register(KeltnerChannels, "KELTNER_CHANNELS", IndicatorType.VOLATILITY)
indicator_registry.register(DonchianChannels, "DONCHIAN_CHANNELS", IndicatorType.VOLATILITY)
indicator_registry.register(VolatilityRatio, "VOL_RATIO", IndicatorType.VOLATILITY)
indicator_registry.register(UlcerIndex, "ULCER_INDEX", IndicatorType.VOLATILITY)

# Register analysis tools
indicator_registry.register(VolatilityBreakout, "VOL_BREAKOUT", IndicatorType.VOLATILITY)
indicator_registry.register(VolatilityRegimeDetector, "VOL_REGIME", IndicatorType.VOLATILITY)
indicator_registry.register(VolatilityClusteringAnalyzer, "VOL_CLUSTERING", IndicatorType.VOLATILITY)

# ============================================
# Utility Functions
# ============================================

def create_volatility_bands_signals(data: Union[pd.DataFrame, pd.Series],
                                   band_type: str = "bollinger",
                                   period: int = 20) -> pd.DataFrame:
    """Create volatility band-based trading signals"""
    
    if band_type.lower() == "bollinger":
        bands_indicator = BollingerBands(config=IndicatorConfig(period=period))
    elif band_type.lower() == "keltner":
        bands_indicator = KeltnerChannels(config=IndicatorConfig(period=period))
    elif band_type.lower() == "donchian":
        bands_indicator = DonchianChannels(config=IndicatorConfig(period=period))
    else:
        raise ValueError(f"Unsupported band type: {band_type}")
    
    bands_result = bands_indicator.calculate(data)
    bands_data = bands_result.values
    
    price_series = data['close'] if isinstance(data, pd.DataFrame) else data
    
    # Create signals
    signals = pd.DataFrame(index=price_series.index)
    
    if band_type.lower() == "bollinger":
        signals['bb_upper'] = bands_data['bb_upper']
        signals['bb_middle'] = bands_data['bb_middle']
        signals['bb_lower'] = bands_data['bb_lower']
        signals['bb_bandwidth'] = bands_data['bb_bandwidth']
        signals['bb_percent_b'] = bands_data['bb_percent_b']
        
        # Bollinger Band signals
        signals['bb_squeeze'] = bands_data['bb_bandwidth'] < bands_data['bb_bandwidth'].rolling(20).quantile(0.1)
        signals['bb_expansion'] = bands_data['bb_bandwidth'] > bands_data['bb_bandwidth'].rolling(20).quantile(0.9)
        signals['price_above_upper'] = price_series > bands_data['bb_upper']
        signals['price_below_lower'] = price_series < bands_data['bb_lower']
        signals['mean_reversion_buy'] = (price_series < bands_data['bb_lower']) & (price_series.shift(1) >= bands_data['bb_lower'].shift(1))
        signals['mean_reversion_sell'] = (price_series > bands_data['bb_upper']) & (price_series.shift(1) <= bands_data['bb_upper'].shift(1))
        
    else:
        # Generic band signals for Keltner and Donchian
        upper_col = 'kc_upper' if band_type.lower() == "keltner" else 'dc_upper'
        middle_col = 'kc_middle' if band_type.lower() == "keltner" else 'dc_middle'
        lower_col = 'kc_lower' if band_type.lower() == "keltner" else 'dc_lower'
        
        signals[f'{band_type}_upper'] = bands_data[upper_col]
        signals[f'{band_type}_middle'] = bands_data[middle_col]
        signals[f'{band_type}_lower'] = bands_data[lower_col]
        
        signals['breakout_buy'] = (price_series > bands_data[upper_col]) & (price_series.shift(1) <= bands_data[upper_col].shift(1))
        signals['breakout_sell'] = (price_series < bands_data[lower_col]) & (price_series.shift(1) >= bands_data[lower_col].shift(1))
    
    return signals

@time_it("volatility_suite_calculation")
def calculate_volatility_suite(data: Union[pd.DataFrame, pd.Series],
                              include_advanced: bool = True) -> Dict[str, IndicatorResult]:
    """Calculate comprehensive volatility indicator suite"""
    
    results = {}
    
    # Core volatility indicators
    bollinger = BollingerBands()
    results['bollinger_bands'] = bollinger.calculate(data)
    
    hist_vol = HistoricalVolatility()
    results['historical_volatility'] = hist_vol.calculate(data)
    
    # OHLC-dependent indicators
    if isinstance(data, pd.DataFrame):
        atr = AverageTrueRange()
        results['atr'] = atr.calculate(data)
        
        chaikin_vol = ChaikinVolatility()
        results['chaikin_volatility'] = chaikin_vol.calculate(data)
        
        # Advanced indicators
        if include_advanced:
            keltner = KeltnerChannels()
            results['keltner_channels'] = keltner.calculate(data)
            
            donchian = DonchianChannels()
            results['donchian_channels'] = donchian.calculate(data)
            
            vol_ratio = VolatilityRatio()
            results['volatility_ratio'] = vol_ratio.calculate(data)
            
            ulcer = UlcerIndex()
            results['ulcer_index'] = ulcer.calculate(data)
            
            # Analysis tools
            vol_regime = VolatilityRegimeDetector()
            results['volatility_regime'] = vol_regime.calculate(data)
    
    logger.info(f"Calculated {len(results)} volatility indicators")
    return results

def analyze_volatility_patterns(data: Union[pd.DataFrame, pd.Series],
                               analysis_window: int = 252) -> Dict[str, Any]:
    """Analyze volatility patterns and characteristics"""
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Volatility pattern analysis requires OHLC data")
    
    # Calculate various volatility measures
    hist_vol = HistoricalVolatility(config=IndicatorConfig(period=20))
    atr = AverageTrueRange(config=IndicatorConfig(period=14))
    vol_ratio = VolatilityRatio(short_period=10, long_period=30)
    
    hist_vol_values = hist_vol.calculate(data).values
    atr_values = atr.calculate(data).values
    vol_ratio_values = vol_ratio.calculate(data).values
    
    # Calculate statistics
    analysis = {
        'historical_volatility': {
            'current': hist_vol_values.iloc[-1] if len(hist_vol_values) > 0 else np.nan,
            'mean': hist_vol_values.tail(analysis_window).mean(),
            'std': hist_vol_values.tail(analysis_window).std(),
            'percentile_25': hist_vol_values.tail(analysis_window).quantile(0.25),
            'percentile_75': hist_vol_values.tail(analysis_window).quantile(0.75),
            'max': hist_vol_values.tail(analysis_window).max(),
            'min': hist_vol_values.tail(analysis_window).min()
        },
        'atr': {
            'current': atr_values.iloc[-1] if len(atr_values) > 0 else np.nan,
            'mean': atr_values.tail(analysis_window).mean(),
            'trend': 'increasing' if atr_values.iloc[-1] > atr_values.tail(20).mean() else 'decreasing'
        },
        'volatility_ratio': {
            'current': vol_ratio_values.iloc[-1] if len(vol_ratio_values) > 0 else np.nan,
            'above_1_count': (vol_ratio_values.tail(analysis_window) > 1).sum(),
            'below_1_count': (vol_ratio_values.tail(analysis_window) < 1).sum()
        }
    }
    
    # Volatility clustering analysis
    clustering_analyzer = VolatilityClusteringAnalyzer()
    clustering_result = clustering_analyzer.calculate(data)
    
    analysis['clustering'] = {
        'total_clusters': clustering_result.metadata['total_clusters'],
        'currently_in_cluster': clustering_result.metadata['current_in_cluster'],
        'cluster_statistics': clustering_result.metadata['cluster_statistics']
    }
    
    # Regime analysis
    regime_detector = VolatilityRegimeDetector()
    regime_result = regime_detector.calculate(data)
    
    analysis['regime'] = {
        'current_regime': regime_result.metadata['current_regime'],
        'low_regime_periods': regime_result.metadata['low_regime_periods'],
        'medium_regime_periods': regime_result.metadata['medium_regime_periods'],
        'high_regime_periods': regime_result.metadata['high_regime_periods']
    }
    
    return analysis

def create_volatility_dashboard(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Create comprehensive volatility analysis dashboard"""
    
    dashboard = pd.DataFrame(index=data.index if isinstance(data, pd.DataFrame) else data.index)
    
    # Calculate volatility suite
    vol_suite = calculate_volatility_suite(data, include_advanced=False)
    
    # Add core indicators
    dashboard['hist_vol'] = vol_suite['historical_volatility'].values
    
    bb_data = vol_suite['bollinger_bands'].values
    dashboard['bb_bandwidth'] = bb_data['bb_bandwidth']
    dashboard['bb_percent_b'] = bb_data['bb_percent_b']
    
    if isinstance(data, pd.DataFrame):
        dashboard['atr'] = vol_suite['atr'].values
        dashboard['chaikin_vol'] = vol_suite['chaikin_volatility'].values
    
    # Add volatility signals
    if isinstance(data, pd.DataFrame):
        bb_signals = create_volatility_bands_signals(data, "bollinger")
        dashboard['bb_squeeze'] = bb_signals['bb_squeeze']
        dashboard['bb_expansion'] = bb_signals['bb_expansion']
        dashboard['mean_reversion_buy'] = bb_signals['mean_reversion_buy']
        dashboard['mean_reversion_sell'] = bb_signals['mean_reversion_sell']
    
    # Add volatility regime
    if isinstance(data, pd.DataFrame):
        regime_detector = VolatilityRegimeDetector()
        regime_result = regime_detector.calculate(data)
        dashboard['vol_regime'] = regime_result.values['regime']
        dashboard['vol_regime_label'] = regime_result.values['regime_label']
    
    # Calculate composite volatility score
    vol_cols = ['hist_vol']
    if isinstance(data, pd.DataFrame):
        vol_cols.extend(['atr', 'chaikin_vol'])
    
    # Normalize volatility measures to 0-100 scale
    for col in vol_cols:
        if col in dashboard.columns:
            normalized_col = f'{col}_normalized'
            rolling_min = dashboard[col].rolling(window=252, min_periods=50).min()
            rolling_max = dashboard[col].rolling(window=252, min_periods=50).max()
            dashboard[normalized_col] = 100 * (dashboard[col] - rolling_min) / (rolling_max - rolling_min)
    
    # Composite volatility score
    normalized_cols = [col for col in dashboard.columns if 'normalized' in col]
    if normalized_cols:
        dashboard['volatility_score'] = dashboard[normalized_cols].mean(axis=1)
    
    return dashboard

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    from .base import create_sample_data
    
    print("Testing Volatility Indicators")
    
    # Create sample data
    sample_data = create_sample_data(300, start_price=100.0, volatility=0.025)
    print(f"Sample data shape: {sample_data.shape}")
    
    # Test individual indicators
    bollinger = BollingerBands()
    bb_result = bollinger.calculate(sample_data)
    print(f"Bollinger Bands last 5 values:\n{bb_result.values.tail()}")
    
    atr = AverageTrueRange()
    atr_result = atr.calculate(sample_data)
    print(f"ATR last 5 values: {atr_result.values.tail()}")
    
    hist_vol = HistoricalVolatility()
    hv_result = hist_vol.calculate(sample_data)
    print(f"Historical Volatility last 5 values: {hv_result.values.tail()}")
    
    # Test volatility suite
    vol_suite = calculate_volatility_suite(sample_data)
    print(f"Volatility suite calculated {len(vol_suite)} indicators")
    
    # Test pattern analysis
    patterns = analyze_volatility_patterns(sample_data)
    print(f"Current volatility regime: {patterns['regime']['current_regime']}")
    print(f"Total volatility clusters: {patterns['clustering']['total_clusters']}")
    
    # Test dashboard
    dashboard = create_volatility_dashboard(sample_data)
    print(f"Dashboard created with {len(dashboard.columns)} columns")
    
    if 'volatility_score' in dashboard.columns:
        current_vol_score = dashboard['volatility_score'].iloc[-1]
        print(f"Current volatility score: {current_vol_score:.2f}")
    
    # Test signals
    bb_signals = create_volatility_bands_signals(sample_data, "bollinger")
    squeeze_periods = bb_signals['bb_squeeze'].sum()
    expansion_periods = bb_signals['bb_expansion'].sum()
    print(f"Bollinger Band squeeze periods: {squeeze_periods}")
    print(f"Bollinger Band expansion periods: {expansion_periods}")
    
    print("Volatility indicators testing completed successfully!")
