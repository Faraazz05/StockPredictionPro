# ============================================
# StockPredictionPro - src/features/indicators/volume.py
# Comprehensive volume indicators for technical analysis with advanced financial domain support
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

logger = get_logger('features.indicators.volume')

# ============================================
# Core Volume Indicators
# ============================================

class OnBalanceVolume(SingleValueIndicator):
    """
    On-Balance Volume (OBV)
    
    OBV is a momentum indicator that uses volume to predict price movements.
    It adds volume on up days and subtracts volume on down days.
    
    Formula:
    - If Close > Previous Close: OBV = Previous OBV + Volume
    - If Close < Previous Close: OBV = Previous OBV - Volume
    - If Close = Previous Close: OBV = Previous OBV
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("OBV", IndicatorType.VOLUME, config or IndicatorConfig())
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("OBV requires OHLCV data with volume column")
        
        close = data['close']
        volume = data['volume']
        
        if len(data) < 2:
            return pd.Series([0] * len(data), index=data.index)
        
        # Calculate price direction
        price_change = close.diff()
        direction = np.sign(price_change)
        direction = direction.fillna(0)  # First value is neutral
        
        # Calculate OBV
        volume_flow = direction * volume
        obv = volume_flow.cumsum()
        
        return obv

class VolumeRateOfChange(SingleValueIndicator):
    """
    Volume Rate of Change (VROC)
    
    Measures the percentage change in volume from one period to another.
    Helps identify unusual volume spikes or drops.
    
    Formula: VROC = ((Volume - Volume n periods ago) / Volume n periods ago) × 100
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("VROC", IndicatorType.VOLUME, config or IndicatorConfig(period=12))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("VROC requires OHLCV data with volume column")
        
        volume = data['volume']
        
        if len(volume) < self.config.period + 1:
            return self._handle_insufficient_data(len(volume), self.config.period + 1)
        
        # Calculate volume rate of change
        vroc = volume.pct_change(periods=self.config.period) * 100
        
        return vroc

class AccumulationDistributionLine(SingleValueIndicator):
    """
    Accumulation/Distribution Line (ADL)
    
    Measures the cumulative flow of money into and out of a security.
    Combines price and volume to show how much of the volume is related to buying vs selling.
    
    Formula:
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    Money Flow Volume = Money Flow Multiplier × Volume
    ADL = Previous ADL + Money Flow Volume
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("ADL", IndicatorType.VOLUME, config or IndicatorConfig())
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("ADL requires OHLCV data with volume column")
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate Money Flow Multiplier
        money_flow_multiplier = MathUtils.money_flow_multiplier(high, low, close)
        
        # Handle division by zero (when high = low)
        money_flow_multiplier = money_flow_multiplier.fillna(0)
        money_flow_multiplier = money_flow_multiplier.replace([np.inf, -np.inf], 0)
        
        # Calculate Money Flow Volume
        money_flow_volume = money_flow_multiplier * volume
        
        # Calculate ADL (cumulative sum)
        adl = money_flow_volume.cumsum()
        
        return adl

class ChaikinMoneyFlow(SingleValueIndicator):
    """
    Chaikin Money Flow (CMF)
    
    Measures the amount of money flow volume over a specific period.
    Values above zero indicate buying pressure, below zero selling pressure.
    
    Formula: CMF = Sum(Money Flow Volume, n) / Sum(Volume, n)
    Where Money Flow Volume = Money Flow Multiplier × Volume
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("CMF", IndicatorType.VOLUME, config or IndicatorConfig(period=20))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("CMF requires OHLCV data with volume column")
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        if len(data) < self.config.period:
            return self._handle_insufficient_data(len(data), self.config.period)
        
        # Calculate Money Flow Multiplier
        money_flow_multiplier = MathUtils.money_flow_multiplier(high, low, close)
        money_flow_multiplier = money_flow_multiplier.fillna(0)
        money_flow_multiplier = money_flow_multiplier.replace([np.inf, -np.inf], 0)
        
        # Calculate Money Flow Volume
        money_flow_volume = money_flow_multiplier * volume
        
        # Calculate rolling sums
        mfv_sum = money_flow_volume.rolling(
            window=self.config.period,
            min_periods=self.config.min_periods
        ).sum()
        
        volume_sum = volume.rolling(
            window=self.config.period,
            min_periods=self.config.min_periods
        ).sum()
        
        # Calculate CMF
        cmf = mfv_sum / volume_sum
        cmf = cmf.fillna(0)  # Handle division by zero
        
        return cmf

class VolumeWeightedAveragePrice(SingleValueIndicator):
    """
    Volume Weighted Average Price (VWAP)
    
    Calculates the average price weighted by volume over a period.
    Often used as a benchmark for execution quality.
    
    Formula: VWAP = Sum(Price × Volume) / Sum(Volume)
    Where Price is typically (High + Low + Close) / 3
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("VWAP", IndicatorType.VOLUME, config or IndicatorConfig(period=20))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("VWAP requires OHLCV data with volume column")
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        if len(data) < self.config.period:
            return self._handle_insufficient_data(len(data), self.config.period)
        
        # Calculate typical price
        typical_price = MathUtils.typical_price(high, low, close)
        
        # Calculate price-volume product
        pv = typical_price * volume
        
        # Calculate rolling VWAP
        pv_sum = pv.rolling(
            window=self.config.period,
            min_periods=self.config.min_periods
        ).sum()
        
        volume_sum = volume.rolling(
            window=self.config.period,
            min_periods=self.config.min_periods
        ).sum()
        
        vwap = pv_sum / volume_sum
        vwap = vwap.fillna(method='ffill')  # Forward fill to handle division by zero
        
        return vwap

class NegativeVolumeIndex(SingleValueIndicator):
    """
    Negative Volume Index (NVI)
    
    Tracks price changes only when volume decreases from the previous period.
    Theory: smart money trades on light volume days.
    
    Formula:
    - If Volume < Previous Volume: NVI = Previous NVI × (1 + % Price Change)
    - If Volume >= Previous Volume: NVI = Previous NVI
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("NVI", IndicatorType.VOLUME, config or IndicatorConfig())
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("NVI requires OHLCV data with volume column")
        
        close = data['close']
        volume = data['volume']
        
        if len(data) < 2:
            return pd.Series([1000] * len(data), index=data.index)
        
        # Initialize NVI
        nvi = pd.Series(index=data.index, dtype=float)
        nvi.iloc[0] = 1000  # Starting value
        
        # Calculate NVI
        for i in range(1, len(data)):
            if volume.iloc[i] < volume.iloc[i-1]:
                # Volume decreased - update NVI
                price_change = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + price_change)
            else:
                # Volume increased or stayed same - keep previous NVI
                nvi.iloc[i] = nvi.iloc[i-1]
        
        return nvi

class PositiveVolumeIndex(SingleValueIndicator):
    """
    Positive Volume Index (PVI)
    
    Tracks price changes only when volume increases from the previous period.
    Theory: uninformed traders are more active on high volume days.
    
    Formula:
    - If Volume > Previous Volume: PVI = Previous PVI × (1 + % Price Change)
    - If Volume <= Previous Volume: PVI = Previous PVI
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("PVI", IndicatorType.VOLUME, config or IndicatorConfig())
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("PVI requires OHLCV data with volume column")
        
        close = data['close']
        volume = data['volume']
        
        if len(data) < 2:
            return pd.Series([1000] * len(data), index=data.index)
        
        # Initialize PVI
        pvi = pd.Series(index=data.index, dtype=float)
        pvi.iloc[0] = 1000  # Starting value
        
        # Calculate PVI
        for i in range(1, len(data)):
            if volume.iloc[i] > volume.iloc[i-1]:
                # Volume increased - update PVI
                price_change = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + price_change)
            else:
                # Volume decreased or stayed same - keep previous PVI
                pvi.iloc[i] = pvi.iloc[i-1]
        
        return pvi

# ============================================
# Advanced Volume Indicators
# ============================================

class VolumeOscillator(SingleValueIndicator):
    """
    Volume Oscillator
    
    Measures the difference between two volume moving averages.
    Positive values indicate short-term volume is higher than long-term average.
    
    Formula: VO = ((Short MA - Long MA) / Long MA) × 100
    """
    
    def __init__(self, 
                 short_period: int = 14,
                 long_period: int = 28,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("VOLUME_OSC", IndicatorType.VOLUME, config or IndicatorConfig())
        self.short_period = short_period
        self.long_period = long_period
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("Volume Oscillator requires OHLCV data with volume column")
        
        volume = data['volume']
        
        if len(volume) < self.long_period:
            return self._handle_insufficient_data(len(volume), self.long_period)
        
        # Calculate moving averages
        short_ma = MathUtils.simple_moving_average(volume, self.short_period)
        long_ma = MathUtils.simple_moving_average(volume, self.long_period)
        
        # Calculate oscillator
        volume_oscillator = ((short_ma - long_ma) / long_ma) * 100
        volume_oscillator = volume_oscillator.fillna(0)
        
        return volume_oscillator

class PriceVolumeRank(SingleValueIndicator):
    """
    Price Volume Rank (PVR)
    
    Ranks volume relative to recent volume history.
    Values from 0-100, where 100 means current volume is highest in the period.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("PVR", IndicatorType.VOLUME, config or IndicatorConfig(period=50))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("PVR requires OHLCV data with volume column")
        
        volume = data['volume']
        
        if len(volume) < self.config.period:
            return self._handle_insufficient_data(len(volume), self.config.period)
        
        def calculate_rank(window_data):
            if len(window_data) < 2:
                return 50.0
            
            current_volume = window_data.iloc[-1]
            rank = (window_data < current_volume).sum() / len(window_data) * 100
            return rank
        
        pvr = volume.rolling(
            window=self.config.period,
            min_periods=self.config.min_periods
        ).apply(calculate_rank, raw=False)
        
        return pvr

class VolumeZoneOscillator(SingleValueIndicator):
    """
    Volume Zone Oscillator (VZO)
    
    Combines price direction with volume to create a volume-based momentum oscillator.
    Positive values indicate volume is flowing into rising prices.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("VZO", IndicatorType.VOLUME, config or IndicatorConfig(period=14))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("VZO requires OHLCV data with volume column")
        
        close = data['close']
        volume = data['volume']
        
        if len(data) < self.config.period + 1:
            return self._handle_insufficient_data(len(data), self.config.period + 1)
        
        # Calculate price direction
        price_change = close.diff()
        direction = np.sign(price_change)
        
        # Calculate signed volume
        signed_volume = direction * volume
        
        # Calculate VZO using exponential moving average
        vzo = signed_volume.ewm(span=self.config.period).mean()
        
        # Normalize to percentage
        volume_ema = volume.ewm(span=self.config.period).mean()
        vzo_normalized = (vzo / volume_ema) * 100
        vzo_normalized = vzo_normalized.fillna(0)
        
        return vzo_normalized

class KleingerVolumeOscillator(MultiValueIndicator):
    """
    Klinger Volume Oscillator (KVO)
    
    Developed by Stephen Klinger, this oscillator uses volume and price to predict price direction.
    Consists of the KVO line and a signal line.
    """
    
    def __init__(self, 
                 fast_period: int = 34,
                 slow_period: int = 55,
                 signal_period: int = 13,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("KVO", IndicatorType.VOLUME, config or IndicatorConfig())
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("KVO requires OHLCV data with volume column")
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        if len(data) < max(self.slow_period, self.signal_period) + 1:
            return self._handle_insufficient_data(len(data), max(self.slow_period, self.signal_period) + 1)
        
        # Calculate typical price
        typical_price = MathUtils.typical_price(high, low, close)
        
        # Calculate trend (simplified version)
        trend = pd.Series(index=data.index, dtype=int)
        trend.iloc[0] = 1
        
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                trend.iloc[i] = 1
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
        
        # Calculate volume force
        volume_force = volume * trend * abs(2 * ((close - low) - (high - close)) / (high - low))
        volume_force = volume_force.fillna(0)
        volume_force = volume_force.replace([np.inf, -np.inf], 0)
        
        # Calculate KVO
        kvo_fast = volume_force.ewm(span=self.fast_period).mean()
        kvo_slow = volume_force.ewm(span=self.slow_period).mean()
        kvo = kvo_fast - kvo_slow
        
        # Calculate signal line
        kvo_signal = kvo.ewm(span=self.signal_period).mean()
        
        # Calculate histogram
        kvo_histogram = kvo - kvo_signal
        
        result = pd.DataFrame({
            'kvo': kvo,
            'kvo_signal': kvo_signal,
            'kvo_histogram': kvo_histogram
        }, index=data.index)
        
        return result

# ============================================
# Volume Analysis Tools
# ============================================

class VolumeProfileAnalyzer(BaseIndicator):
    """
    Volume Profile Analyzer
    
    Analyzes volume distribution across price levels to identify
    support/resistance levels and areas of high/low interest.
    """
    
    def __init__(self, 
                 price_bins: int = 20,
                 volume_threshold: float = 0.7):
        super().__init__("VOLUME_PROFILE", IndicatorType.VOLUME)
        self.price_bins = price_bins
        self.volume_threshold = volume_threshold
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> IndicatorResult:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("Volume Profile requires OHLCV data with volume column")
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate price range and bins
        price_min = low.min()
        price_max = high.max()
        price_range = price_max - price_min
        bin_size = price_range / self.price_bins
        
        # Create price levels
        price_levels = np.linspace(price_min, price_max, self.price_bins + 1)
        
        # Calculate volume at each price level
        volume_profile = pd.Series(index=price_levels[:-1], dtype=float)
        
        for i in range(len(price_levels) - 1):
            level_low = price_levels[i]
            level_high = price_levels[i + 1]
            
            # Find periods where price was in this range
            in_range = ((low <= level_high) & (high >= level_low))
            level_volume = volume[in_range].sum()
            
            volume_profile.iloc[i] = level_volume
        
        # Calculate Point of Control (POC) - price level with highest volume
        poc_price = volume_profile.idxmax()
        poc_volume = volume_profile.max()
        
        # Calculate Value Area (area containing specified percentage of volume)
        total_volume = volume_profile.sum()
        target_volume = total_volume * self.volume_threshold
        
        # Find value area boundaries
        sorted_profile = volume_profile.sort_values(ascending=False)
        cumulative_volume = 0
        value_area_prices = []
        
        for price, vol in sorted_profile.items():
            cumulative_volume += vol
            value_area_prices.append(price)
            if cumulative_volume >= target_volume:
                break
        
        value_area_high = max(value_area_prices)
        value_area_low = min(value_area_prices)
        
        # Create result with volume profile data
        result_data = pd.DataFrame({
            'price_level': price_levels[:-1],
            'volume_at_level': volume_profile.values,
            'volume_percentage': (volume_profile.values / total_volume) * 100
        })
        
        metadata = {
            'poc_price': poc_price,
            'poc_volume': poc_volume,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'value_area_volume_pct': self.volume_threshold * 100,
            'total_volume': total_volume,
            'price_bins': self.price_bins
        }
        
        return self.create_result(result_data, metadata)

class VolumeBreakoutDetector(BaseIndicator):
    """
    Volume Breakout Detector
    
    Identifies significant volume breakouts that often precede price movements.
    Uses statistical analysis to detect unusual volume spikes.
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 breakout_threshold: float = 2.0):
        super().__init__("VOLUME_BREAKOUT", IndicatorType.VOLUME)
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
    
    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> IndicatorResult:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("Volume Breakout Detector requires OHLCV data with volume column")
        
        volume = data['volume']
        close = data['close']
        
        # Calculate rolling volume statistics
        volume_mean = volume.rolling(window=self.lookback_period).mean()
        volume_std = volume.rolling(window=self.lookback_period).std()
        
        # Calculate breakout threshold
        breakout_level = volume_mean + (self.breakout_threshold * volume_std)
        
        # Identify breakouts
        volume_breakout = volume > breakout_level
        
        # Calculate volume ratio
        volume_ratio = volume / volume_mean
        
        # Identify price direction during breakout
        price_change = close.pct_change()
        breakout_with_price_up = volume_breakout & (price_change > 0)
        breakout_with_price_down = volume_breakout & (price_change < 0)
        
        result = pd.DataFrame({
            'volume': volume,
            'volume_mean': volume_mean,
            'volume_std': volume_std,
            'breakout_threshold': breakout_level,
            'volume_ratio': volume_ratio,
            'volume_breakout': volume_breakout,
            'breakout_bullish': breakout_with_price_up,
            'breakout_bearish': breakout_with_price_down,
            'price_change': price_change
        }, index=data.index)
        
        metadata = {
            'lookback_period': self.lookback_period,
            'breakout_threshold': self.breakout_threshold,
            'total_breakouts': volume_breakout.sum(),
            'bullish_breakouts': breakout_with_price_up.sum(),
            'bearish_breakouts': breakout_with_price_down.sum()
        }
        
        return self.create_result(result, metadata)

# ============================================
# Register All Volume Indicators
# ============================================

# Register core volume indicators
indicator_registry.register(OnBalanceVolume, "OBV", IndicatorType.VOLUME)
indicator_registry.register(VolumeRateOfChange, "VROC", IndicatorType.VOLUME)
indicator_registry.register(AccumulationDistributionLine, "ADL", IndicatorType.VOLUME)
indicator_registry.register(ChaikinMoneyFlow, "CMF", IndicatorType.VOLUME)
indicator_registry.register(VolumeWeightedAveragePrice, "VWAP", IndicatorType.VOLUME)
indicator_registry.register(NegativeVolumeIndex, "NVI", IndicatorType.VOLUME)
indicator_registry.register(PositiveVolumeIndex, "PVI", IndicatorType.VOLUME)

# Register advanced volume indicators
indicator_registry.register(VolumeOscillator, "VOLUME_OSC", IndicatorType.VOLUME)
indicator_registry.register(PriceVolumeRank, "PVR", IndicatorType.VOLUME)
indicator_registry.register(VolumeZoneOscillator, "VZO", IndicatorType.VOLUME)
indicator_registry.register(KleingerVolumeOscillator, "KVO", IndicatorType.VOLUME)

# Register analysis tools
indicator_registry.register(VolumeProfileAnalyzer, "VOLUME_PROFILE", IndicatorType.VOLUME)
indicator_registry.register(VolumeBreakoutDetector, "VOLUME_BREAKOUT", IndicatorType.VOLUME)

# ============================================
# Utility Functions
# ============================================

def create_volume_signals(data: Union[pd.DataFrame, pd.Series],
                         obv_period: int = 10,
                         cmf_period: int = 20) -> pd.DataFrame:
    """Create volume-based trading signals"""
    
    # Calculate volume indicators
    obv = OnBalanceVolume()
    cmf = ChaikinMoneyFlow(IndicatorConfig(period=cmf_period))
    vroc = VolumeRateOfChange(IndicatorConfig(period=12))
    
    obv_values = obv.calculate(data).values
    cmf_values = cmf.calculate(data).values
    vroc_values = vroc.calculate(data).values
    
    # Create signals DataFrame
    signals = pd.DataFrame(index=data.index)
    
    # OBV signals
    signals['obv'] = obv_values
    signals['obv_ma'] = MathUtils.simple_moving_average(obv_values, obv_period)
    signals['obv_bullish'] = obv_values > signals['obv_ma']
    signals['obv_bearish'] = obv_values < signals['obv_ma']
    
    # CMF signals
    signals['cmf'] = cmf_values
    signals['cmf_bullish'] = cmf_values > 0.1
    signals['cmf_bearish'] = cmf_values < -0.1
    
    # VROC signals
    signals['vroc'] = vroc_values
    signals['volume_spike'] = vroc_values > 50  # 50% volume increase
    signals['volume_dry_up'] = vroc_values < -30  # 30% volume decrease
    
    # Combined signals
    signals['volume_bullish'] = (
        signals['obv_bullish'] & 
        signals['cmf_bullish'] & 
        signals['volume_spike']
    )
    
    signals['volume_bearish'] = (
        signals['obv_bearish'] & 
        signals['cmf_bearish']
    )
    
    return signals

@time_it("volume_suite_calculation")
def calculate_volume_suite(data: Union[pd.DataFrame, pd.Series],
                          include_advanced: bool = True) -> Dict[str, IndicatorResult]:
    """Calculate comprehensive volume indicator suite"""
    
    if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
        raise ValidationError("Volume indicators require OHLCV data with volume column")
    
    results = {}
    
    # Core volume indicators
    obv = OnBalanceVolume()
    results['obv'] = obv.calculate(data)
    
    vroc = VolumeRateOfChange()
    results['vroc'] = vroc.calculate(data)
    
    adl = AccumulationDistributionLine()
    results['adl'] = adl.calculate(data)
    
    cmf = ChaikinMoneyFlow()
    results['cmf'] = cmf.calculate(data)
    
    vwap = VolumeWeightedAveragePrice()
    results['vwap'] = vwap.calculate(data)
    
    # Advanced indicators
    if include_advanced:
        nvi = NegativeVolumeIndex()
        results['nvi'] = nvi.calculate(data)
        
        pvi = PositiveVolumeIndex()
        results['pvi'] = pvi.calculate(data)
        
        vol_osc = VolumeOscillator()
        results['volume_oscillator'] = vol_osc.calculate(data)
        
        pvr = PriceVolumeRank()
        results['price_volume_rank'] = pvr.calculate(data)
        
        vzo = VolumeZoneOscillator()
        results['volume_zone_oscillator'] = vzo.calculate(data)
        
        kvo = KleingerVolumeOscillator()
        results['klinger_volume_oscillator'] = kvo.calculate(data)
    
    logger.info(f"Calculated {len(results)} volume indicators")
    return results

def analyze_volume_patterns(data: Union[pd.DataFrame, pd.Series],
                           analysis_window: int = 50) -> Dict[str, Any]:
    """Analyze volume patterns and characteristics"""
    
    if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
        raise ValidationError("Volume pattern analysis requires OHLCV data with volume column")
    
    volume = data['volume']
    close = data['close']
    
    # Calculate volume statistics
    recent_volume = volume.tail(analysis_window)
    
    analysis = {
        'volume_statistics': {
            'current': volume.iloc[-1],
            'mean': recent_volume.mean(),
            'median': recent_volume.median(),
            'std': recent_volume.std(),
            'min': recent_volume.min(),
            'max': recent_volume.max(),
            'coefficient_of_variation': recent_volume.std() / recent_volume.mean() if recent_volume.mean() > 0 else np.nan
        },
        'volume_trends': {
            'volume_trend': 'increasing' if volume.iloc[-5:].mean() > volume.iloc[-10:-5].mean() else 'decreasing',
            'volume_ma_20': volume.rolling(20).mean().iloc[-1],
            'volume_above_average': volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]
        }
    }
    
    # Price-volume relationship
    price_change = close.pct_change()
    volume_change = volume.pct_change()
    
    # Calculate correlation
    correlation = price_change.tail(analysis_window).corr(volume_change.tail(analysis_window))
    
    analysis['price_volume_relationship'] = {
        'correlation': correlation,
        'relationship_strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
    }
    
    # Volume breakout analysis
    breakout_detector = VolumeBreakoutDetector()
    breakout_result = breakout_detector.calculate(data)
    
    analysis['volume_breakouts'] = {
        'total_breakouts': breakout_result.metadata['total_breakouts'],
        'bullish_breakouts': breakout_result.metadata['bullish_breakouts'],
        'bearish_breakouts': breakout_result.metadata['bearish_breakouts'],
        'recent_breakout': breakout_result.values['volume_breakout'].iloc[-1]
    }
    
    return analysis

def create_volume_dashboard(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Create comprehensive volume analysis dashboard"""
    
    if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
        raise ValidationError("Volume dashboard requires OHLCV data with volume column")
    
    dashboard = pd.DataFrame(index=data.index)
    
    # Calculate volume suite
    volume_suite = calculate_volume_suite(data, include_advanced=False)
    
    # Add core indicators
    dashboard['volume'] = data['volume']
    dashboard['obv'] = volume_suite['obv'].values
    dashboard['vroc'] = volume_suite['vroc'].values
    dashboard['adl'] = volume_suite['adl'].values
    dashboard['cmf'] = volume_suite['cmf'].values
    dashboard['vwap'] = volume_suite['vwap'].values
    
    # Add volume moving averages
    dashboard['volume_ma_10'] = data['volume'].rolling(10).mean()
    dashboard['volume_ma_20'] = data['volume'].rolling(20).mean()
    dashboard['volume_ma_50'] = data['volume'].rolling(50).mean()
    
    # Volume relative to average
    dashboard['volume_vs_ma_20'] = data['volume'] / dashboard['volume_ma_20']
    
    # Volume signals
    volume_signals = create_volume_signals(data)
    dashboard['volume_bullish'] = volume_signals['volume_bullish']
    dashboard['volume_bearish'] = volume_signals['volume_bearish']
    dashboard['volume_spike'] = volume_signals['volume_spike']
    
    # Calculate volume score (0-100)
    # Normalize volume relative to recent average
    volume_ratio = dashboard['volume_vs_ma_20'].rolling(50).rank(pct=True) * 100
    dashboard['volume_score'] = volume_ratio
    
    return dashboard

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    from .base import create_sample_data
    
    print("Testing Volume Indicators")
    
    # Create sample data with volume
    sample_data = create_sample_data(200, start_price=100.0, volatility=0.02)
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Columns: {sample_data.columns.tolist()}")
    
    # Test individual indicators
    obv = OnBalanceVolume()
    obv_result = obv.calculate(sample_data)
    print(f"OBV last 5 values: {obv_result.values.tail()}")
    
    cmf = ChaikinMoneyFlow()
    cmf_result = cmf.calculate(sample_data)
    print(f"CMF last 5 values: {cmf_result.values.tail()}")
    
    vwap = VolumeWeightedAveragePrice()
    vwap_result = vwap.calculate(sample_data)
    print(f"VWAP last 5 values: {vwap_result.values.tail()}")
    
    # Test volume suite
    volume_suite = calculate_volume_suite(sample_data)
    print(f"Volume suite calculated {len(volume_suite)} indicators")
    
    # Test pattern analysis
    patterns = analyze_volume_patterns(sample_data)
    print(f"Current volume: {patterns['volume_statistics']['current']:.0f}")
    print(f"Volume trend: {patterns['volume_trends']['volume_trend']}")
    print(f"Price-volume correlation: {patterns['price_volume_relationship']['correlation']:.3f}")
    
    # Test signals
    volume_signals = create_volume_signals(sample_data)
    bullish_signals = volume_signals['volume_bullish'].sum()
    bearish_signals = volume_signals['volume_bearish'].sum()
    volume_spikes = volume_signals['volume_spike'].sum()
    
    print(f"Volume signals - Bullish: {bullish_signals}, Bearish: {bearish_signals}, Spikes: {volume_spikes}")
    
    # Test dashboard
    dashboard = create_volume_dashboard(sample_data)
    print(f"Dashboard created with {len(dashboard.columns)} columns")
    
    if 'volume_score' in dashboard.columns:
        current_volume_score = dashboard['volume_score'].iloc[-1]
        print(f"Current volume score: {current_volume_score:.1f}/100")
    
    print("Volume indicators testing completed successfully!")
