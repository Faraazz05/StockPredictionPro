# ============================================
# StockPredictionPro - src/features/indicators/custom.py
# Custom financial indicators with advanced domain-specific implementations
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from .base import (
    BaseIndicator, SingleValueIndicator, MultiValueIndicator,
    IndicatorConfig, IndicatorType, IndicatorResult,
    MathUtils, SmoothingMethods, indicator_registry,
    PriceField, TimeFrame
)

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('features.indicators.custom')

# ============================================
# Market Microstructure Indicators
# ============================================

class BidAskSpread(SingleValueIndicator):
    """
    Bid-Ask Spread Indicator
    
    Measures market liquidity and transaction costs.
    Can be calculated as absolute spread or relative spread.
    
    Formula: 
    - Absolute Spread = Ask - Bid
    - Relative Spread = (Ask - Bid) / Midpoint × 100
    """
    
    def __init__(self, 
                 relative: bool = True,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("BID_ASK_SPREAD", IndicatorType.CUSTOM, config or IndicatorConfig())
        self.relative = relative
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Bid-Ask Spread requires DataFrame with bid/ask columns")
        
        if 'bid' not in data.columns or 'ask' not in data.columns:
            # Fallback: estimate from high/low
            if 'high' in data.columns and 'low' in data.columns:
                logger.warning("Using high/low as proxy for bid/ask")
                bid = data['low']
                ask = data['high']
            else:
                raise ValidationError("No bid/ask or high/low data available")
        else:
            bid = data['bid']
            ask = data['ask']
        
        # Calculate spread
        absolute_spread = ask - bid
        
        if self.relative:
            midpoint = (bid + ask) / 2
            spread = (absolute_spread / midpoint) * 100
            spread = spread.replace([np.inf, -np.inf], np.nan)
        else:
            spread = absolute_spread
        
        return spread

class MarketImpactEstimator(SingleValueIndicator):
    """
    Market Impact Estimator
    
    Estimates the market impact of trades based on volume and volatility.
    Higher values indicate higher expected market impact.
    
    Formula: Impact = √(Volume / Average Volume) × Volatility
    """
    
    def __init__(self, 
                 volume_window: int = 20,
                 volatility_window: int = 20,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("MARKET_IMPACT", IndicatorType.CUSTOM, config or IndicatorConfig())
        self.volume_window = volume_window
        self.volatility_window = volatility_window
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("Market Impact requires OHLCV data with volume")
        
        close = data['close']
        volume = data['volume']
        
        # Calculate average volume
        avg_volume = volume.rolling(window=self.volume_window).mean()
        
        # Calculate volatility (returns standard deviation)
        returns = close.pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()
        
        # Calculate relative volume
        relative_volume = volume / avg_volume
        relative_volume = relative_volume.fillna(1.0)
        
        # Calculate market impact
        impact = np.sqrt(relative_volume) * volatility * 100
        impact = impact.fillna(0.0)
        
        return impact

class OrderFlowImbalance(SingleValueIndicator):
    """
    Order Flow Imbalance (OFI)
    
    Measures the imbalance between buy and sell orders.
    Positive values indicate more buying pressure.
    
    Simplified calculation using price movements and volume.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("ORDER_FLOW_IMBALANCE", IndicatorType.CUSTOM, config or IndicatorConfig(period=10))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("OFI requires OHLCV data with volume")
        
        close = data['close']
        volume = data['volume']
        
        # Estimate buy/sell volume based on price movement
        price_change = close.diff()
        up_volume = volume.where(price_change > 0, 0)
        down_volume = volume.where(price_change < 0, 0)
        
        # Calculate rolling imbalance
        buy_volume_sum = up_volume.rolling(window=self.config.period).sum()
        sell_volume_sum = down_volume.rolling(window=self.config.period).sum()
        total_volume_sum = volume.rolling(window=self.config.period).sum()
        
        # Order flow imbalance
        ofi = (buy_volume_sum - sell_volume_sum) / total_volume_sum
        ofi = ofi.fillna(0.0)
        
        return ofi

# ============================================
# Regime Detection Indicators
# ============================================

class TrendRegimeFilter(SingleValueIndicator):
    """
    Trend Regime Filter
    
    Identifies market regimes: trending vs ranging.
    Uses multiple timeframe analysis and volatility measures.
    
    Returns:
    - 1: Strong uptrend
    - 0.5: Weak uptrend  
    - 0: Ranging/Sideways
    - -0.5: Weak downtrend
    - -1: Strong downtrend
    """
    
    def __init__(self, 
                 short_ma: int = 10,
                 long_ma: int = 50,
                 atr_period: int = 14,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("TREND_REGIME", IndicatorType.CUSTOM, config or IndicatorConfig())
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.atr_period = atr_period
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Trend Regime Filter requires OHLC data")
        
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Calculate moving averages
        ma_short = MathUtils.simple_moving_average(close, self.short_ma)
        ma_long = MathUtils.simple_moving_average(close, self.long_ma)
        
        # Calculate ATR for trend strength
        from .volatility import AverageTrueRange
        atr_indicator = AverageTrueRange(IndicatorConfig(period=self.atr_period))
        atr = atr_indicator._calculate_values(data)
        
        # Price relative to MAs
        price_vs_short = (close - ma_short) / ma_short
        price_vs_long = (close - ma_long) / ma_long
        ma_slope = (ma_short - ma_long) / ma_long
        
        # Trend strength based on ATR
        atr_percentile = atr.rolling(window=50).rank(pct=True)
        
        # Calculate regime
        regime = pd.Series(index=close.index, dtype=float)
        
        for i in range(len(close)):
            if pd.isna(price_vs_short.iloc[i]) or pd.isna(ma_slope.iloc[i]):
                regime.iloc[i] = 0
                continue
            
            trend_score = 0
            
            # Price vs MA signals
            if price_vs_short.iloc[i] > 0.02 and price_vs_long.iloc[i] > 0.02:
                trend_score += 0.4
            elif price_vs_short.iloc[i] < -0.02 and price_vs_long.iloc[i] < -0.02:
                trend_score -= 0.4
            
            # MA slope signal
            if ma_slope.iloc[i] > 0.01:
                trend_score += 0.3
            elif ma_slope.iloc[i] < -0.01:
                trend_score -= 0.3
            
            # Trend strength from ATR
            if not pd.isna(atr_percentile.iloc[i]):
                if atr_percentile.iloc[i] > 0.7:  # High volatility = strong trend
                    trend_score *= 1.3
                elif atr_percentile.iloc[i] < 0.3:  # Low volatility = ranging
                    trend_score *= 0.7
            
            # Clamp to [-1, 1] range
            regime.iloc[i] = np.clip(trend_score, -1, 1)
        
        return regime

class VolatilityRegimeIndicator(SingleValueIndicator):
    """
    Volatility Regime Indicator
    
    Classifies volatility environments into regimes.
    Returns normalized values from 0 (low vol) to 1 (high vol).
    """
    
    def __init__(self, 
                 vol_window: int = 20,
                 regime_window: int = 252,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("VOL_REGIME", IndicatorType.CUSTOM, config or IndicatorConfig())
        self.vol_window = vol_window
        self.regime_window = regime_window
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        close = self.get_price_series(data)
        
        # Calculate realized volatility
        returns = close.pct_change()
        volatility = returns.rolling(window=self.vol_window).std() * np.sqrt(252)
        
        # Calculate rolling percentile rank over longer period
        vol_regime = volatility.rolling(
            window=self.regime_window, 
            min_periods=max(50, self.regime_window // 4)
        ).rank(pct=True)
        
        return vol_regime

# ============================================
# Machine Learning Enhanced Indicators
# ============================================

class AdaptiveTrendIndicator(SingleValueIndicator):
    """
    Adaptive Trend Indicator using ML
    
    Uses linear regression and statistical tests to identify trend strength
    and direction with adaptive parameters.
    """
    
    def __init__(self, 
                 lookback_window: int = 20,
                 min_r_squared: float = 0.3,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("ADAPTIVE_TREND", IndicatorType.CUSTOM, config or IndicatorConfig())
        self.lookback_window = lookback_window
        self.min_r_squared = min_r_squared
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        close = self.get_price_series(data)
        
        if len(close) < self.lookback_window:
            return self._handle_insufficient_data(len(close), self.lookback_window)
        
        adaptive_trend = pd.Series(index=close.index, dtype=float)
        
        for i in range(self.lookback_window - 1, len(close)):
            # Get window data
            window_close = close.iloc[i - self.lookback_window + 1:i + 1]
            X = np.arange(len(window_close)).reshape(-1, 1)
            y = window_close.values
            
            try:
                # Fit linear regression
                lr = LinearRegression()
                lr.fit(X, y)
                
                # Calculate R-squared
                y_pred = lr.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Calculate trend strength
                slope = lr.coef_[0]
                trend_strength = slope / np.std(y) if np.std(y) > 0 else 0
                
                # Apply R-squared filter
                if r_squared >= self.min_r_squared:
                    adaptive_trend.iloc[i] = trend_strength
                else:
                    adaptive_trend.iloc[i] = 0  # No clear trend
                    
            except Exception as e:
                adaptive_trend.iloc[i] = 0
        
        # Normalize to [-1, 1] range
        rolling_std = adaptive_trend.rolling(window=50).std()
        normalized_trend = adaptive_trend / (2 * rolling_std)
        normalized_trend = np.clip(normalized_trend, -1, 1)
        
        return normalized_trend

class PatternRecognitionIndicator(MultiValueIndicator):
    """
    Pattern Recognition Indicator
    
    Identifies common chart patterns using statistical methods.
    Returns probabilities for different pattern types.
    """
    
    def __init__(self, 
                 pattern_window: int = 20,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("PATTERN_RECOGNITION", IndicatorType.CUSTOM, config or IndicatorConfig())
        self.pattern_window = pattern_window
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        close = self.get_price_series(data)
        
        if len(close) < self.pattern_window:
            empty_df = pd.DataFrame({
                'double_top': [0.0] * len(close),
                'double_bottom': [0.0] * len(close),
                'head_shoulders': [0.0] * len(close),
                'triangle': [0.0] * len(close)
            }, index=close.index)
            return empty_df
        
        # Initialize result arrays
        double_top = pd.Series(0.0, index=close.index)
        double_bottom = pd.Series(0.0, index=close.index)
        head_shoulders = pd.Series(0.0, index=close.index)
        triangle = pd.Series(0.0, index=close.index)
        
        for i in range(self.pattern_window - 1, len(close)):
            window_close = close.iloc[i - self.pattern_window + 1:i + 1]
            
            # Detect patterns using statistical measures
            try:
                # Double top/bottom detection
                peaks = self._find_local_extrema(window_close, 'peaks')
                troughs = self._find_local_extrema(window_close, 'troughs')
                
                if len(peaks) >= 2:
                    peak_similarity = 1 - abs(peaks[-1] - peaks[-2]) / (peaks[-1] + 1e-8)
                    double_top.iloc[i] = max(0, min(1, peak_similarity - 0.8) * 5)
                
                if len(troughs) >= 2:
                    trough_similarity = 1 - abs(troughs[-1] - troughs[-2]) / (troughs[-1] + 1e-8)
                    double_bottom.iloc[i] = max(0, min(1, trough_similarity - 0.8) * 5)
                
                # Head and shoulders (simplified)
                if len(peaks) >= 3 and len(troughs) >= 2:
                    # Check if middle peak is highest
                    if peaks[-2] > peaks[-1] and peaks[-2] > peaks[-3]:
                        shoulder_symmetry = 1 - abs(peaks[-1] - peaks[-3]) / (peaks[-2] + 1e-8)
                        head_shoulders.iloc[i] = max(0, min(1, shoulder_symmetry - 0.7) * 3.33)
                
                # Triangle pattern (convergence)
                if len(window_close) >= 10:
                    upper_trend = np.polyfit(range(len(peaks)), peaks, 1)[0] if len(peaks) > 1 else 0
                    lower_trend = np.polyfit(range(len(troughs)), troughs, 1)[0] if len(troughs) > 1 else 0
                    
                    if upper_trend < 0 and lower_trend > 0:  # Converging
                        convergence = abs(upper_trend) + abs(lower_trend)
                        triangle.iloc[i] = max(0, min(1, convergence * 10))
                        
            except Exception:
                continue
        
        result = pd.DataFrame({
            'double_top': double_top,
            'double_bottom': double_bottom,
            'head_shoulders': head_shoulders,
            'triangle': triangle
        }, index=close.index)
        
        return result
    
    def _find_local_extrema(self, series: pd.Series, extrema_type: str) -> List[float]:
        """Find local peaks or troughs in a series"""
        extrema = []
        
        for i in range(1, len(series) - 1):
            if extrema_type == 'peaks':
                if series.iloc[i] > series.iloc[i-1] and series.iloc[i] > series.iloc[i+1]:
                    extrema.append(series.iloc[i])
            else:  # troughs
                if series.iloc[i] < series.iloc[i-1] and series.iloc[i] < series.iloc[i+1]:
                    extrema.append(series.iloc[i])
        
        return extrema

# ============================================
# Risk and Portfolio Indicators
# ============================================

class RiskAdjustedMomentum(SingleValueIndicator):
    """
    Risk-Adjusted Momentum
    
    Calculates momentum adjusted for risk (volatility).
    Higher values indicate stronger risk-adjusted performance.
    
    Formula: RAM = (Return / Volatility) × √252
    """
    
    def __init__(self, 
                 return_window: int = 20,
                 volatility_window: int = 20,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("RISK_ADJ_MOMENTUM", IndicatorType.CUSTOM, config or IndicatorConfig())
        self.return_window = return_window
        self.volatility_window = volatility_window
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        close = self.get_price_series(data)
        
        # Calculate returns
        returns = close.pct_change()
        
        # Calculate rolling return and volatility
        rolling_return = returns.rolling(window=self.return_window).mean()
        rolling_volatility = returns.rolling(window=self.volatility_window).std()
        
        # Calculate risk-adjusted momentum (annualized)
        risk_adj_momentum = (rolling_return / rolling_volatility) * np.sqrt(252)
        risk_adj_momentum = risk_adj_momentum.replace([np.inf, -np.inf], np.nan)
        risk_adj_momentum = risk_adj_momentum.fillna(0)
        
        return risk_adj_momentum

class DrawdownIndicator(SingleValueIndicator):
    """
    Maximum Drawdown Indicator
    
    Calculates rolling maximum drawdown from recent peaks.
    Returns negative values (0 to -100%).
    """
    
    def __init__(self, 
                 lookback_window: int = 252,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("MAX_DRAWDOWN", IndicatorType.CUSTOM, config or IndicatorConfig())
        self.lookback_window = lookback_window
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        close = self.get_price_series(data)
        
        # Calculate rolling maximum
        rolling_max = close.rolling(window=self.lookback_window, min_periods=1).max()
        
        # Calculate drawdown
        drawdown = (close - rolling_max) / rolling_max * 100
        
        return drawdown

class BetaIndicator(SingleValueIndicator):
    """
    Rolling Beta Indicator
    
    Calculates rolling beta relative to a benchmark.
    Requires benchmark data in the DataFrame.
    """
    
    def __init__(self, 
                 benchmark_column: str = 'benchmark',
                 config: Optional[IndicatorConfig] = None):
        super().__init__("BETA", IndicatorType.CUSTOM, config or IndicatorConfig(period=60))
        self.benchmark_column = benchmark_column
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or self.benchmark_column not in data.columns:
            logger.warning(f"Beta calculation requires {self.benchmark_column} column")
            return pd.Series(1.0, index=data.index if isinstance(data, pd.DataFrame) else data.index)
        
        close = self.get_price_series(data)
        benchmark = data[self.benchmark_column]
        
        # Calculate returns
        returns = close.pct_change()
        benchmark_returns = benchmark.pct_change()
        
        # Calculate rolling beta
        def calculate_beta(window_data):
            if len(window_data) < 10:
                return np.nan
            
            stock_returns = window_data['stock']
            bench_returns = window_data['benchmark']
            
            # Remove NaN values
            valid_mask = ~(pd.isna(stock_returns) | pd.isna(bench_returns))
            if valid_mask.sum() < 10:
                return np.nan
            
            stock_clean = stock_returns[valid_mask]
            bench_clean = bench_returns[valid_mask]
            
            if np.var(bench_clean) == 0:
                return np.nan
            
            beta = np.cov(stock_clean, bench_clean)[0, 1] / np.var(bench_clean)
            return beta
        
        combined_returns = pd.DataFrame({
            'stock': returns,
            'benchmark': benchmark_returns
        })
        
        beta = combined_returns.rolling(window=self.config.period).apply(
            lambda x: calculate_beta(x), raw=False
        )
        
        return beta.iloc[:, 0]  # Return just the beta series

# ============================================
# Sentiment and News-Based Indicators
# ============================================

class NewsVolumeIndicator(SingleValueIndicator):
    """
    News Volume Indicator
    
    Estimates market attention based on volume spikes and price movements.
    Higher values indicate periods of high market attention/news flow.
    """
    
    def __init__(self, 
                 volume_threshold: float = 1.5,
                 price_threshold: float = 0.02,
                 config: Optional[IndicatorConfig] = None):
        super().__init__("NEWS_VOLUME", IndicatorType.CUSTOM, config or IndicatorConfig(period=20))
        self.volume_threshold = volume_threshold
        self.price_threshold = price_threshold
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if not isinstance(data, pd.DataFrame) or 'volume' not in data.columns:
            raise ValidationError("News Volume Indicator requires OHLCV data with volume")
        
        close = data['close']
        volume = data['volume']
        
        # Calculate volume ratio
        avg_volume = volume.rolling(window=self.config.period).mean()
        volume_ratio = volume / avg_volume
        
        # Calculate price movement
        price_change = abs(close.pct_change())
        
        # Calculate attention score
        volume_attention = np.where(volume_ratio > self.volume_threshold, 
                                  (volume_ratio - 1) * 2, 0)
        
        price_attention = np.where(price_change > self.price_threshold,
                                 (price_change / self.price_threshold) * 2, 0)
        
        # Combined news/attention indicator
        news_indicator = (volume_attention + price_attention) / 2
        
        # Smooth the indicator
        news_smooth = pd.Series(news_indicator, index=close.index).rolling(window=3).mean()
        
        return news_smooth

class MarketStressIndicator(MultiValueIndicator):
    """
    Market Stress Indicator
    
    Combines multiple stress measures to gauge market conditions.
    Returns stress levels for different market aspects.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__("MARKET_STRESS", IndicatorType.CUSTOM, config or IndicatorConfig(period=20))
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Market Stress Indicator requires OHLC data")
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume'] if 'volume' in data.columns else None
        
        # Volatility stress
        returns = close.pct_change()
        volatility = returns.rolling(window=self.config.period).std() * np.sqrt(252)
        vol_percentile = volatility.rolling(window=252, min_periods=50).rank(pct=True)
        
        # Price stress (based on drawdowns)
        rolling_max = close.rolling(window=self.config.period).max()
        drawdown = (close - rolling_max) / rolling_max
        price_stress = abs(drawdown)
        
        # Gap stress
        gaps = abs(close - close.shift(1)) / close.shift(1)
        gap_stress = gaps.rolling(window=self.config.period).mean()
        
        # Volume stress (if available)
        if volume is not None:
            avg_volume = volume.rolling(window=self.config.period).mean()
            volume_ratio = volume / avg_volume
            volume_stress = np.where(volume_ratio > 2, (volume_ratio - 1) * 0.5, 0)
            volume_stress = pd.Series(volume_stress, index=close.index)
        else:
            volume_stress = pd.Series(0, index=close.index)
        
        # Combine stress measures
        total_stress = (vol_percentile + price_stress + gap_stress + volume_stress) / 4
        
        result = pd.DataFrame({
            'volatility_stress': vol_percentile,
            'price_stress': price_stress,
            'gap_stress': gap_stress,
            'volume_stress': volume_stress,
            'total_stress': total_stress
        }, index=close.index)
        
        return result

# ============================================
# Intermarket Analysis Indicators
# ============================================

class SectorRotationIndicator(MultiValueIndicator):
    """
    Sector Rotation Indicator
    
    Analyzes relative strength between different market sectors.
    Requires multiple sector/asset price series.
    """
    
    def __init__(self, 
                 sector_columns: List[str],
                 config: Optional[IndicatorConfig] = None):
        super().__init__("SECTOR_ROTATION", IndicatorType.CUSTOM, config or IndicatorConfig(period=20))
        self.sector_columns = sector_columns
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Sector Rotation requires DataFrame with multiple sector columns")
        
        available_sectors = [col for col in self.sector_columns if col in data.columns]
        if len(available_sectors) < 2:
            raise ValidationError(f"Need at least 2 sectors from {self.sector_columns}")
        
        # Calculate relative strength for each sector
        result = pd.DataFrame(index=data.index)
        
        # Use first sector as benchmark
        benchmark = data[available_sectors[0]]
        
        for sector in available_sectors:
            sector_price = data[sector]
            
            # Calculate relative strength
            relative_strength = sector_price / benchmark
            
            # Calculate momentum
            rs_momentum = relative_strength.pct_change(self.config.period)
            
            # Calculate percentile rank
            rs_rank = relative_strength.rolling(window=252, min_periods=50).rank(pct=True)
            
            result[f'{sector}_relative_strength'] = relative_strength
            result[f'{sector}_rs_momentum'] = rs_momentum
            result[f'{sector}_rs_rank'] = rs_rank
        
        return result

class CurrencyStrengthIndicator(MultiValueIndicator):
    """
    Currency Strength Indicator
    
    Calculates relative strength of currencies based on multiple pairs.
    Useful for forex and international equity analysis.
    """
    
    def __init__(self, 
                 currency_pairs: Dict[str, str],  # {'EURUSD': 'EUR', 'GBPUSD': 'GBP', ...}
                 config: Optional[IndicatorConfig] = None):
        super().__init__("CURRENCY_STRENGTH", IndicatorType.CUSTOM, config or IndicatorConfig(period=20))
        self.currency_pairs = currency_pairs
    
    def _calculate_values(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Currency Strength requires DataFrame with currency pair columns")
        
        # Extract available pairs
        available_pairs = {pair: currency for pair, currency in self.currency_pairs.items() 
                          if pair in data.columns}
        
        if len(available_pairs) < 2:
            raise ValidationError("Need at least 2 currency pairs")
        
        # Get unique currencies
        currencies = list(set(available_pairs.values()))
        
        # Calculate currency strength
        result = pd.DataFrame(index=data.index)
        
        for currency in currencies:
            strength_components = []
            
            for pair, pair_currency in available_pairs.items():
                pair_data = data[pair]
                pair_returns = pair_data.pct_change()
                
                if pair_currency == currency:
                    # This currency is the base currency
                    strength_components.append(pair_returns)
                else:
                    # This currency might be the quote currency
                    # (simplified approach - in reality need more complex parsing)
                    if currency in pair:
                        strength_components.append(-pair_returns)
            
            if strength_components:
                # Average strength across all pairs involving this currency
                currency_strength = pd.concat(strength_components, axis=1).mean(axis=1)
                
                # Calculate rolling momentum
                momentum = currency_strength.rolling(window=self.config.period).sum()
                
                result[f'{currency}_strength'] = momentum
        
        return result

# ============================================
# Register All Custom Indicators
# ============================================

# Register market microstructure indicators
indicator_registry.register(BidAskSpread, "BID_ASK_SPREAD", IndicatorType.CUSTOM)
indicator_registry.register(MarketImpactEstimator, "MARKET_IMPACT", IndicatorType.CUSTOM)
indicator_registry.register(OrderFlowImbalance, "ORDER_FLOW_IMBALANCE", IndicatorType.CUSTOM)

# Register regime detection indicators
indicator_registry.register(TrendRegimeFilter, "TREND_REGIME", IndicatorType.CUSTOM)
indicator_registry.register(VolatilityRegimeIndicator, "VOL_REGIME", IndicatorType.CUSTOM)

# Register ML-enhanced indicators
indicator_registry.register(AdaptiveTrendIndicator, "ADAPTIVE_TREND", IndicatorType.CUSTOM)
indicator_registry.register(PatternRecognitionIndicator, "PATTERN_RECOGNITION", IndicatorType.CUSTOM)

# Register risk and portfolio indicators
indicator_registry.register(RiskAdjustedMomentum, "RISK_ADJ_MOMENTUM", IndicatorType.CUSTOM)
indicator_registry.register(DrawdownIndicator, "MAX_DRAWDOWN", IndicatorType.CUSTOM)
indicator_registry.register(BetaIndicator, "BETA", IndicatorType.CUSTOM)

# Register sentiment indicators
indicator_registry.register(NewsVolumeIndicator, "NEWS_VOLUME", IndicatorType.CUSTOM)
indicator_registry.register(MarketStressIndicator, "MARKET_STRESS", IndicatorType.CUSTOM)

# Register intermarket indicators
indicator_registry.register(SectorRotationIndicator, "SECTOR_ROTATION", IndicatorType.CUSTOM)
indicator_registry.register(CurrencyStrengthIndicator, "CURRENCY_STRENGTH", IndicatorType.CUSTOM)

# ============================================
# Utility Functions
# ============================================

@time_it("custom_indicator_suite")
def calculate_custom_suite(data: Union[pd.DataFrame, pd.Series],
                          include_ml: bool = True,
                          include_intermarket: bool = False) -> Dict[str, IndicatorResult]:
    """Calculate comprehensive custom indicator suite"""
    
    results = {}
    
    # Market microstructure (if bid/ask data available)
    try:
        if isinstance(data, pd.DataFrame):
            if 'bid' in data.columns and 'ask' in data.columns:
                spread = BidAskSpread()
                results['bid_ask_spread'] = spread.calculate(data)
            
            if 'volume' in data.columns:
                impact = MarketImpactEstimator()
                results['market_impact'] = impact.calculate(data)
                
                ofi = OrderFlowImbalance()
                results['order_flow_imbalance'] = ofi.calculate(data)
    except Exception as e:
        logger.warning(f"Error calculating microstructure indicators: {e}")
    
    # Regime detection
    try:
        if isinstance(data, pd.DataFrame):
            trend_regime = TrendRegimeFilter()
            results['trend_regime'] = trend_regime.calculate(data)
        
        vol_regime = VolatilityRegimeIndicator()
        results['volatility_regime'] = vol_regime.calculate(data)
    except Exception as e:
        logger.warning(f"Error calculating regime indicators: {e}")
    
    # ML-enhanced indicators
    if include_ml:
        try:
            adaptive_trend = AdaptiveTrendIndicator()
            results['adaptive_trend'] = adaptive_trend.calculate(data)
            
            if isinstance(data, pd.DataFrame):
                pattern_recognition = PatternRecognitionIndicator()
                results['pattern_recognition'] = pattern_recognition.calculate(data)
        except Exception as e:
            logger.warning(f"Error calculating ML indicators: {e}")
    
    # Risk indicators
    try:
        risk_adj_momentum = RiskAdjustedMomentum()
        results['risk_adjusted_momentum'] = risk_adj_momentum.calculate(data)
        
        drawdown = DrawdownIndicator()
        results['max_drawdown'] = drawdown.calculate(data)
    except Exception as e:
        logger.warning(f"Error calculating risk indicators: {e}")
    
    # Sentiment indicators
    try:
        if isinstance(data, pd.DataFrame) and 'volume' in data.columns:
            news_volume = NewsVolumeIndicator()
            results['news_volume'] = news_volume.calculate(data)
        
        if isinstance(data, pd.DataFrame):
            market_stress = MarketStressIndicator()
            results['market_stress'] = market_stress.calculate(data)
    except Exception as e:
        logger.warning(f"Error calculating sentiment indicators: {e}")
    
    logger.info(f"Calculated {len(results)} custom indicators")
    return results

def create_regime_analysis(data: Union[pd.DataFrame, pd.Series]) -> Dict[str, Any]:
    """Create comprehensive regime analysis"""
    
    analysis = {}
    
    # Trend regime
    try:
        trend_regime = TrendRegimeFilter()
        trend_result = trend_regime.calculate(data)
        trend_values = trend_result.values
        
        current_trend = trend_values.iloc[-1]
        if current_trend > 0.5:
            trend_label = "Strong Uptrend"
        elif current_trend > 0:
            trend_label = "Weak Uptrend"
        elif current_trend < -0.5:
            trend_label = "Strong Downtrend"
        elif current_trend < 0:
            trend_label = "Weak Downtrend"
        else:
            trend_label = "Ranging/Sideways"
        
        analysis['trend_regime'] = {
            'current_value': current_trend,
            'current_label': trend_label,
            'recent_average': trend_values.tail(10).mean(),
            'regime_persistence': (trend_values.tail(5) > 0).sum() / 5
        }
    except Exception as e:
        logger.warning(f"Error in trend regime analysis: {e}")
    
    # Volatility regime
    try:
        vol_regime = VolatilityRegimeIndicator()
        vol_result = vol_regime.calculate(data)
        vol_values = vol_result.values
        
        current_vol_regime = vol_values.iloc[-1]
        if current_vol_regime > 0.8:
            vol_label = "High Volatility"
        elif current_vol_regime > 0.6:
            vol_label = "Elevated Volatility"
        elif current_vol_regime > 0.4:
            vol_label = "Normal Volatility"
        elif current_vol_regime > 0.2:
            vol_label = "Low Volatility"
        else:
            vol_label = "Very Low Volatility"
        
        analysis['volatility_regime'] = {
            'current_value': current_vol_regime,
            'current_label': vol_label,
            'recent_trend': 'increasing' if vol_values.iloc[-1] > vol_values.tail(5).mean() else 'decreasing'
        }
    except Exception as e:
        logger.warning(f"Error in volatility regime analysis: {e}")
    
    return analysis

def create_risk_dashboard(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Create risk-focused dashboard"""
    
    dashboard = pd.DataFrame(index=data.index if isinstance(data, pd.DataFrame) else data.index)
    
    # Risk-adjusted momentum
    try:
        risk_momentum = RiskAdjustedMomentum()
        ram_result = risk_momentum.calculate(data)
        dashboard['risk_adj_momentum'] = ram_result.values
    except Exception as e:
        logger.warning(f"Error calculating risk-adjusted momentum: {e}")
    
    # Drawdown
    try:
        drawdown = DrawdownIndicator()
        dd_result = drawdown.calculate(data)
        dashboard['max_drawdown'] = dd_result.values
        dashboard['drawdown_severity'] = pd.cut(
            dd_result.values, 
            bins=[-100, -20, -10, -5, 0], 
            labels=['Severe', 'High', 'Moderate', 'Low']
        )
    except Exception as e:
        logger.warning(f"Error calculating drawdown: {e}")
    
    # Market stress
    try:
        if isinstance(data, pd.DataFrame):
            stress = MarketStressIndicator()
            stress_result = stress.calculate(data)
            dashboard['market_stress'] = stress_result.values['total_stress']
            dashboard['vol_stress'] = stress_result.values['volatility_stress']
    except Exception as e:
        logger.warning(f"Error calculating market stress: {e}")
    
    return dashboard

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    from .base import create_sample_data
    
    print("Testing Custom Indicators")
    
    # Create sample data
    sample_data = create_sample_data(300, start_price=100.0, volatility=0.025)
    print(f"Sample data shape: {sample_data.shape}")
    
    # Test individual indicators
    trend_regime = TrendRegimeFilter()
    trend_result = trend_regime.calculate(sample_data)
    print(f"Trend regime last 5 values: {trend_result.values.tail()}")
    
    risk_momentum = RiskAdjustedMomentum()
    ram_result = risk_momentum.calculate(sample_data)
    print(f"Risk-adjusted momentum last 5 values: {ram_result.values.tail()}")
    
    drawdown = DrawdownIndicator()
    dd_result = drawdown.calculate(sample_data)
    print(f"Max drawdown last 5 values: {dd_result.values.tail()}")
    
    # Test custom suite
    custom_suite = calculate_custom_suite(sample_data, include_ml=True)
    print(f"Custom suite calculated {len(custom_suite)} indicators")
    
    # Test regime analysis
    regime_analysis = create_regime_analysis(sample_data)
    if 'trend_regime' in regime_analysis:
        print(f"Current trend regime: {regime_analysis['trend_regime']['current_label']}")
    if 'volatility_regime' in regime_analysis:
        print(f"Current volatility regime: {regime_analysis['volatility_regime']['current_label']}")
    
    # Test risk dashboard
    risk_dashboard = create_risk_dashboard(sample_data)
    print(f"Risk dashboard created with {len(risk_dashboard.columns)} columns")
    
    if 'max_drawdown' in risk_dashboard.columns:
        current_drawdown = risk_dashboard['max_drawdown'].iloc[-1]
        print(f"Current drawdown: {current_drawdown:.2f}%")
    
    print("Custom indicators testing completed successfully!")
