# ============================================
# StockPredictionPro - src/evaluation/metrics/custom_metrics.py
# Custom financial metrics for specialized trading evaluation
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from scipy import stats
import math

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('evaluation.metrics.custom')

# ============================================
# Custom Risk Metrics
# ============================================

@time_it("tail_ratio_calculation")
def calculate_tail_ratio(returns: Union[np.ndarray, pd.Series],
                        percentile: float = 5.0) -> float:
    """
    Calculate Tail Ratio - ratio of right tail to left tail
    Measures return distribution asymmetry
    
    Args:
        returns: Returns series
        percentile: Percentile for tail calculation (default 5%)
        
    Returns:
        Tail ratio (> 1 indicates positive skew in tails)
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 1.0
    
    # Calculate tail values
    upper_tail = np.percentile(returns_clean, 100 - percentile)
    lower_tail = np.percentile(returns_clean, percentile)
    
    if lower_tail == 0:
        return float('inf') if upper_tail > 0 else 1.0
    
    # Tail ratio (right tail / abs(left tail))
    tail_ratio = upper_tail / abs(lower_tail)
    
    return float(tail_ratio)


@time_it("pain_index_calculation")
def calculate_pain_index(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Pain Index - average of squared drawdowns
    Measures persistent pain from drawdowns
    
    Args:
        returns: Returns series
        
    Returns:
        Pain Index value
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns_clean)
    
    # Calculate running maximum (peaks)
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Calculate drawdown percentage
    drawdowns = (cumulative_returns - running_max) / running_max
    
    # Pain index is average of squared drawdowns
    pain_index = np.mean(drawdowns ** 2)
    
    return float(abs(pain_index))


@time_it("ulcer_index_calculation")
def calculate_ulcer_index(returns: Union[np.ndarray, pd.Series],
                         periods_per_year: int = 252) -> float:
    """
    Calculate Ulcer Index - RMS of drawdowns
    Alternative to standard deviation focusing on downside
    
    Args:
        returns: Returns series
        periods_per_year: Trading periods per year
        
    Returns:
        Annualized Ulcer Index
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns_clean)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Calculate drawdown percentage
    drawdowns = (cumulative_returns - running_max) / running_max * 100
    
    # Ulcer index is RMS of drawdowns
    ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
    
    # Annualize if needed
    if periods_per_year != 252:
        ulcer_index = ulcer_index * np.sqrt(periods_per_year / 252)
    
    return float(ulcer_index)


@time_it("martin_ratio_calculation")
def calculate_martin_ratio(returns: Union[np.ndarray, pd.Series],
                          risk_free_rate: float = 0.02,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Martin Ratio - excess return / Ulcer Index
    Risk-adjusted return using Ulcer Index as risk measure
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        
    Returns:
        Martin Ratio
    """
    
    from .trading_metrics import calculate_annualized_return
    
    # Calculate annualized excess return
    annual_return = calculate_annualized_return(returns, periods_per_year)
    excess_return = annual_return - risk_free_rate
    
    # Calculate Ulcer Index
    ulcer_index = calculate_ulcer_index(returns, periods_per_year)
    
    if ulcer_index == 0:
        return float('inf') if excess_return > 0 else 0.0
    
    martin_ratio = excess_return / ulcer_index
    
    return float(martin_ratio)

# ============================================
# Behavioral Finance Metrics
# ============================================

@time_it("gain_pain_ratio_calculation")
def calculate_gain_pain_ratio(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Gain-to-Pain Ratio
    Ratio of positive returns to negative returns
    
    Args:
        returns: Returns series
        
    Returns:
        Gain-to-Pain ratio
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    gains = returns_clean[returns_clean > 0]
    pains = returns_clean[returns_clean < 0]
    
    if len(pains) == 0:
        return float('inf') if len(gains) > 0 else 0.0
    
    if len(gains) == 0:
        return 0.0
    
    gain_sum = np.sum(gains)
    pain_sum = abs(np.sum(pains))
    
    return float(gain_sum / pain_sum)


@time_it("lake_ratio_calculation")
def calculate_lake_ratio(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Lake Ratio - area under water / total area
    Measures time spent in drawdown
    
    Args:
        returns: Returns series
        
    Returns:
        Lake Ratio (0 to 1, lower is better)
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns_clean)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Calculate drawdown
    drawdowns = (cumulative_returns - running_max) / running_max
    
    # Lake ratio is area under water / total area
    underwater_area = abs(np.sum(drawdowns))
    total_area = len(drawdowns)
    
    lake_ratio = underwater_area / total_area if total_area > 0 else 0.0
    
    return float(lake_ratio)

# ============================================
# Momentum and Trend Metrics
# ============================================

@time_it("hurst_exponent_calculation")
def calculate_hurst_exponent(returns: Union[np.ndarray, pd.Series],
                            max_lags: int = 20) -> float:
    """
    Calculate Hurst Exponent - measure of trend persistence
    H > 0.5: trending, H < 0.5: mean reverting, H = 0.5: random walk
    
    Args:
        returns: Returns series
        max_lags: Maximum number of lags to use
        
    Returns:
        Hurst exponent
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < max_lags * 2:
        return 0.5  # Default to random walk
    
    # Convert to price series
    price_series = np.cumsum(returns_clean)
    
    lags = range(2, max_lags + 1)
    rs_values = []
    
    for lag in lags:
        # Calculate R/S statistic
        ts_len = len(price_series)
        sections = ts_len // lag
        
        if sections == 0:
            continue
        
        rs_list = []
        
        for i in range(sections):
            start_idx = i * lag
            end_idx = start_idx + lag
            section = price_series[start_idx:end_idx]
            
            if len(section) < 2:
                continue
            
            # Mean-adjusted series
            mean_adj = section - np.mean(section)
            
            # Cumulative sum
            cum_sum = np.cumsum(mean_adj)
            
            # Range
            R = np.max(cum_sum) - np.min(cum_sum)
            
            # Standard deviation
            S = np.std(section, ddof=1)
            
            if S > 0:
                rs_list.append(R / S)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
    
    if len(rs_values) < 2:
        return 0.5
    
    # Linear regression of log(R/S) vs log(lag)
    log_rs = np.log(rs_values)
    log_lags = np.log(lags[:len(rs_values)])
    
    # Hurst exponent is the slope
    hurst = np.polyfit(log_lags, log_rs, 1)[0]
    
    # Bound between 0 and 1
    hurst = max(0.0, min(1.0, hurst))
    
    return float(hurst)


@time_it("trend_strength_calculation")
def calculate_trend_strength(returns: Union[np.ndarray, pd.Series],
                           window: int = 20) -> float:
    """
    Calculate Trend Strength using linear regression R-squared
    
    Args:
        returns: Returns series
        window: Rolling window for calculation
        
    Returns:
        Average trend strength (0 to 1)
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < window:
        return 0.0
    
    # Convert to price series
    price_series = np.cumprod(1 + returns_clean)
    
    trend_strengths = []
    
    for i in range(window, len(price_series)):
        window_prices = price_series[i-window:i]
        x = np.arange(len(window_prices))
        
        # Linear regression
        correlation = np.corrcoef(x, window_prices)[0, 1]
        
        if not np.isnan(correlation):
            r_squared = correlation ** 2
            trend_strengths.append(r_squared)
    
    if not trend_strengths:
        return 0.0
    
    return float(np.mean(trend_strengths))

# ============================================
# Market Timing Metrics
# ============================================

@time_it("market_timing_calculation")
def calculate_market_timing_metrics(returns: Union[np.ndarray, pd.Series],
                                   benchmark_returns: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate Market Timing Metrics (Treynor-Mazuy)
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Dictionary with timing metrics
    """
    
    returns_array = np.asarray(returns)
    benchmark_array = np.asarray(benchmark_returns)
    
    # Align lengths
    min_length = min(len(returns_array), len(benchmark_array))
    if min_length == 0:
        return {'timing_coefficient': 0.0, 'selectivity': 0.0}
    
    returns_aligned = returns_array[-min_length:]
    benchmark_aligned = benchmark_array[-min_length:]
    
    # Remove NaN values
    mask = ~(np.isnan(returns_aligned) | np.isnan(benchmark_aligned))
    returns_clean = returns_aligned[mask]
    benchmark_clean = benchmark_aligned[mask]
    
    if len(returns_clean) < 10:
        return {'timing_coefficient': 0.0, 'selectivity': 0.0}
    
    # Treynor-Mazuy regression: Rp - Rf = alpha + beta*(Rm - Rf) + gamma*(Rm - Rf)^2
    # Simplified assuming Rf = 0
    
    x1 = benchmark_clean  # Linear term
    x2 = benchmark_clean ** 2  # Quadratic term (timing)
    
    # Multiple regression
    X = np.column_stack([np.ones(len(x1)), x1, x2])
    
    try:
        coefficients = np.linalg.lstsq(X, returns_clean, rcond=None)[0]
        
        alpha = coefficients[0]  # Selectivity (alpha)
        beta = coefficients[1]   # Market sensitivity
        gamma = coefficients[2]  # Timing coefficient
        
        return {
            'selectivity': float(alpha),
            'market_beta': float(beta),
            'timing_coefficient': float(gamma)
        }
    
    except:
        return {'timing_coefficient': 0.0, 'selectivity': 0.0, 'market_beta': 1.0}


@time_it("batting_average_calculation")
def calculate_batting_average(returns: Union[np.ndarray, pd.Series],
                             benchmark_returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Batting Average - percentage of periods outperforming benchmark
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Batting average (0 to 1)
    """
    
    returns_array = np.asarray(returns)
    benchmark_array = np.asarray(benchmark_returns)
    
    # Align lengths
    min_length = min(len(returns_array), len(benchmark_array))
    if min_length == 0:
        return 0.0
    
    returns_aligned = returns_array[-min_length:]
    benchmark_aligned = benchmark_array[-min_length:]
    
    # Remove NaN values
    mask = ~(np.isnan(returns_aligned) | np.isnan(benchmark_aligned))
    returns_clean = returns_aligned[mask]
    benchmark_clean = benchmark_aligned[mask]
    
    if len(returns_clean) == 0:
        return 0.0
    
    # Count outperformance periods
    outperform = returns_clean > benchmark_clean
    batting_avg = np.mean(outperform)
    
    return float(batting_avg)

# ============================================
# Volatility and Stability Metrics
# ============================================

@time_it("volatility_clustering_calculation")
def calculate_volatility_clustering(returns: Union[np.ndarray, pd.Series],
                                   window: int = 20) -> float:
    """
    Calculate Volatility Clustering metric
    Measures persistence of volatility levels
    
    Args:
        returns: Returns series
        window: Window for volatility calculation
        
    Returns:
        Volatility clustering coefficient (0 to 1)
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < window * 2:
        return 0.0
    
    # Calculate rolling volatility
    volatilities = []
    for i in range(window, len(returns_clean)):
        window_returns = returns_clean[i-window:i]
        vol = np.std(window_returns)
        volatilities.append(vol)
    
    if len(volatilities) < 2:
        return 0.0
    
    # Calculate autocorrelation of volatilities
    vol_series = pd.Series(volatilities)
    clustering = vol_series.autocorr(lag=1)
    
    if np.isnan(clustering):
        return 0.0
    
    return float(abs(clustering))


@time_it("stability_ratio_calculation")
def calculate_stability_ratio(returns: Union[np.ndarray, pd.Series],
                             window: int = 252) -> float:
    """
    Calculate Stability Ratio - consistency of returns over time
    
    Args:
        returns: Returns series
        window: Window for stability calculation
        
    Returns:
        Stability ratio (higher is more stable)
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < window:
        return 0.0
    
    # Calculate rolling annual returns
    annual_returns = []
    for i in range(window, len(returns_clean), window // 4):  # Quarterly rolling
        period_returns = returns_clean[i-window:i]
        annual_return = (np.prod(1 + period_returns) ** (252 / len(period_returns))) - 1
        annual_returns.append(annual_return)
    
    if len(annual_returns) < 2:
        return 0.0
    
    # Stability is inverse of coefficient of variation
    mean_return = np.mean(annual_returns)
    std_return = np.std(annual_returns)
    
    if std_return == 0:
        return float('inf') if mean_return >= 0 else 0.0
    
    stability = abs(mean_return) / std_return
    
    return float(stability)

# ============================================
# Advanced Custom Metrics
# ============================================

@time_it("asymmetric_volatility_calculation")
def calculate_asymmetric_volatility(returns: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate Asymmetric Volatility Metrics
    Separate volatility for up and down moves
    
    Args:
        returns: Returns series
        
    Returns:
        Dictionary with upside/downside volatility metrics
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return {
            'upside_volatility': 0.0,
            'downside_volatility': 0.0,
            'volatility_asymmetry': 0.0
        }
    
    # Separate up and down moves
    up_moves = returns_clean[returns_clean > 0]
    down_moves = returns_clean[returns_clean < 0]
    
    upside_vol = np.std(up_moves) if len(up_moves) > 1 else 0.0
    downside_vol = np.std(down_moves) if len(down_moves) > 1 else 0.0
    
    # Asymmetry ratio
    if downside_vol > 0:
        asymmetry = upside_vol / downside_vol
    else:
        asymmetry = float('inf') if upside_vol > 0 else 1.0
    
    return {
        'upside_volatility': float(upside_vol),
        'downside_volatility': float(downside_vol),
        'volatility_asymmetry': float(asymmetry)
    }


@time_it("regime_detection_calculation")
def calculate_regime_metrics(returns: Union[np.ndarray, pd.Series],
                           window: int = 60) -> Dict[str, float]:
    """
    Calculate Regime Detection Metrics
    Identify market regime characteristics
    
    Args:
        returns: Returns series
        window: Window for regime calculation
        
    Returns:
        Dictionary with regime metrics
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) < window:
        return {
            'regime_consistency': 0.0,
            'trend_persistence': 0.0,
            'volatility_regime': 0.0
        }
    
    # Calculate rolling metrics
    rolling_means = []
    rolling_vols = []
    trend_signals = []
    
    for i in range(window, len(returns_clean)):
        window_returns = returns_clean[i-window:i]
        
        # Mean and volatility
        rolling_means.append(np.mean(window_returns))
        rolling_vols.append(np.std(window_returns))
        
        # Trend signal (positive if mean > 0)
        trend_signals.append(1 if np.mean(window_returns) > 0 else -1)
    
    if len(rolling_means) < 2:
        return {
            'regime_consistency': 0.0,
            'trend_persistence': 0.0,
            'volatility_regime': 0.0
        }
    
    # Regime consistency (stability of trend direction)
    trend_changes = np.diff(trend_signals)
    regime_consistency = 1 - (np.sum(trend_changes != 0) / len(trend_changes))
    
    # Trend persistence (autocorrelation of returns)
    persistence = pd.Series(returns_clean).autocorr(lag=1)
    persistence = persistence if not np.isnan(persistence) else 0.0
    
    # Volatility regime (consistency of volatility levels)
    vol_consistency = 1 - (np.std(rolling_vols) / np.mean(rolling_vols)) if np.mean(rolling_vols) > 0 else 0.0
    
    return {
        'regime_consistency': float(regime_consistency),
        'trend_persistence': float(persistence),
        'volatility_regime': float(vol_consistency)
    }

# ============================================
# Custom Metrics Calculator
# ============================================

class CustomMetricsCalculator:
    """
    Calculator for custom financial metrics
    """
    
    def __init__(self, periods_per_year: int = 252):
        self.periods_per_year = periods_per_year
    
    @time_it("custom_metrics_calculation")
    def calculate_all_custom_metrics(self, 
                                   returns: Union[np.ndarray, pd.Series],
                                   benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive set of custom metrics
        
        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns
            
        Returns:
            Dictionary of all custom metrics
        """
        
        metrics = {}
        
        try:
            # Risk metrics
            metrics['tail_ratio'] = calculate_tail_ratio(returns)
            metrics['pain_index'] = calculate_pain_index(returns)
            metrics['ulcer_index'] = calculate_ulcer_index(returns, self.periods_per_year)
            metrics['martin_ratio'] = calculate_martin_ratio(returns, periods_per_year=self.periods_per_year)
            
            # Behavioral metrics
            metrics['gain_pain_ratio'] = calculate_gain_pain_ratio(returns)
            metrics['lake_ratio'] = calculate_lake_ratio(returns)
            
            # Trend metrics
            metrics['hurst_exponent'] = calculate_hurst_exponent(returns)
            metrics['trend_strength'] = calculate_trend_strength(returns)
            
            # Volatility metrics
            metrics['volatility_clustering'] = calculate_volatility_clustering(returns)
            metrics['stability_ratio'] = calculate_stability_ratio(returns)
            
            # Asymmetric volatility
            asym_vol = calculate_asymmetric_volatility(returns)
            metrics.update({f'asym_{k}': v for k, v in asym_vol.items()})
            
            # Regime metrics
            regime_metrics = calculate_regime_metrics(returns)
            metrics.update({f'regime_{k}': v for k, v in regime_metrics.items()})
            
            # Benchmark comparison metrics
            if benchmark_returns is not None:
                timing_metrics = calculate_market_timing_metrics(returns, benchmark_returns)
                metrics.update({f'timing_{k}': v for k, v in timing_metrics.items()})
                
                metrics['batting_average'] = calculate_batting_average(returns, benchmark_returns)
            
        except Exception as e:
            logger.error(f"Error calculating custom metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all custom metrics"""
        
        descriptions = {
            'tail_ratio': 'Ratio of right tail to left tail returns (asymmetry measure)',
            'pain_index': 'Average of squared drawdowns (persistent pain measure)',
            'ulcer_index': 'RMS of drawdowns (downside risk measure)',
            'martin_ratio': 'Excess return divided by Ulcer Index',
            'gain_pain_ratio': 'Ratio of positive to negative returns',
            'lake_ratio': 'Fraction of time spent in drawdown',
            'hurst_exponent': 'Trend persistence measure (>0.5 trending, <0.5 mean reverting)',
            'trend_strength': 'Average R-squared of rolling linear trends',
            'volatility_clustering': 'Persistence of volatility levels',
            'stability_ratio': 'Consistency of returns over time',
            'asym_upside_volatility': 'Volatility of positive returns',
            'asym_downside_volatility': 'Volatility of negative returns',
            'asym_volatility_asymmetry': 'Ratio of upside to downside volatility',
            'regime_consistency': 'Stability of market regime direction',
            'regime_trend_persistence': 'Autocorrelation of returns',
            'regime_volatility_regime': 'Consistency of volatility levels',
            'timing_selectivity': 'Alpha from Treynor-Mazuy model',
            'timing_market_beta': 'Market sensitivity',
            'timing_timing_coefficient': 'Market timing ability',
            'batting_average': 'Percentage of periods outperforming benchmark'
        }
        
        return descriptions

# ============================================
# Utility Functions
# ============================================

def calculate_custom_metric_suite(returns: Union[np.ndarray, pd.Series],
                                 benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
    """
    Quick utility to calculate all custom metrics
    
    Args:
        returns: Strategy returns
        benchmark_returns: Optional benchmark returns
        
    Returns:
        Dictionary of all custom metrics with descriptions
    """
    
    calculator = CustomMetricsCalculator()
    metrics = calculator.calculate_all_custom_metrics(returns, benchmark_returns)
    descriptions = calculator.get_metric_descriptions()
    
    return {
        'metrics': metrics,
        'descriptions': descriptions
    }

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Custom Financial Metrics")
    
    # Generate sample data
    np.random.seed(42)
    n_periods = 252 * 3  # 3 years of daily data
    
    # Strategy returns with some interesting characteristics
    base_returns = np.random.normal(0.0005, 0.015, n_periods)
    
    # Add volatility clustering
    vol_factor = np.ones(n_periods)
    for i in range(1, n_periods):
        vol_factor[i] = 0.95 * vol_factor[i-1] + 0.05 * abs(base_returns[i-1]) * 10
    
    strategy_returns = base_returns * vol_factor
    
    # Add some trend persistence
    for i in range(1, n_periods):
        if i > 20:
            momentum = np.mean(strategy_returns[i-20:i])
            strategy_returns[i] += 0.1 * momentum
    
    # Benchmark returns
    benchmark_returns = np.random.normal(0.0003, 0.012, n_periods)
    
    print("Sample Custom Metrics Calculation:")
    print("=" * 50)
    
    # Test individual metrics
    print(f"Tail Ratio: {calculate_tail_ratio(strategy_returns):.3f}")
    print(f"Pain Index: {calculate_pain_index(strategy_returns):.4f}")
    print(f"Ulcer Index: {calculate_ulcer_index(strategy_returns):.2f}")
    print(f"Martin Ratio: {calculate_martin_ratio(strategy_returns):.3f}")
    print(f"Hurst Exponent: {calculate_hurst_exponent(strategy_returns):.3f}")
    print(f"Volatility Clustering: {calculate_volatility_clustering(strategy_returns):.3f}")
    print(f"Batting Average: {calculate_batting_average(strategy_returns, benchmark_returns):.1%}")
    
    # Test comprehensive calculator
    print("\n" + "=" * 50)
    print("COMPREHENSIVE CUSTOM METRICS:")
    
    calculator = CustomMetricsCalculator()
    all_metrics = calculator.calculate_all_custom_metrics(strategy_returns, benchmark_returns)
    descriptions = calculator.get_metric_descriptions()
    
    for metric_name, value in all_metrics.items():
        if metric_name != 'error':
            description = descriptions.get(metric_name, "No description available")
            if isinstance(value, float):
                print(f"{metric_name}: {value:.4f}")
                print(f"  └─ {description}")
            else:
                print(f"{metric_name}: {value}")
    
    # Test regime metrics
    print("\n" + "=" * 30)
    print("REGIME ANALYSIS:")
    
    regime_metrics = calculate_regime_metrics(strategy_returns)
    for key, value in regime_metrics.items():
        print(f"{key}: {value:.3f}")
    
    # Test asymmetric volatility
    print("\n" + "=" * 30)
    print("ASYMMETRIC VOLATILITY:")
    
    asym_vol = calculate_asymmetric_volatility(strategy_returns)
    for key, value in asym_vol.items():
        print(f"{key}: {value:.4f}")
    
    print("\nCustom metrics testing completed successfully!")
