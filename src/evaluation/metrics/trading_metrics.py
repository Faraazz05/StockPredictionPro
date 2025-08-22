# ============================================
# StockPredictionPro - src/evaluation/metrics/trading_metrics.py
# Comprehensive trading performance metrics for financial machine learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
from datetime import datetime, timedelta

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('evaluation.metrics.trading')

# ============================================
# Core Trading Metrics
# ============================================

@time_it("sharpe_ratio_calculation")
def calculate_sharpe_ratio(returns: Union[np.ndarray, pd.Series], 
                          risk_free_rate: float = 0.02,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe Ratio - risk-adjusted return metric
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Trading periods per year (252 for daily)
        
    Returns:
        Sharpe ratio value
    """
    
    if len(returns) == 0:
        return 0.0
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    # Convert annual risk-free rate to period rate
    period_rf_rate = risk_free_rate / periods_per_year
    
    # Calculate excess returns
    excess_returns = returns_clean - period_rf_rate
    
    # Calculate Sharpe ratio
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns, ddof=1)
    
    if std_excess_return == 0 or np.isnan(std_excess_return):
        return 0.0
    
    sharpe = (mean_excess_return / std_excess_return) * np.sqrt(periods_per_year)
    
    return float(sharpe)


@time_it("sortino_ratio_calculation")
def calculate_sortino_ratio(returns: Union[np.ndarray, pd.Series],
                           risk_free_rate: float = 0.02,
                           target_return: float = 0.0,
                           periods_per_year: int = 252) -> float:
    """
    Calculate Sortino Ratio - downside risk-adjusted return
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        target_return: Target return threshold
        periods_per_year: Trading periods per year
        
    Returns:
        Sortino ratio value
    """
    
    if len(returns) == 0:
        return 0.0
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    # Convert annual rates to period rates
    period_rf_rate = risk_free_rate / periods_per_year
    period_target = target_return / periods_per_year
    
    # Calculate excess returns
    excess_returns = returns_clean - period_rf_rate
    
    # Calculate downside deviation
    downside_returns = excess_returns - period_target
    downside_returns = downside_returns[downside_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    sortino = (np.mean(excess_returns) / downside_deviation) * np.sqrt(periods_per_year)
    
    return float(sortino)


@time_it("max_drawdown_calculation")
def calculate_max_drawdown(returns: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate Maximum Drawdown and related metrics
    
    Args:
        returns: Daily returns series
        
    Returns:
        Dictionary with max_drawdown, max_drawdown_duration, recovery_time
    """
    
    if len(returns) == 0:
        return {'max_drawdown': 0.0, 'max_drawdown_duration': 0, 'recovery_time': 0}
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return {'max_drawdown': 0.0, 'max_drawdown_duration': 0, 'recovery_time': 0}
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns_clean)
    
    # Calculate running maximum (peaks)
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Maximum drawdown
    max_drawdown = np.min(drawdown)
    
    # Find drawdown periods
    is_drawdown = drawdown < 0
    drawdown_periods = []
    start_idx = None
    
    for i, in_drawdown in enumerate(is_drawdown):
        if in_drawdown and start_idx is None:
            start_idx = i
        elif not in_drawdown and start_idx is not None:
            drawdown_periods.append((start_idx, i - 1))
            start_idx = None
    
    # Handle case where drawdown extends to the end
    if start_idx is not None:
        drawdown_periods.append((start_idx, len(is_drawdown) - 1))
    
    # Calculate maximum drawdown duration
    max_drawdown_duration = 0
    if drawdown_periods:
        max_drawdown_duration = max(end - start + 1 for start, end in drawdown_periods)
    
    # Calculate recovery time (time to recover from max drawdown)
    max_dd_idx = np.argmin(drawdown)
    recovery_time = 0
    
    for i in range(max_dd_idx + 1, len(drawdown)):
        if drawdown[i] >= 0:
            recovery_time = i - max_dd_idx
            break
    else:
        # If no recovery by end of series
        recovery_time = len(drawdown) - max_dd_idx - 1
    
    return {
        'max_drawdown': float(abs(max_drawdown)),
        'max_drawdown_duration': int(max_drawdown_duration),
        'recovery_time': int(recovery_time)
    }


@time_it("calmar_ratio_calculation")
def calculate_calmar_ratio(returns: Union[np.ndarray, pd.Series],
                          periods_per_year: int = 252) -> float:
    """
    Calculate Calmar Ratio - return to max drawdown ratio
    
    Args:
        returns: Daily returns series
        periods_per_year: Trading periods per year
        
    Returns:
        Calmar ratio value
    """
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate annualized return
    annual_return = calculate_annualized_return(returns, periods_per_year)
    
    # Calculate max drawdown
    max_dd_info = calculate_max_drawdown(returns)
    max_drawdown = max_dd_info['max_drawdown']
    
    if max_drawdown == 0:
        return float('inf') if annual_return > 0 else 0.0
    
    calmar = annual_return / max_drawdown
    
    return float(calmar)


def calculate_annualized_return(returns: Union[np.ndarray, pd.Series],
                               periods_per_year: int = 252) -> float:
    """
    Calculate annualized return from period returns
    
    Args:
        returns: Period returns series
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized return
    """
    
    if len(returns) == 0:
        return 0.0
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    # Calculate cumulative return
    total_return = np.prod(1 + returns_clean) - 1
    
    # Annualize
    num_periods = len(returns_clean)
    years = num_periods / periods_per_year
    
    if years == 0:
        return 0.0
    
    annualized = (1 + total_return) ** (1 / years) - 1
    
    return float(annualized)


def calculate_annualized_volatility(returns: Union[np.ndarray, pd.Series],
                                   periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility
    
    Args:
        returns: Period returns series
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized volatility
    """
    
    if len(returns) == 0:
        return 0.0
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    volatility = np.std(returns_clean, ddof=1) * np.sqrt(periods_per_year)
    
    return float(volatility)

# ============================================
# Advanced Trading Metrics
# ============================================

def calculate_information_ratio(returns: Union[np.ndarray, pd.Series],
                               benchmark_returns: Union[np.ndarray, pd.Series],
                               periods_per_year: int = 252) -> float:
    """
    Calculate Information Ratio - active return per unit of tracking error
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Trading periods per year
        
    Returns:
        Information ratio
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
    
    # Calculate active returns
    active_returns = returns_clean - benchmark_clean
    
    # Information ratio
    mean_active_return = np.mean(active_returns)
    tracking_error = np.std(active_returns, ddof=1)
    
    if tracking_error == 0:
        return 0.0
    
    info_ratio = (mean_active_return / tracking_error) * np.sqrt(periods_per_year)
    
    return float(info_ratio)


def calculate_omega_ratio(returns: Union[np.ndarray, pd.Series],
                         threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio - probability weighted ratio of gains to losses
    
    Args:
        returns: Returns series
        threshold: Threshold return (default 0%)
        
    Returns:
        Omega ratio
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    # Separate gains and losses relative to threshold
    excess_returns = returns_clean - threshold
    gains = excess_returns[excess_returns > 0]
    losses = excess_returns[excess_returns < 0]
    
    if len(losses) == 0:
        return float('inf') if len(gains) > 0 else 1.0
    
    if len(gains) == 0:
        return 0.0
    
    omega = np.sum(gains) / abs(np.sum(losses))
    
    return float(omega)


def calculate_var(returns: Union[np.ndarray, pd.Series],
                 confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: Returns series
        confidence_level: Confidence level (default 5%)
        
    Returns:
        VaR value (positive number representing loss)
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    var = -np.percentile(returns_clean, confidence_level * 100)
    
    return float(var)


def calculate_cvar(returns: Union[np.ndarray, pd.Series],
                  confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
    
    Args:
        returns: Returns series
        confidence_level: Confidence level (default 5%)
        
    Returns:
        CVaR value (positive number representing expected loss)
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    # Calculate VaR threshold
    var_threshold = -np.percentile(returns_clean, confidence_level * 100)
    
    # Calculate expected value of losses beyond VaR
    tail_losses = returns_clean[returns_clean <= -var_threshold]
    
    if len(tail_losses) == 0:
        return var_threshold
    
    cvar = -np.mean(tail_losses)
    
    return float(cvar)


def calculate_win_rate(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate win rate (percentage of positive returns)
    
    Args:
        returns: Returns series
        
    Returns:
        Win rate as decimal (0.0 to 1.0)
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    win_rate = np.sum(returns_clean > 0) / len(returns_clean)
    
    return float(win_rate)


def calculate_profit_factor(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate profit factor (gross profits / gross losses)
    
    Args:
        returns: Returns series
        
    Returns:
        Profit factor
    """
    
    returns_array = np.asarray(returns)
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        return 0.0
    
    gross_profits = np.sum(returns_clean[returns_clean > 0])
    gross_losses = abs(np.sum(returns_clean[returns_clean < 0]))
    
    if gross_losses == 0:
        return float('inf') if gross_profits > 0 else 0.0
    
    profit_factor = gross_profits / gross_losses
    
    return float(profit_factor)

# ============================================
# Comprehensive Trading Metrics Calculator
# ============================================

class TradingMetricsCalculator:
    """
    Comprehensive calculator for trading performance metrics
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 periods_per_year: int = 252,
                 confidence_level: float = 0.05):
        """
        Initialize calculator with default parameters
        
        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            confidence_level: VaR confidence level
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.confidence_level = confidence_level
    
    @time_it("comprehensive_metrics_calculation")
    def calculate_all_metrics(self, 
                             returns: Union[np.ndarray, pd.Series],
                             benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive set of trading metrics
        
        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns
            
        Returns:
            Dictionary of all calculated metrics
        """
        
        metrics = {}
        
        try:
            # Basic return metrics
            metrics['total_return'] = float(np.prod(1 + np.asarray(returns)) - 1)
            metrics['annualized_return'] = calculate_annualized_return(returns, self.periods_per_year)
            metrics['annualized_volatility'] = calculate_annualized_volatility(returns, self.periods_per_year)
            
            # Risk-adjusted metrics
            metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, self.risk_free_rate, self.periods_per_year)
            metrics['sortino_ratio'] = calculate_sortino_ratio(returns, self.risk_free_rate, 0.0, self.periods_per_year)
            
            # Drawdown metrics
            dd_metrics = calculate_max_drawdown(returns)
            metrics.update({f'drawdown_{k}': v for k, v in dd_metrics.items()})
            
            # Additional risk metrics
            metrics['calmar_ratio'] = calculate_calmar_ratio(returns, self.periods_per_year)
            metrics['omega_ratio'] = calculate_omega_ratio(returns, 0.0)
            metrics['var'] = calculate_var(returns, self.confidence_level)
            metrics['cvar'] = calculate_cvar(returns, self.confidence_level)
            
            # Win/Loss metrics
            metrics['win_rate'] = calculate_win_rate(returns)
            metrics['profit_factor'] = calculate_profit_factor(returns)
            
            # Benchmark comparison metrics
            if benchmark_returns is not None:
                metrics['information_ratio'] = calculate_information_ratio(
                    returns, benchmark_returns, self.periods_per_year
                )
                
                # Alpha and Beta (simplified)
                returns_array = np.asarray(returns)
                benchmark_array = np.asarray(benchmark_returns)
                
                # Align lengths
                min_length = min(len(returns_array), len(benchmark_array))
                if min_length > 0:
                    returns_aligned = returns_array[-min_length:]
                    benchmark_aligned = benchmark_array[-min_length:]
                    
                    # Remove NaN values
                    mask = ~(np.isnan(returns_aligned) | np.isnan(benchmark_aligned))
                    if np.sum(mask) > 1:
                        returns_clean = returns_aligned[mask]
                        benchmark_clean = benchmark_aligned[mask]
                        
                        # Beta calculation
                        covariance = np.cov(returns_clean, benchmark_clean)[0, 1]
                        benchmark_variance = np.var(benchmark_clean, ddof=1)
                        
                        if benchmark_variance > 0:
                            beta = covariance / benchmark_variance
                            metrics['beta'] = float(beta)
                            
                            # Alpha calculation
                            benchmark_return = calculate_annualized_return(benchmark_clean, self.periods_per_year)
                            strategy_return = calculate_annualized_return(returns_clean, self.periods_per_year)
                            alpha = strategy_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
                            metrics['alpha'] = float(alpha)
            
            # Additional descriptive statistics
            returns_clean = np.asarray(returns)[~np.isnan(np.asarray(returns))]
            if len(returns_clean) > 0:
                metrics['skewness'] = float(pd.Series(returns_clean).skew())
                metrics['kurtosis'] = float(pd.Series(returns_clean).kurtosis())
                metrics['num_periods'] = len(returns_clean)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return basic metrics even if advanced ones fail
            metrics = {
                'total_return': float(np.prod(1 + np.asarray(returns)) - 1) if len(returns) > 0 else 0.0,
                'error': str(e)
            }
        
        return metrics
    
    def format_metrics_report(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics into a readable report
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Formatted report string
        """
        
        report = []
        report.append("=" * 50)
        report.append("TRADING PERFORMANCE METRICS REPORT")
        report.append("=" * 50)
        
        # Return metrics
        if 'total_return' in metrics:
            report.append(f"Total Return: {metrics['total_return']:.2%}")
        if 'annualized_return' in metrics:
            report.append(f"Annualized Return: {metrics['annualized_return']:.2%}")
        if 'annualized_volatility' in metrics:
            report.append(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
        
        report.append("")
        
        # Risk-adjusted metrics
        if 'sharpe_ratio' in metrics:
            report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        if 'sortino_ratio' in metrics:
            report.append(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        if 'calmar_ratio' in metrics:
            report.append(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        if 'information_ratio' in metrics:
            report.append(f"Information Ratio: {metrics['information_ratio']:.3f}")
        
        report.append("")
        
        # Risk metrics
        if 'drawdown_max_drawdown' in metrics:
            report.append(f"Maximum Drawdown: {metrics['drawdown_max_drawdown']:.2%}")
        if 'var' in metrics:
            report.append(f"VaR ({self.confidence_level:.0%}): {metrics['var']:.2%}")
        if 'cvar' in metrics:
            report.append(f"CVaR ({self.confidence_level:.0%}): {metrics['cvar']:.2%}")
        
        report.append("")
        
        # Win/Loss metrics
        if 'win_rate' in metrics:
            report.append(f"Win Rate: {metrics['win_rate']:.1%}")
        if 'profit_factor' in metrics:
            report.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Alpha/Beta if available
        if 'alpha' in metrics and 'beta' in metrics:
            report.append("")
            report.append(f"Alpha: {metrics['alpha']:.2%}")
            report.append(f"Beta: {metrics['beta']:.3f}")
        
        return "\n".join(report)

# ============================================
# Utility Functions
# ============================================

def quick_performance_summary(returns: Union[np.ndarray, pd.Series],
                            benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None) -> str:
    """
    Quick utility function to get performance summary
    
    Args:
        returns: Strategy returns
        benchmark_returns: Optional benchmark returns
        
    Returns:
        Formatted performance summary
    """
    
    calculator = TradingMetricsCalculator()
    metrics = calculator.calculate_all_metrics(returns, benchmark_returns)
    return calculator.format_metrics_report(metrics)


# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Trading Metrics")
    
    # Generate sample data
    np.random.seed(42)
    n_periods = 252 * 2  # 2 years of daily data
    
    # Strategy returns (with some drift and volatility)
    strategy_returns = np.random.normal(0.0008, 0.016, n_periods)  # ~20% annual return, 25% volatility
    
    # Benchmark returns (slightly lower return, lower volatility)
    benchmark_returns = np.random.normal(0.0003, 0.012, n_periods)  # ~8% annual return, 19% volatility
    
    print("Sample Trading Metrics Calculation:")
    print("=" * 40)
    
    # Test individual metrics
    print(f"Sharpe Ratio: {calculate_sharpe_ratio(strategy_returns):.3f}")
    print(f"Sortino Ratio: {calculate_sortino_ratio(strategy_returns):.3f}")
    print(f"Calmar Ratio: {calculate_calmar_ratio(strategy_returns):.3f}")
    
    max_dd = calculate_max_drawdown(strategy_returns)
    print(f"Max Drawdown: {max_dd['max_drawdown']:.2%}")
    print(f"Max DD Duration: {max_dd['max_drawdown_duration']} days")
    
    print(f"Win Rate: {calculate_win_rate(strategy_returns):.1%}")
    print(f"Profit Factor: {calculate_profit_factor(strategy_returns):.2f}")
    
    # Test comprehensive calculator
    print("\n" + "=" * 40)
    print("COMPREHENSIVE METRICS REPORT:")
    
    summary = quick_performance_summary(strategy_returns, benchmark_returns)
    print(summary)
    
    print("\nTrading metrics testing completed successfully!")
