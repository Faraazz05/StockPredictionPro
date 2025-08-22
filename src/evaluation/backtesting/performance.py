# ============================================
# StockPredictionPro - src/evaluation/backtesting/performance.py
# Comprehensive performance analytics for backtesting results
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from scipy import stats
import math

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it
from ..metrics.trading_metrics import (
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown,
    calculate_calmar_ratio, calculate_annualized_return, calculate_annualized_volatility
)

logger = get_logger('evaluation.backtesting.performance')

# ============================================
# Performance Data Structures
# ============================================

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Additional metrics
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_trade: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    
    # Time-based metrics
    trading_days: int = 0
    exposure_time: float = 0.0  # Percentage of time in market
    
    # Benchmark comparison
    alpha: Optional[float] = None
    beta: Optional[float] = None
    correlation: Optional[float] = None
    information_ratio: Optional[float] = None

@dataclass
class DrawdownPeriod:
    """Container for drawdown period information"""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    recovery_date: Optional[pd.Timestamp]
    peak_value: float
    trough_value: float
    drawdown_pct: float
    duration_days: int
    recovery_days: Optional[int]

@dataclass
class TradingPeriodStats:
    """Statistics for specific trading periods"""
    period: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    returns: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    trades: int
    win_rate: float

# ============================================
# Performance Analyzer
# ============================================

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtesting results.
    
    This class provides detailed analysis of trading strategy performance
    including returns, risk metrics, drawdown analysis, and benchmarking.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252):
        """
        Initialize performance analyzer
        
        Args:
            risk_free_rate: Annual risk-free rate for calculations
            periods_per_year: Number of trading periods per year
        """
        
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
        # Cached results
        self._equity_returns = None
        self._benchmark_returns = None
        self._metrics_cache = {}
    
    @time_it("performance_analysis")
    def analyze(self, backtest_results: Dict[str, Any], 
                benchmark_data: Optional[Union[pd.Series, np.ndarray]] = None) -> PerformanceMetrics:
        """
        Analyze backtest performance
        
        Args:
            backtest_results: Results from BacktestEngine
            benchmark_data: Optional benchmark returns for comparison
            
        Returns:
            PerformanceMetrics object with comprehensive analysis
        """
        
        if 'equity_curve' not in backtest_results or not backtest_results['equity_curve']:
            raise ValueError("Equity curve data not found in backtest results")
        
        # Extract data
        equity_curve = np.array(backtest_results['equity_curve'])
        trades = backtest_results.get('trades', [])
        initial_capital = backtest_results.get('summary', {}).get('initial_capital', equity_curve[0])
        
        # Calculate returns
        self._equity_returns = pd.Series(equity_curve).pct_change().dropna()
        
        # Calculate benchmark returns if provided
        if benchmark_data is not None:
            benchmark_data = np.array(benchmark_data)
            if len(benchmark_data) == len(equity_curve):
                self._benchmark_returns = pd.Series(benchmark_data).pct_change().dropna()
        
        # Calculate comprehensive metrics
        metrics = PerformanceMetrics()
        
        # Basic returns
        metrics.total_return = (equity_curve[-1] - initial_capital) / initial_capital
        metrics.annualized_return = calculate_annualized_return(self._equity_returns, self.periods_per_year)
        metrics.volatility = calculate_annualized_volatility(self._equity_returns, self.periods_per_year)
        
        # Risk-adjusted returns
        metrics.sharpe_ratio = calculate_sharpe_ratio(self._equity_returns, self.risk_free_rate, self.periods_per_year)
        metrics.sortino_ratio = calculate_sortino_ratio(self._equity_returns, self.risk_free_rate, periods_per_year=self.periods_per_year)
        
        # Drawdown analysis
        dd_info = calculate_max_drawdown(self._equity_returns)
        metrics.max_drawdown = dd_info['max_drawdown']
        metrics.max_drawdown_duration = dd_info['max_drawdown_duration']
        
        # Calmar ratio
        metrics.calmar_ratio = calculate_calmar_ratio(self._equity_returns, self.periods_per_year)
        
        # Risk metrics
        metrics.var_95, metrics.cvar_95 = self._calculate_var_cvar(self._equity_returns)
        
        # Trade analysis
        if trades:
            metrics = self._analyze_trades(metrics, trades)
        
        # Time-based metrics
        metrics.trading_days = len(equity_curve)
        metrics.exposure_time = self._calculate_exposure_time(backtest_results)
        
        # Benchmark comparison
        if self._benchmark_returns is not None:
            metrics = self._analyze_benchmark_comparison(metrics)
        
        return metrics
    
    def analyze_drawdowns(self, equity_curve: Union[List, np.ndarray, pd.Series]) -> List[DrawdownPeriod]:
        """
        Detailed drawdown analysis
        
        Args:
            equity_curve: Portfolio equity over time
            
        Returns:
            List of DrawdownPeriod objects
        """
        
        if isinstance(equity_curve, (list, np.ndarray)):
            equity_series = pd.Series(equity_curve)
        else:
            equity_series = equity_curve.copy()
        
        # Calculate running maximum (peaks)
        rolling_max = equity_series.expanding().max()
        
        # Calculate drawdown
        drawdowns = (equity_series - rolling_max) / rolling_max
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        current_period = None
        
        for i, (date, dd) in enumerate(drawdowns.items()):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                current_period = {
                    'start_idx': i,
                    'start_date': date,
                    'peak_value': rolling_max.iloc[i],
                    'min_dd': dd,
                    'min_dd_idx': i
                }
            
            elif dd < 0 and in_drawdown:
                # Continue drawdown - check if new minimum
                if dd < current_period['min_dd']:
                    current_period['min_dd'] = dd
                    current_period['min_dd_idx'] = i
            
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                
                # Calculate period statistics
                trough_value = equity_series.iloc[current_period['min_dd_idx']]
                duration = current_period['min_dd_idx'] - current_period['start_idx'] + 1
                
                # Find recovery date
                recovery_date = date if abs(dd) < 0.001 else None  # Approximately recovered
                recovery_days = i - current_period['min_dd_idx'] if recovery_date else None
                
                period = DrawdownPeriod(
                    start_date=current_period['start_date'],
                    end_date=equity_series.index[current_period['min_dd_idx']],
                    recovery_date=recovery_date,
                    peak_value=current_period['peak_value'],
                    trough_value=trough_value,
                    drawdown_pct=abs(current_period['min_dd']),
                    duration_days=duration,
                    recovery_days=recovery_days
                )
                
                drawdown_periods.append(period)
        
        # Handle ongoing drawdown
        if in_drawdown and current_period:
            trough_value = equity_series.iloc[current_period['min_dd_idx']]
            duration = current_period['min_dd_idx'] - current_period['start_idx'] + 1
            
            period = DrawdownPeriod(
                start_date=current_period['start_date'],
                end_date=equity_series.index[current_period['min_dd_idx']],
                recovery_date=None,
                peak_value=current_period['peak_value'],
                trough_value=trough_value,
                drawdown_pct=abs(current_period['min_dd']),
                duration_days=duration,
                recovery_days=None
            )
            
            drawdown_periods.append(period)
        
        return sorted(drawdown_periods, key=lambda x: x.drawdown_pct, reverse=True)
    
    def analyze_rolling_performance(self, equity_curve: Union[List, np.ndarray, pd.Series], 
                                  window_days: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics
        
        Args:
            equity_curve: Portfolio equity over time
            window_days: Rolling window size in days
            
        Returns:
            DataFrame with rolling metrics
        """
        
        if isinstance(equity_curve, (list, np.ndarray)):
            equity_series = pd.Series(equity_curve)
        else:
            equity_series = equity_curve.copy()
        
        returns = equity_series.pct_change().dropna()
        
        results = []
        
        for i in range(window_days, len(returns)):
            window_returns = returns.iloc[i-window_days:i]
            end_date = returns.index[i] if hasattr(returns.index, '__getitem__') else i
            
            # Calculate metrics for window
            total_return = (1 + window_returns).prod() - 1
            annualized_return = (1 + total_return) ** (self.periods_per_year / window_days) - 1
            volatility = window_returns.std() * np.sqrt(self.periods_per_year)
            sharpe = calculate_sharpe_ratio(window_returns, self.risk_free_rate, self.periods_per_year)
            
            # Drawdown for window
            window_equity = equity_series.iloc[i-window_days:i]
            window_dd = calculate_max_drawdown(window_returns)['max_drawdown']
            
            results.append({
                'end_date': end_date,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': window_dd
            })
        
        return pd.DataFrame(results)
    
    def analyze_monthly_returns(self, equity_curve: Union[List, np.ndarray, pd.Series], 
                               dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """
        Calculate monthly return matrix
        
        Args:
            equity_curve: Portfolio equity over time
            dates: Date index (if not provided, assumes daily data from start)
            
        Returns:
            DataFrame with monthly returns
        """
        
        if isinstance(equity_curve, (list, np.ndarray)):
            equity_series = pd.Series(equity_curve)
        else:
            equity_series = equity_curve.copy()
        
        if dates is not None:
            equity_series.index = dates
        elif not isinstance(equity_series.index, pd.DatetimeIndex):
            # Create daily date range
            start_date = '2020-01-01'  # Default start
            equity_series.index = pd.date_range(start_date, periods=len(equity_series), freq='D')
        
        # Calculate daily returns
        returns = equity_series.pct_change().dropna()
        
        # Resample to monthly returns
        monthly_returns = (1 + returns).resample('M').prod() - 1
        
        # Create matrix with years as rows and months as columns
        matrix_data = []
        
        for year in monthly_returns.index.year.unique():
            year_data = {'Year': year}
            year_returns = monthly_returns[monthly_returns.index.year == year]
            
            for month in range(1, 13):
                month_name = pd.Timestamp(2000, month, 1).strftime('%b')
                matching_returns = year_returns[year_returns.index.month == month]
                
                if len(matching_returns) > 0:
                    year_data[month_name] = matching_returns.iloc[0]
                else:
                    year_data[month_name] = np.nan
            
            # Calculate yearly return
            year_data['Annual'] = (1 + year_returns).prod() - 1
            
            matrix_data.append(year_data)
        
        return pd.DataFrame(matrix_data).set_index('Year')
    
    def compare_strategies(self, results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple strategy results
        
        Args:
            results_dict: Dictionary of {strategy_name: backtest_results}
            
        Returns:
            DataFrame comparing strategies
        """
        
        comparison_data = []
        
        for strategy_name, results in results_dict.items():
            try:
                metrics = self.analyze(results)
                
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Total Return': f"{metrics.total_return:.2%}",
                    'Annual Return': f"{metrics.annualized_return:.2%}",
                    'Volatility': f"{metrics.volatility:.2%}",
                    'Sharpe Ratio': f"{metrics.sharpe_ratio:.3f}",
                    'Sortino Ratio': f"{metrics.sortino_ratio:.3f}",
                    'Calmar Ratio': f"{metrics.calmar_ratio:.3f}",
                    'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                    'Win Rate': f"{metrics.win_rate:.1%}",
                    'Total Trades': metrics.total_trades,
                    'Profit Factor': f"{metrics.profit_factor:.2f}"
                })
                
            except Exception as e:
                logger.warning(f"Error analyzing strategy {strategy_name}: {e}")
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Error': str(e)
                })
        
        return pd.DataFrame(comparison_data)
    
    def generate_performance_report(self, metrics: PerformanceMetrics, 
                                  strategy_name: str = "Strategy") -> str:
        """
        Generate comprehensive performance report
        
        Args:
            metrics: PerformanceMetrics object
            strategy_name: Name of the strategy
            
        Returns:
            Formatted performance report
        """
        
        report = []
        report.append("=" * 60)
        report.append(f"PERFORMANCE REPORT - {strategy_name.upper()}")
        report.append("=" * 60)
        
        # Returns section
        report.append("\n📈 RETURNS")
        report.append("-" * 20)
        report.append(f"Total Return: {metrics.total_return:>20.2%}")
        report.append(f"Annualized Return: {metrics.annualized_return:>15.2%}")
        report.append(f"Volatility (Annual): {metrics.volatility:>13.2%}")
        
        # Risk-adjusted returns
        report.append("\n⚖️  RISK-ADJUSTED RETURNS")
        report.append("-" * 30)
        report.append(f"Sharpe Ratio: {metrics.sharpe_ratio:>21.3f}")
        report.append(f"Sortino Ratio: {metrics.sortino_ratio:>20.3f}")
        report.append(f"Calmar Ratio: {metrics.calmar_ratio:>21.3f}")
        
        # Risk metrics
        report.append("\n🛡️  RISK METRICS")
        report.append("-" * 20)
        report.append(f"Maximum Drawdown: {metrics.max_drawdown:>15.2%}")
        report.append(f"Max DD Duration: {metrics.max_drawdown_duration:>16d} days")
        report.append(f"Value at Risk (95%): {metrics.var_95:>12.2%}")
        report.append(f"Conditional VaR (95%): {metrics.cvar_95:>9.2%}")
        
        # Trading statistics
        report.append("\n📊 TRADING STATISTICS")
        report.append("-" * 25)
        report.append(f"Total Trades: {metrics.total_trades:>21d}")
        report.append(f"Winning Trades: {metrics.winning_trades:>19d}")
        report.append(f"Losing Trades: {metrics.losing_trades:>20d}")
        report.append(f"Win Rate: {metrics.win_rate:>25.1%}")
        report.append(f"Profit Factor: {metrics.profit_factor:>18.2f}")
        
        # Trade details
        if metrics.avg_trade != 0:
            report.append(f"Average Trade: {metrics.avg_trade:>19.2f}")
            report.append(f"Best Trade: {metrics.best_trade:>21.2f}")
            report.append(f"Worst Trade: {metrics.worst_trade:>20.2f}")
            
            if metrics.avg_winning_trade != 0:
                report.append(f"Avg Winning Trade: {metrics.avg_winning_trade:>14.2f}")
            if metrics.avg_losing_trade != 0:
                report.append(f"Avg Losing Trade: {metrics.avg_losing_trade:>15.2f}")
        
        # Time-based metrics
        report.append("\n⏱️  TIME METRICS")
        report.append("-" * 20)
        report.append(f"Trading Days: {metrics.trading_days:>21d}")
        report.append(f"Market Exposure: {metrics.exposure_time:>16.1%}")
        
        # Benchmark comparison
        if metrics.alpha is not None:
            report.append("\n📊 BENCHMARK COMPARISON")
            report.append("-" * 25)
            report.append(f"Alpha: {metrics.alpha:>29.4f}")
            report.append(f"Beta: {metrics.beta:>30.4f}")
            report.append(f"Correlation: {metrics.correlation:>20.4f}")
            
            if metrics.information_ratio is not None:
                report.append(f"Information Ratio: {metrics.information_ratio:>13.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def _analyze_trades(self, metrics: PerformanceMetrics, trades: List) -> PerformanceMetrics:
        """Analyze individual trades"""
        
        trade_pnls = [trade.pnl for trade in trades if trade.pnl is not None]
        
        if not trade_pnls:
            return metrics
        
        metrics.total_trades = len(trade_pnls)
        
        # Separate winning and losing trades
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        metrics.win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0
        
        # Trade statistics
        metrics.best_trade = max(trade_pnls) if trade_pnls else 0
        metrics.worst_trade = min(trade_pnls) if trade_pnls else 0
        metrics.avg_trade = np.mean(trade_pnls) if trade_pnls else 0
        
        if winning_trades:
            metrics.avg_winning_trade = np.mean(winning_trades)
        
        if losing_trades:
            metrics.avg_losing_trade = np.mean(losing_trades)
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        return metrics
    
    def _calculate_var_cvar(self, returns: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional VaR"""
        
        if len(returns) == 0:
            return 0.0, 0.0
        
        # VaR calculation
        var = -np.percentile(returns, (1 - confidence) * 100)
        
        # CVaR calculation (expected shortfall)
        threshold_returns = returns[returns <= -var]
        cvar = -np.mean(threshold_returns) if len(threshold_returns) > 0 else var
        
        return var, cvar
    
    def _calculate_exposure_time(self, backtest_results: Dict[str, Any]) -> float:
        """Calculate percentage of time in market"""
        
        portfolio_history = backtest_results.get('portfolio_history', [])
        
        if not portfolio_history:
            return 0.0
        
        periods_in_market = 0
        total_periods = len(portfolio_history)
        
        for state in portfolio_history:
            if state.positions:  # Has positions
                periods_in_market += 1
        
        return periods_in_market / total_periods if total_periods > 0 else 0.0
    
    def _analyze_benchmark_comparison(self, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """Analyze performance vs benchmark"""
        
        if self._benchmark_returns is None or len(self._benchmark_returns) == 0:
            return metrics
        
        # Align series
        min_length = min(len(self._equity_returns), len(self._benchmark_returns))
        strategy_returns = self._equity_returns.iloc[-min_length:]
        benchmark_returns = self._benchmark_returns.iloc[-min_length:]
        
        # Calculate beta
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance > 0:
            metrics.beta = covariance / benchmark_variance
            
            # Calculate alpha
            benchmark_return = calculate_annualized_return(benchmark_returns, self.periods_per_year)
            metrics.alpha = metrics.annualized_return - (self.risk_free_rate + metrics.beta * (benchmark_return - self.risk_free_rate))
        
        # Correlation
        metrics.correlation = np.corrcoef(strategy_returns, benchmark_returns)[0, 1]
        
        # Information ratio
        active_returns = strategy_returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(self.periods_per_year)
        
        if tracking_error > 0:
            metrics.information_ratio = np.mean(active_returns) * self.periods_per_year / tracking_error
        
        return metrics

# ============================================
# Performance Visualization
# ============================================

class PerformanceVisualizer:
    """
    Create visualizations for backtesting performance
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style to use
        """
        self.style = style
        plt.style.use(style)
        
    def plot_equity_curve(self, equity_curve: Union[List, np.ndarray, pd.Series],
                         benchmark_curve: Optional[Union[List, np.ndarray, pd.Series]] = None,
                         title: str = "Portfolio Equity Curve",
                         figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot equity curve with optional benchmark
        
        Args:
            equity_curve: Portfolio equity over time
            benchmark_curve: Optional benchmark equity curve
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity curve
        ax.plot(equity_curve, label='Portfolio', linewidth=2, color='blue')
        
        # Plot benchmark if provided
        if benchmark_curve is not None:
            ax.plot(benchmark_curve, label='Benchmark', linewidth=1, color='gray', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(self, equity_curve: Union[List, np.ndarray, pd.Series],
                     title: str = "Drawdown Analysis",
                     figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Plot drawdown curve
        
        Args:
            equity_curve: Portfolio equity over time
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        
        if isinstance(equity_curve, (list, np.ndarray)):
            equity_series = pd.Series(equity_curve)
        else:
            equity_series = equity_curve.copy()
        
        # Calculate drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot drawdown
        ax.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        ax.plot(drawdown, color='red', linewidth=1)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Drawdown')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax.grid(True, alpha=0.3)
        
        # Highlight maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax.annotate(f'Max DD: {max_dd_value:.2%}',
                   xy=(max_dd_idx, max_dd_value),
                   xytext=(max_dd_idx + len(drawdown) * 0.1, max_dd_value * 0.5),
                   arrowprops=dict(arrowstyle='->', color='black'),
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def plot_monthly_returns_heatmap(self, monthly_returns_df: pd.DataFrame,
                                   title: str = "Monthly Returns Heatmap",
                                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot monthly returns heatmap
        
        Args:
            monthly_returns_df: DataFrame from analyze_monthly_returns
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Exclude 'Annual' column for heatmap
        monthly_data = monthly_returns_df.drop('Annual', axis=1, errors='ignore')
        
        # Create heatmap
        sns.heatmap(monthly_data, 
                   annot=True, 
                   fmt='.2%', 
                   cmap='RdYlGn', 
                   center=0,
                   cbar_kws={'format': '%.1%'},
                   ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        plt.tight_layout()
        return fig
    
    def plot_rolling_metrics(self, rolling_df: pd.DataFrame,
                           metrics: List[str] = ['sharpe_ratio', 'max_drawdown'],
                           title: str = "Rolling Performance Metrics",
                           figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Plot rolling performance metrics
        
        Args:
            rolling_df: DataFrame from analyze_rolling_performance
            metrics: List of metrics to plot
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in rolling_df.columns:
                ax = axes[i]
                ax.plot(rolling_df.index, rolling_df[metric], linewidth=1.5)
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                
                # Format y-axis based on metric type
                if 'ratio' in metric.lower():
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                elif 'drawdown' in metric.lower() or 'return' in metric.lower():
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.tight_layout()
        return fig
    
    def plot_trade_analysis(self, trades: List, 
                          title: str = "Trade Analysis",
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot comprehensive trade analysis
        
        Args:
            trades: List of Trade objects
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        
        if not trades:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No trades to analyze', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
        
        # Extract trade data
        trade_pnls = [trade.pnl for trade in trades if trade.pnl is not None]
        trade_dates = [trade.timestamp for trade in trades if trade.pnl is not None]
        
        if not trade_pnls:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No trade P&L data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Cumulative P&L
        cumulative_pnl = np.cumsum(trade_pnls)
        ax1.plot(cumulative_pnl, marker='o', markersize=3, linewidth=1)
        ax1.set_title('Cumulative P&L')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Cumulative P&L')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. P&L Distribution
        ax2.hist(trade_pnls, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(x=np.mean(trade_pnls), color='blue', linestyle='-', alpha=0.7, label=f'Mean: {np.mean(trade_pnls):.2f}')
        ax2.set_title('P&L Distribution')
        ax2.set_xlabel('Trade P&L')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade P&L over time
        colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
        ax3.scatter(range(len(trade_pnls)), trade_pnls, c=colors, alpha=0.6)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title('Individual Trade P&L')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Trade P&L')
        ax3.grid(True, alpha=0.3)
        
        # 4. Win/Loss Statistics
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        win_rate = len(winning_trades) / len(trade_pnls) * 100
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        labels = ['Winning Trades', 'Losing Trades']
        sizes = [len(winning_trades), len(losing_trades)]
        colors = ['green', 'red']
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title(f'Win Rate: {win_rate:.1f}%\nAvg Win: {avg_win:.2f}\nAvg Loss: {avg_loss:.2f}')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

# ============================================
# Utility Functions
# ============================================

def quick_performance_analysis(backtest_results: Dict[str, Any],
                             benchmark_data: Optional[Union[pd.Series, np.ndarray]] = None,
                             strategy_name: str = "Strategy") -> str:
    """
    Quick utility function for performance analysis
    
    Args:
        backtest_results: Results from BacktestEngine
        benchmark_data: Optional benchmark data
        strategy_name: Name of strategy
        
    Returns:
        Formatted performance report
    """
    
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(backtest_results, benchmark_data)
    return analyzer.generate_performance_report(metrics, strategy_name)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Performance Analysis")
    
    # Generate sample backtest results for testing
    np.random.seed(42)
    n_days = 500
    
    # Create synthetic equity curve with some realistic characteristics
    daily_returns = np.random.normal(0.0008, 0.015, n_days)  # ~20% annual return, 24% volatility
    
    # Add some autocorrelation
    for i in range(1, n_days):
        daily_returns[i] += 0.1 * daily_returns[i-1]
    
    # Create equity curve
    initial_capital = 1000000
    equity_curve = [initial_capital]
    
    for ret in daily_returns:
        equity_curve.append(equity_curve[-1] * (1 + ret))
    
    # Create sample trades
    sample_trades = []
    for i in range(0, n_days, 20):  # Trade every 20 days
        pnl = np.random.normal(500, 2000)  # Random P&L
        
        class MockTrade:
            def __init__(self, pnl, timestamp):
                self.pnl = pnl
                self.timestamp = timestamp
        
        sample_trades.append(MockTrade(pnl, pd.Timestamp('2023-01-01') + pd.Timedelta(days=i)))
    
    # Create sample backtest results
    backtest_results = {
        'equity_curve': equity_curve,
        'trades': sample_trades,
        'summary': {
            'initial_capital': initial_capital,
            'final_value': equity_curve[-1]
        }
    }
    
    # Create benchmark data
    benchmark_returns = np.random.normal(0.0003, 0.012, n_days)  # ~8% annual, 19% volatility
    benchmark_curve = [initial_capital]
    for ret in benchmark_returns:
        benchmark_curve.append(benchmark_curve[-1] * (1 + ret))
    
    print(f"Generated sample backtest data: {n_days} days, {len(sample_trades)} trades")
    
    # Test performance analysis
    print("\n1. Testing Performance Analysis")
    
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(backtest_results, benchmark_curve)
    
    print("Key Performance Metrics:")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Annualized Return: {metrics.annualized_return:.2%}")
    print(f"Volatility: {metrics.volatility:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.1%}")
    
    # Test comprehensive report
    print("\n2. Testing Performance Report Generation")
    
    report = analyzer.generate_performance_report(metrics, "Sample Strategy")
    print(report)
    
    # Test drawdown analysis
    print("\n3. Testing Drawdown Analysis")
    
    drawdown_periods = analyzer.analyze_drawdowns(equity_curve)
    print(f"Found {len(drawdown_periods)} drawdown periods")
    
    if drawdown_periods:
        worst_dd = drawdown_periods[0]  # Sorted by severity
        print(f"Worst Drawdown:")
        print(f"  Period: {worst_dd.start_date} to {worst_dd.end_date}")
        print(f"  Drawdown: {worst_dd.drawdown_pct:.2%}")
        print(f"  Duration: {worst_dd.duration_days} days")
        if worst_dd.recovery_days:
            print(f"  Recovery: {worst_dd.recovery_days} days")
    
    # Test rolling performance
    print("\n4. Testing Rolling Performance Analysis")
    
    dates = pd.date_range('2023-01-01', periods=len(equity_curve), freq='D')
    equity_series = pd.Series(equity_curve, index=dates)
    
    rolling_perf = analyzer.analyze_rolling_performance(equity_series, window_days=63)  # ~3 months
    print(f"Rolling performance calculated for {len(rolling_perf)} periods")
    print("Sample rolling metrics (last 5 periods):")
    print(rolling_perf.tail().to_string())
    
    # Test monthly returns analysis
    print("\n5. Testing Monthly Returns Analysis")
    
    monthly_returns = analyzer.analyze_monthly_returns(equity_series)
    print("Monthly Returns Matrix (sample):")
    print(monthly_returns.head().to_string())
    
    # Test strategy comparison
    print("\n6. Testing Strategy Comparison")
    
    # Create second strategy results (slightly different)
    equity_curve2 = [initial_capital * 0.95]  # Start slightly lower
    for ret in daily_returns:
        # Slightly different performance
        modified_ret = ret * 0.9 + np.random.normal(0, 0.002)
        equity_curve2.append(equity_curve2[-1] * (1 + modified_ret))
    
    backtest_results2 = {
        'equity_curve': equity_curve2,
        'trades': sample_trades[:15],  # Fewer trades
        'summary': {
            'initial_capital': initial_capital * 0.95,
            'final_value': equity_curve2[-1]
        }
    }
    
    strategies = {
        'Aggressive Strategy': backtest_results,
        'Conservative Strategy': backtest_results2
    }
    
    comparison_df = analyzer.compare_strategies(strategies)
    print("Strategy Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Test quick analysis function
    print("\n7. Testing Quick Analysis Function")
    
    quick_report = quick_performance_analysis(
        backtest_results, 
        benchmark_curve, 
        "Quick Test Strategy"
    )
    
    print("Quick Analysis Report (first 500 chars):")
    print(quick_report[:500] + "...")
    
    # Test performance visualizer
    print("\n8. Testing Performance Visualization")
    
    try:
        visualizer = PerformanceVisualizer()
        
        # Test equity curve plot
        fig1 = visualizer.plot_equity_curve(equity_curve, benchmark_curve)
        print("✓ Equity curve plot created")
        plt.close(fig1)
        
        # Test drawdown plot
        fig2 = visualizer.plot_drawdown(equity_curve)
        print("✓ Drawdown plot created")
        plt.close(fig2)
        
        # Test monthly returns heatmap
        fig3 = visualizer.plot_monthly_returns_heatmap(monthly_returns)
        print("✓ Monthly returns heatmap created")
        plt.close(fig3)
        
        # Test rolling metrics plot
        fig4 = visualizer.plot_rolling_metrics(rolling_perf, ['sharpe_ratio', 'max_drawdown'])
        print("✓ Rolling metrics plot created")
        plt.close(fig4)
        
        # Test trade analysis plot
        fig5 = visualizer.plot_trade_analysis(sample_trades)
        print("✓ Trade analysis plot created")
        plt.close(fig5)
        
    except Exception as e:
        print(f"Visualization test failed (expected if no display): {e}")
    
    print("\nPerformance analysis testing completed successfully!")
