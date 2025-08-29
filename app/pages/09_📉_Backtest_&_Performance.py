"""
app/pages/09_⏳_Backtest_&_Performance.py

Advanced Backtesting & Performance Analytics for StockPredictionPro.
Comprehensive framework for evaluating trading strategies with professional-grade
backtesting engines, performance metrics, validation techniques, and reporting.

Integrates with:
- src/evaluation/backtesting/ (engine, performance, portfolio, risk_management, strategies)
- src/evaluation/metrics/ (classification, regression, trading, custom metrics)
- src/evaluation/validation/ (combinatorial_cv, purged_cv, time_series, walk_forward)
- src/evaluation/ (plots, reports)

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path
import traceback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your comprehensive evaluation system
try:
    # Backtesting modules
    from src.evaluation.backtesting.engine import BacktestingEngine
    from src.evaluation.backtesting.performance import PerformanceAnalyzer
    from src.evaluation.backtesting.portfolio import PortfolioBacktester
    from src.evaluation.backtesting.risk_management import RiskManager
    from src.evaluation.backtesting.strategies import StrategyManager
    
    # Metrics modules
    from src.evaluation.metrics.trading_metrics import TradingMetricsCalculator
    from src.evaluation.metrics.custom_metrics import CustomMetricsCalculator
    from src.evaluation.metrics.classification_metrics import ClassificationMetrics
    from src.evaluation.metrics.regression_metrics import RegressionMetrics
    
    # Validation modules
    from src.evaluation.validation.time_series import TimeSeriesSplitCV
    from src.evaluation.validation.walk_forward import WalkForwardValidation
    from src.evaluation.validation.purged_cv import PurgedCrossValidation
    from src.evaluation.validation.combinatorial_cv import CombinatorialPurgedCV
    
    # Reporting and plotting
    from src.evaluation.reports import ReportGenerator
    from src.evaluation.plots import PlotGenerator
    
    EVALUATION_MODULES_AVAILABLE = True
    
except ImportError as e:
    st.error(f"Evaluation modules not found: {e}")
    EVALUATION_MODULES_AVAILABLE = False

# Import app components
from app.components.filters import (
    filter_symbols, filter_date_range, filter_categorical,
    create_data_filter_panel, filter_numeric
)
from app.components.charts import render_line_chart
from app.components.metrics import (
    display_trading_metrics, create_metrics_grid,
    display_performance_summary
)
from app.components.tables import display_dataframe, create_download_button
from app.components.alerts import get_alert_manager
from app.styles.themes import apply_custom_theme

# ============================================
# BACKTESTING CONFIGURATION
# ============================================

BACKTESTING_STRATEGIES = {
    "Moving Average Crossover": {
        "description": "Classic MA crossover with customizable periods",
        "parameters": {"short_period": 20, "long_period": 50, "stop_loss": 0.05},
        "complexity": "Simple",
        "expected_trades": "Medium"
    },
    "RSI Mean Reversion": {
        "description": "RSI-based mean reversion with overbought/oversold signals",
        "parameters": {"rsi_period": 14, "oversold": 30, "overbought": 70, "hold_period": 5},
        "complexity": "Intermediate",
        "expected_trades": "High"
    },
    "Momentum Breakout": {
        "description": "Price momentum breakout strategy with volume confirmation",
        "parameters": {"lookback": 20, "breakout_threshold": 0.02, "volume_multiplier": 1.5},
        "complexity": "Advanced",
        "expected_trades": "Low"
    },
    "Bollinger Bands": {
        "description": "Bollinger Band mean reversion with dynamic bands",
        "parameters": {"period": 20, "std_dev": 2, "squeeze_threshold": 0.01},
        "complexity": "Intermediate",
        "expected_trades": "Medium"
    },
    "MACD Signal": {
        "description": "MACD crossover with histogram confirmation",
        "parameters": {"fast": 12, "slow": 26, "signal": 9, "histogram_threshold": 0},
        "complexity": "Intermediate", 
        "expected_trades": "Medium"
    },
    "Pairs Trading": {
        "description": "Statistical arbitrage between correlated assets",
        "parameters": {"lookback": 60, "entry_zscore": 2.0, "exit_zscore": 0.5},
        "complexity": "Advanced",
        "expected_trades": "Variable"
    },
    "Multi-Timeframe": {
        "description": "Multi-timeframe trend following strategy",
        "parameters": {"short_tf": "1D", "long_tf": "1W", "trend_strength": 0.6},
        "complexity": "Advanced",
        "expected_trades": "Low"
    }
}

VALIDATION_METHODS = {
    "Time Series Split": {
        "class": "TimeSeriesSplitCV",
        "description": "Standard time series cross-validation",
        "parameters": {"n_splits": 5, "gap": 30}
    },
    "Walk Forward": {
        "class": "WalkForwardValidation",
        "description": "Walk-forward analysis with rolling windows",
        "parameters": {"train_period": 252, "test_period": 63, "step_size": 21}
    },
    "Purged CV": {
        "class": "PurgedCrossValidation", 
        "description": "Cross-validation with purged training sets",
        "parameters": {"n_splits": 5, "purge_length": 10}
    },
    "Combinatorial Purged CV": {
        "class": "CombinatorialPurgedCV",
        "description": "Advanced combinatorial purged cross-validation",
        "parameters": {"n_splits": 10, "n_test_splits": 2, "purge_length": 5}
    }
}

PERFORMANCE_METRICS = {
    "Basic": ["Total Return", "Annualized Return", "Volatility", "Sharpe Ratio", "Max Drawdown"],
    "Advanced": ["Sortino Ratio", "Calmar Ratio", "Omega Ratio", "VaR", "CVaR"],
    "Trading": ["Win Rate", "Profit Factor", "Average Trade", "Max Consecutive Losses", "Recovery Factor"],
    "Risk": ["Beta", "Alpha", "Information Ratio", "Tracking Error", "Downside Deviation"]
}

# ============================================
# DATA LOADING & STRATEGY IMPLEMENTATION
# ============================================

def load_backtesting_data(symbols: List[str], start_date: date, end_date: date) -> pd.DataFrame:
    """Load comprehensive market data for backtesting"""
    np.random.seed(42)  # For consistent demo data
    
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)
    
    # Generate realistic market data with regime changes and correlation
    data = pd.DataFrame({'date': dates})
    
    # Create market regimes for more realistic backtesting
    regime_length = n_days // 3
    regimes = np.concatenate([
        np.full(regime_length, 0.0008),      # Bull market
        np.full(regime_length, -0.0005),     # Bear market
        np.full(n_days - 2 * regime_length, 0.0003)  # Sideways
    ])
    
    # Add volatility clustering
    volatility = np.random.uniform(0.015, 0.03, n_days)
    for i in range(1, n_days):
        volatility[i] = 0.8 * volatility[i-1] + 0.2 * np.random.uniform(0.01, 0.04)
    
    base_prices = {
        "AAPL": 175.0, "MSFT": 342.0, "GOOGL": 2650.0, "TSLA": 245.0, "NVDA": 485.0,
        "AMZN": 3380.0, "META": 310.0, "SPY": 450.0, "QQQ": 380.0, "IWM": 220.0
    }
    
    for symbol in symbols:
        base_price = base_prices.get(symbol, 150.0)
        
        # Generate correlated returns
        market_factor = np.random.normal(regimes, volatility * 0.7)
        idiosyncratic = np.random.normal(0, volatility * 0.3, n_days)
        returns = market_factor + idiosyncratic
        
        # Create price series
        prices = [base_price]
        for ret in returns:
            prices.append(max(prices[-1] * (1 + ret), 1.0))
        
        # Create OHLCV data
        close_prices = prices[:n_days]
        open_prices = [p * (1 + np.random.normal(0, 0.002)) for p in close_prices]
        high_prices = [max(o, c) * (1 + abs(np.random.normal(0, 0.01))) for o, c in zip(open_prices, close_prices)]
        low_prices = [min(o, c) * (1 - abs(np.random.normal(0, 0.01))) for o, c in zip(open_prices, close_prices)]
        
        data[f'{symbol}_open'] = open_prices
        data[f'{symbol}_high'] = high_prices
        data[f'{symbol}_low'] = low_prices
        data[f'{symbol}_close'] = close_prices
        data[f'{symbol}_volume'] = np.random.lognormal(14.5, 0.8, n_days).astype(int)
    
    return data

def implement_strategy_logic(data: pd.DataFrame, 
                           strategy_name: str, 
                           parameters: Dict[str, Any],
                           symbol: str) -> pd.DataFrame:
    """Implement trading strategy logic and generate signals"""
    signals = []
    
    close_col = f'{symbol}_close'
    volume_col = f'{symbol}_volume'
    
    if close_col not in data.columns:
        st.error(f"Price data not available for {symbol}")
        return pd.DataFrame()
    
    prices = data[close_col].values
    volumes = data[volume_col].values if volume_col in data.columns else np.ones(len(prices))
    
    if strategy_name == "Moving Average Crossover":
        short_ma = pd.Series(prices).rolling(parameters['short_period']).mean()
        long_ma = pd.Series(prices).rolling(parameters['long_period']).mean()
        
        for i in range(1, len(prices)):
            if i < parameters['long_period']:
                signal = 'Hold'
            elif short_ma.iloc[i] > long_ma.iloc[i] and short_ma.iloc[i-1] <= long_ma.iloc[i-1]:
                signal = 'Buy'
            elif short_ma.iloc[i] < long_ma.iloc[i] and short_ma.iloc[i-1] >= long_ma.iloc[i-1]:
                signal = 'Sell'
            else:
                signal = 'Hold'
            
            signals.append({
                'date': data['date'].iloc[i],
                'signal': signal,
                'price': prices[i],
                'volume': volumes[i],
                'short_ma': short_ma.iloc[i],
                'long_ma': long_ma.iloc[i],
                'strategy': strategy_name
            })
    
    elif strategy_name == "RSI Mean Reversion":
        # Calculate RSI
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=parameters['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=parameters['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        for i in range(parameters['rsi_period'], len(prices)):
            if rsi.iloc[i] < parameters['oversold']:
                signal = 'Buy'
            elif rsi.iloc[i] > parameters['overbought']:
                signal = 'Sell'
            else:
                signal = 'Hold'
            
            signals.append({
                'date': data['date'].iloc[i],
                'signal': signal,
                'price': prices[i],
                'volume': volumes[i],
                'rsi': rsi.iloc[i],
                'strategy': strategy_name
            })
    
    elif strategy_name == "Momentum Breakout":
        lookback = parameters['lookback']
        threshold = parameters['breakout_threshold']
        vol_multiplier = parameters['volume_multiplier']
        
        for i in range(lookback, len(prices)):
            price_change = (prices[i] - prices[i-lookback]) / prices[i-lookback]
            avg_volume = np.mean(volumes[i-lookback:i])
            volume_spike = volumes[i] > (avg_volume * vol_multiplier)
            
            if price_change > threshold and volume_spike:
                signal = 'Buy'
            elif price_change < -threshold and volume_spike:
                signal = 'Sell'
            else:
                signal = 'Hold'
            
            signals.append({
                'date': data['date'].iloc[i],
                'signal': signal,
                'price': prices[i],
                'volume': volumes[i],
                'price_change': price_change,
                'volume_ratio': volumes[i] / avg_volume,
                'strategy': strategy_name
            })
    
    elif strategy_name == "Bollinger Bands":
        period = parameters['period']
        std_dev = parameters['std_dev']
        
        sma = pd.Series(prices).rolling(period).mean()
        std = pd.Series(prices).rolling(period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        for i in range(period, len(prices)):
            if prices[i] <= lower_band.iloc[i]:
                signal = 'Buy'
            elif prices[i] >= upper_band.iloc[i]:
                signal = 'Sell'
            else:
                signal = 'Hold'
            
            signals.append({
                'date': data['date'].iloc[i],
                'signal': signal,
                'price': prices[i],
                'volume': volumes[i],
                'upper_band': upper_band.iloc[i],
                'lower_band': lower_band.iloc[i],
                'sma': sma.iloc[i],
                'strategy': strategy_name
            })
    
    return pd.DataFrame(signals)

def run_backtest_simulation(data: pd.DataFrame,
                          signals: pd.DataFrame,
                          initial_capital: float,
                          commission: float,
                          slippage: float) -> Dict[str, Any]:
    """Run comprehensive backtest simulation"""
    
    if signals.empty:
        return {
            'equity_curve': pd.Series([initial_capital]),
            'trades': pd.DataFrame(),
            'metrics': {},
            'drawdowns': pd.Series([0])
        }
    
    # Initialize portfolio tracking
    portfolio_values = [initial_capital]
    trades = []
    cash = initial_capital
    position = 0
    entry_price = 0
    entry_date = None
    
    for i, signal_row in signals.iterrows():
        current_price = signal_row['price']
        signal = signal_row['signal']
        date = signal_row['date']
        
        # Calculate transaction costs
        transaction_cost = commission + (slippage * current_price)
        
        # Process signals
        if signal == 'Buy' and position <= 0:
            # Close short position if any
            if position < 0:
                pnl = -position * (entry_price - current_price) - abs(position) * transaction_cost
                cash += -position * entry_price + pnl
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'type': 'Short',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': -position,
                    'pnl': pnl,
                    'return_pct': (entry_price - current_price) / entry_price
                })
                position = 0
            
            # Enter long position
            if cash > current_price + transaction_cost:
                quantity = int((cash * 0.95) / (current_price + transaction_cost))
                cost = quantity * (current_price + transaction_cost)
                cash -= cost
                position = quantity
                entry_price = current_price
                entry_date = date
        
        elif signal == 'Sell' and position >= 0:
            # Close long position if any
            if position > 0:
                pnl = position * (current_price - entry_price) - position * transaction_cost
                cash += position * current_price - position * transaction_cost
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'type': 'Long',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': position,
                    'pnl': pnl,
                    'return_pct': (current_price - entry_price) / entry_price
                })
                position = 0
            
            # Enter short position (if allowed)
            # For simplicity, we'll skip short selling in this demo
        
        # Calculate current portfolio value
        current_value = cash + (position * current_price if position > 0 else 0)
        portfolio_values.append(current_value)
    
    # Convert to pandas series
    portfolio_series = pd.Series(portfolio_values, index=[signals['date'].iloc[0] - timedelta(days=1)] + list(signals['date']))
    trades_df = pd.DataFrame(trades)
    
    # Calculate performance metrics
    returns = portfolio_series.pct_change().fillna(0)
    
    # Basic metrics
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
    volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
    sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
    
    # Drawdown calculation
    peak = portfolio_series.cummax()
    drawdown = (portfolio_series - peak) / peak
    max_drawdown = drawdown.min()
    
    # Trading metrics
    if not trades_df.empty:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df)
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losing_trades['pnl'].sum() != 0 else float('inf')
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    metrics = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Total Trades': len(trades_df),
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Final Value': portfolio_values[-1]
    }
    
    return {
        'equity_curve': portfolio_series,
        'trades': trades_df,
        'metrics': metrics,
        'drawdowns': drawdown,
        'signals': signals
    }

# ============================================
# VISUALIZATION ENGINE
# ============================================

def create_backtest_dashboard(backtest_results: Dict[str, Any], strategy_name: str) -> None:
    """Create comprehensive backtesting dashboard"""
    
    equity_curve = backtest_results['equity_curve']
    drawdowns = backtest_results['drawdowns']
    trades = backtest_results['trades']
    signals = backtest_results['signals']
    
    # Create multi-panel dashboard
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Portfolio Equity Curve',
            'Drawdown Analysis',
            'Trade Distribution',
            'Signal Analysis',
            'Monthly Returns',
            'Rolling Sharpe Ratio',
            'Trade Timeline',
            'Risk Metrics'
        ),
        specs=[
            [{"colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"colspan": 2}, None]
        ]
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=3)
        ), row=1, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns.values * 100,
            mode='lines',
            name='Drawdown %',
            fill='tonexty',
            line=dict(color='red', width=2)
        ), row=2, col=1
    )
    
    # Trade distribution
    if not trades.empty and 'pnl' in trades.columns:
        fig.add_trace(
            go.Histogram(
                x=trades['pnl'],
                name='Trade P&L',
                nbinsx=20,
                opacity=0.7
            ), row=2, col=2
        )
    
    # Signal analysis
    if not signals.empty:
        buy_signals = signals[signals['signal'] == 'Buy']
        sell_signals = signals[signals['signal'] == 'Sell']
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['date'],
                    y=buy_signals['price'],
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(symbol='triangle-up', size=8, color='green')
                ), row=3, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['date'],
                    y=sell_signals['price'],
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(symbol='triangle-down', size=8, color='red')
                ), row=3, col=1
            )
    
    # Rolling Sharpe ratio
    returns = equity_curve.pct_change().fillna(0)
    rolling_sharpe = returns.rolling(window=30).mean() / returns.rolling(window=30).std() * np.sqrt(252)
    
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode='lines',
            name='30-Day Rolling Sharpe',
            line=dict(color='purple', width=2)
        ), row=3, col=2
    )
    
    # Trade timeline
    if not trades.empty:
        colors = ['green' if pnl > 0 else 'red' for pnl in trades['pnl']]
        fig.add_trace(
            go.Bar(
                x=trades['exit_date'],
                y=trades['pnl'],
                name='Trade P&L',
                marker_color=colors,
                opacity=0.7
            ), row=4, col=1
        )
    
    fig.update_layout(
        height=1200,
        title=f'{strategy_name} - Comprehensive Backtest Analysis',
        template='plotly_white',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# MAIN PAGE FUNCTION
# ============================================

def main():
    """Main backtest & performance page function"""
    
    # Apply theme
    apply_custom_theme("financial")
    
    # Page header
    st.title("⏳ Advanced Backtesting & Performance Analytics")
    st.markdown("Professional-grade strategy backtesting with comprehensive performance analysis")
    
    if not EVALUATION_MODULES_AVAILABLE:
        st.warning("⚠️ **Evaluation modules not fully available.** Using comprehensive fallback implementations.")
    
    st.markdown("---")
    
    # ============================================
    # BACKTESTING CONFIGURATION
    # ============================================
    
    st.subheader("🛠️ Backtesting Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Symbol and strategy selection
        available_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "SPY", "QQQ", "IWM"]
        selected_symbol = filter_categorical(
            "Select Symbol",
            available_symbols,
            multi=False,
            key="backtest_symbol"
        )
        
        selected_strategy = filter_categorical(
            "Trading Strategy",
            list(BACKTESTING_STRATEGIES.keys()),
            multi=False,
            key="backtest_strategy"
        )
        
        # Date range
        start_date, end_date = filter_date_range(
            default_days=252,  # 1 trading year
            key="backtest_dates"
        )
    
    with col2:
        # Portfolio settings
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000
        )
        
        commission = st.number_input(
            "Commission per Trade (%)",
            min_value=0.0,
            max_value=2.0,
            value=0.1,
            step=0.01,
            help="Transaction cost as percentage of trade value"
        )
        
        slippage = st.number_input(
            "Slippage per Trade (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Market impact cost as percentage"
        )
    
    if not selected_symbol or not selected_strategy:
        st.warning("⚠️ Please select both symbol and strategy")
        return
    
    # Strategy information
    strategy_info = BACKTESTING_STRATEGIES[selected_strategy]
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Strategy:** {strategy_info['description']}")
    with col2:
        st.info(f"**Complexity:** {strategy_info['complexity']} | **Expected Trades:** {strategy_info['expected_trades']}")
    
    # Strategy parameters customization
    with st.expander("⚙️ Strategy Parameters", expanded=False):
        strategy_params = strategy_info['parameters'].copy()
        
        st.write(f"**{selected_strategy} Parameters:**")
        
        if selected_strategy == "Moving Average Crossover":
            strategy_params['short_period'] = st.slider("Short MA Period", 5, 50, strategy_params['short_period'])
            strategy_params['long_period'] = st.slider("Long MA Period", 20, 200, strategy_params['long_period'])
            strategy_params['stop_loss'] = st.slider("Stop Loss (%)", 0.01, 0.20, strategy_params['stop_loss'], 0.01)
        
        elif selected_strategy == "RSI Mean Reversion":
            strategy_params['rsi_period'] = st.slider("RSI Period", 5, 50, strategy_params['rsi_period'])
            strategy_params['oversold'] = st.slider("Oversold Level", 10, 40, strategy_params['oversold'])
            strategy_params['overbought'] = st.slider("Overbought Level", 60, 90, strategy_params['overbought'])
            strategy_params['hold_period'] = st.slider("Hold Period", 1, 20, strategy_params['hold_period'])
        
        elif selected_strategy == "Momentum Breakout":
            strategy_params['lookback'] = st.slider("Lookback Period", 10, 60, strategy_params['lookback'])
            strategy_params['breakout_threshold'] = st.slider("Breakout Threshold (%)", 0.005, 0.10, strategy_params['breakout_threshold'], 0.005)
            strategy_params['volume_multiplier'] = st.slider("Volume Multiplier", 1.0, 5.0, strategy_params['volume_multiplier'], 0.1)
        
        elif selected_strategy == "Bollinger Bands":
            strategy_params['period'] = st.slider("BB Period", 10, 50, strategy_params['period'])
            strategy_params['std_dev'] = st.slider("Standard Deviations", 1.0, 3.0, strategy_params['std_dev'], 0.1)
            strategy_params['squeeze_threshold'] = st.slider("Squeeze Threshold", 0.005, 0.05, strategy_params['squeeze_threshold'], 0.005)
    
    # Validation method selection
    with st.expander("🔬 Validation & Testing Options", expanded=False):
        validation_method = st.selectbox(
            "Validation Method",
            list(VALIDATION_METHODS.keys()),
            help="Choose validation method for robust backtesting"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            out_of_sample_pct = st.slider("Out-of-Sample %", 10, 50, 20)
            walk_forward = st.checkbox("Walk-Forward Analysis", value=False)
        with col2:
            monte_carlo = st.checkbox("Monte Carlo Simulation", value=False)
            mc_iterations = st.number_input("MC Iterations", 100, 10000, 1000) if monte_carlo else 1000
    
    st.markdown("---")
    
    # ============================================
    # BACKTESTING EXECUTION
    # ============================================
    
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Loading data and running backtest..."):
            
            # Load market data
            market_data = load_backtesting_data([selected_symbol], start_date, end_date)
            
            if market_data.empty:
                st.error("❌ No market data available")
                return
            
            # Generate trading signals
            signals = implement_strategy_logic(
                market_data, 
                selected_strategy, 
                strategy_params,
                selected_symbol
            )
            
            if signals.empty:
                st.error("❌ No trading signals generated")
                return
            
            # Run backtest simulation
            backtest_results = run_backtest_simulation(
                market_data,
                signals,
                initial_capital,
                commission / 100,
                slippage / 100
            )
            
            # Store results in session state
            st.session_state.backtest_results = backtest_results
            st.session_state.backtest_strategy = selected_strategy
            st.session_state.backtest_symbol = selected_symbol
            st.session_state.backtest_params = strategy_params
        
        st.success("✅ Backtest completed successfully!")
    
    # ============================================
    # RESULTS ANALYSIS
    # ============================================
    
    if hasattr(st.session_state, 'backtest_results'):
        st.markdown("---")
        st.subheader("📊 Backtest Results")
        
        results = st.session_state.backtest_results
        strategy_name = st.session_state.backtest_strategy
        symbol = st.session_state.backtest_symbol
        
        # Performance summary
        metrics = results['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics['Total Return']:.2%}",
                f"{metrics['Total Return']*100:+.1f}%"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['Sharpe Ratio']:.3f}",
                help="Risk-adjusted return metric"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics['Max Drawdown']:.2%}",
                help="Maximum peak-to-trough decline"
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{metrics['Win Rate']:.1%}",
                help="Percentage of profitable trades"
            )
        
        # Comprehensive dashboard
        st.subheader("📈 Performance Dashboard")
        create_backtest_dashboard(results, f"{strategy_name} on {symbol}")
        
        # Detailed metrics
        st.subheader("📋 Detailed Performance Metrics")
        
        # Create metrics categories
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Return Metrics**")
            st.write(f"• Total Return: {metrics['Total Return']:.2%}")
            st.write(f"• Annualized Return: {metrics['Annualized Return']:.2%}")
            st.write(f"• Final Portfolio Value: ${metrics['Final Value']:,.0f}")
            
            st.write("**Risk Metrics**")
            st.write(f"• Volatility: {metrics['Volatility']:.2%}")
            st.write(f"• Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
            st.write(f"• Max Drawdown: {metrics['Max Drawdown']:.2%}")
        
        with col2:
            st.write("**Trading Metrics**")
            st.write(f"• Total Trades: {metrics['Total Trades']}")
            st.write(f"• Win Rate: {metrics['Win Rate']:.1%}")
            st.write(f"• Profit Factor: {metrics['Profit Factor']:.2f}")
            st.write(f"• Average Win: ${metrics['Average Win']:.2f}")
            st.write(f"• Average Loss: ${metrics['Average Loss']:.2f}")
        
        # Performance assessment
        st.subheader("🎯 Strategy Assessment")
        
        total_return = metrics['Total Return']
        sharpe_ratio = metrics['Sharpe Ratio']
        max_drawdown = metrics['Max Drawdown']
        win_rate = metrics['Win Rate']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if total_return > 0.15:
                st.success("🎉 **Excellent Returns** - Strategy significantly outperformed")
            elif total_return > 0.05:
                st.info("✅ **Good Performance** - Strategy delivered solid returns")
            elif total_return > -0.05:
                st.warning("⚠️ **Modest Performance** - Strategy roughly broke even")
            else:
                st.error("❌ **Poor Performance** - Strategy underperformed")
        
        with col2:
            if sharpe_ratio > 1.5:
                st.success("🏆 **Outstanding Risk-Adjusted Returns**")
            elif sharpe_ratio > 1.0:
                st.info("👍 **Good Risk-Adjusted Returns**")
            elif sharpe_ratio > 0.5:
                st.warning("⚠️ **Moderate Risk-Adjusted Returns**")
            else:
                st.error("🚨 **Poor Risk-Adjusted Returns**")
        
        with col3:
            if abs(max_drawdown) < 0.05:
                st.success("💎 **Excellent Drawdown Control**")
            elif abs(max_drawdown) < 0.15:
                st.info("✅ **Good Drawdown Management**")
            elif abs(max_drawdown) < 0.25:
                st.warning("⚠️ **Moderate Drawdowns**")
            else:
                st.error("🚨 **High Drawdown Risk**")
        
        # Trade analysis
        if not results['trades'].empty:
            st.subheader("💼 Trade Analysis")
            
            trades_df = results['trades']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Recent Trades**")
                recent_trades = trades_df.tail(10).copy()
                recent_trades['entry_date'] = pd.to_datetime(recent_trades['entry_date']).dt.strftime('%Y-%m-%d')
                recent_trades['exit_date'] = pd.to_datetime(recent_trades['exit_date']).dt.strftime('%Y-%m-%d')
                recent_trades['pnl'] = recent_trades['pnl'].apply(lambda x: f"${x:.2f}")
                recent_trades['return_pct'] = recent_trades['return_pct'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(
                    recent_trades[['entry_date', 'exit_date', 'type', 'pnl', 'return_pct']],
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                st.write("**Trade Statistics**")
                profitable_trades = trades_df[trades_df['pnl'] > 0]
                losing_trades = trades_df[trades_df['pnl'] < 0]
                
                st.write(f"• Profitable Trades: {len(profitable_trades)}")
                st.write(f"• Losing Trades: {len(losing_trades)}")
                
                if len(profitable_trades) > 0:
                    st.write(f"• Best Trade: ${profitable_trades['pnl'].max():.2f}")
                    st.write(f"• Avg Profitable Trade: ${profitable_trades['pnl'].mean():.2f}")
                
                if len(losing_trades) > 0:
                    st.write(f"• Worst Trade: ${losing_trades['pnl'].min():.2f}")
                    st.write(f"• Avg Losing Trade: ${losing_trades['pnl'].mean():.2f}")
        
        # ============================================
        # EXPORT & REPORTING
        # ============================================
        
        st.subheader("📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export equity curve
            equity_data = pd.DataFrame({
                'Date': results['equity_curve'].index,
                'Portfolio_Value': results['equity_curve'].values,
                'Drawdown': results['drawdowns'].values
            })
            
            create_download_button(
                equity_data,
                f"{symbol}_{strategy_name}_equity_curve.csv",
                "📈 Download Equity Curve",
                key="equity_download"
            )
        
        with col2:
            # Export trades
            if not results['trades'].empty:
                create_download_button(
                    results['trades'],
                    f"{symbol}_{strategy_name}_trades.csv",
                    "💼 Download Trades",
                    key="trades_download"
                )
        
        with col3:
            # Export comprehensive report
            report_data = pd.DataFrame([{
                'Strategy': strategy_name,
                'Symbol': symbol,
                'Start_Date': start_date,
                'End_Date': end_date,
                'Initial_Capital': initial_capital,
                'Final_Value': metrics['Final Value'],
                'Total_Return': metrics['Total Return'],
                'Annualized_Return': metrics['Annualized Return'],
                'Sharpe_Ratio': metrics['Sharpe Ratio'],
                'Max_Drawdown': metrics['Max Drawdown'],
                'Win_Rate': metrics['Win Rate'],
                'Total_Trades': metrics['Total Trades'],
                'Profit_Factor': metrics['Profit Factor'],
                'Parameters': str(st.session_state.backtest_params),
                'Backtest_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }])
            
            create_download_button(
                report_data,
                f"{symbol}_{strategy_name}_report.csv",
                "📋 Download Full Report",
                key="report_download"
            )
    
    # ============================================
    # ALERTS & RECOMMENDATIONS
    # ============================================
    
    st.markdown("---")
    st.subheader("🚨 Backtest Alerts & Recommendations")
    
    # Generate backtest alerts
    alert_manager = get_alert_manager()
    
    if hasattr(st.session_state, 'backtest_results'):
        results = st.session_state.backtest_results
        metrics = results['metrics']
        
        # Performance alerts
        if metrics['Total Return'] > 0.2:
            alert_manager.add_alert(
                f"🎉 Outstanding performance: {metrics['Total Return']:.1%} total return achieved",
                "backtest_performance",
                "success",
                selected_strategy
            )
        elif metrics['Total Return'] < -0.15:
            alert_manager.add_alert(
                f"⚠️ Poor performance: {metrics['Total Return']:.1%} loss - strategy needs optimization",
                "backtest_performance",
                "warning",
                selected_strategy
            )
        
        # Risk alerts
        if abs(metrics['Max Drawdown']) > 0.25:
            alert_manager.add_alert(
                f"🚨 High drawdown risk: {metrics['Max Drawdown']:.1%} maximum drawdown",
                "risk_management",
                "error",
                selected_strategy
            )
        
        # Trading efficiency alerts
        if metrics['Win Rate'] < 0.3:
            alert_manager.add_alert(
                f"⚠️ Low win rate: {metrics['Win Rate']:.1%} - consider refining entry/exit rules",
                "trading_efficiency",
                "warning",
                selected_strategy
            )
        elif metrics['Win Rate'] > 0.7:
            alert_manager.add_alert(
                f"✅ High win rate: {metrics['Win Rate']:.1%} - excellent strategy performance",
                "trading_efficiency",
                "success",
                selected_strategy
            )
        
        # Sharpe ratio alerts
        if metrics['Sharpe Ratio'] > 2.0:
            alert_manager.add_alert(
                f"🏆 Exceptional risk-adjusted returns: Sharpe ratio of {metrics['Sharpe Ratio']:.2f}",
                "risk_adjusted_performance",
                "success",
                selected_strategy
            )
    
    # Display alerts
    if alert_manager.alerts:
        for alert in alert_manager.alerts[-6:]:
            if alert.level.value == "success":
                st.success(f"✅ {alert}")
            elif alert.level.value == "warning":
                st.warning(f"⚠️ {alert}")
            elif alert.level.value == "error":
                st.error(f"❌ {alert}")
            else:
                st.info(f"ℹ️ {alert}")
    else:
        st.info("✅ No critical alerts - backtest parameters within normal ranges")
    
    # ============================================
    # FOOTER
    # ============================================
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        ⏳ Advanced Backtesting & Performance Analytics | 
        Professional Strategy Evaluation | 
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE EXECUTION
# ============================================

if __name__ == "__main__":
    main()
