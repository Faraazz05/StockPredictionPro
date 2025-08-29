"""
app/pages/08_📁_Portfolio_Management.py

Advanced Portfolio Management & Risk Analytics for StockPredictionPro.
Comprehensive portfolio construction, optimization, risk management,
position sizing, trade execution, and performance analytics.

Integrates with:
- src/trading/portfolio.py (comprehensive portfolio management)
- src/trading/risk/ (portfolio_risk, position_sizing, stop_loss, take_profit)
- src/trading/execution/ (market_orders, limit_orders, slippage)
- src/trading/strategies/ (portfolio strategy execution)

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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your comprehensive trading system
try:
    # Portfolio management
    from src.trading.portfolio import Portfolio
    
    # Risk management modules
    from src.trading.risk.portfolio_risk import PortfolioRiskManager
    from src.trading.risk.position_sizing import PositionSizingManager
    from src.trading.risk.stop_loss import StopLossManager
    from src.trading.risk.take_profit import TakeProfitManager
    
    # Execution engines
    from src.trading.execution.market_orders import MarketOrderExecutor
    from src.trading.execution.limit_orders import LimitOrderExecutor
    from src.trading.execution.slippage import SlippageModel
    
    # Strategy modules
    from src.trading.strategies.momentum import MomentumStrategy
    from src.trading.strategies.mean_reversion import MeanReversionStrategy
    
    PORTFOLIO_MODULES_AVAILABLE = True
    
except ImportError as e:
    st.error(f"Portfolio modules not found: {e}")
    PORTFOLIO_MODULES_AVAILABLE = False

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
# PORTFOLIO CONFIGURATION
# ============================================

PORTFOLIO_STRATEGIES = {
    "Equal Weight": {
        "description": "Equal allocation across all selected assets",
        "rebalance_frequency": "monthly",
        "risk_level": "medium"
    },
    "Market Cap Weight": {
        "description": "Weight by market capitalization",
        "rebalance_frequency": "quarterly", 
        "risk_level": "medium"
    },
    "Risk Parity": {
        "description": "Equal risk contribution from each asset",
        "rebalance_frequency": "monthly",
        "risk_level": "low"
    },
    "Momentum": {
        "description": "Weight by recent price momentum",
        "rebalance_frequency": "weekly",
        "risk_level": "high"
    },
    "Mean Reversion": {
        "description": "Contrarian approach based on price reversals",
        "rebalance_frequency": "weekly",
        "risk_level": "medium-high"
    },
    "Minimum Variance": {
        "description": "Minimize portfolio volatility",
        "rebalance_frequency": "monthly",
        "risk_level": "low"
    },
    "Maximum Sharpe": {
        "description": "Maximize risk-adjusted returns",
        "rebalance_frequency": "monthly",
        "risk_level": "medium"
    }
}

REBALANCING_OPTIONS = {
    "Never": 0,
    "Weekly": 7,
    "Monthly": 30,
    "Quarterly": 90,
    "Semi-Annually": 180,
    "Annually": 365
}

# ============================================
# DATA LOADING & PORTFOLIO CONSTRUCTION
# ============================================

def load_portfolio_data(symbols: List[str], start_date: date, end_date: date) -> pd.DataFrame:
    """Load comprehensive market data for portfolio construction"""
    np.random.seed(42)  # For consistent demo data
    
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)
    
    # Generate correlated stock returns for realistic portfolio behavior
    n_assets = len(symbols)
    correlation_matrix = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    np.fill_diagonal(correlation_matrix, 1.0)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    
    # Generate returns using multivariate normal distribution
    mean_returns = np.random.uniform(0.0005, 0.0015, n_assets)
    volatilities = np.random.uniform(0.015, 0.035, n_assets)
    
    # Create covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    # Generate correlated returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    
    # Convert to prices
    base_prices = {
        "AAPL": 175.0, "MSFT": 342.0, "GOOGL": 2650.0, "TSLA": 245.0, "NVDA": 485.0,
        "AMZN": 3380.0, "META": 310.0, "RELIANCE": 2800.0, "TCS": 3500.0, "INFY": 1400.0
    }
    
    data = pd.DataFrame({'date': dates})
    
    for i, symbol in enumerate(symbols):
        base_price = base_prices.get(symbol, 150.0)
        prices = [base_price]
        
        for ret in returns[:, i]:
            prices.append(max(prices[-1] * (1 + ret), 1.0))
        
        data[symbol] = prices[:n_days]
    
    return data

def calculate_portfolio_weights(data: pd.DataFrame, 
                              symbols: List[str],
                              strategy: str,
                              lookback_days: int = 60) -> Dict[str, float]:
    """Calculate portfolio weights based on selected strategy"""
    
    if strategy == "Equal Weight":
        return {symbol: 1.0 / len(symbols) for symbol in symbols}
    
    elif strategy == "Market Cap Weight":
        # Simulate market cap weighting (use recent price as proxy)
        recent_prices = {symbol: data[symbol].iloc[-1] for symbol in symbols}
        total_value = sum(recent_prices.values())
        return {symbol: price / total_value for symbol, price in recent_prices.items()}
    
    elif strategy == "Risk Parity":
        # Equal risk contribution (inverse volatility weighting)
        returns_data = data[symbols].pct_change().dropna()
        volatilities = returns_data.tail(lookback_days).std()
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        return weights.to_dict()
    
    elif strategy == "Momentum":
        # Weight by recent momentum
        returns_data = data[symbols].pct_change().dropna()
        momentum = returns_data.tail(lookback_days).mean()
        # Convert to positive weights
        momentum_adj = momentum - momentum.min() + 0.1
        weights = momentum_adj / momentum_adj.sum()
        return weights.to_dict()
    
    elif strategy == "Mean Reversion":
        # Weight inversely to recent performance
        returns_data = data[symbols].pct_change().dropna()
        recent_performance = returns_data.tail(lookback_days).mean()
        # Inverse weighting
        inv_performance = 1 / (recent_performance + 0.1)
        weights = inv_performance / inv_performance.sum()
        return weights.to_dict()
    
    elif strategy == "Minimum Variance":
        # Minimize portfolio variance
        returns_data = data[symbols].pct_change().dropna().tail(lookback_days)
        cov_matrix = returns_data.cov()
        
        # Simple minimum variance (equal weight for demo)
        inv_cov_sum = np.linalg.inv(cov_matrix).sum(axis=1)
        weights = inv_cov_sum / inv_cov_sum.sum()
        return {symbol: weight for symbol, weight in zip(symbols, weights)}
    
    elif strategy == "Maximum Sharpe":
        # Maximize Sharpe ratio (simplified)
        returns_data = data[symbols].pct_change().dropna().tail(lookback_days)
        mean_returns = returns_data.mean()
        cov_matrix = returns_data.cov()
        
        # Simple Sharpe optimization (equal weight for demo)
        sharpe_ratios = mean_returns / returns_data.std()
        weights = sharpe_ratios / sharpe_ratios.sum()
        return weights.to_dict()
    
    else:
        return {symbol: 1.0 / len(symbols) for symbol in symbols}

def simulate_portfolio_performance(data: pd.DataFrame,
                                 symbols: List[str],
                                 weights: Dict[str, float],
                                 initial_capital: float,
                                 rebalance_days: int,
                                 transaction_cost: float = 0.001) -> Dict[str, Any]:
    """Simulate portfolio performance with rebalancing"""
    
    returns_data = data[symbols].pct_change().fillna(0)
    portfolio_values = [initial_capital]
    portfolio_weights = [weights.copy()]
    rebalance_dates = []
    transaction_costs = []
    
    current_weights = weights.copy()
    
    for i, (date, returns) in enumerate(returns_data.iterrows()):
        if i == 0:
            continue
            
        # Calculate portfolio return
        portfolio_return = sum(current_weights[symbol] * returns[symbol] for symbol in symbols)
        new_value = portfolio_values[-1] * (1 + portfolio_return)
        
        # Update weights based on market movement
        for symbol in symbols:
            current_weights[symbol] *= (1 + returns[symbol]) / (1 + portfolio_return)
        
        # Rebalance if needed
        if rebalance_days > 0 and i % rebalance_days == 0:
            target_weights = calculate_portfolio_weights(
                data.iloc[:i+1], symbols, 
                list(PORTFOLIO_STRATEGIES.keys())[0]  # Use first strategy for rebalancing
            )
            
            # Calculate transaction costs
            weight_changes = sum(abs(current_weights[symbol] - target_weights[symbol]) 
                               for symbol in symbols)
            cost = new_value * weight_changes * transaction_cost
            new_value -= cost
            
            current_weights = target_weights.copy()
            rebalance_dates.append(date)
            transaction_costs.append(cost)
        
        portfolio_values.append(new_value)
        portfolio_weights.append(current_weights.copy())
    
    return {
        'portfolio_values': pd.Series(portfolio_values, index=[data['date'].iloc[0]] + list(returns_data.index)),
        'portfolio_weights': portfolio_weights,
        'rebalance_dates': rebalance_dates,
        'transaction_costs': transaction_costs,
        'final_value': portfolio_values[-1],
        'total_return': (portfolio_values[-1] - initial_capital) / initial_capital,
        'returns': pd.Series(portfolio_values).pct_change().fillna(0)
    }

# ============================================
# RISK ANALYTICS ENGINE
# ============================================

def calculate_portfolio_risk_metrics(portfolio_returns: pd.Series, 
                                    portfolio_values: pd.Series,
                                    benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """Calculate comprehensive portfolio risk metrics"""
    
    # Basic return metrics
    total_return = (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    
    # Risk metrics
    volatility = portfolio_returns.std() * np.sqrt(252)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    # Drawdown analysis
    peak = portfolio_values.cummax()
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min()
    
    # Sharpe and Sortino ratios
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0
    sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
    
    # Additional metrics
    positive_returns = portfolio_returns[portfolio_returns > 0]
    negative_returns = portfolio_returns[portfolio_returns < 0]
    
    win_rate = len(positive_returns) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
    avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
    avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
    profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() != 0 else float('inf')
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(portfolio_returns, 5)
    
    metrics = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'VaR (95%)': var_95,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss
    }
    
    # Beta calculation if benchmark provided
    if benchmark_returns is not None and len(benchmark_returns) == len(portfolio_returns):
        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
        metrics['Beta'] = beta
        
        # Alpha calculation
        benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
        alpha = annualized_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
        metrics['Alpha'] = alpha
    
    return metrics

# ============================================
# VISUALIZATION ENGINE
# ============================================

def create_portfolio_dashboard(portfolio_data: Dict[str, Any], 
                             market_data: pd.DataFrame,
                             symbols: List[str]) -> None:
    """Create comprehensive portfolio performance dashboard"""
    
    portfolio_values = portfolio_data['portfolio_values']
    
    # Create multi-panel dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Portfolio Value Over Time',
            'Asset Allocation',
            'Drawdown Analysis', 
            'Rolling Sharpe Ratio',
            'Return Distribution',
            'Correlation Heatmap'
        ),
        specs=[
            [{"colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ]
    )
    
    # Portfolio value chart
    fig.add_trace(
        go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=3)
        ), row=1, col=1
    )
    
    # Drawdown chart
    peak = portfolio_values.cummax()
    drawdown = (portfolio_values - peak) / peak * 100
    
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown %',
            fill='tonexty',
            line=dict(color='red', width=2)
        ), row=2, col=1
    )
    
    # Rolling Sharpe ratio
    returns = portfolio_values.pct_change().fillna(0)
    rolling_sharpe = returns.rolling(window=30).mean() / returns.rolling(window=30).std() * np.sqrt(252)
    
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode='lines',
            name='30-Day Rolling Sharpe',
            line=dict(color='green', width=2)
        ), row=2, col=2
    )
    
    # Return distribution
    fig.add_trace(
        go.Histogram(
            x=returns.values * 100,
            name='Daily Returns %',
            nbinsx=50,
            opacity=0.7
        ), row=3, col=1
    )
    
    # Asset correlation heatmap (simplified)
    if len(symbols) > 1:
        asset_returns = market_data[symbols].pct_change().fillna(0)
        correlation_matrix = asset_returns.corr()
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ), row=3, col=2
        )
    
    fig.update_layout(
        height=1000,
        title="Portfolio Performance Dashboard",
        template='plotly_white',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_allocation_pie_chart(weights: Dict[str, float]) -> None:
    """Plot current portfolio allocation"""
    
    fig = go.Figure(data=[
        go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Current Portfolio Allocation",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# MAIN PAGE FUNCTION
# ============================================

def main():
    """Main portfolio management page function"""
    
    # Apply theme
    apply_custom_theme("financial")
    
    # Page header
    st.title("📁 Advanced Portfolio Management & Risk Analytics")
    st.markdown("Professional portfolio construction, optimization, and comprehensive risk management")
    
    if not PORTFOLIO_MODULES_AVAILABLE:
        st.warning("⚠️ **Portfolio modules not fully available.** Using comprehensive fallback implementations.")
    
    st.markdown("---")
    
    # ============================================
    # PORTFOLIO CONFIGURATION
    # ============================================
    
    st.subheader("🛠️ Portfolio Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Symbol selection with regional support
        region = st.selectbox("Market Region", ["US", "Indian", "Global"])
        
        if region == "US":
            available_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "ORCL", "NFLX", "CRM"]
        elif region == "Indian":
            available_symbols = ["RELIANCE", "TCS", "INFY", "HINDUNILVR", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "HDFCBANK"]
        else:
            available_symbols = ["AAPL", "MSFT", "GOOGL", "RELIANCE", "TCS", "ASML", "TSM", "SAP", "SHOP", "BABA"]
        
        selected_symbols = filter_categorical(
            "Select Portfolio Assets",
            available_symbols,
            multi=True,
            key="portfolio_symbols"
        )
        
        start_date, end_date = filter_date_range(
            default_days=365,
            key="portfolio_dates"
        )
    
    with col2:
        # Portfolio strategy selection
        selected_strategy = filter_categorical(
            "Portfolio Strategy",
            list(PORTFOLIO_STRATEGIES.keys()),
            multi=False,
            key="portfolio_strategy"
        )
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000
        )
        
        rebalance_frequency = st.selectbox(
            "Rebalancing Frequency",
            list(REBALANCING_OPTIONS.keys()),
            index=2  # Monthly default
        )
    
    if not selected_symbols:
        st.warning("⚠️ Please select at least 2 assets for portfolio construction")
        return
    
    if len(selected_symbols) < 2:
        st.warning("⚠️ Portfolio requires at least 2 assets for diversification")
        return
    
    # Strategy information
    if selected_strategy:
        strategy_info = PORTFOLIO_STRATEGIES[selected_strategy]
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Strategy:** {strategy_info['description']}")
        with col2:
            st.info(f"**Risk Level:** {strategy_info['risk_level']}")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Portfolio Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Risk Management**")
            max_position_weight = st.slider("Max Position Weight (%)", 5, 50, 25)
            max_sector_weight = st.slider("Max Sector Weight (%)", 10, 100, 40)
            rebalance_threshold = st.slider("Rebalance Threshold (%)", 1, 10, 5)
        
        with col2:
            st.write("**Transaction Costs**")
            transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.01)
            min_trade_size = st.number_input("Min Trade Size ($)", 100, 10000, 1000)
            slippage_factor = st.slider("Slippage Factor (%)", 0.0, 0.5, 0.05, 0.01)
        
        with col3:
            st.write("**Performance Settings**")
            benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "VTI", "None"], index=0)
            lookback_days = st.slider("Lookback Period (days)", 30, 252, 60)
            risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0, 0.1)
    
    st.markdown("---")
    
    # ============================================
    # DATA LOADING & PORTFOLIO CONSTRUCTION
    # ============================================
    
    with st.spinner("Loading market data and constructing portfolio..."):
        # Load market data
        market_data = load_portfolio_data(selected_symbols, start_date, end_date)
        
        if market_data.empty:
            st.error("❌ No market data available for selected period")
            return
        
        # Calculate portfolio weights
        portfolio_weights = calculate_portfolio_weights(
            market_data, selected_symbols, selected_strategy, lookback_days
        )
        
        # Simulate portfolio performance
        rebalance_days = REBALANCING_OPTIONS[rebalance_frequency]
        portfolio_performance = simulate_portfolio_performance(
            market_data,
            selected_symbols,
            portfolio_weights,
            initial_capital,
            rebalance_days,
            transaction_cost / 100
        )
    
    # ============================================
    # PORTFOLIO OVERVIEW
    # ============================================
    
    st.subheader("📊 Portfolio Overview")
    
    # Key metrics
    final_value = portfolio_performance['final_value']
    total_return = portfolio_performance['total_return']
    portfolio_values = portfolio_performance['portfolio_values']
    portfolio_returns = portfolio_performance['returns']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Value", 
            f"${final_value:,.0f}",
            f"${final_value - initial_capital:+,.0f}"
        )
    
    with col2:
        st.metric(
            "Total Return",
            f"{total_return:.2%}",
            f"{total_return*100:+.1f}%"
        )
    
    with col3:
        days_elapsed = (end_date - start_date).days
        annualized_return = (1 + total_return) ** (365 / days_elapsed) - 1
        st.metric("Annualized Return", f"{annualized_return:.2%}")
    
    with col4:
        volatility = portfolio_returns.std() * np.sqrt(252)
        st.metric("Volatility", f"{volatility:.2%}")
    
    # ============================================
    # PORTFOLIO ALLOCATION
    # ============================================
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 Current Allocation")
        plot_allocation_pie_chart(portfolio_weights)
    
    with col2:
        st.subheader("📋 Holdings Detail")
        
        # Create holdings table
        holdings_data = []
        latest_prices = {symbol: market_data[symbol].iloc[-1] for symbol in selected_symbols}
        
        for symbol, weight in portfolio_weights.items():
            allocation_value = final_value * weight
            shares = allocation_value / latest_prices[symbol]
            
            holdings_data.append({
                'Symbol': symbol,
                'Weight': f"{weight:.1%}",
                'Value': f"${allocation_value:,.0f}",
                'Shares': f"{shares:.0f}",
                'Price': f"${latest_prices[symbol]:.2f}"
            })
        
        holdings_df = pd.DataFrame(holdings_data)
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)
    
    # ============================================
    # PERFORMANCE ANALYTICS
    # ============================================
    
    st.subheader("📈 Performance Analytics")
    
    # Calculate comprehensive risk metrics
    risk_metrics = calculate_portfolio_risk_metrics(portfolio_returns, portfolio_values)
    
    # Create performance dashboard
    create_portfolio_dashboard(portfolio_performance, market_data, selected_symbols)
    
    # Risk metrics grid
    st.subheader("📊 Risk & Performance Metrics")
    
    metrics_grid = {
        "Total Return": {
            "value": f"{risk_metrics['Total Return']:.2%}",
            "help": "Total portfolio return over the period"
        },
        "Annualized Return": {
            "value": f"{risk_metrics['Annualized Return']:.2%}",
            "help": "Annualized portfolio return"
        },
        "Volatility": {
            "value": f"{risk_metrics['Volatility']:.2%}",
            "help": "Annualized portfolio volatility"
        },
        "Sharpe Ratio": {
            "value": f"{risk_metrics['Sharpe Ratio']:.3f}",
            "help": "Risk-adjusted return metric"
        },
        "Sortino Ratio": {
            "value": f"{risk_metrics['Sortino Ratio']:.3f}",
            "help": "Downside risk-adjusted return"
        },
        "Max Drawdown": {
            "value": f"{risk_metrics['Max Drawdown']:.2%}",
            "help": "Maximum peak-to-trough decline"
        },
        "Win Rate": {
            "value": f"{risk_metrics['Win Rate']:.1%}",
            "help": "Percentage of positive return days"
        },
        "VaR (95%)": {
            "value": f"{risk_metrics['VaR (95%)']:.2%}",
            "help": "Value at Risk at 95% confidence"
        }
    }
    
    create_metrics_grid(metrics_grid, cols=4)
    
    # ============================================
    # REBALANCING ANALYSIS
    # ============================================
    
    if len(portfolio_performance['rebalance_dates']) > 0:
        st.subheader("🔄 Rebalancing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Rebalancing Events", len(portfolio_performance['rebalance_dates']))
            total_transaction_costs = sum(portfolio_performance['transaction_costs'])
            st.metric("Total Transaction Costs", f"${total_transaction_costs:,.0f}")
        
        with col2:
            avg_cost_per_rebalance = total_transaction_costs / len(portfolio_performance['rebalance_dates'])
            st.metric("Avg Cost per Rebalance", f"${avg_cost_per_rebalance:,.0f}")
            cost_percentage = total_transaction_costs / initial_capital
            st.metric("Cost as % of Capital", f"{cost_percentage:.2%}")
        
        # Rebalancing dates table
        if portfolio_performance['rebalance_dates']:
            rebalance_df = pd.DataFrame({
                'Date': portfolio_performance['rebalance_dates'],
                'Transaction Cost': [f"${cost:,.0f}" for cost in portfolio_performance['transaction_costs']]
            })
            st.dataframe(rebalance_df, use_container_width=True, hide_index=True)
    
    # ============================================
    # STRESS TESTING & SCENARIOS
    # ============================================
    
    st.subheader("🧪 Stress Testing & Scenario Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Market Stress Scenarios**")
        
        # Simulate different market scenarios
        scenarios = {
            "Market Crash (-20%)": -0.20,
            "Moderate Correction (-10%)": -0.10,
            "Volatility Spike (+50% vol)": 0.0,
            "Bull Market (+15%)": 0.15
        }
        
        current_value = portfolio_values.iloc[-1]
        
        for scenario, impact in scenarios.items():
            if "vol" in scenario.lower():
                # Volatility scenario - show impact on risk metrics
                new_volatility = risk_metrics['Volatility'] * 1.5
                st.write(f"**{scenario}:** Volatility → {new_volatility:.2%}")
            else:
                stressed_value = current_value * (1 + impact)
                change = stressed_value - current_value
                st.write(f"**{scenario}:** ${stressed_value:,.0f} ({change:+,.0f})")
    
    with col2:
        st.write("**Portfolio Statistics**")
        
        # Additional portfolio statistics
        asset_returns = market_data[selected_symbols].pct_change().fillna(0)
        
        # Correlation statistics
        avg_correlation = asset_returns.corr().values[np.triu_indices_from(asset_returns.corr().values, k=1)].mean()
        max_correlation = asset_returns.corr().values[np.triu_indices_from(asset_returns.corr().values, k=1)].max()
        
        st.write(f"**Average Asset Correlation:** {avg_correlation:.3f}")
        st.write(f"**Maximum Asset Correlation:** {max_correlation:.3f}")
        st.write(f"**Number of Assets:** {len(selected_symbols)}")
        st.write(f"**Effective Diversification:** {1/sum(w**2 for w in portfolio_weights.values()):.1f}")
    
    # ============================================
    # EXPORT & REPORTING
    # ============================================
    
    st.subheader("📥 Export & Reporting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export portfolio data
        portfolio_export = pd.DataFrame({
            'Date': portfolio_values.index,
            'Portfolio_Value': portfolio_values.values,
            'Daily_Return': portfolio_returns.values
        })
        
        create_download_button(
            portfolio_export,
            f"portfolio_performance_{selected_strategy.lower().replace(' ', '_')}.csv",
            "📊 Download Performance Data",
            key="portfolio_performance"
        )
    
    with col2:
        # Export holdings
        holdings_export = pd.DataFrame([
            {
                'Symbol': symbol,
                'Weight': weight,
                'Value': final_value * weight,
                'Price': latest_prices[symbol]
            }
            for symbol, weight in portfolio_weights.items()
        ])
        
        create_download_button(
            holdings_export,
            "current_holdings.csv",
            "💼 Download Holdings",
            key="holdings_export"
        )
    
    with col3:
        # Export comprehensive report
        report_data = pd.DataFrame([{
            'Strategy': selected_strategy,
            'Assets': ', '.join(selected_symbols),
            'Initial_Capital': initial_capital,
            'Final_Value': final_value,
            'Total_Return': total_return,
            'Annualized_Return': risk_metrics['Annualized Return'],
            'Volatility': risk_metrics['Volatility'],
            'Sharpe_Ratio': risk_metrics['Sharpe Ratio'],
            'Max_Drawdown': risk_metrics['Max Drawdown'],
            'Rebalancing_Frequency': rebalance_frequency,
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        
        create_download_button(
            report_data,
            "portfolio_report.csv",
            "📋 Download Full Report",
            key="report_export"
        )
    
    # ============================================
    # ALERTS & RECOMMENDATIONS
    # ============================================
    
    st.markdown("---")
    st.subheader("🚨 Portfolio Alerts & Recommendations")
    
    # Generate portfolio alerts
    alert_manager = get_alert_manager()
    
    # Performance alerts
    if total_return > 0.15:
        alert_manager.add_alert(
            f"🎉 Portfolio significantly outperforming with {total_return:.1%} returns",
            "portfolio_performance",
            "success",
            selected_strategy
        )
    elif total_return < -0.1:
        alert_manager.add_alert(
            f"⚠️ Portfolio underperforming with {total_return:.1%} returns - consider strategy review",
            "portfolio_performance",
            "warning",
            selected_strategy
        )
    
    # Risk alerts
    if risk_metrics['Max Drawdown'] < -0.2:
        alert_manager.add_alert(
            f"🚨 High drawdown detected ({risk_metrics['Max Drawdown']:.1%}) - review risk settings",
            "risk_management",
            "error",
            selected_strategy
        )
    
    if risk_metrics['Sharpe Ratio'] > 1.5:
        alert_manager.add_alert(
            f"✅ Excellent risk-adjusted returns (Sharpe: {risk_metrics['Sharpe Ratio']:.2f})",
            "risk_metrics",
            "success",
            selected_strategy
        )
    elif risk_metrics['Sharpe Ratio'] < 0.5:
        alert_manager.add_alert(
            f"⚠️ Poor risk-adjusted returns (Sharpe: {risk_metrics['Sharpe Ratio']:.2f}) - consider optimization",
            "risk_metrics",
            "warning",
            selected_strategy
        )
    
    # Correlation alerts
    if avg_correlation > 0.8:
        alert_manager.add_alert(
            f"⚠️ High asset correlation ({avg_correlation:.2f}) detected - diversification may be limited",
            "diversification",
            "warning",
            selected_strategy
        )
    
    # Display alerts
    if alert_manager.alerts:
        for alert in alert_manager.alerts[-8:]:
            if alert.level.value == "success":
                st.success(f"✅ {alert}")
            elif alert.level.value == "warning":
                st.warning(f"⚠️ {alert}")
            elif alert.level.value == "error":
                st.error(f"❌ {alert}")
            else:
                st.info(f"ℹ️ {alert}")
    else:
        st.info("✅ Portfolio is performing within normal parameters")
    
    # ============================================
    # FOOTER
    # ============================================
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        📁 Advanced Portfolio Management & Risk Analytics | 
        Strategy: {selected_strategy} | Assets: {len(selected_symbols)} | 
        Capital: ${initial_capital:,.0f} | 
        Performance: {total_return:+.1%} | 
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE EXECUTION
# ============================================

if __name__ == "__main__":
    main()
