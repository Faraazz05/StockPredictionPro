"""
app/pages/01_🏠_Overview.py

Main dashboard overview page for StockPredictionPro.
Displays portfolio summary, key metrics, performance charts,
latest market data, alerts, and quick insights.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any

# Import your components
from app.components.filters import filter_symbols, filter_date_range, create_data_filter_panel
from app.components.charts import (
    plot_candlestick_with_volume, 
    render_line_chart,
    render_correlation_heatmap,
    create_chart_controls
)
from app.components.metrics import (
    display_trading_metrics,
    create_metrics_grid,
    display_performance_summary
)
from app.components.alerts import (
    get_alert_manager, 
    generate_simple_signal_alerts,
    create_alerts_sidebar
)
from app.components.tables import (
    display_stock_data_table,
    display_portfolio_holdings_table,
    create_download_button
)
from app.styles.themes import apply_custom_theme

# ============================================
# DATA LOADING FUNCTIONS (PLACEHOLDER)
# ============================================

def load_portfolio_summary() -> Dict[str, Any]:
    """Load portfolio summary data"""
    # TODO: Replace with real portfolio data from your database
    return {
        "total_value": 125430.50,
        "daily_pnl": 2150.25,
        "total_pnl": 15430.50,
        "cash_balance": 25000.00,
        "positions_count": 8,
        "win_rate": 0.685
    }

def load_stock_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Load historical stock data for given symbol and date range"""
    # TODO: Replace with real data loading from your src/data modules
    np.random.seed(42)  # For consistent demo data
    
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)
    
    # Generate realistic stock price data
    base_price = 150.0
    returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_days)
    })
    
    # Calculate returns
    data['returns'] = data['close'].pct_change().fillna(0)
    
    return data

def load_portfolio_holdings() -> pd.DataFrame:
    """Load current portfolio holdings"""
    # TODO: Replace with real portfolio holdings data
    holdings = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN'],
        'quantity': [100, 50, 25, 75, 40, 30],
        'current_price': [175.25, 342.50, 2650.75, 245.80, 485.20, 3380.45],
        'avg_cost': [165.30, 320.15, 2580.90, 220.45, 470.85, 3250.20],
        'market_value': [17525, 17125, 66269, 18435, 19408, 101413]
    })
    
    # Calculate unrealized P&L
    holdings['unrealized_pnl'] = (holdings['current_price'] - holdings['avg_cost']) * holdings['quantity']
    
    return holdings

def get_market_status() -> Dict[str, str]:
    """Get current market status"""
    now = datetime.now()
    market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if market_open_time <= now <= market_close_time:
        return {"status": "OPEN", "message": "Market is currently open"}
    else:
        return {"status": "CLOSED", "message": "Market is currently closed"}

def calculate_key_metrics(portfolio_data: Dict[str, Any], holdings_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Calculate key portfolio metrics for display"""
    total_unrealized = holdings_df['unrealized_pnl'].sum()
    
    metrics = {
        "Portfolio Value": {
            "value": portfolio_data['total_value'],
            "delta": portfolio_data['daily_pnl'],
            "help": "Total portfolio market value"
        },
        "Daily P&L": {
            "value": portfolio_data['daily_pnl'],
            "delta": portfolio_data['daily_pnl'] / portfolio_data['total_value'] * 100,
            "help": "Today's profit/loss"
        },
        "Total P&L": {
            "value": total_unrealized,
            "delta": total_unrealized / portfolio_data['total_value'] * 100,
            "help": "Total unrealized profit/loss"
        },
        "Win Rate": {
            "value": portfolio_data['win_rate'] * 100,
            "delta": 3.2,  # Example change
            "help": "Percentage of profitable trades"
        },
        "Cash Balance": {
            "value": portfolio_data['cash_balance'],
            "delta": -1500,  # Example change
            "help": "Available cash for trading"
        },
        "Positions": {
            "value": len(holdings_df),
            "delta": 1,  # Example change
            "help": "Number of open positions"
        }
    }
    
    return metrics

# ============================================
# MAIN PAGE FUNCTION
# ============================================

def main():
    """Main overview page function"""
    
    # Apply theme
    apply_custom_theme("financial")
    
    # Page header
    st.title("🏠 StockPredictionPro Dashboard")
    st.markdown("---")
    
    # Market status
    market_status = get_market_status()
    if market_status["status"] == "OPEN":
        st.success(f"🟢 {market_status['message']}")
    else:
        st.info(f"🔴 {market_status['message']}")
    
    # Data loading
    with st.spinner("Loading portfolio data..."):
        portfolio_data = load_portfolio_summary()
        holdings_df = load_portfolio_holdings()
        key_metrics = calculate_key_metrics(portfolio_data, holdings_df)
    
    # ============================================
    # METRICS DASHBOARD
    # ============================================
    
    st.subheader("📊 Key Metrics")
    create_metrics_grid(key_metrics, cols=3)
    
    st.markdown("---")
    
    # ============================================
    # FILTERS & CONTROLS
    # ============================================
    
    st.subheader("🔍 Analysis Controls")
    
    # Get available symbols from holdings
    available_symbols = holdings_df['symbol'].tolist() + ['SPY', 'QQQ', 'VTI']  # Add some ETFs
    
    # Create filter panel
    data_filters = create_data_filter_panel(available_symbols, key_prefix="overview")
    
    # Chart controls
    chart_options = create_chart_controls()
    
    st.markdown("---")
    
    # ============================================
    # MAIN CHARTS SECTION
    # ============================================
    
    st.subheader("📈 Market Analysis")
    
    # Load data for selected symbol
    if data_filters['symbols']:
        selected_symbol = data_filters['symbols'][0]  # Use first selected symbol
        
        with st.spinner(f"Loading data for {selected_symbol}..."):
            stock_data = load_stock_data(
                selected_symbol, 
                data_filters['start_date'], 
                data_filters['end_date']
            )
        
        if not stock_data.empty:
            # Main price chart
            if chart_options['chart_type'] == 'Candlestick':
                plot_candlestick_with_volume(
                    stock_data, 
                    title=f"{selected_symbol} Price Chart",
                    show_volume=chart_options.get('show_volume', True)
                )
            else:
                render_line_chart(
                    stock_data, 
                    'date', 
                    ['close'], 
                    title=f"{selected_symbol} Price Trend"
                )
            
            # Performance metrics for the selected stock
            if len(stock_data) > 1:
                display_trading_metrics(
                    stock_data['returns'], 
                    title=f"{selected_symbol} Performance"
                )
        else:
            st.warning(f"No data available for {selected_symbol}")
    
    st.markdown("---")
    
    # ============================================
    # PORTFOLIO SECTION
    # ============================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💼 Portfolio Holdings")
        display_portfolio_holdings_table(holdings_df, portfolio_data['total_value'])
        
        # Download button for portfolio data
        create_download_button(
            holdings_df, 
            "portfolio_holdings.csv", 
            "📥 Download Portfolio Data",
            key="portfolio_download"
        )
    
    with col2:
        st.subheader("🔔 Recent Alerts")
        
        # Generate some sample alerts
        alert_manager = get_alert_manager()
        
        # Add sample alerts if none exist
        if len(alert_manager.alerts) == 0:
            alert_manager.add_alert("Portfolio up +2.1% today", "success")
            alert_manager.add_alert("AAPL approaching resistance level", "warning")
            alert_manager.add_alert("High volume detected in TSLA", "info")
        
        # Display alerts in sidebar style
        create_alerts_sidebar(alert_manager)
    
    st.markdown("---")
    
    # ============================================
    # CORRELATION ANALYSIS
    # ============================================
    
    if len(data_filters['symbols']) > 1:
        st.subheader("📊 Portfolio Correlation Analysis")
        
        # Load data for multiple symbols for correlation
        correlation_data = {}
        for symbol in data_filters['symbols'][:5]:  # Limit to 5 symbols for performance
            symbol_data = load_stock_data(
                symbol,
                data_filters['start_date'],
                data_filters['end_date']
            )
            correlation_data[symbol] = symbol_data['returns']
        
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data)
            render_correlation_heatmap(corr_df, "Symbol Correlation Matrix")
    
    # ============================================
    # RECENT DATA TABLE
    # ============================================
    
    if data_filters['symbols'] and not stock_data.empty:
        st.subheader(f"📋 Recent Data - {selected_symbol}")
        
        # Show last 10 days of data
        recent_data = stock_data.tail(10).copy()
        display_stock_data_table(recent_data, selected_symbol, max_rows=10)
        
        # Download button for stock data
        create_download_button(
            stock_data, 
            f"{selected_symbol}_price_data.csv",
            f"📥 Download {selected_symbol} Data",
            key="stock_download"
        )
    
    # ============================================
    # FOOTER
    # ============================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        📊 StockPredictionPro Dashboard | Last Updated: {timestamp}
    </div>
    """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 
    unsafe_allow_html=True)

# ============================================
# PAGE EXECUTION
# ============================================

if __name__ == "__main__":
    main()
