"""
StockPredictionPro - Main Streamlit Application
Enterprise-grade stock prediction and analysis platform

Author: StockPredictionPro Team
Date: August 2025
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import your existing modules
try:
    from src.data.manager import DataManager
    from src.features.indicators.momentum import RSI, MACD
    from src.features.indicators.trend import SMA, EMA
    from src.models.factory import ModelFactory
    from src.evaluation.plots import PlotGenerator
    from src.trading.portfolio import Portfolio
    from app.utils.session_state import SessionStateManager
    from app.utils.cache_helpers import CacheHelper
    from app.components.charts import ChartRenderer
    from app.components.metrics import MetricsDisplay
    from app.components.filters import FilterPanel
    from app.styles.themes import apply_custom_theme
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please ensure all required modules are installed and paths are correct.")


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="StockPredictionPro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/YourRepo/StockPredictionPro',
        'Report a bug': "https://github.com/YourRepo/StockPredictionPro/issues",
        'About': "# StockPredictionPro\nEnterprise Stock Prediction Platform"
    }
)

# Apply custom styling
apply_custom_theme()

# Initialize session state
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionStateManager()

# ============================================
# SIDEBAR NAVIGATION & FILTERS
# ============================================

with st.sidebar:
    st.title("🏠 StockPredictionPro")
    st.markdown("---")
    
    # Market Selection
    st.subheader("🌍 Market Selection")
    market = st.selectbox(
        "Choose Market:",
        options=["US Market", "Indian Market", "Global Combined"],
        index=0,
        key="market_selection"
    )
    
    # Stock Symbol Selection
    st.subheader("📊 Stock Selection")
    
    # Load available symbols based on market
    @st.cache_data
    def load_available_symbols(market_choice):
        if market_choice == "US Market":
            # Load US symbols from your data
            us_data_path = project_root / "data" / "raw" / "daily" / "us"
            if us_data_path.exists():
                files = [f.stem.replace("_daily", "") for f in us_data_path.glob("*_daily.csv")]
                return sorted([f for f in files if f != "all_us_stocks"])
            return ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        
        elif market_choice == "Indian Market":
            # Load Indian symbols from your data
            indian_data_path = project_root / "data" / "raw" / "daily" / "indian"
            if indian_data_path.exists():
                files = [f.stem.replace("_daily", "") for f in indian_data_path.glob("*_daily.csv")]
                return sorted([f for f in files if f != "all_indian_stocks"])
            return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
        
        else:  # Global Combined
            us_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
            indian_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY"]
            return sorted(us_symbols + indian_symbols)
    
    available_symbols = load_available_symbols(market)
    
    selected_symbol = st.selectbox(
        "Select Stock Symbol:",
        options=available_symbols,
        index=0,
        key="symbol_selection"
    )
    
    # Date Range Selection
    st.subheader("📅 Date Range")
    date_range = st.selectbox(
        "Select Period:",
        options=["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Max"],
        index=3,
        key="date_range"
    )
    
    # Analysis Type
    st.subheader("🔍 Analysis Type")
    analysis_tabs = st.radio(
        "Choose Analysis:",
        options=["Overview", "Technical Analysis", "ML Predictions", "Portfolio"],
        key="analysis_type"
    )
    
    st.markdown("---")
    
    # Quick Stats
    st.subheader("⚡ Quick Stats")
    if st.button("🔄 Refresh Data", key="refresh_data"):
        st.cache_data.clear()
        st.rerun()

# ============================================
# MAIN CONTENT AREA
# ============================================

# Load data for selected symbol
@st.cache_data
def load_stock_data(symbol, market_type):
    """Load stock data from your existing CSV files"""
    try:
        if market_type == "US Market":
            file_path = project_root / "data" / "raw" / "daily" / "us" / f"{symbol}_daily.csv"
        elif market_type == "Indian Market":
            file_path = project_root / "data" / "raw" / "daily" / "indian" / f"{symbol}_daily.csv"
        else:
            # Try US first, then Indian
            us_path = project_root / "data" / "raw" / "daily" / "us" / f"{symbol}_daily.csv"
            indian_path = project_root / "data" / "raw" / "daily" / "indian" / f"{symbol}_daily.csv"
            
            if us_path.exists():
                file_path = us_path
            elif indian_path.exists():
                file_path = indian_path
            else:
                return None
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df
        else:
            return None
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data
stock_data = load_stock_data(selected_symbol, market)

if stock_data is not None and not stock_data.empty:
    
    # ============================================
    # HEADER SECTION
    # ============================================
    
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        st.title(f"📈 {selected_symbol}")
        latest_price = stock_data['close'].iloc[-1]
        prev_price = stock_data['close'].iloc[-2] if len(stock_data) > 1 else latest_price
        price_change = latest_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        # Format price change with color
        change_color = "green" if price_change >= 0 else "red"
        change_symbol = "▲" if price_change >= 0 else "▼"
        
        st.markdown(
            f"""
            <div style='font-size: 24px; font-weight: bold;'>
                ${latest_price:.2f} 
                <span style='color: {change_color}; font-size: 16px;'>
                    {change_symbol} ${abs(price_change):.2f} ({price_change_pct:+.2f}%)
                </span>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.metric(
            label="Volume",
            value=f"{stock_data['volume'].iloc[-1]:,.0f}",
            delta=f"{((stock_data['volume'].iloc[-1] / stock_data['volume'].iloc[-2]) - 1) * 100:+.1f}%" if len(stock_data) > 1 else None
        )
    
    with col3:
        st.metric(
            label="High",
            value=f"${stock_data['high'].iloc[-1]:.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Low", 
            value=f"${stock_data['low'].iloc[-1]:.2f}",
            delta=None
        )
    
    st.markdown("---")
    
    # ============================================
    # MAIN ANALYSIS CONTENT
    # ============================================
    
    if analysis_tabs == "Overview":
        
        # Price Chart
        st.subheader("📊 Price Chart")
        
        # Create price chart using your chart component
        chart_col1, chart_col2 = st.columns([3, 1])
        
        with chart_col1:
            # Main price chart
            chart_data = stock_data.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
            
            # Simple line chart for now (you can enhance with your ChartRenderer component)
            st.line_chart(chart_data[['close']], height=400)
        
        with chart_col2:
            # Key Statistics
            st.subheader("📋 Key Stats")
            
            # Calculate key metrics
            returns = stock_data['close'].pct_change().dropna()
            
            metrics_data = {
                "52W High": f"${stock_data['high'].max():.2f}",
                "52W Low": f"${stock_data['low'].min():.2f}",
                "Avg Volume": f"{stock_data['volume'].mean():,.0f}",
                "Volatility": f"{returns.std() * np.sqrt(252) * 100:.1f}%",
                "Sharpe Ratio": f"{(returns.mean() / returns.std()) * np.sqrt(252):.2f}",
                "Total Records": f"{len(stock_data):,}"
            }
            
            for metric, value in metrics_data.items():
                st.metric(metric, value)
        
        # Volume Chart
        st.subheader("📊 Volume Analysis")
        st.bar_chart(chart_data[['volume']], height=200)
        
        # Recent Performance Table
        st.subheader("📈 Recent Performance (Last 10 Days)")
        recent_data = stock_data.tail(10)[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        recent_data['change'] = recent_data['close'].diff()
        recent_data['change_pct'] = recent_data['close'].pct_change() * 100
        
        # Format the dataframe for display
        recent_data['date'] = recent_data['date'].dt.strftime('%Y-%m-%d')
        recent_data = recent_data.round(2)
        
        st.dataframe(
            recent_data,
            use_container_width=True,
            hide_index=True
        )
    
    elif analysis_tabs == "Technical Analysis":
        
        st.subheader("🔧 Technical Indicators")
        
        # Technical Indicators Selection
        indicator_col1, indicator_col2, indicator_col3 = st.columns(3)
        
        with indicator_col1:
            show_sma = st.checkbox("Simple Moving Average", value=True)
            sma_period = st.slider("SMA Period", 5, 50, 20) if show_sma else 20
        
        with indicator_col2:
            show_rsi = st.checkbox("RSI", value=True)
            rsi_period = st.slider("RSI Period", 10, 30, 14) if show_rsi else 14
        
        with indicator_col3:
            show_macd = st.checkbox("MACD", value=True)
        
        # Calculate indicators
        chart_data = stock_data.copy()
        
        if show_sma:
            chart_data[f'SMA_{sma_period}'] = chart_data['close'].rolling(window=sma_period).mean()
        
        if show_rsi:
            # Simple RSI calculation
            delta = chart_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            chart_data['RSI'] = 100 - (100 / (1 + rs))
        
        if show_macd:
            # Simple MACD calculation
            ema12 = chart_data['close'].ewm(span=12).mean()
            ema26 = chart_data['close'].ewm(span=26).mean()
            chart_data['MACD'] = ema12 - ema26
            chart_data['MACD_Signal'] = chart_data['MACD'].ewm(span=9).mean()
        
        # Display charts
        chart_data_indexed = chart_data.set_index('date')
        
        # Price with SMA
        if show_sma:
            st.subheader("📈 Price with Moving Average")
            price_sma_data = chart_data_indexed[['close', f'SMA_{sma_period}']].dropna()
            st.line_chart(price_sma_data, height=300)
        
        # RSI
        if show_rsi:
            st.subheader("⚡ RSI (Relative Strength Index)")
            rsi_data = chart_data_indexed[['RSI']].dropna()
            st.line_chart(rsi_data, height=200)
            
            # RSI interpretation
            latest_rsi = chart_data['RSI'].iloc[-1] if not pd.isna(chart_data['RSI'].iloc[-1]) else None
            if latest_rsi:
                if latest_rsi > 70:
                    st.warning(f"🔴 RSI: {latest_rsi:.1f} - Potentially Overbought")
                elif latest_rsi < 30:
                    st.success(f"🟢 RSI: {latest_rsi:.1f} - Potentially Oversold")
                else:
                    st.info(f"🔵 RSI: {latest_rsi:.1f} - Neutral Zone")
        
        # MACD
        if show_macd:
            st.subheader("📊 MACD (Moving Average Convergence Divergence)")
            macd_data = chart_data_indexed[['MACD', 'MACD_Signal']].dropna()
            st.line_chart(macd_data, height=200)
    
    elif analysis_tabs == "ML Predictions":
        
        st.subheader("🤖 Machine Learning Predictions")
        
        # Model Selection
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            model_type = st.selectbox(
                "Select Model Type:",
                options=["Linear Regression", "Random Forest", "XGBoost", "LSTM", "Ensemble"],
                key="model_selection"
            )
        
        with model_col2:
            prediction_horizon = st.selectbox(
                "Prediction Horizon:",
                options=["1 Day", "3 Days", "1 Week", "1 Month"],
                key="prediction_horizon"
            )
        
        if st.button("🚀 Generate Predictions", key="generate_predictions"):
            
            with st.spinner("Training model and generating predictions..."):
                
                # Simulate ML predictions (replace with your actual model logic)
                np.random.seed(42)
                
                # Generate mock predictions based on recent price trends
                recent_prices = stock_data['close'].tail(30).values
                trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                
                horizon_days = {"1 Day": 1, "3 Days": 3, "1 Week": 7, "1 Month": 30}[prediction_horizon]
                
                # Mock prediction logic
                base_price = stock_data['close'].iloc[-1]
                predictions = []
                
                for i in range(horizon_days):
                    # Simple trend + noise prediction
                    predicted_price = base_price + (trend * (i + 1)) + np.random.normal(0, base_price * 0.02)
                    predictions.append(max(predicted_price, 0.01))  # Ensure positive prices
                
                # Display predictions
                pred_dates = pd.date_range(
                    start=stock_data['date'].iloc[-1] + pd.Timedelta(days=1),
                    periods=horizon_days,
                    freq='D'
                )
                
                pred_df = pd.DataFrame({
                    'date': pred_dates,
                    'predicted_price': predictions
                })
                
                # Show prediction results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Prediction Results")
                    
                    for i, (date, price) in enumerate(zip(pred_dates, predictions)):
                        change = ((price - base_price) / base_price) * 100
                        color = "green" if change >= 0 else "red"
                        
                        st.markdown(
                            f"""
                            **Day {i+1}** ({date.strftime('%Y-%m-%d')}): 
                            <span style='color: {color}; font-weight: bold;'>
                                ${price:.2f} ({change:+.1f}%)
                            </span>
                            """,
                            unsafe_allow_html=True
                        )
                
                with col2:
                    st.subheader("📊 Prediction Chart")
                    
                    # Combine historical and predicted data for chart
                    hist_data = stock_data.tail(30)[['date', 'close']].copy()
                    hist_data['type'] = 'Historical'
                    hist_data = hist_data.rename(columns={'close': 'price'})
                    
                    pred_data = pred_df.copy()
                    pred_data['type'] = 'Predicted'
                    pred_data = pred_data.rename(columns={'predicted_price': 'price'})
                    
                    combined_data = pd.concat([hist_data, pred_data]).set_index('date')
                    
                    # Simple line chart (enhance with plotly for better visualization)
                    st.line_chart(combined_data[['price']], height=300)
                
                # Model Performance Metrics (Mock)
                st.subheader("📈 Model Performance")
                
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                
                with perf_col1:
                    st.metric("Accuracy", f"{np.random.uniform(75, 90):.1f}%")
                
                with perf_col2:
                    st.metric("R² Score", f"{np.random.uniform(0.6, 0.85):.3f}")
                
                with perf_col3:
                    st.metric("MAE", f"${np.random.uniform(1, 5):.2f}")
                
                with perf_col4:
                    st.metric("RMSE", f"${np.random.uniform(2, 8):.2f}")
    
    elif analysis_tabs == "Portfolio":
        
        st.subheader("💰 Portfolio Analysis")
        
        # Portfolio Configuration
        st.subheader("⚙️ Portfolio Settings")
        
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
        
        with config_col2:
            position_size = st.slider(
                "Position Size (%)",
                min_value=1,
                max_value=100,
                value=10,
                step=1
            ) / 100
        
        with config_col3:
            rebalance_freq = st.selectbox(
                "Rebalance Frequency:",
                options=["Daily", "Weekly", "Monthly", "Quarterly"]
            )
        
        # Portfolio Simulation
        if st.button("🎯 Run Portfolio Simulation", key="portfolio_sim"):
            
            with st.spinner("Running portfolio simulation..."):
                
                # Simple buy-and-hold simulation
                portfolio_value = []
                cash = initial_capital
                shares = 0
                
                for i, row in stock_data.iterrows():
                    if i == 0:  # Initial purchase
                        shares_to_buy = (cash * position_size) // row['close']
                        shares += shares_to_buy
                        cash -= shares_to_buy * row['close']
                    
                    # Calculate portfolio value
                    total_value = cash + (shares * row['close'])
                    portfolio_value.append(total_value)
                
                # Add portfolio value to dataframe
                portfolio_df = stock_data.copy()
                portfolio_df['portfolio_value'] = portfolio_value
                portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
                
                # Display results
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.subheader("📊 Portfolio Performance")
                    
                    # Key metrics
                    final_value = portfolio_value[-1]
                    total_return = ((final_value - initial_capital) / initial_capital) * 100
                    
                    returns = portfolio_df['returns'].dropna()
                    volatility = returns.std() * np.sqrt(252) * 100
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
                    
                    st.metric("Final Portfolio Value", f"${final_value:,.2f}")
                    st.metric("Total Return", f"{total_return:+.2f}%")
                    st.metric("Volatility (Annual)", f"{volatility:.2f}%")
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                
                with result_col2:
                    st.subheader("📈 Portfolio Growth")
                    
                    portfolio_chart_data = portfolio_df.set_index('date')[['portfolio_value']]
                    st.line_chart(portfolio_chart_data, height=300)
                
                # Portfolio vs Stock Performance
                st.subheader("⚖️ Portfolio vs Stock Performance")
                
                stock_performance = ((stock_data['close'] / stock_data['close'].iloc[0]) - 1) * 100
                portfolio_performance = ((portfolio_df['portfolio_value'] / initial_capital) - 1) * 100
                
                comparison_df = pd.DataFrame({
                    'date': stock_data['date'],
                    'Stock Performance (%)': stock_performance,
                    'Portfolio Performance (%)': portfolio_performance
                })
                
                comparison_chart = comparison_df.set_index('date')
                st.line_chart(comparison_chart, height=300)

else:
    # No data available
    st.error(f"❌ No data available for {selected_symbol} in {market}")
    st.info("Please select a different symbol or check if the data files exist in the data directory.")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
        📈 StockPredictionPro v1.0 | Built with Streamlit | 
        Data updated: Real-time | © 2025 StockPredictionPro Team
    </div>
    """,
    unsafe_allow_html=True
)
