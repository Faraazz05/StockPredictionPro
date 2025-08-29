"""
app/pages/03_📈_Chart_Analysis.py

Advanced chart analysis page for StockPredictionPro.
Provides interactive charts, technical overlays, trade visualization,
drawing tools, and comprehensive market analysis capabilities.

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

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your components
from app.components.filters import (
    filter_symbols, filter_date_range, filter_technical_indicators,
    create_data_filter_panel, filter_categorical
)
from app.components.charts import (
    plot_candlestick_with_volume,
    plot_price_with_indicators,
    plot_technical_indicators_separate,
    render_correlation_heatmap,
    plot_volume_profile,
    create_chart_controls
)
from app.components.metrics import display_trading_metrics, create_metrics_grid
from app.components.tables import display_stock_data_table, create_download_button
from app.components.alerts import get_alert_manager, generate_simple_signal_alerts
from app.styles.themes import apply_custom_theme

# ============================================
# DATA LOADING FUNCTIONS
# ============================================

def load_stock_data_for_charting(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Load comprehensive stock data for chart analysis"""
    # TODO: Replace with real data loading from your src/data modules
    np.random.seed(hash(symbol) % 2**32)
    
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)
    
    # Generate realistic stock data with volatility
    base_price = {"AAPL": 175.0, "MSFT": 342.0, "GOOGL": 2650.0, "TSLA": 245.0, "NVDA": 485.0}.get(symbol, 150.0)
    
    # More realistic price generation with trending and volatility clusters
    trend = np.cumsum(np.random.normal(0.001, 0.005, n_days))
    volatility = np.random.uniform(0.01, 0.03, n_days)
    returns = np.random.normal(trend, volatility)
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(max(prices[-1] * (1 + ret), 1.0))
    
    # Create comprehensive OHLCV data
    data = pd.DataFrame({
        'date': dates,
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0.005, 0.015))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0.005, 0.015))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(14, 0.5, n_days).astype(int)  # More realistic volume distribution
    })
    
    # Ensure valid OHLC relationships
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Add derived fields
    data['returns'] = data['close'].pct_change()
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    
    return data

def calculate_comprehensive_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculate a comprehensive set of technical indicators"""
    indicators = {}
    
    # Moving Averages
    indicators['SMA_10'] = df['close'].rolling(window=10).mean()
    indicators['SMA_20'] = df['close'].rolling(window=20).mean()
    indicators['SMA_50'] = df['close'].rolling(window=50).mean()
    indicators['EMA_12'] = df['close'].ewm(span=12).mean()
    indicators['EMA_26'] = df['close'].ewm(span=26).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    indicators['MACD'] = indicators['EMA_12'] - indicators['EMA_26']
    indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9).mean()
    indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    sma = df['close'].rolling(window=bb_period).mean()
    std = df['close'].rolling(window=bb_period).std()
    indicators['BB_Upper'] = sma + (std * bb_std)
    indicators['BB_Lower'] = sma - (std * bb_std)
    indicators['BB_Middle'] = sma
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    indicators['Stoch_K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    indicators['Stoch_D'] = indicators['Stoch_K'].rolling(window=3).mean()
    
    # Volume indicators
    indicators['Volume_SMA'] = df['volume'].rolling(window=20).mean()
    indicators['Volume_Ratio'] = df['volume'] / indicators['Volume_SMA']
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    indicators['ATR'] = true_range.rolling(window=14).mean()
    
    return indicators

def generate_trade_signals(df: pd.DataFrame, indicators: Dict[str, pd.Series]) -> pd.DataFrame:
    """Generate sample trade signals based on indicators"""
    signals = []
    
    for i in range(1, len(df)):
        # Simple moving average crossover signals
        if ('SMA_10' in indicators and 'SMA_20' in indicators and 
            not pd.isna(indicators['SMA_10'].iloc[i]) and not pd.isna(indicators['SMA_20'].iloc[i])):
            
            if (indicators['SMA_10'].iloc[i] > indicators['SMA_20'].iloc[i] and 
                indicators['SMA_10'].iloc[i-1] <= indicators['SMA_20'].iloc[i-1]):
                signals.append({
                    'date': df['date'].iloc[i],
                    'type': 'Buy',
                    'price': df['close'].iloc[i],
                    'signal': 'SMA Crossover',
                    'strength': 'Medium'
                })
            
            elif (indicators['SMA_10'].iloc[i] < indicators['SMA_20'].iloc[i] and 
                  indicators['SMA_10'].iloc[i-1] >= indicators['SMA_20'].iloc[i-1]):
                signals.append({
                    'date': df['date'].iloc[i],
                    'type': 'Sell',
                    'price': df['close'].iloc[i],
                    'signal': 'SMA Crossover',
                    'strength': 'Medium'
                })
        
        # RSI signals
        if 'RSI' in indicators and not pd.isna(indicators['RSI'].iloc[i]):
            if indicators['RSI'].iloc[i] < 30 and indicators['RSI'].iloc[i-1] >= 30:
                signals.append({
                    'date': df['date'].iloc[i],
                    'type': 'Buy',
                    'price': df['close'].iloc[i],
                    'signal': 'RSI Oversold',
                    'strength': 'Strong'
                })
            elif indicators['RSI'].iloc[i] > 70 and indicators['RSI'].iloc[i-1] <= 70:
                signals.append({
                    'date': df['date'].iloc[i],
                    'type': 'Sell',
                    'price': df['close'].iloc[i],
                    'signal': 'RSI Overbought',
                    'strength': 'Strong'
                })
    
    return pd.DataFrame(signals)

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_advanced_candlestick_chart(df: pd.DataFrame, 
                                    indicators: Dict[str, pd.Series],
                                    signals: pd.DataFrame,
                                    symbol: str) -> None:
    """Create advanced candlestick chart with indicators and signals"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(
            f'{symbol} Price Chart with Indicators',
            'RSI',
            'MACD', 
            'Volume'
        )
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing=dict(fillcolor='#00C851', line=dict(color='#00C851')),
            decreasing=dict(fillcolor='#ff4444', line=dict(color='#ff4444'))
        ), row=1, col=1
    )
    
    # Add moving averages
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ma_indicators = ['SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26']
    
    for i, indicator in enumerate(ma_indicators):
        if indicator in indicators:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=indicators[indicator],
                    mode='lines',
                    name=indicator,
                    line=dict(color=colors[i % len(colors)], width=1.5),
                    opacity=0.8
                ), row=1, col=1
            )
    
    # Bollinger Bands
    if all(bb in indicators for bb in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=indicators['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=indicators['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ), row=1, col=1
        )
    
    # Add trade signals
    if not signals.empty:
        buy_signals = signals[signals['type'] == 'Buy']
        sell_signals = signals[signals['type'] == 'Sell']
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['date'],
                    y=buy_signals['price'],
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    )
                ), row=1, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['date'],
                    y=sell_signals['price'],
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    )
                ), row=1, col=1
            )
    
    # RSI
    if 'RSI' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=indicators['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ), row=2, col=1
        )
        
        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
    
    # MACD
    if all(macd in indicators for macd in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=indicators['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ), row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=indicators['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=2)
            ), row=3, col=1
        )
        
        # MACD Histogram
        colors = ['green' if x >= 0 else 'red' for x in indicators['MACD_Histogram']]
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=indicators['MACD_Histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.7
            ), row=3, col=1
        )
    
    # Volume
    volume_colors = ['green' if close >= open else 'red' 
                    for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volume',
            marker_color=volume_colors,
            opacity=0.7
        ), row=4, col=1
    )
    
    if 'Volume_SMA' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=indicators['Volume_SMA'],
                mode='lines',
                name='Volume SMA',
                line=dict(color='orange', width=2)
            ), row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=1000,
        template='plotly_white',
        title=f'{symbol} - Advanced Technical Analysis',
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    fig.update_xaxes(matches='x')
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# MAIN PAGE FUNCTION
# ============================================

def main():
    """Main chart analysis page function"""
    
    # Apply theme
    apply_custom_theme("financial")
    
    # Page header  
    st.title("📈 Advanced Chart Analysis")
    st.markdown("Professional-grade charting with technical indicators and trade signals")
    st.markdown("---")
    
    # ============================================
    # CONTROLS & FILTERS
    # ============================================
    
    # Data filters
    available_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "SPY", "QQQ"]
    data_filters = create_data_filter_panel(available_symbols, key_prefix="chart_analysis")
    
    if not data_filters['symbols']:
        st.warning("⚠️ Please select at least one symbol to analyze")
        return
    
    # Chart controls
    st.subheader("📊 Chart Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chart_style = st.selectbox(
            "Chart Style",
            ["Candlestick", "OHLC", "Line", "Area"],
            help="Select the main chart visualization style"
        )
        
        show_volume = st.checkbox("Show Volume", value=True)
    
    with col2:
        show_signals = st.checkbox("Show Trade Signals", value=True)
        show_patterns = st.checkbox("Show Patterns", value=False)
    
    with col3:
        time_frame = st.selectbox(
            "Time Frame",
            ["Daily", "Weekly", "Monthly"],
            help="Data aggregation period"
        )
        
        show_grid = st.checkbox("Show Grid", value=True)
    
    # Indicator selection
    st.subheader("🔧 Technical Indicators")
    
    indicator_categories = {
        "Trend": ["SMA_10", "SMA_20", "SMA_50", "EMA_12", "EMA_26", "BB_Upper", "BB_Lower"],
        "Momentum": ["RSI", "MACD", "Stoch_K", "Stoch_D"],
        "Volume": ["Volume_SMA", "Volume_Ratio"],
        "Volatility": ["ATR", "BB_Upper", "BB_Lower"]
    }
    
    selected_indicators = []
    
    indicator_tabs = st.tabs(list(indicator_categories.keys()))
    
    for i, (category, indicators) in enumerate(indicator_categories.items()):
        with indicator_tabs[i]:
            for indicator in indicators:
                if st.checkbox(indicator, key=f"indicator_{indicator}"):
                    selected_indicators.append(indicator)
    
    st.markdown("---")
    
    # ============================================
    # DATA LOADING & PROCESSING
    # ============================================
    
    selected_symbol = data_filters['symbols'][0]
    
    with st.spinner(f"Loading chart data for {selected_symbol}..."):
        stock_data = load_stock_data_for_charting(
            selected_symbol,
            data_filters['start_date'],
            data_filters['end_date']
        )
    
    if stock_data.empty:
        st.error(f"❌ No data available for {selected_symbol}")
        return
    
    # Calculate indicators
    with st.spinner("Calculating technical indicators..."):
        all_indicators = calculate_comprehensive_indicators(stock_data)
        
        # Filter indicators based on user selection
        filtered_indicators = {k: v for k, v in all_indicators.items() 
                             if k in selected_indicators or k in ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram']}
    
    # Generate trade signals
    if show_signals:
        with st.spinner("Generating trade signals..."):
            trade_signals = generate_trade_signals(stock_data, all_indicators)
    else:
        trade_signals = pd.DataFrame()
    
    # ============================================
    # MAIN CHART DISPLAY
    # ============================================
    
    st.subheader(f"📈 {selected_symbol} Chart Analysis")
    
    # Display main chart
    create_advanced_candlestick_chart(
        stock_data, 
        filtered_indicators, 
        trade_signals,
        selected_symbol
    )
    
    # ============================================
    # ANALYSIS PANELS
    # ============================================
    
    # Current values panel
    if not stock_data.empty:
        st.subheader("📊 Current Market Data")
        
        latest_data = stock_data.iloc[-1]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            price_change = latest_data['close'] - stock_data['close'].iloc[-2] if len(stock_data) > 1 else 0
            price_change_pct = (price_change / stock_data['close'].iloc[-2] * 100) if len(stock_data) > 1 else 0
            st.metric(
                "Current Price", 
                f"${latest_data['close']:.2f}",
                f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
            )
        
        with col2:
            st.metric("Volume", f"{latest_data['volume']:,.0f}")
        
        with col3:
            day_range = f"${latest_data['low']:.2f} - ${latest_data['high']:.2f}"
            st.metric("Day Range", day_range)
        
        with col4:
            if 'RSI' in all_indicators and not pd.isna(all_indicators['RSI'].iloc[-1]):
                rsi_value = all_indicators['RSI'].iloc[-1]
                rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                st.metric("RSI", f"{rsi_value:.1f}", rsi_status)
            else:
                st.metric("RSI", "N/A")
        
        with col5:
            if 'MACD' in all_indicators and not pd.isna(all_indicators['MACD'].iloc[-1]):
                macd_value = all_indicators['MACD'].iloc[-1]
                macd_signal = all_indicators['MACD_Signal'].iloc[-1] if 'MACD_Signal' in all_indicators else 0
                macd_status = "Bullish" if macd_value > macd_signal else "Bearish"
                st.metric("MACD", f"{macd_value:.3f}", macd_status)
            else:
                st.metric("MACD", "N/A")
    
    # ============================================
    # TRADE SIGNALS ANALYSIS
    # ============================================
    
    if show_signals and not trade_signals.empty:
        st.subheader("🎯 Trade Signals Analysis")
        
        # Signal summary
        signal_summary = trade_signals.groupby(['type', 'signal']).size().reset_index(name='count')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Signal Distribution**")
            
            buy_signals = len(trade_signals[trade_signals['type'] == 'Buy'])
            sell_signals = len(trade_signals[trade_signals['type'] == 'Sell'])
            
            st.metric("Buy Signals", buy_signals)
            st.metric("Sell Signals", sell_signals)
        
        with col2:
            st.write("**Recent Signals**")
            
            if len(trade_signals) > 0:
                recent_signals = trade_signals.tail(5)[['date', 'type', 'signal', 'strength']]
                recent_signals['date'] = recent_signals['date'].dt.strftime('%Y-%m-%d')
                st.dataframe(recent_signals, use_container_width=True, hide_index=True)
            else:
                st.info("No signals generated for the selected period")
    
    # ============================================
    # PATTERN RECOGNITION
    # ============================================
    
    if show_patterns:
        st.subheader("🔍 Pattern Recognition")
        
        # Simple pattern detection (placeholder)
        patterns_detected = []
        
        # Detect simple patterns
        if len(stock_data) >= 20:
            recent_highs = stock_data['high'].tail(10)
            recent_lows = stock_data['low'].tail(10)
            
            if recent_highs.is_monotonic_increasing:
                patterns_detected.append("Ascending Highs")
            
            if recent_lows.is_monotonic_increasing:
                patterns_detected.append("Ascending Lows")
            
            # Support/Resistance levels
            resistance_level = stock_data['high'].rolling(window=20).max().iloc[-1]
            support_level = stock_data['low'].rolling(window=20).min().iloc[-1]
            
            patterns_detected.append(f"Resistance: ${resistance_level:.2f}")
            patterns_detected.append(f"Support: ${support_level:.2f}")
        
        if patterns_detected:
            for pattern in patterns_detected:
                st.info(f"📈 {pattern}")
        else:
            st.info("No significant patterns detected")
    
    # ============================================
    # MULTI-TIMEFRAME ANALYSIS
    # ============================================
    
    if len(data_filters['symbols']) > 1:
        st.subheader("📊 Multi-Symbol Comparison")
        
        comparison_data = {}
        
        for symbol in data_filters['symbols'][:4]:  # Limit to 4 symbols
            symbol_data = load_stock_data_for_charting(
                symbol,
                data_filters['start_date'],
                data_filters['end_date']
            )
            
            if not symbol_data.empty:
                # Normalize prices for comparison
                normalized_prices = symbol_data['close'] / symbol_data['close'].iloc[0] * 100
                comparison_data[symbol] = normalized_prices
        
        if comparison_data and len(comparison_data) > 1:
            comparison_df = pd.DataFrame(comparison_data, index=stock_data['date'])
            
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, (symbol, data) in enumerate(comparison_data.items()):
                fig.add_trace(
                    go.Scatter(
                        x=stock_data['date'],
                        y=data,
                        mode='lines',
                        name=symbol,
                        line=dict(color=colors[i % len(colors)], width=2)
                    )
                )
            
            fig.update_layout(
                title="Normalized Price Comparison (Base = 100)",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # EXPORT & ALERTS
    # ============================================
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Export Data")
        
        # Prepare export data
        export_data = stock_data.copy()
        for name, series in filtered_indicators.items():
            export_data[name] = series
        
        create_download_button(
            export_data,
            f"{selected_symbol}_chart_analysis.csv",
            "📊 Download Chart Data",
            key="chart_data_download"
        )
        
        if not trade_signals.empty:
            create_download_button(
                trade_signals,
                f"{selected_symbol}_trade_signals.csv",
                "🎯 Download Signals",
                key="signals_download"
            )
    
    with col2:
        st.subheader("🚨 Alerts")
        
        # Generate and display alerts
        alert_manager = get_alert_manager()
        
        # Generate signal-based alerts
        if not trade_signals.empty:
            recent_signals = trade_signals.tail(3)
            for _, signal in recent_signals.iterrows():
                alert_manager.add_alert(
                    f"{signal['type']} signal: {signal['signal']} for {selected_symbol}",
                    "trading_signal",
                    "success" if signal['type'] == 'Buy' else "warning",
                    selected_symbol
                )
        
        # Generate simple alerts
        generate_simple_signal_alerts(stock_data, selected_symbol, alert_manager)
        
        # Display recent alerts
        if alert_manager.alerts:
            for alert in alert_manager.alerts[-5:]:
                if alert.level.value == "success":
                    st.success(f"✅ {alert}")
                elif alert.level.value == "warning":
                    st.warning(f"⚠️ {alert}")
                else:
                    st.info(f"ℹ️ {alert}")
        else:
            st.info("No alerts at this time")
    
    # ============================================
    # FOOTER
    # ============================================
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        📈 Chart Analysis | {len(filtered_indicators)} indicators active | 
        {len(trade_signals)} signals generated | 
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE EXECUTION
# ============================================

if __name__ == "__main__":
    main()
