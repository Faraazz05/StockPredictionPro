"""
app/components/charts.py

Comprehensive charting components for StockPredictionPro Streamlit application.
Provides advanced financial charts with technical indicators integration,
candlestick charts, volume analysis, and interactive visualizations.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Any
import sys
from pathlib import Path
from datetime import timedelta

# Add project root to path for indicator imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your technical indicators
try:
    from src.features.indicators.momentum import RSI, MACD, StochasticOscillator
    from src.features.indicators.trend import SMA, EMA, BollingerBands
    from src.features.indicators.volatility import ATR, VIX
    from src.features.indicators.volume import VolumeProfile, OBV
    from src.features.indicators.custom import WilliamsR, CCI
except ImportError:
    # Mock indicators if not available
    RSI = MACD = SMA = EMA = ATR = None
    StochasticOscillator = BollingerBands = VIX = None
    VolumeProfile = OBV = WilliamsR = CCI = None

# ============================================
# CORE CHART FUNCTIONS
# ============================================

def plot_candlestick_with_volume(df: pd.DataFrame, 
                                title: str = "Stock Price Chart",
                                height: int = 600,
                                show_volume: bool = True) -> None:
    """
    Create candlestick chart with volume bars
    
    Args:
        df: DataFrame with OHLCV data
        title: Chart title
        height: Chart height in pixels
        show_volume: Whether to show volume subplot
    """
    if df is None or df.empty:
        st.warning("📊 No data available for candlestick chart")
        return
    
    # Validate required columns
    required_cols = ['date', 'open', 'high', 'low', 'close']
    if show_volume:
        required_cols.append('volume')
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return
    
    # Create subplots
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, 'Volume')
        )
    else:
        fig = go.Figure()
    
    # Add candlestick chart
    candlestick = go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC',
        increasing=dict(fillcolor='#00ff00', line=dict(color='#00aa00')),
        decreasing=dict(fillcolor='#ff0000', line=dict(color='#aa0000'))
    )
    
    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Add volume bars if requested
    if show_volume and 'volume' in df.columns:
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df['close'], df['open'])]
        
        volume_bars = go.Bar(
            x=df['date'],
            y=df['volume'],
            marker_color=colors,
            name='Volume',
            opacity=0.7
        )
        fig.add_trace(volume_bars, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=height,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    if not show_volume:
        fig.update_layout(title=title)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_price_with_indicators(df: pd.DataFrame,
                              indicators: Dict[str, pd.Series],
                              title: str = "Price with Technical Indicators",
                              height: int = 800) -> None:
    """
    Plot price chart with technical indicators overlay
    
    Args:
        df: DataFrame with OHLCV data
        indicators: Dict of indicator name -> pandas Series
        title: Chart title
        height: Chart height in pixels
    """
    if df is None or df.empty:
        st.warning("📈 No price data available")
        return
    
    # Create subplot structure
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price & Moving Averages', 'Momentum Indicators', 'Volume Indicators')
    )
    
    # Main price chart (candlestick)
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'], 
            low=df['low'],
            close=df['close'],
            name='Price'
        ), row=1, col=1
    )
    
    # Add indicators to appropriate subplots
    for indicator_name, indicator_data in indicators.items():
        if indicator_data is None or indicator_data.empty:
            continue
            
        # Determine which subplot to use based on indicator type
        if any(keyword in indicator_name.lower() for keyword in ['sma', 'ema', 'bollinger', 'bb']):
            # Price overlay indicators (subplot 1)
            fig.add_trace(
                go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data.values,
                    mode='lines',
                    name=indicator_name,
                    line=dict(width=2)
                ), row=1, col=1
            )
        
        elif any(keyword in indicator_name.lower() for keyword in ['rsi', 'macd', 'stoch', 'williams', 'cci']):
            # Momentum indicators (subplot 2)
            fig.add_trace(
                go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data.values,
                    mode='lines',
                    name=indicator_name,
                    line=dict(width=2)
                ), row=2, col=1
            )
        
        elif any(keyword in indicator_name.lower() for keyword in ['volume', 'obv', 'atr']):
            # Volume/volatility indicators (subplot 3)
            fig.add_trace(
                go.Scatter(
                    x=indicator_data.index,
                    y=indicator_data.values,
                    mode='lines',
                    name=indicator_name,
                    line=dict(width=2)
                ), row=3, col=1
            )
    
    # Add RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=height,
        title=title,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_technical_indicators_separate(indicators: Dict[str, pd.Series],
                                     title: str = "Technical Indicators",
                                     height: int = 500) -> None:
    """
    Plot technical indicators in separate chart
    
    Args:
        indicators: Dict of indicator name -> pandas Series
        title: Chart title
        height: Chart height in pixels
    """
    if not indicators:
        st.info("📊 No indicators to display")
        return
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    color_index = 0
    
    for indicator_name, indicator_data in indicators.items():
        if indicator_data is None or indicator_data.empty:
            continue
            
        fig.add_trace(
            go.Scatter(
                x=indicator_data.index,
                y=indicator_data.values,
                mode='lines',
                name=indicator_name,
                line=dict(color=colors[color_index % len(colors)], width=2)
            )
        )
        color_index += 1
    
    # Add reference lines for common indicators
    if any('rsi' in name.lower() for name in indicators.keys()):
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3)
    
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_white',
        hovermode='x unified',
        xaxis_title='Date',
        yaxis_title='Value'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# CORRELATION & COMPARISON CHARTS
# ============================================

def plot_correlation_heatmap(df: pd.DataFrame,
                           title: str = "Asset Correlation Matrix",
                           height: int = 500) -> None:
    """
    Create correlation heatmap for numeric columns
    
    Args:
        df: DataFrame with numeric data
        title: Chart title
        height: Chart height in pixels
    """
    if df is None or df.empty:
        st.warning("📊 No data available for correlation analysis")
        return
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        st.warning("📊 No numeric columns found for correlation")
        return
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=np.around(corr_matrix.values, decimals=2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_performance_comparison(data: Dict[str, pd.Series],
                              title: str = "Performance Comparison",
                              height: int = 400) -> None:
    """
    Compare performance of multiple assets/strategies
    
    Args:
        data: Dict of asset name -> price series
        title: Chart title
        height: Chart height in pixels
    """
    if not data:
        st.info("📈 No data available for comparison")
        return
    
    fig = go.Figure()
    
    for asset_name, price_series in data.items():
        if price_series is None or price_series.empty:
            continue
        
        # Calculate cumulative returns (normalized to start at 100)
        returns = price_series.pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod() * 100
        
        fig.add_trace(
            go.Scatter(
                x=price_series.index,
                y=cumulative_returns,
                mode='lines',
                name=asset_name,
                line=dict(width=2)
            )
        )
    
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_white',
        hovermode='x unified',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# INDICATOR CALCULATION & INTEGRATION
# ============================================

def calculate_and_plot_indicators(df: pd.DataFrame,
                                selected_indicators: List[str],
                                indicator_params: Dict[str, Dict] = None) -> None:
    """
    Calculate selected indicators and plot them
    
    Args:
        df: DataFrame with OHLCV data
        selected_indicators: List of indicator names to calculate
        indicator_params: Parameters for each indicator
    """
    if df is None or df.empty:
        st.error("📊 No data available for indicator calculation")
        return
    
    if indicator_params is None:
        indicator_params = {}
    
    calculated_indicators = {}
    
    # Calculate each selected indicator
    for indicator in selected_indicators:
        try:
            if indicator.lower() == 'rsi' and RSI is not None:
                period = indicator_params.get('rsi', {}).get('period', 14)
                rsi_values = calculate_rsi(df['close'], period=period)
                calculated_indicators['RSI'] = rsi_values
            
            elif indicator.lower() == 'sma' and SMA is not None:
                period = indicator_params.get('sma', {}).get('period', 20)
                sma_values = calculate_sma(df['close'], period=period)
                calculated_indicators[f'SMA_{period}'] = sma_values
            
            elif indicator.lower() == 'ema' and EMA is not None:
                period = indicator_params.get('ema', {}).get('period', 12)
                ema_values = calculate_ema(df['close'], period=period)
                calculated_indicators[f'EMA_{period}'] = ema_values
            
            elif indicator.lower() == 'macd' and MACD is not None:
                macd_data = calculate_macd(df['close'])
                if isinstance(macd_data, pd.DataFrame):
                    calculated_indicators['MACD'] = macd_data['macd']
                    calculated_indicators['MACD_Signal'] = macd_data['signal']
            
            elif indicator.lower() == 'bollinger' and BollingerBands is not None:
                bb_data = calculate_bollinger_bands(df['close'])
                if isinstance(bb_data, pd.DataFrame):
                    calculated_indicators['BB_Upper'] = bb_data['upper']
                    calculated_indicators['BB_Lower'] = bb_data['lower']
                    calculated_indicators['BB_Middle'] = bb_data['middle']
        
        except Exception as e:
            st.warning(f"⚠️ Could not calculate {indicator}: {str(e)}")
    
    if calculated_indicators:
        plot_price_with_indicators(df, calculated_indicators)
    else:
        st.warning("📊 No indicators were successfully calculated")


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (fallback implementation)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average (fallback implementation)"""
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int = 12) -> pd.Series:
    """Calculate Exponential Moving Average (fallback implementation)"""
    return prices.ewm(span=period).mean()


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD (fallback implementation)"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'macd': macd,
        'signal': signal_line,
        'histogram': histogram
    })


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
    """Calculate Bollinger Bands (fallback implementation)"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return pd.DataFrame({
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band
    })

# ============================================
# ADVANCED CHARTS
# ============================================

def plot_volume_profile(df: pd.DataFrame,
                       title: str = "Volume Profile",
                       height: int = 600) -> None:
    """
    Create volume profile chart
    
    Args:
        df: DataFrame with OHLCV data
        title: Chart title
        height: Chart height in pixels
    """
    if df is None or df.empty or 'volume' not in df.columns:
        st.warning("📊 No volume data available for volume profile")
        return
    
    # Create price bins
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_bins = np.linspace(price_min, price_max, 50)
    
    # Calculate volume at each price level
    volume_at_price = []
    
    for i in range(len(price_bins) - 1):
        bin_low = price_bins[i]
        bin_high = price_bins[i + 1]
        
        # Find rows where price range overlaps with bin
        mask = (df['low'] <= bin_high) & (df['high'] >= bin_low)
        volume_sum = df.loc[mask, 'volume'].sum()
        volume_at_price.append(volume_sum)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=volume_at_price,
            y=price_bins[:-1],
            orientation='h',
            name='Volume',
            marker=dict(color='rgba(0, 100, 200, 0.6)')
        )
    )
    
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_white',
        xaxis_title='Volume',
        yaxis_title='Price'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_market_breadth(data: Dict[str, pd.DataFrame],
                       title: str = "Market Breadth Analysis",
                       height: int = 500) -> None:
    """
    Plot market breadth indicators
    
    Args:
        data: Dict of market data for multiple symbols
        title: Chart title
        height: Chart height in pixels
    """
    if not data:
        st.warning("📊 No market data available for breadth analysis")
        return
    
    # Calculate advance/decline ratio
    dates = None
    advancing = []
    declining = []
    
    for symbol, df in data.items():
        if df is None or df.empty:
            continue
        
        if dates is None:
            dates = df['date']
        
        # Calculate daily changes
        daily_change = df['close'].diff()
        advancing.append((daily_change > 0).astype(int))
        declining.append((daily_change < 0).astype(int))
    
    if not advancing:
        st.warning("📊 Insufficient data for market breadth calculation")
        return
    
    # Sum across all symbols
    total_advancing = pd.concat(advancing, axis=1).sum(axis=1)
    total_declining = pd.concat(declining, axis=1).sum(axis=1)
    
    # Calculate advance/decline ratio
    ad_ratio = total_advancing / (total_declining + 1)  # Add 1 to avoid division by zero
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ad_ratio,
            mode='lines',
            name='Advance/Decline Ratio',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add reference line at 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_white',
        xaxis_title='Date',
        yaxis_title='A/D Ratio'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# CHART UTILITY FUNCTIONS
# ============================================

def create_chart_controls() -> Dict[str, Any]:
    """
    Create interactive controls for chart customization
    
    Returns:
        Dict with selected chart options
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            options=["Candlestick", "Line", "Area"],
            index=0
        )
    
    with col2:
        time_period = st.selectbox(
            "Time Period",
            options=["1D", "5D", "1M", "3M", "6M", "1Y", "2Y", "5Y", "Max"],
            index=3
        )
    
    with col3:
        indicators = st.multiselect(
            "Technical Indicators",
            options=["RSI", "MACD", "SMA", "EMA", "Bollinger Bands", "Volume"],
            default=["SMA"]
        )
    
    return {
        "chart_type": chart_type,
        "time_period": time_period,
        "indicators": indicators
    }


def filter_data_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Filter DataFrame by time period
    
    Args:
        df: DataFrame with date column
        period: Time period string ("1D", "1M", etc.)
        
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty or 'date' not in df.columns:
        return df
    
    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Calculate cutoff date
    end_date = df['date'].max()
    
    if period == "1D":
        start_date = end_date - timedelta(days=1)
    elif period == "5D":
        start_date = end_date - timedelta(days=5)
    elif period == "1M":
        start_date = end_date - timedelta(days=30)
    elif period == "3M":
        start_date = end_date - timedelta(days=90)
    elif period == "6M":
        start_date = end_date - timedelta(days=180)
    elif period == "1Y":
        start_date = end_date - timedelta(days=365)
    elif period == "2Y":
        start_date = end_date - timedelta(days=730)
    elif period == "5Y":
        start_date = end_date - timedelta(days=1825)
    else:  # "Max"
        return df
    
    # Filter data
    return df[df['date'] >= start_date]

# ============================================
# EXAMPLE USAGE PATTERNS
# ============================================

if __name__ == "__main__":
    """
    Example usage patterns:
    
    # Basic candlestick chart
    plot_candlestick_with_volume(stock_df, "AAPL Stock Price")
    
    # Chart with technical indicators
    indicators = {
        'RSI': calculate_rsi(stock_df['close']),
        'SMA_20': calculate_sma(stock_df['close'], 20),
        'SMA_50': calculate_sma(stock_df['close'], 50)
    }
    plot_price_with_indicators(stock_df, indicators)
    
    # Interactive controls
    chart_options = create_chart_controls()
    filtered_df = filter_data_by_period(stock_df, chart_options['time_period'])
    
    # Performance comparison
    performance_data = {
        'AAPL': stock_df_aapl['close'],
        'MSFT': stock_df_msft['close'],
        'GOOGL': stock_df_googl['close']
    }
    plot_performance_comparison(performance_data)
    
    # Correlation analysis
    plot_correlation_heatmap(combined_df)
    """
    pass
