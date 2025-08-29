"""
app/pages/02_📊_Data_&_Indicators.py

Advanced Data & Technical Indicators Analysis for StockPredictionPro.
Fully integrated with comprehensive data fetching, processing, and analysis pipeline.

Integrates with:
- src/data/fetchers/ (yahoo_finance, alpha_vantage, polygon, quandl, fred, base_fetcher)
- src/data/processors/ (cleaner, transformer, resampler, splitter)
- src/data/ (cache, manager, validators)
- data/ directory structure (raw, processed, features, targets)

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
import os
import traceback

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your comprehensive data system
try:
    # Data fetchers
    from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
    from src.data.fetchers.alpha_vantage import AlphaVantageFetcher
    from src.data.fetchers.polygon import PolygonFetcher
    from src.data.fetchers.quandl import QuandlFetcher
    from src.data.fetchers.fred import FredFetcher
    from src.data.fetchers.base_fetcher import BaseFetcher
    
    # Data processors
    from src.data.processors.cleaner import DataCleaner
    from src.data.processors.transformer import DataTransformer
    from src.data.processors.resampler import DataResampler
    from src.data.processors.splitter import DataSplitter
    
    # Data management
    from src.data.manager import DataManager
    from src.data.cache import CacheManager
    from src.data.validators import DataValidator
    
    DATA_MODULES_AVAILABLE = True
    
except ImportError as e:
    st.error(f"Data modules not found: {e}")
    DATA_MODULES_AVAILABLE = False

# Import app components
from app.components.filters import (
    filter_symbols, filter_date_range, filter_categorical,
    create_data_filter_panel, filter_numeric
)
from app.components.charts import render_line_chart, plot_candlestick_with_volume
from app.components.metrics import (
    display_trading_metrics, create_metrics_grid,
    display_performance_summary
)
from app.components.tables import display_dataframe, create_download_button
from app.components.alerts import get_alert_manager, generate_simple_signal_alerts
from app.styles.themes import apply_custom_theme

# ============================================
# CONFIGURATION & CONSTANTS
# ============================================

# Available data sources
DATA_SOURCES = {
    "Yahoo Finance": {
        "class": "YahooFinanceFetcher",
        "description": "Free real-time and historical market data",
        "supports": ["stocks", "indices", "forex", "crypto"],
        "rate_limit": "High",
        "data_quality": "Good"
    },
    "Alpha Vantage": {
        "class": "AlphaVantageFetcher", 
        "description": "Premium financial data with fundamentals",
        "supports": ["stocks", "forex", "crypto", "indicators"],
        "rate_limit": "Medium",
        "data_quality": "Excellent"
    },
    "Polygon": {
        "class": "PolygonFetcher",
        "description": "Professional-grade market data",
        "supports": ["stocks", "options", "forex", "crypto"],
        "rate_limit": "High",
        "data_quality": "Excellent"
    },
    "Quandl": {
        "class": "QuandlFetcher",
        "description": "Economic and financial data",
        "supports": ["economic", "commodities", "rates"],
        "rate_limit": "Medium", 
        "data_quality": "High"
    },
    "FRED": {
        "class": "FredFetcher",
        "description": "Federal Reserve economic data",
        "supports": ["economic", "rates", "indicators"],
        "rate_limit": "High",
        "data_quality": "Excellent"
    }
}

# Technical indicators available
TECHNICAL_INDICATORS = {
    "Trend Indicators": {
        "SMA": {"periods": [10, 20, 50, 100, 200]},
        "EMA": {"periods": [12, 26, 50]},
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "Bollinger Bands": {"period": 20, "std": 2},
        "Parabolic SAR": {"af": 0.02, "max_af": 0.2}
    },
    "Momentum Indicators": {
        "RSI": {"period": 14},
        "Stochastic": {"k_period": 14, "d_period": 3},
        "Williams %R": {"period": 14},
        "CCI": {"period": 20},
        "ROC": {"period": 10}
    },
    "Volume Indicators": {
        "Volume SMA": {"period": 20},
        "OBV": {},
        "Volume Profile": {},
        "A/D Line": {},
        "Chaikin MF": {"period": 20}
    },
    "Volatility Indicators": {
        "ATR": {"period": 14},
        "Volatility": {"period": 20},
        "Keltner Channels": {"period": 20, "multiplier": 2}
    }
}

# Data processing options
PROCESSING_OPTIONS = {
    "Cleaning": {
        "remove_outliers": True,
        "fill_missing": "forward_fill",
        "validate_ohlc": True,
        "remove_weekends": True
    },
    "Transformation": {
        "log_returns": False,
        "normalize_volume": True,
        "add_features": True,
        "calculate_returns": True
    },
    "Resampling": {
        "frequency": "daily",
        "aggregation": "last",
        "handle_gaps": True
    }
}

# ============================================
# DATA LOADING & PROCESSING ENGINE
# ============================================

def load_comprehensive_data(symbols: List[str], 
                           start_date: date, 
                           end_date: date,
                           data_source: str = "Yahoo Finance",
                           use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """Load comprehensive market data using available data modules"""
    
    if not DATA_MODULES_AVAILABLE:
        return load_fallback_data(symbols, start_date, end_date)
    
    try:
        # Initialize data manager and cache
        data_manager = DataManager()
        cache_manager = CacheManager() if use_cache else None
        
        # Initialize selected data fetcher
        if data_source == "Yahoo Finance":
            fetcher = YahooFinanceFetcher()
        elif data_source == "Alpha Vantage":
            fetcher = AlphaVantageFetcher()
        elif data_source == "Polygon":
            fetcher = PolygonFetcher()
        elif data_source == "Quandl":
            fetcher = QuandlFetcher()
        elif data_source == "FRED":
            fetcher = FredFetcher()
        else:
            fetcher = YahooFinanceFetcher()  # Default fallback
        
        # Load data for each symbol
        data_dict = {}
        
        for symbol in symbols:
            cache_key = f"{symbol}_{start_date}_{end_date}_{data_source}"
            
            # Check cache first
            if cache_manager:
                cached_data = cache_manager.get(cache_key)
                if cached_data is not None:
                    data_dict[symbol] = cached_data
                    continue
            
            # Fetch fresh data
            with st.spinner(f"Fetching {symbol} data from {data_source}..."):
                raw_data = fetcher.fetch_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval='1d'
                )
                
                if raw_data is not None and not raw_data.empty:
                    data_dict[symbol] = raw_data
                    
                    # Cache the data
                    if cache_manager:
                        cache_manager.set(cache_key, raw_data, ttl=3600)  # 1 hour TTL
                else:
                    st.warning(f"No data available for {symbol}")
        
        return data_dict
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return load_fallback_data(symbols, start_date, end_date)

def load_fallback_data(symbols: List[str], start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
    """Generate fallback data when data modules are not available"""
    data_dict = {}
    
    for symbol in symbols:
        # Generate realistic market data
        np.random.seed(hash(symbol) % 2**32)
        dates = pd.date_range(start_date, end_date, freq='D')
        n_days = len(dates)
        
        # Base price for each symbol
        base_prices = {
            "AAPL": 175.0, "MSFT": 342.0, "GOOGL": 2650.0, "TSLA": 245.0, "NVDA": 485.0,
            "AMZN": 3380.0, "META": 310.0, "RELIANCE": 2800.0, "TCS": 3500.0, "INFY": 1400.0
        }
        base_price = base_prices.get(symbol, 150.0)
        
        # Generate price series with realistic volatility
        returns = np.random.normal(0.0008, 0.025, n_days)
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(max(prices[-1] * (1 + ret), 1.0))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0.005, 0.018))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0.005, 0.018))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(14.0, 0.7, n_days).astype(int)
        })
        
        # Ensure valid OHLC relationships
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        data_dict[symbol] = data
    
    return data_dict

def process_market_data(raw_data: pd.DataFrame, 
                       processing_options: Dict[str, Any]) -> pd.DataFrame:
    """Process raw market data using available processing modules"""
    
    if not DATA_MODULES_AVAILABLE:
        return process_fallback_data(raw_data, processing_options)
    
    try:
        processed_data = raw_data.copy()
        
        # Data cleaning
        if processing_options["Cleaning"]["remove_outliers"]:
            cleaner = DataCleaner()
            processed_data = cleaner.remove_outliers(processed_data)
            processed_data = cleaner.validate_ohlc_data(processed_data)
            processed_data = cleaner.fill_missing_values(
                processed_data, 
                method=processing_options["Cleaning"]["fill_missing"]
            )
        
        # Data transformation
        transformer = DataTransformer()
        
        if processing_options["Transformation"]["calculate_returns"]:
            processed_data = transformer.add_return_features(processed_data)
        
        if processing_options["Transformation"]["log_returns"]:
            processed_data = transformer.add_log_returns(processed_data)
        
        if processing_options["Transformation"]["add_features"]:
            processed_data = transformer.add_time_features(processed_data)
            processed_data = transformer.add_lag_features(processed_data, lags=[1, 5, 10])
        
        # Data resampling if needed
        if processing_options["Resampling"]["frequency"] != "daily":
            resampler = DataResampler()
            processed_data = resampler.resample_data(
                processed_data,
                frequency=processing_options["Resampling"]["frequency"],
                aggregation=processing_options["Resampling"]["aggregation"]
            )
        
        return processed_data
        
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return process_fallback_data(raw_data, processing_options)

def process_fallback_data(raw_data: pd.DataFrame, processing_options: Dict[str, Any]) -> pd.DataFrame:
    """Fallback data processing when modules are not available"""
    processed_data = raw_data.copy()
    
    # Basic return calculation
    if processing_options["Transformation"]["calculate_returns"]:
        processed_data['returns'] = processed_data['close'].pct_change()
        processed_data['log_returns'] = np.log(processed_data['close'] / processed_data['close'].shift(1))
    
    # Basic feature engineering
    if processing_options["Transformation"]["add_features"]:
        processed_data['price_change'] = processed_data['close'].diff()
        processed_data['volume_change'] = processed_data['volume'].pct_change()
        processed_data['high_low_ratio'] = processed_data['high'] / processed_data['low']
        processed_data['close_open_ratio'] = processed_data['close'] / processed_data['open']
    
    # Fill missing values
    processed_data = processed_data.fillna(method='ffill')
    
    return processed_data

# ============================================
# TECHNICAL INDICATORS ENGINE
# ============================================

def calculate_comprehensive_indicators(data: pd.DataFrame, 
                                     selected_indicators: Dict[str, List[str]]) -> Dict[str, pd.Series]:
    """Calculate comprehensive technical indicators"""
    indicators = {}
    
    try:
        # Trend Indicators
        if "SMA" in selected_indicators.get("Trend Indicators", []):
            for period in TECHNICAL_INDICATORS["Trend Indicators"]["SMA"]["periods"]:
                indicators[f"SMA_{period}"] = data['close'].rolling(window=period).mean()
        
        if "EMA" in selected_indicators.get("Trend Indicators", []):
            for period in TECHNICAL_INDICATORS["Trend Indicators"]["EMA"]["periods"]:
                indicators[f"EMA_{period}"] = data['close'].ewm(span=period).mean()
        
        if "MACD" in selected_indicators.get("Trend Indicators", []):
            macd_config = TECHNICAL_INDICATORS["Trend Indicators"]["MACD"]
            ema_fast = data['close'].ewm(span=macd_config["fast"]).mean()
            ema_slow = data['close'].ewm(span=macd_config["slow"]).mean()
            indicators['MACD'] = ema_fast - ema_slow
            indicators['MACD_Signal'] = indicators['MACD'].ewm(span=macd_config["signal"]).mean()
            indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']
        
        if "Bollinger Bands" in selected_indicators.get("Trend Indicators", []):
            bb_config = TECHNICAL_INDICATORS["Trend Indicators"]["Bollinger Bands"]
            sma = data['close'].rolling(window=bb_config["period"]).mean()
            std = data['close'].rolling(window=bb_config["period"]).std()
            indicators['BB_Upper'] = sma + (bb_config["std"] * std)
            indicators['BB_Lower'] = sma - (bb_config["std"] * std)
            indicators['BB_Middle'] = sma
        
        # Momentum Indicators
        if "RSI" in selected_indicators.get("Momentum Indicators", []):
            rsi_period = TECHNICAL_INDICATORS["Momentum Indicators"]["RSI"]["period"]
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))
        
        if "Stochastic" in selected_indicators.get("Momentum Indicators", []):
            stoch_config = TECHNICAL_INDICATORS["Momentum Indicators"]["Stochastic"]
            low_min = data['low'].rolling(window=stoch_config["k_period"]).min()
            high_max = data['high'].rolling(window=stoch_config["k_period"]).max()
            indicators['Stoch_K'] = 100 * ((data['close'] - low_min) / (high_max - low_min))
            indicators['Stoch_D'] = indicators['Stoch_K'].rolling(window=stoch_config["d_period"]).mean()
        
        # Volume Indicators
        if "Volume SMA" in selected_indicators.get("Volume Indicators", []):
            vol_period = TECHNICAL_INDICATORS["Volume Indicators"]["Volume SMA"]["period"]
            indicators['Volume_SMA'] = data['volume'].rolling(window=vol_period).mean()
            indicators['Volume_Ratio'] = data['volume'] / indicators['Volume_SMA']
        
        if "OBV" in selected_indicators.get("Volume Indicators", []):
            obv = [0]
            for i in range(1, len(data)):
                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                    obv.append(obv[-1] + data['volume'].iloc[i])
                elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                    obv.append(obv[-1] - data['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            indicators['OBV'] = pd.Series(obv, index=data.index)
        
        # Volatility Indicators
        if "ATR" in selected_indicators.get("Volatility Indicators", []):
            atr_period = TECHNICAL_INDICATORS["Volatility Indicators"]["ATR"]["period"]
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            indicators['ATR'] = true_range.rolling(window=atr_period).mean()
        
        return indicators
        
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return {}

# ============================================
# VISUALIZATION ENGINE
# ============================================

def create_comprehensive_chart(data: pd.DataFrame, 
                              indicators: Dict[str, pd.Series],
                              symbol: str,
                              chart_options: Dict[str, Any]) -> None:
    """Create comprehensive multi-panel chart with indicators"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Determine number of subplots needed
    subplot_count = 1  # Main price chart
    has_volume = chart_options.get("show_volume", True)
    has_momentum = any(indicator in indicators for indicator in ['RSI', 'Stoch_K', 'Stoch_D'])
    has_macd = 'MACD' in indicators
    
    if has_volume:
        subplot_count += 1
    if has_momentum:
        subplot_count += 1
    if has_macd:
        subplot_count += 1
    
    # Create subplot titles
    subplot_titles = [f'{symbol} Price Chart']
    row_heights = [0.5]
    
    if has_volume:
        subplot_titles.append('Volume')
        row_heights.append(0.15)
    if has_momentum:
        subplot_titles.append('Momentum Indicators')
        row_heights.append(0.15)
    if has_macd:
        subplot_titles.append('MACD')
        row_heights.append(0.2)
    
    # Normalize row heights
    row_heights = [h / sum(row_heights) for h in row_heights]
    
    fig = make_subplots(
        rows=subplot_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )
    
    # Main price chart
    if chart_options.get("chart_type", "Candlestick") == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=data['date'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing=dict(fillcolor='#00C851', line=dict(color='#00C851')),
                decreasing=dict(fillcolor='#ff4444', line=dict(color='#ff4444'))
            ), row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ), row=1, col=1
        )
    
    # Add trend indicators to main chart
    colors = ['orange', 'purple', 'green', 'red', 'brown']
    color_idx = 0
    
    for indicator_name, indicator_series in indicators.items():
        if indicator_name.startswith(('SMA', 'EMA', 'BB')):
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=indicator_series,
                    mode='lines',
                    name=indicator_name,
                    line=dict(color=colors[color_idx % len(colors)], width=1.5),
                    opacity=0.8
                ), row=1, col=1
            )
            color_idx += 1
    
    current_row = 2
    
    # Volume chart
    if has_volume:
        volume_colors = ['green' if close >= open else 'red' 
                        for close, open in zip(data['close'], data['open'])]
        
        fig.add_trace(
            go.Bar(
                x=data['date'],
                y=data['volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7
            ), row=current_row, col=1
        )
        
        # Add volume indicators
        if 'Volume_SMA' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=indicators['Volume_SMA'],
                    mode='lines',
                    name='Volume SMA',
                    line=dict(color='orange', width=2)
                ), row=current_row, col=1
            )
        
        current_row += 1
    
    # Momentum indicators
    if has_momentum:
        if 'RSI' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=indicators['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ), row=current_row, col=1
            )
            
            # RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=current_row, col=1)
        
        if 'Stoch_K' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=indicators['Stoch_K'],
                    mode='lines',
                    name='Stoch %K',
                    line=dict(color='blue', width=2)
                ), row=current_row, col=1
            )
            
            if 'Stoch_D' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data['date'],
                        y=indicators['Stoch_D'],
                        mode='lines',
                        name='Stoch %D',
                        line=dict(color='red', width=2)
                    ), row=current_row, col=1
                )
        
        current_row += 1
    
    # MACD
    if has_macd:
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=indicators['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ), row=current_row, col=1
        )
        
        if 'MACD_Signal' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=indicators['MACD_Signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='red', width=2)
                ), row=current_row, col=1
            )
        
        if 'MACD_Histogram' in indicators:
            colors = ['green' if x >= 0 else 'red' for x in indicators['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=data['date'],
                    y=indicators['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color=colors,
                    opacity=0.6
                ), row=current_row, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=800 + (subplot_count - 1) * 200,
        title=f'{symbol} - Comprehensive Technical Analysis',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    fig.update_xaxes(matches='x')
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# MAIN PAGE FUNCTION
# ============================================

def main():
    """Main data & indicators page function"""
    
    # Apply theme
    apply_custom_theme("financial")
    
    # Page header
    st.title("📊 Advanced Data & Technical Indicators")
    st.markdown("Professional market data analysis with comprehensive indicator calculations")
    
    if not DATA_MODULES_AVAILABLE:
        st.warning("⚠️ **Data modules not fully available.** Using fallback data generation for demonstration.")
    
    st.markdown("---")
    
    # ============================================
    # DATA SOURCE & SYMBOL CONFIGURATION
    # ============================================
    
    st.subheader("🔧 Data Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data source selection
        selected_data_source = filter_categorical(
            "Select Data Source",
            list(DATA_SOURCES.keys()),
            multi=False,
            key="data_source"
        )
        
        # Symbol selection with regional support
        st.write("**Symbol Selection**")
        region = st.selectbox(
            "Market Region",
            ["US", "Indian", "Global"],
            help="Select market region for symbol selection"
        )
        
        if region == "US":
            available_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "ORCL", "NFLX", "CRM"]
        elif region == "Indian":
            available_symbols = ["RELIANCE", "TCS", "INFY", "HINDUNILVR", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "HDFCBANK"]
        else:
            available_symbols = ["AAPL", "MSFT", "GOOGL", "RELIANCE", "TCS", "ASML", "TSM", "SAP", "SHOP", "BABA"]
        
        selected_symbols = filter_categorical(
            "Select Symbols",
            available_symbols,
            multi=True,
            key="indicators_symbols"
        )
    
    with col2:
        # Date range selection
        start_date, end_date = filter_date_range(
            default_days=252,  # 1 trading year
            key="indicators_dates"
        )
        
        # Advanced options
        use_cache = st.checkbox("Use Data Cache", value=True, help="Cache data for faster loading")
        auto_refresh = st.checkbox("Auto Refresh", value=False, help="Automatically refresh data")
    
    if not selected_symbols:
        st.warning("⚠️ Please select at least one symbol to analyze")
        return
    
    # Display data source information
    if selected_data_source:
        source_info = DATA_SOURCES[selected_data_source]
        st.info(f"**{selected_data_source}:** {source_info['description']} | "
               f"Quality: {source_info['data_quality']} | Rate Limit: {source_info['rate_limit']}")
    
    # ============================================
    # TECHNICAL INDICATORS SELECTION
    # ============================================
    
    st.subheader("⚙️ Technical Indicators Configuration")
    
    # Create tabs for different indicator categories
    indicator_tabs = st.tabs(list(TECHNICAL_INDICATORS.keys()))
    selected_indicators = {}
    
    for i, (category, indicators) in enumerate(TECHNICAL_INDICATORS.items()):
        with indicator_tabs[i]:
            selected_indicators[category] = []
            
            st.write(f"**{category}**")
            
            for indicator_name in indicators.keys():
                if st.checkbox(indicator_name, key=f"indicator_{category}_{indicator_name}"):
                    selected_indicators[category].append(indicator_name)
    
    # Processing options
    with st.expander("🛠️ Data Processing Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Data Cleaning**")
            clean_outliers = st.checkbox("Remove Outliers", value=True)
            fill_missing = st.selectbox("Fill Missing Values", ["forward_fill", "backward_fill", "interpolate"])
            validate_ohlc = st.checkbox("Validate OHLC", value=True)
        
        with col2:
            st.write("**Data Transformation**")
            calc_returns = st.checkbox("Calculate Returns", value=True)
            log_returns = st.checkbox("Log Returns", value=False)
            add_features = st.checkbox("Add Time Features", value=True)
        
        with col3:
            st.write("**Advanced Options**")
            frequency = st.selectbox("Resampling Frequency", ["daily", "weekly", "monthly"], index=0)
            normalize_volume = st.checkbox("Normalize Volume", value=True)
            export_processed = st.checkbox("Export Processed Data", value=False)
    
    # Update processing options
    processing_opts = PROCESSING_OPTIONS.copy()
    processing_opts["Cleaning"]["remove_outliers"] = clean_outliers
    processing_opts["Cleaning"]["fill_missing"] = fill_missing
    processing_opts["Cleaning"]["validate_ohlc"] = validate_ohlc
    processing_opts["Transformation"]["calculate_returns"] = calc_returns
    processing_opts["Transformation"]["log_returns"] = log_returns
    processing_opts["Transformation"]["add_features"] = add_features
    processing_opts["Transformation"]["normalize_volume"] = normalize_volume
    processing_opts["Resampling"]["frequency"] = frequency
    
    st.markdown("---")
    
    # ============================================
    # DATA LOADING & PROCESSING
    # ============================================
    
    with st.spinner("Loading and processing market data..."):
        
        # Load comprehensive data
        raw_data_dict = load_comprehensive_data(
            selected_symbols,
            start_date,
            end_date, 
            selected_data_source,
            use_cache
        )
        
        if not raw_data_dict:
            st.error("❌ No data could be loaded for the selected symbols")
            return
        
        # Process data for each symbol
        processed_data_dict = {}
        indicators_dict = {}
        
        for symbol in selected_symbols:
            if symbol in raw_data_dict:
                # Process the data
                processed_data = process_market_data(raw_data_dict[symbol], processing_opts)
                processed_data_dict[symbol] = processed_data
                
                # Calculate indicators
                if any(selected_indicators.values()):  # If any indicators selected
                    symbol_indicators = calculate_comprehensive_indicators(processed_data, selected_indicators)
                    indicators_dict[symbol] = symbol_indicators
                else:
                    indicators_dict[symbol] = {}
    
    # ============================================
    # SUMMARY DASHBOARD
    # ============================================
    
    st.subheader("📈 Data Summary Dashboard")
    
    # Create summary metrics
    total_symbols = len(processed_data_dict)
    total_indicators = sum(len(indicators) for indicators in indicators_dict.values())
    date_range_days = (end_date - start_date).days
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Symbols Loaded", total_symbols)
    with col2:
        st.metric("Indicators Calculated", total_indicators)
    with col3:
        st.metric("Data Points", sum(len(df) for df in processed_data_dict.values()))
    with col4:
        st.metric("Date Range (Days)", date_range_days)
    
    # ============================================
    # INDIVIDUAL SYMBOL ANALYSIS
    # ============================================
    
    for symbol in selected_symbols:
        if symbol not in processed_data_dict:
            continue
            
        st.markdown("---")
        st.subheader(f"📊 {symbol} - Technical Analysis")
        
        data = processed_data_dict[symbol]
        indicators = indicators_dict.get(symbol, {})
        
        # Current metrics
        if not data.empty:
            latest_data = data.iloc[-1]
            previous_data = data.iloc[-2] if len(data) > 1 else latest_data
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                price_change = latest_data['close'] - previous_data['close']
                price_change_pct = (price_change / previous_data['close']) * 100
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
                if 'returns' in data.columns:
                    volatility = data['returns'].std() * np.sqrt(252) * 100
                    st.metric("Annualized Volatility", f"{volatility:.1f}%")
        
        # Chart configuration
        chart_options = {
            "chart_type": st.selectbox(f"Chart Type for {symbol}", ["Candlestick", "Line"], key=f"chart_{symbol}"),
            "show_volume": st.checkbox(f"Show Volume for {symbol}", value=True, key=f"vol_{symbol}"),
            "height": 800
        }
        
        # Create comprehensive chart
        if indicators:
            create_comprehensive_chart(data, indicators, symbol, chart_options)
        else:
            # Basic price chart
            plot_candlestick_with_volume(data, f"{symbol} Price Chart", chart_options["show_volume"])
        
        # Indicator values table
        if indicators:
            st.subheader(f"📋 {symbol} - Latest Indicator Values")
            
            indicator_values = {}
            for ind_name, ind_series in indicators.items():
                if not ind_series.empty and not pd.isna(ind_series.iloc[-1]):
                    indicator_values[ind_name] = ind_series.iloc[-1]
            
            if indicator_values:
                # Create a nicely formatted table
                ind_df = pd.DataFrame(list(indicator_values.items()), columns=['Indicator', 'Current Value'])
                ind_df['Current Value'] = ind_df['Current Value'].apply(lambda x: f"{x:.4f}")
                st.dataframe(ind_df, use_container_width=True, hide_index=True)
        
        # Performance metrics
        if 'returns' in data.columns:
            st.subheader(f"📊 {symbol} - Performance Metrics")
            display_trading_metrics(data['returns'], f"{symbol} Performance")
    
    # ============================================
    # MULTI-SYMBOL ANALYSIS
    # ============================================
    
    if len(selected_symbols) > 1:
        st.markdown("---")
        st.subheader("🔍 Multi-Symbol Analysis")
        
        # Correlation analysis
        price_data = {}
        return_data = {}
        
        for symbol in selected_symbols:
            if symbol in processed_data_dict:
                data = processed_data_dict[symbol]
                price_data[symbol] = data['close']
                if 'returns' in data.columns:
                    return_data[symbol] = data['returns']
        
        if price_data:
            # Price correlation
            price_df = pd.DataFrame(price_data)
            price_corr = price_df.corr()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Price Correlation Matrix**")
                st.dataframe(price_corr.round(3), use_container_width=True)
            
            with col2:
                if return_data:
                    return_df = pd.DataFrame(return_data)
                    return_corr = return_df.corr()
                    st.write("**Returns Correlation Matrix**")
                    st.dataframe(return_corr.round(3), use_container_width=True)
            
            # Performance comparison
            st.subheader("📈 Performance Comparison")
            
            # Normalize prices for comparison
            normalized_prices = price_df.div(price_df.iloc[0]) * 100
            
            import plotly.graph_objects as go
            fig = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            for i, symbol in enumerate(normalized_prices.columns):
                fig.add_trace(
                    go.Scatter(
                        x=processed_data_dict[symbol]['date'],
                        y=normalized_prices[symbol],
                        mode='lines',
                        name=symbol,
                        line=dict(color=colors[i % len(colors)], width=2)
                    )
                )
            
            fig.update_layout(
                title="Normalized Price Comparison (Base = 100)",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # DATA EXPORT & DOWNLOAD
    # ============================================
    
    st.markdown("---")
    st.subheader("📥 Data Export & Download")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Raw Data Export**")
        
        for symbol in selected_symbols:
            if symbol in raw_data_dict:
                create_download_button(
                    raw_data_dict[symbol],
                    f"{symbol}_raw_data.csv",
                    f"📄 Download {symbol} Raw Data",
                    key=f"raw_{symbol}"
                )
    
    with col2:
        st.write("**Processed Data Export**")
        
        for symbol in selected_symbols:
            if symbol in processed_data_dict:
                # Combine processed data with indicators
                export_data = processed_data_dict[symbol].copy()
                
                if symbol in indicators_dict:
                    for ind_name, ind_series in indicators_dict[symbol].items():
                        export_data[ind_name] = ind_series
                
                create_download_button(
                    export_data,
                    f"{symbol}_with_indicators.csv",
                    f"📊 Download {symbol} + Indicators",
                    key=f"processed_{symbol}"
                )
    
    with col3:
        st.write("**Analysis Summary**")
        
        # Create comprehensive summary
        summary_data = []
        
        for symbol in selected_symbols:
            if symbol in processed_data_dict:
                data = processed_data_dict[symbol]
                latest = data.iloc[-1]
                
                summary_row = {
                    'Symbol': symbol,
                    'Latest_Price': latest['close'],
                    'Volume': latest['volume'],
                    'Data_Points': len(data),
                    'Indicators_Count': len(indicators_dict.get(symbol, {})),
                    'Date_Range': f"{start_date} to {end_date}",
                    'Source': selected_data_source
                }
                
                if 'returns' in data.columns:
                    summary_row['Volatility'] = data['returns'].std() * np.sqrt(252)
                    summary_row['Total_Return'] = (latest['close'] / data['close'].iloc[0] - 1)
                
                summary_data.append(summary_row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            create_download_button(
                summary_df,
                "analysis_summary.csv",
                "📋 Download Summary Report",
                key="summary_export"
            )
    
    # ============================================
    # ALERTS & NOTIFICATIONS
    # ============================================
    
    st.markdown("---")
    st.subheader("🚨 Market Alerts & Insights")
    
    # Generate alerts based on data and indicators
    alert_manager = get_alert_manager()
    
    for symbol in selected_symbols:
        if symbol in processed_data_dict and symbol in indicators_dict:
            data = processed_data_dict[symbol]
            indicators = indicators_dict[symbol]
            
            # Generate simple alerts
            generate_simple_signal_alerts(data, symbol, alert_manager)
            
            # Indicator-based alerts
            if 'RSI' in indicators:
                rsi_latest = indicators['RSI'].iloc[-1]
                if not pd.isna(rsi_latest):
                    if rsi_latest > 70:
                        alert_manager.add_alert(
                            f"{symbol} RSI is overbought at {rsi_latest:.1f}",
                            "technical_indicator",
                            "warning",
                            symbol
                        )
                    elif rsi_latest < 30:
                        alert_manager.add_alert(
                            f"{symbol} RSI is oversold at {rsi_latest:.1f}",
                            "technical_indicator", 
                            "info",
                            symbol
                        )
            
            if 'Volume_Ratio' in indicators:
                vol_ratio = indicators['Volume_Ratio'].iloc[-1]
                if not pd.isna(vol_ratio) and vol_ratio > 2.0:
                    alert_manager.add_alert(
                        f"{symbol} showing unusual volume spike ({vol_ratio:.1f}x normal)",
                        "volume_analysis",
                        "info",
                        symbol
                    )
    
    # Display alerts
    if alert_manager.alerts:
        for alert in alert_manager.alerts[-10:]:  # Show last 10 alerts
            if alert.level.value == "success":
                st.success(f"✅ {alert}")
            elif alert.level.value == "warning":
                st.warning(f"⚠️ {alert}")
            elif alert.level.value == "error":
                st.error(f"❌ {alert}")
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
        📊 Advanced Data & Technical Indicators | 
        {len(selected_symbols)} symbols analyzed | 
        {sum(len(indicators) for indicators in indicators_dict.values())} indicators calculated |
        Data Source: {selected_data_source} | 
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE EXECUTION
# ============================================

if __name__ == "__main__":
    main()
