"""
app/utils/ui_helpers.py

UI helper utilities for StockPredictionPro Streamlit application.
Provides reusable UI components, formatting utilities, and layout helpers
to create consistent user interfaces across all pages.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import base64
import io

# ============================================
# DATA DISPLAY HELPERS
# ============================================

def display_dataframe(df: pd.DataFrame, 
                     max_rows: int = 100, 
                     height: int = None,
                     use_container_width: bool = True,
                     hide_index: bool = True) -> None:
    """
    Display DataFrame with enhanced formatting
    
    Args:
        df: DataFrame to display
        max_rows: Maximum rows to show
        height: Fixed height for table
        use_container_width: Use full container width
        hide_index: Hide DataFrame index
    """
    if df is None or df.empty:
        st.info("📊 No data available to display")
        return
    
    # Limit rows if needed
    display_df = df.head(max_rows) if len(df) > max_rows else df
    
    # Format numeric columns
    display_df = format_numeric_columns(display_df)
    
    st.dataframe(
        display_df,
        height=height,
        use_container_width=use_container_width,
        hide_index=hide_index
    )
    
    if len(df) > max_rows:
        st.caption(f"Showing {max_rows} of {len(df)} rows")


def format_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric columns for better display"""
    df_formatted = df.copy()
    
    for col in df_formatted.columns:
        if pd.api.types.is_numeric_dtype(df_formatted[col]):
            # Format based on column name and values
            if any(keyword in col.lower() for keyword in ['price', 'close', 'open', 'high', 'low']):
                df_formatted[col] = df_formatted[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "")
            elif 'volume' in col.lower():
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
            elif 'percent' in col.lower() or '%' in col:
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
            elif any(keyword in col.lower() for keyword in ['ratio', 'rsi', 'macd']):
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    
    return df_formatted


def create_summary_stats(df: pd.DataFrame, numeric_only: bool = True) -> pd.DataFrame:
    """Create summary statistics DataFrame"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    if numeric_only:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return pd.DataFrame()
        
        stats = pd.DataFrame({
            'Count': numeric_df.count(),
            'Mean': numeric_df.mean(),
            'Std': numeric_df.std(),
            'Min': numeric_df.min(),
            'Max': numeric_df.max(),
            'Median': numeric_df.median()
        }).round(3)
        
        return stats
    
    return df.describe()

# ============================================
# CHART HELPERS
# ============================================

def render_line_chart(df: pd.DataFrame, 
                     x_col: str, 
                     y_cols: Union[str, List[str]], 
                     title: str = "",
                     height: int = 400,
                     show_legend: bool = True) -> None:
    """
    Render interactive line chart with Plotly
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_cols: Column name(s) for y-axis
        title: Chart title
        height: Chart height in pixels
        show_legend: Show/hide legend
    """
    if df is None or df.empty:
        st.info("📈 No data available for chart")
        return
    
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    
    fig = go.Figure()
    
    for y_col in y_cols:
        if y_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='lines',
                name=y_col.title(),
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title=title,
        height=height,
        showlegend=show_legend,
        xaxis_title=x_col.title(),
        yaxis_title="Value",
        template="plotly_white",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_candlestick_chart(df: pd.DataFrame, 
                           date_col: str = 'date',
                           open_col: str = 'open',
                           high_col: str = 'high', 
                           low_col: str = 'low',
                           close_col: str = 'close',
                           volume_col: str = 'volume',
                           title: str = "Stock Price",
                           height: int = 600) -> None:
    """
    Render candlestick chart with volume
    
    Args:
        df: DataFrame with OHLCV data
        date_col: Date column name
        open_col: Open price column
        high_col: High price column
        low_col: Low price column
        close_col: Close price column
        volume_col: Volume column
        title: Chart title
        height: Chart height
    """
    if df is None or df.empty:
        st.info("📊 No OHLCV data available for candlestick chart")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(title, 'Volume'),
        row_width=[0.2, 0.7]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df[date_col],
            open=df[open_col],
            high=df[high_col],
            low=df[low_col],
            close=df[close_col],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Volume chart
    colors = ['green' if close >= open else 'red' 
             for close, open in zip(df[close_col], df[open_col])]
    
    fig.add_trace(
        go.Bar(
            x=df[date_col],
            y=df[volume_col],
            marker_color=colors,
            name="Volume",
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=height,
        showlegend=False,
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix") -> None:
    """Render correlation heatmap"""
    if df is None or df.empty:
        st.info("📊 No data available for correlation analysis")
        return
    
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        st.warning("No numeric columns found for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title=title,
        color_continuous_scale="RdBu_r"
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# METRIC DISPLAY HELPERS
# ============================================

def render_metric_card(label: str, 
                      value: Union[int, float, str], 
                      delta: Optional[Union[int, float]] = None,
                      delta_color: str = "normal",
                      help_text: str = None) -> None:
    """
    Render enhanced metric card
    
    Args:
        label: Metric label
        value: Metric value
        delta: Change value
        delta_color: Color for delta ("normal", "inverse", or "off")
        help_text: Tooltip help text
    """
    if isinstance(value, (int, float)):
        if abs(value) >= 1e9:
            formatted_value = f"{value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            formatted_value = f"{value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            formatted_value = f"{value/1e3:.2f}K"
        else:
            formatted_value = f"{value:.2f}"
    else:
        formatted_value = str(value)
    
    st.metric(
        label=label,
        value=formatted_value,
        delta=f"{delta:+.2f}" if delta is not None else None,
        delta_color=delta_color,
        help=help_text
    )


def create_metrics_grid(metrics: Dict[str, Dict[str, Any]], cols: int = 4) -> None:
    """
    Create grid of metrics
    
    Args:
        metrics: Dictionary of metrics with format:
                {label: {"value": val, "delta": delta, "help": help_text}}
        cols: Number of columns in grid
    """
    metric_cols = st.columns(cols)
    
    for i, (label, metric_data) in enumerate(metrics.items()):
        with metric_cols[i % cols]:
            render_metric_card(
                label=label,
                value=metric_data.get("value", 0),
                delta=metric_data.get("delta"),
                delta_color=metric_data.get("delta_color", "normal"),
                help_text=metric_data.get("help")
            )

# ============================================
# LAYOUT HELPERS
# ============================================

def create_two_column_layout(left_content: Callable, 
                            right_content: Callable,
                            left_ratio: float = 0.7) -> None:
    """
    Create two-column layout with custom ratio
    
    Args:
        left_content: Function to render left column content
        right_content: Function to render right column content
        left_ratio: Ratio for left column (0-1)
    """
    left_col, right_col = st.columns([left_ratio, 1-left_ratio])
    
    with left_col:
        left_content()
    
    with right_col:
        right_content()


def create_expandable_section(title: str, 
                            content_func: Callable,
                            expanded: bool = False) -> None:
    """
    Create expandable section
    
    Args:
        title: Section title
        content_func: Function to render content
        expanded: Initially expanded state
    """
    with st.expander(title, expanded=expanded):
        content_func()

# ============================================
# NOTIFICATION HELPERS
# ============================================

def show_success(message: str, icon: str = "✅") -> None:
    """Show success message"""
    st.success(f"{icon} {message}")


def show_warning(message: str, icon: str = "⚠️") -> None:
    """Show warning message"""
    st.warning(f"{icon} {message}")


def show_error(message: str, icon: str = "❌") -> None:
    """Show error message"""
    st.error(f"{icon} {message}")


def show_info(message: str, icon: str = "ℹ️") -> None:
    """Show info message"""
    st.info(f"{icon} {message}")


def show_loading(message: str = "Loading...") -> None:
    """Show loading spinner with message"""
    return st.spinner(message)

# ============================================
# INPUT HELPERS
# ============================================

def create_date_range_picker(default_days: int = 30) -> tuple:
    """
    Create date range picker
    
    Args:
        default_days: Default number of days for range
        
    Returns:
        Tuple of (start_date, end_date)
    """
    col1, col2 = st.columns(2)
    
    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=default_days)
    
    with col1:
        start_date = st.date_input("Start Date", value=default_start)
    
    with col2:
        end_date = st.date_input("End Date", value=default_end)
    
    return start_date, end_date


def create_symbol_selector(symbols: List[str], 
                          default_symbol: str = None,
                          key: str = "symbol_selector") -> str:
    """
    Create stock symbol selector
    
    Args:
        symbols: List of available symbols
        default_symbol: Default selected symbol
        key: Unique key for widget
        
    Returns:
        Selected symbol
    """
    if default_symbol and default_symbol in symbols:
        default_index = symbols.index(default_symbol)
    else:
        default_index = 0
    
    return st.selectbox(
        "Select Symbol",
        options=symbols,
        index=default_index,
        key=key
    )


def create_multi_select_filter(options: List[str],
                             label: str = "Select Options",
                             default: List[str] = None,
                             key: str = None) -> List[str]:
    """
    Create multi-select filter
    
    Args:
        options: List of available options
        label: Filter label
        default: Default selected options
        key: Unique key for widget
        
    Returns:
        List of selected options
    """
    return st.multiselect(
        label,
        options=options,
        default=default or [],
        key=key
    )

# ============================================
# DOWNLOAD HELPERS
# ============================================

def create_download_button(data: pd.DataFrame, 
                          filename: str,
                          button_text: str = "📥 Download CSV",
                          key: str = None) -> None:
    """
    Create download button for DataFrame
    
    Args:
        data: DataFrame to download
        filename: Name for downloaded file
        button_text: Button text
        key: Unique key for button
    """
    if data is None or data.empty:
        st.warning("No data available for download")
        return
    
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label=button_text,
        data=csv_data,
        file_name=filename,
        mime='text/csv',
        key=key
    )


def create_json_download_button(data: Dict[str, Any],
                               filename: str,
                               button_text: str = "📥 Download JSON",
                               key: str = None) -> None:
    """
    Create download button for JSON data
    
    Args:
        data: Dictionary to download as JSON
        filename: Name for downloaded file
        button_text: Button text
        key: Unique key for button
    """
    import json
    
    json_data = json.dumps(data, indent=2, default=str)
    
    st.download_button(
        label=button_text,
        data=json_data,
        file_name=filename,
        mime='application/json',
        key=key
    )

# ============================================
# PROGRESS INDICATORS
# ============================================

def show_progress_bar(progress: float, text: str = "") -> None:
    """
    Show progress bar
    
    Args:
        progress: Progress value (0.0 to 1.0)
        text: Progress text
    """
    st.progress(progress, text=text)


def create_status_indicator(status: str, message: str = "") -> None:
    """
    Create status indicator
    
    Args:
        status: Status type ('running', 'complete', 'error')
        message: Status message
    """
    status_container = st.status(message)
    
    if status == 'running':
        status_container.update(state="running", expanded=True)
    elif status == 'complete':
        status_container.update(state="complete", expanded=False)
    elif status == 'error':
        status_container.update(state="error", expanded=True)

# ============================================
# FORMATTING UTILITIES
# ============================================

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount"""
    if currency == "USD":
        return f"${amount:,.2f}"
    elif currency == "INR":
        return f"₹{amount:,.2f}"
    else:
        return f"{currency} {amount:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage value"""
    return f"{value:.{decimals}f}%"


def format_large_number(number: float) -> str:
    """Format large numbers with K, M, B suffixes"""
    if abs(number) >= 1e9:
        return f"{number/1e9:.2f}B"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.2f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.2f}K"
    else:
        return f"{number:.2f}"

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    """
    Example usage patterns:
    
    # Display DataFrame
    display_dataframe(stock_df, max_rows=50)
    
    # Create metrics grid
    metrics = {
        "Current Price": {"value": 150.25, "delta": 2.15},
        "Volume": {"value": 1500000, "help": "Daily trading volume"},
        "Market Cap": {"value": 2.5e12, "delta": 0.05}
    }
    create_metrics_grid(metrics, cols=3)
    
    # Render charts
    render_line_chart(df, 'date', ['close', 'sma_20'], "Price Chart")
    render_candlestick_chart(df, title="AAPL Stock Price")
    
    # Create layouts
    def left_content():
        st.write("Left column content")
    
    def right_content():
        st.write("Right column content")
    
    create_two_column_layout(left_content, right_content, left_ratio=0.6)
    
    # Download buttons
    create_download_button(df, "stock_data.csv")
    """
    pass
