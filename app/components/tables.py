"""
app/components/tables.py

Advanced table components for StockPredictionPro Streamlit application.
Provides reusable table displays with formatting, sorting, filtering,
and download capabilities for financial data visualization.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Callable
import base64
import io
from datetime import datetime

# ============================================
# BASIC TABLE DISPLAY FUNCTIONS
# ============================================

def display_dataframe(
    df: pd.DataFrame,
    title: str = None,
    max_rows: int = 100,
    show_index: bool = False,
    height: int = 400,
    use_container_width: bool = True,
    key: Optional[str] = None
) -> None:
    """
    Display pandas DataFrame with enhanced formatting
    
    Args:
        df: DataFrame to display
        title: Optional title for the table
        max_rows: Maximum number of rows to display
        show_index: Whether to show DataFrame index
        height: Table height in pixels
        use_container_width: Use full container width
        key: Unique key for Streamlit widget
    """
    if df is None or df.empty:
        st.info("📊 No data available to display")
        return
    
    if title:
        st.subheader(title)
    
    # Show row count info
    if len(df) > max_rows:
        st.warning(f"📋 Displaying first {max_rows} of {len(df):,} total rows")
    else:
        st.info(f"📋 Showing {len(df):,} rows")
    
    # Format the DataFrame
    display_df = format_financial_dataframe(df.head(max_rows))
    
    st.dataframe(
        display_df,
        use_container_width=use_container_width,
        height=height,
        key=key,
        hide_index=not show_index
    )

def display_summary_table(
    data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    key: Optional[str] = None
) -> None:
    """
    Display summary table from list of dictionaries
    
    Args:
        data: List of dictionaries containing table data
        columns: Optional list of columns to display
        title: Optional table title
        key: Unique key for Streamlit widget
    """
    if not data:
        st.info("📊 No summary data available")
        return
    
    df = pd.DataFrame(data)
    
    if columns:
        # Only show specified columns if they exist
        available_columns = [col for col in columns if col in df.columns]
        if available_columns:
            df = df[available_columns]
    
    if title:
        st.subheader(title)
    
    # Format and display
    formatted_df = format_financial_dataframe(df)
    st.dataframe(
        formatted_df,
        use_container_width=True,
        key=key,
        hide_index=True
    )

# ============================================
# SPECIALIZED TABLE DISPLAYS
# ============================================

def display_stock_data_table(
    df: pd.DataFrame,
    symbol: str = "Stock",
    max_rows: int = 50,
    key: Optional[str] = None
) -> None:
    """
    Display stock price data with proper formatting
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol for title
        max_rows: Maximum rows to display
        key: Unique key for widget
    """
    if df is None or df.empty:
        st.info(f"📈 No price data available for {symbol}")
        return
    
    st.subheader(f"📈 {symbol} Price Data")
    
    # Prepare display DataFrame
    display_df = df.copy()
    
    # Format date column
    if 'date' in display_df.columns:
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
    
    # Format price columns
    price_columns = ['open', 'high', 'low', 'close', 'adj_close']
    for col in price_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "-")
    
    # Format volume
    if 'volume' in display_df.columns:
        display_df['volume'] = display_df['volume'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "-")
    
    # Calculate daily change if possible
    if 'close' in df.columns and len(df) > 1:
        df_calc = df.copy()
        df_calc['daily_change'] = df_calc['close'].pct_change() * 100
        display_df['daily_change %'] = df_calc['daily_change'].apply(
            lambda x: f"{x:+.2f}%" if pd.notnull(x) else "-"
        )
    
    st.dataframe(
        display_df.head(max_rows),
        use_container_width=True,
        key=key,
        hide_index=True
    )

def display_trade_records_table(
    df: pd.DataFrame,
    max_rows: int = 100,
    key: Optional[str] = None
) -> None:
    """
    Display trade records with profit/loss highlighting
    
    Args:
        df: DataFrame with trade data
        max_rows: Maximum rows to display
        key: Unique key for widget
    """
    if df is None or df.empty:
        st.info("💼 No trade records to display")
        return
    
    st.subheader("💼 Trade Records")
    
    display_df = df.copy()
    
    # Format date columns
    date_columns = ['date', 'entry_date', 'exit_date', 'timestamp']
    for col in date_columns:
        if col in display_df.columns:
            display_df[col] = pd.to_datetime(display_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
    
    # Format currency columns
    currency_columns = ['entry_price', 'exit_price', 'profit', 'loss', 'pnl', 'commission']
    for col in currency_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "-")
    
    # Format percentage columns
    pct_columns = ['return_pct', 'profit_pct', 'return']
    for col in pct_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
    
    # Apply conditional formatting for profit/loss
    def highlight_pnl(val):
        """Color code profit/loss values"""
        if isinstance(val, str) and val != "-":
            try:
                numeric_val = float(val.replace('$', '').replace(',', ''))
                if numeric_val > 0:
                    return 'background-color: #d4edda; color: #155724'  # Light green
                elif numeric_val < 0:
                    return 'background-color: #f8d7da; color: #721c24'  # Light red
            except:
                pass
        return ''
    
    # Check for profit/PnL columns to highlight
    pnl_columns = [col for col in ['profit', 'pnl', 'profit_pct'] if col in display_df.columns]
    
    if pnl_columns:
        styled_df = display_df.head(max_rows).style.applymap(highlight_pnl, subset=pnl_columns)
        st.dataframe(styled_df, use_container_width=True, key=key)
    else:
        st.dataframe(display_df.head(max_rows), use_container_width=True, key=key, hide_index=True)

def display_portfolio_holdings_table(
    df: pd.DataFrame,
    total_value: float = None,
    key: Optional[str] = None
) -> None:
    """
    Display portfolio holdings with allocation percentages
    
    Args:
        df: DataFrame with portfolio holdings
        total_value: Total portfolio value for allocation calculation
        key: Unique key for widget
    """
    if df is None or df.empty:
        st.info("💰 No portfolio holdings to display")
        return
    
    st.subheader("💰 Portfolio Holdings")
    
    display_df = df.copy()
    
    # Calculate market value if not present
    if 'market_value' not in display_df.columns and 'quantity' in display_df.columns and 'current_price' in display_df.columns:
        display_df['market_value'] = display_df['quantity'] * display_df['current_price']
    
    # Calculate allocation percentages
    if 'market_value' in display_df.columns:
        total_mv = display_df['market_value'].sum() if total_value is None else total_value
        display_df['allocation_pct'] = (display_df['market_value'] / total_mv * 100).round(2)
    
    # Format currency columns
    currency_columns = ['current_price', 'avg_cost', 'market_value', 'unrealized_pnl']
    for col in currency_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "-")
    
    # Format percentage columns
    if 'allocation_pct' in display_df.columns:
        display_df['allocation_pct'] = display_df['allocation_pct'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "-")
    
    # Format quantity
    if 'quantity' in display_df.columns:
        display_df['quantity'] = display_df['quantity'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "-")
    
    st.dataframe(display_df, use_container_width=True, key=key, hide_index=True)

def display_performance_comparison_table(
    data: Dict[str, Dict[str, float]],
    title: str = "Performance Comparison",
    key: Optional[str] = None
) -> None:
    """
    Display performance comparison table for multiple strategies/assets
    
    Args:
        data: Dictionary with format {asset_name: {metric_name: value}}
        title: Table title
        key: Unique key for widget
    """
    if not data:
        st.info("📊 No performance data available for comparison")
        return
    
    st.subheader(title)
    
    # Convert to DataFrame
    df = pd.DataFrame(data).T
    
    # Format percentage columns
    pct_columns = ['Total Return', 'Annual Return', 'Volatility', 'Max Drawdown']
    for col in pct_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
    
    # Format ratio columns
    ratio_columns = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
    for col in ratio_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
    
    st.dataframe(df, use_container_width=True, key=key)

# ============================================
# FORMATTING UTILITIES
# ============================================

def format_financial_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply financial formatting to DataFrame
    
    Args:
        df: DataFrame to format
        
    Returns:
        Formatted DataFrame
    """
    if df.empty:
        return df
    
    formatted_df = df.copy()
    
    for col in formatted_df.columns:
        if pd.api.types.is_numeric_dtype(formatted_df[col]):
            # Price columns
            if any(keyword in col.lower() for keyword in ['price', 'value', 'cost', 'pnl', 'profit', 'loss']):
                formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "-")
            
            # Volume columns
            elif 'volume' in col.lower():
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "-")
            
            # Percentage columns
            elif any(keyword in col.lower() for keyword in ['pct', 'percent', 'ratio']) or '%' in col:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
            
            # Ratio columns
            elif any(keyword in col.lower() for keyword in ['sharpe', 'sortino', 'calmar']):
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
            
            # General numeric columns
            else:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.3f}" if pd.notnull(x) else "-")
    
    return formatted_df

# ============================================
# DOWNLOAD FUNCTIONALITY
# ============================================

def create_download_button(
    df: pd.DataFrame,
    filename: str = "data.csv",
    button_text: str = "📥 Download CSV",
    key: Optional[str] = None
) -> None:
    """
    Create download button for DataFrame
    
    Args:
        df: DataFrame to download
        filename: Name for downloaded file
        button_text: Button display text
        key: Unique key for button
    """
    if df is None or df.empty:
        st.warning("⚠️ No data available for download")
        return
    
    # Convert DataFrame to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label=button_text,
        data=csv_data,
        file_name=filename,
        mime='text/csv',
        key=key
    )

def generate_download_link(df: pd.DataFrame, filename: str = "data.csv") -> str:
    """
    Generate HTML download link for DataFrame (legacy method)
    
    Args:
        df: DataFrame to download
        filename: Name for downloaded file
        
    Returns:
        HTML string for download link
    """
    if df is None or df.empty:
        return "<p>No data available for download</p>"
    
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download {filename}</a>'

# ============================================
# INTERACTIVE TABLE FEATURES
# ============================================

def create_sortable_table(
    df: pd.DataFrame,
    default_sort_column: str = None,
    ascending: bool = False,
    key: Optional[str] = None
) -> pd.DataFrame:
    """
    Create interactive sortable table
    
    Args:
        df: DataFrame to display
        default_sort_column: Column to sort by initially
        ascending: Sort order
        key: Unique key for widgets
        
    Returns:
        Sorted DataFrame
    """
    if df is None or df.empty:
        st.info("📊 No data available for sorting")
        return pd.DataFrame()
    
    # Sorting controls
    sort_col1, sort_col2 = st.columns(2)
    
    with sort_col1:
        sort_column = st.selectbox(
            "Sort by Column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_sort_column) if default_sort_column in df.columns else 0,
            key=f"{key}_sort_column" if key else "sort_column"
        )
    
    with sort_col2:
        sort_ascending = st.checkbox(
            "Ascending Order",
            value=ascending,
            key=f"{key}_sort_ascending" if key else "sort_ascending"
        )
    
    # Sort DataFrame
    sorted_df = df.sort_values(by=sort_column, ascending=sort_ascending)
    
    return sorted_df

def create_filterable_table(
    df: pd.DataFrame,
    filterable_columns: List[str] = None,
    key: Optional[str] = None
) -> pd.DataFrame:
    """
    Create table with column-based filtering
    
    Args:
        df: DataFrame to filter
        filterable_columns: Columns that can be filtered
        key: Unique key for widgets
        
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty:
        st.info("📊 No data available for filtering")
        return pd.DataFrame()
    
    if filterable_columns is None:
        filterable_columns = [col for col in df.columns if df[col].dtype in ['object', 'string']]
    
    if not filterable_columns:
        return df
    
    filtered_df = df.copy()
    
    # Create filters
    st.write("**Filter Options:**")
    filter_cols = st.columns(min(len(filterable_columns), 3))
    
    for i, column in enumerate(filterable_columns[:3]):  # Limit to 3 filters for UI
        with filter_cols[i % 3]:
            unique_values = sorted(df[column].dropna().unique())
            selected_values = st.multiselect(
                f"Filter {column}",
                options=unique_values,
                default=unique_values,
                key=f"{key}_filter_{column}" if key else f"filter_{column}"
            )
            
            if selected_values:
                filtered_df = filtered_df[filtered_df[column].isin(selected_values)]
    
    return filtered_df

# ============================================
# EXAMPLE USAGE AND TESTING
# ============================================

if __name__ == "__main__":
    st.title("📋 Tables Demo - StockPredictionPro")
    
    # Generate sample data
    np.random.seed(42)
    
    # Sample stock data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    stock_data = pd.DataFrame({
        'date': dates,
        'symbol': 'AAPL',
        'open': 150 + np.random.randn(100) * 5,
        'high': 155 + np.random.randn(100) * 5,
        'low': 145 + np.random.randn(100) * 5,
        'close': 150 + np.random.randn(100) * 5,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Sample trade data
    trade_data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=20, freq='W'),
        'symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL'], 20),
        'type': np.random.choice(['Buy', 'Sell'], 20),
        'quantity': np.random.randint(10, 1000, 20),
        'entry_price': 100 + np.random.randn(20) * 10,
        'exit_price': 105 + np.random.randn(20) * 10,
        'profit': np.random.randn(20) * 500
    })
    
    # Sample portfolio data
    portfolio_data = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        'quantity': [100, 50, 25, 75, 40],
        'current_price': [150, 300, 2500, 200, 800],
        'avg_cost': [140, 280, 2400, 180, 750]
    })
    
    # Demo tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Stock Data", "Trade Records", "Portfolio", "Interactive"])
    
    with tab1:
        display_stock_data_table(stock_data, "AAPL")
        create_download_button(stock_data, "AAPL_data.csv", key="stock_download")
    
    with tab2:
        display_trade_records_table(trade_data)
        create_download_button(trade_data, "trades.csv", key="trade_download")
    
    with tab3:
        display_portfolio_holdings_table(portfolio_data)
        
        # Performance comparison example
        performance_data = {
            'Portfolio': {
                'Total Return': 0.15,
                'Annual Return': 0.12,
                'Volatility': 0.18,
                'Sharpe Ratio': 0.67,
                'Max Drawdown': -0.08
            },
            'S&P 500': {
                'Total Return': 0.10,
                'Annual Return': 0.08,
                'Volatility': 0.16,
                'Sharpe Ratio': 0.50,
                'Max Drawdown': -0.12
            }
        }
        display_performance_comparison_table(performance_data, "Portfolio vs Benchmark")
    
    with tab4:
        st.subheader("Interactive Table Features")
        
        # Sortable table
        sorted_data = create_sortable_table(trade_data, default_sort_column='profit', key="sort_demo")
        display_dataframe(sorted_data, "Sorted Trade Data", max_rows=10, key="sorted_table")
        
        # Filterable table
        filtered_data = create_filterable_table(trade_data, ['symbol', 'type'], key="filter_demo")
        display_dataframe(filtered_data, "Filtered Trade Data", max_rows=10, key="filtered_table")
