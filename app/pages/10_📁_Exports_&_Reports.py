"""
app/pages/10_📁_Exports_&_Reports.py

Advanced Exports & Reports Management for StockPredictionPro.
Comprehensive data export, report generation, analysis summaries,
and professional documentation system.

Integrates with:
- src/evaluation/reports.py (comprehensive report generation)
- src/evaluation/plots.py (chart export and visualization)
- data/exports/ (organized export storage)
- All analysis modules for data aggregation

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
import json
import io
import zipfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your comprehensive reporting system
try:
    # Evaluation and reporting modules
    from src.evaluation.reports import ReportGenerator
    from src.evaluation.plots import PlotGenerator
    
    # Data export modules (if available)
    from src.data.manager import DataManager
    from src.data.cache import CacheManager
    
    REPORTING_MODULES_AVAILABLE = True
    
except ImportError as e:
    st.warning(f"Some reporting modules not found: {e}")
    REPORTING_MODULES_AVAILABLE = False

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
# EXPORT & REPORT CONFIGURATIONS
# ============================================

EXPORT_CATEGORIES = {
    "Market Data": {
        "description": "Raw and processed market data exports",
        "formats": ["CSV", "Excel", "JSON", "Parquet"],
        "data_types": ["OHLCV", "Technical Indicators", "Fundamental Data"]
    },
    "Analysis Results": {
        "description": "Model predictions and analysis outputs",
        "formats": ["CSV", "Excel", "JSON", "PDF"],
        "data_types": ["Predictions", "Signals", "Performance Metrics"]
    },
    "Backtesting Reports": {
        "description": "Strategy backtesting and performance analysis",
        "formats": ["PDF", "Excel", "HTML", "JSON"],
        "data_types": ["Equity Curves", "Trade History", "Risk Metrics"]
    },
    "Portfolio Analytics": {
        "description": "Portfolio management and risk analysis",
        "formats": ["PDF", "Excel", "CSV"],
        "data_types": ["Holdings", "Performance", "Risk Metrics", "Attribution"]
    },
    "Research Reports": {
        "description": "Comprehensive research and analysis reports",
        "formats": ["PDF", "HTML", "Word"],
        "data_types": ["Full Analysis", "Executive Summary", "Technical Details"]
    }
}

REPORT_TEMPLATES = {
    "Executive Summary": {
        "sections": ["Key Metrics", "Performance Overview", "Risk Assessment", "Recommendations"],
        "target_audience": "Executives",
        "length": "2-3 pages"
    },
    "Technical Analysis": {
        "sections": ["Data Quality", "Model Performance", "Validation Results", "Methodology"],
        "target_audience": "Analysts",
        "length": "5-10 pages"
    },
    "Investment Report": {
        "sections": ["Market Analysis", "Portfolio Performance", "Risk Metrics", "Outlook"],
        "target_audience": "Investors",
        "length": "3-5 pages"
    },
    "Compliance Report": {
        "sections": ["Risk Limits", "Validation Results", "Model Governance", "Audit Trail"],
        "target_audience": "Compliance",
        "length": "10-15 pages"
    },
    "Custom Report": {
        "sections": ["Customizable"],
        "target_audience": "Any",
        "length": "Variable"
    }
}

# ============================================
# DATA AGGREGATION FUNCTIONS
# ============================================

def aggregate_session_data() -> Dict[str, Any]:
    """Aggregate all data from current session for reporting"""
    
    session_data = {
        "timestamp": datetime.now(),
        "user_session": st.session_state.get("session_id", "unknown"),
        "analysis_summary": {}
    }
    
    # Collect data from various session states
    if hasattr(st.session_state, 'backtest_results'):
        session_data["backtest_results"] = {
            "strategy": st.session_state.get("backtest_strategy", "Unknown"),
            "symbol": st.session_state.get("backtest_symbol", "Unknown"),
            "metrics": st.session_state.backtest_results.get("metrics", {}),
            "total_trades": len(st.session_state.backtest_results.get("trades", [])),
            "final_value": st.session_state.backtest_results.get("metrics", {}).get("Final Value", 0)
        }
    
    if hasattr(st.session_state, 'portfolio_results'):
        session_data["portfolio_results"] = {
            "assets": st.session_state.get("portfolio_symbols", []),
            "strategy": st.session_state.get("portfolio_strategy", "Unknown"),
            "performance": st.session_state.portfolio_results
        }
    
    if hasattr(st.session_state, 'model_results'):
        session_data["model_results"] = {
            "model_type": st.session_state.get("selected_model", "Unknown"),
            "performance": st.session_state.model_results
        }
    
    return session_data

def generate_comprehensive_dataset(symbols: List[str], 
                                 start_date: date, 
                                 end_date: date) -> Dict[str, pd.DataFrame]:
    """Generate comprehensive dataset for export"""
    
    # Simulate comprehensive data (replace with actual data loading)
    datasets = {}
    
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        dates = pd.date_range(start_date, end_date, freq='D')
        n_days = len(dates)
        
        # Generate realistic market data
        returns = np.random.normal(0.0008, 0.02, n_days)
        prices = [100.0]
        for ret in returns[1:]:
            prices.append(max(prices[-1] * (1 + ret), 1.0))
        
        # Create comprehensive dataset
        data = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0.005, 0.015))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0.005, 0.015))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(14.0, 0.7, n_days).astype(int),
            'returns': [0] + list(np.diff(prices) / prices[:-1]),
            'volatility': pd.Series([0] + list(returns[1:])).rolling(20).std().fillna(0),
            'sma_20': pd.Series(prices).rolling(20).mean().fillna(prices[0]),
            'sma_50': pd.Series(prices).rolling(50).mean().fillna(prices[0]),
        })
        
        # Calculate RSI
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema_12 = pd.Series(prices).ewm(span=12).mean()
        ema_26 = pd.Series(prices).ewm(span=26).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Ensure valid OHLC relationships
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        datasets[symbol] = data.fillna(method='ffill').fillna(method='bfill')
    
    return datasets

def create_executive_summary(session_data: Dict[str, Any]) -> str:
    """Generate executive summary text"""
    
    summary = f"""
    # Executive Summary - StockPredictionPro Analysis
    
    **Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## Key Highlights
    
    """
    
    if "backtest_results" in session_data:
        backtest = session_data["backtest_results"]
        summary += f"""
    ### Strategy Performance
    - **Strategy:** {backtest.get('strategy', 'N/A')}
    - **Asset:** {backtest.get('symbol', 'N/A')}
    - **Total Trades:** {backtest.get('total_trades', 0)}
    - **Final Portfolio Value:** ${backtest.get('final_value', 0):,.0f}
    """
    
    if "portfolio_results" in session_data:
        portfolio = session_data["portfolio_results"]
        summary += f"""
    ### Portfolio Analysis
    - **Strategy:** {portfolio.get('strategy', 'N/A')}
    - **Assets:** {', '.join(portfolio.get('assets', []))}
    """
    
    summary += """
    
    ## Risk Assessment
    - Portfolio diversification appears adequate
    - Risk metrics within acceptable ranges
    - Regular monitoring recommended
    
    ## Recommendations
    1. Continue monitoring key performance metrics
    2. Review strategy parameters quarterly
    3. Maintain diversification across asset classes
    4. Consider rebalancing if significant drift occurs
    
    ---
    *This report was generated automatically by StockPredictionPro*
    """
    
    return summary

# ============================================
# EXPORT FUNCTIONS
# ============================================

def create_excel_export(datasets: Dict[str, pd.DataFrame], filename: str) -> bytes:
    """Create Excel file with multiple sheets"""
    
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = []
        for symbol, data in datasets.items():
            if not data.empty:
                summary_data.append({
                    'Symbol': symbol,
                    'Records': len(data),
                    'Start_Date': data['date'].min().strftime('%Y-%m-%d'),
                    'End_Date': data['date'].max().strftime('%Y-%m-%d'),
                    'Latest_Price': data['close'].iloc[-1],
                    'Total_Return': (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100,
                    'Volatility': data['returns'].std() * 100 * np.sqrt(252)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Individual symbol sheets
        for symbol, data in datasets.items():
            sheet_name = symbol[:31]  # Excel sheet name limit
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    
    buffer.seek(0)
    return buffer.getvalue()

def create_json_export(datasets: Dict[str, pd.DataFrame]) -> str:
    """Create JSON export of all datasets"""
    
    export_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source": "StockPredictionPro",
            "version": "1.0"
        },
        "data": {}
    }
    
    for symbol, data in datasets.items():
        export_data["data"][symbol] = data.to_dict('records')
    
    return json.dumps(export_data, indent=2, default=str)

def create_zip_export(datasets: Dict[str, pd.DataFrame], 
                     session_data: Dict[str, Any]) -> bytes:
    """Create comprehensive ZIP export"""
    
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        
        # Add individual CSV files
        for symbol, data in datasets.items():
            csv_data = data.to_csv(index=False)
            zf.writestr(f"data/{symbol}_data.csv", csv_data)
        
        # Add JSON metadata
        json_data = create_json_export(datasets)
        zf.writestr("metadata.json", json_data)
        
        # Add executive summary
        summary = create_executive_summary(session_data)
        zf.writestr("executive_summary.md", summary)
        
        # Add session data
        session_json = json.dumps(session_data, indent=2, default=str)
        zf.writestr("session_data.json", session_json)
    
    buffer.seek(0)
    return buffer.getvalue()

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_export_dashboard(datasets: Dict[str, pd.DataFrame]) -> None:
    """Create comprehensive export overview dashboard"""
    
    if not datasets:
        st.warning("No data available for dashboard")
        return
    
    # Create summary statistics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Portfolio Performance Overview',
            'Data Quality Summary',
            'Export Statistics',
            'Analysis Coverage'
        )
    )
    
    # Portfolio performance
    symbols = list(datasets.keys())
    performance_data = []
    
    for symbol in symbols[:5]:  # Limit to 5 symbols for readability
        data = datasets[symbol]
        if len(data) > 1:
            total_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
            performance_data.append(total_return)
    
    if performance_data:
        fig.add_trace(
            go.Bar(
                x=symbols[:len(performance_data)],
                y=performance_data,
                name='Total Return %',
                marker_color=['green' if x > 0 else 'red' for x in performance_data]
            ), row=1, col=1
        )
    
    # Data quality summary
    data_quality = []
    for symbol, data in datasets.items():
        completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        data_quality.append(completeness)
    
    fig.add_trace(
        go.Bar(
            x=list(datasets.keys()),
            y=data_quality,
            name='Data Completeness %',
            marker_color='blue'
        ), row=1, col=2
    )
    
    # Export statistics
    export_stats = {
        'Total Symbols': len(datasets),
        'Total Records': sum(len(data) for data in datasets.values()),
        'Date Range (Days)': max(len(data) for data in datasets.values()) if datasets else 0,
        'Data Points': sum(len(data) * len(data.columns) for data in datasets.values())
    }
    
    fig.add_trace(
        go.Bar(
            x=list(export_stats.keys()),
            y=list(export_stats.values()),
            name='Statistics',
            marker_color='orange'
        ), row=2, col=1
    )
    
    # Analysis coverage pie chart
    analysis_types = ['Price Data', 'Technical Indicators', 'Volume Data', 'Calculated Metrics']
    coverage_values = [100, 85, 95, 75]  # Sample coverage percentages
    
    fig.add_trace(
        go.Pie(
            labels=analysis_types,
            values=coverage_values,
            name='Coverage'
        ), row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title="Export & Analysis Dashboard",
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# MAIN PAGE FUNCTION
# ============================================

def main():
    """Main exports & reports page function"""
    
    # Apply theme
    apply_custom_theme("financial")
    
    # Page header
    st.title("📁 Advanced Exports & Reports Management")
    st.markdown("Comprehensive data export, report generation, and analysis documentation system")
    
    if not REPORTING_MODULES_AVAILABLE:
        st.info("ℹ️ Using built-in export capabilities. Enhanced reporting features available with full module integration.")
    
    st.markdown("---")
    
    # ============================================
    # EXPORT CONFIGURATION
    # ============================================
    
    st.subheader("🛠️ Export Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data selection
        export_category = filter_categorical(
            "Export Category",
            list(EXPORT_CATEGORIES.keys()),
            multi=False,
            key="export_category"
        )
        
        # Symbol selection
        available_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "SPY", "QQQ", "IWM"]
        selected_symbols = filter_categorical(
            "Select Symbols",
            available_symbols,
            multi=True,
            key="export_symbols"
        )
        
        # Date range
        start_date, end_date = filter_date_range(
            default_days=90,
            key="export_dates"
        )
    
    with col2:
        # Export format
        if export_category:
            category_info = EXPORT_CATEGORIES[export_category]
            export_format = st.selectbox(
                "Export Format",
                category_info["formats"],
                help=f"Available formats for {export_category}"
            )
        else:
            export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON", "PDF"])
        
        # Report template
        if export_category == "Research Reports":
            report_template = filter_categorical(
                "Report Template",
                list(REPORT_TEMPLATES.keys()),
                multi=False,
                key="report_template"
            )
        else:
            report_template = None
        
        # Export options
        include_charts = st.checkbox("Include Charts", value=True)
        include_metadata = st.checkbox("Include Metadata", value=True)
        compress_export = st.checkbox("Compress (ZIP)", value=False)
    
    # Display category information
    if export_category:
        category_info = EXPORT_CATEGORIES[export_category]
        st.info(f"**{export_category}:** {category_info['description']}")
        st.write(f"**Data Types:** {', '.join(category_info['data_types'])}")
    
    # ============================================
    # SESSION DATA OVERVIEW
    # ============================================
    
    st.subheader("📊 Session Data Overview")
    
    # Collect session data
    session_data = aggregate_session_data()
    
    # Display current session summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Session Duration", "Current Session")
    
    with col2:
        backtest_count = 1 if hasattr(st.session_state, 'backtest_results') else 0
        st.metric("Backtests Run", backtest_count)
    
    with col3:
        portfolio_count = 1 if hasattr(st.session_state, 'portfolio_results') else 0
        st.metric("Portfolios Analyzed", portfolio_count)
    
    with col4:
        model_count = 1 if hasattr(st.session_state, 'model_results') else 0
        st.metric("Models Trained", model_count)
    
    # Show session data details
    if st.expander("📋 Detailed Session Information", expanded=False):
        
        if hasattr(st.session_state, 'backtest_results'):
            st.write("**Latest Backtest Results:**")
            backtest_info = session_data.get("backtest_results", {})
            st.json(backtest_info)
        
        if hasattr(st.session_state, 'portfolio_results'):
            st.write("**Portfolio Analysis:**")
            st.write("Portfolio analysis data available for export")
        
        if hasattr(st.session_state, 'model_results'):
            st.write("**Model Results:**")
            st.write("Machine learning model results available")
    
    st.markdown("---")
    
    # ============================================
    # DATA GENERATION & EXPORT
    # ============================================
    
    if selected_symbols:
        
        st.subheader("📈 Export Preview & Generation")
        
        # Generate datasets
        with st.spinner("Generating comprehensive datasets..."):
            datasets = generate_comprehensive_dataset(selected_symbols, start_date, end_date)
        
        # Create export dashboard
        create_export_dashboard(datasets)
        
        # Export summary
        st.subheader("📋 Export Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data summary table
            summary_data = []
            for symbol, data in datasets.items():
                summary_data.append({
                    'Symbol': symbol,
                    'Records': len(data),
                    'Columns': len(data.columns),
                    'Date Range': f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}",
                    'Latest Price': f"${data['close'].iloc[-1]:.2f}",
                    'Total Return': f"{((data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100):+.1f}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Export options and buttons
            st.write("**Available Exports:**")
            
            # Individual CSV exports
            for symbol in selected_symbols:
                if symbol in datasets:
                    create_download_button(
                        datasets[symbol],
                        f"{symbol}_data_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv",
                        f"📄 Download {symbol} CSV",
                        key=f"csv_{symbol}"
                    )
            
            # Combined Excel export
            if len(datasets) > 1:
                excel_data = create_excel_export(datasets, "combined_analysis")
                st.download_button(
                    label="📊 Download Combined Excel",
                    data=excel_data,
                    file_name=f"combined_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="excel_export"
                )
            
            # JSON export
            json_data = create_json_export(datasets)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="json_export"
            )
            
            # Comprehensive ZIP export
            if compress_export:
                zip_data = create_zip_export(datasets, session_data)
                st.download_button(
                    label="🗜️ Download ZIP Archive",
                    data=zip_data,
                    file_name=f"stockpredictionpro_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    key="zip_export"
                )
    
    # ============================================
    # REPORT GENERATION
    # ============================================
    
    if export_category == "Research Reports" and report_template:
        
        st.markdown("---")
        st.subheader("📝 Report Generation")
        
        template_info = REPORT_TEMPLATES[report_template]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Template:** {report_template}")
            st.write(f"**Target Audience:** {template_info['target_audience']}")
            st.write(f"**Expected Length:** {template_info['length']}")
            st.write(f"**Sections:** {', '.join(template_info['sections'])}")
        
        with col2:
            # Report customization
            st.write("**Report Customization:**")
            
            report_title = st.text_input("Report Title", value=f"StockPredictionPro Analysis - {datetime.now().strftime('%Y-%m-%d')}")
            include_executive_summary = st.checkbox("Include Executive Summary", value=True)
            include_detailed_analysis = st.checkbox("Include Detailed Analysis", value=True)
            include_appendices = st.checkbox("Include Data Appendices", value=False)
        
        # Generate report
        if st.button("📋 Generate Report", type="primary"):
            
            with st.spinner("Generating comprehensive report..."):
                
                # Create executive summary
                if include_executive_summary:
                    summary_text = create_executive_summary(session_data)
                    
                    st.success("✅ Report generated successfully!")
                    
                    # Display report preview
                    st.subheader("📄 Report Preview")
                    st.markdown(summary_text)
                    
                    # Download report
                    st.download_button(
                        label="📥 Download Report (Markdown)",
                        data=summary_text,
                        file_name=f"{report_title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown",
                        key="report_download"
                    )
                else:
                    st.info("Please select report sections to generate")
    
    # ============================================
    # EXPORT HISTORY & MANAGEMENT
    # ============================================
    
    st.markdown("---")
    st.subheader("📚 Export History & Management")
    
    # Simulated export history
    export_history = [
        {
            "Date": "2025-08-29 20:45",
            "Type": "Portfolio Analysis",
            "Symbols": "AAPL, MSFT, GOOGL",
            "Format": "Excel",
            "Size": "2.3 MB",
            "Status": "Completed"
        },
        {
            "Date": "2025-08-29 18:22",
            "Type": "Backtest Report",
            "Symbols": "TSLA",
            "Format": "PDF",
            "Size": "1.8 MB",
            "Status": "Completed"
        },
        {
            "Date": "2025-08-29 16:10",
            "Type": "Market Data",
            "Symbols": "SPY, QQQ",
            "Format": "CSV",
            "Size": "0.9 MB",
            "Status": "Completed"
        }
    ]
    
    history_df = pd.DataFrame(export_history)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("**Recent Export History:**")
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Export Statistics:**")
        st.metric("Total Exports", len(export_history))
        st.metric("Total Size", "5.0 MB")
        st.metric("Success Rate", "100%")
    
    # ============================================
    # ALERTS & NOTIFICATIONS
    # ============================================
    
    st.markdown("---")
    st.subheader("🚨 Export Alerts & Notifications")
    
    # Generate export-related alerts
    alert_manager = get_alert_manager()
    
    # Data quality alerts
    if selected_symbols and datasets:
        total_records = sum(len(data) for data in datasets.values())
        if total_records > 10000:
            alert_manager.add_alert(
                f"📊 Large dataset export: {total_records:,} total records across {len(datasets)} symbols",
                "data_export",
                "info",
                "Export"
            )
        
        # Data completeness check
        for symbol, data in datasets.items():
            null_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            if null_percentage > 5:
                alert_manager.add_alert(
                    f"⚠️ Data quality concern: {symbol} has {null_percentage:.1f}% missing values",
                    "data_quality",
                    "warning",
                    symbol
                )
    
    # Export recommendations
    if export_category and selected_symbols:
        alert_manager.add_alert(
            f"✅ Ready to export {export_category} for {len(selected_symbols)} symbols in {export_format} format",
            "export_ready",
            "success",
            "Export"
        )
    
    # Performance alerts
    if hasattr(st.session_state, 'backtest_results'):
        results = st.session_state.backtest_results
        if "metrics" in results:
            total_return = results["metrics"].get("Total Return", 0)
            if total_return > 0.15:
                alert_manager.add_alert(
                    f"🎉 Strong backtest performance ({total_return:.1%}) - Consider including in report",
                    "performance_highlight",
                    "success",
                    "Backtest"
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
        st.info("✅ No export alerts - All systems ready for data export")
    
    # ============================================
    # FOOTER
    # ============================================
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        📁 Advanced Exports & Reports Management | 
        {len(selected_symbols) if selected_symbols else 0} symbols selected | 
        {export_format if 'export_format' in locals() else 'Format not selected'} export format | 
        {export_category if export_category else 'Category not selected'} | 
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE EXECUTION
# ============================================

if __name__ == "__main__":
    main()
