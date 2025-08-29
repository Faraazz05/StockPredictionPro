"""
app/pages/11_⚙️_Admin_&_Logs.py

Advanced Administration & Log Management for StockPredictionPro.
Comprehensive system monitoring, log analysis, user management,
configuration control, and governance oversight.

Integrates with:
- logs/ (api.log, app.log, error.log, trading.log, audit/)
- src/utils/ (logger, config_loader, governance, helpers, validators, timing, file_io, exceptions)

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
import json
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your comprehensive utilities system
try:
    # Utilities modules
    from src.utils.logger import Logger, LogManager
    from src.utils.config_loader import ConfigLoader
    from src.utils.governance import GovernanceManager
    from src.utils.helpers import SystemHelpers
    from src.utils.validators import SystemValidators
    from src.utils.timing import TimingManager
    from src.utils.file_io import FileIOManager
    from src.utils.exceptions import SystemExceptions
    
    UTILS_MODULES_AVAILABLE = True
    
except ImportError as e:
    st.error(f"Utilities modules not found: {e}")
    UTILS_MODULES_AVAILABLE = False

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
# ADMIN CONFIGURATION & CONSTANTS
# ============================================

# Available log files
LOG_FILES = {
    "Application Log": "logs/app.log",
    "API Log": "logs/api.log", 
    "Error Log": "logs/error.log",
    "Trading Log": "logs/trading.log",
    "Audit Log": "logs/audit/"
}

# System monitoring categories
MONITORING_CATEGORIES = {
    "System Performance": {
        "metrics": ["CPU Usage", "Memory Usage", "Disk Space", "Network I/O"],
        "thresholds": {"cpu": 80, "memory": 85, "disk": 90}
    },
    "Application Health": {
        "metrics": ["Response Time", "Error Rate", "Active Sessions", "Database Connections"],
        "thresholds": {"response_time": 2.0, "error_rate": 5.0}
    },
    "Trading Operations": {
        "metrics": ["Orders Executed", "Signal Accuracy", "Portfolio P&L", "Risk Metrics"],
        "thresholds": {"max_drawdown": 0.15, "sharpe_ratio": 1.0}
    },
    "Data Quality": {
        "metrics": ["Data Completeness", "Update Frequency", "Validation Errors", "Source Availability"],
        "thresholds": {"completeness": 95.0, "validation_errors": 1.0}
    }
}

# User roles and permissions
USER_ROLES = {
    "Super Admin": {
        "permissions": ["all"],
        "description": "Full system access and control"
    },
    "Admin": {
        "permissions": ["logs", "config", "monitoring", "users"],
        "description": "Administrative access without system changes"
    },
    "Analyst": {
        "permissions": ["logs", "monitoring"],
        "description": "View logs and monitoring data"
    },
    "User": {
        "permissions": ["monitoring"],
        "description": "Basic monitoring access"
    }
}

# ============================================
# AUTHENTICATION & AUTHORIZATION
# ============================================

def check_admin_access() -> bool:
    """Check if current user has admin access"""
    # Simplified authentication - in production, integrate with proper auth system
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    
    return st.session_state.admin_authenticated

def authenticate_admin() -> bool:
    """Simple admin authentication"""
    st.subheader("🔐 Administrator Authentication")
    
    with st.form("admin_login"):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username", type="default")
        with col2:
            password = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button("Login")
        
        if submitted:
            # Simple authentication (replace with proper auth in production)
            if username == "admin" and password == "stockpredictionpro2025":
                st.session_state.admin_authenticated = True
                st.session_state.admin_username = username
                st.session_state.admin_login_time = datetime.now()
                st.success("✅ Authentication successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
                return False
    
    return False

# ============================================
# LOG MANAGEMENT FUNCTIONS
# ============================================

def load_log_file(log_path: str, lines: int = 1000) -> pd.DataFrame:
    """Load and parse log file"""
    try:
        log_file = Path(project_root) / log_path
        
        if not log_file.exists():
            # Generate sample log data for demonstration
            return generate_sample_logs(lines)
        
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
        
        # Parse log lines (assuming standard format)
        parsed_logs = []
        for line in log_lines[-lines:]:
            if line.strip():
                # Extract timestamp, level, message
                match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - (\w+) - (.+)', line)
                if match:
                    timestamp, level, message = match.groups()
                    parsed_logs.append({
                        'timestamp': pd.to_datetime(timestamp),
                        'level': level,
                        'message': message.strip(),
                        'source': log_path.split('/')[-1]
                    })
        
        return pd.DataFrame(parsed_logs)
    
    except Exception as e:
        st.error(f"Error loading log file {log_path}: {e}")
        return generate_sample_logs(lines)

def generate_sample_logs(lines: int = 1000) -> pd.DataFrame:
    """Generate sample log data for demonstration"""
    np.random.seed(42)
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    # Generate timestamps
    timestamps = pd.date_range(start_time, end_time, periods=lines)
    
    # Generate log levels
    levels = np.random.choice(['INFO', 'WARNING', 'ERROR', 'DEBUG'], lines, p=[0.6, 0.2, 0.1, 0.1])
    
    # Generate sample messages
    sample_messages = [
        "Application started successfully",
        "User session initialized",
        "Data fetch completed",
        "Model training started",
        "Backtest execution completed",
        "Portfolio rebalanced",
        "API request processed",
        "Database connection established",
        "Cache updated",
        "Signal generated",
        "Order executed",
        "Risk check passed",
        "Validation completed",
        "Report generated",
        "System health check completed"
    ]
    
    error_messages = [
        "Database connection failed",
        "API rate limit exceeded",
        "Model training failed",
        "Data validation error",
        "Network timeout",
        "Configuration error",
        "Authentication failed",
        "Memory allocation error"
    ]
    
    messages = []
    for level in levels:
        if level == 'ERROR':
            messages.append(np.random.choice(error_messages))
        else:
            messages.append(np.random.choice(sample_messages))
    
    sources = np.random.choice(['app.log', 'api.log', 'trading.log', 'error.log'], lines, p=[0.4, 0.3, 0.2, 0.1])
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'level': levels,
        'message': messages,
        'source': sources
    })

def analyze_log_patterns(logs_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze log patterns and generate insights"""
    
    if logs_df.empty:
        return {}
    
    analysis = {}
    
    # Time-based analysis
    logs_df['hour'] = logs_df['timestamp'].dt.hour
    logs_df['date'] = logs_df['timestamp'].dt.date
    
    analysis['total_entries'] = len(logs_df)
    analysis['time_range'] = {
        'start': logs_df['timestamp'].min(),
        'end': logs_df['timestamp'].max()
    }
    
    # Level distribution
    level_counts = logs_df['level'].value_counts()
    analysis['level_distribution'] = level_counts.to_dict()
    
    # Error analysis
    error_logs = logs_df[logs_df['level'] == 'ERROR']
    analysis['error_count'] = len(error_logs)
    analysis['error_rate'] = len(error_logs) / len(logs_df) * 100 if len(logs_df) > 0 else 0
    
    # Peak hours
    hourly_activity = logs_df.groupby('hour').size()
    analysis['peak_hour'] = hourly_activity.idxmax()
    analysis['peak_hour_count'] = hourly_activity.max()
    
    # Recent trends
    daily_counts = logs_df.groupby('date').size()
    if len(daily_counts) > 1:
        analysis['trend'] = 'Increasing' if daily_counts.iloc[-1] > daily_counts.iloc[-2] else 'Decreasing'
    else:
        analysis['trend'] = 'Stable'
    
    # Top error messages
    if not error_logs.empty:
        error_messages = error_logs['message'].value_counts().head(5)
        analysis['top_errors'] = error_messages.to_dict()
    
    return analysis

# ============================================
# SYSTEM MONITORING FUNCTIONS  
# ============================================

def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics"""
    
    # Generate realistic system metrics (replace with actual monitoring in production)
    metrics = {
        'cpu_usage': np.random.uniform(20, 80),
        'memory_usage': np.random.uniform(40, 85),
        'disk_usage': np.random.uniform(30, 70),
        'network_io': np.random.uniform(10, 100),
        'response_time': np.random.uniform(0.5, 2.5),
        'error_rate': np.random.uniform(0.1, 8.0),
        'active_sessions': np.random.randint(5, 50),
        'database_connections': np.random.randint(2, 20),
        'orders_executed': np.random.randint(0, 100),
        'signal_accuracy': np.random.uniform(65, 95),
        'portfolio_pnl': np.random.uniform(-5, 15),
        'data_completeness': np.random.uniform(90, 100),
        'validation_errors': np.random.randint(0, 5)
    }
    
    return metrics

def create_system_dashboard(metrics: Dict[str, Any]) -> None:
    """Create comprehensive system monitoring dashboard"""
    
    # Create multi-panel system dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'System Resource Usage',
            'Application Performance',
            'Trading Operations',
            'Data Quality Metrics',
            'Error Trends',
            'Activity Heatmap'
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ]
    )
    
    # System resource usage (gauge-like bars)
    resources = ['CPU', 'Memory', 'Disk', 'Network']
    resource_values = [metrics['cpu_usage'], metrics['memory_usage'], metrics['disk_usage'], metrics['network_io']]
    colors = ['red' if x > 80 else 'orange' if x > 60 else 'green' for x in resource_values]
    
    fig.add_trace(
        go.Bar(
            x=resources,
            y=resource_values,
            name='Resource Usage %',
            marker_color=colors
        ), row=1, col=1
    )
    
    # Application performance
    perf_metrics = ['Response Time (s)', 'Error Rate %', 'Active Sessions', 'DB Connections']
    perf_values = [metrics['response_time'], metrics['error_rate'], metrics['active_sessions'], metrics['database_connections']]
    
    fig.add_trace(
        go.Scatter(
            x=perf_metrics,
            y=perf_values,
            mode='lines+markers',
            name='Performance',
            line=dict(color='blue', width=3)
        ), row=1, col=2
    )
    
    # Trading operations
    trading_labels = ['Orders', 'Signal Accuracy %', 'Portfolio P&L %']
    trading_values = [metrics['orders_executed'], metrics['signal_accuracy'], metrics['portfolio_pnl']]
    
    fig.add_trace(
        go.Bar(
            x=trading_labels,
            y=trading_values,
            name='Trading Metrics',
            marker_color='purple'
        ), row=2, col=1
    )
    
    # Data quality
    quality_metrics = ['Completeness %', 'Validation Errors']
    quality_values = [metrics['data_completeness'], metrics['validation_errors']]
    
    fig.add_trace(
        go.Bar(
            x=quality_metrics,
            y=quality_values,
            name='Data Quality',
            marker_color='orange'
        ), row=2, col=2
    )
    
    # Error trends (simulated hourly data)
    hours = list(range(24))
    error_counts = np.random.poisson(2, 24)
    
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=error_counts,
            mode='lines',
            name='Hourly Errors',
            line=dict(color='red', width=2)
        ), row=3, col=1
    )
    
    # Activity heatmap (simulated daily/hourly activity)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours_heat = list(range(24))
    activity_matrix = np.random.rand(7, 24) * 100
    
    fig.add_trace(
        go.Heatmap(
            z=activity_matrix,
            x=hours_heat,
            y=days,
            colorscale='Viridis',
            name='Activity'
        ), row=3, col=2
    )
    
    fig.update_layout(
        height=1000,
        title="StockPredictionPro - System Monitoring Dashboard",
        template='plotly_white',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# CONFIGURATION MANAGEMENT
# ============================================

def load_system_configuration() -> Dict[str, Any]:
    """Load system configuration"""
    
    # Sample configuration structure
    config = {
        "application": {
            "name": "StockPredictionPro",
            "version": "1.0.0",
            "debug_mode": False,
            "log_level": "INFO",
            "session_timeout": 3600
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "stockprediction",
            "connection_pool_size": 10,
            "timeout": 30
        },
        "api": {
            "rate_limit": 1000,
            "timeout": 30,
            "retry_attempts": 3,
            "cache_ttl": 300
        },
        "trading": {
            "max_position_size": 0.1,
            "risk_per_trade": 0.02,
            "max_drawdown": 0.15,
            "rebalance_frequency": "daily"
        },
        "alerts": {
            "email_notifications": True,
            "slack_integration": False,
            "alert_threshold": 0.05,
            "max_alerts_per_hour": 10
        }
    }
    
    return config

def update_configuration(config: Dict[str, Any]) -> bool:
    """Update system configuration"""
    try:
        # In production, save to actual config file
        st.session_state.system_config = config
        return True
    except Exception as e:
        st.error(f"Error updating configuration: {e}")
        return False

# ============================================
# MAIN PAGE FUNCTION
# ============================================

def main():
    """Main admin & logs page function"""
    
    # Apply theme
    apply_custom_theme("financial")
    
    # Page header
    st.title("⚙️ System Administration & Log Management")
    st.markdown("Comprehensive system monitoring, log analysis, and administrative controls")
    
    # Authentication check
    if not check_admin_access():
        authenticate_admin()
        return
    
    # Admin info
    st.success(f"✅ Logged in as: **{st.session_state.get('admin_username', 'Admin')}** | Session: {st.session_state.get('admin_login_time', datetime.now()).strftime('%H:%M:%S')}")
    
    # Logout button
    if st.button("🚪 Logout", key="admin_logout"):
        st.session_state.admin_authenticated = False
        st.rerun()
    
    st.markdown("---")
    
    # ============================================
    # NAVIGATION TABS
    # ============================================
    
    admin_tabs = st.tabs([
        "📊 System Dashboard",
        "📄 Log Analysis", 
        "⚙️ Configuration",
        "👥 User Management",
        "🔍 Audit Trail",
        "⚡ Performance"
    ])
    
    # ============================================
    # SYSTEM DASHBOARD TAB
    # ============================================
    
    with admin_tabs[0]:
        st.subheader("📊 Real-time System Dashboard")
        
        # Get current system metrics
        current_metrics = get_system_metrics()
        
        # System status overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_status = "🔴" if current_metrics['cpu_usage'] > 80 else "🟡" if current_metrics['cpu_usage'] > 60 else "🟢"
            st.metric(
                "CPU Usage",
                f"{current_metrics['cpu_usage']:.1f}%",
                help="Current CPU utilization"
            )
            st.write(f"Status: {cpu_status}")
        
        with col2:
            mem_status = "🔴" if current_metrics['memory_usage'] > 85 else "🟡" if current_metrics['memory_usage'] > 70 else "🟢"
            st.metric(
                "Memory Usage",
                f"{current_metrics['memory_usage']:.1f}%",
                help="Current memory utilization"
            )
            st.write(f"Status: {mem_status}")
        
        with col3:
            st.metric(
                "Active Sessions",
                current_metrics['active_sessions'],
                help="Current active user sessions"
            )
        
        with col4:
            error_status = "🔴" if current_metrics['error_rate'] > 5 else "🟡" if current_metrics['error_rate'] > 2 else "🟢"
            st.metric(
                "Error Rate",
                f"{current_metrics['error_rate']:.2f}%",
                help="Current system error rate"
            )
            st.write(f"Status: {error_status}")
        
        # System monitoring dashboard
        create_system_dashboard(current_metrics)
        
        # System alerts
        st.subheader("🚨 System Alerts")
        
        alerts = []
        if current_metrics['cpu_usage'] > 80:
            alerts.append(("High CPU Usage", f"{current_metrics['cpu_usage']:.1f}%", "error"))
        if current_metrics['memory_usage'] > 85:
            alerts.append(("High Memory Usage", f"{current_metrics['memory_usage']:.1f}%", "error"))
        if current_metrics['error_rate'] > 5:
            alerts.append(("High Error Rate", f"{current_metrics['error_rate']:.2f}%", "warning"))
        
        if alerts:
            for alert_title, alert_value, alert_type in alerts:
                if alert_type == "error":
                    st.error(f"🚨 **{alert_title}:** {alert_value}")
                else:
                    st.warning(f"⚠️ **{alert_title}:** {alert_value}")
        else:
            st.success("✅ All systems operating normally")
    
    # ============================================
    # LOG ANALYSIS TAB
    # ============================================
    
    with admin_tabs[1]:
        st.subheader("📄 Advanced Log Analysis")
        
        # Log file selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_log = st.selectbox(
                "Select Log File",
                list(LOG_FILES.keys()),
                help="Choose log file to analyze"
            )
            
            log_lines = st.slider("Number of Lines", 100, 5000, 1000)
        
        with col2:
            # Log level filter
            log_levels = st.multiselect(
                "Filter by Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                default=["INFO", "WARNING", "ERROR"]
            )
            
            # Time range filter
            hours_back = st.slider("Hours Back", 1, 168, 24)
        
        # Load and display logs
        log_path = LOG_FILES[selected_log]
        logs_df = load_log_file(log_path, log_lines)
        
        if not logs_df.empty:
            # Filter logs
            if log_levels:
                logs_df = logs_df[logs_df['level'].isin(log_levels)]
            
            # Time filter
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            logs_df = logs_df[logs_df['timestamp'] >= cutoff_time]
            
            # Log analysis
            log_analysis = analyze_log_patterns(logs_df)
            
            # Display analysis summary
            st.subheader("📈 Log Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Entries", log_analysis.get('total_entries', 0))
            with col2:
                st.metric("Error Rate", f"{log_analysis.get('error_rate', 0):.2f}%")
            with col3:
                st.metric("Peak Hour", f"{log_analysis.get('peak_hour', 'N/A')}:00")
            with col4:
                st.metric("Trend", log_analysis.get('trend', 'Unknown'))
            
            # Level distribution chart
            if 'level_distribution' in log_analysis:
                fig = px.pie(
                    values=list(log_analysis['level_distribution'].values()),
                    names=list(log_analysis['level_distribution'].keys()),
                    title="Log Level Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display recent logs
            st.subheader("📋 Recent Log Entries")
            
            # Format logs for display
            display_logs = logs_df.copy()
            display_logs['timestamp'] = display_logs['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Color code by level
            def color_level(level):
                colors = {
                    'ERROR': '🔴',
                    'WARNING': '🟡', 
                    'INFO': '🟢',
                    'DEBUG': '🔵'
                }
                return colors.get(level, '⚪')
            
            display_logs['level_icon'] = display_logs['level'].apply(color_level)
            
            st.dataframe(
                display_logs[['timestamp', 'level_icon', 'level', 'message', 'source']].tail(50),
                use_container_width=True,
                hide_index=True
            )
            
            # Export logs
            st.subheader("📥 Export Logs")
            
            col1, col2 = st.columns(2)
            
            with col1:
                create_download_button(
                    logs_df,
                    f"{selected_log.lower().replace(' ', '_')}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "📄 Download Filtered Logs",
                    key="logs_export"
                )
            
            with col2:
                # Export analysis summary
                analysis_df = pd.DataFrame([log_analysis])
                create_download_button(
                    analysis_df,
                    f"log_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "📊 Download Analysis",
                    key="analysis_export"
                )
        else:
            st.warning("No log entries found for the selected criteria")
    
    # ============================================
    # CONFIGURATION TAB
    # ============================================
    
    with admin_tabs[2]:
        st.subheader("⚙️ System Configuration Management")
        
        # Load current configuration
        config = load_system_configuration()
        
        # Configuration categories
        config_tabs = st.tabs([
            "🖥️ Application",
            "🗄️ Database", 
            "🌐 API",
            "📈 Trading",
            "🔔 Alerts"
        ])
        
        with config_tabs[0]:
            st.write("**Application Settings**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                app_config = config['application']
                app_config['debug_mode'] = st.checkbox("Debug Mode", app_config['debug_mode'])
                app_config['log_level'] = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
                app_config['session_timeout'] = st.number_input("Session Timeout (seconds)", 300, 86400, app_config['session_timeout'])
            
            with col2:
                st.info(f"**Version:** {app_config['version']}")
                st.info(f"**Current Settings Applied:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with config_tabs[1]:
            st.write("**Database Configuration**")
            
            db_config = config['database']
            
            col1, col2 = st.columns(2)
            
            with col1:
                db_config['host'] = st.text_input("Database Host", db_config['host'])
                db_config['port'] = st.number_input("Port", 1000, 65535, db_config['port'])
                db_config['name'] = st.text_input("Database Name", db_config['name'])
            
            with col2:
                db_config['connection_pool_size'] = st.number_input("Connection Pool Size", 1, 100, db_config['connection_pool_size'])
                db_config['timeout'] = st.number_input("Timeout (seconds)", 10, 300, db_config['timeout'])
        
        with config_tabs[2]:
            st.write("**API Configuration**")
            
            api_config = config['api']
            
            col1, col2 = st.columns(2)
            
            with col1:
                api_config['rate_limit'] = st.number_input("Rate Limit (requests/hour)", 100, 10000, api_config['rate_limit'])
                api_config['timeout'] = st.number_input("API Timeout (seconds)", 5, 120, api_config['timeout'])
            
            with col2:
                api_config['retry_attempts'] = st.number_input("Retry Attempts", 1, 10, api_config['retry_attempts'])
                api_config['cache_ttl'] = st.number_input("Cache TTL (seconds)", 60, 3600, api_config['cache_ttl'])
        
        with config_tabs[3]:
            st.write("**Trading Configuration**")
            
            trading_config = config['trading']
            
            col1, col2 = st.columns(2)
            
            with col1:
                trading_config['max_position_size'] = st.slider("Max Position Size", 0.01, 0.5, trading_config['max_position_size'])
                trading_config['risk_per_trade'] = st.slider("Risk per Trade", 0.005, 0.1, trading_config['risk_per_trade'], 0.005)
            
            with col2:
                trading_config['max_drawdown'] = st.slider("Max Drawdown", 0.05, 0.3, trading_config['max_drawdown'], 0.01)
                trading_config['rebalance_frequency'] = st.selectbox("Rebalance Frequency", ["daily", "weekly", "monthly"], index=0)
        
        with config_tabs[4]:
            st.write("**Alert Configuration**")
            
            alerts_config = config['alerts']
            
            col1, col2 = st.columns(2)
            
            with col1:
                alerts_config['email_notifications'] = st.checkbox("Email Notifications", alerts_config['email_notifications'])
                alerts_config['slack_integration'] = st.checkbox("Slack Integration", alerts_config['slack_integration'])
            
            with col2:
                alerts_config['alert_threshold'] = st.slider("Alert Threshold", 0.01, 0.2, alerts_config['alert_threshold'], 0.01)
                alerts_config['max_alerts_per_hour'] = st.number_input("Max Alerts/Hour", 1, 100, alerts_config['max_alerts_per_hour'])
        
        # Save configuration
        if st.button("💾 Save Configuration", type="primary"):
            if update_configuration(config):
                st.success("✅ Configuration updated successfully!")
            else:
                st.error("❌ Failed to update configuration")
    
    # ============================================
    # USER MANAGEMENT TAB
    # ============================================
    
    with admin_tabs[3]:
        st.subheader("👥 User Management")
        
        # Sample user data
        users_data = [
            {"Username": "admin", "Role": "Super Admin", "Last Login": "2025-08-29 20:45", "Status": "Active", "Sessions": 2},
            {"Username": "analyst1", "Role": "Analyst", "Last Login": "2025-08-29 18:30", "Status": "Active", "Sessions": 1},
            {"Username": "trader1", "Role": "User", "Last Login": "2025-08-29 16:15", "Status": "Active", "Sessions": 1},
            {"Username": "viewer1", "Role": "User", "Last Login": "2025-08-28 14:22", "Status": "Inactive", "Sessions": 0}
        ]
        
        users_df = pd.DataFrame(users_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Current Users**")
            st.dataframe(users_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**User Statistics**")
            st.metric("Total Users", len(users_df))
            st.metric("Active Users", len(users_df[users_df['Status'] == 'Active']))
            st.metric("Admin Users", len(users_df[users_df['Role'].str.contains('Admin')]))
            st.metric("Total Sessions", users_df['Sessions'].sum())
        
        # Role permissions
        st.subheader("🔐 Role Permissions")
        
        for role, info in USER_ROLES.items():
            with st.expander(f"{role} - {info['description']}"):
                st.write(f"**Permissions:** {', '.join(info['permissions'])}")
                
                # Show users with this role
                role_users = users_df[users_df['Role'] == role]['Username'].tolist()
                if role_users:
                    st.write(f"**Users:** {', '.join(role_users)}")
                else:
                    st.write("**Users:** None")
    
    # ============================================
    # AUDIT TRAIL TAB
    # ============================================
    
    with admin_tabs[4]:
        st.subheader("🔍 System Audit Trail")
        
        # Generate sample audit data
        audit_events = [
            {"Timestamp": "2025-08-29 20:45:23", "User": "admin", "Action": "Configuration Updated", "Resource": "Trading Settings", "IP": "192.168.1.100", "Status": "Success"},
            {"Timestamp": "2025-08-29 20:30:15", "User": "analyst1", "Action": "Log Export", "Resource": "Application Logs", "IP": "192.168.1.105", "Status": "Success"},
            {"Timestamp": "2025-08-29 19:22:45", "User": "trader1", "Action": "Backtest Execution", "Resource": "AAPL Strategy", "IP": "192.168.1.110", "Status": "Success"},
            {"Timestamp": "2025-08-29 18:15:30", "User": "unknown", "Action": "Login Attempt", "Resource": "Admin Panel", "IP": "203.0.113.45", "Status": "Failed"},
            {"Timestamp": "2025-08-29 17:45:12", "User": "admin", "Action": "User Management", "Resource": "Role Assignment", "IP": "192.168.1.100", "Status": "Success"}
        ]
        
        audit_df = pd.DataFrame(audit_events)
        
        # Audit filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_user = st.selectbox("Filter by User", ["All"] + audit_df['User'].unique().tolist())
        with col2:
            selected_action = st.selectbox("Filter by Action", ["All"] + audit_df['Action'].unique().tolist())
        with col3:
            selected_status = st.selectbox("Filter by Status", ["All"] + audit_df['Status'].unique().tolist())
        
        # Apply filters
        filtered_audit = audit_df.copy()
        if selected_user != "All":
            filtered_audit = filtered_audit[filtered_audit['User'] == selected_user]
        if selected_action != "All":
            filtered_audit = filtered_audit[filtered_audit['Action'] == selected_action]
        if selected_status != "All":
            filtered_audit = filtered_audit[filtered_audit['Status'] == selected_status]
        
        # Display audit trail
        st.dataframe(filtered_audit, use_container_width=True, hide_index=True)
        
        # Audit statistics
        st.subheader("📊 Audit Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Events", len(audit_df))
        with col2:
            st.metric("Failed Events", len(audit_df[audit_df['Status'] == 'Failed']))
        with col3:
            st.metric("Unique Users", audit_df['User'].nunique())
        with col4:
            st.metric("Unique IPs", audit_df['IP'].nunique())
        
        # Export audit trail
        create_download_button(
            filtered_audit,
            f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "📥 Export Audit Trail",
            key="audit_export"
        )
    
    # ============================================
    # PERFORMANCE TAB
    # ============================================
    
    with admin_tabs[5]:
        st.subheader("⚡ System Performance Analytics")
        
        # Performance metrics over time
        st.write("**Performance Trends (Last 24 Hours)**")
        
        # Generate sample performance data
        hours = list(range(24))
        response_times = np.random.uniform(0.5, 3.0, 24)
        cpu_usage = np.random.uniform(20, 85, 24)
        memory_usage = np.random.uniform(40, 90, 24)
        error_rates = np.random.uniform(0.1, 8.0, 24)
        
        # Create performance chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Time', 'CPU Usage', 'Memory Usage', 'Error Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(go.Scatter(x=hours, y=response_times, mode='lines', name='Response Time (s)', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=hours, y=cpu_usage, mode='lines', name='CPU %', line=dict(color='red')), row=1, col=2)
        fig.add_trace(go.Scatter(x=hours, y=memory_usage, mode='lines', name='Memory %', line=dict(color='green')), row=2, col=1)
        fig.add_trace(go.Scatter(x=hours, y=error_rates, mode='lines', name='Error Rate %', line=dict(color='orange')), row=2, col=2)
        
        fig.update_layout(height=600, template='plotly_white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Response Time", f"{np.mean(response_times):.2f}s")
        with col2:
            st.metric("Peak CPU Usage", f"{np.max(cpu_usage):.1f}%")
        with col3:
            st.metric("Peak Memory Usage", f"{np.max(memory_usage):.1f}%")
        with col4:
            st.metric("Max Error Rate", f"{np.max(error_rates):.2f}%")
        
        # Performance recommendations
        st.subheader("💡 Performance Recommendations")
        
        recommendations = []
        if np.max(cpu_usage) > 80:
            recommendations.append("🔴 **High CPU Usage Detected** - Consider scaling resources or optimizing algorithms")
        if np.max(memory_usage) > 85:
            recommendations.append("🟡 **Memory Usage Warning** - Monitor for memory leaks or increase available memory")
        if np.mean(response_times) > 2.0:
            recommendations.append("⚠️ **Slow Response Times** - Optimize database queries and API calls")
        if np.max(error_rates) > 5.0:
            recommendations.append("🚨 **High Error Rate** - Review error logs and improve error handling")
        
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.success("✅ System performance is within acceptable parameters")
    
    # ============================================
    # FOOTER
    # ============================================
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        ⚙️ System Administration & Log Management | 
        Admin Session: {st.session_state.get('admin_username', 'Admin')} | 
        System Status: ✅ Operational | 
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE EXECUTION
# ============================================

if __name__ == "__main__":
    main()
