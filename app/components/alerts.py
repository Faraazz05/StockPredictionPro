"""
app/components/alerts.py

Advanced alerting system for StockPredictionPro Streamlit application.
Integrates with trading signals, risk management, portfolio monitoring,
and strategy notifications to provide real-time alerts to users.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import sys
from pathlib import Path

# Add project root to path for trading imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your trading modules
try:
    from src.trading.signals.technical_signals import TechnicalSignalGenerator
    from src.trading.signals.classification_signals import ClassificationSignals
    from src.trading.signals.regression_signals import RegressionSignals
    from src.trading.signals.composite_signals import CompositeSignals
    from src.trading.risk.portfolio_risk import PortfolioRiskManager
    from src.trading.risk.position_sizing import PositionSizer
    from src.trading.risk.stop_loss import StopLossManager
    from src.trading.risk.take_profit import TakeProfitManager
    from src.trading.strategies.momentum import MomentumStrategy
    from src.trading.strategies.mean_reversion import MeanReversionStrategy
    from src.trading.strategies.trend_following import TrendFollowingStrategy
    from src.trading.portfolio import Portfolio
except ImportError:
    # Mock imports if not available
    TechnicalSignalGenerator = ClassificationSignals = RegressionSignals = None
    CompositeSignals = PortfolioRiskManager = PositionSizer = None
    StopLossManager = TakeProfitManager = MomentumStrategy = None
    MeanReversionStrategy = TrendFollowingStrategy = Portfolio = None

# ============================================
# ALERT TYPES AND LEVELS
# ============================================

class AlertType(Enum):
    """Alert types for categorization"""
    TRADING_SIGNAL = "trading_signal"
    RISK_WARNING = "risk_warning"
    PORTFOLIO_UPDATE = "portfolio_update"
    STRATEGY_ALERT = "strategy_alert"
    MARKET_NEWS = "market_news"
    SYSTEM_NOTIFICATION = "system_notification"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# ============================================
# ALERT CLASSES
# ============================================

class Alert:
    """Represents a single alert with metadata"""
    
    def __init__(self, 
                 message: str,
                 alert_type: AlertType = AlertType.SYSTEM_NOTIFICATION,
                 level: AlertLevel = AlertLevel.INFO,
                 symbol: str = None,
                 timestamp: datetime = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize alert
        
        Args:
            message: Alert message
            alert_type: Type of alert
            level: Alert severity level
            symbol: Associated stock symbol
            timestamp: Alert timestamp
            metadata: Additional alert data
        """
        self.message = message
        self.alert_type = alert_type
        self.level = level
        self.symbol = symbol
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.id = f"{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(message) % 10000}"
    
    def __str__(self) -> str:
        """String representation"""
        symbol_part = f"[{self.symbol}] " if self.symbol else ""
        return f"{symbol_part}{self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'message': self.message,
            'type': self.alert_type.value,
            'level': self.level.value,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class AlertManager:
    """Manages collection of alerts and their display"""
    
    def __init__(self, max_alerts: int = 100):
        """
        Initialize alert manager
        
        Args:
            max_alerts: Maximum number of alerts to keep
        """
        self.alerts: List[Alert] = []
        self.max_alerts = max_alerts
        self.alert_counts = {level: 0 for level in AlertLevel}
    
    def add_alert(self, 
                  message: str,
                  alert_type: AlertType = AlertType.SYSTEM_NOTIFICATION,
                  level: AlertLevel = AlertLevel.INFO,
                  symbol: str = None,
                  metadata: Dict[str, Any] = None) -> Alert:
        """
        Add new alert
        
        Args:
            message: Alert message
            alert_type: Type of alert
            level: Alert severity level
            symbol: Associated stock symbol
            metadata: Additional alert data
            
        Returns:
            Created alert
        """
        alert = Alert(message, alert_type, level, symbol, metadata=metadata)
        self.alerts.append(alert)
        self.alert_counts[level] += 1
        
        # Keep only max_alerts
        if len(self.alerts) > self.max_alerts:
            removed_alert = self.alerts.pop(0)
            self.alert_counts[removed_alert.level] -= 1
        
        return alert
    
    def get_alerts_by_type(self, alert_type: AlertType) -> List[Alert]:
        """Get alerts by type"""
        return [alert for alert in self.alerts if alert.alert_type == alert_type]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by level"""
        return [alert for alert in self.alerts if alert.level == level]
    
    def get_alerts_by_symbol(self, symbol: str) -> List[Alert]:
        """Get alerts for specific symbol"""
        return [alert for alert in self.alerts if alert.symbol == symbol]
    
    def clear_alerts(self, alert_type: AlertType = None, level: AlertLevel = None):
        """
        Clear alerts with optional filtering
        
        Args:
            alert_type: Only clear alerts of this type
            level: Only clear alerts of this level
        """
        if alert_type is None and level is None:
            # Clear all
            self.alerts.clear()
            self.alert_counts = {level: 0 for level in AlertLevel}
        else:
            # Filter and remove
            remaining_alerts = []
            for alert in self.alerts:
                should_remove = True
                if alert_type and alert.alert_type != alert_type:
                    should_remove = False
                if level and alert.level != level:
                    should_remove = False
                
                if should_remove:
                    self.alert_counts[alert.level] -= 1
                else:
                    remaining_alerts.append(alert)
            
            self.alerts = remaining_alerts

# ============================================
# ALERT RENDERING
# ============================================

def render_alert(alert: Alert, show_timestamp: bool = True, show_symbol: bool = True) -> None:
    """
    Render single alert in Streamlit
    
    Args:
        alert: Alert to render
        show_timestamp: Whether to show timestamp
        show_symbol: Whether to show symbol
    """
    # Format message
    message_parts = []
    
    if show_timestamp:
        message_parts.append(f"**{alert.timestamp.strftime('%H:%M:%S')}**")
    
    if show_symbol and alert.symbol:
        message_parts.append(f"**[{alert.symbol}]**")
    
    message_parts.append(alert.message)
    
    formatted_message = " ".join(message_parts)
    
    # Display based on level
    if alert.level == AlertLevel.INFO:
        st.info(formatted_message)
    elif alert.level == AlertLevel.SUCCESS:
        st.success(formatted_message)
    elif alert.level == AlertLevel.WARNING:
        st.warning(formatted_message)
    elif alert.level == AlertLevel.ERROR:
        st.error(formatted_message)
    elif alert.level == AlertLevel.CRITICAL:
        st.error(f"🚨 **CRITICAL**: {formatted_message}")
    else:
        st.write(formatted_message)

def render_alert_summary(alert_manager: AlertManager) -> None:
    """Render alert summary statistics"""
    if not alert_manager.alerts:
        st.info("🔕 No alerts")
        return
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", len(alert_manager.alerts))
    
    with col2:
        critical_count = alert_manager.alert_counts[AlertLevel.CRITICAL]
        error_count = alert_manager.alert_counts[AlertLevel.ERROR]
        st.metric("Critical/Error", critical_count + error_count)
    
    with col3:
        warning_count = alert_manager.alert_counts[AlertLevel.WARNING]
        st.metric("Warnings", warning_count)
    
    with col4:
        success_count = alert_manager.alert_counts[AlertLevel.SUCCESS]
        st.metric("Success", success_count)

def render_alerts_panel(alert_manager: AlertManager, 
                       max_display: int = 10,
                       filter_type: AlertType = None,
                       filter_level: AlertLevel = None,
                       filter_symbol: str = None) -> None:
    """
    Render alerts panel with filtering
    
    Args:
        alert_manager: Alert manager instance
        max_display: Maximum alerts to display
        filter_type: Filter by alert type
        filter_level: Filter by alert level
        filter_symbol: Filter by symbol
    """
    if not alert_manager.alerts:
        st.info("🔕 No alerts to display")
        return
    
    # Apply filters
    filtered_alerts = alert_manager.alerts
    
    if filter_type:
        filtered_alerts = [a for a in filtered_alerts if a.alert_type == filter_type]
    
    if filter_level:
        filtered_alerts = [a for a in filtered_alerts if a.level == filter_level]
    
    if filter_symbol:
        filtered_alerts = [a for a in filtered_alerts if a.symbol == filter_symbol]
    
    # Sort by timestamp (newest first)
    filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
    
    # Display alerts
    for i, alert in enumerate(filtered_alerts[:max_display]):
        render_alert(alert)
    
    if len(filtered_alerts) > max_display:
        st.caption(f"Showing {max_display} of {len(filtered_alerts)} alerts")

# ============================================
# TRADING SIGNAL ALERTS
# ============================================

def generate_signal_alerts(df: pd.DataFrame, 
                          symbol: str,
                          alert_manager: AlertManager) -> None:
    """
    Generate alerts from trading signals
    
    Args:
        df: Price data DataFrame
        symbol: Stock symbol
        alert_manager: Alert manager to add alerts to
    """
    if df is None or df.empty:
        return
    
    try:
        # Technical signals
        if TechnicalSignalGenerator:
            tech_generator = TechnicalSignalGenerator()
            signals = tech_generator.generate_signals(df)
            
            for signal in signals:
                level = AlertLevel.INFO
                if signal.get('strength', 0) > 0.7:
                    level = AlertLevel.SUCCESS if signal.get('direction') == 'BUY' else AlertLevel.WARNING
                
                message = f"{signal.get('type', 'Signal')}: {signal.get('description', 'Unknown signal')}"
                alert_manager.add_alert(
                    message=message,
                    alert_type=AlertType.TRADING_SIGNAL,
                    level=level,
                    symbol=symbol,
                    metadata=signal
                )
    
    except Exception as e:
        alert_manager.add_alert(
            message=f"Error generating signals: {str(e)}",
            alert_type=AlertType.SYSTEM_NOTIFICATION,
            level=AlertLevel.ERROR,
            symbol=symbol
        )

def generate_simple_signal_alerts(df: pd.DataFrame,
                                 symbol: str,
                                 alert_manager: AlertManager) -> None:
    """
    Generate simple signal alerts (fallback implementation)
    
    Args:
        df: Price data DataFrame
        symbol: Stock symbol
        alert_manager: Alert manager to add alerts to
    """
    if df is None or df.empty or len(df) < 20:
        return
    
    try:
        # Simple RSI signal
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        latest_rsi = rsi.iloc[-1]
        if not pd.isna(latest_rsi):
            if latest_rsi > 70:
                alert_manager.add_alert(
                    message=f"RSI Overbought: {latest_rsi:.1f}",
                    alert_type=AlertType.TRADING_SIGNAL,
                    level=AlertLevel.WARNING,
                    symbol=symbol,
                    metadata={'rsi': latest_rsi, 'type': 'overbought'}
                )
            elif latest_rsi < 30:
                alert_manager.add_alert(
                    message=f"RSI Oversold: {latest_rsi:.1f}",
                    alert_type=AlertType.TRADING_SIGNAL,
                    level=AlertLevel.SUCCESS,
                    symbol=symbol,
                    metadata={'rsi': latest_rsi, 'type': 'oversold'}
                )
        
        # Simple price movement alert
        if len(df) >= 2:
            price_change_pct = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100
            
            if abs(price_change_pct) > 5:  # 5% move
                level = AlertLevel.SUCCESS if price_change_pct > 0 else AlertLevel.WARNING
                direction = "up" if price_change_pct > 0 else "down"
                
                alert_manager.add_alert(
                    message=f"Large price move: {price_change_pct:+.2f}% {direction}",
                    alert_type=AlertType.TRADING_SIGNAL,
                    level=level,
                    symbol=symbol,
                    metadata={'price_change_pct': price_change_pct}
                )
    
    except Exception as e:
        alert_manager.add_alert(
            message=f"Error in signal calculation: {str(e)}",
            alert_type=AlertType.SYSTEM_NOTIFICATION,
            level=AlertLevel.ERROR,
            symbol=symbol
        )

# ============================================
# RISK MANAGEMENT ALERTS
# ============================================

def generate_risk_alerts(portfolio_data: Dict[str, Any],
                        alert_manager: AlertManager) -> None:
    """
    Generate risk management alerts
    
    Args:
        portfolio_data: Portfolio information
        alert_manager: Alert manager to add alerts to
    """
    if not portfolio_data:
        return
    
    try:
        # Portfolio risk analysis
        if PortfolioRiskManager:
            risk_manager = PortfolioRiskManager()
            risk_metrics = risk_manager.assess_risk(portfolio_data)
            
            for metric_name, metric_value in risk_metrics.items():
                if metric_name == 'var_95' and metric_value > 0.1:  # 10% VaR threshold
                    alert_manager.add_alert(
                        message=f"High portfolio VaR: {metric_value:.2%}",
                        alert_type=AlertType.RISK_WARNING,
                        level=AlertLevel.WARNING,
                        metadata={'var_95': metric_value}
                    )
                
                elif metric_name == 'volatility' and metric_value > 0.3:  # 30% volatility
                    alert_manager.add_alert(
                        message=f"High portfolio volatility: {metric_value:.2%}",
                        alert_type=AlertType.RISK_WARNING,
                        level=AlertLevel.WARNING,
                        metadata={'volatility': metric_value}
                    )
    
    except Exception as e:
        alert_manager.add_alert(
            message=f"Error in risk analysis: {str(e)}",
            alert_type=AlertType.SYSTEM_NOTIFICATION,
            level=AlertLevel.ERROR
        )

def generate_position_alerts(positions: List[Dict[str, Any]],
                           alert_manager: AlertManager) -> None:
    """
    Generate position-specific alerts
    
    Args:
        positions: List of position dictionaries
        alert_manager: Alert manager to add alerts to
    """
    if not positions:
        return
    
    for position in positions:
        try:
            symbol = position.get('symbol', 'Unknown')
            current_price = position.get('current_price', 0)
            entry_price = position.get('entry_price', 0)
            stop_loss = position.get('stop_loss')
            take_profit = position.get('take_profit')
            
            if entry_price > 0:
                pnl_pct = ((current_price / entry_price) - 1) * 100
                
                # Stop loss alerts
                if stop_loss and current_price <= stop_loss:
                    alert_manager.add_alert(
                        message=f"Stop loss triggered at ${current_price:.2f}",
                        alert_type=AlertType.RISK_WARNING,
                        level=AlertLevel.CRITICAL,
                        symbol=symbol,
                        metadata={'stop_loss': stop_loss, 'current_price': current_price}
                    )
                
                # Take profit alerts
                if take_profit and current_price >= take_profit:
                    alert_manager.add_alert(
                        message=f"Take profit target reached at ${current_price:.2f}",
                        alert_type=AlertType.TRADING_SIGNAL,
                        level=AlertLevel.SUCCESS,
                        symbol=symbol,
                        metadata={'take_profit': take_profit, 'current_price': current_price}
                    )
                
                # Large P&L alerts
                if abs(pnl_pct) > 10:  # 10% move
                    level = AlertLevel.SUCCESS if pnl_pct > 0 else AlertLevel.WARNING
                    alert_manager.add_alert(
                        message=f"Large P&L: {pnl_pct:+.2f}%",
                        alert_type=AlertType.PORTFOLIO_UPDATE,
                        level=level,
                        symbol=symbol,
                        metadata={'pnl_pct': pnl_pct}
                    )
        
        except Exception as e:
            alert_manager.add_alert(
                message=f"Error processing position {position.get('symbol', 'Unknown')}: {str(e)}",
                alert_type=AlertType.SYSTEM_NOTIFICATION,
                level=AlertLevel.ERROR
            )

# ============================================
# STREAMLIT INTEGRATION
# ============================================

def create_alerts_sidebar(alert_manager: AlertManager) -> None:
    """Create alerts sidebar for Streamlit app"""
    with st.sidebar:
        st.subheader("🚨 Alerts")
        
        # Alert summary
        render_alert_summary(alert_manager)
        
        # Quick filters
        st.write("**Filter Alerts:**")
        filter_level = st.selectbox(
            "Alert Level",
            options=[None] + [level.value for level in AlertLevel],
            format_func=lambda x: "All Levels" if x is None else x.title()
        )
        
        filter_type = st.selectbox(
            "Alert Type", 
            options=[None] + [alert_type.value for alert_type in AlertType],
            format_func=lambda x: "All Types" if x is None else x.replace('_', ' ').title()
        )
        
        # Convert back to enums
        filter_level_enum = AlertLevel(filter_level) if filter_level else None
        filter_type_enum = AlertType(filter_type) if filter_type else None
        
        # Display recent alerts
        st.write("**Recent Alerts:**")
        render_alerts_panel(
            alert_manager,
            max_display=5,
            filter_level=filter_level_enum,
            filter_type=filter_type_enum
        )
        
        # Clear alerts button
        if st.button("🗑️ Clear All Alerts"):
            alert_manager.clear_alerts()
            st.rerun()

def create_alerts_main_panel(alert_manager: AlertManager) -> None:
    """Create main alerts panel"""
    st.subheader("🚨 Alert Center")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_timestamp = st.checkbox("Show Timestamps", value=True)
    
    with col2:
        show_symbol = st.checkbox("Show Symbols", value=True)
    
    with col3:
        max_display = st.slider("Max Alerts", min_value=5, max_value=50, value=20)
    
    st.markdown("---")
    
    # Alerts display
    render_alerts_panel(alert_manager, max_display=max_display)

# ============================================
# GLOBAL ALERT MANAGER
# ============================================

# Global alert manager instance
_global_alert_manager = None

def get_alert_manager() -> AlertManager:
    """Get global alert manager instance"""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
    return _global_alert_manager

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    """
    Example usage patterns:
    
    # Get global alert manager
    alert_manager = get_alert_manager()
    
    # Add alerts
    alert_manager.add_alert(
        "Strong buy signal detected",
        AlertType.TRADING_SIGNAL,
        AlertLevel.SUCCESS,
        "AAPL"
    )
    
    # Generate signals from data
    generate_simple_signal_alerts(stock_df, "AAPL", alert_manager)
    
    # Generate risk alerts
    portfolio_data = {"total_value": 100000, "positions": [...]}
    generate_risk_alerts(portfolio_data, alert_manager)
    
    # Display in Streamlit
    create_alerts_sidebar(alert_manager)
    create_alerts_main_panel(alert_manager)
    """
    pass
