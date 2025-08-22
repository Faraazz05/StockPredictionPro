# ============================================
# StockPredictionPro - src/evaluation/backtesting/risk_management.py
# Advanced risk management system for financial backtesting and trading strategies
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import math

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('evaluation.backtesting.risk_management')

# ============================================
# Risk Management Data Structures
# ============================================

class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskType(Enum):
    """Types of financial risks"""
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    OPERATIONAL_RISK = "operational_risk"
    CONCENTRATION_RISK = "concentration_risk"
    LEVERAGE_RISK = "leverage_risk"
    DRAWDOWN_RISK = "drawdown_risk"
    VOLATILITY_RISK = "volatility_risk"

@dataclass
class RiskAlert:
    """Container for risk alerts"""
    timestamp: pd.Timestamp
    risk_type: RiskType
    risk_level: RiskLevel
    symbol: Optional[str]
    message: str
    metric_value: float
    threshold: float
    suggested_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    # Portfolio-level risks
    portfolio_var: float = 0.0          # Value at Risk
    portfolio_cvar: float = 0.0         # Conditional Value at Risk
    max_drawdown: float = 0.0           # Maximum drawdown
    current_drawdown: float = 0.0       # Current drawdown
    
    # Concentration risks
    max_position_weight: float = 0.0    # Largest position as % of portfolio
    sector_concentration: Dict[str, float] = field(default_factory=dict)
    
    # Leverage and margin
    gross_leverage: float = 1.0         # Gross leverage ratio
    net_leverage: float = 1.0           # Net leverage ratio
    margin_utilization: float = 0.0     # Margin usage percentage
    
    # Volatility measures
    portfolio_volatility: float = 0.0   # Annualized volatility
    rolling_volatility: float = 0.0     # Short-term volatility
    volatility_regime: str = "normal"   # Current volatility regime
    
    # Liquidity measures
    liquidity_score: float = 1.0        # Overall liquidity score
    days_to_liquidate: float = 1.0      # Estimated days to liquidate portfolio
    
    # Time-based measures
    sharpe_ratio: float = 0.0           # Risk-adjusted return
    sortino_ratio: float = 0.0          # Downside risk-adjusted return
    calmar_ratio: float = 0.0           # Return to max drawdown ratio

# ============================================
# Risk Rule Base Classes
# ============================================

class RiskRule(ABC):
    """Base class for risk management rules"""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.alerts_generated = 0
        self.last_triggered = None
    
    @abstractmethod
    def evaluate(self, portfolio_state: Any, market_data: Dict[str, Any]) -> List[RiskAlert]:
        """Evaluate the risk rule and return alerts if triggered"""
        pass
    
    def is_triggered(self, current_value: float, threshold: float, comparison: str = "greater") -> bool:
        """Check if rule is triggered based on comparison"""
        if comparison == "greater":
            return current_value > threshold
        elif comparison == "less":
            return current_value < threshold
        elif comparison == "greater_equal":
            return current_value >= threshold
        elif comparison == "less_equal":
            return current_value <= threshold
        else:
            raise ValueError(f"Unknown comparison: {comparison}")

class PositionSizeRule(RiskRule):
    """Rule for maximum position size limits"""
    
    def __init__(self, max_position_pct: float = 0.1, max_sector_pct: float = 0.3):
        super().__init__("Position Size Rule")
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
    
    def evaluate(self, portfolio_state: Any, market_data: Dict[str, Any]) -> List[RiskAlert]:
        alerts = []
        
        if not hasattr(portfolio_state, 'positions') or not portfolio_state.positions:
            return alerts
        
        total_value = portfolio_state.total_value
        if total_value <= 0:
            return alerts
        
        # Check individual position sizes
        for symbol, position in portfolio_state.positions.items():
            position_value = abs(position.quantity * position.market_price)
            position_weight = position_value / total_value
            
            if position_weight > self.max_position_pct:
                alerts.append(RiskAlert(
                    timestamp=portfolio_state.timestamp,
                    risk_type=RiskType.CONCENTRATION_RISK,
                    risk_level=RiskLevel.HIGH if position_weight > self.max_position_pct * 1.5 else RiskLevel.MEDIUM,
                    symbol=symbol,
                    message=f"Position size exceeds limit: {position_weight:.2%} > {self.max_position_pct:.2%}",
                    metric_value=position_weight,
                    threshold=self.max_position_pct,
                    suggested_action="Reduce position size or increase diversification"
                ))
        
        return alerts

class DrawdownRule(RiskRule):
    """Rule for maximum drawdown limits"""
    
    def __init__(self, max_drawdown_pct: float = 0.15, warning_drawdown_pct: float = 0.10):
        super().__init__("Drawdown Rule")
        self.max_drawdown_pct = max_drawdown_pct
        self.warning_drawdown_pct = warning_drawdown_pct
    
    def evaluate(self, portfolio_state: Any, market_data: Dict[str, Any]) -> List[RiskAlert]:
        alerts = []
        
        if not hasattr(portfolio_state, 'drawdown'):
            return alerts
        
        current_dd = portfolio_state.drawdown
        
        if current_dd > self.max_drawdown_pct:
            alerts.append(RiskAlert(
                timestamp=portfolio_state.timestamp,
                risk_type=RiskType.DRAWDOWN_RISK,
                risk_level=RiskLevel.CRITICAL,
                symbol=None,
                message=f"Maximum drawdown exceeded: {current_dd:.2%} > {self.max_drawdown_pct:.2%}",
                metric_value=current_dd,
                threshold=self.max_drawdown_pct,
                suggested_action="Consider reducing risk exposure or stopping trading"
            ))
        
        elif current_dd > self.warning_drawdown_pct:
            alerts.append(RiskAlert(
                timestamp=portfolio_state.timestamp,
                risk_type=RiskType.DRAWDOWN_RISK,
                risk_level=RiskLevel.HIGH,
                symbol=None,
                message=f"Drawdown warning: {current_dd:.2%} > {self.warning_drawdown_pct:.2%}",
                metric_value=current_dd,
                threshold=self.warning_drawdown_pct,
                suggested_action="Monitor closely and prepare risk reduction measures"
            ))
        
        return alerts

class LeverageRule(RiskRule):
    """Rule for leverage limits"""
    
    def __init__(self, max_gross_leverage: float = 2.0, max_net_leverage: float = 1.0):
        super().__init__("Leverage Rule")
        self.max_gross_leverage = max_gross_leverage
        self.max_net_leverage = max_net_leverage
    
    def evaluate(self, portfolio_state: Any, market_data: Dict[str, Any]) -> List[RiskAlert]:
        alerts = []
        
        if not hasattr(portfolio_state, 'positions') or not portfolio_state.positions:
            return alerts
        
        total_value = portfolio_state.total_value
        if total_value <= 0:
            return alerts
        
        # Calculate leverage
        gross_exposure = sum(abs(pos.quantity * pos.market_price) for pos in portfolio_state.positions.values())
        net_exposure = sum(pos.quantity * pos.market_price for pos in portfolio_state.positions.values())
        
        gross_leverage = gross_exposure / total_value
        net_leverage = abs(net_exposure) / total_value
        
        # Check gross leverage
        if gross_leverage > self.max_gross_leverage:
            alerts.append(RiskAlert(
                timestamp=portfolio_state.timestamp,
                risk_type=RiskType.LEVERAGE_RISK,
                risk_level=RiskLevel.HIGH,
                symbol=None,
                message=f"Gross leverage exceeded: {gross_leverage:.2f}x > {self.max_gross_leverage:.2f}x",
                metric_value=gross_leverage,
                threshold=self.max_gross_leverage,
                suggested_action="Reduce position sizes to lower leverage"
            ))
        
        # Check net leverage
        if net_leverage > self.max_net_leverage:
            alerts.append(RiskAlert(
                timestamp=portfolio_state.timestamp,
                risk_type=RiskType.LEVERAGE_RISK,
                risk_level=RiskLevel.MEDIUM,
                symbol=None,
                message=f"Net leverage exceeded: {net_leverage:.2f}x > {self.max_net_leverage:.2f}x",
                metric_value=net_leverage,
                threshold=self.max_net_leverage,
                suggested_action="Rebalance long/short positions"
            ))
        
        return alerts

class VolatilityRule(RiskRule):
    """Rule for portfolio volatility limits"""
    
    def __init__(self, max_portfolio_vol: float = 0.25, vol_lookback: int = 20):
        super().__init__("Volatility Rule")
        self.max_portfolio_vol = max_portfolio_vol
        self.vol_lookback = vol_lookback
        self.return_history = []
    
    def evaluate(self, portfolio_state: Any, market_data: Dict[str, Any]) -> List[RiskAlert]:
        alerts = []
        
        # Calculate portfolio return for this period
        if hasattr(self, 'prev_value'):
            if self.prev_value > 0:
                portfolio_return = (portfolio_state.total_value - self.prev_value) / self.prev_value
                self.return_history.append(portfolio_return)
                
                # Keep only recent history
                if len(self.return_history) > self.vol_lookback:
                    self.return_history = self.return_history[-self.vol_lookback:]
                
                # Calculate volatility if we have enough data
                if len(self.return_history) >= 10:
                    returns_array = np.array(self.return_history)
                    daily_vol = np.std(returns_array, ddof=1)
                    annualized_vol = daily_vol * np.sqrt(252)  # Assuming daily data
                    
                    if annualized_vol > self.max_portfolio_vol:
                        alerts.append(RiskAlert(
                            timestamp=portfolio_state.timestamp,
                            risk_type=RiskType.VOLATILITY_RISK,
                            risk_level=RiskLevel.MEDIUM,
                            symbol=None,
                            message=f"Portfolio volatility high: {annualized_vol:.2%} > {self.max_portfolio_vol:.2%}",
                            metric_value=annualized_vol,
                            threshold=self.max_portfolio_vol,
                            suggested_action="Consider reducing position sizes or adding hedges"
                        ))
        
        self.prev_value = portfolio_state.total_value
        return alerts

class VaRRule(RiskRule):
    """Rule for Value at Risk limits"""
    
    def __init__(self, max_var_pct: float = 0.05, confidence_level: float = 0.95, 
                 lookback_days: int = 252):
        super().__init__("VaR Rule")
        self.max_var_pct = max_var_pct
        self.confidence_level = confidence_level
        self.lookback_days = lookback_days
        self.return_history = []
    
    def evaluate(self, portfolio_state: Any, market_data: Dict[str, Any]) -> List[RiskAlert]:
        alerts = []
        
        # Calculate portfolio return
        if hasattr(self, 'prev_value'):
            if self.prev_value > 0:
                portfolio_return = (portfolio_state.total_value - self.prev_value) / self.prev_value
                self.return_history.append(portfolio_return)
                
                # Keep lookback period
                if len(self.return_history) > self.lookback_days:
                    self.return_history = self.return_history[-self.lookback_days:]
                
                # Calculate VaR if we have enough data
                if len(self.return_history) >= 50:
                    returns_array = np.array(self.return_history)
                    var_percentile = (1 - self.confidence_level) * 100
                    var_value = -np.percentile(returns_array, var_percentile)
                    
                    if var_value > self.max_var_pct:
                        alerts.append(RiskAlert(
                            timestamp=portfolio_state.timestamp,
                            risk_type=RiskType.MARKET_RISK,
                            risk_level=RiskLevel.HIGH,
                            symbol=None,
                            message=f"VaR exceeded: {var_value:.2%} > {self.max_var_pct:.2%}",
                            metric_value=var_value,
                            threshold=self.max_var_pct,
                            suggested_action="Reduce risk exposure or add hedges"
                        ))
        
        self.prev_value = portfolio_state.total_value
        return alerts

# ============================================
# Risk Manager
# ============================================

class RiskManager:
    """
    Comprehensive risk management system for backtesting and trading.
    
    This class monitors various risk factors and generates alerts when
    thresholds are breached. It supports customizable risk rules and
    provides detailed risk analytics.
    """
    
    def __init__(self, rules: Optional[List[RiskRule]] = None):
        """
        Initialize risk manager
        
        Args:
            rules: List of risk rules to apply. If None, uses default rules.
        """
        
        self.rules = rules or self._get_default_rules()
        self.alerts_history: List[RiskAlert] = []
        self.risk_metrics_history: List[RiskMetrics] = []
        
        # Performance tracking
        self.equity_curve_history = []
        self.return_history = []
        self.peak_value = 0
        
        logger.info(f"RiskManager initialized with {len(self.rules)} rules")
    
    def _get_default_rules(self) -> List[RiskRule]:
        """Get default set of risk rules"""
        return [
            PositionSizeRule(max_position_pct=0.15, max_sector_pct=0.4),
            DrawdownRule(max_drawdown_pct=0.20, warning_drawdown_pct=0.12),
            LeverageRule(max_gross_leverage=2.0, max_net_leverage=1.2),
            VolatilityRule(max_portfolio_vol=0.30),
            VaRRule(max_var_pct=0.06, confidence_level=0.95)
        ]
    
    @time_it("risk_evaluation")
    def evaluate_risk(self, portfolio_state: Any, market_data: Dict[str, Any]) -> Tuple[List[RiskAlert], RiskMetrics]:
        """
        Evaluate all risk rules and calculate risk metrics
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data
            
        Returns:
            Tuple of (alerts, risk_metrics)
        """
        
        alerts = []
        
        # Evaluate all risk rules
        for rule in self.rules:
            if rule.enabled:
                try:
                    rule_alerts = rule.evaluate(portfolio_state, market_data)
                    alerts.extend(rule_alerts)
                    rule.alerts_generated += len(rule_alerts)
                    
                    if rule_alerts:
                        rule.last_triggered = portfolio_state.timestamp
                        
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.name}: {e}")
        
        # Calculate comprehensive risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio_state, market_data)
        
        # Store history
        self.alerts_history.extend(alerts)
        self.risk_metrics_history.append(risk_metrics)
        
        # Update performance tracking
        self._update_performance_tracking(portfolio_state)
        
        # Log significant alerts
        for alert in alerts:
            if alert.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                logger.warning(f"Risk Alert [{alert.risk_level.value.upper()}]: {alert.message}")
        
        return alerts, risk_metrics
    
    def _calculate_risk_metrics(self, portfolio_state: Any, market_data: Dict[str, Any]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        metrics = RiskMetrics()
        
        if not hasattr(portfolio_state, 'positions') or not portfolio_state.positions:
            return metrics
        
        total_value = portfolio_state.total_value
        if total_value <= 0:
            return metrics
        
        # Position concentration analysis
        position_weights = []
        gross_exposure = 0
        net_exposure = 0
        
        for symbol, position in portfolio_state.positions.items():
            position_value = position.quantity * position.market_price
            position_weight = abs(position_value) / total_value
            position_weights.append(position_weight)
            
            gross_exposure += abs(position_value)
            net_exposure += position_value
        
        if position_weights:
            metrics.max_position_weight = max(position_weights)
        
        # Leverage metrics
        metrics.gross_leverage = gross_exposure / total_value
        metrics.net_leverage = abs(net_exposure) / total_value
        
        # Drawdown metrics
        if hasattr(portfolio_state, 'drawdown'):
            metrics.current_drawdown = portfolio_state.drawdown
        
        # Calculate max drawdown from history
        if len(self.equity_curve_history) > 1:
            equity_series = pd.Series(self.equity_curve_history)
            running_max = equity_series.expanding().max()
            drawdowns = (equity_series - running_max) / running_max
            metrics.max_drawdown = abs(drawdowns.min())
        
        # Volatility and risk-adjusted metrics
        if len(self.return_history) > 10:
            returns_array = np.array(self.return_history[-252:])  # Last year of data
            
            # Portfolio volatility
            daily_vol = np.std(returns_array, ddof=1)
            metrics.portfolio_volatility = daily_vol * np.sqrt(252)
            
            # Rolling volatility (last 20 days)
            if len(returns_array) >= 20:
                recent_vol = np.std(returns_array[-20:], ddof=1)
                metrics.rolling_volatility = recent_vol * np.sqrt(252)
                
                # Volatility regime detection
                if recent_vol > daily_vol * 1.5:
                    metrics.volatility_regime = "high"
                elif recent_vol < daily_vol * 0.7:
                    metrics.volatility_regime = "low"
                else:
                    metrics.volatility_regime = "normal"
            
            # Risk-adjusted returns
            if daily_vol > 0:
                mean_return = np.mean(returns_array)
                metrics.sharpe_ratio = (mean_return / daily_vol) * np.sqrt(252)
                
                # Sortino ratio (downside deviation)
                negative_returns = returns_array[returns_array < 0]
                if len(negative_returns) > 0:
                    downside_deviation = np.std(negative_returns, ddof=1)
                    if downside_deviation > 0:
                        metrics.sortino_ratio = (mean_return / downside_deviation) * np.sqrt(252)
                
                # Calmar ratio
                if metrics.max_drawdown > 0:
                    annual_return = (1 + mean_return) ** 252 - 1
                    metrics.calmar_ratio = annual_return / metrics.max_drawdown
            
            # VaR calculation
            var_95 = -np.percentile(returns_array, 5)
            cvar_95 = -np.mean(returns_array[returns_array <= -var_95])
            
            metrics.portfolio_var = var_95
            metrics.portfolio_cvar = cvar_95
        
        # Liquidity assessment (simplified)
        metrics.liquidity_score = self._assess_liquidity(portfolio_state, market_data)
        
        return metrics
    
    def _assess_liquidity(self, portfolio_state: Any, market_data: Dict[str, Any]) -> float:
        """Assess overall portfolio liquidity"""
        
        if not hasattr(portfolio_state, 'positions') or not portfolio_state.positions:
            return 1.0
        
        total_value = 0
        liquidity_weighted_value = 0
        
        for symbol, position in portfolio_state.positions.items():
            position_value = abs(position.quantity * position.market_price)
            total_value += position_value
            
            # Simple liquidity score based on market data
            symbol_data = market_data.get(symbol, {})
            volume = symbol_data.get('volume', 1000000)  # Default volume
            
            # Higher volume = better liquidity (simplified model)
            liquidity_score = min(1.0, volume / 1000000)  # Normalize to 1M volume
            liquidity_weighted_value += position_value * liquidity_score
        
        return liquidity_weighted_value / total_value if total_value > 0 else 1.0
    
    def _update_performance_tracking(self, portfolio_state: Any):
        """Update performance tracking data"""
        
        current_value = portfolio_state.total_value
        self.equity_curve_history.append(current_value)
        
        # Update peak value
        self.peak_value = max(self.peak_value, current_value)
        
        # Calculate return
        if len(self.equity_curve_history) > 1:
            prev_value = self.equity_curve_history[-2]
            if prev_value > 0:
                portfolio_return = (current_value - prev_value) / prev_value
                self.return_history.append(portfolio_return)
                
                # Keep reasonable history size
                if len(self.return_history) > 1000:
                    self.return_history = self.return_history[-1000:]
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        
        if not self.risk_metrics_history:
            return {"error": "No risk metrics available"}
        
        latest_metrics = self.risk_metrics_history[-1]
        
        # Alert statistics
        alert_counts = {}
        for alert in self.alerts_history:
            risk_type = alert.risk_type.value
            alert_counts[risk_type] = alert_counts.get(risk_type, 0) + 1
        
        # Recent alerts (last 30)
        recent_alerts = self.alerts_history[-30:] if len(self.alerts_history) > 30 else self.alerts_history
        
        return {
            "current_metrics": {
                "max_drawdown": latest_metrics.max_drawdown,
                "current_drawdown": latest_metrics.current_drawdown,
                "portfolio_volatility": latest_metrics.portfolio_volatility,
                "sharpe_ratio": latest_metrics.sharpe_ratio,
                "var_95": latest_metrics.portfolio_var,
                "max_position_weight": latest_metrics.max_position_weight,
                "gross_leverage": latest_metrics.gross_leverage,
                "liquidity_score": latest_metrics.liquidity_score
            },
            "alert_statistics": {
                "total_alerts": len(self.alerts_history),
                "alerts_by_type": alert_counts,
                "recent_alerts": len(recent_alerts)
            },
            "rule_performance": [
                {
                    "rule_name": rule.name,
                    "enabled": rule.enabled,
                    "alerts_generated": rule.alerts_generated,
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
                }
                for rule in self.rules
            ]
        }
    
    def generate_risk_report(self) -> str:
        """Generate detailed risk report"""
        
        if not self.risk_metrics_history:
            return "No risk data available for report generation."
        
        latest_metrics = self.risk_metrics_history[-1]
        summary = self.get_risk_summary()
        
        report = []
        report.append("=" * 60)
        report.append("RISK MANAGEMENT REPORT")
        report.append("=" * 60)
        
        # Current risk metrics
        report.append("\n📊 CURRENT RISK METRICS")
        report.append("-" * 30)
        report.append(f"Maximum Drawdown: {latest_metrics.max_drawdown:>15.2%}")
        report.append(f"Current Drawdown: {latest_metrics.current_drawdown:>15.2%}")
        report.append(f"Portfolio Volatility: {latest_metrics.portfolio_volatility:>13.2%}")
        report.append(f"Value at Risk (95%): {latest_metrics.portfolio_var:>12.2%}")
        report.append(f"Conditional VaR (95%): {latest_metrics.portfolio_cvar:>9.2%}")
        
        # Position and leverage risks
        report.append("\n📍 POSITION & LEVERAGE RISKS")
        report.append("-" * 35)
        report.append(f"Max Position Weight: {latest_metrics.max_position_weight:>14.2%}")
        report.append(f"Gross Leverage: {latest_metrics.gross_leverage:>19.2f}x")
        report.append(f"Net Leverage: {latest_metrics.net_leverage:>21.2f}x")
        report.append(f"Liquidity Score: {latest_metrics.liquidity_score:>18.2f}")
        
        # Risk-adjusted performance
        report.append("\n⚖️  RISK-ADJUSTED PERFORMANCE")
        report.append("-" * 35)
        report.append(f"Sharpe Ratio: {latest_metrics.sharpe_ratio:>23.3f}")
        report.append(f"Sortino Ratio: {latest_metrics.sortino_ratio:>22.3f}")
        report.append(f"Calmar Ratio: {latest_metrics.calmar_ratio:>23.3f}")
        
        # Alert summary
        report.append("\n🚨 ALERT SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Alerts Generated: {summary['alert_statistics']['total_alerts']:>11d}")
        
        for risk_type, count in summary['alert_statistics']['alerts_by_type'].items():
            formatted_type = risk_type.replace('_', ' ').title()
            report.append(f"{formatted_type}: {count:>25d}")
        
        # Recent critical alerts
        critical_alerts = [a for a in self.alerts_history[-10:] 
                          if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        
        if critical_alerts:
            report.append("\n⚠️  RECENT CRITICAL ALERTS")
            report.append("-" * 30)
            for alert in critical_alerts:
                report.append(f"{alert.timestamp.strftime('%Y-%m-%d %H:%M')}: {alert.message}")
        
        # Rule performance
        report.append("\n📋 RULE PERFORMANCE")
        report.append("-" * 25)
        for rule_info in summary['rule_performance']:
            status = "✓" if rule_info['enabled'] else "✗"
            report.append(f"{status} {rule_info['rule_name']}: {rule_info['alerts_generated']} alerts")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def add_rule(self, rule: RiskRule):
        """Add a new risk rule"""
        self.rules.append(rule)
        logger.info(f"Added risk rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a risk rule by name"""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                logger.info(f"Removed risk rule: {rule_name}")
                return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a risk rule"""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"Enabled risk rule: {rule_name}")
                return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a risk rule"""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"Disabled risk rule: {rule_name}")
                return True
        return False

# ============================================
# Risk Monitoring Integration
# ============================================

class RiskMonitoringEngine:
    """
    Engine that integrates risk management with backtesting
    """
    
    def __init__(self, risk_manager: RiskManager, 
                 alert_callback: Optional[Callable[[List[RiskAlert]], None]] = None):
        """
        Initialize risk monitoring engine
        
        Args:
            risk_manager: RiskManager instance
            alert_callback: Optional callback function for handling alerts
        """
        
        self.risk_manager = risk_manager
        self.alert_callback = alert_callback
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start risk monitoring"""
        self.monitoring_active = True
        logger.info("Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop risk monitoring"""
        self.monitoring_active = False
        logger.info("Risk monitoring stopped")
    
    def process_portfolio_update(self, portfolio_state: Any, market_data: Dict[str, Any]):
        """
        Process portfolio update and evaluate risks
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Current market data
        """
        
        if not self.monitoring_active:
            return
        
        try:
            alerts, risk_metrics = self.risk_manager.evaluate_risk(portfolio_state, market_data)
            
            # Handle alerts
            if alerts and self.alert_callback:
                self.alert_callback(alerts)
            
            # Log critical alerts
            critical_alerts = [a for a in alerts if a.risk_level == RiskLevel.CRITICAL]
            if critical_alerts:
                logger.critical(f"CRITICAL RISK ALERTS: {len(critical_alerts)} alerts triggered")
                for alert in critical_alerts:
                    logger.critical(f"  - {alert.message}")
        
        except Exception as e:
            logger.error(f"Error in risk monitoring: {e}")

# ============================================
# Utility Functions
# ============================================

def create_default_risk_manager() -> RiskManager:
    """Create risk manager with sensible default settings"""
    
    rules = [
        PositionSizeRule(max_position_pct=0.10, max_sector_pct=0.30),
        DrawdownRule(max_drawdown_pct=0.15, warning_drawdown_pct=0.08),
        LeverageRule(max_gross_leverage=1.5, max_net_leverage=1.0),
        VolatilityRule(max_portfolio_vol=0.25),
        VaRRule(max_var_pct=0.05, confidence_level=0.95)
    ]
    
    return RiskManager(rules)

def create_conservative_risk_manager() -> RiskManager:
    """Create conservative risk manager for low-risk strategies"""
    
    rules = [
        PositionSizeRule(max_position_pct=0.05, max_sector_pct=0.20),
        DrawdownRule(max_drawdown_pct=0.08, warning_drawdown_pct=0.05),
        LeverageRule(max_gross_leverage=1.2, max_net_leverage=1.0),
        VolatilityRule(max_portfolio_vol=0.15),
        VaRRule(max_var_pct=0.03, confidence_level=0.95)
    ]
    
    return RiskManager(rules)

def create_aggressive_risk_manager() -> RiskManager:
    """Create aggressive risk manager for high-risk strategies"""
    
    rules = [
        PositionSizeRule(max_position_pct=0.20, max_sector_pct=0.50),
        DrawdownRule(max_drawdown_pct=0.25, warning_drawdown_pct=0.15),
        LeverageRule(max_gross_leverage=3.0, max_net_leverage=2.0),
        VolatilityRule(max_portfolio_vol=0.40),
        VaRRule(max_var_pct=0.10, confidence_level=0.95)
    ]
    
    return RiskManager(rules)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Risk Management System")
    
    # Create mock portfolio state for testing
    class MockPosition:
        def __init__(self, quantity, market_price):
            self.quantity = quantity
            self.market_price = market_price
    
    class MockPortfolioState:
        def __init__(self, total_value, positions, drawdown=0.0):
            self.total_value = total_value
            self.positions = positions
            self.drawdown = drawdown
            self.timestamp = pd.Timestamp.now()
    
    # Create test scenario
    print("\n1. Testing Default Risk Manager")
    
    risk_manager = create_default_risk_manager()
    
    # Test normal scenario
    positions = {
        'AAPL': MockPosition(100, 150.0),  # $15,000 position
        'MSFT': MockPosition(200, 250.0),  # $50,000 position
        'GOOGL': MockPosition(50, 2800.0)  # $140,000 position (large)
    }
    
    portfolio_state = MockPortfolioState(
        total_value=1000000,  # $1M portfolio
        positions=positions,
        drawdown=0.05  # 5% drawdown
    )
    
    market_data = {
        'AAPL': {'volume': 50000000, 'close': 150.0},
        'MSFT': {'volume': 30000000, 'close': 250.0},
        'GOOGL': {'volume': 1500000, 'close': 2800.0}
    }
    
    alerts, metrics = risk_manager.evaluate_risk(portfolio_state, market_data)
    
    print(f"Generated {len(alerts)} alerts")
    for alert in alerts:
        print(f"  {alert.risk_level.value.upper()}: {alert.message}")
    
    print(f"\nRisk Metrics:")
    print(f"  Max Position Weight: {metrics.max_position_weight:.2%}")
    print(f"  Gross Leverage: {metrics.gross_leverage:.2f}x")
    print(f"  Current Drawdown: {metrics.current_drawdown:.2%}")
    print(f"  Liquidity Score: {metrics.liquidity_score:.2f}")
    
    # Test high-risk scenario
    print("\n2. Testing High-Risk Scenario")
    
    # Create scenario with large positions and high drawdown
    risky_positions = {
        'AAPL': MockPosition(1000, 150.0),   # $150,000 (15% of portfolio)
        'MSFT': MockPosition(2000, 250.0),   # $500,000 (50% of portfolio!)
        'TSLA': MockPosition(500, 800.0)     # $400,000 (40% of portfolio!)
    }
    
    risky_portfolio = MockPortfolioState(
        total_value=1000000,
        positions=risky_positions,
        drawdown=0.18  # 18% drawdown (above warning level)
    )
    
    alerts, metrics = risk_manager.evaluate_risk(risky_portfolio, market_data)
    
    print(f"Generated {len(alerts)} alerts in risky scenario")
    for alert in alerts:
        level_indicator = "🚨" if alert.risk_level == RiskLevel.CRITICAL else "⚠️"
        print(f"  {level_indicator} {alert.risk_level.value.upper()}: {alert.message}")
    
    # Test different risk manager configurations
    print("\n3. Testing Different Risk Manager Configurations")
    
    configs = {
        "Conservative": create_conservative_risk_manager(),
        "Default": create_default_risk_manager(),
        "Aggressive": create_aggressive_risk_manager()
    }
    
    for config_name, rm in configs.items():
        alerts, _ = rm.evaluate_risk(risky_portfolio, market_data)
        print(f"{config_name} Risk Manager: {len(alerts)} alerts")
    
    # Test risk monitoring engine
    print("\n4. Testing Risk Monitoring Engine")
    
    def alert_handler(alerts):
        print(f"🚨 ALERT HANDLER: Received {len(alerts)} alerts")
        for alert in alerts:
            if alert.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                print(f"    {alert.risk_type.value}: {alert.message}")
    
    monitoring_engine = RiskMonitoringEngine(risk_manager, alert_handler)
    monitoring_engine.start_monitoring()
    
    # Simulate portfolio updates
    for i in range(3):
        # Gradually increase drawdown
        test_portfolio = MockPortfolioState(
            total_value=1000000 * (1 - 0.05 * (i + 1)),  # Decreasing value
            positions=risky_positions,
            drawdown=0.05 * (i + 2)  # Increasing drawdown
        )
        
        print(f"\nUpdate {i + 1}: Portfolio Value = ${test_portfolio.total_value:,.0f}, "
              f"Drawdown = {test_portfolio.drawdown:.1%}")
        
        monitoring_engine.process_portfolio_update(test_portfolio, market_data)
    
    monitoring_engine.stop_monitoring()
    
    # Test comprehensive risk report
    print("\n5. Testing Risk Report Generation")
    
    # Add some history to risk manager
    for i in range(10):
        test_dd = 0.02 + i * 0.01  # Gradually increasing drawdown
        test_portfolio = MockPortfolioState(
            total_value=1000000 * (1 - test_dd),
            positions=positions,
            drawdown=test_dd
        )
        
        risk_manager.evaluate_risk(test_portfolio, market_data)
    
    # Generate and display report
    risk_report = risk_manager.generate_risk_report()
    print(risk_report)
    
    # Test risk summary
    print("\n6. Testing Risk Summary")
    
    summary = risk_manager.get_risk_summary()
    print("Risk Summary:")
    print(f"  Total Alerts: {summary['alert_statistics']['total_alerts']}")
    print(f"  Current Max Drawdown: {summary['current_metrics']['max_drawdown']:.2%}")
    print(f"  Portfolio Volatility: {summary['current_metrics']['portfolio_volatility']:.2%}")
    
    print("Rule Performance:")
    for rule in summary['rule_performance']:
        status = "✓" if rule['enabled'] else "✗"
        print(f"  {status} {rule['rule_name']}: {rule['alerts_generated']} alerts")
    
    # Test rule management
    print("\n7. Testing Rule Management")
    
    print(f"Initial rules: {len(risk_manager.rules)}")
    
    # Add custom rule
    class CustomRule(RiskRule):
        def evaluate(self, portfolio_state, market_data):
            if portfolio_state.total_value < 900000:  # Below $900K
                return [RiskAlert(
                    timestamp=portfolio_state.timestamp,
                    risk_type=RiskType.MARKET_RISK,
                    risk_level=RiskLevel.MEDIUM,
                    symbol=None,
                    message="Portfolio value below $900K threshold",
                    metric_value=portfolio_state.total_value,
                    threshold=900000,
                    suggested_action="Review strategy performance"
                )]
            return []
    
    custom_rule = CustomRule("Custom Value Rule")
    risk_manager.add_rule(custom_rule)
    print(f"After adding custom rule: {len(risk_manager.rules)}")
    
    # Test custom rule
    low_value_portfolio = MockPortfolioState(850000, positions)
    alerts, _ = risk_manager.evaluate_risk(low_value_portfolio, market_data)
    
    custom_alerts = [a for a in alerts if "Portfolio value below" in a.message]
    print(f"Custom rule triggered: {len(custom_alerts) > 0}")
    
    print("\nRisk management system testing completed successfully!")
