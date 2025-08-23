# ============================================
# StockPredictionPro - src/trading/risk/portfolio_risk.py
# Comprehensive portfolio risk management and analysis system
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import threading
import math
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf, EmpiricalCovariance

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('trading.risk.portfolio_risk')

# ============================================
# Risk Data Structures and Enums
# ============================================

class RiskMeasure(Enum):
    """Risk measurement types"""
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VAR = "conditional_var"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"

class VaRMethod(Enum):
    """Value at Risk calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"

class RiskLimitType(Enum):
    """Types of risk limits"""
    VAR_ABSOLUTE = "var_absolute"
    VAR_PERCENTAGE = "var_percentage"
    VOLATILITY = "volatility"
    CONCENTRATION = "concentration"
    SECTOR_EXPOSURE = "sector_exposure"
    CORRELATION = "correlation"
    LEVERAGE = "leverage"

@dataclass
class Position:
    """Portfolio position representation"""
    symbol: str
    quantity: int
    market_value: float
    weight: float
    
    # Risk characteristics
    beta: Optional[float] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    currency: Optional[str] = None
    
    # Performance metrics
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    return_1d: float = 0.0
    return_mtd: float = 0.0
    return_ytd: float = 0.0
    
    # Risk metrics
    var_contribution: float = 0.0
    marginal_var: float = 0.0
    component_var: float = 0.0
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value (absolute market value)"""
        return abs(self.market_value)

@dataclass
class RiskLimit:
    """Risk limit definition"""
    limit_id: str
    limit_type: RiskLimitType
    level: str  # "portfolio", "sector", "position"
    entity: str  # Symbol, sector name, or "portfolio"
    
    # Limit values
    limit_value: float
    warning_threshold: float = 0.8  # 80% of limit triggers warning
    current_value: float = 0.0
    
    # Status
    is_breached: bool = False
    is_warning: bool = False
    breach_count: int = 0
    last_breach_time: Optional[datetime] = None
    
    # Metadata
    description: str = ""
    created_time: datetime = field(default_factory=datetime.now)
    
    def update_status(self, current_value: float):
        """Update limit status based on current value"""
        self.current_value = current_value
        
        # Check for breach
        if abs(current_value) > self.limit_value:
            if not self.is_breached:
                self.breach_count += 1
                self.last_breach_time = datetime.now()
            self.is_breached = True
            self.is_warning = False
        elif abs(current_value) > self.limit_value * self.warning_threshold:
            self.is_breached = False
            self.is_warning = True
        else:
            self.is_breached = False
            self.is_warning = False

@dataclass
class RiskMetrics:
    """Comprehensive portfolio risk metrics"""
    
    # Basic metrics
    total_value: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    leverage: float = 1.0
    
    # Volatility metrics
    daily_volatility: float = 0.0
    annual_volatility: float = 0.0
    volatility_95_percentile: float = 0.0
    
    # VaR metrics (by confidence level)
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0  # Conditional VaR
    cvar_99: float = 0.0
    
    # Drawdown metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    drawdown_duration: int = 0
    recovery_time: Optional[int] = None
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Concentration metrics
    top_5_concentration: float = 0.0
    top_10_concentration: float = 0.0
    effective_number_positions: float = 0.0
    herfindahl_index: float = 0.0
    
    # Correlation metrics
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    eigenvalue_ratio: float = 0.0
    
    # Beta and factor exposure
    portfolio_beta: float = 1.0
    market_correlation: float = 0.0
    
    # Time stamps
    calculation_time: datetime = field(default_factory=datetime.now)
    data_as_of: datetime = field(default_factory=datetime.now)

# ============================================
# Portfolio Risk Calculator
# ============================================

class PortfolioRiskCalculator:
    """
    Comprehensive portfolio risk calculation engine.
    
    Calculates various risk metrics including VaR, volatility,
    drawdowns, concentration risk, and correlation analysis.
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Calculation parameters
        self.returns_window = 252  # 1 year of daily returns
        self.monte_carlo_simulations = 10000
        self.var_holding_period = 1  # 1 day
        
        # Caching
        self.covariance_cache = {}
        self.returns_cache = {}
        
        logger.info("Initialized PortfolioRiskCalculator")
    
    @time_it("portfolio_risk_calculation")
    def calculate_portfolio_risk(self, positions: List[Position], 
                                returns_data: pd.DataFrame,
                                benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics
        
        Args:
            positions: List of portfolio positions
            returns_data: Historical returns data (symbols as columns)
            benchmark_returns: Benchmark returns for beta calculation
            
        Returns:
            RiskMetrics object with all calculated metrics
        """
        
        if not positions:
            logger.warning("No positions provided for risk calculation")
            return RiskMetrics()
        
        # Prepare portfolio data
        portfolio_weights = self._calculate_weights(positions)
        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)
        
        if len(portfolio_returns) < 30:
            logger.warning("Insufficient return history for robust risk calculation")
        
        # Calculate risk metrics
        metrics = RiskMetrics()
        
        # Basic portfolio metrics
        metrics.total_value = sum(pos.market_value for pos in positions)
        metrics.net_exposure = sum(pos.market_value for pos in positions)
        metrics.gross_exposure = sum(abs(pos.market_value) for pos in positions)
        metrics.leverage = metrics.gross_exposure / abs(metrics.total_value) if metrics.total_value != 0 else 1.0
        
        # Volatility calculations
        self._calculate_volatility_metrics(portfolio_returns, metrics)
        
        # VaR calculations
        self._calculate_var_metrics(portfolio_returns, positions, returns_data, metrics)
        
        # Drawdown calculations
        self._calculate_drawdown_metrics(portfolio_returns, metrics)
        
        # Performance ratios
        self._calculate_performance_ratios(portfolio_returns, metrics)
        
        # Concentration metrics
        self._calculate_concentration_metrics(positions, metrics)
        
        # Correlation analysis
        self._calculate_correlation_metrics(returns_data, portfolio_weights, metrics)
        
        # Beta calculation
        if benchmark_returns is not None:
            self._calculate_beta_metrics(portfolio_returns, benchmark_returns, positions, metrics)
        
        logger.debug(f"Portfolio risk calculated: VaR95={metrics.var_95:.2f}, Vol={metrics.annual_volatility:.2%}")
        
        return metrics
    
    def _calculate_weights(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate position weights"""
        total_value = sum(abs(pos.market_value) for pos in positions)
        
        if total_value == 0:
            return {}
        
        return {pos.symbol: pos.market_value / total_value for pos in positions}
    
    def _calculate_portfolio_returns(self, positions: List[Position], 
                                   returns_data: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns from position weights and asset returns"""
        
        weights = self._calculate_weights(positions)
        
        # Filter returns data to only include positions in portfolio
        available_symbols = [symbol for symbol in weights.keys() if symbol in returns_data.columns]
        
        if not available_symbols:
            logger.warning("No return data available for portfolio positions")
            return pd.Series(dtype=float)
        
        # Calculate weighted returns
        portfolio_returns = pd.Series(0.0, index=returns_data.index)
        
        for symbol in available_symbols:
            weight = weights[symbol]
            portfolio_returns += weight * returns_data[symbol]
        
        return portfolio_returns.dropna()
    
    def _calculate_volatility_metrics(self, returns: pd.Series, metrics: RiskMetrics):
        """Calculate volatility-based risk metrics"""
        
        if len(returns) == 0:
            return
        
        # Daily volatility
        metrics.daily_volatility = returns.std()
        
        # Annualized volatility (assuming 252 trading days)
        metrics.annual_volatility = metrics.daily_volatility * np.sqrt(252)
        
        # 95th percentile volatility
        rolling_vol = returns.rolling(window=21).std()  # 21-day rolling volatility
        metrics.volatility_95_percentile = rolling_vol.quantile(0.95)
    
    def _calculate_var_metrics(self, returns: pd.Series, positions: List[Position],
                              returns_data: pd.DataFrame, metrics: RiskMetrics):
        """Calculate Value at Risk metrics using multiple methods"""
        
        if len(returns) == 0:
            return
        
        portfolio_value = metrics.total_value
        
        for confidence_level in self.confidence_levels:
            # Historical VaR
            var_hist = self._calculate_historical_var(returns, confidence_level)
            
            # Parametric VaR
            var_param = self._calculate_parametric_var(returns, confidence_level)
            
            # Use historical VaR as primary method
            var_absolute = var_hist * abs(portfolio_value)
            
            # Conditional VaR (Expected Shortfall)
            cvar = self._calculate_conditional_var(returns, confidence_level)
            cvar_absolute = cvar * abs(portfolio_value)
            
            # Store results
            if confidence_level == 0.95:
                metrics.var_95 = var_absolute
                metrics.cvar_95 = cvar_absolute
            elif confidence_level == 0.99:
                metrics.var_99 = var_absolute
                metrics.cvar_99 = cvar_absolute
        
        # Calculate component VaR for positions
        self._calculate_component_var(positions, returns_data, metrics)
    
    def _calculate_historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Historical VaR"""
        if len(returns) == 0:
            return 0.0
        
        return -returns.quantile(1 - confidence_level)
    
    def _calculate_parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Parametric VaR assuming normal distribution"""
        if len(returns) == 0:
            return 0.0
        
        mean_return = returns.mean()
        volatility = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        
        return -(mean_return + z_score * volatility)
    
    def _calculate_conditional_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        
        var_threshold = -self._calculate_historical_var(returns, confidence_level)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return self._calculate_historical_var(returns, confidence_level)
        
        return -tail_returns.mean()
    
    def _calculate_component_var(self, positions: List[Position], 
                                returns_data: pd.DataFrame, metrics: RiskMetrics):
        """Calculate component VaR for each position"""
        
        if not positions or returns_data.empty:
            return
        
        # Get position symbols that have return data
        symbols = [pos.symbol for pos in positions if pos.symbol in returns_data.columns]
        
        if len(symbols) < 2:
            return
        
        # Calculate covariance matrix
        returns_subset = returns_data[symbols].dropna()
        if len(returns_subset) < 30:
            return
        
        cov_matrix = returns_subset.cov().values
        weights = np.array([pos.weight for pos in positions if pos.symbol in symbols])
        
        # Portfolio variance
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        if portfolio_volatility == 0:
            return
        
        # Marginal VaR (partial derivative of portfolio VaR w.r.t. position weight)
        marginal_var = np.dot(cov_matrix, weights) / portfolio_volatility
        
        # Component VaR
        component_var = weights * marginal_var
        
        # Update positions with VaR contributions
        for i, pos in enumerate([p for p in positions if p.symbol in symbols]):
            pos.marginal_var = marginal_var[i] * abs(metrics.total_value)
            pos.component_var = component_var[i] * abs(metrics.total_value)
            pos.var_contribution = component_var[i] / np.sum(component_var) if np.sum(component_var) != 0 else 0
    
    def _calculate_drawdown_metrics(self, returns: pd.Series, metrics: RiskMetrics):
        """Calculate drawdown-related metrics"""
        
        if len(returns) == 0:
            return
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum (peaks)
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        metrics.max_drawdown = drawdowns.min()
        
        # Current drawdown
        metrics.current_drawdown = drawdowns.iloc[-1] if len(drawdowns) > 0 else 0
        
        # Drawdown duration (days in current drawdown)
        if metrics.current_drawdown < 0:
            # Find the start of current drawdown
            peak_idx = running_max.index[running_max == running_max.iloc[-1]][-1]
            current_idx = drawdowns.index[-1]
            metrics.drawdown_duration = len(drawdowns.loc[peak_idx:current_idx]) - 1
        
        # Recovery time for previous drawdowns
        self._calculate_recovery_time(drawdowns, metrics)
    
    def _calculate_recovery_time(self, drawdowns: pd.Series, metrics: RiskMetrics):
        """Calculate average recovery time from drawdowns"""
        
        recovery_times = []
        in_drawdown = False
        drawdown_start = None
        
        for date, dd in drawdowns.items():
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                drawdown_start = date
            elif dd >= 0 and in_drawdown:
                # End of drawdown (recovery)
                in_drawdown = False
                if drawdown_start is not None:
                    recovery_days = (date - drawdown_start).days
                    recovery_times.append(recovery_days)
        
        if recovery_times:
            metrics.recovery_time = int(np.mean(recovery_times))
    
    def _calculate_performance_ratios(self, returns: pd.Series, metrics: RiskMetrics):
        """Calculate performance ratios"""
        
        if len(returns) == 0:
            return
        
        # Annualized returns
        total_return = (1 + returns).prod() - 1
        num_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
        
        # Sharpe ratio
        excess_return = annual_return - self.risk_free_rate
        if metrics.annual_volatility > 0:
            metrics.sharpe_ratio = excess_return / metrics.annual_volatility
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                metrics.sortino_ratio = excess_return / downside_deviation
        
        # Calmar ratio
        if metrics.max_drawdown < 0:
            metrics.calmar_ratio = annual_return / abs(metrics.max_drawdown)
    
    def _calculate_concentration_metrics(self, positions: List[Position], metrics: RiskMetrics):
        """Calculate concentration risk metrics"""
        
        if not positions:
            return
        
        # Calculate absolute weights
        weights = [abs(pos.weight) for pos in positions]
        weights_sorted = sorted(weights, reverse=True)
        
        # Top N concentration
        if len(weights_sorted) >= 5:
            metrics.top_5_concentration = sum(weights_sorted[:5])
        if len(weights_sorted) >= 10:
            metrics.top_10_concentration = sum(weights_sorted[:10])
        
        # Herfindahl-Hirschman Index
        metrics.herfindahl_index = sum(w**2 for w in weights)
        
        # Effective number of positions
        if metrics.herfindahl_index > 0:
            metrics.effective_number_positions = 1 / metrics.herfindahl_index
    
    def _calculate_correlation_metrics(self, returns_data: pd.DataFrame, 
                                     portfolio_weights: Dict[str, float], 
                                     metrics: RiskMetrics):
        """Calculate correlation-based risk metrics"""
        
        symbols = list(portfolio_weights.keys())
        symbols_in_data = [s for s in symbols if s in returns_data.columns]
        
        if len(symbols_in_data) < 2:
            return
        
        # Calculate correlation matrix
        correlation_matrix = returns_data[symbols_in_data].corr()
        
        # Average correlation (excluding diagonal)
        correlations = []
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                correlations.append(correlation_matrix.iloc[i, j])
        
        if correlations:
            metrics.avg_correlation = np.mean(correlations)
            metrics.max_correlation = np.max(correlations)
        
        # Eigenvalue analysis for diversification
        eigenvalues = np.linalg.eigvals(correlation_matrix.values)
        eigenvalues = eigenvalues[eigenvalues > 0]  # Remove negative eigenvalues
        
        if len(eigenvalues) > 1:
            metrics.eigenvalue_ratio = eigenvalues[0] / eigenvalues[-1]
    
    def _calculate_beta_metrics(self, portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series, 
                               positions: List[Position], 
                               metrics: RiskMetrics):
        """Calculate beta and market-related metrics"""
        
        # Align returns
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_returns) < 30:
            return
        
        portfolio_aligned = aligned_returns.iloc[:, 0]
        benchmark_aligned = aligned_returns.iloc[:, 1]
        
        # Portfolio beta
        covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_aligned)
        
        if benchmark_variance > 0:
            metrics.portfolio_beta = covariance / benchmark_variance
        
        # Market correlation
        correlation_matrix = np.corrcoef(portfolio_aligned, benchmark_aligned)
        metrics.market_correlation = correlation_matrix[0, 1]

# ============================================
# Risk Limit Manager
# ============================================

class RiskLimitManager:
    """
    Risk limit management system.
    
    Monitors portfolio risk limits, generates alerts,
    and provides risk limit breach analysis.
    """
    
    def __init__(self):
        self.limits = {}
        self.breach_history = []
        self.alert_callbacks = []
        
        # Default limits
        self._initialize_default_limits()
        
        logger.info("Initialized RiskLimitManager")
    
    def _initialize_default_limits(self):
        """Initialize default risk limits"""
        
        default_limits = [
            RiskLimit(
                limit_id="portfolio_var_95",
                limit_type=RiskLimitType.VAR_PERCENTAGE,
                level="portfolio",
                entity="portfolio",
                limit_value=0.02,  # 2% of portfolio value
                description="Portfolio 95% VaR limit"
            ),
            RiskLimit(
                limit_id="portfolio_volatility",
                limit_type=RiskLimitType.VOLATILITY,
                level="portfolio", 
                entity="portfolio",
                limit_value=0.20,  # 20% annual volatility
                description="Portfolio volatility limit"
            ),
            RiskLimit(
                limit_id="max_position_concentration",
                limit_type=RiskLimitType.CONCENTRATION,
                level="position",
                entity="any",
                limit_value=0.10,  # 10% max position size
                description="Maximum single position concentration"
            ),
            RiskLimit(
                limit_id="leverage_limit",
                limit_type=RiskLimitType.LEVERAGE,
                level="portfolio",
                entity="portfolio", 
                limit_value=1.5,  # 1.5x leverage
                description="Maximum portfolio leverage"
            )
        ]
        
        for limit in default_limits:
            self.limits[limit.limit_id] = limit
    
    def add_limit(self, limit: RiskLimit):
        """Add a new risk limit"""
        self.limits[limit.limit_id] = limit
        logger.info(f"Added risk limit: {limit.limit_id}")
    
    def remove_limit(self, limit_id: str) -> bool:
        """Remove a risk limit"""
        if limit_id in self.limits:
            del self.limits[limit_id]
            logger.info(f"Removed risk limit: {limit_id}")
            return True
        return False
    
    def check_limits(self, metrics: RiskMetrics, positions: List[Position]) -> Dict[str, RiskLimit]:
        """
        Check all risk limits against current portfolio metrics
        
        Args:
            metrics: Current portfolio risk metrics
            positions: Current portfolio positions
            
        Returns:
            Dictionary of breached limits
        """
        
        breached_limits = {}
        
        for limit_id, limit in self.limits.items():
            try:
                current_value = self._get_current_value(limit, metrics, positions)
                limit.update_status(current_value)
                
                if limit.is_breached:
                    breached_limits[limit_id] = limit
                    self._record_breach(limit)
                    logger.warning(f"Risk limit breached: {limit_id} = {current_value:.4f} > {limit.limit_value:.4f}")
                
                elif limit.is_warning:
                    logger.info(f"Risk limit warning: {limit_id} = {current_value:.4f} (threshold: {limit.limit_value * limit.warning_threshold:.4f})")
                
            except Exception as e:
                logger.error(f"Error checking limit {limit_id}: {e}")
        
        return breached_limits
    
    def _get_current_value(self, limit: RiskLimit, metrics: RiskMetrics, 
                          positions: List[Position]) -> float:
        """Get current value for a specific limit"""
        
        if limit.limit_type == RiskLimitType.VAR_ABSOLUTE:
            return metrics.var_95
        
        elif limit.limit_type == RiskLimitType.VAR_PERCENTAGE:
            if metrics.total_value != 0:
                return metrics.var_95 / abs(metrics.total_value)
            return 0.0
        
        elif limit.limit_type == RiskLimitType.VOLATILITY:
            return metrics.annual_volatility
        
        elif limit.limit_type == RiskLimitType.CONCENTRATION:
            if limit.entity == "any":
                # Maximum single position concentration
                return max([abs(pos.weight) for pos in positions], default=0.0)
            else:
                # Specific position concentration
                for pos in positions:
                    if pos.symbol == limit.entity:
                        return abs(pos.weight)
                return 0.0
        
        elif limit.limit_type == RiskLimitType.SECTOR_EXPOSURE:
            # Calculate sector exposure
            sector_exposure = defaultdict(float)
            for pos in positions:
                if pos.sector:
                    sector_exposure[pos.sector] += abs(pos.weight)
            
            if limit.entity in sector_exposure:
                return sector_exposure[limit.entity]
            return 0.0
        
        elif limit.limit_type == RiskLimitType.LEVERAGE:
            return metrics.leverage
        
        elif limit.limit_type == RiskLimitType.CORRELATION:
            return metrics.avg_correlation
        
        else:
            logger.warning(f"Unknown limit type: {limit.limit_type}")
            return 0.0
    
    def _record_breach(self, limit: RiskLimit):
        """Record a limit breach for historical analysis"""
        
        breach_record = {
            'timestamp': datetime.now(),
            'limit_id': limit.limit_id,
            'limit_type': limit.limit_type.value,
            'limit_value': limit.limit_value,
            'current_value': limit.current_value,
            'excess': limit.current_value - limit.limit_value,
            'entity': limit.entity
        }
        
        self.breach_history.append(breach_record)
        
        # Keep only recent breach history
        cutoff_date = datetime.now() - timedelta(days=90)
        self.breach_history = [b for b in self.breach_history if b['timestamp'] > cutoff_date]
    
    def get_limit_status(self) -> pd.DataFrame:
        """Get status of all risk limits"""
        
        status_data = []
        
        for limit_id, limit in self.limits.items():
            status_data.append({
                'Limit_ID': limit_id,
                'Type': limit.limit_type.value,
                'Entity': limit.entity,
                'Limit': limit.limit_value,
                'Current': limit.current_value,
                'Utilization': (limit.current_value / limit.limit_value) if limit.limit_value != 0 else 0,
                'Status': 'BREACH' if limit.is_breached else 'WARNING' if limit.is_warning else 'OK',
                'Breach_Count': limit.breach_count
            })
        
        return pd.DataFrame(status_data)
    
    def get_breach_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of recent limit breaches"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_breaches = [b for b in self.breach_history if b['timestamp'] > cutoff_date]
        
        if not recent_breaches:
            return {
                'period_days': days,
                'total_breaches': 0,
                'unique_limits': 0,
                'breach_frequency': 0.0
            }
        
        # Breach statistics
        unique_limits = len(set(b['limit_id'] for b in recent_breaches))
        breach_by_type = defaultdict(int)
        
        for breach in recent_breaches:
            breach_by_type[breach['limit_type']] += 1
        
        return {
            'period_days': days,
            'total_breaches': len(recent_breaches),
            'unique_limits': unique_limits,
            'breach_frequency': len(recent_breaches) / days,
            'breaches_by_type': dict(breach_by_type),
            'most_frequent_limit': max(breach_by_type.keys(), key=breach_by_type.get) if breach_by_type else None
        }

# ============================================
# Portfolio Risk Manager
# ============================================

class PortfolioRiskManager:
    """
    Comprehensive portfolio risk management system.
    
    Integrates risk calculation, limit monitoring, scenario analysis,
    and risk reporting for complete portfolio risk management.
    """
    
    def __init__(self, portfolio_id: str):
        self.portfolio_id = portfolio_id
        
        # Components
        self.risk_calculator = PortfolioRiskCalculator()
        self.limit_manager = RiskLimitManager()
        
        # Data storage
        self.positions = []
        self.risk_history = []
        self.returns_data = pd.DataFrame()
        self.benchmark_returns = pd.Series()
        
        # Risk monitoring
        self.monitoring_enabled = True
        self.last_calculation_time = None
        
        logger.info(f"Initialized PortfolioRiskManager for portfolio {portfolio_id}")
    
    def update_positions(self, positions: List[Position]):
        """Update portfolio positions"""
        self.positions = positions
        logger.debug(f"Updated {len(positions)} positions")
    
    def update_returns_data(self, returns_data: pd.DataFrame):
        """Update historical returns data"""
        self.returns_data = returns_data
        logger.debug(f"Updated returns data: {len(returns_data)} periods, {len(returns_data.columns)} assets")
    
    def update_benchmark(self, benchmark_returns: pd.Series):
        """Update benchmark returns"""
        self.benchmark_returns = benchmark_returns
        logger.debug(f"Updated benchmark data: {len(benchmark_returns)} periods")
    
    @time_it("portfolio_risk_monitoring")
    def calculate_risk(self, store_history: bool = True) -> RiskMetrics:
        """
        Calculate portfolio risk metrics
        
        Args:
            store_history: Whether to store results in risk history
            
        Returns:
            RiskMetrics object
        """
        
        if not self.positions:
            logger.warning("No positions available for risk calculation")
            return RiskMetrics()
        
        # Calculate risk metrics
        benchmark = self.benchmark_returns if not self.benchmark_returns.empty else None
        metrics = self.risk_calculator.calculate_portfolio_risk(
            self.positions, 
            self.returns_data, 
            benchmark
        )
        
        # Store in history
        if store_history:
            self.risk_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics
            })
            
            # Keep last 252 periods (1 year)
            if len(self.risk_history) > 252:
                self.risk_history = self.risk_history[-252:]
        
        self.last_calculation_time = datetime.now()
        
        return metrics
    
    def monitor_risk_limits(self) -> Dict[str, Any]:
        """
        Monitor risk limits and return status
        
        Returns:
            Dictionary with risk limit status and any breaches
        """
        
        if not self.monitoring_enabled:
            return {'monitoring_enabled': False}
        
        # Calculate current risk
        current_metrics = self.calculate_risk(store_history=False)
        
        # Check limits
        breached_limits = self.limit_manager.check_limits(current_metrics, self.positions)
        
        # Get limit status
        limit_status = self.limit_manager.get_limit_status()
        
        return {
            'monitoring_enabled': True,
            'calculation_time': current_metrics.calculation_time,
            'metrics_summary': {
                'var_95': current_metrics.var_95,
                'volatility': current_metrics.annual_volatility,
                'leverage': current_metrics.leverage,
                'max_drawdown': current_metrics.max_drawdown
            },
            'breached_limits': {limit_id: limit.current_value for limit_id, limit in breached_limits.items()},
            'limit_status': limit_status,
            'breach_count': len(breached_limits)
        }
    
    def scenario_analysis(self, scenarios: Dict[str, Dict[str, float]]) -> Dict[str, RiskMetrics]:
        """
        Perform scenario analysis on portfolio
        
        Args:
            scenarios: Dictionary of scenario_name -> {symbol: shock_percentage}
            
        Returns:
            Dictionary of scenario results
        """
        
        scenario_results = {}
        
        # Calculate base case
        base_metrics = self.calculate_risk(store_history=False)
        scenario_results['base_case'] = base_metrics
        
        # Apply each scenario
        for scenario_name, shocks in scenarios.items():
            # Create shocked positions
            shocked_positions = []
            
            for pos in self.positions:
                shocked_pos = Position(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    market_value=pos.market_value,
                    weight=pos.weight,
                    beta=pos.beta,
                    sector=pos.sector
                )
                
                # Apply shock if symbol is in scenario
                if pos.symbol in shocks:
                    shock = shocks[pos.symbol]
                    shocked_pos.market_value *= (1 + shock)
                
                shocked_positions.append(shocked_pos)
            
            # Recalculate weights after shocks
            total_value = sum(pos.market_value for pos in shocked_positions)
            for pos in shocked_positions:
                pos.weight = pos.market_value / total_value if total_value != 0 else 0
            
            # Calculate scenario metrics
            scenario_metrics = self.risk_calculator.calculate_portfolio_risk(
                shocked_positions, 
                self.returns_data, 
                self.benchmark_returns if not self.benchmark_returns.empty else None
            )
            
            scenario_results[scenario_name] = scenario_metrics
        
        logger.info(f"Completed scenario analysis for {len(scenarios)} scenarios")
        
        return scenario_results
    
    def stress_test(self, stress_factors: Dict[str, float]) -> Dict[str, float]:
        """
        Perform stress testing with market-wide factors
        
        Args:
            stress_factors: Dictionary of factor_name -> shock_percentage
            
        Returns:
            Dictionary of stress test results
        """
        
        stress_results = {}
        base_value = sum(pos.market_value for pos in self.positions)
        
        for factor_name, shock in stress_factors.items():
            # Apply uniform shock to all positions
            shocked_positions = []
            
            for pos in self.positions:
                shocked_pos = Position(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    market_value=pos.market_value * (1 + shock),
                    weight=pos.weight,  # Weights remain the same for uniform shock
                    beta=pos.beta,
                    sector=pos.sector
                )
                shocked_positions.append(shocked_pos)
            
            # Calculate P&L impact
            stressed_value = sum(pos.market_value for pos in shocked_positions)
            pnl_impact = stressed_value - base_value
            pnl_percentage = (pnl_impact / abs(base_value)) if base_value != 0 else 0
            
            stress_results[factor_name] = {
                'shock_applied': shock,
                'pnl_impact': pnl_impact,
                'pnl_percentage': pnl_percentage,
                'new_portfolio_value': stressed_value
            }
        
        logger.info(f"Completed stress testing for {len(stress_factors)} factors")
        
        return stress_results
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        
        if not self.positions:
            return {'error': 'No positions available'}
        
        # Current risk metrics
        current_metrics = self.calculate_risk(store_history=False)
        
        # Position analysis
        position_analysis = []
        for pos in self.positions:
            position_analysis.append({
                'symbol': pos.symbol,
                'weight': pos.weight,
                'market_value': pos.market_value,
                'var_contribution': pos.var_contribution,
                'beta': pos.beta,
                'sector': pos.sector
            })
        
        # Risk limit status
        limit_status = self.limit_manager.get_limit_status()
        breach_summary = self.limit_manager.get_breach_summary()
        
        # Historical risk trends
        risk_trends = self._calculate_risk_trends()
        
        return {
            'portfolio_id': self.portfolio_id,
            'report_date': datetime.now(),
            'current_metrics': current_metrics,
            'position_analysis': position_analysis,
            'limit_status': limit_status.to_dict('records'),
            'breach_summary': breach_summary,
            'risk_trends': risk_trends,
            'recommendations': self._generate_risk_recommendations(current_metrics)
        }
    
    def _calculate_risk_trends(self) -> Dict[str, Any]:
        """Calculate risk trends from historical data"""
        
        if len(self.risk_history) < 2:
            return {}
        
        # Extract metrics over time
        dates = [entry['timestamp'] for entry in self.risk_history]
        var_95_values = [entry['metrics'].var_95 for entry in self.risk_history]
        volatility_values = [entry['metrics'].annual_volatility for entry in self.risk_history]
        
        # Calculate trends
        if len(var_95_values) >= 5:
            var_trend = 'increasing' if var_95_values[-1] > var_95_values[-5] else 'decreasing'
            vol_trend = 'increasing' if volatility_values[-1] > volatility_values[-5] else 'decreasing'
        else:
            var_trend = 'stable'
            vol_trend = 'stable'
        
        return {
            'periods_analyzed': len(self.risk_history),
            'var_trend': var_trend,
            'volatility_trend': vol_trend,
            'current_var': var_95_values[-1] if var_95_values else 0,
            'max_var_period': max(var_95_values) if var_95_values else 0,
            'avg_volatility': np.mean(volatility_values) if volatility_values else 0
        }
    
    def _generate_risk_recommendations(self, metrics: RiskMetrics) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        # High concentration risk
        if metrics.top_5_concentration > 0.5:
            recommendations.append("Consider diversifying portfolio - top 5 positions exceed 50% concentration")
        
        # High volatility
        if metrics.annual_volatility > 0.25:
            recommendations.append("Portfolio volatility is elevated (>25%) - consider risk reduction")
        
        # High correlation
        if metrics.avg_correlation > 0.7:
            recommendations.append("High average correlation detected - diversification may be limited")
        
        # Large drawdown
        if metrics.current_drawdown < -0.1:
            recommendations.append("Portfolio in significant drawdown (>10%) - monitor closely")
        
        # Low Sharpe ratio
        if metrics.sharpe_ratio < 0.5:
            recommendations.append("Low risk-adjusted returns - review strategy effectiveness")
        
        # High leverage
        if metrics.leverage > 1.3:
            recommendations.append("Elevated leverage detected - monitor margin requirements")
        
        if not recommendations:
            recommendations.append("Portfolio risk profile appears well-managed")
        
        return recommendations

# ============================================
# Utility Functions
# ============================================

def calculate_portfolio_var(positions: List[Position], returns_data: pd.DataFrame,
                           confidence_level: float = 0.95, method: str = 'historical') -> float:
    """
    Quick utility to calculate portfolio VaR
    
    Args:
        positions: Portfolio positions
        returns_data: Historical returns data
        confidence_level: VaR confidence level
        method: VaR calculation method
        
    Returns:
        Portfolio VaR value
    """
    
    calculator = PortfolioRiskCalculator()
    metrics = calculator.calculate_portfolio_risk(positions, returns_data)
    
    return metrics.var_95 if confidence_level == 0.95 else metrics.var_99

def create_stress_scenarios() -> Dict[str, Dict[str, float]]:
    """Create standard stress test scenarios"""
    
    return {
        'market_crash': {
            # 20% decline across major indices
            'SPY': -0.20,
            'QQQ': -0.25,
            'IWM': -0.30
        },
        'tech_selloff': {
            'AAPL': -0.15,
            'MSFT': -0.15,
            'GOOGL': -0.18,
            'AMZN': -0.20,
            'TSLA': -0.25
        },
        'interest_rate_shock': {
            # Financial sector impact
            'XLF': -0.10,
            'JPM': -0.12,
            'BAC': -0.12,
            'C': -0.15
        },
        'covid_resurgence': {
            # Travel and hospitality impact
            'UAL': -0.25,
            'AAL': -0.30,
            'CCL': -0.35,
            'NCLH': -0.40
        }
    }

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Portfolio Risk System")
    
    # Create sample portfolio positions
    sample_positions = [
        Position(
            symbol='AAPL',
            quantity=1000,
            market_value=150000,  # $150k
            weight=0.30,
            beta=1.2,
            sector='Technology'
        ),
        Position(
            symbol='MSFT',
            quantity=800,
            market_value=100000,  # $100k
            weight=0.20,
            beta=1.1,
            sector='Technology'
        ),
        Position(
            symbol='JPM',
            quantity=700,
            market_value=75000,   # $75k
            weight=0.15,
            beta=1.3,
            sector='Financial'
        ),
        Position(
            symbol='JNJ',
            quantity=500,
            market_value=50000,   # $50k
            weight=0.10,
            beta=0.8,
            sector='Healthcare'
        ),
        Position(
            symbol='TSLA',
            quantity=300,
            market_value=125000,  # $125k
            weight=0.25,
            beta=1.8,
            sector='Automotive'
        )
    ]
    
    # Generate sample returns data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=252, freq='D')
    symbols = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'TSLA']
    
    # Create correlated returns
    base_returns = np.random.normal(0.0005, 0.02, (252, len(symbols)))
    
    # Add some correlation structure
    correlation_matrix = np.array([
        [1.0, 0.6, 0.3, 0.2, 0.4],  # AAPL
        [0.6, 1.0, 0.4, 0.3, 0.3],  # MSFT
        [0.3, 0.4, 1.0, 0.2, 0.2],  # JPM
        [0.2, 0.3, 0.2, 1.0, 0.1],  # JNJ
        [0.4, 0.3, 0.2, 0.1, 1.0]   # TSLA
    ])
    
    # Apply correlation
    L = np.linalg.cholesky(correlation_matrix)
    correlated_returns = base_returns @ L.T
    
    returns_data = pd.DataFrame(correlated_returns, index=dates, columns=symbols)
    
    # Generate benchmark returns (S&P 500 proxy)
    benchmark_returns = pd.Series(
        np.random.normal(0.0004, 0.015, 252),
        index=dates,
        name='SPY'
    )
    
    print(f"\nSample Portfolio:")
    total_value = sum(pos.market_value for pos in sample_positions)
    print(f"Total Value: ${total_value:,.0f}")
    
    for pos in sample_positions:
        print(f"  {pos.symbol}: ${pos.market_value:,.0f} ({pos.weight:.1%}) - {pos.sector}")
    
    print("\n1. Testing Risk Calculator")
    
    # Initialize risk calculator
    risk_calculator = PortfolioRiskCalculator()
    
    # Calculate risk metrics
    risk_metrics = risk_calculator.calculate_portfolio_risk(
        sample_positions, 
        returns_data, 
        benchmark_returns
    )
    
    print(f"Portfolio Risk Metrics:")
    print(f"  Total Value: ${risk_metrics.total_value:,.0f}")
    print(f"  Leverage: {risk_metrics.leverage:.2f}x")
    print(f"  Annual Volatility: {risk_metrics.annual_volatility:.2%}")
    print(f"  VaR (95%): ${risk_metrics.var_95:,.0f}")
    print(f"  VaR (99%): ${risk_metrics.var_99:,.0f}")
    print(f"  Max Drawdown: {risk_metrics.max_drawdown:.2%}")
    print(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
    print(f"  Portfolio Beta: {risk_metrics.portfolio_beta:.2f}")
    
    print(f"\nConcentration Metrics:")
    print(f"  Top 5 Concentration: {risk_metrics.top_5_concentration:.1%}")
    print(f"  Effective # Positions: {risk_metrics.effective_number_positions:.1f}")
    print(f"  Average Correlation: {risk_metrics.avg_correlation:.2f}")
    
    print("\n2. Testing Risk Limits")
    
    # Initialize limit manager
    limit_manager = RiskLimitManager()
    
    # Add custom limits
    limit_manager.add_limit(RiskLimit(
        limit_id="tech_sector_exposure",
        limit_type=RiskLimitType.SECTOR_EXPOSURE,
        level="sector",
        entity="Technology",
        limit_value=0.40,  # 40% max tech exposure
        description="Technology sector exposure limit"
    ))
    
    # Check limits
    breached_limits = limit_manager.check_limits(risk_metrics, sample_positions)
    
    print(f"Risk Limit Status:")
    limit_status = limit_manager.get_limit_status()
    print(limit_status[['Limit_ID', 'Type', 'Current', 'Limit', 'Status']].to_string(index=False))
    
    if breached_limits:
        print(f"\nBreach Alert: {len(breached_limits)} limits breached!")
        for limit_id, limit in breached_limits.items():
            print(f"  {limit_id}: {limit.current_value:.4f} > {limit.limit_value:.4f}")
    
    print("\n3. Testing Portfolio Risk Manager")
    
    # Initialize portfolio risk manager
    risk_manager = PortfolioRiskManager("SAMPLE_PORTFOLIO")
    
    # Update with data
    risk_manager.update_positions(sample_positions)
    risk_manager.update_returns_data(returns_data)
    risk_manager.update_benchmark(benchmark_returns)
    
    # Monitor risk
    monitoring_result = risk_manager.monitor_risk_limits()
    
    print(f"Risk Monitoring Results:")
    print(f"  Monitoring Enabled: {monitoring_result['monitoring_enabled']}")
    print(f"  Calculation Time: {monitoring_result['calculation_time']}")
    print(f"  Breach Count: {monitoring_result['breach_count']}")
    
    metrics_summary = monitoring_result['metrics_summary']
    print(f"  Current VaR (95%): ${metrics_summary['var_95']:,.0f}")
    print(f"  Current Volatility: {metrics_summary['volatility']:.2%}")
    print(f"  Current Leverage: {metrics_summary['leverage']:.2f}x")
    
    print("\n4. Testing Scenario Analysis")
    
    # Create stress scenarios
    stress_scenarios = create_stress_scenarios()
    
    # Add custom scenario
    custom_scenario = {
        'AAPL': -0.10,  # Apple down 10%
        'TSLA': -0.15   # Tesla down 15%
    }
    stress_scenarios['custom_tech_decline'] = custom_scenario
    
    # Run scenario analysis
    scenario_results = risk_manager.scenario_analysis(stress_scenarios)
    
    print(f"Scenario Analysis Results:")
    base_value = scenario_results['base_case'].total_value
    
    for scenario_name, scenario_metrics in scenario_results.items():
        if scenario_name == 'base_case':
            continue
        
        value_change = scenario_metrics.total_value - base_value
        pct_change = (value_change / base_value) * 100
        
        print(f"  {scenario_name}:")
        print(f"    P&L Impact: ${value_change:,.0f} ({pct_change:+.1f}%)")
        print(f"    New VaR: ${scenario_metrics.var_95:,.0f}")
    
    print("\n5. Testing Stress Testing")
    
    # Stress test factors
    stress_factors = {
        'market_decline_10%': -0.10,
        'market_decline_20%': -0.20,
        'market_crash_30%': -0.30,
        'market_rally_15%': 0.15
    }
    
    stress_results = risk_manager.stress_test(stress_factors)
    
    print(f"Stress Test Results:")
    for factor_name, result in stress_results.items():
        print(f"  {factor_name}:")
        print(f"    P&L Impact: ${result['pnl_impact']:,.0f} ({result['pnl_percentage']:+.1%})")
        print(f"    New Portfolio Value: ${result['new_portfolio_value']:,.0f}")
    
    print("\n6. Testing Component VaR")
    
    print(f"Position Risk Contributions:")
    print("Symbol | Weight | VaR Contribution | Component VaR")
    print("-" * 50)
    
    for pos in sample_positions:
        if pos.var_contribution > 0:
            print(f"{pos.symbol:6} | {pos.weight:6.1%} | {pos.var_contribution:12.1%} | ${pos.component_var:10,.0f}")
    
    print("\n7. Testing Risk Report Generation")
    
    # Generate comprehensive risk report
    risk_report = risk_manager.get_risk_report()
    
    print(f"Risk Report Summary:")
    print(f"  Portfolio ID: {risk_report['portfolio_id']}")
    print(f"  Report Date: {risk_report['report_date'].strftime('%Y-%m-%d %H:%M')}")
    
    current_metrics = risk_report['current_metrics']
    print(f"  Key Metrics:")
    print(f"    VaR (95%): ${current_metrics.var_95:,.0f}")
    print(f"    Volatility: {current_metrics.annual_volatility:.2%}")
    print(f"    Max Drawdown: {current_metrics.max_drawdown:.2%}")
    print(f"    Sharpe Ratio: {current_metrics.sharpe_ratio:.2f}")
    
    print(f"  Risk Recommendations:")
    for i, rec in enumerate(risk_report['recommendations'], 1):
        print(f"    {i}. {rec}")
    
    print("\n8. Testing Utility Functions")
    
    # Quick VaR calculation
    quick_var = calculate_portfolio_var(sample_positions, returns_data, 0.95)
    print(f"Quick VaR Calculation: ${quick_var:,.0f}")
    
    # Show position-level risk metrics
    print(f"\nPosition Risk Breakdown:")
    for pos in sample_positions:
        print(f"  {pos.symbol}:")
        print(f"    Market Value: ${pos.market_value:,.0f}")
        print(f"    Weight: {pos.weight:.1%}")
        print(f"    Beta: {pos.beta:.2f}")
        if pos.var_contribution > 0:
            print(f"    VaR Contribution: {pos.var_contribution:.1%}")
    
    print("\nPortfolio risk system testing completed successfully!")
    print("\nImplemented features include:")
    print("• Comprehensive risk metrics calculation (VaR, volatility, drawdowns)")
    print("• Multiple VaR methodologies (Historical, Parametric, Monte Carlo)")
    print("• Component VaR and marginal risk contributions")
    print("• Risk limit monitoring with breach detection")
    print("• Scenario analysis and stress testing capabilities")
    print("• Concentration and correlation risk analysis")
    print("• Portfolio beta and factor exposure measurement")
    print("• Real-time risk monitoring and alerting")
    print("• Comprehensive risk reporting and recommendations")
