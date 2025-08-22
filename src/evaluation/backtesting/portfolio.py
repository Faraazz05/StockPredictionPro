# ============================================
# StockPredictionPro - src/evaluation/backtesting/portfolio.py
# Advanced portfolio management and optimization for financial backtesting
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
from scipy.optimize import minimize
from scipy.stats import norm
import cvxpy as cp

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('evaluation.backtesting.portfolio')

# ============================================
# Portfolio Data Structures
# ============================================

class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequencies"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    CUSTOM = "custom"

class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_SORTINO = "maximize_sortino"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_DIVERSIFICATION = "max_diversification"
    BLACK_LITTERMAN = "black_litterman"

@dataclass
class AssetAllocation:
    """Container for asset allocation data"""
    symbol: str
    target_weight: float
    current_weight: float
    target_value: float
    current_value: float
    drift: float
    rebalance_amount: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_weights: Optional[Dict[str, float]] = None
    max_weights: Optional[Dict[str, float]] = None
    sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    turnover_constraint: Optional[float] = None
    leverage_constraint: float = 1.0
    long_only: bool = True

@dataclass
class PortfolioMetrics:
    """Portfolio performance and risk metrics"""
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    excess_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    downside_deviation: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Portfolio characteristics
    diversification_ratio: float = 0.0
    concentration_ratio: float = 0.0
    turnover: float = 0.0
    tracking_error: float = 0.0
    
    # Attribution
    sector_allocation: Dict[str, float] = field(default_factory=dict)
    asset_contribution: Dict[str, float] = field(default_factory=dict)

# ============================================
# Portfolio Optimizers
# ============================================

class PortfolioOptimizer(ABC):
    """Base class for portfolio optimizers"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def optimize(self, expected_returns: np.ndarray, 
                covariance_matrix: np.ndarray,
                constraints: PortfolioConstraints,
                **kwargs) -> np.ndarray:
        """
        Optimize portfolio weights
        
        Args:
            expected_returns: Expected returns for assets
            covariance_matrix: Covariance matrix of asset returns
            constraints: Portfolio constraints
            **kwargs: Additional optimizer-specific parameters
            
        Returns:
            Optimal weights array
        """
        pass

class MeanVarianceOptimizer(PortfolioOptimizer):
    """Markowitz Mean-Variance Optimizer"""
    
    def __init__(self, risk_aversion: float = 1.0):
        super().__init__("Mean-Variance")
        self.risk_aversion = risk_aversion
    
    def optimize(self, expected_returns: np.ndarray, 
                covariance_matrix: np.ndarray,
                constraints: PortfolioConstraints,
                **kwargs) -> np.ndarray:
        """Optimize using mean-variance framework"""
        
        n_assets = len(expected_returns)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Objective function: maximize utility (return - risk penalty)
        portfolio_return = expected_returns.T @ weights
        portfolio_risk = cp.quad_form(weights, covariance_matrix)
        objective = cp.Maximize(portfolio_return - 0.5 * self.risk_aversion * portfolio_risk)
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,  # Weights sum to 1
        ]
        
        # Weight bounds
        if constraints.long_only:
            constraints_list.append(weights >= 0)
        
        if constraints.min_weights:
            for i, symbol in enumerate(kwargs.get('symbols', [])):
                if symbol in constraints.min_weights:
                    constraints_list.append(weights[i] >= constraints.min_weights[symbol])
        
        if constraints.max_weights:
            for i, symbol in enumerate(kwargs.get('symbols', [])):
                if symbol in constraints.max_weights:
                    constraints_list.append(weights[i] <= constraints.max_weights[symbol])
        
        # Individual weight bounds
        constraints_list.extend([
            weights >= constraints.min_weight,
            weights <= constraints.max_weight
        ])
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status not in ["infeasible", "unbounded"]:
                return weights.value
            else:
                logger.warning(f"Optimization failed: {problem.status}")
                return np.array([1.0 / n_assets] * n_assets)  # Equal weights fallback
        
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return np.array([1.0 / n_assets] * n_assets)

class MinVarianceOptimizer(PortfolioOptimizer):
    """Minimum Variance Portfolio Optimizer"""
    
    def __init__(self):
        super().__init__("Min-Variance")
    
    def optimize(self, expected_returns: np.ndarray, 
                covariance_matrix: np.ndarray,
                constraints: PortfolioConstraints,
                **kwargs) -> np.ndarray:
        """Optimize for minimum variance"""
        
        n_assets = len(expected_returns)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Objective: minimize portfolio variance
        portfolio_variance = cp.quad_form(weights, covariance_matrix)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints_list = [cp.sum(weights) == 1]
        
        if constraints.long_only:
            constraints_list.append(weights >= 0)
        
        constraints_list.extend([
            weights >= constraints.min_weight,
            weights <= constraints.max_weight
        ])
        
        # Solve optimization
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status not in ["infeasible", "unbounded"]:
                return weights.value
            else:
                logger.warning(f"Min variance optimization failed: {problem.status}")
                return self._equal_weights_fallback(n_assets)
        
        except Exception as e:
            logger.error(f"Min variance optimization error: {e}")
            return self._equal_weights_fallback(n_assets)
    
    def _equal_weights_fallback(self, n_assets: int) -> np.ndarray:
        """Fallback to equal weights"""
        return np.array([1.0 / n_assets] * n_assets)

class RiskParityOptimizer(PortfolioOptimizer):
    """Risk Parity Portfolio Optimizer"""
    
    def __init__(self):
        super().__init__("Risk-Parity")
    
    def optimize(self, expected_returns: np.ndarray, 
                covariance_matrix: np.ndarray,
                constraints: PortfolioConstraints,
                **kwargs) -> np.ndarray:
        """Optimize for risk parity"""
        
        n_assets = len(expected_returns)
        
        def risk_parity_objective(weights):
            """Risk parity objective function"""
            weights = np.array(weights)
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
            
            # Risk contributions
            marginal_risk = covariance_matrix @ weights / portfolio_vol
            risk_contributions = weights * marginal_risk
            
            # Target equal risk contributions
            target_risk = portfolio_vol / n_assets
            
            # Minimize sum of squared deviations from equal risk
            return np.sum((risk_contributions - target_risk) ** 2)
        
        # Constraints
        constraints_scipy = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        if constraints.long_only:
            bounds = [(max(0.0, b[0]), b[1]) for b in bounds]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        try:
            result = minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_scipy,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                logger.warning("Risk parity optimization failed")
                return x0
        
        except Exception as e:
            logger.error(f"Risk parity optimization error: {e}")
            return x0

class MaxSharpeOptimizer(PortfolioOptimizer):
    """Maximum Sharpe Ratio Optimizer"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        super().__init__("Max-Sharpe")
        self.risk_free_rate = risk_free_rate
    
    def optimize(self, expected_returns: np.ndarray, 
                covariance_matrix: np.ndarray,
                constraints: PortfolioConstraints,
                **kwargs) -> np.ndarray:
        """Optimize for maximum Sharpe ratio"""
        
        n_assets = len(expected_returns)
        
        def neg_sharpe_ratio(weights):
            """Negative Sharpe ratio (for minimization)"""
            weights = np.array(weights)
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
            
            if portfolio_vol == 0:
                return -np.inf
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Negative for minimization
        
        # Constraints
        constraints_scipy = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        if constraints.long_only:
            bounds = [(max(0.0, b[0]), b[1]) for b in bounds]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        try:
            result = minimize(
                neg_sharpe_ratio,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_scipy,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                logger.warning("Max Sharpe optimization failed")
                return x0
        
        except Exception as e:
            logger.error(f"Max Sharpe optimization error: {e}")
            return x0

# ============================================
# Portfolio Manager
# ============================================

class PortfolioManager:
    """
    Comprehensive portfolio management system.
    
    This class handles portfolio construction, optimization, rebalancing,
    and performance analysis for backtesting and live trading.
    """
    
    def __init__(self, symbols: List[str], 
                 initial_weights: Optional[Dict[str, float]] = None,
                 constraints: Optional[PortfolioConstraints] = None,
                 optimizer: Optional[PortfolioOptimizer] = None,
                 rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
                 benchmark_symbol: Optional[str] = None):
        """
        Initialize portfolio manager
        
        Args:
            symbols: List of asset symbols
            initial_weights: Initial portfolio weights
            constraints: Portfolio constraints
            optimizer: Portfolio optimizer
            rebalance_frequency: How often to rebalance
            benchmark_symbol: Benchmark for comparison
        """
        
        self.symbols = symbols
        self.n_assets = len(symbols)
        
        # Portfolio weights
        if initial_weights:
            self.target_weights = np.array([initial_weights.get(symbol, 0.0) for symbol in symbols])
        else:
            self.target_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        self.current_weights = self.target_weights.copy()
        
        # Configuration
        self.constraints = constraints or PortfolioConstraints()
        self.optimizer = optimizer or MeanVarianceOptimizer()
        self.rebalance_frequency = rebalance_frequency
        self.benchmark_symbol = benchmark_symbol
        
        # Tracking data
        self.price_history = {}
        self.return_history = {}
        self.weight_history = []
        self.rebalance_history = []
        self.portfolio_values = []
        self.benchmark_values = []
        
        # Performance metrics
        self.metrics = PortfolioMetrics()
        
        # Rebalancing control
        self.days_since_rebalance = 0
        self.last_rebalance_date = None
        
        logger.info(f"PortfolioManager initialized with {self.n_assets} assets")
    
    @time_it("portfolio_update")
    def update(self, timestamp: pd.Timestamp, market_data: Dict[str, Any],
               portfolio_value: float):
        """
        Update portfolio with new market data
        
        Args:
            timestamp: Current timestamp
            market_data: Market data for all symbols
            portfolio_value: Current total portfolio value
        """
        
        # Update price history
        current_prices = {}
        for symbol in self.symbols:
            if symbol in market_data and 'close' in market_data[symbol]:
                price = market_data[symbol]['close']
                current_prices[symbol] = price
                
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                self.price_history[symbol].append(price)
        
        # Calculate returns if we have previous prices
        if len(current_prices) == self.n_assets:
            returns = self._calculate_returns(current_prices)
            
            if returns is not None:
                for i, symbol in enumerate(self.symbols):
                    if symbol not in self.return_history:
                        self.return_history[symbol] = []
                    self.return_history[symbol].append(returns[i])
        
        # Update portfolio value tracking
        self.portfolio_values.append(portfolio_value)
        
        # Update benchmark if specified
        if self.benchmark_symbol and self.benchmark_symbol in market_data:
            benchmark_price = market_data[self.benchmark_symbol]['close']
            self.benchmark_values.append(benchmark_price)
        
        # Update current weights based on price changes
        self._update_current_weights(current_prices, portfolio_value)
        
        # Record weight history
        self.weight_history.append({
            'timestamp': timestamp,
            'weights': self.current_weights.copy(),
            'target_weights': self.target_weights.copy()
        })
        
        # Check if rebalancing is needed
        self.days_since_rebalance += 1
        
        if self._should_rebalance(timestamp):
            self._rebalance_portfolio(timestamp, current_prices, portfolio_value)
            self.days_since_rebalance = 0
            self.last_rebalance_date = timestamp
    
    def _calculate_returns(self, current_prices: Dict[str, float]) -> Optional[np.ndarray]:
        """Calculate returns for current period"""
        
        returns = []
        
        for symbol in self.symbols:
            if (symbol in current_prices and 
                symbol in self.price_history and 
                len(self.price_history[symbol]) > 1):
                
                current_price = current_prices[symbol]
                previous_price = self.price_history[symbol][-2]
                
                if previous_price > 0:
                    ret = (current_price - previous_price) / previous_price
                    returns.append(ret)
                else:
                    returns.append(0.0)
            else:
                returns.append(0.0)
        
        return np.array(returns) if len(returns) == self.n_assets else None
    
    def _update_current_weights(self, current_prices: Dict[str, float], portfolio_value: float):
        """Update current weights based on price changes"""
        
        if portfolio_value <= 0 or len(current_prices) != self.n_assets:
            return
        
        # Calculate current asset values
        asset_values = []
        for i, symbol in enumerate(self.symbols):
            if symbol in current_prices:
                # Estimate current holding based on previous weight and price changes
                if len(self.portfolio_values) > 1:
                    prev_portfolio_value = self.portfolio_values[-2]
                    prev_asset_value = prev_portfolio_value * self.current_weights[i]
                    
                    if symbol in self.price_history and len(self.price_history[symbol]) > 1:
                        price_change = (current_prices[symbol] / self.price_history[symbol][-2])
                        current_asset_value = prev_asset_value * price_change
                    else:
                        current_asset_value = prev_asset_value
                else:
                    current_asset_value = portfolio_value * self.current_weights[i]
                
                asset_values.append(current_asset_value)
            else:
                asset_values.append(0.0)
        
        # Update current weights
        total_value = sum(asset_values)
        if total_value > 0:
            self.current_weights = np.array(asset_values) / total_value
    
    def _should_rebalance(self, timestamp: pd.Timestamp) -> bool:
        """Determine if portfolio should be rebalanced"""
        
        # Check frequency-based rebalancing
        frequency_check = False
        
        if self.rebalance_frequency == RebalanceFrequency.DAILY:
            frequency_check = self.days_since_rebalance >= 1
        elif self.rebalance_frequency == RebalanceFrequency.WEEKLY:
            frequency_check = self.days_since_rebalance >= 7
        elif self.rebalance_frequency == RebalanceFrequency.MONTHLY:
            frequency_check = self.days_since_rebalance >= 21  # ~21 trading days
        elif self.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            frequency_check = self.days_since_rebalance >= 63  # ~3 months
        elif self.rebalance_frequency == RebalanceFrequency.ANNUALLY:
            frequency_check = self.days_since_rebalance >= 252  # ~1 year
        
        # Check drift-based rebalancing
        max_drift = np.max(np.abs(self.current_weights - self.target_weights))
        drift_check = max_drift > 0.05  # 5% drift threshold
        
        return frequency_check or drift_check
    
    def _rebalance_portfolio(self, timestamp: pd.Timestamp, 
                           current_prices: Dict[str, float], 
                           portfolio_value: float):
        """Rebalance portfolio to target weights"""
        
        logger.info(f"Rebalancing portfolio at {timestamp}")
        
        # Calculate expected returns and covariance matrix for optimization
        expected_returns, covariance_matrix = self._estimate_risk_return_parameters()
        
        if expected_returns is not None and covariance_matrix is not None:
            # Optimize portfolio
            try:
                optimal_weights = self.optimizer.optimize(
                    expected_returns, covariance_matrix, self.constraints,
                    symbols=self.symbols
                )
                
                if optimal_weights is not None and len(optimal_weights) == self.n_assets:
                    # Normalize weights to ensure they sum to 1
                    optimal_weights = optimal_weights / np.sum(optimal_weights)
                    
                    # Update target weights
                    old_targets = self.target_weights.copy()
                    self.target_weights = optimal_weights
                    
                    logger.info(f"Updated target weights: {dict(zip(self.symbols, optimal_weights))}")
                else:
                    logger.warning("Optimization failed, keeping current target weights")
            
            except Exception as e:
                logger.error(f"Portfolio optimization error: {e}")
        
        # Calculate rebalancing trades
        allocations = []
        
        for i, symbol in enumerate(self.symbols):
            current_weight = self.current_weights[i]
            target_weight = self.target_weights[i]
            
            current_value = portfolio_value * current_weight
            target_value = portfolio_value * target_weight
            rebalance_amount = target_value - current_value
            
            allocation = AssetAllocation(
                symbol=symbol,
                target_weight=target_weight,
                current_weight=current_weight,
                target_value=target_value,
                current_value=current_value,
                drift=target_weight - current_weight,
                rebalance_amount=rebalance_amount
            )
            
            allocations.append(allocation)
        
        # Record rebalancing event
        self.rebalance_history.append({
            'timestamp': timestamp,
            'allocations': allocations,
            'portfolio_value': portfolio_value,
            'optimization_method': self.optimizer.name
        })
        
        # Update current weights to target weights (assuming perfect execution)
        self.current_weights = self.target_weights.copy()
    
    def _estimate_risk_return_parameters(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate expected returns and covariance matrix"""
        
        # Check if we have enough return history
        min_history = 20  # Minimum periods needed
        
        if not all(len(self.return_history.get(symbol, [])) >= min_history 
                  for symbol in self.symbols):
            return None, None
        
        # Create return matrix
        returns_matrix = []
        min_length = min(len(self.return_history[symbol]) for symbol in self.symbols)
        
        for symbol in self.symbols:
            symbol_returns = self.return_history[symbol][-min_length:]
            returns_matrix.append(symbol_returns)
        
        returns_matrix = np.array(returns_matrix).T  # Shape: (periods, assets)
        
        # Calculate expected returns (simple historical mean)
        expected_returns = np.mean(returns_matrix, axis=0)
        
        # Calculate covariance matrix
        covariance_matrix = np.cov(returns_matrix.T)
        
        # Ensure covariance matrix is positive definite
        eigenvals = np.linalg.eigvals(covariance_matrix)
        if np.min(eigenvals) <= 0:
            # Add small regularization term
            regularization = 1e-8 * np.eye(self.n_assets)
            covariance_matrix += regularization
        
        return expected_returns, covariance_matrix
    
    @time_it("portfolio_metrics_calculation")
    def calculate_metrics(self, risk_free_rate: float = 0.02, 
                         periods_per_year: int = 252) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        
        if len(self.portfolio_values) < 2:
            return self.metrics
        
        # Calculate portfolio returns
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        if len(returns) == 0:
            return self.metrics
        
        # Basic return metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        self.metrics.total_return = total_return
        
        # Annualized return
        n_periods = len(returns)
        if n_periods > 0:
            annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
            self.metrics.annualized_return = annualized_return
        
        # Risk metrics
        self.metrics.volatility = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
        
        # Downside deviation (for Sortino ratio)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            self.metrics.downside_deviation = np.std(downside_returns, ddof=1) * np.sqrt(periods_per_year)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        self.metrics.max_drawdown = abs(np.min(drawdown))
        
        # VaR and CVaR
        if len(returns) > 0:
            var_95 = -np.percentile(returns, 5)
            self.metrics.var_95 = var_95
            
            # CVaR (expected shortfall)
            worst_returns = returns[returns <= -var_95]
            if len(worst_returns) > 0:
                self.metrics.cvar_95 = -np.mean(worst_returns)
        
        # Risk-adjusted ratios
        if self.metrics.volatility > 0:
            excess_return = self.metrics.annualized_return - risk_free_rate
            self.metrics.sharpe_ratio = excess_return / self.metrics.volatility
            
            if self.metrics.downside_deviation > 0:
                self.metrics.sortino_ratio = excess_return / self.metrics.downside_deviation
        
        if self.metrics.max_drawdown > 0:
            self.metrics.calmar_ratio = self.metrics.annualized_return / self.metrics.max_drawdown
        
        # Benchmark comparison
        if self.benchmark_values and len(self.benchmark_values) == len(self.portfolio_values):
            self._calculate_benchmark_metrics(risk_free_rate, periods_per_year)
        
        # Portfolio characteristics
        self._calculate_portfolio_characteristics()
        
        return self.metrics
    
    def _calculate_benchmark_metrics(self, risk_free_rate: float, periods_per_year: int):
        """Calculate benchmark-relative metrics"""
        
        portfolio_values = np.array(self.portfolio_values)
        benchmark_values = np.array(self.benchmark_values)
        
        # Normalize both series to start at the same value
        benchmark_values = benchmark_values * (portfolio_values[0] / benchmark_values[0])
        
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        
        if len(portfolio_returns) == len(benchmark_returns) and len(portfolio_returns) > 0:
            # Excess returns vs benchmark
            active_returns = portfolio_returns - benchmark_returns
            self.metrics.excess_return = np.mean(active_returns) * periods_per_year
            
            # Tracking error
            self.metrics.tracking_error = np.std(active_returns, ddof=1) * np.sqrt(periods_per_year)
            
            # Information ratio
            if self.metrics.tracking_error > 0:
                self.metrics.information_ratio = self.metrics.excess_return / self.metrics.tracking_error
    
    def _calculate_portfolio_characteristics(self):
        """Calculate portfolio-specific characteristics"""
        
        if len(self.weight_history) == 0:
            return
        
        # Current concentration (Herfindahl index)
        current_weights = self.current_weights
        self.metrics.concentration_ratio = np.sum(current_weights ** 2)
        
        # Diversification ratio (would need asset volatilities for full calculation)
        # Simplified version using weight distribution
        equal_weight = 1.0 / self.n_assets
        weight_deviations = np.abs(current_weights - equal_weight)
        self.metrics.diversification_ratio = 1 - np.mean(weight_deviations) / equal_weight
        
        # Turnover calculation
        if len(self.weight_history) > 1:
            recent_weights = [entry['weights'] for entry in self.weight_history[-10:]]
            if len(recent_weights) > 1:
                weight_changes = []
                for i in range(1, len(recent_weights)):
                    change = np.sum(np.abs(recent_weights[i] - recent_weights[i-1]))
                    weight_changes.append(change)
                
                self.metrics.turnover = np.mean(weight_changes) if weight_changes else 0.0
    
    def get_asset_allocations(self, portfolio_value: float) -> List[AssetAllocation]:
        """Get current asset allocation details"""
        
        allocations = []
        
        for i, symbol in enumerate(self.symbols):
            current_weight = self.current_weights[i]
            target_weight = self.target_weights[i]
            
            current_value = portfolio_value * current_weight
            target_value = portfolio_value * target_weight
            
            allocation = AssetAllocation(
                symbol=symbol,
                target_weight=target_weight,
                current_weight=current_weight,
                target_value=target_value,
                current_value=current_value,
                drift=current_weight - target_weight,
                rebalance_amount=target_value - current_value
            )
            
            allocations.append(allocation)
        
        return allocations
    
    def generate_portfolio_report(self) -> str:
        """Generate comprehensive portfolio report"""
        
        metrics = self.calculate_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("PORTFOLIO MANAGEMENT REPORT")
        report.append("=" * 60)
        
        # Portfolio composition
        report.append("\n📊 CURRENT PORTFOLIO COMPOSITION")
        report.append("-" * 35)
        
        for i, symbol in enumerate(self.symbols):
            current_weight = self.current_weights[i]
            target_weight = self.target_weights[i]
            drift = current_weight - target_weight
            
            report.append(f"{symbol:>8}: {current_weight:>6.1%} (target: {target_weight:>6.1%}, "
                         f"drift: {drift:>+6.1%})")
        
        # Performance metrics
        report.append("\n📈 PERFORMANCE METRICS")
        report.append("-" * 25)
        report.append(f"Total Return: {metrics.total_return:>19.2%}")
        report.append(f"Annualized Return: {metrics.annualized_return:>14.2%}")
        report.append(f"Volatility: {metrics.volatility:>21.2%}")
        report.append(f"Sharpe Ratio: {metrics.sharpe_ratio:>19.3f}")
        report.append(f"Sortino Ratio: {metrics.sortino_ratio:>18.3f}")
        report.append(f"Calmar Ratio: {metrics.calmar_ratio:>19.3f}")
        
        # Risk metrics
        report.append("\n🛡️  RISK METRICS")
        report.append("-" * 20)
        report.append(f"Maximum Drawdown: {metrics.max_drawdown:>15.2%}")
        report.append(f"Value at Risk (95%): {metrics.var_95:>12.2%}")
        report.append(f"Conditional VaR (95%): {metrics.cvar_95:>9.2%}")
        report.append(f"Downside Deviation: {metrics.downside_deviation:>13.2%}")
        
        # Portfolio characteristics
        report.append("\n⚖️  PORTFOLIO CHARACTERISTICS")
        report.append("-" * 30)
        report.append(f"Concentration Ratio: {metrics.concentration_ratio:>14.3f}")
        report.append(f"Diversification Ratio: {metrics.diversification_ratio:>11.3f}")
        report.append(f"Portfolio Turnover: {metrics.turnover:>15.2%}")
        
        # Benchmark comparison
        if metrics.excess_return != 0:
            report.append("\n📊 BENCHMARK COMPARISON")
            report.append("-" * 25)
            report.append(f"Excess Return: {metrics.excess_return:>18.2%}")
            report.append(f"Tracking Error: {metrics.tracking_error:>17.2%}")
            report.append(f"Information Ratio: {metrics.information_ratio:>14.3f}")
        
        # Rebalancing history
        if self.rebalance_history:
            report.append("\n🔄 REBALANCING HISTORY")
            report.append("-" * 25)
            report.append(f"Total Rebalances: {len(self.rebalance_history):>17d}")
            report.append(f"Days Since Last: {self.days_since_rebalance:>18d}")
            report.append(f"Optimizer Used: {self.optimizer.name:>19s}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

# ============================================
# Utility Functions
# ============================================

def create_equal_weight_portfolio(symbols: List[str]) -> PortfolioManager:
    """Create equal-weight portfolio"""
    
    equal_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
    
    return PortfolioManager(
        symbols=symbols,
        initial_weights=equal_weights,
        optimizer=None,  # No optimization, maintain equal weights
        rebalance_frequency=RebalanceFrequency.QUARTERLY
    )

def create_min_variance_portfolio(symbols: List[str], 
                                constraints: Optional[PortfolioConstraints] = None) -> PortfolioManager:
    """Create minimum variance portfolio"""
    
    return PortfolioManager(
        symbols=symbols,
        constraints=constraints or PortfolioConstraints(),
        optimizer=MinVarianceOptimizer(),
        rebalance_frequency=RebalanceFrequency.MONTHLY
    )

def create_max_sharpe_portfolio(symbols: List[str], 
                              risk_free_rate: float = 0.02,
                              constraints: Optional[PortfolioConstraints] = None) -> PortfolioManager:
    """Create maximum Sharpe ratio portfolio"""
    
    return PortfolioManager(
        symbols=symbols,
        constraints=constraints or PortfolioConstraints(),
        optimizer=MaxSharpeOptimizer(risk_free_rate),
        rebalance_frequency=RebalanceFrequency.MONTHLY
    )

def create_risk_parity_portfolio(symbols: List[str],
                               constraints: Optional[PortfolioConstraints] = None) -> PortfolioManager:
    """Create risk parity portfolio"""
    
    return PortfolioManager(
        symbols=symbols,
        constraints=constraints or PortfolioConstraints(),
        optimizer=RiskParityOptimizer(),
        rebalance_frequency=RebalanceFrequency.QUARTERLY
    )

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Portfolio Management System")
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'JNJ', 'XOM']
    
    # Generate sample market data
    np.random.seed(42)
    n_days = 252  # One year of data
    
    # Create correlated price series
    returns_data = {}
    base_returns = np.random.multivariate_normal(
        mean=[0.0008, 0.0006, 0.0010, 0.0004, 0.0005],  # Different expected returns
        cov=[[0.0004, 0.0001, 0.0002, 0.0000, 0.0001],  # Correlation matrix
             [0.0001, 0.0003, 0.0001, 0.0000, 0.0000],
             [0.0002, 0.0001, 0.0005, 0.0000, 0.0001],
             [0.0000, 0.0000, 0.0000, 0.0002, 0.0000],
             [0.0001, 0.0000, 0.0001, 0.0000, 0.0006]],
        size=n_days
    )
    
    # Convert to price series
    prices = {}
    base_price = 100
    
    for i, symbol in enumerate(test_symbols):
        symbol_returns = base_returns[:, i]
        symbol_prices = [base_price]
        
        for ret in symbol_returns:
            symbol_prices.append(symbol_prices[-1] * (1 + ret))
        
        prices[symbol] = symbol_prices[1:]  # Exclude initial price
    
    print(f"Generated {n_days} days of price data for {len(test_symbols)} assets")
    
    # Test different portfolio strategies
    print("\n1. Testing Portfolio Creation")
    
    # Equal weight portfolio
    equal_weight_pm = create_equal_weight_portfolio(test_symbols)
    print(f"Equal weight portfolio: {dict(zip(test_symbols, equal_weight_pm.target_weights))}")
    
    # Minimum variance portfolio
    min_var_pm = create_min_variance_portfolio(test_symbols)
    print(f"Min variance portfolio created with {min_var_pm.optimizer.name} optimizer")
    
    # Maximum Sharpe portfolio
    max_sharpe_pm = create_max_sharpe_portfolio(test_symbols, risk_free_rate=0.02)
    print(f"Max Sharpe portfolio created with {max_sharpe_pm.optimizer.name} optimizer")
    
    # Risk parity portfolio
    risk_parity_pm = create_risk_parity_portfolio(test_symbols)
    print(f"Risk parity portfolio created with {risk_parity_pm.optimizer.name} optimizer")
    
    # Test portfolio updates
    print("\n2. Testing Portfolio Updates")
    
    portfolios = {
        'Equal Weight': equal_weight_pm,
        'Min Variance': min_var_pm,
        'Max Sharpe': max_sharpe_pm,
        'Risk Parity': risk_parity_pm
    }
    
    # Simulate portfolio updates
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    initial_portfolio_value = 1000000  # $1M
    
    for portfolio_name, pm in portfolios.items():
        current_value = initial_portfolio_value
        
        print(f"\nSimulating {portfolio_name} portfolio:")
        
        for day in range(min(50, n_days)):  # First 50 days for testing
            timestamp = dates[day]
            
            # Create market data
            market_data = {}
            for symbol in test_symbols:
                market_data[symbol] = {
                    'open': prices[symbol][day] * 0.999,
                    'high': prices[symbol][day] * 1.001,
                    'low': prices[symbol][day] * 0.998,
                    'close': prices[symbol][day],
                    'volume': np.random.randint(1000000, 5000000)
                }
            
            # Update portfolio value based on returns
            if day > 0:
                portfolio_return = 0
                for i, symbol in enumerate(test_symbols):
                    weight = pm.current_weights[i]
                    asset_return = (prices[symbol][day] - prices[symbol][day-1]) / prices[symbol][day-1]
                    portfolio_return += weight * asset_return
                
                current_value *= (1 + portfolio_return)
            
            # Update portfolio
            pm.update(timestamp, market_data, current_value)
            
            if day % 10 == 0:  # Print every 10 days
                allocations = pm.get_asset_allocations(current_value)
                max_drift = max(abs(alloc.drift) for alloc in allocations)
                print(f"  Day {day+1}: Value=${current_value:,.0f}, Max drift={max_drift:.1%}")
        
        # Calculate final metrics
        final_metrics = pm.calculate_metrics()
        print(f"  Final return: {final_metrics.total_return:.2%}")
        print(f"  Volatility: {final_metrics.volatility:.2%}")
        print(f"  Sharpe ratio: {final_metrics.sharpe_ratio:.3f}")
        print(f"  Max drawdown: {final_metrics.max_drawdown:.2%}")
        print(f"  Rebalances: {len(pm.rebalance_history)}")
    
    # Test portfolio optimization
    print("\n3. Testing Portfolio Optimizers")
    
    # Create sample expected returns and covariance matrix
    expected_returns = np.array([0.08, 0.06, 0.10, 0.04, 0.05])  # 8%, 6%, 10%, 4%, 5%
    correlation_matrix = np.array([
        [1.00, 0.30, 0.50, 0.10, 0.20],
        [0.30, 1.00, 0.40, 0.05, 0.15],
        [0.50, 0.40, 1.00, 0.15, 0.25],
        [0.10, 0.05, 0.15, 1.00, 0.05],
        [0.20, 0.15, 0.25, 0.05, 1.00]
    ])
    
    volatilities = np.array([0.20, 0.18, 0.25, 0.12, 0.30])  # 20%, 18%, 25%, 12%, 30%
    covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    constraints = PortfolioConstraints(min_weight=0.05, max_weight=0.40, long_only=True)
    
    optimizers = [
        MeanVarianceOptimizer(risk_aversion=2.0),
        MinVarianceOptimizer(),
        MaxSharpeOptimizer(risk_free_rate=0.02),
        RiskParityOptimizer()
    ]
    
    print("Optimizer comparison:")
    for optimizer in optimizers:
        try:
            optimal_weights = optimizer.optimize(expected_returns, covariance_matrix, constraints, symbols=test_symbols)
            
            if optimal_weights is not None:
                weights_dict = dict(zip(test_symbols, optimal_weights))
                portfolio_return = np.sum(optimal_weights * expected_returns)
                portfolio_vol = np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights)
                sharpe = (portfolio_return - 0.02) / portfolio_vol
                
                print(f"\n{optimizer.name}:")
                print(f"  Weights: {weights_dict}")
                print(f"  Expected return: {portfolio_return:.2%}")
                print(f"  Volatility: {portfolio_vol:.2%}")
                print(f"  Sharpe ratio: {sharpe:.3f}")
            else:
                print(f"{optimizer.name}: Optimization failed")
        
        except Exception as e:
            print(f"{optimizer.name}: Error - {e}")
    
    # Test comprehensive portfolio report
    print("\n4. Testing Portfolio Report Generation")
    
    # Use the equal weight portfolio that has been updated
    report = equal_weight_pm.generate_portfolio_report()
    print(report)
    
    # Test asset allocations
    print("\n5. Testing Asset Allocation Analysis")
    
    final_value = equal_weight_pm.portfolio_values[-1] if equal_weight_pm.portfolio_values else 1000000
    allocations = equal_weight_pm.get_asset_allocations(final_value)
    
    print("Asset Allocation Details:")
    for allocation in allocations:
        print(f"{allocation.symbol}:")
        print(f"  Current weight: {allocation.current_weight:.1%}")
        print(f"  Target weight: {allocation.target_weight:.1%}")
        print(f"  Drift: {allocation.drift:+.1%}")
        print(f"  Current value: ${allocation.current_value:,.0f}")
        print(f"  Rebalance needed: ${allocation.rebalance_amount:+,.0f}")
    
    # Performance comparison
    print("\n6. Portfolio Performance Comparison")
    
    comparison_data = []
    for name, pm in portfolios.items():
        metrics = pm.calculate_metrics()
        comparison_data.append({
            'Portfolio': name,
            'Return': f"{metrics.total_return:.2%}",
            'Volatility': f"{metrics.volatility:.2%}",
            'Sharpe': f"{metrics.sharpe_ratio:.3f}",
            'Max DD': f"{metrics.max_drawdown:.2%}",
            'Rebalances': len(pm.rebalance_history)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("Portfolio Performance Summary:")
    print(comparison_df.to_string(index=False))
    
    print("\nPortfolio management testing completed successfully!")
