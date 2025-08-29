"""
app/components/metrics.py

Comprehensive metrics and evaluation components for StockPredictionPro.
Provides ML model evaluation metrics, trading performance metrics,
and financial analysis metrics with Streamlit integration.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# ============================================
# REGRESSION METRICS
# ============================================

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE)"""
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error (MSE)"""
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate coefficient of determination (R²)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate directional accuracy for price prediction"""
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return np.mean(true_direction == pred_direction)

# ============================================
# CLASSIFICATION METRICS
# ============================================

def classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy"""
    return np.mean(y_true == y_pred)

def confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate precision, recall, F1-score, and specificity"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn
    }

# ============================================
# TRADING PERFORMANCE METRICS
# ============================================

def total_return(returns: pd.Series) -> float:
    """Calculate total return"""
    return (1 + returns).prod() - 1

def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized return"""
    total_ret = total_return(returns)
    num_periods = len(returns)
    return (1 + total_ret) ** (periods_per_year / num_periods) - 1 if num_periods > 0 else 0

def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility"""
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / periods_per_year
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year) if excess_returns.std() != 0 else 0

def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate Sortino ratio (downside deviation)"""
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year) if downside_std != 0 else 0

def maximum_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Calmar ratio (annual return / max drawdown)"""
    annual_ret = annualized_return(returns, periods_per_year)
    max_dd = abs(maximum_drawdown(returns))
    return annual_ret / max_dd if max_dd != 0 else 0

def value_at_risk(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk (VaR)"""
    return np.percentile(returns, confidence_level * 100)

def expected_shortfall(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Expected Shortfall (Conditional VaR)"""
    var = value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()

def win_rate(trades: pd.DataFrame, profit_column: str = 'profit') -> float:
    """Calculate win rate of trades"""
    if trades.empty or profit_column not in trades.columns:
        return 0.0
    profitable_trades = trades[trades[profit_column] > 0]
    return len(profitable_trades) / len(trades) if len(trades) > 0 else 0

def profit_factor(trades: pd.DataFrame, profit_column: str = 'profit') -> float:
    """Calculate profit factor (gross profit / gross loss)"""
    if trades.empty or profit_column not in trades.columns:
        return 0.0
    
    gross_profit = trades[trades[profit_column] > 0][profit_column].sum()
    gross_loss = abs(trades[trades[profit_column] < 0][profit_column].sum())
    
    return gross_profit / gross_loss if gross_loss != 0 else float('inf') if gross_profit > 0 else 0

def average_trade(trades: pd.DataFrame, profit_column: str = 'profit') -> float:
    """Calculate average trade profit/loss"""
    if trades.empty or profit_column not in trades.columns:
        return 0.0
    return trades[profit_column].mean()

# ============================================
# STREAMLIT DISPLAY FUNCTIONS
# ============================================

def display_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Regression Metrics") -> None:
    """Display regression metrics in Streamlit"""
    st.subheader(f"📊 {title}")
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    dir_acc = directional_accuracy(y_true, y_pred)
    
    # Display in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("MAE", f"{mae:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
    
    with col2:
        st.metric("MAPE", f"{mape:.2f}%")
        st.metric("R² Score", f"{r2:.4f}")
    
    with col3:
        st.metric("MSE", f"{mse:.4f}")
        st.metric("Directional Accuracy", f"{dir_acc:.2%}")

def display_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Classification Metrics") -> None:
    """Display classification metrics in Streamlit"""
    st.subheader(f"🎯 {title}")
    
    # Calculate metrics
    metrics = confusion_matrix_metrics(y_true, y_pred)
    
    # Display main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    
    with col4:
        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    
    # Confusion matrix
    with st.expander("Confusion Matrix Details"):
        conf_col1, conf_col2 = st.columns(2)
        
        with conf_col1:
            st.metric("True Positives", int(metrics['true_positives']))
            st.metric("False Positives", int(metrics['false_positives']))
        
        with conf_col2:
            st.metric("True Negatives", int(metrics['true_negatives']))
            st.metric("False Negatives", int(metrics['false_negatives']))

def display_trading_metrics(returns: pd.Series, trades: pd.DataFrame = None, title: str = "Trading Performance") -> None:
    """Display trading performance metrics in Streamlit"""
    st.subheader(f"💰 {title}")
    
    if returns.empty:
        st.warning("No return data available")
        return
    
    # Calculate performance metrics
    total_ret = total_return(returns)
    annual_ret = annualized_return(returns)
    vol = volatility(returns)
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    max_dd = maximum_drawdown(returns)
    calmar = calmar_ratio(returns)
    var_95 = value_at_risk(returns, 0.05)
    
    # Display main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{total_ret:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
    
    with col2:
        st.metric("Annual Return", f"{annual_ret:.2%}")
        st.metric("Sortino Ratio", f"{sortino:.3f}")
    
    with col3:
        st.metric("Volatility", f"{vol:.2%}")
        st.metric("Calmar Ratio", f"{calmar:.3f}")
    
    with col4:
        st.metric("Max Drawdown", f"{max_dd:.2%}")
        st.metric("VaR (95%)", f"{var_95:.2%}")
    
    # Trade-specific metrics if trades provided
    if trades is not None and not trades.empty:
        st.subheader("📈 Trade Statistics")
        
        win_rt = win_rate(trades)
        profit_fct = profit_factor(trades)
        avg_trade = average_trade(trades)
        
        trade_col1, trade_col2, trade_col3 = st.columns(3)
        
        with trade_col1:
            st.metric("Win Rate", f"{win_rt:.1%}")
        
        with trade_col2:
            st.metric("Profit Factor", f"{profit_fct:.2f}")
        
        with trade_col3:
            st.metric("Avg Trade", f"${avg_trade:.2f}")

def create_performance_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
    title: str = "Performance Summary"
) -> Dict[str, float]:
    """Create comprehensive performance summary"""
    
    summary = {
        "Total Return": total_return(returns),
        "Annualized Return": annualized_return(returns),
        "Volatility": volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns),
        "Sortino Ratio": sortino_ratio(returns),
        "Maximum Drawdown": maximum_drawdown(returns),
        "Calmar Ratio": calmar_ratio(returns),
        "VaR (95%)": value_at_risk(returns, 0.05),
        "Expected Shortfall": expected_shortfall(returns, 0.05)
    }
    
    # Add benchmark comparison if provided
    if benchmark_returns is not None:
        summary["Alpha"] = annualized_return(returns) - annualized_return(benchmark_returns)
        summary["Beta"] = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        
        # Information Ratio
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
        summary["Information Ratio"] = summary["Alpha"] / tracking_error if tracking_error != 0 else 0
    
    return summary

def display_performance_summary(summary: Dict[str, float], title: str = "Performance Summary") -> None:
    """Display performance summary in Streamlit"""
    st.subheader(f"📋 {title}")
    
    # Create DataFrame for better display
    df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
    
    # Format values
    percentage_metrics = ['Total Return', 'Annualized Return', 'Volatility', 'Maximum Drawdown', 'VaR (95%)', 'Expected Shortfall', 'Alpha']
    
    for idx, row in df.iterrows():
        if row['Metric'] in percentage_metrics:
            df.at[idx, 'Value'] = f"{row['Value']:.2%}"
        else:
            df.at[idx, 'Value'] = f"{row['Value']:.3f}"
    
    st.dataframe(df, use_container_width=True, hide_index=True)

# ============================================
# EXAMPLE USAGE AND TESTING
# ============================================

if __name__ == "__main__":
    st.title("📊 Metrics Demo - StockPredictionPro")
    
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Regression data
    y_true_reg = np.random.randn(n_samples)
    y_pred_reg = y_true_reg + np.random.randn(n_samples) * 0.3
    
    # Classification data
    y_true_clf = np.random.choice([0, 1], n_samples)
    y_pred_clf = np.random.choice([0, 1], n_samples)
    
    # Returns data
    returns = pd.Series(np.random.randn(252) * 0.02)  # Daily returns for 1 year
    
    # Sample trades
    trades = pd.DataFrame({
        'profit': np.random.randn(100) * 100
    })
    
    # Display tabs
    tab1, tab2, tab3 = st.tabs(["Regression Metrics", "Classification Metrics", "Trading Metrics"])
    
    with tab1:
        display_regression_metrics(y_true_reg, y_pred_reg)
    
    with tab2:
        display_classification_metrics(y_true_clf, y_pred_clf)
    
    with tab3:
        display_trading_metrics(returns, trades)
        
        # Performance summary
        summary = create_performance_summary(returns)
        display_performance_summary(summary)
