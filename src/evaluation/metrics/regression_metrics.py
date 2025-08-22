# ============================================
# StockPredictionPro - src/evaluation/metrics/regression_metrics.py
# Comprehensive regression metrics for financial machine learning models
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
from datetime import datetime
from scipy import stats
import math

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('evaluation.metrics.regression')

# ============================================
# Core Regression Metrics
# ============================================

@time_it("mse_calculation")
def calculate_mse(y_true: Union[np.ndarray, pd.Series], 
                 y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Squared Error (MSE)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MSE value
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align lengths
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    mse = np.mean((y_true_clean - y_pred_clean) ** 2)
    
    return float(mse)


@time_it("rmse_calculation")
def calculate_rmse(y_true: Union[np.ndarray, pd.Series], 
                  y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Root Mean Squared Error (RMSE)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    
    mse = calculate_mse(y_true, y_pred)
    return float(np.sqrt(mse))


@time_it("mae_calculation")
def calculate_mae(y_true: Union[np.ndarray, pd.Series], 
                 y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Error (MAE)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align lengths
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    
    return float(mae)


@time_it("r2_calculation")
def calculate_r2_score(y_true: Union[np.ndarray, pd.Series], 
                      y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate R-squared (coefficient of determination)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        R² score
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align lengths
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate R²
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)  # Total sum of squares
    
    if ss_tot == 0:
        return 0.0  # Perfect prediction when variance is 0
    
    r2 = 1 - (ss_res / ss_tot)
    
    return float(r2)


@time_it("adjusted_r2_calculation")
def calculate_adjusted_r2(y_true: Union[np.ndarray, pd.Series], 
                         y_pred: Union[np.ndarray, pd.Series],
                         n_features: int) -> float:
    """
    Calculate Adjusted R-squared
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        n_features: Number of features used in the model
        
    Returns:
        Adjusted R² score
    """
    
    r2 = calculate_r2_score(y_true, y_pred)
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    n_samples = np.sum(mask)
    
    if n_samples <= n_features + 1:
        return 0.0
    
    # Adjusted R² formula
    adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
    
    return float(adjusted_r2)

# ============================================
# Financial-Specific Regression Metrics
# ============================================

@time_it("directional_accuracy_calculation")
def calculate_directional_accuracy(y_true: Union[np.ndarray, pd.Series], 
                                  y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Directional Accuracy - percentage of correct direction predictions
    Important for financial applications where direction matters more than magnitude
    
    Args:
        y_true: Actual values (typically returns)
        y_pred: Predicted values (typically returns)
        
    Returns:
        Directional accuracy as percentage (0 to 1)
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align lengths
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate directional accuracy
    true_direction = np.sign(y_true_clean)
    pred_direction = np.sign(y_pred_clean)
    
    correct_directions = (true_direction == pred_direction)
    directional_accuracy = np.mean(correct_directions)
    
    return float(directional_accuracy)


@time_it("hit_rate_calculation")
def calculate_hit_rate(y_true: Union[np.ndarray, pd.Series], 
                      y_pred: Union[np.ndarray, pd.Series],
                      threshold: float = 0.0) -> float:
    """
    Calculate Hit Rate - percentage of predictions above threshold when actual is above threshold
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        threshold: Threshold for determining "hits"
        
    Returns:
        Hit rate as percentage (0 to 1)
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate hit rate
    true_hits = y_true_clean > threshold
    pred_hits = y_pred_clean > threshold
    
    if np.sum(true_hits) == 0:
        return 0.0  # No actual hits to measure
    
    # Hit rate = correct positive predictions / total actual positives
    hit_rate = np.sum(true_hits & pred_hits) / np.sum(true_hits)
    
    return float(hit_rate)


@time_it("prediction_profit_calculation")
def calculate_prediction_profit(y_true: Union[np.ndarray, pd.Series], 
                               y_pred: Union[np.ndarray, pd.Series],
                               transaction_cost: float = 0.001) -> float:
    """
    Calculate theoretical profit from predictions
    Assumes long when prediction > 0, short when prediction < 0
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        transaction_cost: Cost per transaction (default 0.1%)
        
    Returns:
        Total profit/loss from predictions
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Trading signals based on predictions
    positions = np.sign(y_pred_clean)  # 1 for long, -1 for short, 0 for neutral
    
    # Calculate returns from positions
    strategy_returns = positions * y_true_clean
    
    # Calculate position changes for transaction costs
    position_changes = np.diff(positions, prepend=0)  # Changes in position
    n_transactions = np.sum(np.abs(position_changes) > 0)
    
    # Total profit minus transaction costs
    gross_profit = np.sum(strategy_returns)
    net_profit = gross_profit - (n_transactions * transaction_cost)
    
    return float(net_profit)

# ============================================
# Advanced Regression Metrics
# ============================================

@time_it("mape_calculation")
def calculate_mape(y_true: Union[np.ndarray, pd.Series], 
                  y_pred: Union[np.ndarray, pd.Series],
                  epsilon: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE as percentage
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Avoid division by zero
    y_true_safe = np.where(np.abs(y_true_clean) < epsilon, epsilon, y_true_clean)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_safe)) * 100
    
    return float(mape)


@time_it("smape_calculation")
def calculate_smape(y_true: Union[np.ndarray, pd.Series], 
                   y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        SMAPE as percentage
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate SMAPE
    numerator = np.abs(y_true_clean - y_pred_clean)
    denominator = (np.abs(y_true_clean) + np.abs(y_pred_clean)) / 2
    
    # Avoid division by zero
    mask_nonzero = denominator > 0
    if not np.any(mask_nonzero):
        return 0.0
    
    smape = np.mean(numerator[mask_nonzero] / denominator[mask_nonzero]) * 100
    
    return float(smape)


@time_it("rmsle_calculation")
def calculate_rmsle(y_true: Union[np.ndarray, pd.Series], 
                   y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Root Mean Squared Logarithmic Error (RMSLE)
    Useful when you care about relative differences
    
    Args:
        y_true: Actual values (must be positive)
        y_pred: Predicted values (must be positive)
        
    Returns:
        RMSLE value
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Ensure positive values for logarithm
    y_true_positive = np.maximum(y_true_clean, 1e-8)
    y_pred_positive = np.maximum(y_pred_clean, 1e-8)
    
    # Calculate RMSLE
    log_diff = np.log1p(y_true_positive) - np.log1p(y_pred_positive)
    rmsle = np.sqrt(np.mean(log_diff ** 2))
    
    return float(rmsle)


@time_it("max_error_calculation")
def calculate_max_error(y_true: Union[np.ndarray, pd.Series], 
                       y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Maximum Error - worst case prediction error
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Maximum absolute error
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate maximum error
    errors = np.abs(y_true_clean - y_pred_clean)
    max_error = np.max(errors)
    
    return float(max_error)

# ============================================
# Time Series Specific Metrics
# ============================================

@time_it("temporal_correlation_calculation")
def calculate_temporal_correlation(y_true: Union[np.ndarray, pd.Series], 
                                  y_pred: Union[np.ndarray, pd.Series],
                                  lag: int = 1) -> float:
    """
    Calculate temporal correlation between predictions and actuals
    Measures how well the model captures temporal patterns
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        lag: Lag for correlation calculation
        
    Returns:
        Temporal correlation coefficient
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length <= lag:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) <= lag:
        return 0.0
    
    # Calculate lagged correlation
    y_true_lagged = y_true_clean[:-lag]
    y_pred_current = y_pred_clean[lag:]
    
    if len(y_true_lagged) == 0 or len(y_pred_current) == 0:
        return 0.0
    
    correlation = np.corrcoef(y_true_lagged, y_pred_current)[0, 1]
    
    if np.isnan(correlation):
        return 0.0
    
    return float(correlation)


@time_it("forecast_bias_calculation")
def calculate_forecast_bias(y_true: Union[np.ndarray, pd.Series], 
                           y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Forecast Bias - systematic over/under prediction
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Mean forecast error (positive = over-prediction)
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate bias (mean error)
    bias = np.mean(y_pred_clean - y_true_clean)
    
    return float(bias)

# ============================================
# Comprehensive Regression Metrics Calculator
# ============================================

class RegressionMetricsCalculator:
    """
    Comprehensive calculator for regression performance metrics
    """
    
    def __init__(self, n_features: Optional[int] = None):
        self.n_features = n_features
    
    @time_it("comprehensive_regression_metrics")
    def calculate_all_metrics(self, 
                             y_true: Union[np.ndarray, pd.Series],
                             y_pred: Union[np.ndarray, pd.Series],
                             n_features: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate comprehensive set of regression metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            n_features: Number of features (for adjusted R²)
            
        Returns:
            Dictionary of all calculated metrics
        """
        
        metrics = {}
        
        try:
            # Use provided n_features or class default
            n_feat = n_features or self.n_features or 1
            
            # Basic regression metrics
            metrics['mse'] = calculate_mse(y_true, y_pred)
            metrics['rmse'] = calculate_rmse(y_true, y_pred)
            metrics['mae'] = calculate_mae(y_true, y_pred)
            metrics['r2_score'] = calculate_r2_score(y_true, y_pred)
            metrics['adjusted_r2'] = calculate_adjusted_r2(y_true, y_pred, n_feat)
            
            # Financial-specific metrics
            metrics['directional_accuracy'] = calculate_directional_accuracy(y_true, y_pred)
            metrics['hit_rate'] = calculate_hit_rate(y_true, y_pred)
            metrics['prediction_profit'] = calculate_prediction_profit(y_true, y_pred)
            
            # Advanced regression metrics
            metrics['mape'] = calculate_mape(y_true, y_pred)
            metrics['smape'] = calculate_smape(y_true, y_pred)
            metrics['max_error'] = calculate_max_error(y_true, y_pred)
            
            # Time series metrics
            metrics['temporal_correlation'] = calculate_temporal_correlation(y_true, y_pred)
            metrics['forecast_bias'] = calculate_forecast_bias(y_true, y_pred)
            
            # Additional statistics
            y_true_array = np.asarray(y_true)
            y_pred_array = np.asarray(y_pred)
            
            min_length = min(len(y_true_array), len(y_pred_array))
            if min_length > 0:
                y_true_aligned = y_true_array[-min_length:]
                y_pred_aligned = y_pred_array[-min_length:]
                
                mask = ~(np.isnan(y_true_aligned) | np.isnan(y_pred_aligned))
                y_true_clean = y_true_aligned[mask]
                y_pred_clean = y_pred_aligned[mask]
                
                if len(y_true_clean) > 0:
                    # Correlation coefficient
                    correlation = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
                    metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
                    
                    # Explained variance
                    explained_var = 1 - (np.var(y_true_clean - y_pred_clean) / np.var(y_true_clean))
                    metrics['explained_variance'] = explained_var if not np.isnan(explained_var) else 0.0
                    
                    # Number of samples
                    metrics['n_samples'] = len(y_true_clean)
        
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def format_metrics_report(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics into a readable report
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Formatted report string
        """
        
        report = []
        report.append("=" * 50)
        report.append("REGRESSION METRICS REPORT")
        report.append("=" * 50)
        
        # Basic metrics
        if 'mse' in metrics:
            report.append(f"Mean Squared Error (MSE): {metrics['mse']:.6f}")
        if 'rmse' in metrics:
            report.append(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
        if 'mae' in metrics:
            report.append(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}")
        
        report.append("")
        
        # R-squared metrics
        if 'r2_score' in metrics:
            report.append(f"R-squared (R²): {metrics['r2_score']:.4f}")
        if 'adjusted_r2' in metrics:
            report.append(f"Adjusted R²: {metrics['adjusted_r2']:.4f}")
        if 'explained_variance' in metrics:
            report.append(f"Explained Variance: {metrics['explained_variance']:.4f}")
        
        report.append("")
        
        # Financial metrics
        if 'directional_accuracy' in metrics:
            report.append(f"Directional Accuracy: {metrics['directional_accuracy']:.1%}")
        if 'hit_rate' in metrics:
            report.append(f"Hit Rate: {metrics['hit_rate']:.1%}")
        if 'prediction_profit' in metrics:
            report.append(f"Prediction Profit: {metrics['prediction_profit']:.4f}")
        
        report.append("")
        
        # Advanced metrics
        if 'mape' in metrics:
            report.append(f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
        if 'smape' in metrics:
            report.append(f"Symmetric MAPE: {metrics['smape']:.2f}%")
        if 'max_error' in metrics:
            report.append(f"Maximum Error: {metrics['max_error']:.6f}")
        
        # Time series metrics
        if 'temporal_correlation' in metrics:
            report.append(f"Temporal Correlation: {metrics['temporal_correlation']:.4f}")
        if 'forecast_bias' in metrics:
            report.append(f"Forecast Bias: {metrics['forecast_bias']:.6f}")
        
        # Sample information
        if 'n_samples' in metrics:
            report.append("")
            report.append(f"Number of Samples: {metrics['n_samples']}")
        
        return "\n".join(report)

# ============================================
# Utility Functions
# ============================================

def quick_regression_evaluation(y_true: Union[np.ndarray, pd.Series],
                               y_pred: Union[np.ndarray, pd.Series],
                               n_features: Optional[int] = None) -> str:
    """
    Quick utility function to get regression evaluation summary
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        n_features: Number of features used
        
    Returns:
        Formatted evaluation summary
    """
    
    calculator = RegressionMetricsCalculator(n_features)
    metrics = calculator.calculate_all_metrics(y_true, y_pred, n_features)
    return calculator.format_metrics_report(metrics)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Regression Metrics")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic actual values (e.g., stock returns)
    y_true = np.random.normal(0.001, 0.02, n_samples)  # Daily returns
    
    # Create predictions with some noise and bias
    noise = np.random.normal(0, 0.005, n_samples)
    bias = 0.0002  # Slight positive bias
    correlation_factor = 0.7
    
    y_pred = correlation_factor * y_true + noise + bias
    
    print("Sample Regression Metrics Calculation:")
    print("=" * 50)
    
    # Test individual metrics
    print(f"MSE: {calculate_mse(y_true, y_pred):.8f}")
    print(f"RMSE: {calculate_rmse(y_true, y_pred):.8f}")
    print(f"MAE: {calculate_mae(y_true, y_pred):.8f}")
    print(f"R² Score: {calculate_r2_score(y_true, y_pred):.4f}")
    print(f"Directional Accuracy: {calculate_directional_accuracy(y_true, y_pred):.1%}")
    print(f"Hit Rate: {calculate_hit_rate(y_true, y_pred):.1%}")
    print(f"MAPE: {calculate_mape(y_true, y_pred):.2f}%")
    print(f"Forecast Bias: {calculate_forecast_bias(y_true, y_pred):.6f}")
    
    # Test comprehensive calculator
    print("\n" + "=" * 50)
    print("COMPREHENSIVE REGRESSION EVALUATION:")
    
    summary = quick_regression_evaluation(y_true, y_pred, n_features=5)
    print(summary)
    
    # Test with different scenarios
    print("\n" + "=" * 30)
    print("PERFECT PREDICTION SCENARIO:")
    
    perfect_summary = quick_regression_evaluation(y_true, y_true)
    print(perfect_summary)
    
    print("\nRegression metrics testing completed successfully!")
