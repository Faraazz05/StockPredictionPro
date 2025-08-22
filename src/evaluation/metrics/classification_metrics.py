# ============================================
# StockPredictionPro - src/evaluation/metrics/classification_metrics.py
# Comprehensive classification metrics for financial machine learning models
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

logger = get_logger('evaluation.metrics.classification')

# ============================================
# Core Classification Metrics
# ============================================

@time_it("accuracy_calculation")
def calculate_accuracy(y_true: Union[np.ndarray, pd.Series], 
                      y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Accuracy - fraction of correct predictions
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        
    Returns:
        Accuracy score (0 to 1)
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
    mask = ~(pd.isna(y_true_aligned) | pd.isna(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate accuracy
    accuracy = np.mean(y_true_clean == y_pred_clean)
    
    return float(accuracy)


@time_it("precision_calculation")
def calculate_precision(y_true: Union[np.ndarray, pd.Series], 
                       y_pred: Union[np.ndarray, pd.Series],
                       positive_class: Union[int, str] = 1) -> float:
    """
    Calculate Precision - TP / (TP + FP)
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        positive_class: Label of positive class
        
    Returns:
        Precision score (0 to 1)
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(pd.isna(y_true_aligned) | pd.isna(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate precision for positive class
    tp = np.sum((y_true_clean == positive_class) & (y_pred_clean == positive_class))
    fp = np.sum((y_true_clean != positive_class) & (y_pred_clean == positive_class))
    
    if tp + fp == 0:
        return 0.0  # No positive predictions
    
    precision = tp / (tp + fp)
    
    return float(precision)


@time_it("recall_calculation")
def calculate_recall(y_true: Union[np.ndarray, pd.Series], 
                    y_pred: Union[np.ndarray, pd.Series],
                    positive_class: Union[int, str] = 1) -> float:
    """
    Calculate Recall (Sensitivity) - TP / (TP + FN)
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        positive_class: Label of positive class
        
    Returns:
        Recall score (0 to 1)
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(pd.isna(y_true_aligned) | pd.isna(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate recall for positive class
    tp = np.sum((y_true_clean == positive_class) & (y_pred_clean == positive_class))
    fn = np.sum((y_true_clean == positive_class) & (y_pred_clean != positive_class))
    
    if tp + fn == 0:
        return 0.0  # No actual positives
    
    recall = tp / (tp + fn)
    
    return float(recall)


@time_it("f1_calculation")
def calculate_f1_score(y_true: Union[np.ndarray, pd.Series], 
                      y_pred: Union[np.ndarray, pd.Series],
                      positive_class: Union[int, str] = 1) -> float:
    """
    Calculate F1 Score - harmonic mean of precision and recall
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        positive_class: Label of positive class
        
    Returns:
        F1 score (0 to 1)
    """
    
    precision = calculate_precision(y_true, y_pred, positive_class)
    recall = calculate_recall(y_true, y_pred, positive_class)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return float(f1)


@time_it("specificity_calculation")
def calculate_specificity(y_true: Union[np.ndarray, pd.Series], 
                         y_pred: Union[np.ndarray, pd.Series],
                         positive_class: Union[int, str] = 1) -> float:
    """
    Calculate Specificity (True Negative Rate) - TN / (TN + FP)
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        positive_class: Label of positive class
        
    Returns:
        Specificity score (0 to 1)
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(pd.isna(y_true_aligned) | pd.isna(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate specificity
    tn = np.sum((y_true_clean != positive_class) & (y_pred_clean != positive_class))
    fp = np.sum((y_true_clean != positive_class) & (y_pred_clean == positive_class))
    
    if tn + fp == 0:
        return 0.0  # No actual negatives
    
    specificity = tn / (tn + fp)
    
    return float(specificity)

# ============================================
# Financial-Specific Classification Metrics
# ============================================

@time_it("trading_accuracy_calculation")
def calculate_trading_accuracy(y_true: Union[np.ndarray, pd.Series], 
                              y_pred: Union[np.ndarray, pd.Series],
                              returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, float]:
    """
    Calculate Trading-Specific Accuracy Metrics
    
    Args:
        y_true: True direction labels (e.g., 0=down, 1=up)
        y_pred: Predicted direction labels
        returns: Actual returns (optional, for profit calculation)
        
    Returns:
        Dictionary with trading accuracy metrics
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return {'trading_accuracy': 0.0, 'long_accuracy': 0.0, 'short_accuracy': 0.0}
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    if returns is not None:
        returns_array = np.asarray(returns)
        returns_aligned = returns_array[-min_length:]
    else:
        returns_aligned = None
    
    # Remove NaN values
    if returns_aligned is not None:
        mask = ~(pd.isna(y_true_aligned) | pd.isna(y_pred_aligned) | pd.isna(returns_aligned))
        returns_clean = returns_aligned[mask]
    else:
        mask = ~(pd.isna(y_true_aligned) | pd.isna(y_pred_aligned))
        returns_clean = None
    
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return {'trading_accuracy': 0.0, 'long_accuracy': 0.0, 'short_accuracy': 0.0}
    
    # Overall trading accuracy
    trading_accuracy = calculate_accuracy(y_true_clean, y_pred_clean)
    
    # Long position accuracy (when predicted up)
    long_mask = y_pred_clean == 1
    if np.sum(long_mask) > 0:
        long_accuracy = np.mean(y_true_clean[long_mask] == y_pred_clean[long_mask])
    else:
        long_accuracy = 0.0
    
    # Short position accuracy (when predicted down)
    short_mask = y_pred_clean == 0
    if np.sum(short_mask) > 0:
        short_accuracy = np.mean(y_true_clean[short_mask] == y_pred_clean[short_mask])
    else:
        short_accuracy = 0.0
    
    result = {
        'trading_accuracy': float(trading_accuracy),
        'long_accuracy': float(long_accuracy),
        'short_accuracy': float(short_accuracy)
    }
    
    # Add profit-based metrics if returns provided
    if returns_clean is not None:
        # Calculate profit from predictions
        position_returns = np.where(y_pred_clean == 1, returns_clean, -returns_clean)
        total_profit = np.sum(position_returns)
        avg_profit_per_trade = np.mean(position_returns)
        
        # Win rate (profitable trades)
        win_rate = np.mean(position_returns > 0)
        
        result.update({
            'total_profit': float(total_profit),
            'avg_profit_per_trade': float(avg_profit_per_trade),
            'win_rate': float(win_rate)
        })
    
    return result


@time_it("regime_classification_metrics")
def calculate_regime_classification_metrics(y_true: Union[np.ndarray, pd.Series], 
                                          y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate metrics specific to market regime classification
    
    Args:
        y_true: True regime labels
        y_pred: Predicted regime labels
        
    Returns:
        Dictionary with regime classification metrics
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return {'regime_accuracy': 0.0, 'regime_consistency': 0.0}
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(pd.isna(y_true_aligned) | pd.isna(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return {'regime_accuracy': 0.0, 'regime_consistency': 0.0}
    
    # Overall regime accuracy
    regime_accuracy = calculate_accuracy(y_true_clean, y_pred_clean)
    
    # Regime consistency (how often predictions don't change)
    if len(y_pred_clean) > 1:
        pred_changes = np.sum(np.diff(y_pred_clean) != 0)
        regime_consistency = 1 - (pred_changes / (len(y_pred_clean) - 1))
    else:
        regime_consistency = 1.0
    
    # Calculate per-regime metrics
    unique_regimes = np.unique(y_true_clean)
    regime_metrics = {}
    
    for regime in unique_regimes:
        regime_mask = y_true_clean == regime
        if np.sum(regime_mask) > 0:
            regime_precision = calculate_precision(y_true_clean, y_pred_clean, regime)
            regime_recall = calculate_recall(y_true_clean, y_pred_clean, regime)
            regime_metrics[f'regime_{regime}_precision'] = regime_precision
            regime_metrics[f'regime_{regime}_recall'] = regime_recall
    
    result = {
        'regime_accuracy': float(regime_accuracy),
        'regime_consistency': float(regime_consistency)
    }
    result.update(regime_metrics)
    
    return result

# ============================================
# Advanced Classification Metrics
# ============================================

@time_it("confusion_matrix_calculation")
def calculate_confusion_matrix(y_true: Union[np.ndarray, pd.Series], 
                              y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
    """
    Calculate Confusion Matrix and derived metrics
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        
    Returns:
        Dictionary with confusion matrix and derived metrics
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return {'confusion_matrix': np.array([[0]]), 'classes': []}
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(pd.isna(y_true_aligned) | pd.isna(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return {'confusion_matrix': np.array([[0]]), 'classes': []}
    
    # Get unique classes
    classes = np.unique(np.concatenate([y_true_clean, y_pred_clean]))
    n_classes = len(classes)
    
    # Create confusion matrix
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            confusion_matrix[i, j] = np.sum(
                (y_true_clean == true_class) & (y_pred_clean == pred_class)
            )
    
    # Calculate per-class metrics
    per_class_metrics = {}
    
    for i, class_label in enumerate(classes):
        tp = confusion_matrix[i, i]
        fn = np.sum(confusion_matrix[i, :]) - tp
        fp = np.sum(confusion_matrix[:, i]) - tp
        tn = np.sum(confusion_matrix) - tp - fn - fp
        
        # Per-class precision, recall, f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[f'class_{class_label}_precision'] = precision
        per_class_metrics[f'class_{class_label}_recall'] = recall
        per_class_metrics[f'class_{class_label}_f1'] = f1
    
    return {
        'confusion_matrix': confusion_matrix,
        'classes': classes.tolist(),
        'per_class_metrics': per_class_metrics
    }


@time_it("balanced_accuracy_calculation")
def calculate_balanced_accuracy(y_true: Union[np.ndarray, pd.Series], 
                               y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Balanced Accuracy - average of per-class recalls
    Better for imbalanced datasets
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        
    Returns:
        Balanced accuracy score (0 to 1)
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(pd.isna(y_true_aligned) | pd.isna(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate balanced accuracy
    classes = np.unique(y_true_clean)
    recalls = []
    
    for class_label in classes:
        recall = calculate_recall(y_true_clean, y_pred_clean, class_label)
        recalls.append(recall)
    
    balanced_accuracy = np.mean(recalls)
    
    return float(balanced_accuracy)


@time_it("matthews_correlation_calculation")
def calculate_matthews_correlation(y_true: Union[np.ndarray, pd.Series], 
                                  y_pred: Union[np.ndarray, pd.Series],
                                  positive_class: Union[int, str] = 1) -> float:
    """
    Calculate Matthews Correlation Coefficient (MCC)
    Good metric for imbalanced datasets
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        positive_class: Label of positive class
        
    Returns:
        MCC score (-1 to 1)
    """
    
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_pred_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_pred_aligned = y_pred_array[-min_length:]
    
    mask = ~(pd.isna(y_true_aligned) | pd.isna(y_pred_aligned))
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate confusion matrix elements
    tp = np.sum((y_true_clean == positive_class) & (y_pred_clean == positive_class))
    tn = np.sum((y_true_clean != positive_class) & (y_pred_clean != positive_class))
    fp = np.sum((y_true_clean != positive_class) & (y_pred_clean == positive_class))
    fn = np.sum((y_true_clean == positive_class) & (y_pred_clean != positive_class))
    
    # Calculate MCC
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    
    mcc = numerator / denominator
    
    return float(mcc)

# ============================================
# Probability-Based Metrics (for probabilistic predictions)
# ============================================

@time_it("log_loss_calculation")
def calculate_log_loss(y_true: Union[np.ndarray, pd.Series], 
                      y_prob: Union[np.ndarray, pd.Series],
                      epsilon: float = 1e-15) -> float:
    """
    Calculate Logarithmic Loss (Cross-entropy loss)
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        epsilon: Small value to avoid log(0)
        
    Returns:
        Log loss value (lower is better)
    """
    
    y_true_array = np.asarray(y_true)
    y_prob_array = np.asarray(y_prob)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_prob_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_prob_aligned = y_prob_array[-min_length:]
    
    mask = ~(pd.isna(y_true_aligned) | pd.isna(y_prob_aligned))
    y_true_clean = y_true_aligned[mask]
    y_prob_clean = y_prob_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Clip probabilities to avoid log(0)
    y_prob_clipped = np.clip(y_prob_clean, epsilon, 1 - epsilon)
    
    # Calculate log loss
    log_loss = -np.mean(
        y_true_clean * np.log(y_prob_clipped) + 
        (1 - y_true_clean) * np.log(1 - y_prob_clipped)
    )
    
    return float(log_loss)


@time_it("brier_score_calculation")
def calculate_brier_score(y_true: Union[np.ndarray, pd.Series], 
                         y_prob: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Brier Score - mean squared difference between probabilities and outcomes
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        
    Returns:
        Brier score (lower is better, range 0 to 1)
    """
    
    y_true_array = np.asarray(y_true)
    y_prob_array = np.asarray(y_prob)
    
    # Align and clean data
    min_length = min(len(y_true_array), len(y_prob_array))
    if min_length == 0:
        return 0.0
    
    y_true_aligned = y_true_array[-min_length:]
    y_prob_aligned = y_prob_array[-min_length:]
    
    mask = ~(pd.isna(y_true_aligned) | pd.isna(y_prob_aligned))
    y_true_clean = y_true_aligned[mask]
    y_prob_clean = y_prob_aligned[mask]
    
    if len(y_true_clean) == 0:
        return 0.0
    
    # Calculate Brier score
    brier_score = np.mean((y_prob_clean - y_true_clean) ** 2)
    
    return float(brier_score)

# ============================================
# Comprehensive Classification Metrics Calculator
# ============================================

class ClassificationMetricsCalculator:
    """
    Comprehensive calculator for classification performance metrics
    """
    
    def __init__(self, positive_class: Union[int, str] = 1):
        self.positive_class = positive_class
    
    @time_it("comprehensive_classification_metrics")
    def calculate_all_metrics(self, 
                             y_true: Union[np.ndarray, pd.Series],
                             y_pred: Union[np.ndarray, pd.Series],
                             y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
                             returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive set of classification metrics
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            y_prob: Predicted probabilities (optional)
            returns: Actual returns for trading metrics (optional)
            
        Returns:
            Dictionary of all calculated metrics
        """
        
        metrics = {}
        
        try:
            # Basic classification metrics
            metrics['accuracy'] = calculate_accuracy(y_true, y_pred)
            metrics['precision'] = calculate_precision(y_true, y_pred, self.positive_class)
            metrics['recall'] = calculate_recall(y_true, y_pred, self.positive_class)
            metrics['f1_score'] = calculate_f1_score(y_true, y_pred, self.positive_class)
            metrics['specificity'] = calculate_specificity(y_true, y_pred, self.positive_class)
            
            # Advanced metrics
            metrics['balanced_accuracy'] = calculate_balanced_accuracy(y_true, y_pred)
            metrics['matthews_correlation'] = calculate_matthews_correlation(y_true, y_pred, self.positive_class)
            
            # Confusion matrix and per-class metrics
            cm_result = calculate_confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm_result['confusion_matrix'].tolist()
            metrics['classes'] = cm_result['classes']
            metrics.update(cm_result['per_class_metrics'])
            
            # Financial-specific metrics
            trading_metrics = calculate_trading_accuracy(y_true, y_pred, returns)
            metrics.update({f'trading_{k}': v for k, v in trading_metrics.items()})
            
            # Regime classification metrics if multi-class
            unique_classes = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
            if len(unique_classes) > 2:
                regime_metrics = calculate_regime_classification_metrics(y_true, y_pred)
                metrics.update(regime_metrics)
            
            # Probability-based metrics if probabilities provided
            if y_prob is not None:
                metrics['log_loss'] = calculate_log_loss(y_true, y_prob)
                metrics['brier_score'] = calculate_brier_score(y_true, y_prob)
            
            # Additional statistics
            y_true_array = np.asarray(y_true)
            y_pred_array = np.asarray(y_pred)
            
            min_length = min(len(y_true_array), len(y_pred_array))
            if min_length > 0:
                y_true_aligned = y_true_array[-min_length:]
                y_pred_aligned = y_pred_array[-min_length:]
                
                mask = ~(pd.isna(y_true_aligned) | pd.isna(y_pred_aligned))
                y_true_clean = y_true_aligned[mask]
                y_pred_clean = y_pred_aligned[mask]
                
                if len(y_true_clean) > 0:
                    # Class distribution
                    unique_true, counts_true = np.unique(y_true_clean, return_counts=True)
                    class_distribution = dict(zip(unique_true, counts_true))
                    metrics['class_distribution'] = class_distribution
                    
                    # Number of samples
                    metrics['n_samples'] = len(y_true_clean)
                    
                    # Class balance ratio (for binary classification)
                    if len(unique_true) == 2:
                        minority_count = np.min(counts_true)
                        majority_count = np.max(counts_true)
                        metrics['class_balance_ratio'] = minority_count / majority_count
        
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def format_metrics_report(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics into a readable report
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Formatted report string
        """
        
        report = []
        report.append("=" * 50)
        report.append("CLASSIFICATION METRICS REPORT")
        report.append("=" * 50)
        
        # Basic metrics
        if 'accuracy' in metrics:
            report.append(f"Accuracy: {metrics['accuracy']:.3f}")
        if 'balanced_accuracy' in metrics:
            report.append(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        
        report.append("")
        
        # Precision, Recall, F1
        if 'precision' in metrics:
            report.append(f"Precision: {metrics['precision']:.3f}")
        if 'recall' in metrics:
            report.append(f"Recall (Sensitivity): {metrics['recall']:.3f}")
        if 'specificity' in metrics:
            report.append(f"Specificity: {metrics['specificity']:.3f}")
        if 'f1_score' in metrics:
            report.append(f"F1 Score: {metrics['f1_score']:.3f}")
        
        report.append("")
        
        # Advanced metrics
        if 'matthews_correlation' in metrics:
            report.append(f"Matthews Correlation: {metrics['matthews_correlation']:.3f}")
        
        # Trading metrics
        if 'trading_trading_accuracy' in metrics:
            report.append("")
            report.append("Trading Metrics:")
            report.append(f"  Trading Accuracy: {metrics['trading_trading_accuracy']:.3f}")
            
            if 'trading_long_accuracy' in metrics:
                report.append(f"  Long Accuracy: {metrics['trading_long_accuracy']:.3f}")
            if 'trading_short_accuracy' in metrics:
                report.append(f"  Short Accuracy: {metrics['trading_short_accuracy']:.3f}")
            if 'trading_win_rate' in metrics:
                report.append(f"  Win Rate: {metrics['trading_win_rate']:.1%}")
            if 'trading_total_profit' in metrics:
                report.append(f"  Total Profit: {metrics['trading_total_profit']:.4f}")
        
        # Probability metrics
        if 'log_loss' in metrics:
            report.append("")
            report.append(f"Log Loss: {metrics['log_loss']:.4f}")
        if 'brier_score' in metrics:
            report.append(f"Brier Score: {metrics['brier_score']:.4f}")
        
        # Class distribution
        if 'class_distribution' in metrics:
            report.append("")
            report.append("Class Distribution:")
            for class_label, count in metrics['class_distribution'].items():
                percentage = count / metrics.get('n_samples', 1) * 100
                report.append(f"  Class {class_label}: {count} ({percentage:.1f}%)")
        
        # Sample information
        if 'n_samples' in metrics:
            report.append("")
            report.append(f"Number of Samples: {metrics['n_samples']}")
        
        if 'class_balance_ratio' in metrics:
            report.append(f"Class Balance Ratio: {metrics['class_balance_ratio']:.3f}")
        
        return "\n".join(report)

# ============================================
# Utility Functions
# ============================================

def quick_classification_evaluation(y_true: Union[np.ndarray, pd.Series],
                                   y_pred: Union[np.ndarray, pd.Series],
                                   y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
                                   returns: Optional[Union[np.ndarray, pd.Series]] = None) -> str:
    """
    Quick utility function to get classification evaluation summary
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        y_prob: Predicted probabilities (optional)
        returns: Actual returns for trading metrics (optional)
        
    Returns:
        Formatted evaluation summary
    """
    
    calculator = ClassificationMetricsCalculator()
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_prob, returns)
    return calculator.format_metrics_report(metrics)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Classification Metrics")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic classification data
    # True labels (0 = down, 1 = up)
    y_true = np.random.binomial(1, 0.52, n_samples)  # Slightly more ups than downs
    
    # Predictions with some accuracy
    accuracy_rate = 0.65
    y_pred = y_true.copy()
    
    # Add some errors
    error_indices = np.random.choice(n_samples, size=int(n_samples * (1 - accuracy_rate)), replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]  # Flip labels
    
    # Generate probabilities
    y_prob = np.where(y_pred == 1, 
                     np.random.uniform(0.6, 0.9, n_samples),  # Higher prob for predicted ups
                     np.random.uniform(0.1, 0.4, n_samples))  # Lower prob for predicted downs
    
    # Generate returns
    returns = np.where(y_true == 1, 
                      np.random.lognormal(-0.001, 0.02, n_samples),  # Positive returns
                      -np.random.lognormal(-0.001, 0.02, n_samples))  # Negative returns
    
    print("Sample Classification Metrics Calculation:")
    print("=" * 50)
    
    # Test individual metrics
    print(f"Accuracy: {calculate_accuracy(y_true, y_pred):.3f}")
    print(f"Precision: {calculate_precision(y_true, y_pred):.3f}")
    print(f"Recall: {calculate_recall(y_true, y_pred):.3f}")
    print(f"F1 Score: {calculate_f1_score(y_true, y_pred):.3f}")
    print(f"Balanced Accuracy: {calculate_balanced_accuracy(y_true, y_pred):.3f}")
    print(f"Matthews Correlation: {calculate_matthews_correlation(y_true, y_pred):.3f}")
    print(f"Log Loss: {calculate_log_loss(y_true, y_prob):.4f}")
    
    # Test trading metrics
    trading_metrics = calculate_trading_accuracy(y_true, y_pred, returns)
    print(f"Trading Accuracy: {trading_metrics['trading_accuracy']:.3f}")
    if 'win_rate' in trading_metrics:
        print(f"Win Rate: {trading_metrics['win_rate']:.1%}")
    
    # Test comprehensive calculator
    print("\n" + "=" * 50)
    print("COMPREHENSIVE CLASSIFICATION EVALUATION:")
    
    summary = quick_classification_evaluation(y_true, y_pred, y_prob, returns)
    print(summary)
    
    # Test multi-class scenario
    print("\n" + "=" * 30)
    print("MULTI-CLASS SCENARIO:")
    
    # Generate 3-class data (bear, sideways, bull market)
    y_true_multi = np.random.choice([0, 1, 2], size=500, p=[0.3, 0.4, 0.3])
    y_pred_multi = y_true_multi.copy()
    
    # Add errors
    error_indices = np.random.choice(500, size=150, replace=False)
    y_pred_multi[error_indices] = np.random.choice([0, 1, 2], size=150)
    
    regime_metrics = calculate_regime_classification_metrics(y_true_multi, y_pred_multi)
    print("Regime Classification Metrics:")
    for key, value in regime_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
    
    print("\nClassification metrics testing completed successfully!")
