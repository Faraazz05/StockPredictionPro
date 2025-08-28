"""
scripts/models/evaluate_models.py

Comprehensive model evaluation and performance analysis for StockPredictionPro.
Evaluates trained models on multiple metrics, generates detailed reports, and compares model performance.
Supports backtesting, feature importance analysis, and prediction quality assessment.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import logging
import joblib
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Plotting configuration
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockPredictionPro.ModelEvaluation')

# Directory configuration
MODELS_DIR = Path('./models/trained')
EVALUATION_DIR = Path('./models/evaluation')
REPORTS_DIR = Path('./outputs/reports')
PLOTS_DIR = Path('./outputs/visualizations')

# Ensure directories exist
for dir_path in [EVALUATION_DIR, REPORTS_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# EVALUATION METRICS AND UTILITIES
# ============================================

@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics"""
    # Basic regression metrics
    rmse: float
    mae: float
    mape: float
    r2_score: float
    explained_variance: float
    
    # Statistical metrics
    pearson_correlation: float
    spearman_correlation: float
    
    # Distribution metrics
    prediction_std: float
    residual_mean: float
    residual_std: float
    
    # Directional accuracy
    directional_accuracy: float
    hit_rate: float
    
    # Risk metrics for financial data
    sharpe_ratio: float = None
    max_drawdown: float = None
    volatility: float = None
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for a model"""
    model_name: str
    model_path: str
    evaluation_date: str
    dataset_info: Dict[str, Any]
    metrics: ModelMetrics
    feature_importance: Dict[str, float]
    cross_validation_scores: Dict[str, List[float]]
    prediction_analysis: Dict[str, Any]
    residual_analysis: Dict[str, Any]
    plots_generated: List[str]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            f.write(self.to_json())

class MetricsCalculator:
    """Calculate comprehensive evaluation metrics"""
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate standard regression metrics"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'r2_score': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred)
        }
    
    @staticmethod
    def calculate_correlation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate correlation metrics"""
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr
        }
    
    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, 
                                     threshold: float = 0.0) -> Dict[str, float]:
        """Calculate directional prediction accuracy"""
        if len(y_true) < 2:
            return {'directional_accuracy': 0.0, 'hit_rate': 0.0}
        
        # Calculate direction changes
        true_direction = np.diff(y_true) > threshold
        pred_direction = np.diff(y_pred) > threshold
        
        directional_accuracy = np.mean(true_direction == pred_direction)
        
        # Hit rate (predictions within acceptable range)
        hit_rate = np.mean(np.abs(y_true - y_pred) / np.abs(y_true) < 0.05)  # 5% tolerance
        
        return {
            'directional_accuracy': directional_accuracy,
            'hit_rate': hit_rate
        }
    
    @staticmethod
    def calculate_residual_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate residual analysis metrics"""
        residuals = y_true - y_pred
        
        return {
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'prediction_std': np.std(y_pred)
        }
    
    @staticmethod
    def calculate_financial_metrics(returns: np.ndarray, benchmark_returns: np.ndarray = None) -> Dict[str, float]:
        """Calculate financial performance metrics"""
        if len(returns) == 0:
            return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'volatility': 0.0}
        
        # Annualized metrics (assuming daily returns)
        volatility = np.std(returns) * np.sqrt(252)
        mean_return = np.mean(returns) * 252
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility
        }

# ============================================
# VISUALIZATION UTILITIES
# ============================================

class EvaluationPlotter:
    """Generate evaluation plots and visualizations"""
    
    def __init__(self, output_dir: Path = PLOTS_DIR):
        self.output_dir = output_dir
        self.generated_plots = []
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  model_name: str, save: bool = True) -> str:
        """Plot predictions vs actual values"""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual')
        plt.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # Residuals plot
        residuals = y_true - y_pred
        plt.subplot(2, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Residuals histogram
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot for residuals
        plt.subplot(2, 2, 4)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Residuals Normality)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = self.output_dir / f'{model_name}_predictions_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.generated_plots.append(str(plot_path))
            plt.close()
            logger.info(f"📊 Saved predictions analysis plot: {plot_path}")
            return str(plot_path)
        else:
            plt.show()
            return ""
    
    def plot_time_series_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   dates: np.ndarray = None, model_name: str = "",
                                   save: bool = True) -> str:
        """Plot time series predictions"""
        if dates is None:
            dates = np.arange(len(y_true))
        
        plt.figure(figsize=(15, 10))
        
        # Full time series
        plt.subplot(3, 1, 1)
        plt.plot(dates, y_true, label='Actual', linewidth=1.5, alpha=0.8)
        plt.plot(dates, y_pred, label='Predicted', linewidth=1.5, alpha=0.8)
        plt.fill_between(dates, y_true, y_pred, alpha=0.2, color='gray')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title(f'{model_name} - Time Series Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Recent period (last 100 points)
        recent_idx = max(0, len(y_true) - 100)
        plt.subplot(3, 1, 2)
        plt.plot(dates[recent_idx:], y_true[recent_idx:], label='Actual', linewidth=2)
        plt.plot(dates[recent_idx:], y_pred[recent_idx:], label='Predicted', linewidth=2)
        plt.fill_between(dates[recent_idx:], y_true[recent_idx:], y_pred[recent_idx:], 
                        alpha=0.2, color='gray')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title('Recent Predictions (Last 100 Points)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Error over time
        errors = np.abs(y_true - y_pred)
        plt.subplot(3, 1, 3)
        plt.plot(dates, errors, label='Absolute Error', alpha=0.7)
        plt.plot(dates, pd.Series(errors).rolling(window=20).mean(), 
                label='20-period Moving Average', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Absolute Error')
        plt.title('Prediction Error Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = self.output_dir / f'{model_name}_time_series_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.generated_plots.append(str(plot_path))
            plt.close()
            logger.info(f"📊 Saved time series analysis plot: {plot_path}")
            return str(plot_path)
        else:
            plt.show()
            return ""
    
    def plot_feature_importance(self, importance_dict: Dict[str, float], 
                              model_name: str, top_n: int = 20, save: bool = True) -> str:
        """Plot feature importance"""
        if not importance_dict:
            logger.warning("No feature importance data available")
            return ""
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importance = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        
        # Horizontal bar plot
        y_pos = np.arange(len(features))
        bars = plt.barh(y_pos, importance, alpha=0.8)
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.yticks(y_pos, features)
        plt.xlabel('Importance Score')
        plt.title(f'{model_name} - Feature Importance (Top {len(features)})')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars, importance)):
            plt.text(bar.get_width() + max(importance) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{imp:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            plot_path = self.output_dir / f'{model_name}_feature_importance.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.generated_plots.append(str(plot_path))
            plt.close()
            logger.info(f"📊 Saved feature importance plot: {plot_path}")
            return str(plot_path)
        else:
            plt.show()
            return ""
    
    def plot_cross_validation_results(self, cv_scores: Dict[str, List[float]], 
                                    model_name: str, save: bool = True) -> str:
        """Plot cross-validation results"""
        plt.figure(figsize=(12, 6))
        
        metrics = list(cv_scores.keys())
        n_metrics = len(metrics)
        
        for i, (metric, scores) in enumerate(cv_scores.items()):
            plt.subplot(1, n_metrics, i + 1)
            
            # Box plot for CV scores
            plt.boxplot(scores, labels=[metric])
            plt.scatter([1] * len(scores), scores, alpha=0.6, s=30)
            
            # Add mean and std
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            plt.axhline(mean_score, color='red', linestyle='--', alpha=0.7)
            
            plt.title(f'{metric}\nMean: {mean_score:.4f}\nStd: {std_score:.4f}')
            plt.ylabel('Score')
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Cross-Validation Results', fontsize=14)
        plt.tight_layout()
        
        if save:
            plot_path = self.output_dir / f'{model_name}_cross_validation.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.generated_plots.append(str(plot_path))
            plt.close()
            logger.info(f"📊 Saved cross-validation plot: {plot_path}")
            return str(plot_path)
        else:
            plt.show()
            return ""
    
    def plot_model_comparison(self, comparison_data: Dict[str, Dict[str, float]], 
                            save: bool = True) -> str:
        """Plot comparison between multiple models"""
        if len(comparison_data) < 2:
            logger.warning("Need at least 2 models for comparison")
            return ""
        
        # Prepare data for plotting
        models = list(comparison_data.keys())
        metrics = list(comparison_data[models[0]].keys())
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data).T
        
        plt.figure(figsize=(15, 10))
        
        # Radar chart for key metrics
        key_metrics = ['rmse', 'mae', 'r2_score', 'pearson_correlation', 'directional_accuracy']
        available_metrics = [m for m in key_metrics if m in metrics]
        
        if len(available_metrics) >= 3:
            plt.subplot(2, 2, 1)
            angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for model in models:
                values = [df.loc[model, metric] for metric in available_metrics]
                values += values[:1]  # Complete the circle
                
                plt.polar(angles, values, 'o-', linewidth=2, label=model)
                plt.fill(angles, values, alpha=0.25)
            
            plt.thetagrids(np.degrees(angles[:-1]), available_metrics)
            plt.title('Model Performance Radar Chart')
            plt.legend()
        
        # Bar comparison for RMSE
        plt.subplot(2, 2, 2)
        rmse_values = [comparison_data[model].get('rmse', 0) for model in models]
        bars = plt.bar(models, rmse_values, alpha=0.8)
        plt.title('RMSE Comparison')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, rmse_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values) * 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # R² comparison
        plt.subplot(2, 2, 3)
        r2_values = [comparison_data[model].get('r2_score', 0) for model in models]
        bars = plt.bar(models, r2_values, alpha=0.8, color='green')
        plt.title('R² Score Comparison')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        
        for bar, value in zip(bars, r2_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(r2_values) * 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Metrics heatmap
        plt.subplot(2, 2, 4)
        heatmap_data = df[available_metrics].T
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='viridis', cbar=True)
        plt.title('Metrics Heatmap')
        plt.xlabel('Models')
        plt.ylabel('Metrics')
        
        plt.tight_layout()
        
        if save:
            plot_path = self.output_dir / 'model_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.generated_plots.append(str(plot_path))
            plt.close()
            logger.info(f"📊 Saved model comparison plot: {plot_path}")
            return str(plot_path)
        else:
            plt.show()
            return ""

# ============================================
# MODEL EVALUATOR
# ============================================

class ModelEvaluator:
    """Comprehensive model evaluation system"""
    
    def __init__(self, output_dir: Path = EVALUATION_DIR):
        self.output_dir = output_dir
        self.plotter = EvaluationPlotter()
        self.metrics_calculator = MetricsCalculator()
    
    def load_model_and_data(self, model_path: str, data_path: str, 
                           scaler_path: str = None) -> Tuple[Any, np.ndarray, np.ndarray, List[str]]:
        """Load model and prepare evaluation data"""
        # Load model
        model = joblib.load(model_path)
        logger.info(f"📥 Loaded model from: {model_path}")
        
        # Load scaler if provided
        scaler = None
        if scaler_path and Path(scaler_path).exists():
            scaler = joblib.load(scaler_path)
            logger.info(f"📥 Loaded scaler from: {scaler_path}")
        
        # Load and prepare data
        df = pd.read_csv(data_path)
        
        # Identify target column (assuming it's 'close' or last column)
        if 'close' in df.columns:
            target_col = 'close'
        else:
            target_col = df.columns[-1]
        
        # Prepare features and target
        exclude_cols = {target_col, 'symbol', 'timestamp', 'date'}
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Apply scaling if scaler is available
        if scaler is not None:
            X = scaler.transform(X)
            logger.info("✅ Applied feature scaling")
        
        logger.info(f"📊 Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return model, X, y, feature_cols
    
    def evaluate_model(self, model_path: str, data_path: str, 
                      scaler_path: str = None, generate_plots: bool = True) -> EvaluationReport:
        """Comprehensive model evaluation"""
        logger.info(f"🔍 Starting evaluation of model: {Path(model_path).name}")
        
        # Load model and data
        model, X, y, feature_names = self.load_model_and_data(model_path, data_path, scaler_path)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate comprehensive metrics
        metrics_dict = {}
        
        # Basic regression metrics
        regression_metrics = self.metrics_calculator.calculate_regression_metrics(y, y_pred)
        metrics_dict.update(regression_metrics)
        
        # Correlation metrics
        correlation_metrics = self.metrics_calculator.calculate_correlation_metrics(y, y_pred)
        metrics_dict.update(correlation_metrics)
        
        # Directional accuracy
        directional_metrics = self.metrics_calculator.calculate_directional_accuracy(y, y_pred)
        metrics_dict.update(directional_metrics)
        
        # Residual analysis
        residual_metrics = self.metrics_calculator.calculate_residual_metrics(y, y_pred)
        metrics_dict.update(residual_metrics)
        
        # Financial metrics (if applicable)
        try:
            returns = np.diff(y) / y[:-1]  # Simple returns
            pred_returns = np.diff(y_pred) / y_pred[:-1]
            financial_metrics = self.metrics_calculator.calculate_financial_metrics(pred_returns)
            metrics_dict.update(financial_metrics)
        except:
            logger.warning("Could not calculate financial metrics")
            financial_metrics = {'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'volatility': 0.0}
            metrics_dict.update(financial_metrics)
        
        # Create ModelMetrics object
        model_metrics = ModelMetrics(**metrics_dict)
        
        # Cross-validation analysis
        cv_scores = self._perform_cross_validation(model, X, y)
        
        # Feature importance
        feature_importance = self._extract_feature_importance(model, feature_names)
        
        # Prediction analysis
        prediction_analysis = {
            'prediction_range': {'min': float(np.min(y_pred)), 'max': float(np.max(y_pred))},
            'actual_range': {'min': float(np.min(y)), 'max': float(np.max(y))},
            'prediction_mean': float(np.mean(y_pred)),
            'actual_mean': float(np.mean(y)),
            'prediction_variance': float(np.var(y_pred)),
            'actual_variance': float(np.var(y))
        }
        
        # Residual analysis
        residuals = y - y_pred
        residual_analysis = {
            'normality_test': self._test_residual_normality(residuals),
            'autocorrelation': self._test_residual_autocorrelation(residuals),
            'heteroscedasticity': self._test_heteroscedasticity(y_pred, residuals)
        }
        
        # Generate plots
        plots_generated = []
        if generate_plots:
            model_name = Path(model_path).stem
            
            # Predictions vs actual plot
            plot_path = self.plotter.plot_predictions_vs_actual(y, y_pred, model_name)
            if plot_path:
                plots_generated.append(plot_path)
            
            # Time series plot (if data has temporal structure)
            plot_path = self.plotter.plot_time_series_predictions(y, y_pred, model_name=model_name)
            if plot_path:
                plots_generated.append(plot_path)
            
            # Feature importance plot
            if feature_importance:
                plot_path = self.plotter.plot_feature_importance(feature_importance, model_name)
                if plot_path:
                    plots_generated.append(plot_path)
            
            # Cross-validation plot
            plot_path = self.plotter.plot_cross_validation_results(cv_scores, model_name)
            if plot_path:
                plots_generated.append(plot_path)
        
        # Dataset information
        dataset_info = {
            'data_path': data_path,
            'n_samples': len(X),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'target_statistics': {
                'mean': float(np.mean(y)),
                'std': float(np.std(y)),
                'min': float(np.min(y)),
                'max': float(np.max(y))
            }
        }
        
        # Create evaluation report
        report = EvaluationReport(
            model_name=Path(model_path).stem,
            model_path=model_path,
            evaluation_date=datetime.now().isoformat(),
            dataset_info=dataset_info,
            metrics=model_metrics,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores,
            prediction_analysis=prediction_analysis,
            residual_analysis=residual_analysis,
            plots_generated=plots_generated
        )
        
        # Save report
        report_path = self.output_dir / f"{Path(model_path).stem}_evaluation_report.json"
        report.save(report_path)
        logger.info(f"📄 Saved evaluation report: {report_path}")
        
        # Print summary
        self._print_evaluation_summary(report)
        
        return report
    
    def _perform_cross_validation(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Perform cross-validation analysis"""
        logger.info("🔄 Performing cross-validation...")
        
        # Use TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=5)
        
        cv_results = {}
        
        # RMSE scores
        rmse_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_scores.append(rmse)
        
        cv_results['rmse'] = rmse_scores
        
        # R² scores
        r2_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        cv_results['r2'] = r2_scores.tolist()
        
        # MAE scores
        mae_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        cv_results['mae'] = (-mae_scores).tolist()
        
        logger.info(f"✅ Cross-validation completed: RMSE {np.mean(rmse_scores):.4f}±{np.std(rmse_scores):.4f}")
        
        return cv_results
    
    def _extract_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (XGBoost, LightGBM, RandomForest)
                importance = model.feature_importances_
                importance_dict = dict(zip(feature_names, importance))
            elif hasattr(model, 'coef_'):
                # Linear models
                coef = np.abs(model.coef_)
                if coef.ndim > 1:
                    coef = coef.flatten()
                importance_dict = dict(zip(feature_names, coef))
            else:
                logger.warning("Model does not support feature importance extraction")
        
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        return importance_dict
    
    def _test_residual_normality(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test residuals for normality"""
        try:
            from scipy.stats import shapiro, jarque_bera
            
            # Shapiro-Wilk test (for smaller samples)
            if len(residuals) <= 5000:
                stat, p_value = shapiro(residuals)
                test_name = "Shapiro-Wilk"
            else:
                # Jarque-Bera test (for larger samples)
                stat, p_value = jarque_bera(residuals)
                test_name = "Jarque-Bera"
            
            return {
                'test_name': test_name,
                'statistic': float(stat),
                'p_value': float(p_value),
                'is_normal': p_value > 0.05
            }
        except Exception as e:
            logger.warning(f"Normality test failed: {e}")
            return {'test_name': 'Failed', 'is_normal': False}
    
    def _test_residual_autocorrelation(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test residuals for autocorrelation"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            # Ljung-Box test
            result = acorr_ljungbox(residuals, lags=10, return_df=True)
            p_values = result['lb_pvalue'].values
            
            return {
                'test_name': 'Ljung-Box',
                'min_p_value': float(np.min(p_values)),
                'has_autocorrelation': np.any(p_values < 0.05)
            }
        except Exception as e:
            logger.warning(f"Autocorrelation test failed: {e}")
            return {'test_name': 'Failed', 'has_autocorrelation': False}
    
    def _test_heteroscedasticity(self, y_pred: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Test for heteroscedasticity"""
        try:
            # Breusch-Pagan test approximation
            correlation, p_value = pearsonr(np.abs(residuals), y_pred)
            
            return {
                'test_name': 'Correlation-based',
                'correlation': float(correlation),
                'p_value': float(p_value),
                'has_heteroscedasticity': p_value < 0.05
            }
        except Exception as e:
            logger.warning(f"Heteroscedasticity test failed: {e}")
            return {'test_name': 'Failed', 'has_heteroscedasticity': False}
    
    def _print_evaluation_summary(self, report: EvaluationReport) -> None:
        """Print evaluation summary to console"""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {report.model_name}")
        print(f"Evaluation Date: {report.evaluation_date}")
        print(f"Dataset: {report.dataset_info['n_samples']} samples, {report.dataset_info['n_features']} features")
        
        print(f"\nPerformance Metrics:")
        print("-" * 30)
        metrics = report.metrics
        print(f"RMSE:                    {metrics.rmse:.4f}")
        print(f"MAE:                     {metrics.mae:.4f}")
        print(f"MAPE:                    {metrics.mape:.2f}%")
        print(f"R² Score:                {metrics.r2_score:.4f}")
        print(f"Pearson Correlation:     {metrics.pearson_correlation:.4f}")
        print(f"Directional Accuracy:    {metrics.directional_accuracy:.2%}")
        
        if metrics.sharpe_ratio is not None:
            print(f"\nFinancial Metrics:")
            print("-" * 30)
            print(f"Sharpe Ratio:            {metrics.sharpe_ratio:.4f}")
            print(f"Max Drawdown:            {metrics.max_drawdown:.2%}")
            print(f"Volatility:              {metrics.volatility:.2%}")
        
        print(f"\nCross-Validation Results:")
        print("-" * 30)
        for metric, scores in report.cross_validation_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"{metric.upper():20}: {mean_score:.4f} ± {std_score:.4f}")
        
        if report.plots_generated:
            print(f"\nGenerated Plots: {len(report.plots_generated)}")
            for plot in report.plots_generated:
                print(f"  • {Path(plot).name}")

    def compare_models(self, model_reports: List[EvaluationReport]) -> Dict[str, Any]:
        """Compare multiple model evaluation reports"""
        if len(model_reports) < 2:
            raise ValueError("At least 2 models required for comparison")
        
        logger.info(f"📊 Comparing {len(model_reports)} models...")
        
        # Extract metrics for comparison
        comparison_data = {}
        for report in model_reports:
            comparison_data[report.model_name] = report.metrics.to_dict()
        
        # Generate comparison plot
        self.plotter.plot_model_comparison(comparison_data)
        
        # Determine best model for each metric
        best_models = {}
        metrics_to_minimize = {'rmse', 'mae', 'mape', 'residual_std', 'max_drawdown'}
        
        for metric in comparison_data[model_reports[0].model_name].keys():
            if metric in metrics_to_minimize:
                # Lower is better
                best_model = min(comparison_data.keys(), 
                               key=lambda x: comparison_data[x].get(metric, float('inf')))
            else:
                # Higher is better
                best_model = max(comparison_data.keys(), 
                               key=lambda x: comparison_data[x].get(metric, float('-inf')))
            
            best_models[metric] = best_model
        
        # Create comparison summary
        comparison_summary = {
            'models_compared': len(model_reports),
            'comparison_date': datetime.now().isoformat(),
            'metrics_comparison': comparison_data,
            'best_models_by_metric': best_models,
            'overall_ranking': self._rank_models(comparison_data)
        }
        
        # Save comparison report
        comparison_path = self.output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_summary, f, indent=2, default=str)
        
        logger.info(f"📄 Saved comparison report: {comparison_path}")
        
        return comparison_summary
    
    def _rank_models(self, comparison_data: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Rank models based on multiple metrics"""
        # Define weights for different metrics
        metric_weights = {
            'rmse': -0.3,  # Lower is better
            'mae': -0.2,   # Lower is better
            'r2_score': 0.3,  # Higher is better
            'pearson_correlation': 0.2,  # Higher is better
            'directional_accuracy': 0.2   # Higher is better
        }
        
        model_scores = {}
        
        for model_name, metrics in comparison_data.items():
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric, weight in metric_weights.items():
                if metric in metrics:
                    weighted_score += weight * metrics[metric]
                    total_weight += abs(weight)
            
            if total_weight > 0:
                model_scores[model_name] = weighted_score / total_weight
            else:
                model_scores[model_name] = 0.0
        
        # Rank models
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        ranking = []
        for i, (model_name, score) in enumerate(ranked_models):
            ranking.append({
                'rank': i + 1,
                'model_name': model_name,
                'weighted_score': score,
                'metrics': comparison_data[model_name]
            })
        
        return ranking

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate models for StockPredictionPro')
    parser.add_argument('--model', required=True, help='Path to trained model file')
    parser.add_argument('--data', required=True, help='Path to evaluation data CSV')
    parser.add_argument('--scaler', help='Path to scaler file (optional)')
    parser.add_argument('--output-dir', default=str(EVALUATION_DIR), help='Output directory for reports')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--compare', nargs='+', help='Additional models to compare (model paths)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(Path(args.output_dir))
    
    # Evaluate main model
    logger.info(f"🎯 Evaluating model: {args.model}")
    main_report = evaluator.evaluate_model(
        args.model, 
        args.data, 
        args.scaler,
        generate_plots=not args.no_plots
    )
    
    # Evaluate additional models for comparison
    if args.compare:
        logger.info(f"📊 Evaluating {len(args.compare)} additional models for comparison...")
        
        all_reports = [main_report]
        
        for model_path in args.compare:
            try:
                report = evaluator.evaluate_model(
                    model_path,
                    args.data,
                    args.scaler,
                    generate_plots=False  # Skip plots for comparison models
                )
                all_reports.append(report)
            except Exception as e:
                logger.error(f"Failed to evaluate {model_path}: {e}")
        
        if len(all_reports) > 1:
            comparison_summary = evaluator.compare_models(all_reports)
            
            print("\n" + "="*60)
            print("MODEL COMPARISON RESULTS")
            print("="*60)
            
            for ranking in comparison_summary['overall_ranking']:
                print(f"{ranking['rank']}. {ranking['model_name']} (Score: {ranking['weighted_score']:.4f})")
                print(f"   RMSE: {ranking['metrics'].get('rmse', 'N/A'):.4f}, "
                      f"R²: {ranking['metrics'].get('r2_score', 'N/A'):.4f}")
    
    logger.info("🎉 Model evaluation completed!")

if __name__ == '__main__':
    main()
