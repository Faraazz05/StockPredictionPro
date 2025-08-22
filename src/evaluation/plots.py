# ============================================
# StockPredictionPro - src/evaluation/plots.py
# Comprehensive visualization system for financial machine learning evaluation
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import base64
import io
from pathlib import Path

from ..utils.exceptions import ValidationError, CalculationError
from ..utils.logger import get_logger
from ..utils.timing import time_it

logger = get_logger('evaluation.plots')

# ============================================
# Plot Configuration and Enums
# ============================================

class PlotStyle(Enum):
    """Available plot styles"""
    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"
    SEABORN = "seaborn"

class PlotTheme(Enum):
    """Plot themes"""
    DEFAULT = "default"
    DARK = "dark"
    PROFESSIONAL = "professional"
    COLORFUL = "colorful"
    MINIMAL = "minimal"

@dataclass
class PlotConfig:
    """Configuration for plot generation"""
    style: PlotStyle = PlotStyle.PLOTLY
    theme: PlotTheme = PlotTheme.PROFESSIONAL
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    save_format: str = "png"
    color_palette: Optional[List[str]] = None
    font_size: int = 12
    title_size: int = 16
    show_grid: bool = True
    interactive: bool = True

# ============================================
# Base Plot Generator
# ============================================

class BasePlotGenerator:
    """
    Base class for all plot generators.
    
    This class provides common functionality for creating
    high-quality visualizations for financial ML evaluation.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        
        # Set up matplotlib style
        if self.config.style == PlotStyle.MATPLOTLIB:
            plt.rcParams.update({
                'figure.figsize': self.config.figure_size,
                'font.size': self.config.font_size,
                'axes.titlesize': self.config.title_size,
                'axes.grid': self.config.show_grid,
                'savefig.dpi': self.config.dpi,
                'savefig.format': self.config.save_format
            })
        
        # Set up seaborn style
        if self.config.style == PlotStyle.SEABORN:
            sns.set_style("whitegrid" if self.config.show_grid else "white")
            sns.set_palette(self.config.color_palette or "husl")
        
        # Color palettes
        self.colors = self._get_color_palette()
    
    def _get_color_palette(self) -> Dict[str, str]:
        """Get color palette based on theme"""
        
        if self.config.theme == PlotTheme.PROFESSIONAL:
            return {
                'primary': '#2E86AB',
                'secondary': '#A23B72', 
                'success': '#21A086',
                'warning': '#F18F01',
                'danger': '#C73E1D',
                'neutral': '#6C757D',
                'background': '#FFFFFF',
                'text': '#212529'
            }
        elif self.config.theme == PlotTheme.DARK:
            return {
                'primary': '#00D2FF',
                'secondary': '#FF0080',
                'success': '#00FF88',
                'warning': '#FFAA00',
                'danger': '#FF4444',
                'neutral': '#CCCCCC',
                'background': '#1E1E1E',
                'text': '#FFFFFF'
            }
        else:  # Default
            return {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'success': '#2ca02c',
                'warning': '#d62728',
                'danger': '#9467bd',
                'neutral': '#8c564b',
                'background': '#ffffff',
                'text': '#000000'
            }
    
    def save_plot(self, fig, filename: str, output_dir: str = "plots") -> str:
        """Save plot to file"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        full_path = output_path / filename
        
        if self.config.style == PlotStyle.PLOTLY:
            fig.write_image(str(full_path))
        else:
            fig.savefig(str(full_path), dpi=self.config.dpi, bbox_inches='tight')
        
        logger.info(f"Plot saved to: {full_path}")
        return str(full_path)
    
    def _apply_plotly_theme(self, fig):
        """Apply theme to Plotly figure"""
        
        if self.config.theme == PlotTheme.PROFESSIONAL:
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=self.config.font_size, color=self.colors['text']),
                title_font_size=self.config.title_size,
                showlegend=True,
                legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1)
            )
        elif self.config.theme == PlotTheme.DARK:
            fig.update_layout(
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(family="Arial, sans-serif", size=self.config.font_size, color=self.colors['text']),
                title_font_size=self.config.title_size,
                showlegend=True
            )
        
        return fig

# ============================================
# Model Performance Plots
# ============================================

class ModelPerformancePlots(BasePlotGenerator):
    """
    Generates plots for machine learning model performance evaluation.
    
    This class creates visualizations for model metrics, confusion matrices,
    feature importance, learning curves, and prediction analysis.
    """
    
    @time_it("confusion_matrix_plot")
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "Confusion Matrix") -> Union[plt.Figure, go.Figure]:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            class_names: Names of classes
            title: Plot title
            
        Returns:
            Figure object
        """
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=class_names,
                y=class_names,
                color_continuous_scale='Blues',
                title=title
            )
            
            # Add text annotations
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    fig.add_annotation(
                        x=j, y=i,
                        text=str(cm[i, j]),
                        showarrow=False,
                        font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                    )
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
            
            ax.set_title(title, fontsize=self.config.title_size)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            return fig
    
    @time_it("roc_curve_plot")
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                      title: str = "ROC Curve") -> Union[plt.Figure, go.Figure]:
        """
        Plot ROC curve
        
        Args:
            y_true: True binary labels
            y_scores: Target scores (probabilities)
            title: Plot title
            
        Returns:
            Figure object
        """
        
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color=self.colors['primary'], width=2)
            ))
            
            # Add diagonal reference line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color=self.colors['neutral'], width=1, dash='dash')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            ax.plot(fpr, tpr, color=self.colors['primary'], linewidth=2,
                   label=f'ROC Curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color=self.colors['neutral'], linestyle='--',
                   label='Random Classifier')
            
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(title, fontsize=self.config.title_size)
            ax.legend()
            ax.grid(self.config.show_grid)
            
            return fig
    
    @time_it("feature_importance_plot")
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray,
                              title: str = "Feature Importance", top_n: int = 20) -> Union[plt.Figure, go.Figure]:
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores for each feature
            title: Plot title
            top_n: Number of top features to show
            
        Returns:
            Figure object
        """
        
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1][:top_n]
        sorted_features = [feature_names[i] for i in indices]
        sorted_scores = importance_scores[indices]
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = go.Figure(go.Bar(
                x=sorted_scores,
                y=sorted_features,
                orientation='h',
                marker_color=self.colors['primary']
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=max(400, len(sorted_features) * 20)
            )
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, ax = plt.subplots(figsize=(self.config.figure_size[0], max(6, len(sorted_features) * 0.3)))
            
            ax.barh(range(len(sorted_features)), sorted_scores, color=self.colors['primary'])
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel('Importance Score')
            ax.set_title(title, fontsize=self.config.title_size)
            ax.grid(self.config.show_grid, axis='x')
            
            return fig
    
    @time_it("learning_curve_plot")
    def plot_learning_curve(self, train_sizes: np.ndarray, train_scores: np.ndarray,
                           validation_scores: np.ndarray, title: str = "Learning Curve") -> Union[plt.Figure, go.Figure]:
        """
        Plot learning curve
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores for each size
            validation_scores: Validation scores for each size
            title: Plot title
            
        Returns:
            Figure object
        """
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=np.mean(train_scores, axis=1),
                mode='lines+markers',
                name='Training Score',
                line=dict(color=self.colors['primary']),
                error_y=dict(
                    type='data',
                    array=np.std(train_scores, axis=1),
                    visible=True
                )
            ))
            
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=np.mean(validation_scores, axis=1),
                mode='lines+markers', 
                name='Validation Score',
                line=dict(color=self.colors['warning']),
                error_y=dict(
                    type='data',
                    array=np.std(validation_scores, axis=1),
                    visible=True
                )
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Training Set Size',
                yaxis_title='Score'
            )
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(validation_scores, axis=1)
            val_std = np.std(validation_scores, axis=1)
            
            ax.plot(train_sizes, train_mean, 'o-', color=self.colors['primary'], 
                   label='Training Score')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                           alpha=0.2, color=self.colors['primary'])
            
            ax.plot(train_sizes, val_mean, 'o-', color=self.colors['warning'], 
                   label='Validation Score')
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                           alpha=0.2, color=self.colors['warning'])
            
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('Score')
            ax.set_title(title, fontsize=self.config.title_size)
            ax.legend()
            ax.grid(self.config.show_grid)
            
            return fig

# ============================================
# Backtesting Performance Plots
# ============================================

class BacktestingPlots(BasePlotGenerator):
    """
    Generates plots for backtesting and trading strategy evaluation.
    
    This class creates visualizations for equity curves, drawdowns,
    returns analysis, and trading performance metrics.
    """
    
    @time_it("equity_curve_plot")
    def plot_equity_curve(self, dates: pd.DatetimeIndex, portfolio_values: np.ndarray,
                         benchmark_values: Optional[np.ndarray] = None,
                         title: str = "Portfolio Equity Curve") -> Union[plt.Figure, go.Figure]:
        """
        Plot portfolio equity curve
        
        Args:
            dates: Date index
            portfolio_values: Portfolio values over time
            benchmark_values: Optional benchmark values
            title: Plot title
            
        Returns:
            Figure object
        """
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio_values,
                mode='lines',
                name='Portfolio',
                line=dict(color=self.colors['primary'], width=2)
            ))
            
            if benchmark_values is not None:
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=benchmark_values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.colors['neutral'], width=1, dash='dash')
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Portfolio Value',
                hovermode='x unified'
            )
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            ax.plot(dates, portfolio_values, color=self.colors['primary'], 
                   linewidth=2, label='Portfolio')
            
            if benchmark_values is not None:
                ax.plot(dates, benchmark_values, color=self.colors['neutral'], 
                       linestyle='--', label='Benchmark')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value')
            ax.set_title(title, fontsize=self.config.title_size)
            ax.legend()
            ax.grid(self.config.show_grid)
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            return fig
    
    @time_it("drawdown_plot")
    def plot_drawdown(self, dates: pd.DatetimeIndex, drawdowns: np.ndarray,
                     title: str = "Portfolio Drawdown") -> Union[plt.Figure, go.Figure]:
        """
        Plot portfolio drawdown
        
        Args:
            dates: Date index
            drawdowns: Drawdown values over time
            title: Plot title
            
        Returns:
            Figure object
        """
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=drawdowns,
                fill='tonexty',
                mode='lines',
                name='Drawdown',
                line=dict(color=self.colors['danger']),
                fillcolor=f"rgba{tuple(list(plt.colors.to_rgba(self.colors['danger']))[:3] + [0.3])}"
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color=self.colors['neutral'])
            
            fig.update_layout(
                title=title,
                xaxis_title='Date', 
                yaxis_title='Drawdown (%)',
                hovermode='x unified'
            )
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            ax.fill_between(dates, drawdowns, 0, color=self.colors['danger'], alpha=0.3)
            ax.plot(dates, drawdowns, color=self.colors['danger'], linewidth=1)
            ax.axhline(y=0, color=self.colors['neutral'], linestyle='--', alpha=0.7)
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.set_title(title, fontsize=self.config.title_size)
            ax.grid(self.config.show_grid)
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            return fig
    
    @time_it("returns_distribution_plot")
    def plot_returns_distribution(self, returns: np.ndarray, 
                                 title: str = "Returns Distribution") -> Union[plt.Figure, go.Figure]:
        """
        Plot returns distribution
        
        Args:
            returns: Array of returns
            title: Plot title
            
        Returns:
            Figure object
        """
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
                marker_color=self.colors['primary'],
                opacity=0.7
            ))
            
            # Add normal distribution overlay
            x_norm = np.linspace(returns.min(), returns.max(), 100)
            y_norm = len(returns) * np.diff(x_norm)[0] * stats.norm.pdf(x_norm, returns.mean(), returns.std())
            
            fig.add_trace(go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name='Normal Fit',
                line=dict(color=self.colors['warning'], width=2)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Returns',
                yaxis_title='Frequency'
            )
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            ax.hist(returns, bins=50, alpha=0.7, color=self.colors['primary'], 
                   density=True, label='Returns')
            
            # Add normal distribution overlay
            x_norm = np.linspace(returns.min(), returns.max(), 100)
            y_norm = stats.norm.pdf(x_norm, returns.mean(), returns.std())
            ax.plot(x_norm, y_norm, color=self.colors['warning'], linewidth=2, 
                   label='Normal Fit')
            
            ax.set_xlabel('Returns')
            ax.set_ylabel('Density')
            ax.set_title(title, fontsize=self.config.title_size)
            ax.legend()
            ax.grid(self.config.show_grid)
            
            return fig
    
    @time_it("rolling_metrics_plot")
    def plot_rolling_metrics(self, dates: pd.DatetimeIndex, 
                            metrics_dict: Dict[str, np.ndarray],
                            title: str = "Rolling Performance Metrics") -> Union[plt.Figure, go.Figure]:
        """
        Plot rolling performance metrics
        
        Args:
            dates: Date index
            metrics_dict: Dictionary of metric_name -> values
            title: Plot title
            
        Returns:
            Figure object
        """
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = make_subplots(
                rows=len(metrics_dict), cols=1,
                shared_xaxes=True,
                subplot_titles=list(metrics_dict.keys()),
                vertical_spacing=0.05
            )
            
            colors = [self.colors['primary'], self.colors['secondary'], 
                     self.colors['success'], self.colors['warning']]
            
            for i, (metric_name, values) in enumerate(metrics_dict.items()):
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines',
                        name=metric_name,
                        line=dict(color=colors[i % len(colors)])
                    ),
                    row=i+1, col=1
                )
            
            fig.update_layout(title=title, height=200 * len(metrics_dict))
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, axes = plt.subplots(len(metrics_dict), 1, figsize=(self.config.figure_size[0], 
                                                                   self.config.figure_size[1] * len(metrics_dict) / 2),
                                   sharex=True)
            
            if len(metrics_dict) == 1:
                axes = [axes]
            
            colors = [self.colors['primary'], self.colors['secondary'], 
                     self.colors['success'], self.colors['warning']]
            
            for i, (metric_name, values) in enumerate(metrics_dict.items()):
                axes[i].plot(dates, values, color=colors[i % len(colors)], linewidth=2)
                axes[i].set_ylabel(metric_name)
                axes[i].grid(self.config.show_grid)
            
            axes[-1].set_xlabel('Date')
            fig.suptitle(title, fontsize=self.config.title_size)
            plt.tight_layout()
            
            return fig
    
    @time_it("monthly_returns_heatmap")
    def plot_monthly_returns_heatmap(self, monthly_returns_df: pd.DataFrame,
                                   title: str = "Monthly Returns Heatmap") -> Union[plt.Figure, go.Figure]:
        """
        Plot monthly returns heatmap
        
        Args:
            monthly_returns_df: DataFrame with years as index and months as columns
            title: Plot title
            
        Returns:
            Figure object
        """
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = go.Figure(data=go.Heatmap(
                z=monthly_returns_df.values,
                x=monthly_returns_df.columns,
                y=monthly_returns_df.index,
                colorscale='RdYlGn',
                zmid=0,
                text=monthly_returns_df.values,
                texttemplate="%{text:.2%}",
                textfont={"size": 10},
                colorbar=dict(title="Returns")
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Month',
                yaxis_title='Year'
            )
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            sns.heatmap(monthly_returns_df, annot=True, fmt='.2%', cmap='RdYlGn', 
                       center=0, cbar_kws={'format': '%.1%'}, ax=ax)
            
            ax.set_title(title, fontsize=self.config.title_size)
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            
            return fig

# ============================================
# Strategy Comparison Plots
# ============================================

class StrategyComparisonPlots(BasePlotGenerator):
    """
    Generates plots for comparing multiple trading strategies.
    
    This class creates visualizations for strategy performance comparison,
    risk-return scatter plots, and relative performance analysis.
    """
    
    @time_it("strategy_performance_comparison")
    def plot_strategy_comparison(self, strategy_results: Dict[str, Dict[str, float]],
                               title: str = "Strategy Performance Comparison") -> Union[plt.Figure, go.Figure]:
        """
        Plot strategy performance comparison
        
        Args:
            strategy_results: Dictionary of strategy_name -> metrics
            title: Plot title
            
        Returns:
            Figure object
        """
        
        strategies = list(strategy_results.keys())
        metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Volatility']
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=metrics,
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            colors = px.colors.qualitative.Set1[:len(strategies)]
            
            for i, metric in enumerate(metrics):
                row, col = (i // 2) + 1, (i % 2) + 1
                values = [strategy_results[strategy].get(metric, 0) for strategy in strategies]
                
                fig.add_trace(
                    go.Bar(x=strategies, y=values, name=metric, 
                          marker_color=colors, showlegend=False),
                    row=row, col=col
                )
            
            fig.update_layout(title=title, height=600)
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
            axes = axes.ravel()
            
            for i, metric in enumerate(metrics):
                values = [strategy_results[strategy].get(metric, 0) for strategy in strategies]
                
                axes[i].bar(strategies, values, color=self.colors['primary'])
                axes[i].set_title(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(self.config.show_grid, axis='y')
            
            fig.suptitle(title, fontsize=self.config.title_size)
            plt.tight_layout()
            
            return fig
    
    @time_it("risk_return_scatter")
    def plot_risk_return_scatter(self, strategy_data: Dict[str, Tuple[float, float]],
                               title: str = "Risk-Return Analysis") -> Union[plt.Figure, go.Figure]:
        """
        Plot risk-return scatter plot
        
        Args:
            strategy_data: Dictionary of strategy_name -> (risk, return)
            title: Plot title
            
        Returns:
            Figure object  
        """
        
        strategies = list(strategy_data.keys())
        risks = [strategy_data[strategy][0] for strategy in strategies]
        returns = [strategy_data[strategy][1] for strategy in strategies]
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=risks,
                y=returns,
                mode='markers+text',
                text=strategies,
                textposition="top center",
                marker=dict(
                    size=10,
                    color=self.colors['primary'],
                    line=dict(width=2, color=self.colors['text'])
                )
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Risk (Volatility)',
                yaxis_title='Return'
            )
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            ax.scatter(risks, returns, s=100, color=self.colors['primary'], 
                      alpha=0.7, edgecolors=self.colors['text'])
            
            for i, strategy in enumerate(strategies):
                ax.annotate(strategy, (risks[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Risk (Volatility)')
            ax.set_ylabel('Return')
            ax.set_title(title, fontsize=self.config.title_size)
            ax.grid(self.config.show_grid)
            
            return fig

# ============================================
# Portfolio Analysis Plots
# ============================================

class PortfolioAnalysisPlots(BasePlotGenerator):
    """
    Generates plots for portfolio analysis and optimization.
    
    This class creates visualizations for asset allocation,
    correlation analysis, and portfolio optimization results.
    """
    
    @time_it("asset_allocation_plot")
    def plot_asset_allocation(self, allocations: Dict[str, float],
                            title: str = "Portfolio Asset Allocation") -> Union[plt.Figure, go.Figure]:
        """
        Plot portfolio asset allocation
        
        Args:
            allocations: Dictionary of asset -> weight
            title: Plot title
            
        Returns:
            Figure object
        """
        
        assets = list(allocations.keys())
        weights = list(allocations.values())
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = go.Figure(data=[go.Pie(
                labels=assets,
                values=weights,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(title=title)
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            wedges, texts, autotexts = ax.pie(weights, labels=assets, autopct='%1.1f%%',
                                             startangle=90)
            
            ax.set_title(title, fontsize=self.config.title_size)
            
            return fig
    
    @time_it("correlation_heatmap")
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                               title: str = "Asset Correlation Matrix") -> Union[plt.Figure, go.Figure]:
        """
        Plot correlation heatmap
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Plot title
            
        Returns:
            Figure object
        """
        
        if self.config.style == PlotStyle.PLOTLY:
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.values,
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Assets',
                yaxis_title='Assets'
            )
            
            return self._apply_plotly_theme(fig)
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=ax)
            
            ax.set_title(title, fontsize=self.config.title_size)
            
            return fig

# ============================================
# Interactive Dashboard
# ============================================

class InteractiveDashboard:
    """
    Creates interactive dashboards combining multiple plots.
    
    This class generates comprehensive dashboards for model evaluation,
    backtesting analysis, and portfolio management.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.model_plots = ModelPerformancePlots(config)
        self.backtest_plots = BacktestingPlots(config)
        self.strategy_plots = StrategyComparisonPlots(config)
        self.portfolio_plots = PortfolioAnalysisPlots(config)
    
    @time_it("model_evaluation_dashboard")
    def create_model_evaluation_dashboard(self, model_results: Dict[str, Any]) -> go.Figure:
        """
        Create comprehensive model evaluation dashboard
        
        Args:
            model_results: Dictionary containing model evaluation results
            
        Returns:
            Plotly figure with multiple subplots
        """
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['ROC Curve', 'Feature Importance', 'Confusion Matrix', 'Learning Curve'],
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Add ROC curve
        if 'roc_data' in model_results:
            roc_data = model_results['roc_data']
            fig.add_trace(
                go.Scatter(x=roc_data['fpr'], y=roc_data['tpr'],
                          mode='lines', name='ROC Curve'),
                row=1, col=1
            )
        
        # Add feature importance
        if 'feature_importance' in model_results:
            fi_data = model_results['feature_importance']
            fig.add_trace(
                go.Bar(x=fi_data['scores'][:10], y=fi_data['features'][:10],
                      orientation='h', name='Top Features'),
                row=1, col=2
            )
        
        fig.update_layout(
            title="Model Evaluation Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    @time_it("backtesting_dashboard")
    def create_backtesting_dashboard(self, backtest_results: Dict[str, Any]) -> go.Figure:
        """
        Create comprehensive backtesting dashboard
        
        Args:
            backtest_results: Dictionary containing backtest results
            
        Returns:
            Plotly figure with multiple subplots
        """
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Equity Curve', 'Drawdown', 'Returns Distribution', 
                           'Monthly Returns', 'Rolling Sharpe', 'Trade Analysis'],
            specs=[[{"secondary_y": True}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Extract data
        dates = backtest_results.get('dates', [])
        equity_curve = backtest_results.get('equity_curve', [])
        
        # Add equity curve
        if dates and equity_curve:
            fig.add_trace(
                go.Scatter(x=dates, y=equity_curve, mode='lines', name='Portfolio'),
                row=1, col=1
            )
        
        fig.update_layout(
            title="Backtesting Performance Dashboard",
            height=1200,
            showlegend=True
        )
        
        return fig

# ============================================
# Utility Functions
# ============================================

def create_model_plots(model_results: Dict[str, Any], 
                      config: Optional[PlotConfig] = None) -> Dict[str, Union[plt.Figure, go.Figure]]:
    """Quick utility to create model performance plots"""
    
    plotter = ModelPerformancePlots(config)
    plots = {}
    
    if 'confusion_matrix' in model_results:
        cm_data = model_results['confusion_matrix']
        plots['confusion_matrix'] = plotter.plot_confusion_matrix(
            cm_data['y_true'], cm_data['y_pred'], cm_data.get('class_names')
        )
    
    if 'roc_data' in model_results:
        roc_data = model_results['roc_data']
        plots['roc_curve'] = plotter.plot_roc_curve(
            roc_data['y_true'], roc_data['y_scores']
        )
    
    if 'feature_importance' in model_results:
        fi_data = model_results['feature_importance']
        plots['feature_importance'] = plotter.plot_feature_importance(
            fi_data['features'], fi_data['scores']
        )
    
    return plots

def create_backtesting_plots(backtest_results: Dict[str, Any],
                           config: Optional[PlotConfig] = None) -> Dict[str, Union[plt.Figure, go.Figure]]:
    """Quick utility to create backtesting plots"""
    
    plotter = BacktestingPlots(config)
    plots = {}
    
    dates = backtest_results.get('dates')
    equity_curve = backtest_results.get('equity_curve')
    
    if dates is not None and equity_curve is not None:
        plots['equity_curve'] = plotter.plot_equity_curve(dates, equity_curve)
        
        # Calculate drawdowns
        equity_series = pd.Series(equity_curve, index=dates)
        running_max = equity_series.expanding().max()
        drawdowns = (equity_series - running_max) / running_max
        
        plots['drawdown'] = plotter.plot_drawdown(dates, drawdowns.values)
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        plots['returns_distribution'] = plotter.plot_returns_distribution(returns.values)
    
    return plots

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Evaluation Plots System")
    
    # Generate sample data for testing
    np.random.seed(42)
    
    # Sample model results
    sample_model_results = {
        'confusion_matrix': {
            'y_true': np.random.choice([0, 1], 1000),
            'y_pred': np.random.choice([0, 1], 1000),
            'class_names': ['Down', 'Up']
        },
        'roc_data': {
            'y_true': np.random.choice([0, 1], 1000),
            'y_scores': np.random.random(1000)
        },
        'feature_importance': {
            'features': [f'Feature_{i}' for i in range(20)],
            'scores': np.random.random(20)
        }
    }
    
    # Sample backtesting results
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns = np.random.normal(0.001, 0.02, 252)
    equity_curve = 1000000 * (1 + returns).cumprod()
    
    sample_backtest_results = {
        'dates': dates,
        'equity_curve': equity_curve
    }
    
    print("\n1. Testing Model Performance Plots")
    
    model_plots = create_model_plots(sample_model_results)
    print(f"Created {len(model_plots)} model plots:")
    for plot_name in model_plots.keys():
        print(f"  - {plot_name}")
    
    print("\n2. Testing Backtesting Plots")
    
    backtest_plots = create_backtesting_plots(sample_backtest_results)
    print(f"Created {len(backtest_plots)} backtesting plots:")
    for plot_name in backtest_plots.keys():
        print(f"  - {plot_name}")
    
    print("\n3. Testing Individual Plot Generators")
    
    # Test model performance plots
    model_plotter = ModelPerformancePlots()
    
    # Test confusion matrix
    cm_fig = model_plotter.plot_confusion_matrix(
        sample_model_results['confusion_matrix']['y_true'],
        sample_model_results['confusion_matrix']['y_pred']
    )
    print("✓ Confusion matrix plot created")
    
    # Test ROC curve
    roc_fig = model_plotter.plot_roc_curve(
        sample_model_results['roc_data']['y_true'],
        sample_model_results['roc_data']['y_scores']
    )
    print("✓ ROC curve plot created")
    
    # Test feature importance
    fi_fig = model_plotter.plot_feature_importance(
        sample_model_results['feature_importance']['features'],
        sample_model_results['feature_importance']['scores']
    )
    print("✓ Feature importance plot created")
    
    print("\n4. Testing Backtesting Plot Generators")
    
    backtest_plotter = BacktestingPlots()
    
    # Test equity curve
    equity_fig = backtest_plotter.plot_equity_curve(
        sample_backtest_results['dates'],
        sample_backtest_results['equity_curve']
    )
    print("✓ Equity curve plot created")
    
    # Test drawdown plot
    equity_series = pd.Series(equity_curve, index=dates)
    running_max = equity_series.expanding().max()
    drawdowns = (equity_series - running_max) / running_max
    
    dd_fig = backtest_plotter.plot_drawdown(dates, drawdowns.values)
    print("✓ Drawdown plot created")
    
    # Test returns distribution
    returns_vals = equity_series.pct_change().dropna().values
    returns_fig = backtest_plotter.plot_returns_distribution(returns_vals)
    print("✓ Returns distribution plot created")
    
    print("\n5. Testing Strategy Comparison Plots")
    
    strategy_plotter = StrategyComparisonPlots()
    
    # Sample strategy comparison data
    strategy_results = {
        'Strategy A': {'Total Return': 0.15, 'Sharpe Ratio': 1.2, 'Max Drawdown': 0.08, 'Volatility': 0.18},
        'Strategy B': {'Total Return': 0.12, 'Sharpe Ratio': 1.1, 'Max Drawdown': 0.06, 'Volatility': 0.15},
        'Strategy C': {'Total Return': 0.18, 'Sharpe Ratio': 0.9, 'Max Drawdown': 0.12, 'Volatility': 0.22}
    }
    
    comparison_fig = strategy_plotter.plot_strategy_comparison(strategy_results)
    print("✓ Strategy comparison plot created")
    
    # Test risk-return scatter
    risk_return_data = {
        'Strategy A': (0.18, 0.15),
        'Strategy B': (0.15, 0.12), 
        'Strategy C': (0.22, 0.18)
    }
    
    scatter_fig = strategy_plotter.plot_risk_return_scatter(risk_return_data)
    print("✓ Risk-return scatter plot created")
    
    print("\n6. Testing Portfolio Analysis Plots")
    
    portfolio_plotter = PortfolioAnalysisPlots()
    
    # Test asset allocation
    allocations = {'AAPL': 0.3, 'MSFT': 0.25, 'GOOGL': 0.2, 'TSLA': 0.15, 'BND': 0.1}
    allocation_fig = portfolio_plotter.plot_asset_allocation(allocations)
    print("✓ Asset allocation plot created")
    
    # Test correlation heatmap
    assets = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BND']
    corr_matrix = pd.DataFrame(
        np.random.uniform(0.1, 0.9, (5, 5)), 
        index=assets, 
        columns=assets
    )
    np.fill_diagonal(corr_matrix.values, 1.0)
    
    corr_fig = portfolio_plotter.plot_correlation_heatmap(corr_matrix)
    print("✓ Correlation heatmap created")
    
    print("\n7. Testing Interactive Dashboard")
    
    dashboard = InteractiveDashboard()
    
    # Create model dashboard
    model_dashboard = dashboard.create_model_evaluation_dashboard(sample_model_results)
    print("✓ Model evaluation dashboard created")
    
    # Create backtesting dashboard  
    backtest_dashboard = dashboard.create_backtesting_dashboard(sample_backtest_results)
    print("✓ Backtesting dashboard created")
    
    print("\n8. Testing Different Plot Configurations")
    
    # Test different themes
    themes = [PlotTheme.PROFESSIONAL, PlotTheme.DARK, PlotTheme.DEFAULT]
    
    for theme in themes:
        config = PlotConfig(theme=theme, style=PlotStyle.PLOTLY)
        themed_plotter = ModelPerformancePlots(config)
        themed_fig = themed_plotter.plot_confusion_matrix(
            sample_model_results['confusion_matrix']['y_true'][:100],
            sample_model_results['confusion_matrix']['y_pred'][:100]
        )
        print(f"✓ {theme.value} theme plot created")
    
    print("\nEvaluation plots system testing completed successfully!")
    print("\nGenerated visualizations include:")
    print("• Model Performance: ROC curves, confusion matrices, feature importance, learning curves")
    print("• Backtesting Analysis: Equity curves, drawdowns, returns distribution, rolling metrics")
    print("• Strategy Comparison: Performance comparison, risk-return analysis")
    print("• Portfolio Analysis: Asset allocation, correlation matrices, optimization results")
    print("• Interactive Dashboards: Multi-plot comprehensive analysis views")
    print("• Multiple Themes: Professional, dark, and default styling options")
    print("• Export Capabilities: PNG, HTML, PDF formats with high DPI")
