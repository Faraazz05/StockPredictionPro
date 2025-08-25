"""
plotting-helper.py

Advanced plotting utilities for StockPredictionPro notebooks.
Includes standardized financial charts, technical analysis plots, model evaluation visualizations,
and professional dashboard layouts.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ============================================
# GLOBAL PLOTTING CONFIGURATION
# ============================================

class PlottingConfig:
    """Global configuration for consistent plotting styles"""
    
    # Color schemes
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff9800',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40'
    }
    
    FINANCIAL_COLORS = {
        'bullish': '#26a69a',
        'bearish': '#ef5350',
        'volume': '#78909c',
        'ma_short': '#ff9800',
        'ma_long': '#9c27b0',
        'support': '#4caf50',
        'resistance': '#f44336'
    }
    
    # Default figure settings
    FIGURE_SIZE = (14, 8)
    DPI = 100
    STYLE = 'seaborn-v0_8-darkgrid'
    FONT_SIZE = 10
    TITLE_SIZE = 14
    LABEL_SIZE = 12

# Initialize plotting environment
def setup_plotting_style():
    """Configure matplotlib and seaborn for consistent styling"""
    plt.style.use('default')  # Reset to default first
    sns.set_theme(style='darkgrid', palette='deep', font_scale=1.0)
    
    plt.rcParams.update({
        'figure.figsize': PlottingConfig.FIGURE_SIZE,
        'figure.dpi': PlottingConfig.DPI,
        'font.size': PlottingConfig.FONT_SIZE,
        'axes.titlesize': PlottingConfig.TITLE_SIZE,
        'axes.labelsize': PlottingConfig.LABEL_SIZE,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'grid.alpha': 0.3,
        'figure.autolayout': True
    })

# Initialize on import
setup_plotting_style()

# ============================================
# BASIC PLOTTING UTILITIES
# ============================================

def create_figure(figsize: Tuple[int, int] = None, title: str = None, 
                 tight_layout: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a standardized figure with consistent styling
    
    Args:
        figsize: Figure size tuple (width, height)
        title: Figure title
        tight_layout: Whether to use tight layout
        
    Returns:
        Tuple of (figure, axes)
    """
    figsize = figsize or PlottingConfig.FIGURE_SIZE
    fig, ax = plt.subplots(figsize=figsize)
    
    if title:
        fig.suptitle(title, fontsize=PlottingConfig.TITLE_SIZE, fontweight='bold')
    
    if tight_layout:
        plt.tight_layout()
    
    return fig, ax

def plot_line_chart(x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray],
                   title: str = '', xlabel: str = '', ylabel: str = '',
                   color: str = None, style: str = '-', linewidth: float = 2,
                   alpha: float = 1.0, label: str = None, figsize: Tuple = None,
                   grid: bool = True, show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a professional line chart
    
    Args:
        x, y: Data for x and y axes
        title: Chart title
        xlabel, ylabel: Axis labels
        color: Line color
        style: Line style ('-', '--', ':', etc.)
        linewidth: Line width
        alpha: Transparency
        label: Legend label
        figsize: Figure size
        grid: Show grid
        show: Whether to display the plot
        
    Returns:
        Figure and axes objects
    """
    fig, ax = create_figure(figsize, title)
    
    color = color or PlottingConfig.COLORS['primary']
    
    ax.plot(x, y, color=color, linestyle=style, linewidth=linewidth, 
           alpha=alpha, label=label)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if label:
        ax.legend()
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return fig, ax

def plot_multiple_series(data: Dict[str, pd.Series], title: str = '',
                        xlabel: str = '', ylabel: str = '', colors: List[str] = None,
                        styles: List[str] = None, figsize: Tuple = None,
                        grid: bool = True, show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple time series on the same chart
    
    Args:
        data: Dictionary of {label: series} pairs
        title: Chart title
        xlabel, ylabel: Axis labels
        colors: List of colors for each series
        styles: List of line styles
        figsize: Figure size
        grid: Show grid
        show: Whether to display the plot
        
    Returns:
        Figure and axes objects
    """
    fig, ax = create_figure(figsize, title)
    
    colors = colors or [PlottingConfig.COLORS['primary'], PlottingConfig.COLORS['secondary'],
                       PlottingConfig.COLORS['success'], PlottingConfig.COLORS['danger']]
    styles = styles or ['-'] * len(data)
    
    for i, (label, series) in enumerate(data.items()):
        color = colors[i % len(colors)]
        style = styles[i % len(styles)]
        
        ax.plot(series.index, series.values, label=label, color=color, 
               linestyle=style, linewidth=2, alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return fig, ax

# ============================================
# FINANCIAL CHART UTILITIES
# ============================================

def plot_candlestick_chart(df: pd.DataFrame, title: str = 'Candlestick Chart',
                          volume: bool = True, ma_periods: List[int] = None,
                          figsize: Tuple = None, show: bool = True) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create professional candlestick chart with volume and moving averages
    
    Args:
        df: DataFrame with OHLCV data
        title: Chart title
        volume: Whether to show volume subplot
        ma_periods: List of moving average periods to plot
        figsize: Figure size
        show: Whether to display the plot
        
    Returns:
        Figure and list of axes objects
    """
    figsize = figsize or (16, 10)
    
    if volume:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        axes = [ax1]
    
    fig.suptitle(title, fontsize=PlottingConfig.TITLE_SIZE, fontweight='bold')
    
    # Candlestick plot
    for i in range(len(df)):
        row = df.iloc[i]
        date = i  # Use integer index for simplicity
        
        open_price, close_price = row['open'], row['close']
        high_price, low_price = row['high'], row['low']
        
        # Determine color
        color = PlottingConfig.FINANCIAL_COLORS['bullish'] if close_price >= open_price else PlottingConfig.FINANCIAL_COLORS['bearish']
        
        # Draw high-low line
        ax1.plot([date, date], [low_price, high_price], color='black', linewidth=1)
        
        # Draw candlestick body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        ax1.bar(date, body_height, bottom=body_bottom, width=0.6, 
               color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add moving averages
    if ma_periods:
        colors = [PlottingConfig.FINANCIAL_COLORS['ma_short'], PlottingConfig.FINANCIAL_COLORS['ma_long']]
        for i, period in enumerate(ma_periods):
            ma = df['close'].rolling(window=period).mean()
            ax1.plot(range(len(ma)), ma, label=f'MA-{period}', 
                    color=colors[i % len(colors)], linewidth=2, alpha=0.7)
        ax1.legend()
    
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    # Volume subplot
    if volume and 'volume' in df.columns:
        colors = [PlottingConfig.FINANCIAL_COLORS['bullish'] if df.iloc[i]['close'] >= df.iloc[i]['open'] 
                 else PlottingConfig.FINANCIAL_COLORS['bearish'] for i in range(len(df))]
        
        ax2.bar(range(len(df)), df['volume'], color=colors, alpha=0.6)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Time Period')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, axes

def plot_technical_indicators(df: pd.DataFrame, price_col: str = 'close',
                             indicators: Dict[str, pd.Series] = None,
                             title: str = 'Technical Analysis Chart',
                             figsize: Tuple = None, show: bool = True) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot price with technical indicators in subplots
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        indicators: Dictionary of indicator name and series
        title: Chart title
        figsize: Figure size
        show: Whether to display the plot
        
    Returns:
        Figure and axes objects
    """
    figsize = figsize or (16, 12)
    n_indicators = len(indicators) if indicators else 0
    n_subplots = 1 + n_indicators
    
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, 
                            gridspec_kw={'height_ratios': [3] + [1] * n_indicators})
    
    if n_subplots == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=PlottingConfig.TITLE_SIZE, fontweight='bold')
    
    # Price chart
    axes[0].plot(df.index, df[price_col], color=PlottingConfig.COLORS['primary'], 
                linewidth=2, label='Price')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Technical indicators
    if indicators:
        colors = list(PlottingConfig.COLORS.values())
        for i, (name, series) in enumerate(indicators.items()):
            ax = axes[i + 1]
            color = colors[i % len(colors)]
            
            ax.plot(series.index, series.values, color=color, linewidth=2, label=name)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, axes

# ============================================
# STATISTICAL VISUALIZATION UTILITIES
# ============================================

def plot_correlation_heatmap(df: pd.DataFrame, title: str = 'Correlation Heatmap',
                            figsize: Tuple = None, annot: bool = True,
                            cmap: str = 'RdBu_r', center: float = 0,
                            show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create correlation heatmap with professional styling
    
    Args:
        df: DataFrame to calculate correlations
        title: Chart title
        figsize: Figure size
        annot: Show correlation values
        cmap: Color map
        center: Center value for color map
        show: Whether to display the plot
        
    Returns:
        Figure and axes objects
    """
    figsize = figsize or (12, 10)
    fig, ax = create_figure(figsize, title)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.2f',
               cmap=cmap, center=center, square=True, ax=ax,
               cbar_kws={'shrink': 0.8})
    
    ax.set_title(title, fontsize=PlottingConfig.TITLE_SIZE, fontweight='bold')
    
    if show:
        plt.show()
    
    return fig, ax

def plot_distribution_analysis(series: pd.Series, title: str = 'Distribution Analysis',
                              bins: int = 50, kde: bool = True, rug: bool = True,
                              figsize: Tuple = None, show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create comprehensive distribution analysis plot
    
    Args:
        series: Data series to analyze
        title: Chart title
        bins: Number of histogram bins
        kde: Show kernel density estimate
        rug: Show rug plot
        figsize: Figure size
        show: Whether to display the plot
        
    Returns:
        Figure and axes objects
    """
    fig, ax = create_figure(figsize, title)
    
    # Histogram with KDE
    sns.histplot(series, bins=bins, kde=kde, stat='density', 
                alpha=0.7, color=PlottingConfig.COLORS['primary'], ax=ax)
    
    # Add rug plot
    if rug:
        sns.rugplot(series, color=PlottingConfig.COLORS['secondary'], ax=ax)
    
    # Add statistics
    mean_val = series.mean()
    std_val = series.std()
    median_val = series.median()
    
    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.3f}')
    
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return fig, ax

def plot_boxplot_analysis(df: pd.DataFrame, x_col: str = None, y_col: str = None,
                         title: str = 'Box Plot Analysis', figsize: Tuple = None,
                         show_outliers: bool = True, show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create professional box plot with outlier analysis
    
    Args:
        df: DataFrame containing the data
        x_col: Column for x-axis (categorical)
        y_col: Column for y-axis (numerical)
        title: Chart title
        figsize: Figure size
        show_outliers: Whether to show outliers
        show: Whether to display the plot
        
    Returns:
        Figure and axes objects
    """
    fig, ax = create_figure(figsize, title)
    
    if x_col and y_col:
        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax, showfliers=show_outliers)
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
    else:
        # Single variable box plot
        data_col = y_col or df.select_dtypes(include=[np.number]).columns[0]
        sns.boxplot(y=df[data_col], ax=ax, showfliers=show_outliers)
        ax.set_ylabel(data_col.replace('_', ' ').title())
    
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return fig, ax

# ============================================
# MODEL EVALUATION VISUALIZATIONS
# ============================================

def plot_model_performance(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = 'Model Performance Analysis',
                          figsize: Tuple = None, show: bool = True) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create comprehensive model performance visualization
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Chart title
        figsize: Figure size
        show: Whether to display the plot
        
    Returns:
        Figure and axes objects
    """
    figsize = figsize or (16, 12)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=PlottingConfig.TITLE_SIZE, fontweight='bold')
    
    # Predictions vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color=PlottingConfig.COLORS['primary'])
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                   'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Predictions vs Actual')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color=PlottingConfig.COLORS['secondary'])
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, color=PlottingConfig.COLORS['success'], 
                   density=True, edgecolor='black')
    axes[1, 0].axvline(residuals.mean(), color='red', linestyle='--', 
                      label=f'Mean: {residuals.mean():.4f}')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # QQ plot for residuals normality
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Residuals Normality)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, axes

def plot_learning_curves(train_scores: List[float], val_scores: List[float],
                        title: str = 'Learning Curves', xlabel: str = 'Epoch',
                        ylabel: str = 'Score', figsize: Tuple = None,
                        show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot training and validation learning curves
    
    Args:
        train_scores: Training scores over epochs
        val_scores: Validation scores over epochs
        title: Chart title
        xlabel, ylabel: Axis labels
        figsize: Figure size
        show: Whether to display the plot
        
    Returns:
        Figure and axes objects
    """
    fig, ax = create_figure(figsize, title)
    
    epochs = range(1, len(train_scores) + 1)
    
    ax.plot(epochs, train_scores, label='Training', 
           color=PlottingConfig.COLORS['primary'], linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_scores, label='Validation', 
           color=PlottingConfig.COLORS['secondary'], linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight best validation score
    best_epoch = np.argmin(val_scores) if 'loss' in ylabel.lower() else np.argmax(val_scores)
    best_score = val_scores[best_epoch]
    ax.axvline(best_epoch + 1, color='red', linestyle='--', alpha=0.7, 
              label=f'Best: Epoch {best_epoch + 1}')
    ax.legend()
    
    if show:
        plt.show()
    
    return fig, ax

def plot_feature_importance(importance_dict: Dict[str, float], title: str = 'Feature Importance',
                           top_n: int = 20, figsize: Tuple = None,
                           show: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot feature importance with professional styling
    
    Args:
        importance_dict: Dictionary of feature names and importance scores
        title: Chart title
        top_n: Number of top features to show
        figsize: Figure size
        show: Whether to display the plot
        
    Returns:
        Figure and axes objects
    """
    figsize = figsize or (12, 8)
    fig, ax = create_figure(figsize, title)
    
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importance = zip(*sorted_features)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importance, color=PlottingConfig.COLORS['primary'], alpha=0.8)
    
    # Customize appearance
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score')
    ax.set_title(title, fontsize=PlottingConfig.TITLE_SIZE, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, importance)):
        ax.text(bar.get_width() + max(importance) * 0.01, bar.get_y() + bar.get_height()/2,
               f'{imp:.3f}', ha='left', va='center', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, ax

# ============================================
# DASHBOARD AND MULTI-PANEL UTILITIES
# ============================================

def create_dashboard(n_rows: int, n_cols: int, figsize: Tuple = None,
                    title: str = None, hspace: float = 0.3,
                    wspace: float = 0.3) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create multi-panel dashboard layout
    
    Args:
        n_rows: Number of rows
        n_cols: Number of columns
        figsize: Figure size
        title: Dashboard title
        hspace: Height spacing between subplots
        wspace: Width spacing between subplots
        
    Returns:
        Figure and axes array
    """
    figsize = figsize or (16, 12)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    
    # Ensure axes is always 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    return fig, axes

def save_plot(fig: plt.Figure, filepath: str, dpi: int = 300,
             bbox_inches: str = 'tight', facecolor: str = 'white') -> bool:
    """
    Save plot with high quality settings
    
    Args:
        fig: Figure object to save
        filepath: Output file path
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box in inches
        facecolor: Background color
        
    Returns:
        True if saved successfully
    """
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, 
                   facecolor=facecolor, edgecolor='none')
        print(f"✅ Plot saved: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Failed to save plot: {e}")
        return False

# ============================================
# UTILITY FUNCTIONS
# ============================================

def format_axis_labels(ax: plt.Axes, x_rotation: int = 0, y_rotation: int = 0,
                      x_format: str = None, y_format: str = None) -> None:
    """
    Format axis labels with rotation and number formatting
    
    Args:
        ax: Axes object to format
        x_rotation: X-axis label rotation angle
        y_rotation: Y-axis label rotation angle
        x_format: X-axis number format (e.g., '%.2f')
        y_format: Y-axis number format (e.g., '%.2f')
    """
    if x_rotation != 0:
        ax.tick_params(axis='x', rotation=x_rotation)
    
    if y_rotation != 0:
        ax.tick_params(axis='y', rotation=y_rotation)
    
    if x_format:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: x_format % x))
    
    if y_format:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: y_format % y))

def add_watermark(ax: plt.Axes, text: str = 'StockPredictionPro',
                 alpha: float = 0.1, fontsize: int = 20) -> None:
    """
    Add watermark to plot
    
    Args:
        ax: Axes object
        text: Watermark text
        alpha: Transparency
        fontsize: Font size
    """
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=fontsize,
           color='gray', alpha=alpha, ha='center', va='center',
           rotation=30, weight='bold')

def get_available_functions() -> List[str]:
    """Get list of all available plotting functions"""
    import sys
    current_module = sys.modules[__name__]
    functions = [name for name, obj in vars(current_module).items() 
                if callable(obj) and not name.startswith('_') and name != 'get_available_functions']
    return sorted(functions)

def print_plotting_help() -> None:
    """Print help information about available plotting utilities"""
    print("\n" + "="*60)
    print("STOCKPREDICTIONPRO PLOTTING UTILITIES")
    print("="*60)
    
    categories = {
        'Basic Charts': ['plot_line_chart', 'plot_multiple_series', 'create_figure'],
        'Financial Charts': ['plot_candlestick_chart', 'plot_technical_indicators'],
        'Statistical Plots': ['plot_correlation_heatmap', 'plot_distribution_analysis', 'plot_boxplot_analysis'],
        'Model Evaluation': ['plot_model_performance', 'plot_learning_curves', 'plot_feature_importance'],
        'Dashboard Tools': ['create_dashboard', 'save_plot'],
        'Utilities': ['format_axis_labels', 'add_watermark', 'setup_plotting_style']
    }
    
    for category, functions in categories.items():
        print(f"\n{category}:")
        for func in functions:
            print(f"  • {func}")
    
    print(f"\n📚 Total functions: {len(get_available_functions())}")
    print("💡 Use help(function_name) for detailed documentation")

# Success message
print("✅ StockPredictionPro Plotting Utilities loaded successfully")
print("📊 Use print_plotting_help() for available functions")
