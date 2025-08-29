"""
app/pages/04_🔮_Regression_Models.py

Advanced regression modeling page for StockPredictionPro.
Provides model selection, training, hyperparameter tuning, evaluation,
and comprehensive performance analysis for stock price prediction.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your components
from app.components.filters import (
    filter_symbols, filter_date_range, filter_categorical,
    create_data_filter_panel, filter_numeric
)
from app.components.charts import render_line_chart, render_correlation_heatmap
from app.components.metrics import (
    display_regression_metrics, create_metrics_grid,
    display_performance_summary
)
from app.components.tables import display_dataframe, create_download_button
from app.components.alerts import get_alert_manager
from app.components.forms import model_parameters_form
from app.styles.themes import apply_custom_theme

# Try importing your ML models from src/models/
try:
    from src.models.regression.linear import LinearRegressionModel
    from src.models.regression.random_forest import RandomForestModel
    from src.models.regression.neural_network import NeuralNetworkModel
    from src.models.factory import ModelFactory
    CUSTOM_MODELS_AVAILABLE = True
except ImportError:
    CUSTOM_MODELS_AVAILABLE = False

# ============================================
# MODEL CONFIGURATIONS
# ============================================

AVAILABLE_MODELS = {
    "Linear Regression": {
        "class": LinearRegression,
        "params": {},
        "hyperparams": {},
        "description": "Simple linear relationship between features and target"
    },
    "Ridge Regression": {
        "class": Ridge,
        "params": {"alpha": 1.0},
        "hyperparams": {"alpha": [0.1, 1.0, 10.0]},
        "description": "Linear regression with L2 regularization"
    },
    "Lasso Regression": {
        "class": Lasso,
        "params": {"alpha": 1.0},
        "hyperparams": {"alpha": [0.1, 1.0, 10.0]},
        "description": "Linear regression with L1 regularization and feature selection"
    },
    "Random Forest": {
        "class": RandomForestRegressor,
        "params": {"n_estimators": 100, "random_state": 42},
        "hyperparams": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        },
        "description": "Ensemble of decision trees with bagging"
    },
    "Gradient Boosting": {
        "class": GradientBoostingRegressor,
        "params": {"n_estimators": 100, "random_state": 42},
        "hyperparams": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        },
        "description": "Sequential ensemble with gradient boosting"
    },
    "Support Vector Machine": {
        "class": SVR,
        "params": {"kernel": "rbf"},
        "hyperparams": {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear"]
        },
        "description": "Support Vector Machine for regression"
    },
    "Neural Network": {
        "class": MLPRegressor,
        "params": {"hidden_layer_sizes": (100,), "random_state": 42, "max_iter": 500},
        "hyperparams": {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "alpha": [0.0001, 0.001, 0.01]
        },
        "description": "Multi-layer perceptron neural network"
    }
}

# ============================================
# DATA LOADING & PREPARATION
# ============================================

def load_stock_data_with_features(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Load stock data and create features for regression modeling"""
    # TODO: Replace with real data loading from your src/data modules
    np.random.seed(hash(symbol) % 2**32)
    
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)
    
    # Generate realistic stock data
    base_price = {"AAPL": 175.0, "MSFT": 342.0, "GOOGL": 2650.0, "TSLA": 245.0, "NVDA": 485.0}.get(symbol, 150.0)
    
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(max(prices[-1] * (1 + ret), 1.0))
    
    # Create comprehensive dataset
    data = pd.DataFrame({
        'date': dates,
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    # Ensure price relationships are valid
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def create_features(df: pd.DataFrame, lookback_days: int = 5) -> pd.DataFrame:
    """Create features for regression modeling"""
    features_df = df.copy()
    
    # Price-based features
    features_df['returns'] = features_df['close'].pct_change()
    features_df['volatility'] = features_df['returns'].rolling(window=10).std()
    features_df['price_change'] = features_df['close'].diff()
    
    # Technical indicators (simple implementations)
    features_df['sma_5'] = features_df['close'].rolling(window=5).mean()
    features_df['sma_20'] = features_df['close'].rolling(window=20).mean()
    features_df['ema_12'] = features_df['close'].ewm(span=12).mean()
    
    # RSI
    delta = features_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features_df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume features
    features_df['volume_sma'] = features_df['volume'].rolling(window=10).mean()
    features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
    
    # Lag features
    for i in range(1, lookback_days + 1):
        features_df[f'close_lag_{i}'] = features_df['close'].shift(i)
        features_df[f'volume_lag_{i}'] = features_df['volume'].shift(i)
        features_df[f'returns_lag_{i}'] = features_df['returns'].shift(i)
    
    # Target variable (next day's close price)
    features_df['target'] = features_df['close'].shift(-1)
    
    # Drop rows with NaN values
    features_df = features_df.dropna()
    
    return features_df

def prepare_model_data(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and target for model training"""
    X = df[feature_columns].values
    y = df['target'].values
    
    return X, y

# ============================================
# MODEL TRAINING & EVALUATION
# ============================================

def train_model_with_hyperparameter_tuning(model_name: str, 
                                          X_train: np.ndarray, 
                                          y_train: np.ndarray,
                                          use_grid_search: bool = True) -> Any:
    """Train model with optional hyperparameter tuning"""
    model_config = AVAILABLE_MODELS[model_name]
    
    if use_grid_search and model_config["hyperparams"]:
        # Grid search for best parameters
        base_model = model_config["class"]()
        grid_search = GridSearchCV(
            base_model,
            model_config["hyperparams"],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        with st.spinner(f"Performing hyperparameter tuning for {model_name}..."):
            grid_search.fit(X_train, y_train)
        
        st.success(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        # Train with default parameters
        model = model_config["class"](**model_config["params"])
        with st.spinner(f"Training {model_name}..."):
            model.fit(X_train, y_train)
        return model

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    return metrics, y_pred

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def plot_predictions_vs_actual(dates: pd.Series, 
                               y_actual: np.ndarray, 
                               y_predicted: np.ndarray,
                               model_name: str) -> None:
    """Plot actual vs predicted values"""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_actual,
        mode='lines',
        name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    # Predicted values
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_predicted,
        mode='lines',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{model_name} - Actual vs Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_residuals(y_actual: np.ndarray, y_predicted: np.ndarray, model_name: str) -> None:
    """Plot residuals analysis"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    residuals = y_actual - y_predicted
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals vs Predicted', 'Residuals Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_predicted,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', opacity=0.6)
        ),
        row=1, col=1
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Residuals histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name='Distribution',
            nbinsx=30,
            marker=dict(color='lightblue', opacity=0.7)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f'{model_name} - Residuals Analysis',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Residual Value", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# MAIN PAGE FUNCTION
# ============================================

def main():
    """Main regression models page function"""
    
    # Apply theme
    apply_custom_theme("financial")
    
    # Page header
    st.title("🔮 Regression Models")
    st.markdown("Advanced machine learning models for stock price prediction")
    st.markdown("---")
    
    # ============================================
    # MODEL CONFIGURATION
    # ============================================
    
    st.subheader("🛠️ Model Configuration")
    
    # Model selection and data filters
    col1, col2 = st.columns(2)
    
    with col1:
        available_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META"]
        selected_symbol = filter_categorical(
            "Select Symbol",
            available_symbols,
            multi=False,
            key="regression_symbol"
        )
        
        selected_model = filter_categorical(
            "Select Model",
            list(AVAILABLE_MODELS.keys()),
            multi=False,
            key="regression_model"
        )
    
    with col2:
        start_date, end_date = filter_date_range(
            default_days=365,
            key="regression_dates"
        )
        
        test_size = st.slider(
            "Test Size (%)",
            min_value=10,
            max_value=50,
            value=20,
            help="Percentage of data used for testing"
        )
    
    if not selected_symbol or not selected_model:
        st.warning("⚠️ Please select both symbol and model to proceed")
        return
    
    # Advanced options
    with st.expander("🔧 Advanced Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lookback_days = st.slider("Lookback Period", 1, 10, 5)
            use_scaling = st.checkbox("Feature Scaling", value=True)
        
        with col2:
            use_hyperparameter_tuning = st.checkbox("Hyperparameter Tuning", value=False)
            cross_validation = st.checkbox("Cross Validation", value=True)
        
        with col3:
            scaler_type = st.selectbox("Scaler Type", ["StandardScaler", "MinMaxScaler"])
            random_state = st.number_input("Random State", value=42, min_value=0)
    
    st.markdown("---")
    
    # ============================================
    # MODEL INFORMATION
    # ============================================
    
    st.subheader(f"📊 {selected_model} Information")
    
    model_info = AVAILABLE_MODELS[selected_model]
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Description:** {model_info['description']}")
    with col2:
        st.info(f"**Default Parameters:** {model_info['params']}")
    
    # ============================================
    # DATA LOADING & PREPARATION
    # ============================================
    
    with st.spinner("Loading and preparing data..."):
        # Load stock data
        stock_data = load_stock_data_with_features(selected_symbol, start_date, end_date)
        
        if stock_data.empty:
            st.error("❌ No data available for the selected period")
            return
        
        # Create features
        features_data = create_features(stock_data, lookback_days)
        
        # Select feature columns (exclude date, target, and some raw columns)
        exclude_cols = ['date', 'target', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in features_data.columns if col not in exclude_cols]
        
        # Prepare model data
        X, y = prepare_model_data(features_data, feature_columns)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state, shuffle=False
        )
        
        # Feature scaling
        if use_scaling:
            if scaler_type == "StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
    
    # Display data info
    st.subheader("📋 Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(features_data))
    with col2:
        st.metric("Features", len(feature_columns))
    with col3:
        st.metric("Training Size", len(X_train))
    with col4:
        st.metric("Test Size", len(X_test))
    
    # ============================================
    # MODEL TRAINING
    # ============================================
    
    if st.button("🚀 Train Model", type="primary"):
        
        # Train the model
        trained_model = train_model_with_hyperparameter_tuning(
            selected_model,
            X_train_scaled,
            y_train,
            use_hyperparameter_tuning
        )
        
        # Evaluate the model
        metrics, y_pred = evaluate_model(trained_model, X_test_scaled, y_test)
        
        # Store results in session state
        st.session_state.trained_model = trained_model
        st.session_state.model_metrics = metrics
        st.session_state.predictions = y_pred
        st.session_state.y_test = y_test
        st.session_state.test_dates = features_data['date'].iloc[-len(y_test):].reset_index(drop=True)
        
        st.success(f"✅ {selected_model} trained successfully!")
    
    # ============================================
    # RESULTS DISPLAY
    # ============================================
    
    if hasattr(st.session_state, 'trained_model') and st.session_state.trained_model is not None:
        
        st.markdown("---")
        st.subheader("📊 Model Results")
        
        # Display metrics
        metrics = st.session_state.model_metrics
        y_pred = st.session_state.predictions
        y_test = st.session_state.y_test
        test_dates = st.session_state.test_dates
        
        # Metrics grid
        metrics_grid = {
            "R² Score": {
                "value": metrics["r2"],
                "delta": None,
                "help": "Coefficient of determination (higher is better)"
            },
            "RMSE": {
                "value": metrics["rmse"],
                "delta": None,
                "help": "Root Mean Square Error (lower is better)"
            },
            "MAE": {
                "value": metrics["mae"],
                "delta": None,
                "help": "Mean Absolute Error (lower is better)"
            },
            "MAPE": {
                "value": f"{metrics['mape']:.2f}%",
                "delta": None,
                "help": "Mean Absolute Percentage Error"
            }
        }
        
        create_metrics_grid(metrics_grid, cols=4)
        
        # Detailed metrics
        with st.expander("📈 Detailed Performance Metrics"):
            display_regression_metrics(y_test, y_pred, f"{selected_model} Performance")
        
        # ============================================
        # VISUALIZATIONS
        # ============================================
        
        st.subheader("📈 Prediction Analysis")
        
        # Predictions vs Actual
        plot_predictions_vs_actual(test_dates, y_test, y_pred, selected_model)
        
        # Residuals analysis
        st.subheader("🔍 Residuals Analysis")
        plot_residuals(y_test, y_pred, selected_model)
        
        # Feature importance (if available)
        if hasattr(st.session_state.trained_model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': st.session_state.trained_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            import plotly.express as px
            fig = px.bar(
                importance_df.head(15),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Feature Importances'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # ============================================
        # CROSS VALIDATION
        # ============================================
        
        if cross_validation:
            st.subheader("🔄 Cross Validation Results")
            
            with st.spinner("Performing cross validation..."):
                cv_scores = cross_val_score(
                    st.session_state.trained_model,
                    X_train_scaled,
                    y_train,
                    cv=5,
                    scoring='neg_mean_squared_error'
                )
                cv_rmse_scores = np.sqrt(-cv_scores)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CV RMSE Mean", f"{cv_rmse_scores.mean():.4f}")
            with col2:
                st.metric("CV RMSE Std", f"{cv_rmse_scores.std():.4f}")
            with col3:
                st.metric("CV Score Range", f"{cv_rmse_scores.min():.3f} - {cv_rmse_scores.max():.3f}")
        
        # ============================================
        # DATA EXPORT
        # ============================================
        
        st.subheader("📥 Export Results")
        
        # Prepare results DataFrame
        results_df = pd.DataFrame({
            'date': test_dates,
            'actual': y_test,
            'predicted': y_pred,
            'residual': y_test - y_pred,
            'abs_error': np.abs(y_test - y_pred),
            'pct_error': np.abs((y_test - y_pred) / y_test) * 100
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_download_button(
                results_df,
                f"{selected_symbol}_{selected_model}_results.csv",
                "📊 Download Predictions",
                key="predictions_download"
            )
        
        with col2:
            # Create model summary
            model_summary = pd.DataFrame([{
                'symbol': selected_symbol,
                'model': selected_model,
                'r2_score': metrics['r2'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape'],
                'test_size': test_size,
                'features_count': len(feature_columns),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }])
            
            create_download_button(
                model_summary,
                f"{selected_symbol}_{selected_model}_summary.csv",
                "📋 Download Summary",
                key="summary_download"
            )
    
    # ============================================
    # FOOTER
    # ============================================
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        🔮 Regression Models | Advanced ML for Stock Prediction | 
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE EXECUTION
# ============================================

if __name__ == "__main__":
    main()
