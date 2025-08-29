"""
app/pages/06_⚖️_Model_Comparison.py

Advanced model comparison dashboard for StockPredictionPro.
Compares regression and classification models side-by-side,
provides comprehensive performance analysis and visualization.

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your components
from app.components.filters import (
    filter_symbols, filter_date_range, filter_categorical,
    create_data_filter_panel, filter_numeric
)
from app.components.charts import render_correlation_heatmap
from app.components.metrics import (
    display_regression_metrics, display_classification_metrics,
    create_metrics_grid, display_performance_summary
)
from app.components.tables import display_dataframe, create_download_button
from app.components.alerts import get_alert_manager
from app.styles.themes import apply_custom_theme

# ============================================
# MODEL CONFIGURATIONS
# ============================================

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf'),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(100,), random_state=42, max_iter=500)
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVC": SVC(kernel='rbf', random_state=42, probability=True),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=500)
}

# ============================================
# DATA LOADING & PREPARATION
# ============================================

def load_stock_data_for_comparison(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Load stock data for model comparison"""
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
        'volume': np.random.lognormal(14, 0.5, n_days).astype(int)
    })
    
    # Ensure valid OHLC relationships
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def create_features_for_comparison(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create features for both regression and classification tasks"""
    features_df = df.copy()
    
    # Price-based features
    features_df['returns'] = features_df['close'].pct_change()
    features_df['volatility'] = features_df['returns'].rolling(window=10).std()
    features_df['price_change'] = features_df['close'].diff()
    
    # Technical indicators
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
    for i in range(1, 4):
        features_df[f'returns_lag_{i}'] = features_df['returns'].shift(i)
        features_df[f'volume_ratio_lag_{i}'] = features_df['volume_ratio'].shift(i)
    
    # Regression target (next day's close price)
    features_df['regression_target'] = features_df['close'].shift(-1)
    
    # Classification target (next day price direction)
    features_df['classification_target'] = (features_df['close'].shift(-1) > features_df['close']).astype(int)
    
    # Drop rows with NaN values
    features_df = features_df.dropna()
    
    # Split into regression and classification datasets
    exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'regression_target', 'classification_target']
    feature_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    regression_df = features_df[feature_columns + ['regression_target']].copy()
    classification_df = features_df[feature_columns + ['classification_target']].copy()
    
    return regression_df, classification_df

# ============================================
# MODEL TRAINING & EVALUATION
# ============================================

def train_and_evaluate_regression_models(regression_df: pd.DataFrame, 
                                       selected_models: List[str],
                                       test_size: float = 0.2) -> Dict[str, Dict[str, Any]]:
    """Train and evaluate multiple regression models"""
    results = {}
    
    # Prepare data
    X = regression_df.drop('regression_target', axis=1)
    y = regression_df['regression_target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name in selected_models:
        if model_name in REGRESSION_MODELS:
            with st.spinner(f"Training {model_name}..."):
                model = REGRESSION_MODELS[model_name]
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'actual': y_test,
                    'metrics': {
                        'MSE': mse,
                        'MAE': mae,
                        'RMSE': rmse,
                        'R²': r2,
                        'MAPE': mape
                    }
                }
    
    return results

def train_and_evaluate_classification_models(classification_df: pd.DataFrame,
                                           selected_models: List[str],
                                           test_size: float = 0.2) -> Dict[str, Dict[str, Any]]:
    """Train and evaluate multiple classification models"""
    results = {}
    
    # Prepare data
    X = classification_df.drop('classification_target', axis=1)
    y = classification_df['classification_target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name in selected_models:
        if model_name in CLASSIFICATION_MODELS:
            with st.spinner(f"Training {model_name}..."):
                model = CLASSIFICATION_MODELS[model_name]
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'actual': y_test,
                    'metrics': {
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1-Score': f1
                    }
                }
    
    return results

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_regression_comparison_chart(results: Dict[str, Dict[str, Any]], symbol: str) -> None:
    """Create comparison chart for regression models"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('R² Score', 'RMSE', 'MAE', 'MAPE'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    models = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # R² Score
    r2_scores = [results[model]['metrics']['R²'] for model in models]
    fig.add_trace(
        go.Bar(x=models, y=r2_scores, name='R²', marker_color=colors[0]),
        row=1, col=1
    )
    
    # RMSE
    rmse_scores = [results[model]['metrics']['RMSE'] for model in models]
    fig.add_trace(
        go.Bar(x=models, y=rmse_scores, name='RMSE', marker_color=colors[1]),
        row=1, col=2
    )
    
    # MAE
    mae_scores = [results[model]['metrics']['MAE'] for model in models]
    fig.add_trace(
        go.Bar(x=models, y=mae_scores, name='MAE', marker_color=colors[2]),
        row=2, col=1
    )
    
    # MAPE
    mape_scores = [results[model]['metrics']['MAPE'] for model in models]
    fig.add_trace(
        go.Bar(x=models, y=mape_scores, name='MAPE (%)', marker_color=colors[3]),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title=f'{symbol} - Regression Models Comparison',
        showlegend=False,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_classification_comparison_chart(results: Dict[str, Dict[str, Any]], symbol: str) -> None:
    """Create comparison chart for classification models"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    models = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Accuracy
    accuracy_scores = [results[model]['metrics']['Accuracy'] for model in models]
    fig.add_trace(
        go.Bar(x=models, y=accuracy_scores, name='Accuracy', marker_color=colors[0]),
        row=1, col=1
    )
    
    # Precision
    precision_scores = [results[model]['metrics']['Precision'] for model in models]
    fig.add_trace(
        go.Bar(x=models, y=precision_scores, name='Precision', marker_color=colors[1]),
        row=1, col=2
    )
    
    # Recall
    recall_scores = [results[model]['metrics']['Recall'] for model in models]
    fig.add_trace(
        go.Bar(x=models, y=recall_scores, name='Recall', marker_color=colors[2]),
        row=2, col=1
    )
    
    # F1-Score
    f1_scores = [results[model]['metrics']['F1-Score'] for model in models]
    fig.add_trace(
        go.Bar(x=models, y=f1_scores, name='F1-Score', marker_color=colors[3]),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title=f'{symbol} - Classification Models Comparison',
        showlegend=False,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_performance_radar_chart(regression_results: Dict[str, Dict[str, Any]], 
                                 classification_results: Dict[str, Dict[str, Any]]) -> None:
    """Create radar chart comparing model performance"""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Normalize metrics for radar chart (0-1 scale)
    def normalize_metric(values, higher_is_better=True):
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        if higher_is_better:
            return [(v - min_val) / (max_val - min_val) for v in values]
        else:
            return [1 - (v - min_val) / (max_val - min_val) for v in values]
    
    # Get best model from each category
    if regression_results:
        best_regression_model = max(regression_results.keys(), 
                                   key=lambda x: regression_results[x]['metrics']['R²'])
        reg_metrics = regression_results[best_regression_model]['metrics']
        
        reg_values = [
            reg_metrics['R²'],
            1 - (reg_metrics['RMSE'] / max([r['metrics']['RMSE'] for r in regression_results.values()])),
            1 - (reg_metrics['MAE'] / max([r['metrics']['MAE'] for r in regression_results.values()])),
            1 - (reg_metrics['MAPE'] / 100)  # Convert to 0-1 scale
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=reg_values,
            theta=['R² Score', 'RMSE (inv)', 'MAE (inv)', 'MAPE (inv)'],
            fill='toself',
            name=f'Best Regression ({best_regression_model})',
            line_color='blue'
        ))
    
    if classification_results:
        best_classification_model = max(classification_results.keys(),
                                       key=lambda x: classification_results[x]['metrics']['Accuracy'])
        clf_metrics = classification_results[best_classification_model]['metrics']
        
        clf_values = [
            clf_metrics['Accuracy'],
            clf_metrics['Precision'],
            clf_metrics['Recall'],
            clf_metrics['F1-Score']
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=clf_values,
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            fill='toself',
            name=f'Best Classification ({best_classification_model})',
            line_color='red'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Best Model Performance Comparison",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# MAIN PAGE FUNCTION
# ============================================

def main():
    """Main model comparison page function"""
    
    # Apply theme
    apply_custom_theme("financial")
    
    # Page header
    st.title("⚖️ Model Comparison Dashboard")
    st.markdown("Compare regression and classification models side-by-side for stock prediction")
    st.markdown("---")
    
    # ============================================
    # CONFIGURATION
    # ============================================
    
    st.subheader("🛠️ Comparison Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        available_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META"]
        selected_symbol = filter_categorical(
            "Select Symbol",
            available_symbols,
            multi=False,
            key="comparison_symbol"
        )
        
        start_date, end_date = filter_date_range(
            default_days=365,
            key="comparison_dates"
        )
    
    with col2:
        test_size = st.slider("Test Size (%)", 10, 50, 20)
        
        comparison_type = st.selectbox(
            "Comparison Type",
            ["Both", "Regression Only", "Classification Only"],
            help="Choose which model types to compare"
        )
    
    if not selected_symbol:
        st.warning("⚠️ Please select a symbol to proceed")
        return
    
    # Model selection
    st.subheader("📊 Model Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if comparison_type in ["Both", "Regression Only"]:
            st.write("**Regression Models**")
            selected_regression_models = []
            for model_name in REGRESSION_MODELS.keys():
                if st.checkbox(model_name, key=f"reg_{model_name}", value=(model_name in ["Linear Regression", "Random Forest"])):
                    selected_regression_models.append(model_name)
        else:
            selected_regression_models = []
    
    with col2:
        if comparison_type in ["Both", "Classification Only"]:
            st.write("**Classification Models**")
            selected_classification_models = []
            for model_name in CLASSIFICATION_MODELS.keys():
                if st.checkbox(model_name, key=f"clf_{model_name}", value=(model_name in ["Logistic Regression", "Random Forest"])):
                    selected_classification_models.append(model_name)
        else:
            selected_classification_models = []
    
    if not selected_regression_models and not selected_classification_models:
        st.warning("⚠️ Please select at least one model to compare")
        return
    
    st.markdown("---")
    
    # ============================================
    # DATA LOADING & PREPARATION
    # ============================================
    
    with st.spinner("Loading and preparing data..."):
        stock_data = load_stock_data_for_comparison(selected_symbol, start_date, end_date)
        
        if stock_data.empty:
            st.error("❌ No data available for the selected period")
            return
        
        regression_df, classification_df = create_features_for_comparison(stock_data)
    
    # Display data info
    st.subheader("📋 Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(stock_data))
    with col2:
        st.metric("Features", len(regression_df.columns) - 1)
    with col3:
        st.metric("Regression Samples", len(regression_df))
    with col4:
        st.metric("Classification Samples", len(classification_df))
    
    # ============================================
    # MODEL TRAINING & COMPARISON
    # ============================================
    
    if st.button("🚀 Run Model Comparison", type="primary"):
        
        # Train regression models
        regression_results = {}
        if selected_regression_models:
            st.subheader("📈 Regression Models Training")
            regression_results = train_and_evaluate_regression_models(
                regression_df, selected_regression_models, test_size/100
            )
        
        # Train classification models
        classification_results = {}
        if selected_classification_models:
            st.subheader("🎯 Classification Models Training")
            classification_results = train_and_evaluate_classification_models(
                classification_df, selected_classification_models, test_size/100
            )
        
        # Store results in session state
        st.session_state.regression_results = regression_results
        st.session_state.classification_results = classification_results
        st.session_state.selected_symbol = selected_symbol
        
        st.success("✅ Model comparison completed!")
    
    # ============================================
    # RESULTS DISPLAY
    # ============================================
    
    if (hasattr(st.session_state, 'regression_results') or 
        hasattr(st.session_state, 'classification_results')):
        
        st.markdown("---")
        st.subheader("📊 Comparison Results")
        
        regression_results = getattr(st.session_state, 'regression_results', {})
        classification_results = getattr(st.session_state, 'classification_results', {})
        symbol = getattr(st.session_state, 'selected_symbol', selected_symbol)
        
        # ============================================
        # REGRESSION RESULTS
        # ============================================
        
        if regression_results:
            st.subheader("📈 Regression Models Performance")
            
            # Create metrics comparison table
            reg_metrics_df = pd.DataFrame({
                model: results['metrics'] for model, results in regression_results.items()
            }).T
            
            st.dataframe(reg_metrics_df.round(4), use_container_width=True)
            
            # Regression charts
            create_regression_comparison_chart(regression_results, symbol)
            
            # Best regression model
            best_reg_model = max(regression_results.keys(), 
                               key=lambda x: regression_results[x]['metrics']['R²'])
            st.success(f"🏆 Best Regression Model: **{best_reg_model}** (R² = {regression_results[best_reg_model]['metrics']['R²']:.4f})")
        
        # ============================================
        # CLASSIFICATION RESULTS
        # ============================================
        
        if classification_results:
            st.subheader("🎯 Classification Models Performance")
            
            # Create metrics comparison table
            clf_metrics_df = pd.DataFrame({
                model: results['metrics'] for model, results in classification_results.items()
            }).T
            
            st.dataframe(clf_metrics_df.round(4), use_container_width=True)
            
            # Classification charts
            create_classification_comparison_chart(classification_results, symbol)
            
            # Best classification model
            best_clf_model = max(classification_results.keys(),
                               key=lambda x: classification_results[x]['metrics']['Accuracy'])
            st.success(f"🏆 Best Classification Model: **{best_clf_model}** (Accuracy = {classification_results[best_clf_model]['metrics']['Accuracy']:.4f})")
        
        # ============================================
        # COMBINED ANALYSIS
        # ============================================
        
        if regression_results and classification_results:
            st.subheader("🔍 Combined Analysis")
            
            # Radar chart comparison
            create_performance_radar_chart(regression_results, classification_results)
            
            # Summary recommendations
            st.subheader("💡 Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**For Price Prediction:**")
                best_reg = max(regression_results.keys(), 
                             key=lambda x: regression_results[x]['metrics']['R²'])
                st.info(f"Use **{best_reg}** for continuous price forecasting")
            
            with col2:
                st.write("**For Direction Prediction:**")
                best_clf = max(classification_results.keys(),
                             key=lambda x: classification_results[x]['metrics']['Accuracy'])
                st.info(f"Use **{best_clf}** for buy/sell signal generation")
        
        # ============================================
        # DETAILED ANALYSIS
        # ============================================
        
        with st.expander("🔬 Detailed Model Analysis"):
            
            if regression_results:
                st.write("**Regression Models Detailed Metrics:**")
                for model_name, results in regression_results.items():
                    st.write(f"**{model_name}:**")
                    display_regression_metrics(
                        results['actual'].values, 
                        results['predictions'], 
                        f"{model_name} Performance"
                    )
                    st.markdown("---")
            
            if classification_results:
                st.write("**Classification Models Detailed Metrics:**")
                for model_name, results in classification_results.items():
                    st.write(f"**{model_name}:**")
                    display_classification_metrics(
                        results['actual'].values,
                        results['predictions'],
                        f"{model_name} Performance"
                    )
                    st.markdown("---")
        
        # ============================================
        # EXPORT RESULTS
        # ============================================
        
        st.subheader("📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if regression_results:
                reg_summary = pd.DataFrame({
                    'Model': list(regression_results.keys()),
                    'Task': ['Regression'] * len(regression_results),
                    'R²': [r['metrics']['R²'] for r in regression_results.values()],
                    'RMSE': [r['metrics']['RMSE'] for r in regression_results.values()],
                    'MAE': [r['metrics']['MAE'] for r in regression_results.values()],
                    'MAPE': [r['metrics']['MAPE'] for r in regression_results.values()]
                })
                
                create_download_button(
                    reg_summary,
                    f"{symbol}_regression_comparison.csv",
                    "📈 Download Regression Results",
                    key="reg_comparison_download"
                )
        
        with col2:
            if classification_results:
                clf_summary = pd.DataFrame({
                    'Model': list(classification_results.keys()),
                    'Task': ['Classification'] * len(classification_results),
                    'Accuracy': [r['metrics']['Accuracy'] for r in classification_results.values()],
                    'Precision': [r['metrics']['Precision'] for r in classification_results.values()],
                    'Recall': [r['metrics']['Recall'] for r in classification_results.values()],
                    'F1-Score': [r['metrics']['F1-Score'] for r in classification_results.values()]
                })
                
                create_download_button(
                    clf_summary,
                    f"{symbol}_classification_comparison.csv",
                    "🎯 Download Classification Results",
                    key="clf_comparison_download"
                )
    
    # ============================================
    # FOOTER
    # ============================================
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        ⚖️ Model Comparison Dashboard | Advanced ML Performance Analysis | 
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE EXECUTION
# ============================================

if __name__ == "__main__":
    main()
