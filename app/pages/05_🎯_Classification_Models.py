"""
app/pages/05_🎯_Classification_Models.py

Advanced classification modeling page for StockPredictionPro.
Provides binary and multi-class classification for stock direction prediction,
trading signal classification, and market regime identification.

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score, roc_curve
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
    display_classification_metrics, create_metrics_grid,
    confusion_matrix_metrics
)
from app.components.tables import display_dataframe, create_download_button
from app.components.alerts import get_alert_manager
from app.components.forms import model_parameters_form
from app.styles.themes import apply_custom_theme

# Try importing your ML models from src/models/
try:
    from src.models.classification.logistic import LogisticClassifier
    from src.models.classification.random_forest import RandomForestClassifier as CustomRF
    from src.models.classification.neural_network import NeuralNetworkClassifier
    from src.models.factory import ModelFactory
    CUSTOM_MODELS_AVAILABLE = True
except ImportError:
    CUSTOM_MODELS_AVAILABLE = False

# ============================================
# MODEL CONFIGURATIONS
# ============================================

AVAILABLE_MODELS = {
    "Logistic Regression": {
        "class": LogisticRegression,
        "params": {"random_state": 42, "max_iter": 1000},
        "hyperparams": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "lbfgs"]
        },
        "description": "Linear classifier using logistic function for probability estimation"
    },
    "Random Forest": {
        "class": RandomForestClassifier,
        "params": {"n_estimators": 100, "random_state": 42},
        "hyperparams": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "description": "Ensemble of decision trees with bootstrap aggregating"
    },
    "Gradient Boosting": {
        "class": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "random_state": 42},
        "hyperparams": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        },
        "description": "Sequential ensemble with gradient boosting for classification"
    },
    "Support Vector Machine": {
        "class": SVC,
        "params": {"random_state": 42, "probability": True},
        "hyperparams": {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear", "poly"]
        },
        "description": "Support Vector Machine for non-linear classification"
    },
    "Neural Network": {
        "class": MLPClassifier,
        "params": {"hidden_layer_sizes": (100,), "random_state": 42, "max_iter": 500},
        "hyperparams": {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"]
        },
        "description": "Multi-layer perceptron neural network classifier"
    },
    "Naive Bayes": {
        "class": GaussianNB,
        "params": {},
        "hyperparams": {
            "var_smoothing": [1e-9, 1e-8, 1e-7]
        },
        "description": "Gaussian Naive Bayes classifier assuming feature independence"
    }
}

# ============================================
# DATA LOADING & PREPARATION
# ============================================

def load_stock_data_for_classification(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Load stock data for classification modeling"""
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

def create_classification_features(df: pd.DataFrame, 
                                 prediction_horizon: int = 1,
                                 lookback_days: int = 5) -> pd.DataFrame:
    """Create features for classification modeling"""
    features_df = df.copy()
    
    # Price-based features
    features_df['returns'] = features_df['close'].pct_change()
    features_df['volatility'] = features_df['returns'].rolling(window=10).std()
    features_df['price_change'] = features_df['close'].diff()
    features_df['price_momentum'] = features_df['close'].pct_change(periods=5)
    
    # Technical indicators
    features_df['sma_5'] = features_df['close'].rolling(window=5).mean()
    features_df['sma_20'] = features_df['close'].rolling(window=20).mean()
    features_df['ema_12'] = features_df['close'].ewm(span=12).mean()
    
    # Price relative to moving averages
    features_df['price_vs_sma5'] = features_df['close'] / features_df['sma_5'] - 1
    features_df['price_vs_sma20'] = features_df['close'] / features_df['sma_20'] - 1
    
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
        features_df[f'returns_lag_{i}'] = features_df['returns'].shift(i)
        features_df[f'volume_ratio_lag_{i}'] = features_df['volume_ratio'].shift(i)
    
    # Create target variable based on prediction type
    if prediction_horizon == 1:
        # Next day price direction (up/down)
        features_df['target'] = (features_df['close'].shift(-1) > features_df['close']).astype(int)
    else:
        # Multi-day ahead prediction
        future_returns = features_df['close'].pct_change(periods=prediction_horizon).shift(-prediction_horizon)
        features_df['target'] = (future_returns > 0).astype(int)
    
    # Multi-class target (for regime classification)
    returns_quantiles = features_df['returns'].quantile([0.33, 0.67])
    conditions = [
        features_df['returns'] <= returns_quantiles.iloc[0],
        (features_df['returns'] > returns_quantiles.iloc[0]) & (features_df['returns'] <= returns_quantiles.iloc[1]),
        features_df['returns'] > returns_quantiles.iloc[1]
    ]
    choices = [0, 1, 2]  # Bear, Neutral, Bull
    features_df['target_regime'] = np.select(conditions, choices, default=1)
    
    # Drop rows with NaN values
    features_df = features_df.dropna()
    
    return features_df

def prepare_classification_data(df: pd.DataFrame, 
                              feature_columns: List[str],
                              target_type: str = 'binary') -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and target for classification"""
    X = df[feature_columns].values
    
    if target_type == 'binary':
        y = df['target'].values
    elif target_type == 'regime':
        y = df['target_regime'].values
    else:
        y = df['target'].values
    
    return X, y

# ============================================
# MODEL TRAINING & EVALUATION
# ============================================

def train_classification_model(model_name: str,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              use_grid_search: bool = True) -> Any:
    """Train classification model with optional hyperparameter tuning"""
    model_config = AVAILABLE_MODELS[model_name]
    
    if use_grid_search and model_config["hyperparams"]:
        base_model = model_config["class"]()
        grid_search = GridSearchCV(
            base_model,
            model_config["hyperparams"],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        with st.spinner(f"Performing hyperparameter tuning for {model_name}..."):
            grid_search.fit(X_train, y_train)
        
        st.success(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        model = model_config["class"](**model_config["params"])
        with st.spinner(f"Training {model_name}..."):
            model.fit(X_train, y_train)
        return model

def evaluate_classification_model(model: Any, 
                                X_test: np.ndarray, 
                                y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate classification model performance"""
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    except:
        y_proba = None
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    }
    
    # Add AUC if binary classification
    if len(np.unique(y_test)) == 2 and y_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_test, y_proba)
    
    return metrics, y_pred

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, class_names: List[str] = None) -> None:
    """Plot confusion matrix"""
    import plotly.graph_objects as go
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title=f'{model_name} - Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str) -> None:
    """Plot ROC curve for binary classification"""
    import plotly.graph_objects as go
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc:.3f})',
        line=dict(color='blue', width=3)
    ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.5)',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{model_name} - ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(model: Any, feature_names: List[str], model_name: str) -> None:
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        import plotly.express as px
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'{model_name} - Top 15 Feature Importances'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type")

# ============================================
# MAIN PAGE FUNCTION
# ============================================

def main():
    """Main classification models page function"""
    
    # Apply theme
    apply_custom_theme("financial")
    
    # Page header
    st.title("🎯 Classification Models")
    st.markdown("Machine learning classification for stock direction prediction and market regime identification")
    st.markdown("---")
    
    # ============================================
    # MODEL CONFIGURATION
    # ============================================
    
    st.subheader("🛠️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        available_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META"]
        selected_symbol = filter_categorical(
            "Select Symbol",
            available_symbols,
            multi=False,
            key="classification_symbol"
        )
        
        selected_model = filter_categorical(
            "Select Model",
            list(AVAILABLE_MODELS.keys()),
            multi=False,
            key="classification_model"
        )
    
    with col2:
        start_date, end_date = filter_date_range(
            default_days=365,
            key="classification_dates"
        )
        
        prediction_type = st.selectbox(
            "Prediction Type",
            ["Binary (Up/Down)", "Multi-class (Regime)"],
            help="Choose between binary direction prediction or multi-class regime classification"
        )
    
    if not selected_symbol or not selected_model:
        st.warning("⚠️ Please select both symbol and model to proceed")
        return
    
    # Advanced options
    with st.expander("🔧 Advanced Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prediction_horizon = st.slider("Prediction Horizon (days)", 1, 10, 1)
            lookback_days = st.slider("Lookback Period", 1, 10, 5)
        
        with col2:
            test_size = st.slider("Test Size (%)", 10, 50, 20)
            use_scaling = st.checkbox("Feature Scaling", value=True)
        
        with col3:
            use_hyperparameter_tuning = st.checkbox("Hyperparameter Tuning", value=False)
            cross_validation = st.checkbox("Cross Validation", value=True)
    
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
        stock_data = load_stock_data_for_classification(selected_symbol, start_date, end_date)
        
        if stock_data.empty:
            st.error("❌ No data available for the selected period")
            return
        
        # Create features
        features_data = create_classification_features(
            stock_data, 
            prediction_horizon, 
            lookback_days
        )
        
        # Select feature columns
        exclude_cols = ['date', 'target', 'target_regime', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in features_data.columns if col not in exclude_cols]
        
        # Prepare model data
        target_type = 'binary' if prediction_type == "Binary (Up/Down)" else 'regime'
        X, y = prepare_classification_data(features_data, feature_columns, target_type)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42, stratify=y
        )
        
        # Feature scaling
        if use_scaling:
            scaler = StandardScaler()
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
    
    # Class distribution
    st.subheader("📊 Class Distribution")
    class_counts = pd.Series(y).value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if target_type == 'binary':
            st.write("**Binary Classes:**")
            st.write(f"Down (0): {class_counts.get(0, 0)} samples")
            st.write(f"Up (1): {class_counts.get(1, 0)} samples")
        else:
            st.write("**Market Regimes:**")
            st.write(f"Bear (0): {class_counts.get(0, 0)} samples")
            st.write(f"Neutral (1): {class_counts.get(1, 0)} samples")
            st.write(f"Bull (2): {class_counts.get(2, 0)} samples")
    
    with col2:
        import plotly.express as px
        
        class_labels = ["Down", "Up"] if target_type == 'binary' else ["Bear", "Neutral", "Bull"]
        
        fig = px.pie(
            values=class_counts.values,
            names=[class_labels[i] for i in class_counts.index],
            title="Class Distribution"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # MODEL TRAINING
    # ============================================
    
    if st.button("🚀 Train Model", type="primary"):
        
        # Train the model
        trained_model = train_classification_model(
            selected_model,
            X_train_scaled,
            y_train,
            use_hyperparameter_tuning
        )
        
        # Evaluate the model
        metrics, y_pred = evaluate_classification_model(trained_model, X_test_scaled, y_test)
        
        # Store results in session state
        st.session_state.trained_model = trained_model
        st.session_state.model_metrics = metrics
        st.session_state.predictions = y_pred
        st.session_state.y_test = y_test
        st.session_state.feature_names = feature_columns
        st.session_state.target_type = target_type
        st.session_state.scaler = scaler if use_scaling else None
        
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
        feature_names = st.session_state.feature_names
        target_type = st.session_state.target_type
        
        # Metrics grid
        metrics_grid = {
            "Accuracy": {
                "value": f"{metrics['accuracy']:.3f}",
                "delta": None,
                "help": "Overall classification accuracy"
            },
            "Precision": {
                "value": f"{metrics['precision']:.3f}",
                "delta": None,
                "help": "Weighted precision across classes"
            },
            "Recall": {
                "value": f"{metrics['recall']:.3f}",
                "delta": None,
                "help": "Weighted recall across classes"
            },
            "F1-Score": {
                "value": f"{metrics['f1_score']:.3f}",
                "delta": None,
                "help": "Weighted F1-score across classes"
            }
        }
        
        if "auc_roc" in metrics:
            metrics_grid["AUC-ROC"] = {
                "value": f"{metrics['auc_roc']:.3f}",
                "delta": None,
                "help": "Area Under ROC Curve"
            }
        
        create_metrics_grid(metrics_grid, cols=4)
        
        # Detailed metrics
        with st.expander("📈 Detailed Performance Metrics"):
            display_classification_metrics(y_test, y_pred, f"{selected_model} Performance")
        
        # ============================================
        # VISUALIZATIONS
        # ============================================
        
        st.subheader("📈 Model Analysis")
        
        # Confusion Matrix
        class_names = ["Down", "Up"] if target_type == 'binary' else ["Bear", "Neutral", "Bull"]
        plot_confusion_matrix(y_test, y_pred, selected_model, class_names)
        
        # ROC Curve (for binary classification)
        if target_type == 'binary' and hasattr(st.session_state.trained_model, 'predict_proba'):
            try:
                y_proba = st.session_state.trained_model.predict_proba(X_test_scaled)[:, 1]
                st.subheader("📊 ROC Analysis")
                plot_roc_curve(y_test, y_proba, selected_model)
            except:
                st.info("ROC curve not available for this model configuration")
        
        # Feature importance
        st.subheader("🎯 Feature Analysis")
        plot_feature_importance(st.session_state.trained_model, feature_names, selected_model)
        
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
                    scoring='accuracy'
                )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CV Accuracy Mean", f"{cv_scores.mean():.4f}")
            with col2:
                st.metric("CV Accuracy Std", f"{cv_scores.std():.4f}")
            with col3:
                st.metric("CV Score Range", f"{cv_scores.min():.3f} - {cv_scores.max():.3f}")
        
        # ============================================
        # PREDICTION EXAMPLES
        # ============================================
        
        st.subheader("🔮 Recent Predictions")
        
        # Show recent predictions with actual outcomes
        recent_predictions = pd.DataFrame({
            'Actual': y_test[-10:],
            'Predicted': y_pred[-10:],
            'Correct': y_test[-10:] == y_pred[-10:]
        })
        
        if target_type == 'binary':
            recent_predictions['Actual'] = recent_predictions['Actual'].map({0: 'Down', 1: 'Up'})
            recent_predictions['Predicted'] = recent_predictions['Predicted'].map({0: 'Down', 1: 'Up'})
        else:
            recent_predictions['Actual'] = recent_predictions['Actual'].map({0: 'Bear', 1: 'Neutral', 2: 'Bull'})
            recent_predictions['Predicted'] = recent_predictions['Predicted'].map({0: 'Bear', 1: 'Neutral', 2: 'Bull'})
        
        st.dataframe(recent_predictions, use_container_width=True)
        
        # ============================================
        # DATA EXPORT
        # ============================================
        
        st.subheader("📥 Export Results")
        
        # Prepare results DataFrame
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'correct': y_test == y_pred
        })
        
        # Add probabilities if available
        if hasattr(st.session_state.trained_model, 'predict_proba'):
            try:
                proba = st.session_state.trained_model.predict_proba(X_test_scaled)
                for i in range(proba.shape[1]):
                    results_df[f'prob_class_{i}'] = proba[:, i]
            except:
                pass
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_download_button(
                results_df,
                f"{selected_symbol}_{selected_model}_predictions.csv",
                "📊 Download Predictions",
                key="predictions_download"
            )
        
        with col2:
            # Create model summary
            model_summary = pd.DataFrame([{
                'symbol': selected_symbol,
                'model': selected_model,
                'prediction_type': prediction_type,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'test_size': test_size,
                'features_count': len(feature_names),
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
        🎯 Classification Models | Advanced ML for Market Direction Prediction | 
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE EXECUTION
# ============================================

if __name__ == "__main__":
    main()
