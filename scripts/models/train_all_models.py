"""
scripts/models/train_all_models.py

Comprehensive model training pipeline for StockPredictionPro.
Trains multiple ML models (XGBoost, LightGBM, JAX/Flax neural networks) with automated 
hyperparameter optimization, cross-validation, and model persistence.

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

# ML Models
import xgboost as xgb
import lightgbm as lgb

# JAX/Flax imports (if available)
try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import optax
    from flax.training import train_state
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    logger = logging.getLogger('StockPredictionPro.TrainModels')
    logger.warning("JAX/Flax not available. JAX models will be skipped.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockPredictionPro.TrainModels')

# Configuration
DATA_DIR = Path('./data/processed')
MODELS_DIR = Path('./models/trained')
EXPERIMENTS_DIR = Path('./models/experiments')
LOGS_DIR = Path('./logs')

# Ensure directories exist
for dir_path in [MODELS_DIR, EXPERIMENTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Data settings
    target_column: str = 'close'
    feature_columns: List[str] = None
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Training settings
    cv_folds: int = 5
    n_trials: int = 50  # For hyperparameter optimization
    timeout: int = 3600  # 1 hour timeout per model
    
    # Model selection
    train_xgboost: bool = True
    train_lightgbm: bool = True
    train_neural_network: bool = True
    
    # Feature engineering
    create_lags: bool = True
    lag_periods: List[int] = None
    create_technical_indicators: bool = True
    
    # Evaluation
    scoring_metric: str = 'rmse'  # rmse, mae, r2
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = ['open', 'high', 'low', 'volume', 'returns']
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 5, 10]

@dataclass
class ModelResult:
    """Results from training a single model"""
    model_name: str
    model_type: str
    train_score: float
    validation_score: float
    test_score: float
    cv_scores: List[float]
    best_params: Dict[str, Any]
    training_time: float
    model_path: str
    feature_importance: Dict[str, float] = None
    
    @property
    def cv_mean(self) -> float:
        return np.mean(self.cv_scores)
    
    @property
    def cv_std(self) -> float:
        return np.std(self.cv_scores)

@dataclass
class TrainingReport:
    """Comprehensive training report"""
    training_timestamp: str
    config: TrainingConfig
    dataset_info: Dict[str, Any]
    model_results: List[ModelResult]
    best_model: str
    total_training_time: float
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

# ============================================
# FEATURE ENGINEERING
# ============================================

class FeatureEngineer:
    """Feature engineering for financial time series"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        logger.info("🔧 Creating features...")
        
        df_features = df.copy()
        
        # Create lag features
        if self.config.create_lags:
            df_features = self._create_lag_features(df_features)
        
        # Create technical indicators
        if self.config.create_technical_indicators:
            df_features = self._create_technical_indicators(df_features)
        
        # Create date/time features
        df_features = self._create_temporal_features(df_features)
        
        # Remove rows with NaN (due to lags/rolling calculations)
        initial_rows = len(df_features)
        df_features = df_features.dropna()
        final_rows = len(df_features)
        
        logger.info(f"✅ Feature engineering completed: {len(df_features.columns)} features, "
                   f"{initial_rows - final_rows} rows dropped due to NaN")
        
        return df_features
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for time series"""
        target_col = self.config.target_column
        
        for lag in self.config.lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
            # Rolling statistics
            df[f'{target_col}_rolling_mean_{lag}'] = df[target_col].rolling(window=lag).mean()
            df[f'{target_col}_rolling_std_{lag}'] = df[target_col].rolling(window=lag).std()
        
        return df
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis indicators"""
        if 'close' in df.columns:
            # Simple Moving Averages
            for window in [5, 10, 20, 50]:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            
            # Exponential Moving Averages
            for span in [12, 26]:
                df[f'ema_{span}'] = df['close'].ewm(span=span).mean()
            
            # Price ratios
            df['price_sma_ratio_20'] = df['close'] / df['sma_20']
            
            # Volatility
            df['volatility_10'] = df['returns'].rolling(window=10).std() if 'returns' in df.columns else np.nan
            
            # RSI approximation
            if 'returns' in df.columns:
                gains = df['returns'].where(df['returns'] > 0, 0)
                losses = -df['returns'].where(df['returns'] < 0, 0)
                
                avg_gain = gains.rolling(window=14).mean()
                avg_loss = losses.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create date/time based features"""
        if df.index.name == 'timestamp' or 'timestamp' in df.columns:
            if 'timestamp' in df.columns:
                timestamps = pd.to_datetime(df['timestamp'])
            else:
                timestamps = df.index
                
            df['day_of_week'] = timestamps.dayofweek
            df['month'] = timestamps.month
            df['quarter'] = timestamps.quarter
            
            # Cyclical encoding
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df

# ============================================
# MODEL TRAINERS
# ============================================

class XGBoostTrainer:
    """XGBoost model trainer with hyperparameter optimization"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.best_params = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Train XGBoost model with Optuna optimization"""
        logger.info("🚀 Training XGBoost model...")
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.config.random_state
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred, squared=False)  # RMSE
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
        
        self.best_params = study.best_params
        
        # Train final model with best parameters
        final_model = xgb.XGBRegressor(**self.best_params)
        final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        self.model = final_model
        logger.info(f"✅ XGBoost training completed. Best RMSE: {study.best_value:.4f}")
        
        return final_model, self.best_params

class LightGBMTrainer:
    """LightGBM model trainer with hyperparameter optimization"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.best_params = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Train LightGBM model with Optuna optimization"""
        logger.info("🚀 Training LightGBM model...")
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': self.config.random_state,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred, squared=False)  # RMSE
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
        
        self.best_params = study.best_params
        
        # Train final model with best parameters
        final_model = lgb.LGBMRegressor(**self.best_params)
        final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        self.model = final_model
        logger.info(f"✅ LightGBM training completed. Best RMSE: {study.best_value:.4f}")
        
        return final_model, self.best_params

class NeuralNetworkTrainer:
    """JAX/Flax neural network trainer"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.best_params = {}
        self.train_state = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Train JAX/Flax neural network"""
        if not HAS_JAX:
            logger.warning("JAX not available, skipping neural network training")
            return None, {}
        
        logger.info("🚀 Training JAX/Flax Neural Network...")
        
        # Define neural network architecture
        class MLP(nn.Module):
            features: List[int]
            
            @nn.compact
            def __call__(self, x):
                for i, feat in enumerate(self.features[:-1]):
                    x = nn.Dense(feat)(x)
                    x = nn.relu(x)
                    x = nn.Dropout(0.1)(x, deterministic=False)
                x = nn.Dense(self.features[-1])(x)
                return x
        
        # Hyperparameter optimization with Optuna
        def objective(trial):
            # Network architecture
            n_layers = trial.suggest_int('n_layers', 2, 4)
            layer_sizes = []
            for i in range(n_layers):
                layer_sizes.append(trial.suggest_int(f'layer_{i}_size', 32, 256))
            layer_sizes.append(1)  # Output layer
            
            # Training hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            
            # Initialize model
            key = jax.random.PRNGKey(self.config.random_state)
            model = MLP(features=layer_sizes)
            
            # Initialize parameters
            dummy_input = jnp.ones((1, X_train.shape[1]))
            params = model.init(key, dummy_input)
            
            # Optimizer
            optimizer = optax.adam(learning_rate)
            state = train_state.TrainState.create(
                apply_fn=model.apply, params=params, tx=optimizer
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            n_epochs = 100
            for epoch in range(n_epochs):
                # Training step
                for i in range(0, len(X_train), batch_size):
                    batch_X = jnp.array(X_train[i:i+batch_size])
                    batch_y = jnp.array(y_train[i:i+batch_size].reshape(-1, 1))
                    
                    def loss_fn(params):
                        pred = model.apply(params, batch_X)
                        return jnp.mean((pred - batch_y) ** 2)
                    
                    loss_val, grads = jax.value_and_grad(loss_fn)(state.params)
                    state = state.apply_gradients(grads=grads)
                
                # Validation
                if epoch % 5 == 0:
                    val_pred = model.apply(state.params, jnp.array(X_val))
                    val_loss = jnp.mean((val_pred - jnp.array(y_val.reshape(-1, 1))) ** 2)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        break
            
            return float(jnp.sqrt(best_val_loss))  # Return RMSE
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=min(20, self.config.n_trials), timeout=self.config.timeout)
        
        self.best_params = study.best_params
        
        # Train final model with best parameters
        # (Implementation similar to objective function but with full training)
        logger.info(f"✅ Neural Network training completed. Best RMSE: {study.best_value:.4f}")
        
        return None, self.best_params  # Simplified for template

# ============================================
# MAIN TRAINING PIPELINE
# ============================================

class ModelTrainingPipeline:
    """Main training pipeline coordinating all models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.results = []
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and prepare data for training"""
        logger.info(f"📂 Loading data from {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Feature engineering
        df_features = self.feature_engineer.create_features(df)
        
        # Prepare features and target
        target_col = self.config.target_column
        if target_col not in df_features.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        y = df_features[target_col].values
        
        # Select feature columns (exclude target and non-predictive columns)
        exclude_cols = {target_col, 'symbol'} if 'symbol' in df_features.columns else {target_col}
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        X = df_features[feature_cols].values
        
        logger.info(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_cols
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train/validation/test sets"""
        # Use time series split to maintain temporal order
        test_size = int(len(X) * self.config.test_size)
        val_size = int(len(X) * self.config.validation_size)
        
        # Split sequentially for time series
        X_temp, X_test = X[:-test_size], X[-test_size:]
        y_temp, y_test = y[:-test_size], y[-test_size:]
        
        X_train, X_val = X_temp[:-val_size], X_temp[-val_size:]
        y_train, y_val = y_temp[:-val_size], y_temp[-val_size:]
        
        logger.info(f"📊 Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def evaluate_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray, 
                      feature_names: List[str]) -> Tuple[float, float, float, List[float], Dict[str, float]]:
        """Evaluate model performance"""
        
        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Scores
        train_score = mean_squared_error(y_train, train_pred, squared=False)
        val_score = mean_squared_error(y_val, val_pred, squared=False)
        test_score = mean_squared_error(y_test, test_pred, squared=False)
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                                  scoring='neg_mean_squared_error')
        cv_scores = np.sqrt(-cv_scores)  # Convert to RMSE
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(feature_names, importance))
        
        return train_score, val_score, test_score, cv_scores.tolist(), feature_importance
    
    def train_all_models(self, data_path: str) -> TrainingReport:
        """Train all configured models"""
        logger.info("🚀 Starting comprehensive model training pipeline...")
        start_time = datetime.now()
        
        # Load and prepare data
        X, y, feature_names = self.load_and_prepare_data(data_path)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Dataset info
        dataset_info = {
            'data_path': data_path,
            'total_samples': len(X),
            'n_features': len(feature_names),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_names': feature_names
        }
        
        # Train models
        models_to_train = []
        if self.config.train_xgboost:
            models_to_train.append(('XGBoost', XGBoostTrainer(self.config)))
        if self.config.train_lightgbm:
            models_to_train.append(('LightGBM', LightGBMTrainer(self.config)))
        if self.config.train_neural_network and HAS_JAX:
            models_to_train.append(('NeuralNetwork', NeuralNetworkTrainer(self.config)))
        
        for model_name, trainer in models_to_train:
            try:
                model_start_time = datetime.now()
                logger.info(f"📈 Training {model_name}...")
                
                # Train model
                model, best_params = trainer.train(X_train_scaled, y_train, X_val_scaled, y_val)
                
                if model is not None:
                    # Evaluate model
                    train_score, val_score, test_score, cv_scores, feature_importance = self.evaluate_model(
                        model, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, feature_names
                    )
                    
                    # Save model
                    model_path = MODELS_DIR / f"{model_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    joblib.dump(model, model_path)
                    
                    # Save scaler
                    scaler_path = MODELS_DIR / f"scaler_{model_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    joblib.dump(self.scaler, scaler_path)
                    
                    training_time = (datetime.now() - model_start_time).total_seconds()
                    
                    # Create result
                    result = ModelResult(
                        model_name=model_name,
                        model_type=type(model).__name__,
                        train_score=train_score,
                        validation_score=val_score,
                        test_score=test_score,
                        cv_scores=cv_scores,
                        best_params=best_params,
                        training_time=training_time,
                        model_path=str(model_path),
                        feature_importance=feature_importance
                    )
                    
                    self.results.append(result)
                    
                    logger.info(f"✅ {model_name} completed - Test RMSE: {test_score:.4f}, CV RMSE: {np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                continue
        
        # Determine best model
        best_model = min(self.results, key=lambda x: x.test_score).model_name if self.results else "None"
        
        # Create training report
        total_time = (datetime.now() - start_time).total_seconds()
        report = TrainingReport(
            training_timestamp=datetime.now().isoformat(),
            config=self.config,
            dataset_info=dataset_info,
            model_results=self.results,
            best_model=best_model,
            total_training_time=total_time
        )
        
        # Save report
        report_path = EXPERIMENTS_DIR / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            f.write(report.to_json())
        
        logger.info(f"🎉 Training pipeline completed in {total_time:.2f} seconds!")
        logger.info(f"📊 Best model: {best_model}")
        logger.info(f"📄 Report saved to: {report_path}")
        
        return report

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all models for StockPredictionPro')
    parser.add_argument('--data', '-d', required=True, help='Path to processed data CSV file')
    parser.add_argument('--config', '-c', help='Path to training configuration JSON file')
    parser.add_argument('--target', default='close', help='Target column name')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    parser.add_argument('--n-trials', type=int, default=50, help='Hyperparameter optimization trials')
    parser.add_argument('--models', nargs='+', choices=['xgboost', 'lightgbm', 'neural'], 
                       default=['xgboost', 'lightgbm'], help='Models to train')
    parser.add_argument('--timeout', type=int, default=3600, help='Training timeout per model (seconds)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        # Create configuration from arguments
        config = TrainingConfig(
            target_column=args.target,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            n_trials=args.n_trials,
            timeout=args.timeout,
            train_xgboost='xgboost' in args.models,
            train_lightgbm='lightgbm' in args.models,
            train_neural_network='neural' in args.models
        )
    
    # Initialize and run training pipeline
    pipeline = ModelTrainingPipeline(config)
    report = pipeline.train_all_models(args.data)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    print(f"Best Model: {report.best_model}")
    print(f"Total Training Time: {report.total_training_time:.2f} seconds")
    print("\nModel Performance (Test RMSE):")
    print("-" * 40)
    
    for result in sorted(report.model_results, key=lambda x: x.test_score):
        print(f"{result.model_name:15}: {result.test_score:.4f} (CV: {result.cv_mean:.4f}±{result.cv_std:.4f})")
    
    print(f"\n📊 Detailed report saved to experiments directory")

if __name__ == '__main__':
    main()
