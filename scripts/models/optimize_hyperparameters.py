"""
scripts/models/optimize_hyperparameters.py

Advanced hyperparameter optimization for StockPredictionPro models.
Supports multiple optimization strategies: Optuna, GridSearch, RandomSearch, and Bayesian optimization.
Includes early stopping, cross-validation, and automated model comparison.

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
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, 
    TimeSeriesSplit, StratifiedKFold
)
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner

# ML Models
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Advanced optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockPredictionPro.HyperparameterOptimization')

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Directory configuration
DATA_DIR = Path('./data/processed')
MODELS_DIR = Path('./models/trained')
OPTIMIZATION_DIR = Path('./models/optimization')
LOGS_DIR = Path('./logs')

# Ensure directories exist
for dir_path in [OPTIMIZATION_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    # Model settings
    model_type: str = 'xgboost'  # xgboost, lightgbm, random_forest, ridge, lasso
    target_metric: str = 'rmse'  # rmse, mae, r2, custom
    
    # Optimization settings
    optimization_method: str = 'optuna'  # optuna, grid_search, random_search, bayesian
    n_trials: int = 100
    n_jobs: int = -1
    timeout: int = 3600  # 1 hour
    
    # Cross-validation settings
    cv_method: str = 'time_series'  # time_series, k_fold
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 50
    patience: int = 10
    
    # Advanced settings
    use_pruning: bool = True
    random_state: int = 42
    verbose: bool = True
    
    # Resource limits
    max_memory_gb: float = 8.0
    max_cpu_percent: float = 80.0

@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization"""
    model_type: str
    optimization_method: str
    best_params: Dict[str, Any]
    best_score: float
    best_std: float
    optimization_time: float
    n_trials_completed: int
    cv_results: Dict[str, List[float]]
    study_summary: Dict[str, Any] = None
    model_path: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

# ============================================
# HYPERPARAMETER SEARCH SPACES
# ============================================

class SearchSpaceManager:
    """Manage hyperparameter search spaces for different models"""
    
    @staticmethod
    def get_xgboost_space(method: str = 'optuna') -> Dict[str, Any]:
        """XGBoost hyperparameter search space"""
        if method == 'optuna':
            return {
                'n_estimators': ('suggest_int', 50, 1000),
                'max_depth': ('suggest_int', 3, 12),
                'learning_rate': ('suggest_float', 0.01, 0.3, True),  # log=True
                'subsample': ('suggest_float', 0.6, 1.0),
                'colsample_bytree': ('suggest_float', 0.6, 1.0),
                'reg_alpha': ('suggest_float', 0.0, 10.0),
                'reg_lambda': ('suggest_float', 0.0, 10.0),
                'min_child_weight': ('suggest_int', 1, 10),
                'gamma': ('suggest_float', 0.0, 5.0)
            }
        elif method == 'sklearn':
            return {
                'n_estimators': [50, 100, 200, 500, 1000],
                'max_depth': [3, 4, 5, 6, 7, 8, 10, 12],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0.0, 0.1, 1.0, 10.0],
                'reg_lambda': [0.0, 0.1, 1.0, 10.0]
            }
        elif method == 'bayesian' and HAS_SKOPT:
            return {
                'n_estimators': Integer(50, 1000),
                'max_depth': Integer(3, 12),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'reg_alpha': Real(0.0, 10.0),
                'reg_lambda': Real(0.0, 10.0)
            }
    
    @staticmethod
    def get_lightgbm_space(method: str = 'optuna') -> Dict[str, Any]:
        """LightGBM hyperparameter search space"""
        if method == 'optuna':
            return {
                'n_estimators': ('suggest_int', 50, 1000),
                'num_leaves': ('suggest_int', 20, 300),
                'learning_rate': ('suggest_float', 0.01, 0.3, True),
                'feature_fraction': ('suggest_float', 0.4, 1.0),
                'bagging_fraction': ('suggest_float', 0.4, 1.0),
                'bagging_freq': ('suggest_int', 1, 7),
                'min_child_samples': ('suggest_int', 5, 100),
                'reg_alpha': ('suggest_float', 0.0, 10.0),
                'reg_lambda': ('suggest_float', 0.0, 10.0)
            }
        elif method == 'sklearn':
            return {
                'n_estimators': [50, 100, 200, 500, 1000],
                'num_leaves': [20, 31, 50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                'feature_fraction': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'bagging_fraction': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'min_child_samples': [5, 10, 20, 50, 100]
            }
        elif method == 'bayesian' and HAS_SKOPT:
            return {
                'n_estimators': Integer(50, 1000),
                'num_leaves': Integer(20, 300),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'feature_fraction': Real(0.4, 1.0),
                'bagging_fraction': Real(0.4, 1.0),
                'min_child_samples': Integer(5, 100)
            }
    
    @staticmethod
    def get_random_forest_space(method: str = 'optuna') -> Dict[str, Any]:
        """Random Forest hyperparameter search space"""
        if method == 'optuna':
            return {
                'n_estimators': ('suggest_int', 50, 500),
                'max_depth': ('suggest_int', 3, 20),
                'min_samples_split': ('suggest_int', 2, 20),
                'min_samples_leaf': ('suggest_int', 1, 10),
                'max_features': ('suggest_categorical', ['sqrt', 'log2', None]),
                'bootstrap': ('suggest_categorical', [True, False])
            }
        elif method == 'sklearn':
            return {
                'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 15, 20],
                'min_samples_leaf': [1, 2, 4, 6, 8, 10],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
    
    @staticmethod
    def get_ridge_space(method: str = 'optuna') -> Dict[str, Any]:
        """Ridge regression hyperparameter search space"""
        if method == 'optuna':
            return {
                'alpha': ('suggest_float', 0.001, 100.0, True),
                'fit_intercept': ('suggest_categorical', [True, False]),
                'solver': ('suggest_categorical', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
            }
        elif method == 'sklearn':
            return {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'fit_intercept': [True, False],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }

# ============================================
# OPTIMIZATION STRATEGIES
# ============================================

class OptunaOptimizer:
    """Optuna-based hyperparameter optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.search_space = SearchSpaceManager()
        self.study = None
        self.best_model = None
    
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                model_class: Any, param_space: Dict[str, Any]) -> OptimizationResult:
        """Optimize hyperparameters using Optuna"""
        logger.info(f"🔍 Starting Optuna optimization for {self.config.model_type}")
        
        start_time = time.time()
        
        # Create study
        sampler = TPESampler(seed=self.config.random_state)
        pruner = MedianPruner() if self.config.use_pruning else None
        
        self.study = optuna.create_study(
            direction='minimize' if self.config.target_metric in ['rmse', 'mae'] else 'maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Define objective function
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                method_name = param_config[0]
                args = param_config[1:]
                params[param_name] = getattr(trial, method_name)(param_name, *args)
            
            # Add fixed parameters
            if self.config.model_type == 'xgboost':
                params.update({
                    'objective': 'reg:squarederror',
                    'random_state': self.config.random_state,
                    'n_jobs': 1,  # Use 1 for nested parallelization
                    'verbosity': 0
                })
            elif self.config.model_type == 'lightgbm':
                params.update({
                    'objective': 'regression',
                    'metric': 'rmse',
                    'random_state': self.config.random_state,
                    'n_jobs': 1,
                    'verbosity': -1
                })
            elif self.config.model_type == 'random_forest':
                params.update({
                    'random_state': self.config.random_state,
                    'n_jobs': 1
                })
            
            # Create model
            model = model_class(**params)
            
            # Cross-validation
            cv_scores = self._cross_validate(model, X, y, trial)
            
            return np.mean(cv_scores)
        
        # Run optimization
        self.study.optimize(
            objective, 
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=self.config.verbose
        )
        
        optimization_time = time.time() - start_time
        
        # Train best model
        best_params = self.study.best_params.copy()
        
        # Add fixed parameters for best model
        if self.config.model_type == 'xgboost':
            best_params.update({
                'objective': 'reg:squarederror',
                'random_state': self.config.random_state,
                'verbosity': 0
            })
        elif self.config.model_type == 'lightgbm':
            best_params.update({
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': self.config.random_state,
                'verbosity': -1
            })
        elif self.config.model_type == 'random_forest':
            best_params.update({
                'random_state': self.config.random_state
            })
        
        # Train final model with best parameters
        self.best_model = model_class(**best_params)
        final_cv_scores = self._cross_validate(self.best_model, X, y)
        
        # Save best model
        model_path = self._save_best_model(self.best_model, best_params)
        
        # Create result
        result = OptimizationResult(
            model_type=self.config.model_type,
            optimization_method='optuna',
            best_params=best_params,
            best_score=np.mean(final_cv_scores),
            best_std=np.std(final_cv_scores),
            optimization_time=optimization_time,
            n_trials_completed=len(self.study.trials),
            cv_results={'cv_scores': final_cv_scores.tolist()},
            study_summary=self._get_study_summary(),
            model_path=model_path
        )
        
        logger.info(f"✅ Optuna optimization completed: {result.best_score:.4f} ± {result.best_std:.4f}")
        
        return result
    
    def _cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray, 
                       trial=None) -> np.ndarray:
        """Perform cross-validation with optional pruning"""
        if self.config.cv_method == 'time_series':
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                               random_state=self.config.random_state)
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            if self.config.model_type in ['xgboost', 'lightgbm'] and self.config.enable_early_stopping:
                if self.config.model_type == 'xgboost':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=self.config.early_stopping_rounds,
                        verbose=False
                    )
                else:  # lightgbm
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]
                    )
            else:
                model.fit(X_train, y_train)
            
            # Predict and score
            y_pred = model.predict(X_val)
            
            if self.config.target_metric == 'rmse':
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            elif self.config.target_metric == 'mae':
                score = mean_absolute_error(y_val, y_pred)
            elif self.config.target_metric == 'r2':
                score = r2_score(y_val, y_pred)
            else:
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            
            scores.append(score)
            
            # Pruning for Optuna
            if trial is not None and self.config.use_pruning:
                trial.report(score, fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        return np.array(scores)
    
    def _save_best_model(self, model: Any, params: Dict[str, Any]) -> str:
        """Save the best model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{self.config.model_type}_optuna_optimized_{timestamp}.pkl"
        model_path = OPTIMIZATION_DIR / model_filename
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save parameters
        params_filename = f"{self.config.model_type}_optuna_params_{timestamp}.json"
        params_path = OPTIMIZATION_DIR / params_filename
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2, default=str)
        
        logger.info(f"💾 Saved optimized model: {model_path}")
        
        return str(model_path)
    
    def _get_study_summary(self) -> Dict[str, Any]:
        """Get study summary statistics"""
        if not self.study:
            return {}
        
        return {
            'best_value': self.study.best_value,
            'best_trial_number': self.study.best_trial.number,
            'n_trials': len(self.study.trials),
            'state_counts': {
                'COMPLETE': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'PRUNED': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'FAIL': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
            }
        }

class SklearnOptimizer:
    """Scikit-learn based optimization (Grid/Random Search)"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.search_space = SearchSpaceManager()
        self.search_cv = None
        self.best_model = None
    
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                model_class: Any, param_space: Dict[str, Any]) -> OptimizationResult:
        """Optimize hyperparameters using sklearn methods"""
        method_name = self.config.optimization_method
        logger.info(f"🔍 Starting {method_name} optimization for {self.config.model_type}")
        
        start_time = time.time()
        
        # Create base model
        base_params = {}
        if self.config.model_type == 'xgboost':
            base_params = {
                'objective': 'reg:squarederror',
                'random_state': self.config.random_state,
                'verbosity': 0
            }
        elif self.config.model_type == 'lightgbm':
            base_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': self.config.random_state,
                'verbosity': -1
            }
        elif self.config.model_type == 'random_forest':
            base_params = {'random_state': self.config.random_state}
        
        base_model = model_class(**base_params)
        
        # Create cross-validation strategy
        if self.config.cv_method == 'time_series':
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                               random_state=self.config.random_state)
        
        # Create scorer
        if self.config.target_metric == 'rmse':
            scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))
        elif self.config.target_metric == 'mae':
            scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        elif self.config.target_metric == 'r2':
            scorer = make_scorer(r2_score)
        else:
            scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))
        
        # Choose optimization method
        if method_name == 'grid_search':
            self.search_cv = GridSearchCV(
                base_model,
                param_space,
                cv=cv,
                scoring=scorer,
                n_jobs=self.config.n_jobs,
                verbose=1 if self.config.verbose else 0
            )
        elif method_name == 'random_search':
            self.search_cv = RandomizedSearchCV(
                base_model,
                param_space,
                n_iter=self.config.n_trials,
                cv=cv,
                scoring=scorer,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=1 if self.config.verbose else 0
            )
        
        # Run optimization
        self.search_cv.fit(X, y)
        
        optimization_time = time.time() - start_time
        
        # Get best model
        self.best_model = self.search_cv.best_estimator_
        
        # Save best model
        model_path = self._save_best_model(self.best_model, self.search_cv.best_params_)
        
        # Calculate cross-validation scores for best model
        cv_scores = cross_val_score(self.best_model, X, y, cv=cv, scoring=scorer)
        if self.config.target_metric in ['rmse', 'mae']:
            cv_scores = -cv_scores  # Convert back to positive values
        
        # Create result
        result = OptimizationResult(
            model_type=self.config.model_type,
            optimization_method=method_name,
            best_params=self.search_cv.best_params_,
            best_score=np.mean(cv_scores),
            best_std=np.std(cv_scores),
            optimization_time=optimization_time,
            n_trials_completed=len(self.search_cv.cv_results_['params']),
            cv_results={'cv_scores': cv_scores.tolist()},
            model_path=model_path
        )
        
        logger.info(f"✅ {method_name} optimization completed: {result.best_score:.4f} ± {result.best_std:.4f}")
        
        return result
    
    def _save_best_model(self, model: Any, params: Dict[str, Any]) -> str:
        """Save the best model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{self.config.model_type}_{self.config.optimization_method}_{timestamp}.pkl"
        model_path = OPTIMIZATION_DIR / model_filename
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save parameters
        params_filename = f"{self.config.model_type}_{self.config.optimization_method}_params_{timestamp}.json"
        params_path = OPTIMIZATION_DIR / params_filename
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2, default=str)
        
        logger.info(f"💾 Saved optimized model: {model_path}")
        
        return str(model_path)

class BayesianOptimizer:
    """Bayesian optimization using scikit-optimize"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.search_space = SearchSpaceManager()
        self.search_cv = None
        self.best_model = None
    
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                model_class: Any, param_space: Dict[str, Any]) -> OptimizationResult:
        """Optimize hyperparameters using Bayesian optimization"""
        if not HAS_SKOPT:
            raise ImportError("scikit-optimize required for Bayesian optimization")
        
        logger.info(f"🔍 Starting Bayesian optimization for {self.config.model_type}")
        
        start_time = time.time()
        
        # Create base model
        base_params = {}
        if self.config.model_type == 'xgboost':
            base_params = {
                'objective': 'reg:squarederror',
                'random_state': self.config.random_state,
                'verbosity': 0
            }
        elif self.config.model_type == 'lightgbm':
            base_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': self.config.random_state,
                'verbosity': -1
            }
        
        base_model = model_class(**base_params)
        
        # Create cross-validation strategy
        if self.config.cv_method == 'time_series':
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                               random_state=self.config.random_state)
        
        # Create scorer
        if self.config.target_metric == 'rmse':
            scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))
        elif self.config.target_metric == 'mae':
            scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        elif self.config.target_metric == 'r2':
            scorer = make_scorer(r2_score)
        else:
            scorer = make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))
        
        # Run Bayesian optimization
        self.search_cv = BayesSearchCV(
            base_model,
            param_space,
            n_iter=self.config.n_trials,
            cv=cv,
            scoring=scorer,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbose=1 if self.config.verbose else 0
        )
        
        self.search_cv.fit(X, y)
        
        optimization_time = time.time() - start_time
        
        # Get best model
        self.best_model = self.search_cv.best_estimator_
        
        # Save best model
        model_path = self._save_best_model(self.best_model, self.search_cv.best_params_)
        
        # Calculate cross-validation scores for best model
        cv_scores = cross_val_score(self.best_model, X, y, cv=cv, scoring=scorer)
        if self.config.target_metric in ['rmse', 'mae']:
            cv_scores = -cv_scores  # Convert back to positive values
        
        # Create result
        result = OptimizationResult(
            model_type=self.config.model_type,
            optimization_method='bayesian',
            best_params=self.search_cv.best_params_,
            best_score=np.mean(cv_scores),
            best_std=np.std(cv_scores),
            optimization_time=optimization_time,
            n_trials_completed=self.config.n_trials,
            cv_results={'cv_scores': cv_scores.tolist()},
            model_path=model_path
        )
        
        logger.info(f"✅ Bayesian optimization completed: {result.best_score:.4f} ± {result.best_std:.4f}")
        
        return result
    
    def _save_best_model(self, model: Any, params: Dict[str, Any]) -> str:
        """Save the best model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{self.config.model_type}_bayesian_{timestamp}.pkl"
        model_path = OPTIMIZATION_DIR / model_filename
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save parameters
        params_filename = f"{self.config.model_type}_bayesian_params_{timestamp}.json"
        params_path = OPTIMIZATION_DIR / params_filename
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2, default=str)
        
        logger.info(f"💾 Saved optimized model: {model_path}")
        
        return str(model_path)

# ============================================
# MAIN OPTIMIZATION ORCHESTRATOR
# ============================================

class HyperparameterOptimizer:
    """Main orchestrator for hyperparameter optimization"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.search_space = SearchSpaceManager()
        self.results = []
    
    def optimize_model(self, data_path: str, target_column: str = 'close') -> OptimizationResult:
        """Optimize hyperparameters for specified model"""
        logger.info(f"🚀 Starting hyperparameter optimization for {self.config.model_type}")
        
        # Load and prepare data
        X, y = self._load_and_prepare_data(data_path, target_column)
        
        # Get model class and parameter space
        model_class, param_space = self._get_model_and_space()
        
        # Choose optimization strategy
        if self.config.optimization_method == 'optuna':
            optimizer = OptunaOptimizer(self.config)
        elif self.config.optimization_method in ['grid_search', 'random_search']:
            optimizer = SklearnOptimizer(self.config)
        elif self.config.optimization_method == 'bayesian':
            optimizer = BayesianOptimizer(self.config)
        else:
            raise ValueError(f"Unsupported optimization method: {self.config.optimization_method}")
        
        # Run optimization
        result = optimizer.optimize(X, y, model_class, param_space)
        self.results.append(result)
        
        # Save result
        result_path = OPTIMIZATION_DIR / f"{self.config.model_type}_optimization_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result.save(result_path)
        logger.info(f"📄 Saved optimization result: {result_path}")
        
        return result
    
    def _load_and_prepare_data(self, data_path: str, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare data for optimization"""
        df = pd.read_csv(data_path)
        
        # Handle missing values
        df = df.dropna()
        
        # Prepare features and target
        exclude_cols = {target_column, 'symbol', 'timestamp', 'date'}
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[target_column].values
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Save scaler for later use
        scaler_path = OPTIMIZATION_DIR / f"scaler_{self.config.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"📊 Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def _get_model_and_space(self) -> Tuple[Any, Dict[str, Any]]:
        """Get model class and parameter space"""
        model_classes = {
            'xgboost': xgb.XGBRegressor,
            'lightgbm': lgb.LGBMRegressor,
            'random_forest': RandomForestRegressor,
            'ridge': Ridge,
            'lasso': Lasso
        }
        
        space_methods = {
            'xgboost': self.search_space.get_xgboost_space,
            'lightgbm': self.search_space.get_lightgbm_space,
            'random_forest': self.search_space.get_random_forest_space,
            'ridge': self.search_space.get_ridge_space,
            'lasso': self.search_space.get_ridge_space  # Same as Ridge
        }
        
        if self.config.model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        model_class = model_classes[self.config.model_type]
        
        # Get parameter space based on optimization method
        space_method = self.config.optimization_method
        if space_method in ['grid_search', 'random_search']:
            space_method = 'sklearn'
        elif space_method == 'bayesian':
            space_method = 'bayesian'
        else:  # optuna
            space_method = 'optuna'
        
        param_space = space_methods[self.config.model_type](space_method)
        
        return model_class, param_space
    
    def compare_optimization_methods(self, data_path: str, target_column: str = 'close',
                                   methods: List[str] = None) -> Dict[str, OptimizationResult]:
        """Compare different optimization methods"""
        if methods is None:
            methods = ['optuna', 'random_search']
            if HAS_SKOPT:
                methods.append('bayesian')
        
        logger.info(f"🔄 Comparing optimization methods: {methods}")
        
        results = {}
        original_method = self.config.optimization_method
        
        for method in methods:
            try:
                logger.info(f"Testing {method} optimization...")
                self.config.optimization_method = method
                
                # Reduce trials for comparison
                original_trials = self.config.n_trials
                self.config.n_trials = min(50, self.config.n_trials)
                
                result = self.optimize_model(data_path, target_column)
                results[method] = result
                
                # Restore original settings
                self.config.n_trials = original_trials
                
            except Exception as e:
                logger.error(f"Failed to optimize with {method}: {e}")
                continue
        
        # Restore original method
        self.config.optimization_method = original_method
        
        # Print comparison
        self._print_comparison_results(results)
        
        return results
    
    def _print_comparison_results(self, results: Dict[str, OptimizationResult]) -> None:
        """Print comparison results"""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION COMPARISON")
        print("="*60)
        
        # Sort by best score
        sorted_results = sorted(results.items(), key=lambda x: x[1].best_score, 
                              reverse=self.config.target_metric == 'r2')
        
        for method, result in sorted_results:
            print(f"\n{method.upper()}:")
            print(f"  Best {self.config.target_metric.upper()}: {result.best_score:.4f} ± {result.best_std:.4f}")
            print(f"  Optimization Time: {result.optimization_time:.1f}s")
            print(f"  Trials Completed: {result.n_trials_completed}")
            print(f"  Best Parameters: {json.dumps(result.best_params, indent=4)}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize hyperparameters for StockPredictionPro models')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--model', choices=['xgboost', 'lightgbm', 'random_forest', 'ridge', 'lasso'],
                       default='xgboost', help='Model type to optimize')
    parser.add_argument('--method', choices=['optuna', 'grid_search', 'random_search', 'bayesian'],
                       default='optuna', help='Optimization method')
    parser.add_argument('--target', default='close', help='Target column name')
    parser.add_argument('--metric', choices=['rmse', 'mae', 'r2'], default='rmse', help='Target metric')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    parser.add_argument('--timeout', type=int, default=3600, help='Optimization timeout (seconds)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple optimization methods')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create configuration
    config = OptimizationConfig(
        model_type=args.model,
        optimization_method=args.method,
        target_metric=args.metric,
        n_trials=args.trials,
        cv_folds=args.cv_folds,
        timeout=args.timeout,
        verbose=args.verbose
    )
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(config)
    
    if args.compare:
        # Compare methods
        results = optimizer.compare_optimization_methods(args.data, args.target)
        
        # Find best method
        best_method = min(results.keys(), key=lambda x: results[x].best_score)
        best_score = results[best_method].best_score
        
        print(f"\n🏆 Best method: {best_method} with {args.metric.upper()}: {best_score:.4f}")
        
    else:
        # Single optimization
        result = optimizer.optimize_model(args.data, args.target)
        
        print(f"\n🎉 Optimization completed!")
        print(f"Best {args.metric.upper()}: {result.best_score:.4f} ± {result.best_std:.4f}")
        print(f"Best parameters: {json.dumps(result.best_params, indent=2)}")
        print(f"Model saved to: {result.model_path}")

if __name__ == '__main__':
    main()
