# ============================================
# StockPredictionPro - src/models/optimization/optuna_opt.py
# Advanced Optuna optimization with financial domain specializations and multi-objective tuning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
import warnings
import json
from collections import defaultdict
import time
import sqlite3

# Core ML imports
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.base import BaseEstimator, clone

# Optuna imports
try:
    import optuna
    from optuna import Trial, Study
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, NSGAIISampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
    from optuna.visualization import (
        plot_optimization_history, plot_param_importances, 
        plot_contour, plot_slice, plot_parallel_coordinate
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Optuna optimization will use fallback methods.")

# Import our model factory functions
from ..classification.gradient_boosting import create_gradient_boosting_classifier
from ..classification.random_forest import create_random_forest_classifier
from ..classification.svm import create_svm_classifier
from ..classification.logistic import create_logistic_classifier
from ..classification.neural_network import create_neural_network_classifier

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it

logger = get_logger('models.optimization.optuna_opt')

# ============================================
# Optuna Optimization Framework
# ============================================

class OptunaOptimizer:
    """Advanced Optuna optimization with intelligent parameter exploration and pruning"""
    
    def __init__(self,
                 model_factory: Callable,
                 parameter_space: Dict[str, Dict[str, Any]],
                 scoring: str = 'accuracy',
                 cv_folds: int = 5,
                 n_trials: int = 100,
                 timeout: Optional[int] = None,
                 sampler: str = 'TPE',
                 pruner: str = 'median',
                 time_aware_cv: bool = True,
                 study_name: Optional[str] = None,
                 storage: Optional[str] = None,
                 direction: str = 'maximize',
                 random_state: int = 42,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize Optuna Optimizer
        
        Args:
            model_factory: Function that creates model instances
            parameter_space: Dictionary defining parameter search space
            scoring: Scoring metric to optimize
            cv_folds: Number of cross-validation folds
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
            sampler: Sampling algorithm ('TPE', 'Random', 'CmaEs')
            pruner: Pruning algorithm ('median', 'successive_halving', 'hyperband')
            time_aware_cv: Whether to use time-aware cross-validation
            study_name: Name for the Optuna study
            storage: Database URL for persistent storage
            direction: Optimization direction ('maximize' or 'minimize')
            random_state: Random seed
            verbose: Whether to print progress
        """
        self.model_factory = model_factory
        self.parameter_space = parameter_space
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.timeout = timeout
        self.sampler_name = sampler
        self.pruner_name = pruner
        self.time_aware_cv = time_aware_cv
        self.study_name = study_name or f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        self.direction = direction
        self.random_state = random_state
        self.verbose = verbose
        
        # Optimization results
        self.study_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        self.optimization_history_ = []
        self.optuna_analysis_ = None
        
        # Validation
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Using fallback random search.")
        
        self._validate_parameters()
        
        logger.info(f"Initialized Optuna optimizer: {self.study_name}")
    
    def _validate_parameters(self):
        """Validate Optuna parameters"""
        valid_samplers = ['TPE', 'Random', 'CmaEs']
        if self.sampler_name not in valid_samplers:
            raise ValueError(f"sampler must be one of {valid_samplers}")
        
        valid_pruners = ['median', 'successive_halving', 'hyperband', 'none']
        if self.pruner_name not in valid_pruners:
            raise ValueError(f"pruner must be one of {valid_pruners}")
        
        valid_directions = ['maximize', 'minimize']
        if self.direction not in valid_directions:
            raise ValueError(f"direction must be one of {valid_directions}")
        
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
    
    def _create_sampler(self):
        """Create Optuna sampler"""
        if not OPTUNA_AVAILABLE:
            return None
        
        if self.sampler_name == 'TPE':
            return TPESampler(seed=self.random_state)
        elif self.sampler_name == 'Random':
            return RandomSampler(seed=self.random_state)
        elif self.sampler_name == 'CmaEs':
            return CmaEsSampler(seed=self.random_state)
        else:
            return TPESampler(seed=self.random_state)
    
    def _create_pruner(self):
        """Create Optuna pruner"""
        if not OPTUNA_AVAILABLE:
            return None
        
        if self.pruner_name == 'median':
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.pruner_name == 'successive_halving':
            return SuccessiveHalvingPruner()
        elif self.pruner_name == 'hyperband':
            return HyperbandPruner()
        elif self.pruner_name == 'none':
            return None
        else:
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    def _create_cv_splitter(self, X: np.ndarray, y: np.ndarray):
        """Create appropriate cross-validation splitter"""
        if self.time_aware_cv:
            # Use TimeSeriesSplit for financial data
            return TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            # Use standard CV
            if len(np.unique(y)) <= 10:  # Classification
                return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            else:  # Regression
                return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
    
    def _suggest_parameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest parameters using Optuna trial"""
        suggested_params = {}
        
        for param_name, param_config in self.parameter_space.items():
            param_type = param_config.get('type', 'categorical')
            
            if param_type == 'float':
                low = param_config['low']
                high = param_config['high']
                log = param_config.get('log', False)
                step = param_config.get('step', None)
                
                if step is not None:
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, low, high, step=step, log=log
                    )
                else:
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, low, high, log=log
                    )
            
            elif param_type == 'int':
                low = param_config['low']
                high = param_config['high']
                step = param_config.get('step', 1)
                log = param_config.get('log', False)
                
                suggested_params[param_name] = trial.suggest_int(
                    param_name, low, high, step=step, log=log
                )
            
            elif param_type == 'categorical':
                choices = param_config['choices']
                suggested_params[param_name] = trial.suggest_categorical(
                    param_name, choices
                )
            
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        return suggested_params
    
    def _objective_function(self, trial: Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for Optuna optimization"""
        
        # Suggest parameters
        params = self._suggest_parameters(trial)
        
        try:
            # Create model with suggested parameters
            model = self.model_factory(**params)
            
            # Create CV splitter
            cv_splitter = self._create_cv_splitter(X, y)
            
            # Perform cross-validation with pruning support
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit model
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                # Evaluate
                if self.scoring == 'accuracy':
                    val_pred = model_clone.predict(X_val)
                    fold_score = accuracy_score(y_val, val_pred)
                elif self.scoring == 'roc_auc':
                    val_pred_proba = model_clone.predict_proba(X_val)[:, 1]
                    fold_score = roc_auc_score(y_val, val_pred_proba)
                elif self.scoring == 'r2':
                    val_pred = model_clone.predict(X_val)
                    fold_score = r2_score(y_val, val_pred)
                else:
                    # Use sklearn cross_val_score for other metrics
                    scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=self.scoring, n_jobs=1)
                    return np.mean(scores)
                
                cv_scores.append(fold_score)
                
                # Report intermediate score for pruning
                intermediate_score = np.mean(cv_scores)
                trial.report(intermediate_score, fold)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Calculate final score
            final_score = np.mean(cv_scores)
            score_std = np.std(cv_scores)
            
            # Store trial information
            trial_info = {
                'trial_number': trial.number,
                'params': params.copy(),
                'score': final_score,
                'score_std': score_std,
                'cv_scores': cv_scores.copy(),
                'timestamp': datetime.now().isoformat(),
                'state': 'COMPLETE'
            }
            self.optimization_history_.append(trial_info)
            
            if self.verbose:
                logger.info(f"Trial {trial.number}: {final_score:.4f} ± {score_std:.4f} with {params}")
            
            return final_score
            
        except optuna.TrialPruned:
            # Record pruned trial
            trial_info = {
                'trial_number': trial.number,
                'params': params.copy(),
                'score': None,
                'score_std': None,
                'cv_scores': [],
                'timestamp': datetime.now().isoformat(),
                'state': 'PRUNED'
            }
            self.optimization_history_.append(trial_info)
            
            if self.verbose:
                logger.info(f"Trial {trial.number}: PRUNED with {params}")
            
            raise
            
        except Exception as e:
            logger.warning(f"Error in trial {trial.number} with params {params}: {e}")
            
            # Record failed trial
            trial_info = {
                'trial_number': trial.number,
                'params': params.copy(),
                'score': None,
                'score_std': None,
                'cv_scores': [],
                'timestamp': datetime.now().isoformat(),
                'state': 'FAILED',
                'error': str(e)
            }
            self.optimization_history_.append(trial_info)
            
            # Return worst possible score
            return -np.inf if self.direction == 'maximize' else np.inf
    
    @time_it("optuna_optimization", include_args=True)
    def optimize(self, X: Union[pd.DataFrame, np.ndarray], 
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Run Optuna optimization"""
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        logger.info(f"Starting Optuna optimization with {self.n_trials} trials")
        
        if not OPTUNA_AVAILABLE:
            return self._fallback_optimization(X, y)
        
        try:
            # Create sampler and pruner
            sampler = self._create_sampler()
            pruner = self._create_pruner()
            
            # Create or load study
            if self.storage:
                self.study_ = optuna.create_study(
                    study_name=self.study_name,
                    storage=self.storage,
                    direction=self.direction,
                    sampler=sampler,
                    pruner=pruner,
                    load_if_exists=True
                )
            else:
                self.study_ = optuna.create_study(
                    direction=self.direction,
                    sampler=sampler,
                    pruner=pruner
                )
            
            # Define objective with data binding
            def objective(trial):
                return self._objective_function(trial, X, y)
            
            # Run optimization
            self.study_.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=self.verbose
            )
            
            # Extract best results
            self.best_params_ = self.study_.best_params.copy()
            self.best_score_ = self.study_.best_value
            
            # Create best model
            self.best_model_ = self.model_factory(**self.best_params_)
            self.best_model_.fit(X, y)
            
            # Analyze results
            self._analyze_optuna_results()
            
            logger.info(f"Optuna optimization completed. Best score: {self.best_score_:.4f}")
            
            return self._create_optimization_results()
            
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}")
            return self._fallback_optimization(X, y)
    
    def _fallback_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fallback to random search if Optuna fails"""
        logger.info("Using fallback random search optimization")
        
        from .random_search import RandomSearchOptimizer
        
        # Convert parameter space to random search format
        random_param_space = {}
        for param_name, param_config in self.parameter_space.items():
            param_type = param_config.get('type', 'categorical')
            
            if param_type == 'float':
                random_param_space[param_name] = (param_config['low'], param_config['high'])
            elif param_type == 'int':
                random_param_space[param_name] = list(range(param_config['low'], param_config['high'] + 1))
            elif param_type == 'categorical':
                random_param_space[param_name] = param_config['choices']
        
        # Run random search
        random_optimizer = RandomSearchOptimizer(
            model_factory=self.model_factory,
            parameter_space=random_param_space,
            scoring=self.scoring,
            cv_folds=self.cv_folds,
            n_iter=min(self.n_trials, 50),
            random_state=self.random_state
        )
        
        results = random_optimizer.optimize(X, y)
        
        # Update our results
        self.best_params_ = results['best_params']
        self.best_score_ = results['best_score']
        self.best_model_ = results['best_model']
        self.optimization_history_ = results.get('optimization_history', [])
        
        return results
    
    def _analyze_optuna_results(self):
        """Analyze Optuna optimization results"""
        
        if not self.study_:
            return
        
        # Basic study statistics
        completed_trials = [t for t in self.study_.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in self.study_.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in self.study_.trials if t.state == optuna.trial.TrialState.FAIL]
        
        self.optuna_analysis_ = {
            'n_trials_total': len(self.study_.trials),
            'n_trials_completed': len(completed_trials),
            'n_trials_pruned': len(pruned_trials),
            'n_trials_failed': len(failed_trials),
            'best_trial': self.study_.best_trial.number,
            'best_score': self.best_score_,
            'best_params': self.best_params_.copy(),
            'sampler': self.sampler_name,
            'pruner': self.pruner_name
        }
        
        if completed_trials:
            # Score statistics
            scores = [t.value for t in completed_trials]
            self.optuna_analysis_['score_statistics'] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores)),
                'q25': float(np.percentile(scores, 25)),
                'q75': float(np.percentile(scores, 75))
            }
            
            # Parameter importance (if available)
            try:
                param_importance = optuna.importance.get_param_importances(self.study_)
                self.optuna_analysis_['parameter_importance'] = {
                    k: float(v) for k, v in param_importance.items()
                }
            except Exception as e:
                logger.warning(f"Could not compute parameter importance: {e}")
                self.optuna_analysis_['parameter_importance'] = {}
        
        # Convergence analysis
        if len(completed_trials) > 1:
            # Best score so far over trials
            best_scores_so_far = []
            current_best = -np.inf if self.direction == 'maximize' else np.inf
            
            for trial in self.study_.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    if self.direction == 'maximize':
                        current_best = max(current_best, trial.value)
                    else:
                        current_best = min(current_best, trial.value)
                best_scores_so_far.append(current_best)
            
            self.optuna_analysis_['convergence_analysis'] = {
                'best_scores_progression': best_scores_so_far,
                'improvement_rate': self._calculate_improvement_rate(best_scores_so_far),
                'convergence_detected': self._detect_convergence(best_scores_so_far)
            }
    
    def _calculate_improvement_rate(self, scores: List[float]) -> float:
        """Calculate improvement rate over trials"""
        if len(scores) < 2:
            return 0.0
        
        # Calculate improvement per trial
        improvements = []
        for i in range(1, len(scores)):
            improvement = scores[i] - scores[i-1]
            improvements.append(improvement)
        
        # Return average improvement rate
        return float(np.mean(improvements))
    
    def _detect_convergence(self, scores: List[float], window: int = 20, threshold: float = 1e-4) -> bool:
        """Detect if optimization has converged"""
        if len(scores) < window:
            return False
        
        # Check if improvement in last window is below threshold
        recent_improvement = scores[-1] - scores[-window]
        return abs(recent_improvement) < threshold
    
    def _create_optimization_results(self) -> Dict[str, Any]:
        """Create comprehensive optimization results"""
        
        results = {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'best_model': self.best_model_,
            'optimization_history': self.optimization_history_,
            'optuna_analysis': self.optuna_analysis_,
            'optimization_method': 'optuna',
            'study_name': self.study_name,
            'sampler': self.sampler_name,
            'pruner': self.pruner_name
        }
        
        # Add study object if available
        if self.study_:
            results['study'] = self.study_
            results['trials_dataframe'] = self.study_.trials_dataframe()
        
        return results
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance from Optuna analysis"""
        if self.optuna_analysis_ and 'parameter_importance' in self.optuna_analysis_:
            return self.optuna_analysis_['parameter_importance'].copy()
        return {}
    
    def get_trial_history(self) -> pd.DataFrame:
        """Get trial history as DataFrame"""
        if self.study_:
            return self.study_.trials_dataframe()
        else:
            # Create DataFrame from optimization history
            if self.optimization_history_:
                return pd.DataFrame(self.optimization_history_)
            else:
                return pd.DataFrame()
    
    def plot_optimization_analysis(self) -> List[Any]:
        """Create Optuna visualization plots"""
        if not OPTUNA_AVAILABLE or not self.study_:
            logger.warning("Optuna study not available for plotting")
            return []
        
        plots = []
        
        try:
            # Optimization history
            fig1 = plot_optimization_history(self.study_)
            fig1.update_layout(title=f"Optimization History - {self.study_name}")
            plots.append(('optimization_history', fig1))
            
            # Parameter importance
            if len(self.study_.trials) > 1:
                fig2 = plot_param_importances(self.study_)
                fig2.update_layout(title=f"Parameter Importance - {self.study_name}")
                plots.append(('parameter_importance', fig2))
            
            # Parameter relationships (for top 2 parameters)
            param_importance = self.get_parameter_importance()
            if len(param_importance) >= 2:
                top_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)[:2]
                param_names = [p[0] for p in top_params]
                
                # Contour plot
                try:
                    fig3 = plot_contour(self.study_, params=param_names)
                    fig3.update_layout(title=f"Parameter Contour - {self.study_name}")
                    plots.append(('contour', fig3))
                except Exception as e:
                    logger.warning(f"Could not create contour plot: {e}")
                
                # Slice plot
                try:
                    fig4 = plot_slice(self.study_, params=param_names)
                    fig4.update_layout(title=f"Parameter Slice - {self.study_name}")
                    plots.append(('slice', fig4))
                except Exception as e:
                    logger.warning(f"Could not create slice plot: {e}")
            
            # Parallel coordinate plot
            if len(self.parameter_space) > 1:
                try:
                    fig5 = plot_parallel_coordinate(self.study_)
                    fig5.update_layout(title=f"Parallel Coordinate - {self.study_name}")
                    plots.append(('parallel_coordinate', fig5))
                except Exception as e:
                    logger.warning(f"Could not create parallel coordinate plot: {e}")
            
        except Exception as e:
            logger.warning(f"Error creating Optuna plots: {e}")
        
        return plots
    
    def save_study(self, filepath: str):
        """Save Optuna study to file"""
        if not self.study_:
            logger.warning("No study to save")
            return
        
        try:
            # Save study using joblib
            import joblib
            joblib.dump(self.study_, filepath)
            logger.info(f"Study saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving study: {e}")
    
    def load_study(self, filepath: str):
        """Load Optuna study from file"""
        try:
            import joblib
            self.study_ = joblib.load(filepath)
            
            # Update results
            if self.study_.best_trial:
                self.best_params_ = self.study_.best_params.copy()
                self.best_score_ = self.study_.best_value
                
                # Recreate best model
                self.best_model_ = self.model_factory(**self.best_params_)
            
            logger.info(f"Study loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading study: {e}")

# ============================================
# Multi-Objective Optuna Optimization
# ============================================

class MultiObjectiveOptunaOptimizer:
    """Multi-objective optimization using Optuna's NSGA-II implementation"""
    
    def __init__(self,
                 model_factory: Callable,
                 parameter_space: Dict[str, Dict[str, Any]],
                 objectives: List[str] = ['accuracy', 'precision'],
                 directions: List[str] = ['maximize', 'maximize'],
                 cv_folds: int = 5,
                 n_trials: int = 100,
                 study_name: Optional[str] = None,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Multi-Objective Optuna Optimizer
        
        Args:
            model_factory: Function that creates model instances
            parameter_space: Dictionary defining parameter search space
            objectives: List of objective metrics
            directions: Optimization directions for each objective
            cv_folds: Number of cross-validation folds
            n_trials: Number of optimization trials
            study_name: Name for the Optuna study
            random_state: Random seed
        """
        self.model_factory = model_factory
        self.parameter_space = parameter_space
        self.objectives = objectives
        self.directions = directions
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.study_name = study_name or f"multi_objective_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.random_state = random_state
        
        # Results
        self.study_ = None
        self.pareto_front_ = []
        self.optimization_history_ = []
        
        # Validation
        if len(objectives) != len(directions):
            raise ValueError("Number of objectives must match number of directions")
        
        logger.info(f"Initialized multi-objective Optuna optimizer with objectives: {objectives}")
    
    def _multi_objective_function(self, trial: Trial, X: np.ndarray, y: np.ndarray) -> Tuple[float, ...]:
        """Multi-objective function for Optuna"""
        
        # Suggest parameters (reuse from single-objective)
        single_optimizer = OptunaOptimizer(
            model_factory=self.model_factory,
            parameter_space=self.parameter_space
        )
        params = single_optimizer._suggest_parameters(trial)
        
        try:
            # Create model
            model = self.model_factory(**params)
            
            # Cross-validation
            cv_splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            # Evaluate each objective
            objective_scores = []
            for objective in self.objectives:
                scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=objective, n_jobs=1)
                objective_scores.append(np.mean(scores))
            
            # Store trial information
            trial_info = {
                'trial_number': trial.number,
                'params': params.copy(),
                'objective_scores': objective_scores.copy(),
                'objectives': self.objectives.copy(),
                'timestamp': datetime.now().isoformat()
            }
            self.optimization_history_.append(trial_info)
            
            return tuple(objective_scores)
            
        except Exception as e:
            logger.warning(f"Error in multi-objective trial {trial.number}: {e}")
            # Return worst possible scores
            worst_scores = []
            for direction in self.directions:
                if direction == 'maximize':
                    worst_scores.append(-np.inf)
                else:
                    worst_scores.append(np.inf)
            return tuple(worst_scores)
    
    def optimize(self, X: Union[pd.DataFrame, np.ndarray], 
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Run multi-objective Optuna optimization"""
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        logger.info(f"Starting multi-objective Optuna optimization")
        
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not available for multi-objective optimization")
            raise RuntimeError("Optuna required for multi-objective optimization")
        
        try:
            # Create multi-objective study
            self.study_ = optuna.create_study(
                directions=self.directions,
                sampler=NSGAIISampler(seed=self.random_state),
                study_name=self.study_name
            )
            
            # Define objective with data binding
            def objective(trial):
                return self._multi_objective_function(trial, X, y)
            
            # Run optimization
            self.study_.optimize(objective, n_trials=self.n_trials)
            
            # Extract Pareto front
            self.pareto_front_ = []
            for trial in self.study_.best_trials:
                solution = {
                    'params': trial.params.copy(),
                    'objective_values': trial.values.copy(),
                    'trial_number': trial.number
                }
                self.pareto_front_.append(solution)
            
            logger.info(f"Multi-objective optimization completed. Found {len(self.pareto_front_)} Pareto optimal solutions")
            
            return self._create_multi_objective_results()
            
        except Exception as e:
            logger.error(f"Multi-objective Optuna optimization failed: {e}")
            raise
    
    def _create_multi_objective_results(self) -> Dict[str, Any]:
        """Create multi-objective optimization results"""
        
        results = {
            'pareto_front': self.pareto_front_,
            'optimization_history': self.optimization_history_,
            'objectives': self.objectives,
            'directions': self.directions,
            'n_pareto_solutions': len(self.pareto_front_),
            'study': self.study_,
            'optimization_method': 'multi_objective_optuna'
        }
        
        if self.optimization_history_:
            # Analyze objective trade-offs
            all_scores = [trial['objective_scores'] for trial in self.optimization_history_]
            all_scores = np.array(all_scores)
            
            results['objective_analysis'] = {
                'objective_means': np.mean(all_scores, axis=0).tolist(),
                'objective_stds': np.std(all_scores, axis=0).tolist(),
                'objective_correlations': np.corrcoef(all_scores.T).tolist(),
                'pareto_front_size': len(self.pareto_front_)
            }
        
        return results

# ============================================
# Financial Domain Specialized Optimizer
# ============================================

class FinancialOptunaOptimizer(OptunaOptimizer):
    """Optuna optimizer specialized for financial models with domain-specific objectives"""
    
    def __init__(self,
                 model_factory: Callable,
                 parameter_space: Dict[str, Dict[str, Any]],
                 financial_objectives: List[str] = ['accuracy', 'precision', 'recall'],
                 risk_metrics: List[str] = ['sharpe_ratio', 'max_drawdown'],
                 volatility_penalty: float = 0.1,
                 **kwargs):
        """
        Initialize Financial Optuna Optimizer
        
        Args:
            model_factory: Function that creates model instances
            parameter_space: Dictionary defining parameter search space
            financial_objectives: Financial-specific objectives
            risk_metrics: Risk-based metrics to consider
            volatility_penalty: Penalty for high prediction volatility
        """
        super().__init__(model_factory, parameter_space, **kwargs)
        
        self.financial_objectives = financial_objectives
        self.risk_metrics = risk_metrics
        self.volatility_penalty = volatility_penalty
        
    def _calculate_financial_score(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate comprehensive financial score with risk adjustments"""
        
        # Get predictions and probabilities
        predictions = model.predict(X)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            confidence = np.max(probabilities, axis=1)
        else:
            confidence = np.ones(len(predictions))
        
        # Base accuracy
        base_score = accuracy_score(y, predictions)
        
        # Risk adjustments
        
        # 1. Volatility penalty (lower prediction variance is better)
        confidence_volatility = np.std(confidence)
        volatility_adjustment = -confidence_volatility * self.volatility_penalty
        
        # 2. Consistency bonus (stable predictions across folds)
        prediction_variance = np.var(predictions.astype(float))
        consistency_bonus = -prediction_variance * 0.05
        
        # 3. Confidence-based adjustment
        mean_confidence = np.mean(confidence)
        confidence_adjustment = (mean_confidence - 0.5) * 0.1
        
        # 4. Financial domain penalty for extreme predictions
        extreme_penalty = 0
        if hasattr(model, 'predict_proba'):
            # Penalize overconfident predictions (common in financial markets)
            overconfident_ratio = np.mean(np.max(probabilities, axis=1) > 0.95)
            extreme_penalty = -overconfident_ratio * 0.05
        
        # Combine all adjustments
        financial_score = (base_score + 
                          volatility_adjustment + 
                          consistency_bonus + 
                          confidence_adjustment + 
                          extreme_penalty)
        
        return max(0, min(1, financial_score))  # Clamp to [0, 1]
    
    def _objective_function(self, trial: Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Financial domain-specific objective function"""
        
        # Suggest parameters
        params = self._suggest_parameters(trial)
        
        try:
            # Create model
            model = self.model_factory(**params)
            
            # Time-series cross-validation for financial data
            cv_splitter = TimeSeriesSplit(n_splits=self.cv_folds)
            
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit model
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                # Calculate financial score
                financial_score = self._calculate_financial_score(model_clone, X_val, y_val)
                cv_scores.append(financial_score)
                
                # Report intermediate score for pruning
                intermediate_score = np.mean(cv_scores)
                trial.report(intermediate_score, fold)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            final_score = np.mean(cv_scores)
            
            # Store financial trial information
            trial_info = {
                'trial_number': trial.number,
                'params': params.copy(),
                'financial_score': final_score,
                'base_accuracy': accuracy_score(y, model.fit(X, y).predict(X)),
                'cv_scores': cv_scores,
                'timestamp': datetime.now().isoformat(),
                'financial_adjustments': True
            }
            self.optimization_history_.append(trial_info)
            
            return final_score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Error in financial trial {trial.number}: {e}")
            return -np.inf

# ============================================
# Factory Functions and Utilities
# ============================================

def optimize_model_optuna(model_name: str,
                         parameter_space: Dict[str, Dict[str, Any]],
                         X: Union[pd.DataFrame, np.ndarray],
                         y: Union[pd.Series, np.ndarray],
                         **optimizer_kwargs) -> Dict[str, Any]:
    """Convenient function to optimize specific model types with Optuna"""
    
    # Model factory functions
    model_factories = {
        'gradient_boosting': create_gradient_boosting_classifier,
        'random_forest': create_random_forest_classifier,
        'svm': create_svm_classifier,
        'logistic': create_logistic_classifier,
        'neural_network': create_neural_network_classifier
    }
    
    if model_name not in model_factories:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_factories.keys())}")
    
    model_factory = model_factories[model_name]
    
    # Create optimizer
    optimizer = OptunaOptimizer(
        model_factory=model_factory,
        parameter_space=parameter_space,
        **optimizer_kwargs
    )
    
    # Run optimization
    results = optimizer.optimize(X, y)
    
    return results

def create_financial_optuna_spaces(model_name: str) -> Dict[str, Dict[str, Any]]:
    """Create Optuna parameter spaces optimized for financial models"""
    
    spaces = {
        'gradient_boosting': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500, 'log': False},
            'max_depth': {'type': 'int', 'low': 3, 'high': 15, 'log': False},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0, 'log': False},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20, 'log': False},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10, 'log': False}
        },
        'random_forest': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300, 'log': False},
            'max_depth': {'type': 'int', 'low': 5, 'high': 25, 'log': False},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 15, 'log': False},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 8, 'log': False},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', 0.5, 0.8]}
        },
        'svm': {
            'C': {'type': 'float', 'low': 0.1, 'high': 100, 'log': True},
            'gamma': {'type': 'float', 'low': 1e-4, 'high': 1, 'log': True},
            'kernel': {'type': 'categorical', 'choices': ['rbf', 'poly', 'sigmoid']}
        },
        'logistic': {
            'C': {'type': 'float', 'low': 0.01, 'high': 100, 'log': True},
            'penalty': {'type': 'categorical', 'choices': ['l1', 'l2', 'elasticnet']},
            'solver': {'type': 'categorical', 'choices': ['liblinear', 'saga', 'lbfgs']}
        },
        'neural_network': {
            'hidden_layer_sizes': {'type': 'categorical', 'choices': [(50,), (100,), (100, 50), (150, 100, 50)]},
            'alpha': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'log': True},
            'learning_rate_init': {'type': 'float', 'low': 1e-4, 'high': 1e-1, 'log': True},
            'max_iter': {'type': 'int', 'low': 100, 'high': 1000, 'log': False}
        }
    }
    
    if model_name not in spaces:
        raise ValueError(f"No Optuna parameter space for model: {model_name}")
    
    return spaces[model_name]

def compare_optuna_strategies(X: Union[pd.DataFrame, np.ndarray],
                            y: Union[pd.Series, np.ndarray],
                            model_factory: Callable,
                            parameter_space: Dict[str, Dict[str, Any]],
                            strategies: Dict[str, Dict] = None) -> Dict[str, Any]:
    """Compare different Optuna optimization strategies"""
    
    if strategies is None:
        strategies = {
            'tpe_median': {'sampler': 'TPE', 'pruner': 'median'},
            'tpe_successive': {'sampler': 'TPE', 'pruner': 'successive_halving'},
            'tpe_hyperband': {'sampler': 'TPE', 'pruner': 'hyperband'},
            'random_median': {'sampler': 'Random', 'pruner': 'median'},
            'cmaes_none': {'sampler': 'CmaEs', 'pruner': 'none'}
        }
    
    results = {}
    
    for strategy_name, strategy_config in strategies.items():
        logger.info(f"Testing Optuna strategy: {strategy_name}")
        
        try:
            optimizer = OptunaOptimizer(
                model_factory=model_factory,
                parameter_space=parameter_space,
                n_trials=50,  # Reduced for comparison
                verbose=False,
                **strategy_config
            )
            
            strategy_results = optimizer.optimize(X, y)
            
            results[strategy_name] = {
                'best_score': strategy_results['best_score'],
                'best_params': strategy_results['best_params'],
                'n_trials_completed': strategy_results['optuna_analysis']['n_trials_completed'],
                'n_trials_pruned': strategy_results['optuna_analysis']['n_trials_pruned'],
                'parameter_importance': strategy_results['optuna_analysis'].get('parameter_importance', {}),
                'strategy_config': strategy_config
            }
            
        except Exception as e:
            logger.warning(f"Error with strategy {strategy_name}: {e}")
            results[strategy_name] = {'error': str(e)}
    
    # Find best strategy
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_strategy = max(valid_results.keys(), key=lambda k: valid_results[k]['best_score'])
        
        results['comparison'] = {
            'best_strategy': best_strategy,
            'strategy_rankings': sorted(valid_results.keys(), 
                                      key=lambda k: valid_results[k]['best_score'], reverse=True),
            'pruning_efficiency': {
                strategy: {
                    'pruning_rate': v['n_trials_pruned'] / (v['n_trials_completed'] + v['n_trials_pruned']),
                    'efficiency_score': v['best_score'] / max(1, v['n_trials_completed'] + v['n_trials_pruned'])
                }
                for strategy, v in valid_results.items()
            }
        }
    
    return results
