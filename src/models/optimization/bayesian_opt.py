# ============================================
# StockPredictionPro - src/models/optimization/bayesian_opt.py
# Advanced Bayesian optimization for hyperparameter tuning with financial domain optimizations
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
import warnings
import json
from collections import defaultdict

# Core ML imports
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.base import BaseEstimator, clone

# Bayesian optimization imports
try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
    from skopt import dump, load
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn("scikit-optimize not available. Bayesian optimization will use fallback methods.")

# Import our model factory functions
from ..classification.gradient_boosting import create_gradient_boosting_classifier
from ..classification.random_forest import create_random_forest_classifier
from ..classification.svm import create_svm_classifier
from ..classification.logistic import create_logistic_classifier
from ..classification.neural_network import create_neural_network_classifier

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it

logger = get_logger('models.optimization.bayesian_opt')

# ============================================
# Bayesian Optimization Framework
# ============================================

class BayesianOptimizer:
    """Advanced Bayesian optimization with financial domain optimizations"""
    
    def __init__(self,
                 model_factory: Callable,
                 parameter_space: Dict[str, Any],
                 scoring: str = 'accuracy',
                 cv_folds: int = 5,
                 n_calls: int = 100,
                 n_initial_points: int = 10,
                 acquisition_function: str = 'EI',  # Expected Improvement
                 surrogate_model: str = 'gp',  # Gaussian Process
                 time_aware_cv: bool = True,
                 early_stopping_rounds: Optional[int] = None,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize Bayesian Optimizer
        
        Args:
            model_factory: Function that creates model instances
            parameter_space: Dictionary defining parameter search space
            scoring: Scoring metric to optimize
            cv_folds: Number of cross-validation folds
            n_calls: Number of optimization iterations
            n_initial_points: Number of random initialization points
            acquisition_function: Acquisition function ('EI', 'PI', 'LCB')
            surrogate_model: Surrogate model type ('gp', 'rf', 'gbrt')
            time_aware_cv: Whether to use time-aware cross-validation
            early_stopping_rounds: Early stopping for convergence
            random_state: Random seed
            verbose: Whether to print progress
        """
        self.model_factory = model_factory
        self.parameter_space = parameter_space
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        self.surrogate_model = surrogate_model
        self.time_aware_cv = time_aware_cv
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.verbose = verbose
        
        # Optimization results
        self.optimization_result_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        self.optimization_history_ = []
        self.convergence_history_ = []
        
        # Validation
        if not SKOPT_AVAILABLE:
            logger.warning("scikit-optimize not available. Using fallback random search.")
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate optimization parameters"""
        valid_acquisitions = ['EI', 'PI', 'LCB']
        if self.acquisition_function not in valid_acquisitions:
            raise ValueError(f"acquisition_function must be one of {valid_acquisitions}")
        
        valid_surrogates = ['gp', 'rf', 'gbrt']
        if self.surrogate_model not in valid_surrogates:
            raise ValueError(f"surrogate_model must be one of {valid_surrogates}")
        
        if self.n_calls <= 0:
            raise ValueError("n_calls must be positive")
        
        if self.n_initial_points >= self.n_calls:
            raise ValueError("n_initial_points must be less than n_calls")
    
    def _create_search_space(self) -> List[Any]:
        """Convert parameter space to scikit-optimize format"""
        if not SKOPT_AVAILABLE:
            return []
        
        search_space = []
        
        for param_name, param_config in self.parameter_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'real')
                
                if param_type == 'real':
                    low = param_config['low']
                    high = param_config['high']
                    prior = param_config.get('prior', 'uniform')
                    search_space.append(Real(low, high, prior=prior, name=param_name))
                
                elif param_type == 'integer':
                    low = param_config['low']
                    high = param_config['high']
                    search_space.append(Integer(low, high, name=param_name))
                
                elif param_type == 'categorical':
                    categories = param_config['categories']
                    search_space.append(Categorical(categories, name=param_name))
                
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
            else:
                # Simple list format - assume categorical
                search_space.append(Categorical(param_config, name=param_name))
        
        return search_space
    
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
    
    def _objective_function(self, params: List[Any], X: np.ndarray, y: np.ndarray) -> float:
        """Objective function to minimize (negative score for maximization)"""
        
        # Convert parameter list to dictionary
        param_dict = {}
        param_names = list(self.parameter_space.keys())
        
        for i, param_value in enumerate(params):
            if i < len(param_names):
                param_dict[param_names[i]] = param_value
        
        try:
            # Create model with parameters
            model = self.model_factory(**param_dict)
            
            # Create CV splitter
            cv_splitter = self._create_cv_splitter(X, y)
            
            # Evaluate model
            cv_scores = cross_val_score(
                model, X, y, 
                cv=cv_splitter, 
                scoring=self.scoring,
                n_jobs=1,  # Avoid nested parallelization
                error_score='raise'
            )
            
            score = np.mean(cv_scores)
            score_std = np.std(cv_scores)
            
            # Store evaluation result
            evaluation_result = {
                'params': param_dict.copy(),
                'score_mean': float(score),
                'score_std': float(score_std),
                'cv_scores': cv_scores.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            self.optimization_history_.append(evaluation_result)
            
            if self.verbose:
                logger.info(f"Evaluation: {param_dict} -> {score:.4f} ± {score_std:.4f}")
            
            # Return negative score for minimization
            return -score
            
        except Exception as e:
            logger.warning(f"Error evaluating parameters {param_dict}: {e}")
            # Return large positive value for failed evaluations
            return 1000.0
    
    def _get_acquisition_function(self):
        """Get acquisition function"""
        if not SKOPT_AVAILABLE:
            return None
        
        acquisition_functions = {
            'EI': gaussian_ei,
            'PI': gaussian_pi, 
            'LCB': gaussian_lcb
        }
        
        return acquisition_functions.get(self.acquisition_function, gaussian_ei)
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged"""
        if not self.early_stopping_rounds or len(self.convergence_history_) < self.early_stopping_rounds:
            return False
        
        # Check if best score hasn't improved in last N rounds
        recent_scores = self.convergence_history_[-self.early_stopping_rounds:]
        best_recent = max(recent_scores)
        
        if len(self.convergence_history_) > self.early_stopping_rounds:
            previous_best = max(self.convergence_history_[:-self.early_stopping_rounds])
            improvement = best_recent - previous_best
            
            if improvement < 1e-6:  # Very small improvement threshold
                logger.info(f"Early stopping: no improvement in {self.early_stopping_rounds} rounds")
                return True
        
        return False
    
    @time_it("bayesian_optimization", include_args=True)
    def optimize(self, X: Union[pd.DataFrame, np.ndarray], 
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Run Bayesian optimization"""
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        logger.info(f"Starting Bayesian optimization with {self.n_calls} evaluations")
        
        if not SKOPT_AVAILABLE:
            return self._fallback_optimization(X, y)
        
        try:
            # Create search space
            search_space = self._create_search_space()
            
            if not search_space:
                raise ValueError("Empty search space")
            
            # Create objective function with data binding
            @use_named_args(search_space)
            def objective(**params):
                param_values = [params[dim.name] for dim in search_space]
                score = self._objective_function(param_values, X, y)
                
                # Update convergence tracking
                self.convergence_history_.append(-score)  # Convert back to positive
                
                # Check for early stopping
                if self._check_convergence():
                    # Note: scikit-optimize doesn't support early stopping directly
                    # This check is for future implementations
                    pass
                
                return score
            
            # Select optimization algorithm
            if self.surrogate_model == 'gp':
                optimizer_func = gp_minimize
            elif self.surrogate_model == 'rf':
                optimizer_func = forest_minimize  
            elif self.surrogate_model == 'gbrt':
                optimizer_func = gbrt_minimize
            else:
                optimizer_func = gp_minimize
            
            # Run optimization
            self.optimization_result_ = optimizer_func(
                func=objective,
                dimensions=search_space,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                acquisition_func=self.acquisition_function.lower(),
                random_state=self.random_state,
                verbose=self.verbose
            )
            
            # Extract best parameters and score
            best_params_list = self.optimization_result_.x
            self.best_score_ = -self.optimization_result_.fun  # Convert back to positive
            
            # Convert best parameters to dictionary
            self.best_params_ = {}
            for i, dim in enumerate(search_space):
                self.best_params_[dim.name] = best_params_list[i]
            
            # Create best model
            self.best_model_ = self.model_factory(**self.best_params_)
            self.best_model_.fit(X, y)
            
            logger.info(f"Bayesian optimization completed. Best score: {self.best_score_:.4f}")
            
            return self._create_optimization_results()
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return self._fallback_optimization(X, y)
    
    def _fallback_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fallback to random search if Bayesian optimization fails"""
        logger.info("Using fallback random search optimization")
        
        from .random_search import RandomSearchOptimizer
        
        # Convert parameter space to random search format
        random_param_space = {}
        for param_name, param_config in self.parameter_space.items():
            if isinstance(param_config, dict):
                if param_config.get('type') == 'real':
                    random_param_space[param_name] = (param_config['low'], param_config['high'])
                elif param_config.get('type') == 'integer':
                    random_param_space[param_name] = list(range(param_config['low'], param_config['high'] + 1))
                elif param_config.get('type') == 'categorical':
                    random_param_space[param_name] = param_config['categories']
            else:
                random_param_space[param_name] = param_config
        
        # Run random search
        random_optimizer = RandomSearchOptimizer(
            model_factory=self.model_factory,
            parameter_space=random_param_space,
            scoring=self.scoring,
            cv_folds=self.cv_folds,
            n_iter=min(self.n_calls, 50),  # Limit iterations for fallback
            random_state=self.random_state
        )
        
        results = random_optimizer.optimize(X, y)
        
        # Update our results
        self.best_params_ = results['best_params']
        self.best_score_ = results['best_score']
        self.best_model_ = results['best_model']
        self.optimization_history_ = results.get('optimization_history', [])
        
        return results
    
    def _create_optimization_results(self) -> Dict[str, Any]:
        """Create comprehensive optimization results"""
        
        results = {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'best_model': self.best_model_,
            'optimization_history': self.optimization_history_,
            'convergence_history': self.convergence_history_,
            'n_evaluations': len(self.optimization_history_),
            'optimization_method': 'bayesian' if SKOPT_AVAILABLE else 'random_fallback',
            'surrogate_model': self.surrogate_model,
            'acquisition_function': self.acquisition_function
        }
        
        # Add scikit-optimize specific results
        if self.optimization_result_ is not None and SKOPT_AVAILABLE:
            results.update({
                'func_vals': self.optimization_result_.func_vals.tolist(),
                'x_iters': [x for x in self.optimization_result_.x_iters],
                'optimization_time': getattr(self.optimization_result_, 'optimization_time', None)
            })
        
        # Convergence analysis
        if self.convergence_history_:
            results['convergence_analysis'] = {
                'final_score': self.convergence_history_[-1],
                'initial_score': self.convergence_history_[0] if self.convergence_history_ else None,
                'improvement': self.convergence_history_[-1] - self.convergence_history_[0] if len(self.convergence_history_) > 1 else 0,
                'best_iteration': np.argmax(self.convergence_history_),
                'convergence_rate': self._calculate_convergence_rate()
            }
        
        return results
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate"""
        if len(self.convergence_history_) < 2:
            return 0.0
        
        # Simple convergence rate: improvement per iteration
        total_improvement = self.convergence_history_[-1] - self.convergence_history_[0]
        n_iterations = len(self.convergence_history_) - 1
        
        return total_improvement / n_iterations if n_iterations > 0 else 0.0
    
    def save_optimization_results(self, filepath: str):
        """Save optimization results to file"""
        try:
            results = self._create_optimization_results()
            
            # Remove non-serializable objects
            serializable_results = results.copy()
            serializable_results.pop('best_model', None)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Optimization results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving optimization results: {e}")
    
    def load_optimization_results(self, filepath: str):
        """Load optimization results from file"""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            self.best_params_ = results.get('best_params')
            self.best_score_ = results.get('best_score')
            self.optimization_history_ = results.get('optimization_history', [])
            self.convergence_history_ = results.get('convergence_history', [])
            
            # Recreate best model
            if self.best_params_:
                self.best_model_ = self.model_factory(**self.best_params_)
            
            logger.info(f"Optimization results loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading optimization results: {e}")

# ============================================
# Multi-Objective Bayesian Optimization
# ============================================

class MultiObjectiveBayesianOptimizer:
    """Multi-objective Bayesian optimization for trading off multiple metrics"""
    
    def __init__(self,
                 model_factory: Callable,
                 parameter_space: Dict[str, Any],
                 objectives: List[str] = ['accuracy', 'precision'],
                 weights: Optional[List[float]] = None,
                 cv_folds: int = 5,
                 n_calls: int = 100,
                 random_state: int = 42):
        """
        Initialize Multi-Objective Bayesian Optimizer
        
        Args:
            model_factory: Function that creates model instances
            parameter_space: Dictionary defining parameter search space
            objectives: List of objectives to optimize
            weights: Weights for combining objectives (if None, equal weights)
            cv_folds: Number of cross-validation folds
            n_calls: Number of optimization iterations
            random_state: Random seed
        """
        self.model_factory = model_factory
        self.parameter_space = parameter_space
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
        self.cv_folds = cv_folds
        self.n_calls = n_calls
        self.random_state = random_state
        
        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
        
        # Results
        self.pareto_front_ = []
        self.optimization_history_ = []
        self.best_compromise_params_ = None
        self.best_compromise_model_ = None
        
        logger.info(f"Initialized multi-objective optimization with objectives: {objectives}")
    
    def _evaluate_objectives(self, params: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> List[float]:
        """Evaluate all objectives for given parameters"""
        
        model = self.model_factory(**params)
        
        cv_splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        objective_scores = []
        
        for objective in self.objectives:
            try:
                scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=objective, n_jobs=1)
                objective_scores.append(np.mean(scores))
            except Exception as e:
                logger.warning(f"Error evaluating objective {objective}: {e}")
                objective_scores.append(0.0)
        
        return objective_scores
    
    def _calculate_weighted_score(self, objective_scores: List[float]) -> float:
        """Calculate weighted combination of objectives"""
        return sum(score * weight for score, weight in zip(objective_scores, self.weights))
    
    def _is_pareto_optimal(self, scores: List[float], existing_front: List[List[float]]) -> bool:
        """Check if scores are Pareto optimal"""
        for existing_scores in existing_front:
            # Check if existing solution dominates current solution
            dominates = all(existing >= current for existing, current in zip(existing_scores, scores))
            strictly_better = any(existing > current for existing, current in zip(existing_scores, scores))
            
            if dominates and strictly_better:
                return False
        
        return True
    
    def _update_pareto_front(self, params: Dict[str, Any], scores: List[float]):
        """Update Pareto front with new solution"""
        
        # Remove dominated solutions
        new_front = []
        for existing_params, existing_scores in self.pareto_front_:
            # Check if new solution dominates existing
            dominates = all(new >= existing for new, existing in zip(scores, existing_scores))
            strictly_better = any(new > existing for new, existing in zip(scores, existing_scores))
            
            if not (dominates and strictly_better):
                new_front.append((existing_params, existing_scores))
        
        # Add new solution if it's Pareto optimal
        existing_scores = [scores for _, scores in new_front]
        if self._is_pareto_optimal(scores, existing_scores):
            new_front.append((params.copy(), scores.copy()))
        
        self.pareto_front_ = new_front
    
    def optimize(self, X: Union[pd.DataFrame, np.ndarray], 
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Run multi-objective Bayesian optimization"""
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        logger.info(f"Starting multi-objective Bayesian optimization")
        
        # Use single-objective optimizer with weighted sum
        single_optimizer = BayesianOptimizer(
            model_factory=self.model_factory,
            parameter_space=self.parameter_space,
            scoring='accuracy',  # Will be overridden
            cv_folds=self.cv_folds,
            n_calls=self.n_calls,
            random_state=self.random_state
        )
        
        # Custom objective function for multi-objective
        def multi_objective_function(params_list: List[Any]) -> float:
            # Convert parameter list to dictionary
            param_dict = {}
            param_names = list(self.parameter_space.keys())
            
            for i, param_value in enumerate(params_list):
                if i < len(param_names):
                    param_dict[param_names[i]] = param_value
            
            # Evaluate all objectives
            objective_scores = self._evaluate_objectives(param_dict, X, y)
            
            # Update Pareto front
            self._update_pareto_front(param_dict, objective_scores)
            
            # Calculate weighted score
            weighted_score = self._calculate_weighted_score(objective_scores)
            
            # Store evaluation
            evaluation = {
                'params': param_dict.copy(),
                'objective_scores': objective_scores.copy(),
                'weighted_score': weighted_score,
                'timestamp': datetime.now().isoformat()
            }
            self.optimization_history_.append(evaluation)
            
            # Return negative for minimization
            return -weighted_score
        
        # Replace objective function
        single_optimizer._objective_function = multi_objective_function
        
        # Run optimization
        results = single_optimizer.optimize(X, y)
        
        # Find best compromise solution (highest weighted score)
        if self.optimization_history_:
            best_evaluation = max(self.optimization_history_, key=lambda x: x['weighted_score'])
            self.best_compromise_params_ = best_evaluation['params']
            self.best_compromise_model_ = self.model_factory(**self.best_compromise_params_)
            self.best_compromise_model_.fit(X, y)
        
        return self._create_multi_objective_results()
    
    def _create_multi_objective_results(self) -> Dict[str, Any]:
        """Create multi-objective optimization results"""
        
        results = {
            'pareto_front': self.pareto_front_,
            'best_compromise_params': self.best_compromise_params_,
            'best_compromise_model': self.best_compromise_model_,
            'optimization_history': self.optimization_history_,
            'objectives': self.objectives,
            'weights': self.weights,
            'n_pareto_solutions': len(self.pareto_front_)
        }
        
        if self.optimization_history_:
            # Analysis of objective trade-offs
            all_scores = [eval_data['objective_scores'] for eval_data in self.optimization_history_]
            all_scores = np.array(all_scores)
            
            results['objective_analysis'] = {
                'objective_means': np.mean(all_scores, axis=0).tolist(),
                'objective_stds': np.std(all_scores, axis=0).tolist(),
                'objective_correlations': np.corrcoef(all_scores.T).tolist(),
                'best_individual_objectives': np.max(all_scores, axis=0).tolist()
            }
        
        return results

# ============================================
# Specialized Financial Bayesian Optimization
# ============================================

class FinancialBayesianOptimizer(BayesianOptimizer):
    """Bayesian optimizer specialized for financial models"""
    
    def __init__(self, 
                 model_factory: Callable,
                 parameter_space: Dict[str, Any],
                 financial_metrics: List[str] = ['accuracy', 'precision', 'recall'],
                 risk_adjusted_scoring: bool = True,
                 volatility_penalty: float = 0.1,
                 **kwargs):
        """
        Initialize Financial Bayesian Optimizer
        
        Args:
            model_factory: Function that creates model instances
            parameter_space: Dictionary defining parameter search space
            financial_metrics: Financial-specific metrics to consider
            risk_adjusted_scoring: Whether to adjust scores for risk
            volatility_penalty: Penalty factor for prediction volatility
        """
        super().__init__(model_factory, parameter_space, **kwargs)
        
        self.financial_metrics = financial_metrics
        self.risk_adjusted_scoring = risk_adjusted_scoring
        self.volatility_penalty = volatility_penalty
        
    def _calculate_financial_score(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate risk-adjusted financial score"""
        
        # Get base predictions
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X)
            pred_classes = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
        else:
            pred_classes = model.predict(X)
            confidence = np.ones(len(pred_classes))
        
        # Calculate base accuracy
        base_score = accuracy_score(y, pred_classes)
        
        if not self.risk_adjusted_scoring:
            return base_score
        
        # Risk adjustments
        
        # 1. Confidence-based adjustment
        mean_confidence = np.mean(confidence)
        confidence_adjustment = (mean_confidence - 0.5) * 0.1  # Bonus for high confidence
        
        # 2. Volatility penalty
        if len(confidence) > 1:
            confidence_volatility = np.std(confidence)
            volatility_adjustment = -confidence_volatility * self.volatility_penalty
        else:
            volatility_adjustment = 0
        
        # 3. Consistency bonus (lower prediction variance is better)
        if hasattr(model, 'predict_proba') and len(np.unique(y)) == 2:
            prob_variance = np.var(predictions[:, 1])
            consistency_bonus = -prob_variance * 0.05
        else:
            consistency_bonus = 0
        
        # Calculate risk-adjusted score
        risk_adjusted_score = base_score + confidence_adjustment + volatility_adjustment + consistency_bonus
        
        return max(0, min(1, risk_adjusted_score))  # Clamp to [0, 1]
    
    def _objective_function(self, params: List[Any], X: np.ndarray, y: np.ndarray) -> float:
        """Financial-specific objective function"""
        
        # Convert parameter list to dictionary
        param_dict = {}
        param_names = list(self.parameter_space.keys())
        
        for i, param_value in enumerate(params):
            if i < len(param_names):
                param_dict[param_names[i]] = param_value
        
        try:
            # Create model with parameters
            model = self.model_factory(**param_dict)
            
            # Create CV splitter
            cv_splitter = self._create_cv_splitter(X, y)
            
            # Evaluate with financial scoring
            cv_scores = []
            for train_idx, val_idx in cv_splitter.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit model
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                # Calculate financial score
                financial_score = self._calculate_financial_score(model_clone, X_val, y_val)
                cv_scores.append(financial_score)
            
            score = np.mean(cv_scores)
            score_std = np.std(cv_scores)
            
            # Store evaluation result
            evaluation_result = {
                'params': param_dict.copy(),
                'score_mean': float(score),
                'score_std': float(score_std),
                'cv_scores': cv_scores,
                'risk_adjusted': self.risk_adjusted_scoring,
                'timestamp': datetime.now().isoformat()
            }
            self.optimization_history_.append(evaluation_result)
            
            if self.verbose:
                logger.info(f"Financial evaluation: {param_dict} -> {score:.4f} ± {score_std:.4f}")
            
            # Return negative score for minimization
            return -score
            
        except Exception as e:
            logger.warning(f"Error evaluating parameters {param_dict}: {e}")
            return 1000.0

# ============================================
# Factory Functions and Utilities
# ============================================

def optimize_model_bayesian(model_name: str,
                           parameter_space: Dict[str, Any],
                           X: Union[pd.DataFrame, np.ndarray],
                           y: Union[pd.Series, np.ndarray],
                           **optimizer_kwargs) -> Dict[str, Any]:
    """Convenient function to optimize specific model types"""
    
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
    optimizer = BayesianOptimizer(
        model_factory=model_factory,
        parameter_space=parameter_space,
        **optimizer_kwargs
    )
    
    # Run optimization
    results = optimizer.optimize(X, y)
    
    return results

def create_financial_parameter_space(model_name: str) -> Dict[str, Any]:
    """Create default parameter spaces optimized for financial models"""
    
    spaces = {
        'gradient_boosting': {
            'n_estimators': {'type': 'integer', 'low': 50, 'high': 500},
            'max_depth': {'type': 'integer', 'low': 3, 'high': 15},
            'learning_rate': {'type': 'real', 'low': 0.01, 'high': 0.3, 'prior': 'log-uniform'},
            'subsample': {'type': 'real', 'low': 0.6, 'high': 1.0},
            'min_samples_split': {'type': 'integer', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'integer', 'low': 1, 'high': 10}
        },
        'random_forest': {
            'n_estimators': {'type': 'integer', 'low': 50, 'high': 300},
            'max_depth': {'type': 'integer', 'low': 5, 'high': 25},
            'min_samples_split': {'type': 'integer', 'low': 2, 'high': 15},
            'min_samples_leaf': {'type': 'integer', 'low': 1, 'high': 8},
            'max_features': {'type': 'categorical', 'categories': ['sqrt', 'log2', 0.5, 0.8]}
        },
        'svm': {
            'C': {'type': 'real', 'low': 0.1, 'high': 100, 'prior': 'log-uniform'},
            'gamma': {'type': 'real', 'low': 1e-4, 'high': 1, 'prior': 'log-uniform'},
            'kernel': {'type': 'categorical', 'categories': ['rbf', 'poly', 'sigmoid']}
        },
        'logistic': {
            'C': {'type': 'real', 'low': 0.01, 'high': 100, 'prior': 'log-uniform'},
            'penalty': {'type': 'categorical', 'categories': ['l1', 'l2', 'elasticnet']},
            'solver': {'type': 'categorical', 'categories': ['liblinear', 'saga', 'lbfgs']}
        },
        'neural_network': {
            'hidden_layer_sizes': {'type': 'categorical', 'categories': [(50,), (100,), (100, 50), (150, 100, 50)]},
            'alpha': {'type': 'real', 'low': 1e-5, 'high': 1e-1, 'prior': 'log-uniform'},
            'learning_rate_init': {'type': 'real', 'low': 1e-4, 'high': 1e-1, 'prior': 'log-uniform'},
            'max_iter': {'type': 'integer', 'low': 100, 'high': 1000}
        }
    }
    
    if model_name not in spaces:
        raise ValueError(f"No default parameter space for model: {model_name}")
    
    return spaces[model_name]

def compare_bayesian_strategies(X: Union[pd.DataFrame, np.ndarray],
                              y: Union[pd.Series, np.ndarray],
                              model_factory: Callable,
                              parameter_space: Dict[str, Any],
                              strategies: Dict[str, Dict] = None) -> Dict[str, Any]:
    """Compare different Bayesian optimization strategies"""
    
    if strategies is None:
        strategies = {
            'gp_ei': {'surrogate_model': 'gp', 'acquisition_function': 'EI'},
            'gp_pi': {'surrogate_model': 'gp', 'acquisition_function': 'PI'},
            'gp_lcb': {'surrogate_model': 'gp', 'acquisition_function': 'LCB'},
            'rf_ei': {'surrogate_model': 'rf', 'acquisition_function': 'EI'},
            'gbrt_ei': {'surrogate_model': 'gbrt', 'acquisition_function': 'EI'}
        }
    
    results = {}
    
    for strategy_name, strategy_config in strategies.items():
        logger.info(f"Testing Bayesian strategy: {strategy_name}")
        
        try:
            optimizer = BayesianOptimizer(
                model_factory=model_factory,
                parameter_space=parameter_space,
                n_calls=50,  # Reduced for comparison
                verbose=False,
                **strategy_config
            )
            
            strategy_results = optimizer.optimize(X, y)
            
            results[strategy_name] = {
                'best_score': strategy_results['best_score'],
                'best_params': strategy_results['best_params'],
                'n_evaluations': strategy_results['n_evaluations'],
                'convergence_rate': strategy_results.get('convergence_analysis', {}).get('convergence_rate', 0),
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
            'score_range': {
                'best': max(v['best_score'] for v in valid_results.values()),
                'worst': min(v['best_score'] for v in valid_results.values())
            }
        }
    
    return results
