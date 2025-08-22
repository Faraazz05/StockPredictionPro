# ============================================
# StockPredictionPro - src/models/optimization/random_search.py
# Advanced random search optimization with intelligent sampling and adaptive strategies
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
import warnings
import json
from collections import defaultdict
import time
from scipy import stats

# Core ML imports
from sklearn.model_selection import (
    RandomizedSearchCV, cross_val_score, ParameterSampler,
    StratifiedKFold, KFold, TimeSeriesSplit
)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.base import BaseEstimator, clone

# Import our model factory functions
from ..classification.gradient_boosting import create_gradient_boosting_classifier
from ..classification.random_forest import create_random_forest_classifier
from ..classification.svm import create_svm_classifier
from ..classification.logistic import create_logistic_classifier
from ..classification.neural_network import create_neural_network_classifier

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it

logger = get_logger('models.optimization.random_search')

# ============================================
# Advanced Random Search Framework
# ============================================

class RandomSearchOptimizer:
    """Advanced random search with intelligent sampling strategies and adaptive exploration"""
    
    def __init__(self,
                 model_factory: Callable,
                 parameter_space: Dict[str, Union[List, Tuple, Any]],
                 scoring: str = 'accuracy',
                 cv_folds: int = 5,
                 n_iter: int = 100,
                 sampling_strategy: str = 'random',
                 time_aware_cv: bool = True,
                 early_stopping: bool = False,
                 early_stopping_patience: int = 10,
                 improvement_threshold: float = 0.001,
                 n_jobs: Optional[int] = -1,
                 random_state: int = 42,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize Random Search Optimizer
        
        Args:
            model_factory: Function that creates model instances
            parameter_space: Dictionary defining parameter search space
            scoring: Scoring metric to optimize
            cv_folds: Number of cross-validation folds
            n_iter: Number of random iterations
            sampling_strategy: Sampling strategy ('random', 'latin_hypercube', 'sobol', 'halton')
            time_aware_cv: Whether to use time-aware cross-validation
            early_stopping: Whether to use early stopping
            early_stopping_patience: Number of iterations to wait for improvement
            improvement_threshold: Minimum improvement threshold
            n_jobs: Number of parallel jobs
            random_state: Random seed
            verbose: Whether to print progress
        """
        self.model_factory = model_factory
        self.parameter_space = parameter_space
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.sampling_strategy = sampling_strategy
        self.time_aware_cv = time_aware_cv
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.improvement_threshold = improvement_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        # Results
        self.random_search_cv_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        self.optimization_history_ = []
        self.parameter_analysis_ = None
        self.sampling_analysis_ = None
        
        # Initialize random state
        np.random.seed(self.random_state)
        
        # Validation
        self._validate_parameters()
        
        logger.info(f"Initialized random search optimizer with {self.n_iter} iterations")
    
    def _validate_parameters(self):
        """Validate random search parameters"""
        valid_strategies = ['random', 'latin_hypercube', 'sobol', 'halton']
        if self.sampling_strategy not in valid_strategies:
            raise ValueError(f"sampling_strategy must be one of {valid_strategies}")
        
        if self.n_iter <= 0:
            raise ValueError("n_iter must be positive")
        
        if self.cv_folds <= 1:
            raise ValueError("cv_folds must be greater than 1")
    
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
    
    def _create_parameter_sampler(self) -> List[Dict[str, Any]]:
        """Create parameter sampler based on strategy"""
        
        if self.sampling_strategy == 'random':
            return self._random_sampling()
        elif self.sampling_strategy == 'latin_hypercube':
            return self._latin_hypercube_sampling()
        elif self.sampling_strategy == 'sobol':
            return self._sobol_sampling()
        elif self.sampling_strategy == 'halton':
            return self._halton_sampling()
        else:
            return self._random_sampling()
    
    def _random_sampling(self) -> List[Dict[str, Any]]:
        """Standard random parameter sampling"""
        sampler = ParameterSampler(
            self.parameter_space, 
            n_iter=self.n_iter, 
            random_state=self.random_state
        )
        return list(sampler)
    
    def _latin_hypercube_sampling(self) -> List[Dict[str, Any]]:
        """Latin Hypercube Sampling for better space coverage"""
        try:
            from scipy.stats import qmc
            
            # Separate numeric and categorical parameters
            numeric_params = {}
            categorical_params = {}
            
            for param_name, param_values in self.parameter_space.items():
                if isinstance(param_values, (tuple, list)):
                    if len(param_values) == 2 and all(isinstance(x, (int, float)) for x in param_values):
                        # Numeric range (low, high)
                        numeric_params[param_name] = param_values
                    else:
                        # Categorical values
                        categorical_params[param_name] = param_values
                else:
                    categorical_params[param_name] = param_values
            
            parameter_combinations = []
            
            if numeric_params:
                # Generate LHS samples for numeric parameters
                n_numeric = len(numeric_params)
                sampler = qmc.LatinHypercube(d=n_numeric, seed=self.random_state)
                lhs_samples = sampler.random(n=self.n_iter)
                
                # Scale to parameter ranges
                param_names = list(numeric_params.keys())
                for i in range(self.n_iter):
                    params = {}
                    
                    # Scale numeric parameters
                    for j, param_name in enumerate(param_names):
                        low, high = numeric_params[param_name]
                        scaled_value = low + lhs_samples[i, j] * (high - low)
                        
                        # Check if parameter should be integer
                        if isinstance(low, int) and isinstance(high, int):
                            params[param_name] = int(round(scaled_value))
                        else:
                            params[param_name] = scaled_value
                    
                    # Add categorical parameters randomly
                    for param_name, param_values in categorical_params.items():
                        params[param_name] = np.random.choice(param_values)
                    
                    parameter_combinations.append(params)
            else:
                # Only categorical parameters - use random sampling
                parameter_combinations = self._random_sampling()
            
            return parameter_combinations
            
        except ImportError:
            logger.warning("scipy.stats.qmc not available. Falling back to random sampling.")
            return self._random_sampling()
    
    def _sobol_sampling(self) -> List[Dict[str, Any]]:
        """Sobol sequence sampling for quasi-random coverage"""
        try:
            from scipy.stats import qmc
            
            # Separate numeric and categorical parameters
            numeric_params = {}
            categorical_params = {}
            
            for param_name, param_values in self.parameter_space.items():
                if isinstance(param_values, (tuple, list)):
                    if len(param_values) == 2 and all(isinstance(x, (int, float)) for x in param_values):
                        numeric_params[param_name] = param_values
                    else:
                        categorical_params[param_name] = param_values
                else:
                    categorical_params[param_name] = param_values
            
            parameter_combinations = []
            
            if numeric_params:
                # Generate Sobol samples for numeric parameters
                n_numeric = len(numeric_params)
                sampler = qmc.Sobol(d=n_numeric, seed=self.random_state)
                sobol_samples = sampler.random(n=self.n_iter)
                
                # Scale to parameter ranges
                param_names = list(numeric_params.keys())
                for i in range(self.n_iter):
                    params = {}
                    
                    # Scale numeric parameters
                    for j, param_name in enumerate(param_names):
                        low, high = numeric_params[param_name]
                        scaled_value = low + sobol_samples[i, j] * (high - low)
                        
                        # Check if parameter should be integer
                        if isinstance(low, int) and isinstance(high, int):
                            params[param_name] = int(round(scaled_value))
                        else:
                            params[param_name] = scaled_value
                    
                    # Add categorical parameters randomly
                    for param_name, param_values in categorical_params.items():
                        params[param_name] = np.random.choice(param_values)
                    
                    parameter_combinations.append(params)
            else:
                # Only categorical parameters - use random sampling
                parameter_combinations = self._random_sampling()
            
            return parameter_combinations
            
        except ImportError:
            logger.warning("scipy.stats.qmc not available. Falling back to random sampling.")
            return self._random_sampling()
    
    def _halton_sampling(self) -> List[Dict[str, Any]]:
        """Halton sequence sampling for quasi-random coverage"""
        try:
            from scipy.stats import qmc
            
            # Separate numeric and categorical parameters
            numeric_params = {}
            categorical_params = {}
            
            for param_name, param_values in self.parameter_space.items():
                if isinstance(param_values, (tuple, list)):
                    if len(param_values) == 2 and all(isinstance(x, (int, float)) for x in param_values):
                        numeric_params[param_name] = param_values
                    else:
                        categorical_params[param_name] = param_values
                else:
                    categorical_params[param_name] = param_values
            
            parameter_combinations = []
            
            if numeric_params:
                # Generate Halton samples for numeric parameters
                n_numeric = len(numeric_params)
                sampler = qmc.Halton(d=n_numeric, seed=self.random_state)
                halton_samples = sampler.random(n=self.n_iter)
                
                # Scale to parameter ranges
                param_names = list(numeric_params.keys())
                for i in range(self.n_iter):
                    params = {}
                    
                    # Scale numeric parameters
                    for j, param_name in enumerate(param_names):
                        low, high = numeric_params[param_name]
                        scaled_value = low + halton_samples[i, j] * (high - low)
                        
                        # Check if parameter should be integer
                        if isinstance(low, int) and isinstance(high, int):
                            params[param_name] = int(round(scaled_value))
                        else:
                            params[param_name] = scaled_value
                    
                    # Add categorical parameters randomly
                    for param_name, param_values in categorical_params.items():
                        params[param_name] = np.random.choice(param_values)
                    
                    parameter_combinations.append(params)
            else:
                # Only categorical parameters - use random sampling
                parameter_combinations = self._random_sampling()
            
            return parameter_combinations
            
        except ImportError:
            logger.warning("scipy.stats.qmc not available. Falling back to random sampling.")
            return self._random_sampling()
    
    def _evaluate_parameter_combination(self, params: Dict[str, Any], 
                                      X: np.ndarray, y: np.ndarray) -> Tuple[float, float, List[float]]:
        """Evaluate a single parameter combination"""
        
        try:
            model = self.model_factory(**params)
            cv_splitter = self._create_cv_splitter(X, y)
            
            cv_scores = cross_val_score(
                model, X, y,
                cv=cv_splitter,
                scoring=self.scoring,
                n_jobs=1,  # Avoid nested parallelization
                error_score='raise'
            )
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            return mean_score, std_score, cv_scores.tolist()
            
        except Exception as e:
            logger.warning(f"Error evaluating params {params}: {e}")
            return -np.inf, 0.0, []
    
    def _random_search_with_early_stopping(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run random search with early stopping"""
        
        # Generate parameter combinations
        parameter_combinations = self._create_parameter_sampler()
        
        best_score = -np.inf
        best_params = None
        no_improvement_count = 0
        
        results = []
        
        logger.info(f"Starting random search with early stopping over {len(parameter_combinations)} combinations")
        
        for i, params in enumerate(parameter_combinations):
            start_time = time.time()
            
            # Evaluate current parameters
            mean_score, std_score, cv_scores = self._evaluate_parameter_combination(params, X, y)
            
            evaluation_time = time.time() - start_time
            
            # Store result
            result = {
                'params': params.copy(),
                'mean_test_score': mean_score,
                'std_test_score': std_score,
                'cv_scores': cv_scores,
                'evaluation_time': evaluation_time,
                'iteration': i + 1
            }
            results.append(result)
            self.optimization_history_.append(result)
            
            # Check for improvement
            if mean_score > best_score + self.improvement_threshold:
                best_score = mean_score
                best_params = params.copy()
                no_improvement_count = 0
                
                if self.verbose:
                    logger.info(f"New best score: {best_score:.4f} with params {params}")
            else:
                no_improvement_count += 1
            
            # Early stopping check
            if self.early_stopping and no_improvement_count >= self.early_stopping_patience:
                logger.info(f"Early stopping at iteration {i + 1}: no improvement for {self.early_stopping_patience} iterations")
                break
            
            if self.verbose and i % 10 == 0:
                logger.info(f"Iteration {i + 1}/{len(parameter_combinations)}: {mean_score:.4f} ± {std_score:.4f}")
        
        # Find best result
        best_result = max(results, key=lambda x: x['mean_test_score'])
        
        return {
            'best_params_': best_result['params'],
            'best_score_': best_result['mean_test_score'],
            'cv_results_': {
                'params': [r['params'] for r in results],
                'mean_test_score': [r['mean_test_score'] for r in results],
                'std_test_score': [r['std_test_score'] for r in results],
                'mean_fit_time': [r['evaluation_time'] for r in results]
            },
            'n_evaluations_': len(results)
        }
    
    @time_it("random_search_optimization", include_args=True)
    def optimize(self, X: Union[pd.DataFrame, np.ndarray], 
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Run random search optimization"""
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        logger.info(f"Starting random search optimization with {self.sampling_strategy} sampling")
        
        try:
            if self.early_stopping:
                # Use custom random search with early stopping
                search_results = self._random_search_with_early_stopping(X, y)
                
                # Create a mock RandomizedSearchCV object for compatibility
                class MockRandomizedSearchCV:
                    def __init__(self, results):
                        self.best_params_ = results['best_params_']
                        self.best_score_ = results['best_score_']
                        self.cv_results_ = results['cv_results_']
                        self.n_evaluations_ = results['n_evaluations_']
                
                self.random_search_cv_ = MockRandomizedSearchCV(search_results)
                
            else:
                # Use custom parameter sampler with scikit-learn RandomizedSearchCV
                parameter_combinations = self._create_parameter_sampler()
                
                # Convert to format expected by RandomizedSearchCV
                param_distributions = {}
                for param_name in self.parameter_space.keys():
                    # Create a custom distribution that samples from our pre-generated combinations
                    param_distributions[param_name] = [combo[param_name] for combo in parameter_combinations]
                
                base_model = self.model_factory()
                cv_splitter = self._create_cv_splitter(X, y)
                
                self.random_search_cv_ = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_distributions,
                    n_iter=min(self.n_iter, len(parameter_combinations)),
                    cv=cv_splitter,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    verbose=1 if self.verbose else 0,
                    random_state=self.random_state,
                    return_train_score=True,
                    error_score='raise'
                )
                
                self.random_search_cv_.fit(X, y)
            
            # Extract results
            self.best_params_ = self.random_search_cv_.best_params_
            self.best_score_ = self.random_search_cv_.best_score_
            
            # Create best model
            self.best_model_ = self.model_factory(**self.best_params_)
            self.best_model_.fit(X, y)
            
            # Analyze results
            self._analyze_random_search_results()
            
            logger.info(f"Random search completed. Best score: {self.best_score_:.4f}")
            
            return self._create_optimization_results()
            
        except Exception as e:
            logger.error(f"Random search optimization failed: {e}")
            raise
    
    def _analyze_random_search_results(self):
        """Analyze random search results for insights"""
        
        if not hasattr(self.random_search_cv_, 'cv_results_'):
            return
        
        cv_results = self.random_search_cv_.cv_results_
        
        # Parameter analysis
        self.parameter_analysis_ = self._calculate_parameter_statistics(cv_results)
        
        # Sampling analysis
        self.sampling_analysis_ = {
            'sampling_strategy': self.sampling_strategy,
            'total_evaluations': len(cv_results['mean_test_score']),
            'best_score': self.best_score_,
            'worst_score': min(cv_results['mean_test_score']),
            'score_range': max(cv_results['mean_test_score']) - min(cv_results['mean_test_score']),
            'mean_score': np.mean(cv_results['mean_test_score']),
            'score_std': np.std(cv_results['mean_test_score']),
            'parameter_analysis': self.parameter_analysis_
        }
        
        # Convergence analysis
        if self.optimization_history_:
            scores = [result['mean_test_score'] for result in self.optimization_history_]
            best_scores_so_far = np.maximum.accumulate(scores)
            
            self.sampling_analysis_['convergence_analysis'] = {
                'scores_progression': scores,
                'best_scores_progression': best_scores_so_far.tolist(),
                'improvement_iterations': self._find_improvement_iterations(best_scores_so_far),
                'convergence_rate': self._calculate_convergence_rate(best_scores_so_far)
            }
        
        # Top performing combinations
        scores = cv_results['mean_test_score']
        top_indices = np.argsort(scores)[-10:][::-1]  # Top 10
        
        self.sampling_analysis_['top_combinations'] = []
        for idx in top_indices:
            combination = {
                'params': cv_results['params'][idx],
                'score': scores[idx],
                'std': cv_results['std_test_score'][idx],
                'rank': len(top_indices) - list(top_indices).index(idx)
            }
            self.sampling_analysis_['top_combinations'].append(combination)
    
    def _calculate_parameter_statistics(self, cv_results: Dict[str, List]) -> Dict[str, Any]:
        """Calculate parameter statistics across all evaluations"""
        
        params_list = cv_results['params']
        scores = np.array(cv_results['mean_test_score'])
        
        # Get all parameter names
        all_param_names = set()
        for params in params_list:
            all_param_names.update(params.keys())
        
        parameter_stats = {}
        
        for param_name in all_param_names:
            param_values = []
            param_scores = []
            
            for params, score in zip(params_list, scores):
                if param_name in params:
                    param_values.append(params[param_name])
                    param_scores.append(score)
            
            if param_values:
                # Determine if parameter is numeric or categorical
                if all(isinstance(v, (int, float)) for v in param_values):
                    # Numeric parameter
                    parameter_stats[param_name] = {
                        'type': 'numeric',
                        'mean': float(np.mean(param_values)),
                        'std': float(np.std(param_values)),
                        'min': float(np.min(param_values)),
                        'max': float(np.max(param_values)),
                        'correlation_with_score': float(np.corrcoef(param_values, param_scores)[0, 1]) if len(set(param_values)) > 1 else 0.0,
                        'best_value': param_values[np.argmax(param_scores)],
                        'values_explored': len(set(param_values))
                    }
                else:
                    # Categorical parameter
                    from collections import Counter
                    value_counts = Counter(param_values)
                    value_scores = defaultdict(list)
                    
                    for val, score in zip(param_values, param_scores):
                        value_scores[val].append(score)
                    
                    value_stats = {}
                    for val, scores_for_val in value_scores.items():
                        value_stats[str(val)] = {
                            'count': len(scores_for_val),
                            'mean_score': float(np.mean(scores_for_val)),
                            'std_score': float(np.std(scores_for_val))
                        }
                    
                    parameter_stats[param_name] = {
                        'type': 'categorical',
                        'value_counts': dict(value_counts),
                        'value_statistics': value_stats,
                        'best_value': param_values[np.argmax(param_scores)],
                        'values_explored': len(set(param_values))
                    }
        
        return parameter_stats
    
    def _find_improvement_iterations(self, best_scores_so_far: np.ndarray) -> List[int]:
        """Find iterations where score improved"""
        improvements = []
        
        for i in range(1, len(best_scores_so_far)):
            if best_scores_so_far[i] > best_scores_so_far[i-1]:
                improvements.append(i)
        
        return improvements
    
    def _calculate_convergence_rate(self, best_scores_so_far: np.ndarray) -> float:
        """Calculate convergence rate"""
        if len(best_scores_so_far) < 2:
            return 0.0
        
        # Calculate improvement rate over iterations
        total_improvement = best_scores_so_far[-1] - best_scores_so_far[0]
        n_iterations = len(best_scores_so_far) - 1
        
        return total_improvement / n_iterations if n_iterations > 0 else 0.0
    
    def _create_optimization_results(self) -> Dict[str, Any]:
        """Create comprehensive optimization results"""
        
        results = {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'best_model': self.best_model_,
            'optimization_history': self.optimization_history_,
            'parameter_analysis': self.parameter_analysis_,
            'sampling_analysis': self.sampling_analysis_,
            'optimization_method': 'random_search',
            'sampling_strategy': self.sampling_strategy,
            'early_stopping_used': self.early_stopping
        }
        
        # Add CV results if available
        if hasattr(self.random_search_cv_, 'cv_results_'):
            results['cv_results'] = self.random_search_cv_.cv_results_
        
        # Add evaluation count
        if hasattr(self.random_search_cv_, 'n_evaluations_'):
            results['n_evaluations'] = self.random_search_cv_.n_evaluations_
        else:
            results['n_evaluations'] = len(self.optimization_history_)
        
        return results
    
    def get_parameter_statistics(self) -> Dict[str, Any]:
        """Get parameter statistics"""
        return self.parameter_analysis_.copy() if self.parameter_analysis_ else {}
    
    def get_sampling_analysis(self) -> Dict[str, Any]:
        """Get sampling strategy analysis"""
        return self.sampling_analysis_.copy() if self.sampling_analysis_ else {}
    
    def get_top_combinations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N parameter combinations"""
        if self.sampling_analysis_ and 'top_combinations' in self.sampling_analysis_:
            return self.sampling_analysis_['top_combinations'][:n]
        return []
    
    def plot_random_search_analysis(self) -> Any:
        """Plot random search analysis results"""
        if not self.sampling_analysis_:
            logger.warning("Sampling analysis not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            analysis = self.sampling_analysis_
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Parameter exploration analysis
            if self.parameter_analysis_:
                param_names = list(self.parameter_analysis_.keys())[:10]  # Top 10 parameters
                exploration_scores = []
                
                for param_name in param_names:
                    param_stats = self.parameter_analysis_[param_name]
                    if param_stats['type'] == 'numeric':
                        # Use correlation with score as exploration quality
                        exploration_scores.append(abs(param_stats['correlation_with_score']))
                    else:
                        # Use score variance across categorical values
                        value_stats = param_stats['value_statistics']
                        scores = [stats['mean_score'] for stats in value_stats.values()]
                        exploration_scores.append(np.std(scores) if len(scores) > 1 else 0)
                
                if exploration_scores:
                    bars = axes[0, 0].bar(range(len(param_names)), exploration_scores, alpha=0.7, color='steelblue')
                    axes[0, 0].set_title('Parameter Exploration Quality')
                    axes[0, 0].set_xticks(range(len(param_names)))
                    axes[0, 0].set_xticklabels(param_names, rotation=45, ha='right')
                    axes[0, 0].set_ylabel('Exploration Score')
                    
                    # Add values on bars
                    for bar, score in zip(bars, exploration_scores):
                        axes[0, 0].text(bar.get_x() + bar.get_width()/2, score + max(exploration_scores) * 0.01,
                                       f'{score:.3f}', ha='center', va='bottom', fontsize=8)
                else:
                    axes[0, 0].text(0.5, 0.5, 'Parameter exploration\nanalysis not available', 
                                   ha='center', va='center', transform=axes[0, 0].transAxes)
                    axes[0, 0].set_title('Parameter Exploration Quality')
            else:
                axes[0, 0].axis('off')
            
            # Score distribution
            if hasattr(self.random_search_cv_, 'cv_results_'):
                scores = self.random_search_cv_.cv_results_['mean_test_score']
                
                axes[0, 1].hist(scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[0, 1].axvline(self.best_score_, color='red', linestyle='--', linewidth=2,
                                  label=f'Best: {self.best_score_:.3f}')
                axes[0, 1].axvline(np.mean(scores), color='blue', linestyle='--', linewidth=2,
                                  label=f'Mean: {np.mean(scores):.3f}')
                axes[0, 1].set_title(f'Score Distribution ({self.sampling_strategy})')
                axes[0, 1].set_xlabel('CV Score')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, 'Score distribution\nnot available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Score Distribution')
            
            # Top combinations
            top_combinations = self.get_top_combinations(10)
            if top_combinations:
                top_scores = [combo['score'] for combo in top_combinations]
                top_labels = [f"#{combo['rank']}" for combo in top_combinations]
                
                bars = axes[0, 2].bar(range(len(top_scores)), top_scores, alpha=0.7, color='gold')
                axes[0, 2].set_title('Top 10 Combinations')
                axes[0, 2].set_xticks(range(len(top_scores)))
                axes[0, 2].set_xticklabels(top_labels)
                axes[0, 2].set_ylabel('CV Score')
                
                # Add values on bars
                for bar, score in zip(bars, top_scores):
                    axes[0, 2].text(bar.get_x() + bar.get_width()/2, score + 0.005,
                                   f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                axes[0, 2].axis('off')
            
            # Convergence analysis
            if 'convergence_analysis' in analysis:
                convergence = analysis['convergence_analysis']
                iterations = range(1, len(convergence['scores_progression']) + 1)
                
                axes[1, 0].plot(iterations, convergence['scores_progression'], 'o-', alpha=0.6, 
                               label='Individual Scores', markersize=3)
                axes[1, 0].plot(iterations, convergence['best_scores_progression'], 'r-', linewidth=2, 
                               label='Best So Far')
                
                # Mark improvement iterations
                improvements = convergence['improvement_iterations']
                if improvements:
                    improvement_scores = [convergence['best_scores_progression'][i] for i in improvements]
                    axes[1, 0].scatter([i+1 for i in improvements], improvement_scores, 
                                      color='green', s=50, marker='^', label='Improvements', zorder=5)
                
                axes[1, 0].set_title('Convergence Analysis')
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('CV Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Convergence analysis\nnot available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Convergence Analysis')
            
            # Parameter correlation heatmap (for numeric parameters)
            numeric_params = {k: v for k, v in (self.parameter_analysis_ or {}).items() 
                            if v.get('type') == 'numeric'}
            
            if len(numeric_params) >= 2:
                # Create correlation matrix
                param_names = list(numeric_params.keys())[:5]  # Limit to 5 parameters
                correlations = []
                
                for i, param1 in enumerate(param_names):
                    row = []
                    for j, param2 in enumerate(param_names):
                        if i == j:
                            row.append(1.0)
                        else:
                            # Get correlation from parameter analysis
                            corr1 = numeric_params[param1]['correlation_with_score']
                            corr2 = numeric_params[param2]['correlation_with_score']
                            # Simple approximation - in real analysis would compute cross-correlation
                            row.append((corr1 * corr2))
                    correlations.append(row)
                
                correlations = np.array(correlations)
                sns.heatmap(correlations, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                           xticklabels=param_names, yticklabels=param_names, ax=axes[1, 1])
                axes[1, 1].set_title('Parameter Score Correlations')
            else:
                axes[1, 1].text(0.5, 0.5, 'Not enough numeric\nparameters for correlation', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Parameter Correlations')
            
            # Random search summary
            summary_text = f"Random Search Results\n"
            summary_text += f"Strategy: {analysis['sampling_strategy'].title()}\n"
            summary_text += f"Evaluations: {analysis['total_evaluations']}\n"
            summary_text += f"Best Score: {analysis['best_score']:.4f}\n"
            summary_text += f"Score Range: {analysis['score_range']:.4f}\n"
            summary_text += f"Mean Score: {analysis['mean_score']:.4f}\n"
            summary_text += f"Score Std: {analysis['score_std']:.4f}\n"
            
            if self.early_stopping:
                summary_text += f"\nEarly Stopping: Enabled\n"
                if hasattr(self.random_search_cv_, 'n_evaluations_'):
                    efficiency = self.random_search_cv_.n_evaluations_ / self.n_iter
                    summary_text += f"Efficiency: {efficiency:.1%}"
            
            if 'convergence_analysis' in analysis:
                convergence = analysis['convergence_analysis']
                summary_text += f"\nImprovements: {len(convergence['improvement_iterations'])}\n"
                summary_text += f"Convergence Rate: {convergence['convergence_rate']:.4f}"
            
            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            axes[1, 2].set_title('Random Search Summary')
            axes[1, 2].axis('off')
            
            plt.suptitle(f'Random Search Analysis - {self.sampling_strategy.title()} Sampling', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            logger.warning(f"Error creating random search analysis plot: {e}")
            return None
    
    def save_optimization_results(self, filepath: str):
        """Save optimization results to file"""
        try:
            results = self._create_optimization_results()
            
            # Remove non-serializable objects
            serializable_results = results.copy()
            serializable_results.pop('best_model', None)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Random search results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving random search results: {e}")

# ============================================
# Adaptive Random Search
# ============================================

class AdaptiveRandomSearchOptimizer:
    """Adaptive random search that focuses on promising parameter regions"""
    
    def __init__(self,
                 model_factory: Callable,
                 parameter_space: Dict[str, Union[List, Tuple, Any]],
                 scoring: str = 'accuracy',
                 cv_folds: int = 5,
                 n_iter: int = 100,
                 adaptation_rounds: int = 3,
                 elite_ratio: float = 0.2,
                 exploration_ratio: float = 0.3,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Adaptive Random Search Optimizer
        
        Args:
            model_factory: Function that creates model instances
            parameter_space: Dictionary defining parameter search space
            scoring: Scoring metric to optimize
            cv_folds: Number of cross-validation folds
            n_iter: Total number of iterations
            adaptation_rounds: Number of adaptation rounds
            elite_ratio: Ratio of elite solutions to use for adaptation
            exploration_ratio: Ratio of random exploration vs focused search
            random_state: Random seed
        """
        self.model_factory = model_factory
        self.parameter_space = parameter_space
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.adaptation_rounds = adaptation_rounds
        self.elite_ratio = elite_ratio
        self.exploration_ratio = exploration_ratio
        self.random_state = random_state
        
        # Results
        self.adaptation_history_ = []
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        self.all_evaluations_ = []
        
        logger.info(f"Initialized adaptive random search with {adaptation_rounds} adaptation rounds")
    
    def _adapt_parameter_space(self, elite_solutions: List[Dict[str, Any]]) -> Dict[str, Union[List, Tuple, Any]]:
        """Adapt parameter space based on elite solutions"""
        
        adapted_space = {}
        
        for param_name, param_values in self.parameter_space.items():
            elite_values = [sol['params'][param_name] for sol in elite_solutions if param_name in sol['params']]
            
            if not elite_values:
                # Keep original space if no elite values
                adapted_space[param_name] = param_values
                continue
            
            # Check if parameter is numeric or categorical
            if isinstance(param_values, (tuple, list)) and len(param_values) == 2 and all(isinstance(x, (int, float)) for x in param_values):
                # Numeric range - adapt based on elite values
                elite_mean = np.mean(elite_values)
                elite_std = np.std(elite_values)
                
                # Expand range around elite region
                expansion_factor = 2.0
                new_low = max(param_values[0], elite_mean - expansion_factor * elite_std)
                new_high = min(param_values[1], elite_mean + expansion_factor * elite_std)
                
                # Ensure valid range
                if new_low >= new_high:
                    new_low = param_values[0]
                    new_high = param_values[1]
                
                adapted_space[param_name] = (new_low, new_high)
                
            else:
                # Categorical - weight towards elite values
                from collections import Counter
                elite_counts = Counter(elite_values)
                
                # Get top elite values
                top_elite = [val for val, count in elite_counts.most_common()]
                
                # Include some original values for exploration
                if isinstance(param_values, (list, tuple)):
                    original_values = list(param_values)
                else:
                    original_values = [param_values]
                
                # Combine elite and original values (elite values have higher probability)
                adapted_values = top_elite + original_values
                adapted_space[param_name] = adapted_values
        
        return adapted_space
    
    def optimize(self, X: Union[pd.DataFrame, np.ndarray], 
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Run adaptive random search optimization"""
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        logger.info("Starting adaptive random search optimization")
        
        current_param_space = self.parameter_space.copy()
        iterations_per_round = self.n_iter // self.adaptation_rounds
        
        for round_num in range(self.adaptation_rounds):
            logger.info(f"Adaptation round {round_num + 1}/{self.adaptation_rounds}")
            
            # Determine iterations for this round
            if round_num == self.adaptation_rounds - 1:
                # Last round - use remaining iterations
                round_iterations = self.n_iter - (iterations_per_round * (self.adaptation_rounds - 1))
            else:
                round_iterations = iterations_per_round
            
            # Split iterations between exploration and focused search
            exploration_iterations = int(round_iterations * self.exploration_ratio)
            focused_iterations = round_iterations - exploration_iterations
            
            round_evaluations = []
            
            # Exploration phase - use original parameter space
            if exploration_iterations > 0:
                explorer = RandomSearchOptimizer(
                    model_factory=self.model_factory,
                    parameter_space=self.parameter_space,
                    scoring=self.scoring,
                    cv_folds=self.cv_folds,
                    n_iter=exploration_iterations,
                    sampling_strategy='latin_hypercube',
                    verbose=False,
                    random_state=self.random_state + round_num
                )
                
                exploration_results = explorer.optimize(X, y)
                
                # Add exploration evaluations
                for eval_data in exploration_results.get('optimization_history', []):
                    eval_data['phase'] = 'exploration'
                    eval_data['round'] = round_num + 1
                    round_evaluations.append(eval_data)
            
            # Focused phase - use adapted parameter space
            if focused_iterations > 0:
                focuser = RandomSearchOptimizer(
                    model_factory=self.model_factory,
                    parameter_space=current_param_space,
                    scoring=self.scoring,
                    cv_folds=self.cv_folds,
                    n_iter=focused_iterations,
                    sampling_strategy='sobol',
                    verbose=False,
                    random_state=self.random_state + round_num + 100
                )
                
                focused_results = focuser.optimize(X, y)
                
                # Add focused evaluations
                for eval_data in focused_results.get('optimization_history', []):
                    eval_data['phase'] = 'focused'
                    eval_data['round'] = round_num + 1
                    round_evaluations.append(eval_data)
            
            # Store round results
            round_info = {
                'round': round_num + 1,
                'parameter_space': current_param_space.copy(),
                'evaluations': round_evaluations,
                'exploration_iterations': exploration_iterations,
                'focused_iterations': focused_iterations,
                'best_score_this_round': max(eval_data['mean_test_score'] for eval_data in round_evaluations)
            }
            self.adaptation_history_.append(round_info)
            
            # Add to all evaluations
            self.all_evaluations_.extend(round_evaluations)
            
            # Update best overall
            for eval_data in round_evaluations:
                if eval_data['mean_test_score'] > (self.best_score_ or -np.inf):
                    self.best_score_ = eval_data['mean_test_score']
                    self.best_params_ = eval_data['params']
            
            # Adapt parameter space for next round based on elite solutions
            if round_num < self.adaptation_rounds - 1:
                # Get elite solutions from all evaluations so far
                all_scores = [eval_data['mean_test_score'] for eval_data in self.all_evaluations_]
                elite_threshold = np.percentile(all_scores, (1 - self.elite_ratio) * 100)
                
                elite_solutions = [eval_data for eval_data in self.all_evaluations_ 
                                 if eval_data['mean_test_score'] >= elite_threshold]
                
                if elite_solutions:
                    current_param_space = self._adapt_parameter_space(elite_solutions)
                    logger.info(f"Adapted parameter space based on {len(elite_solutions)} elite solutions")
        
        # Create best model
        if self.best_params_:
            self.best_model_ = self.model_factory(**self.best_params_)
            self.best_model_.fit(X, y)
        
        logger.info(f"Adaptive random search completed. Best score: {self.best_score_:.4f}")
        
        return self._create_adaptive_results()
    
    def _create_adaptive_results(self) -> Dict[str, Any]:
        """Create comprehensive adaptive optimization results"""
        
        results = {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'best_model': self.best_model_,
            'adaptation_history': self.adaptation_history_,
            'all_evaluations': self.all_evaluations_,
            'optimization_method': 'adaptive_random_search',
            'adaptation_rounds': self.adaptation_rounds,
            'total_evaluations': len(self.all_evaluations_)
        }
        
        # Analysis across adaptation rounds
        round_analysis = {}
        for round_info in self.adaptation_history_:
            round_num = round_info['round']
            round_analysis[f'round_{round_num}'] = {
                'best_score': round_info['best_score_this_round'],
                'n_evaluations': len(round_info['evaluations']),
                'exploration_ratio': round_info['exploration_iterations'] / len(round_info['evaluations']) if round_info['evaluations'] else 0
            }
        
        results['round_analysis'] = round_analysis
        
        # Convergence analysis
        round_scores = [info['best_score_this_round'] for info in self.adaptation_history_]
        if len(round_scores) > 1:
            improvements = [round_scores[i] - round_scores[i-1] for i in range(1, len(round_scores))]
            results['convergence_analysis'] = {
                'round_scores': round_scores,
                'improvements': improvements,
                'total_improvement': round_scores[-1] - round_scores[0] if round_scores else 0,
                'adaptation_effectiveness': np.mean([imp for imp in improvements if imp > 0]) if any(imp > 0 for imp in improvements) else 0
            }
        
        return results

# ============================================
# Factory Functions and Utilities
# ============================================

def optimize_model_random_search(model_name: str,
                                parameter_space: Dict[str, Union[List, Tuple, Any]],
                                X: Union[pd.DataFrame, np.ndarray],
                                y: Union[pd.Series, np.ndarray],
                                **optimizer_kwargs) -> Dict[str, Any]:
    """Convenient function to optimize specific model types with random search"""
    
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
    optimizer = RandomSearchOptimizer(
        model_factory=model_factory,
        parameter_space=parameter_space,
        **optimizer_kwargs
    )
    
    # Run optimization
    results = optimizer.optimize(X, y)
    
    return results

def create_financial_random_spaces(model_name: str) -> Dict[str, Union[List, Tuple, Any]]:
    """Create random search spaces optimized for financial models"""
    
    spaces = {
        'gradient_boosting': {
            'n_estimators': (50, 500),
            'max_depth': (3, 15),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.6, 1.0),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10)
        },
        'random_forest': {
            'n_estimators': (50, 300),
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': (2, 15),
            'min_samples_leaf': (1, 8),
            'max_features': ['sqrt', 'log2', 0.5, 0.8]
        },
        'svm': {
            'C': (0.1, 100),
            'gamma': (1e-4, 1),
            'kernel': ['rbf', 'poly', 'sigmoid']
        },
        'logistic': {
            'C': (0.01, 100),
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga', 'lbfgs']
        },
        'neural_network': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
            'alpha': (1e-5, 1e-1),
            'learning_rate_init': (1e-4, 1e-1),
            'max_iter': (100, 1000)
        }
    }
    
    if model_name not in spaces:
        raise ValueError(f"No random search space for model: {model_name}")
    
    return spaces[model_name]

def compare_sampling_strategies(X: Union[pd.DataFrame, np.ndarray],
                              y: Union[pd.Series, np.ndarray],
                              model_factory: Callable,
                              parameter_space: Dict[str, Union[List, Tuple, Any]],
                              strategies: List[str] = ['random', 'latin_hypercube', 'sobol', 'halton']) -> Dict[str, Any]:
    """Compare different random sampling strategies"""
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"Testing sampling strategy: {strategy}")
        
        try:
            optimizer = RandomSearchOptimizer(
                model_factory=model_factory,
                parameter_space=parameter_space,
                sampling_strategy=strategy,
                n_iter=50,  # Reduced for comparison
                verbose=False,
                random_state=42
            )
            
            strategy_results = optimizer.optimize(X, y)
            
            results[strategy] = {
                'best_score': strategy_results['best_score'],
                'best_params': strategy_results['best_params'],
                'sampling_analysis': strategy_results['sampling_analysis'],
                'convergence_rate': strategy_results['sampling_analysis'].get('convergence_analysis', {}).get('convergence_rate', 0)
            }
            
        except Exception as e:
            logger.warning(f"Error with strategy {strategy}: {e}")
            results[strategy] = {'error': str(e)}
    
    # Find best strategy
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_strategy = max(valid_results.keys(), key=lambda k: valid_results[k]['best_score'])
        
        results['comparison'] = {
            'best_strategy': best_strategy,
            'strategy_rankings': sorted(valid_results.keys(), 
                                      key=lambda k: valid_results[k]['best_score'], reverse=True),
            'convergence_analysis': {
                strategy: {
                    'convergence_rate': v['convergence_rate'],
                    'score_improvement': v['best_score']
                }
                for strategy, v in valid_results.items()
            }
        }
    
    return results
