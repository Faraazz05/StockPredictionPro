# ============================================
# StockPredictionPro - src/models/optimization/grid_search.py
# Advanced grid search optimization with intelligent parameter exploration
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
import warnings
import json
from collections import defaultdict
from itertools import product, combinations
import time

# Core ML imports
from sklearn.model_selection import (
    GridSearchCV, ParameterGrid, cross_val_score, 
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

logger = get_logger('models.optimization.grid_search')

# ============================================
# Advanced Grid Search Framework
# ============================================

class GridSearchOptimizer:
    """Advanced grid search with intelligent parameter exploration and early stopping"""
    
    def __init__(self,
                 model_factory: Callable,
                 parameter_grid: Dict[str, List[Any]],
                 scoring: str = 'accuracy',
                 cv_folds: int = 5,
                 n_jobs: Optional[int] = -1,
                 time_aware_cv: bool = True,
                 early_stopping: bool = False,
                 early_stopping_patience: int = 10,
                 min_improvement: float = 0.001,
                 refit: bool = True,
                 verbose: int = 1,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Grid Search Optimizer
        
        Args:
            model_factory: Function that creates model instances
            parameter_grid: Dictionary or list of dictionaries with parameter grids
            scoring: Scoring metric to optimize
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all cores)
            time_aware_cv: Whether to use time-aware cross-validation
            early_stopping: Whether to use early stopping
            early_stopping_patience: Number of iterations to wait for improvement
            min_improvement: Minimum improvement threshold for early stopping
            refit: Whether to refit on full dataset with best parameters
            verbose: Verbosity level
            random_state: Random seed
        """
        self.model_factory = model_factory
        self.parameter_grid = parameter_grid
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.time_aware_cv = time_aware_cv
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.min_improvement = min_improvement
        self.refit = refit
        self.verbose = verbose
        self.random_state = random_state
        
        # Results
        self.grid_search_cv_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        self.optimization_history_ = []
        self.grid_analysis_ = None
        self.parameter_importance_ = {}
        
        # Validation
        self._validate_parameters()
        
        logger.info(f"Initialized grid search optimizer with {self._count_parameter_combinations()} combinations")
    
    def _validate_parameters(self):
        """Validate grid search parameters"""
        if not isinstance(self.parameter_grid, (dict, list)):
            raise ValueError("parameter_grid must be a dictionary or list of dictionaries")
        
        if self.cv_folds <= 1:
            raise ValueError("cv_folds must be greater than 1")
        
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
    
    def _count_parameter_combinations(self) -> int:
        """Count total number of parameter combinations"""
        if isinstance(self.parameter_grid, dict):
            return len(list(ParameterGrid(self.parameter_grid)))
        elif isinstance(self.parameter_grid, list):
            total = 0
            for grid in self.parameter_grid:
                total += len(list(ParameterGrid(grid)))
            return total
        else:
            return 0
    
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
    
    def _create_model_for_params(self, params: Dict[str, Any]) -> BaseEstimator:
        """Create model instance with specific parameters"""
        try:
            return self.model_factory(**params)
        except Exception as e:
            logger.warning(f"Error creating model with params {params}: {e}")
            # Return model with default parameters
            return self.model_factory()
    
    def _evaluate_parameter_combination(self, params: Dict[str, Any], 
                                      X: np.ndarray, y: np.ndarray) -> Tuple[float, float, List[float]]:
        """Evaluate a single parameter combination"""
        
        model = self._create_model_for_params(params)
        cv_splitter = self._create_cv_splitter(X, y)
        
        try:
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
    
    def _grid_search_with_early_stopping(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run grid search with early stopping"""
        
        # Generate all parameter combinations
        if isinstance(self.parameter_grid, dict):
            param_combinations = list(ParameterGrid(self.parameter_grid))
        else:
            param_combinations = []
            for grid in self.parameter_grid:
                param_combinations.extend(list(ParameterGrid(grid)))
        
        # Randomize order for better early stopping
        np.random.seed(self.random_state)
        np.random.shuffle(param_combinations)
        
        best_score = -np.inf
        best_params = None
        no_improvement_count = 0
        
        results = []
        
        logger.info(f"Starting grid search with early stopping over {len(param_combinations)} combinations")
        
        for i, params in enumerate(param_combinations):
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
            if mean_score > best_score + self.min_improvement:
                best_score = mean_score
                best_params = params.copy()
                no_improvement_count = 0
                
                if self.verbose >= 1:
                    logger.info(f"New best score: {best_score:.4f} with params {params}")
            else:
                no_improvement_count += 1
            
            # Early stopping check
            if self.early_stopping and no_improvement_count >= self.early_stopping_patience:
                logger.info(f"Early stopping at iteration {i + 1}: no improvement for {self.early_stopping_patience} iterations")
                break
            
            if self.verbose >= 2:
                logger.info(f"Iteration {i + 1}/{len(param_combinations)}: {mean_score:.4f} ± {std_score:.4f}")
        
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
    
    @time_it("grid_search_optimization", include_args=True)
    def optimize(self, X: Union[pd.DataFrame, np.ndarray], 
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Run grid search optimization"""
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        logger.info(f"Starting grid search optimization")
        
        try:
            if self.early_stopping:
                # Use custom grid search with early stopping
                grid_results = self._grid_search_with_early_stopping(X, y)
                
                # Create a mock GridSearchCV object for compatibility
                class MockGridSearchCV:
                    def __init__(self, results):
                        self.best_params_ = results['best_params_']
                        self.best_score_ = results['best_score_']
                        self.cv_results_ = results['cv_results_']
                        self.n_evaluations_ = results['n_evaluations_']
                
                self.grid_search_cv_ = MockGridSearchCV(grid_results)
                
            else:
                # Use standard scikit-learn GridSearchCV
                base_model = self.model_factory()
                cv_splitter = self._create_cv_splitter(X, y)
                
                self.grid_search_cv_ = GridSearchCV(
                    estimator=base_model,
                    param_grid=self.parameter_grid,
                    scoring=self.scoring,
                    cv=cv_splitter,
                    n_jobs=self.n_jobs,
                    refit=self.refit,
                    verbose=self.verbose,
                    error_score='raise',
                    return_train_score=True
                )
                
                self.grid_search_cv_.fit(X, y)
            
            # Extract results
            self.best_params_ = self.grid_search_cv_.best_params_
            self.best_score_ = self.grid_search_cv_.best_score_
            
            # Create best model
            if self.refit and hasattr(self.grid_search_cv_, 'best_estimator_'):
                self.best_model_ = self.grid_search_cv_.best_estimator_
            else:
                self.best_model_ = self._create_model_for_params(self.best_params_)
                self.best_model_.fit(X, y)
            
            # Analyze grid search results
            self._analyze_grid_results()
            
            logger.info(f"Grid search completed. Best score: {self.best_score_:.4f}")
            
            return self._create_optimization_results()
            
        except Exception as e:
            logger.error(f"Grid search optimization failed: {e}")
            raise
    
    def _analyze_grid_results(self):
        """Analyze grid search results for insights"""
        
        if not hasattr(self.grid_search_cv_, 'cv_results_'):
            return
        
        cv_results = self.grid_search_cv_.cv_results_
        
        # Parameter importance analysis
        self.parameter_importance_ = self._calculate_parameter_importance(cv_results)
        
        # Grid analysis
        self.grid_analysis_ = {
            'total_combinations': len(cv_results['mean_test_score']),
            'best_score': self.best_score_,
            'worst_score': min(cv_results['mean_test_score']),
            'score_range': max(cv_results['mean_test_score']) - min(cv_results['mean_test_score']),
            'mean_score': np.mean(cv_results['mean_test_score']),
            'std_score': np.std(cv_results['mean_test_score']),
            'parameter_importance': self.parameter_importance_
        }
        
        # Top performing parameter combinations
        scores = cv_results['mean_test_score']
        top_indices = np.argsort(scores)[-10:][::-1]  # Top 10
        
        self.grid_analysis_['top_combinations'] = []
        for idx in top_indices:
            combination = {
                'params': cv_results['params'][idx],
                'score': scores[idx],
                'std': cv_results['std_test_score'][idx]
            }
            self.grid_analysis_['top_combinations'].append(combination)
    
    def _calculate_parameter_importance(self, cv_results: Dict[str, List]) -> Dict[str, float]:
        """Calculate parameter importance based on score variance"""
        
        params_list = cv_results['params']
        scores = np.array(cv_results['mean_test_score'])
        
        # Get all parameter names
        all_param_names = set()
        for params in params_list:
            all_param_names.update(params.keys())
        
        importance = {}
        
        for param_name in all_param_names:
            # Group scores by parameter value
            param_groups = defaultdict(list)
            for params, score in zip(params_list, scores):
                if param_name in params:
                    param_groups[params[param_name]].append(score)
            
            if len(param_groups) > 1:
                # Calculate variance between groups
                group_means = [np.mean(group_scores) for group_scores in param_groups.values()]
                between_group_variance = np.var(group_means)
                
                # Calculate within-group variance
                within_group_variances = [np.var(group_scores) for group_scores in param_groups.values()]
                within_group_variance = np.mean(within_group_variances)
                
                # Importance as ratio of between to within variance
                if within_group_variance > 0:
                    importance[param_name] = between_group_variance / within_group_variance
                else:
                    importance[param_name] = between_group_variance
            else:
                importance[param_name] = 0.0
        
        # Normalize importance scores
        max_importance = max(importance.values()) if importance else 1.0
        if max_importance > 0:
            importance = {k: v / max_importance for k, v in importance.items()}
        
        return importance
    
    def _create_optimization_results(self) -> Dict[str, Any]:
        """Create comprehensive optimization results"""
        
        results = {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'best_model': self.best_model_,
            'optimization_history': self.optimization_history_,
            'grid_analysis': self.grid_analysis_,
            'parameter_importance': self.parameter_importance_,
            'optimization_method': 'grid_search',
            'early_stopping_used': self.early_stopping
        }
        
        # Add CV results if available
        if hasattr(self.grid_search_cv_, 'cv_results_'):
            results['cv_results'] = self.grid_search_cv_.cv_results_
        
        # Add evaluation count
        if hasattr(self.grid_search_cv_, 'n_evaluations_'):
            results['n_evaluations'] = self.grid_search_cv_.n_evaluations_
        else:
            results['n_evaluations'] = len(self.optimization_history_)
        
        return results
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance scores"""
        return self.parameter_importance_.copy()
    
    def get_top_combinations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N parameter combinations"""
        if self.grid_analysis_ and 'top_combinations' in self.grid_analysis_:
            return self.grid_analysis_['top_combinations'][:n]
        return []
    
    def plot_grid_analysis(self) -> Any:
        """Plot grid search analysis results"""
        if not self.grid_analysis_:
            logger.warning("Grid analysis not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            analysis = self.grid_analysis_
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Parameter importance
            if self.parameter_importance_:
                params = list(self.parameter_importance_.keys())
                importances = list(self.parameter_importance_.values())
                
                bars = axes[0, 0].bar(range(len(params)), importances, alpha=0.7, color='steelblue')
                axes[0, 0].set_title('Parameter Importance')
                axes[0, 0].set_xticks(range(len(params)))
                axes[0, 0].set_xticklabels(params, rotation=45, ha='right')
                axes[0, 0].set_ylabel('Importance Score')
                
                # Add values on bars
                for bar, imp in zip(bars, importances):
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, imp + max(importances) * 0.01,
                                   f'{imp:.3f}', ha='center', va='bottom', fontsize=9)
            else:
                axes[0, 0].text(0.5, 0.5, 'Parameter importance\nnot available', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Parameter Importance')
            
            # Score distribution
            if hasattr(self.grid_search_cv_, 'cv_results_'):
                scores = self.grid_search_cv_.cv_results_['mean_test_score']
                
                axes[0, 1].hist(scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[0, 1].axvline(self.best_score_, color='red', linestyle='--', linewidth=2,
                                  label=f'Best: {self.best_score_:.3f}')
                axes[0, 1].axvline(np.mean(scores), color='blue', linestyle='--', linewidth=2,
                                  label=f'Mean: {np.mean(scores):.3f}')
                axes[0, 1].set_title('Score Distribution')
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
                top_labels = [f"Top {i+1}" for i in range(len(top_scores))]
                
                bars = axes[0, 2].bar(range(len(top_scores)), top_scores, alpha=0.7, color='gold')
                axes[0, 2].set_title('Top 10 Combinations')
                axes[0, 2].set_xticks(range(len(top_scores)))
                axes[0, 2].set_xticklabels(top_labels, rotation=45)
                axes[0, 2].set_ylabel('CV Score')
                
                # Add values on bars
                for bar, score in zip(bars, top_scores):
                    axes[0, 2].text(bar.get_x() + bar.get_width()/2, score + 0.005,
                                   f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                axes[0, 2].axis('off')
            
            # Convergence plot (if optimization history available)
            if self.optimization_history_:
                iterations = range(1, len(self.optimization_history_) + 1)
                scores = [eval_data['mean_test_score'] for eval_data in self.optimization_history_]
                best_so_far = np.maximum.accumulate(scores)
                
                axes[1, 0].plot(iterations, scores, 'o-', alpha=0.6, label='Individual Scores')
                axes[1, 0].plot(iterations, best_so_far, 'r-', linewidth=2, label='Best So Far')
                axes[1, 0].set_title('Optimization Progress')
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('CV Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Optimization history\nnot available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Optimization Progress')
            
            # Parameter exploration heatmap (for 2 parameters)
            if (hasattr(self.grid_search_cv_, 'cv_results_') and 
                self.parameter_importance_ and len(self.parameter_importance_) >= 2):
                
                # Get top 2 most important parameters
                sorted_params = sorted(self.parameter_importance_.items(), 
                                     key=lambda x: x[1], reverse=True)[:2]
                
                if len(sorted_params) == 2:
                    param1_name, param2_name = sorted_params[0][0], sorted_params[1][0]
                    
                    # Create heatmap data
                    param_combinations = self.grid_search_cv_.cv_results_['params']
                    scores = self.grid_search_cv_.cv_results_['mean_test_score']
                    
                    # Extract unique parameter values
                    param1_values = sorted(set(p.get(param1_name) for p in param_combinations if param1_name in p))
                    param2_values = sorted(set(p.get(param2_name) for p in param_combinations if param2_name in p))
                    
                    if param1_values and param2_values and len(param1_values) * len(param2_values) <= 100:
                        # Create score matrix
                        score_matrix = np.full((len(param2_values), len(param1_values)), np.nan)
                        
                        for params, score in zip(param_combinations, scores):
                            if param1_name in params and param2_name in params:
                                p1_idx = param1_values.index(params[param1_name])
                                p2_idx = param2_values.index(params[param2_name])
                                score_matrix[p2_idx, p1_idx] = score
                        
                        # Plot heatmap
                        sns.heatmap(score_matrix, 
                                   xticklabels=param1_values, 
                                   yticklabels=param2_values,
                                   annot=True, fmt='.3f', cmap='viridis',
                                   ax=axes[1, 1])
                        axes[1, 1].set_title(f'Parameter Heatmap\n{param1_name} vs {param2_name}')
                        axes[1, 1].set_xlabel(param1_name)
                        axes[1, 1].set_ylabel(param2_name)
                    else:
                        axes[1, 1].text(0.5, 0.5, 'Too many parameter\ncombinations for heatmap', 
                                       ha='center', va='center', transform=axes[1, 1].transAxes)
                        axes[1, 1].set_title('Parameter Heatmap')
                else:
                    axes[1, 1].axis('off')
            else:
                axes[1, 1].axis('off')
            
            # Grid search summary
            summary_text = f"Grid Search Results\n"
            summary_text += f"Total Combinations: {analysis['total_combinations']}\n"
            summary_text += f"Best Score: {analysis['best_score']:.4f}\n"
            summary_text += f"Score Range: {analysis['score_range']:.4f}\n"
            summary_text += f"Mean Score: {analysis['mean_score']:.4f}\n"
            summary_text += f"Score Std: {analysis['std_score']:.4f}\n"
            
            if self.early_stopping:
                summary_text += f"\nEarly Stopping: Enabled\n"
                if hasattr(self.grid_search_cv_, 'n_evaluations_'):
                    total_combinations = self._count_parameter_combinations()
                    evaluated = self.grid_search_cv_.n_evaluations_
                    summary_text += f"Evaluated: {evaluated}/{total_combinations}\n"
                    summary_text += f"Efficiency: {evaluated/total_combinations:.1%}"
            
            # Most important parameter
            if self.parameter_importance_:
                most_important = max(self.parameter_importance_.items(), key=lambda x: x[1])
                summary_text += f"\nMost Important Parameter:\n{most_important[0]} ({most_important[1]:.3f})"
            
            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            axes[1, 2].set_title('Grid Search Summary')
            axes[1, 2].axis('off')
            
            plt.suptitle('Grid Search Optimization Analysis', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            logger.warning(f"Error creating grid search analysis plot: {e}")
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
            
            logger.info(f"Grid search results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving grid search results: {e}")

# ============================================
# Adaptive Grid Search
# ============================================

class AdaptiveGridSearchOptimizer:
    """Adaptive grid search that refines promising regions"""
    
    def __init__(self,
                 model_factory: Callable,
                 initial_parameter_grid: Dict[str, List[Any]],
                 scoring: str = 'accuracy',
                 cv_folds: int = 5,
                 refinement_levels: int = 2,
                 top_k_refine: int = 3,
                 expansion_factor: float = 0.5,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Adaptive Grid Search Optimizer
        
        Args:
            model_factory: Function that creates model instances
            initial_parameter_grid: Initial coarse grid
            scoring: Scoring metric to optimize
            cv_folds: Number of cross-validation folds
            refinement_levels: Number of refinement iterations
            top_k_refine: Number of top combinations to refine around
            expansion_factor: Factor to expand around best points
            random_state: Random seed
        """
        self.model_factory = model_factory
        self.initial_parameter_grid = initial_parameter_grid
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.refinement_levels = refinement_levels
        self.top_k_refine = top_k_refine
        self.expansion_factor = expansion_factor
        self.random_state = random_state
        
        # Results
        self.refinement_history_ = []
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        
        logger.info(f"Initialized adaptive grid search with {refinement_levels} refinement levels")
    
    def _refine_parameter_grid(self, best_combinations: List[Dict[str, Any]], 
                              original_grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Create refined parameter grid around best combinations"""
        
        refined_grid = {}
        
        for param_name, param_values in original_grid.items():
            # Get values from best combinations
            best_values = []
            for combo in best_combinations:
                if param_name in combo['params']:
                    best_values.append(combo['params'][param_name])
            
            if not best_values:
                # Keep original values if no best values found
                refined_grid[param_name] = param_values
                continue
            
            # Determine parameter type
            if all(isinstance(v, (int, float)) for v in param_values):
                # Numeric parameter - create refined range
                numeric_values = [v for v in param_values if isinstance(v, (int, float))]
                
                if numeric_values:
                    min_val, max_val = min(numeric_values), max(numeric_values)
                    value_range = max_val - min_val
                    
                    # Create refined values around best
                    refined_values = set()
                    for best_val in best_values:
                        # Add expansion around best value
                        expansion = value_range * self.expansion_factor / len(param_values)
                        
                        if isinstance(best_val, int):
                            # Integer parameter
                            expansion = max(1, int(expansion))
                            for offset in range(-expansion, expansion + 1):
                                new_val = best_val + offset
                                if min_val <= new_val <= max_val:
                                    refined_values.add(new_val)
                        else:
                            # Float parameter
                            for i in range(5):  # 5 points around best
                                offset = (i - 2) * expansion / 2
                                new_val = best_val + offset
                                if min_val <= new_val <= max_val:
                                    refined_values.add(round(new_val, 6))
                    
                    refined_grid[param_name] = sorted(list(refined_values))
                else:
                    refined_grid[param_name] = param_values
            else:
                # Categorical parameter - keep only values that appeared in best
                refined_values = list(set(best_values))
                if not refined_values:
                    refined_values = param_values
                refined_grid[param_name] = refined_values
        
        return refined_grid
    
    def optimize(self, X: Union[pd.DataFrame, np.ndarray], 
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """Run adaptive grid search optimization"""
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        logger.info("Starting adaptive grid search optimization")
        
        current_grid = self.initial_parameter_grid.copy()
        all_results = []
        
        for level in range(self.refinement_levels + 1):
            logger.info(f"Refinement level {level + 1}/{self.refinement_levels + 1}")
            
            # Run grid search on current grid
            optimizer = GridSearchOptimizer(
                model_factory=self.model_factory,
                parameter_grid=current_grid,
                scoring=self.scoring,
                cv_folds=self.cv_folds,
                verbose=0,
                random_state=self.random_state
            )
            
            level_results = optimizer.optimize(X, y)
            
            # Store level results
            level_info = {
                'level': level + 1,
                'parameter_grid': current_grid.copy(),
                'best_score': level_results['best_score'],
                'best_params': level_results['best_params'],
                'n_combinations': optimizer._count_parameter_combinations(),
                'results': level_results
            }
            self.refinement_history_.append(level_info)
            
            # Collect all results
            if hasattr(optimizer.grid_search_cv_, 'cv_results_'):
                cv_results = optimizer.grid_search_cv_.cv_results_
                for i, params in enumerate(cv_results['params']):
                    result = {
                        'params': params,
                        'score': cv_results['mean_test_score'][i],
                        'std': cv_results['std_test_score'][i],
                        'level': level + 1
                    }
                    all_results.append(result)
            
            # Update best overall
            if level_results['best_score'] > (self.best_score_ or -np.inf):
                self.best_score_ = level_results['best_score']
                self.best_params_ = level_results['best_params']
                self.best_model_ = level_results['best_model']
            
            # Prepare for next refinement level
            if level < self.refinement_levels:
                # Get top combinations for refinement
                sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
                top_combinations = sorted_results[:self.top_k_refine]
                
                # Refine grid around top combinations
                current_grid = self._refine_parameter_grid(top_combinations, current_grid)
                
                logger.info(f"Refined grid: {current_grid}")
        
        logger.info(f"Adaptive grid search completed. Best score: {self.best_score_:.4f}")
        
        return self._create_adaptive_results(all_results)
    
    def _create_adaptive_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive adaptive optimization results"""
        
        results = {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'best_model': self.best_model_,
            'refinement_history': self.refinement_history_,
            'all_evaluations': all_results,
            'optimization_method': 'adaptive_grid_search',
            'refinement_levels': self.refinement_levels,
            'total_evaluations': len(all_results)
        }
        
        # Analysis across refinement levels
        level_analysis = {}
        for level_info in self.refinement_history_:
            level = level_info['level']
            level_analysis[f'level_{level}'] = {
                'best_score': level_info['best_score'],
                'n_combinations': level_info['n_combinations'],
                'improvement': level_info['best_score'] - (self.refinement_history_[0]['best_score'] if level > 1 else 0)
            }
        
        results['level_analysis'] = level_analysis
        
        # Convergence analysis
        level_scores = [info['best_score'] for info in self.refinement_history_]
        if len(level_scores) > 1:
            improvements = [level_scores[i] - level_scores[i-1] for i in range(1, len(level_scores))]
            results['convergence_analysis'] = {
                'level_scores': level_scores,
                'improvements': improvements,
                'total_improvement': level_scores[-1] - level_scores[0],
                'converged': all(imp < 0.001 for imp in improvements[-2:]) if len(improvements) >= 2 else False
            }
        
        return results

# ============================================
# Factory Functions and Utilities
# ============================================

def optimize_model_grid_search(model_name: str,
                              parameter_grid: Dict[str, List[Any]],
                              X: Union[pd.DataFrame, np.ndarray],
                              y: Union[pd.Series, np.ndarray],
                              **optimizer_kwargs) -> Dict[str, Any]:
    """Convenient function to optimize specific model types with grid search"""
    
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
    optimizer = GridSearchOptimizer(
        model_factory=model_factory,
        parameter_grid=parameter_grid,
        **optimizer_kwargs
    )
    
    # Run optimization
    results = optimizer.optimize(X, y)
    
    return results

def create_financial_grid_spaces(model_name: str, resolution: str = 'coarse') -> Dict[str, List[Any]]:
    """Create parameter grids optimized for financial models"""
    
    if resolution == 'coarse':
        spaces = {
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2']
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            },
            'logistic': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            }
        }
    elif resolution == 'fine':
        spaces = {
            'gradient_boosting': {
                'n_estimators': [50, 100, 150, 200, 300],
                'max_depth': [3, 5, 7, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            },
            'random_forest': {
                'n_estimators': [50, 100, 150, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5, 0.8]
            },
            'svm': {
                'C': [0.01, 0.1, 1, 10, 50, 100],
                'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'logistic': {
                'C': [0.001, 0.01, 0.1, 1, 10, 50, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga', 'lbfgs'],
                'max_iter': [100, 500, 1000]
            }
        }
    else:
        raise ValueError(f"Unknown resolution: {resolution}. Use 'coarse' or 'fine'")
    
    if model_name not in spaces:
        raise ValueError(f"No grid space for model: {model_name}")
    
    return spaces[model_name]

def compare_grid_strategies(X: Union[pd.DataFrame, np.ndarray],
                          y: Union[pd.Series, np.ndarray],
                          model_factory: Callable,
                          parameter_grid: Dict[str, List[Any]],
                          strategies: List[str] = ['standard', 'early_stopping', 'adaptive']) -> Dict[str, Any]:
    """Compare different grid search strategies"""
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"Testing grid search strategy: {strategy}")
        
        try:
            if strategy == 'standard':
                optimizer = GridSearchOptimizer(
                    model_factory=model_factory,
                    parameter_grid=parameter_grid,
                    early_stopping=False,
                    verbose=0
                )
                strategy_results = optimizer.optimize(X, y)
                
            elif strategy == 'early_stopping':
                optimizer = GridSearchOptimizer(
                    model_factory=model_factory,
                    parameter_grid=parameter_grid,
                    early_stopping=True,
                    early_stopping_patience=10,
                    verbose=0
                )
                strategy_results = optimizer.optimize(X, y)
                
            elif strategy == 'adaptive':
                optimizer = AdaptiveGridSearchOptimizer(
                    model_factory=model_factory,
                    initial_parameter_grid=parameter_grid,
                    refinement_levels=2,
                    top_k_refine=3
                )
                strategy_results = optimizer.optimize(X, y)
            
            results[strategy] = {
                'best_score': strategy_results['best_score'],
                'best_params': strategy_results['best_params'],
                'n_evaluations': strategy_results.get('total_evaluations', strategy_results.get('n_evaluations', 0)),
                'strategy_results': strategy_results
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
            'efficiency_analysis': {
                strategy: {
                    'score_per_evaluation': v['best_score'] / max(1, v['n_evaluations']),
                    'total_evaluations': v['n_evaluations']
                }
                for strategy, v in valid_results.items()
            }
        }
    
    return results
