# ============================================
# StockPredictionPro - src/evaluation/validation/walk_forward.py
# Advanced walk-forward validation for robust financial machine learning model evaluation
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator, Callable
import warnings
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.metrics import get_scorer

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('evaluation.validation.walk_forward')

# ============================================
# Basic Walk-Forward Validation
# ============================================

class WalkForwardSplit(BaseCrossValidator):
    """
    Walk-Forward Cross-Validation for time series data.
    
    This method simulates realistic trading conditions by:
    1. Training on historical data
    2. Testing on the immediate future period
    3. Moving forward in time and retraining
    4. Aggregating results across all forward steps
    
    Based on the methodology commonly used in algorithmic trading
    and quantitative finance.
    
    Parameters:
    -----------
    train_size : int or float
        Size of the training window. If int, represents number of samples.
        If float (0-1), represents fraction of total data.
    test_size : int or float, default=1
        Size of the test window. If int, represents number of samples.
        If float (0-1), represents fraction of total data.
    step_size : int, default=1
        Number of samples to step forward each iteration.
    window_type : str, default='expanding'
        Type of training window: 'expanding' or 'sliding'.
        - 'expanding': Training window grows with each step
        - 'sliding': Training window maintains fixed size
    min_train_size : int, optional
        Minimum required training samples.
    """
    
    def __init__(self, train_size: Union[int, float],
                 test_size: Union[int, float] = 1,
                 step_size: int = 1,
                 window_type: str = 'expanding',
                 min_train_size: Optional[int] = None):
        
        self.train_size = train_size
        self.test_size = test_size  
        self.step_size = step_size
        self.window_type = window_type
        self.min_train_size = min_train_size
        
        if window_type not in ['expanding', 'sliding']:
            raise ValueError("window_type must be 'expanding' or 'sliding'")
    
    def split(self, X, y=None, groups=None):
        """Generate walk-forward splits."""
        
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        # Convert fractional sizes to absolute
        if isinstance(self.train_size, float):
            initial_train_size = int(n_samples * self.train_size)
        else:
            initial_train_size = self.train_size
            
        if isinstance(self.test_size, float):
            test_window_size = max(1, int(n_samples * self.test_size))
        else:
            test_window_size = self.test_size
        
        # Set minimum training size
        min_train = self.min_train_size or max(1, initial_train_size // 4)
        
        # Start walk-forward process
        current_pos = initial_train_size
        
        while current_pos + test_window_size <= n_samples:
            # Calculate test window
            test_start = current_pos
            test_end = min(current_pos + test_window_size, n_samples)
            
            # Calculate training window
            if self.window_type == 'expanding':
                # Training window grows from start to current position
                train_start = 0
                train_end = current_pos
            else:  # sliding window
                # Training window maintains fixed size
                train_start = max(0, current_pos - initial_train_size)
                train_end = current_pos
            
            # Ensure minimum training size
            if train_end - train_start < min_train:
                current_pos += self.step_size
                continue
            
            # Generate indices
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            # Skip if either set is empty
            if len(train_indices) == 0 or len(test_indices) == 0:
                current_pos += self.step_size
                continue
            
            yield train_indices, test_indices
            
            # Move forward
            current_pos += self.step_size
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        if X is not None:
            n_samples = _num_samples(X)
            
            if isinstance(self.train_size, float):
                initial_train_size = int(n_samples * self.train_size)
            else:
                initial_train_size = self.train_size
                
            if isinstance(self.test_size, float):
                test_window_size = max(1, int(n_samples * self.test_size))
            else:
                test_window_size = self.test_size
            
            remaining_samples = n_samples - initial_train_size
            n_splits = max(0, (remaining_samples - test_window_size) // self.step_size + 1)
            
            return n_splits
        
        return 0

# ============================================
# Anchored Walk-Forward Validation
# ============================================

class AnchoredWalkForward(BaseCrossValidator):
    """
    Anchored Walk-Forward Validation.
    
    Training window always starts from the beginning (anchored),
    while test windows move forward. This simulates having access
    to all historical data when making predictions.
    
    Parameters:
    -----------
    initial_train_size : int or float
        Initial size of training data.
    test_size : int or float, default=1
        Size of each test window.
    step_size : int, default=1
        Number of samples to step forward.
    max_train_size : int, optional
        Maximum size of training window (for memory/computation limits).
    """
    
    def __init__(self, initial_train_size: Union[int, float],
                 test_size: Union[int, float] = 1,
                 step_size: int = 1,
                 max_train_size: Optional[int] = None):
        
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.max_train_size = max_train_size
    
    def split(self, X, y=None, groups=None):
        """Generate anchored walk-forward splits."""
        
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        # Convert fractional sizes
        if isinstance(self.initial_train_size, float):
            initial_train = int(n_samples * self.initial_train_size)
        else:
            initial_train = self.initial_train_size
            
        if isinstance(self.test_size, float):
            test_window_size = max(1, int(n_samples * self.test_size))
        else:
            test_window_size = self.test_size
        
        current_pos = initial_train
        
        while current_pos + test_window_size <= n_samples:
            # Test window
            test_start = current_pos
            test_end = min(current_pos + test_window_size, n_samples)
            
            # Training window (anchored at beginning)
            train_start = 0
            train_end = current_pos
            
            # Apply max training size constraint
            if self.max_train_size is not None and train_end - train_start > self.max_train_size:
                train_start = train_end - self.max_train_size
            
            # Generate indices
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
            
            current_pos += self.step_size
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        if X is not None:
            n_samples = _num_samples(X)
            
            if isinstance(self.initial_train_size, float):
                initial_train = int(n_samples * self.initial_train_size)
            else:
                initial_train = self.initial_train_size
                
            if isinstance(self.test_size, float):
                test_window_size = max(1, int(n_samples * self.test_size))
            else:
                test_window_size = self.test_size
            
            remaining_samples = n_samples - initial_train
            n_splits = max(0, (remaining_samples - test_window_size) // self.step_size + 1)
            
            return n_splits
        
        return 0

# ============================================
# Multi-Step Walk-Forward Validation
# ============================================

class MultiStepWalkForward(BaseCrossValidator):
    """
    Multi-Step Walk-Forward Validation for multi-horizon forecasting.
    
    Instead of single-step predictions, this method evaluates
    multi-step-ahead forecasts at each walk-forward step.
    
    Parameters:
    -----------
    train_size : int or float
        Size of training window.
    forecast_horizon : int, default=5
        Number of steps ahead to forecast.
    step_size : int, default=1
        Number of samples to step forward.
    window_type : str, default='sliding'
        Type of training window.
    overlap_strategy : str, default='skip'
        How to handle overlapping forecasts: 'skip' or 'average'.
    """
    
    def __init__(self, train_size: Union[int, float],
                 forecast_horizon: int = 5,
                 step_size: int = 1,
                 window_type: str = 'sliding',
                 overlap_strategy: str = 'skip'):
        
        self.train_size = train_size
        self.forecast_horizon = forecast_horizon
        self.step_size = step_size
        self.window_type = window_type
        self.overlap_strategy = overlap_strategy
    
    def split(self, X, y=None, groups=None):
        """Generate multi-step walk-forward splits."""
        
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        # Convert fractional sizes
        if isinstance(self.train_size, float):
            train_window_size = int(n_samples * self.train_size)
        else:
            train_window_size = self.train_size
        
        current_pos = train_window_size
        
        while current_pos + self.forecast_horizon <= n_samples:
            # Test window (multi-step forecast)
            test_start = current_pos
            test_end = current_pos + self.forecast_horizon
            
            # Training window
            if self.window_type == 'expanding':
                train_start = 0
                train_end = current_pos
            else:  # sliding
                train_start = current_pos - train_window_size
                train_end = current_pos
            
            # Generate indices
            train_indices = np.arange(max(0, train_start), train_end)
            test_indices = np.arange(test_start, min(test_end, n_samples))
            
            yield train_indices, test_indices
            
            # Move forward based on overlap strategy
            if self.overlap_strategy == 'skip':
                # No overlap - step by forecast horizon
                current_pos += self.forecast_horizon
            else:  # average overlaps
                # Step by step_size allowing overlaps
                current_pos += self.step_size
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        if X is not None:
            n_samples = _num_samples(X)
            
            if isinstance(self.train_size, float):
                train_window_size = int(n_samples * self.train_size)
            else:
                train_window_size = self.train_size
            
            remaining_samples = n_samples - train_window_size
            
            if self.overlap_strategy == 'skip':
                n_splits = remaining_samples // self.forecast_horizon
            else:
                n_splits = max(0, (remaining_samples - self.forecast_horizon) // self.step_size + 1)
            
            return n_splits
        
        return 0

# ============================================
# Grouped Walk-Forward Validation
# ============================================

class GroupedWalkForward(BaseCrossValidator):
    """
    Walk-Forward Validation for grouped/multi-asset data.
    
    Applies walk-forward validation separately to each group
    (e.g., different assets, sectors) while maintaining temporal order.
    
    Parameters:
    -----------
    train_size : int or float
        Size of training window per group.
    test_size : int or float, default=1
        Size of test window per group.
    step_size : int, default=1
        Step size for walk-forward.
    min_group_size : int, default=50
        Minimum samples required per group.
    sync_groups : bool, default=True
        Whether to synchronize walk-forward across groups.
    """
    
    def __init__(self, train_size: Union[int, float],
                 test_size: Union[int, float] = 1,
                 step_size: int = 1,
                 min_group_size: int = 50,
                 sync_groups: bool = True):
        
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.min_group_size = min_group_size
        self.sync_groups = sync_groups
    
    def split(self, X, y=None, groups=None):
        """Generate grouped walk-forward splits."""
        
        if groups is None:
            raise ValueError("groups parameter is required for GroupedWalkForward")
        
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        unique_groups = np.unique(groups)
        
        # Filter groups with sufficient samples
        valid_groups = []
        group_indices = {}
        
        for group in unique_groups:
            group_mask = groups == group
            group_idx = np.where(group_mask)[0]
            
            if len(group_idx) >= self.min_group_size:
                valid_groups.append(group)
                group_indices[group] = group_idx
        
        if len(valid_groups) == 0:
            raise ValueError("No groups have sufficient samples")
        
        # Generate walk-forward splits for each group
        group_splits = {}
        
        for group in valid_groups:
            group_idx = group_indices[group]
            
            # Create walk-forward CV for this group
            wf_cv = WalkForwardSplit(
                train_size=self.train_size,
                test_size=self.test_size,
                step_size=self.step_size
            )
            
            # Generate splits (relative to group indices)
            group_X = np.arange(len(group_idx))  # Dummy X for splitting
            relative_splits = list(wf_cv.split(group_X))
            
            # Convert relative indices to absolute indices
            absolute_splits = []
            for train_rel, test_rel in relative_splits:
                train_abs = group_idx[train_rel]
                test_abs = group_idx[test_rel]
                absolute_splits.append((train_abs, test_abs))
            
            group_splits[group] = absolute_splits
        
        # Combine splits across groups
        if self.sync_groups:
            # Synchronize: each split combines same step from all groups
            max_splits = max(len(splits) for splits in group_splits.values())
            
            for step in range(max_splits):
                combined_train = []
                combined_test = []
                
                for group, splits in group_splits.items():
                    if step < len(splits):
                        train_abs, test_abs = splits[step]
                        combined_train.extend(train_abs)
                        combined_test.extend(test_abs)
                
                if len(combined_train) > 0 and len(combined_test) > 0:
                    yield np.array(combined_train), np.array(combined_test)
        else:
            # Separate: yield splits from each group independently
            for group, splits in group_splits.items():
                for train_abs, test_abs in splits:
                    yield train_abs, test_abs
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        # This is approximate since it depends on the actual data
        if X is not None and groups is not None:
            unique_groups = np.unique(groups)
            
            # Estimate based on first valid group
            for group in unique_groups:
                group_mask = groups == group
                group_size = np.sum(group_mask)
                
                if group_size >= self.min_group_size:
                    dummy_cv = WalkForwardSplit(
                        train_size=self.train_size,
                        test_size=self.test_size,
                        step_size=self.step_size
                    )
                    return dummy_cv.get_n_splits(np.arange(group_size))
        
        return 0

# ============================================
# Walk-Forward Optimization
# ============================================

class WalkForwardOptimizer:
    """
    Walk-Forward Optimization for systematic parameter tuning.
    
    This class combines walk-forward validation with hyperparameter
    optimization, retraining and reoptimizing the model at each step.
    
    Parameters:
    -----------
    estimator : BaseEstimator
        Base model to optimize.
    param_grid : dict or list of dicts
        Parameter grid for optimization.
    cv_split : BaseCrossValidator
        Cross-validator for walk-forward splits.
    optimization_cv : BaseCrossValidator, optional
        Cross-validator for parameter optimization within each window.
    scoring : str or callable
        Scoring function.
    refit : bool, default=True
        Whether to refit using best parameters.
    n_jobs : int, default=None
        Number of parallel jobs.
    """
    
    def __init__(self, estimator: BaseEstimator,
                 param_grid: Union[Dict, List[Dict]],
                 cv_split: BaseCrossValidator,
                 optimization_cv: Optional[BaseCrossValidator] = None,
                 scoring: Optional[Union[str, callable]] = None,
                 refit: bool = True,
                 n_jobs: Optional[int] = None):
        
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv_split = cv_split
        self.optimization_cv = optimization_cv
        self.scoring = scoring
        self.refit = refit
        self.n_jobs = n_jobs
        
        self.optimization_history_ = []
        self.performance_history_ = []
    
    @time_it("walk_forward_optimization")
    def fit_predict(self, X, y):
        """
        Perform walk-forward optimization and return predictions.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
            
        Returns:
        --------
        Dictionary with predictions and optimization history
        """
        
        X, y = indexable(X, y)
        
        predictions = []
        actuals = []
        
        for step, (train_idx, test_idx) in enumerate(self.cv_split.split(X, y)):
            step_start = datetime.now()
            
            # Get training and test data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Optimize hyperparameters on training window
            best_params, best_score = self._optimize_parameters(X_train, y_train)
            
            # Train model with best parameters
            optimized_model = clone(self.estimator)
            optimized_model.set_params(**best_params)
            optimized_model.fit(X_train, y_train)
            
            # Make predictions on test window
            step_predictions = optimized_model.predict(X_test)
            
            # Store results
            predictions.extend(step_predictions)
            actuals.extend(y_test)
            
            # Record optimization history
            step_time = (datetime.now() - step_start).total_seconds()
            
            self.optimization_history_.append({
                'step': step + 1,
                'best_params': best_params,
                'best_score': best_score,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'optimization_time': step_time
            })
            
            # Calculate step performance
            if self.scoring:
                if callable(self.scoring):
                    step_performance = self.scoring(y_test, step_predictions)
                else:
                    scorer = get_scorer(self.scoring)
                    step_performance = scorer._score_func(y_test, step_predictions)
            else:
                step_performance = optimized_model.score(X_test, y_test)
            
            self.performance_history_.append({
                'step': step + 1,
                'performance': step_performance,
                'test_period': [test_idx[0], test_idx[-1]] if len(test_idx) > 0 else []
            })
            
            logger.info(f"Step {step + 1}: params={best_params}, "
                       f"performance={step_performance:.4f}")
        
        return {
            'predictions': np.array(predictions),
            'actuals': np.array(actuals),
            'optimization_history': self.optimization_history_,
            'performance_history': self.performance_history_
        }
    
    def _optimize_parameters(self, X_train, y_train):
        """Optimize hyperparameters on training data."""
        
        from sklearn.model_selection import GridSearchCV
        
        # Use provided CV or default time series split
        if self.optimization_cv is not None:
            cv = self.optimization_cv
        else:
            from sklearn.model_selection import TimeSeriesSplit
            cv = TimeSeriesSplit(n_splits=3)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=cv,
            scoring=self.scoring,
            refit=self.refit,
            n_jobs=self.n_jobs
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_params_, grid_search.best_score_
    
    def get_optimization_summary(self):
        """Get summary of optimization across all steps."""
        
        if not self.optimization_history_:
            return "No optimization history available"
        
        # Parameter stability analysis
        all_params = {}
        for step_info in self.optimization_history_:
            for param, value in step_info['best_params'].items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)
        
        param_stability = {}
        for param, values in all_params.items():
            if isinstance(values[0], (int, float)):
                param_stability[param] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                # Categorical parameters
                unique_values, counts = np.unique(values, return_counts=True)
                most_common = unique_values[np.argmax(counts)]
                param_stability[param] = {
                    'most_common': most_common,
                    'frequency': np.max(counts) / len(values),
                    'unique_values': unique_values.tolist()
                }
        
        # Performance summary
        performances = [step['performance'] for step in self.performance_history_]
        
        summary = {
            'n_steps': len(self.optimization_history_),
            'parameter_stability': param_stability,
            'performance_summary': {
                'mean': np.mean(performances),
                'std': np.std(performances),
                'min': np.min(performances),
                'max': np.max(performances)
            },
            'average_optimization_time': np.mean([
                step['optimization_time'] for step in self.optimization_history_
            ])
        }
        
        return summary

# ============================================
# Utility Functions
# ============================================

@time_it("walk_forward_validation")
def walk_forward_validate(model: BaseEstimator,
                         X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series],
                         train_size: Union[int, float],
                         test_size: Union[int, float] = 1,
                         step_size: int = 1,
                         window_type: str = 'expanding',
                         scoring: Optional[Union[str, callable]] = None,
                         return_predictions: bool = False) -> Dict[str, Any]:
    """
    Perform walk-forward validation.
    
    Parameters:
    -----------
    model : BaseEstimator
        Model to validate
    X : array-like
        Features
    y : array-like
        Target values
    train_size : int or float
        Training window size
    test_size : int or float
        Test window size
    step_size : int
        Step size for walk-forward
    window_type : str
        Type of training window ('expanding' or 'sliding')
    scoring : str or callable, optional
        Scoring function
    return_predictions : bool
        Whether to return predictions
        
    Returns:
    --------
    Dictionary with validation results
    """
    
    # Create walk-forward splitter
    wf_split = WalkForwardSplit(
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        window_type=window_type
    )
    
    results = {
        'method': 'walk_forward',
        'window_type': window_type,
        'n_steps': 0,
        'step_scores': [],
        'step_details': [],
        'train_sizes': [],
        'test_sizes': [],
        'fit_times': []
    }
    
    if return_predictions:
        results['predictions'] = []
        results['actuals'] = []
        results['step_indices'] = []
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    for step, (train_idx, test_idx) in enumerate(wf_split.split(X, y)):
        step_start = datetime.now()
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model_copy = clone(model)
        model_copy.fit(X_train, y_train)
        fit_time = (datetime.now() - step_start).total_seconds()
        
        # Make predictions
        predictions = model_copy.predict(X_test)
        
        # Score model
        if scoring is None:
            score = model_copy.score(X_test, y_test)
        elif callable(scoring):
            score = scoring(y_test, predictions)
        else:
            scorer = get_scorer(scoring)
            score = scorer(model_copy, X_test, y_test)
        
        # Store results
        results['step_scores'].append(score)
        results['train_sizes'].append(len(train_idx))
        results['test_sizes'].append(len(test_idx))
        results['fit_times'].append(fit_time)
        
        results['step_details'].append({
            'step': step + 1,
            'score': score,
            'train_period': [train_idx[0], train_idx[-1]] if len(train_idx) > 0 else [],
            'test_period': [test_idx[0], test_idx[-1]] if len(test_idx) > 0 else [],
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'fit_time': fit_time
        })
        
        if return_predictions:
            results['predictions'].extend(predictions.tolist())
            results['actuals'].extend(y_test.tolist())
            results['step_indices'].append({
                'step': step + 1,
                'test_indices': test_idx.tolist()
            })
        
        logger.debug(f"Step {step + 1}: score={score:.4f}, "
                    f"train_size={len(train_idx)}, test_size={len(test_idx)}")
    
    # Calculate summary statistics
    results['n_steps'] = len(results['step_scores'])
    
    if results['n_steps'] > 0:
        results['mean_score'] = np.mean(results['step_scores'])
        results['std_score'] = np.std(results['step_scores'])
        results['min_score'] = np.min(results['step_scores'])
        results['max_score'] = np.max(results['step_scores'])
        results['median_score'] = np.median(results['step_scores'])
        
        # Average training window size
        results['avg_train_size'] = np.mean(results['train_sizes'])
        results['avg_test_size'] = np.mean(results['test_sizes'])
        results['avg_fit_time'] = np.mean(results['fit_times'])
    
    return results

def compare_window_types(model: BaseEstimator,
                        X: Union[np.ndarray, pd.DataFrame],
                        y: Union[np.ndarray, pd.Series],
                        train_size: Union[int, float],
                        scoring: Optional[Union[str, callable]] = None) -> pd.DataFrame:
    """
    Compare expanding vs sliding window walk-forward validation.
    
    Parameters:
    -----------
    model : BaseEstimator
        Model to compare
    X : array-like
        Features
    y : array-like
        Target values
    train_size : int or float
        Training window size
    scoring : str or callable, optional
        Scoring function
        
    Returns:
    --------
    DataFrame comparing the two approaches
    """
    
    results = []
    
    for window_type in ['expanding', 'sliding']:
        try:
            wf_results = walk_forward_validate(
                model=model, X=X, y=y,
                train_size=train_size,
                window_type=window_type,
                scoring=scoring
            )
            
            results.append({
                'window_type': window_type,
                'n_steps': wf_results['n_steps'],
                'mean_score': wf_results.get('mean_score', np.nan),
                'std_score': wf_results.get('std_score', np.nan),
                'min_score': wf_results.get('min_score', np.nan),
                'max_score': wf_results.get('max_score', np.nan),
                'avg_train_size': wf_results.get('avg_train_size', np.nan),
                'avg_fit_time': wf_results.get('avg_fit_time', np.nan)
            })
            
        except Exception as e:
            logger.warning(f"Error with {window_type} window: {e}")
            results.append({
                'window_type': window_type,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Walk-Forward Validation Methods")
    
    # Generate sample financial time series data
    np.random.seed(42)
    n_samples = 2000
    n_features = 4
    
    # Create synthetic time series with trend and seasonality
    time_index = np.arange(n_samples)
    trend = 0.0001 * time_index
    seasonality = 0.01 * np.sin(2 * np.pi * time_index / 252)  # Annual cycle
    noise = np.random.normal(0, 0.02, n_samples)
    
    # Base return series
    returns = trend + seasonality + noise
    
    # Add autocorrelation
    for i in range(1, n_samples):
        returns[i] += 0.1 * returns[i-1]
    
    # Create features (lagged returns, moving averages, volatility)
    features = []
    for lag in range(1, n_features + 1):
        features.append(np.roll(returns, lag))
    
    X = np.column_stack(features)[n_features:]
    y = returns[n_features:]
    
    print(f"Generated time series data: X={X.shape}, y={len(y)}")
    
    # Test basic WalkForwardSplit
    print("\n1. Testing WalkForwardSplit")
    
    wf_split = WalkForwardSplit(
        train_size=500,
        test_size=50,
        step_size=25,
        window_type='expanding'
    )
    
    n_splits = wf_split.get_n_splits(X)
    print(f"Number of walk-forward steps: {n_splits}")
    
    for i, (train_idx, test_idx) in enumerate(wf_split.split(X)):
        if i < 5:  # Show first 5 steps
            print(f"Step {i+1}: Train=[{train_idx[0]}:{train_idx[-1]}] ({len(train_idx)}), "
                  f"Test=[{test_idx[0]}:{test_idx[-1]}] ({len(test_idx)})")
    
    # Test AnchoredWalkForward
    print("\n2. Testing AnchoredWalkForward")
    
    anchored_wf = AnchoredWalkForward(
        initial_train_size=500,
        test_size=50,
        step_size=50,
        max_train_size=1000
    )
    
    for i, (train_idx, test_idx) in enumerate(anchored_wf.split(X)):
        if i < 3:  # Show first 3 steps
            print(f"Step {i+1}: Train=[{train_idx[0]}:{train_idx[-1]}] ({len(train_idx)}), "
                  f"Test=[{test_idx[0]}:{test_idx[-1]}] ({len(test_idx)})")
    
    # Test MultiStepWalkForward
    print("\n3. Testing MultiStepWalkForward")
    
    multi_step_wf = MultiStepWalkForward(
        train_size=500,
        forecast_horizon=10,
        step_size=5,
        window_type='sliding'
    )
    
    for i, (train_idx, test_idx) in enumerate(multi_step_wf.split(X)):
        if i < 3:  # Show first 3 steps
            print(f"Step {i+1}: Train=[{train_idx[0]}:{train_idx[-1]}] ({len(train_idx)}), "
                  f"Test=[{test_idx[0]}:{test_idx[-1]}] ({len(test_idx)})")
    
    # Test comprehensive walk-forward validation
    print("\n4. Testing Comprehensive Walk-Forward Validation")
    
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    
    model = Ridge(alpha=0.1)
    
    def neg_mse_scorer(y_true, y_pred):
        return -mean_squared_error(y_true, y_pred)
    
    # Run walk-forward validation
    wf_results = walk_forward_validate(
        model=model,
        X=X, y=y,
        train_size=0.3,  # 30% for initial training
        test_size=50,    # Test on 50 samples
        step_size=25,    # Step forward 25 samples
        window_type='expanding',
        scoring=neg_mse_scorer,
        return_predictions=True
    )
    
    print("Walk-Forward Validation Results:")
    print(f"  Number of steps: {wf_results['n_steps']}")
    print(f"  Mean score: {wf_results['mean_score']:.6f} ± {wf_results['std_score']:.6f}")
    print(f"  Score range: [{wf_results['min_score']:.6f}, {wf_results['max_score']:.6f}]")
    print(f"  Average training size: {wf_results['avg_train_size']:.0f}")
    print(f"  Average fit time: {wf_results['avg_fit_time']:.4f}s")
    
    # Test GroupedWalkForward
    print("\n5. Testing GroupedWalkForward")
    
    # Create artificial groups (e.g., different assets)
    n_groups = 5
    groups = np.repeat(np.arange(n_groups), len(X) // n_groups + 1)[:len(X)]
    
    grouped_wf = GroupedWalkForward(
        train_size=100,
        test_size=20,
        step_size=10,
        min_group_size=200,
        sync_groups=True
    )
    
    try:
        for i, (train_idx, test_idx) in enumerate(grouped_wf.split(X, y, groups)):
            if i < 3:  # Show first 3 steps
                train_groups = len(np.unique(groups[train_idx]))
                test_groups = len(np.unique(groups[test_idx]))
                print(f"Step {i+1}: Train groups={train_groups}, Test groups={test_groups}, "
                      f"Train size={len(train_idx)}, Test size={len(test_idx)}")
    except ValueError as e:
        print(f"Grouped walk-forward failed: {e}")
    
    # Test Walk-Forward Optimization
    print("\n6. Testing Walk-Forward Optimization")
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    
    # Parameter grid for optimization
    param_grid = {
        'n_estimators': [10, 20, 50],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }
    
    # Create walk-forward optimizer
    wf_optimizer = WalkForwardOptimizer(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv_split=WalkForwardSplit(train_size=500, test_size=100, step_size=50),
        optimization_cv=TimeSeriesSplit(n_splits=3),
        scoring=neg_mse_scorer
    )
    
    # Run optimization (on smaller dataset for speed)
    small_X = X[:1000]
    small_y = y[:1000]
    
    opt_results = wf_optimizer.fit_predict(small_X, small_y)
    
    print("Walk-Forward Optimization Results:")
    print(f"  Total predictions: {len(opt_results['predictions'])}")
    print(f"  Optimization steps: {len(opt_results['optimization_history'])}")
    
    # Show optimization summary
    opt_summary = wf_optimizer.get_optimization_summary()
    print(f"  Average performance: {opt_summary['performance_summary']['mean']:.6f}")
    print(f"  Parameter stability example:")
    
    if 'n_estimators' in opt_summary['parameter_stability']:
        n_est_stats = opt_summary['parameter_stability']['n_estimators']
        print(f"    n_estimators - mean: {n_est_stats['mean']:.1f}, std: {n_est_stats['std']:.1f}")
    
    # Compare window types
    print("\n7. Comparing Window Types")
    
    comparison_df = compare_window_types(
        model=Ridge(alpha=0.1),
        X=X[:1000], y=y[:1000],  # Smaller dataset for speed
        train_size=200,
        scoring=neg_mse_scorer
    )
    
    print("Window Type Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Performance analysis across steps
    print("\n8. Step-by-Step Performance Analysis")
    
    print("Individual Step Performance (First 10 steps):")
    for i, step_detail in enumerate(wf_results['step_details'][:10]):
        print(f"  Step {step_detail['step']}: Score={step_detail['score']:.6f}, "
              f"Train=[{step_detail['train_period'][0]}:{step_detail['train_period'][1]}], "
              f"Test=[{step_detail['test_period'][0]}:{step_detail['test_period'][1]}]")
    
    # Overall statistics
    print(f"\n9. Summary Statistics:")
    print(f"  Total walk-forward steps: {wf_results['n_steps']}")
    print(f"  Training window growth: expanding")
    print(f"  Final training size: {wf_results['train_sizes'][-1]}")
    print(f"  Consistent test size: {wf_results['test_sizes'][0]}")
    print(f"  Performance stability (CV): {wf_results['std_score'] / abs(wf_results['mean_score']):.3f}")
    
    print("\nWalk-forward validation testing completed successfully!")
