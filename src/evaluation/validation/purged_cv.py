# ============================================
# StockPredictionPro - src/evaluation/validation/purged_cv.py
# Advanced purged cross-validation for financial machine learning with embargo and gap controls
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
import warnings
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from itertools import combinations

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('evaluation.validation.purged_cv')

# ============================================
# Basic Purged Cross-Validation
# ============================================

class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold cross-validator for financial time series.
    
    This class implements the purging and embargoing methodology from
    "Advances in Financial Machine Learning" by Marcos Lopez de Prado.
    
    Purging removes training samples whose labels overlapped in time with
    the test labels, preventing information leakage.
    
    Embargoing removes training samples that immediately follow the test
    period to account for the non-IID nature of financial data.
    
    Parameters:
    -----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    times : array-like, optional
        Times when the observation was made (prediction times).
        If None, uses integer indices.
    eval_times : array-like, optional  
        Times when the label was evaluated (evaluation times).
        If None, assumes eval_times = times.
    purge : pd.Timedelta or int, default=0
        Duration to purge before test set.
    embargo : pd.Timedelta or int, default=0
        Duration to embargo after test set.
    """
    
    def __init__(self, n_splits: int = 3, 
                 times: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
                 eval_times: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
                 purge: Union[pd.Timedelta, int] = 0,
                 embargo: Union[pd.Timedelta, int] = 0):
        
        self.n_splits = n_splits
        self.times = times
        self.eval_times = eval_times
        self.purge = purge
        self.embargo = embargo
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set with purging."""
        
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        if self.n_splits > n_samples:
            raise ValueError(f"Cannot have number of splits n_splits={self.n_splits} "
                           f"greater than the number of samples: n_samples={n_samples}.")
        
        # Set up times
        if self.times is not None:
            times = pd.Series(self.times, index=np.arange(n_samples))
        else:
            times = pd.Series(np.arange(n_samples), index=np.arange(n_samples))
        
        if self.eval_times is not None:
            eval_times = pd.Series(self.eval_times, index=np.arange(n_samples))
        else:
            eval_times = times.copy()
        
        # Calculate fold boundaries
        test_fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Calculate test indices
            test_start_idx = i * test_fold_size
            test_end_idx = test_start_idx + test_fold_size
            
            if test_end_idx > n_samples:
                test_end_idx = n_samples
            
            if test_start_idx >= test_end_idx:
                break
            
            test_indices = np.arange(test_start_idx, test_end_idx)
            
            # Get train indices (all except test)
            train_indices = np.concatenate([
                np.arange(0, test_start_idx),
                np.arange(test_end_idx, n_samples)
            ])
            
            # Apply purging and embargoing
            if len(test_indices) > 0:
                train_indices = self._purge_embargo(
                    train_indices, test_indices, times, eval_times
                )
            
            # Skip if no training data left
            if len(train_indices) == 0:
                continue
            
            yield train_indices, test_indices
    
    def _purge_embargo(self, train_indices, test_indices, times, eval_times):
        """Apply purging and embargoing to training indices."""
        
        if len(test_indices) == 0:
            return train_indices
        
        # Get test period boundaries
        test_times = times.iloc[test_indices]
        test_eval_times = eval_times.iloc[test_indices]
        
        test_start_time = test_times.min()
        test_end_time = test_times.max()
        test_eval_start = test_eval_times.min()
        test_eval_end = test_eval_times.max()
        
        # Apply purging - remove training samples that overlap with test evaluation times
        if self.purge > 0:
            purge_mask = self._get_purge_mask(
                train_indices, times, eval_times, 
                test_eval_start, test_eval_end
            )
            train_indices = train_indices[purge_mask]
        
        # Apply embargoing - remove training samples immediately after test period
        if self.embargo > 0:
            embargo_mask = self._get_embargo_mask(
                train_indices, times, test_end_time
            )
            train_indices = train_indices[embargo_mask]
        
        return train_indices
    
    def _get_purge_mask(self, train_indices, times, eval_times, test_eval_start, test_eval_end):
        """Get mask for purging overlapping samples."""
        
        train_times = times.iloc[train_indices]
        train_eval_times = eval_times.iloc[train_indices]
        
        # Check for overlaps between train evaluation times and test evaluation times
        if isinstance(self.purge, pd.Timedelta):
            # Time-based purging
            overlap_mask = ~(
                (train_eval_times + self.purge < test_eval_start) |
                (train_times > test_eval_end + self.purge)
            )
        else:
            # Index-based purging
            purge_value = self.purge
            overlap_mask = ~(
                (train_eval_times < test_eval_start - purge_value) |
                (train_times > test_eval_end + purge_value)
            )
        
        return ~overlap_mask
    
    def _get_embargo_mask(self, train_indices, times, test_end_time):
        """Get mask for embargoing samples after test period."""
        
        train_times = times.iloc[train_indices]
        
        if isinstance(self.embargo, pd.Timedelta):
            # Time-based embargo
            embargo_mask = train_times <= test_end_time + self.embargo
        else:
            # Index-based embargo
            embargo_mask = train_indices <= train_indices.max() + self.embargo
        
        return ~embargo_mask
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

# ============================================
# Combinatorial Purged Cross-Validation
# ============================================

class CombinatorialPurgedKFold(BaseCrossValidator):
    """
    Combinatorial Purged K-Fold cross-validator.
    
    Generates multiple backtest paths by selecting different combinations
    of groups for testing, while applying purging and embargoing.
    
    Based on the methodology from "Advances in Financial Machine Learning"
    by Marcos Lopez de Prado.
    
    Parameters:
    -----------
    n_splits : int, default=6
        Number of groups to divide the data into.
    n_test_groups : int, default=2
        Number of groups to use for testing in each split.
    times : array-like, optional
        Times when the observation was made.
    eval_times : array-like, optional
        Times when the label was evaluated.
    purge : pd.Timedelta or int, default=0
        Duration to purge before test sets.
    embargo : pd.Timedelta or int, default=0
        Duration to embargo after test sets.
    """
    
    def __init__(self, n_splits: int = 6, n_test_groups: int = 2,
                 times: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
                 eval_times: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
                 purge: Union[pd.Timedelta, int] = 0,
                 embargo: Union[pd.Timedelta, int] = 0):
        
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.times = times
        self.eval_times = eval_times
        self.purge = purge
        self.embargo = embargo
    
    def split(self, X, y=None, groups=None):
        """Generate combinatorial splits with purging and embargoing."""
        
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        # Set up times
        if self.times is not None:
            times = pd.Series(self.times, index=np.arange(n_samples))
        else:
            times = pd.Series(np.arange(n_samples), index=np.arange(n_samples))
        
        if self.eval_times is not None:
            eval_times = pd.Series(self.eval_times, index=np.arange(n_samples))
        else:
            eval_times = times.copy()
        
        # Generate backtest paths
        backtest_paths = self._generate_backtest_paths()
        
        # Divide data into groups
        group_boundaries = self._get_group_boundaries(n_samples)
        
        for path in backtest_paths:
            test_groups = path
            train_groups = [i for i in range(self.n_splits) if i not in test_groups]
            
            # Get test indices
            test_indices = np.concatenate([
                np.arange(group_boundaries[group], group_boundaries[group + 1])
                for group in test_groups
                if group_boundaries[group] < group_boundaries[group + 1]
            ])
            
            # Get train indices
            train_indices = np.concatenate([
                np.arange(group_boundaries[group], group_boundaries[group + 1])
                for group in train_groups
                if group_boundaries[group] < group_boundaries[group + 1]
            ])
            
            # Apply purging and embargoing
            if len(test_indices) > 0 and len(train_indices) > 0:
                train_indices = self._purge_embargo_combinatorial(
                    train_indices, test_indices, times, eval_times, 
                    test_groups, group_boundaries
                )
            
            # Skip if no training data left
            if len(train_indices) == 0:
                continue
            
            yield train_indices, test_indices
    
    def _generate_backtest_paths(self):
        """Generate all possible combinations of test groups."""
        return list(combinations(range(self.n_splits), self.n_test_groups))
    
    def _get_group_boundaries(self, n_samples):
        """Calculate group boundaries."""
        group_size = n_samples // self.n_splits
        boundaries = []
        
        for i in range(self.n_splits + 1):
            if i == self.n_splits:
                boundaries.append(n_samples)
            else:
                boundaries.append(i * group_size)
        
        return boundaries
    
    def _purge_embargo_combinatorial(self, train_indices, test_indices, 
                                   times, eval_times, test_groups, group_boundaries):
        """Apply purging and embargoing for combinatorial splits."""
        
        # Sort test groups to identify continuous blocks
        test_groups_sorted = sorted(test_groups)
        
        # Apply purging and embargoing for each test block
        for i, test_group in enumerate(test_groups_sorted):
            test_block_start = group_boundaries[test_group]
            test_block_end = group_boundaries[test_group + 1]
            
            test_block_indices = test_indices[
                (test_indices >= test_block_start) & 
                (test_indices < test_block_end)
            ]
            
            if len(test_block_indices) > 0:
                # Apply purging
                if self.purge > 0:
                    train_indices = self._apply_purge_combinatorial(
                        train_indices, test_block_indices, times, eval_times
                    )
                
                # Apply embargoing
                if self.embargo > 0:
                    train_indices = self._apply_embargo_combinatorial(
                        train_indices, test_block_indices, times
                    )
        
        return train_indices
    
    def _apply_purge_combinatorial(self, train_indices, test_indices, times, eval_times):
        """Apply purging for combinatorial method."""
        
        if len(test_indices) == 0:
            return train_indices
        
        test_eval_start = eval_times.iloc[test_indices].min()
        test_eval_end = eval_times.iloc[test_indices].max()
        
        train_times = times.iloc[train_indices]
        train_eval_times = eval_times.iloc[train_indices]
        
        if isinstance(self.purge, pd.Timedelta):
            overlap_mask = (
                (train_eval_times + self.purge >= test_eval_start) &
                (train_times <= test_eval_end + self.purge)
            )
        else:
            overlap_mask = (
                (train_eval_times >= test_eval_start - self.purge) &
                (train_times <= test_eval_end + self.purge)
            )
        
        return train_indices[~overlap_mask]
    
    def _apply_embargo_combinatorial(self, train_indices, test_indices, times):
        """Apply embargoing for combinatorial method."""
        
        if len(test_indices) == 0:
            return train_indices
        
        test_end_time = times.iloc[test_indices].max()
        train_times = times.iloc[train_indices]
        
        if isinstance(self.embargo, pd.Timedelta):
            embargo_mask = train_times <= test_end_time + self.embargo
        else:
            embargo_mask = train_indices <= test_indices.max() + self.embargo
        
        return train_indices[~embargo_mask]
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        from math import comb
        return comb(self.n_splits, self.n_test_groups)

# ============================================
# Group-Aware Purged Cross-Validation
# ============================================

class GroupPurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Group-aware purged time series split.
    
    Useful when you have groups (e.g., different assets, strategies) that
    should not be split across train/test sets.
    
    Parameters:
    -----------
    n_splits : int, default=5
        Number of splits.
    purge : pd.Timedelta or int, default=0
        Duration to purge between splits.
    embargo : pd.Timedelta or int, default=0
        Duration to embargo after test sets.
    max_train_group_size : int, optional
        Maximum number of groups in training set.
    """
    
    def __init__(self, n_splits: int = 5,
                 purge: Union[pd.Timedelta, int] = 0,
                 embargo: Union[pd.Timedelta, int] = 0,
                 max_train_group_size: Optional[int] = None):
        
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.max_train_group_size = max_train_group_size
    
    def split(self, X, y=None, groups=None):
        """Generate group-aware splits with purging."""
        
        if groups is None:
            raise ValueError("groups parameter is required for GroupPurgedTimeSeriesSplit")
        
        X, y, groups = indexable(X, y, groups)
        
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if self.n_splits > n_groups:
            raise ValueError(f"Cannot have number of splits n_splits={self.n_splits} "
                           f"greater than the number of groups: n_groups={n_groups}.")
        
        # Calculate group fold size
        test_fold_size = n_groups // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Select test groups
            test_group_start = (i + 1) * test_fold_size
            test_group_end = test_group_start + test_fold_size
            
            if test_group_end > n_groups:
                test_group_end = n_groups
            
            test_groups = unique_groups[test_group_start:test_group_end]
            
            # Select train groups
            train_groups = unique_groups[
                ~np.isin(unique_groups, test_groups)
            ]
            
            # Apply max train group size constraint
            if self.max_train_group_size is not None:
                if len(train_groups) > self.max_train_group_size:
                    # Take most recent groups
                    train_groups = train_groups[-self.max_train_group_size:]
            
            # Get indices
            test_indices = np.where(np.isin(groups, test_groups))[0]
            train_indices = np.where(np.isin(groups, train_groups))[0]
            
            # Apply purging and embargoing based on group boundaries
            if self.purge > 0 or self.embargo > 0:
                train_indices = self._apply_group_purge_embargo(
                    train_indices, test_indices, groups
                )
            
            if len(train_indices) == 0:
                continue
            
            yield train_indices, test_indices
    
    def _apply_group_purge_embargo(self, train_indices, test_indices, groups):
        """Apply purging and embargoing at group level."""
        
        if len(test_indices) == 0:
            return train_indices
        
        test_groups = np.unique(groups[test_indices])
        
        # For simplicity, remove train groups that are "close" to test groups
        # This is a simplified implementation - in practice, you'd use actual time information
        
        if self.purge > 0:
            # Remove training groups that are within purge distance of test groups
            purge_mask = np.ones(len(train_indices), dtype=bool)
            train_groups = groups[train_indices]
            
            for test_group in test_groups:
                if isinstance(self.purge, int):
                    # Assume groups are ordered and use integer distance
                    close_groups = np.where(
                        np.abs(groups - test_group) <= self.purge
                    )[0]
                    close_mask = np.isin(train_indices, close_groups)
                    purge_mask &= ~close_mask
            
            train_indices = train_indices[purge_mask]
        
        return train_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        return self.n_splits

# ============================================
# Advanced Purged Cross-Validation with Custom Logic
# ============================================

class AdvancedPurgedCV:
    """
    Advanced purged cross-validation with custom purging and embargoing logic.
    
    Provides maximum flexibility for complex financial validation scenarios.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 test_size: float = 0.2,
                 purge_func: Optional[callable] = None,
                 embargo_func: Optional[callable] = None,
                 validation_func: Optional[callable] = None):
        """
        Initialize advanced purged CV.
        
        Parameters:
        -----------
        n_splits : int
            Number of splits
        test_size : float
            Fraction of data to use for testing
        purge_func : callable, optional
            Custom function to determine which samples to purge
        embargo_func : callable, optional
            Custom function to determine which samples to embargo
        validation_func : callable, optional
            Custom validation function for splits
        """
        
        self.n_splits = n_splits
        self.test_size = test_size
        self.purge_func = purge_func or self._default_purge_func
        self.embargo_func = embargo_func or self._default_embargo_func
        self.validation_func = validation_func or self._default_validation_func
    
    def split(self, X, y=None, times=None, eval_times=None, **kwargs):
        """Generate advanced purged splits."""
        
        n_samples = len(X)
        test_size = int(n_samples * self.test_size)
        
        # Generate base splits
        split_size = (n_samples - test_size) // self.n_splits
        
        for i in range(self.n_splits):
            # Calculate test boundaries
            test_start = i * split_size + split_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                test_end = n_samples
            
            if test_start >= test_end:
                break
            
            test_indices = np.arange(test_start, test_end)
            train_indices = np.concatenate([
                np.arange(0, test_start),
                np.arange(test_end, n_samples)
            ])
            
            # Apply custom purging
            if self.purge_func:
                train_indices = self.purge_func(
                    train_indices, test_indices, times, eval_times, **kwargs
                )
            
            # Apply custom embargoing
            if self.embargo_func:
                train_indices = self.embargo_func(
                    train_indices, test_indices, times, eval_times, **kwargs
                )
            
            # Apply custom validation
            if self.validation_func:
                is_valid = self.validation_func(
                    train_indices, test_indices, X, y, **kwargs
                )
                if not is_valid:
                    continue
            
            yield train_indices, test_indices
    
    def _default_purge_func(self, train_indices, test_indices, times, eval_times, **kwargs):
        """Default purging function."""
        return train_indices
    
    def _default_embargo_func(self, train_indices, test_indices, times, eval_times, **kwargs):
        """Default embargoing function."""
        return train_indices
    
    def _default_validation_func(self, train_indices, test_indices, X, y, **kwargs):
        """Default validation function."""
        return len(train_indices) > 0 and len(test_indices) > 0

# ============================================
# Utility Functions
# ============================================

@time_it("purged_cross_validation")
def purged_cross_validate(model: BaseEstimator,
                         X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series],
                         times: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
                         eval_times: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
                         cv: Optional[BaseCrossValidator] = None,
                         scoring: Optional[Union[str, callable]] = None,
                         purge: Union[pd.Timedelta, int] = 0,
                         embargo: Union[pd.Timedelta, int] = 0,
                         n_splits: int = 3) -> Dict[str, Any]:
    """
    Perform purged cross-validation.
    
    Parameters:
    -----------
    model : BaseEstimator
        Model to validate
    X : array-like
        Features
    y : array-like
        Target values
    times : array-like, optional
        Prediction times
    eval_times : array-like, optional
        Evaluation times
    cv : BaseCrossValidator, optional
        Cross-validator to use
    scoring : str or callable, optional
        Scoring function
    purge : pd.Timedelta or int
        Purge duration
    embargo : pd.Timedelta or int
        Embargo duration
    n_splits : int
        Number of splits
        
    Returns:
    --------
    Dictionary with validation results
    """
    
    if cv is None:
        cv = PurgedKFold(
            n_splits=n_splits,
            times=times,
            eval_times=eval_times,
            purge=purge,
            embargo=embargo
        )
    
    results = {
        'scores': [],
        'train_sizes': [],
        'test_sizes': [],
        'purge_stats': [],
        'fit_times': []
    }
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_start = datetime.now()
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        fit_time = (datetime.now() - fold_start).total_seconds()
        
        # Score model
        if scoring is None:
            score = model.score(X_test, y_test)
        elif callable(scoring):
            predictions = model.predict(X_test)
            score = scoring(y_test, predictions)
        else:
            from sklearn.metrics import get_scorer
            scorer = get_scorer(scoring)
            score = scorer(model, X_test, y_test)
        
        # Store results
        results['scores'].append(score)
        results['train_sizes'].append(len(train_idx))
        results['test_sizes'].append(len(test_idx))
        results['fit_times'].append(fit_time)
        
        # Calculate purge statistics
        original_train_size = len(X) - len(test_idx)
        purged_samples = original_train_size - len(train_idx)
        purge_ratio = purged_samples / original_train_size if original_train_size > 0 else 0
        
        results['purge_stats'].append({
            'fold': fold + 1,
            'purged_samples': purged_samples,
            'purge_ratio': purge_ratio
        })
        
        logger.debug(f"Fold {fold + 1}: score={score:.4f}, train_size={len(train_idx)}, "
                    f"test_size={len(test_idx)}, purged={purged_samples}")
    
    # Summary statistics
    results['mean_score'] = np.mean(results['scores'])
    results['std_score'] = np.std(results['scores'])
    results['mean_purge_ratio'] = np.mean([s['purge_ratio'] for s in results['purge_stats']])
    
    return results

def visualize_purged_splits(cv: BaseCrossValidator, 
                          X: Union[np.ndarray, pd.DataFrame],
                          times: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """
    Create a visualization dataframe for purged splits.
    
    Parameters:
    -----------
    cv : BaseCrossValidator
        Cross-validator
    X : array-like
        Data to split
    times : pd.DatetimeIndex, optional
        Time index for visualization
        
    Returns:
    --------
    DataFrame with split visualization data
    """
    
    n_samples = len(X)
    
    if times is None:
        times = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create visualization dataframe
    viz_data = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        # Mark all samples
        for i in range(n_samples):
            if i in train_idx:
                split_type = 'train'
            elif i in test_idx:
                split_type = 'test'
            else:
                split_type = 'purged'
            
            viz_data.append({
                'fold': fold + 1,
                'index': i,
                'time': times[i],
                'split_type': split_type
            })
    
    return pd.DataFrame(viz_data)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Purged Cross-Validation Methods")
    
    # Generate sample financial time series data
    np.random.seed(42)
    n_samples = 1000
    
    # Create time index
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate autocorrelated returns
    returns = np.zeros(n_samples)
    returns[0] = np.random.randn() * 0.01
    
    for i in range(1, n_samples):
        returns[i] = 0.05 * returns[i-1] + np.random.randn() * 0.01
    
    # Create features (lagged returns and technical indicators)
    X = np.column_stack([
        np.roll(returns, 1),  # t-1 return
        np.roll(returns, 2),  # t-2 return
        pd.Series(returns).rolling(10).mean().fillna(0),  # 10-day MA
        pd.Series(returns).rolling(20).std().fillna(0)    # 20-day vol
    ])[2:]  # Remove first 2 rows due to lags
    
    # Create forward-looking targets (next 5 days return)
    y = pd.Series(returns).rolling(5, min_periods=1).sum().shift(-5).fillna(0).values[2:]
    
    times = dates[2:]  # Align with X
    eval_times = dates[2:] + pd.Timedelta(days=5)  # Evaluation 5 days later
    
    print(f"Generated financial data: X={X.shape}, y={len(y)}")
    
    # Test basic PurgedKFold
    print("\n1. Testing PurgedKFold")
    
    purged_cv = PurgedKFold(
        n_splits=5,
        times=times,
        eval_times=eval_times,
        purge=pd.Timedelta(days=3),
        embargo=pd.Timedelta(days=2)
    )
    
    split_info = []
    for fold, (train_idx, test_idx) in enumerate(purged_cv.split(X, y)):
        purged_samples = len(X) - len(test_idx) - len(train_idx)
        print(f"Fold {fold + 1}: Train={len(train_idx)}, Test={len(test_idx)}, "
              f"Purged={purged_samples}")
        split_info.append((train_idx, test_idx))
    
    # Test CombinatorialPurgedKFold
    print("\n2. Testing CombinatorialPurgedKFold")
    
    comb_cv = CombinatorialPurgedKFold(
        n_splits=6,
        n_test_groups=2,
        times=times,
        eval_times=eval_times,
        purge=pd.Timedelta(days=2),
        embargo=pd.Timedelta(days=1)
    )
    
    n_combinations = comb_cv.get_n_splits()
    print(f"Number of combinations: {n_combinations}")
    
    for fold, (train_idx, test_idx) in enumerate(comb_cv.split(X, y)):
        purged_samples = len(X) - len(test_idx) - len(train_idx)
        print(f"Path {fold + 1}: Train={len(train_idx)}, Test={len(test_idx)}, "
              f"Purged={purged_samples}")
        if fold >= 4:  # Show only first 5 paths
            break
    
    # Test GroupPurgedTimeSeriesSplit
    print("\n3. Testing GroupPurgedTimeSeriesSplit")
    
    # Create artificial groups (e.g., different assets)
    groups = np.repeat(np.arange(20), len(X) // 20 + 1)[:len(X)]
    
    group_cv = GroupPurgedTimeSeriesSplit(
        n_splits=5,
        purge=2,  # Integer-based purge for groups
        max_train_group_size=15
    )
    
    for fold, (train_idx, test_idx) in enumerate(group_cv.split(X, y, groups)):
        train_groups = len(np.unique(groups[train_idx]))
        test_groups = len(np.unique(groups[test_idx]))
        print(f"Fold {fold + 1}: Train groups={train_groups}, Test groups={test_groups}")
    
    # Test comprehensive purged cross-validation
    print("\n4. Testing Comprehensive Purged Cross-Validation")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    model = LinearRegression()
    
    def neg_mse_scorer(y_true, y_pred):
        return -mean_squared_error(y_true, y_pred)
    
    # Run purged cross-validation
    purged_results = purged_cross_validate(
        model=model,
        X=X, y=y,
        times=times,
        eval_times=eval_times,
        purge=pd.Timedelta(days=3),
        embargo=pd.Timedelta(days=2),
        scoring=neg_mse_scorer,
        n_splits=5
    )
    
    print("Purged Cross-Validation Results:")
    print(f"  Mean Score: {purged_results['mean_score']:.6f} ± {purged_results['std_score']:.6f}")
    print(f"  Mean Purge Ratio: {purged_results['mean_purge_ratio']:.1%}")
    
    # Show purge statistics
    print("\nPurge Statistics by Fold:")
    for stat in purged_results['purge_stats']:
        print(f"  Fold {stat['fold']}: {stat['purged_samples']} samples purged "
              f"({stat['purge_ratio']:.1%})")
    
    # Compare with standard cross-validation
    print("\n5. Comparison with Standard CV")
    
    from sklearn.model_selection import TimeSeriesSplit
    
    standard_cv = TimeSeriesSplit(n_splits=5)
    standard_scores = []
    
    for train_idx, test_idx in standard_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = neg_mse_scorer(y_test, pred)
        standard_scores.append(score)
    
    print("Standard vs Purged CV Comparison:")
    print(f"  Standard CV Score: {np.mean(standard_scores):.6f} ± {np.std(standard_scores):.6f}")
    print(f"  Purged CV Score:   {purged_results['mean_score']:.6f} ± {purged_results['std_score']:.6f}")
    print(f"  Difference: {np.mean(standard_scores) - purged_results['mean_score']:.6f}")
    
    # Create visualization data
    print("\n6. Creating Visualization Data")
    
    viz_df = visualize_purged_splits(purged_cv, X, times[:len(X)])
    
    print("Split Visualization Sample:")
    print(viz_df.head(20))
    
    # Summary of purging effectiveness
    print(f"\nPurging Summary:")
    print(f"  Total samples per fold: {len(X)}")
    
    for fold in viz_df['fold'].unique():
        fold_data = viz_df[viz_df['fold'] == fold]
        train_count = len(fold_data[fold_data['split_type'] == 'train'])
        test_count = len(fold_data[fold_data['split_type'] == 'test'])
        purged_count = len(fold_data[fold_data['split_type'] == 'purged'])
        
        print(f"  Fold {fold}: Train={train_count}, Test={test_count}, "
              f"Purged={purged_count} ({purged_count/len(X):.1%})")
    
    print("\nPurged cross-validation testing completed successfully!")
