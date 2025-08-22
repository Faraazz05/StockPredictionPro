# ============================================
# StockPredictionPro - src/evaluation/validation/time_series.py
# Advanced time series validation methods for financial machine learning
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

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('evaluation.validation.time_series')

# ============================================
# Basic Time Series Cross-Validation
# ============================================

class TimeSeriesSplit(BaseCrossValidator):
    """
    Time Series cross-validator that respects temporal ordering.
    Similar to sklearn's TimeSeriesSplit but with more financial-focused features.
    
    Parameters:
    -----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, optional
        Maximum size for a single training set.
    test_size : int, optional
        Used to limit the size of test sets.
    gap : int, default=0
        Number of samples to exclude from the end of each train set
        before the test set.
    """
    
    def __init__(self, n_splits: int = 5, max_train_size: Optional[int] = None,
                 test_size: Optional[int] = None, gap: int = 0):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        if self.n_splits > n_samples:
            raise ValueError(f"Cannot have number of splits n_splits={self.n_splits} "
                           f"greater than the number of samples: n_samples={n_samples}.")
        
        # Calculate test size
        test_size = self.test_size
        if test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Calculate test start and end
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                test_end = n_samples
            
            # Calculate train start and end
            train_end = test_start - self.gap
            
            if self.max_train_size is None:
                train_start = 0
            else:
                train_start = max(0, train_end - self.max_train_size)
            
            # Get indices
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            # Skip if empty
            if len(test_indices) == 0:
                break
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits

# ============================================
# Purged Time Series Cross-Validation
# ============================================

class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Purged Time Series cross-validator for financial data.
    
    Ensures no information leakage by purging observations that could
    contain forward-looking information. Essential for financial time series
    where information can leak across time due to autocorrelation.
    
    Parameters:
    -----------
    n_splits : int, default=5
        Number of splits.
    purge : int, default=0
        Number of observations to exclude between train and test sets
        to prevent information leakage.
    max_train_size : int, optional
        Maximum size for training set.
    min_test_size : int, default=1
        Minimum size for test set.
    """
    
    def __init__(self, n_splits: int = 5, purge: int = 0, 
                 max_train_size: Optional[int] = None, min_test_size: int = 1):
        self.n_splits = n_splits
        self.purge = purge
        self.max_train_size = max_train_size
        self.min_test_size = min_test_size
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set with purging."""
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        if self.n_splits > n_samples:
            raise ValueError(f"Cannot have number of splits n_splits={self.n_splits} "
                           f"greater than the number of samples: n_samples={n_samples}.")
        
        # Calculate approximate test fold size
        test_fold_size = n_samples // (self.n_splits + 1)
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Calculate test boundaries
            test_start = (i + 1) * test_fold_size
            test_end = test_start + test_fold_size
            
            if test_end > n_samples:
                test_end = n_samples
            
            # Apply purge - exclude samples before test that could contain leakage
            purged_end = test_start - self.purge
            
            # Calculate train boundaries
            if self.max_train_size is None:
                train_start = 0
            else:
                train_start = max(0, purged_end - self.max_train_size)
            
            train_end = purged_end
            
            # Get indices
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            # Skip if test set is too small
            if len(test_indices) < self.min_test_size:
                continue
            
            # Skip if no training data
            if len(train_indices) == 0:
                continue
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits

# ============================================
# Anchored Time Series Cross-Validation
# ============================================

class AnchoredTimeSeriesSplit(BaseCrossValidator):
    """
    Anchored Time Series cross-validator.
    
    All training sets start from the same point (anchored), but test sets
    progress through time. This mimics a more realistic scenario where
    you have access to all historical data.
    
    Parameters:
    -----------
    n_splits : int, default=5
        Number of splits.
    test_size : int, optional
        Size of each test set.
    gap : int, default=0
        Number of samples to exclude between train and test sets.
    anchor_size : int, optional
        Minimum size of training set (anchor point).
    """
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None,
                 gap: int = 0, anchor_size: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.anchor_size = anchor_size
    
    def split(self, X, y=None, groups=None):
        """Generate indices for anchored time series split."""
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        if self.n_splits > n_samples:
            raise ValueError(f"Cannot have number of splits n_splits={self.n_splits} "
                           f"greater than the number of samples: n_samples={n_samples}.")
        
        # Calculate test size
        test_size = self.test_size
        if test_size is None:
            test_size = max(1, n_samples // (self.n_splits * 2))
        
        # Calculate anchor size
        anchor_size = self.anchor_size
        if anchor_size is None:
            anchor_size = max(1, n_samples // 3)  # Use first third as minimum training
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Test boundaries
            test_start = anchor_size + self.gap + i * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                test_end = n_samples
            
            if test_start >= test_end:
                break
            
            # Train always starts from beginning (anchored)
            train_start = 0
            train_end = test_start - self.gap
            
            # Get indices
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            # Skip if sets are too small
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits

# ============================================
# Blocking Time Series Cross-Validation
# ============================================

class BlockingTimeSeriesSplit(BaseCrossValidator):
    """
    Blocking Time Series cross-validator for highly autocorrelated data.
    
    Creates non-overlapping blocks of data with gaps to prevent leakage
    in highly autocorrelated time series (common in high-frequency financial data).
    
    Parameters:
    -----------
    n_splits : int, default=5
        Number of splits.
    block_size : int
        Size of each block (both training and test).
    gap : int, default=0
        Gap between training and test blocks.
    """
    
    def __init__(self, n_splits: int = 5, block_size: int = 100, gap: int = 0):
        self.n_splits = n_splits
        self.block_size = block_size
        self.gap = gap
    
    def split(self, X, y=None, groups=None):
        """Generate indices for blocking time series split."""
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        # Total space needed per split
        total_block_size = 2 * self.block_size + self.gap
        
        if total_block_size * self.n_splits > n_samples:
            raise ValueError(f"Not enough samples for {self.n_splits} splits. "
                           f"Need at least {total_block_size * self.n_splits} samples, "
                           f"but got {n_samples}.")
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Calculate block start
            block_start = i * total_block_size
            
            # Training block
            train_start = block_start
            train_end = train_start + self.block_size
            
            # Test block (after gap)
            test_start = train_end + self.gap
            test_end = test_start + self.block_size
            
            if test_end > n_samples:
                break
            
            # Get indices
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits

# ============================================
# Custom Financial Time Series Validator
# ============================================

class FinancialTimeSeriesValidator:
    """
    Comprehensive validator for financial time series models.
    
    Provides multiple validation strategies and metrics specific to
    financial applications.
    """
    
    def __init__(self, 
                 method: str = 'purged',
                 n_splits: int = 5,
                 purge: int = 0,
                 gap: int = 0,
                 max_train_size: Optional[int] = None,
                 test_size: Optional[int] = None):
        """
        Initialize financial time series validator.
        
        Parameters:
        -----------
        method : str, default='purged'
            Validation method ('standard', 'purged', 'anchored', 'blocking')
        n_splits : int, default=5
            Number of splits
        purge : int, default=0
            Number of observations to purge (for purged method)
        gap : int, default=0
            Gap between train and test sets
        max_train_size : int, optional
            Maximum training set size
        test_size : int, optional
            Test set size
        """
        
        self.method = method
        self.n_splits = n_splits
        self.purge = purge
        self.gap = gap
        self.max_train_size = max_train_size
        self.test_size = test_size
        
        # Initialize cross-validator based on method
        self.cv = self._get_cross_validator()
    
    def _get_cross_validator(self) -> BaseCrossValidator:
        """Get the appropriate cross-validator based on method."""
        
        if self.method == 'standard':
            return TimeSeriesSplit(
                n_splits=self.n_splits,
                max_train_size=self.max_train_size,
                test_size=self.test_size,
                gap=self.gap
            )
        
        elif self.method == 'purged':
            return PurgedTimeSeriesSplit(
                n_splits=self.n_splits,
                purge=self.purge,
                max_train_size=self.max_train_size
            )
        
        elif self.method == 'anchored':
            return AnchoredTimeSeriesSplit(
                n_splits=self.n_splits,
                test_size=self.test_size,
                gap=self.gap
            )
        
        elif self.method == 'blocking':
            block_size = self.test_size or 100
            return BlockingTimeSeriesSplit(
                n_splits=self.n_splits,
                block_size=block_size,
                gap=self.gap
            )
        
        else:
            raise ValueError(f"Unknown validation method: {self.method}")
    
    @time_it("cross_validation")
    def cross_validate(self, 
                      model: BaseEstimator,
                      X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.Series],
                      scoring: Optional[Union[str, callable]] = None,
                      return_predictions: bool = False) -> Dict[str, Any]:
        """
        Perform cross-validation with the configured method.
        
        Parameters:
        -----------
        model : BaseEstimator
            Model to validate
        X : array-like
            Features
        y : array-like
            Target values
        scoring : str or callable, optional
            Scoring function
        return_predictions : bool, default=False
            Whether to return out-of-sample predictions
            
        Returns:
        --------
        Dict containing validation results
        """
        
        results = {
            'method': self.method,
            'n_splits': self.n_splits,
            'scores': [],
            'train_sizes': [],
            'test_sizes': [],
            'fit_times': [],
            'score_times': []
        }
        
        if return_predictions:
            results['predictions'] = []
            results['test_indices'] = []
        
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(X, y)):
            fold_start_time = datetime.now()
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit model
            fit_start = datetime.now()
            model.fit(X_train, y_train)
            fit_time = (datetime.now() - fit_start).total_seconds()
            
            # Score model
            score_start = datetime.now()
            if scoring is None:
                score = model.score(X_test, y_test)
            elif callable(scoring):
                predictions = model.predict(X_test)
                score = scoring(y_test, predictions)
            else:
                # Assume sklearn scoring string
                from sklearn.metrics import get_scorer
                scorer = get_scorer(scoring)
                score = scorer(model, X_test, y_test)
            
            score_time = (datetime.now() - score_start).total_seconds()
            
            # Store results
            results['scores'].append(score)
            results['train_sizes'].append(len(train_idx))
            results['test_sizes'].append(len(test_idx))
            results['fit_times'].append(fit_time)
            results['score_times'].append(score_time)
            
            if return_predictions:
                predictions = model.predict(X_test)
                results['predictions'].append(predictions)
                results['test_indices'].append(test_idx)
            
            logger.debug(f"Fold {fold + 1}/{self.n_splits}: score={score:.4f}, "
                        f"train_size={len(train_idx)}, test_size={len(test_idx)}")
        
        # Calculate summary statistics
        results['mean_score'] = np.mean(results['scores'])
        results['std_score'] = np.std(results['scores'])
        results['mean_fit_time'] = np.mean(results['fit_times'])
        results['mean_score_time'] = np.mean(results['score_times'])
        
        return results
    
    def get_split_info(self, X: Union[np.ndarray, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Get information about train/test splits without running validation.
        
        Parameters:
        -----------
        X : array-like
            Data to split (only used for determining size)
            
        Returns:
        --------
        List of dictionaries with split information
        """
        
        X = np.asarray(X)
        split_info = []
        
        for i, (train_idx, test_idx) in enumerate(self.cv.split(X)):
            split_info.append({
                'split': i + 1,
                'train_start': train_idx[0] if len(train_idx) > 0 else None,
                'train_end': train_idx[-1] if len(train_idx) > 0 else None,
                'train_size': len(train_idx),
                'test_start': test_idx[0] if len(test_idx) > 0 else None,
                'test_end': test_idx[-1] if len(test_idx) > 0 else None,
                'test_size': len(test_idx),
                'gap': (test_idx[0] - train_idx[-1] - 1) if len(train_idx) > 0 and len(test_idx) > 0 else 0
            })
        
        return split_info

# ============================================
# Utility Functions
# ============================================

def validate_time_series_model(model: BaseEstimator,
                               X: Union[np.ndarray, pd.DataFrame],
                               y: Union[np.ndarray, pd.Series],
                               method: str = 'purged',
                               n_splits: int = 5,
                               purge: int = 0,
                               scoring: Optional[Union[str, callable]] = None) -> Dict[str, Any]:
    """
    Quick utility function for time series model validation.
    
    Parameters:
    -----------
    model : BaseEstimator
        Model to validate
    X : array-like
        Features
    y : array-like
        Target values
    method : str, default='purged'
        Validation method
    n_splits : int, default=5
        Number of splits
    purge : int, default=0
        Number of observations to purge
    scoring : str or callable, optional
        Scoring function
        
    Returns:
    --------
    Dictionary with validation results
    """
    
    validator = FinancialTimeSeriesValidator(
        method=method,
        n_splits=n_splits,
        purge=purge
    )
    
    return validator.cross_validate(model, X, y, scoring=scoring)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Time Series Validation Methods")
    
    # Generate sample time series data
    np.random.seed(42)
    n_samples = 1000
    
    # Create autocorrelated time series
    y = np.zeros(n_samples)
    y[0] = np.random.randn()
    
    for i in range(1, n_samples):
        y[i] = 0.7 * y[i-1] + 0.3 * np.random.randn()
    
    # Create some features
    X = np.column_stack([
        np.roll(y, 1),  # Lagged y
        np.roll(y, 2),  # Double lagged y
        np.random.randn(n_samples),  # Random feature
        np.cumsum(np.random.randn(n_samples))  # Random walk feature
    ])[2:]  # Remove first 2 rows due to lags
    
    y = y[2:]  # Align y with X
    
    print(f"Generated time series data: {X.shape}")
    
    # Test different cross-validation methods
    print("\n1. Testing TimeSeriesSplit")
    
    tscv = TimeSeriesSplit(n_splits=5, gap=5)
    split_info = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1}: Train={len(train_idx)}, Test={len(test_idx)}, "
              f"Gap={test_idx[0] - train_idx[-1] - 1}")
        split_info.append((train_idx, test_idx))
    
    # Test purged cross-validation
    print("\n2. Testing PurgedTimeSeriesSplit")
    
    purged_cv = PurgedTimeSeriesSplit(n_splits=5, purge=10)
    
    for fold, (train_idx, test_idx) in enumerate(purged_cv.split(X)):
        print(f"Fold {fold + 1}: Train={len(train_idx)}, Test={len(test_idx)}, "
              f"Purge={test_idx[0] - train_idx[-1] - 1}")
    
    # Test anchored cross-validation
    print("\n3. Testing AnchoredTimeSeriesSplit")
    
    anchored_cv = AnchoredTimeSeriesSplit(n_splits=5, test_size=50)
    
    for fold, (train_idx, test_idx) in enumerate(anchored_cv.split(X)):
        print(f"Fold {fold + 1}: Train={len(train_idx)} (start=0), Test={len(test_idx)}")
    
    # Test blocking cross-validation
    print("\n4. Testing BlockingTimeSeriesSplit")
    
    blocking_cv = BlockingTimeSeriesSplit(n_splits=3, block_size=100, gap=20)
    
    for fold, (train_idx, test_idx) in enumerate(blocking_cv.split(X)):
        print(f"Fold {fold + 1}: Train={len(train_idx)}, Test={len(test_idx)}, "
              f"Gap={test_idx[0] - train_idx[-1] - 1}")
    
    # Test comprehensive validator
    print("\n5. Testing FinancialTimeSeriesValidator")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    model = LinearRegression()
    
    validator = FinancialTimeSeriesValidator(
        method='purged',
        n_splits=5,
        purge=5
    )
    
    # Get split information
    splits = validator.get_split_info(X)
    print("Split Information:")
    for split in splits:
        print(f"  Split {split['split']}: Train [{split['train_start']}:{split['train_end']}] "
              f"Test [{split['test_start']}:{split['test_end']}] Gap={split['gap']}")
    
    # Run cross-validation
    def neg_mse_scorer(y_true, y_pred):
        return -mean_squared_error(y_true, y_pred)
    
    results = validator.cross_validate(
        model, X, y, 
        scoring=neg_mse_scorer,
        return_predictions=True
    )
    
    print(f"\nCross-Validation Results:")
    print(f"  Method: {results['method']}")
    print(f"  Mean Score: {results['mean_score']:.6f} ± {results['std_score']:.6f}")
    print(f"  Mean Fit Time: {results['mean_fit_time']:.4f}s")
    print(f"  Individual Scores: {[f'{s:.6f}' for s in results['scores']]}")
    
    # Test utility function
    print("\n6. Testing Utility Function")
    
    quick_results = validate_time_series_model(
        model=LinearRegression(),
        X=X, y=y,
        method='purged',
        n_splits=3,
        purge=10,
        scoring=neg_mse_scorer
    )
    
    print(f"Quick Validation Results:")
    print(f"  Mean Score: {quick_results['mean_score']:.6f}")
    print(f"  Std Score: {quick_results['std_score']:.6f}")
    
    print("\nTime series validation testing completed successfully!")
