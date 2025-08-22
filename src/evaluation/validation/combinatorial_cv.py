# ============================================
# StockPredictionPro - src/evaluation/validation/combinatorial_cv.py
# Advanced combinatorial cross-validation for generating multiple backtest paths in financial ML
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
import warnings
from datetime import datetime, timedelta
from itertools import combinations, permutations
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from math import comb

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('evaluation.validation.combinatorial_cv')

# ============================================
# Basic Combinatorial Cross-Validation
# ============================================

class CombinatorialCV(BaseCrossValidator):
    """
    Combinatorial Cross-Validation for generating multiple backtest paths.
    
    This method generates multiple train/test splits by selecting different
    combinations of data groups, providing more robust model evaluation
    than single-path backtesting.
    
    Based on the methodology from "Advances in Financial Machine Learning"
    by Marcos Lopez de Prado.
    
    Parameters:
    -----------
    n_groups : int, default=10
        Number of groups to divide the data into.
    n_test_groups : int, default=2
        Number of groups to use for testing in each combination.
    n_train_groups : int, optional
        Number of groups to use for training (if None, uses all non-test groups).
    shuffle_groups : bool, default=False
        Whether to shuffle the order of groups.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(self, n_groups: int = 10, n_test_groups: int = 2,
                 n_train_groups: Optional[int] = None, shuffle_groups: bool = False,
                 random_state: Optional[int] = None):
        
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.n_train_groups = n_train_groups
        self.shuffle_groups = shuffle_groups
        self.random_state = random_state
        
        if self.n_test_groups >= self.n_groups:
            raise ValueError("n_test_groups must be less than n_groups")
    
    def split(self, X, y=None, groups=None):
        """Generate combinatorial splits."""
        
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        # Create group boundaries
        group_boundaries = self._create_group_boundaries(n_samples)
        
        # Generate all possible combinations of test groups
        test_combinations = list(combinations(range(self.n_groups), self.n_test_groups))
        
        # Optionally shuffle combinations for randomness
        if self.shuffle_groups and self.random_state is not None:
            np.random.seed(self.random_state)
            np.random.shuffle(test_combinations)
        
        for test_groups in test_combinations:
            # Get available train groups
            all_groups = set(range(self.n_groups))
            available_train_groups = list(all_groups - set(test_groups))
            
            # Select train groups
            if self.n_train_groups is not None:
                if len(available_train_groups) >= self.n_train_groups:
                    # Take the most recent groups (highest indices)
                    train_groups = sorted(available_train_groups)[-self.n_train_groups:]
                else:
                    train_groups = available_train_groups
            else:
                train_groups = available_train_groups
            
            # Skip if no training groups available
            if len(train_groups) == 0:
                continue
            
            # Convert groups to indices
            test_indices = self._groups_to_indices(test_groups, group_boundaries)
            train_indices = self._groups_to_indices(train_groups, group_boundaries)
            
            # Skip if either set is empty
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
            
            yield train_indices, test_indices
    
    def _create_group_boundaries(self, n_samples: int) -> List[int]:
        """Create boundaries for dividing data into groups."""
        
        group_size = n_samples // self.n_groups
        boundaries = []
        
        for i in range(self.n_groups + 1):
            if i == self.n_groups:
                boundaries.append(n_samples)
            else:
                boundaries.append(i * group_size)
        
        return boundaries
    
    def _groups_to_indices(self, groups: List[int], boundaries: List[int]) -> np.ndarray:
        """Convert group numbers to sample indices."""
        
        indices = []
        for group in groups:
            start = boundaries[group]
            end = boundaries[group + 1]
            indices.extend(range(start, end))
        
        return np.array(indices, dtype=int)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        return comb(self.n_groups, self.n_test_groups)

# ============================================
# Stacked Combinatorial Cross-Validation
# ============================================

class StackedCombinatorialCV(BaseCrossValidator):
    """
    Stacked Combinatorial Cross-Validation for multi-asset datasets.
    
    This method applies combinatorial CV separately to different assets/instruments
    and then combines the results, useful for portfolio-level validation.
    
    Parameters:
    -----------
    n_groups : int, default=6
        Number of time groups to divide each asset's data into.
    n_test_groups : int, default=2
        Number of groups to use for testing.
    asset_groups : array-like, optional
        Asset identifiers for each sample.
    min_asset_samples : int, default=50
        Minimum number of samples required per asset.
    """
    
    def __init__(self, n_groups: int = 6, n_test_groups: int = 2,
                 asset_groups: Optional[np.ndarray] = None,
                 min_asset_samples: int = 50):
        
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.asset_groups = asset_groups
        self.min_asset_samples = min_asset_samples
    
    def split(self, X, y=None, groups=None):
        """Generate stacked combinatorial splits."""
        
        X, y, groups = indexable(X, y, groups)
        
        # Use provided asset groups or the groups parameter
        if self.asset_groups is not None:
            asset_ids = self.asset_groups
        elif groups is not None:
            asset_ids = groups
        else:
            raise ValueError("Must provide asset_groups or groups parameter")
        
        # Get unique assets and their sample counts
        unique_assets = np.unique(asset_ids)
        asset_sample_counts = {asset: np.sum(asset_ids == asset) for asset in unique_assets}
        
        # Filter assets with sufficient samples
        valid_assets = [asset for asset, count in asset_sample_counts.items() 
                       if count >= self.min_asset_samples]
        
        if len(valid_assets) == 0:
            raise ValueError("No assets have sufficient samples for combinatorial CV")
        
        # Create combinatorial CV for each asset
        asset_cvs = {}
        asset_splits = {}
        
        for asset in valid_assets:
            asset_mask = asset_ids == asset
            asset_indices = np.where(asset_mask)[0]
            
            # Create CV for this asset
            cv = CombinatorialCV(
                n_groups=min(self.n_groups, len(asset_indices) // 10),  # Adjust for small assets
                n_test_groups=self.n_test_groups
            )
            
            # Generate splits for this asset (relative to asset data)
            asset_X = X[asset_mask] if hasattr(X, '__getitem__') else X[asset_indices]
            asset_y = y[asset_mask] if y is not None else None
            
            splits = list(cv.split(asset_X, asset_y))
            
            # Convert relative indices back to absolute indices
            absolute_splits = []
            for train_rel, test_rel in splits:
                train_abs = asset_indices[train_rel]
                test_abs = asset_indices[test_rel]
                absolute_splits.append((train_abs, test_abs))
            
            asset_splits[asset] = absolute_splits
        
        # Generate combined splits across all assets
        # Each split combines one path from each asset
        max_splits = max(len(splits) for splits in asset_splits.values())
        
        for split_idx in range(max_splits):
            train_indices = []
            test_indices = []
            
            for asset, splits in asset_splits.items():
                if split_idx < len(splits):
                    train_abs, test_abs = splits[split_idx]
                    train_indices.extend(train_abs)
                    test_indices.extend(test_abs)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield np.array(train_indices), np.array(test_indices)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        # This is approximate since it depends on the data
        return comb(self.n_groups, self.n_test_groups)

# ============================================
# Probability-Based Combinatorial CV
# ============================================

class ProbabilisticCombinatorialCV(BaseCrossValidator):
    """
    Probabilistic Combinatorial Cross-Validation.
    
    Instead of using all possible combinations, this method randomly samples
    combinations based on specified probabilities, useful for large datasets
    where exhaustive combinatorial CV would be computationally prohibitive.
    
    Parameters:
    -----------
    n_groups : int, default=10
        Number of groups to divide the data into.
    n_test_groups : int, default=2
        Number of groups to use for testing.
    n_combinations : int, default=50
        Number of random combinations to sample.
    test_group_probs : array-like, optional
        Probabilities for selecting each group for testing.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(self, n_groups: int = 10, n_test_groups: int = 2,
                 n_combinations: int = 50, 
                 test_group_probs: Optional[np.ndarray] = None,
                 random_state: Optional[int] = None):
        
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.n_combinations = n_combinations
        self.test_group_probs = test_group_probs
        self.random_state = random_state
        
        if self.test_group_probs is not None:
            if len(self.test_group_probs) != self.n_groups:
                raise ValueError("test_group_probs must have length equal to n_groups")
            if not np.allclose(np.sum(self.test_group_probs), 1.0):
                self.test_group_probs = self.test_group_probs / np.sum(self.test_group_probs)
    
    def split(self, X, y=None, groups=None):
        """Generate probabilistic combinatorial splits."""
        
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Create group boundaries
        group_boundaries = self._create_group_boundaries(n_samples)
        
        # Generate random combinations
        generated_combinations = set()
        
        for _ in range(self.n_combinations * 2):  # Try more to avoid duplicates
            if len(generated_combinations) >= self.n_combinations:
                break
            
            # Sample test groups
            if self.test_group_probs is not None:
                test_groups = tuple(np.random.choice(
                    self.n_groups, 
                    size=self.n_test_groups, 
                    replace=False,
                    p=self.test_group_probs
                ))
            else:
                test_groups = tuple(np.random.choice(
                    self.n_groups, 
                    size=self.n_test_groups, 
                    replace=False
                ))
            
            generated_combinations.add(test_groups)
        
        # Convert to splits
        for test_groups in generated_combinations:
            # Get train groups
            all_groups = set(range(self.n_groups))
            train_groups = list(all_groups - set(test_groups))
            
            if len(train_groups) == 0:
                continue
            
            # Convert to indices
            test_indices = self._groups_to_indices(test_groups, group_boundaries)
            train_indices = self._groups_to_indices(train_groups, group_boundaries)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def _create_group_boundaries(self, n_samples: int) -> List[int]:
        """Create boundaries for dividing data into groups."""
        
        group_size = n_samples // self.n_groups
        boundaries = []
        
        for i in range(self.n_groups + 1):
            if i == self.n_groups:
                boundaries.append(n_samples)
            else:
                boundaries.append(i * group_size)
        
        return boundaries
    
    def _groups_to_indices(self, groups: Tuple[int, ...], boundaries: List[int]) -> np.ndarray:
        """Convert group numbers to sample indices."""
        
        indices = []
        for group in groups:
            start = boundaries[group]
            end = boundaries[group + 1]
            indices.extend(range(start, end))
        
        return np.array(indices, dtype=int)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        return self.n_combinations

# ============================================
# Adaptive Combinatorial CV
# ============================================

class AdaptiveCombinatorialCV(BaseCrossValidator):
    """
    Adaptive Combinatorial Cross-Validation.
    
    Automatically adjusts the number of groups and combinations based on
    data characteristics and computational constraints.
    
    Parameters:
    -----------
    max_combinations : int, default=100
        Maximum number of combinations to generate.
    min_test_size : int, default=100
        Minimum size of test sets.
    max_test_size : int, default=1000
        Maximum size of test sets.
    auto_adjust : bool, default=True
        Whether to automatically adjust parameters based on data size.
    """
    
    def __init__(self, max_combinations: int = 100,
                 min_test_size: int = 100, max_test_size: int = 1000,
                 auto_adjust: bool = True):
        
        self.max_combinations = max_combinations
        self.min_test_size = min_test_size
        self.max_test_size = max_test_size
        self.auto_adjust = auto_adjust
    
    def split(self, X, y=None, groups=None):
        """Generate adaptive combinatorial splits."""
        
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        # Automatically determine optimal parameters
        if self.auto_adjust:
            n_groups, n_test_groups = self._determine_optimal_groups(n_samples)
        else:
            n_groups = max(6, min(20, n_samples // self.max_test_size))
            n_test_groups = 2
        
        # Limit the number of actual combinations
        total_combinations = comb(n_groups, n_test_groups)
        
        if total_combinations <= self.max_combinations:
            # Use all combinations
            cv = CombinatorialCV(n_groups=n_groups, n_test_groups=n_test_groups)
            yield from cv.split(X, y, groups)
        else:
            # Use probabilistic sampling
            cv = ProbabilisticCombinatorialCV(
                n_groups=n_groups,
                n_test_groups=n_test_groups,
                n_combinations=self.max_combinations
            )
            yield from cv.split(X, y, groups)
    
    def _determine_optimal_groups(self, n_samples: int) -> Tuple[int, int]:
        """Determine optimal number of groups and test groups."""
        
        # Target test size range
        target_test_size = max(self.min_test_size, min(self.max_test_size, n_samples // 10))
        
        # Calculate optimal number of groups
        n_groups = max(6, min(20, n_samples // target_test_size))
        
        # Determine test groups (typically 15-25% of total groups)
        n_test_groups = max(1, min(n_groups // 4, 3))
        
        # Ensure we don't exceed max combinations
        while comb(n_groups, n_test_groups) > self.max_combinations and n_groups > 6:
            if n_test_groups > 1:
                n_test_groups -= 1
            else:
                n_groups -= 1
        
        return n_groups, n_test_groups
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        if X is not None:
            n_samples = _num_samples(X)
            n_groups, n_test_groups = self._determine_optimal_groups(n_samples)
            total_combinations = comb(n_groups, n_test_groups)
            return min(total_combinations, self.max_combinations)
        return self.max_combinations

# ============================================
# Combinatorial CV with Purging and Embargo
# ============================================

class CombinatorialPurgedCV(BaseCrossValidator):
    """
    Combinatorial Cross-Validation with Purging and Embargo.
    
    Combines combinatorial backtesting with purging and embargo mechanisms
    to prevent information leakage in financial time series.
    
    Parameters:
    -----------
    n_groups : int, default=6
        Number of groups to divide the data into.
    n_test_groups : int, default=2
        Number of groups to use for testing.
    purge_pct : float, default=0.01
        Percentage of samples to purge between train and test.
    embargo_pct : float, default=0.01
        Percentage of samples to embargo after test periods.
    """
    
    def __init__(self, n_groups: int = 6, n_test_groups: int = 2,
                 purge_pct: float = 0.01, embargo_pct: float = 0.01):
        
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
    
    def split(self, X, y=None, groups=None):
        """Generate combinatorial splits with purging and embargo."""
        
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        
        # Create base combinatorial CV
        base_cv = CombinatorialCV(
            n_groups=self.n_groups,
            n_test_groups=self.n_test_groups
        )
        
        # Apply purging and embargo to each split
        for train_indices, test_indices in base_cv.split(X, y, groups):
            
            # Apply purging and embargo
            purged_train_indices = self._apply_purge_embargo(
                train_indices, test_indices, n_samples
            )
            
            if len(purged_train_indices) > 0:
                yield purged_train_indices, test_indices
    
    def _apply_purge_embargo(self, train_indices: np.ndarray, 
                           test_indices: np.ndarray, n_samples: int) -> np.ndarray:
        """Apply purging and embargo to training indices."""
        
        if len(test_indices) == 0:
            return train_indices
        
        # Calculate purge and embargo sizes
        purge_size = int(n_samples * self.purge_pct)
        embargo_size = int(n_samples * self.embargo_pct)
        
        # Find test period boundaries
        test_start = np.min(test_indices)
        test_end = np.max(test_indices)
        
        # Apply purging (remove samples before test that are too close)
        purge_mask = ~(
            (train_indices >= test_start - purge_size) & 
            (train_indices < test_start)
        )
        
        # Apply embargo (remove samples after test that are too close)
        embargo_mask = ~(
            (train_indices > test_end) & 
            (train_indices <= test_end + embargo_size)
        )
        
        # Combine masks
        final_mask = purge_mask & embargo_mask
        
        return train_indices[final_mask]
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        return comb(self.n_groups, self.n_test_groups)

# ============================================
# Utility Functions
# ============================================

@time_it("combinatorial_cross_validation")
def combinatorial_cross_validate(model: BaseEstimator,
                                X: Union[np.ndarray, pd.DataFrame],
                                y: Union[np.ndarray, pd.Series],
                                cv: Optional[BaseCrossValidator] = None,
                                scoring: Optional[Union[str, callable]] = None,
                                n_groups: int = 6,
                                n_test_groups: int = 2,
                                return_paths: bool = True) -> Dict[str, Any]:
    """
    Perform combinatorial cross-validation.
    
    Parameters:
    -----------
    model : BaseEstimator
        Model to validate
    X : array-like
        Features
    y : array-like
        Target values
    cv : BaseCrossValidator, optional
        Cross-validator to use
    scoring : str or callable, optional
        Scoring function
    n_groups : int, default=6
        Number of groups for default CV
    n_test_groups : int, default=2
        Number of test groups for default CV
    return_paths : bool, default=True
        Whether to return individual path results
        
    Returns:
    --------
    Dictionary with validation results including all backtest paths
    """
    
    if cv is None:
        cv = CombinatorialCV(n_groups=n_groups, n_test_groups=n_test_groups)
    
    results = {
        'method': 'combinatorial',
        'n_paths': 0,
        'path_scores': [],
        'path_details': [],
        'combined_predictions': [],
        'combined_actuals': [],
        'fit_times': []
    }
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    for path_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        path_start = datetime.now()
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        fit_time = (datetime.now() - path_start).total_seconds()
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Score model
        if scoring is None:
            score = model.score(X_test, y_test)
        elif callable(scoring):
            score = scoring(y_test, predictions)
        else:
            from sklearn.metrics import get_scorer
            scorer = get_scorer(scoring)
            score = scorer(model, X_test, y_test)
        
        # Store path results
        results['path_scores'].append(score)
        results['fit_times'].append(fit_time)
        
        if return_paths:
            results['path_details'].append({
                'path': path_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'score': score,
                'fit_time': fit_time,
                'train_period': [train_idx[0], train_idx[-1]] if len(train_idx) > 0 else [],
                'test_period': [test_idx[0], test_idx[-1]] if len(test_idx) > 0 else []
            })
        
        # Collect predictions for combined analysis
        results['combined_predictions'].extend(predictions.tolist())
        results['combined_actuals'].extend(y_test.tolist())
        
        logger.debug(f"Path {path_idx + 1}: score={score:.4f}, "
                    f"train_size={len(train_idx)}, test_size={len(test_idx)}")
    
    # Calculate summary statistics
    results['n_paths'] = len(results['path_scores'])
    
    if results['n_paths'] > 0:
        results['mean_score'] = np.mean(results['path_scores'])
        results['std_score'] = np.std(results['path_scores'])
        results['min_score'] = np.min(results['path_scores'])
        results['max_score'] = np.max(results['path_scores'])
        results['median_score'] = np.median(results['path_scores'])
        
        # Combined score across all paths
        if scoring is None or isinstance(scoring, str):
            # Use model's default score on combined data
            if len(results['combined_predictions']) > 0:
                combined_pred = np.array(results['combined_predictions'])
                combined_actual = np.array(results['combined_actuals'])
                if callable(scoring):
                    results['combined_score'] = scoring(combined_actual, combined_pred)
                else:
                    # Correlation for regression-like problems
                    results['combined_score'] = np.corrcoef(combined_actual, combined_pred)[0, 1]
        elif callable(scoring):
            combined_pred = np.array(results['combined_predictions'])
            combined_actual = np.array(results['combined_actuals'])
            results['combined_score'] = scoring(combined_actual, combined_pred)
    
    return results

def analyze_backtest_paths(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the stability and consistency of backtest paths.
    
    Parameters:
    -----------
    cv_results : dict
        Results from combinatorial_cross_validate
        
    Returns:
    --------
    Dictionary with path analysis metrics
    """
    
    if 'path_scores' not in cv_results or len(cv_results['path_scores']) == 0:
        return {'error': 'No path scores found in results'}
    
    scores = np.array(cv_results['path_scores'])
    
    analysis = {
        'n_paths': len(scores),
        'score_distribution': {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores),
            'q25': np.percentile(scores, 25),
            'q75': np.percentile(scores, 75)
        },
        'stability_metrics': {
            'coefficient_of_variation': np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else np.inf,
            'score_range': np.max(scores) - np.min(scores),
            'interquartile_range': np.percentile(scores, 75) - np.percentile(scores, 25)
        }
    }
    
    # Consistency analysis
    positive_paths = np.sum(scores > 0)
    analysis['consistency'] = {
        'positive_paths': positive_paths,
        'positive_ratio': positive_paths / len(scores),
        'negative_paths': len(scores) - positive_paths,
        'consistent_positive': positive_paths / len(scores) > 0.6,
        'consistent_negative': positive_paths / len(scores) < 0.4
    }
    
    # Outlier detection
    q1, q3 = np.percentile(scores, [25, 75])
    iqr = q3 - q1
    outlier_threshold_low = q1 - 1.5 * iqr
    outlier_threshold_high = q3 + 1.5 * iqr
    
    outliers = scores[(scores < outlier_threshold_low) | (scores > outlier_threshold_high)]
    analysis['outliers'] = {
        'count': len(outliers),
        'ratio': len(outliers) / len(scores),
        'values': outliers.tolist()
    }
    
    return analysis

def compare_combinatorial_methods(model: BaseEstimator,
                                 X: Union[np.ndarray, pd.DataFrame],
                                 y: Union[np.ndarray, pd.Series],
                                 methods: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare different combinatorial CV methods.
    
    Parameters:
    -----------
    model : BaseEstimator
        Model to validate
    X : array-like
        Features
    y : array-like
        Target values
    methods : list of str, optional
        Methods to compare
        
    Returns:
    --------
    DataFrame with comparison results
    """
    
    if methods is None:
        methods = ['basic', 'probabilistic', 'adaptive', 'purged']
    
    results = []
    
    for method in methods:
        try:
            if method == 'basic':
                cv = CombinatorialCV(n_groups=6, n_test_groups=2)
            elif method == 'probabilistic':
                cv = ProbabilisticCombinatorialCV(n_groups=8, n_test_groups=2, n_combinations=20)
            elif method == 'adaptive':
                cv = AdaptiveCombinatorialCV(max_combinations=25)
            elif method == 'purged':
                cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
            else:
                continue
            
            # Run cross-validation
            cv_results = combinatorial_cross_validate(model, X, y, cv=cv)
            
            # Extract key metrics
            results.append({
                'method': method,
                'n_paths': cv_results['n_paths'],
                'mean_score': cv_results.get('mean_score', np.nan),
                'std_score': cv_results.get('std_score', np.nan),
                'min_score': cv_results.get('min_score', np.nan),
                'max_score': cv_results.get('max_score', np.nan),
                'combined_score': cv_results.get('combined_score', np.nan)
            })
            
        except Exception as e:
            logger.warning(f"Error running {method} combinatorial CV: {e}")
            results.append({
                'method': method,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Combinatorial Cross-Validation Methods")
    
    # Generate sample financial time series data
    np.random.seed(42)
    n_samples = 2000
    n_features = 5
    
    # Create synthetic financial features
    returns = np.random.normal(0.001, 0.02, n_samples)
    
    # Add autocorrelation
    for i in range(1, n_samples):
        returns[i] += 0.1 * returns[i-1]
    
    # Create features (lags, moving averages, etc.)
    X = np.column_stack([
        np.roll(returns, i+1) for i in range(n_features)
    ])[n_features:]
    
    # Create forward-looking target
    y = np.roll(returns, -1)[n_features:-1]
    
    print(f"Generated data: X={X.shape}, y={len(y)}")
    
    # Test basic CombinatorialCV
    print("\n1. Testing CombinatorialCV")
    
    basic_cv = CombinatorialCV(n_groups=6, n_test_groups=2)
    n_splits = basic_cv.get_n_splits()
    print(f"Number of combinations: {n_splits}")
    
    split_info = []
    for i, (train_idx, test_idx) in enumerate(basic_cv.split(X)):
        split_info.append({
            'path': i + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_period': [train_idx[0], train_idx[-1]] if len(train_idx) > 0 else [],
            'test_period': [test_idx[0], test_idx[-1]] if len(test_idx) > 0 else []
        })
        
        if i < 5:  # Show first 5 paths
            print(f"Path {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Test ProbabilisticCombinatorialCV
    print("\n2. Testing ProbabilisticCombinatorialCV")
    
    prob_cv = ProbabilisticCombinatorialCV(
        n_groups=10,
        n_test_groups=3,
        n_combinations=15,
        random_state=42
    )
    
    for i, (train_idx, test_idx) in enumerate(prob_cv.split(X)):
        print(f"Path {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
        if i >= 4:  # Show first 5 paths
            break
    
    # Test AdaptiveCombinatorialCV
    print("\n3. Testing AdaptiveCombinatorialCV")
    
    adaptive_cv = AdaptiveCombinatorialCV(max_combinations=20)
    adaptive_splits = adaptive_cv.get_n_splits(X)
    print(f"Adaptive CV will generate {adaptive_splits} splits")
    
    for i, (train_idx, test_idx) in enumerate(adaptive_cv.split(X)):
        if i < 3:  # Show first 3 paths
            print(f"Path {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Test CombinatorialPurgedCV
    print("\n4. Testing CombinatorialPurgedCV")
    
    purged_cv = CombinatorialPurgedCV(
        n_groups=6,
        n_test_groups=2,
        purge_pct=0.02,
        embargo_pct=0.01
    )
    
    for i, (train_idx, test_idx) in enumerate(purged_cv.split(X)):
        original_train_size = len(X) - len(test_idx)
        purged_samples = original_train_size - len(train_idx)
        
        if i < 3:  # Show first 3 paths
            print(f"Path {i+1}: Train={len(train_idx)}, Test={len(test_idx)}, "
                  f"Purged={purged_samples}")
    
    # Test comprehensive combinatorial cross-validation
    print("\n5. Testing Comprehensive Combinatorial Cross-Validation")
    
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    
    model = Ridge(alpha=0.1)
    
    def neg_mse_scorer(y_true, y_pred):
        return -mean_squared_error(y_true, y_pred)
    
    # Run combinatorial cross-validation
    cv_results = combinatorial_cross_validate(
        model=model,
        X=X, y=y,
        cv=basic_cv,
        scoring=neg_mse_scorer,
        return_paths=True
    )
    
    print("Combinatorial CV Results:")
    print(f"  Number of paths: {cv_results['n_paths']}")
    print(f"  Mean score: {cv_results['mean_score']:.6f} ± {cv_results['std_score']:.6f}")
    print(f"  Score range: [{cv_results['min_score']:.6f}, {cv_results['max_score']:.6f}]")
    print(f"  Combined score: {cv_results.get('combined_score', 'N/A'):.6f}")
    
    # Analyze backtest paths
    print("\n6. Analyzing Backtest Paths")
    
    path_analysis = analyze_backtest_paths(cv_results)
    
    print("Path Analysis:")
    print(f"  Score distribution: mean={path_analysis['score_distribution']['mean']:.6f}, "
          f"std={path_analysis['score_distribution']['std']:.6f}")
    print(f"  Coefficient of variation: {path_analysis['stability_metrics']['coefficient_of_variation']:.4f}")
    print(f"  Positive paths: {path_analysis['consistency']['positive_paths']}/{path_analysis['n_paths']} "
          f"({path_analysis['consistency']['positive_ratio']:.1%})")
    print(f"  Outliers: {path_analysis['outliers']['count']}/{path_analysis['n_paths']} "
          f"({path_analysis['outliers']['ratio']:.1%})")
    
    # Compare different methods
    print("\n7. Comparing Combinatorial Methods")
    
    comparison_df = compare_combinatorial_methods(
        model, X[:1000], y[:1000],  # Use smaller dataset for speed
        methods=['basic', 'probabilistic', 'adaptive']
    )
    
    print("Method Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Show individual path details
    print("\n8. Individual Path Details (First 5 paths):")
    
    for i, path_detail in enumerate(cv_results['path_details'][:5]):
        print(f"Path {path_detail['path']}: Score={path_detail['score']:.6f}, "
              f"Train=[{path_detail['train_period'][0]}:{path_detail['train_period'][1]}], "
              f"Test=[{path_detail['test_period'][0]}:{path_detail['test_period'][1]}]")
    
    # Performance statistics
    print(f"\n9. Performance Summary:")
    print(f"Total paths evaluated: {cv_results['n_paths']}")
    print(f"Average fit time per path: {np.mean(cv_results['fit_times']):.4f}s")
    print(f"Total validation time: {np.sum(cv_results['fit_times']):.2f}s")
    
    print("\nCombinatorial cross-validation testing completed successfully!")
