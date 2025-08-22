# ============================================
# StockPredictionPro - src/data/processors/splitter.py
# Advanced data splitting for time series with financial market awareness
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.base import BaseEstimator
import warnings

from ...utils.exceptions import DataValidationError, BusinessLogicError, InvalidParameterError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ...utils.config_loader import get
from ...utils.helpers import validate_numeric_range

logger = get_logger('data.processors.splitter')

# ============================================
# Time Series Splitting Strategies
# ============================================

class FinancialTimeSeriesSplitter:
    """
    Advanced time series splitting for financial data
    
    Features:
    - Multiple splitting strategies
    - Market regime awareness
    - Walk-forward analysis support
    - Purging and embargo periods
    - Financial calendar awareness
    """
    
    def __init__(self,
                 strategy: str = 'time_series',
                 test_size: float = 0.2,
                 validation_size: float = 0.2,
                 gap_days: int = 0,
                 embargo_days: int = 0,
                 min_train_size: int = 252,  # 1 year of daily data
                 max_train_size: Optional[int] = None,
                 preserve_order: bool = True):
        """
        Initialize financial time series splitter
        
        Args:
            strategy: Splitting strategy ('time_series', 'walk_forward', 'purged', 'blocking')
            test_size: Proportion or absolute size of test set
            validation_size: Proportion of validation set from remaining data
            gap_days: Gap between train and test sets (prevent data leakage)
            embargo_days: Embargo period to prevent look-ahead bias
            min_train_size: Minimum training set size
            max_train_size: Maximum training set size (None for no limit)
            preserve_order: Whether to preserve temporal order
        """
        self.strategy = strategy
        self.test_size = test_size
        self.validation_size = validation_size
        self.gap_days = gap_days
        self.embargo_days = embargo_days
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.preserve_order = preserve_order
        
        # Validate parameters
        self._validate_parameters()
        
        # Market calendar awareness
        self.trading_calendar = None
        self.market_regimes = None
    
    def _validate_parameters(self):
        """Validate initialization parameters"""
        if self.strategy not in ['time_series', 'walk_forward', 'purged', 'blocking', 'expanding']:
            raise InvalidParameterError(f"Unknown splitting strategy: {self.strategy}")
        
        if not 0 < self.test_size < 1:
            if self.test_size >= 1 and not isinstance(self.test_size, int):
                raise InvalidParameterError("test_size must be between 0 and 1, or an integer >= 1")
        
        if not 0 <= self.validation_size < 1:
            raise InvalidParameterError("validation_size must be between 0 and 1")
        
        if self.gap_days < 0:
            raise InvalidParameterError("gap_days must be non-negative")
        
        if self.min_train_size < 1:
            raise InvalidParameterError("min_train_size must be positive")
    
    @time_it("financial_data_split")
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
              groups: Optional[pd.Series] = None) -> Dict[str, Tuple[pd.Index, pd.Index]]:
        """
        Split data using specified strategy
        
        Args:
            X: Feature data
            y: Target data (optional)
            groups: Group labels (optional)
            
        Returns:
            Dictionary with split indices
        """
        logger.info(f"Splitting data using {self.strategy} strategy")
        
        if not isinstance(X.index, pd.DatetimeIndex):
            raise DataValidationError("Data must have DatetimeIndex for time series splitting")
        
        if len(X) < self.min_train_size + 50:  # Need minimum data
            raise DataValidationError(f"Insufficient data: {len(X)} rows, need at least {self.min_train_size + 50}")
        
        # Route to appropriate splitting method
        if self.strategy == 'time_series':
            return self._time_series_split(X, y)
        elif self.strategy == 'walk_forward':
            return self._walk_forward_split(X, y)
        elif self.strategy == 'purged':
            return self._purged_split(X, y)
        elif self.strategy == 'blocking':
            return self._blocking_split(X, y)
        elif self.strategy == 'expanding':
            return self._expanding_window_split(X, y)
        else:
            raise InvalidParameterError(f"Strategy {self.strategy} not implemented")
    
    def _time_series_split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Tuple[pd.Index, pd.Index]]:
        """Standard time series split with gap and embargo"""
        
        # Calculate split points
        total_size = len(X)
        
        if self.test_size < 1:
            test_size = int(total_size * self.test_size)
        else:
            test_size = int(self.test_size)
        
        # Apply embargo and gap
        effective_test_start = total_size - test_size - self.embargo_days
        train_end = effective_test_start - self.gap_days
        
        # Ensure minimum training size
        if train_end < self.min_train_size:
            raise DataValidationError(f"Insufficient training data after applying gaps and embargo: {train_end} < {self.min_train_size}")
        
        # Apply maximum training size if specified
        train_start = 0
        if self.max_train_size and train_end > self.max_train_size:
            train_start = train_end - self.max_train_size
        
        # Create indices
        train_idx = X.index[train_start:train_end]
        test_idx = X.index[effective_test_start + self.embargo_days:]
        
        # Create validation split if requested
        if self.validation_size > 0:
            val_size = int(len(train_idx) * self.validation_size)
            val_start = len(train_idx) - val_size
            
            val_idx = train_idx[val_start:]
            train_idx = train_idx[:val_start - self.gap_days]  # Apply gap before validation too
            
            return {
                'train': (train_idx, train_idx),
                'validation': (val_idx, val_idx),
                'test': (test_idx, test_idx)
            }
        else:
            return {
                'train': (train_idx, train_idx),
                'test': (test_idx, test_idx)
            }
    
    def _walk_forward_split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, List[Tuple[pd.Index, pd.Index]]]:
        """Walk-forward analysis splitting"""
        
        splits = []
        total_size = len(X)
        
        if self.test_size < 1:
            test_size = int(total_size * self.test_size)
        else:
            test_size = int(self.test_size)
        
        # Calculate step size (could be configurable)
        step_size = max(1, test_size // 4)  # Move forward by 1/4 of test size
        
        current_pos = self.min_train_size
        split_count = 0
        
        while current_pos + test_size + self.embargo_days <= total_size:
            # Training set
            train_start = max(0, current_pos - self.max_train_size) if self.max_train_size else 0
            train_end = current_pos
            
            # Apply gap
            test_start = current_pos + self.gap_days + self.embargo_days
            test_end = min(test_start + test_size, total_size)
            
            # Ensure we have enough test data
            if test_end - test_start < test_size // 2:
                break
            
            train_idx = X.index[train_start:train_end]
            test_idx = X.index[test_start:test_end]
            
            splits.append((train_idx, test_idx))
            split_count += 1
            
            # Move forward
            current_pos += step_size
            
            # Limit number of splits to prevent excessive computation
            if split_count >= 20:
                logger.warning("Limiting walk-forward splits to 20 to prevent excessive computation")
                break
        
        if not splits:
            raise DataValidationError("Could not create any walk-forward splits with given parameters")
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return {'walk_forward_splits': splits}
    
    def _purged_split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Tuple[pd.Index, pd.Index]]:
        """Purged time series split to prevent data leakage"""
        
        # This is similar to time_series_split but with additional purging
        # Purging removes samples whose labels are determined after samples in test set
        
        total_size = len(X)
        
        if self.test_size < 1:
            test_size = int(total_size * self.test_size)
        else:
            test_size = int(self.test_size)
        
        # Test set at the end
        test_start = total_size - test_size
        test_idx = X.index[test_start:]
        
        # Purge period (samples to remove before test set)
        purge_period = max(self.gap_days, self.embargo_days, 5)  # At least 5 days
        purged_end = test_start - purge_period
        
        # Training set
        train_start = 0
        if self.max_train_size and purged_end > self.max_train_size:
            train_start = purged_end - self.max_train_size
        
        train_idx = X.index[train_start:purged_end]
        
        if len(train_idx) < self.min_train_size:
            raise DataValidationError(f"Insufficient training data after purging: {len(train_idx)} < {self.min_train_size}")
        
        # Create validation split if requested
        if self.validation_size > 0:
            val_size = int(len(train_idx) * self.validation_size)
            val_start = len(train_idx) - val_size - purge_period
            
            val_idx = train_idx[val_start:val_start + val_size]
            train_idx = train_idx[:val_start]
            
            return {
                'train': (train_idx, train_idx),
                'validation': (val_idx, val_idx),
                'test': (test_idx, test_idx)
            }
        else:
            return {
                'train': (train_idx, train_idx),
                'test': (test_idx, test_idx)
            }
    
    def _blocking_split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, List[Tuple[pd.Index, pd.Index]]]:
        """Blocking time series split for cross-validation"""
        
        # Divide data into blocks and use some for training, others for testing
        total_size = len(X)
        n_blocks = 10  # Number of blocks
        
        block_size = total_size // n_blocks
        if block_size < self.min_train_size // 5:  # Each block should have reasonable size
            n_blocks = max(3, total_size // (self.min_train_size // 5))
            block_size = total_size // n_blocks
        
        splits = []
        
        # Use sliding window of blocks
        for i in range(n_blocks - 2):  # Leave at least 2 blocks for testing
            # Training blocks (first i+1 blocks)
            train_start = 0
            train_end = (i + 1) * block_size
            
            # Skip gap blocks
            gap_blocks = max(1, self.gap_days // block_size)
            
            # Test block
            test_start = train_end + gap_blocks * block_size
            test_end = min(test_start + block_size, total_size)
            
            if test_start >= total_size:
                break
            
            train_idx = X.index[train_start:train_end]
            test_idx = X.index[test_start:test_end]
            
            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        if not splits:
            raise DataValidationError("Could not create any blocking splits with given parameters")
        
        logger.info(f"Created {len(splits)} blocking splits")
        return {'blocking_splits': splits}
    
    def _expanding_window_split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, List[Tuple[pd.Index, pd.Index]]]:
        """Expanding window split for time series"""
        
        splits = []
        total_size = len(X)
        
        if self.test_size < 1:
            test_size = int(total_size * self.test_size)
        else:
            test_size = int(self.test_size)
        
        # Start with minimum training size and expand
        for train_end in range(self.min_train_size, total_size - test_size, test_size // 4):
            
            train_idx = X.index[:train_end]
            
            # Apply gap
            test_start = train_end + self.gap_days + self.embargo_days
            test_end = min(test_start + test_size, total_size)
            
            if test_end - test_start < test_size // 2:
                break
            
            test_idx = X.index[test_start:test_end]
            splits.append((train_idx, test_idx))
        
        if not splits:
            raise DataValidationError("Could not create any expanding window splits")
        
        logger.info(f"Created {len(splits)} expanding window splits")
        return {'expanding_splits': splits}
    
    def get_split_info(self, splits: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about the splits"""
        info = {
            'strategy': self.strategy,
            'total_splits': 0,
            'split_details': {}
        }
        
        for split_name, split_data in splits.items():
            if isinstance(split_data, list):
                # Multiple splits (walk-forward, blocking, etc.)
                info['total_splits'] += len(split_data)
                
                train_sizes = [len(train_idx) for train_idx, test_idx in split_data]
                test_sizes = [len(test_idx) for train_idx, test_idx in split_data]
                
                info['split_details'][split_name] = {
                    'count': len(split_data),
                    'train_size_range': (min(train_sizes), max(train_sizes)),
                    'test_size_range': (min(test_sizes), max(test_sizes)),
                    'avg_train_size': np.mean(train_sizes),
                    'avg_test_size': np.mean(test_sizes)
                }
            else:
                # Single split
                train_idx, test_idx = split_data
                info['total_splits'] += 1
                
                info['split_details'][split_name] = {
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'train_start': train_idx[0].isoformat() if len(train_idx) > 0 else None,
                    'train_end': train_idx[-1].isoformat() if len(train_idx) > 0 else None,
                    'test_start': test_idx[0].isoformat() if len(test_idx) > 0 else None,
                    'test_end': test_idx[-1].isoformat() if len(test_idx) > 0 else None
                }
        
        return info

# ============================================
# Market Regime Aware Splitting
# ============================================

class MarketRegimeAwareSplitter(FinancialTimeSeriesSplitter):
    """
    Market regime-aware splitting that considers market conditions
    
    Features:
    - Bull/bear market detection
    - Volatility regime awareness
    - Crisis period handling
    - Balanced regime representation
    """
    
    def __init__(self, regime_column: str = 'market_regime', 
                 ensure_regime_balance: bool = True, **kwargs):
        """
        Initialize market regime aware splitter
        
        Args:
            regime_column: Column name containing regime labels
            ensure_regime_balance: Whether to ensure balanced regime representation
            **kwargs: Arguments for parent class
        """
        super().__init__(**kwargs)
        self.regime_column = regime_column
        self.ensure_regime_balance = ensure_regime_balance
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
              groups: Optional[pd.Series] = None) -> Dict[str, Tuple[pd.Index, pd.Index]]:
        """Split data with market regime awareness"""
        
        if self.regime_column not in X.columns:
            logger.warning(f"Regime column '{self.regime_column}' not found, falling back to standard splitting")
            return super().split(X, y, groups)
        
        # Get base splits
        base_splits = super().split(X, y, groups)
        
        if not self.ensure_regime_balance:
            return base_splits
        
        # Adjust splits to ensure regime balance
        return self._balance_regime_representation(X, base_splits)
    
    def _balance_regime_representation(self, X: pd.DataFrame, 
                                     splits: Dict[str, Tuple[pd.Index, pd.Index]]) -> Dict[str, Tuple[pd.Index, pd.Index]]:
        """Balance regime representation in splits"""
        
        balanced_splits = {}
        
        for split_name, (train_idx, test_idx) in splits.items():
            if isinstance(train_idx, list):
                # Multiple splits - skip balancing for now
                balanced_splits[split_name] = (train_idx, test_idx)
                continue
            
            # Get regime distribution in full dataset
            full_regime_dist = X[self.regime_column].value_counts(normalize=True)
            
            # Check train set regime balance
            train_regime_dist = X.loc[train_idx, self.regime_column].value_counts(normalize=True)
            
            # If severely imbalanced, try to rebalance
            max_deviation = max(abs(full_regime_dist - train_regime_dist).fillna(1))
            
            if max_deviation > 0.2:  # More than 20% deviation
                logger.info(f"Rebalancing regime representation in {split_name} split")
                
                # Simple rebalancing: extend training period if possible
                extended_train_idx = self._extend_for_balance(X, train_idx, full_regime_dist)
                balanced_splits[split_name] = (extended_train_idx, test_idx)
            else:
                balanced_splits[split_name] = (train_idx, test_idx)
        
        return balanced_splits
    
    def _extend_for_balance(self, X: pd.DataFrame, train_idx: pd.Index, 
                           target_distribution: pd.Series) -> pd.Index:
        """Extend training set to achieve better regime balance"""
        
        # This is a simplified implementation
        # In practice, you might want more sophisticated balancing
        
        current_dist = X.loc[train_idx, self.regime_column].value_counts(normalize=True)
        
        # Find underrepresented regimes
        underrepresented = target_distribution - current_dist.fillna(0)
        underrepresented = underrepresented[underrepresented > 0.1]  # Significant underrepresentation
        
        if underrepresented.empty:
            return train_idx
        
        # Try to extend training period to include more of underrepresented regimes
        # Look backwards from training start
        train_start_pos = X.index.get_loc(train_idx[0])
        extension_start = max(0, train_start_pos - 252)  # Look back up to 1 year
        
        extension_idx = X.index[extension_start:train_start_pos]
        combined_idx = extension_idx.union(train_idx)
        
        return combined_idx

# ============================================
# Cross-Validation Splitters
# ============================================

class TimeSeriesCrossValidator:
    """
    Time series cross-validation with multiple strategies
    
    Features:
    - Multiple CV strategies
    - Financial calendar awareness
    - Purging and embargo
    - Performance tracking
    """
    
    def __init__(self, n_splits: int = 5, strategy: str = 'time_series',
                 gap_days: int = 0, embargo_days: int = 0,
                 min_train_size: int = 252):
        """
        Initialize time series cross-validator
        
        Args:
            n_splits: Number of splits
            strategy: CV strategy ('time_series', 'sliding_window', 'expanding')
            gap_days: Gap between train and validation sets
            embargo_days: Embargo period
            min_train_size: Minimum training size
        """
        self.n_splits = n_splits
        self.strategy = strategy
        self.gap_days = gap_days
        self.embargo_days = embargo_days
        self.min_train_size = min_train_size
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Iterator[Tuple[pd.Index, pd.Index]]:
        """Generate train/validation splits"""
        
        if not isinstance(X.index, pd.DatetimeIndex):
            raise DataValidationError("Data must have DatetimeIndex")
        
        if self.strategy == 'time_series':
            yield from self._time_series_cv(X, y)
        elif self.strategy == 'sliding_window':
            yield from self._sliding_window_cv(X, y)
        elif self.strategy == 'expanding':
            yield from self._expanding_window_cv(X, y)
        else:
            raise InvalidParameterError(f"Unknown CV strategy: {self.strategy}")
    
    def _time_series_cv(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Iterator[Tuple[pd.Index, pd.Index]]:
        """Time series cross-validation"""
        
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Test set
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                test_end = n_samples
            
            # Training set (all data before test set minus gap and embargo)
            train_end = test_start - self.gap_days - self.embargo_days
            
            if train_end < self.min_train_size:
                continue
            
            train_start = max(0, train_end - self.min_train_size * 2)  # Limit training size
            
            train_idx = X.index[train_start:train_end]
            test_idx = X.index[test_start:test_end]
            
            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                yield train_idx, test_idx
    
    def _sliding_window_cv(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Iterator[Tuple[pd.Index, pd.Index]]:
        """Sliding window cross-validation"""
        
        n_samples = len(X)
        window_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Sliding training window
            train_start = i * (window_size // 2)  # 50% overlap
            train_end = train_start + window_size
            
            if train_end >= n_samples - window_size // 4:  # Leave space for test
                break
            
            # Test set after gap
            test_start = train_end + self.gap_days + self.embargo_days
            test_end = min(test_start + window_size // 4, n_samples)
            
            if test_start >= n_samples:
                break
            
            train_idx = X.index[train_start:train_end]
            test_idx = X.index[test_start:test_end]
            
            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                yield train_idx, test_idx
    
    def _expanding_window_cv(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Iterator[Tuple[pd.Index, pd.Index]]:
        """Expanding window cross-validation"""
        
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 2)  # Smaller test sets for expanding
        
        for i in range(self.n_splits):
            # Expanding training window
            train_start = 0
            train_end = self.min_train_size + i * test_size
            
            if train_end >= n_samples - test_size:
                break
            
            # Test set
            test_start = train_end + self.gap_days + self.embargo_days
            test_end = min(test_start + test_size, n_samples)
            
            if test_start >= n_samples:
                break
            
            train_idx = X.index[train_start:train_end]
            test_idx = X.index[test_start:test_end]
            
            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                yield train_idx, test_idx

# ============================================
# Data Leakage Prevention
# ============================================

class LeakagePreventionSplitter:
    """
    Advanced splitting with data leakage prevention
    
    Features:
    - Purging overlapping samples
    - Embargo periods
    - Group-based splitting
    - Label-based purging
    """
    
    def __init__(self, embargo_days: int = 1, purge_days: int = 0,
                 group_column: Optional[str] = None):
        """
        Initialize leakage prevention splitter
        
        Args:
            embargo_days: Days to embargo after training period
            purge_days: Days to purge before test period
            group_column: Column for group-based splitting
        """
        self.embargo_days = embargo_days
        self.purge_days = purge_days
        self.group_column = group_column
    
    def purged_group_time_series_split(self, X: pd.DataFrame, y: pd.Series,
                                      test_size: float = 0.2) -> Tuple[pd.Index, pd.Index]:
        """
        Purged group time series split
        
        Prevents data leakage by:
        1. Purging samples whose labels are determined after test samples
        2. Adding embargo period
        3. Ensuring group integrity
        """
        
        if not isinstance(X.index, pd.DatetimeIndex):
            raise DataValidationError("Data must have DatetimeIndex")
        
        n_samples = len(X)
        test_samples = int(n_samples * test_size)
        
        # Test set at the end
        test_start_idx = n_samples - test_samples
        test_idx = X.index[test_start_idx:]
        
        # Purge and embargo period
        purge_embargo_days = max(self.purge_days, self.embargo_days)
        purge_end_idx = test_start_idx - purge_embargo_days
        
        # Training set
        train_idx = X.index[:purge_end_idx]
        
        # Handle groups if specified
        if self.group_column and self.group_column in X.columns:
            train_idx, test_idx = self._handle_group_constraints(X, train_idx, test_idx)
        
        return train_idx, test_idx
    
    def _handle_group_constraints(self, X: pd.DataFrame, train_idx: pd.Index, 
                                test_idx: pd.Index) -> Tuple[pd.Index, pd.Index]:
        """Handle group constraints to prevent leakage"""
        
        # Get groups in test set
        test_groups = set(X.loc[test_idx, self.group_column].unique())
        
        # Remove samples from training set if they belong to test groups
        train_mask = ~X.loc[train_idx, self.group_column].isin(test_groups)
        clean_train_idx = train_idx[train_mask]
        
        logger.info(f"Group-based purging removed {len(train_idx) - len(clean_train_idx)} samples from training set")
        
        return clean_train_idx, test_idx

# ============================================
# Factory Functions
# ============================================

def create_financial_splitter(split_type: str = 'time_series', **kwargs) -> FinancialTimeSeriesSplitter:
    """
    Create pre-configured financial data splitter
    
    Args:
        split_type: Type of splitter ('basic', 'research', 'production', 'walk_forward')
        **kwargs: Additional arguments
        
    Returns:
        Configured splitter
    """
    
    if split_type == 'basic':
        config = {
            'strategy': 'time_series',
            'test_size': 0.2,
            'validation_size': 0.2,
            'gap_days': 1,
            'embargo_days': 0
        }
    elif split_type == 'research':
        config = {
            'strategy': 'purged',
            'test_size': 0.2,
            'validation_size': 0.2,
            'gap_days': 5,
            'embargo_days': 1,
            'min_train_size': 252
        }
    elif split_type == 'production':
        config = {
            'strategy': 'time_series',
            'test_size': 0.15,
            'validation_size': 0.15,
            'gap_days': 1,
            'embargo_days': 0,
            'min_train_size': 504,  # 2 years
            'max_train_size': 1260  # 5 years
        }
    elif split_type == 'walk_forward':
        config = {
            'strategy': 'walk_forward',
            'test_size': 0.1,
            'gap_days': 1,
            'embargo_days': 0,
            'min_train_size': 252
        }
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    
    # Override with provided kwargs
    config.update(kwargs)
    
    return FinancialTimeSeriesSplitter(**config)

def create_cv_splitter(cv_type: str = 'time_series', n_splits: int = 5, **kwargs) -> TimeSeriesCrossValidator:
    """
    Create cross-validation splitter
    
    Args:
        cv_type: Type of CV ('time_series', 'sliding_window', 'expanding')
        n_splits: Number of splits
        **kwargs: Additional arguments
        
    Returns:
        Configured CV splitter
    """
    
    base_config = {
        'n_splits': n_splits,
        'strategy': cv_type,
        'gap_days': 1,
        'embargo_days': 0,
        'min_train_size': 252
    }
    
    base_config.update(kwargs)
    
    return TimeSeriesCrossValidator(**base_config)

# ============================================
# Utility Functions
# ============================================

def validate_split_quality(X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: Optional[pd.Series] = None, 
                          y_test: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Validate the quality of a data split
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets (optional)
        y_test: Test targets (optional)
        
    Returns:
        Validation report
    """
    
    report = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_date_range': None,
        'test_date_range': None,
        'time_gap_days': None,
        'data_leakage_risk': 'unknown',
        'target_distribution': None,
        'warnings': []
    }
    
    # Date range analysis
    if isinstance(X_train.index, pd.DatetimeIndex) and isinstance(X_test.index, pd.DatetimeIndex):
        train_start = X_train.index.min()
        train_end = X_train.index.max()
        test_start = X_test.index.min()
        test_end = X_test.index.max()
        
        report['train_date_range'] = (train_start.isoformat(), train_end.isoformat())
        report['test_date_range'] = (test_start.isoformat(), test_end.isoformat())
        
        # Check for time gap
        if test_start > train_end:
            gap_days = (test_start - train_end).days
            report['time_gap_days'] = gap_days
            report['data_leakage_risk'] = 'low' if gap_days > 0 else 'high'
        else:
            report['data_leakage_risk'] = 'high'
            report['warnings'].append("Test period overlaps with training period - high data leakage risk")
    
    # Target distribution analysis
    if y_train is not None and y_test is not None:
        if y_train.dtype == 'object' or len(y_train.unique()) < 20:
            # Classification target
            train_dist = y_train.value_counts(normalize=True)
            test_dist = y_test.value_counts(normalize=True)
            
            report['target_distribution'] = {
                'train': train_dist.to_dict(),
                'test': test_dist.to_dict(),
                'distribution_shift': (train_dist - test_dist.reindex(train_dist.index, fill_value=0)).abs().mean()
            }
        else:
            # Regression target
            report['target_distribution'] = {
                'train_mean': float(y_train.mean()),
                'test_mean': float(y_test.mean()),
                'train_std': float(y_train.std()),
                'test_std': float(y_test.std())
            }
    
    # Size validation
    if len(X_train) < 100:
        report['warnings'].append("Small training set size - may lead to poor model performance")
    
    if len(X_test) < 20:
        report['warnings'].append("Small test set size - may lead to unreliable evaluation")
    
    # Ratio validation
    total_size = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total_size
    
    if test_ratio < 0.1:
        report['warnings'].append("Very small test set ratio - may not be representative")
    elif test_ratio > 0.5:
        report['warnings'].append("Large test set ratio - may not leave enough training data")
    
    return report

def quick_time_series_split(X: pd.DataFrame, y: Optional[pd.Series] = None,
                           test_size: float = 0.2, gap_days: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """
    Quick time series split function
    
    Args:
        X: Feature data
        y: Target data (optional)
        test_size: Test set size
        gap_days: Gap between train and test
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    
    splitter = FinancialTimeSeriesSplitter(
        strategy='time_series',
        test_size=test_size,
        gap_days=gap_days,
        validation_size=0
    )
    
    splits = splitter.split(X, y)
    train_idx, test_idx = splits['train'][0], splits['test'][0]
    
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    
    if y is not None:
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test, None, None
