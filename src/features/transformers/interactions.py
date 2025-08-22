# ============================================
# StockPredictionPro - src/features/transformers/interactions.py
# Advanced feature interaction generators for financial machine learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
from itertools import combinations, product
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_array, check_is_fitted

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('features.transformers.interactions')

# ============================================
# Base Interaction Classes
# ============================================

@dataclass
class InteractionConfig:
    """Configuration for interaction transformers"""
    max_degree: int = 2
    include_bias: bool = False
    interaction_only: bool = False
    feature_names: Optional[List[str]] = None
    include_original: bool = True
    min_correlation: Optional[float] = None
    max_correlation: Optional[float] = None
    
    def __post_init__(self):
        if self.max_degree < 1:
            raise ValueError("max_degree must be at least 1")

class BaseInteractionTransformer(BaseEstimator, TransformerMixin):
    """Base class for all interaction transformers"""
    
    def __init__(self, config: Optional[InteractionConfig] = None):
        self.config = config or InteractionConfig()
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.interaction_names_ = None
        self.is_fitted_ = False
    
    def _validate_input(self, X):
        """Validate input data"""
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        if not self.is_fitted_:
            self.n_features_in_ = X.shape[1]
            if self.config.feature_names and len(self.config.feature_names) != X.shape[1]:
                raise ValueError("Length of feature_names must match number of features")
        else:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        
        return X
    
    def _get_feature_names(self):
        """Get feature names for input features"""
        if self.config.feature_names:
            return self.config.feature_names
        else:
            return [f'feature_{i}' for i in range(self.n_features_in_)]
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        check_is_fitted(self, 'is_fitted_')
        
        if input_features is None:
            input_features = self._get_feature_names()
        
        output_names = []
        if self.config.include_original:
            output_names.extend(input_features)
        
        if hasattr(self, 'interaction_names_') and self.interaction_names_:
            output_names.extend(self.interaction_names_)
        
        return output_names

# ============================================
# Polynomial Interactions
# ============================================

class PolynomialInteractionTransformer(BaseInteractionTransformer):
    """
    Generates polynomial feature interactions
    
    Creates polynomial features up to specified degree, with options to include
    only interaction terms or bias terms.
    """
    
    def __init__(self, config: Optional[InteractionConfig] = None):
        super().__init__(config)
        self.polynomial_features_ = None
    
    def fit(self, X, y=None):
        """Fit the polynomial interaction transformer"""
        X = self._validate_input(X)
        
        # Create sklearn PolynomialFeatures
        self.polynomial_features_ = PolynomialFeatures(
            degree=self.config.max_degree,
            include_bias=self.config.include_bias,
            interaction_only=self.config.interaction_only
        )
        
        # Fit the transformer
        self.polynomial_features_.fit(X)
        
        # Generate feature names for interactions
        input_names = self._get_feature_names()
        poly_names = self.polynomial_features_.get_feature_names_out(input_names)
        
        if self.config.include_original:
            # Keep original features
            self.interaction_names_ = poly_names[len(input_names):]
        else:
            # All polynomial features
            self.interaction_names_ = list(poly_names)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with polynomial interactions"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        # Generate polynomial features
        X_poly = self.polynomial_features_.transform(X)
        
        if self.config.include_original:
            # Combine original features with new interactions
            X_interactions = X_poly[:, self.n_features_in_:]
            return np.hstack([X, X_interactions])
        else:
            return X_poly
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

# ============================================
# Financial-Specific Interactions
# ============================================

class FinancialRatioTransformer(BaseInteractionTransformer):
    """
    Creates financial ratio interactions
    
    Generates ratios between features that make financial sense,
    such as price/volume, return/volatility, etc.
    """
    
    def __init__(self, 
                 ratio_pairs: Optional[List[Tuple[str, str]]] = None,
                 auto_detect_ratios: bool = True,
                 config: Optional[InteractionConfig] = None):
        super().__init__(config)
        self.ratio_pairs = ratio_pairs or []
        self.auto_detect_ratios = auto_detect_ratios
        self.computed_ratios_ = None
    
    def fit(self, X, y=None):
        """Fit the financial ratio transformer"""
        X = self._validate_input(X)
        
        feature_names = self._get_feature_names()
        self.computed_ratios_ = []
        
        # Use provided ratio pairs
        for num_name, den_name in self.ratio_pairs:
            if num_name in feature_names and den_name in feature_names:
                num_idx = feature_names.index(num_name)
                den_idx = feature_names.index(den_name)
                self.computed_ratios_.append((num_idx, den_idx, f"{num_name}_over_{den_name}"))
        
        # Auto-detect meaningful ratios
        if self.auto_detect_ratios:
            self._detect_financial_ratios(X, feature_names)
        
        # Generate interaction names
        self.interaction_names_ = [name for _, _, name in self.computed_ratios_]
        
        self.is_fitted_ = True
        return self
    
    def _detect_financial_ratios(self, X, feature_names):
        """Automatically detect meaningful financial ratios"""
        
        # Common financial ratio patterns
        price_keywords = ['price', 'close', 'high', 'low', 'open', 'vwap']
        volume_keywords = ['volume', 'vol']
        volatility_keywords = ['volatility', 'vol', 'std', 'atr']
        return_keywords = ['return', 'ret', 'change', 'pct']
        
        # Price/Volume ratios
        price_features = [i for i, name in enumerate(feature_names) 
                         if any(keyword in name.lower() for keyword in price_keywords)]
        volume_features = [i for i, name in enumerate(feature_names) 
                          if any(keyword in name.lower() for keyword in volume_keywords)]
        
        for p_idx in price_features:
            for v_idx in volume_features:
                if p_idx != v_idx:
                    ratio_name = f"{feature_names[p_idx]}_over_{feature_names[v_idx]}"
                    if (p_idx, v_idx, ratio_name) not in self.computed_ratios_:
                        self.computed_ratios_.append((p_idx, v_idx, ratio_name))
        
        # Return/Risk ratios (Sharpe-like)
        return_features = [i for i, name in enumerate(feature_names) 
                          if any(keyword in name.lower() for keyword in return_keywords)]
        risk_features = [i for i, name in enumerate(feature_names) 
                        if any(keyword in name.lower() for keyword in volatility_keywords)]
        
        for r_idx in return_features:
            for risk_idx in risk_features:
                if r_idx != risk_idx:
                    ratio_name = f"{feature_names[r_idx]}_over_{feature_names[risk_idx]}"
                    if (r_idx, risk_idx, ratio_name) not in self.computed_ratios_:
                        self.computed_ratios_.append((r_idx, risk_idx, ratio_name))
        
        logger.info(f"Auto-detected {len(self.computed_ratios_)} financial ratios")
    
    def transform(self, X):
        """Transform data with financial ratios"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        if not self.computed_ratios_:
            return X if self.config.include_original else np.array([]).reshape(X.shape[0], 0)
        
        # Calculate ratios
        ratios = []
        for num_idx, den_idx, _ in self.computed_ratios_:
            numerator = X[:, num_idx]
            denominator = X[:, den_idx]
            
            # Handle division by zero
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ratio = np.divide(numerator, denominator,
                                out=np.zeros_like(numerator),
                                where=denominator!=0)
            
            # Replace inf and -inf with reasonable values
            ratio = np.where(np.isinf(ratio), 0, ratio)
            ratios.append(ratio)
        
        X_ratios = np.column_stack(ratios)
        
        if self.config.include_original:
            return np.hstack([X, X_ratios])
        else:
            return X_ratios

# ============================================
# Correlation-Based Interactions
# ============================================

class CorrelationInteractionTransformer(BaseInteractionTransformer):
    """
    Creates interactions based on feature correlations
    
    Generates multiplicative interactions between features with
    correlations within specified thresholds.
    """
    
    def __init__(self, 
                 min_correlation: float = 0.3,
                 max_correlation: float = 0.9,
                 interaction_type: str = 'multiply',
                 config: Optional[InteractionConfig] = None):
        super().__init__(config)
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.interaction_type = interaction_type
        self.selected_pairs_ = None
        self.correlation_matrix_ = None
    
    def fit(self, X, y=None):
        """Fit the correlation-based interaction transformer"""
        X = self._validate_input(X)
        
        # Calculate correlation matrix
        X_df = pd.DataFrame(X)
        self.correlation_matrix_ = X_df.corr().abs()
        
        # Find feature pairs within correlation threshold
        self.selected_pairs_ = []
        feature_names = self._get_feature_names()
        
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                correlation = self.correlation_matrix_.iloc[i, j]
                
                if self.min_correlation <= correlation <= self.max_correlation:
                    interaction_name = f"{feature_names[i]}_{self.interaction_type}_{feature_names[j]}"
                    self.selected_pairs_.append((i, j, interaction_name, correlation))
        
        # Generate interaction names
        self.interaction_names_ = [name for _, _, name, _ in self.selected_pairs_]
        
        logger.info(f"Selected {len(self.selected_pairs_)} feature pairs based on correlation")
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with correlation-based interactions"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        if not self.selected_pairs_:
            return X if self.config.include_original else np.array([]).reshape(X.shape[0], 0)
        
        # Generate interactions
        interactions = []
        for i, j, _, _ in self.selected_pairs_:
            if self.interaction_type == 'multiply':
                interaction = X[:, i] * X[:, j]
            elif self.interaction_type == 'add':
                interaction = X[:, i] + X[:, j]
            elif self.interaction_type == 'subtract':
                interaction = X[:, i] - X[:, j]
            elif self.interaction_type == 'divide':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    interaction = np.divide(X[:, i], X[:, j],
                                          out=np.zeros_like(X[:, i]),
                                          where=X[:, j]!=0)
            else:
                raise ValueError(f"Unknown interaction type: {self.interaction_type}")
            
            interactions.append(interaction)
        
        X_interactions = np.column_stack(interactions)
        
        if self.config.include_original:
            return np.hstack([X, X_interactions])
        else:
            return X_interactions

# ============================================
# Binned Interactions
# ============================================

class BinnedInteractionTransformer(BaseInteractionTransformer):
    """
    Creates interactions with binned features
    
    Bins continuous features and creates interactions between
    binned and continuous features.
    """
    
    def __init__(self, 
                 n_bins: int = 5,
                 binning_strategy: str = 'quantile',
                 bin_features: Optional[List[int]] = None,
                 config: Optional[InteractionConfig] = None):
        super().__init__(config)
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy
        self.bin_features = bin_features
        self.bin_edges_ = None
        self.bin_feature_indices_ = None
    
    def fit(self, X, y=None):
        """Fit the binned interaction transformer"""
        X = self._validate_input(X)
        
        # Determine which features to bin
        if self.bin_features is None:
            # Bin all features
            self.bin_feature_indices_ = list(range(X.shape[1]))
        else:
            self.bin_feature_indices_ = self.bin_features
        
        # Calculate bin edges for each feature
        self.bin_edges_ = {}
        feature_names = self._get_feature_names()
        
        for feature_idx in self.bin_feature_indices_:
            feature_data = X[:, feature_idx]
            
            if self.binning_strategy == 'quantile':
                # Quantile-based binning
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                bin_edges = np.quantile(feature_data, quantiles)
            elif self.binning_strategy == 'uniform':
                # Uniform width binning
                bin_edges = np.linspace(feature_data.min(), feature_data.max(), self.n_bins + 1)
            else:
                raise ValueError(f"Unknown binning strategy: {self.binning_strategy}")
            
            # Ensure unique bin edges
            bin_edges = np.unique(bin_edges)
            self.bin_edges_[feature_idx] = bin_edges
        
        # Generate interaction names
        self.interaction_names_ = []
        for bin_idx in self.bin_feature_indices_:
            for cont_idx in range(X.shape[1]):
                if bin_idx != cont_idx:
                    for bin_num in range(len(self.bin_edges_[bin_idx]) - 1):
                        interaction_name = f"{feature_names[cont_idx]}_x_{feature_names[bin_idx]}_bin_{bin_num}"
                        self.interaction_names_.append(interaction_name)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with binned interactions"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        interactions = []
        
        for bin_idx in self.bin_feature_indices_:
            bin_edges = self.bin_edges_[bin_idx]
            
            # Create bin indicators
            binned_feature = np.digitize(X[:, bin_idx], bin_edges) - 1
            binned_feature = np.clip(binned_feature, 0, len(bin_edges) - 2)
            
            # Create interactions with all other features
            for cont_idx in range(X.shape[1]):
                if bin_idx != cont_idx:
                    continuous_feature = X[:, cont_idx]
                    
                    # Create interaction for each bin
                    for bin_num in range(len(bin_edges) - 1):
                        bin_mask = (binned_feature == bin_num)
                        interaction = continuous_feature * bin_mask.astype(float)
                        interactions.append(interaction)
        
        if not interactions:
            X_interactions = np.array([]).reshape(X.shape[0], 0)
        else:
            X_interactions = np.column_stack(interactions)
        
        if self.config.include_original:
            return np.hstack([X, X_interactions])
        else:
            return X_interactions

# ============================================
# Time-Series Specific Interactions
# ============================================

class TimeSeriesInteractionTransformer(BaseInteractionTransformer):
    """
    Creates time-series specific interactions
    
    Generates interactions between current values and lagged values,
    trend interactions, and seasonal interactions.
    """
    
    def __init__(self, 
                 max_lag: int = 5,
                 include_trend: bool = True,
                 include_seasonal: bool = False,
                 seasonal_period: int = 252,
                 config: Optional[InteractionConfig] = None):
        super().__init__(config)
        self.max_lag = max_lag
        self.include_trend = include_trend
        self.include_seasonal = include_seasonal
        self.seasonal_period = seasonal_period
        self.interaction_indices_ = None
    
    def fit(self, X, y=None):
        """Fit the time-series interaction transformer"""
        X = self._validate_input(X)
        
        feature_names = self._get_feature_names()
        self.interaction_indices_ = []
        self.interaction_names_ = []
        
        # Current x Lag interactions
        for feature_idx in range(X.shape[1]):
            for lag in range(1, min(self.max_lag + 1, X.shape[0])):
                interaction_name = f"{feature_names[feature_idx]}_x_lag_{lag}"
                self.interaction_indices_.append(('current_lag', feature_idx, lag))
                self.interaction_names_.append(interaction_name)
        
        # Trend interactions
        if self.include_trend:
            for feature_idx in range(X.shape[1]):
                interaction_name = f"{feature_names[feature_idx]}_x_trend"
                self.interaction_indices_.append(('trend', feature_idx, None))
                self.interaction_names_.append(interaction_name)
        
        # Seasonal interactions
        if self.include_seasonal and X.shape[0] >= self.seasonal_period:
            for feature_idx in range(X.shape[1]):
                interaction_name = f"{feature_names[feature_idx]}_x_seasonal"
                self.interaction_indices_.append(('seasonal', feature_idx, self.seasonal_period))
                self.interaction_names_.append(interaction_name)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with time-series interactions"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        interactions = []
        
        for interaction_type, feature_idx, param in self.interaction_indices_:
            if interaction_type == 'current_lag':
                # Current value x lagged value
                lag = param
                current_values = X[lag:, feature_idx]
                lagged_values = X[:-lag, feature_idx]
                
                # Pad with zeros to maintain original length
                interaction = np.zeros(X.shape[0])
                interaction[lag:] = current_values * lagged_values
                
            elif interaction_type == 'trend':
                # Current value x trend (time index)
                trend = np.arange(X.shape[0])
                interaction = X[:, feature_idx] * trend
                
            elif interaction_type == 'seasonal':
                # Current value x seasonal component
                seasonal_lag = param
                if X.shape[0] >= seasonal_lag:
                    current_values = X[seasonal_lag:, feature_idx]
                    seasonal_values = X[:-seasonal_lag, feature_idx]
                    
                    interaction = np.zeros(X.shape[0])
                    interaction[seasonal_lag:] = current_values * seasonal_values
                else:
                    interaction = np.zeros(X.shape[0])
            
            interactions.append(interaction)
        
        if not interactions:
            X_interactions = np.array([]).reshape(X.shape[0], 0)
        else:
            X_interactions = np.column_stack(interactions)
        
        if self.config.include_original:
            return np.hstack([X, X_interactions])
        else:
            return X_interactions

# ============================================
# Composite Interaction Transformer
# ============================================

class CompositeInteractionTransformer(BaseInteractionTransformer):
    """
    Combines multiple interaction transformers
    
    Applies multiple types of interactions and combines results.
    """
    
    def __init__(self, 
                 transformers: List[Tuple[str, BaseInteractionTransformer]],
                 config: Optional[InteractionConfig] = None):
        super().__init__(config)
        self.transformers = transformers
        self.fitted_transformers_ = []
    
    def fit(self, X, y=None):
        """Fit all component transformers"""
        X = self._validate_input(X)
        
        self.fitted_transformers_ = []
        all_interaction_names = []
        
        for name, transformer in self.transformers:
            # Fit transformer
            fitted_transformer = transformer.fit(X, y)
            self.fitted_transformers_.append((name, fitted_transformer))
            
            # Collect interaction names
            if hasattr(fitted_transformer, 'interaction_names_'):
                transformer_names = [f"{name}_{iname}" for iname in fitted_transformer.interaction_names_]
                all_interaction_names.extend(transformer_names)
        
        self.interaction_names_ = all_interaction_names
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data using all component transformers"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        all_interactions = []
        
        for name, transformer in self.fitted_transformers_:
            # Get interactions from this transformer
            transformer_interactions = transformer.transform(X)
            
            # Remove original features if they're included
            if transformer.config.include_original and transformer_interactions.shape[1] > X.shape[1]:
                interaction_part = transformer_interactions[:, X.shape[1]:]
                all_interactions.append(interaction_part)
            elif not transformer.config.include_original:
                all_interactions.append(transformer_interactions)
        
        if not all_interactions:
            X_interactions = np.array([]).reshape(X.shape[0], 0)
        else:
            X_interactions = np.hstack(all_interactions)
        
        if self.config.include_original:
            return np.hstack([X, X_interactions])
        else:
            return X_interactions

# ============================================
# Utility Functions
# ============================================

@time_it("interaction_generation")
def create_financial_interactions(X: Union[pd.DataFrame, np.ndarray],
                                feature_names: Optional[List[str]] = None,
                                include_polynomial: bool = True,
                                include_ratios: bool = True,
                                include_correlations: bool = True,
                                max_degree: int = 2) -> Tuple[np.ndarray, List[str]]:
    """
    Create comprehensive financial feature interactions
    
    Args:
        X: Input feature matrix
        feature_names: Names of input features
        include_polynomial: Whether to include polynomial interactions
        include_ratios: Whether to include financial ratios
        include_correlations: Whether to include correlation-based interactions
        max_degree: Maximum degree for polynomial features
        
    Returns:
        Tuple of (transformed_features, feature_names)
    """
    
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or X.columns.tolist()
        X = X.values
    
    config = InteractionConfig(
        max_degree=max_degree,
        feature_names=feature_names,
        include_original=True
    )
    
    transformers = []
    
    # Polynomial interactions
    if include_polynomial:
        poly_transformer = PolynomialInteractionTransformer(config)
        transformers.append(('polynomial', poly_transformer))
    
    # Financial ratios
    if include_ratios:
        ratio_transformer = FinancialRatioTransformer(config=config)
        transformers.append(('ratios', ratio_transformer))
    
    # Correlation-based interactions
    if include_correlations:
        corr_transformer = CorrelationInteractionTransformer(
            min_correlation=0.3,
            max_correlation=0.9,
            config=config
        )
        transformers.append(('correlations', corr_transformer))
    
    if not transformers:
        return X, feature_names or [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create composite transformer
    composite = CompositeInteractionTransformer(transformers, config)
    X_transformed = composite.fit_transform(X)
    output_names = composite.get_feature_names_out(feature_names)
    
    logger.info(f"Created {X_transformed.shape[1] - X.shape[1]} interaction features")
    return X_transformed, output_names

def analyze_interaction_importance(X: np.ndarray, 
                                 y: np.ndarray,
                                 interaction_names: List[str],
                                 method: str = 'mutual_info') -> pd.DataFrame:
    """
    Analyze importance of interaction features
    
    Args:
        X: Feature matrix with interactions
        y: Target variable
        interaction_names: Names of all features
        method: Method for importance calculation
        
    Returns:
        DataFrame with feature importance scores
    """
    
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    
    if method == 'mutual_info':
        # Determine if classification or regression
        if len(np.unique(y)) < 20:  # Heuristic for classification
            importance_scores = mutual_info_classif(X, y, random_state=42)
        else:
            importance_scores = mutual_info_regression(X, y, random_state=42)
    
    elif method == 'random_forest':
        if len(np.unique(y)) < 20:  # Classification
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Regression
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rf.fit(X, y)
        importance_scores = rf.feature_importances_
    
    else:
        raise ValueError(f"Unknown importance method: {method}")
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature_name': interaction_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    return importance_df

def filter_interactions_by_importance(X: np.ndarray,
                                    y: np.ndarray,
                                    feature_names: List[str],
                                    top_k: Optional[int] = None,
                                    threshold: Optional[float] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Filter interactions by importance scores
    
    Args:
        X: Feature matrix with interactions
        y: Target variable
        feature_names: Names of features
        top_k: Keep top k features
        threshold: Keep features above threshold
        
    Returns:
        Filtered features and names
    """
    
    importance_df = analyze_interaction_importance(X, y, feature_names)
    
    if top_k is not None:
        selected_features = importance_df.head(top_k)
    elif threshold is not None:
        selected_features = importance_df[importance_df['importance'] >= threshold]
    else:
        raise ValueError("Must specify either top_k or threshold")
    
    # Get selected feature indices
    selected_indices = [feature_names.index(name) for name in selected_features['feature_name']]
    
    X_filtered = X[:, selected_indices]
    filtered_names = selected_features['feature_name'].tolist()
    
    logger.info(f"Filtered to {len(selected_indices)} features from {len(feature_names)}")
    return X_filtered, filtered_names

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Interaction Transformers")
    
    # Create sample financial data
    np.random.seed(42)
    n_samples, n_features = 1000, 6
    
    # Generate correlated features mimicking financial data
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.cumprod(1 + returns)
    volume = np.random.lognormal(10, 0.5, n_samples)
    volatility = np.abs(np.random.normal(0.2, 0.05, n_samples))
    
    X = np.column_stack([
        prices,
        returns,
        volume,
        volatility,
        prices * volume,  # Price-volume interaction
        returns / volatility  # Sharpe-like ratio
    ])
    
    feature_names = ['price', 'returns', 'volume', 'volatility', 'price_vol', 'risk_adj_return']
    y = returns * 100 + np.random.normal(0, 0.1, n_samples)  # Target
    
    print(f"Original data shape: {X.shape}")
    
    # Test polynomial interactions
    print("\n1. Testing Polynomial Interactions")
    poly_config = InteractionConfig(max_degree=2, interaction_only=True, feature_names=feature_names)
    poly_transformer = PolynomialInteractionTransformer(poly_config)
    X_poly = poly_transformer.fit_transform(X)
    poly_names = poly_transformer.get_feature_names_out()
    print(f"After polynomial interactions: {X_poly.shape}")
    print(f"New features: {len(poly_names) - len(feature_names)}")
    
    # Test financial ratios
    print("\n2. Testing Financial Ratios")
    ratio_transformer = FinancialRatioTransformer(
        auto_detect_ratios=True,
        config=InteractionConfig(feature_names=feature_names)
    )
    X_ratios = ratio_transformer.fit_transform(X)
    ratio_names = ratio_transformer.get_feature_names_out()
    print(f"After ratio interactions: {X_ratios.shape}")
    print(f"Detected ratios: {len(ratio_transformer.computed_ratios_)}")
    
    # Test correlation interactions
    print("\n3. Testing Correlation Interactions")
    corr_transformer = CorrelationInteractionTransformer(
        min_correlation=0.1,
        max_correlation=0.8,
        config=InteractionConfig(feature_names=feature_names)
    )
    X_corr = corr_transformer.fit_transform(X)
    corr_names = corr_transformer.get_feature_names_out()
    print(f"After correlation interactions: {X_corr.shape}")
    print(f"Selected pairs: {len(corr_transformer.selected_pairs_)}")
    
    # Test comprehensive interactions
    print("\n4. Testing Comprehensive Interactions")
    X_comprehensive, comprehensive_names = create_financial_interactions(
        X, feature_names, 
        include_polynomial=True,
        include_ratios=True,
        include_correlations=True,
        max_degree=2
    )
    print(f"Comprehensive interactions shape: {X_comprehensive.shape}")
    print(f"Total features created: {len(comprehensive_names) - len(feature_names)}")
    
    # Test importance analysis
    print("\n5. Testing Importance Analysis")
    importance_df = analyze_interaction_importance(X_comprehensive, y, comprehensive_names)
    print(f"Top 5 most important features:")
    print(importance_df.head())
    
    # Test filtering by importance
    print("\n6. Testing Importance Filtering")
    X_filtered, filtered_names = filter_interactions_by_importance(
        X_comprehensive, y, comprehensive_names, top_k=15
    )
    print(f"Filtered to top 15 features: {X_filtered.shape}")
    print(f"Selected features: {filtered_names[:5]}...")  # Show first 5
    
    print("\nInteraction transformers testing completed successfully!")
