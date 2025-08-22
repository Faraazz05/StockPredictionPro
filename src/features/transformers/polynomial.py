# ============================================
# StockPredictionPro - src/features/transformers/polynomial.py
# Advanced polynomial feature transformations for financial machine learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from itertools import combinations_with_replacement, combinations
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.special import comb
from scipy.stats import pearsonr

from ...utils.exceptions import ValidationError, CalculationError
from ...utils.logger import get_logger
from ...utils.timing import time_it

logger = get_logger('features.transformers.polynomial')

# ============================================
# Configuration and Base Classes
# ============================================

@dataclass
class PolynomialConfig:
    """Configuration for polynomial transformers"""
    degree: int = 2
    include_bias: bool = False
    interaction_only: bool = False
    include_original: bool = True
    feature_names: Optional[List[str]] = None
    regularization_alpha: float = 0.0
    max_features: Optional[int] = None
    min_correlation_threshold: Optional[float] = None
    exclude_features: Optional[List[int]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.degree < 1:
            raise ValueError("degree must be at least 1")
        if self.regularization_alpha < 0:
            raise ValueError("regularization_alpha must be non-negative")

class BasePolynomialTransformer(BaseEstimator, TransformerMixin):
    """Base class for polynomial transformers"""
    
    def __init__(self, config: Optional[PolynomialConfig] = None):
        self.config = config or PolynomialConfig()
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.polynomial_names_ = None
        self.selected_features_ = None
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
            return [f'x{i}' for i in range(self.n_features_in_)]
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        check_is_fitted(self, 'is_fitted_')
        
        if input_features is None:
            input_features = self._get_feature_names()
        
        output_names = []
        if self.config.include_original:
            output_names.extend(input_features)
        
        if hasattr(self, 'polynomial_names_') and self.polynomial_names_:
            output_names.extend(self.polynomial_names_)
        
        return output_names

# ============================================
# Standard Polynomial Features
# ============================================

class StandardPolynomialTransformer(BasePolynomialTransformer):
    """
    Standard polynomial feature transformer using sklearn's PolynomialFeatures
    with additional financial-specific enhancements.
    """
    
    def __init__(self, config: Optional[PolynomialConfig] = None):
        super().__init__(config)
        self.sklearn_poly_ = None
        self.feature_importance_scores_ = None
    
    def fit(self, X, y=None):
        """Fit the polynomial transformer"""
        X = self._validate_input(X)
        
        # Create sklearn PolynomialFeatures
        self.sklearn_poly_ = PolynomialFeatures(
            degree=self.config.degree,
            include_bias=self.config.include_bias,
            interaction_only=self.config.interaction_only
        )
        
        # Fit the transformer
        self.sklearn_poly_.fit(X)
        
        # Get feature names
        input_names = self._get_feature_names()
        all_names = self.sklearn_poly_.get_feature_names_out(input_names)
        
        if self.config.include_original:
            # Polynomial names exclude original features
            self.polynomial_names_ = list(all_names[len(input_names):])
        else:
            # All polynomial features
            self.polynomial_names_ = list(all_names)
        
        # Feature selection based on correlation (if y is provided)
        if y is not None and self.config.min_correlation_threshold is not None:
            self._select_features_by_correlation(X, y)
        
        # Limit number of features if specified
        if self.config.max_features is not None and len(self.polynomial_names_) > self.config.max_features:
            self._select_top_features(X, y)
        
        self.is_fitted_ = True
        return self
    
    def _select_features_by_correlation(self, X, y):
        """Select polynomial features based on correlation with target"""
        # Generate all polynomial features
        X_poly_full = self.sklearn_poly_.transform(X)
        
        if self.config.include_original:
            polynomial_features = X_poly_full[:, self.n_features_in_:]
        else:
            polynomial_features = X_poly_full
        
        # Calculate correlations
        correlations = []
        for i in range(polynomial_features.shape[1]):
            try:
                corr, _ = pearsonr(polynomial_features[:, i], y)
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            except:
                correlations.append(0)
        
        # Select features above threshold
        selected_indices = [i for i, corr in enumerate(correlations) 
                          if corr >= self.config.min_correlation_threshold]
        
        self.selected_features_ = selected_indices
        self.polynomial_names_ = [self.polynomial_names_[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_indices)} polynomial features based on correlation")
    
    def _select_top_features(self, X, y):
        """Select top polynomial features based on importance"""
        if y is None:
            # Random selection if no target
            n_select = min(self.config.max_features, len(self.polynomial_names_))
            selected_indices = np.random.choice(len(self.polynomial_names_), n_select, replace=False)
        else:
            # Importance-based selection
            X_poly_full = self.sklearn_poly_.transform(X)
            
            if self.config.include_original:
                polynomial_features = X_poly_full[:, self.n_features_in_:]
            else:
                polynomial_features = X_poly_full
            
            # Use correlation as importance measure
            importances = []
            for i in range(polynomial_features.shape[1]):
                try:
                    corr, _ = pearsonr(polynomial_features[:, i], y)
                    importances.append(abs(corr) if not np.isnan(corr) else 0)
                except:
                    importances.append(0)
            
            # Select top features
            selected_indices = np.argsort(importances)[-self.config.max_features:]
        
        self.selected_features_ = selected_indices
        self.polynomial_names_ = [self.polynomial_names_[i] for i in selected_indices]
        self.feature_importance_scores_ = [importances[i] for i in selected_indices] if y is not None else None
        
        logger.info(f"Selected top {len(selected_indices)} polynomial features")
    
    def transform(self, X):
        """Transform data with polynomial features"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        # Generate polynomial features
        X_poly = self.sklearn_poly_.transform(X)
        
        if self.config.include_original:
            original_features = X
            polynomial_features = X_poly[:, self.n_features_in_:]
        else:
            original_features = None
            polynomial_features = X_poly
        
        # Apply feature selection if performed
        if self.selected_features_ is not None:
            polynomial_features = polynomial_features[:, self.selected_features_]
        
        # Combine original and polynomial features
        if self.config.include_original and original_features is not None:
            return np.hstack([original_features, polynomial_features])
        else:
            return polynomial_features

# ============================================
# Financial-Specific Polynomial Features
# ============================================

class FinancialPolynomialTransformer(BasePolynomialTransformer):
    """
    Financial-specific polynomial transformer that creates
    economically meaningful polynomial features.
    """
    
    def __init__(self, 
                 momentum_degree: int = 2,
                 volatility_degree: int = 2,
                 volume_degree: int = 2,
                 cross_terms: bool = True,
                 config: Optional[PolynomialConfig] = None):
        super().__init__(config)
        self.momentum_degree = momentum_degree
        self.volatility_degree = volatility_degree
        self.volume_degree = volume_degree
        self.cross_terms = cross_terms
        self.feature_categories_ = None
        self.polynomial_generators_ = None
    
    def fit(self, X, y=None):
        """Fit the financial polynomial transformer"""
        X = self._validate_input(X)
        
        # Categorize features based on names
        self.feature_categories_ = self._categorize_features()
        
        # Generate polynomial terms
        self.polynomial_generators_ = self._create_polynomial_generators()
        
        # Generate feature names
        self.polynomial_names_ = []
        for generator_name, _ in self.polynomial_generators_:
            self.polynomial_names_.append(generator_name)
        
        self.is_fitted_ = True
        return self
    
    def _categorize_features(self):
        """Categorize features into financial types"""
        feature_names = self._get_feature_names()
        categories = {
            'price': [],
            'return': [],
            'momentum': [],
            'volatility': [],
            'volume': [],
            'other': []
        }
        
        for i, name in enumerate(feature_names):
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in ['price', 'close', 'open', 'high', 'low']):
                categories['price'].append(i)
            elif any(keyword in name_lower for keyword in ['return', 'ret', 'change', 'pct']):
                categories['return'].append(i)
            elif any(keyword in name_lower for keyword in ['rsi', 'macd', 'momentum', 'roc']):
                categories['momentum'].append(i)
            elif any(keyword in name_lower for keyword in ['vol', 'atr', 'std', 'volatility']):
                categories['volatility'].append(i)
            elif any(keyword in name_lower for keyword in ['volume', 'vol']):
                categories['volume'].append(i)
            else:
                categories['other'].append(i)
        
        return categories
    
    def _create_polynomial_generators(self):
        """Create polynomial feature generators"""
        generators = []
        feature_names = self._get_feature_names()
        
        # Price polynomial features (usually degree 1-2)
        for feature_idx in self.feature_categories_['price']:
            for degree in range(2, min(3, self.config.degree + 1)):
                generator_name = f"{feature_names[feature_idx]}_power_{degree}"
                generators.append((generator_name, ('power', feature_idx, degree)))
        
        # Return polynomial features (higher degrees for momentum effects)
        for feature_idx in self.feature_categories_['return']:
            for degree in range(2, min(self.momentum_degree + 1, self.config.degree + 1)):
                generator_name = f"{feature_names[feature_idx]}_power_{degree}"
                generators.append((generator_name, ('power', feature_idx, degree)))
        
        # Momentum squared terms (momentum persistence)
        for feature_idx in self.feature_categories_['momentum']:
            for degree in range(2, min(self.momentum_degree + 1, self.config.degree + 1)):
                generator_name = f"{feature_names[feature_idx]}_power_{degree}"
                generators.append((generator_name, ('power', feature_idx, degree)))
        
        # Volatility terms (volatility clustering)
        for feature_idx in self.feature_categories_['volatility']:
            for degree in range(2, min(self.volatility_degree + 1, self.config.degree + 1)):
                generator_name = f"{feature_names[feature_idx]}_power_{degree}"
                generators.append((generator_name, ('power', feature_idx, degree)))
        
        # Volume polynomial terms
        for feature_idx in self.feature_categories_['volume']:
            for degree in range(2, min(self.volume_degree + 1, self.config.degree + 1)):
                generator_name = f"{feature_names[feature_idx]}_power_{degree}"
                generators.append((generator_name, ('power', feature_idx, degree)))
        
        # Cross-terms between categories
        if self.cross_terms:
            generators.extend(self._create_cross_terms())
        
        return generators
    
    def _create_cross_terms(self):
        """Create meaningful cross-terms between feature categories"""
        cross_generators = []
        feature_names = self._get_feature_names()
        
        # Return × Volatility interactions (risk-adjusted returns)
        for ret_idx in self.feature_categories_['return']:
            for vol_idx in self.feature_categories_['volatility']:
                generator_name = f"{feature_names[ret_idx]}_times_{feature_names[vol_idx]}"
                cross_generators.append((generator_name, ('multiply', ret_idx, vol_idx)))
        
        # Price × Volume interactions (dollar volume)
        for price_idx in self.feature_categories_['price']:
            for vol_idx in self.feature_categories_['volume']:
                generator_name = f"{feature_names[price_idx]}_times_{feature_names[vol_idx]}"
                cross_generators.append((generator_name, ('multiply', price_idx, vol_idx)))
        
        # Momentum × Volatility interactions
        for mom_idx in self.feature_categories_['momentum']:
            for vol_idx in self.feature_categories_['volatility']:
                generator_name = f"{feature_names[mom_idx]}_times_{feature_names[vol_idx]}"
                cross_generators.append((generator_name, ('multiply', mom_idx, vol_idx)))
        
        # Return squared (momentum effects)
        for ret_idx in self.feature_categories_['return']:
            generator_name = f"{feature_names[ret_idx]}_squared"
            cross_generators.append((generator_name, ('power', ret_idx, 2)))
        
        return cross_generators
    
    def transform(self, X):
        """Transform data with financial polynomial features"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        # Generate polynomial features
        polynomial_features = []
        
        for generator_name, (operation, *params) in self.polynomial_generators_:
            if operation == 'power':
                feature_idx, degree = params
                feature_values = X[:, feature_idx] ** degree
            elif operation == 'multiply':
                feature_idx1, feature_idx2 = params
                feature_values = X[:, feature_idx1] * X[:, feature_idx2]
            else:
                continue  # Skip unknown operations
            
            polynomial_features.append(feature_values)
        
        if not polynomial_features:
            polynomial_array = np.array([]).reshape(X.shape[0], 0)
        else:
            polynomial_array = np.column_stack(polynomial_features)
        
        # Combine with original features if requested
        if self.config.include_original:
            return np.hstack([X, polynomial_array])
        else:
            return polynomial_array

# ============================================
# Orthogonal Polynomial Features
# ============================================

class OrthogonalPolynomialTransformer(BasePolynomialTransformer):
    """
    Creates orthogonal polynomial features to reduce multicollinearity.
    Uses Gram-Schmidt orthogonalization process.
    """
    
    def __init__(self, config: Optional[PolynomialConfig] = None):
        super().__init__(config)
        self.orthogonal_basis_ = None
        self.mean_features_ = None
        self.std_features_ = None
    
    def fit(self, X, y=None):
        """Fit orthogonal polynomial transformer"""
        X = self._validate_input(X)
        
        # Standardize features
        self.mean_features_ = np.mean(X, axis=0)
        self.std_features_ = np.std(X, axis=0)
        self.std_features_[self.std_features_ == 0] = 1  # Avoid division by zero
        
        X_standardized = (X - self.mean_features_) / self.std_features_
        
        # Generate standard polynomial features
        standard_poly = PolynomialFeatures(
            degree=self.config.degree,
            include_bias=self.config.include_bias,
            interaction_only=self.config.interaction_only
        )
        
        X_poly = standard_poly.fit_transform(X_standardized)
        
        # Apply Gram-Schmidt orthogonalization
        self.orthogonal_basis_ = self._gram_schmidt_orthogonalization(X_poly)
        
        # Generate feature names
        input_names = self._get_feature_names()
        standard_names = standard_poly.get_feature_names_out(input_names)
        
        if self.config.include_original:
            self.polynomial_names_ = [f"orth_{name}" for name in standard_names[X.shape[1]:]]
        else:
            self.polynomial_names_ = [f"orth_{name}" for name in standard_names]
        
        self.is_fitted_ = True
        return self
    
    def _gram_schmidt_orthogonalization(self, X):
        """Apply Gram-Schmidt orthogonalization to polynomial features"""
        n_samples, n_features = X.shape
        orthogonal_basis = np.zeros_like(X)
        
        # First vector is normalized version of the first column
        orthogonal_basis[:, 0] = X[:, 0] / np.linalg.norm(X[:, 0])
        
        # Orthogonalize remaining vectors
        for i in range(1, n_features):
            # Start with the original vector
            vector = X[:, i].copy()
            
            # Subtract projections onto all previous orthogonal vectors
            for j in range(i):
                projection = np.dot(vector, orthogonal_basis[:, j])
                vector = vector - projection * orthogonal_basis[:, j]
            
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 1e-10:  # Avoid numerical issues
                orthogonal_basis[:, i] = vector / norm
            else:
                orthogonal_basis[:, i] = 0  # Linearly dependent vector
        
        return orthogonal_basis
    
    def transform(self, X):
        """Transform data with orthogonal polynomial features"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        # Standardize features
        X_standardized = (X - self.mean_features_) / self.std_features_
        
        # Generate standard polynomial features
        standard_poly = PolynomialFeatures(
            degree=self.config.degree,
            include_bias=self.config.include_bias,
            interaction_only=self.config.interaction_only
        )
        
        X_poly = standard_poly.fit_transform(X_standardized)
        
        # Project onto orthogonal basis
        X_orthogonal = X_poly @ self.orthogonal_basis_.T
        
        if self.config.include_original:
            return np.hstack([X, X_orthogonal[:, X.shape[1]:]])
        else:
            return X_orthogonal

# ============================================
# Regularized Polynomial Features
# ============================================

class RegularizedPolynomialTransformer(BasePolynomialTransformer):
    """
    Creates polynomial features with built-in regularization to
    prevent overfitting and reduce multicollinearity.
    """
    
    def __init__(self, 
                 regularization_type: str = 'ridge',
                 config: Optional[PolynomialConfig] = None):
        super().__init__(config)
        self.regularization_type = regularization_type
        self.regularization_matrix_ = None
        self.polynomial_features_ = None
    
    def fit(self, X, y=None):
        """Fit regularized polynomial transformer"""
        X = self._validate_input(X)
        
        # Generate polynomial features
        self.polynomial_features_ = PolynomialFeatures(
            degree=self.config.degree,
            include_bias=self.config.include_bias,
            interaction_only=self.config.interaction_only
        )
        
        X_poly = self.polynomial_features_.fit_transform(X)
        
        # Create regularization matrix
        if self.regularization_type == 'ridge':
            self.regularization_matrix_ = np.eye(X_poly.shape[1]) * self.config.regularization_alpha
        elif self.regularization_type == 'lasso':
            # For LASSO, we'll use feature selection approach
            self.regularization_matrix_ = None
        else:
            raise ValueError(f"Unknown regularization type: {self.regularization_type}")
        
        # Generate feature names
        input_names = self._get_feature_names()
        all_names = self.polynomial_features_.get_feature_names_out(input_names)
        
        if self.config.include_original:
            self.polynomial_names_ = [f"reg_{name}" for name in all_names[X.shape[1]:]]
        else:
            self.polynomial_names_ = [f"reg_{name}" for name in all_names]
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data with regularized polynomial features"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        # Generate polynomial features
        X_poly = self.polynomial_features_.transform(X)
        
        # Apply regularization (for Ridge, this is implicit in the matrix)
        # For actual regularization, this would be applied during model training
        
        if self.config.include_original:
            return np.hstack([X, X_poly[:, X.shape[1]:]])
        else:
            return X_poly

# ============================================
# Composite Polynomial Transformer
# ============================================

class CompositePolynomialTransformer(BasePolynomialTransformer):
    """
    Combines multiple polynomial transformers with different strategies.
    """
    
    def __init__(self, 
                 transformers: List[Tuple[str, BasePolynomialTransformer]],
                 config: Optional[PolynomialConfig] = None):
        super().__init__(config)
        self.transformers = transformers
        self.fitted_transformers_ = []
    
    def fit(self, X, y=None):
        """Fit all component transformers"""
        X = self._validate_input(X)
        
        self.fitted_transformers_ = []
        all_polynomial_names = []
        
        for name, transformer in self.transformers:
            # Fit transformer
            fitted_transformer = transformer.fit(X, y)
            self.fitted_transformers_.append((name, fitted_transformer))
            
            # Collect polynomial names
            if hasattr(fitted_transformer, 'polynomial_names_'):
                transformer_names = [f"{name}_{pname}" for pname in fitted_transformer.polynomial_names_]
                all_polynomial_names.extend(transformer_names)
        
        self.polynomial_names_ = all_polynomial_names
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform data using all component transformers"""
        check_is_fitted(self, 'is_fitted_')
        X = self._validate_input(X)
        
        all_polynomials = []
        
        for name, transformer in self.fitted_transformers_:
            # Get polynomials from this transformer
            transformer_output = transformer.transform(X)
            
            # Extract only polynomial features (exclude original if included)
            if transformer.config.include_original and transformer_output.shape[1] > X.shape[1]:
                polynomial_part = transformer_output[:, X.shape[1]:]
                all_polynomials.append(polynomial_part)
            elif not transformer.config.include_original:
                all_polynomials.append(transformer_output)
        
        if not all_polynomials:
            X_polynomials = np.array([]).reshape(X.shape[0], 0)
        else:
            X_polynomials = np.hstack(all_polynomials)
        
        if self.config.include_original:
            return np.hstack([X, X_polynomials])
        else:
            return X_polynomials

# ============================================
# Utility Functions
# ============================================

@time_it("polynomial_generation")
def create_financial_polynomials(X: Union[pd.DataFrame, np.ndarray],
                                feature_names: Optional[List[str]] = None,
                                degree: int = 2,
                                include_financial: bool = True,
                                include_orthogonal: bool = False,
                                max_features: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Create comprehensive polynomial features for financial data
    
    Args:
        X: Input feature matrix
        feature_names: Names of input features
        degree: Maximum polynomial degree
        include_financial: Whether to include financial-specific polynomials
        include_orthogonal: Whether to include orthogonal polynomials
        max_features: Maximum number of polynomial features to create
        
    Returns:
        Tuple of (transformed_features, feature_names)
    """
    
    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or X.columns.tolist()
        X = X.values
    
    config = PolynomialConfig(
        degree=degree,
        feature_names=feature_names,
        include_original=True,
        max_features=max_features
    )
    
    transformers = []
    
    # Standard polynomial features
    standard_transformer = StandardPolynomialTransformer(config)
    transformers.append(('standard', standard_transformer))
    
    # Financial-specific polynomials
    if include_financial:
        financial_transformer = FinancialPolynomialTransformer(config=config)
        transformers.append(('financial', financial_transformer))
    
    # Orthogonal polynomials
    if include_orthogonal:
        orthogonal_transformer = OrthogonalPolynomialTransformer(config)
        transformers.append(('orthogonal', orthogonal_transformer))
    
    # Create composite transformer
    composite = CompositePolynomialTransformer(transformers, config)
    X_transformed = composite.fit_transform(X)
    output_names = composite.get_feature_names_out(feature_names)
    
    logger.info(f"Created {X_transformed.shape[1] - X.shape[1]} polynomial features")
    return X_transformed, output_names

def analyze_polynomial_importance(X: np.ndarray, 
                                y: np.ndarray,
                                polynomial_names: List[str]) -> pd.DataFrame:
    """Analyze importance of polynomial features"""
    
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    
    # Determine if classification or regression
    if len(np.unique(y)) < 20:  # Heuristic for classification
        importance_scores = mutual_info_classif(X, y, random_state=42)
    else:
        importance_scores = mutual_info_regression(X, y, random_state=42)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature_name': polynomial_names,
        'importance': importance_scores,
        'feature_type': ['original' if not any(keyword in name for keyword in ['power', 'times', 'orth', 'reg']) 
                        else 'polynomial' for name in polynomial_names]
    }).sort_values('importance', ascending=False)
    
    return importance_df

def detect_multicollinearity(X: np.ndarray, 
                           feature_names: List[str],
                           threshold: float = 0.9) -> pd.DataFrame:
    """Detect multicollinearity in polynomial features"""
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    n_features = len(feature_names)
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            correlation = abs(corr_matrix[i, j])
            if correlation > threshold:
                high_corr_pairs.append({
                    'feature_1': feature_names[i],
                    'feature_2': feature_names[j],
                    'correlation': correlation
                })
    
    multicollinearity_df = pd.DataFrame(high_corr_pairs)
    if not multicollinearity_df.empty:
        multicollinearity_df = multicollinearity_df.sort_values('correlation', ascending=False)
    
    return multicollinearity_df

def optimize_polynomial_features(X: np.ndarray,
                                y: np.ndarray,
                                feature_names: List[str],
                                max_features: int = 50,
                                correlation_threshold: float = 0.9) -> Tuple[np.ndarray, List[str]]:
    """
    Optimize polynomial features by removing multicollinear features
    and selecting the most important ones.
    """
    
    # Detect multicollinearity
    multicollinear_df = detect_multicollinearity(X, feature_names, correlation_threshold)
    
    # Remove multicollinear features (keep first occurrence)
    features_to_remove = set()
    for _, row in multicollinear_df.iterrows():
        # Remove the feature with lower correlation to target
        feature_1_idx = feature_names.index(row['feature_1'])
        feature_2_idx = feature_names.index(row['feature_2'])
        
        corr_1 = abs(np.corrcoef(X[:, feature_1_idx], y)[0, 1])
        corr_2 = abs(np.corrcoef(X[:, feature_2_idx], y)[0, 1])
        
        if corr_1 < corr_2:
            features_to_remove.add(feature_1_idx)
        else:
            features_to_remove.add(feature_2_idx)
    
    # Keep features not marked for removal
    keep_indices = [i for i in range(len(feature_names)) if i not in features_to_remove]
    X_filtered = X[:, keep_indices]
    filtered_names = [feature_names[i] for i in keep_indices]
    
    # Select top features if still too many
    if len(filtered_names) > max_features:
        importance_df = analyze_polynomial_importance(X_filtered, y, filtered_names)
        top_features = importance_df.head(max_features)
        
        final_indices = [filtered_names.index(name) for name in top_features['feature_name']]
        X_final = X_filtered[:, final_indices]
        final_names = top_features['feature_name'].tolist()
    else:
        X_final = X_filtered
        final_names = filtered_names
    
    logger.info(f"Optimized from {len(feature_names)} to {len(final_names)} polynomial features")
    return X_final, final_names

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Polynomial Transformers")
    
    # Create sample financial data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic financial features
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.cumprod(1 + returns)
    volume = np.random.lognormal(10, 0.5, n_samples)
    volatility = np.abs(np.random.normal(0.2, 0.05, n_samples))
    rsi = 50 + 30 * np.tanh(np.cumsum(returns) * 10)
    
    X = np.column_stack([prices, returns, volume, volatility, rsi])
    feature_names = ['price', 'returns', 'volume', 'volatility', 'rsi']
    y = returns * 100 + np.random.normal(0, 0.1, n_samples)
    
    print(f"Original data shape: {X.shape}")
    
    # Test standard polynomial transformer
    print("\n1. Testing Standard Polynomial Transformer")
    config = PolynomialConfig(degree=2, feature_names=feature_names, max_features=20)
    standard_transformer = StandardPolynomialTransformer(config)
    X_standard = standard_transformer.fit_transform(X, y)
    standard_names = standard_transformer.get_feature_names_out()
    print(f"Standard polynomial shape: {X_standard.shape}")
    print(f"Created {len(standard_transformer.polynomial_names_)} polynomial features")
    
    # Test financial polynomial transformer
    print("\n2. Testing Financial Polynomial Transformer")
    financial_transformer = FinancialPolynomialTransformer(
        momentum_degree=2,
        volatility_degree=2,
        cross_terms=True,
        config=config
    )
    X_financial = financial_transformer.fit_transform(X)
    financial_names = financial_transformer.get_feature_names_out()
    print(f"Financial polynomial shape: {X_financial.shape}")
    print(f"Created {len(financial_transformer.polynomial_names_)} financial polynomial features")
    
    # Test orthogonal polynomial transformer
    print("\n3. Testing Orthogonal Polynomial Transformer")
    orthogonal_transformer = OrthogonalPolynomialTransformer(config)
    X_orthogonal = orthogonal_transformer.fit_transform(X)
    orthogonal_names = orthogonal_transformer.get_feature_names_out()
    print(f"Orthogonal polynomial shape: {X_orthogonal.shape}")
    
    # Test comprehensive polynomial creation
    print("\n4. Testing Comprehensive Polynomial Creation")
    X_comprehensive, comprehensive_names = create_financial_polynomials(
        X, feature_names,
        degree=2,
        include_financial=True,
        include_orthogonal=True,
        max_features=50
    )
    print(f"Comprehensive polynomial shape: {X_comprehensive.shape}")
    print(f"Total features: {len(comprehensive_names)}")
    
    # Test importance analysis
    print("\n5. Testing Polynomial Importance Analysis")
    importance_df = analyze_polynomial_importance(X_comprehensive, y, comprehensive_names)
    print("Top 10 most important polynomial features:")
    print(importance_df.head(10)[['feature_name', 'importance', 'feature_type']])
    
    # Test multicollinearity detection
    print("\n6. Testing Multicollinearity Detection")
    multicollinear_df = detect_multicollinearity(X_comprehensive, comprehensive_names, threshold=0.8)
    if not multicollinear_df.empty:
        print(f"Found {len(multicollinear_df)} highly correlated feature pairs:")
        print(multicollinear_df.head())
    else:
        print("No highly correlated features found")
    
    # Test feature optimization
    print("\n7. Testing Feature Optimization")
    X_optimized, optimized_names = optimize_polynomial_features(
        X_comprehensive, y, comprehensive_names,
        max_features=30,
        correlation_threshold=0.9
    )
    print(f"Optimized polynomial shape: {X_optimized.shape}")
    print(f"Optimized to {len(optimized_names)} features")
    
    print("\nPolynomial transformers testing completed successfully!")
