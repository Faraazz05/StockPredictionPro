# ============================================
# StockPredictionPro - src/models/regression/ridge.py
# Ridge regression models for financial prediction with L2 regularization
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_squared_error, r2_score
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_regressor import BaseFinancialRegressor, RegressionStrategy

logger = get_logger('models.regression.ridge')

# ============================================
# Ridge Regression Model
# ============================================

class FinancialRidgeRegression(BaseFinancialRegressor):
    """
    Ridge regression model optimized for financial data with L2 regularization
    
    Features:
    - L2 regularization for multicollinearity handling
    - Cross-validation for optimal alpha selection
    - Regularization path analysis
    - Financial domain optimizations
    - Automatic feature scaling
    """
    
    def __init__(self,
                 name: str = "ridge_regression",
                 alpha: Union[float, List[float]] = 1.0,
                 fit_intercept: bool = True,
                 normalize: bool = False,
                 copy_X: bool = True,
                 max_iter: Optional[int] = None,
                 tol: float = 1e-3,
                 solver: str = 'auto',
                 positive: bool = False,
                 random_state: Optional[int] = 42,
                 auto_alpha: bool = True,
                 alpha_range: Optional[Tuple[float, float]] = None,
                 cv_folds: int = 5,
                 **kwargs):
        """
        Initialize Financial Ridge Regression
        
        Args:
            name: Model name
            alpha: Regularization strength (or list for CV)
            fit_intercept: Whether to calculate intercept
            normalize: Whether to normalize features (deprecated in sklearn)
            copy_X: Whether to copy X or overwrite
            max_iter: Maximum iterations for solver
            tol: Tolerance for solver
            solver: Solver to use ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga')
            positive: Whether to force positive coefficients
            random_state: Random state for reproducibility
            auto_alpha: Whether to automatically select best alpha via CV
            alpha_range: Range for alpha search (min_alpha, max_alpha)
            cv_folds: Number of cross-validation folds for alpha selection
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="ridge_regression",
            regression_strategy=RegressionStrategy.PRICE_PREDICTION,
            **kwargs
        )
        
        # Ridge regression parameters
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.positive = positive
        self.random_state = random_state
        self.auto_alpha = auto_alpha
        self.alpha_range = alpha_range or (0.001, 1000.0)
        self.cv_folds = cv_folds
        
        # Store parameters for model creation
        self.model_params.update({
            'alpha': alpha,
            'fit_intercept': fit_intercept,
            'copy_X': copy_X,
            'max_iter': max_iter,
            'tol': tol,
            'solver': solver,
            'positive': positive,
            'random_state': random_state
        })
        
        # Ridge-specific attributes
        self.scaler_: Optional[StandardScaler] = None
        self.best_alpha_: Optional[float] = None
        self.alpha_scores_: Optional[Dict[float, float]] = None
        self.regularization_path_: Optional[Dict[str, Any]] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        
        logger.info(f"Initialized Ridge regression: {self.name} (auto_alpha={self.auto_alpha})")
    
    def _create_model(self) -> Union[Ridge, RidgeCV]:
        """Create the Ridge regression model"""
        
        if self.auto_alpha:
            # Use RidgeCV for automatic alpha selection
            if isinstance(self.alpha, list):
                alphas = self.alpha
            else:
                # Generate alpha range for cross-validation
                alphas = np.logspace(
                    np.log10(self.alpha_range[0]), 
                    np.log10(self.alpha_range[1]), 
                    50
                )
            
            cv_params = {
                'alphas': alphas,
                'fit_intercept': self.fit_intercept,
                'normalize': self.normalize,
                'copy_X': self.copy_X,
                'cv': self.cv_folds,
                'scoring': None,  # Use default (negative MSE)
                'gcv_mode': None,
                'store_cv_values': True,
                'alpha_per_target': False
            }
            
            return RidgeCV(**cv_params)
        else:
            # Use regular Ridge with fixed alpha
            ridge_params = {k: v for k, v in self.model_params.items() 
                           if k not in ['random_state']}  # Ridge doesn't use random_state
            return Ridge(**ridge_params)
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with automatic scaling"""
        
        # Basic preprocessing
        X_processed = super()._preprocess_features(X)
        
        # Always scale features for Ridge regression
        if self.scaler_ is None:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_processed)
            logger.debug("Fitted feature scaler for Ridge regression")
        else:
            X_scaled = self.scaler_.transform(X_processed)
        
        return X_scaled
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Enhanced post-training processing for Ridge regression"""
        
        # Call parent processing
        super()._post_training_processing(X, y)
        
        # Extract Ridge-specific information
        if isinstance(self.model, RidgeCV):
            self.best_alpha_ = self.model.alpha_
            
            # Store CV scores for different alphas
            if hasattr(self.model, 'cv_values_'):
                cv_scores = np.mean(self.model.cv_values_, axis=0)
                self.alpha_scores_ = dict(zip(self.model.alphas, cv_scores))
            
            logger.info(f"Best alpha selected via CV: {self.best_alpha_:.6f}")
        else:
            self.best_alpha_ = self.model.alpha
        
        # Extract coefficients
        self.coefficients_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        # Set feature importances (absolute coefficients)
        self.feature_importances_ = np.abs(self.coefficients_)
        
        # Calculate regularization path if reasonable number of features
        if X.shape[1] <= 100:  # Avoid expensive computation for high-dimensional data
            self._calculate_regularization_path(X, y)
    
    def _calculate_regularization_path(self, X: np.ndarray, y: np.ndarray):
        """Calculate regularization path for different alpha values"""
        
        try:
            from sklearn.linear_model import ridge_regression
            
            # Generate alpha values
            n_alphas = 100
            alpha_min, alpha_max = self.alpha_range
            alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas)
            
            # Calculate coefficients for each alpha
            coefs = []
            train_scores = []
            
            for alpha in alphas:
                # Fit Ridge with current alpha
                coef = ridge_regression(X, y, alpha=alpha, fit_intercept=self.fit_intercept)
                coefs.append(coef)
                
                # Calculate training score
                if self.fit_intercept:
                    y_pred = X @ coef[:-1] + coef[-1]  # Last element is intercept
                else:
                    y_pred = X @ coef
                
                train_score = r2_score(y, y_pred)
                train_scores.append(train_score)
            
            self.regularization_path_ = {
                'alphas': alphas,
                'coefficients': np.array(coefs),
                'train_scores': np.array(train_scores),
                'best_alpha_idx': np.argmin(np.abs(alphas - self.best_alpha_))
            }
            
        except Exception as e:
            logger.debug(f"Could not calculate regularization path: {e}")
            self.regularization_path_ = None
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients with feature names and regularization info
        
        Returns:
            DataFrame with coefficients and statistics
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get coefficients")
        
        if self.coefficients_ is None:
            raise BusinessLogicError("Coefficients not available")
        
        # Create coefficients DataFrame
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.coefficients_,
            'abs_coefficient': np.abs(self.coefficients_),
            'regularization_effect': self._calculate_regularization_effect()
        })
        
        # Add intercept if fitted
        if self.fit_intercept and self.intercept_ is not None:
            intercept_row = pd.DataFrame({
                'feature': ['intercept'],
                'coefficient': [self.intercept_],
                'abs_coefficient': [abs(self.intercept_)],
                'regularization_effect': [0.0]  # Intercept not regularized
            })
            coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
        
        # Sort by absolute coefficient value
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        return coef_df
    
    def _calculate_regularization_effect(self) -> np.ndarray:
        """Calculate the effect of regularization on each coefficient"""
        
        if self.regularization_path_ is None:
            return np.zeros(len(self.coefficients_))
        
        try:
            # Find coefficient values without regularization (smallest alpha)
            unregularized_coefs = self.regularization_path_['coefficients'][0]
            if self.fit_intercept:
                unregularized_coefs = unregularized_coefs[:-1]  # Remove intercept
            
            # Calculate shrinkage percentage
            regularization_effect = 1.0 - (np.abs(self.coefficients_) / 
                                         (np.abs(unregularized_coefs) + 1e-10))
            
            return regularization_effect
            
        except Exception as e:
            logger.debug(f"Could not calculate regularization effect: {e}")
            return np.zeros(len(self.coefficients_))
    
    def get_alpha_validation_curve(self, X: pd.DataFrame, y: pd.Series, 
                                  n_alphas: int = 50) -> Dict[str, np.ndarray]:
        """
        Generate validation curve for alpha parameter
        
        Args:
            X: Feature matrix
            y: Target values
            n_alphas: Number of alpha values to test
            
        Returns:
            Dictionary with alphas, train scores, and validation scores
        """
        
        logger.info(f"Generating alpha validation curve with {n_alphas} points")
        
        # Generate alpha range
        alpha_min, alpha_max = self.alpha_range
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas)
        
        # Preprocess data
        X_processed = self._preprocess_features(X)
        y_processed = self._preprocess_targets(y)
        
        # Create base model for validation curve
        base_model = Ridge(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
            positive=self.positive
        )
        
        # Generate validation curve
        train_scores, val_scores = validation_curve(
            base_model, X_processed, y_processed,
            param_name='alpha',
            param_range=alphas,
            cv=self.cv_folds,
            scoring='r2',
            n_jobs=-1
        )
        
        return {
            'alphas': alphas,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1),
            'best_alpha': alphas[np.argmax(np.mean(val_scores, axis=1))],
            'best_score': np.max(np.mean(val_scores, axis=1))
        }
    
    def plot_regularization_path(self, top_features: int = 20) -> Any:
        """
        Plot regularization path showing coefficient evolution
        
        Args:
            top_features: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        if self.regularization_path_ is None:
            logger.warning("Regularization path not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            path = self.regularization_path_
            alphas = path['alphas']
            coefficients = path['coefficients']
            
            # Remove intercept if present
            if self.fit_intercept:
                coefficients = coefficients[:, :-1]
            
            # Select top features based on final coefficient magnitude
            final_coefs = coefficients[-1, :]
            top_indices = np.argsort(np.abs(final_coefs))[-top_features:]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            for i in top_indices:
                plt.plot(alphas, coefficients[:, i], 
                        label=self.feature_names[i] if i < len(self.feature_names) else f'Feature_{i}',
                        linewidth=2)
            
            # Mark best alpha
            plt.axvline(x=self.best_alpha_, color='red', linestyle='--', 
                       label=f'Best α = {self.best_alpha_:.4f}')
            
            plt.xscale('log')
            plt.xlabel('Regularization Parameter (α)')
            plt.ylabel('Coefficient Value')
            plt.title(f'Ridge Regularization Path - {self.name}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_alpha_validation_curve(self, X: pd.DataFrame, y: pd.Series, 
                                   n_alphas: int = 50) -> Any:
        """
        Plot validation curve for alpha parameter selection
        
        Args:
            X: Feature matrix
            y: Target values
            n_alphas: Number of alpha values to test
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            # Generate validation curve
            curve_data = self.get_alpha_validation_curve(X, y, n_alphas)
            
            alphas = curve_data['alphas']
            train_mean = curve_data['train_scores_mean']
            train_std = curve_data['train_scores_std']
            val_mean = curve_data['val_scores_mean']
            val_std = curve_data['val_scores_std']
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot training scores
            plt.plot(alphas, train_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(alphas, train_mean - train_std, train_mean + train_std, 
                           alpha=0.2, color='blue')
            
            # Plot validation scores
            plt.plot(alphas, val_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(alphas, val_mean - val_std, val_mean + val_std, 
                           alpha=0.2, color='red')
            
            # Mark best alpha
            best_alpha = curve_data['best_alpha']
            best_score = curve_data['best_score']
            plt.axvline(x=best_alpha, color='green', linestyle='--',
                       label=f'Best α = {best_alpha:.4f} (Score = {best_score:.3f})')
            
            # Mark current model's alpha if different
            if hasattr(self, 'best_alpha_') and self.best_alpha_ != best_alpha:
                plt.axvline(x=self.best_alpha_, color='purple', linestyle=':',
                           label=f'Model α = {self.best_alpha_:.4f}')
            
            plt.xscale('log')
            plt.xlabel('Regularization Parameter (α)')
            plt.ylabel('R² Score')
            plt.title(f'Ridge Alpha Validation Curve - {self.name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_coefficient_comparison(self, other_model=None) -> Any:
        """
        Plot coefficient comparison with and without regularization
        
        Args:
            other_model: Another model to compare with (e.g., LinearRegression)
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from .linear import FinancialLinearRegression
            
            # If no comparison model provided, fit a linear regression
            if other_model is None:
                linear_model = FinancialLinearRegression(
                    name="comparison_linear",
                    auto_scale=True
                )
                # We need the original training data for this - this is a limitation
                logger.warning("No comparison model provided and training data not stored")
                return None
            
            # Get coefficients from both models
            ridge_coefs = self.get_coefficients()
            if hasattr(other_model, 'get_coefficients'):
                other_coefs = other_model.get_coefficients()
            else:
                logger.warning("Comparison model doesn't have get_coefficients method")
                return None
            
            # Merge coefficients (excluding intercept for plotting)
            ridge_features = ridge_coefs[ridge_coefs['feature'] != 'intercept']
            other_features = other_coefs[other_coefs['feature'] != 'intercept']
            
            merged = pd.merge(ridge_features[['feature', 'coefficient']], 
                            other_features[['feature', 'coefficient']], 
                            on='feature', suffixes=('_ridge', '_other'))
            
            # Create comparison plot
            plt.figure(figsize=(12, 8))
            
            plt.scatter(merged['coefficient_other'], merged['coefficient_ridge'], 
                       alpha=0.7, s=50)
            
            # Add diagonal line
            min_coef = min(merged['coefficient_other'].min(), merged['coefficient_ridge'].min())
            max_coef = max(merged['coefficient_other'].max(), merged['coefficient_ridge'].max())
            plt.plot([min_coef, max_coef], [min_coef, max_coef], 'r--', alpha=0.8)
            
            plt.xlabel('Linear Regression Coefficients')
            plt.ylabel('Ridge Regression Coefficients')
            plt.title(f'Coefficient Comparison: Ridge vs Linear - {self.name}')
            plt.grid(True, alpha=0.3)
            
            # Add text with regularization info
            plt.text(0.05, 0.95, f'Ridge α = {self.best_alpha_:.4f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def get_regularization_summary(self) -> Dict[str, Any]:
        """Get summary of regularization effects"""
        
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get regularization summary")
        
        coef_df = self.get_coefficients()
        
        # Calculate regularization statistics
        summary = {
            'best_alpha': float(self.best_alpha_),
            'n_features': len(self.feature_names),
            'n_zero_coefficients': int((np.abs(self.coefficients_) < 1e-10).sum()),
            'max_coefficient': float(np.max(np.abs(self.coefficients_))),
            'mean_coefficient': float(np.mean(np.abs(self.coefficients_))),
            'coefficient_l2_norm': float(np.linalg.norm(self.coefficients_))
        }
        
        # Add regularization effect statistics if available
        if 'regularization_effect' in coef_df.columns:
            reg_effects = coef_df['regularization_effect'][coef_df['feature'] != 'intercept']
            summary.update({
                'mean_regularization_effect': float(reg_effects.mean()),
                'max_regularization_effect': float(reg_effects.max()),
                'features_heavily_regularized': int((reg_effects > 0.5).sum())
            })
        
        # Add alpha selection info if available
        if self.alpha_scores_:
            summary['alpha_cv_scores'] = {
                'best_score': float(max(self.alpha_scores_.values())),
                'score_range': [float(min(self.alpha_scores_.values())), 
                               float(max(self.alpha_scores_.values()))],
                'n_alphas_tested': len(self.alpha_scores_)
            }
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add Ridge-specific information
        summary.update({
            'regularization_type': 'L2 (Ridge)',
            'alpha': float(self.best_alpha_) if self.best_alpha_ else self.alpha,
            'auto_alpha_selection': self.auto_alpha,
            'cv_folds': self.cv_folds,
            'solver': self.solver,
            'positive_coefficients': self.positive,
            'intercept': float(self.intercept_) if self.intercept_ is not None else None
        })
        
        # Add regularization summary
        if self.is_fitted:
            try:
                summary['regularization_summary'] = self.get_regularization_summary()
            except Exception as e:
                logger.debug(f"Could not generate regularization summary: {e}")
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_ridge_regressor(alpha: Union[float, str] = 'auto', 
                          cv_folds: int = 5,
                          **kwargs) -> FinancialRidgeRegression:
    """
    Create a Ridge regression model
    
    Args:
        alpha: Regularization strength ('auto' for CV selection, float for fixed)
        cv_folds: Number of CV folds for alpha selection
        **kwargs: Additional model parameters
        
    Returns:
        Configured Ridge regression model
    """
    
    if alpha == 'auto':
        # Use cross-validation for alpha selection
        default_params = {
            'name': 'ridge_regression_cv',
            'auto_alpha': True,
            'alpha_range': (0.001, 1000.0),
            'cv_folds': cv_folds,
            'solver': 'auto',
            'fit_intercept': True,
            'random_state': 42
        }
    else:
        # Use fixed alpha
        default_params = {
            'name': 'ridge_regression',
            'alpha': alpha,
            'auto_alpha': False,
            'solver': 'auto',
            'fit_intercept': True,
            'random_state': 42
        }
    
    # Override with provided kwargs
    default_params.update(kwargs)
    
    return FinancialRidgeRegression(**default_params)

def create_ridge_cv_regressor(alpha_range: Tuple[float, float] = (0.001, 1000.0),
                             cv_folds: int = 10,
                             **kwargs) -> FinancialRidgeRegression:
    """Create Ridge regression with cross-validation for alpha selection"""
    
    return create_ridge_regressor(
        alpha='auto',
        alpha_range=alpha_range,
        cv_folds=cv_folds,
        **kwargs
    )

def create_positive_ridge_regressor(alpha: Union[float, str] = 'auto',
                                   **kwargs) -> FinancialRidgeRegression:
    """Create Ridge regression with positive coefficient constraint"""
    
    return create_ridge_regressor(
        alpha=alpha,
        positive=True,
        solver='lbfgs',  # Required for positive constraint
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def find_optimal_alpha(X: pd.DataFrame, y: pd.Series,
                      alpha_range: Tuple[float, float] = (0.001, 1000.0),
                      n_alphas: int = 100,
                      cv_folds: int = 5,
                      scoring: str = 'r2') -> Dict[str, Any]:
    """
    Find optimal alpha for Ridge regression using cross-validation
    
    Args:
        X: Feature matrix
        y: Target values
        alpha_range: Range of alpha values to test
        n_alphas: Number of alpha values to test
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Dictionary with optimal alpha and CV results
    """
    
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    
    logger.info(f"Finding optimal alpha with {n_alphas} candidates and {cv_folds}-fold CV")
    
    # Generate alpha candidates
    alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform cross-validation
    ridge_cv = RidgeCV(
        alphas=alphas,
        cv=cv_folds,
        scoring=scoring,
        store_cv_values=True
    )
    
    ridge_cv.fit(X_scaled, y)
    
    # Extract results
    cv_scores = np.mean(ridge_cv.cv_values_, axis=0)
    
    result = {
        'optimal_alpha': ridge_cv.alpha_,
        'optimal_score': ridge_cv.best_score_ if hasattr(ridge_cv, 'best_score_') else np.max(cv_scores),
        'alphas_tested': alphas,
        'cv_scores': cv_scores,
        'cv_scores_std': np.std(ridge_cv.cv_values_, axis=0),
        'alpha_index': np.argmin(np.abs(alphas - ridge_cv.alpha_))
    }
    
    logger.info(f"Optimal alpha found: {result['optimal_alpha']:.6f} (score: {result['optimal_score']:.4f})")
    
    return result

def compare_ridge_alphas(X: pd.DataFrame, y: pd.Series,
                        alphas: List[float],
                        cv_folds: int = 5) -> pd.DataFrame:
    """
    Compare Ridge regression performance across different alpha values
    
    Args:
        X: Feature matrix
        y: Target values
        alphas: List of alpha values to compare
        cv_folds: Number of cross-validation folds
        
    Returns:
        DataFrame with comparison results
    """
    
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    logger.info(f"Comparing {len(alphas)} alpha values")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    
    for alpha in alphas:
        # Create Ridge model
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        
        # Perform cross-validation
        cv_scores = cross_val_score(ridge, X_scaled, y, cv=cv_folds, scoring='r2')
        
        results.append({
            'alpha': alpha,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max()
        })
    
    results_df = pd.DataFrame(results).sort_values('mean_score', ascending=False)
    
    logger.info(f"Best alpha: {results_df.iloc[0]['alpha']:.6f} (score: {results_df.iloc[0]['mean_score']:.4f})")
    
    return results_df

def analyze_ridge_vs_linear(X: pd.DataFrame, y: pd.Series,
                           alpha: float = 1.0) -> Dict[str, Any]:
    """
    Analyze differences between Ridge and Linear regression
    
    Args:
        X: Feature matrix
        y: Target values
        alpha: Ridge regularization parameter
        
    Returns:
        Dictionary with comparison analysis
    """
    
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit both models
    linear_model = LinearRegression()
    ridge_model = Ridge(alpha=alpha)
    
    linear_model.fit(X_scaled, y)
    ridge_model.fit(X_scaled, y)
    
    # Compare coefficients
    coef_comparison = pd.DataFrame({
        'feature': X.columns,
        'linear_coef': linear_model.coef_,
        'ridge_coef': ridge_model.coef_,
        'abs_linear': np.abs(linear_model.coef_),
        'abs_ridge': np.abs(ridge_model.coef_)
    })
    
    coef_comparison['shrinkage'] = 1 - (coef_comparison['abs_ridge'] / 
                                       (coef_comparison['abs_linear'] + 1e-10))
    
    # Cross-validation comparison
    linear_cv = cross_val_score(linear_model, X_scaled, y, cv=5, scoring='r2')
    ridge_cv = cross_val_score(ridge_model, X_scaled, y, cv=5, scoring='r2')
    
    analysis = {
        'coefficient_comparison': coef_comparison.sort_values('shrinkage', ascending=False),
        'performance_comparison': {
            'linear_cv_mean': linear_cv.mean(),
            'linear_cv_std': linear_cv.std(),
            'ridge_cv_mean': ridge_cv.mean(),
            'ridge_cv_std': ridge_cv.std(),
            'performance_difference': ridge_cv.mean() - linear_cv.mean()
        },
        'regularization_effects': {
            'mean_shrinkage': coef_comparison['shrinkage'].mean(),
            'max_shrinkage': coef_comparison['shrinkage'].max(),
            'features_heavily_shrunk': (coef_comparison['shrinkage'] > 0.5).sum(),
            'l2_norm_linear': np.linalg.norm(linear_model.coef_),
            'l2_norm_ridge': np.linalg.norm(ridge_model.coef_),
            'norm_reduction': 1 - (np.linalg.norm(ridge_model.coef_) / 
                                  np.linalg.norm(linear_model.coef_))
        }
    }
    
    return analysis
