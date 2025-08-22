# ============================================
# StockPredictionPro - src/models/regression/lasso.py
# Lasso regression models for financial prediction with L1 regularization and feature selection
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.linear_model import Lasso, LassoCV, LassoLars, LassoLarsCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_regressor import BaseFinancialRegressor, RegressionStrategy

logger = get_logger('models.regression.lasso')

# ============================================
# Lasso Regression Model
# ============================================

class FinancialLassoRegression(BaseFinancialRegressor):
    """
    Lasso regression model optimized for financial data with L1 regularization
    
    Features:
    - L1 regularization for automatic feature selection
    - Cross-validation for optimal alpha selection
    - Regularization path analysis with sparsity tracking
    - Feature selection and importance ranking
    - Multiple Lasso variants (standard, LARS)
    """
    
    def __init__(self,
                 name: str = "lasso_regression",
                 alpha: Union[float, List[float]] = 1.0,
                 fit_intercept: bool = True,
                 normalize: bool = False,
                 precompute: Union[bool, str] = False,
                 copy_X: bool = True,
                 max_iter: int = 1000,
                 tol: float = 1e-4,
                 warm_start: bool = False,
                 positive: bool = False,
                 random_state: Optional[int] = 42,
                 selection: str = 'cyclic',
                 auto_alpha: bool = True,
                 alpha_range: Optional[Tuple[float, float]] = None,
                 cv_folds: int = 5,
                 lasso_variant: str = 'standard',
                 feature_selection_threshold: Optional[float] = None,
                 **kwargs):
        """
        Initialize Financial Lasso Regression
        
        Args:
            name: Model name
            alpha: Regularization strength (or list for CV)
            fit_intercept: Whether to calculate intercept
            normalize: Whether to normalize features (deprecated in sklearn)
            precompute: Whether to use precomputed Gram matrix
            copy_X: Whether to copy X or overwrite
            max_iter: Maximum iterations
            tol: Tolerance for optimization
            warm_start: When set to True, reuse solution of previous call
            positive: Whether to force positive coefficients
            random_state: Random state for reproducibility
            selection: Algorithm to use ('cyclic', 'random')
            auto_alpha: Whether to automatically select best alpha via CV
            alpha_range: Range for alpha search (min_alpha, max_alpha)
            cv_folds: Number of cross-validation folds for alpha selection
            lasso_variant: Type of Lasso ('standard', 'lars')
            feature_selection_threshold: Threshold for feature selection
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="lasso_regression",
            regression_strategy=RegressionStrategy.PRICE_PREDICTION,
            **kwargs
        )
        
        # Lasso regression parameters
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
        self.auto_alpha = auto_alpha
        self.alpha_range = alpha_range or (0.0001, 10.0)
        self.cv_folds = cv_folds
        self.lasso_variant = lasso_variant
        self.feature_selection_threshold = feature_selection_threshold
        
        # Store parameters for model creation
        self.model_params.update({
            'alpha': alpha,
            'fit_intercept': fit_intercept,
            'precompute': precompute,
            'copy_X': copy_X,
            'max_iter': max_iter,
            'tol': tol,
            'warm_start': warm_start,
            'positive': positive,
            'random_state': random_state,
            'selection': selection
        })
        
        # Lasso-specific attributes
        self.scaler_: Optional[StandardScaler] = None
        self.best_alpha_: Optional[float] = None
        self.alpha_scores_: Optional[Dict[float, float]] = None
        self.regularization_path_: Optional[Dict[str, Any]] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.selected_features_: Optional[List[str]] = None
        self.feature_selector_: Optional[SelectFromModel] = None
        self.sparsity_stats_: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized {self.lasso_variant} Lasso regression: {self.name} (auto_alpha={self.auto_alpha})")
    
    def _create_model(self) -> Union[Lasso, LassoCV, LassoLars, LassoLarsCV]:
        """Create the Lasso regression model"""
        
        if self.lasso_variant == 'lars':
            # Use LARS (Least Angle Regression) algorithm
            if self.auto_alpha:
                lars_cv_params = {
                    'fit_intercept': self.fit_intercept,
                    'verbose': False,
                    'normalize': self.normalize,
                    'precompute': self.precompute,
                    'max_iter': self.max_iter,
                    'cv': self.cv_folds,
                    'max_n_alphas': 1000,
                    'n_jobs': -1,
                    'eps': 1e-16,
                    'copy_X': self.copy_X
                }
                return LassoLarsCV(**lars_cv_params)
            else:
                lars_params = {
                    'alpha': self.alpha if not isinstance(self.alpha, list) else self.alpha[0],
                    'fit_intercept': self.fit_intercept,
                    'verbose': False,
                    'normalize': self.normalize,
                    'precompute': self.precompute,
                    'max_iter': self.max_iter,
                    'eps': 1e-16,
                    'copy_X': self.copy_X,
                    'positive': self.positive
                }
                return LassoLars(**lars_params)
        
        else:  # standard Lasso
            if self.auto_alpha:
                # Use LassoCV for automatic alpha selection
                if isinstance(self.alpha, list):
                    alphas = self.alpha
                else:
                    # Generate alpha range for cross-validation
                    alphas = np.logspace(
                        np.log10(self.alpha_range[0]), 
                        np.log10(self.alpha_range[1]), 
                        100
                    )
                
                cv_params = {
                    'alphas': alphas,
                    'fit_intercept': self.fit_intercept,
                    'normalize': self.normalize,
                    'precompute': self.precompute,
                    'copy_X': self.copy_X,
                    'max_iter': self.max_iter,
                    'tol': self.tol,
                    'cv': self.cv_folds,
                    'verbose': False,
                    'n_jobs': -1,
                    'positive': self.positive,
                    'random_state': self.random_state,
                    'selection': self.selection
                }
                
                return LassoCV(**cv_params)
            else:
                # Use regular Lasso with fixed alpha
                lasso_params = {k: v for k, v in self.model_params.items() 
                               if k not in ['normalize']}  # normalize deprecated
                return Lasso(**lasso_params)
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with automatic scaling"""
        
        # Basic preprocessing
        X_processed = super()._preprocess_features(X)
        
        # Always scale features for Lasso regression
        if self.scaler_ is None:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_processed)
            logger.debug("Fitted feature scaler for Lasso regression")
        else:
            X_scaled = self.scaler_.transform(X_processed)
        
        return X_scaled
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Enhanced post-training processing for Lasso regression"""
        
        # Call parent processing
        super()._post_training_processing(X, y)
        
        # Extract Lasso-specific information
        if isinstance(self.model, (LassoCV, LassoLarsCV)):
            self.best_alpha_ = self.model.alpha_
            
            # Store CV scores for different alphas
            if hasattr(self.model, 'mse_path_'):
                # LassoCV stores MSE path
                cv_scores = -np.mean(self.model.mse_path_, axis=1)  # Convert MSE to negative for "higher is better"
                if hasattr(self.model, 'alphas_'):
                    self.alpha_scores_ = dict(zip(self.model.alphas_, cv_scores))
            elif hasattr(self.model, 'cv_alphas_'):
                # LassoLarsCV
                cv_scores = -np.mean(self.model.cv_mse_path_, axis=1)
                self.alpha_scores_ = dict(zip(self.model.cv_alphas_, cv_scores))
            
            logger.info(f"Best alpha selected via CV: {self.best_alpha_:.6f}")
        else:
            self.best_alpha_ = self.model.alpha
        
        # Extract coefficients
        self.coefficients_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        # Set feature importances (absolute coefficients)
        self.feature_importances_ = np.abs(self.coefficients_)
        
        # Analyze sparsity
        self._analyze_sparsity()
        
        # Identify selected features (non-zero coefficients)
        self._identify_selected_features()
        
        # Calculate regularization path if reasonable number of features
        if X.shape[1] <= 200:  # Lasso path can be expensive for high-dimensional data
            self._calculate_regularization_path(X, y)
    
    def _analyze_sparsity(self):
        """Analyze sparsity patterns in Lasso coefficients"""
        
        n_features = len(self.coefficients_)
        non_zero_coefs = np.sum(np.abs(self.coefficients_) > 1e-10)
        zero_coefs = n_features - non_zero_coefs
        sparsity_ratio = zero_coefs / n_features
        
        # Calculate coefficient statistics
        non_zero_indices = np.abs(self.coefficients_) > 1e-10
        if non_zero_indices.any():
            active_coefs = self.coefficients_[non_zero_indices]
            max_coef = np.max(np.abs(active_coefs))
            mean_coef = np.mean(np.abs(active_coefs))
            l1_norm = np.sum(np.abs(active_coefs))
        else:
            max_coef = mean_coef = l1_norm = 0.0
        
        self.sparsity_stats_ = {
            'n_features_total': n_features,
            'n_features_selected': int(non_zero_coefs),
            'n_features_eliminated': int(zero_coefs),
            'sparsity_ratio': float(sparsity_ratio),
            'selection_ratio': float(non_zero_coefs / n_features),
            'max_coefficient': float(max_coef),
            'mean_active_coefficient': float(mean_coef),
            'l1_norm': float(l1_norm)
        }
        
        logger.info(f"Lasso feature selection: {non_zero_coefs}/{n_features} features selected "
                   f"({sparsity_ratio:.1%} sparsity)")
    
    def _identify_selected_features(self):
        """Identify features selected by Lasso (non-zero coefficients)"""
        
        selected_indices = np.abs(self.coefficients_) > 1e-10
        self.selected_features_ = [self.feature_names[i] for i in range(len(self.feature_names)) 
                                  if i < len(selected_indices) and selected_indices[i]]
        
        logger.debug(f"Selected features: {len(self.selected_features_)}")
    
    def _calculate_regularization_path(self, X: np.ndarray, y: np.ndarray):
        """Calculate Lasso regularization path"""
        
        try:
            from sklearn.linear_model import lasso_path
            
            # Generate alpha values
            n_alphas = 100
            alpha_min, alpha_max = self.alpha_range
            
            # Use lasso_path to compute the entire regularization path
            alphas_path, coefs_path, _ = lasso_path(
                X, y, 
                alphas=None,  # Let sklearn choose alphas
                l1_ratio=1.0,  # Pure Lasso (not Elastic Net)
                fit_intercept=self.fit_intercept,
                normalize=False,  # We already scaled
                copy_X=True,
                max_iter=self.max_iter,
                tol=self.tol,
                return_n_iter=True,
                positive=self.positive
            )
            
            # Calculate number of selected features for each alpha
            n_selected = np.sum(np.abs(coefs_path) > 1e-10, axis=0)
            
            # Calculate training scores for each alpha
            train_scores = []
            for i, alpha in enumerate(alphas_path):
                coef = coefs_path[:, i]
                if self.fit_intercept:
                    y_pred = X @ coef + self.intercept_
                else:
                    y_pred = X @ coef
                train_score = r2_score(y, y_pred)
                train_scores.append(train_score)
            
            self.regularization_path_ = {
                'alphas': alphas_path,
                'coefficients': coefs_path,
                'n_selected_features': n_selected,
                'train_scores': np.array(train_scores),
                'best_alpha_idx': np.argmin(np.abs(alphas_path - self.best_alpha_))
            }
            
            logger.debug(f"Calculated regularization path with {len(alphas_path)} alpha values")
            
        except Exception as e:
            logger.debug(f"Could not calculate regularization path: {e}")
            self.regularization_path_ = None
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients with feature selection information
        
        Returns:
            DataFrame with coefficients and selection status
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
            'selected': np.abs(self.coefficients_) > 1e-10,
            'importance_rank': np.argsort(np.argsort(-np.abs(self.coefficients_))) + 1
        })
        
        # Add intercept if fitted
        if self.fit_intercept and self.intercept_ is not None:
            intercept_row = pd.DataFrame({
                'feature': ['intercept'],
                'coefficient': [self.intercept_],
                'abs_coefficient': [abs(self.intercept_)],
                'selected': [True],
                'importance_rank': [0]  # Intercept not ranked
            })
            coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
        
        # Sort by absolute coefficient value
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        return coef_df
    
    def get_selected_features(self) -> List[str]:
        """
        Get list of features selected by Lasso (non-zero coefficients)
        
        Returns:
            List of selected feature names
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get selected features")
        
        return self.selected_features_.copy() if self.selected_features_ else []
    
    def create_feature_selector(self) -> SelectFromModel:
        """
        Create a feature selector based on Lasso coefficients
        
        Returns:
            Fitted SelectFromModel instance
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to create feature selector")
        
        # Create feature selector
        threshold = self.feature_selection_threshold or 1e-10
        self.feature_selector_ = SelectFromModel(
            estimator=self.model,
            threshold=threshold,
            prefit=True
        )
        
        return self.feature_selector_
    
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using Lasso feature selection
        
        Args:
            X: Input features
            
        Returns:
            Transformed features with only selected columns
        """
        if self.feature_selector_ is None:
            self.create_feature_selector()
        
        # Get selected feature indices
        selected_mask = self.feature_selector_.get_support()
        selected_features = [name for name, selected in zip(X.columns, selected_mask) if selected]
        
        return X[selected_features]
    
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
        
        logger.info(f"Generating Lasso alpha validation curve with {n_alphas} points")
        
        # Generate alpha range
        alpha_min, alpha_max = self.alpha_range
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas)
        
        # Preprocess data
        X_processed = self._preprocess_features(X)
        y_processed = self._preprocess_targets(y)
        
        # Create base model for validation curve
        base_model = Lasso(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            max_iter=self.max_iter,
            tol=self.tol,
            positive=self.positive,
            random_state=self.random_state,
            selection=self.selection
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
        Plot Lasso regularization path showing coefficient evolution and sparsity
        
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
            n_selected = path['n_selected_features']
            
            # Create subplot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Coefficient paths
            # Select top features based on final coefficient magnitude
            final_coefs = coefficients[:, -1]
            top_indices = np.argsort(np.abs(final_coefs))[-top_features:]
            
            for i in top_indices:
                if i < len(self.feature_names):
                    ax1.plot(alphas, coefficients[i, :], 
                            label=self.feature_names[i], linewidth=2)
            
            # Mark best alpha
            ax1.axvline(x=self.best_alpha_, color='red', linestyle='--', 
                       label=f'Best α = {self.best_alpha_:.4f}')
            
            ax1.set_xscale('log')
            ax1.set_xlabel('Regularization Parameter (α)')
            ax1.set_ylabel('Coefficient Value')
            ax1.set_title(f'Lasso Regularization Path - {self.name}')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Number of selected features
            ax2.plot(alphas, n_selected, 'bo-', linewidth=2, markersize=4)
            ax2.axvline(x=self.best_alpha_, color='red', linestyle='--',
                       label=f'Best α = {self.best_alpha_:.4f}')
            
            ax2.set_xscale('log')
            ax2.set_xlabel('Regularization Parameter (α)')
            ax2.set_ylabel('Number of Selected Features')
            ax2.set_title('Feature Selection vs Regularization')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_feature_selection(self) -> Any:
        """
        Plot feature selection results
        
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            coef_df = self.get_coefficients()
            
            # Remove intercept for plotting
            if 'intercept' in coef_df['feature'].values:
                coef_df = coef_df[coef_df['feature'] != 'intercept']
            
            # Split selected and eliminated features
            selected = coef_df[coef_df['selected']]
            eliminated = coef_df[~coef_df['selected']]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot 1: Selected features
            if not selected.empty:
                y_pos = np.arange(len(selected))
                colors = ['red' if x < 0 else 'blue' for x in selected['coefficient']]
                
                ax1.barh(y_pos, selected['coefficient'], color=colors, alpha=0.7)
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(selected['feature'], fontsize=8)
                ax1.set_xlabel('Coefficient Value')
                ax1.set_title(f'Selected Features ({len(selected)}/{len(coef_df)})')
                ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax1.grid(True, alpha=0.3)
                
                # Add coefficient values as text
                for i, (idx, row) in enumerate(selected.iterrows()):
                    coef = row['coefficient']
                    ax1.text(coef + (0.01 * np.sign(coef) if coef != 0 else 0.01), i, 
                            f'{coef:.3f}', ha='left' if coef >= 0 else 'right', 
                            va='center', fontsize=8)
            else:
                ax1.text(0.5, 0.5, 'No features selected', ha='center', va='center',
                        transform=ax1.transAxes, fontsize=14)
                ax1.set_title('Selected Features (0)')
            
            # Plot 2: Sparsity statistics
            if self.sparsity_stats_:
                stats = self.sparsity_stats_
                
                # Create pie chart of feature selection
                sizes = [stats['n_features_selected'], stats['n_features_eliminated']]
                labels = ['Selected', 'Eliminated']
                colors = ['lightblue', 'lightcoral']
                
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Feature Selection Summary')
                
                # Add text with statistics
                text = f"Total features: {stats['n_features_total']}\n"
                text += f"Selected: {stats['n_features_selected']}\n"
                text += f"Eliminated: {stats['n_features_eliminated']}\n"
                text += f"Sparsity: {stats['sparsity_ratio']:.1%}\n"
                text += f"L1 norm: {stats['l1_norm']:.3f}"
                
                ax2.text(1.3, 0.5, text, transform=ax2.transAxes, fontsize=10,
                        verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle(f'Lasso Feature Selection Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_alpha_vs_sparsity(self, X: pd.DataFrame, y: pd.Series, 
                              n_alphas: int = 50) -> Any:
        """
        Plot relationship between alpha and sparsity
        
        Args:
            X: Feature matrix
            y: Target values
            n_alphas: Number of alpha values to test
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.linear_model import Lasso
            
            # Generate alpha range
            alpha_min, alpha_max = self.alpha_range
            alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas)
            
            # Preprocess data
            X_processed = self._preprocess_features(X)
            y_processed = self._preprocess_targets(y)
            
            # Calculate sparsity for each alpha
            sparsities = []
            scores = []
            
            for alpha in alphas:
                lasso = Lasso(alpha=alpha, fit_intercept=self.fit_intercept,
                             max_iter=self.max_iter, tol=self.tol)
                lasso.fit(X_processed, y_processed)
                
                # Calculate sparsity (fraction of zero coefficients)
                n_zero = np.sum(np.abs(lasso.coef_) < 1e-10)
                sparsity = n_zero / len(lasso.coef_)
                sparsities.append(sparsity)
                
                # Calculate R² score
                score = lasso.score(X_processed, y_processed)
                scores.append(score)
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Alpha vs Sparsity
            ax1.plot(alphas, sparsities, 'bo-', linewidth=2, markersize=4)
            ax1.axvline(x=self.best_alpha_, color='red', linestyle='--',
                       label=f'Best α = {self.best_alpha_:.4f}')
            ax1.set_xscale('log')
            ax1.set_xlabel('Regularization Parameter (α)')
            ax1.set_ylabel('Sparsity (Fraction of Zero Coefficients)')
            ax1.set_title('Alpha vs Sparsity')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Alpha vs Score
            ax2.plot(alphas, scores, 'go-', linewidth=2, markersize=4)
            ax2.axvline(x=self.best_alpha_, color='red', linestyle='--',
                       label=f'Best α = {self.best_alpha_:.4f}')
            ax2.set_xscale('log')
            ax2.set_xlabel('Regularization Parameter (α)')
            ax2.set_ylabel('R² Score')
            ax2.set_title('Alpha vs Performance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f'Lasso Alpha Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def get_sparsity_summary(self) -> Dict[str, Any]:
        """Get comprehensive sparsity and feature selection summary"""
        
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get sparsity summary")
        
        summary = {
            'regularization_info': {
                'best_alpha': float(self.best_alpha_),
                'lasso_variant': self.lasso_variant,
                'regularization_type': 'L1 (Lasso)'
            }
        }
        
        # Add sparsity statistics
        if self.sparsity_stats_:
            summary['sparsity_stats'] = self.sparsity_stats_
        
        # Add selected features information
        if self.selected_features_:
            coef_df = self.get_coefficients()
            selected_df = coef_df[coef_df['selected'] & (coef_df['feature'] != 'intercept')]
            
            summary['feature_selection'] = {
                'selected_features': self.selected_features_,
                'n_selected': len(self.selected_features_),
                'selection_ratio': len(self.selected_features_) / len(self.feature_names),
                'top_5_features': selected_df.head(5)['feature'].tolist(),
                'coefficient_range': [float(selected_df['coefficient'].min()), 
                                    float(selected_df['coefficient'].max())]
            }
        
        # Add alpha selection info if available
        if self.alpha_scores_:
            summary['alpha_selection'] = {
                'best_score': float(max(self.alpha_scores_.values())),
                'score_range': [float(min(self.alpha_scores_.values())), 
                               float(max(self.alpha_scores_.values()))],
                'n_alphas_tested': len(self.alpha_scores_)
            }
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add Lasso-specific information
        summary.update({
            'regularization_type': 'L1 (Lasso)',
            'lasso_variant': self.lasso_variant,
            'alpha': float(self.best_alpha_) if self.best_alpha_ else self.alpha,
            'auto_alpha_selection': self.auto_alpha,
            'cv_folds': self.cv_folds,
            'max_iter': self.max_iter,
            'positive_coefficients': self.positive,
            'intercept': float(self.intercept_) if self.intercept_ is not None else None
        })
        
        # Add sparsity and feature selection summary
        if self.is_fitted:
            try:
                summary['sparsity_summary'] = self.get_sparsity_summary()
            except Exception as e:
                logger.debug(f"Could not generate sparsity summary: {e}")
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_lasso_regressor(alpha: Union[float, str] = 'auto',
                          variant: str = 'standard',
                          cv_folds: int = 5,
                          **kwargs) -> FinancialLassoRegression:
    """
    Create a Lasso regression model
    
    Args:
        alpha: Regularization strength ('auto' for CV selection, float for fixed)
        variant: Lasso variant ('standard', 'lars')
        cv_folds: Number of CV folds for alpha selection
        **kwargs: Additional model parameters
        
    Returns:
        Configured Lasso regression model
    """
    
    if alpha == 'auto':
        # Use cross-validation for alpha selection
        default_params = {
            'name': f'lasso_{variant}_cv',
            'auto_alpha': True,
            'alpha_range': (0.0001, 10.0),
            'cv_folds': cv_folds,
            'lasso_variant': variant,
            'fit_intercept': True,
            'max_iter': 1000,
            'random_state': 42
        }
    else:
        # Use fixed alpha
        default_params = {
            'name': f'lasso_{variant}',
            'alpha': alpha,
            'auto_alpha': False,
            'lasso_variant': variant,
            'fit_intercept': True,
            'max_iter': 1000,
            'random_state': 42
        }
    
    # Override with provided kwargs
    default_params.update(kwargs)
    
    return FinancialLassoRegression(**default_params)

def create_lasso_cv_regressor(alpha_range: Tuple[float, float] = (0.0001, 10.0),
                             cv_folds: int = 10,
                             **kwargs) -> FinancialLassoRegression:
    """Create Lasso regression with cross-validation for alpha selection"""
    
    return create_lasso_regressor(
        alpha='auto',
        variant='standard',
        alpha_range=alpha_range,
        cv_folds=cv_folds,
        **kwargs
    )

def create_lasso_lars_regressor(alpha: Union[float, str] = 'auto',
                               **kwargs) -> FinancialLassoRegression:
    """Create Lasso regression using LARS algorithm"""
    
    return create_lasso_regressor(
        alpha=alpha,
        variant='lars',
        **kwargs
    )

def create_feature_selector_lasso(alpha: float = 1.0,
                                 selection_threshold: float = 1e-5,
                                 **kwargs) -> FinancialLassoRegression:
    """Create Lasso model optimized for feature selection"""
    
    return create_lasso_regressor(
        alpha=alpha,
        name='lasso_feature_selector',
        feature_selection_threshold=selection_threshold,
        max_iter=2000,  # More iterations for better convergence
        tol=1e-6,      # Tighter tolerance
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def find_optimal_lasso_alpha(X: pd.DataFrame, y: pd.Series,
                             alpha_range: Tuple[float, float] = (0.0001, 10.0),
                             n_alphas: int = 100,
                             cv_folds: int = 5,
                             scoring: str = 'r2') -> Dict[str, Any]:
    """
    Find optimal alpha for Lasso regression using cross-validation
    
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
    
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    
    logger.info(f"Finding optimal Lasso alpha with {n_alphas} candidates and {cv_folds}-fold CV")
    
    # Generate alpha candidates
    alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform cross-validation
    lasso_cv = LassoCV(
        alphas=alphas,
        cv=cv_folds,
        n_jobs=-1,
        max_iter=1000,
        random_state=42
    )
    
    lasso_cv.fit(X_scaled, y)
    
    # Calculate sparsity at optimal alpha
    n_selected = np.sum(np.abs(lasso_cv.coef_) > 1e-10)
    sparsity = 1 - (n_selected / len(lasso_cv.coef_))
    
    result = {
        'optimal_alpha': lasso_cv.alpha_,
        'optimal_score': lasso_cv.score(X_scaled, y),
        'n_features_selected': int(n_selected),
        'sparsity': float(sparsity),
        'alphas_tested': alphas,
        'cv_scores_mean': -np.mean(lasso_cv.mse_path_, axis=1),  # Convert MSE to score
        'cv_scores_std': np.std(lasso_cv.mse_path_, axis=1),
        'alpha_index': np.argmin(np.abs(alphas - lasso_cv.alpha_))
    }
    
    logger.info(f"Optimal Lasso alpha: {result['optimal_alpha']:.6f} "
               f"(score: {result['optimal_score']:.4f}, "
               f"selected: {result['n_features_selected']}/{len(X.columns)} features)")
    
    return result

def perform_lasso_feature_selection(X: pd.DataFrame, y: pd.Series,
                                   alpha: Optional[float] = None,
                                   selection_threshold: float = 1e-5) -> Dict[str, Any]:
    """
    Perform feature selection using Lasso regression
    
    Args:
        X: Feature matrix
        y: Target values
        alpha: Regularization parameter (auto-selected if None)
        selection_threshold: Threshold for feature selection
        
    Returns:
        Dictionary with feature selection results
    """
    
    logger.info(f"Performing Lasso feature selection on {X.shape[1]} features")
    
    # Create Lasso feature selector
    if alpha is None:
        lasso_selector = create_lasso_cv_regressor(name='feature_selector')
    else:
        lasso_selector = create_feature_selector_lasso(
            alpha=alpha,
            selection_threshold=selection_threshold
        )
    
    # Fit the model
    lasso_selector.fit(X, y)
    
    # Get selected features
    selected_features = lasso_selector.get_selected_features()
    coef_df = lasso_selector.get_coefficients()
    selected_coef = coef_df[coef_df['selected'] & (coef_df['feature'] != 'intercept')]
    
    # Create results
    result = {
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'selection_ratio': len(selected_features) / len(X.columns),
        'alpha_used': lasso_selector.best_alpha_,
        'model_score': lasso_selector.training_score,
        'feature_coefficients': selected_coef[['feature', 'coefficient', 'abs_coefficient']].to_dict('records'),
        'sparsity_stats': lasso_selector.sparsity_stats_
    }
    
    logger.info(f"Feature selection complete: {result['n_selected']}/{X.shape[1]} features selected "
               f"({result['selection_ratio']:.1%} selection ratio)")
    
    return result

def compare_lasso_variants(X: pd.DataFrame, y: pd.Series,
                          alpha: float = 1.0) -> Dict[str, Any]:
    """
    Compare different Lasso variants (standard vs LARS)
    
    Args:
        X: Feature matrix
        y: Target values
        alpha: Regularization parameter
        
    Returns:
        Dictionary with comparison results
    """
    
    logger.info("Comparing Lasso variants (Standard vs LARS)")
    
    # Create both variants
    lasso_standard = create_lasso_regressor(alpha=alpha, variant='standard')
    lasso_lars = create_lasso_regressor(alpha=alpha, variant='lars')
    
    # Fit both models
    lasso_standard.fit(X, y)
    lasso_lars.fit(X, y)
    
    # Compare results
    comparison = {
        'standard': {
            'training_score': lasso_standard.training_score,
            'n_features_selected': lasso_standard.sparsity_stats_['n_features_selected'],
            'sparsity': lasso_standard.sparsity_stats_['sparsity_ratio'],
            'l1_norm': lasso_standard.sparsity_stats_['l1_norm'],
            'selected_features': lasso_standard.get_selected_features()
        },
        'lars': {
            'training_score': lasso_lars.training_score,
            'n_features_selected': lasso_lars.sparsity_stats_['n_features_selected'],
            'sparsity': lasso_lars.sparsity_stats_['sparsity_ratio'],
            'l1_norm': lasso_lars.sparsity_stats_['l1_norm'],
            'selected_features': lasso_lars.get_selected_features()
        }
    }
    
    # Calculate differences
    comparison['differences'] = {
        'score_diff': comparison['lars']['training_score'] - comparison['standard']['training_score'],
        'selection_diff': comparison['lars']['n_features_selected'] - comparison['standard']['n_features_selected'],
        'common_features': len(set(comparison['standard']['selected_features']) & 
                              set(comparison['lars']['selected_features'])),
        'unique_to_standard': len(set(comparison['standard']['selected_features']) - 
                                 set(comparison['lars']['selected_features'])),
        'unique_to_lars': len(set(comparison['lars']['selected_features']) - 
                            set(comparison['standard']['selected_features']))
    }
    
    logger.info(f"Variant comparison complete: "
               f"Standard selected {comparison['standard']['n_features_selected']} features, "
               f"LARS selected {comparison['lars']['n_features_selected']} features")
    
    return comparison
