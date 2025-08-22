# ============================================
# StockPredictionPro - src/models/regression/elastic_net.py
# Elastic Net regression models with combined L1 and L2 regularization
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_regressor import BaseFinancialRegressor, RegressionStrategy

logger = get_logger('models.regression.elastic_net')

# ============================================
# Elastic Net Regression Model
# ============================================

class FinancialElasticNetRegression(BaseFinancialRegressor):
    """
    Elastic Net regression model with combined L1 and L2 regularization
    
    Features:
    - Combined L1 (Lasso) and L2 (Ridge) regularization
    - Automatic hyperparameter selection (alpha and l1_ratio)
    - Feature selection with controlled sparsity
    - Multicollinearity handling
    - Regularization path analysis
    """
    
    def __init__(self,
                 name: str = "elastic_net_regression",
                 alpha: Union[float, List[float]] = 1.0,
                 l1_ratio: Union[float, List[float]] = 0.5,
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
                 auto_hyperparams: bool = True,
                 alpha_range: Optional[Tuple[float, float]] = None,
                 l1_ratio_range: Optional[Tuple[float, float]] = None,
                 cv_folds: int = 5,
                 feature_selection_threshold: Optional[float] = None,
                 **kwargs):
        """
        Initialize Financial Elastic Net Regression
        
        Args:
            name: Model name
            alpha: Regularization strength (or list for CV)
            l1_ratio: Mixing parameter (0=Ridge, 1=Lasso, 0<l1_ratio<1=Elastic Net)
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
            auto_hyperparams: Whether to automatically select alpha and l1_ratio via CV
            alpha_range: Range for alpha search (min_alpha, max_alpha)
            l1_ratio_range: Range for l1_ratio search (min_ratio, max_ratio)
            cv_folds: Number of cross-validation folds
            feature_selection_threshold: Threshold for feature selection
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="elastic_net_regression",
            regression_strategy=RegressionStrategy.PRICE_PREDICTION,
            **kwargs
        )
        
        # Elastic Net parameters
        self.alpha = alpha
        self.l1_ratio = l1_ratio
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
        self.auto_hyperparams = auto_hyperparams
        self.alpha_range = alpha_range or (0.0001, 10.0)
        self.l1_ratio_range = l1_ratio_range or (0.1, 0.9)
        self.cv_folds = cv_folds
        self.feature_selection_threshold = feature_selection_threshold
        
        # Store parameters for model creation
        self.model_params.update({
            'alpha': alpha,
            'l1_ratio': l1_ratio,
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
        
        # Elastic Net-specific attributes
        self.scaler_: Optional[StandardScaler] = None
        self.best_alpha_: Optional[float] = None
        self.best_l1_ratio_: Optional[float] = None
        self.hyperparameter_scores_: Optional[Dict[str, Any]] = None
        self.regularization_path_: Optional[Dict[str, Any]] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.selected_features_: Optional[List[str]] = None
        self.feature_selector_: Optional[SelectFromModel] = None
        self.regularization_stats_: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized Elastic Net regression: {self.name} (auto_hyperparams={self.auto_hyperparams})")
    
    def _create_model(self) -> Union[ElasticNet, ElasticNetCV]:
        """Create the Elastic Net regression model"""
        
        if self.auto_hyperparams:
            # Use ElasticNetCV for automatic hyperparameter selection
            if isinstance(self.alpha, list) and isinstance(self.l1_ratio, list):
                alphas = self.alpha
                l1_ratios = self.l1_ratio
            else:
                # Generate parameter grids for cross-validation
                alphas = np.logspace(
                    np.log10(self.alpha_range[0]), 
                    np.log10(self.alpha_range[1]), 
                    20
                )
                l1_ratios = np.linspace(
                    self.l1_ratio_range[0],
                    self.l1_ratio_range[1],
                    10
                )
            
            cv_params = {
                'l1_ratio': l1_ratios,
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
            
            return ElasticNetCV(**cv_params)
        else:
            # Use regular ElasticNet with fixed hyperparameters
            elastic_params = {k: v for k, v in self.model_params.items() 
                             if k not in ['normalize']}  # normalize deprecated
            return ElasticNet(**elastic_params)
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with automatic scaling"""
        
        # Basic preprocessing
        X_processed = super()._preprocess_features(X)
        
        # Always scale features for Elastic Net regression
        if self.scaler_ is None:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_processed)
            logger.debug("Fitted feature scaler for Elastic Net regression")
        else:
            X_scaled = self.scaler_.transform(X_processed)
        
        return X_scaled
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Enhanced post-training processing for Elastic Net regression"""
        
        # Call parent processing
        super()._post_training_processing(X, y)
        
        # Extract Elastic Net-specific information
        if isinstance(self.model, ElasticNetCV):
            self.best_alpha_ = self.model.alpha_
            self.best_l1_ratio_ = self.model.l1_ratio_
            
            # Store CV scores for different hyperparameter combinations
            if hasattr(self.model, 'mse_path_'):
                self.hyperparameter_scores_ = {
                    'alphas': self.model.alphas,
                    'l1_ratios': self.model.l1_ratio,
                    'mse_path': self.model.mse_path_,
                    'best_alpha_idx': np.where(self.model.alphas == self.best_alpha_)[0][0],
                    'best_l1_ratio_idx': np.where(self.model.l1_ratio == self.best_l1_ratio_)[0][0]
                }
            
            logger.info(f"Best hyperparameters selected via CV: "
                       f"alpha={self.best_alpha_:.6f}, l1_ratio={self.best_l1_ratio_:.3f}")
        else:
            self.best_alpha_ = self.model.alpha
            self.best_l1_ratio_ = self.model.l1_ratio
        
        # Extract coefficients
        self.coefficients_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        # Set feature importances (absolute coefficients)
        self.feature_importances_ = np.abs(self.coefficients_)
        
        # Analyze regularization effects
        self._analyze_regularization_effects()
        
        # Identify selected features (non-zero coefficients)
        self._identify_selected_features()
        
        # Calculate regularization path if reasonable number of features
        if X.shape[1] <= 150:
            self._calculate_regularization_path(X, y)
    
    def _analyze_regularization_effects(self):
        """Analyze the combined L1 and L2 regularization effects"""
        
        n_features = len(self.coefficients_)
        non_zero_coefs = np.sum(np.abs(self.coefficients_) > 1e-10)
        zero_coefs = n_features - non_zero_coefs
        sparsity_ratio = zero_coefs / n_features
        
        # Calculate coefficient statistics
        non_zero_indices = np.abs(self.coefficients_) > 1e-10
        if non_zero_indices.any():
            active_coefs = self.coefficients_[non_zero_indices]
            l1_norm = np.sum(np.abs(active_coefs))
            l2_norm = np.sqrt(np.sum(active_coefs ** 2))
            max_coef = np.max(np.abs(active_coefs))
            mean_coef = np.mean(np.abs(active_coefs))
        else:
            l1_norm = l2_norm = max_coef = mean_coef = 0.0
        
        # Calculate regularization contribution
        l1_contribution = self.best_l1_ratio_
        l2_contribution = 1 - self.best_l1_ratio_
        
        self.regularization_stats_ = {
            'alpha': float(self.best_alpha_),
            'l1_ratio': float(self.best_l1_ratio_),
            'l1_contribution': float(l1_contribution),
            'l2_contribution': float(l2_contribution),
            'regularization_type': self._get_regularization_type(),
            'n_features_total': n_features,
            'n_features_selected': int(non_zero_coefs),
            'n_features_eliminated': int(zero_coefs),
            'sparsity_ratio': float(sparsity_ratio),
            'selection_ratio': float(non_zero_coefs / n_features),
            'l1_norm': float(l1_norm),
            'l2_norm': float(l2_norm),
            'max_coefficient': float(max_coef),
            'mean_active_coefficient': float(mean_coef)
        }
        
        logger.info(f"Elastic Net regularization: {self._get_regularization_type()} "
                   f"({non_zero_coefs}/{n_features} features, {sparsity_ratio:.1%} sparsity)")
    
    def _get_regularization_type(self) -> str:
        """Determine the effective regularization type based on l1_ratio"""
        if self.best_l1_ratio_ == 0.0:
            return "Pure Ridge (L2)"
        elif self.best_l1_ratio_ == 1.0:
            return "Pure Lasso (L1)"
        elif self.best_l1_ratio_ < 0.3:
            return "Ridge-dominated Elastic Net"
        elif self.best_l1_ratio_ > 0.7:
            return "Lasso-dominated Elastic Net"
        else:
            return "Balanced Elastic Net"
    
    def _identify_selected_features(self):
        """Identify features selected by Elastic Net (non-zero coefficients)"""
        
        selected_indices = np.abs(self.coefficients_) > 1e-10
        self.selected_features_ = [self.feature_names[i] for i in range(len(self.feature_names)) 
                                  if i < len(selected_indices) and selected_indices[i]]
        
        logger.debug(f"Selected features: {len(self.selected_features_)}")
    
    def _calculate_regularization_path(self, X: np.ndarray, y: np.ndarray):
        """Calculate Elastic Net regularization path for visualization"""
        
        try:
            from sklearn.linear_model import enet_path
            
            # Use fixed l1_ratio for path calculation
            l1_ratio = self.best_l1_ratio_
            
            # Generate alpha values
            n_alphas = 100
            alpha_min, alpha_max = self.alpha_range
            
            # Calculate the regularization path
            alphas_path, coefs_path, _ = enet_path(
                X, y,
                l1_ratio=l1_ratio,
                alphas=None,  # Let sklearn choose alphas
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
                'l1_ratio_used': l1_ratio,
                'best_alpha_idx': np.argmin(np.abs(alphas_path - self.best_alpha_))
            }
            
            logger.debug(f"Calculated Elastic Net regularization path with {len(alphas_path)} alpha values")
            
        except Exception as e:
            logger.debug(f"Could not calculate regularization path: {e}")
            self.regularization_path_ = None
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients with regularization information
        
        Returns:
            DataFrame with coefficients and regularization effects
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get coefficients")
        
        if self.coefficients_ is None:
            raise BusinessLogicError("Coefficients not available")
        
        # Calculate regularization effects
        l1_effect = self._calculate_l1_effect()
        l2_effect = self._calculate_l2_effect()
        
        # Create coefficients DataFrame
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.coefficients_,
            'abs_coefficient': np.abs(self.coefficients_),
            'selected': np.abs(self.coefficients_) > 1e-10,
            'l1_effect': l1_effect,
            'l2_effect': l2_effect,
            'total_regularization': l1_effect + l2_effect,
            'importance_rank': np.argsort(np.argsort(-np.abs(self.coefficients_))) + 1
        })
        
        # Add intercept if fitted
        if self.fit_intercept and self.intercept_ is not None:
            intercept_row = pd.DataFrame({
                'feature': ['intercept'],
                'coefficient': [self.intercept_],
                'abs_coefficient': [abs(self.intercept_)],
                'selected': [True],
                'l1_effect': [0.0],
                'l2_effect': [0.0],
                'total_regularization': [0.0],
                'importance_rank': [0]
            })
            coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
        
        # Sort by absolute coefficient value
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        return coef_df
    
    def _calculate_l1_effect(self) -> np.ndarray:
        """Calculate L1 regularization effect on each coefficient"""
        # L1 effect is related to sparsity - coefficients that would be zero with pure L1
        l1_strength = self.best_alpha_ * self.best_l1_ratio_
        # Simplified estimation of L1 effect
        return np.minimum(np.abs(self.coefficients_), l1_strength)
    
    def _calculate_l2_effect(self) -> np.ndarray:
        """Calculate L2 regularization effect on each coefficient"""
        # L2 effect is shrinkage without sparsity
        l2_strength = self.best_alpha_ * (1 - self.best_l1_ratio_)
        # Simplified estimation of L2 effect
        return l2_strength * np.abs(self.coefficients_) / (1 + l2_strength)
    
    def get_selected_features(self) -> List[str]:
        """Get list of features selected by Elastic Net"""
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get selected features")
        
        return self.selected_features_.copy() if self.selected_features_ else []
    
    def create_feature_selector(self) -> SelectFromModel:
        """Create a feature selector based on Elastic Net coefficients"""
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to create feature selector")
        
        threshold = self.feature_selection_threshold or 1e-10
        self.feature_selector_ = SelectFromModel(
            estimator=self.model,
            threshold=threshold,
            prefit=True
        )
        
        return self.feature_selector_
    
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using Elastic Net feature selection"""
        if self.feature_selector_ is None:
            self.create_feature_selector()
        
        selected_mask = self.feature_selector_.get_support()
        selected_features = [name for name, selected in zip(X.columns, selected_mask) if selected]
        
        return X[selected_features]
    
    def get_hyperparameter_validation_curve(self, X: pd.DataFrame, y: pd.Series,
                                          param_name: str = 'alpha',
                                          n_values: int = 20) -> Dict[str, np.ndarray]:
        """
        Generate validation curve for hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target values
            param_name: Parameter to vary ('alpha' or 'l1_ratio')
            n_values: Number of parameter values to test
            
        Returns:
            Dictionary with parameter values and scores
        """
        
        logger.info(f"Generating {param_name} validation curve with {n_values} points")
        
        # Generate parameter range
        if param_name == 'alpha':
            param_range = np.logspace(
                np.log10(self.alpha_range[0]), 
                np.log10(self.alpha_range[1]), 
                n_values
            )
            fixed_params = {'l1_ratio': self.best_l1_ratio_ or 0.5}
        elif param_name == 'l1_ratio':
            param_range = np.linspace(
                self.l1_ratio_range[0],
                self.l1_ratio_range[1],
                n_values
            )
            fixed_params = {'alpha': self.best_alpha_ or 1.0}
        else:
            raise ValueError(f"Unsupported parameter: {param_name}")
        
        # Preprocess data
        X_processed = self._preprocess_features(X)
        y_processed = self._preprocess_targets(y)
        
        # Create base model for validation curve
        base_model = ElasticNet(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            max_iter=self.max_iter,
            tol=self.tol,
            positive=self.positive,
            random_state=self.random_state,
            selection=self.selection,
            **fixed_params
        )
        
        # Generate validation curve
        train_scores, val_scores = validation_curve(
            base_model, X_processed, y_processed,
            param_name=param_name,
            param_range=param_range,
            cv=self.cv_folds,
            scoring='r2',
            n_jobs=-1
        )
        
        return {
            f'{param_name}_values': param_range,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1),
            f'best_{param_name}': param_range[np.argmax(np.mean(val_scores, axis=1))],
            'best_score': np.max(np.mean(val_scores, axis=1))
        }
    
    def plot_regularization_path(self, top_features: int = 20) -> Any:
        """Plot Elastic Net regularization path"""
        if self.regularization_path_ is None:
            logger.warning("Regularization path not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            path = self.regularization_path_
            alphas = path['alphas']
            coefficients = path['coefficients']
            n_selected = path['n_selected_features']
            l1_ratio = path['l1_ratio_used']
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Coefficient paths
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
            ax1.set_title(f'Elastic Net Regularization Path (l1_ratio = {l1_ratio:.2f}) - {self.name}')
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
    
    def plot_hyperparameter_heatmap(self, X: pd.DataFrame, y: pd.Series,
                                   n_alphas: int = 10, n_l1_ratios: int = 10) -> Any:
        """Plot hyperparameter optimization heatmap"""
        try:
            import matplotlib.pyplot as plt
            
            # Generate parameter grids
            alphas = np.logspace(np.log10(self.alpha_range[0]), np.log10(self.alpha_range[1]), n_alphas)
            l1_ratios = np.linspace(self.l1_ratio_range[0], self.l1_ratio_range[1], n_l1_ratios)
            
            # Preprocess data
            X_processed = self._preprocess_features(X)
            y_processed = self._preprocess_targets(y)
            
            # Calculate scores for each combination
            scores = np.zeros((len(l1_ratios), len(alphas)))
            
            for i, l1_ratio in enumerate(l1_ratios):
                for j, alpha in enumerate(alphas):
                    elastic = ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        fit_intercept=self.fit_intercept,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        random_state=self.random_state
                    )
                    
                    elastic.fit(X_processed, y_processed)
                    scores[i, j] = elastic.score(X_processed, y_processed)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            im = ax.imshow(scores, cmap='viridis', aspect='auto', origin='lower')
            
            # Set ticks and labels
            ax.set_xticks(range(len(alphas)))
            ax.set_xticklabels([f'{a:.3f}' for a in alphas], rotation=45)
            ax.set_yticks(range(len(l1_ratios)))
            ax.set_yticklabels([f'{r:.2f}' for r in l1_ratios])
            
            ax.set_xlabel('Alpha (Regularization Strength)')
            ax.set_ylabel('L1 Ratio (Lasso vs Ridge Mix)')
            ax.set_title(f'Elastic Net Hyperparameter Optimization - {self.name}')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('R² Score')
            
            # Mark best parameters if available
            if self.best_alpha_ and self.best_l1_ratio_:
                best_alpha_idx = np.argmin(np.abs(alphas - self.best_alpha_))
                best_l1_idx = np.argmin(np.abs(l1_ratios - self.best_l1_ratio_))
                ax.plot(best_alpha_idx, best_l1_idx, 'r*', markersize=15, 
                       label=f'Best: α={self.best_alpha_:.3f}, l1={self.best_l1_ratio_:.2f}')
                ax.legend()
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_regularization_effects(self) -> Any:
        """Plot combined L1 and L2 regularization effects"""
        try:
            import matplotlib.pyplot as plt
            
            coef_df = self.get_coefficients()
            
            # Remove intercept for plotting
            if 'intercept' in coef_df['feature'].values:
                coef_df = coef_df[coef_df['feature'] != 'intercept']
            
            # Select top features for visualization
            top_features = coef_df.head(20)
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot 1: Coefficient values with regularization breakdown
            y_pos = np.arange(len(top_features))
            
            ax1.barh(y_pos, top_features['coefficient'], color='steelblue', alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_features['feature'], fontsize=8)
            ax1.set_xlabel('Coefficient Value')
            ax1.set_title('Coefficient Values')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: L1 vs L2 effects
            ax2.scatter(top_features['l1_effect'], top_features['l2_effect'], 
                       s=top_features['abs_coefficient'] * 100, alpha=0.7, color='red')
            ax2.set_xlabel('L1 Effect (Sparsity)')
            ax2.set_ylabel('L2 Effect (Shrinkage)')
            ax2.set_title('L1 vs L2 Regularization Effects')
            ax2.grid(True, alpha=0.3)
            
            # Add diagonal line
            max_effect = max(top_features[['l1_effect', 'l2_effect']].max())
            ax2.plot([0, max_effect], [0, max_effect], 'k--', alpha=0.5)
            
            # Plot 3: Regularization summary
            if self.regularization_stats_:
                stats = self.regularization_stats_
                
                # Pie chart of regularization contributions
                sizes = [stats['l1_contribution'], stats['l2_contribution']]
                labels = ['L1 (Lasso)', 'L2 (Ridge)']
                colors = ['lightcoral', 'lightblue']
                
                ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax3.set_title('Regularization Mix')
                
                # Add text with statistics
                text = f"Type: {stats['regularization_type']}\n"
                text += f"Alpha: {stats['alpha']:.4f}\n"
                text += f"L1 ratio: {stats['l1_ratio']:.3f}\n"
                text += f"Features: {stats['n_features_selected']}/{stats['n_features_total']}\n"
                text += f"Sparsity: {stats['sparsity_ratio']:.1%}"
                
                ax3.text(1.3, 0.5, text, transform=ax3.transAxes, fontsize=10,
                        verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle(f'Elastic Net Regularization Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def get_regularization_summary(self) -> Dict[str, Any]:
        """Get comprehensive regularization summary"""
        
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get regularization summary")
        
        summary = {
            'hyperparameters': {
                'best_alpha': float(self.best_alpha_),
                'best_l1_ratio': float(self.best_l1_ratio_),
                'auto_hyperparams': self.auto_hyperparams
            }
        }
        
        # Add regularization statistics
        if self.regularization_stats_:
            summary['regularization_analysis'] = self.regularization_stats_
        
        # Add feature selection information
        if self.selected_features_:
            coef_df = self.get_coefficients()
            selected_df = coef_df[coef_df['selected'] & (coef_df['feature'] != 'intercept')]
            
            summary['feature_selection'] = {
                'selected_features': self.selected_features_,
                'n_selected': len(self.selected_features_),
                'selection_ratio': len(self.selected_features_) / len(self.feature_names),
                'top_5_features': selected_df.head(5)['feature'].tolist(),
                'coefficient_range': [float(selected_df['coefficient'].min()), 
                                    float(selected_df['coefficient'].max())],
                'mean_l1_effect': float(selected_df['l1_effect'].mean()),
                'mean_l2_effect': float(selected_df['l2_effect'].mean())
            }
        
        # Add hyperparameter optimization info if available
        if self.hyperparameter_scores_:
            summary['hyperparameter_optimization'] = {
                'n_alphas_tested': len(self.hyperparameter_scores_['alphas']),
                'n_l1_ratios_tested': len(self.hyperparameter_scores_['l1_ratios']),
                'cv_folds': self.cv_folds
            }
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add Elastic Net-specific information
        summary.update({
            'regularization_type': 'L1 + L2 (Elastic Net)',
            'alpha': float(self.best_alpha_) if self.best_alpha_ else self.alpha,
            'l1_ratio': float(self.best_l1_ratio_) if self.best_l1_ratio_ else self.l1_ratio,
            'effective_regularization': self._get_regularization_type() if self.best_l1_ratio_ else None,
            'auto_hyperparams': self.auto_hyperparams,
            'cv_folds': self.cv_folds,
            'max_iter': self.max_iter,
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

def create_elastic_net_regressor(alpha: Union[float, str] = 'auto',
                                l1_ratio: Union[float, str] = 'auto',
                                cv_folds: int = 5,
                                **kwargs) -> FinancialElasticNetRegression:
    """
    Create an Elastic Net regression model
    
    Args:
        alpha: Regularization strength ('auto' for CV selection, float for fixed)
        l1_ratio: L1/L2 mix ratio ('auto' for CV selection, float for fixed)
        cv_folds: Number of CV folds for hyperparameter selection
        **kwargs: Additional model parameters
        
    Returns:
        Configured Elastic Net regression model
    """
    
    if alpha == 'auto' or l1_ratio == 'auto':
        # Use cross-validation for hyperparameter selection
        default_params = {
            'name': 'elastic_net_cv',
            'auto_hyperparams': True,
            'alpha_range': (0.0001, 10.0),
            'l1_ratio_range': (0.1, 0.9),
            'cv_folds': cv_folds,
            'fit_intercept': True,
            'max_iter': 1000,
            'random_state': 42
        }
    else:
        # Use fixed hyperparameters
        default_params = {
            'name': 'elastic_net',
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'auto_hyperparams': False,
            'fit_intercept': True,
            'max_iter': 1000,
            'random_state': 42
        }
    
    # Override with provided kwargs
    default_params.update(kwargs)
    
    return FinancialElasticNetRegression(**default_params)

def create_balanced_elastic_net(alpha: Union[float, str] = 'auto',
                               **kwargs) -> FinancialElasticNetRegression:
    """Create Elastic Net with balanced L1/L2 regularization"""
    
    return create_elastic_net_regressor(
        alpha=alpha,
        l1_ratio=0.5,  # Equal L1 and L2
        name='balanced_elastic_net',
        **kwargs
    )

def create_lasso_dominated_elastic_net(alpha: Union[float, str] = 'auto',
                                      **kwargs) -> FinancialElasticNetRegression:
    """Create Elastic Net with Lasso-dominated regularization"""
    
    return create_elastic_net_regressor(
        alpha=alpha,
        l1_ratio=0.8,  # 80% L1, 20% L2
        name='lasso_dominated_elastic_net',
        **kwargs
    )

def create_ridge_dominated_elastic_net(alpha: Union[float, str] = 'auto',
                                      **kwargs) -> FinancialElasticNetRegression:
    """Create Elastic Net with Ridge-dominated regularization"""
    
    return create_elastic_net_regressor(
        alpha=alpha,
        l1_ratio=0.2,  # 20% L1, 80% L2
        name='ridge_dominated_elastic_net',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def find_optimal_elastic_net_params(X: pd.DataFrame, y: pd.Series,
                                   alpha_range: Tuple[float, float] = (0.0001, 10.0),
                                   l1_ratio_range: Tuple[float, float] = (0.1, 0.9),
                                   n_alphas: int = 20,
                                   n_l1_ratios: int = 10,
                                   cv_folds: int = 5) -> Dict[str, Any]:
    """
    Find optimal hyperparameters for Elastic Net regression
    
    Args:
        X: Feature matrix
        y: Target values
        alpha_range: Range of alpha values to test
        l1_ratio_range: Range of l1_ratio values to test
        n_alphas: Number of alpha values to test
        n_l1_ratios: Number of l1_ratio values to test
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with optimal hyperparameters and CV results
    """
    
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    
    logger.info(f"Finding optimal Elastic Net hyperparameters with "
               f"{n_alphas} alphas × {n_l1_ratios} l1_ratios = {n_alphas * n_l1_ratios} combinations")
    
    # Generate parameter grids
    alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas)
    l1_ratios = np.linspace(l1_ratio_range[0], l1_ratio_range[1], n_l1_ratios)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform cross-validation
    elastic_cv = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=cv_folds,
        n_jobs=-1,
        max_iter=1000,
        random_state=42
    )
    
    elastic_cv.fit(X_scaled, y)
    
    # Calculate sparsity and regularization type
    n_selected = np.sum(np.abs(elastic_cv.coef_) > 1e-10)
    sparsity = 1 - (n_selected / len(elastic_cv.coef_))
    
    # Determine regularization type
    if elastic_cv.l1_ratio_ == 0.0:
        reg_type = "Pure Ridge (L2)"
    elif elastic_cv.l1_ratio_ == 1.0:
        reg_type = "Pure Lasso (L1)"
    elif elastic_cv.l1_ratio_ < 0.3:
        reg_type = "Ridge-dominated Elastic Net"
    elif elastic_cv.l1_ratio_ > 0.7:
        reg_type = "Lasso-dominated Elastic Net"
    else:
        reg_type = "Balanced Elastic Net"
    
    result = {
        'optimal_alpha': elastic_cv.alpha_,
        'optimal_l1_ratio': elastic_cv.l1_ratio_,
        'optimal_score': elastic_cv.score(X_scaled, y),
        'regularization_type': reg_type,
        'n_features_selected': int(n_selected),
        'sparsity': float(sparsity),
        'l1_contribution': float(elastic_cv.l1_ratio_),
        'l2_contribution': float(1 - elastic_cv.l1_ratio_),
        'alphas_tested': alphas,
        'l1_ratios_tested': l1_ratios,
        'cv_folds': cv_folds
    }
    
    logger.info(f"Optimal Elastic Net hyperparameters: "
               f"alpha={result['optimal_alpha']:.6f}, l1_ratio={result['optimal_l1_ratio']:.3f} "
               f"({result['regularization_type']}, {result['n_features_selected']} features selected)")
    
    return result

def compare_regularization_methods(X: pd.DataFrame, y: pd.Series,
                                 alpha: float = 1.0) -> Dict[str, Any]:
    """
    Compare Ridge, Lasso, and Elastic Net regularization
    
    Args:
        X: Feature matrix
        y: Target values
        alpha: Regularization strength to use for comparison
        
    Returns:
        Dictionary with comparison results
    """
    
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
    logger.info("Comparing regularization methods: Ridge, Lasso, Elastic Net")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create models
    models = {
        'ridge': Ridge(alpha=alpha, random_state=42),
        'lasso': Lasso(alpha=alpha, random_state=42, max_iter=1000),
        'elastic_net': ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        # Fit model
        model.fit(X_scaled, y)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        
        # Calculate statistics
        n_selected = np.sum(np.abs(model.coef_) > 1e-10)
        sparsity = 1 - (n_selected / len(model.coef_))
        l1_norm = np.sum(np.abs(model.coef_))
        l2_norm = np.sqrt(np.sum(model.coef_ ** 2))
        
        results[name] = {
            'training_score': model.score(X_scaled, y),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_features_selected': int(n_selected),
            'sparsity': float(sparsity),
            'l1_norm': float(l1_norm),
            'l2_norm': float(l2_norm),
            'max_coefficient': float(np.max(np.abs(model.coef_))),
            'mean_coefficient': float(np.mean(np.abs(model.coef_[model.coef_ != 0]))) if n_selected > 0 else 0.0
        }
    
    # Add comparison summary
    results['comparison'] = {
        'best_cv_method': max(results.keys(), key=lambda k: results[k]['cv_mean']),
        'sparsest_method': max(results.keys(), key=lambda k: results[k]['sparsity']),
        'most_stable_method': min(results.keys(), key=lambda k: results[k]['cv_std'])
    }
    
    logger.info(f"Regularization comparison complete. Best CV: {results['comparison']['best_cv_method']}")
    
    return results
