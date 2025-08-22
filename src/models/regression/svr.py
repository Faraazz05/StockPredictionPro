# ============================================
# StockPredictionPro - src/models/regression/svr.py
# Support Vector Regression models for financial prediction with kernel methods
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.svm import SVR, NuSVR
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_regressor import BaseFinancialRegressor, RegressionStrategy

logger = get_logger('models.regression.svr')

# ============================================
# Support Vector Regression Model
# ============================================

class FinancialSVRegressor(BaseFinancialRegressor):
    """
    Support Vector Regression model optimized for financial data
    
    Features:
    - Multiple kernel functions (linear, poly, rbf, sigmoid)
    - Epsilon-insensitive loss function
    - Nu-SVR variant for automatic epsilon tuning
    - Advanced hyperparameter optimization
    - Support vector analysis
    - Kernel parameter exploration
    """
    
    def __init__(self,
                 name: str = "svr_regressor",
                 kernel: str = 'rbf',
                 degree: int = 3,
                 gamma: Union[str, float] = 'scale',
                 coef0: float = 0.0,
                 tol: float = 1e-3,
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 shrinking: bool = True,
                 cache_size: float = 200,
                 verbose: bool = False,
                 max_iter: int = -1,
                 svr_variant: str = 'epsilon_svr',
                 nu: float = 0.5,
                 scaler_type: str = 'standard',
                 auto_scale: bool = True,
                 **kwargs):
        """
        Initialize Financial Support Vector Regressor
        
        Args:
            name: Model name
            kernel: Kernel function ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
            degree: Degree for polynomial kernel
            gamma: Kernel coefficient for rbf, poly, sigmoid
            coef0: Independent term in kernel function
            tol: Tolerance for stopping criterion
            C: Regularization parameter
            epsilon: Epsilon parameter in epsilon-SVR model
            shrinking: Whether to use shrinking heuristic
            cache_size: Cache size in MB
            verbose: Enable verbose output
            max_iter: Hard limit on iterations (-1 for no limit)
            svr_variant: SVR variant ('epsilon_svr', 'nu_svr')
            nu: Upper bound on fraction of training errors (Nu-SVR only)
            scaler_type: Type of scaler ('standard', 'robust', 'minmax')
            auto_scale: Whether to automatically scale features
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="svr_regressor",
            regression_strategy=RegressionStrategy.PRICE_PREDICTION,
            **kwargs
        )
        
        # SVR parameters
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.svr_variant = svr_variant
        self.nu = nu
        self.scaler_type = scaler_type
        self.auto_scale = auto_scale
        
        # Store parameters for model creation
        self.model_params = {
            'kernel': kernel,
            'degree': degree,
            'gamma': gamma,
            'coef0': coef0,
            'tol': tol,
            'C': C,
            'shrinking': shrinking,
            'cache_size': cache_size,
            'verbose': verbose,
            'max_iter': max_iter
        }
        
        # Add variant-specific parameters
        if svr_variant == 'epsilon_svr':
            self.model_params['epsilon'] = epsilon
        elif svr_variant == 'nu_svr':
            self.model_params['nu'] = nu
        
        # SVR-specific attributes
        self.scaler_: Optional[Union[StandardScaler, RobustScaler]] = None
        self.support_vectors_: Optional[np.ndarray] = None
        self.support_: Optional[np.ndarray] = None
        self.n_support_: Optional[int] = None
        self.dual_coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.kernel_params_: Optional[Dict[str, Any]] = None
        self.hyperparameter_analysis_: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized {svr_variant.replace('_', '-').upper()} with {kernel} kernel: {self.name}")
    
    def _create_model(self) -> Union[SVR, NuSVR]:
        """Create the Support Vector Regression model"""
        
        if self.svr_variant == 'nu_svr':
            # Nu-SVR: automatically determines epsilon
            nu_params = {k: v for k, v in self.model_params.items() if k != 'epsilon'}
            return NuSVR(**nu_params)
        else:
            # Standard epsilon-SVR
            epsilon_params = {k: v for k, v in self.model_params.items() if k != 'nu'}
            return SVR(**epsilon_params)
    
    def _create_scaler(self) -> Union[StandardScaler, RobustScaler]:
        """Create appropriate scaler based on scaler_type"""
        
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        elif self.scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with scaling (essential for SVR)"""
        
        # Basic preprocessing
        X_processed = super()._preprocess_features(X)
        
        # SVR requires feature scaling
        if self.auto_scale:
            if self.scaler_ is None:
                self.scaler_ = self._create_scaler()
                X_scaled = self.scaler_.fit_transform(X_processed)
                logger.debug(f"Fitted {self.scaler_type} scaler for SVR")
            else:
                X_scaled = self.scaler_.transform(X_processed)
            
            return X_scaled
        else:
            logger.warning("SVR without feature scaling may perform poorly")
            return X_processed
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Enhanced post-training processing for SVR"""
        
        # Call parent processing
        super()._post_training_processing(X, y)
        
        # Extract SVR-specific information
        if hasattr(self.model, 'support_vectors_'):
            self.support_vectors_ = self.model.support_vectors_
        
        if hasattr(self.model, 'support_'):
            self.support_ = self.model.support_
            self.n_support_ = len(self.model.support_)
        
        if hasattr(self.model, 'dual_coef_'):
            self.dual_coef_ = self.model.dual_coef_
        
        if hasattr(self.model, 'intercept_'):
            self.intercept_ = self.model.intercept_[0] if hasattr(self.model.intercept_, '__len__') else self.model.intercept_
        
        # Store kernel parameters
        self.kernel_params_ = {
            'kernel': self.kernel,
            'gamma': self.model.gamma if hasattr(self.model, 'gamma') else self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'C': self.C,
            'epsilon': getattr(self.model, 'epsilon', self.epsilon)
        }
        
        # Calculate support vector statistics
        self._analyze_support_vectors()
        
        logger.info(f"SVR training complete: {self.n_support_} support vectors ({self.n_support_/len(X)*100:.1f}% of training data)")
    
    def _analyze_support_vectors(self):
        """Analyze support vector characteristics"""
        
        if self.support_vectors_ is None or self.dual_coef_ is None:
            return
        
        # Calculate support vector statistics
        sv_analysis = {
            'n_support_vectors': self.n_support_,
            'support_vector_ratio': float(self.n_support_ / len(self.support_vectors_)) if len(self.support_vectors_) > 0 else 0.0,
            'dual_coef_mean': float(np.mean(np.abs(self.dual_coef_))),
            'dual_coef_std': float(np.std(self.dual_coef_)),
            'dual_coef_max': float(np.max(np.abs(self.dual_coef_))),
            'intercept': float(self.intercept_) if self.intercept_ is not None else 0.0
        }
        
        # Analyze dual coefficients distribution
        if len(self.dual_coef_.flatten()) > 0:
            dual_coef_flat = self.dual_coef_.flatten()
            sv_analysis.update({
                'dual_coef_skewness': float(pd.Series(dual_coef_flat).skew()),
                'dual_coef_kurtosis': float(pd.Series(dual_coef_flat).kurtosis()),
                'bound_support_vectors': int(np.sum(np.abs(dual_coef_flat) >= self.C * 0.99))  # Nearly at bound
            })
        
        self.support_vector_analysis_ = sv_analysis
    
    def get_support_vector_info(self) -> Dict[str, Any]:
        """
        Get comprehensive support vector information
        
        Returns:
            Dictionary with support vector analysis
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get support vector info")
        
        if not hasattr(self, 'support_vector_analysis_'):
            return {'error': 'Support vector analysis not available'}
        
        return self.support_vector_analysis_.copy()
    
    def get_kernel_matrix(self, X: pd.DataFrame, X2: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Compute kernel matrix for given data
        
        Args:
            X: First set of samples
            X2: Second set of samples (if None, compute X vs X)
            
        Returns:
            Kernel matrix
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to compute kernel matrix")
        
        from sklearn.metrics.pairwise import pairwise_kernels
        
        X_processed = self._preprocess_features(X)
        
        if X2 is not None:
            X2_processed = self._preprocess_features(X2)
        else:
            X2_processed = X_processed
        
        # Get kernel parameters
        kernel_params = {}
        if self.kernel == 'rbf' or self.kernel == 'sigmoid':
            kernel_params['gamma'] = self.model.gamma
        elif self.kernel == 'poly':
            kernel_params['gamma'] = self.model.gamma
            kernel_params['degree'] = self.degree
            kernel_params['coef0'] = self.coef0
        elif self.kernel == 'sigmoid':
            kernel_params['gamma'] = self.model.gamma
            kernel_params['coef0'] = self.coef0
        
        return pairwise_kernels(X_processed, X2_processed, 
                               metric=self.kernel, **kernel_params)
    
    def predict_with_decision_function(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and return decision function values
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, decision_values)
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        X_processed = self._preprocess_features(X)
        
        predictions = self.model.predict(X_processed)
        decision_values = self.model.decision_function(X_processed)
        
        # Inverse transform predictions if target transformation was applied
        if self.target_transformer_ is not None:
            predictions = self._inverse_transform_targets(predictions)
        
        return predictions, decision_values
    
    def get_hyperparameter_validation_curve(self, X: pd.DataFrame, y: pd.Series,
                                          param_name: str, param_range: List[Any],
                                          cv: int = 5) -> Dict[str, np.ndarray]:
        """
        Generate validation curve for hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target values
            param_name: Parameter name to vary
            param_range: Range of parameter values
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with parameter values and scores
        """
        
        logger.info(f"Generating SVR validation curve for {param_name}")
        
        # Preprocess data
        X_processed = self._preprocess_features(X)
        y_processed = self._preprocess_targets(y)
        
        # Create base model
        base_model = self._create_model()
        
        # Generate validation curve
        train_scores, val_scores = validation_curve(
            base_model, X_processed, y_processed,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring='r2',
            n_jobs=1  # SVR doesn't parallelize well
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
    
    def plot_support_vectors(self, X: pd.DataFrame, y: pd.Series, 
                           feature_indices: Tuple[int, int] = (0, 1)) -> Any:
        """
        Plot support vectors in 2D feature space
        
        Args:
            X: Feature matrix
            y: Target values
            feature_indices: Indices of features to plot
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to plot support vectors")
        
        try:
            import matplotlib.pyplot as plt
            
            X_processed = self._preprocess_features(X)
            
            if X_processed.shape[1] < 2:
                logger.warning("Need at least 2 features for 2D support vector plot")
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get feature indices
            feat1, feat2 = feature_indices
            
            # Plot all points
            scatter = ax.scatter(X_processed[:, feat1], X_processed[:, feat2], 
                               c=y, cmap='viridis', alpha=0.6, s=30, label='Training Data')
            
            # Highlight support vectors
            if self.support_ is not None and len(self.support_) > 0:
                support_points = X_processed[self.support_]
                ax.scatter(support_points[:, feat1], support_points[:, feat2], 
                          c='red', s=100, alpha=0.8, marker='o', 
                          facecolors='none', edgecolors='red', linewidths=2,
                          label=f'Support Vectors ({len(self.support_)})')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Target Value')
            
            # Labels and title
            feature_names = self.feature_names if self.feature_names else [f'Feature {i}' for i in range(X_processed.shape[1])]
            ax.set_xlabel(feature_names[feat1] if feat1 < len(feature_names) else f'Feature {feat1}')
            ax.set_ylabel(feature_names[feat2] if feat2 < len(feature_names) else f'Feature {feat2}')
            ax.set_title(f'SVR Support Vectors - {self.name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_validation_curve(self, X: pd.DataFrame, y: pd.Series,
                             param_name: str, param_range: List[Any]) -> Any:
        """
        Plot validation curve for hyperparameter
        
        Args:
            X: Feature matrix
            y: Target values
            param_name: Parameter name
            param_range: Parameter range
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            # Generate validation curve
            curve_data = self.get_hyperparameter_validation_curve(
                X, y, param_name, param_range
            )
            
            param_values = curve_data[f'{param_name}_values']
            train_mean = curve_data['train_scores_mean']
            train_std = curve_data['train_scores_std']
            val_mean = curve_data['val_scores_mean']
            val_std = curve_data['val_scores_std']
            
            plt.figure(figsize=(12, 8))
            
            # Plot training scores
            plt.plot(param_values, train_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(param_values, train_mean - train_std, train_mean + train_std,
                           alpha=0.2, color='blue')
            
            # Plot validation scores
            plt.plot(param_values, val_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(param_values, val_mean - val_std, val_mean + val_std,
                           alpha=0.2, color='red')
            
            # Mark best parameter
            best_param = curve_data[f'best_{param_name}']
            best_score = curve_data['best_score']
            plt.axvline(x=best_param, color='green', linestyle='--',
                       label=f'Best {param_name} = {best_param} (Score = {best_score:.3f})')
            
            # Handle log scale for certain parameters
            if param_name in ['C', 'epsilon', 'gamma'] and isinstance(param_values[0], (int, float)):
                plt.xscale('log')
            
            plt.xlabel(param_name.upper())
            plt.ylabel('R² Score')
            plt.title(f'SVR Validation Curve - {param_name} - {self.name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_decision_surface(self, X: pd.DataFrame, y: pd.Series,
                             feature_indices: Tuple[int, int] = (0, 1),
                             resolution: int = 100) -> Any:
        """
        Plot decision surface for 2D feature space
        
        Args:
            X: Feature matrix
            y: Target values
            feature_indices: Indices of features to plot
            resolution: Grid resolution for decision surface
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to plot decision surface")
        
        try:
            import matplotlib.pyplot as plt
            
            X_processed = self._preprocess_features(X)
            
            if X_processed.shape[1] < 2:
                logger.warning("Need at least 2 features for 2D decision surface plot")
                return None
            
            feat1, feat2 = feature_indices
            
            # Create mesh grid
            x1_min, x1_max = X_processed[:, feat1].min() - 0.1, X_processed[:, feat1].max() + 0.1
            x2_min, x2_max = X_processed[:, feat2].min() - 0.1, X_processed[:, feat2].max() + 0.1
            
            xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, resolution),
                                  np.linspace(x2_min, x2_max, resolution))
            
            # Create grid points with other features set to mean values
            grid_points = np.zeros((resolution * resolution, X_processed.shape[1]))
            grid_points[:, feat1] = xx1.ravel()
            grid_points[:, feat2] = xx2.ravel()
            
            # Set other features to mean values
            for i in range(X_processed.shape[1]):
                if i not in [feat1, feat2]:
                    grid_points[:, i] = np.mean(X_processed[:, i])
            
            # Make predictions on grid
            Z = self.model.predict(grid_points)
            Z = Z.reshape(xx1.shape)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot decision surface
            contour = ax.contourf(xx1, xx2, Z, levels=20, alpha=0.8, cmap='viridis')
            plt.colorbar(contour, ax=ax, label='Predicted Value')
            
            # Plot training points
            scatter = ax.scatter(X_processed[:, feat1], X_processed[:, feat2],
                               c=y, cmap='viridis', s=50, alpha=0.8, edgecolors='black')
            
            # Highlight support vectors
            if self.support_ is not None and len(self.support_) > 0:
                support_points = X_processed[self.support_]
                ax.scatter(support_points[:, feat1], support_points[:, feat2],
                          c='red', s=100, alpha=1.0, marker='o',
                          facecolors='none', edgecolors='red', linewidths=2,
                          label=f'Support Vectors ({len(self.support_)})')
            
            # Labels and title
            feature_names = self.feature_names if self.feature_names else [f'Feature {i}' for i in range(X_processed.shape[1])]
            ax.set_xlabel(feature_names[feat1] if feat1 < len(feature_names) else f'Feature {feat1}')
            ax.set_ylabel(feature_names[feat2] if feat2 < len(feature_names) else f'Feature {feat2}')
            ax.set_title(f'SVR Decision Surface ({self.kernel} kernel) - {self.name}')
            
            if self.support_ is not None and len(self.support_) > 0:
                ax.legend()
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def get_svr_summary(self) -> Dict[str, Any]:
        """Get comprehensive SVR summary"""
        
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get SVR summary")
        
        summary = {
            'svr_variant': self.svr_variant,
            'kernel_info': self.kernel_params_,
            'hyperparameters': {
                'C': self.C,
                'epsilon': getattr(self, 'epsilon', None),
                'nu': getattr(self, 'nu', None),
                'gamma': self.gamma,
                'degree': self.degree if self.kernel == 'poly' else None,
                'coef0': self.coef0 if self.kernel in ['poly', 'sigmoid'] else None
            },
            'scaler_info': {
                'scaler_type': self.scaler_type,
                'auto_scale': self.auto_scale
            }
        }
        
        # Add support vector information
        if hasattr(self, 'support_vector_analysis_'):
            summary['support_vectors'] = self.support_vector_analysis_
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add SVR-specific information
        summary.update({
            'model_family': 'Support Vector Machine',
            'svr_variant': self.svr_variant.replace('_', '-').upper(),
            'kernel': self.kernel,
            'regularization_parameter': self.C,
            'epsilon': getattr(self, 'epsilon', None),
            'nu': getattr(self, 'nu', None) if self.svr_variant == 'nu_svr' else None,
            'gamma': self.gamma,
            'scaler_type': self.scaler_type,
            'n_support_vectors': self.n_support_ if hasattr(self, 'n_support_') else None,
            'support_vector_ratio': getattr(self, 'support_vector_analysis_', {}).get('support_vector_ratio'),
            'kernel_complexity': self._get_kernel_complexity()
        })
        
        # Add SVR summary
        if self.is_fitted:
            try:
                summary['svr_summary'] = self.get_svr_summary()
            except Exception as e:
                logger.debug(f"Could not generate SVR summary: {e}")
        
        return summary
    
    def _get_kernel_complexity(self) -> str:
        """Get kernel complexity description"""
        if self.kernel == 'linear':
            return 'Low (Linear)'
        elif self.kernel == 'poly':
            return f'Medium (Polynomial degree {self.degree})'
        elif self.kernel == 'rbf':
            return 'High (Radial Basis Function)'
        elif self.kernel == 'sigmoid':
            return 'Medium (Sigmoid)'
        else:
            return 'Unknown'

# ============================================
# Factory Functions
# ============================================

def create_svr_regressor(kernel: str = 'rbf',
                        svr_variant: str = 'epsilon_svr',
                        performance_preset: str = 'balanced',
                        **kwargs) -> FinancialSVRegressor:
    """
    Create a Support Vector Regression model
    
    Args:
        kernel: Kernel function ('linear', 'poly', 'rbf', 'sigmoid')
        svr_variant: SVR variant ('epsilon_svr', 'nu_svr')
        performance_preset: Performance preset ('fast', 'balanced', 'accurate')
        **kwargs: Additional model parameters
        
    Returns:
        Configured SVR regression model
    """
    
    # Base configuration
    base_config = {
        'name': f'{svr_variant}_{kernel}',
        'kernel': kernel,
        'svr_variant': svr_variant,
        'auto_scale': True,
        'scaler_type': 'standard'
    }
    
    # Performance presets
    if performance_preset == 'fast':
        preset_config = {
            'C': 1.0,
            'epsilon': 0.1,
            'nu': 0.5,
            'gamma': 'scale',
            'tol': 1e-2,
            'cache_size': 100
        }
    elif performance_preset == 'balanced':
        preset_config = {
            'C': 1.0,
            'epsilon': 0.1,
            'nu': 0.5,
            'gamma': 'scale',
            'tol': 1e-3,
            'cache_size': 200
        }
    elif performance_preset == 'accurate':
        preset_config = {
            'C': 1.0,
            'epsilon': 0.01,
            'nu': 0.3,
            'gamma': 'scale',
            'tol': 1e-4,
            'cache_size': 500,
            'scaler_type': 'robust'  # More robust to outliers
        }
    else:
        raise ValueError(f"Unknown performance preset: {performance_preset}")
    
    # Kernel-specific adjustments
    if kernel == 'linear':
        preset_config.update({
            'gamma': 'auto',  # Not used for linear kernel
        })
    elif kernel == 'poly':
        preset_config.update({
            'degree': 3,
            'coef0': 1.0
        })
    elif kernel == 'sigmoid':
        preset_config.update({
            'coef0': 1.0
        })
    
    # Combine configurations
    config = {**base_config, **preset_config}
    config.update(kwargs)  # Override with user parameters
    
    return FinancialSVRegressor(**config)

def create_linear_svr(**kwargs) -> FinancialSVRegressor:
    """Create Linear SVR for high-dimensional data"""
    
    return create_svr_regressor(
        kernel='linear',
        performance_preset='balanced',
        name='linear_svr',
        **kwargs
    )

def create_rbf_svr(**kwargs) -> FinancialSVRegressor:
    """Create RBF SVR for non-linear patterns"""
    
    return create_svr_regressor(
        kernel='rbf',
        performance_preset='balanced',
        name='rbf_svr',
        **kwargs
    )

def create_polynomial_svr(degree: int = 3, **kwargs) -> FinancialSVRegressor:
    """Create Polynomial SVR"""
    
    return create_svr_regressor(
        kernel='poly',
        degree=degree,
        performance_preset='balanced',
        name=f'poly{degree}_svr',
        **kwargs
    )

def create_nu_svr(kernel: str = 'rbf', **kwargs) -> FinancialSVRegressor:
    """Create Nu-SVR with automatic epsilon tuning"""
    
    return create_svr_regressor(
        kernel=kernel,
        svr_variant='nu_svr',
        performance_preset='balanced',
        name=f'nu_svr_{kernel}',
        **kwargs
    )

def create_robust_svr(**kwargs) -> FinancialSVRegressor:
    """Create robust SVR for noisy financial data"""
    
    return create_svr_regressor(
        kernel='rbf',
        performance_preset='accurate',
        scaler_type='robust',
        epsilon=0.05,  # More tolerant to noise
        C=0.1,         # Less regularization
        name='robust_svr',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def tune_svr_hyperparameters(X: pd.DataFrame, y: pd.Series,
                            kernel: str = 'rbf',
                            param_grid: Optional[Dict[str, List[Any]]] = None,
                            cv: int = 5,
                            scoring: str = 'r2',
                            n_jobs: int = 1) -> Dict[str, Any]:
    """
    Tune SVR hyperparameters using grid search
    
    Args:
        X: Feature matrix
        y: Target values
        kernel: Kernel function
        param_grid: Parameter grid for search
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs (SVR doesn't parallelize well)
        
    Returns:
        Dictionary with best parameters and scores
    """
    
    logger.info(f"Starting SVR hyperparameter tuning for {kernel} kernel")
    
    # Default parameter grids for different kernels
    if param_grid is None:
        if kernel == 'linear':
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'epsilon': [0.01, 0.1, 0.2]
            }
        elif kernel == 'rbf':
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'epsilon': [0.01, 0.1, 0.2]
            }
        elif kernel == 'poly':
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 0.01, 0.1, 1.0],
                'degree': [2, 3, 4],
                'epsilon': [0.01, 0.1, 0.2]
            }
        else:
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'epsilon': [0.01, 0.1, 0.2]
            }
    
    # Create base model
    base_model = SVR(kernel=kernel)
    
    # Scale data for SVR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_scaled, y)
    
    # Extract results
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_,
        'kernel': kernel,
        'n_support_vectors': len(grid_search.best_estimator_.support_)
    }
    
    logger.info(f"SVR hyperparameter tuning complete. Best score: {results['best_score']:.4f}")
    logger.info(f"Best parameters: {results['best_params']}")
    logger.info(f"Support vectors: {results['n_support_vectors']}")
    
    return results

def compare_svr_kernels(X: pd.DataFrame, y: pd.Series,
                       kernels: List[str] = ['linear', 'poly', 'rbf', 'sigmoid'],
                       cv: int = 5) -> Dict[str, Any]:
    """
    Compare different SVR kernels
    
    Args:
        X: Feature matrix
        y: Target values
        kernels: List of kernels to compare
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with comparison results
    """
    
    from sklearn.model_selection import cross_val_score
    
    logger.info(f"Comparing SVR kernels: {kernels}")
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for kernel in kernels:
        logger.info(f"Evaluating {kernel} kernel")
        
        # Create model with default parameters
        if kernel == 'poly':
            model = SVR(kernel=kernel, degree=3, gamma='scale')
        else:
            model = SVR(kernel=kernel, gamma='scale')
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
        
        # Fit model to get support vector information
        model.fit(X_scaled, y)
        
        results[kernel] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'n_support_vectors': len(model.support_),
            'support_vector_ratio': len(model.support_) / len(X),
            'model': model
        }
    
    # Add comparison summary
    results['comparison'] = {
        'best_kernel': max(results.keys(), key=lambda k: results[k]['cv_mean'] if k != 'comparison' else -np.inf),
        'most_sparse': min(results.keys(), key=lambda k: results[k]['support_vector_ratio'] if k != 'comparison' else np.inf),
        'most_stable': min(results.keys(), key=lambda k: results[k]['cv_std'] if k != 'comparison' else np.inf)
    }
    
    logger.info(f"Kernel comparison complete. Best kernel: {results['comparison']['best_kernel']}")
    
    return results

def analyze_svr_complexity(X: pd.DataFrame, y: pd.Series,
                          C_range: List[float] = [0.01, 0.1, 1.0, 10.0, 100.0],
                          kernel: str = 'rbf') -> Dict[str, Any]:
    """
    Analyze SVR model complexity vs performance trade-off
    
    Args:
        X: Feature matrix
        y: Target values
        C_range: Range of C values to test
        kernel: Kernel function
        
    Returns:
        Dictionary with complexity analysis
    """
    
    from sklearn.model_selection import cross_val_score
    
    logger.info(f"Analyzing SVR complexity for {kernel} kernel")
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {
        'C_values': C_range,
        'cv_scores': [],
        'n_support_vectors': [],
        'support_vector_ratios': [],
        'training_scores': []
    }
    
    for C in C_range:
        logger.info(f"Testing C = {C}")
        
        # Create model
        model = SVR(kernel=kernel, C=C, gamma='scale')
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        results['cv_scores'].append(cv_scores.mean())
        
        # Fit model to get complexity metrics
        model.fit(X_scaled, y)
        
        # Training score
        train_score = model.score(X_scaled, y)
        results['training_scores'].append(train_score)
        
        # Support vector information
        n_sv = len(model.support_)
        sv_ratio = n_sv / len(X)
        
        results['n_support_vectors'].append(n_sv)
        results['support_vector_ratios'].append(sv_ratio)
    
    # Find optimal C
    best_idx = np.argmax(results['cv_scores'])
    results['optimal_C'] = C_range[best_idx]
    results['optimal_score'] = results['cv_scores'][best_idx]
    results['optimal_n_sv'] = results['n_support_vectors'][best_idx]
    
    logger.info(f"Complexity analysis complete. Optimal C: {results['optimal_C']}")
    
    return results

def create_svr_ensemble(X: pd.DataFrame, y: pd.Series,
                       kernels: List[str] = ['linear', 'rbf', 'poly'],
                       n_models: int = 3) -> Dict[str, Any]:
    """
    Create ensemble of SVR models with different kernels
    
    Args:
        X: Feature matrix
        y: Target values
        kernels: List of kernels to use
        n_models: Number of models per kernel
        
    Returns:
        Dictionary with ensemble models and predictions
    """
    
    logger.info(f"Creating SVR ensemble with kernels: {kernels}")
    
    ensemble_models = {}
    predictions = {}
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for kernel in kernels:
        kernel_models = []
        kernel_predictions = []
        
        for i in range(n_models):
            # Create model with slight parameter variations
            if kernel == 'linear':
                C_values = [0.1, 1.0, 10.0]
                model = SVR(kernel=kernel, C=C_values[i % len(C_values)])
            elif kernel == 'rbf':
                gamma_values = [0.01, 0.1, 1.0]
                model = SVR(kernel=kernel, gamma=gamma_values[i % len(gamma_values)])
            elif kernel == 'poly':
                degree_values = [2, 3, 4]
                model = SVR(kernel=kernel, degree=degree_values[i % len(degree_values)])
            else:
                model = SVR(kernel=kernel)
            
            # Fit model
            model.fit(X_scaled, y)
            
            # Make predictions
            pred = model.predict(X_scaled)
            
            kernel_models.append(model)
            kernel_predictions.append(pred)
        
        ensemble_models[kernel] = kernel_models
        predictions[kernel] = np.array(kernel_predictions)
    
    # Calculate ensemble predictions
    all_predictions = np.concatenate([predictions[k] for k in kernels], axis=0)
    ensemble_prediction = np.mean(all_predictions, axis=0)
    
    # Calculate individual and ensemble scores
    scores = {}
    for kernel in kernels:
        kernel_scores = [r2_score(y, pred) for pred in predictions[kernel]]
        scores[kernel] = {
            'individual_scores': kernel_scores,
            'mean_score': np.mean(kernel_scores),
            'std_score': np.std(kernel_scores)
        }
    
    ensemble_score = r2_score(y, ensemble_prediction)
    scores['ensemble'] = ensemble_score
    
    results = {
        'models': ensemble_models,
        'predictions': predictions,
        'ensemble_prediction': ensemble_prediction,
        'scores': scores,
        'ensemble_score': ensemble_score,
        'scaler': scaler
    }
    
    logger.info(f"SVR ensemble created. Ensemble score: {ensemble_score:.4f}")
    
    return results
