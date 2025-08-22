# ============================================
# StockPredictionPro - src/models/regression/gradient_boosting.py
# Gradient boosting regression models for financial prediction with ensemble learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_regressor import BaseFinancialRegressor, RegressionStrategy

logger = get_logger('models.regression.gradient_boosting')

# ============================================
# Gradient Boosting Regression Model
# ============================================

class FinancialGradientBoostingRegressor(BaseFinancialRegressor):
    """
    Gradient Boosting regression model optimized for financial data
    
    Features:
    - Ensemble of weak decision trees
    - Multiple loss functions (squared_error, absolute_error, huber, quantile)
    - Advanced hyperparameter optimization
    - Feature importance analysis
    - Learning curve analysis
    - Early stopping and regularization
    """
    
    def __init__(self,
                 name: str = "gradient_boosting_regressor",
                 loss: str = 'squared_error',
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 subsample: float = 1.0,
                 criterion: str = 'friedman_mse',
                 min_samples_split: Union[int, float] = 2,
                 min_samples_leaf: Union[int, float] = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 max_depth: Optional[int] = 3,
                 min_impurity_decrease: float = 0.0,
                 random_state: Optional[int] = 42,
                 max_features: Optional[Union[str, int, float]] = None,
                 alpha: float = 0.9,
                 verbose: int = 0,
                 max_leaf_nodes: Optional[int] = None,
                 warm_start: bool = False,
                 validation_fraction: float = 0.1,
                 n_iter_no_change: Optional[int] = None,
                 tol: float = 1e-4,
                 ccp_alpha: float = 0.0,
                 use_histogram_boosting: bool = False,
                 auto_scale: bool = True,
                 **kwargs):
        """
        Initialize Financial Gradient Boosting Regressor
        
        Args:
            name: Model name
            loss: Loss function ('squared_error', 'absolute_error', 'huber', 'quantile')
            learning_rate: Learning rate (shrinkage)
            n_estimators: Number of boosting stages
            subsample: Fraction of samples for stochastic gradient boosting
            criterion: Split quality measure ('friedman_mse', 'squared_error')
            min_samples_split: Minimum samples to split internal node
            min_samples_leaf: Minimum samples at leaf node
            min_weight_fraction_leaf: Minimum weighted fraction at leaf
            max_depth: Maximum depth of trees
            min_impurity_decrease: Minimum impurity decrease for splits
            random_state: Random state for reproducibility
            max_features: Number of features for best split
            alpha: Alpha quantile for huber/quantile loss
            verbose: Verbosity level
            max_leaf_nodes: Maximum leaf nodes
            warm_start: Whether to reuse previous solution
            validation_fraction: Fraction for early stopping validation
            n_iter_no_change: Early stopping patience
            tol: Tolerance for early stopping
            ccp_alpha: Complexity parameter for pruning
            use_histogram_boosting: Whether to use HistGradientBoostingRegressor
            auto_scale: Whether to automatically scale features
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="gradient_boosting_regressor",
            regression_strategy=RegressionStrategy.PRICE_PREDICTION,
            **kwargs
        )
        
        # Gradient Boosting parameters
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.max_features = max_features
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        self.use_histogram_boosting = use_histogram_boosting
        self.auto_scale = auto_scale
        
        # Store parameters for model creation
        self.model_params.update({
            'loss': loss,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'criterion': criterion,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'min_weight_fraction_leaf': min_weight_fraction_leaf,
            'max_depth': max_depth,
            'min_impurity_decrease': min_impurity_decrease,
            'random_state': random_state,
            'max_features': max_features,
            'alpha': alpha,
            'verbose': verbose,
            'max_leaf_nodes': max_leaf_nodes,
            'warm_start': warm_start,
            'validation_fraction': validation_fraction,
            'n_iter_no_change': n_iter_no_change,
            'tol': tol,
            'ccp_alpha': ccp_alpha
        })
        
        # Gradient Boosting-specific attributes
        self.scaler_: Optional[StandardScaler] = None
        self.train_scores_: Optional[np.ndarray] = None
        self.oob_scores_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.learning_curve_: Optional[Dict[str, Any]] = None
        self.boosting_stats_: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized {'Histogram ' if use_histogram_boosting else ''}Gradient Boosting regressor: {self.name}")
    
    def _create_model(self) -> Union[GradientBoostingRegressor, HistGradientBoostingRegressor]:
        """Create the Gradient Boosting regression model"""
        
        if self.use_histogram_boosting:
            # Use HistGradientBoostingRegressor for larger datasets
            hist_params = {
                'loss': self.loss,
                'learning_rate': self.learning_rate,
                'max_iter': self.n_estimators,  # Different parameter name
                'max_leaf_nodes': self.max_leaf_nodes or 31,  # Default for HistGradient
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_samples_leaf,
                'l2_regularization': self.ccp_alpha,
                'max_bins': 255,  # Default
                'categorical_features': None,
                'monotonic_cst': None,
                'warm_start': self.warm_start,
                'early_stopping': self.n_iter_no_change is not None,
                'scoring': 'loss',
                'validation_fraction': self.validation_fraction if self.n_iter_no_change else None,
                'n_iter_no_change': self.n_iter_no_change,
                'tol': self.tol,
                'verbose': self.verbose,
                'random_state': self.random_state
            }
            
            # Remove None values
            hist_params = {k: v for k, v in hist_params.items() if v is not None}
            
            return HistGradientBoostingRegressor(**hist_params)
        else:
            # Use standard GradientBoostingRegressor
            return GradientBoostingRegressor(**self.model_params)
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with optional scaling"""
        
        # Basic preprocessing
        X_processed = super()._preprocess_features(X)
        
        # Apply feature scaling if enabled (generally not needed for tree-based models)
        if self.auto_scale:
            if self.scaler_ is None:
                self.scaler_ = StandardScaler()
                X_scaled = self.scaler_.fit_transform(X_processed)
                logger.debug("Fitted feature scaler for Gradient Boosting regression")
            else:
                X_scaled = self.scaler_.transform(X_processed)
            
            return X_scaled
        
        return X_processed
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Enhanced post-training processing for Gradient Boosting regression"""
        
        # Call parent processing
        super()._post_training_processing(X, y)
        
        # Extract Gradient Boosting-specific information
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        
        # Extract training scores
        if hasattr(self.model, 'train_score_'):
            self.train_scores_ = self.model.train_score_
        
        # Extract out-of-bag scores if available
        if hasattr(self.model, 'oob_scores_') and self.model.oob_scores_ is not None:
            self.oob_scores_ = self.model.oob_scores_
        elif hasattr(self.model, 'oob_improvement_') and self.model.oob_improvement_ is not None:
            # Calculate cumulative OOB scores from improvements
            self.oob_scores_ = np.cumsum(self.model.oob_improvement_)
        
        # Calculate boosting statistics
        self._calculate_boosting_statistics()
        
        # Analyze learning curve
        self._analyze_learning_curve(X, y)
    
    def _calculate_boosting_statistics(self):
        """Calculate statistics about the boosting process"""
        
        self.boosting_stats_ = {
            'n_estimators_used': getattr(self.model, 'n_estimators_', self.n_estimators),
            'early_stopping_used': hasattr(self.model, 'n_estimators_') and self.model.n_estimators_ < self.n_estimators,
            'loss_function': self.loss,
            'learning_rate': self.learning_rate,
            'subsample_ratio': self.subsample,
            'max_depth': self.max_depth,
            'use_histogram_boosting': self.use_histogram_boosting
        }
        
        # Add training performance metrics
        if self.train_scores_ is not None:
            self.boosting_stats_.update({
                'initial_train_score': float(self.train_scores_[0]),
                'final_train_score': float(self.train_scores_[-1]),
                'train_score_improvement': float(self.train_scores_[-1] - self.train_scores_[0]),
                'best_iteration': int(np.argmin(self.train_scores_)) if self.loss in ['squared_error', 'absolute_error'] else int(np.argmax(self.train_scores_))
            })
        
        # Add OOB performance metrics if available
        if self.oob_scores_ is not None:
            self.boosting_stats_.update({
                'oob_available': True,
                'final_oob_score': float(self.oob_scores_[-1]),
                'best_oob_iteration': int(np.argmin(self.oob_scores_)) if self.loss in ['squared_error', 'absolute_error'] else int(np.argmax(self.oob_scores_))
            })
        else:
            self.boosting_stats_['oob_available'] = False
    
    def _analyze_learning_curve(self, X: np.ndarray, y: np.ndarray):
        """Analyze learning curve for the model"""
        
        try:
            if len(X) > 1000:  # Only for reasonably sized datasets
                train_sizes = np.linspace(0.1, 1.0, 10)
                
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    estimator=self._create_model(),
                    X=X, y=y,
                    train_sizes=train_sizes,
                    cv=3,  # Faster with fewer folds
                    scoring='r2',
                    n_jobs=-1,
                    random_state=self.random_state
                )
                
                self.learning_curve_ = {
                    'train_sizes': train_sizes_abs,
                    'train_scores_mean': np.mean(train_scores, axis=1),
                    'train_scores_std': np.std(train_scores, axis=1),
                    'val_scores_mean': np.mean(val_scores, axis=1),
                    'val_scores_std': np.std(val_scores, axis=1)
                }
                
                logger.debug("Calculated learning curve for Gradient Boosting model")
            
        except Exception as e:
            logger.debug(f"Could not calculate learning curve: {e}")
            self.learning_curve_ = None
    
    def get_feature_importance(self, top_n: Optional[int] = None, 
                              importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance rankings with different importance types
        
        Args:
            top_n: Number of top features to return
            importance_type: Type of importance ('gain', 'split', 'permutation')
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get feature importance")
        
        if importance_type == 'gain' and self.feature_importances_ is not None:
            # Standard impurity-based feature importance
            importance_scores = self.feature_importances_
            
        elif importance_type == 'split':
            # Count-based feature importance (number of splits)
            if hasattr(self.model, 'estimators_'):
                split_counts = np.zeros(len(self.feature_names))
                for estimator_stage in self.model.estimators_:
                    for estimator in estimator_stage:
                        if hasattr(estimator, 'tree_'):
                            # Count splits for each feature
                            feature_counts = np.bincount(
                                estimator.tree_.feature[estimator.tree_.feature >= 0],
                                minlength=len(self.feature_names)
                            )
                            split_counts += feature_counts[:len(self.feature_names)]
                
                # Normalize
                importance_scores = split_counts / np.sum(split_counts) if np.sum(split_counts) > 0 else split_counts
            else:
                importance_scores = self.feature_importances_
                
        elif importance_type == 'permutation':
            # Permutation importance (requires fitted model and data)
            logger.warning("Permutation importance requires X and y data - using gain importance instead")
            importance_scores = self.feature_importances_
            
        else:
            raise ValueError(f"Unknown importance type: {importance_type}")
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores,
            'importance_type': importance_type
        }).sort_values('importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def get_boosting_progress(self) -> Dict[str, np.ndarray]:
        """
        Get boosting progress metrics
        
        Returns:
            Dictionary with training and OOB scores over iterations
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get boosting progress")
        
        progress = {
            'iterations': np.arange(1, len(self.train_scores_) + 1) if self.train_scores_ is not None else None,
            'train_scores': self.train_scores_,
            'oob_scores': self.oob_scores_
        }
        
        return progress
    
    def plot_boosting_progress(self) -> Any:
        """
        Plot training progress over boosting iterations
        
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            progress = self.get_boosting_progress()
            
            if progress['train_scores'] is None:
                logger.warning("No training scores available for plotting")
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            iterations = progress['iterations']
            
            # Plot training scores
            ax.plot(iterations, progress['train_scores'], 'b-', label='Training Score', linewidth=2)
            
            # Plot OOB scores if available
            if progress['oob_scores'] is not None:
                ax.plot(iterations, progress['oob_scores'], 'r-', label='Out-of-Bag Score', linewidth=2)
            
            # Mark best iteration if available
            if self.boosting_stats_ and 'best_iteration' in self.boosting_stats_:
                best_iter = self.boosting_stats_['best_iteration']
                ax.axvline(x=best_iter + 1, color='green', linestyle='--', 
                          label=f'Best Iteration: {best_iter + 1}')
            
            ax.set_xlabel('Boosting Iteration')
            ax.set_ylabel('Score')
            ax.set_title(f'Gradient Boosting Training Progress - {self.name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text with final statistics
            if self.boosting_stats_:
                stats_text = f"Loss: {self.loss}\n"
                stats_text += f"Learning Rate: {self.learning_rate}\n"
                stats_text += f"Estimators: {self.boosting_stats_['n_estimators_used']}\n"
                if self.boosting_stats_['early_stopping_used']:
                    stats_text += "Early Stopping: Yes\n"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_feature_importance(self, top_n: int = 20, 
                               importance_type: str = 'gain') -> Any:
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to show
            importance_type: Type of importance to plot
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            importance_df = self.get_feature_importance(top_n, importance_type)
            
            if importance_df.empty:
                logger.warning("No feature importance data available")
                return None
            
            plt.figure(figsize=(12, 8))
            
            y_pos = np.arange(len(importance_df))
            
            plt.barh(y_pos, importance_df['importance'], alpha=0.7, color='steelblue')
            plt.yticks(y_pos, importance_df['feature'], fontsize=10)
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.title(f'Gradient Boosting Feature Importance ({importance_type.title()}) - {self.name}')
            plt.grid(True, alpha=0.3, axis='x')
            
            # Add importance values as text
            for i, (idx, row) in enumerate(importance_df.iterrows()):
                importance = row['importance']
                plt.text(importance + 0.001, i, f'{importance:.3f}', 
                        va='center', fontsize=8)
            
            plt.tight_layout()
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_learning_curve(self) -> Any:
        """
        Plot learning curve if available
        
        Returns:
            Matplotlib figure
        """
        if self.learning_curve_ is None:
            logger.warning("Learning curve not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            curve = self.learning_curve_
            
            plt.figure(figsize=(12, 8))
            
            train_sizes = curve['train_sizes']
            train_mean = curve['train_scores_mean']
            train_std = curve['train_scores_std']
            val_mean = curve['val_scores_mean']
            val_std = curve['val_scores_std']
            
            # Plot training scores
            plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                           alpha=0.2, color='blue')
            
            # Plot validation scores
            plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                           alpha=0.2, color='red')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('R² Score')
            plt.title(f'Learning Curve - {self.name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def get_validation_curve(self, X: pd.DataFrame, y: pd.Series,
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
        
        logger.info(f"Generating validation curve for {param_name}")
        
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
    
    def get_boosting_summary(self) -> Dict[str, Any]:
        """Get comprehensive boosting summary"""
        
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get boosting summary")
        
        summary = {
            'boosting_stats': self.boosting_stats_,
            'model_type': 'HistGradientBoosting' if self.use_histogram_boosting else 'GradientBoosting',
            'hyperparameters': {
                'loss': self.loss,
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'subsample': self.subsample,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf
            }
        }
        
        # Add feature importance summary
        if self.feature_importances_ is not None:
            importance_df = self.get_feature_importance(top_n=10)
            summary['top_features'] = {
                'features': importance_df['feature'].tolist(),
                'importance_scores': importance_df['importance'].tolist(),
                'top_3_features': importance_df.head(3)['feature'].tolist()
            }
        
        # Add training progress summary
        if self.train_scores_ is not None:
            summary['training_progress'] = {
                'initial_score': float(self.train_scores_[0]),
                'final_score': float(self.train_scores_[-1]),
                'score_improvement': float(self.train_scores_[-1] - self.train_scores_[0]),
                'convergence_iteration': int(len(self.train_scores_))
            }
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add Gradient Boosting-specific information
        summary.update({
            'ensemble_type': 'Gradient Boosting',
            'boosting_variant': 'Histogram' if self.use_histogram_boosting else 'Standard',
            'loss_function': self.loss,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'early_stopping': self.n_iter_no_change is not None,
            'feature_subsampling': self.max_features is not None,
            'auto_scaling': self.auto_scale
        })
        
        # Add boosting summary
        if self.is_fitted:
            try:
                summary['boosting_summary'] = self.get_boosting_summary()
            except Exception as e:
                logger.debug(f"Could not generate boosting summary: {e}")
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_gradient_boosting_regressor(variant: str = 'standard',
                                      performance_preset: str = 'balanced',
                                      **kwargs) -> FinancialGradientBoostingRegressor:
    """
    Create a Gradient Boosting regression model
    
    Args:
        variant: Boosting variant ('standard', 'histogram')
        performance_preset: Performance preset ('fast', 'balanced', 'accurate')
        **kwargs: Additional model parameters
        
    Returns:
        Configured Gradient Boosting regression model
    """
    
    # Base configuration
    base_config = {
        'name': f'gradient_boosting_{variant}',
        'use_histogram_boosting': variant == 'histogram',
        'random_state': 42,
        'auto_scale': False  # Tree-based models don't need scaling
    }
    
    # Performance presets
    if performance_preset == 'fast':
        preset_config = {
            'n_estimators': 50,
            'learning_rate': 0.2,
            'max_depth': 3,
            'subsample': 0.8,
            'min_samples_split': 20,
            'min_samples_leaf': 10
        }
    elif performance_preset == 'balanced':
        preset_config = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 4,
            'subsample': 0.9,
            'min_samples_split': 5,
            'min_samples_leaf': 5
        }
    elif performance_preset == 'accurate':
        preset_config = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 1.0,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'n_iter_no_change': 20,  # Early stopping
            'validation_fraction': 0.1
        }
    else:
        raise ValueError(f"Unknown performance preset: {performance_preset}")
    
    # Combine configurations
    config = {**base_config, **preset_config}
    config.update(kwargs)  # Override with user parameters
    
    return FinancialGradientBoostingRegressor(**config)

def create_robust_gradient_boosting(**kwargs) -> FinancialGradientBoostingRegressor:
    """Create Gradient Boosting with robust loss function"""
    
    return create_gradient_boosting_regressor(
        performance_preset='balanced',
        loss='huber',  # Robust to outliers
        alpha=0.9,
        name='robust_gradient_boosting',
        **kwargs
    )

def create_quantile_gradient_boosting(quantile: float = 0.5, **kwargs) -> FinancialGradientBoostingRegressor:
    """Create Gradient Boosting for quantile regression"""
    
    return create_gradient_boosting_regressor(
        performance_preset='balanced',
        loss='quantile',
        alpha=quantile,
        name=f'quantile_gradient_boosting_{int(quantile*100)}',
        **kwargs
    )

def create_fast_gradient_boosting(**kwargs) -> FinancialGradientBoostingRegressor:
    """Create fast Gradient Boosting with HistGradientBoosting"""
    
    return create_gradient_boosting_regressor(
        variant='histogram',
        performance_preset='fast',
        name='fast_gradient_boosting',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def tune_gradient_boosting_hyperparameters(X: pd.DataFrame, y: pd.Series,
                                          param_grid: Optional[Dict[str, List[Any]]] = None,
                                          cv: int = 5,
                                          scoring: str = 'r2',
                                          n_jobs: int = -1) -> Dict[str, Any]:
    """
    Tune Gradient Boosting hyperparameters using grid search
    
    Args:
        X: Feature matrix
        y: Target values
        param_grid: Parameter grid for search
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with best parameters and scores
    """
    
    from sklearn.model_selection import GridSearchCV
    
    logger.info("Starting Gradient Boosting hyperparameter tuning")
    
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 4, 6],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5]
        }
    
    # Create base model
    base_model = GradientBoostingRegressor(random_state=42)
    
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
    
    grid_search.fit(X, y)
    
    # Extract results
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_,
        'feature_importances': grid_search.best_estimator_.feature_importances_
    }
    
    logger.info(f"Hyperparameter tuning complete. Best score: {results['best_score']:.4f}")
    logger.info(f"Best parameters: {results['best_params']}")
    
    return results

def compare_gradient_boosting_variants(X: pd.DataFrame, y: pd.Series,
                                     cv: int = 5) -> Dict[str, Any]:
    """
    Compare different Gradient Boosting variants and configurations
    
    Args:
        X: Feature matrix
        y: Target values
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with comparison results
    """
    
    from sklearn.model_selection import cross_val_score
    
    logger.info("Comparing Gradient Boosting variants")
    
    # Create different models
    models = {
        'standard_fast': create_gradient_boosting_regressor('standard', 'fast'),
        'standard_balanced': create_gradient_boosting_regressor('standard', 'balanced'),
        'standard_accurate': create_gradient_boosting_regressor('standard', 'accurate'),
        'histogram_fast': create_gradient_boosting_regressor('histogram', 'fast'),
        'histogram_balanced': create_gradient_boosting_regressor('histogram', 'balanced'),
        'robust_huber': create_robust_gradient_boosting()
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        
        # Get the underlying sklearn model
        sklearn_model = model._create_model()
        
        # Perform cross-validation
        cv_scores = cross_val_score(sklearn_model, X, y, cv=cv, scoring='r2', n_jobs=-1)
        
        # Time a single fit for performance comparison
        import time
        start_time = time.time()
        sklearn_model.fit(X, y)
        fit_time = time.time() - start_time
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'fit_time': fit_time,
            'model_config': model.get_model_summary()
        }
    
    # Add comparison summary
    results['comparison'] = {
        'best_accuracy': max(results.keys(), key=lambda k: results[k]['cv_mean']),
        'fastest_model': min(results.keys(), key=lambda k: results[k]['fit_time']),
        'most_stable': min(results.keys(), key=lambda k: results[k]['cv_std'])
    }
    
    logger.info(f"Variant comparison complete. Best accuracy: {results['comparison']['best_accuracy']}")
    
    return results

def analyze_gradient_boosting_convergence(X: pd.DataFrame, y: pd.Series,
                                        max_estimators: int = 500,
                                        early_stopping: bool = True) -> Dict[str, Any]:
    """
    Analyze convergence behavior of Gradient Boosting
    
    Args:
        X: Feature matrix
        y: Target values
        max_estimators: Maximum number of estimators to test
        early_stopping: Whether to use early stopping
        
    Returns:
        Dictionary with convergence analysis
    """
    
    logger.info(f"Analyzing Gradient Boosting convergence up to {max_estimators} estimators")
    
    # Create model with early stopping if requested
    model_params = {
        'n_estimators': max_estimators,
        'learning_rate': 0.1,
        'max_depth': 4,
        'subsample': 0.9,
        'validation_fraction': 0.2,
        'random_state': 42
    }
    
    if early_stopping:
        model_params.update({
            'n_iter_no_change': 20,
            'tol': 1e-4
        })
    
    # Fit model
    model = GradientBoostingRegressor(**model_params)
    model.fit(X, y)
    
    # Extract training progress
    train_scores = model.train_score_
    n_estimators_used = len(train_scores)
    
    # Calculate convergence metrics
    score_differences = np.diff(train_scores)
    convergence_rate = np.mean(np.abs(score_differences[-10:]))  # Last 10 iterations
    
    # Find optimal stopping point
    optimal_n_estimators = np.argmin(train_scores) + 1 if model.loss in ['squared_error'] else np.argmax(train_scores) + 1
    
    analysis = {
        'n_estimators_used': n_estimators_used,
        'early_stopping_triggered': n_estimators_used < max_estimators,
        'optimal_n_estimators': optimal_n_estimators,
        'final_train_score': train_scores[-1],
        'best_train_score': np.min(train_scores) if model.loss in ['squared_error'] else np.max(train_scores),
        'convergence_rate': convergence_rate,
        'train_scores': train_scores,
        'score_improvement': train_scores[-1] - train_scores[0],
        'convergence_stability': np.std(score_differences[-20:]) if len(score_differences) >= 20 else np.nan
    }
    
    logger.info(f"Convergence analysis complete. Used {n_estimators_used}/{max_estimators} estimators")
    logger.info(f"Optimal stopping point: {optimal_n_estimators} estimators")
    
    return analysis
