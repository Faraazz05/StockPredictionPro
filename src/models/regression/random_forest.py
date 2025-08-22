# ============================================
# StockPredictionPro - src/models/regression/random_forest.py
# Random Forest regression models for financial prediction with ensemble learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_regressor import BaseFinancialRegressor, RegressionStrategy

logger = get_logger('models.regression.random_forest')

# ============================================
# Random Forest Regression Model
# ============================================

class FinancialRandomForestRegressor(BaseFinancialRegressor):
    """
    Random Forest regression model optimized for financial data
    
    Features:
    - Ensemble of decision trees with bagging
    - Multiple splitting criteria and bootstrap options
    - Advanced feature importance analysis
    - Out-of-bag error estimation
    - Proximity matrix analysis
    - Tree diversity metrics
    """
    
    def __init__(self,
                 name: str = "random_forest_regressor",
                 n_estimators: int = 100,
                 criterion: str = 'squared_error',
                 max_depth: Optional[int] = None,
                 min_samples_split: Union[int, float] = 2,
                 min_samples_leaf: Union[int, float] = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 max_features: Union[str, int, float] = 'sqrt',
                 max_leaf_nodes: Optional[int] = None,
                 min_impurity_decrease: float = 0.0,
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 n_jobs: Optional[int] = -1,
                 random_state: Optional[int] = 42,
                 verbose: int = 0,
                 warm_start: bool = False,
                 ccp_alpha: float = 0.0,
                 max_samples: Optional[Union[int, float]] = None,
                 forest_variant: str = 'random_forest',
                 auto_scale: bool = False,
                 **kwargs):
        """
        Initialize Financial Random Forest Regressor
        
        Args:
            name: Model name
            n_estimators: Number of trees in the forest
            criterion: Split quality measure ('squared_error', 'absolute_error', 'friedman_mse', 'poisson')
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split internal node
            min_samples_leaf: Minimum samples at leaf node
            min_weight_fraction_leaf: Minimum weighted fraction at leaf
            max_features: Number of features for best split
            max_leaf_nodes: Maximum leaf nodes
            min_impurity_decrease: Minimum impurity decrease for splits
            bootstrap: Whether to use bootstrap sampling
            oob_score: Whether to compute out-of-bag score
            n_jobs: Number of parallel jobs
            random_state: Random state for reproducibility
            verbose: Verbosity level
            warm_start: Whether to reuse previous solution
            ccp_alpha: Complexity parameter for pruning
            max_samples: Number of samples for bootstrap
            forest_variant: Forest variant ('random_forest', 'extra_trees')
            auto_scale: Whether to automatically scale features
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="random_forest_regressor",
            regression_strategy=RegressionStrategy.PRICE_PREDICTION,
            **kwargs
        )
        
        # Random Forest parameters
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.forest_variant = forest_variant
        self.auto_scale = auto_scale
        
        # Store parameters for model creation
        self.model_params.update({
            'n_estimators': n_estimators,
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'min_weight_fraction_leaf': min_weight_fraction_leaf,
            'max_features': max_features,
            'max_leaf_nodes': max_leaf_nodes,
            'min_impurity_decrease': min_impurity_decrease,
            'bootstrap': bootstrap,
            'oob_score': oob_score,
            'n_jobs': n_jobs,
            'random_state': random_state,
            'verbose': verbose,
            'warm_start': warm_start,
            'ccp_alpha': ccp_alpha,
            'max_samples': max_samples
        })
        
        # Random Forest-specific attributes
        self.scaler_: Optional[StandardScaler] = None
        self.oob_score_: Optional[float] = None
        self.oob_prediction_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.permutation_importances_: Optional[Dict[str, np.ndarray]] = None
        self.tree_stats_: Optional[Dict[str, Any]] = None
        self.proximity_matrix_: Optional[np.ndarray] = None
        self.learning_curve_: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized {forest_variant.replace('_', ' ').title()} regressor: {self.name}")
    
    def _create_model(self) -> Union[RandomForestRegressor, ExtraTreesRegressor]:
        """Create the Random Forest regression model"""
        
        if self.forest_variant == 'extra_trees':
            # Extra Trees (Extremely Randomized Trees)
            return ExtraTreesRegressor(**self.model_params)
        else:
            # Standard Random Forest
            return RandomForestRegressor(**self.model_params)
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with optional scaling"""
        
        # Basic preprocessing
        X_processed = super()._preprocess_features(X)
        
        # Apply feature scaling if enabled (generally not needed for tree-based models)
        if self.auto_scale:
            if self.scaler_ is None:
                self.scaler_ = StandardScaler()
                X_scaled = self.scaler_.fit_transform(X_processed)
                logger.debug("Fitted feature scaler for Random Forest regression")
            else:
                X_scaled = self.scaler_.transform(X_processed)
            
            return X_scaled
        
        return X_processed
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Enhanced post-training processing for Random Forest regression"""
        
        # Call parent processing
        super()._post_training_processing(X, y)
        
        # Extract Random Forest-specific information
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        
        # Extract out-of-bag score if available
        if hasattr(self.model, 'oob_score_') and self.oob_score:
            self.oob_score_ = self.model.oob_score_
            logger.info(f"Out-of-bag R² score: {self.oob_score_:.4f}")
        
        # Extract out-of-bag predictions if available
        if hasattr(self.model, 'oob_prediction_'):
            self.oob_prediction_ = self.model.oob_prediction_
        
        # Calculate tree statistics
        self._calculate_tree_statistics()
        
        # Calculate permutation importance (on a sample for large datasets)
        if X.shape[0] <= 1000:  # Only for reasonably sized datasets
            self._calculate_permutation_importance(X, y)
        
        # Analyze learning curve
        self._analyze_learning_curve(X, y)
    
    def _calculate_tree_statistics(self):
        """Calculate statistics about the trees in the forest"""
        
        if not hasattr(self.model, 'estimators_'):
            return
        
        # Extract tree statistics
        tree_depths = []
        tree_nodes = []
        tree_leaves = []
        
        for estimator in self.model.estimators_:
            if hasattr(estimator, 'tree_'):
                tree = estimator.tree_
                tree_depths.append(tree.max_depth)
                tree_nodes.append(tree.node_count)
                tree_leaves.append(tree.n_leaves)
        
        self.tree_stats_ = {
            'n_trees': len(self.model.estimators_),
            'mean_depth': float(np.mean(tree_depths)) if tree_depths else 0.0,
            'std_depth': float(np.std(tree_depths)) if tree_depths else 0.0,
            'max_depth': int(np.max(tree_depths)) if tree_depths else 0,
            'min_depth': int(np.min(tree_depths)) if tree_depths else 0,
            'mean_nodes': float(np.mean(tree_nodes)) if tree_nodes else 0.0,
            'mean_leaves': float(np.mean(tree_leaves)) if tree_leaves else 0.0,
            'total_nodes': int(np.sum(tree_nodes)) if tree_nodes else 0,
            'total_leaves': int(np.sum(tree_leaves)) if tree_leaves else 0
        }
        
        # Calculate tree diversity
        if len(tree_depths) > 1:
            self.tree_stats_['depth_diversity'] = np.std(tree_depths) / np.mean(tree_depths) if np.mean(tree_depths) > 0 else 0.0
            self.tree_stats_['size_diversity'] = np.std(tree_nodes) / np.mean(tree_nodes) if np.mean(tree_nodes) > 0 else 0.0
        
        logger.debug(f"Tree statistics: {self.tree_stats_['n_trees']} trees, "
                    f"mean depth: {self.tree_stats_['mean_depth']:.1f}")
    
    def _calculate_permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                                        n_repeats: int = 5):
        """Calculate permutation importance for features"""
        
        try:
            # Sample data if too large
            if len(X) > 1000:
                indices = np.random.choice(len(X), 1000, replace=False)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model, X_sample, y_sample,
                n_repeats=n_repeats,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            
            self.permutation_importances_ = {
                'importances_mean': perm_importance.importances_mean,
                'importances_std': perm_importance.importances_std,
                'importances': perm_importance.importances
            }
            
            logger.debug("Calculated permutation importance")
            
        except Exception as e:
            logger.debug(f"Could not calculate permutation importance: {e}")
            self.permutation_importances_ = None
    
    def _analyze_learning_curve(self, X: np.ndarray, y: np.ndarray):
        """Analyze learning curve for the model"""
        
        try:
            if len(X) > 500:  # Only for reasonably sized datasets
                train_sizes = np.linspace(0.1, 1.0, 10)
                
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    estimator=self._create_model(),
                    X=X, y=y,
                    train_sizes=train_sizes,
                    cv=3,  # Faster with fewer folds
                    scoring='r2',
                    n_jobs=self.n_jobs,
                    random_state=self.random_state
                )
                
                self.learning_curve_ = {
                    'train_sizes': train_sizes_abs,
                    'train_scores_mean': np.mean(train_scores, axis=1),
                    'train_scores_std': np.std(train_scores, axis=1),
                    'val_scores_mean': np.mean(val_scores, axis=1),
                    'val_scores_std': np.std(val_scores, axis=1)
                }
                
                logger.debug("Calculated learning curve for Random Forest model")
            
        except Exception as e:
            logger.debug(f"Could not calculate learning curve: {e}")
            self.learning_curve_ = None
    
    def get_feature_importance(self, top_n: Optional[int] = None, 
                              importance_type: str = 'gini') -> pd.DataFrame:
        """
        Get feature importance rankings with different importance types
        
        Args:
            top_n: Number of top features to return
            importance_type: Type of importance ('gini', 'permutation')
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get feature importance")
        
        if importance_type == 'gini' and self.feature_importances_ is not None:
            # Standard impurity-based feature importance
            importance_scores = self.feature_importances_
            importance_std = None
            
        elif importance_type == 'permutation' and self.permutation_importances_ is not None:
            # Permutation-based feature importance
            importance_scores = self.permutation_importances_['importances_mean']
            importance_std = self.permutation_importances_['importances_std']
            
        else:
            # Fallback to gini importance
            if self.feature_importances_ is not None:
                importance_scores = self.feature_importances_
                importance_std = None
            else:
                raise BusinessLogicError("Feature importance not available")
        
        # Create importance dataframe
        importance_data = {
            'feature': self.feature_names,
            'importance': importance_scores,
            'importance_type': importance_type
        }
        
        if importance_std is not None:
            importance_data['importance_std'] = importance_std
        
        importance_df = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def get_oob_predictions(self) -> Optional[np.ndarray]:
        """
        Get out-of-bag predictions if available
        
        Returns:
            Out-of-bag predictions array or None
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get OOB predictions")
        
        return self.oob_prediction_
    
    def calculate_proximity_matrix(self, X: pd.DataFrame, 
                                  max_samples: int = 500) -> np.ndarray:
        """
        Calculate proximity matrix between samples
        
        Args:
            X: Feature matrix
            max_samples: Maximum samples to include (for computational efficiency)
            
        Returns:
            Proximity matrix
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to calculate proximity matrix")
        
        X_processed = self._preprocess_features(X)
        
        # Sample data if too large
        if len(X_processed) > max_samples:
            indices = np.random.choice(len(X_processed), max_samples, replace=False)
            X_sample = X_processed[indices]
        else:
            X_sample = X_processed
            indices = np.arange(len(X_processed))
        
        n_samples = len(X_sample)
        proximity_matrix = np.zeros((n_samples, n_samples))
        
        # Calculate proximity based on leaf co-occurrence
        for estimator in self.model.estimators_:
            # Get leaf indices for each sample
            leaf_indices = estimator.apply(X_sample)
            
            # Increment proximity for samples in same leaf
            for i in range(n_samples):
                for j in range(n_samples):
                    if leaf_indices[i] == leaf_indices[j]:
                        proximity_matrix[i, j] += 1
        
        # Normalize by number of trees
        proximity_matrix /= self.n_estimators
        
        self.proximity_matrix_ = proximity_matrix
        logger.debug(f"Calculated proximity matrix for {n_samples} samples")
        
        return proximity_matrix
    
    def get_tree_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from individual trees
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, n_estimators) with individual tree predictions
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get tree predictions")
        
        X_processed = self._preprocess_features(X)
        
        # Get predictions from each tree
        tree_predictions = np.zeros((len(X_processed), self.n_estimators))
        
        for i, estimator in enumerate(self.model.estimators_):
            tree_predictions[:, i] = estimator.predict(X_processed)
        
        return tree_predictions
    
    def analyze_prediction_variance(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Analyze prediction variance across trees
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with variance analysis
        """
        tree_predictions = self.get_tree_predictions(X)
        
        # Calculate statistics across trees for each sample
        analysis = {
            'mean_prediction': np.mean(tree_predictions, axis=1),
            'prediction_std': np.std(tree_predictions, axis=1),
            'prediction_var': np.var(tree_predictions, axis=1),
            'min_prediction': np.min(tree_predictions, axis=1),
            'max_prediction': np.max(tree_predictions, axis=1),
            'prediction_range': np.max(tree_predictions, axis=1) - np.min(tree_predictions, axis=1)
        }
        
        # Calculate prediction confidence (inverse of variance)
        analysis['prediction_confidence'] = 1.0 / (1.0 + analysis['prediction_var'])
        
        return analysis
    
    def plot_feature_importance(self, top_n: int = 20, 
                               importance_type: str = 'gini',
                               show_std: bool = True) -> Any:
        """
        Plot feature importance with optional standard deviation
        
        Args:
            top_n: Number of top features to show
            importance_type: Type of importance to plot
            show_std: Whether to show standard deviation (for permutation importance)
            
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
            
            # Plot bars
            bars = plt.barh(y_pos, importance_df['importance'], alpha=0.7, color='forestgreen')
            
            # Add error bars if standard deviation is available
            if show_std and 'importance_std' in importance_df.columns:
                plt.errorbar(importance_df['importance'], y_pos, 
                           xerr=importance_df['importance_std'],
                           fmt='none', color='black', capsize=3)
            
            plt.yticks(y_pos, importance_df['feature'], fontsize=10)
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.title(f'Random Forest Feature Importance ({importance_type.title()}) - {self.name}')
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
    
    def plot_oob_error(self) -> Any:
        """
        Plot out-of-bag error evolution (requires warm_start=True)
        
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.warm_start:
                logger.warning("OOB error evolution requires warm_start=True")
                return None
            
            # This would require fitting the model incrementally
            # For now, show final OOB score
            if self.oob_score_ is None:
                logger.warning("No OOB score available")
                return None
            
            plt.figure(figsize=(10, 6))
            plt.bar(['Out-of-Bag Score'], [self.oob_score_], color='forestgreen', alpha=0.7)
            plt.ylabel('R² Score')
            plt.title(f'Random Forest Out-of-Bag Performance - {self.name}')
            plt.ylim(0, 1)
            
            # Add text with score
            plt.text(0, self.oob_score_ + 0.02, f'{self.oob_score_:.4f}', 
                    ha='center', fontsize=12, fontweight='bold')
            
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
    
    def plot_prediction_variance(self, X: pd.DataFrame, max_samples: int = 100) -> Any:
        """
        Plot prediction variance analysis
        
        Args:
            X: Feature matrix
            max_samples: Maximum samples to analyze
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            # Sample data if too large
            if len(X) > max_samples:
                sample_indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X.iloc[sample_indices]
            else:
                X_sample = X
            
            variance_analysis = self.analyze_prediction_variance(X_sample)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Prediction std vs mean
            ax1.scatter(variance_analysis['mean_prediction'], 
                       variance_analysis['prediction_std'], alpha=0.6)
            ax1.set_xlabel('Mean Prediction')
            ax1.set_ylabel('Prediction Std Dev')
            ax1.set_title('Prediction Uncertainty vs Mean')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Histogram of prediction std
            ax2.hist(variance_analysis['prediction_std'], bins=20, alpha=0.7, color='orange')
            ax2.set_xlabel('Prediction Std Dev')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Prediction Uncertainty')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Prediction range
            ax3.scatter(range(len(variance_analysis['prediction_range'])), 
                       variance_analysis['prediction_range'], alpha=0.6, color='green')
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Prediction Range (Max - Min)')
            ax3.set_title('Prediction Range Across Trees')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Prediction confidence
            ax4.hist(variance_analysis['prediction_confidence'], bins=20, alpha=0.7, color='purple')
            ax4.set_xlabel('Prediction Confidence')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Prediction Confidence')
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'Random Forest Prediction Variance Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
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
            n_jobs=self.n_jobs
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
    
    def get_forest_summary(self) -> Dict[str, Any]:
        """Get comprehensive forest summary"""
        
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get forest summary")
        
        summary = {
            'forest_variant': self.forest_variant,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'criterion': self.criterion,
                'max_depth': self.max_depth,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'oob_score': self.oob_score,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf
            }
        }
        
        # Add tree statistics
        if self.tree_stats_:
            summary['tree_statistics'] = self.tree_stats_
        
        # Add out-of-bag information
        if self.oob_score_ is not None:
            summary['oob_performance'] = {
                'oob_score': float(self.oob_score_),
                'oob_available': True
            }
        else:
            summary['oob_performance'] = {'oob_available': False}
        
        # Add feature importance summary
        if self.feature_importances_ is not None:
            importance_df = self.get_feature_importance(top_n=10)
            summary['top_features'] = {
                'features': importance_df['feature'].tolist(),
                'importance_scores': importance_df['importance'].tolist(),
                'top_3_features': importance_df.head(3)['feature'].tolist()
            }
        
        # Add permutation importance if available
        if self.permutation_importances_ is not None:
            perm_importance_df = self.get_feature_importance(top_n=5, importance_type='permutation')
            summary['permutation_importance'] = {
                'top_5_features': perm_importance_df['feature'].tolist(),
                'importance_scores': perm_importance_df['importance'].tolist()
            }
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add Random Forest-specific information
        summary.update({
            'ensemble_type': 'Random Forest',
            'forest_variant': self.forest_variant.replace('_', ' ').title(),
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'oob_score_enabled': self.oob_score,
            'oob_score_value': float(self.oob_score_) if self.oob_score_ else None,
            'feature_subsampling': self.max_features != 'auto',
            'auto_scaling': self.auto_scale
        })
        
        # Add forest summary
        if self.is_fitted:
            try:
                summary['forest_summary'] = self.get_forest_summary()
            except Exception as e:
                logger.debug(f"Could not generate forest summary: {e}")
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_random_forest_regressor(variant: str = 'random_forest',
                                  performance_preset: str = 'balanced',
                                  **kwargs) -> FinancialRandomForestRegressor:
    """
    Create a Random Forest regression model
    
    Args:
        variant: Forest variant ('random_forest', 'extra_trees')
        performance_preset: Performance preset ('fast', 'balanced', 'accurate')
        **kwargs: Additional model parameters
        
    Returns:
        Configured Random Forest regression model
    """
    
    # Base configuration
    base_config = {
        'name': f'{variant}_regressor',
        'forest_variant': variant,
        'random_state': 42,
        'n_jobs': -1,
        'auto_scale': False  # Tree-based models don't need scaling
    }
    
    # Performance presets
    if performance_preset == 'fast':
        preset_config = {
            'n_estimators': 50,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True
        }
    elif performance_preset == 'balanced':
        preset_config = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True
        }
    elif performance_preset == 'accurate':
        preset_config = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'max_samples': 0.8  # Subsample for diversity
        }
    else:
        raise ValueError(f"Unknown performance preset: {performance_preset}")
    
    # Combine configurations
    config = {**base_config, **preset_config}
    config.update(kwargs)  # Override with user parameters
    
    return FinancialRandomForestRegressor(**config)

def create_extra_trees_regressor(performance_preset: str = 'balanced',
                                **kwargs) -> FinancialRandomForestRegressor:
    """Create Extra Trees regressor (Extremely Randomized Trees)"""
    
    return create_random_forest_regressor(
        variant='extra_trees',
        performance_preset=performance_preset,
        **kwargs
    )

def create_oob_random_forest(**kwargs) -> FinancialRandomForestRegressor:
    """Create Random Forest optimized for out-of-bag evaluation"""
    
    return create_random_forest_regressor(
        performance_preset='balanced',
        bootstrap=True,
        oob_score=True,
        max_samples=0.8,  # Ensure OOB samples
        name='oob_random_forest',
        **kwargs
    )

def create_feature_selection_forest(**kwargs) -> FinancialRandomForestRegressor:
    """Create Random Forest optimized for feature selection"""
    
    return create_random_forest_regressor(
        performance_preset='accurate',
        n_estimators=300,
        max_features='log2',  # More selective feature sampling
        bootstrap=True,
        oob_score=True,
        name='feature_selection_forest',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def tune_random_forest_hyperparameters(X: pd.DataFrame, y: pd.Series,
                                      param_grid: Optional[Dict[str, List[Any]]] = None,
                                      cv: int = 5,
                                      scoring: str = 'r2',
                                      n_jobs: int = -1) -> Dict[str, Any]:
    """
    Tune Random Forest hyperparameters using grid search
    
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
    
    logger.info("Starting Random Forest hyperparameter tuning")
    
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
    
    # Create base model
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
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

def compare_forest_variants(X: pd.DataFrame, y: pd.Series,
                           cv: int = 5) -> Dict[str, Any]:
    """
    Compare Random Forest vs Extra Trees variants
    
    Args:
        X: Feature matrix
        y: Target values
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with comparison results
    """
    
    from sklearn.model_selection import cross_val_score
    
    logger.info("Comparing forest variants: Random Forest vs Extra Trees")
    
    # Create different models
    models = {
        'random_forest_fast': create_random_forest_regressor('random_forest', 'fast'),
        'random_forest_balanced': create_random_forest_regressor('random_forest', 'balanced'),
        'random_forest_accurate': create_random_forest_regressor('random_forest', 'accurate'),
        'extra_trees_fast': create_extra_trees_regressor('fast'),
        'extra_trees_balanced': create_extra_trees_regressor('balanced'),
        'extra_trees_accurate': create_extra_trees_regressor('accurate')
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
            'oob_score': getattr(sklearn_model, 'oob_score_', None),
            'feature_importances': sklearn_model.feature_importances_
        }
    
    # Add comparison summary
    results['comparison'] = {
        'best_accuracy': max(results.keys(), key=lambda k: results[k]['cv_mean']),
        'fastest_model': min(results.keys(), key=lambda k: results[k]['fit_time']),
        'most_stable': min(results.keys(), key=lambda k: results[k]['cv_std'])
    }
    
    logger.info(f"Variant comparison complete. Best accuracy: {results['comparison']['best_accuracy']}")
    
    return results

def analyze_forest_diversity(model: FinancialRandomForestRegressor, 
                            X: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze diversity of trees in the forest
    
    Args:
        model: Fitted Random Forest model
        X: Feature matrix for analysis
        
    Returns:
        Dictionary with diversity analysis
    """
    
    if not model.is_fitted:
        raise BusinessLogicError("Model must be fitted for diversity analysis")
    
    logger.info("Analyzing forest diversity")
    
    # Get predictions from individual trees
    tree_predictions = model.get_tree_predictions(X)
    
    # Calculate pairwise correlations between trees
    tree_correlations = np.corrcoef(tree_predictions.T)
    
    # Remove diagonal (self-correlations)
    mask = ~np.eye(tree_correlations.shape[0], dtype=bool)
    correlation_values = tree_correlations[mask]
    
    # Calculate diversity metrics
    diversity_analysis = {
        'mean_tree_correlation': float(np.mean(correlation_values)),
        'std_tree_correlation': float(np.std(correlation_values)),
        'min_tree_correlation': float(np.min(correlation_values)),
        'max_tree_correlation': float(np.max(correlation_values)),
        'diversity_score': 1.0 - np.mean(correlation_values),  # Higher is more diverse
        'prediction_variance': np.var(tree_predictions, axis=1).mean()
    }
    
    # Add tree statistics if available
    if model.tree_stats_:
        diversity_analysis.update({
            'structural_diversity': {
                'depth_diversity': model.tree_stats_.get('depth_diversity', 0.0),
                'size_diversity': model.tree_stats_.get('size_diversity', 0.0),
                'mean_depth': model.tree_stats_['mean_depth'],
                'depth_range': model.tree_stats_['max_depth'] - model.tree_stats_['min_depth']
            }
        })
    
    logger.info(f"Forest diversity analysis complete. Diversity score: {diversity_analysis['diversity_score']:.3f}")
    
    return diversity_analysis

def perform_feature_selection_with_forest(X: pd.DataFrame, y: pd.Series,
                                        importance_threshold: float = 0.001,
                                        n_estimators: int = 200) -> Dict[str, Any]:
    """
    Perform feature selection using Random Forest importance
    
    Args:
        X: Feature matrix
        y: Target values
        importance_threshold: Minimum importance threshold
        n_estimators: Number of trees for feature selection
        
    Returns:
        Dictionary with feature selection results
    """
    
    logger.info(f"Performing feature selection with Random Forest (threshold: {importance_threshold})")
    
    # Create feature selection forest
    selector_forest = create_feature_selection_forest(n_estimators=n_estimators)
    
    # Fit the model
    selector_forest.fit(X, y)
    
    # Get feature importance
    importance_df = selector_forest.get_feature_importance()
    
    # Select features above threshold
    selected_features = importance_df[importance_df['importance'] >= importance_threshold]['feature'].tolist()
    eliminated_features = importance_df[importance_df['importance'] < importance_threshold]['feature'].tolist()
    
    # Create results
    results = {
        'selected_features': selected_features,
        'eliminated_features': eliminated_features,
        'n_selected': len(selected_features),
        'n_eliminated': len(eliminated_features),
        'selection_ratio': len(selected_features) / len(X.columns),
        'importance_threshold': importance_threshold,
        'forest_oob_score': selector_forest.oob_score_,
        'top_10_features': importance_df.head(10)['feature'].tolist(),
        'feature_importance_stats': {
            'mean_importance': importance_df['importance'].mean(),
            'std_importance': importance_df['importance'].std(),
            'max_importance': importance_df['importance'].max(),
            'min_importance': importance_df['importance'].min()
        }
    }
    
    logger.info(f"Feature selection complete: {results['n_selected']}/{len(X.columns)} features selected "
               f"({results['selection_ratio']:.1%} selection ratio)")
    
    return results
