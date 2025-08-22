# ============================================
# StockPredictionPro - src/models/classification/svm.py
# Support Vector Machine classification models for financial prediction with kernel methods
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.svm import SVC, NuSVC
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, log_loss
)
from sklearn.calibration import CalibratedClassifierCV
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_classifier import BaseFinancialClassifier, ClassificationStrategy

logger = get_logger('models.classification.svm')

# ============================================
# Support Vector Machine Classification Model
# ============================================

class FinancialSVMClassifier(BaseFinancialClassifier):
    """
    Support Vector Machine classification model optimized for financial data
    
    Features:
    - Multiple kernel functions (linear, polynomial, RBF, sigmoid)
    - Advanced hyperparameter optimization with kernel-specific tuning
    - Support vector analysis and decision boundary visualization
    - Probability calibration for reliable confidence estimates
    - Nu-SVM variant for automatic margin tuning
    - Financial domain optimizations (volatility-aware, trend-sensitive)
    - Comprehensive kernel parameter exploration
    """
    
    def __init__(self,
                 name: str = "svm_classifier",
                 kernel: str = 'rbf',
                 degree: int = 3,
                 gamma: Union[str, float] = 'scale',
                 coef0: float = 0.0,
                 tol: float = 1e-3,
                 C: float = 1.0,
                 nu: float = 0.5,
                 shrinking: bool = True,
                 probability: bool = True,
                 cache_size: float = 200,
                 class_weight: Optional[Union[str, Dict]] = None,
                 verbose: bool = False,
                 max_iter: int = -1,
                 decision_function_shape: str = 'ovr',
                 break_ties: bool = False,
                 random_state: Optional[int] = 42,
                 svm_variant: str = 'svc',
                 scaler_type: str = 'standard',
                 calibrate_probabilities: bool = True,
                 auto_scale: bool = True,
                 **kwargs):
        """
        Initialize Financial Support Vector Machine Classifier
        
        Args:
            name: Model name
            kernel: Kernel function ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
            degree: Degree for polynomial kernel
            gamma: Kernel coefficient for rbf, poly, sigmoid
            coef0: Independent term in kernel function
            tol: Tolerance for stopping criterion
            C: Regularization parameter
            nu: Upper bound on fraction of training errors (Nu-SVC only)
            shrinking: Whether to use shrinking heuristic
            probability: Whether to enable probability estimates
            cache_size: Cache size in MB
            class_weight: Weights associated with classes
            verbose: Enable verbose output
            max_iter: Hard limit on iterations (-1 for no limit)
            decision_function_shape: Decision function shape ('ovo', 'ovr')
            break_ties: Break ties according to confidence values
            random_state: Random state for reproducibility
            svm_variant: SVM variant ('svc', 'nu_svc')
            scaler_type: Type of scaler ('standard', 'robust', 'minmax')
            calibrate_probabilities: Whether to calibrate prediction probabilities
            auto_scale: Whether to automatically scale features
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="svm_classifier",
            classification_strategy=ClassificationStrategy.DIRECTION_PREDICTION,
            **kwargs
        )
        
        # SVM parameters
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.nu = nu
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        self.svm_variant = svm_variant
        self.scaler_type = scaler_type
        self.calibrate_probabilities = calibrate_probabilities
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
            'probability': probability,
            'cache_size': cache_size,
            'class_weight': class_weight,
            'verbose': verbose,
            'max_iter': max_iter,
            'decision_function_shape': decision_function_shape,
            'break_ties': break_ties,
            'random_state': random_state
        }
        
        # Add variant-specific parameters
        if svm_variant == 'nu_svc':
            self.model_params['nu'] = nu
        
        # SVM-specific attributes
        self.scaler_: Optional[Union[StandardScaler, RobustScaler]] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.calibrated_model_: Optional[CalibratedClassifierCV] = None
        self.support_vectors_: Optional[np.ndarray] = None
        self.support_: Optional[np.ndarray] = None
        self.n_support_: Optional[np.ndarray] = None
        self.dual_coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.kernel_params_: Optional[Dict[str, Any]] = None
        self.hyperparameter_analysis_: Optional[Dict[str, Any]] = None
        self.class_weights_: Optional[Dict[Any, float]] = None
        
        logger.info(f"Initialized {svm_variant.replace('_', '-').upper()} with {kernel} kernel: {self.name}")
    
    def _create_model(self) -> Union[SVC, NuSVC]:
        """Create the Support Vector Machine classification model"""
        
        if self.svm_variant == 'nu_svc':
            # Nu-SVC: automatically determines regularization
            nu_params = {k: v for k, v in self.model_params.items() if k != 'C'}
            return NuSVC(**nu_params)
        else:
            # Standard SVC
            svc_params = {k: v for k, v in self.model_params.items() if k != 'nu'}
            return SVC(**svc_params)
    
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
        """Preprocess features with scaling (essential for SVM)"""
        
        # Basic preprocessing
        X_processed = super()._preprocess_features(X)
        
        # SVM requires feature scaling
        if self.auto_scale:
            if self.scaler_ is None:
                self.scaler_ = self._create_scaler()
                X_scaled = self.scaler_.fit_transform(X_processed)
                logger.debug(f"Fitted {self.scaler_type} scaler for SVM classifier")
            else:
                X_scaled = self.scaler_.transform(X_processed)
            
            return X_scaled
        else:
            logger.warning("SVM without feature scaling may perform poorly")
            return X_processed
    
    def _preprocess_targets(self, y: pd.Series) -> np.ndarray:
        """Preprocess target labels with encoding"""
        
        # Convert to numpy array
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        # Encode labels if necessary
        if self.label_encoder_ is None:
            self.label_encoder_ = LabelEncoder()
            y_encoded = self.label_encoder_.fit_transform(y_array)
            self.classes_ = self.label_encoder_.classes_
            logger.debug(f"Fitted label encoder. Classes: {self.classes_}")
        else:
            y_encoded = self.label_encoder_.transform(y_array)
        
        return y_encoded
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Enhanced post-training processing for SVM classification"""
        
        # Call parent processing
        super()._post_training_processing(X, y)
        
        # Extract SVM-specific information
        if hasattr(self.model, 'support_vectors_'):
            self.support_vectors_ = self.model.support_vectors_
        
        if hasattr(self.model, 'support_'):
            self.support_ = self.model.support_
        
        if hasattr(self.model, 'n_support_'):
            self.n_support_ = self.model.n_support_
        
        if hasattr(self.model, 'dual_coef_'):
            self.dual_coef_ = self.model.dual_coef_
        
        if hasattr(self.model, 'intercept_'):
            self.intercept_ = self.model.intercept_
        
        # Calculate class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.class_weights_ = {cls: count / len(y) for cls, count in zip(unique_classes, class_counts)}
        
        # Store kernel parameters
        self.kernel_params_ = {
            'kernel': self.kernel,
            'gamma': self.model.gamma if hasattr(self.model, 'gamma') else self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'C': self.C if self.svm_variant == 'svc' else None,
            'nu': self.nu if self.svm_variant == 'nu_svc' else None
        }
        
        # Calculate support vector statistics
        self._analyze_support_vectors()
        
        # Calibrate probabilities if requested
        if self.calibrate_probabilities and self.probability:
            self._calibrate_probabilities(X, y)
        
        logger.info(f"SVM training complete: {np.sum(self.n_support_) if self.n_support_ is not None else 0} "
                   f"support vectors ({np.sum(self.n_support_)/len(X)*100:.1f}% of training data)")
    
    def _analyze_support_vectors(self):
        """Analyze support vector characteristics"""
        
        if self.support_vectors_ is None or self.dual_coef_ is None:
            return
        
        # Calculate support vector statistics
        n_total_sv = np.sum(self.n_support_) if self.n_support_ is not None else len(self.support_vectors_)
        sv_ratio = n_total_sv / len(self.support_vectors_) if len(self.support_vectors_) > 0 else 0.0
        
        sv_analysis = {
            'n_support_vectors': int(n_total_sv),
            'n_support_per_class': self.n_support_.tolist() if self.n_support_ is not None else [],
            'support_vector_ratio': float(sv_ratio),
            'dual_coef_mean': float(np.mean(np.abs(self.dual_coef_))),
            'dual_coef_std': float(np.std(self.dual_coef_)),
            'dual_coef_max': float(np.max(np.abs(self.dual_coef_))),
            'intercept': self.intercept_.tolist() if self.intercept_ is not None else []
        }
        
        # Analyze dual coefficients distribution
        if self.dual_coef_.size > 0:
            dual_coef_flat = self.dual_coef_.flatten()
            sv_analysis.update({
                'dual_coef_skewness': float(pd.Series(dual_coef_flat).skew()),
                'dual_coef_kurtosis': float(pd.Series(dual_coef_flat).kurtosis()),
                'bound_support_vectors': int(np.sum(np.abs(dual_coef_flat) >= self.C * 0.99)) if self.svm_variant == 'svc' else 0
            })
        
        # Kernel-specific analysis
        if self.kernel == 'linear' and hasattr(self.model, 'coef_'):
            # For linear kernel, we can analyze feature weights
            feature_weights = self.model.coef_[0] if len(self.model.coef_) == 1 else self.model.coef_
            sv_analysis['linear_weights_norm'] = float(np.linalg.norm(feature_weights))
            sv_analysis['max_feature_weight'] = float(np.max(np.abs(feature_weights)))
        
        self.support_vector_analysis_ = sv_analysis
    
    def _calibrate_probabilities(self, X: np.ndarray, y: np.ndarray):
        """Calibrate prediction probabilities using cross-validation"""
        
        try:
            # Use Platt scaling for SVM (sigmoid calibration)
            self.calibrated_model_ = CalibratedClassifierCV(
                base_estimator=self.model,
                method='sigmoid',  # Platt scaling, works well for SVM
                cv=3
            )
            self.calibrated_model_.fit(X, y)
            logger.debug("Calibrated prediction probabilities using Platt scaling")
        except Exception as e:
            logger.warning(f"Could not calibrate probabilities: {e}")
            self.calibrated_model_ = None
    
    @time_it("svm_fit", include_args=True)
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'FinancialSVMClassifier':
        """
        Fit the SVM classification model
        
        Args:
            X: Feature matrix
            y: Target labels
            **kwargs: Additional fitting parameters
            
        Returns:
            Fitted model instance
        """
        logger.info(f"Fitting SVM Classifier on {len(X)} samples with {X.shape[1]} features")
        
        # Validate input data
        validation_result = self._validate_input_data(X, y)
        if not validation_result.is_valid:
            raise ModelValidationError(f"Input validation failed: {validation_result.errors}")
        
        try:
            # Update status
            from ..base.base_model import ModelStatus
            self.status = ModelStatus.TRAINING
            self.last_training_time = datetime.now()
            
            # Store feature names
            self.feature_names = list(X.columns)
            self.target_name = y.name or 'target'
            
            # Preprocess features and targets
            X_processed = self._preprocess_features(X)
            y_processed = self._preprocess_targets(y)
            
            # Create model if not exists
            if self.model is None:
                self.model = self._create_model()
            
            # Fit the model
            fit_start = datetime.now()
            self.model.fit(X_processed, y_processed)
            fit_duration = (datetime.now() - fit_start).total_seconds()
            self.training_duration = fit_duration
            
            # Post-training processing
            self._post_training_processing(X_processed, y_processed)
            
            # Update model metadata
            self.update_metadata({
                'training_samples': len(X),
                'training_features': X.shape[1],
                'n_classes': len(self.classes_),
                'class_names': self.classes_.tolist(),
                'training_duration_seconds': fit_duration,
                'target_name': self.target_name,
                'svm_variant': self.svm_variant,
                'kernel': self.kernel,
                'n_support_vectors': int(np.sum(self.n_support_)) if self.n_support_ is not None else 0
            })
            
            # Calculate training score
            self.training_score = self.model.score(X_processed, y_processed)
            
            # Update status
            self.status = ModelStatus.TRAINED
            self.is_fitted = True
            
            logger.info(f"SVM Classifier trained successfully in {fit_duration:.2f}s")
            
            return self
            
        except Exception as e:
            from ..base.base_model import ModelStatus
            self.status = ModelStatus.ERROR
            self.last_error = str(e)
            logger.error(f"SVM Classifier training failed: {e}")
            raise
    
    @time_it("svm_predict", include_args=True)
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions using the fitted SVM classifier
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        logger.debug(f"Making SVM predictions for {len(X)} samples")
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            # Make predictions
            if self.calibrated_model_ is not None:
                predictions_encoded = self.calibrated_model_.predict(X_processed)
            else:
                predictions_encoded = self.model.predict(X_processed)
            
            # Decode predictions
            predictions = self.label_encoder_.inverse_transform(predictions_encoded)
            
            # Log prediction
            self.log_prediction()
            
            return predictions
        
        except Exception as e:
            logger.error(f"SVM prediction failed: {e}")
            raise
    
    @time_it("svm_predict_proba", include_args=True)
    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        if not self.probability:
            raise BusinessLogicError("Probability estimation not enabled. Set probability=True during initialization.")
        
        logger.debug(f"Making SVM probability predictions for {len(X)} samples")
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            # Get probabilities
            if self.calibrated_model_ is not None:
                probabilities = self.calibrated_model_.predict_proba(X_processed)
            else:
                probabilities = self.model.predict_proba(X_processed)
            
            return probabilities
        
        except Exception as e:
            logger.error(f"SVM probability prediction failed: {e}")
            raise
    
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
        
        if self.calibrated_model_ is not None:
            predictions_encoded = self.calibrated_model_.predict(X_processed)
            # Decision function from base estimator
            decision_values = self.model.decision_function(X_processed)
        else:
            predictions_encoded = self.model.predict(X_processed)
            decision_values = self.model.decision_function(X_processed)
        
        # Decode predictions
        predictions = self.label_encoder_.inverse_transform(predictions_encoded)
        
        return predictions, decision_values
    
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
        if self.kernel in ['rbf', 'sigmoid']:
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
            y_encoded = self.label_encoder_.transform(y)
            
            if X_processed.shape[1] < 2:
                logger.warning("Need at least 2 features for 2D support vector plot")
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get feature indices
            feat1, feat2 = feature_indices
            
            # Plot all points with different colors for each class
            unique_classes = np.unique(y_encoded)
            colors = ['red', 'blue', 'green', 'orange', 'purple'][:len(unique_classes)]
            class_names = [self.label_encoder_.inverse_transform([c])[0] for c in unique_classes]
            
            for i, (cls, color, name) in enumerate(zip(unique_classes, colors, class_names)):
                mask = y_encoded == cls
                ax.scatter(X_processed[mask, feat1], X_processed[mask, feat2], 
                          c=color, alpha=0.6, s=30, label=f'Class {name}')
            
            # Highlight support vectors
            if self.support_ is not None and len(self.support_) > 0:
                support_points = X_processed[self.support_]
                ax.scatter(support_points[:, feat1], support_points[:, feat2], 
                          c='black', s=100, alpha=0.8, marker='o', 
                          facecolors='none', edgecolors='black', linewidths=2,
                          label=f'Support Vectors ({len(self.support_)})')
            
            # Labels and title
            feature_names = self.feature_names if self.feature_names else [f'Feature {i}' for i in range(X_processed.shape[1])]
            ax.set_xlabel(feature_names[feat1] if feat1 < len(feature_names) else f'Feature {feat1}')
            ax.set_ylabel(feature_names[feat2] if feat2 < len(feature_names) else f'Feature {feat2}')
            ax.set_title(f'SVM Support Vectors ({self.kernel} kernel) - {self.name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_decision_boundary(self, X: pd.DataFrame, y: pd.Series,
                              feature_indices: Tuple[int, int] = (0, 1),
                              resolution: int = 100) -> Any:
        """
        Plot decision boundary for 2D feature space
        
        Args:
            X: Feature matrix
            y: Target values
            feature_indices: Indices of features to plot
            resolution: Grid resolution for decision boundary
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to plot decision boundary")
        
        try:
            import matplotlib.pyplot as plt
            
            X_processed = self._preprocess_features(X)
            y_encoded = self.label_encoder_.transform(y)
            
            if X_processed.shape[1] < 2:
                logger.warning("Need at least 2 features for 2D decision boundary plot")
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
            
            # Plot decision boundary
            contour = ax.contourf(xx1, xx2, Z, alpha=0.3, cmap='RdYlBu')
            
            # Plot training points
            unique_classes = np.unique(y_encoded)
            colors = ['red', 'blue', 'green', 'orange', 'purple'][:len(unique_classes)]
            class_names = [self.label_encoder_.inverse_transform([c])[0] for c in unique_classes]
            
            for i, (cls, color, name) in enumerate(zip(unique_classes, colors, class_names)):
                mask = y_encoded == cls
                ax.scatter(X_processed[mask, feat1], X_processed[mask, feat2],
                          c=color, s=50, alpha=0.8, edgecolors='black', label=f'Class {name}')
            
            # Highlight support vectors
            if self.support_ is not None and len(self.support_) > 0:
                support_points = X_processed[self.support_]
                ax.scatter(support_points[:, feat1], support_points[:, feat2],
                          c='yellow', s=100, alpha=1.0, marker='o',
                          facecolors='none', edgecolors='black', linewidths=3,
                          label=f'Support Vectors ({len(self.support_)})')
            
            # Labels and title
            feature_names = self.feature_names if self.feature_names else [f'Feature {i}' for i in range(X_processed.shape[1])]
            ax.set_xlabel(feature_names[feat1] if feat1 < len(feature_names) else f'Feature {feat1}')
            ax.set_ylabel(feature_names[feat2] if feat2 < len(feature_names) else f'Feature {feat2}')
            ax.set_title(f'SVM Decision Boundary ({self.kernel} kernel) - {self.name}')
            ax.legend()
            
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
            if param_name in ['C', 'gamma'] and isinstance(param_values[0], (int, float)):
                plt.xscale('log')
            
            plt.xlabel(param_name.upper())
            plt.ylabel('Accuracy Score')
            plt.title(f'SVM Validation Curve - {param_name} - {self.name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_class_distribution(self) -> Any:
        """
        Plot class distribution
        
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.class_weights_:
                logger.warning("Class distribution not available")
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot
            classes = list(self.class_weights_.keys())
            weights = list(self.class_weights_.values())
            
            bars = ax1.bar(range(len(classes)), weights, alpha=0.7, color='steelblue')
            ax1.set_xticks(range(len(classes)))
            ax1.set_xticklabels([self.label_encoder_.inverse_transform([c])[0] for c in classes])
            ax1.set_ylabel('Proportion')
            ax1.set_title('Class Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Add values on bars
            for bar, weight in zip(bars, weights):
                ax1.text(bar.get_x() + bar.get_width()/2, weight + 0.01, 
                        f'{weight:.3f}', ha='center', va='bottom')
            
            # Support vectors per class
            if hasattr(self, 'support_vector_analysis_') and 'n_support_per_class' in self.support_vector_analysis_:
                sv_per_class = self.support_vector_analysis_['n_support_per_class']
                if len(sv_per_class) == len(classes):
                    bars2 = ax2.bar(range(len(classes)), sv_per_class, alpha=0.7, color='orange')
                    ax2.set_xticks(range(len(classes)))
                    ax2.set_xticklabels([self.label_encoder_.inverse_transform([c])[0] for c in classes])
                    ax2.set_ylabel('Number of Support Vectors')
                    ax2.set_title('Support Vectors per Class')
                    ax2.grid(True, alpha=0.3)
                    
                    # Add values on bars
                    for bar, count in zip(bars2, sv_per_class):
                        ax2.text(bar.get_x() + bar.get_width()/2, count + max(sv_per_class) * 0.01, 
                                f'{count}', ha='center', va='bottom')
                else:
                    ax2.text(0.5, 0.5, 'Support Vectors\nper Class\nNot Available', 
                           ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'Support Vectors\nper Class\nNot Available', 
                       ha='center', va='center', transform=ax2.transAxes)
            
            plt.suptitle(f'Class Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_confusion_matrix(self, X: pd.DataFrame, y: pd.Series, 
                             normalize: str = 'true') -> Any:
        """
        Plot confusion matrix
        
        Args:
            X: Feature matrix
            y: True labels
            normalize: Normalization option ('true', 'pred', 'all', None)
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Make predictions
            y_pred = self.predict(X)
            y_true_encoded = self.label_encoder_.transform(y)
            y_pred_encoded = self.label_encoder_.transform(y_pred)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true_encoded, y_pred_encoded, normalize=normalize)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            # Use class names for labels
            class_names = [self.label_encoder_.inverse_transform([i])[0] for i in range(len(self.classes_))]
            
            sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd', 
                       cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            
            plt.title(f'Confusion Matrix - {self.name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return None
    
    def plot_roc_curve(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Plot ROC curve (for binary classification)
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import auc
            
            if len(self.classes_) != 2:
                logger.warning("ROC curve is only available for binary classification")
                return None
            
            if not self.probability:
                logger.warning("ROC curve requires probability estimates. Set probability=True.")
                return None
            
            # Get probabilities
            y_proba = self.predict_proba(X)[:, 1]  # Positive class probability
            y_true_encoded = self.label_encoder_.transform(y)
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true_encoded, y_proba)
            auc_score = auc(fpr, tpr)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
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
        
        logger.info(f"Generating SVM validation curve for {param_name}")
        
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
            scoring='accuracy',
            n_jobs=1  # SVM doesn't parallelize well
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
    
    def get_svm_summary(self) -> Dict[str, Any]:
        """Get comprehensive SVM summary"""
        
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get SVM summary")
        
        summary = {
            'svm_variant': self.svm_variant,
            'kernel_info': self.kernel_params_,
            'hyperparameters': {
                'C': self.C if self.svm_variant == 'svc' else None,
                'nu': self.nu if self.svm_variant == 'nu_svc' else None,
                'gamma': self.gamma,
                'degree': self.degree if self.kernel == 'poly' else None,
                'coef0': self.coef0 if self.kernel in ['poly', 'sigmoid'] else None,
                'class_weight': self.class_weight
            },
            'scaler_info': {
                'scaler_type': self.scaler_type,
                'auto_scale': self.auto_scale
            },
            'calibration_info': {
                'probabilities_calibrated': self.calibrated_model_ is not None,
                'calibration_method': 'sigmoid' if self.calibrated_model_ else None,
                'probability_estimation': self.probability
            },
            'class_distribution': self.class_weights_
        }
        
        # Add support vector information
        if hasattr(self, 'support_vector_analysis_'):
            summary['support_vectors'] = self.support_vector_analysis_
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add SVM-specific information
        summary.update({
            'model_family': 'Support Vector Machine',
            'svm_variant': self.svm_variant.replace('_', '-').upper(),
            'kernel': self.kernel,
            'regularization_parameter': self.C if self.svm_variant == 'svc' else self.nu,
            'gamma': self.gamma,
            'degree': self.degree if self.kernel == 'poly' else None,
            'scaler_type': self.scaler_type,
            'probability_estimation': self.probability,
            'probability_calibration': self.calibrate_probabilities,
            'n_support_vectors': int(np.sum(self.n_support_)) if self.n_support_ is not None else None,
            'support_vector_ratio': getattr(self, 'support_vector_analysis_', {}).get('support_vector_ratio'),
            'kernel_complexity': self._get_kernel_complexity(),
            'n_classes': len(self.classes_) if hasattr(self, 'classes_') else None,
            'class_weight': self.class_weight
        })
        
        # Add SVM summary
        if self.is_fitted:
            try:
                summary['svm_summary'] = self.get_svm_summary()
            except Exception as e:
                logger.debug(f"Could not generate SVM summary: {e}")
        
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

def create_svm_classifier(kernel: str = 'rbf',
                         svm_variant: str = 'svc',
                         performance_preset: str = 'balanced',
                         **kwargs) -> FinancialSVMClassifier:
    """
    Create a Support Vector Machine classification model
    
    Args:
        kernel: Kernel function ('linear', 'poly', 'rbf', 'sigmoid')
        svm_variant: SVM variant ('svc', 'nu_svc')
        performance_preset: Performance preset ('fast', 'balanced', 'accurate')
        **kwargs: Additional model parameters
        
    Returns:
        Configured SVM classification model
    """
    
    # Base configuration
    base_config = {
        'name': f'{svm_variant}_{kernel}_classifier',
        'kernel': kernel,
        'svm_variant': svm_variant,
        'auto_scale': True,
        'scaler_type': 'standard',
        'probability': True,
        'calibrate_probabilities': True
    }
    
    # Performance presets
    if performance_preset == 'fast':
        preset_config = {
            'C': 1.0,
            'nu': 0.5,
            'gamma': 'scale',
            'tol': 1e-2,
            'cache_size': 100
        }
    elif performance_preset == 'balanced':
        preset_config = {
            'C': 1.0,
            'nu': 0.5,
            'gamma': 'scale',
            'tol': 1e-3,
            'cache_size': 200
        }
    elif performance_preset == 'accurate':
        preset_config = {
            'C': 1.0,
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
    
    return FinancialSVMClassifier(**config)

def create_linear_svm(**kwargs) -> FinancialSVMClassifier:
    """Create Linear SVM for high-dimensional data"""
    
    return create_svm_classifier(
        kernel='linear',
        performance_preset='balanced',
        name='linear_svm_classifier',
        **kwargs
    )

def create_rbf_svm(**kwargs) -> FinancialSVMClassifier:
    """Create RBF SVM for non-linear patterns"""
    
    return create_svm_classifier(
        kernel='rbf',
        performance_preset='balanced',
        name='rbf_svm_classifier',
        **kwargs
    )

def create_polynomial_svm(degree: int = 3, **kwargs) -> FinancialSVMClassifier:
    """Create Polynomial SVM"""
    
    return create_svm_classifier(
        kernel='poly',
        degree=degree,
        performance_preset='balanced',
        name=f'poly{degree}_svm_classifier',
        **kwargs
    )

def create_nu_svm(kernel: str = 'rbf', **kwargs) -> FinancialSVMClassifier:
    """Create Nu-SVM with automatic regularization tuning"""
    
    return create_svm_classifier(
        kernel=kernel,
        svm_variant='nu_svc',
        performance_preset='balanced',
        name=f'nu_svm_{kernel}_classifier',
        **kwargs
    )

def create_binary_svm(**kwargs) -> FinancialSVMClassifier:
    """Create SVM optimized for binary classification"""
    
    return create_svm_classifier(
        kernel='rbf',
        performance_preset='balanced',
        name='binary_svm_classifier',
        **kwargs
    )

def create_multiclass_svm(**kwargs) -> FinancialSVMClassifier:
    """Create SVM optimized for multiclass classification"""
    
    return create_svm_classifier(
        kernel='rbf',
        performance_preset='balanced',
        decision_function_shape='ovr',  # One-vs-rest for multiclass
        name='multiclass_svm_classifier',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def tune_svm_hyperparameters(X: pd.DataFrame, y: pd.Series,
                            kernel: str = 'rbf',
                            param_grid: Optional[Dict[str, List[Any]]] = None,
                            cv: int = 5,
                            scoring: str = 'accuracy',
                            n_jobs: int = 1) -> Dict[str, Any]:
    """
    Tune SVM hyperparameters using grid search
    
    Args:
        X: Feature matrix
        y: Target values
        kernel: Kernel function
        param_grid: Parameter grid for search
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs (SVM doesn't parallelize well)
        
    Returns:
        Dictionary with best parameters and scores
    """
    
    logger.info(f"Starting SVM hyperparameter tuning for {kernel} kernel")
    
    # Default parameter grids for different kernels
    if param_grid is None:
        if kernel == 'linear':
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'class_weight': [None, 'balanced']
            }
        elif kernel == 'rbf':
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'class_weight': [None, 'balanced']
            }
        elif kernel == 'poly':
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 0.01, 0.1, 1.0],
                'degree': [2, 3, 4],
                'class_weight': [None, 'balanced']
            }
        else:
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'class_weight': [None, 'balanced']
            }
    
    # Create base model
    base_model = SVC(kernel=kernel, probability=True, random_state=42)
    
    # Scale data for SVM
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
        'n_support_vectors': np.sum(grid_search.best_estimator_.n_support_)
    }
    
    logger.info(f"SVM hyperparameter tuning complete. Best score: {results['best_score']:.4f}")
    logger.info(f"Best parameters: {results['best_params']}")
    logger.info(f"Support vectors: {results['n_support_vectors']}")
    
    return results

def compare_svm_kernels(X: pd.DataFrame, y: pd.Series,
                       kernels: List[str] = ['linear', 'poly', 'rbf', 'sigmoid'],
                       cv: int = 5) -> Dict[str, Any]:
    """
    Compare different SVM kernels
    
    Args:
        X: Feature matrix
        y: Target values
        kernels: List of kernels to compare
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with comparison results
    """
    
    from sklearn.model_selection import cross_val_score
    
    logger.info(f"Comparing SVM kernels: {kernels}")
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for kernel in kernels:
        logger.info(f"Evaluating {kernel} kernel")
        
        # Create model with default parameters
        if kernel == 'poly':
            model = SVC(kernel=kernel, degree=3, gamma='scale', probability=True, random_state=42)
        else:
            model = SVC(kernel=kernel, gamma='scale', probability=True, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        
        # Fit model to get support vector information
        model.fit(X_scaled, y)
        
        results[kernel] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'n_support_vectors': np.sum(model.n_support_),
            'support_vector_ratio': np.sum(model.n_support_) / len(X),
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

def analyze_svm_complexity(X: pd.DataFrame, y: pd.Series,
                          C_range: List[float] = [0.01, 0.1, 1.0, 10.0, 100.0],
                          kernel: str = 'rbf') -> Dict[str, Any]:
    """
    Analyze SVM model complexity vs performance trade-off
    
    Args:
        X: Feature matrix
        y: Target values
        C_range: Range of C values to test
        kernel: Kernel function
        
    Returns:
        Dictionary with complexity analysis
    """
    
    from sklearn.model_selection import cross_val_score
    
    logger.info(f"Analyzing SVM complexity for {kernel} kernel")
    
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
        model = SVC(kernel=kernel, C=C, gamma='scale', probability=True, random_state=42)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        results['cv_scores'].append(cv_scores.mean())
        
        # Fit model to get complexity metrics
        model.fit(X_scaled, y)
        
        # Training score
        train_score = model.score(X_scaled, y)
        results['training_scores'].append(train_score)
        
        # Support vector information
        n_sv = np.sum(model.n_support_)
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

def create_svm_ensemble(X: pd.DataFrame, y: pd.Series,
                       kernels: List[str] = ['linear', 'rbf', 'poly'],
                       n_models: int = 3) -> Dict[str, Any]:
    """
    Create ensemble of SVM models with different kernels
    
    Args:
        X: Feature matrix
        y: Target values
        kernels: List of kernels to use
        n_models: Number of models per kernel
        
    Returns:
        Dictionary with ensemble models and predictions
    """
    
    logger.info(f"Creating SVM ensemble with kernels: {kernels}")
    
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
                model = SVC(kernel=kernel, C=C_values[i % len(C_values)], 
                           probability=True, random_state=42+i)
            elif kernel == 'rbf':
                gamma_values = [0.01, 0.1, 1.0]
                model = SVC(kernel=kernel, gamma=gamma_values[i % len(gamma_values)], 
                           probability=True, random_state=42+i)
            elif kernel == 'poly':
                degree_values = [2, 3, 4]
                model = SVC(kernel=kernel, degree=degree_values[i % len(degree_values)], 
                           probability=True, random_state=42+i)
            else:
                model = SVC(kernel=kernel, probability=True, random_state=42+i)
            
            # Fit model
            model.fit(X_scaled, y)
            
            # Make predictions
            pred = model.predict(X_scaled)
            
            kernel_models.append(model)
            kernel_predictions.append(pred)
        
        ensemble_models[kernel] = kernel_models
        predictions[kernel] = np.array(kernel_predictions)
    
    # Calculate ensemble predictions using majority voting
    all_predictions = np.concatenate([predictions[k] for k in kernels], axis=0)
    ensemble_prediction = []
    
    for i in range(len(X)):
        # Get all predictions for sample i
        sample_predictions = all_predictions[:, i]
        # Use majority voting
        unique, counts = np.unique(sample_predictions, return_counts=True)
        ensemble_prediction.append(unique[np.argmax(counts)])
    
    ensemble_prediction = np.array(ensemble_prediction)
    
    # Calculate individual and ensemble scores
    scores = {}
    for kernel in kernels:
        kernel_scores = [accuracy_score(y, pred) for pred in predictions[kernel]]
        scores[kernel] = {
            'individual_scores': kernel_scores,
            'mean_score': np.mean(kernel_scores),
            'std_score': np.std(kernel_scores)
        }
    
    ensemble_score = accuracy_score(y, ensemble_prediction)
    scores['ensemble'] = ensemble_score
    
    results = {
        'models': ensemble_models,
        'predictions': predictions,
        'ensemble_prediction': ensemble_prediction,
        'scores': scores,
        'ensemble_score': ensemble_score,
        'scaler': scaler
    }
    
    logger.info(f"SVM ensemble created. Ensemble score: {ensemble_score:.4f}")
    
    return results

def compare_svm_variants(X: pd.DataFrame, y: pd.Series,
                        kernel: str = 'rbf',
                        cv: int = 5) -> Dict[str, Any]:
    """
    Compare SVC vs Nu-SVC variants
    
    Args:
        X: Feature matrix
        y: Target values
        kernel: Kernel to use for comparison
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with comparison results
    """
    
    from sklearn.model_selection import cross_val_score
    
    logger.info(f"Comparing SVM variants (SVC vs Nu-SVC) with {kernel} kernel")
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'svc': create_svm_classifier(kernel=kernel, svm_variant='svc'),
        'nu_svc': create_svm_classifier(kernel=kernel, svm_variant='nu_svc')
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        
        # Get the underlying sklearn model
        sklearn_model = model._create_model()
        
        # Perform cross-validation
        cv_scores = cross_val_score(sklearn_model, X_scaled, y, cv=cv, scoring='accuracy')
        
        # Time a single fit for performance comparison
        import time
        start_time = time.time()
        sklearn_model.fit(X_scaled, y)
        fit_time = time.time() - start_time
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'fit_time': fit_time,
            'n_support_vectors': np.sum(sklearn_model.n_support_),
            'support_vector_ratio': np.sum(sklearn_model.n_support_) / len(X)
        }
    
    # Add comparison summary
    results['comparison'] = {
        'best_accuracy': max(results.keys(), key=lambda k: results[k]['cv_mean'] if k != 'comparison' else -1),
        'fastest_model': min(results.keys(), key=lambda k: results[k]['fit_time'] if k != 'comparison' else float('inf')),
        'most_sparse': min(results.keys(), key=lambda k: results[k]['support_vector_ratio'] if k != 'comparison' else float('inf'))
    }
    
    logger.info(f"SVM variant comparison complete. Best accuracy: {results['comparison']['best_accuracy']}")
    
    return results
