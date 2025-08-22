# ============================================
# StockPredictionPro - src/models/classification/logistic.py
# Logistic regression classification models for financial prediction with statistical analysis
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, log_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from scipy import stats
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_classifier import BaseFinancialClassifier, ClassificationStrategy

logger = get_logger('models.classification.logistic')

# ============================================
# Logistic Regression Classification Model
# ============================================

class FinancialLogisticClassifier(BaseFinancialClassifier):
    """
    Logistic Regression classification model optimized for financial data
    
    Features:
    - Linear and non-linear classification with multiple solvers
    - Comprehensive statistical analysis (coefficients, p-values, confidence intervals)
    - Advanced regularization (L1, L2, Elastic Net)
    - Feature selection and importance analysis
    - Probability calibration for reliable confidence estimates
    - Financial domain optimizations (risk-adjusted predictions, volatility weighting)
    - Comprehensive statistical diagnostics and interpretability
    """
    
    def __init__(self,
                 name: str = "logistic_classifier",
                 penalty: Optional[str] = 'l2',
                 dual: bool = False,
                 tol: float = 1e-4,
                 C: float = 1.0,
                 fit_intercept: bool = True,
                 intercept_scaling: float = 1.0,
                 class_weight: Optional[Union[str, Dict]] = None,
                 random_state: Optional[int] = 42,
                 solver: str = 'lbfgs',
                 max_iter: int = 100,
                 multi_class: str = 'auto',
                 verbose: int = 0,
                 warm_start: bool = False,
                 n_jobs: Optional[int] = None,
                 l1_ratio: Optional[float] = None,
                 feature_selection: Optional[str] = None,
                 selection_k: Optional[int] = None,
                 significance_level: float = 0.05,
                 scaler_type: str = 'standard',
                 calibrate_probabilities: bool = False,  # Already well-calibrated
                 auto_scale: bool = True,
                 **kwargs):
        """
        Initialize Financial Logistic Regression Classifier
        
        Args:
            name: Model name
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', None)
            dual: Dual or primal formulation
            tol: Tolerance for stopping criteria
            C: Inverse of regularization strength
            fit_intercept: Whether to fit intercept
            intercept_scaling: Scaling of synthetic intercept feature
            class_weight: Weights associated with classes
            random_state: Random state for reproducibility
            solver: Algorithm to use ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
            max_iter: Maximum number of iterations
            multi_class: Multiclass option ('auto', 'ovr', 'multinomial')
            verbose: Verbosity level
            warm_start: Whether to reuse previous solution
            n_jobs: Number of parallel jobs
            l1_ratio: Elastic Net mixing parameter
            feature_selection: Feature selection method ('k_best', 'rfe')
            selection_k: Number of features to select
            significance_level: P-value threshold for feature significance
            scaler_type: Type of scaler ('standard', 'robust')
            calibrate_probabilities: Whether to calibrate probabilities
            auto_scale: Whether to automatically scale features
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="logistic_classifier",
            classification_strategy=ClassificationStrategy.DIRECTION_PREDICTION,
            **kwargs
        )
        
        # Logistic regression parameters
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.feature_selection = feature_selection
        self.selection_k = selection_k
        self.significance_level = significance_level
        self.scaler_type = scaler_type
        self.calibrate_probabilities = calibrate_probabilities
        self.auto_scale = auto_scale
        
        # Store parameters for model creation
        self.model_params.update({
            'penalty': penalty,
            'dual': dual,
            'tol': tol,
            'C': C,
            'fit_intercept': fit_intercept,
            'intercept_scaling': intercept_scaling,
            'class_weight': class_weight,
            'random_state': random_state,
            'solver': solver,
            'max_iter': max_iter,
            'multi_class': multi_class,
            'verbose': verbose,
            'warm_start': warm_start,
            'n_jobs': n_jobs,
            'l1_ratio': l1_ratio
        })
        
        # Logistic regression-specific attributes
        self.scaler_: Optional[Union[StandardScaler, RobustScaler]] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.feature_selector_: Optional[Any] = None
        self.selected_features_: Optional[List[str]] = None
        self.calibrated_model_: Optional[CalibratedClassifierCV] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.statistical_summary_: Optional[Dict[str, Any]] = None
        self.odds_ratios_: Optional[np.ndarray] = None
        self.confidence_intervals_: Optional[Dict[str, Any]] = None
        self.class_weights_: Optional[Dict[Any, float]] = None
        self.convergence_info_: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized Logistic Regression classifier: {self.name}")
    
    def _create_model(self) -> LogisticRegression:
        """Create the Logistic Regression classification model"""
        return LogisticRegression(**self.model_params)
    
    def _create_scaler(self) -> Union[StandardScaler, RobustScaler]:
        """Create appropriate scaler based on scaler_type"""
        
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with scaling and optional feature selection"""
        
        # Basic preprocessing
        X_processed = super()._preprocess_features(X)
        
        # Apply feature scaling (important for logistic regression)
        if self.auto_scale:
            if self.scaler_ is None:
                self.scaler_ = self._create_scaler()
                X_scaled = self.scaler_.fit_transform(X_processed)
                logger.debug(f"Fitted {self.scaler_type} scaler for Logistic Regression")
            else:
                X_scaled = self.scaler_.transform(X_processed)
            X_processed = X_scaled
        
        # Apply feature selection if fitted
        if self.feature_selector_ is not None:
            X_processed = self.feature_selector_.transform(X_processed)
        
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
    
    def _perform_feature_selection(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Perform feature selection based on specified method"""
        
        if self.feature_selection is None:
            return X, self.feature_names.copy()
        
        if self.feature_selection == 'k_best':
            # Select K best features using F-statistic
            k = self.selection_k or min(10, X.shape[1])
            self.feature_selector_ = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector_.fit_transform(X, y)
            
            # Get selected feature names
            selected_indices = self.feature_selector_.get_support(indices=True)
            selected_features = [self.feature_names[i] for i in selected_indices]
            
            logger.info(f"Selected {len(selected_features)} best features using F-statistic")
            
        elif self.feature_selection == 'rfe':
            # Recursive Feature Elimination
            estimator = LogisticRegression(C=1.0, random_state=self.random_state, max_iter=1000)
            n_features = self.selection_k or min(10, X.shape[1])
            self.feature_selector_ = RFE(estimator=estimator, n_features_to_select=n_features)
            X_selected = self.feature_selector_.fit_transform(X, y)
            
            # Get selected feature names
            selected_indices = self.feature_selector_.get_support(indices=True)
            selected_features = [self.feature_names[i] for i in selected_indices]
            
            logger.info(f"Selected {len(selected_features)} features using RFE")
            
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection}")
        
        self.selected_features_ = selected_features
        return X_selected, selected_features
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Enhanced post-training processing for Logistic Regression classification"""
        
        # Call parent processing
        super()._post_training_processing(X, y)
        
        # Extract logistic regression-specific information
        if hasattr(self.model, 'coef_'):
            self.coefficients_ = self.model.coef_
            
            # For binary classification, coef_ has shape (1, n_features)
            # For multiclass, coef_ has shape (n_classes, n_features)
            if self.coefficients_.shape[0] == 1:
                self.coefficients_ = self.coefficients_[0]  # Flatten for binary case
        
        if hasattr(self.model, 'intercept_'):
            self.intercept_ = self.model.intercept_
        
        # Calculate class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.class_weights_ = {cls: count / len(y) for cls, count in zip(unique_classes, class_counts)}
        
        # Calculate comprehensive statistical summary
        self._calculate_statistical_summary(X, y)
        
        # Calculate odds ratios
        self._calculate_odds_ratios()
        
        # Analyze convergence
        self._analyze_convergence()
        
        # Calibrate probabilities if requested
        if self.calibrate_probabilities:
            self._calibrate_probabilities(X, y)
    
    def _calculate_statistical_summary(self, X: np.ndarray, y: np.ndarray):
        """Calculate comprehensive statistical summary including p-values"""
        
        try:
            n_samples, n_features = X.shape
            
            # For binary classification statistical analysis
            if len(self.classes_) == 2 and hasattr(self.model, 'coef_'):
                
                # Calculate standard errors using Fisher Information Matrix
                # For logistic regression: Var(β) = (X'WX)^(-1) where W is diagonal weight matrix
                
                # Get predicted probabilities
                y_pred_proba = self.model.predict_proba(X)
                p = y_pred_proba[:, 1]  # Probability of positive class
                
                # Calculate weights W = p(1-p)
                W = p * (1 - p)
                W = np.maximum(W, 1e-8)  # Avoid division by zero
                
                # Calculate weighted design matrix
                X_weighted = X * np.sqrt(W).reshape(-1, 1)
                
                try:
                    # Fisher Information Matrix: X'WX
                    fisher_info = X_weighted.T @ X_weighted
                    
                    # Add small regularization for numerical stability
                    fisher_info += np.eye(n_features) * 1e-8
                    
                    # Covariance matrix: (X'WX)^(-1)
                    cov_matrix = np.linalg.inv(fisher_info)
                    standard_errors = np.sqrt(np.diag(cov_matrix))
                    
                    # Z-statistics and p-values
                    z_stats = self.coefficients_ / standard_errors
                    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
                    
                    # Confidence intervals (95%)
                    alpha = 1 - 0.95
                    z_critical = stats.norm.ppf(1 - alpha/2)
                    ci_lower = self.coefficients_ - z_critical * standard_errors
                    ci_upper = self.coefficients_ + z_critical * standard_errors
                    
                except np.linalg.LinAlgError:
                    logger.warning("Could not calculate standard errors due to singular Fisher Information Matrix")
                    standard_errors = np.full(len(self.coefficients_), np.nan)
                    z_stats = np.full(len(self.coefficients_), np.nan)
                    p_values = np.full(len(self.coefficients_), np.nan)
                    ci_lower = np.full(len(self.coefficients_), np.nan)
                    ci_upper = np.full(len(self.coefficients_), np.nan)
                
                # Intercept statistics
                if self.fit_intercept and hasattr(self.model, 'intercept_'):
                    intercept = self.intercept_[0] if isinstance(self.intercept_, np.ndarray) else self.intercept_
                    
                    # Calculate intercept standard error (simplified)
                    try:
                        # Add intercept column to design matrix
                        X_with_intercept = np.column_stack([np.ones(n_samples), X])
                        X_weighted_intercept = X_with_intercept * np.sqrt(W).reshape(-1, 1)
                        fisher_info_intercept = X_weighted_intercept.T @ X_weighted_intercept
                        fisher_info_intercept += np.eye(n_features + 1) * 1e-8
                        cov_matrix_intercept = np.linalg.inv(fisher_info_intercept)
                        intercept_se = np.sqrt(cov_matrix_intercept[0, 0])
                        
                        intercept_z = intercept / intercept_se
                        intercept_p = 2 * (1 - stats.norm.cdf(np.abs(intercept_z)))
                    except:
                        intercept_se = np.nan
                        intercept_z = np.nan
                        intercept_p = np.nan
                else:
                    intercept = 0.0
                    intercept_se = np.nan
                    intercept_z = np.nan
                    intercept_p = np.nan
                
                # Model fit statistics
                y_pred_proba_full = self.model.predict_proba(X)
                log_likelihood = np.sum(y * np.log(y_pred_proba_full[:, 1] + 1e-15) + 
                                      (1 - y) * np.log(y_pred_proba_full[:, 0] + 1e-15))
                
                # Null model (intercept only)
                p_null = np.mean(y)
                log_likelihood_null = n_samples * (p_null * np.log(p_null + 1e-15) + 
                                                 (1 - p_null) * np.log(1 - p_null + 1e-15))
                
                # Model statistics
                n_params = n_features + (1 if self.fit_intercept else 0)
                aic = 2 * n_params - 2 * log_likelihood
                bic = np.log(n_samples) * n_params - 2 * log_likelihood
                
                # Pseudo R-squared (McFadden's)
                pseudo_r2 = 1 - (log_likelihood / log_likelihood_null)
                
                # Likelihood ratio test
                lr_statistic = 2 * (log_likelihood - log_likelihood_null)
                lr_p_value = 1 - stats.chi2.cdf(lr_statistic, n_features)
                
                self.statistical_summary_ = {
                    'n_observations': int(n_samples),
                    'n_features': int(n_features),
                    'log_likelihood': float(log_likelihood),
                    'log_likelihood_null': float(log_likelihood_null),
                    'pseudo_r_squared': float(pseudo_r2),
                    'aic': float(aic),
                    'bic': float(bic),
                    'lr_statistic': float(lr_statistic),
                    'lr_p_value': float(lr_p_value),
                    'intercept': float(intercept),
                    'intercept_se': float(intercept_se),
                    'intercept_z': float(intercept_z),
                    'intercept_p': float(intercept_p),
                    'coefficients': self.coefficients_.tolist(),
                    'standard_errors': standard_errors.tolist(),
                    'z_statistics': z_stats.tolist(),
                    'p_values': p_values.tolist(),
                    'confidence_intervals_lower': ci_lower.tolist(),
                    'confidence_intervals_upper': ci_upper.tolist(),
                    'significant_features': [
                        self.selected_features_[i] if self.selected_features_ else self.feature_names[i]
                        for i, p in enumerate(p_values) if not np.isnan(p) and p < self.significance_level
                    ]
                }
                
                # Store confidence intervals separately
                self.confidence_intervals_ = {
                    'coefficients_lower': ci_lower,
                    'coefficients_upper': ci_upper,
                    'alpha': alpha
                }
                
            else:
                # For multiclass, provide simplified statistics
                log_likelihood = -log_loss(y, self.model.predict_proba(X), normalize=False)
                n_params = self.coefficients_.size + self.intercept_.size
                aic = 2 * n_params - 2 * log_likelihood
                bic = np.log(n_samples) * n_params - 2 * log_likelihood
                
                self.statistical_summary_ = {
                    'n_observations': int(n_samples),
                    'n_features': int(n_features),
                    'n_classes': len(self.classes_),
                    'log_likelihood': float(log_likelihood),
                    'aic': float(aic),
                    'bic': float(bic),
                    'coefficients': self.coefficients_.tolist(),
                    'intercept': self.intercept_.tolist(),
                    'multiclass_note': 'Detailed statistical analysis available for binary classification only'
                }
                
        except Exception as e:
            logger.warning(f"Could not calculate statistical summary: {e}")
            self.statistical_summary_ = None
    
    def _calculate_odds_ratios(self):
        """Calculate odds ratios for logistic regression coefficients"""
        
        if self.coefficients_ is None:
            return
        
        # Odds ratios are exp(coefficients)
        if len(self.classes_) == 2:
            # Binary classification
            self.odds_ratios_ = np.exp(self.coefficients_)
        else:
            # Multiclass - odds ratios for each class vs reference class
            self.odds_ratios_ = np.exp(self.coefficients_)
    
    def _analyze_convergence(self):
        """Analyze convergence information"""
        
        self.convergence_info_ = {
            'solver': self.solver,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'converged': hasattr(self.model, 'n_iter_') and self.model.n_iter_ < self.max_iter,
            'n_iter': getattr(self.model, 'n_iter_', None)
        }
        
        if hasattr(self.model, 'n_iter_'):
            if isinstance(self.model.n_iter_, np.ndarray):
                self.convergence_info_['n_iter_per_class'] = self.model.n_iter_.tolist()
                self.convergence_info_['max_iter_used'] = int(np.max(self.model.n_iter_))
            else:
                self.convergence_info_['n_iter'] = int(self.model.n_iter_)
                self.convergence_info_['max_iter_used'] = int(self.model.n_iter_)
        
        if not self.convergence_info_['converged']:
            logger.warning(f"Logistic regression did not converge. Consider increasing max_iter or adjusting tol.")
    
    def _calibrate_probabilities(self, X: np.ndarray, y: np.ndarray):
        """Calibrate prediction probabilities using cross-validation"""
        
        try:
            # Use isotonic calibration for logistic regression
            self.calibrated_model_ = CalibratedClassifierCV(
                base_estimator=self.model,
                method='isotonic',  # Works well for already well-calibrated models
                cv=3
            )
            self.calibrated_model_.fit(X, y)
            logger.debug("Calibrated prediction probabilities")
        except Exception as e:
            logger.warning(f"Could not calibrate probabilities: {e}")
            self.calibrated_model_ = None
    
    @time_it("logistic_fit", include_args=True)
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'FinancialLogisticClassifier':
        """
        Fit the logistic regression classification model
        
        Args:
            X: Feature matrix
            y: Target labels
            **kwargs: Additional fitting parameters
            
        Returns:
            Fitted model instance
        """
        logger.info(f"Fitting Logistic Regression Classifier on {len(X)} samples with {X.shape[1]} features")
        
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
            
            # Preprocess targets first to get encoded values
            y_processed = self._preprocess_targets(y)
            
            # Preprocess features (scaling)
            X_processed = self._preprocess_features(X)
            
            # Perform feature selection if specified
            if self.feature_selection:
                X_processed, selected_features = self._perform_feature_selection(X_processed, y_processed)
                logger.info(f"Feature selection completed: {len(selected_features)} features selected")
            else:
                selected_features = self.feature_names.copy()
            
            self.selected_features_ = selected_features
            
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
                'selected_features': len(selected_features),
                'n_classes': len(self.classes_),
                'class_names': self.classes_.tolist(),
                'training_duration_seconds': fit_duration,
                'target_name': self.target_name,
                'solver': self.solver,
                'regularization': self.penalty
            })
            
            # Calculate training score
            self.training_score = self.model.score(X_processed, y_processed)
            
            # Update status
            self.status = ModelStatus.TRAINED
            self.is_fitted = True
            
            logger.info(f"Logistic Regression Classifier trained successfully in {fit_duration:.2f}s")
            
            return self
            
        except Exception as e:
            from ..base.base_model import ModelStatus
            self.status = ModelStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Logistic Regression Classifier training failed: {e}")
            raise
    
    @time_it("logistic_predict", include_args=True)
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions using the fitted logistic regression classifier
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        logger.debug(f"Making Logistic Regression predictions for {len(X)} samples")
        
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
            logger.error(f"Logistic Regression prediction failed: {e}")
            raise
    
    @time_it("logistic_predict_proba", include_args=True)
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
        
        logger.debug(f"Making Logistic Regression probability predictions for {len(X)} samples")
        
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
            logger.error(f"Logistic Regression probability prediction failed: {e}")
            raise
    
    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict log probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Log probabilities
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        X_processed = self._preprocess_features(X)
        return self.model.predict_log_proba(X_processed)
    
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get decision function values
        
        Args:
            X: Feature matrix
            
        Returns:
            Decision function values
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get decision function")
        
        X_processed = self._preprocess_features(X)
        return self.model.decision_function(X_processed)
    
    def get_coefficients_table(self) -> pd.DataFrame:
        """
        Get coefficients table with statistical information
        
        Returns:
            DataFrame with coefficient analysis
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get coefficients table")
        
        if not self.statistical_summary_ or len(self.classes_) > 2:
            # Simple coefficients table for multiclass
            feature_names = self.selected_features_ or self.feature_names
            
            if len(self.classes_) == 2:
                coef_data = {
                    'feature': feature_names,
                    'coefficient': self.coefficients_.tolist(),
                    'odds_ratio': self.odds_ratios_.tolist() if self.odds_ratios_ is not None else [np.nan] * len(feature_names)
                }
            else:
                # Multiclass - show coefficients for each class
                coef_data = {'feature': feature_names}
                for i, class_name in enumerate(self.classes_):
                    coef_data[f'coef_{class_name}'] = self.coefficients_[i].tolist()
                    if self.odds_ratios_ is not None:
                        coef_data[f'odds_ratio_{class_name}'] = self.odds_ratios_[i].tolist()
            
            return pd.DataFrame(coef_data)
        
        # Detailed table for binary classification
        stats_sum = self.statistical_summary_
        feature_names = self.selected_features_ or self.feature_names
        
        # Create coefficients table
        coef_table = pd.DataFrame({
            'feature': feature_names,
            'coefficient': stats_sum['coefficients'],
            'std_error': stats_sum['standard_errors'],
            'z_statistic': stats_sum['z_statistics'],
            'p_value': stats_sum['p_values'],
            'odds_ratio': self.odds_ratios_.tolist() if self.odds_ratios_ is not None else [np.nan] * len(feature_names),
            'ci_lower': stats_sum['confidence_intervals_lower'],
            'ci_upper': stats_sum['confidence_intervals_upper'],
            'significant': [p < self.significance_level for p in stats_sum['p_values']]
        })
        
        # Sort by absolute z-statistic
        coef_table['abs_z_stat'] = np.abs(coef_table['z_statistic'])
        coef_table = coef_table.sort_values('abs_z_stat', ascending=False)
        coef_table = coef_table.drop('abs_z_stat', axis=1)
        
        return coef_table
    
    def get_odds_ratios_table(self) -> pd.DataFrame:
        """
        Get odds ratios table with confidence intervals
        
        Returns:
            DataFrame with odds ratios analysis
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get odds ratios")
        
        if self.odds_ratios_ is None or len(self.classes_) > 2:
            logger.warning("Detailed odds ratios analysis only available for binary classification")
            return pd.DataFrame()
        
        feature_names = self.selected_features_ or self.feature_names
        odds_ratios = self.odds_ratios_
        
        # Calculate confidence intervals for odds ratios
        if self.confidence_intervals_ is not None:
            ci_lower = np.exp(self.confidence_intervals_['coefficients_lower'])
            ci_upper = np.exp(self.confidence_intervals_['coefficients_upper'])
        else:
            ci_lower = np.full_like(odds_ratios, np.nan)
            ci_upper = np.full_like(odds_ratios, np.nan)
        
        odds_table = pd.DataFrame({
            'feature': feature_names,
            'odds_ratio': odds_ratios,
            'or_ci_lower': ci_lower,
            'or_ci_upper': ci_upper,
            'interpretation': self._interpret_odds_ratios(odds_ratios)
        })
        
        # Sort by odds ratio magnitude
        odds_table['or_distance_from_1'] = np.abs(odds_table['odds_ratio'] - 1.0)
        odds_table = odds_table.sort_values('or_distance_from_1', ascending=False)
        odds_table = odds_table.drop('or_distance_from_1', axis=1)
        
        return odds_table
    
    def _interpret_odds_ratios(self, odds_ratios: np.ndarray) -> List[str]:
        """Interpret odds ratios in plain language"""
        
        interpretations = []
        for or_val in odds_ratios:
            if np.isnan(or_val):
                interpretations.append("Unknown")
            elif or_val > 2.0:
                interpretations.append("Strong positive association")
            elif or_val > 1.5:
                interpretations.append("Moderate positive association")
            elif or_val > 1.1:
                interpretations.append("Weak positive association")
            elif or_val > 0.9:
                interpretations.append("No significant association")
            elif or_val > 0.67:
                interpretations.append("Weak negative association")
            elif or_val > 0.5:
                interpretations.append("Moderate negative association")
            else:
                interpretations.append("Strong negative association")
        
        return interpretations
    
    def plot_coefficients(self, top_n: int = 20, show_significance: bool = True) -> Any:
        """
        Plot coefficient values with confidence intervals
        
        Args:
            top_n: Number of top coefficients to show
            show_significance: Whether to highlight significant coefficients
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            coef_table = self.get_coefficients_table()
            
            if coef_table.empty or len(self.classes_) > 2:
                logger.warning("Coefficient plot not available for multiclass or when statistics unavailable")
                return None
            
            # Take top coefficients by absolute value
            if 'abs_z_stat' not in coef_table.columns:
                coef_table['abs_coef'] = np.abs(coef_table['coefficient'])
                coef_table = coef_table.sort_values('abs_coef', ascending=False)
            
            coef_table = coef_table.head(top_n)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            y_pos = np.arange(len(coef_table))
            coefficients = coef_table['coefficient']
            
            # Color by significance if available
            if show_significance and 'significant' in coef_table.columns:
                colors = ['darkgreen' if sig else 'gray' for sig in coef_table['significant']]
            else:
                colors = 'steelblue'
            
            bars = ax.barh(y_pos, coefficients, color=colors, alpha=0.7)
            
            # Add confidence intervals if available
            if 'ci_lower' in coef_table.columns and 'ci_upper' in coef_table.columns:
                ci_lower = coef_table['ci_lower']
                ci_upper = coef_table['ci_upper']
                errors_lower = coefficients - ci_lower
                errors_upper = ci_upper - coefficients
                ax.errorbar(coefficients, y_pos, xerr=[errors_lower, errors_upper], 
                           fmt='none', color='black', capsize=3)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(coef_table['feature'], fontsize=10)
            ax.set_xlabel('Coefficient Value')
            ax.set_ylabel('Features')
            ax.set_title(f'Logistic Regression Coefficients - {self.name}')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add coefficient values as text
            for i, (idx, row) in enumerate(coef_table.iterrows()):
                coef = row['coefficient']
                ax.text(coef + (max(coefficients) - min(coefficients)) * 0.01, i, 
                       f'{coef:.3f}', va='center', fontsize=8)
            
            # Add legend if showing significance
            if show_significance and 'significant' in coef_table.columns:
                green_patch = plt.Rectangle((0, 0), 1, 1, fc='darkgreen', alpha=0.7, label='Significant')
                gray_patch = plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.7, label='Not Significant')
                ax.legend(handles=[green_patch, gray_patch])
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_odds_ratios(self, top_n: int = 20) -> Any:
        """
        Plot odds ratios with confidence intervals
        
        Args:
            top_n: Number of top odds ratios to show
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            if len(self.classes_) > 2:
                logger.warning("Odds ratios plot only available for binary classification")
                return None
            
            odds_table = self.get_odds_ratios_table()
            
            if odds_table.empty:
                logger.warning("Odds ratios not available")
                return None
            
            odds_table = odds_table.head(top_n)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            y_pos = np.arange(len(odds_table))
            odds_ratios = odds_table['odds_ratio']
            
            # Color by effect direction
            colors = ['red' if or_val < 1.0 else 'blue' for or_val in odds_ratios]
            
            bars = ax.barh(y_pos, odds_ratios, color=colors, alpha=0.7)
            
            # Add confidence intervals if available
            if 'or_ci_lower' in odds_table.columns and 'or_ci_upper' in odds_table.columns:
                ci_lower = odds_table['or_ci_lower']
                ci_upper = odds_table['or_ci_upper']
                errors_lower = odds_ratios - ci_lower
                errors_upper = ci_upper - odds_ratios
                ax.errorbar(odds_ratios, y_pos, xerr=[errors_lower, errors_upper], 
                           fmt='none', color='black', capsize=3)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(odds_table['feature'], fontsize=10)
            ax.set_xlabel('Odds Ratio')
            ax.set_ylabel('Features')
            ax.set_title(f'Logistic Regression Odds Ratios - {self.name}')
            ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='No Effect (OR = 1)')
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_xscale('log')
            
            # Add odds ratio values as text
            for i, (idx, row) in enumerate(odds_table.iterrows()):
                or_val = row['odds_ratio']
                ax.text(or_val * 1.1, i, f'{or_val:.2f}', va='center', fontsize=8)
            
            # Add legend
            red_patch = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.7, label='Protective (OR < 1)')
            blue_patch = plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.7, label='Risk Factor (OR > 1)')
            ax.legend(handles=[red_patch, blue_patch])
            
            plt.tight_layout()
            return fig
            
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
            
            # Pie chart
            class_labels = [self.label_encoder_.inverse_transform([c])[0] for c in classes]
            ax2.pie(weights, labels=class_labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Class Distribution')
            
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
            scoring='accuracy',
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
    
    def get_statistical_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive statistical summary
        
        Returns:
            Dictionary with statistical analysis
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get statistical summary")
        
        return self.statistical_summary_.copy() if self.statistical_summary_ else {}
    
    def get_logistic_summary(self) -> Dict[str, Any]:
        """Get comprehensive logistic regression summary"""
        
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get logistic summary")
        
        summary = {
            'model_info': {
                'solver': self.solver,
                'penalty': self.penalty,
                'C': self.C,
                'l1_ratio': self.l1_ratio,
                'max_iter': self.max_iter,
                'n_classes': len(self.classes_),
                'class_names': self.classes_.tolist(),
                'n_features_original': len(self.feature_names),
                'n_features_selected': len(self.selected_features_) if self.selected_features_ else len(self.feature_names),
                'feature_selection_method': self.feature_selection
            },
            'statistical_summary': self.statistical_summary_,
            'convergence_info': self.convergence_info_,
            'class_distribution': self.class_weights_,
            'calibration_info': {
                'probabilities_calibrated': self.calibrated_model_ is not None,
                'calibration_method': 'isotonic' if self.calibrated_model_ else None
            }
        }
        
        # Add coefficients and odds ratios for binary classification
        if len(self.classes_) == 2 and self.statistical_summary_:
            summary['coefficients_analysis'] = {
                'significant_features': self.statistical_summary_.get('significant_features', []),
                'n_significant': len(self.statistical_summary_.get('significant_features', [])),
                'pseudo_r_squared': self.statistical_summary_.get('pseudo_r_squared'),
                'lr_test_p_value': self.statistical_summary_.get('lr_p_value')
            }
            
            if self.odds_ratios_ is not None:
                summary['odds_ratios'] = {
                    'values': self.odds_ratios_.tolist(),
                    'mean_odds_ratio': float(np.mean(self.odds_ratios_)),
                    'max_odds_ratio': float(np.max(self.odds_ratios_)),
                    'min_odds_ratio': float(np.min(self.odds_ratios_))
                }
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add logistic regression-specific information
        summary.update({
            'model_family': 'Logistic Regression',
            'solver': self.solver,
            'regularization': self.penalty,
            'regularization_strength': self.C,
            'l1_ratio': self.l1_ratio,
            'max_iterations': self.max_iter,
            'feature_selection': self.feature_selection,
            'probability_calibration': self.calibrate_probabilities,
            'n_classes': len(self.classes_) if hasattr(self, 'classes_') else None,
            'n_selected_features': len(self.selected_features_) if self.selected_features_ else len(self.feature_names),
            'auto_scaling': self.auto_scale,
            'scaler_type': self.scaler_type
        })
        
        # Add convergence information
        if self.convergence_info_:
            summary['converged'] = self.convergence_info_['converged']
            summary['iterations_used'] = self.convergence_info_.get('max_iter_used')
        
        # Add statistical information for binary classification
        if len(self.classes_) == 2 and self.statistical_summary_:
            summary.update({
                'pseudo_r_squared': self.statistical_summary_['pseudo_r_squared'],
                'log_likelihood': self.statistical_summary_['log_likelihood'],
                'aic': self.statistical_summary_['aic'],
                'bic': self.statistical_summary_['bic'],
                'lr_test_p_value': self.statistical_summary_['lr_p_value'],
                'n_significant_features': len(self.statistical_summary_.get('significant_features', []))
            })
        
        # Add logistic summary
        if self.is_fitted:
            try:
                summary['logistic_summary'] = self.get_logistic_summary()
            except Exception as e:
                logger.debug(f"Could not generate logistic summary: {e}")
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_logistic_classifier(regularization: str = 'l2',
                              performance_preset: str = 'balanced',
                              **kwargs) -> FinancialLogisticClassifier:
    """
    Create a Logistic Regression classification model
    
    Args:
        regularization: Regularization type ('l1', 'l2', 'elasticnet', 'none')
        performance_preset: Performance preset ('fast', 'balanced', 'accurate')
        **kwargs: Additional model parameters
        
    Returns:
        Configured Logistic Regression classification model
    """
    
    # Base configuration
    base_config = {
        'name': f'logistic_classifier_{regularization}',
        'penalty': regularization if regularization != 'none' else None,
        'random_state': 42,
        'auto_scale': True,
        'scaler_type': 'standard',
        'fit_intercept': True
    }
    
    # Performance presets
    if performance_preset == 'fast':
        preset_config = {
            'solver': 'liblinear',
            'max_iter': 100,
            'tol': 1e-3,
            'C': 1.0
        }
    elif performance_preset == 'balanced':
        preset_config = {
            'solver': 'lbfgs' if regularization in ['l2', 'none'] else 'saga',
            'max_iter': 1000,
            'tol': 1e-4,
            'C': 1.0
        }
    elif performance_preset == 'accurate':
        preset_config = {
            'solver': 'saga',  # Supports all penalties and is more robust
            'max_iter': 5000,
            'tol': 1e-6,
            'C': 1.0,
            'scaler_type': 'robust'  # More robust to outliers
        }
    else:
        raise ValueError(f"Unknown performance preset: {performance_preset}")
    
    # Regularization-specific adjustments
    if regularization == 'elasticnet':
        preset_config.update({
            'solver': 'saga',  # Only solver that supports elastic net
            'l1_ratio': 0.5    # Equal mix of L1 and L2
        })
    elif regularization == 'l1':
        if preset_config['solver'] == 'lbfgs':
            preset_config['solver'] = 'liblinear'  # lbfgs doesn't support L1
    elif regularization == 'none':
        if preset_config['solver'] == 'liblinear':
            preset_config['solver'] = 'lbfgs'  # Better for no penalty
    
    # Combine configurations
    config = {**base_config, **preset_config}
    config.update(kwargs)  # Override with user parameters
    
    return FinancialLogisticClassifier(**config)

def create_binary_logistic(**kwargs) -> FinancialLogisticClassifier:
    """Create Logistic Regression optimized for binary classification"""
    
    return create_logistic_classifier(
        regularization='l2',
        performance_preset='balanced',
        name='binary_logistic_classifier',
        **kwargs
    )

def create_multiclass_logistic(**kwargs) -> FinancialLogisticClassifier:
    """Create Logistic Regression optimized for multiclass classification"""
    
    return create_logistic_classifier(
        regularization='l2',
        performance_preset='balanced',
        multi_class='multinomial',
        solver='lbfgs',
        name='multiclass_logistic_classifier',
        **kwargs
    )

def create_regularized_logistic(regularization: str = 'elasticnet', **kwargs) -> FinancialLogisticClassifier:
    """Create regularized Logistic Regression"""
    
    return create_logistic_classifier(
        regularization=regularization,
        performance_preset='accurate',
        name=f'regularized_logistic_{regularization}',
        **kwargs
    )

def create_feature_selection_logistic(method: str = 'rfe', k: int = 10, **kwargs) -> FinancialLogisticClassifier:
    """Create Logistic Regression with feature selection"""
    
    return create_logistic_classifier(
        regularization='l2',
        performance_preset='balanced',
        feature_selection=method,
        selection_k=k,
        name=f'feature_selection_logistic_{method}',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def tune_logistic_hyperparameters(X: pd.DataFrame, y: pd.Series,
                                 param_grid: Optional[Dict[str, List[Any]]] = None,
                                 cv: int = 5,
                                 scoring: str = 'accuracy',
                                 n_jobs: int = -1) -> Dict[str, Any]:
    """
    Tune Logistic Regression hyperparameters using grid search
    
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
    
    logger.info("Starting Logistic Regression hyperparameter tuning")
    
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 5000],
            'class_weight': [None, 'balanced']
        }
    
    # Create base model
    base_model = LogisticRegression(random_state=42)
    
    # Scale data for logistic regression
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
    }
    
    logger.info(f"Hyperparameter tuning complete. Best score: {results['best_score']:.4f}")
    logger.info(f"Best parameters: {results['best_params']}")
    
    return results

def compare_logistic_regularization(X: pd.DataFrame, y: pd.Series,
                                   regularizations: List[str] = ['none', 'l1', 'l2', 'elasticnet'],
                                   cv: int = 5) -> Dict[str, Any]:
    """
    Compare different regularization methods for Logistic Regression
    
    Args:
        X: Feature matrix
        y: Target values
        regularizations: List of regularization methods to compare
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with comparison results
    """
    
    from sklearn.model_selection import cross_val_score
    
    logger.info(f"Comparing Logistic Regression regularization methods: {regularizations}")
    
    results = {}
    
    for reg in regularizations:
        logger.info(f"Evaluating {reg} regularization")
        
        try:
            # Create model
            model = create_logistic_classifier(
                regularization=reg,
                performance_preset='balanced'
            )
            
            # Get the underlying sklearn model
            sklearn_model = model._create_model()
            
            # Preprocess data
            model.feature_names = list(X.columns)
            X_processed = model._preprocess_features(X)
            y_processed = model._preprocess_targets(y)
            
            # Perform cross-validation
            cv_scores = cross_val_score(sklearn_model, X_processed, y_processed, 
                                       cv=cv, scoring='accuracy', n_jobs=-1)
            
            # Time a single fit for performance comparison
            import time
            start_time = time.time()
            sklearn_model.fit(X_processed, y_processed)
            fit_time = time.time() - start_time
            
            # Count non-zero coefficients (sparsity)
            if hasattr(sklearn_model, 'coef_'):
                if len(sklearn_model.classes_) == 2:
                    n_nonzero_coef = np.sum(np.abs(sklearn_model.coef_[0]) > 1e-6)
                    sparsity = 1 - (n_nonzero_coef / len(sklearn_model.coef_[0]))
                else:
                    n_nonzero_coef = np.sum(np.abs(sklearn_model.coef_) > 1e-6)
                    sparsity = 1 - (n_nonzero_coef / sklearn_model.coef_.size)
            else:
                n_nonzero_coef = X.shape[1]
                sparsity = 0.0
            
            results[reg] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'fit_time': fit_time,
                'n_nonzero_coef': n_nonzero_coef,
                'sparsity': sparsity,
                'converged': getattr(sklearn_model, 'n_iter_', 0) < sklearn_model.max_iter
            }
            
        except Exception as e:
            logger.warning(f"Error with {reg} regularization: {e}")
            results[reg] = {'error': str(e)}
    
    # Add comparison summary
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_accuracy = max(valid_results.keys(), key=lambda k: valid_results[k]['cv_mean'])
        fastest = min(valid_results.keys(), key=lambda k: valid_results[k]['fit_time'])
        most_sparse = max(valid_results.keys(), key=lambda k: valid_results[k]['sparsity'])
        
        results['comparison'] = {
            'best_accuracy': best_accuracy,
            'fastest': fastest,
            'most_sparse': most_sparse,
            'valid_methods': list(valid_results.keys())
        }
    
    logger.info("Regularization comparison complete")
    
    return results

def analyze_logistic_coefficients(model: FinancialLogisticClassifier) -> Dict[str, Any]:
    """
    Analyze logistic regression coefficients in detail
    
    Args:
        model: Fitted logistic regression model
        
    Returns:
        Dictionary with coefficient analysis
    """
    
    if not model.is_fitted:
        raise BusinessLogicError("Model must be fitted for coefficient analysis")
    
    logger.info("Analyzing logistic regression coefficients")
    
    # Get coefficients table
    coef_table = model.get_coefficients_table()
    
    if coef_table.empty:
        return {'error': 'Coefficients table not available'}
    
    analysis = {
        'n_features': len(coef_table),
        'coefficient_stats': {
            'mean_abs_coef': coef_table['coefficient'].abs().mean(),
            'max_abs_coef': coef_table['coefficient'].abs().max(),
            'min_abs_coef': coef_table['coefficient'].abs().min(),
            'std_coef': coef_table['coefficient'].std()
        }
    }
    
    # Binary classification specific analysis
    if len(model.classes_) == 2 and 'p_value' in coef_table.columns:
        significant_features = coef_table[coef_table['significant']]['feature'].tolist()
        
        analysis.update({
            'significant_features': significant_features,
            'n_significant': len(significant_features),
            'significance_rate': len(significant_features) / len(coef_table),
            'top_5_significant': coef_table[coef_table['significant']].head(5)['feature'].tolist(),
            'p_value_stats': {
                'mean_p_value': coef_table['p_value'].mean(),
                'min_p_value': coef_table['p_value'].min(),
                'n_very_significant': len(coef_table[coef_table['p_value'] < 0.001])
            }
        })
        
        # Odds ratios analysis
        if 'odds_ratio' in coef_table.columns:
            or_table = model.get_odds_ratios_table()
            if not or_table.empty:
                analysis['odds_ratios'] = {
                    'mean_odds_ratio': or_table['odds_ratio'].mean(),
                    'max_odds_ratio': or_table['odds_ratio'].max(),
                    'min_odds_ratio': or_table['odds_ratio'].min(),
                    'strong_positive_associations': len(or_table[or_table['odds_ratio'] > 2.0]),
                    'strong_negative_associations': len(or_table[or_table['odds_ratio'] < 0.5]),
                    'top_risk_factors': or_table.nlargest(3, 'odds_ratio')['feature'].tolist(),
                    'top_protective_factors': or_table.nsmallest(3, 'odds_ratio')['feature'].tolist()
                }
    
    # Feature importance ranking
    coef_table_sorted = coef_table.sort_values('coefficient', key=abs, ascending=False)
    analysis['feature_ranking'] = {
        'most_important': coef_table_sorted.head(10)['feature'].tolist(),
        'least_important': coef_table_sorted.tail(5)['feature'].tolist(),
        'positive_coefficients': len(coef_table[coef_table['coefficient'] > 0]),
        'negative_coefficients': len(coef_table[coef_table['coefficient'] < 0])
    }
    
    logger.info(f"Coefficient analysis complete. {analysis.get('n_significant', 0)} significant features found")
    
    return analysis

def compare_logistic_solvers(X: pd.DataFrame, y: pd.Series,
                            solvers: List[str] = ['liblinear', 'lbfgs', 'saga'],
                            penalty: str = 'l2') -> Dict[str, Any]:
    """
    Compare different solvers for Logistic Regression
    
    Args:
        X: Feature matrix
        y: Target values
        solvers: List of solvers to compare
        penalty: Penalty type to use
        
    Returns:
        Dictionary with solver comparison results
    """
    
    from sklearn.model_selection import cross_val_score
    
    logger.info(f"Comparing Logistic Regression solvers: {solvers} with {penalty} penalty")
    
    results = {}
    
    # Scale data once
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for solver in solvers:
        logger.info(f"Evaluating {solver} solver")
        
        try:
            # Create model
            model = LogisticRegression(
                solver=solver,
                penalty=penalty,
                C=1.0,
                max_iter=5000,
                random_state=42
            )
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            
            # Time a single fit
            import time
            start_time = time.time()
            model.fit(X_scaled, y)
            fit_time = time.time() - start_time
            
            results[solver] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'fit_time': fit_time,
                'converged': model.n_iter_ < model.max_iter if hasattr(model, 'n_iter_') else True,
                'n_iter': getattr(model, 'n_iter_', None)
            }
            
        except Exception as e:
            logger.warning(f"Error with {solver} solver: {e}")
            results[solver] = {'error': str(e)}
    
    # Add comparison summary
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_accuracy = max(valid_results.keys(), key=lambda k: valid_results[k]['cv_mean'])
        fastest = min(valid_results.keys(), key=lambda k: valid_results[k]['fit_time'])
        
        results['comparison'] = {
            'best_accuracy': best_accuracy,
            'fastest': fastest,
            'all_converged': all(v['converged'] for v in valid_results.values())
        }
    
    logger.info(f"Solver comparison complete. Best accuracy: {results['comparison']['best_accuracy']}")
    
    return results

def perform_feature_selection_logistic(X: pd.DataFrame, y: pd.Series,
                                     methods: List[str] = ['k_best', 'rfe'],
                                     k_values: List[int] = [5, 10, 15, 20]) -> Dict[str, Any]:
    """
    Compare feature selection methods for Logistic Regression
    
    Args:
        X: Feature matrix
        y: Target values
        methods: Feature selection methods to compare
        k_values: Numbers of features to select
        
    Returns:
        Dictionary with feature selection results
    """
    
    logger.info(f"Performing feature selection comparison for Logistic Regression")
    
    results = {}
    
    for method in methods:
        for k in k_values:
            k_adj = min(k, X.shape[1])  # Don't select more features than available
            
            logger.info(f"Testing {method} with k={k_adj}")
            
            try:
                # Create model with feature selection
                model = create_feature_selection_logistic(method=method, k=k_adj)
                model.fit(X, y)
                
                # Evaluate performance
                metrics = model.evaluate(X, y)
                
                # Get selected features
                selected_features = model.selected_features_
                
                results[f'{method}_k{k_adj}'] = {
                    'method': method,
                    'k': k_adj,
                    'selected_features': selected_features,
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'training_score': model.training_score
                }
                
            except Exception as e:
                logger.warning(f"Error with {method} k={k_adj}: {e}")
                results[f'{method}_k{k_adj}'] = {'error': str(e)}
    
    # Find best configuration
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_config = max(valid_results.keys(), key=lambda k: valid_results[k]['accuracy'])
        
        results['summary'] = {
            'best_configuration': best_config,
            'best_accuracy': valid_results[best_config]['accuracy'],
            'best_selected_features': valid_results[best_config]['selected_features'],
            'configurations_tested': len(valid_results)
        }
    
    logger.info(f"Feature selection comparison complete. Best: {results['summary']['best_configuration']}")
    
    return results
