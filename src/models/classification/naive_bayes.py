# ============================================
# StockPredictionPro - src/models/classification/naive_bayes.py
# Naive Bayes classification models for financial prediction with probabilistic analysis
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.naive_bayes import (
    GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
)
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, log_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_classifier import BaseFinancialClassifier, ClassificationStrategy

logger = get_logger('models.classification.naive_bayes')

# ============================================
# Naive Bayes Classification Model
# ============================================

class FinancialNaiveBayesClassifier(BaseFinancialClassifier):
    """
    Naive Bayes classification model optimized for financial data
    
    Features:
    - Multiple Naive Bayes variants (Gaussian, Multinomial, Bernoulli, Complement, Categorical)
    - Comprehensive probabilistic analysis and Bayesian statistics
    - Feature independence analysis and conditional probability examination
    - Advanced smoothing and regularization techniques
    - Financial domain optimizations (market regime detection, sentiment analysis)
    - Probability calibration for reliable confidence estimates
    - Class prior analysis and posterior probability interpretation
    """
    
    def __init__(self,
                 name: str = "naive_bayes_classifier",
                 nb_variant: str = 'gaussian',
                 alpha: float = 1.0,
                 fit_prior: bool = True,
                 class_prior: Optional[np.ndarray] = None,
                 var_smoothing: float = 1e-9,
                 min_categories: Optional[int] = None,
                 force_alpha: bool = True,
                 norm: bool = False,
                 binarize: Optional[float] = 0.0,
                 feature_selection: Optional[str] = None,
                 selection_k: Optional[int] = None,
                 scaler_type: str = 'none',
                 calibrate_probabilities: bool = True,
                 auto_scale: bool = False,
                 **kwargs):
        """
        Initialize Financial Naive Bayes Classifier
        
        Args:
            name: Model name
            nb_variant: Naive Bayes variant ('gaussian', 'multinomial', 'bernoulli', 'complement', 'categorical')
            alpha: Additive (Laplace/Lidstone) smoothing parameter
            fit_prior: Whether to learn class prior probabilities
            class_prior: Prior probabilities of the classes
            var_smoothing: Portion of largest variance added to variances (Gaussian only)
            min_categories: Minimum number of categories per feature (Categorical only)
            force_alpha: If False and alpha is less than 1e-10, use alpha = 1e-10
            norm: Whether to perform a second normalization (Complement only)
            binarize: Threshold for binarizing features (Bernoulli only)
            feature_selection: Feature selection method ('k_best_chi2', 'k_best_f')
            selection_k: Number of features to select
            scaler_type: Type of scaler ('none', 'standard', 'minmax')
            calibrate_probabilities: Whether to calibrate prediction probabilities
            auto_scale: Whether to automatically scale features
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="naive_bayes_classifier",
            classification_strategy=ClassificationStrategy.DIRECTION_PREDICTION,
            **kwargs
        )
        
        # Naive Bayes parameters
        self.nb_variant = nb_variant
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.var_smoothing = var_smoothing
        self.min_categories = min_categories
        self.force_alpha = force_alpha
        self.norm = norm
        self.binarize = binarize
        self.feature_selection = feature_selection
        self.selection_k = selection_k
        self.scaler_type = scaler_type
        self.calibrate_probabilities = calibrate_probabilities
        self.auto_scale = auto_scale
        
        # Naive Bayes-specific attributes
        self.scaler_: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.feature_selector_: Optional[Any] = None
        self.selected_features_: Optional[List[str]] = None
        self.calibrated_model_: Optional[CalibratedClassifierCV] = None
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None
        self.class_count_: Optional[np.ndarray] = None
        self.feature_count_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None  # For Gaussian NB
        self.var_: Optional[np.ndarray] = None    # For Gaussian NB
        self.bayesian_analysis_: Optional[Dict[str, Any]] = None
        self.independence_analysis_: Optional[Dict[str, Any]] = None
        self.class_weights_: Optional[Dict[Any, float]] = None
        
        logger.info(f"Initialized {nb_variant.replace('_', ' ').title()} Naive Bayes classifier: {self.name}")
    
    def _create_model(self) -> Union[GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB]:
        """Create the Naive Bayes classification model based on variant"""
        
        if self.nb_variant == 'gaussian':
            return GaussianNB(
                priors=self.class_prior,
                var_smoothing=self.var_smoothing
            )
        elif self.nb_variant == 'multinomial':
            return MultinomialNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior,
                force_alpha=self.force_alpha
            )
        elif self.nb_variant == 'bernoulli':
            return BernoulliNB(
                alpha=self.alpha,
                binarize=self.binarize,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior,
                force_alpha=self.force_alpha
            )
        elif self.nb_variant == 'complement':
            return ComplementNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior,
                norm=self.norm,
                force_alpha=self.force_alpha
            )
        elif self.nb_variant == 'categorical':
            return CategoricalNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior,
                min_categories=self.min_categories,
                force_alpha=self.force_alpha
            )
        else:
            raise ValueError(f"Unknown Naive Bayes variant: {self.nb_variant}")
    
    def _create_scaler(self) -> Optional[Union[StandardScaler, MinMaxScaler]]:
        """Create appropriate scaler based on scaler_type"""
        
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with optional scaling and selection"""
        
        # Basic preprocessing
        X_processed = super()._preprocess_features(X)
        
        # Apply feature scaling if enabled
        if self.auto_scale and self.scaler_type != 'none':
            if self.scaler_ is None:
                self.scaler_ = self._create_scaler()
                if self.scaler_ is not None:
                    X_scaled = self.scaler_.fit_transform(X_processed)
                    logger.debug(f"Fitted {self.scaler_type} scaler for Naive Bayes")
                else:
                    X_scaled = X_processed
            else:
                if self.scaler_ is not None:
                    X_scaled = self.scaler_.transform(X_processed)
                else:
                    X_scaled = X_processed
            X_processed = X_scaled
        
        # Ensure non-negative values for Multinomial/Complement NB
        if self.nb_variant in ['multinomial', 'complement']:
            if np.any(X_processed < 0):
                logger.warning(f"{self.nb_variant} NB requires non-negative features. Applying shift transformation.")
                X_processed = X_processed - np.min(X_processed, axis=0) + 1e-6
        
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
        
        k = self.selection_k or min(10, X.shape[1])
        
        if self.feature_selection == 'k_best_chi2':
            # Chi-squared test for categorical features
            # Ensure non-negative values
            X_nonneg = X.copy()
            if np.any(X_nonneg < 0):
                X_nonneg = X_nonneg - np.min(X_nonneg, axis=0) + 1e-6
            
            self.feature_selector_ = SelectKBest(score_func=chi2, k=k)
            X_selected = self.feature_selector_.fit_transform(X_nonneg, y)
            
        elif self.feature_selection == 'k_best_f':
            # F-statistic for continuous features
            self.feature_selector_ = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector_.fit_transform(X, y)
            
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection}")
        
        # Get selected feature names
        selected_indices = self.feature_selector_.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        self.selected_features_ = selected_features
        logger.info(f"Selected {len(selected_features)} features using {self.feature_selection}")
        
        return X_selected, selected_features
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Enhanced post-training processing for Naive Bayes classification"""
        
        # Call parent processing
        super()._post_training_processing(X, y)
        
        # Extract Naive Bayes-specific information
        if hasattr(self.model, 'class_log_prior_'):
            self.class_log_prior_ = self.model.class_log_prior_
        
        if hasattr(self.model, 'feature_log_prob_'):
            self.feature_log_prob_ = self.model.feature_log_prob_
        
        if hasattr(self.model, 'class_count_'):
            self.class_count_ = self.model.class_count_
        
        if hasattr(self.model, 'feature_count_'):
            self.feature_count_ = self.model.feature_count_
        
        # Gaussian NB specific attributes
        if hasattr(self.model, 'theta_'):
            self.theta_ = self.model.theta_  # Mean of each feature per class
        
        if hasattr(self.model, 'var_'):
            self.var_ = self.model.var_    # Variance of each feature per class
        
        # Calculate class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.class_weights_ = {cls: count / len(y) for cls, count in zip(unique_classes, class_counts)}
        
        # Perform Bayesian analysis
        self._analyze_bayesian_properties(X, y)
        
        # Analyze feature independence assumption
        self._analyze_feature_independence(X, y)
        
        # Calibrate probabilities if requested
        if self.calibrate_probabilities:
            self._calibrate_probabilities(X, y)
    
    def _analyze_bayesian_properties(self, X: np.ndarray, y: np.ndarray):
        """Analyze Bayesian properties of the model"""
        
        try:
            # Class priors
            if self.class_log_prior_ is not None:
                class_priors = np.exp(self.class_log_prior_)
            else:
                class_priors = np.array([self.class_weights_[i] for i in range(len(self.classes_))])
            
            # Prior entropy
            prior_entropy = -np.sum(class_priors * np.log2(class_priors + 1e-15))
            
            # Posterior analysis
            y_pred_proba = self.model.predict_proba(X)
            
            # Average posterior entropy
            posterior_entropies = -np.sum(y_pred_proba * np.log2(y_pred_proba + 1e-15), axis=1)
            avg_posterior_entropy = np.mean(posterior_entropies)
            
            # Information gain
            information_gain = prior_entropy - avg_posterior_entropy
            
            # Confidence analysis
            max_probabilities = np.max(y_pred_proba, axis=1)
            confidence_stats = {
                'mean_confidence': float(np.mean(max_probabilities)),
                'std_confidence': float(np.std(max_probabilities)),
                'high_confidence_ratio': float(np.mean(max_probabilities > 0.8)),
                'low_confidence_ratio': float(np.mean(max_probabilities < 0.6))
            }
            
            self.bayesian_analysis_ = {
                'class_priors': class_priors.tolist(),
                'prior_entropy': float(prior_entropy),
                'avg_posterior_entropy': float(avg_posterior_entropy),
                'information_gain': float(information_gain),
                'confidence_stats': confidence_stats,
                'nb_variant': self.nb_variant,
                'smoothing_parameter': self.alpha if self.nb_variant != 'gaussian' else self.var_smoothing
            }
            
            # Variant-specific analysis
            if self.nb_variant == 'gaussian':
                self._analyze_gaussian_assumptions(X, y)
            elif self.nb_variant in ['multinomial', 'complement']:
                self._analyze_discrete_assumptions(X, y)
            
        except Exception as e:
            logger.warning(f"Could not perform Bayesian analysis: {e}")
            self.bayesian_analysis_ = None
    
    def _analyze_gaussian_assumptions(self, X: np.ndarray, y: np.ndarray):
        """Analyze Gaussian assumptions for Gaussian Naive Bayes"""
        
        if self.theta_ is None or self.var_ is None:
            return
        
        try:
            from scipy import stats
            
            # Test normality for each feature-class combination
            normality_tests = []
            
            for class_idx, class_label in enumerate(self.classes_):
                class_mask = y == class_idx
                X_class = X[class_mask]
                
                for feature_idx in range(X.shape[1]):
                    feature_data = X_class[:, feature_idx]
                    
                    if len(feature_data) > 3:  # Need at least 3 samples for normality test
                        # Shapiro-Wilk test for normality
                        try:
                            stat, p_value = stats.shapiro(feature_data)
                            is_normal = p_value > 0.05
                        except:
                            stat, p_value, is_normal = np.nan, np.nan, False
                        
                        # Skewness and kurtosis
                        skewness = stats.skew(feature_data)
                        kurt = stats.kurtosis(feature_data)
                        
                        normality_tests.append({
                            'class': class_label,
                            'feature_idx': feature_idx,
                            'feature': self.selected_features_[feature_idx] if self.selected_features_ else f'feature_{feature_idx}',
                            'shapiro_stat': float(stat) if not np.isnan(stat) else None,
                            'shapiro_p_value': float(p_value) if not np.isnan(p_value) else None,
                            'is_normal': is_normal,
                            'skewness': float(skewness),
                            'kurtosis': float(kurt),
                            'mean': float(self.theta_[class_idx, feature_idx]),
                            'variance': float(self.var_[class_idx, feature_idx])
                        })
            
            # Summary statistics
            normal_count = sum(1 for test in normality_tests if test['is_normal'])
            total_tests = len(normality_tests)
            normality_ratio = normal_count / total_tests if total_tests > 0 else 0.0
            
            self.bayesian_analysis_['gaussian_assumptions'] = {
                'normality_tests': normality_tests,
                'normality_ratio': normality_ratio,
                'assumption_violation_severity': 'Low' if normality_ratio > 0.7 else 'Moderate' if normality_ratio > 0.4 else 'High'
            }
            
        except Exception as e:
            logger.debug(f"Could not analyze Gaussian assumptions: {e}")
    
    def _analyze_discrete_assumptions(self, X: np.ndarray, y: np.ndarray):
        """Analyze discrete feature assumptions for Multinomial/Complement NB"""
        
        try:
            # Check if features look discrete vs continuous
            discrete_analysis = []
            
            for feature_idx in range(X.shape[1]):
                feature_data = X[:, feature_idx]
                
                # Check discreteness indicators
                unique_values = len(np.unique(feature_data))
                total_values = len(feature_data)
                uniqueness_ratio = unique_values / total_values
                
                # Check if values are integers
                is_integer = np.allclose(feature_data, np.round(feature_data))
                
                # Check sparsity
                sparsity = np.mean(feature_data == 0)
                
                discrete_analysis.append({
                    'feature_idx': feature_idx,
                    'feature': self.selected_features_[feature_idx] if self.selected_features_ else f'feature_{feature_idx}',
                    'unique_values': unique_values,
                    'uniqueness_ratio': float(uniqueness_ratio),
                    'is_integer': is_integer,
                    'sparsity': float(sparsity),
                    'appears_discrete': is_integer and uniqueness_ratio < 0.1
                })
            
            # Summary
            discrete_count = sum(1 for analysis in discrete_analysis if analysis['appears_discrete'])
            discrete_ratio = discrete_count / len(discrete_analysis) if discrete_analysis else 0.0
            
            self.bayesian_analysis_['discrete_assumptions'] = {
                'feature_analysis': discrete_analysis,
                'discrete_ratio': discrete_ratio,
                'assumption_match': 'Good' if discrete_ratio > 0.7 else 'Moderate' if discrete_ratio > 0.4 else 'Poor'
            }
            
        except Exception as e:
            logger.debug(f"Could not analyze discrete assumptions: {e}")
    
    def _analyze_feature_independence(self, X: np.ndarray, y: np.ndarray):
        """Analyze the feature independence assumption"""
        
        try:
            # Calculate pairwise correlations
            correlations = np.corrcoef(X.T)
            
            # Get upper triangle (excluding diagonal)
            n_features = X.shape[1]
            upper_triangle = np.triu(correlations, k=1)
            correlation_values = upper_triangle[upper_triangle != 0]
            
            # Independence violation analysis
            high_corr_threshold = 0.7
            moderate_corr_threshold = 0.4
            
            high_correlations = np.sum(np.abs(correlation_values) > high_corr_threshold)
            moderate_correlations = np.sum(np.abs(correlation_values) > moderate_corr_threshold)
            total_pairs = len(correlation_values)
            
            # Find most correlated feature pairs
            correlated_pairs = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr_value = correlations[i, j]
                    if abs(corr_value) > moderate_corr_threshold:
                        feature_i = self.selected_features_[i] if self.selected_features_ else f'feature_{i}'
                        feature_j = self.selected_features_[j] if self.selected_features_ else f'feature_{j}'
                        correlated_pairs.append({
                            'feature1': feature_i,
                            'feature2': feature_j,
                            'correlation': float(corr_value),
                            'abs_correlation': float(abs(corr_value))
                        })
            
            # Sort by absolute correlation
            correlated_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
            
            # Independence assumption assessment
            high_corr_ratio = high_correlations / total_pairs if total_pairs > 0 else 0.0
            
            if high_corr_ratio > 0.2:
                independence_assessment = "Severely Violated"
            elif high_corr_ratio > 0.1:
                independence_assessment = "Moderately Violated"
            elif high_corr_ratio > 0.05:
                independence_assessment = "Mildly Violated"
            else:
                independence_assessment = "Approximately Satisfied"
            
            self.independence_analysis_ = {
                'correlation_matrix': correlations.tolist(),
                'correlation_stats': {
                    'mean_abs_correlation': float(np.mean(np.abs(correlation_values))),
                    'max_abs_correlation': float(np.max(np.abs(correlation_values))),
                    'std_correlation': float(np.std(correlation_values))
                },
                'violation_counts': {
                    'high_correlations': high_correlations,
                    'moderate_correlations': moderate_correlations,
                    'total_pairs': total_pairs
                },
                'violation_ratios': {
                    'high_correlation_ratio': float(high_corr_ratio),
                    'moderate_correlation_ratio': float(moderate_correlations / total_pairs) if total_pairs > 0 else 0.0
                },
                'top_correlated_pairs': correlated_pairs[:10],  # Top 10 most correlated pairs
                'independence_assessment': independence_assessment
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze feature independence: {e}")
            self.independence_analysis_ = None
    
    def _calibrate_probabilities(self, X: np.ndarray, y: np.ndarray):
        """Calibrate prediction probabilities using cross-validation"""
        
        try:
            # Use isotonic calibration for Naive Bayes
            self.calibrated_model_ = CalibratedClassifierCV(
                base_estimator=self.model,
                method='isotonic',
                cv=3
            )
            self.calibrated_model_.fit(X, y)
            logger.debug("Calibrated prediction probabilities")
        except Exception as e:
            logger.warning(f"Could not calibrate probabilities: {e}")
            self.calibrated_model_ = None
    
    @time_it("naive_bayes_fit", include_args=True)
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'FinancialNaiveBayesClassifier':
        """
        Fit the Naive Bayes classification model
        
        Args:
            X: Feature matrix
            y: Target labels
            **kwargs: Additional fitting parameters
            
        Returns:
            Fitted model instance
        """
        logger.info(f"Fitting Naive Bayes Classifier ({self.nb_variant}) on {len(X)} samples with {X.shape[1]} features")
        
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
            
            # Preprocess features
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
                'nb_variant': self.nb_variant,
                'smoothing_parameter': self.alpha if self.nb_variant != 'gaussian' else self.var_smoothing
            })
            
            # Calculate training score
            self.training_score = self.model.score(X_processed, y_processed)
            
            # Update status
            self.status = ModelStatus.TRAINED
            self.is_fitted = True
            
            logger.info(f"Naive Bayes Classifier trained successfully in {fit_duration:.2f}s")
            
            return self
            
        except Exception as e:
            from ..base.base_model import ModelStatus
            self.status = ModelStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Naive Bayes Classifier training failed: {e}")
            raise
    
    @time_it("naive_bayes_predict", include_args=True)
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions using the fitted Naive Bayes classifier
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        logger.debug(f"Making Naive Bayes predictions for {len(X)} samples")
        
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
            logger.error(f"Naive Bayes prediction failed: {e}")
            raise
    
    @time_it("naive_bayes_predict_proba", include_args=True)
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
        
        logger.debug(f"Making Naive Bayes probability predictions for {len(X)} samples")
        
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
            logger.error(f"Naive Bayes probability prediction failed: {e}")
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
    
    def get_class_priors(self) -> Dict[str, float]:
        """
        Get class prior probabilities
        
        Returns:
            Dictionary with class names and prior probabilities
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get class priors")
        
        if self.class_log_prior_ is not None:
            priors = np.exp(self.class_log_prior_)
        else:
            # Fallback to empirical class frequencies
            priors = np.array([self.class_weights_[i] for i in range(len(self.classes_))])
        
        return {
            class_name: float(prior) 
            for class_name, prior in zip(self.classes_, priors)
        }
    
    def get_feature_probabilities(self) -> Optional[Dict[str, Any]]:
        """
        Get feature probability parameters
        
        Returns:
            Dictionary with feature probability information
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get feature probabilities")
        
        feature_names = self.selected_features_ or self.feature_names
        
        if self.nb_variant == 'gaussian':
            if self.theta_ is not None and self.var_ is not None:
                return {
                    'type': 'gaussian',
                    'means': {
                        class_name: {
                            feature_names[i]: float(self.theta_[class_idx, i])
                            for i in range(len(feature_names))
                        }
                        for class_idx, class_name in enumerate(self.classes_)
                    },
                    'variances': {
                        class_name: {
                            feature_names[i]: float(self.var_[class_idx, i])
                            for i in range(len(feature_names))
                        }
                        for class_idx, class_name in enumerate(self.classes_)
                    }
                }
        
        elif self.feature_log_prob_ is not None:
            return {
                'type': self.nb_variant,
                'log_probabilities': {
                    class_name: self.feature_log_prob_[class_idx].tolist()
                    for class_idx, class_name in enumerate(self.classes_)
                },
                'feature_names': feature_names
            }
        
        return None
    
    def get_bayesian_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive Bayesian analysis
        
        Returns:
            Dictionary with Bayesian analysis
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get Bayesian analysis")
        
        return self.bayesian_analysis_.copy() if self.bayesian_analysis_ else {}
    
    def get_independence_analysis(self) -> Dict[str, Any]:
        """
        Get feature independence analysis
        
        Returns:
            Dictionary with independence analysis
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get independence analysis")
        
        return self.independence_analysis_.copy() if self.independence_analysis_ else {}
    
    def plot_class_priors(self) -> Any:
        """
        Plot class prior probabilities
        
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            priors = self.get_class_priors()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot
            classes = list(priors.keys())
            prior_values = list(priors.values())
            
            bars = ax1.bar(range(len(classes)), prior_values, alpha=0.7, color='steelblue')
            ax1.set_xticks(range(len(classes)))
            ax1.set_xticklabels(classes)
            ax1.set_ylabel('Prior Probability')
            ax1.set_title('Class Prior Probabilities')
            ax1.grid(True, alpha=0.3)
            
            # Add values on bars
            for bar, value in zip(bars, prior_values):
                ax1.text(bar.get_x() + bar.get_width()/2, value + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
            
            # Pie chart
            ax2.pie(prior_values, labels=classes, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Class Prior Distribution')
            
            plt.suptitle(f'Naive Bayes Class Priors - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_feature_distributions(self, max_features: int = 12) -> Any:
        """
        Plot feature distributions by class (for Gaussian NB)
        
        Args:
            max_features: Maximum number of features to plot
            
        Returns:
            Matplotlib figure
        """
        if self.nb_variant != 'gaussian' or self.theta_ is None:
            logger.warning("Feature distribution plots only available for Gaussian Naive Bayes")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            feature_names = self.selected_features_ or self.feature_names
            n_features = min(len(feature_names), max_features)
            
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            axes = axes.flatten()
            
            for i in range(n_features):
                ax = axes[i]
                
                # Plot Gaussian distributions for each class
                x_min = np.min(self.theta_[:, i] - 3 * np.sqrt(self.var_[:, i]))
                x_max = np.max(self.theta_[:, i] + 3 * np.sqrt(self.var_[:, i]))
                x = np.linspace(x_min, x_max, 100)
                
                colors = ['red', 'blue', 'green', 'orange', 'purple'][:len(self.classes_)]
                
                for class_idx, (class_name, color) in enumerate(zip(self.classes_, colors)):
                    mean = self.theta_[class_idx, i]
                    var = self.var_[class_idx, i]
                    std = np.sqrt(var)
                    
                    # Gaussian PDF
                    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
                    
                    ax.plot(x, y, color=color, label=f'Class {class_name}', linewidth=2)
                    ax.axvline(mean, color=color, linestyle='--', alpha=0.7)
                
                ax.set_title(f'{feature_names[i][:20]}...', fontsize=10)
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'Feature Distributions by Class - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_independence_analysis(self) -> Any:
        """
        Plot feature independence analysis
        
        Returns:
            Matplotlib figure
        """
        if self.independence_analysis_ is None:
            logger.warning("Independence analysis not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Correlation matrix heatmap
            correlation_matrix = np.array(self.independence_analysis_['correlation_matrix'])
            feature_names = self.selected_features_ or self.feature_names
            
            # Limit size for readability
            if len(feature_names) > 20:
                correlation_matrix = correlation_matrix[:20, :20]
                feature_names = feature_names[:20]
            
            sns.heatmap(correlation_matrix, 
                       xticklabels=feature_names, 
                       yticklabels=feature_names,
                       annot=False, cmap='RdBu_r', center=0,
                       ax=axes[0, 0])
            axes[0, 0].set_title('Feature Correlation Matrix')
            
            # Correlation distribution
            corr_values = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            axes[0, 1].hist(corr_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].set_xlabel('Correlation Coefficient')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Pairwise Correlations')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Top correlated pairs
            top_pairs = self.independence_analysis_['top_correlated_pairs'][:10]
            if top_pairs:
                pair_labels = [f"{pair['feature1'][:10]}...\n{pair['feature2'][:10]}..." for pair in top_pairs]
                correlations = [abs(pair['correlation']) for pair in top_pairs]
                
                bars = axes[1, 0].barh(range(len(pair_labels)), correlations, 
                                      color=['red' if c > 0.7 else 'orange' if c > 0.4 else 'yellow' for c in correlations])
                axes[1, 0].set_yticks(range(len(pair_labels)))
                axes[1, 0].set_yticklabels(pair_labels, fontsize=8)
                axes[1, 0].set_xlabel('Absolute Correlation')
                axes[1, 0].set_title('Top Correlated Feature Pairs')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add correlation values
                for i, (bar, corr) in enumerate(zip(bars, correlations)):
                    axes[1, 0].text(corr + 0.01, i, f'{corr:.3f}', va='center', fontsize=8)
            else:
                axes[1, 0].text(0.5, 0.5, 'No High Correlations Found', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Top Correlated Feature Pairs')
            
            # Independence assessment summary
            assessment = self.independence_analysis_['independence_assessment']
            violation_stats = self.independence_analysis_['violation_ratios']
            
            summary_text = f"Independence Assessment: {assessment}\n\n"
            summary_text += f"High Correlation Ratio: {violation_stats['high_correlation_ratio']:.1%}\n"
            summary_text += f"Moderate Correlation Ratio: {violation_stats['moderate_correlation_ratio']:.1%}\n\n"
            summary_text += f"Mean Abs Correlation: {self.independence_analysis_['correlation_stats']['mean_abs_correlation']:.3f}\n"
            summary_text += f"Max Abs Correlation: {self.independence_analysis_['correlation_stats']['max_abs_correlation']:.3f}"
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1, 1].set_title('Independence Assessment Summary')
            axes[1, 1].axis('off')
            
            plt.suptitle(f'Feature Independence Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return None
    
    def plot_bayesian_analysis(self) -> Any:
        """
        Plot Bayesian analysis results
        
        Returns:
            Matplotlib figure
        """
        if self.bayesian_analysis_ is None:
            logger.warning("Bayesian analysis not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Class priors
            priors = self.bayesian_analysis_['class_priors']
            class_names = self.classes_
            
            bars = axes[0, 0].bar(range(len(class_names)), priors, alpha=0.7, color='steelblue')
            axes[0, 0].set_xticks(range(len(class_names)))
            axes[0, 0].set_xticklabels(class_names)
            axes[0, 0].set_ylabel('Prior Probability')
            axes[0, 0].set_title('Class Prior Probabilities')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add values on bars
            for bar, prior in zip(bars, priors):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, prior + 0.01, 
                               f'{prior:.3f}', ha='center', va='bottom')
            
            # Information gain visualization
            prior_entropy = self.bayesian_analysis_['prior_entropy']
            posterior_entropy = self.bayesian_analysis_['avg_posterior_entropy']
            info_gain = self.bayesian_analysis_['information_gain']
            
            entropies = [prior_entropy, posterior_entropy]
            labels = ['Prior Entropy', 'Avg Posterior Entropy']
            colors = ['orange', 'green']
            
            bars2 = axes[0, 1].bar(labels, entropies, color=colors, alpha=0.7)
            axes[0, 1].set_ylabel('Entropy (bits)')
            axes[0, 1].set_title(f'Information Gain: {info_gain:.3f} bits')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add values on bars
            for bar, entropy in zip(bars2, entropies):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, entropy + 0.01, 
                               f'{entropy:.3f}', ha='center', va='bottom')
            
            # Confidence statistics
            conf_stats = self.bayesian_analysis_['confidence_stats']
            
            conf_metrics = ['Mean', 'Std', 'High Conf %', 'Low Conf %']
            conf_values = [
                conf_stats['mean_confidence'],
                conf_stats['std_confidence'],
                conf_stats['high_confidence_ratio'] * 100,
                conf_stats['low_confidence_ratio'] * 100
            ]
            
            bars3 = axes[1, 0].bar(conf_metrics, conf_values, alpha=0.7, color='purple')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('Prediction Confidence Statistics')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add values on bars
            for bar, value in zip(bars3, conf_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, value + max(conf_values) * 0.01, 
                               f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Model summary
            summary_text = f"Naive Bayes Variant: {self.nb_variant.title()}\n\n"
            
            if self.nb_variant == 'gaussian':
                summary_text += f"Variance Smoothing: {self.var_smoothing}\n"
            else:
                summary_text += f"Smoothing Parameter (α): {self.alpha}\n"
            
            summary_text += f"Classes: {len(self.classes_)}\n"
            summary_text += f"Selected Features: {len(self.selected_features_) if self.selected_features_ else len(self.feature_names)}\n\n"
            
            # Add assumption analysis if available
            if 'gaussian_assumptions' in self.bayesian_analysis_:
                gauss_analysis = self.bayesian_analysis_['gaussian_assumptions']
                summary_text += f"Normality Ratio: {gauss_analysis['normality_ratio']:.1%}\n"
                summary_text += f"Assumption Violation: {gauss_analysis['assumption_violation_severity']}\n"
            
            if 'discrete_assumptions' in self.bayesian_analysis_:
                discrete_analysis = self.bayesian_analysis_['discrete_assumptions']
                summary_text += f"Discrete Ratio: {discrete_analysis['discrete_ratio']:.1%}\n"
                summary_text += f"Assumption Match: {discrete_analysis['assumption_match']}\n"
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 1].set_title('Model Summary')
            axes[1, 1].axis('off')
            
            plt.suptitle(f'Bayesian Analysis - {self.name}', fontsize=16)
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
    
    def get_naive_bayes_summary(self) -> Dict[str, Any]:
        """Get comprehensive Naive Bayes summary"""
        
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get Naive Bayes summary")
        
        summary = {
            'model_info': {
                'nb_variant': self.nb_variant,
                'smoothing_parameter': self.alpha if self.nb_variant != 'gaussian' else self.var_smoothing,
                'fit_prior': self.fit_prior,
                'n_classes': len(self.classes_),
                'class_names': self.classes_.tolist(),
                'n_features_original': len(self.feature_names),
                'n_features_selected': len(self.selected_features_) if self.selected_features_ else len(self.feature_names),
                'feature_selection_method': self.feature_selection
            },
            'class_priors': self.get_class_priors(),
            'bayesian_analysis': self.bayesian_analysis_,
            'independence_analysis': self.independence_analysis_,
            'calibration_info': {
                'probabilities_calibrated': self.calibrated_model_ is not None,
                'calibration_method': 'isotonic' if self.calibrated_model_ else None
            }
        }
        
        # Add variant-specific information
        feature_prob_info = self.get_feature_probabilities()
        if feature_prob_info:
            summary['feature_probabilities'] = feature_prob_info
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add Naive Bayes-specific information
        summary.update({
            'model_family': 'Naive Bayes',
            'nb_variant': self.nb_variant.replace('_', ' ').title(),
            'smoothing_parameter': self.alpha if self.nb_variant != 'gaussian' else self.var_smoothing,
            'fit_prior': self.fit_prior,
            'feature_selection': self.feature_selection,
            'probability_calibration': self.calibrate_probabilities,
            'scaler_type': self.scaler_type,
            'n_classes': len(self.classes_) if hasattr(self, 'classes_') else None,
            'n_selected_features': len(self.selected_features_) if self.selected_features_ else len(self.feature_names),
            'auto_scaling': self.auto_scale
        })
        
        # Add Bayesian information
        if self.bayesian_analysis_:
            summary.update({
                'prior_entropy': self.bayesian_analysis_['prior_entropy'],
                'information_gain': self.bayesian_analysis_['information_gain'],
                'mean_confidence': self.bayesian_analysis_['confidence_stats']['mean_confidence']
            })
        
        # Add independence assessment
        if self.independence_analysis_:
            summary.update({
                'independence_assessment': self.independence_analysis_['independence_assessment'],
                'mean_abs_correlation': self.independence_analysis_['correlation_stats']['mean_abs_correlation']
            })
        
        # Add Naive Bayes summary
        if self.is_fitted:
            try:
                summary['naive_bayes_summary'] = self.get_naive_bayes_summary()
            except Exception as e:
                logger.debug(f"Could not generate Naive Bayes summary: {e}")
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_naive_bayes_classifier(nb_variant: str = 'gaussian',
                                 performance_preset: str = 'balanced',
                                 **kwargs) -> FinancialNaiveBayesClassifier:
    """
    Create a Naive Bayes classification model
    
    Args:
        nb_variant: Naive Bayes variant ('gaussian', 'multinomial', 'bernoulli', 'complement', 'categorical')
        performance_preset: Performance preset ('fast', 'balanced', 'accurate')
        **kwargs: Additional model parameters
        
    Returns:
        Configured Naive Bayes classification model
    """
    
    # Base configuration
    base_config = {
        'name': f'{nb_variant}_naive_bayes_classifier',
        'nb_variant': nb_variant,
        'calibrate_probabilities': True,
        'fit_prior': True
    }
    
    # Performance presets
    if performance_preset == 'fast':
        preset_config = {
            'alpha': 1.0,
            'var_smoothing': 1e-7,
            'auto_scale': False,
            'feature_selection': None
        }
    elif performance_preset == 'balanced':
        preset_config = {
            'alpha': 1.0,
            'var_smoothing': 1e-9,
            'auto_scale': nb_variant == 'gaussian',
            'scaler_type': 'standard' if nb_variant == 'gaussian' else 'none',
            'feature_selection': None
        }
    elif performance_preset == 'accurate':
        preset_config = {
            'alpha': 0.1,  # Less smoothing for more accuracy
            'var_smoothing': 1e-11,
            'auto_scale': True,
            'scaler_type': 'standard' if nb_variant == 'gaussian' else 'minmax',
            'feature_selection': 'k_best_f' if nb_variant == 'gaussian' else 'k_best_chi2',
            'selection_k': 15
        }
    else:
        raise ValueError(f"Unknown performance preset: {performance_preset}")
    
    # Variant-specific adjustments
    if nb_variant == 'gaussian':
        preset_config.update({
            'scaler_type': 'standard' if preset_config.get('auto_scale', False) else 'none'
        })
    elif nb_variant in ['multinomial', 'complement']:
        preset_config.update({
            'scaler_type': 'none',  # These variants work with count data
            'auto_scale': False
        })
    elif nb_variant == 'bernoulli':
        preset_config.update({
            'binarize': 0.0,
            'scaler_type': 'none',
            'auto_scale': False
        })
    
    # Combine configurations
    config = {**base_config, **preset_config}
    config.update(kwargs)  # Override with user parameters
    
    return FinancialNaiveBayesClassifier(**config)

def create_gaussian_nb(**kwargs) -> FinancialNaiveBayesClassifier:
    """Create Gaussian Naive Bayes for continuous features"""
    
    return create_naive_bayes_classifier(
        nb_variant='gaussian',
        performance_preset='balanced',
        name='gaussian_naive_bayes',
        **kwargs
    )

def create_multinomial_nb(**kwargs) -> FinancialNaiveBayesClassifier:
    """Create Multinomial Naive Bayes for count/frequency features"""
    
    return create_naive_bayes_classifier(
        nb_variant='multinomial',
        performance_preset='balanced',
        name='multinomial_naive_bayes',
        **kwargs
    )

def create_bernoulli_nb(**kwargs) -> FinancialNaiveBayesClassifier:
    """Create Bernoulli Naive Bayes for binary features"""
    
    return create_naive_bayes_classifier(
        nb_variant='bernoulli',
        performance_preset='balanced',
        name='bernoulli_naive_bayes',
        **kwargs
    )

def create_complement_nb(**kwargs) -> FinancialNaiveBayesClassifier:
    """Create Complement Naive Bayes (better for imbalanced data)"""
    
    return create_naive_bayes_classifier(
        nb_variant='complement',
        performance_preset='balanced',
        name='complement_naive_bayes',
        **kwargs
    )

def create_binary_nb(nb_variant: str = 'gaussian', **kwargs) -> FinancialNaiveBayesClassifier:
    """Create Naive Bayes optimized for binary classification"""
    
    return create_naive_bayes_classifier(
        nb_variant=nb_variant,
        performance_preset='balanced',
        name=f'binary_{nb_variant}_nb',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def tune_naive_bayes_hyperparameters(X: pd.DataFrame, y: pd.Series,
                                    nb_variant: str = 'gaussian',
                                    param_grid: Optional[Dict[str, List[Any]]] = None,
                                    cv: int = 5,
                                    scoring: str = 'accuracy',
                                    n_jobs: int = -1) -> Dict[str, Any]:
    """
    Tune Naive Bayes hyperparameters using grid search
    
    Args:
        X: Feature matrix
        y: Target values
        nb_variant: Naive Bayes variant
        param_grid: Parameter grid for search
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with best parameters and scores
    """
    
    from sklearn.model_selection import GridSearchCV
    
    logger.info(f"Starting Naive Bayes hyperparameter tuning for {nb_variant}")
    
    # Default parameter grids for different variants
    if param_grid is None:
        if nb_variant == 'gaussian':
            param_grid = {
                'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
                'priors': [None]  # Let it learn from data
            }
        elif nb_variant in ['multinomial', 'bernoulli', 'complement']:
            param_grid = {
                'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                'fit_prior': [True, False]
            }
        else:
            param_grid = {'alpha': [0.1, 1.0, 10.0]}
    
    # Create base model
    model = create_naive_bayes_classifier(nb_variant=nb_variant)
    base_model = model._create_model()
    
    # Preprocess data
    X_processed = model._preprocess_features(X)
    y_processed = model._preprocess_targets(y)
    
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
    
    grid_search.fit(X_processed, y_processed)
    
    # Extract results
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_,
        'nb_variant': nb_variant
    }
    
    logger.info(f"Hyperparameter tuning complete. Best score: {results['best_score']:.4f}")
    logger.info(f"Best parameters: {results['best_params']}")
    
    return results

def compare_naive_bayes_variants(X: pd.DataFrame, y: pd.Series,
                               variants: List[str] = ['gaussian', 'multinomial', 'bernoulli', 'complement'],
                               cv: int = 5) -> Dict[str, Any]:
    """
    Compare different Naive Bayes variants
    
    Args:
        X: Feature matrix
        y: Target values
        variants: List of Naive Bayes variants to compare
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with comparison results
    """
    
    from sklearn.model_selection import cross_val_score
    
    logger.info(f"Comparing Naive Bayes variants: {variants}")
    
    results = {}
    
    for variant in variants:
        logger.info(f"Evaluating {variant} Naive Bayes")
        
        try:
            # Create model
            model = create_naive_bayes_classifier(
                nb_variant=variant,
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
            
            results[variant] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'fit_time': fit_time,
                'model': sklearn_model
            }
            
        except Exception as e:
            logger.warning(f"Error with {variant} Naive Bayes: {e}")
            results[variant] = {'error': str(e)}
    
    # Add comparison summary
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_accuracy = max(valid_results.keys(), key=lambda k: valid_results[k]['cv_mean'])
        fastest = min(valid_results.keys(), key=lambda k: valid_results[k]['fit_time'])
        most_stable = min(valid_results.keys(), key=lambda k: valid_results[k]['cv_std'])
        
        results['comparison'] = {
            'best_accuracy': best_accuracy,
            'fastest': fastest,
            'most_stable': most_stable,
            'valid_variants': list(valid_results.keys())
        }
    
    logger.info(f"Variant comparison complete. Best accuracy: {results['comparison']['best_accuracy']}")
    
    return results

def analyze_naive_bayes_assumptions(model: FinancialNaiveBayesClassifier,
                                   X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Analyze Naive Bayes assumptions in detail
    
    Args:
        model: Fitted Naive Bayes model
        X: Feature matrix
        y: Target values
        
    Returns:
        Dictionary with assumption analysis
    """
    
    if not model.is_fitted:
        raise BusinessLogicError("Model must be fitted for assumption analysis")
    
    logger.info(f"Analyzing Naive Bayes assumptions for {model.nb_variant} variant")
    
    # Get Bayesian and independence analysis
    bayesian_analysis = model.get_bayesian_analysis()
    independence_analysis = model.get_independence_analysis()
    
    analysis = {
        'nb_variant': model.nb_variant,
        'bayesian_properties': bayesian_analysis,
        'independence_properties': independence_analysis
    }
    
    # Overall assessment
    assessment_scores = []
    
    # Assess independence assumption
    if independence_analysis:
        independence_score = independence_analysis['independence_assessment']
        if independence_score == "Approximately Satisfied":
            assessment_scores.append(4)
        elif independence_score == "Mildly Violated":
            assessment_scores.append(3)
        elif independence_score == "Moderately Violated":
            assessment_scores.append(2)
        else:  # Severely Violated
            assessment_scores.append(1)
    
    # Assess distributional assumptions
    if model.nb_variant == 'gaussian' and 'gaussian_assumptions' in bayesian_analysis:
        gauss_analysis = bayesian_analysis['gaussian_assumptions']
        normality_ratio = gauss_analysis['normality_ratio']
        
        if normality_ratio > 0.8:
            assessment_scores.append(4)
        elif normality_ratio > 0.6:
            assessment_scores.append(3)
        elif normality_ratio > 0.4:
            assessment_scores.append(2)
        else:
            assessment_scores.append(1)
    
    elif model.nb_variant in ['multinomial', 'complement'] and 'discrete_assumptions' in bayesian_analysis:
        discrete_analysis = bayesian_analysis['discrete_assumptions']
        discrete_match = discrete_analysis['assumption_match']
        
        if discrete_match == "Good":
            assessment_scores.append(4)
        elif discrete_match == "Moderate":
            assessment_scores.append(3)
        else:  # Poor
            assessment_scores.append(2)
    
    # Overall assumption satisfaction
    if assessment_scores:
        avg_score = np.mean(assessment_scores)
        if avg_score >= 3.5:
            overall_assessment = "Assumptions Well Satisfied"
        elif avg_score >= 2.5:
            overall_assessment = "Assumptions Moderately Satisfied"
        elif avg_score >= 1.5:
            overall_assessment = "Assumptions Poorly Satisfied"
        else:
            overall_assessment = "Assumptions Severely Violated"
    else:
        overall_assessment = "Cannot Assess"
    
    analysis['overall_assessment'] = {
        'assessment': overall_assessment,
        'recommendation': _get_assumption_recommendation(overall_assessment, model.nb_variant),
        'scores': assessment_scores,
        'average_score': np.mean(assessment_scores) if assessment_scores else None
    }
    
    logger.info(f"Assumption analysis complete. Overall: {overall_assessment}")
    
    return analysis

def _get_assumption_recommendation(assessment: str, nb_variant: str) -> str:
    """Get recommendation based on assumption analysis"""
    
    if assessment == "Assumptions Well Satisfied":
        return f"{nb_variant.title()} Naive Bayes is well-suited for this data."
    
    elif assessment == "Assumptions Moderately Satisfied":
        return f"{nb_variant.title()} Naive Bayes should work reasonably well. Consider feature engineering or alternative variants."
    
    elif assessment == "Assumptions Poorly Satisfied":
        if nb_variant == 'gaussian':
            return "Consider feature transformation (log, Box-Cox) or alternative Naive Bayes variants (Multinomial, Complement)."
        else:
            return "Consider Gaussian Naive Bayes or more flexible models (Random Forest, SVM)."
    
    else:  # Severely Violated
        return "Naive Bayes assumptions are severely violated. Consider alternative algorithms (Random Forest, SVM, Neural Networks)."

def perform_naive_bayes_feature_analysis(model: FinancialNaiveBayesClassifier,
                                       X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Perform comprehensive feature analysis for Naive Bayes
    
    Args:
        model: Fitted Naive Bayes model
        X: Feature matrix
        y: Target values
        
    Returns:
        Dictionary with feature analysis
    """
    
    if not model.is_fitted:
        raise BusinessLogicError("Model must be fitted for feature analysis")
    
    logger.info("Performing comprehensive Naive Bayes feature analysis")
    
    analysis = {
        'nb_variant': model.nb_variant,
        'n_features': len(model.selected_features_) if model.selected_features_ else len(model.feature_names),
        'selected_features': model.selected_features_
    }
    
    # Feature probability analysis
    feature_probs = model.get_feature_probabilities()
    if feature_probs:
        analysis['feature_probabilities'] = feature_probs
        
        # Analyze feature discriminative power
        if model.nb_variant == 'gaussian' and 'means' in feature_probs:
            discriminative_analysis = []
            
            for feature_idx, feature_name in enumerate(model.selected_features_ or model.feature_names):
                if feature_name in feature_probs['means'][model.classes_[0]]:
                    # Calculate class separation for this feature
                    class_means = [feature_probs['means'][class_name][feature_name] for class_name in model.classes_]
                    class_vars = [feature_probs['variances'][class_name][feature_name] for class_name in model.classes_]
                    
                    # Simple discriminative power metric: mean separation / average std
                    mean_separation = np.max(class_means) - np.min(class_means)
                    avg_std = np.sqrt(np.mean(class_vars))
                    discriminative_power = mean_separation / (avg_std + 1e-8)
                    
                    discriminative_analysis.append({
                        'feature': feature_name,
                        'discriminative_power': discriminative_power,
                        'mean_separation': mean_separation,
                        'avg_std': avg_std,
                        'class_means': class_means,
                        'class_vars': class_vars
                    })
            
            # Sort by discriminative power
            discriminative_analysis.sort(key=lambda x: x['discriminative_power'], reverse=True)
            analysis['discriminative_analysis'] = discriminative_analysis[:20]  # Top 20
    
    # Independence violations by feature
    independence_analysis = model.get_independence_analysis()
    if independence_analysis and 'top_correlated_pairs' in independence_analysis:
        feature_violations = {}
        
        for pair in independence_analysis['top_correlated_pairs']:
            feature1 = pair['feature1']
            feature2 = pair['feature2']
            correlation = abs(pair['correlation'])
            
            for feature in [feature1, feature2]:
                if feature not in feature_violations:
                    feature_violations[feature] = []
                feature_violations[feature].append({
                    'correlated_with': feature2 if feature == feature1 else feature1,
                    'correlation': correlation
                })
        
        # Sort features by number of high correlations
        feature_violation_summary = [
            {
                'feature': feature,
                'n_violations': len(violations),
                'max_correlation': max(v['correlation'] for v in violations),
                'violations': violations
            }
            for feature, violations in feature_violations.items()
        ]
        
        feature_violation_summary.sort(key=lambda x: (x['n_violations'], x['max_correlation']), reverse=True)
        analysis['independence_violations'] = feature_violation_summary[:15]  # Top 15
    
    logger.info(f"Feature analysis complete for {analysis['n_features']} features")
    
    return analysis

def optimize_naive_bayes_for_data(X: pd.DataFrame, y: pd.Series,
                                 test_variants: List[str] = ['gaussian', 'multinomial', 'complement'],
                                 analyze_assumptions: bool = True) -> Dict[str, Any]:
    """
    Optimize Naive Bayes variant and parameters for given data
    
    Args:
        X: Feature matrix
        y: Target values
        test_variants: Naive Bayes variants to test
        analyze_assumptions: Whether to perform assumption analysis
        
    Returns:
        Dictionary with optimization results and recommendations
    """
    
    logger.info("Optimizing Naive Bayes for given data")
    
    # Compare variants
    variant_comparison = compare_naive_bayes_variants(X, y, test_variants)
    
    # Get best variant
    if 'comparison' in variant_comparison:
        best_variant = variant_comparison['comparison']['best_accuracy']
        logger.info(f"Best performing variant: {best_variant}")
        
        # Tune hyperparameters for best variant
        tuning_results = tune_naive_bayes_hyperparameters(X, y, best_variant)
        
        # Create optimized model
        optimized_model = create_naive_bayes_classifier(
            nb_variant=best_variant,
            **tuning_results['best_params']
        )
        optimized_model.fit(X, y)
        
        results = {
            'variant_comparison': variant_comparison,
            'best_variant': best_variant,
            'hyperparameter_tuning': tuning_results,
            'optimized_model': optimized_model,
            'optimization_summary': {
                'recommended_variant': best_variant,
                'best_cv_score': tuning_results['best_score'],
                'best_params': tuning_results['best_params']
            }
        }
        
        # Assumption analysis
        if analyze_assumptions:
            assumption_analysis = analyze_naive_bayes_assumptions(optimized_model, X, y)
            results['assumption_analysis'] = assumption_analysis
            results['optimization_summary']['assumption_assessment'] = assumption_analysis['overall_assessment']['assessment']
            results['optimization_summary']['recommendation'] = assumption_analysis['overall_assessment']['recommendation']
        
        # Feature analysis
        feature_analysis = perform_naive_bayes_feature_analysis(optimized_model, X, y)
        results['feature_analysis'] = feature_analysis
        
        logger.info(f"Optimization complete. Recommended: {best_variant} with score {tuning_results['best_score']:.4f}")
        
        return results
    
    else:
        raise ValueError("No valid Naive Bayes variants found for comparison")
