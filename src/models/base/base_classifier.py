# ============================================
# StockPredictionPro - src/models/base/base_classifier.py
# Base classifier interface for financial prediction models
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
from enum import Enum
from dataclasses import dataclass, field


from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ...utils.config_loader import get
from ...utils.validators import ValidationResult
from .base_model import BaseModel, ModelMetadata, ModelStatus

logger = get_logger('models.base.classifier')

# ============================================
# Classification-Specific Types and Enums
# ============================================

class ClassificationStrategy(Enum):
    """Classification prediction strategies"""
    DIRECTIONAL = "directional"        # Up/Down prediction
    MULTICLASS = "multiclass"         # Multiple price movement classes
    PROBABILITY = "probability"       # Probability-based predictions

@dataclass
class ClassificationMetrics:
    """Classification model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    precision_recall_auc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'precision_recall_auc': self.precision_recall_auc,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'classification_report': self.classification_report
        }

@dataclass
class PredictionResult:
    """Classification prediction result"""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    prediction_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_confident_predictions(self, confidence_threshold: float = 0.6) -> np.ndarray:
        """Get predictions with confidence above threshold"""
        if self.confidence_scores is None:
            return self.predictions
        
        confident_mask = self.confidence_scores >= confidence_threshold
        return self.predictions[confident_mask]

# ============================================
# Base Financial Classifier
# ============================================

class BaseFinancialClassifier(BaseModel, ClassifierMixin, ABC):
    """
    Abstract base class for financial classification models
    
    Features:
    - Financial domain-specific interface
    - Comprehensive performance tracking
    - Probability and confidence estimation
    - Cross-validation support
    - Feature importance analysis
    - Model interpretability
    """
    
    def __init__(self,
                 name: str,
                 model_type: str = "classifier",
                 classification_strategy: ClassificationStrategy = ClassificationStrategy.DIRECTIONAL,
                 n_classes: int = 2,
                 class_names: Optional[List[str]] = None,
                 prediction_horizon: int = 1,
                 min_confidence: float = 0.5,
                 **kwargs):
        """
        Initialize base financial classifier
        
        Args:
            name: Model name
            model_type: Specific model type
            classification_strategy: Classification approach
            n_classes: Number of classes
            class_names: Names for classes
            prediction_horizon: Days ahead to predict
            min_confidence: Minimum confidence threshold
            **kwargs: Additional model parameters
        """
        super().__init__(
            name=name,
            model_type=model_type,
            prediction_type='classification',
            **kwargs
        )
        
        self.classification_strategy = classification_strategy
        self.n_classes = n_classes
        self.class_names = class_names or [f"class_{i}" for i in range(n_classes)]
        self.prediction_horizon = prediction_horizon
        self.min_confidence = min_confidence
        
        # Classification-specific attributes
        self.label_encoder: Optional[LabelEncoder] = None
        self.class_weights_: Optional[Dict[str, float]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        
        # Performance tracking
        self.classification_metrics_: Optional[ClassificationMetrics] = None
        self.cross_validation_scores_: Optional[Dict[str, np.ndarray]] = None
        
        # Probability calibration
        self.is_probability_calibrated: bool = False
        self.calibration_method: Optional[str] = None
        
        logger.debug(f"Initialized {self.name} classifier with {self.n_classes} classes")
    
    def _validate_input_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> ValidationResult:
        """Validate input data for classification"""
        result = super()._validate_input_data(X, y)
        
        if y is not None:
            # Check target variable
            unique_classes = len(y.unique())
            
            if unique_classes < 2:
                result.add_error("Target variable must have at least 2 classes for classification")
            
            if unique_classes != self.n_classes:
                result.add_warning(f"Expected {self.n_classes} classes but found {unique_classes}")
            
            # Check class balance
            class_counts = y.value_counts()
            min_class_size = class_counts.min()
            max_class_size = class_counts.max()
            
            if min_class_size / max_class_size < 0.1:  # Severe imbalance
                result.add_warning(f"Severe class imbalance detected: {class_counts.to_dict()}")
        
        return result
    
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create the underlying model instance"""
        pass
    
    def _preprocess_targets(self, y: pd.Series) -> np.ndarray:
        """Preprocess target labels for classification"""
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            encoded_targets = self.label_encoder.fit_transform(y)
        else:
            encoded_targets = self.label_encoder.transform(y)
        
        # Update class names if not provided
        if len(self.class_names) != len(self.label_encoder.classes_):
            self.class_names = self.label_encoder.classes_.tolist()
        
        return encoded_targets
    
    @time_it("classifier_fit", include_args=True)
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[np.ndarray] = None,
            **fit_params) -> 'BaseFinancialClassifier':
        """
        Fit the classification model
        
        Args:
            X: Feature matrix
            y: Target labels
            sample_weight: Sample weights (optional)
            **fit_params: Additional fitting parameters
            
        Returns:
            Fitted model instance
        """
        logger.info(f"Fitting {self.name} classifier on {len(X)} samples with {X.shape[1]} features")
        
        # Validate input data
        validation_result = self._validate_input_data(X, y)
        if not validation_result.is_valid:
            raise ModelValidationError(f"Input validation failed: {validation_result.errors}")
        
        try:
            # Update status
            self.status = ModelStatus.TRAINING
            self.last_training_time = datetime.now()
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Preprocess features and targets
            X_processed = self._preprocess_features(X)
            y_processed = self._preprocess_targets(y)
            
            # Create model if not exists
            if self.model is None:
                self.model = self._create_model()
            
            # Fit the model
            fit_start = datetime.now()
            
            if sample_weight is not None:
                self.model.fit(X_processed, y_processed, sample_weight=sample_weight, **fit_params)
            else:
                self.model.fit(X_processed, y_processed, **fit_params)
            
            fit_duration = (datetime.now() - fit_start).total_seconds()
            
            # Post-training processing
            self._post_training_processing(X_processed, y_processed)
            
            # Update model metadata
            self.update_metadata({
                'training_samples': len(X),
                'training_features': X.shape[1],
                'training_duration_seconds': fit_duration,
                'n_classes_actual': len(np.unique(y_processed)),
                'class_distribution': pd.Series(y_processed).value_counts().to_dict()
            })
            
            # Evaluate on training data
            self._evaluate_training_performance(X_processed, y_processed)
            
            # Update status
            self.status = ModelStatus.TRAINED
            self.is_fitted = True
            
            logger.info(f"Model {self.name} trained successfully in {fit_duration:.2f}s")
            
            return self
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Training failed for {self.name}: {e}")
            raise
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Post-training processing and analysis"""
        
        # Extract feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients as importance
            if self.model.coef_.ndim > 1:
                self.feature_importances_ = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                self.feature_importances_ = np.abs(self.model.coef_)
        
        # Calculate class weights if needed
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        self.class_weights_ = {}
        for cls, count in zip(unique_classes, class_counts):
            self.class_weights_[str(cls)] = total_samples / (len(unique_classes) * count)
    
    def _evaluate_training_performance(self, X: np.ndarray, y: np.ndarray):
        """Evaluate model performance on training data"""
        
        try:
            # Make predictions
            y_pred = self.model.predict(X)
            
            # Get probabilities if available
            y_proba = None
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X)
            
            # Calculate metrics
            self.classification_metrics_ = self._calculate_classification_metrics(
                y_true=y, 
                y_pred=y_pred, 
                y_proba=y_proba
            )
            
            # Update training score
            self.training_score = self.classification_metrics_.accuracy
            
        except Exception as e:
            logger.warning(f"Failed to evaluate training performance: {e}")
    
    def _calculate_classification_metrics(self, 
                                        y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        y_proba: Optional[np.ndarray] = None) -> ClassificationMetrics:
        """Calculate comprehensive classification metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # ROC AUC (for binary classification or with probabilities)
        roc_auc = None
        pr_auc = None
        
        if y_proba is not None:
            try:
                if self.n_classes == 2:
                    # Binary classification
                    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                    
                    # Precision-Recall AUC
                    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba[:, 1])
                    pr_auc = np.trapz(precision_vals, recall_vals)
                    
                else:
                    # Multi-class ROC AUC
                    roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                    
            except Exception as e:
                logger.debug(f"Could not calculate AUC metrics: {e}")
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            precision_recall_auc=pr_auc,
            confusion_matrix=conf_matrix,
            classification_report=class_report
        )
    
    @time_it("classifier_predict", include_args=True)
    def predict(self, X: pd.DataFrame, 
                return_probabilities: bool = False,
                confidence_threshold: Optional[float] = None) -> Union[np.ndarray, PredictionResult]:
        """
        Make predictions using the fitted model
        
        Args:
            X: Feature matrix for prediction
            return_probabilities: Whether to return prediction probabilities
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Predictions or PredictionResult object
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        logger.debug(f"Making predictions for {len(X)} samples")
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            # Make predictions
            predictions = self.model.predict(X_processed)
            
            # Get probabilities if requested or available
            probabilities = None
            confidence_scores = None
            
            if return_probabilities or confidence_threshold is not None:
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X_processed)
                    
                    # Calculate confidence scores (max probability)
                    confidence_scores = np.max(probabilities, axis=1)
                    
                    # Apply confidence threshold if specified
                    if confidence_threshold is not None:
                        low_confidence_mask = confidence_scores < confidence_threshold
                        predictions[low_confidence_mask] = -1  # Mark as uncertain
                
                elif hasattr(self.model, 'decision_function'):
                    # Use decision function for confidence (e.g., SVM)
                    decision_scores = self.model.decision_function(X_processed)
                    if decision_scores.ndim == 1:  # Binary classification
                        confidence_scores = np.abs(decision_scores)
                    else:  # Multi-class
                        confidence_scores = np.max(decision_scores, axis=1)
            
            # Decode predictions if label encoder was used
            if self.label_encoder is not None:
                # Handle uncertain predictions
                certain_mask = predictions != -1
                decoded_predictions = np.full(len(predictions), 'uncertain', dtype=object)
                
                if np.any(certain_mask):
                    decoded_predictions[certain_mask] = self.label_encoder.inverse_transform(
                        predictions[certain_mask]
                    )
                
                predictions = decoded_predictions
            
            # Return simple predictions or detailed result
            if not return_probabilities and confidence_threshold is None:
                return predictions
            else:
                return PredictionResult(
                    predictions=predictions,
                    probabilities=probabilities,
                    confidence_scores=confidence_scores,
                    prediction_metadata={
                        'model_name': self.name,
                        'prediction_time': datetime.now().isoformat(),
                        'n_samples': len(X),
                        'confidence_threshold': confidence_threshold
                    }
                )
        
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise BusinessLogicError(f"Model {self.model_type} does not support probability predictions")
        
        X_processed = self._preprocess_features(X)
        return self.model.predict_proba(X_processed)
    
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Get decision function scores"""
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before getting decision scores")
        
        if not hasattr(self.model, 'decision_function'):
            raise BusinessLogicError(f"Model {self.model_type} does not support decision function")
        
        X_processed = self._preprocess_features(X)
        return self.model.decision_function(X_processed)
    
    @time_it("classifier_evaluate")
    def evaluate(self, X: pd.DataFrame, y: pd.Series, 
                sample_weight: Optional[np.ndarray] = None) -> ClassificationMetrics:
        """
        Evaluate model performance on test data
        
        Args:
            X: Test features
            y: True labels
            sample_weight: Sample weights (optional)
            
        Returns:
            Classification metrics
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before evaluation")
        
        logger.info(f"Evaluating {self.name} on {len(X)} test samples")
        
        # Make predictions
        X_processed = self._preprocess_features(X)
        y_processed = self._preprocess_targets(y)
        
        y_pred = self.model.predict(X_processed)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_processed)
        
        # Calculate metrics
        metrics = self._calculate_classification_metrics(y_processed, y_pred, y_proba)
        
        # Update validation score
        self.validation_score = metrics.accuracy
        
        logger.info(f"Evaluation complete - Accuracy: {metrics.accuracy:.4f}, F1: {metrics.f1_score:.4f}")
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      cv: int = 5,
                      scoring: Union[str, List[str]] = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      n_jobs: int = -1) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            scoring: Scoring metrics
            n_jobs: Number of parallel jobs
            
        Returns:
            Cross-validation scores
        """
        if not self.is_fitted and self.model is None:
            self.model = self._create_model()
        
        logger.info(f"Performing {cv}-fold cross-validation for {self.name}")
        
        # Preprocess data
        X_processed = self._preprocess_features(X)
        y_processed = self._preprocess_targets(y)
        
        # Perform cross-validation
        cv_results = {}
        
        if isinstance(scoring, str):
            scoring = [scoring]
        
        for metric in scoring:
            try:
                scores = cross_val_score(
                    self.model, X_processed, y_processed,
                    cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                    scoring=metric,
                    n_jobs=n_jobs
                )
                cv_results[metric] = scores
                
                logger.debug(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                
            except Exception as e:
                logger.warning(f"Cross-validation failed for metric {metric}: {e}")
        
        # Store results
        self.cross_validation_scores_ = cv_results
        
        return cv_results
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance rankings
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get feature importance")
        
        if self.feature_importances_ is None:
            raise BusinessLogicError("Feature importance not available for this model type")
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             normalize: bool = False) -> Any:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the matrix
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       cmap='Blues')
            
            plt.title(f'Confusion Matrix - {self.name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return None
    
    def plot_roc_curve(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Plot ROC curve (binary classification only)
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Matplotlib figure
        """
        if self.n_classes != 2:
            raise BusinessLogicError("ROC curve plotting only supported for binary classification")
        
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, auc
            
            # Get probabilities
            y_proba = self.predict_proba(X)[:, 1]
            y_processed = self._preprocess_targets(y)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_processed, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Create plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {self.name}')
            plt.legend(loc="lower right")
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add classification-specific information
        summary.update({
            'classification_strategy': self.classification_strategy.value,
            'n_classes': self.n_classes,
            'class_names': self.class_names,
            'prediction_horizon': self.prediction_horizon,
            'min_confidence': self.min_confidence,
            'has_feature_importance': self.feature_importances_ is not None,
            'supports_probabilities': hasattr(self.model, 'predict_proba') if self.model else False,
            'is_probability_calibrated': self.is_probability_calibrated
        })
        
        # Add performance metrics
        if self.classification_metrics_:
            summary['performance_metrics'] = self.classification_metrics_.to_dict()
        
        # Add cross-validation results
        if self.cross_validation_scores_:
            cv_summary = {}
            for metric, scores in self.cross_validation_scores_.items():
                cv_summary[metric] = {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'scores': scores.tolist()
                }
            summary['cross_validation'] = cv_summary
        
        return summary

# ============================================
# Utility Functions for Classification
# ============================================

def create_directional_targets(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Create directional (up/down) targets from price series
    
    Args:
        prices: Price series (usually closing prices)
        horizon: Number of periods to look ahead
        
    Returns:
        Binary targets (1 for up, 0 for down)
    """
    future_returns = prices.pct_change(horizon).shift(-horizon)
    return (future_returns > 0).astype(int)

def create_multiclass_targets(returns: pd.Series, 
                            thresholds: List[float] = [-0.02, 0.02]) -> pd.Series:
    """
    Create multi-class targets from returns
    
    Args:
        returns: Return series
        thresholds: Threshold values for class boundaries
        
    Returns:
        Multi-class targets
    """
    if len(thresholds) != 2:
        raise ValueError("Expected exactly 2 thresholds for 3-class classification")
    
    low_thresh, high_thresh = sorted(thresholds)
    
    conditions = [
        returns <= low_thresh,    # Down
        returns > high_thresh     # Up
    ]
    
    choices = [0, 2]  # Down, Up
    
    return pd.Series(
        np.select(conditions, choices, default=1),  # Default is sideways (1)
        index=returns.index
    )

def balance_classification_data(X: pd.DataFrame, y: pd.Series, 
                              strategy: str = 'undersample') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance classification dataset
    
    Args:
        X: Feature matrix
        y: Target labels
        strategy: Balancing strategy ('undersample', 'oversample', 'smote')
        
    Returns:
        Balanced features and targets
    """
    try:
        if strategy == 'undersample':
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=42)
            
        elif strategy == 'oversample':
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=42)
            
        elif strategy == 'smote':
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=42)
            
        else:
            raise ValueError(f"Unknown balancing strategy: {strategy}")
        
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        # Convert back to DataFrame and Series
        X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
        y_balanced = pd.Series(y_balanced)
        
        logger.info(f"Balanced data using {strategy}: {len(X)} -> {len(X_balanced)} samples")
        
        return X_balanced, y_balanced
        
    except ImportError:
        logger.warning("imbalanced-learn not available, returning original data")
        return X, y
