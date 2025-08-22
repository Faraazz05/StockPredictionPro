# ============================================
# StockPredictionPro - src/models/classification/ensemble.py
# Advanced ensemble classification methods for financial prediction with meta-learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
from collections import defaultdict
import warnings

# Core ensemble imports
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, AdaBoostClassifier,
    StackingClassifier, ExtraTreesClassifier
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, log_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseClassifier, clone

# Import our custom classifiers
from .gradient_boosting import create_gradient_boosting_classifier
from .random_forest import create_random_forest_classifier
from .svm import create_svm_classifier
from .logistic import create_logistic_classifier
from .naive_bayes import create_naive_bayes_classifier
from .knn import create_knn_classifier
from .neural_network import create_neural_network_classifier

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_classifier import BaseFinancialClassifier, ClassificationStrategy

logger = get_logger('models.classification.ensemble')

# ============================================
# Advanced Meta-Learner Implementations
# ============================================

class DynamicWeightedVoting:
    """Dynamic voting with performance-based weights and confidence scoring"""
    
    def __init__(self, models: List[Any], weighting_method: str = 'performance'):
        self.models = models
        self.weighting_method = weighting_method
        self.weights_ = None
        self.performance_history_ = defaultdict(list)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all models and calculate weights"""
        # Fit all models
        for model in self.models:
            model.fit(X, y)
        
        # Calculate weights based on cross-validation performance
        if self.weighting_method == 'performance':
            self.weights_ = self._calculate_performance_weights(X, y)
        elif self.weighting_method == 'diversity':
            self.weights_ = self._calculate_diversity_weights(X, y)
        elif self.weighting_method == 'confidence':
            self.weights_ = self._calculate_confidence_weights(X, y)
        else:
            self.weights_ = np.ones(len(self.models)) / len(self.models)
    
    def _calculate_performance_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate weights based on individual model performance"""
        cv_scores = []
        
        for model in self.models:
            try:
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                cv_scores.append(np.mean(scores))
            except:
                cv_scores.append(0.5)  # Default score for failed models
        
        # Convert to weights (softmax-like normalization)
        cv_scores = np.array(cv_scores)
        exp_scores = np.exp(cv_scores - np.max(cv_scores))  # Numerical stability
        weights = exp_scores / np.sum(exp_scores)
        
        return weights
    
    def _calculate_diversity_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate weights based on prediction diversity"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Calculate pairwise disagreement
        disagreements = []
        for i, pred_i in enumerate(predictions):
            disagreement = 0
            for j, pred_j in enumerate(predictions):
                if i != j:
                    disagreement += np.mean(pred_i != pred_j)
            disagreements.append(disagreement / (len(predictions) - 1))
        
        # Higher disagreement gets higher weight (diversity bonus)
        disagreements = np.array(disagreements)
        weights = disagreements / np.sum(disagreements) if np.sum(disagreements) > 0 else np.ones(len(self.models)) / len(self.models)
        
        return weights
    
    def _calculate_confidence_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate weights based on prediction confidence"""
        confidences = []
        
        for model in self.models:
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    confidence = np.mean(np.max(proba, axis=1))
                else:
                    confidence = 0.5  # Default for models without probability
                confidences.append(confidence)
            except:
                confidences.append(0.5)
        
        # Normalize confidences to weights
        confidences = np.array(confidences)
        weights = confidences / np.sum(confidences) if np.sum(confidences) > 0 else np.ones(len(self.models)) / len(self.models)
        
        return weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted predictions"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted majority voting
        predictions = np.array(predictions).T  # Shape: (n_samples, n_models)
        weighted_predictions = []
        
        for sample_preds in predictions:
            unique_preds, counts = np.unique(sample_preds, return_counts=True)
            
            # Weight the votes
            weighted_counts = np.zeros_like(counts, dtype=float)
            for i, pred in enumerate(unique_preds):
                for j, model_pred in enumerate(sample_preds):
                    if model_pred == pred:
                        weighted_counts[i] += self.weights_[j]
            
            # Select prediction with highest weighted count
            best_pred_idx = np.argmax(weighted_counts)
            weighted_predictions.append(unique_preds[best_pred_idx])
        
        return np.array(weighted_predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make weighted probability predictions"""
        if not all(hasattr(model, 'predict_proba') for model in self.models):
            raise ValueError("All models must support predict_proba for probability prediction")
        
        weighted_probas = None
        
        for i, model in enumerate(self.models):
            proba = model.predict_proba(X)
            if weighted_probas is None:
                weighted_probas = self.weights_[i] * proba
            else:
                weighted_probas += self.weights_[i] * proba
        
        return weighted_probas

class AdaptiveStacking:
    """Advanced stacking with adaptive meta-learner selection"""
    
    def __init__(self, base_models: List[Any], meta_learners: List[Any], 
                 cv_folds: int = 5, selection_metric: str = 'accuracy'):
        self.base_models = base_models
        self.meta_learners = meta_learners
        self.cv_folds = cv_folds
        self.selection_metric = selection_metric
        self.best_meta_learner_ = None
        self.meta_features_ = None
        self.base_model_scores_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit base models and select best meta-learner"""
        # Generate meta-features using cross-validation
        self.meta_features_ = self._generate_meta_features(X, y)
        
        # Evaluate meta-learners and select best
        self.best_meta_learner_ = self._select_best_meta_learner(self.meta_features_, y)
        
        # Fit all base models on full data
        for model in self.base_models:
            model.fit(X, y)
        
        # Fit best meta-learner on meta-features
        self.best_meta_learner_.fit(self.meta_features_, y)
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate meta-features using cross-validation"""
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]
            
            for model_idx, model in enumerate(self.base_models):
                # Clone model to avoid fitting on same instance
                fold_model = clone(model)
                fold_model.fit(X_train_fold, y_train_fold)
                
                # Generate predictions for validation fold
                if hasattr(fold_model, 'predict_proba'):
                    # Use probabilities as meta-features (more informative)
                    proba = fold_model.predict_proba(X_val_fold)
                    # Use max probability or entropy as meta-feature
                    meta_features[val_idx, model_idx] = np.max(proba, axis=1)
                else:
                    # Use predictions as meta-features
                    pred = fold_model.predict(X_val_fold)
                    meta_features[val_idx, model_idx] = pred
        
        return meta_features
    
    def _select_best_meta_learner(self, meta_features: np.ndarray, y: np.ndarray) -> Any:
        """Select best meta-learner using cross-validation"""
        best_score = -np.inf
        best_learner = None
        
        for meta_learner in self.meta_learners:
            try:
                scores = cross_val_score(meta_learner, meta_features, y, 
                                       cv=3, scoring=self.selection_metric)
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_learner = clone(meta_learner)
            except Exception as e:
                logger.warning(f"Error evaluating meta-learner {type(meta_learner).__name__}: {e}")
        
        if best_learner is None:
            # Fallback to first meta-learner
            best_learner = clone(self.meta_learners[0])
        
        return best_learner
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make stacked predictions"""
        # Generate meta-features from base models
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for model_idx, model in enumerate(self.base_models):
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                meta_features[:, model_idx] = np.max(proba, axis=1)
            else:
                pred = model.predict(X)
                meta_features[:, model_idx] = pred
        
        # Make final prediction with meta-learner
        return self.best_meta_learner_.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make stacked probability predictions"""
        if not hasattr(self.best_meta_learner_, 'predict_proba'):
            raise ValueError("Best meta-learner does not support predict_proba")
        
        # Generate meta-features from base models
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for model_idx, model in enumerate(self.base_models):
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                meta_features[:, model_idx] = np.max(proba, axis=1)
            else:
                pred = model.predict(X)
                meta_features[:, model_idx] = pred
        
        return self.best_meta_learner_.predict_proba(meta_features)

class EnsembleDiversityAnalyzer:
    """Analyze ensemble diversity and performance characteristics"""
    
    def __init__(self, models: List[Any]):
        self.models = models
        self.diversity_metrics_ = {}
        self.performance_metrics_ = {}
        
    def analyze(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Comprehensive ensemble analysis"""
        # Generate predictions from all models
        predictions = []
        probabilities = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probabilities.append(proba)
        
        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        
        # Calculate diversity metrics
        self._calculate_diversity_metrics(predictions, y)
        
        # Calculate performance metrics
        self._calculate_performance_metrics(predictions, probabilities, y)
        
        # Analyze pairwise model relationships
        pairwise_analysis = self._analyze_pairwise_relationships(predictions, y)
        
        return {
            'diversity_metrics': self.diversity_metrics_,
            'performance_metrics': self.performance_metrics_,
            'pairwise_analysis': pairwise_analysis,
            'ensemble_characteristics': self._get_ensemble_characteristics()
        }
    
    def _calculate_diversity_metrics(self, predictions: np.ndarray, y: np.ndarray):
        """Calculate various diversity metrics"""
        n_models, n_samples = predictions.shape
        
        # Q-statistic (Yule's Q)
        q_statistics = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                q_stat = self._calculate_q_statistic(predictions[i], predictions[j], y)
                q_statistics.append(q_stat)
        
        # Correlation coefficient
        correlations = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                correlations.append(corr)
        
        # Disagreement measure
        disagreements = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreement = np.mean(predictions[i] != predictions[j])
                disagreements.append(disagreement)
        
        # Double fault measure
        double_faults = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                both_wrong = (predictions[i] != y) & (predictions[j] != y)
                double_fault = np.mean(both_wrong)
                double_faults.append(double_fault)
        
        self.diversity_metrics_ = {
            'q_statistic_mean': np.mean(q_statistics),
            'q_statistic_std': np.std(q_statistics),
            'correlation_mean': np.mean(correlations),
            'correlation_std': np.std(correlations),
            'disagreement_mean': np.mean(disagreements),
            'disagreement_std': np.std(disagreements),
            'double_fault_mean': np.mean(double_faults),
            'double_fault_std': np.std(double_faults),
            'diversity_score': np.mean(disagreements) - np.mean(double_faults)  # Custom diversity score
        }
    
    def _calculate_q_statistic(self, pred1: np.ndarray, pred2: np.ndarray, y: np.ndarray) -> float:
        """Calculate Q-statistic between two classifiers"""
        # Create contingency table
        both_correct = (pred1 == y) & (pred2 == y)
        first_correct = (pred1 == y) & (pred2 != y)
        second_correct = (pred1 != y) & (pred2 == y)
        both_wrong = (pred1 != y) & (pred2 != y)
        
        n11 = np.sum(both_correct)
        n10 = np.sum(first_correct)
        n01 = np.sum(second_correct)
        n00 = np.sum(both_wrong)
        
        # Calculate Q-statistic
        numerator = (n11 * n00) - (n01 * n10)
        denominator = (n11 * n00) + (n01 * n10)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_performance_metrics(self, predictions: np.ndarray, 
                                     probabilities: List[np.ndarray], y: np.ndarray):
        """Calculate individual and ensemble performance metrics"""
        n_models = len(predictions)
        
        # Individual model performance
        individual_accuracies = []
        individual_f1_scores = []
        
        for i in range(n_models):
            acc = accuracy_score(y, predictions[i])
            f1 = f1_score(y, predictions[i], average='weighted')
            individual_accuracies.append(acc)
            individual_f1_scores.append(f1)
        
        # Ensemble performance (majority voting)
        ensemble_pred = []
        for sample_idx in range(predictions.shape[1]):
            sample_preds = predictions[:, sample_idx]
            unique_preds, counts = np.unique(sample_preds, return_counts=True)
            majority_pred = unique_preds[np.argmax(counts)]
            ensemble_pred.append(majority_pred)
        
        ensemble_pred = np.array(ensemble_pred)
        ensemble_accuracy = accuracy_score(y, ensemble_pred)
        ensemble_f1 = f1_score(y, ensemble_pred, average='weighted')
        
        self.performance_metrics_ = {
            'individual_accuracies': individual_accuracies,
            'individual_f1_scores': individual_f1_scores,
            'mean_individual_accuracy': np.mean(individual_accuracies),
            'std_individual_accuracy': np.std(individual_accuracies),
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_f1': ensemble_f1,
            'ensemble_improvement': ensemble_accuracy - np.mean(individual_accuracies)
        }
    
    def _analyze_pairwise_relationships(self, predictions: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze pairwise model relationships"""
        n_models = predictions.shape[0]
        model_names = [f"Model_{i}" for i in range(n_models)]
        
        # Create pairwise analysis matrix
        pairwise_correlations = np.zeros((n_models, n_models))
        pairwise_q_stats = np.zeros((n_models, n_models))
        pairwise_disagreements = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    pairwise_correlations[i, j] = 1.0
                    pairwise_q_stats[i, j] = 1.0
                    pairwise_disagreements[i, j] = 0.0
                else:
                    corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                    q_stat = self._calculate_q_statistic(predictions[i], predictions[j], y)
                    disagreement = np.mean(predictions[i] != predictions[j])
                    
                    pairwise_correlations[i, j] = corr
                    pairwise_q_stats[i, j] = q_stat
                    pairwise_disagreements[i, j] = disagreement
        
        return {
            'model_names': model_names,
            'correlations': pairwise_correlations,
            'q_statistics': pairwise_q_stats,
            'disagreements': pairwise_disagreements
        }
    
    def _get_ensemble_characteristics(self) -> Dict[str, str]:
        """Get high-level ensemble characteristics"""
        characteristics = {}
        
        # Diversity assessment
        div_score = self.diversity_metrics_.get('diversity_score', 0)
        if div_score > 0.3:
            characteristics['diversity'] = 'High'
        elif div_score > 0.1:
            characteristics['diversity'] = 'Moderate'
        else:
            characteristics['diversity'] = 'Low'
        
        # Correlation assessment
        corr_mean = self.diversity_metrics_.get('correlation_mean', 0)
        if corr_mean < 0.3:
            characteristics['correlation'] = 'Low (Good)'
        elif corr_mean < 0.7:
            characteristics['correlation'] = 'Moderate'
        else:
            characteristics['correlation'] = 'High (Concerning)'
        
        # Performance improvement
        improvement = self.performance_metrics_.get('ensemble_improvement', 0)
        if improvement > 0.05:
            characteristics['ensemble_benefit'] = 'Significant'
        elif improvement > 0.01:
            characteristics['ensemble_benefit'] = 'Moderate'
        else:
            characteristics['ensemble_benefit'] = 'Minimal'
        
        return characteristics

# ============================================
# Main Ensemble Classifier
# ============================================

class FinancialEnsembleClassifier(BaseFinancialClassifier):
    """
    Advanced ensemble classification model optimized for financial data
    
    Features:
    - Multiple ensemble methods: Voting, Bagging, Boosting, Stacking
    - Automatic model selection and optimization
    - Dynamic weighting based on performance and diversity
    - Comprehensive ensemble analysis and diagnostics
    - Financial domain optimizations (volatility weighting, sector-aware ensembles)
    - Advanced meta-learning with adaptive stacking
    """
    
    def __init__(self,
                 name: str = "ensemble_classifier",
                 ensemble_method: str = 'voting',
                 base_models: Optional[List[str]] = None,
                 n_models: Optional[int] = None,
                 voting_type: str = 'soft',
                 weighting_method: str = 'performance',
                 stacking_meta_learner: str = 'logistic',
                 cv_folds: int = 5,
                 optimize_weights: bool = True,
                 diversity_threshold: float = 0.1,
                 performance_threshold: float = 0.6,
                 calibrate_probabilities: bool = True,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Financial Ensemble Classifier
        
        Args:
            name: Model name
            ensemble_method: Ensemble method ('voting', 'bagging', 'boosting', 'stacking', 'dynamic')
            base_models: List of base model types to include
            n_models: Number of models (for auto-generated ensembles)
            voting_type: Voting type ('hard', 'soft')
            weighting_method: Method for calculating weights ('performance', 'diversity', 'confidence')
            stacking_meta_learner: Meta-learner for stacking
            cv_folds: Cross-validation folds
            optimize_weights: Whether to optimize ensemble weights
            diversity_threshold: Minimum diversity threshold
            performance_threshold: Minimum performance threshold
            calibrate_probabilities: Whether to calibrate probabilities
            random_state: Random seed
        """
        super().__init__(
            name=name,
            model_type="ensemble_classifier",
            classification_strategy=ClassificationStrategy.DIRECTION_PREDICTION,
            **kwargs
        )
        
        self.ensemble_method = ensemble_method
        self.base_models = base_models or ['gradient_boosting', 'random_forest', 'svm', 'logistic', 'neural_network']
        self.n_models = n_models
        self.voting_type = voting_type
        self.weighting_method = weighting_method
        self.stacking_meta_learner = stacking_meta_learner
        self.cv_folds = cv_folds
        self.optimize_weights = optimize_weights
        self.diversity_threshold = diversity_threshold
        self.performance_threshold = performance_threshold
        self.calibrate_probabilities = calibrate_probabilities
        self.random_state = random_state
        
        # Ensemble components
        self.label_encoder_: Optional[LabelEncoder] = None
        self.scaler_: Optional[StandardScaler] = None
        self.base_models_: List[Any] = []
        self.ensemble_model_: Optional[Any] = None
        self.calibrated_model_: Optional[CalibratedClassifierCV] = None
        self.ensemble_weights_: Optional[np.ndarray] = None
        self.diversity_analyzer_: Optional[EnsembleDiversityAnalyzer] = None
        self.ensemble_analysis_: Optional[Dict[str, Any]] = None
        self.model_selection_results_: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized {ensemble_method} ensemble classifier: {self.name}")
    
    def _create_base_models(self) -> List[Any]:
        """Create base models based on configuration"""
        models = []
        
        for model_name in self.base_models:
            try:
                if model_name == 'gradient_boosting':
                    model = create_gradient_boosting_classifier(
                        performance_preset='balanced',
                        random_state=self.random_state
                    )
                elif model_name == 'random_forest':
                    model = create_random_forest_classifier(
                        performance_preset='balanced',
                        random_state=self.random_state
                    )
                elif model_name == 'svm':
                    model = create_svm_classifier(
                        kernel='rbf',
                        performance_preset='balanced',
                        random_state=self.random_state
                    )
                elif model_name == 'logistic':
                    model = create_logistic_classifier(
                        performance_preset='balanced',
                        random_state=self.random_state
                    )
                elif model_name == 'naive_bayes':
                    model = create_naive_bayes_classifier(
                        performance_preset='balanced',
                        random_state=self.random_state
                    )
                elif model_name == 'knn':
                    model = create_knn_classifier(
                        k='auto',
                        performance_preset='balanced'
                    )
                elif model_name == 'neural_network':
                    model = create_neural_network_classifier(
                        architecture='balanced',
                        epochs=100,  # Reduced for ensemble training
                        verbose=0,
                        random_state=self.random_state
                    )
                else:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue
                
                models.append(model)
                logger.debug(f"Created {model_name} base model")
                
            except Exception as e:
                logger.warning(f"Failed to create {model_name}: {e}")
        
        return models
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with scaling"""
        X_processed = super()._preprocess_features(X)
        
        if self.scaler_ is None:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_processed)
            logger.debug("Fitted feature scaler for ensemble")
        else:
            X_scaled = self.scaler_.transform(X_processed)
        
        return X_scaled
    
    def _preprocess_targets(self, y: pd.Series) -> np.ndarray:
        """Preprocess target labels with encoding"""
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        if self.label_encoder_ is None:
            self.label_encoder_ = LabelEncoder()
            y_encoded = self.label_encoder_.fit_transform(y_array)
            self.classes_ = self.label_encoder_.classes_
            logger.debug(f"Fitted label encoder. Classes: {self.classes_}")
        else:
            y_encoded = self.label_encoder_.transform(y_array)
        
        return y_encoded
    
    def _select_best_models(self, X: np.ndarray, y: np.ndarray) -> List[Any]:
        """Select best performing and diverse models"""
        candidate_models = self._create_base_models()
        
        # Evaluate all models
        model_scores = []
        model_predictions = []
        
        for model in candidate_models:
            try:
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                avg_score = np.mean(scores)
                
                if avg_score >= self.performance_threshold:
                    model_scores.append((model, avg_score))
                    
                    # Get predictions for diversity analysis
                    model.fit(X, y)
                    pred = model.predict(X)
                    model_predictions.append(pred)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate model {type(model).__name__}: {e}")
        
        if not model_scores:
            logger.warning("No models meet performance threshold, using all available models")
            return candidate_models[:min(5, len(candidate_models))]  # Limit to prevent overfitting
        
        # Sort by performance
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select diverse models
        selected_models = [model_scores[0][0]]  # Start with best model
        selected_predictions = [model_predictions[0]]
        
        for model, score in model_scores[1:]:
            # Check diversity with already selected models
            model_idx = [m for m, _ in model_scores].index(model)
            current_pred = model_predictions[model_idx]
            
            is_diverse = True
            for selected_pred in selected_predictions:
                diversity = np.mean(current_pred != selected_pred)
                if diversity < self.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected_models.append(model)
                selected_predictions.append(current_pred)
            
            # Limit ensemble size
            if len(selected_models) >= (self.n_models or 7):
                break
        
        self.model_selection_results_ = {
            'candidate_scores': [(type(m).__name__, s) for m, s in model_scores],
            'selected_models': [type(m).__name__ for m in selected_models],
            'selection_criteria': {
                'performance_threshold': self.performance_threshold,
                'diversity_threshold': self.diversity_threshold,
                'max_models': self.n_models or 7
            }
        }
        
        logger.info(f"Selected {len(selected_models)} models for ensemble: {[type(m).__name__ for m in selected_models]}")
        
        return selected_models
    
    def _create_ensemble_model(self, base_models: List[Any]):
        """Create the ensemble model based on method"""
        
        if self.ensemble_method == 'voting':
            if self.voting_type == 'soft' and all(hasattr(model, 'predict_proba') for model in base_models):
                ensemble = VotingClassifier(
                    estimators=[(f"model_{i}", model) for i, model in enumerate(base_models)],
                    voting='soft'
                )
            else:
                ensemble = VotingClassifier(
                    estimators=[(f"model_{i}", model) for i, model in enumerate(base_models)],
                    voting='hard'
                )
            
        elif self.ensemble_method == 'bagging':
            # Use the best performing model as base for bagging
            best_model = base_models[0]  # Already sorted by performance
            ensemble = BaggingClassifier(
                base_estimator=best_model,
                n_estimators=min(10, len(base_models) * 2),
                random_state=self.random_state
            )
            
        elif self.ensemble_method == 'boosting':
            # Use simple model as base for boosting
            from sklearn.tree import DecisionTreeClassifier
            base_estimator = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
            ensemble = AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=50,
                random_state=self.random_state
            )
            
        elif self.ensemble_method == 'stacking':
            # Create meta-learner
            if self.stacking_meta_learner == 'logistic':
                meta_learner = create_logistic_classifier(random_state=self.random_state)
            elif self.stacking_meta_learner == 'svm':
                meta_learner = create_svm_classifier(random_state=self.random_state)
            elif self.stacking_meta_learner == 'neural_network':
                meta_learner = create_neural_network_classifier(
                    architecture='simple',
                    epochs=50,
                    verbose=0,
                    random_state=self.random_state
                )
            else:
                meta_learner = create_logistic_classifier(random_state=self.random_state)
            
            ensemble = StackingClassifier(
                estimators=[(f"model_{i}", model) for i, model in enumerate(base_models)],
                final_estimator=meta_learner,
                cv=self.cv_folds,
                passthrough=False
            )
            
        elif self.ensemble_method == 'dynamic':
            # Use dynamic weighted voting
            ensemble = DynamicWeightedVoting(base_models, self.weighting_method)
            
        elif self.ensemble_method == 'adaptive_stacking':
            # Create multiple meta-learners for adaptive selection
            meta_learners = [
                create_logistic_classifier(random_state=self.random_state),
                create_svm_classifier(kernel='linear', random_state=self.random_state),
            ]
            try:
                meta_learners.append(create_neural_network_classifier(
                    architecture='simple', epochs=50, verbose=0, random_state=self.random_state
                ))
            except:
                pass  # Skip if neural network fails
            
            ensemble = AdaptiveStacking(base_models, meta_learners, self.cv_folds)
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble
    
    def _analyze_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Analyze ensemble diversity and performance"""
        if len(self.base_models_) > 1:
            self.diversity_analyzer_ = EnsembleDiversityAnalyzer(self.base_models_)
            self.ensemble_analysis_ = self.diversity_analyzer_.analyze(X, y)
            
            logger.info(f"Ensemble diversity score: {self.ensemble_analysis_['diversity_metrics']['diversity_score']:.3f}")
            logger.info(f"Ensemble improvement: {self.ensemble_analysis_['performance_metrics']['ensemble_improvement']:.3f}")
    
    def _calibrate_probabilities(self, X: np.ndarray, y: np.ndarray):
        """Calibrate ensemble probabilities"""
        try:
            self.calibrated_model_ = CalibratedClassifierCV(
                base_estimator=self.ensemble_model_,
                method='isotonic',
                cv=3
            )
            self.calibrated_model_.fit(X, y)
            logger.debug("Calibrated ensemble probabilities")
        except Exception as e:
            logger.warning(f"Could not calibrate probabilities: {e}")
            self.calibrated_model_ = None
    
    @time_it("ensemble_fit", include_args=True)
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'FinancialEnsembleClassifier':
        """Fit the ensemble classification model"""
        logger.info(f"Fitting Ensemble Classifier ({self.ensemble_method}) on {len(X)} samples with {X.shape[1]} features")
        
        # Validate input
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
            
            # Preprocess data
            X_processed = self._preprocess_features(X)
            y_processed = self._preprocess_targets(y)
            
            # Select best models
            self.base_models_ = self._select_best_models(X_processed, y_processed)
            
            # Create ensemble model
            fit_start = datetime.now()
            self.ensemble_model_ = self._create_ensemble_model(self.base_models_)
            
            # Fit ensemble
            self.ensemble_model_.fit(X_processed, y_processed)
            
            fit_duration = (datetime.now() - fit_start).total_seconds()
            self.training_duration = fit_duration
            
            # Analyze ensemble
            self._analyze_ensemble(X_processed, y_processed)
            
            # Calibrate probabilities
            if self.calibrate_probabilities and hasattr(self.ensemble_model_, 'predict_proba'):
                self._calibrate_probabilities(X_processed, y_processed)
            
            # Update metadata
            self.update_metadata({
                'training_samples': len(X),
                'training_features': X.shape[1],
                'n_classes': len(self.classes_),
                'class_names': self.classes_.tolist(),
                'ensemble_method': self.ensemble_method,
                'n_base_models': len(self.base_models_),
                'base_model_types': [type(m).__name__ for m in self.base_models_],
                'training_duration_seconds': fit_duration
            })
            
            # Calculate training score
            predictions = self.predict(X)
            self.training_score = accuracy_score(y, predictions)
            
            # Update status
            self.status = ModelStatus.TRAINED
            self.is_fitted = True
            
            logger.info(f"Ensemble training complete in {fit_duration:.2f}s")
            return self
            
        except Exception as e:
            from ..base.base_model import ModelStatus
            self.status = ModelStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Ensemble training failed: {e}")
            raise
    
    @time_it("ensemble_predict", include_args=True)
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        logger.debug(f"Making ensemble predictions for {len(X)} samples")
        
        try:
            X_processed = self._preprocess_features(X)
            
            if self.calibrated_model_ is not None:
                predictions_encoded = self.calibrated_model_.predict(X_processed)
            else:
                predictions_encoded = self.ensemble_model_.predict(X_processed)
            
            predictions = self.label_encoder_.inverse_transform(predictions_encoded)
            
            self.log_prediction()
            return predictions
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise
    
    @time_it("ensemble_predict_proba", include_args=True)
    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict ensemble probabilities"""
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        logger.debug(f"Making ensemble probability predictions for {len(X)} samples")
        
        try:
            X_processed = self._preprocess_features(X)
            
            if self.calibrated_model_ is not None:
                probabilities = self.calibrated_model_.predict_proba(X_processed)
            elif hasattr(self.ensemble_model_, 'predict_proba'):
                probabilities = self.ensemble_model_.predict_proba(X_processed)
            else:
                raise ValueError("Ensemble model does not support probability prediction")
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Ensemble probability prediction failed: {e}")
            raise
    
    def get_base_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from individual base models"""
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get base model predictions")
        
        X_processed = self._preprocess_features(X)
        predictions = {}
        
        for i, model in enumerate(self.base_models_):
            try:
                pred = model.predict(X_processed)
                predictions[f"{type(model).__name__}_{i}"] = self.label_encoder_.inverse_transform(pred)
            except Exception as e:
                logger.warning(f"Could not get predictions from base model {i}: {e}")
        
        return predictions
    
    def get_ensemble_analysis(self) -> Dict[str, Any]:
        """Get comprehensive ensemble analysis"""
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get ensemble analysis")
        
        return self.ensemble_analysis_.copy() if self.ensemble_analysis_ else {}
    
    def plot_ensemble_analysis(self) -> Any:
        """Plot ensemble analysis results"""
        if not self.ensemble_analysis_:
            logger.warning("Ensemble analysis not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            analysis = self.ensemble_analysis_
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # Model performance comparison
            perf_metrics = analysis['performance_metrics']
            model_names = [type(m).__name__ for m in self.base_models_]
            
            axes[0, 0].bar(model_names, perf_metrics['individual_accuracies'], alpha=0.7, color='steelblue')
            axes[0, 0].axhline(y=perf_metrics['ensemble_accuracy'], color='red', linestyle='--', 
                              label=f'Ensemble: {perf_metrics["ensemble_accuracy"]:.3f}')
            axes[0, 0].set_title('Model Accuracies')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Diversity metrics
            div_metrics = analysis['diversity_metrics']
            metric_names = ['Q-Statistic', 'Correlation', 'Disagreement', 'Double Fault']
            metric_values = [
                div_metrics['q_statistic_mean'],
                div_metrics['correlation_mean'],
                div_metrics['disagreement_mean'],
                div_metrics['double_fault_mean']
            ]
            
            bars = axes[0, 1].bar(metric_names, metric_values, alpha=0.7, color='orange')
            axes[0, 1].set_title('Diversity Metrics')
            axes[0, 1].set_ylabel('Metric Value')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add values on bars
            for bar, value in zip(bars, metric_values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, value + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            # Pairwise correlations heatmap
            pairwise = analysis['pairwise_analysis']
            correlations = pairwise['correlations']
            model_names = pairwise['model_names']
            
            sns.heatmap(correlations, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                       xticklabels=model_names, yticklabels=model_names, ax=axes[0, 2])
            axes[0, 2].set_title('Model Correlation Matrix')
            
            # Q-statistics heatmap
            q_stats = pairwise['q_statistics']
            sns.heatmap(q_stats, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                       xticklabels=model_names, yticklabels=model_names, ax=axes[1, 0])
            axes[1, 0].set_title('Q-Statistics Matrix')
            
            # Ensemble characteristics
            characteristics = analysis['ensemble_characteristics']
            char_names = list(characteristics.keys())
            char_values = list(characteristics.values())
            
            # Create color mapping for characteristics
            colors = []
            for value in char_values:
                if 'High' in value or 'Significant' in value:
                    colors.append('green')
                elif 'Moderate' in value:
                    colors.append('orange')
                elif 'Low' in value or 'Minimal' in value:
                    colors.append('red')
                else:
                    colors.append('gray')
            
            axes[1, 1].bar(char_names, [1]*len(char_names), color=colors, alpha=0.7)
            axes[1, 1].set_title('Ensemble Characteristics')
            axes[1, 1].set_ylim(0, 1.2)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add characteristic labels
            for i, (name, value) in enumerate(zip(char_names, char_values)):
                axes[1, 1].text(i, 0.5, value, ha='center', va='center', 
                               fontweight='bold', fontsize=10)
            
            # Performance improvement analysis
            improvement = perf_metrics['ensemble_improvement']
            individual_mean = perf_metrics['mean_individual_accuracy']
            ensemble_acc = perf_metrics['ensemble_accuracy']
            
            categories = ['Individual Mean', 'Ensemble']
            values = [individual_mean, ensemble_acc]
            colors = ['lightblue', 'darkblue']
            
            bars = axes[1, 2].bar(categories, values, color=colors, alpha=0.7)
            axes[1, 2].set_title(f'Performance Improvement: +{improvement:.3f}')
            axes[1, 2].set_ylabel('Accuracy')
            
            # Add values on bars
            for bar, value in zip(bars, values):
                axes[1, 2].text(bar.get_x() + bar.get_width()/2, value + 0.005,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle(f'Ensemble Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return None
    
    def plot_confusion_matrix(self, X: pd.DataFrame, y: pd.Series, 
                             normalize: str = 'true') -> Any:
        """Plot confusion matrix"""
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
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble summary"""
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get ensemble summary")
        
        summary = {
            'ensemble_config': {
                'method': self.ensemble_method,
                'n_base_models': len(self.base_models_),
                'base_model_types': [type(m).__name__ for m in self.base_models_],
                'voting_type': self.voting_type,
                'weighting_method': self.weighting_method,
                'cv_folds': self.cv_folds
            },
            'model_selection': self.model_selection_results_,
            'ensemble_analysis': self.ensemble_analysis_,
            'performance_summary': {
                'training_score': self.training_score,
                'training_duration': self.training_duration,
                'calibrated': self.calibrated_model_ is not None
            }
        }
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        summary.update({
            'model_family': 'Ensemble',
            'ensemble_method': self.ensemble_method,
            'n_base_models': len(self.base_models_) if self.base_models_ else 0,
            'base_model_types': [type(m).__name__ for m in self.base_models_] if self.base_models_ else [],
            'voting_type': self.voting_type,
            'weighting_method': self.weighting_method,
            'diversity_optimization': self.optimize_weights,
            'probability_calibration': self.calibrate_probabilities,
            'n_classes': len(self.classes_) if hasattr(self, 'classes_') else None
        })
        
        # Add ensemble analysis if available
        if self.ensemble_analysis_:
            diversity_metrics = self.ensemble_analysis_['diversity_metrics']
            performance_metrics = self.ensemble_analysis_['performance_metrics']
            characteristics = self.ensemble_analysis_['ensemble_characteristics']
            
            summary.update({
                'diversity_score': diversity_metrics['diversity_score'],
                'ensemble_improvement': performance_metrics['ensemble_improvement'],
                'diversity_level': characteristics.get('diversity', 'Unknown'),
                'correlation_level': characteristics.get('correlation', 'Unknown'),
                'ensemble_benefit': characteristics.get('ensemble_benefit', 'Unknown')
            })
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_ensemble_classifier(ensemble_method: str = 'voting',
                              complexity: str = 'balanced',
                              **kwargs) -> FinancialEnsembleClassifier:
    """Create ensemble classifier with different complexity levels"""
    
    # Base configurations by complexity
    complexity_configs = {
        'simple': {
            'base_models': ['random_forest', 'logistic', 'svm'],
            'n_models': 3,
            'cv_folds': 3
        },
        'balanced': {
            'base_models': ['gradient_boosting', 'random_forest', 'svm', 'logistic', 'neural_network'],
            'n_models': 5,
            'cv_folds': 5
        },
        'comprehensive': {
            'base_models': ['gradient_boosting', 'random_forest', 'svm', 'logistic', 
                           'naive_bayes', 'knn', 'neural_network'],
            'n_models': 7,
            'cv_folds': 5
        }
    }
    
    # Method-specific configurations
    method_configs = {
        'voting': {'voting_type': 'soft', 'weighting_method': 'performance'},
        'stacking': {'stacking_meta_learner': 'logistic'},
        'dynamic': {'weighting_method': 'performance', 'optimize_weights': True},
        'adaptive_stacking': {'cv_folds': 5}
    }
    
    config = {
        'name': f'{ensemble_method}_ensemble_{complexity}',
        'ensemble_method': ensemble_method,
        'calibrate_probabilities': True,
        'random_state': 42
    }
    
    # Add complexity configuration
    config.update(complexity_configs.get(complexity, complexity_configs['balanced']))
    
    # Add method configuration
    config.update(method_configs.get(ensemble_method, {}))
    
    # Override with user parameters
    config.update(kwargs)
    
    return FinancialEnsembleClassifier(**config)

def create_voting_ensemble(**kwargs) -> FinancialEnsembleClassifier:
    """Create voting ensemble classifier"""
    return create_ensemble_classifier(
        ensemble_method='voting',
        complexity='balanced',
        name='voting_ensemble_classifier',
        **kwargs
    )

def create_stacking_ensemble(**kwargs) -> FinancialEnsembleClassifier:
    """Create stacking ensemble classifier"""
    return create_ensemble_classifier(
        ensemble_method='stacking',
        complexity='balanced',
        name='stacking_ensemble_classifier',
        **kwargs
    )

def create_dynamic_ensemble(**kwargs) -> FinancialEnsembleClassifier:
    """Create dynamic weighted ensemble classifier"""
    return create_ensemble_classifier(
        ensemble_method='dynamic',
        complexity='balanced',
        name='dynamic_ensemble_classifier',
        **kwargs
    )

def create_adaptive_stacking_ensemble(**kwargs) -> FinancialEnsembleClassifier:
    """Create adaptive stacking ensemble classifier"""
    return create_ensemble_classifier(
        ensemble_method='adaptive_stacking',
        complexity='comprehensive',
        name='adaptive_stacking_ensemble',
        **kwargs
    )

def create_binary_ensemble(**kwargs) -> FinancialEnsembleClassifier:
    """Create ensemble optimized for binary classification"""
    return create_ensemble_classifier(
        ensemble_method='voting',
        complexity='balanced',
        voting_type='soft',
        name='binary_ensemble_classifier',
        **kwargs
    )

def create_multiclass_ensemble(**kwargs) -> FinancialEnsembleClassifier:
    """Create ensemble optimized for multiclass classification"""
    return create_ensemble_classifier(
        ensemble_method='stacking',
        complexity='comprehensive',
        stacking_meta_learner='neural_network',
        name='multiclass_ensemble_classifier',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def compare_ensemble_methods(X: pd.DataFrame, y: pd.Series,
                           methods: List[str] = ['voting', 'stacking', 'dynamic'],
                           cv: int = 5) -> Dict[str, Any]:
    """Compare different ensemble methods"""
    
    logger.info(f"Comparing ensemble methods: {methods}")
    
    results = {}
    
    for method in methods:
        logger.info(f"Evaluating {method} ensemble")
        
        try:
            # Create ensemble
            ensemble = create_ensemble_classifier(
                ensemble_method=method,
                complexity='balanced'
            )
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
            
            # Time fitting
            import time
            start_time = time.time()
            ensemble.fit(X, y)
            fit_time = time.time() - start_time
            
            # Get ensemble analysis if available
            ensemble_analysis = None
            try:
                ensemble_analysis = ensemble.get_ensemble_analysis()
            except:
                pass
            
            results[method] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'fit_time': fit_time,
                'ensemble_analysis': ensemble_analysis,
                'n_base_models': len(ensemble.base_models_) if ensemble.base_models_ else 0
            }
            
        except Exception as e:
            logger.warning(f"Error with {method} ensemble: {e}")
            results[method] = {'error': str(e)}
    
    # Add comparison summary
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_method = max(valid_results.keys(), key=lambda k: valid_results[k]['cv_mean'])
        fastest_method = min(valid_results.keys(), key=lambda k: valid_results[k]['fit_time'])
        
        results['comparison'] = {
            'best_accuracy': best_method,
            'fastest': fastest_method,
            'method_rankings': sorted(valid_results.keys(), 
                                   key=lambda k: valid_results[k]['cv_mean'], reverse=True)
        }
    
    logger.info(f"Ensemble comparison complete. Best method: {results['comparison']['best_accuracy']}")
    
    return results

def optimize_ensemble_composition(X: pd.DataFrame, y: pd.Series,
                                candidate_models: List[str] = None,
                                max_models: int = 7,
                                diversity_weight: float = 0.3) -> Dict[str, Any]:
    """Optimize ensemble composition using performance and diversity"""
    
    if candidate_models is None:
        candidate_models = ['gradient_boosting', 'random_forest', 'svm', 'logistic', 
                          'naive_bayes', 'knn', 'neural_network']
    
    logger.info(f"Optimizing ensemble composition from {len(candidate_models)} candidate models")
    
    # Create and evaluate all candidate models
    model_performances = {}
    model_predictions = {}
    
    for model_name in candidate_models:
        logger.info(f"Evaluating {model_name}")
        
        try:
            # Create model
            if model_name == 'gradient_boosting':
                model = create_gradient_boosting_classifier(performance_preset='balanced')
            elif model_name == 'random_forest':
                model = create_random_forest_classifier(performance_preset='balanced')
            elif model_name == 'svm':
                model = create_svm_classifier(performance_preset='balanced')
            elif model_name == 'logistic':
                model = create_logistic_classifier(performance_preset='balanced')
            elif model_name == 'naive_bayes':
                model = create_naive_bayes_classifier(performance_preset='balanced')
            elif model_name == 'knn':
                model = create_knn_classifier(performance_preset='balanced')
            elif model_name == 'neural_network':
                model = create_neural_network_classifier(architecture='balanced', epochs=50, verbose=0)
            else:
                continue
            
            # Evaluate with cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            
            model_performances[model_name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'model': model
            }
            
            # Get predictions for diversity analysis
            model.fit(X, y)
            predictions = model.predict(X)
            model_predictions[model_name] = predictions
            
        except Exception as e:
            logger.warning(f"Failed to evaluate {model_name}: {e}")
    
    # Select optimal ensemble using greedy algorithm
    selected_models = []
    selected_names = []
    remaining_models = list(model_performances.keys())
    
    # Start with best performing model
    best_model = max(remaining_models, key=lambda m: model_performances[m]['mean_score'])
    selected_models.append(model_performances[best_model]['model'])
    selected_names.append(best_model)
    remaining_models.remove(best_model)
    
    # Greedily add models that maximize performance + diversity
    while len(selected_models) < max_models and remaining_models:
        best_addition = None
        best_score = -np.inf
        
        for candidate in remaining_models:
            # Calculate ensemble performance with this addition
            temp_models = selected_models + [model_performances[candidate]['model']]
            temp_names = selected_names + [candidate]
            
            # Estimate ensemble performance (simple majority voting)
            temp_predictions = []
            for name in temp_names:
                temp_predictions.append(model_predictions[name])
            
            # Majority vote
            ensemble_pred = []
            for i in range(len(X)):
                votes = [pred[i] for pred in temp_predictions]
                ensemble_pred.append(max(set(votes), key=votes.count))
            
            ensemble_accuracy = accuracy_score(y, ensemble_pred)
            
            # Calculate diversity bonus
            diversity_scores = []
            candidate_pred = model_predictions[candidate]
            for selected_name in selected_names:
                selected_pred = model_predictions[selected_name]
                diversity = np.mean(candidate_pred != selected_pred)
                diversity_scores.append(diversity)
            
            avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
            
            # Combined score
            combined_score = (1 - diversity_weight) * ensemble_accuracy + diversity_weight * avg_diversity
            
            if combined_score > best_score:
                best_score = combined_score
                best_addition = candidate
        
        if best_addition:
            selected_models.append(model_performances[best_addition]['model'])
            selected_names.append(best_addition)
            remaining_models.remove(best_addition)
        else:
            break  # No more beneficial additions
    
    # Create optimized ensemble
    optimized_ensemble = FinancialEnsembleClassifier(
        name='optimized_ensemble',
        ensemble_method='voting',
        voting_type='soft'
    )
    
    # Manually set the selected base models
    optimized_ensemble.base_models_ = selected_models
    optimized_ensemble.fit(X, y)
    
    # Evaluate optimized ensemble
    final_score = optimized_ensemble.evaluate(X, y).accuracy
    
    results = {
        'selected_models': selected_names,
        'model_performances': {name: perf['mean_score'] for name, perf in model_performances.items()},
        'optimized_ensemble': optimized_ensemble,
        'final_ensemble_score': final_score,
        'optimization_params': {
            'max_models': max_models,
            'diversity_weight': diversity_weight,
            'selection_algorithm': 'greedy'
        }
    }
    
    logger.info(f"Ensemble optimization complete. Selected {len(selected_names)} models: {selected_names}")
    logger.info(f"Final ensemble score: {final_score:.4f}")
    
    return results

def analyze_ensemble_diversity(models: List[Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Comprehensive ensemble diversity analysis"""
    
    logger.info(f"Analyzing diversity of {len(models)} models")
    
    # Fit all models and get predictions
    predictions = []
    model_names = []
    
    for i, model in enumerate(models):
        try:
            model.fit(X, y)
            pred = model.predict(X)
            predictions.append(pred)
            model_names.append(f"{type(model).__name__}_{i}")
        except Exception as e:
            logger.warning(f"Failed to fit model {i}: {e}")
    
    if len(predictions) < 2:
        logger.warning("Need at least 2 models for diversity analysis")
        return {}
    
    predictions = np.array(predictions)
    
    # Create analyzer
    analyzer = EnsembleDiversityAnalyzer(models[:len(predictions)])
    analysis = analyzer.analyze(X.values, y.values)
    
    # Additional analysis
    additional_analysis = {
        'model_names': model_names,
        'n_models_analyzed': len(predictions),
        'prediction_agreement_matrix': np.zeros((len(predictions), len(predictions))),
        'model_strengths': {},
        'complementarity_analysis': {}
    }
    
    # Calculate prediction agreement matrix
    for i in range(len(predictions)):
        for j in range(len(predictions)):
            agreement = np.mean(predictions[i] == predictions[j])
            additional_analysis['prediction_agreement_matrix'][i, j] = agreement
    
    # Analyze model strengths (accuracy per class)
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        class_accuracies = {}
        for class_label in np.unique(y):
            class_mask = y == class_label
            if np.any(class_mask):
                class_acc = accuracy_score(y[class_mask], pred[class_mask])
                class_accuracies[str(class_label)] = class_acc
        
        additional_analysis['model_strengths'][name] = {
            'overall_accuracy': accuracy_score(y, pred),
            'class_accuracies': class_accuracies
        }
    
    # Combine analyses
    analysis.update(additional_analysis)
    
    return analysis

def create_ensemble_from_models(models: List[Any], 
                               ensemble_method: str = 'voting',
                               **kwargs) -> FinancialEnsembleClassifier:
    """Create ensemble from pre-trained models"""
    
    ensemble = FinancialEnsembleClassifier(
        name=f'custom_{ensemble_method}_ensemble',
        ensemble_method=ensemble_method,
        **kwargs
    )
    
    # Set the base models directly
    ensemble.base_models_ = models
    
    return ensemble
