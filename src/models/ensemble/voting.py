# ============================================
# StockPredictionPro - src/models/ensemble/voting.py
# Advanced voting ensemble methods for financial prediction with consensus-based learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
from collections import defaultdict, Counter
import warnings

# Core ML imports
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, roc_curve, log_loss
)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.calibration import CalibratedClassifierCV

# Import our model factory functions
from ..classification.gradient_boosting import create_gradient_boosting_classifier
from ..classification.random_forest import create_random_forest_classifier
from ..classification.svm import create_svm_classifier
from ..classification.logistic import create_logistic_classifier
from ..classification.naive_bayes import create_naive_bayes_classifier
from ..classification.knn import create_knn_classifier
from ..classification.neural_network import create_neural_network_classifier

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it

logger = get_logger('models.ensemble.voting')

# ============================================
# Advanced Voting Strategies
# ============================================

class WeightedVotingStrategy:
    """Advanced weighted voting with multiple weighting schemes"""
    
    def __init__(self, weighting_method: str = 'performance', 
                 cv_folds: int = 5, weight_decay: float = 0.0):
        self.weighting_method = weighting_method
        self.cv_folds = cv_folds
        self.weight_decay = weight_decay
        self.weights_ = None
        self.weight_history_ = []
        
    def calculate_weights(self, models: List[Any], X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate optimal weights for models"""
        
        if self.weighting_method == 'performance':
            return self._performance_weights(models, X, y)
        elif self.weighting_method == 'diversity':
            return self._diversity_weights(models, X, y)
        elif self.weighting_method == 'confidence':
            return self._confidence_weights(models, X, y)
        elif self.weighting_method == 'accuracy_diversity':
            return self._accuracy_diversity_weights(models, X, y)
        elif self.weighting_method == 'dynamic':
            return self._dynamic_weights(models, X, y)
        else:
            # Equal weights
            return np.ones(len(models)) / len(models)
    
    def _performance_weights(self, models: List[Any], X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate weights based on cross-validation performance"""
        
        scores = []
        
        for model in models:
            try:
                cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='accuracy')
                scores.append(np.mean(cv_scores))
            except Exception as e:
                logger.warning(f"Error evaluating model {type(model).__name__}: {e}")
                scores.append(0.5)  # Default score
        
        scores = np.array(scores)
        
        # Apply softmax to convert scores to weights
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        weights = exp_scores / np.sum(exp_scores)
        
        return weights
    
    def _diversity_weights(self, models: List[Any], X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate weights based on prediction diversity"""
        
        # Get predictions from all models
        predictions = []
        for model in models:
            try:
                model_clone = clone(model)
                model_clone.fit(X, y)
                pred = model_clone.predict(X)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Error getting predictions from {type(model).__name__}: {e}")
                predictions.append(np.random.choice(np.unique(y), size=len(y)))
        
        # Calculate diversity scores (average disagreement with other models)
        diversity_scores = []
        for i, pred_i in enumerate(predictions):
            disagreements = []
            for j, pred_j in enumerate(predictions):
                if i != j:
                    disagreement = np.mean(pred_i != pred_j)
                    disagreements.append(disagreement)
            diversity_scores.append(np.mean(disagreements))
        
        diversity_scores = np.array(diversity_scores)
        
        # Normalize to weights (higher diversity = higher weight)
        if np.sum(diversity_scores) > 0:
            weights = diversity_scores / np.sum(diversity_scores)
        else:
            weights = np.ones(len(models)) / len(models)
        
        return weights
    
    def _confidence_weights(self, models: List[Any], X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate weights based on prediction confidence"""
        
        confidences = []
        
        for model in models:
            try:
                model_clone = clone(model)
                model_clone.fit(X, y)
                
                if hasattr(model_clone, 'predict_proba'):
                    proba = model_clone.predict_proba(X)
                    confidence = np.mean(np.max(proba, axis=1))
                elif hasattr(model_clone, 'decision_function'):
                    decision = model_clone.decision_function(X)
                    if decision.ndim > 1:
                        confidence = np.mean(np.max(np.abs(decision), axis=1))
                    else:
                        confidence = np.mean(np.abs(decision))
                else:
                    confidence = 0.5  # Default confidence
                    
                confidences.append(confidence)
                
            except Exception as e:
                logger.warning(f"Error calculating confidence for {type(model).__name__}: {e}")
                confidences.append(0.5)
        
        confidences = np.array(confidences)
        
        # Normalize to weights
        if np.sum(confidences) > 0:
            weights = confidences / np.sum(confidences)
        else:
            weights = np.ones(len(models)) / len(models)
        
        return weights
    
    def _accuracy_diversity_weights(self, models: List[Any], X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Combine accuracy and diversity for balanced weighting"""
        
        performance_weights = self._performance_weights(models, X, y)
        diversity_weights = self._diversity_weights(models, X, y)
        
        # Combine with equal importance
        combined_weights = 0.7 * performance_weights + 0.3 * diversity_weights
        
        # Normalize
        return combined_weights / np.sum(combined_weights)
    
    def _dynamic_weights(self, models: List[Any], X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Dynamic weights that adapt based on recent performance"""
        
        # Start with performance-based weights
        current_weights = self._performance_weights(models, X, y)
        
        # Apply weight decay if we have history
        if self.weight_history_ and self.weight_decay > 0:
            previous_weights = self.weight_history_[-1]
            # Exponential moving average
            current_weights = (1 - self.weight_decay) * current_weights + \
                            self.weight_decay * previous_weights
        
        # Store in history
        self.weight_history_.append(current_weights.copy())
        
        # Maintain reasonable history size
        if len(self.weight_history_) > 10:
            self.weight_history_ = self.weight_history_[-10:]
        
        return current_weights

class ConsensusAnalyzer:
    """Analyze voting consensus and agreement patterns"""
    
    def __init__(self):
        self.consensus_history_ = []
        self.agreement_matrix_ = None
        self.model_reliability_ = {}
        
    def analyze_predictions(self, predictions: List[np.ndarray], 
                          true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Comprehensive consensus analysis"""
        
        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        n_models, n_samples = predictions.shape
        
        analysis = {
            'n_models': n_models,
            'n_samples': n_samples,
            'consensus_metrics': {},
            'agreement_analysis': {},
            'reliability_analysis': {}
        }
        
        # Calculate consensus strength for each prediction
        consensus_strengths = []
        for i in range(n_samples):
            sample_predictions = predictions[:, i]
            unique_preds, counts = np.unique(sample_predictions, return_counts=True)
            max_agreement = np.max(counts) / n_models
            consensus_strengths.append(max_agreement)
        
        consensus_strengths = np.array(consensus_strengths)
        
        # Consensus metrics
        analysis['consensus_metrics'] = {
            'mean_consensus': float(np.mean(consensus_strengths)),
            'std_consensus': float(np.std(consensus_strengths)),
            'min_consensus': float(np.min(consensus_strengths)),
            'max_consensus': float(np.max(consensus_strengths)),
            'high_consensus_ratio': float(np.mean(consensus_strengths > 0.8)),
            'low_consensus_ratio': float(np.mean(consensus_strengths < 0.6)),
            'unanimous_ratio': float(np.mean(consensus_strengths == 1.0))
        }
        
        # Pairwise agreement analysis
        agreement_matrix = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    agreement = np.mean(predictions[i] == predictions[j])
                    agreement_matrix[i, j] = agreement
        
        self.agreement_matrix_ = agreement_matrix
        
        # Agreement statistics
        upper_triangle = agreement_matrix[np.triu_indices_from(agreement_matrix, k=1)]
        analysis['agreement_analysis'] = {
            'mean_pairwise_agreement': float(np.mean(upper_triangle)),
            'std_pairwise_agreement': float(np.std(upper_triangle)),
            'min_agreement': float(np.min(upper_triangle)),
            'max_agreement': float(np.max(upper_triangle)),
            'agreement_matrix': agreement_matrix.tolist()
        }
        
        # Model reliability analysis (if true labels provided)
        if true_labels is not None:
            model_accuracies = []
            for i in range(n_models):
                accuracy = np.mean(predictions[i] == true_labels)
                model_accuracies.append(accuracy)
            
            # Reliability metrics
            analysis['reliability_analysis'] = {
                'individual_accuracies': model_accuracies,
                'accuracy_variance': float(np.var(model_accuracies)),
                'best_model_accuracy': float(np.max(model_accuracies)),
                'worst_model_accuracy': float(np.min(model_accuracies)),
                'accuracy_range': float(np.max(model_accuracies) - np.min(model_accuracies))
            }
            
            # Correlation between consensus and accuracy
            majority_predictions = []
            for i in range(n_samples):
                sample_preds = predictions[:, i]
                majority_pred = Counter(sample_preds).most_common(1)[0][0]
                majority_predictions.append(majority_pred)
            
            majority_predictions = np.array(majority_predictions)
            ensemble_accuracy = np.mean(majority_predictions == true_labels)
            
            analysis['ensemble_performance'] = {
                'ensemble_accuracy': float(ensemble_accuracy),
                'consensus_accuracy_correlation': float(
                    np.corrcoef(consensus_strengths, 
                               (majority_predictions == true_labels).astype(float))[0, 1]
                    if len(np.unique(majority_predictions == true_labels)) > 1 else 0.0
                )
            }
        
        return analysis
    
    def get_disagreement_cases(self, predictions: List[np.ndarray], 
                              indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze cases where models disagree most"""
        
        predictions = np.array(predictions)
        n_models, n_samples = predictions.shape
        
        if indices is None:
            indices = np.arange(n_samples)
        
        # Calculate disagreement for each sample
        disagreements = []
        for i in range(n_samples):
            sample_preds = predictions[:, i]
            unique_preds = len(np.unique(sample_preds))
            disagreement_score = (unique_preds - 1) / (n_models - 1)  # Normalize to [0, 1]
            disagreements.append(disagreement_score)
        
        disagreements = np.array(disagreements)
        
        # Find most disagreement cases
        most_disagreement_idx = np.argsort(disagreements)[-10:]  # Top 10 most disagreement
        least_disagreement_idx = np.argsort(disagreements)[:10]   # Top 10 least disagreement
        
        return {
            'disagreement_scores': disagreements.tolist(),
            'most_disagreement_indices': indices[most_disagreement_idx].tolist(),
            'least_disagreement_indices': indices[least_disagreement_idx].tolist(),
            'mean_disagreement': float(np.mean(disagreements)),
            'disagreement_distribution': {
                'high_disagreement': float(np.mean(disagreements > 0.7)),
                'moderate_disagreement': float(np.mean((disagreements > 0.3) & (disagreements <= 0.7))),
                'low_disagreement': float(np.mean(disagreements <= 0.3))
            }
        }

class AdaptiveVotingEnsemble(BaseEstimator):
    """Adaptive voting ensemble with dynamic weight adjustment"""
    
    def __init__(self, base_models: List[Any], voting_strategy: str = 'soft',
                 weighting_method: str = 'performance', adaptation_rate: float = 0.1,
                 min_consensus: float = 0.6):
        self.base_models = base_models
        self.voting_strategy = voting_strategy
        self.weighting_method = weighting_method
        self.adaptation_rate = adaptation_rate
        self.min_consensus = min_consensus
        
        self.fitted_models_ = []
        self.voting_weights_ = None
        self.weight_strategy_ = None
        self.consensus_analyzer_ = ConsensusAnalyzer()
        self.performance_history_ = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all base models and calculate initial weights"""
        
        # Fit all base models
        self.fitted_models_ = []
        for model in self.base_models:
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            self.fitted_models_.append(fitted_model)
        
        # Initialize weight strategy
        self.weight_strategy_ = WeightedVotingStrategy(
            weighting_method=self.weighting_method,
            cv_folds=3
        )
        
        # Calculate initial weights
        self.voting_weights_ = self.weight_strategy_.calculate_weights(
            self.fitted_models_, X, y
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted voting predictions"""
        
        if not self.fitted_models_:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from all models
        predictions = []
        probabilities = []
        
        for model in self.fitted_models_:
            pred = model.predict(X)
            predictions.append(pred)
            
            if hasattr(model, 'predict_proba') and self.voting_strategy == 'soft':
                proba = model.predict_proba(X)
                probabilities.append(proba)
        
        predictions = np.array(predictions)
        
        if self.voting_strategy == 'soft' and probabilities:
            # Weighted soft voting
            probabilities = np.array(probabilities)  # Shape: (n_models, n_samples, n_classes)
            
            # Weight the probabilities
            weighted_probabilities = np.zeros_like(probabilities[0])
            for i, (proba, weight) in enumerate(zip(probabilities, self.voting_weights_)):
                weighted_probabilities += weight * proba
            
            # Make final predictions
            final_predictions = np.argmax(weighted_probabilities, axis=1)
        else:
            # Weighted hard voting
            final_predictions = []
            
            for sample_idx in range(predictions.shape[1]):
                sample_predictions = predictions[:, sample_idx]
                
                # Count weighted votes
                vote_counts = defaultdict(float)
                for pred, weight in zip(sample_predictions, self.voting_weights_):
                    vote_counts[pred] += weight
                
                # Select prediction with highest weighted vote
                final_pred = max(vote_counts.keys(), key=lambda k: vote_counts[k])
                final_predictions.append(final_pred)
            
            final_predictions = np.array(final_predictions)
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make weighted probability predictions"""
        
        if not self.fitted_models_:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.voting_strategy != 'soft':
            raise ValueError("Probability prediction requires soft voting")
        
        # Get probabilities from all models
        probabilities = []
        for model in self.fitted_models_:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probabilities.append(proba)
            else:
                raise ValueError("All models must support predict_proba for soft voting")
        
        probabilities = np.array(probabilities)
        
        # Weighted average of probabilities
        weighted_probabilities = np.zeros_like(probabilities[0])
        for i, (proba, weight) in enumerate(zip(probabilities, self.voting_weights_)):
            weighted_probabilities += weight * proba
        
        return weighted_probabilities
    
    def update_weights(self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray):
        """Update voting weights based on recent performance"""
        
        # Calculate individual model performance
        individual_accuracies = []
        for model in self.fitted_models_:
            model_pred = model.predict(X)
            accuracy = np.mean(model_pred == y)
            individual_accuracies.append(accuracy)
        
        individual_accuracies = np.array(individual_accuracies)
        
        # Calculate new weights based on performance
        new_weights = individual_accuracies / np.sum(individual_accuracies)
        
        # Adaptive weight update
        self.voting_weights_ = (1 - self.adaptation_rate) * self.voting_weights_ + \
                              self.adaptation_rate * new_weights
        
        # Store performance history
        ensemble_accuracy = np.mean(predictions == y)
        self.performance_history_.append({
            'ensemble_accuracy': ensemble_accuracy,
            'individual_accuracies': individual_accuracies.tolist(),
            'weights': self.voting_weights_.copy()
        })

# ============================================
# Main Voting Ensemble Model
# ============================================

class FinancialVotingEnsemble(BaseEstimator):
    """
    Advanced voting ensemble for financial prediction with consensus analysis
    
    Features:
    - Multiple voting strategies: hard, soft, weighted, adaptive
    - Comprehensive consensus analysis and disagreement detection
    - Dynamic weight adjustment based on performance
    - Financial domain optimizations (confidence-based voting, sector-aware weights)
    - Advanced voting diagnostics and visualization
    """
    
    def __init__(self,
                 name: str = "voting_ensemble",
                 base_models: Optional[List[Any]] = None,
                 voting_strategy: str = 'soft',
                 weighting_method: str = 'performance',
                 weights: Optional[List[float]] = None,
                 adaptive_weights: bool = True,
                 min_consensus_threshold: float = 0.6,
                 confidence_threshold: float = 0.8,
                 calibrate_probabilities: bool = True,
                 task_type: str = 'classification',
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Financial Voting Ensemble
        
        Args:
            name: Model name
            base_models: List of base models for voting
            voting_strategy: Voting strategy ('hard', 'soft', 'weighted', 'adaptive')
            weighting_method: Method for calculating weights ('performance', 'diversity', 'confidence')
            weights: Manual weights for models (if None, calculated automatically)
            adaptive_weights: Whether to adapt weights during prediction
            min_consensus_threshold: Minimum consensus required for confident prediction
            confidence_threshold: Threshold for high-confidence predictions
            calibrate_probabilities: Whether to calibrate probabilities
            task_type: Type of task ('classification' or 'regression')
            random_state: Random seed
        """
        self.name = name
        self.base_models = base_models or self._create_default_models()
        self.voting_strategy = voting_strategy
        self.weighting_method = weighting_method
        self.weights = weights
        self.adaptive_weights = adaptive_weights
        self.min_consensus_threshold = min_consensus_threshold
        self.confidence_threshold = confidence_threshold
        self.calibrate_probabilities = calibrate_probabilities
        self.task_type = task_type
        self.random_state = random_state
        
        # Fitted components
        self.fitted_models_ = []
        self.voting_ensemble_ = None
        self.adaptive_ensemble_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.calibrated_ensemble_ = None
        self.voting_weights_ = None
        self.consensus_analyzer_ = ConsensusAnalyzer()
        self.voting_analysis_ = None
        self.is_fitted_ = False
        
        # Performance tracking
        self.consensus_history_ = []
        self.weight_history_ = []
        self.performance_metrics_ = {}
        
        logger.info(f"Initialized {voting_strategy} voting ensemble: {self.name}")
    
    def _create_default_models(self) -> List[Any]:
        """Create default set of diverse base models"""
        try:
            models = [
                create_gradient_boosting_classifier(performance_preset='balanced'),
                create_random_forest_classifier(performance_preset='balanced'),
                create_svm_classifier(performance_preset='balanced'),
                create_logistic_classifier(performance_preset='balanced')
            ]
            
            # Add additional models if possible
            try:
                models.append(create_naive_bayes_classifier(performance_preset='balanced'))
                models.append(create_knn_classifier(performance_preset='balanced'))
            except:
                logger.info("Some models not available, using core models")
            
            # Add neural network if possible
            try:
                nn_model = create_neural_network_classifier(
                    architecture='balanced',
                    epochs=100,
                    verbose=0
                )
                models.append(nn_model)
            except:
                logger.info("Neural network not available")
            
            return models
            
        except Exception as e:
            logger.error(f"Error creating default models: {e}")
            return []
    
    def _preprocess_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Preprocess features with scaling"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.scaler_ is None:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = self.scaler_.transform(X)
        
        return X_scaled
    
    def _preprocess_targets(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Preprocess targets with encoding for classification"""
        if isinstance(y, pd.Series):
            y = y.values
        
        if self.task_type == 'classification':
            if self.label_encoder_ is None:
                self.label_encoder_ = LabelEncoder()
                y_encoded = self.label_encoder_.fit_transform(y)
                self.classes_ = self.label_encoder_.classes_
            else:
                y_encoded = self.label_encoder_.transform(y)
            return y_encoded
        else:
            return y.astype(float)
    
    def _create_voting_ensemble(self) -> Union[VotingClassifier, VotingRegressor]:
        """Create the appropriate voting ensemble"""
        
        # Prepare estimators list
        estimators = [(f"model_{i}", model) for i, model in enumerate(self.fitted_models_)]
        
        if self.task_type == 'classification':
            voting_type = 'soft' if self.voting_strategy in ['soft', 'weighted', 'adaptive'] else 'hard'
            return VotingClassifier(
                estimators=estimators,
                voting=voting_type,
                weights=self.voting_weights_
            )
        else:
            return VotingRegressor(
                estimators=estimators,
                weights=self.voting_weights_
            )
    
    def _calculate_voting_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate optimal voting weights"""
        
        if self.weights is not None:
            # Use manual weights
            weights = np.array(self.weights)
            return weights / np.sum(weights)  # Normalize
        
        # Calculate weights using strategy
        weight_strategy = WeightedVotingStrategy(
            weighting_method=self.weighting_method,
            cv_folds=3
        )
        
        weights = weight_strategy.calculate_weights(self.fitted_models_, X, y)
        return weights
    
    def _analyze_voting_consensus(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze voting consensus and model agreement"""
        
        # Get predictions from all models
        all_predictions = []
        for model in self.fitted_models_:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        # Perform consensus analysis
        consensus_analysis = self.consensus_analyzer_.analyze_predictions(
            all_predictions, y
        )
        
        # Add voting-specific analysis
        voting_analysis = {
            'voting_strategy': self.voting_strategy,
            'weighting_method': self.weighting_method,
            'n_base_models': len(self.fitted_models_),
            'voting_weights': self.voting_weights_.tolist() if self.voting_weights_ is not None else None,
            'consensus_analysis': consensus_analysis
        }
        
        # Model contribution analysis
        if self.voting_weights_ is not None:
            model_contributions = []
            for i, (model, weight) in enumerate(zip(self.fitted_models_, self.voting_weights_)):
                contribution = {
                    'model_index': i,
                    'model_type': type(model).__name__,
                    'voting_weight': float(weight),
                    'contribution_level': 'High' if weight > np.mean(self.voting_weights_) + np.std(self.voting_weights_) else
                                        'Low' if weight < np.mean(self.voting_weights_) - np.std(self.voting_weights_) else 'Medium'
                }
                model_contributions.append(contribution)
            
            voting_analysis['model_contributions'] = model_contributions
        
        # Disagreement analysis
        disagreement_analysis = self.consensus_analyzer_.get_disagreement_cases(all_predictions)
        voting_analysis['disagreement_analysis'] = disagreement_analysis
        
        return voting_analysis
    
    def _calibrate_probabilities(self, X: np.ndarray, y: np.ndarray):
        """Calibrate voting ensemble probabilities"""
        if self.task_type != 'classification' or not self.calibrate_probabilities:
            return
        
        try:
            self.calibrated_ensemble_ = CalibratedClassifierCV(
                base_estimator=self.voting_ensemble_,
                method='isotonic',
                cv=3
            )
            self.calibrated_ensemble_.fit(X, y)
            logger.debug("Calibrated voting ensemble probabilities")
        except Exception as e:
            logger.warning(f"Could not calibrate probabilities: {e}")
            self.calibrated_ensemble_ = None
    
    @time_it("voting_fit", include_args=True)
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], **kwargs):
        """Fit the voting ensemble"""
        
        logger.info(f"Fitting voting ensemble with {len(self.base_models)} base models")
        
        try:
            # Preprocess data
            X_processed = self._preprocess_features(X)
            y_processed = self._preprocess_targets(y)
            
            # Fit all base models
            self.fitted_models_ = []
            for i, model in enumerate(self.base_models):
                logger.info(f"Fitting base model {i+1}/{len(self.base_models)}: {type(model).__name__}")
                try:
                    fitted_model = clone(model)
                    fitted_model.fit(X_processed, y_processed)
                    self.fitted_models_.append(fitted_model)
                except Exception as e:
                    logger.warning(f"Failed to fit model {type(model).__name__}: {e}")
            
            if not self.fitted_models_:
                raise ValueError("No base models were successfully fitted")
            
            # Calculate voting weights
            self.voting_weights_ = self._calculate_voting_weights(X_processed, y_processed)
            
            # Create main voting ensemble
            self.voting_ensemble_ = self._create_voting_ensemble()
            self.voting_ensemble_.fit(X_processed, y_processed)
            
            # Create adaptive ensemble if requested
            if self.adaptive_weights:
                self.adaptive_ensemble_ = AdaptiveVotingEnsemble(
                    base_models=self.fitted_models_,
                    voting_strategy=self.voting_strategy,
                    weighting_method=self.weighting_method
                )
                self.adaptive_ensemble_.fit(X_processed, y_processed)
            
            # Analyze voting consensus
            self.voting_analysis_ = self._analyze_voting_consensus(X_processed, y_processed)
            
            # Calibrate probabilities
            if self.task_type == 'classification':
                self._calibrate_probabilities(X_processed, y_processed)
            
            # Calculate performance metrics
            train_predictions = self.predict(X)
            if self.task_type == 'classification':
                train_accuracy = accuracy_score(y, train_predictions)
                self.performance_metrics_['train_accuracy'] = train_accuracy
            else:
                train_r2 = r2_score(y, train_predictions)
                self.performance_metrics_['train_r2'] = train_r2
            
            self.is_fitted_ = True
            logger.info("Voting ensemble fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"Voting ensemble fitting failed: {e}")
            raise
    
    @time_it("voting_predict", include_args=True)
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make voting predictions"""
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            # Choose ensemble based on adaptive setting
            if self.adaptive_weights and self.adaptive_ensemble_ is not None:
                predictions = self.adaptive_ensemble_.predict(X_processed)
            else:
                predictions = self.voting_ensemble_.predict(X_processed)
            
            # Decode predictions for classification
            if self.task_type == 'classification':
                return self.label_encoder_.inverse_transform(predictions)
            else:
                return predictions
                
        except Exception as e:
            logger.error(f"Voting prediction failed: {e}")
            raise
    
    @time_it("voting_predict_proba", include_args=True)
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make voting probability predictions"""
        
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            # Get probabilities
            if self.calibrated_ensemble_ is not None:
                probabilities = self.calibrated_ensemble_.predict_proba(X_processed)
            elif self.adaptive_weights and self.adaptive_ensemble_ is not None:
                probabilities = self.adaptive_ensemble_.predict_proba(X_processed)
            else:
                probabilities = self.voting_ensemble_.predict_proba(X_processed)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Voting probability prediction failed: {e}")
            raise
    
    def predict_with_consensus(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with consensus analysis"""
        
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # Get predictions from all base models
        all_predictions = []
        for model in self.fitted_models_:
            pred = model.predict(X_processed)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Calculate consensus scores
        consensus_scores = []
        final_predictions = []
        
        for i in range(X_processed.shape[0]):
            sample_predictions = all_predictions[:, i]
            unique_preds, counts = np.unique(sample_predictions, return_counts=True)
            
            # Consensus score (fraction agreeing with majority)
            max_agreement = np.max(counts) / len(self.fitted_models_)
            consensus_scores.append(max_agreement)
            
            # Final prediction (majority vote or weighted)
            if self.voting_weights_ is not None:
                # Weighted voting
                vote_counts = defaultdict(float)
                for pred, weight in zip(sample_predictions, self.voting_weights_):
                    vote_counts[pred] += weight
                final_pred = max(vote_counts.keys(), key=lambda k: vote_counts[k])
            else:
                # Simple majority
                final_pred = unique_preds[np.argmax(counts)]
            
            final_predictions.append(final_pred)
        
        final_predictions = np.array(final_predictions)
        consensus_scores = np.array(consensus_scores)
        
        # High confidence mask
        high_confidence = consensus_scores >= self.confidence_threshold
        
        # Decode predictions for classification
        if self.task_type == 'classification':
            final_predictions = self.label_encoder_.inverse_transform(final_predictions)
        
        return final_predictions, consensus_scores, high_confidence
    
    def get_voting_analysis(self) -> Dict[str, Any]:
        """Get comprehensive voting analysis"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to get voting analysis")
        
        return self.voting_analysis_.copy() if self.voting_analysis_ else {}
    
    def get_model_contributions(self) -> Dict[str, float]:
        """Get individual model contributions to ensemble"""
        if not self.is_fitted_ or self.voting_weights_ is None:
            return {}
        
        contributions = {}
        for i, (model, weight) in enumerate(zip(self.fitted_models_, self.voting_weights_)):
            model_name = f"{type(model).__name__}_{i}"
            contributions[model_name] = float(weight)
        
        return contributions
    
    def plot_voting_analysis(self) -> Any:
        """Plot comprehensive voting analysis"""
        if not self.voting_analysis_:
            logger.warning("Voting analysis not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            analysis = self.voting_analysis_
            consensus_analysis = analysis['consensus_analysis']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Model weights/contributions
            if 'model_contributions' in analysis:
                contributions = analysis['model_contributions']
                model_names = [contrib['model_type'] for contrib in contributions]
                weights = [contrib['voting_weight'] for contrib in contributions]
                
                bars = axes[0, 0].bar(range(len(model_names)), weights, alpha=0.7, color='steelblue')
                axes[0, 0].set_title('Model Voting Weights')
                axes[0, 0].set_xticks(range(len(model_names)))
                axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
                axes[0, 0].set_ylabel('Weight')
                
                # Add values on bars
                for bar, weight in zip(bars, weights):
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, weight + 0.005,
                                   f'{weight:.3f}', ha='center', va='bottom')
            else:
                axes[0, 0].text(0.5, 0.5, 'Equal Weights Used', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Model Voting Weights')
            
            # Consensus metrics
            consensus_metrics = consensus_analysis['consensus_metrics']
            metric_names = ['Mean', 'Std', 'High Ratio', 'Low Ratio', 'Unanimous']
            metric_values = [
                consensus_metrics['mean_consensus'],
                consensus_metrics['std_consensus'],
                consensus_metrics['high_consensus_ratio'],
                consensus_metrics['low_consensus_ratio'],
                consensus_metrics['unanimous_ratio']
            ]
            
            bars = axes[0, 1].bar(metric_names, metric_values, alpha=0.7, color='orange')
            axes[0, 1].set_title('Consensus Metrics')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add values on bars
            for bar, value in zip(bars, metric_values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, value + max(metric_values) * 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Agreement matrix heatmap
            if 'agreement_analysis' in consensus_analysis:
                agreement_matrix = np.array(consensus_analysis['agreement_analysis']['agreement_matrix'])
                n_models = len(self.fitted_models_)
                model_labels = [f"M{i}" for i in range(n_models)]
                
                sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                           xticklabels=model_labels, yticklabels=model_labels, ax=axes[0, 2])
                axes[0, 2].set_title('Model Agreement Matrix')
            
            # Individual model performance (if available)
            if 'reliability_analysis' in consensus_analysis:
                reliability = consensus_analysis['reliability_analysis']
                individual_accs = reliability['individual_accuracies']
                model_names = [f"Model {i}" for i in range(len(individual_accs))]
                
                bars = axes[1, 0].bar(range(len(model_names)), individual_accs, alpha=0.7, color='green')
                
                # Add ensemble performance line
                if 'ensemble_performance' in consensus_analysis:
                    ensemble_acc = consensus_analysis['ensemble_performance']['ensemble_accuracy']
                    axes[1, 0].axhline(y=ensemble_acc, color='red', linestyle='--', 
                                      label=f'Ensemble: {ensemble_acc:.3f}')
                    axes[1, 0].legend()
                
                axes[1, 0].set_title('Individual Model Performance')
                axes[1, 0].set_xticks(range(len(model_names)))
                axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
                axes[1, 0].set_ylabel('Accuracy')
                
                # Add values on bars
                for bar, acc in zip(bars, individual_accs):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, acc + 0.005,
                                   f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
            else:
                axes[1, 0].text(0.5, 0.5, 'Performance analysis\nnot available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Individual Model Performance')
            
            # Disagreement distribution
            if 'disagreement_analysis' in analysis:
                disagreement = analysis['disagreement_analysis']
                disagree_dist = disagreement['disagreement_distribution']
                
                categories = ['High', 'Moderate', 'Low']
                values = [
                    disagree_dist['high_disagreement'],
                    disagree_dist['moderate_disagreement'],
                    disagree_dist['low_disagreement']
                ]
                colors = ['red', 'orange', 'green']
                
                wedges, texts, autotexts = axes[1, 1].pie(values, labels=categories, autopct='%1.1f%%',
                                                         colors=colors, startangle=90)
                axes[1, 1].set_title('Disagreement Distribution')
            
            # Voting summary
            summary_text = f"Voting Strategy: {analysis['voting_strategy'].title()}\n"
            summary_text += f"Weighting Method: {analysis['weighting_method'].title()}\n"
            summary_text += f"Base Models: {analysis['n_base_models']}\n\n"
            
            if 'consensus_analysis' in analysis:
                consensus = analysis['consensus_analysis']['consensus_metrics']
                summary_text += f"Mean Consensus: {consensus['mean_consensus']:.3f}\n"
                summary_text += f"High Consensus: {consensus['high_consensus_ratio']:.1%}\n"
                summary_text += f"Unanimous: {consensus['unanimous_ratio']:.1%}\n"
            
            if 'ensemble_performance' in consensus_analysis:
                ensemble_perf = consensus_analysis['ensemble_performance']
                summary_text += f"\nEnsemble Accuracy: {ensemble_perf['ensemble_accuracy']:.3f}\n"
            
            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            axes[1, 2].set_title('Voting Summary')
            axes[1, 2].axis('off')
            
            plt.suptitle(f'Voting Ensemble Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = {
            'model_name': self.name,
            'model_family': 'Voting Ensemble',
            'task_type': self.task_type,
            'voting_strategy': self.voting_strategy,
            'weighting_method': self.weighting_method,
            'n_base_models': len(self.base_models),
            'adaptive_weights': self.adaptive_weights,
            'min_consensus_threshold': self.min_consensus_threshold,
            'confidence_threshold': self.confidence_threshold,
            'probability_calibration': self.calibrate_probabilities,
            'is_fitted': self.is_fitted_
        }
        
        if self.is_fitted_:
            summary.update({
                'base_model_types': [type(model).__name__ for model in self.fitted_models_],
                'voting_weights': self.voting_weights_.tolist() if self.voting_weights_ is not None else None,
                'performance_metrics': self.performance_metrics_,
                'voting_analysis': self.voting_analysis_
            })
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_voting_classifier(voting_strategy: str = 'soft',
                            base_models: Optional[List[str]] = None,
                            weighting_method: str = 'performance',
                            complexity: str = 'balanced',
                            **kwargs) -> FinancialVotingEnsemble:
    """Create voting classifier with different strategies"""
    
    # Default base models by complexity
    complexity_models = {
        'simple': ['gradient_boosting', 'random_forest', 'logistic'],
        'balanced': ['gradient_boosting', 'random_forest', 'svm', 'logistic', 'naive_bayes'],
        'comprehensive': ['gradient_boosting', 'random_forest', 'svm', 'logistic', 
                         'naive_bayes', 'knn', 'neural_network']
    }
    
    if base_models is None:
        base_models = complexity_models.get(complexity, complexity_models['balanced'])
    
    # Create base model instances
    model_instances = []
    for model_name in base_models:
        try:
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
                model = create_neural_network_classifier(
                    architecture='balanced', epochs=100, verbose=0
                )
            else:
                continue
            
            model_instances.append(model)
        except Exception as e:
            logger.warning(f"Could not create {model_name}: {e}")
    
    config = {
        'name': f'{voting_strategy}_voting_{complexity}',
        'base_models': model_instances,
        'voting_strategy': voting_strategy,
        'weighting_method': weighting_method,
        'task_type': 'classification',
        'calibrate_probabilities': True,
        'random_state': 42
    }
    
    config.update(kwargs)
    
    return FinancialVotingEnsemble(**config)

def create_voting_regressor(voting_strategy: str = 'weighted',
                          base_models: Optional[List[str]] = None,
                          complexity: str = 'balanced',
                          **kwargs) -> FinancialVotingEnsemble:
    """Create voting regressor"""
    
    classifier = create_voting_classifier(
        voting_strategy=voting_strategy,
        base_models=base_models,
        complexity=complexity,
        **kwargs
    )
    
    # Convert to regression  
    classifier.task_type = 'regression'
    classifier.calibrate_probabilities = False
    classifier.name = classifier.name.replace('voting', 'voting_regressor')
    
    return classifier

def create_soft_voting(**kwargs) -> FinancialVotingEnsemble:
    """Create soft voting ensemble"""
    return create_voting_classifier(
        voting_strategy='soft',
        name='soft_voting_ensemble',
        **kwargs
    )

def create_hard_voting(**kwargs) -> FinancialVotingEnsemble:
    """Create hard voting ensemble"""
    return create_voting_classifier(
        voting_strategy='hard',
        name='hard_voting_ensemble',
        **kwargs
    )

def create_weighted_voting(**kwargs) -> FinancialVotingEnsemble:
    """Create weighted voting ensemble"""
    return create_voting_classifier(
        voting_strategy='weighted',
        weighting_method='performance',
        name='weighted_voting_ensemble',
        **kwargs
    )

def create_adaptive_voting(**kwargs) -> FinancialVotingEnsemble:
    """Create adaptive voting ensemble"""
    return create_voting_classifier(
        voting_strategy='adaptive',
        adaptive_weights=True,
        weighting_method='dynamic',
        name='adaptive_voting_ensemble',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def compare_voting_strategies(X: Union[pd.DataFrame, np.ndarray],
                            y: Union[pd.Series, np.ndarray],
                            strategies: List[str] = ['hard', 'soft', 'weighted', 'adaptive'],
                            task_type: str = 'classification') -> Dict[str, Any]:
    """Compare different voting strategies"""
    
    logger.info(f"Comparing voting strategies: {strategies}")
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"Evaluating {strategy} voting")
        
        try:
            if task_type == 'classification':
                voter = create_voting_classifier(voting_strategy=strategy)
            else:
                voter = create_voting_regressor(voting_strategy=strategy)
            
            # Fit and evaluate
            voter.fit(X, y)
            
            # Get voting analysis
            analysis = voter.get_voting_analysis()
            
            # Get performance metrics
            predictions = voter.predict(X)
            if task_type == 'classification':
                score = accuracy_score(y, predictions)
                
                # Get consensus information
                if hasattr(voter, 'predict_with_consensus'):
                    _, consensus_scores, high_confidence = voter.predict_with_consensus(X)
                    consensus_info = {
                        'mean_consensus': float(np.mean(consensus_scores)),
                        'high_confidence_ratio': float(np.mean(high_confidence))
                    }
                else:
                    consensus_info = {}
            else:
                score = r2_score(y, predictions)
                consensus_info = {}
            
            results[strategy] = {
                'score': score,
                'voting_analysis': analysis,
                'consensus_info': consensus_info,
                'model': voter
            }
            
        except Exception as e:
            logger.warning(f"Error with {strategy} voting: {e}")
            results[strategy] = {'error': str(e)}
    
    # Add comparison summary
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_strategy = max(valid_results.keys(), key=lambda k: valid_results[k]['score'])
        
        # Find best consensus if available
        consensus_strategies = {k: v for k, v in valid_results.items() 
                             if v['consensus_info'] and 'mean_consensus' in v['consensus_info']}
        
        best_consensus = None
        if consensus_strategies:
            best_consensus = max(consensus_strategies.keys(), 
                               key=lambda k: consensus_strategies[k]['consensus_info']['mean_consensus'])
        
        results['comparison'] = {
            'best_score': best_strategy,
            'best_consensus': best_consensus,
            'strategy_rankings': sorted(valid_results.keys(), 
                                      key=lambda k: valid_results[k]['score'], reverse=True)
        }
    
    logger.info(f"Voting comparison complete. Best strategy: {results['comparison']['best_score']}")
    
    return results

def analyze_voting_consensus(models: List[Any], X: Union[pd.DataFrame, np.ndarray],
                           y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """Analyze consensus among a set of models"""
    
    logger.info(f"Analyzing voting consensus for {len(models)} models")
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Fit all models and get predictions
    predictions = []
    model_names = []
    
    for i, model in enumerate(models):
        try:
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            pred = fitted_model.predict(X)
            predictions.append(pred)
            model_names.append(f"{type(model).__name__}_{i}")
        except Exception as e:
            logger.warning(f"Error with model {i}: {e}")
    
    if len(predictions) < 2:
        logger.warning("Need at least 2 models for consensus analysis")
        return {}
    
    # Create consensus analyzer
    analyzer = ConsensusAnalyzer()
    analysis = analyzer.analyze_predictions(predictions, y)
    
    # Add model names
    analysis['model_names'] = model_names
    analysis['n_models_analyzed'] = len(predictions)
    
    # Additional analysis
    additional_analysis = {
        'voting_scenarios': {},
        'consensus_quality': {},
        'model_diversity': {}
    }
    
    # Analyze different voting scenarios
    predictions_array = np.array(predictions)
    
    # Majority voting results
    majority_predictions = []
    for i in range(predictions_array.shape[1]):
        sample_preds = predictions_array[:, i]
        majority_pred = Counter(sample_preds).most_common(1)[0][0]
        majority_predictions.append(majority_pred)
    
    majority_predictions = np.array(majority_predictions)
    majority_accuracy = accuracy_score(y, majority_predictions)
    
    # Unanimous voting results
    unanimous_predictions = []
    unanimous_mask = []
    for i in range(predictions_array.shape[1]):
        sample_preds = predictions_array[:, i]
        if len(np.unique(sample_preds)) == 1:
            unanimous_predictions.append(sample_preds[0])
            unanimous_mask.append(True)
        else:
            unanimous_predictions.append(majority_predictions[i])  # Fallback
            unanimous_mask.append(False)
    
    unanimous_predictions = np.array(unanimous_predictions)
    unanimous_ratio = np.mean(unanimous_mask)
    unanimous_accuracy = accuracy_score(y[unanimous_mask], 
                                       unanimous_predictions[unanimous_mask]) if np.any(unanimous_mask) else 0.0
    
    additional_analysis['voting_scenarios'] = {
        'majority_voting_accuracy': float(majority_accuracy),
        'unanimous_voting_accuracy': float(unanimous_accuracy),
        'unanimous_prediction_ratio': float(unanimous_ratio)
    }
    
    # Consensus quality metrics
    individual_accuracies = [accuracy_score(y, pred) for pred in predictions]
    best_individual = max(individual_accuracies)
    consensus_improvement = majority_accuracy - best_individual
    
    additional_analysis['consensus_quality'] = {
        'consensus_improvement': float(consensus_improvement),
        'best_individual_accuracy': float(best_individual),
        'consensus_vs_best': 'Better' if consensus_improvement > 0 else 
                           'Same' if consensus_improvement == 0 else 'Worse'
    }
    
    # Model diversity metrics
    pairwise_disagreements = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            disagreement = np.mean(predictions[i] != predictions[j])
            pairwise_disagreements.append(disagreement)
    
    additional_analysis['model_diversity'] = {
        'mean_pairwise_disagreement': float(np.mean(pairwise_disagreements)),
        'diversity_score': float(np.mean(pairwise_disagreements)),
        'diversity_level': 'High' if np.mean(pairwise_disagreements) > 0.3 else
                          'Moderate' if np.mean(pairwise_disagreements) > 0.1 else 'Low'
    }
    
    # Combine analyses
    analysis.update(additional_analysis)
    
    logger.info(f"Consensus analysis complete. Majority accuracy: {majority_accuracy:.4f}")
    
    return analysis
