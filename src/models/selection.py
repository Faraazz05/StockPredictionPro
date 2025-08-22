# ============================================
# StockPredictionPro - src/models/selection.py
# Advanced model selection with automated evaluation, comparison, and ensemble optimization
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
import warnings
from itertools import combinations
from dataclasses import dataclass
from enum import Enum
import json

# Core ML imports
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, TimeSeriesSplit,
    train_test_split, validation_curve, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, make_scorer
)
from sklearn.preprocessing import StandardScaler

# Statistical tests
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon

# Import our modules
from .factory import model_factory, get_available_models, get_model_recommendations
from .persistence import default_registry, ModelMetadata, ModelStatus
from ..utils.exceptions import ModelValidationError, BusinessLogicError
from ..utils.logger import get_logger
from ..utils.timing import Timer, time_it

logger = get_logger('models.selection')

# ============================================
# Selection Criteria and Configuration
# ============================================

class SelectionCriterion(Enum):
    """Model selection criteria"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    R2_SCORE = "r2_score"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    TRAINING_TIME = "training_time"
    PREDICTION_TIME = "prediction_time"
    MODEL_SIZE = "model_size"
    INTERPRETABILITY = "interpretability"
    STABILITY = "stability"
    GENERALIZATION = "generalization"

@dataclass
class SelectionConfig:
    """Configuration for model selection"""
    primary_metric: str = 'accuracy'
    secondary_metrics: List[str] = None
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    time_aware_cv: bool = True
    n_jobs: int = -1
    scoring_weights: Dict[str, float] = None
    min_improvement_threshold: float = 0.01
    statistical_significance_level: float = 0.05
    max_models_to_evaluate: int = 20
    include_ensemble_methods: bool = True
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = []
        if self.scoring_weights is None:
            self.scoring_weights = {self.primary_metric: 1.0}

@dataclass 
class ModelEvaluationResult:
    """Results from model evaluation"""
    model_name: str
    model_instance: BaseEstimator
    cv_scores: Dict[str, List[float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    training_time: float
    prediction_time: float
    model_size: int
    feature_importance: Optional[Dict[str, float]]
    hyperparameters: Dict[str, Any]
    metadata: Dict[str, Any]
    
    @property
    def primary_score(self) -> float:
        """Get primary evaluation score"""
        return self.mean_scores.get('accuracy', 0.0)  # Default to accuracy
    
    def get_score(self, metric: str) -> float:
        """Get score for specific metric"""
        return self.mean_scores.get(metric, 0.0)
    
    def get_weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted score across multiple metrics"""
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in self.mean_scores:
                weighted_score += self.mean_scores[metric] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0

# ============================================
# Model Evaluator
# ============================================

class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics and validation strategies"""
    
    def __init__(self, config: SelectionConfig = None):
        self.config = config or SelectionConfig()
        self.evaluation_cache = {}
        
        logger.info("Initialized ModelEvaluator")
    
    def _create_cv_splitter(self, X: np.ndarray, y: np.ndarray, task_type: str = 'classification'):
        """Create appropriate cross-validation splitter"""
        if self.config.time_aware_cv:
            return TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            if task_type == 'classification':
                return StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            else:
                return KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
    
    def _get_scoring_metrics(self, task_type: str = 'classification') -> Dict[str, Any]:
        """Get scoring metrics for evaluation"""
        if task_type == 'classification':
            metrics = {
                'accuracy': 'accuracy',
                'precision': make_scorer(precision_score, average='weighted', zero_division=0),
                'recall': make_scorer(recall_score, average='weighted', zero_division=0),
                'f1_score': make_scorer(f1_score, average='weighted', zero_division=0)
            }
            
            # Add ROC AUC for binary classification
            try:
                metrics['roc_auc'] = 'roc_auc'
            except:
                pass
                
        else:  # regression
            metrics = {
                'r2_score': 'r2',
                'mae': 'neg_mean_absolute_error',
                'mse': 'neg_mean_squared_error'
            }
        
        return metrics
    
    def _calculate_feature_importance(self, model: BaseEstimator) -> Optional[Dict[str, float]]:
        """Extract feature importance from model if available"""
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = {f'feature_{i}': float(imp) for i, imp in enumerate(model.feature_importances_)}
        elif hasattr(model, 'coef_'):
            # Linear models
            coef = model.coef_
            if coef.ndim > 1:
                coef = coef[0]  # Take first class for multiclass
            importance = {f'feature_{i}': float(abs(c)) for i, c in enumerate(coef)}
        
        return importance
    
    def _estimate_model_size(self, model: BaseEstimator) -> int:
        """Estimate model size in bytes"""
        try:
            import pickle
            return len(pickle.dumps(model))
        except:
            return 0
    
    @time_it("model_evaluation", include_args=True)
    def evaluate_model(self, 
                      model: BaseEstimator,
                      model_name: str,
                      X: np.ndarray,
                      y: np.ndarray,
                      task_type: str = 'classification') -> ModelEvaluationResult:
        """Evaluate a single model comprehensively"""
        
        logger.info(f"Evaluating model: {model_name}")
        
        # Create cache key
        cache_key = f"{model_name}_{hash(str(model.get_params()))}"
        if cache_key in self.evaluation_cache:
            logger.debug(f"Using cached evaluation for {model_name}")
            return self.evaluation_cache[cache_key]
        
        try:
            # Time model training
            train_start = datetime.now()
            model.fit(X, y)
            training_time = (datetime.now() - train_start).total_seconds()
            
            # Time prediction
            pred_start = datetime.now()
            predictions = model.predict(X[:100] if len(X) > 100 else X)  # Sample for timing
            prediction_time = (datetime.now() - pred_start).total_seconds()
            
            # Cross-validation evaluation
            cv_splitter = self._create_cv_splitter(X, y, task_type)
            scoring_metrics = self._get_scoring_metrics(task_type)
            
            cv_scores = {}
            mean_scores = {}
            std_scores = {}
            
            # Evaluate each metric
            for metric_name, scorer in scoring_metrics.items():
                try:
                    scores = cross_val_score(
                        model, X, y, 
                        cv=cv_splitter, 
                        scoring=scorer,
                        n_jobs=1  # Avoid nested parallelization
                    )
                    
                    # Convert negative scores back to positive for error metrics
                    if metric_name in ['mae', 'mse']:
                        scores = -scores
                    
                    cv_scores[metric_name] = scores.tolist()
                    mean_scores[metric_name] = float(np.mean(scores))
                    std_scores[metric_name] = float(np.std(scores))
                    
                except Exception as e:
                    logger.warning(f"Error evaluating {metric_name} for {model_name}: {e}")
                    cv_scores[metric_name] = []
                    mean_scores[metric_name] = 0.0
                    std_scores[metric_name] = 0.0
            
            # Calculate additional metrics
            feature_importance = self._calculate_feature_importance(model)
            model_size = self._estimate_model_size(model)
            
            # Create evaluation result
            result = ModelEvaluationResult(
                model_name=model_name,
                model_instance=model,
                cv_scores=cv_scores,
                mean_scores=mean_scores,
                std_scores=std_scores,
                training_time=training_time,
                prediction_time=prediction_time,
                model_size=model_size,
                feature_importance=feature_importance,
                hyperparameters=model.get_params(),
                metadata={
                    'task_type': task_type,
                    'cv_folds': self.config.cv_folds,
                    'evaluation_date': datetime.now().isoformat()
                }
            )
            
            # Cache result
            self.evaluation_cache[cache_key] = result
            
            logger.info(f"Completed evaluation of {model_name}: "
                       f"primary_score={result.primary_score:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_name}: {e}")
            raise ModelValidationError(f"Model evaluation failed: {e}")
    
    def evaluate_multiple_models(self,
                                models: Dict[str, BaseEstimator],
                                X: np.ndarray,
                                y: np.ndarray,
                                task_type: str = 'classification') -> List[ModelEvaluationResult]:
        """Evaluate multiple models"""
        
        logger.info(f"Evaluating {len(models)} models")
        
        results = []
        for model_name, model in models.items():
            try:
                result = self.evaluate_model(model, model_name, X, y, task_type)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {e}")
        
        # Sort by primary metric (descending)
        results.sort(key=lambda x: x.primary_score, reverse=True)
        
        logger.info(f"Completed evaluation of {len(results)} models")
        return results

# ============================================
# Statistical Model Comparison
# ============================================

class StatisticalModelComparator:
    """Statistical comparison of model performance"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def compare_two_models(self, 
                          result1: ModelEvaluationResult,
                          result2: ModelEvaluationResult,
                          metric: str = 'accuracy') -> Dict[str, Any]:
        """Compare two models statistically"""
        
        scores1 = result1.cv_scores.get(metric, [])
        scores2 = result2.cv_scores.get(metric, [])
        
        if not scores1 or not scores2:
            return {
                'statistic': None,
                'p_value': None,
                'significant': False,
                'better_model': None,
                'effect_size': None
            }
        
        # Paired t-test (if same CV folds)
        if len(scores1) == len(scores2):
            try:
                statistic, p_value = stats.ttest_rel(scores1, scores2)
                
                # Effect size (Cohen's d for paired samples)
                diff = np.array(scores1) - np.array(scores2)
                effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                
                significant = p_value < self.significance_level
                better_model = result1.model_name if np.mean(scores1) > np.mean(scores2) else result2.model_name
                
                return {
                    'test_type': 'paired_t_test',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': significant,
                    'better_model': better_model,
                    'effect_size': float(effect_size),
                    'mean_difference': float(np.mean(scores1) - np.mean(scores2))
                }
                
            except Exception as e:
                logger.warning(f"Error in paired t-test: {e}")
        
        # Fallback to independent t-test
        try:
            statistic, p_value = stats.ttest_ind(scores1, scores2)
            
            # Effect size (Cohen's d for independent samples)
            pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1) + 
                                 (len(scores2) - 1) * np.var(scores2)) / 
                                (len(scores1) + len(scores2) - 2))
            effect_size = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
            
            significant = p_value < self.significance_level
            better_model = result1.model_name if np.mean(scores1) > np.mean(scores2) else result2.model_name
            
            return {
                'test_type': 'independent_t_test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': significant,
                'better_model': better_model,
                'effect_size': float(effect_size),
                'mean_difference': float(np.mean(scores1) - np.mean(scores2))
            }
            
        except Exception as e:
            logger.error(f"Error in statistical comparison: {e}")
            return {
                'statistic': None,
                'p_value': None,
                'significant': False,
                'better_model': None,
                'effect_size': None
            }
    
    def compare_multiple_models(self,
                               results: List[ModelEvaluationResult],
                               metric: str = 'accuracy') -> Dict[str, Any]:
        """Compare multiple models using Friedman test"""
        
        # Prepare data for Friedman test
        model_scores = []
        model_names = []
        
        for result in results:
            scores = result.cv_scores.get(metric, [])
            if scores:
                model_scores.append(scores)
                model_names.append(result.model_name)
        
        if len(model_scores) < 3:
            return {
                'test_type': 'friedman_test',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'post_hoc_comparisons': None
            }
        
        try:
            # Ensure all score lists have the same length
            min_length = min(len(scores) for scores in model_scores)
            model_scores = [scores[:min_length] for scores in model_scores]
            
            # Friedman test
            statistic, p_value = friedmanchisquare(*model_scores)
            
            significant = p_value < self.significance_level
            
            # Post-hoc pairwise comparisons if significant
            post_hoc = None
            if significant:
                post_hoc = {}
                for i, j in combinations(range(len(model_scores)), 2):
                    name1, name2 = model_names[i], model_names[j]
                    scores1, scores2 = model_scores[i], model_scores[j]
                    
                    # Wilcoxon signed-rank test for pairwise comparison
                    try:
                        w_stat, w_p = wilcoxon(scores1, scores2)
                        post_hoc[f"{name1}_vs_{name2}"] = {
                            'statistic': float(w_stat),
                            'p_value': float(w_p),
                            'significant': w_p < self.significance_level / len(list(combinations(range(len(model_scores)), 2))),  # Bonferroni correction
                            'better_model': name1 if np.mean(scores1) > np.mean(scores2) else name2
                        }
                    except Exception as e:
                        logger.warning(f"Error in post-hoc comparison {name1} vs {name2}: {e}")
            
            return {
                'test_type': 'friedman_test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': significant,
                'model_names': model_names,
                'post_hoc_comparisons': post_hoc
            }
            
        except Exception as e:
            logger.error(f"Error in Friedman test: {e}")
            return {
                'test_type': 'friedman_test',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'post_hoc_comparisons': None
            }

# ============================================
# Advanced Model Selector
# ============================================

class AdvancedModelSelector:
    """Advanced model selection with multiple criteria and optimization strategies"""
    
    def __init__(self, config: SelectionConfig = None):
        self.config = config or SelectionConfig()
        self.evaluator = ModelEvaluator(self.config)
        self.comparator = StatisticalModelComparator(self.config.statistical_significance_level)
        
        # Selection history
        self.selection_history = []
        
        logger.info("Initialized AdvancedModelSelector")
    
    def _generate_candidate_models(self, 
                                  data_characteristics: Dict[str, Any],
                                  task_type: str = 'classification') -> Dict[str, BaseEstimator]:
        """Generate candidate models based on data characteristics"""
        
        # Get model recommendations
        recommendations = get_model_recommendations(
            data_characteristics=data_characteristics,
            task_type=task_type,
            priority='accuracy'
        )
        
        # Limit number of models
        top_recommendations = recommendations[:self.config.max_models_to_evaluate]
        
        candidate_models = {}
        
        # Create individual models
        for rec in top_recommendations:
            model_name = rec['model_name']
            preset = rec['recommended_preset']
            
            try:
                model = model_factory.create_model(
                    model_name=model_name,
                    task_type=task_type,
                    performance_preset=preset
                )
                candidate_models[f"{model_name}_{preset}"] = model
                
            except Exception as e:
                logger.warning(f"Failed to create model {model_name}: {e}")
        
        # Add ensemble models if requested
        if self.config.include_ensemble_methods and len(candidate_models) >= 2:
            try:
                # Get base models for ensembles
                base_model_names = list(top_recommendations[:3])  # Top 3 for ensembles
                base_model_names = [rec['model_name'] for rec in base_model_names[:3]]
                
                # Voting ensemble
                voting_ensemble = model_factory.create_ensemble_with_base_models(
                    ensemble_type='voting',
                    base_model_names=base_model_names,
                    task_type=task_type
                )
                candidate_models['voting_ensemble'] = voting_ensemble
                
                # Stacking ensemble
                stacking_ensemble = model_factory.create_ensemble_with_base_models(
                    ensemble_type='stacking',
                    base_model_names=base_model_names,
                    task_type=task_type
                )
                candidate_models['stacking_ensemble'] = stacking_ensemble
                
            except Exception as e:
                logger.warning(f"Failed to create ensemble models: {e}")
        
        logger.info(f"Generated {len(candidate_models)} candidate models")
        return candidate_models
    
    def _calculate_composite_score(self, result: ModelEvaluationResult) -> float:
        """Calculate composite score based on multiple criteria"""
        
        # Primary metric score
        primary_score = result.get_score(self.config.primary_metric)
        composite_score = primary_score * self.config.scoring_weights.get(self.config.primary_metric, 1.0)
        
        # Add secondary metrics
        for metric in self.config.secondary_metrics:
            weight = self.config.scoring_weights.get(metric, 0.1)
            score = result.get_score(metric)
            composite_score += score * weight
        
        # Penalty for complexity (optional)
        if 'simplicity_penalty' in self.config.scoring_weights:
            # Penalize based on training time and model size
            time_penalty = min(result.training_time / 100.0, 1.0)  # Normalize to [0, 1]
            size_penalty = min(result.model_size / 1e6, 1.0)  # Normalize to [0, 1]
            
            complexity_penalty = (time_penalty + size_penalty) / 2.0
            composite_score -= complexity_penalty * self.config.scoring_weights['simplicity_penalty']
        
        return composite_score
    
    @time_it("model_selection", include_args=True)
    def select_best_model(self,
                         X: Union[pd.DataFrame, np.ndarray],
                         y: Union[pd.Series, np.ndarray],
                         task_type: str = 'classification',
                         candidate_models: Optional[Dict[str, BaseEstimator]] = None) -> Dict[str, Any]:
        """Select the best model from candidates"""
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        logger.info(f"Starting model selection for {task_type} task")
        
        # Generate candidates if not provided
        if candidate_models is None:
            data_characteristics = {
                'n_samples': len(X),
                'n_features': X.shape[1] if len(X.shape) > 1 else 1,
                'has_missing': pd.isna(pd.DataFrame(X)).any().any(),
                'has_categorical': False  # Assuming preprocessed data
            }
            candidate_models = self._generate_candidate_models(data_characteristics, task_type)
        
        if not candidate_models:
            raise ValueError("No candidate models available for selection")
        
        # Evaluate all models
        results = self.evaluator.evaluate_multiple_models(candidate_models, X, y, task_type)
        
        if not results:
            raise ValueError("No models could be evaluated successfully")
        
        # Calculate composite scores
        for result in results:
            result.composite_score = self._calculate_composite_score(result)
        
        # Sort by composite score
        results.sort(key=lambda x: getattr(x, 'composite_score', x.primary_score), reverse=True)
        
        # Get best model
        best_result = results[0]
        
        # Statistical comparison with second best (if available)
        statistical_comparison = None
        if len(results) > 1:
            second_best = results[1]
            statistical_comparison = self.comparator.compare_two_models(
                best_result, second_best, self.config.primary_metric
            )
        
        # Overall statistical comparison
        overall_comparison = None
        if len(results) > 2:
            overall_comparison = self.comparator.compare_multiple_models(
                results, self.config.primary_metric
            )
        
        # Create selection result
        selection_result = {
            'best_model': best_result.model_instance,
            'best_model_name': best_result.model_name,
            'best_score': best_result.primary_score,
            'composite_score': getattr(best_result, 'composite_score', best_result.primary_score),
            'all_results': results,
            'statistical_comparison': statistical_comparison,
            'overall_comparison': overall_comparison,
            'selection_config': self.config,
            'selection_date': datetime.now(),
            'n_candidates_evaluated': len(results)
        }
        
        # Store in history
        self.selection_history.append(selection_result)
        
        logger.info(f"Model selection completed. Best model: {best_result.model_name} "
                   f"with score: {best_result.primary_score:.4f}")
        
        return selection_result
    
    def select_top_k_models(self,
                           X: Union[pd.DataFrame, np.ndarray],
                           y: Union[pd.Series, np.ndarray],
                           k: int = 3,
                           task_type: str = 'classification',
                           candidate_models: Optional[Dict[str, BaseEstimator]] = None) -> Dict[str, Any]:
        """Select top k models"""
        
        selection_result = self.select_best_model(X, y, task_type, candidate_models)
        all_results = selection_result['all_results']
        
        # Get top k results
        top_k_results = all_results[:k]
        
        # Statistical comparison among top k
        top_k_comparison = None
        if len(top_k_results) > 2:
            top_k_comparison = self.comparator.compare_multiple_models(
                top_k_results, self.config.primary_metric
            )
        
        return {
            'top_k_models': [(r.model_instance, r.model_name) for r in top_k_results],
            'top_k_results': top_k_results,
            'top_k_comparison': top_k_comparison,
            'selection_config': self.config,
            'selection_date': datetime.now()
        }

# ============================================
# Ensemble Selection and Optimization
# ============================================

class EnsembleSelector:
    """Advanced ensemble selection and optimization"""
    
    def __init__(self, config: SelectionConfig = None):
        self.config = config or SelectionConfig()
        self.base_selector = AdvancedModelSelector(self.config)
        
    def _evaluate_ensemble_combinations(self,
                                      models: List[Tuple[BaseEstimator, str]],
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      task_type: str = 'classification',
                                      max_combinations: int = 10) -> List[Dict[str, Any]]:
        """Evaluate different ensemble combinations"""
        
        ensemble_results = []
        
        # Try different combination sizes
        for combo_size in range(2, min(len(models) + 1, 6)):  # 2 to 5 models
            model_combinations = list(combinations(models, combo_size))
            
            # Limit number of combinations to evaluate
            if len(model_combinations) > max_combinations:
                np.random.seed(self.config.random_state)
                model_combinations = np.random.choice(
                    model_combinations, 
                    size=max_combinations, 
                    replace=False
                ).tolist()
            
            for combo in model_combinations:
                combo_models = [model for model, name in combo]
                combo_names = [name for model, name in combo]
                
                try:
                    # Create voting ensemble
                    voting_ensemble = model_factory.create_ensemble_with_base_models(
                        ensemble_type='voting',
                        base_model_names=combo_names[:3],  # Limit to 3 for practical reasons
                        task_type=task_type
                    )
                    
                    # Evaluate ensemble
                    ensemble_name = f"voting_{'_'.join(combo_names[:3])}"
                    result = self.base_selector.evaluator.evaluate_model(
                        voting_ensemble, ensemble_name, X, y, task_type
                    )
                    
                    ensemble_results.append({
                        'ensemble_type': 'voting',
                        'base_models': combo_names,
                        'result': result,
                        'ensemble_model': voting_ensemble
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate ensemble {combo_names}: {e}")
        
        return ensemble_results
    
    def optimize_ensemble_selection(self,
                                  X: Union[pd.DataFrame, np.ndarray],
                                  y: Union[pd.Series, np.ndarray],
                                  task_type: str = 'classification',
                                  n_base_models: int = 5) -> Dict[str, Any]:
        """Optimize ensemble selection by finding best base model combinations"""
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        logger.info("Starting ensemble selection optimization")
        
        # First, select top individual models
        top_models_result = self.base_selector.select_top_k_models(
            X, y, k=n_base_models, task_type=task_type
        )
        
        top_models = top_models_result['top_k_models']
        
        # Evaluate ensemble combinations
        ensemble_evaluations = self._evaluate_ensemble_combinations(
            top_models, X, y, task_type
        )
        
        if not ensemble_evaluations:
            logger.warning("No ensemble combinations could be evaluated")
            return top_models_result
        
        # Find best ensemble
        best_ensemble = max(ensemble_evaluations, key=lambda x: x['result'].primary_score)
        
        # Compare best ensemble with best individual model
        best_individual = top_models_result['top_k_results'][0]
        ensemble_vs_individual = self.base_selector.comparator.compare_two_models(
            best_ensemble['result'], best_individual, self.config.primary_metric
        )
        
        return {
            'best_ensemble': best_ensemble['ensemble_model'],
            'best_ensemble_result': best_ensemble['result'],
            'best_individual': best_individual.model_instance,
            'best_individual_result': best_individual,
            'ensemble_vs_individual_comparison': ensemble_vs_individual,
            'all_ensemble_evaluations': ensemble_evaluations,
            'top_individual_models': top_models_result,
            'recommendation': 'ensemble' if best_ensemble['result'].primary_score > best_individual.primary_score else 'individual'
        }

# ============================================
# Model Selection Pipeline
# ============================================

class ModelSelectionPipeline:
    """Complete model selection pipeline with automated workflow"""
    
    def __init__(self, config: SelectionConfig = None):
        self.config = config or SelectionConfig()
        self.selector = AdvancedModelSelector(self.config)
        self.ensemble_selector = EnsembleSelector(self.config)
        
        # Pipeline history
        self.pipeline_runs = []
        
    @time_it("model_selection_pipeline", include_args=True)
    def run_full_selection_pipeline(self,
                                   X: Union[pd.DataFrame, np.ndarray],
                                   y: Union[pd.Series, np.ndarray],
                                   task_type: str = 'classification',
                                   include_ensemble_optimization: bool = True,
                                   register_best_model: bool = True) -> Dict[str, Any]:
        """Run complete model selection pipeline"""
        
        logger.info("Starting full model selection pipeline")
        
        pipeline_start = datetime.now()
        
        # Step 1: Individual model selection
        logger.info("Step 1: Individual model selection")
        individual_selection = self.selector.select_best_model(X, y, task_type)
        
        # Step 2: Ensemble optimization (if requested)
        ensemble_optimization = None
        if include_ensemble_optimization:
            logger.info("Step 2: Ensemble optimization")
            try:
                ensemble_optimization = self.ensemble_selector.optimize_ensemble_selection(
                    X, y, task_type
                )
            except Exception as e:
                logger.warning(f"Ensemble optimization failed: {e}")
        
        # Step 3: Final recommendation
        if ensemble_optimization and ensemble_optimization['recommendation'] == 'ensemble':
            final_model = ensemble_optimization['best_ensemble']
            final_result = ensemble_optimization['best_ensemble_result']
            selection_type = 'ensemble'
        else:
            final_model = individual_selection['best_model']
            final_result = individual_selection['all_results'][0]
            selection_type = 'individual'
        
        # Step 4: Model registration (if requested)
        registered_model_id = None
        if register_best_model:
            try:
                registered_model_id = default_registry.register_model(
                    model=final_model,
                    model_name=f"selected_{final_result.model_name}",
                    task_type=task_type,
                    training_data=(X, y),
                    performance_metrics=final_result.mean_scores,
                    hyperparameters=final_result.hyperparameters,
                    description=f"Model selected through automated pipeline on {datetime.now().strftime('%Y-%m-%d')}",
                    tags=['auto_selected', task_type, selection_type],
                    cross_validation_scores=final_result.cv_scores.get(self.config.primary_metric, []),
                    feature_importance=final_result.feature_importance,
                    training_duration=final_result.training_time
                )
                logger.info(f"Registered selected model with ID: {registered_model_id}")
            except Exception as e:
                logger.warning(f"Failed to register selected model: {e}")
        
        # Create pipeline result
        pipeline_result = {
            'final_model': final_model,
            'final_result': final_result,
            'selection_type': selection_type,
            'individual_selection': individual_selection,
            'ensemble_optimization': ensemble_optimization,
            'registered_model_id': registered_model_id,
            'pipeline_config': self.config,
            'pipeline_duration': (datetime.now() - pipeline_start).total_seconds(),
            'pipeline_date': pipeline_start
        }
        
        # Store in history
        self.pipeline_runs.append(pipeline_result)
        
        logger.info(f"Model selection pipeline completed. "
                   f"Selected: {final_result.model_name} ({selection_type}) "
                   f"with score: {final_result.primary_score:.4f}")
        
        return pipeline_result
    
    def create_selection_report(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive selection report"""
        
        individual_results = pipeline_result['individual_selection']['all_results']
        ensemble_optimization = pipeline_result.get('ensemble_optimization')
        
        # Summary statistics
        individual_scores = [r.primary_score for r in individual_results]
        
        report = {
            'pipeline_summary': {
                'selection_date': pipeline_result['pipeline_date'].isoformat(),
                'pipeline_duration': pipeline_result['pipeline_duration'],
                'task_type': individual_results[0].metadata['task_type'],
                'primary_metric': self.config.primary_metric,
                'n_models_evaluated': len(individual_results)
            },
            
            'final_selection': {
                'model_name': pipeline_result['final_result'].model_name,
                'model_type': pipeline_result['selection_type'],
                'score': pipeline_result['final_result'].primary_score,
                'cv_scores': pipeline_result['final_result'].cv_scores,
                'hyperparameters': pipeline_result['final_result'].hyperparameters
            },
            
            'individual_models_summary': {
                'best_score': max(individual_scores),
                'worst_score': min(individual_scores),
                'mean_score': np.mean(individual_scores),
                'std_score': np.std(individual_scores),
                'top_3_models': [
                    {
                        'name': r.model_name,
                        'score': r.primary_score,
                        'rank': i + 1
                    }
                    for i, r in enumerate(individual_results[:3])
                ]
            },
            
            'statistical_analysis': pipeline_result['individual_selection'].get('overall_comparison'),
            
            'recommendations': {
                'selected_model': pipeline_result['final_result'].model_name,
                'selection_confidence': 'high' if pipeline_result['final_result'].primary_score > np.mean(individual_scores) + np.std(individual_scores) else 'medium',
                'alternative_models': [r.model_name for r in individual_results[1:4]]
            }
        }
        
        # Add ensemble analysis if available
        if ensemble_optimization:
            ensemble_scores = [e['result'].primary_score for e in ensemble_optimization['all_ensemble_evaluations']]
            
            report['ensemble_analysis'] = {
                'ensembles_evaluated': len(ensemble_scores),
                'best_ensemble_score': max(ensemble_scores) if ensemble_scores else None,
                'ensemble_improvement': ensemble_optimization['ensemble_vs_individual_comparison'],
                'recommendation': ensemble_optimization['recommendation']
            }
        
        return report

# ============================================
# Global Instances and Convenience Functions
# ============================================

# Global instances
default_selector = AdvancedModelSelector()
default_pipeline = ModelSelectionPipeline()

# Convenience functions
def select_best_model(X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray],
                     task_type: str = 'classification',
                     **kwargs) -> Dict[str, Any]:
    """Select best model using default selector"""
    return default_selector.select_best_model(X, y, task_type, **kwargs)

def run_model_selection_pipeline(X: Union[pd.DataFrame, np.ndarray],
                                y: Union[pd.Series, np.ndarray],
                                task_type: str = 'classification',
                                **kwargs) -> Dict[str, Any]:
    """Run full model selection pipeline"""
    return default_pipeline.run_full_selection_pipeline(X, y, task_type, **kwargs)

def compare_models_statistically(results: List[ModelEvaluationResult],
                                metric: str = 'accuracy') -> Dict[str, Any]:
    """Compare models statistically"""
    comparator = StatisticalModelComparator()
    return comparator.compare_multiple_models(results, metric)

def create_selection_config(primary_metric: str = 'accuracy',
                           secondary_metrics: List[str] = None,
                           **kwargs) -> SelectionConfig:
    """Create selection configuration"""
    return SelectionConfig(
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics or [],
        **kwargs
    )
