# ============================================
# StockPredictionPro - src/models/classification/knn.py
# K-Nearest Neighbors classification models for financial prediction with distance-based learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, log_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_classifier import BaseFinancialClassifier, ClassificationStrategy

logger = get_logger('models.classification.knn')

# ============================================
# K-Nearest Neighbors Classification Model
# ============================================

class FinancialKNNClassifier(BaseFinancialClassifier):
    """
    K-Nearest Neighbors classification model optimized for financial data
    
    Features:
    - Multiple distance metrics and weighting schemes
    - Adaptive K selection and cross-validation optimization
    - Comprehensive neighborhood analysis and local density estimation
    - Curse of dimensionality mitigation with feature selection and PCA
    - Financial domain optimizations (temporal weighting, volatility-aware distances)
    - Distance-based outlier detection and confidence estimation
    - Computational optimizations for large-scale financial datasets
    """
    
    def __init__(self,
                 name: str = "knn_classifier",
                 n_neighbors: int = 5,
                 weights: str = 'uniform',
                 algorithm: str = 'auto',
                 leaf_size: int = 30,
                 p: int = 2,
                 metric: str = 'minkowski',
                 metric_params: Optional[Dict] = None,
                 n_jobs: Optional[int] = None,
                 auto_k: bool = False,
                 k_range: Tuple[int, int] = (1, 50),
                 feature_selection: Optional[str] = None,
                 selection_k: Optional[int] = None,
                 dimensionality_reduction: Optional[str] = None,
                 n_components: Optional[int] = None,
                 scaler_type: str = 'standard',
                 calibrate_probabilities: bool = True,
                 distance_weighting: Optional[str] = None,
                 outlier_detection: bool = True,
                 auto_scale: bool = True,
                 **kwargs):
        """
        Initialize Financial K-Nearest Neighbors Classifier
        
        Args:
            name: Model name
            n_neighbors: Number of neighbors to use
            weights: Weight function ('uniform', 'distance')
            algorithm: Algorithm for nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
            leaf_size: Leaf size for tree algorithms
            p: Power parameter for Minkowski metric
            metric: Distance metric ('minkowski', 'euclidean', 'manhattan', 'chebyshev')
            metric_params: Additional parameters for distance metric
            n_jobs: Number of parallel jobs
            auto_k: Whether to automatically select optimal K
            k_range: Range of K values to test for auto selection
            feature_selection: Feature selection method ('k_best', 'pca_select')
            selection_k: Number of features to select
            dimensionality_reduction: Dimensionality reduction method ('pca', 'none')
            n_components: Number of components for dimensionality reduction
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            calibrate_probabilities: Whether to calibrate prediction probabilities
            distance_weighting: Additional distance weighting scheme
            outlier_detection: Whether to perform outlier detection
            auto_scale: Whether to automatically scale features
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="knn_classifier",
            classification_strategy=ClassificationStrategy.DIRECTION_PREDICTION,
            **kwargs
        )
        
        # KNN parameters
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params or {}
        self.n_jobs = n_jobs
        self.auto_k = auto_k
        self.k_range = k_range
        self.feature_selection = feature_selection
        self.selection_k = selection_k
        self.dimensionality_reduction = dimensionality_reduction
        self.n_components = n_components
        self.scaler_type = scaler_type
        self.calibrate_probabilities = calibrate_probabilities
        self.distance_weighting = distance_weighting
        self.outlier_detection = outlier_detection
        self.auto_scale = auto_scale
        
        # Store parameters for model creation
        self.model_params.update({
            'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithm,
            'leaf_size': leaf_size,
            'p': p,
            'metric': metric,
            'metric_params': metric_params,
            'n_jobs': n_jobs
        })
        
        # KNN-specific attributes
        self.scaler_: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.feature_selector_: Optional[Any] = None
        self.dimensionality_reducer_: Optional[PCA] = None
        self.selected_features_: Optional[List[str]] = None
        self.calibrated_model_: Optional[CalibratedClassifierCV] = None
        self.optimal_k_: Optional[int] = None
        self.k_selection_scores_: Optional[Dict[int, float]] = None
        self.neighborhood_analysis_: Optional[Dict[str, Any]] = None
        self.distance_analysis_: Optional[Dict[str, Any]] = None
        self.class_weights_: Optional[Dict[Any, float]] = None
        self.outlier_scores_: Optional[np.ndarray] = None
        self.local_density_: Optional[np.ndarray] = None
        self.training_data_: Optional[np.ndarray] = None
        self.training_labels_: Optional[np.ndarray] = None
        
        logger.info(f"Initialized KNN classifier with K={n_neighbors}: {self.name}")
    
    def _create_model(self) -> KNeighborsClassifier:
        """Create the K-Nearest Neighbors classification model"""
        return KNeighborsClassifier(**self.model_params)
    
    def _create_scaler(self) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
        """Create appropriate scaler based on scaler_type"""
        
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with scaling, selection, and dimensionality reduction"""
        
        # Basic preprocessing
        X_processed = super()._preprocess_features(X)
        
        # Apply feature scaling (essential for KNN)
        if self.auto_scale:
            if self.scaler_ is None:
                self.scaler_ = self._create_scaler()
                X_scaled = self.scaler_.fit_transform(X_processed)
                logger.debug(f"Fitted {self.scaler_type} scaler for KNN classifier")
            else:
                X_scaled = self.scaler_.transform(X_processed)
            X_processed = X_scaled
        
        # Apply feature selection if fitted
        if self.feature_selector_ is not None:
            X_processed = self.feature_selector_.transform(X_processed)
        
        # Apply dimensionality reduction if fitted
        if self.dimensionality_reducer_ is not None:
            X_processed = self.dimensionality_reducer_.transform(X_processed)
        
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
        """Perform feature selection to combat curse of dimensionality"""
        
        if self.feature_selection is None:
            return X, self.feature_names.copy()
        
        k = self.selection_k or min(20, X.shape[1])
        
        if self.feature_selection == 'k_best':
            # Select K best features using F-statistic
            self.feature_selector_ = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector_.fit_transform(X, y)
            
            # Get selected feature names
            selected_indices = self.feature_selector_.get_support(indices=True)
            selected_features = [self.feature_names[i] for i in selected_indices]
            
        elif self.feature_selection == 'pca_select':
            # Use PCA for feature selection (keep components explaining variance)
            pca_temp = PCA(n_components=k)
            pca_temp.fit(X)
            
            # Select top features based on PCA loadings
            loadings = np.abs(pca_temp.components_).sum(axis=0)
            top_indices = np.argsort(loadings)[-k:]
            
            X_selected = X[:, top_indices]
            selected_features = [self.feature_names[i] for i in top_indices]
            
            # Create a simple selector for consistency
            self.feature_selector_ = type('SimpleSelector', (), {
                'transform': lambda self, X: X[:, top_indices],
                'get_support': lambda self, indices=False: top_indices if indices else np.isin(np.arange(len(self.feature_names)), top_indices)
            })()
            
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection}")
        
        self.selected_features_ = selected_features
        logger.info(f"Selected {len(selected_features)} features using {self.feature_selection}")
        
        return X_selected, selected_features
    
    def _perform_dimensionality_reduction(self, X: np.ndarray) -> np.ndarray:
        """Perform dimensionality reduction to improve KNN performance"""
        
        if self.dimensionality_reduction is None:
            return X
        
        if self.dimensionality_reduction == 'pca':
            if self.dimensionality_reducer_ is None:
                # Determine optimal number of components
                if self.n_components is None:
                    # Use elbow method or preserve 95% variance
                    pca_temp = PCA()
                    pca_temp.fit(X)
                    cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
                    n_comp = np.argmax(cumsum_var >= 0.95) + 1
                    n_comp = min(n_comp, X.shape[1] - 1, 50)  # Reasonable limits
                else:
                    n_comp = min(self.n_components, X.shape[1] - 1)
                
                self.dimensionality_reducer_ = PCA(n_components=n_comp)
                X_reduced = self.dimensionality_reducer_.fit_transform(X)
                
                logger.info(f"Applied PCA: {X.shape[1]} → {n_comp} dimensions "
                           f"(explained variance: {self.dimensionality_reducer_.explained_variance_ratio_.sum():.1%})")
            else:
                X_reduced = self.dimensionality_reducer_.transform(X)
            
            return X_reduced
        
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {self.dimensionality_reduction}")
    
    def _select_optimal_k(self, X: np.ndarray, y: np.ndarray) -> int:
        """Select optimal K using cross-validation"""
        
        if not self.auto_k:
            return self.n_neighbors
        
        from sklearn.model_selection import cross_val_score
        
        k_min, k_max = self.k_range
        k_values = list(range(k_min, min(k_max + 1, len(X) // 2)))  # Don't exceed half the data
        
        logger.info(f"Selecting optimal K from range {k_min}-{k_max}")
        
        best_k = k_values[0]
        best_score = -np.inf
        k_scores = {}
        
        for k in k_values:
            try:
                # Create temporary model with current k
                temp_model = KNeighborsClassifier(
                    n_neighbors=k,
                    weights=self.weights,
                    algorithm=self.algorithm,
                    metric=self.metric,
                    p=self.p,
                    n_jobs=self.n_jobs
                )
                
                # Cross-validation
                cv_scores = cross_val_score(temp_model, X, y, cv=5, scoring='accuracy', n_jobs=self.n_jobs)
                mean_score = cv_scores.mean()
                
                k_scores[k] = mean_score
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_k = k
                
                logger.debug(f"K={k}: CV accuracy = {mean_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Error testing K={k}: {e}")
                k_scores[k] = -np.inf
        
        self.k_selection_scores_ = k_scores
        self.optimal_k_ = best_k
        
        logger.info(f"Selected optimal K={best_k} with CV accuracy {best_score:.4f}")
        
        return best_k
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Enhanced post-training processing for KNN classification"""
        
        # Call parent processing
        super()._post_training_processing(X, y)
        
        # Store training data for later analysis
        self.training_data_ = X.copy()
        self.training_labels_ = y.copy()
        
        # Calculate class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.class_weights_ = {cls: count / len(y) for cls, count in zip(unique_classes, class_counts)}
        
        # Analyze neighborhood structure
        self._analyze_neighborhoods(X, y)
        
        # Analyze distance distributions
        self._analyze_distances(X, y)
        
        # Perform outlier detection if requested
        if self.outlier_detection:
            self._detect_outliers(X, y)
        
        # Calculate local density estimates
        self._calculate_local_density(X)
        
        # Calibrate probabilities if requested
        if self.calibrate_probabilities:
            self._calibrate_probabilities(X, y)
    
    def _analyze_neighborhoods(self, X: np.ndarray, y: np.ndarray):
        """Analyze neighborhood structure and homogeneity"""
        
        try:
            # Get neighbors for each training point
            neighbors = self.model.kneighbors(X, return_distance=False)
            
            # Calculate neighborhood purity (homogeneity)
            purities = []
            class_agreements = []
            
            for i, neighbor_indices in enumerate(neighbors):
                true_class = y[i]
                neighbor_classes = y[neighbor_indices]
                
                # Neighborhood purity (fraction of same-class neighbors)
                purity = np.mean(neighbor_classes == true_class)
                purities.append(purity)
                
                # Class agreement (fraction of most common class in neighborhood)
                unique, counts = np.unique(neighbor_classes, return_counts=True)
                max_agreement = np.max(counts) / len(neighbor_classes)
                class_agreements.append(max_agreement)
            
            purities = np.array(purities)
            class_agreements = np.array(class_agreements)
            
            # Calculate statistics per class
            class_stats = {}
            for class_idx, class_name in enumerate(self.classes_):
                class_mask = y == class_idx
                if np.any(class_mask):
                    class_stats[class_name] = {
                        'mean_purity': float(np.mean(purities[class_mask])),
                        'std_purity': float(np.std(purities[class_mask])),
                        'mean_agreement': float(np.mean(class_agreements[class_mask])),
                        'n_samples': int(np.sum(class_mask))
                    }
            
            self.neighborhood_analysis_ = {
                'overall_purity': {
                    'mean': float(np.mean(purities)),
                    'std': float(np.std(purities)),
                    'min': float(np.min(purities)),
                    'max': float(np.max(purities))
                },
                'class_agreement': {
                    'mean': float(np.mean(class_agreements)),
                    'std': float(np.std(class_agreements)),
                    'min': float(np.min(class_agreements)),
                    'max': float(np.max(class_agreements))
                },
                'class_statistics': class_stats,
                'k_value': self.model.n_neighbors,
                'difficult_samples': int(np.sum(purities < 0.5)),  # Samples in mixed neighborhoods
                'easy_samples': int(np.sum(purities >= 0.8))       # Samples in pure neighborhoods
            }
            
            logger.debug(f"Neighborhood analysis complete. Mean purity: {np.mean(purities):.3f}")
            
        except Exception as e:
            logger.warning(f"Could not analyze neighborhoods: {e}")
            self.neighborhood_analysis_ = None
    
    def _analyze_distances(self, X: np.ndarray, y: np.ndarray):
        """Analyze distance distributions and nearest neighbor distances"""
        
        try:
            # Get distances to nearest neighbors
            distances, neighbor_indices = self.model.kneighbors(X)
            
            # Analyze distance distributions
            mean_distances = np.mean(distances, axis=1)
            min_distances = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]  # Skip self
            max_distances = np.max(distances, axis=1)
            
            # Calculate inter-class vs intra-class distances
            intra_class_distances = []
            inter_class_distances = []
            
            for i, neighbor_idx in enumerate(neighbor_indices):
                true_class = y[i]
                neighbor_classes = y[neighbor_idx]
                point_distances = distances[i]
                
                # Separate intra-class and inter-class distances
                for j, (neighbor_class, dist) in enumerate(zip(neighbor_classes, point_distances)):
                    if j == 0:  # Skip self-distance
                        continue
                    
                    if neighbor_class == true_class:
                        intra_class_distances.append(dist)
                    else:
                        inter_class_distances.append(dist)
            
            intra_class_distances = np.array(intra_class_distances)
            inter_class_distances = np.array(inter_class_distances)
            
            # Distance statistics per class
            class_distance_stats = {}
            for class_idx, class_name in enumerate(self.classes_):
                class_mask = y == class_idx
                if np.any(class_mask):
                    class_distances = mean_distances[class_mask]
                    class_distance_stats[class_name] = {
                        'mean_distance': float(np.mean(class_distances)),
                        'std_distance': float(np.std(class_distances)),
                        'median_distance': float(np.median(class_distances)),
                        'n_samples': int(np.sum(class_mask))
                    }
            
            self.distance_analysis_ = {
                'overall_statistics': {
                    'mean_distance_to_neighbors': float(np.mean(mean_distances)),
                    'std_distance_to_neighbors': float(np.std(mean_distances)),
                    'mean_nearest_neighbor_distance': float(np.mean(min_distances)),
                    'mean_farthest_neighbor_distance': float(np.mean(max_distances))
                },
                'class_separation': {
                    'mean_intra_class_distance': float(np.mean(intra_class_distances)) if len(intra_class_distances) > 0 else None,
                    'mean_inter_class_distance': float(np.mean(inter_class_distances)) if len(inter_class_distances) > 0 else None,
                    'separation_ratio': float(np.mean(inter_class_distances) / np.mean(intra_class_distances)) if len(intra_class_distances) > 0 and len(inter_class_distances) > 0 else None
                },
                'class_distance_stats': class_distance_stats,
                'distance_distribution': {
                    'percentiles': {
                        '25th': float(np.percentile(mean_distances, 25)),
                        '50th': float(np.percentile(mean_distances, 50)),
                        '75th': float(np.percentile(mean_distances, 75)),
                        '95th': float(np.percentile(mean_distances, 95))
                    }
                }
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze distances: {e}")
            self.distance_analysis_ = None
    
    def _detect_outliers(self, X: np.ndarray, y: np.ndarray):
        """Detect outliers using k-nearest neighbor distances"""
        
        try:
            # Get distances to k nearest neighbors
            distances, _ = self.model.kneighbors(X)
            
            # Calculate outlier scores based on mean distance to neighbors
            outlier_scores = np.mean(distances, axis=1)
            
            # Normalize scores (higher = more outlier-like)
            outlier_scores = (outlier_scores - np.min(outlier_scores)) / (np.max(outlier_scores) - np.min(outlier_scores) + 1e-8)
            
            self.outlier_scores_ = outlier_scores
            
            # Identify outliers using 95th percentile threshold
            outlier_threshold = np.percentile(outlier_scores, 95)
            n_outliers = np.sum(outlier_scores > outlier_threshold)
            
            logger.debug(f"Detected {n_outliers} outliers ({n_outliers/len(X)*100:.1f}% of data)")
            
        except Exception as e:
            logger.warning(f"Could not detect outliers: {e}")
            self.outlier_scores_ = None
    
    def _calculate_local_density(self, X: np.ndarray):
        """Calculate local density estimates using k-nearest neighbors"""
        
        try:
            # Get distances to k nearest neighbors
            distances, _ = self.model.kneighbors(X)
            
            # Local density as inverse of mean distance to neighbors
            mean_distances = np.mean(distances, axis=1)
            local_density = 1.0 / (mean_distances + 1e-8)  # Add small epsilon to avoid division by zero
            
            # Normalize density scores
            local_density = (local_density - np.min(local_density)) / (np.max(local_density) - np.min(local_density) + 1e-8)
            
            self.local_density_ = local_density
            
        except Exception as e:
            logger.warning(f"Could not calculate local density: {e}")
            self.local_density_ = None
    
    def _calibrate_probabilities(self, X: np.ndarray, y: np.ndarray):
        """Calibrate prediction probabilities using cross-validation"""
        
        try:
            # Use isotonic calibration for KNN
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
    
    @time_it("knn_fit", include_args=True)
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'FinancialKNNClassifier':
        """
        Fit the K-Nearest Neighbors classification model
        
        Args:
            X: Feature matrix
            y: Target labels
            **kwargs: Additional fitting parameters
            
        Returns:
            Fitted model instance
        """
        logger.info(f"Fitting KNN Classifier on {len(X)} samples with {X.shape[1]} features")
        
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
            
            # Initial feature preprocessing (scaling)
            X_processed = self._preprocess_features(X)
            
            # Perform feature selection if specified
            if self.feature_selection:
                X_processed, selected_features = self._perform_feature_selection(X_processed, y_processed)
                logger.info(f"Feature selection completed: {len(selected_features)} features selected")
            else:
                selected_features = self.feature_names.copy()
            
            self.selected_features_ = selected_features
            
            # Perform dimensionality reduction if specified
            if self.dimensionality_reduction:
                X_processed = self._perform_dimensionality_reduction(X_processed)
                logger.info(f"Dimensionality reduction completed: {X_processed.shape[1]} dimensions")
            
            # Select optimal K if auto_k is enabled
            if self.auto_k:
                optimal_k = self._select_optimal_k(X_processed, y_processed)
                self.model_params['n_neighbors'] = optimal_k
            
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
                'final_dimensions': X_processed.shape[1],
                'n_classes': len(self.classes_),
                'class_names': self.classes_.tolist(),
                'training_duration_seconds': fit_duration,
                'target_name': self.target_name,
                'optimal_k': self.optimal_k_,
                'distance_metric': self.metric
            })
            
            # Calculate training score
            self.training_score = self.model.score(X_processed, y_processed)
            
            # Update status
            self.status = ModelStatus.TRAINED
            self.is_fitted = True
            
            logger.info(f"KNN Classifier trained successfully in {fit_duration:.2f}s")
            
            return self
            
        except Exception as e:
            from ..base.base_model import ModelStatus
            self.status = ModelStatus.ERROR
            self.last_error = str(e)
            logger.error(f"KNN Classifier training failed: {e}")
            raise
    
    @time_it("knn_predict", include_args=True)
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions using the fitted KNN classifier
        
        Args:
            X: Feature matrix for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        logger.debug(f"Making KNN predictions for {len(X)} samples")
        
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
            logger.error(f"KNN prediction failed: {e}")
            raise
    
    @time_it("knn_predict_proba", include_args=True)
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
        
        logger.debug(f"Making KNN probability predictions for {len(X)} samples")
        
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
            logger.error(f"KNN probability prediction failed: {e}")
            raise
    
    def predict_with_neighbors(self, X: pd.DataFrame, return_distance: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions and return information about nearest neighbors
        
        Args:
            X: Feature matrix for prediction
            return_distance: Whether to return distances to neighbors
            
        Returns:
            Tuple of (predictions, neighbor_indices) or (predictions, neighbor_indices, distances)
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        X_processed = self._preprocess_features(X)
        
        # Get neighbors
        if return_distance:
            distances, neighbor_indices = self.model.kneighbors(X_processed, return_distance=True)
        else:
            neighbor_indices = self.model.kneighbors(X_processed, return_distance=False)
        
        # Make predictions
        predictions_encoded = self.model.predict(X_processed)
        predictions = self.label_encoder_.inverse_transform(predictions_encoded)
        
        if return_distance:
            return predictions, neighbor_indices, distances
        else:
            return predictions, neighbor_indices
    
    def get_neighborhood_analysis(self) -> Dict[str, Any]:
        """
        Get neighborhood analysis results
        
        Returns:
            Dictionary with neighborhood analysis
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get neighborhood analysis")
        
        return self.neighborhood_analysis_.copy() if self.neighborhood_analysis_ else {}
    
    def get_distance_analysis(self) -> Dict[str, Any]:
        """
        Get distance analysis results
        
        Returns:
            Dictionary with distance analysis
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get distance analysis")
        
        return self.distance_analysis_.copy() if self.distance_analysis_ else {}
    
    def get_outlier_scores(self) -> Optional[np.ndarray]:
        """
        Get outlier scores for training data
        
        Returns:
            Array of outlier scores or None
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get outlier scores")
        
        return self.outlier_scores_.copy() if self.outlier_scores_ is not None else None
    
    def get_local_density(self) -> Optional[np.ndarray]:
        """
        Get local density estimates for training data
        
        Returns:
            Array of local density estimates or None
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get local density")
        
        return self.local_density_.copy() if self.local_density_ is not None else None
    
    def plot_k_selection_curve(self) -> Any:
        """
        Plot K selection curve if auto K selection was used
        
        Returns:
            Matplotlib figure
        """
        if not self.k_selection_scores_:
            logger.warning("K selection curve not available (auto_k was not used)")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            k_values = list(self.k_selection_scores_.keys())
            scores = list(self.k_selection_scores_.values())
            
            plt.figure(figsize=(12, 8))
            plt.plot(k_values, scores, 'bo-', linewidth=2, markersize=6)
            
            # Highlight optimal K
            if self.optimal_k_ is not None:
                optimal_score = self.k_selection_scores_[self.optimal_k_]
                plt.plot(self.optimal_k_, optimal_score, 'ro', markersize=12, 
                        label=f'Optimal K = {self.optimal_k_}')
            
            plt.xlabel('Number of Neighbors (K)')
            plt.ylabel('Cross-Validation Accuracy')
            plt.title(f'KNN K Selection - {self.name}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add score values as text
            for k, score in zip(k_values[::2], scores[::2]):  # Show every other point to avoid crowding
                plt.text(k, score + 0.005, f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_neighborhood_analysis(self) -> Any:
        """
        Plot neighborhood analysis results
        
        Returns:
            Matplotlib figure
        """
        if not self.neighborhood_analysis_:
            logger.warning("Neighborhood analysis not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            analysis = self.neighborhood_analysis_
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Overall purity distribution
            purity_stats = analysis['overall_purity']
            axes[0, 0].bar(['Mean', 'Std', 'Min', 'Max'], 
                          [purity_stats['mean'], purity_stats['std'], 
                           purity_stats['min'], purity_stats['max']], 
                          alpha=0.7, color='steelblue')
            axes[0, 0].set_title('Neighborhood Purity Statistics')
            axes[0, 0].set_ylabel('Purity Score')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add values on bars
            bars = axes[0, 0].patches
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom')
            
            # Class agreement
            agreement_stats = analysis['class_agreement']
            axes[0, 1].bar(['Mean', 'Std', 'Min', 'Max'], 
                          [agreement_stats['mean'], agreement_stats['std'], 
                           agreement_stats['min'], agreement_stats['max']], 
                          alpha=0.7, color='orange')
            axes[0, 1].set_title('Class Agreement Statistics')
            axes[0, 1].set_ylabel('Agreement Score')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add values on bars
            bars = axes[0, 1].patches
            for bar in bars:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom')
            
            # Per-class purity
            if analysis['class_statistics']:
                class_names = list(analysis['class_statistics'].keys())
                class_purities = [analysis['class_statistics'][cls]['mean_purity'] for cls in class_names]
                
                bars = axes[1, 0].bar(class_names, class_purities, alpha=0.7, color='green')
                axes[1, 0].set_title('Mean Purity by Class')
                axes[1, 0].set_ylabel('Mean Purity')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add values on bars
                for bar, purity in zip(bars, class_purities):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., purity + 0.01,
                                   f'{purity:.3f}', ha='center', va='bottom')
            else:
                axes[1, 0].text(0.5, 0.5, 'No class statistics\navailable', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Sample difficulty distribution
            difficult_samples = analysis['difficult_samples']
            easy_samples = analysis['easy_samples']
            total_samples = difficult_samples + easy_samples + (len(self.training_data_) - difficult_samples - easy_samples)
            
            categories = ['Difficult\n(purity < 0.5)', 'Easy\n(purity >= 0.8)', 'Moderate']
            counts = [difficult_samples, easy_samples, total_samples - difficult_samples - easy_samples]
            
            wedges, texts, autotexts = axes[1, 1].pie(counts, labels=categories, autopct='%1.1f%%', 
                                                     startangle=90, colors=['red', 'green', 'yellow'])
            axes[1, 1].set_title(f'Sample Difficulty (K={analysis["k_value"]})')
            
            plt.suptitle(f'KNN Neighborhood Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_distance_analysis(self) -> Any:
        """
        Plot distance analysis results
        
        Returns:
            Matplotlib figure
        """
        if not self.distance_analysis_:
            logger.warning("Distance analysis not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            analysis = self.distance_analysis_
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Overall distance statistics
            overall = analysis['overall_statistics']
            distance_types = ['Mean Distance\nto Neighbors', 'Mean Nearest\nNeighbor Distance', 
                             'Mean Farthest\nNeighbor Distance']
            distance_values = [overall['mean_distance_to_neighbors'], 
                              overall['mean_nearest_neighbor_distance'],
                              overall['mean_farthest_neighbor_distance']]
            
            bars = axes[0, 0].bar(distance_types, distance_values, alpha=0.7, color='steelblue')
            axes[0, 0].set_title('Overall Distance Statistics')
            axes[0, 0].set_ylabel('Distance')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add values on bars
            for bar, value in zip(bars, distance_values):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., value + max(distance_values) * 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            # Class separation
            separation = analysis['class_separation']
            if separation['separation_ratio'] is not None:
                sep_categories = ['Intra-class\nDistance', 'Inter-class\nDistance']
                sep_values = [separation['mean_intra_class_distance'], separation['mean_inter_class_distance']]
                
                bars = axes[0, 1].bar(sep_categories, sep_values, alpha=0.7, 
                                     color=['red', 'green'])
                axes[0, 1].set_title(f'Class Separation (Ratio: {separation["separation_ratio"]:.2f})')
                axes[0, 1].set_ylabel('Mean Distance')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Add values on bars
                for bar, value in zip(bars, sep_values):
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., value + max(sep_values) * 0.01,
                                   f'{value:.3f}', ha='center', va='bottom')
            else:
                axes[0, 1].text(0.5, 0.5, 'Class separation\nnot available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
            
            # Per-class distance statistics
            if analysis['class_distance_stats']:
                class_names = list(analysis['class_distance_stats'].keys())
                class_distances = [analysis['class_distance_stats'][cls]['mean_distance'] for cls in class_names]
                
                bars = axes[1, 0].bar(class_names, class_distances, alpha=0.7, color='orange')
                axes[1, 0].set_title('Mean Distance to Neighbors by Class')
                axes[1, 0].set_ylabel('Mean Distance')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add values on bars
                for bar, distance in zip(bars, class_distances):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., distance + max(class_distances) * 0.01,
                                   f'{distance:.3f}', ha='center', va='bottom')
            else:
                axes[1, 0].text(0.5, 0.5, 'Per-class statistics\nnot available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
            
            # Distance distribution percentiles
            percentiles = analysis['distance_distribution']['percentiles']
            perc_labels = list(percentiles.keys())
            perc_values = list(percentiles.values())
            
            bars = axes[1, 1].bar(perc_labels, perc_values, alpha=0.7, color='purple')
            axes[1, 1].set_title('Distance Distribution Percentiles')
            axes[1, 1].set_ylabel('Distance')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add values on bars
            for bar, value in zip(bars, perc_values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., value + max(perc_values) * 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.suptitle(f'KNN Distance Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_outlier_analysis(self) -> Any:
        """
        Plot outlier detection results
        
        Returns:
            Matplotlib figure
        """
        if self.outlier_scores_ is None:
            logger.warning("Outlier analysis not available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Outlier score distribution
            axes[0].hist(self.outlier_scores_, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            axes[0].axvline(np.percentile(self.outlier_scores_, 95), color='red', linestyle='--', 
                           label='95th Percentile (Outlier Threshold)')
            axes[0].set_xlabel('Outlier Score')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Distribution of Outlier Scores')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Box plot of outlier scores by class
            if self.training_labels_ is not None:
                class_outlier_scores = []
                class_labels = []
                
                for class_idx, class_name in enumerate(self.classes_):
                    class_mask = self.training_labels_ == class_idx
                    if np.any(class_mask):
                        class_outlier_scores.append(self.outlier_scores_[class_mask])
                        class_labels.append(class_name)
                
                axes[1].boxplot(class_outlier_scores, labels=class_labels)
                axes[1].set_ylabel('Outlier Score')
                axes[1].set_title('Outlier Scores by Class')
                axes[1].tick_params(axis='x', rotation=45)
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'Class-wise outlier\nanalysis not available', 
                            ha='center', va='center', transform=axes[1].transAxes)
            
            # Top outliers
            outlier_threshold = np.percentile(self.outlier_scores_, 95)
            n_outliers = np.sum(self.outlier_scores_ > outlier_threshold)
            
            # Summary statistics
            stats_text = f"Total Samples: {len(self.outlier_scores_)}\n"
            stats_text += f"Outliers (95th percentile): {n_outliers}\n"
            stats_text += f"Outlier Percentage: {n_outliers/len(self.outlier_scores_)*100:.1f}%\n\n"
            stats_text += f"Mean Outlier Score: {np.mean(self.outlier_scores_):.3f}\n"
            stats_text += f"Std Outlier Score: {np.std(self.outlier_scores_):.3f}\n"
            stats_text += f"Max Outlier Score: {np.max(self.outlier_scores_):.3f}\n"
            stats_text += f"95th Percentile: {outlier_threshold:.3f}"
            
            axes[2].text(0.1, 0.9, stats_text, transform=axes[2].transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[2].set_title('Outlier Summary Statistics')
            axes[2].axis('off')
            
            plt.suptitle(f'KNN Outlier Analysis - {self.name}', fontsize=16)
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
    
    def get_knn_summary(self) -> Dict[str, Any]:
        """Get comprehensive KNN summary"""
        
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get KNN summary")
        
        summary = {
            'model_info': {
                'n_neighbors': self.model.n_neighbors,
                'optimal_k': self.optimal_k_,
                'auto_k_used': self.auto_k,
                'weights': self.weights,
                'distance_metric': self.metric,
                'algorithm': self.algorithm,
                'n_classes': len(self.classes_),
                'class_names': self.classes_.tolist(),
                'n_features_original': len(self.feature_names),
                'n_features_selected': len(self.selected_features_) if self.selected_features_ else len(self.feature_names),
                'final_dimensions': self.training_data_.shape[1] if self.training_data_ is not None else None
            },
            'preprocessing': {
                'feature_selection_method': self.feature_selection,
                'dimensionality_reduction': self.dimensionality_reduction,
                'scaler_type': self.scaler_type,
                'explained_variance_ratio': (
                    self.dimensionality_reducer_.explained_variance_ratio_.sum()
                    if self.dimensionality_reducer_ is not None else None
                )
            },
            'neighborhood_analysis': self.neighborhood_analysis_,
            'distance_analysis': self.distance_analysis_,
            'outlier_detection': {
                'outlier_detection_enabled': self.outlier_detection,
                'n_outliers': int(np.sum(self.outlier_scores_ > np.percentile(self.outlier_scores_, 95))) if self.outlier_scores_ is not None else None,
                'outlier_percentage': float(np.mean(self.outlier_scores_ > np.percentile(self.outlier_scores_, 95)) * 100) if self.outlier_scores_ is not None else None
            },
            'calibration_info': {
                'probabilities_calibrated': self.calibrated_model_ is not None,
                'calibration_method': 'isotonic' if self.calibrated_model_ else None
            }
        }
        
        # Add K selection results if available
        if self.k_selection_scores_:
            summary['k_selection'] = {
                'k_range_tested': list(self.k_selection_scores_.keys()),
                'scores': self.k_selection_scores_,
                'optimal_k': self.optimal_k_,
                'optimal_score': self.k_selection_scores_.get(self.optimal_k_) if self.optimal_k_ else None
            }
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add KNN-specific information
        summary.update({
            'model_family': 'K-Nearest Neighbors',
            'n_neighbors': self.model.n_neighbors if self.is_fitted else self.n_neighbors,
            'optimal_k': self.optimal_k_,
            'distance_metric': self.metric,
            'weight_function': self.weights,
            'algorithm': self.algorithm,
            'feature_selection': self.feature_selection,
            'dimensionality_reduction': self.dimensionality_reduction,
            'probability_calibration': self.calibrate_probabilities,
            'outlier_detection': self.outlier_detection,
            'scaler_type': self.scaler_type,
            'n_classes': len(self.classes_) if hasattr(self, 'classes_') else None,
            'n_selected_features': len(self.selected_features_) if self.selected_features_ else len(self.feature_names),
            'final_dimensions': self.training_data_.shape[1] if self.training_data_ is not None else None,
            'auto_scaling': self.auto_scale
        })
        
        # Add neighborhood quality metrics
        if self.neighborhood_analysis_:
            summary.update({
                'mean_neighborhood_purity': self.neighborhood_analysis_['overall_purity']['mean'],
                'difficult_samples_ratio': (
                    self.neighborhood_analysis_['difficult_samples'] / len(self.training_data_)
                    if self.training_data_ is not None else None
                )
            })
        
        # Add distance metrics
        if self.distance_analysis_:
            summary.update({
                'class_separation_ratio': self.distance_analysis_['class_separation'].get('separation_ratio'),
                'mean_neighbor_distance': self.distance_analysis_['overall_statistics']['mean_distance_to_neighbors']
            })
        
        # Add KNN summary
        if self.is_fitted:
            try:
                summary['knn_summary'] = self.get_knn_summary()
            except Exception as e:
                logger.debug(f"Could not generate KNN summary: {e}")
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_knn_classifier(k: Union[int, str] = 'auto',
                         distance_metric: str = 'euclidean',
                         performance_preset: str = 'balanced',
                         **kwargs) -> FinancialKNNClassifier:
    """
    Create a K-Nearest Neighbors classification model
    
    Args:
        k: Number of neighbors ('auto' for automatic selection or integer)
        distance_metric: Distance metric ('euclidean', 'manhattan', 'minkowski', 'chebyshev')
        performance_preset: Performance preset ('fast', 'balanced', 'accurate')
        **kwargs: Additional model parameters
        
    Returns:
        Configured KNN classification model
    """
    
    # Base configuration
    base_config = {
        'name': f'knn_classifier_{distance_metric}',
        'metric': distance_metric,
        'auto_k': k == 'auto',
        'n_neighbors': 5 if k == 'auto' else k,
        'auto_scale': True,
        'scaler_type': 'standard',
        'calibrate_probabilities': True,
        'outlier_detection': True
    }
    
    # Performance presets
    if performance_preset == 'fast':
        preset_config = {
            'algorithm': 'kd_tree',
            'weights': 'uniform',
            'feature_selection': 'k_best',
            'selection_k': 15,
            'k_range': (3, 20),
            'outlier_detection': False
        }
    elif performance_preset == 'balanced':
        preset_config = {
            'algorithm': 'auto',
            'weights': 'distance',
            'feature_selection': 'k_best',
            'selection_k': 25,
            'k_range': (1, 30),
            'dimensionality_reduction': None
        }
    elif performance_preset == 'accurate':
        preset_config = {
            'algorithm': 'auto',
            'weights': 'distance',
            'feature_selection': 'k_best',
            'selection_k': 30,
            'dimensionality_reduction': 'pca',
            'n_components': None,  # Will be auto-determined
            'k_range': (1, 50),
            'scaler_type': 'robust'
        }
    else:
        raise ValueError(f"Unknown performance preset: {performance_preset}")
    
    # Distance metric specific adjustments
    if distance_metric == 'manhattan':
        preset_config.update({
            'p': 1,
            'metric': 'minkowski'
        })
    elif distance_metric == 'euclidean':
        preset_config.update({
            'p': 2,
            'metric': 'minkowski'
        })
    elif distance_metric == 'minkowski':
        preset_config.update({
            'p': 2  # Default to Euclidean
        })
    
    # Combine configurations
    config = {**base_config, **preset_config}
    config.update(kwargs)  # Override with user parameters
    
    return FinancialKNNClassifier(**config)

def create_binary_knn(k: Union[int, str] = 'auto', **kwargs) -> FinancialKNNClassifier:
    """Create KNN optimized for binary classification"""
    
    return create_knn_classifier(
        k=k,
        distance_metric='euclidean',
        performance_preset='balanced',
        name='binary_knn_classifier',
        **kwargs
    )

def create_multiclass_knn(k: Union[int, str] = 'auto', **kwargs) -> FinancialKNNClassifier:
    """Create KNN optimized for multiclass classification"""
    
    return create_knn_classifier(
        k=k,
        distance_metric='euclidean',
        performance_preset='accurate',
        weights='distance',  # Better for multiclass
        name='multiclass_knn_classifier',
        **kwargs
    )

def create_high_dimensional_knn(**kwargs) -> FinancialKNNClassifier:
    """Create KNN optimized for high-dimensional data"""
    
    return create_knn_classifier(
        k='auto',
        distance_metric='euclidean',
        performance_preset='accurate',
        feature_selection='k_best',
        selection_k=20,
        dimensionality_reduction='pca',
        scaler_type='robust',
        name='high_dim_knn_classifier',
        **kwargs
    )

def create_fast_knn(k: int = 5, **kwargs) -> FinancialKNNClassifier:
    """Create fast KNN for real-time predictions"""
    
    return create_knn_classifier(
        k=k,
        distance_metric='euclidean',
        performance_preset='fast',
        algorithm='kd_tree',
        name='fast_knn_classifier',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def tune_knn_hyperparameters(X: pd.DataFrame, y: pd.Series,
                            param_grid: Optional[Dict[str, List[Any]]] = None,
                            cv: int = 5,
                            scoring: str = 'accuracy',
                            n_jobs: int = -1) -> Dict[str, Any]:
    """
    Tune KNN hyperparameters using grid search
    
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
    
    logger.info("Starting KNN hyperparameter tuning")
    
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'kd_tree', 'ball_tree'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]  # For minkowski metric
        }
    
    # Create base model
    base_model = KNeighborsClassifier(n_jobs=n_jobs)
    
    # Scale data for KNN
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
        'scaler': scaler
    }
    
    logger.info(f"Hyperparameter tuning complete. Best score: {results['best_score']:.4f}")
    logger.info(f"Best parameters: {results['best_params']}")
    
    return results

def compare_distance_metrics(X: pd.DataFrame, y: pd.Series,
                           metrics: List[str] = ['euclidean', 'manhattan', 'chebyshev'],
                           k_values: List[int] = [3, 5, 7, 9],
                           cv: int = 5) -> Dict[str, Any]:
    """
    Compare different distance metrics for KNN
    
    Args:
        X: Feature matrix
        y: Target values
        metrics: List of distance metrics to compare
        k_values: List of K values to test
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with comparison results
    """
    
    from sklearn.model_selection import cross_val_score
    
    logger.info(f"Comparing KNN distance metrics: {metrics}")
    
    # Scale data once
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for metric in metrics:
        metric_results = {}
        
        for k in k_values:
            logger.info(f"Testing {metric} metric with K={k}")
            
            try:
                # Create model
                if metric in ['euclidean', 'manhattan']:
                    model = KNeighborsClassifier(
                        n_neighbors=k,
                        metric='minkowski',
                        p=2 if metric == 'euclidean' else 1,
                        weights='distance'
                    )
                else:
                    model = KNeighborsClassifier(
                        n_neighbors=k,
                        metric=metric,
                        weights='distance'
                    )
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
                
                # Fit for timing
                import time
                start_time = time.time()
                model.fit(X_scaled, y)
                fit_time = time.time() - start_time
                
                metric_results[f'k_{k}'] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist(),
                    'fit_time': fit_time
                }
                
            except Exception as e:
                logger.warning(f"Error with {metric} metric K={k}: {e}")
                metric_results[f'k_{k}'] = {'error': str(e)}
        
        results[metric] = metric_results
    
    # Find best combinations
    best_combinations = []
    for metric, metric_results in results.items():
        for k_config, k_results in metric_results.items():
            if 'error' not in k_results:
                best_combinations.append({
                    'metric': metric,
                    'k': int(k_config.split('_')[1]),
                    'cv_mean': k_results['cv_mean'],
                    'cv_std': k_results['cv_std'],
                    'fit_time': k_results['fit_time']
                })
    
    # Sort by accuracy
    best_combinations.sort(key=lambda x: x['cv_mean'], reverse=True)
    
    results['comparison'] = {
        'best_combination': best_combinations[0] if best_combinations else None,
        'top_5_combinations': best_combinations[:5],
        'best_metric': best_combinations[0]['metric'] if best_combinations else None,
        'best_k': best_combinations[0]['k'] if best_combinations else None
    }
    
    logger.info(f"Distance metric comparison complete. Best: {results['comparison']['best_metric']} with K={results['comparison']['best_k']}")
    
    return results

def analyze_curse_of_dimensionality(X: pd.DataFrame, y: pd.Series,
                                   max_dimensions: int = 50,
                                   k: int = 5) -> Dict[str, Any]:
    """
    Analyze the curse of dimensionality effect on KNN performance
    
    Args:
        X: Feature matrix
        y: Target values
        max_dimensions: Maximum number of dimensions to test
        k: Number of neighbors to use
        
    Returns:
        Dictionary with dimensionality analysis
    """
    
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import SelectKBest, f_classif
    
    logger.info(f"Analyzing curse of dimensionality for KNN (max dimensions: {max_dimensions})")
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different numbers of dimensions
    n_features_to_test = list(range(5, min(max_dimensions + 1, X.shape[1] + 1), 5))
    
    results = {
        'dimensions': n_features_to_test,
        'accuracies': [],
        'fit_times': [],
        'selected_features': []
    }
    
    for n_features in n_features_to_test:
        logger.info(f"Testing with {n_features} features")
        
        try:
            # Feature selection
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = selector.fit_transform(X_scaled, y)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_feature_names = [X.columns[i] for i in selected_indices]
            
            # Create and evaluate model
            model = KNeighborsClassifier(n_neighbors=k, weights='distance')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
            
            # Time fitting
            import time
            start_time = time.time()
            model.fit(X_selected, y)
            fit_time = time.time() - start_time
            
            results['accuracies'].append(cv_scores.mean())
            results['fit_times'].append(fit_time)
            results['selected_features'].append(selected_feature_names)
            
        except Exception as e:
            logger.warning(f"Error with {n_features} features: {e}")
            results['accuracies'].append(np.nan)
            results['fit_times'].append(np.nan)
            results['selected_features'].append([])
    
    # Find optimal number of features
    valid_accuracies = [acc for acc in results['accuracies'] if not np.isnan(acc)]
    if valid_accuracies:
        best_idx = np.argmax([acc for acc in results['accuracies'] if not np.isnan(acc)])
        results['optimal_dimensions'] = results['dimensions'][best_idx]
        results['optimal_accuracy'] = results['accuracies'][best_idx]
        results['optimal_features'] = results['selected_features'][best_idx]
    else:
        results['optimal_dimensions'] = None
        results['optimal_accuracy'] = None
        results['optimal_features'] = []
    
    # Calculate dimensionality effect metrics
    if len(valid_accuracies) > 1:
        max_acc = max(valid_accuracies)
        min_acc = min(valid_accuracies)
        results['accuracy_degradation'] = max_acc - min_acc
        results['relative_degradation'] = (max_acc - min_acc) / max_acc if max_acc > 0 else 0
    else:
        results['accuracy_degradation'] = 0
        results['relative_degradation'] = 0
    
    logger.info(f"Dimensionality analysis complete. Optimal dimensions: {results['optimal_dimensions']}")
    
    return results

def optimize_knn_for_data(X: pd.DataFrame, y: pd.Series,
                         test_preprocessing: bool = True,
                         test_dimensionality: bool = True) -> Dict[str, Any]:
    """
    Comprehensive optimization of KNN for given data
    
    Args:
        X: Feature matrix
        y: Target values
        test_preprocessing: Whether to test different preprocessing options
        test_dimensionality: Whether to test dimensionality reduction
        
    Returns:
        Dictionary with optimization results and recommendations
    """
    
    logger.info("Starting comprehensive KNN optimization")
    
    optimization_results = {}
    
    # 1. Basic hyperparameter tuning
    logger.info("Step 1: Basic hyperparameter tuning")
    hyperparameter_results = tune_knn_hyperparameters(X, y)
    optimization_results['hyperparameter_tuning'] = hyperparameter_results
    
    # 2. Distance metric comparison
    logger.info("Step 2: Distance metric comparison")
    distance_results = compare_distance_metrics(X, y)
    optimization_results['distance_comparison'] = distance_results
    
    # 3. Curse of dimensionality analysis
    if test_dimensionality:
        logger.info("Step 3: Dimensionality analysis")
        dimensionality_results = analyze_curse_of_dimensionality(X, y)
        optimization_results['dimensionality_analysis'] = dimensionality_results
    
    # 4. Test different preprocessing approaches
    if test_preprocessing:
        logger.info("Step 4: Preprocessing comparison")
        preprocessing_results = _compare_knn_preprocessing(X, y)
        optimization_results['preprocessing_comparison'] = preprocessing_results
    
    # 5. Create optimized model based on results
    best_params = hyperparameter_results['best_params'].copy()
    best_metric = distance_results['comparison']['best_metric']
    best_k = distance_results['comparison']['best_k']
    
    # Override with distance metric comparison results
    if best_metric == 'euclidean':
        best_params.update({'metric': 'minkowski', 'p': 2})
    elif best_metric == 'manhattan':
        best_params.update({'metric': 'minkowski', 'p': 1})
    else:
        best_params.update({'metric': best_metric})
    
    best_params['n_neighbors'] = best_k
    
    # Add dimensionality recommendations
    if test_dimensionality and 'optimal_dimensions' in dimensionality_results:
        if dimensionality_results['optimal_dimensions']:
            recommended_features = dimensionality_results['optimal_features']
        else:
            recommended_features = None
    else:
        recommended_features = None
    
    # Create optimized model
    optimized_model = FinancialKNNClassifier(**best_params)
    
    if recommended_features:
        # Use feature selection
        optimized_model.feature_selection = 'k_best'
        optimized_model.selection_k = len(recommended_features)
    
    optimized_model.fit(X, y)
    
    # Final evaluation
    final_metrics = optimized_model.evaluate(X, y)
    
    optimization_results['optimized_model'] = optimized_model
    optimization_results['optimization_summary'] = {
        'best_params': best_params,
        'best_metric': best_metric,
        'best_k': best_k,
        'recommended_features': recommended_features,
        'final_accuracy': final_metrics.accuracy,
        'final_f1_score': final_metrics.f1_score,
        'optimization_improvement': final_metrics.accuracy - 0.5  # Compared to random
    }
    
    logger.info(f"KNN optimization complete. Final accuracy: {final_metrics.accuracy:.4f}")
    
    return optimization_results

def _compare_knn_preprocessing(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Compare different preprocessing approaches for KNN"""
    
    from sklearn.model_selection import cross_val_score
    
    preprocessing_options = {
        'standard_scaling': StandardScaler(),
        'minmax_scaling': MinMaxScaler(),
        'robust_scaling': RobustScaler()
    }
    
    results = {}
    
    for name, scaler in preprocessing_options.items():
        try:
            X_scaled = scaler.fit_transform(X)
            
            # Test with optimal K=5
            model = KNeighborsClassifier(n_neighbors=5, weights='distance')
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Error with {name}: {e}")
            results[name] = {'error': str(e)}
    
    # Find best preprocessing
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_preprocessing = max(valid_results.keys(), key=lambda k: valid_results[k]['cv_mean'])
        results['best_preprocessing'] = best_preprocessing
        results['best_score'] = valid_results[best_preprocessing]['cv_mean']
    
    return results
