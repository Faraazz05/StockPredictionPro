# ============================================
# StockPredictionPro - src/models/base/base_regressor.py
# Base regressor interface for financial prediction models
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error, mean_squared_log_error
)
import warnings
from enum import Enum
from dataclasses import dataclass

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ...utils.config_loader import get
from ...utils.validators import ValidationResult
from ...utils.helpers import safe_divide
from .base_model import BaseModel, ModelMetadata, ModelStatus

logger = get_logger('models.base.regressor')

# ============================================
# Regression-Specific Types and Enums
# ============================================

class RegressionStrategy(Enum):
    """Regression prediction strategies"""
    PRICE_PREDICTION = "price_prediction"      # Direct price prediction
    RETURN_PREDICTION = "return_prediction"    # Return/change prediction
    VOLATILITY_PREDICTION = "volatility_prediction"  # Volatility forecasting
    MULTI_TARGET = "multi_target"              # Multiple targets (OHLC)

@dataclass
class RegressionMetrics:
    """Regression model performance metrics"""
    mse: float
    rmse: float
    mae: float
    r2_score: float
    explained_variance: float
    mape: Optional[float] = None
    max_error: Optional[float] = None
    mean_squared_log_error: Optional[float] = None
    
    # Financial-specific metrics
    directional_accuracy: Optional[float] = None
    prediction_bias: Optional[float] = None
    hit_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2_score': self.r2_score,
            'explained_variance': self.explained_variance,
            'mape': self.mape,
            'max_error': self.max_error,
            'mean_squared_log_error': self.mean_squared_log_error,
            'directional_accuracy': self.directional_accuracy,
            'prediction_bias': self.prediction_bias,
            'hit_ratio': self.hit_ratio
        }

@dataclass
class PredictionInterval:
    """Prediction with confidence intervals"""
    predictions: np.ndarray
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None
    prediction_std: Optional[np.ndarray] = None
    confidence_level: float = 0.95
    
    def get_prediction_ranges(self) -> np.ndarray:
        """Get prediction range widths"""
        if self.lower_bounds is not None and self.upper_bounds is not None:
            return self.upper_bounds - self.lower_bounds
        return None
    
    def is_within_bounds(self, actual_values: np.ndarray) -> np.ndarray:
        """Check if actual values are within prediction bounds"""
        if self.lower_bounds is None or self.upper_bounds is None:
            return np.ones(len(actual_values), dtype=bool)
        
        return (actual_values >= self.lower_bounds) & (actual_values <= self.upper_bounds)

# ============================================
# Base Financial Regressor
# ============================================

class BaseFinancialRegressor(BaseModel, RegressorMixin, ABC):
    """
    Abstract base class for financial regression models
    
    Features:
    - Financial domain-specific interface
    - Comprehensive regression metrics
    - Prediction intervals and uncertainty quantification
    - Cross-validation support
    - Feature importance analysis
    - Financial performance evaluation
    """
    
    def __init__(self,
                 name: str,
                 model_type: str = "regressor",
                 regression_strategy: RegressionStrategy = RegressionStrategy.PRICE_PREDICTION,
                 target_transform: Optional[str] = None,
                 prediction_horizon: int = 1,
                 uncertainty_estimation: bool = False,
                 **kwargs):
        """
        Initialize base financial regressor
        
        Args:
            name: Model name
            model_type: Specific model type
            regression_strategy: Regression approach
            target_transform: Target transformation ('log', 'sqrt', 'box-cox')
            prediction_horizon: Days ahead to predict
            uncertainty_estimation: Whether to estimate prediction uncertainty
            **kwargs: Additional model parameters
        """
        super().__init__(
            name=name,
            model_type=model_type,
            prediction_type='regression',
            **kwargs
        )
        
        self.regression_strategy = regression_strategy
        self.target_transform = target_transform
        self.prediction_horizon = prediction_horizon
        self.uncertainty_estimation = uncertainty_estimation
        
        # Regression-specific attributes
        self.target_transformer_: Optional[Any] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.prediction_std_: Optional[np.ndarray] = None
        
        # Performance tracking
        self.regression_metrics_: Optional[RegressionMetrics] = None
        self.cross_validation_scores_: Optional[Dict[str, np.ndarray]] = None
        
        # Financial metrics
        self.residual_analysis_: Optional[Dict[str, Any]] = None
        self.prediction_intervals_: Optional[Dict[str, float]] = None
        
        logger.debug(f"Initialized {self.name} regressor with {self.regression_strategy.value} strategy")
    
    def _validate_input_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> ValidationResult:
        """Validate input data for regression"""
        result = super()._validate_input_data(X, y)
        
        if y is not None:
            # Check target variable for regression
            if not pd.api.types.is_numeric_dtype(y):
                result.add_error("Target variable must be numeric for regression")
            
            # Check for infinite values
            if np.isinf(y).any():
                result.add_warning("Target variable contains infinite values")
            
            # Check target range
            if y.nunique() < 10:
                result.add_warning("Target variable has very few unique values - consider classification")
            
            # Check for extreme outliers
            if len(y) > 10:
                q1, q3 = y.quantile([0.25, 0.75])
                iqr = q3 - q1
                outlier_threshold = 3 * iqr
                outliers = ((y < (q1 - outlier_threshold)) | (y > (q3 + outlier_threshold))).sum()
                
                if outliers > len(y) * 0.1:  # More than 10% outliers
                    result.add_warning(f"High number of outliers in target variable: {outliers} ({outliers/len(y)*100:.1f}%)")
        
        return result
    
    def _preprocess_targets(self, y: pd.Series) -> np.ndarray:
        """Preprocess target variable for regression"""
        
        # Handle missing values
        if y.isnull().any():
            logger.warning("Target variable contains missing values - forward filling")
            y = y.fillna(method='ffill').fillna(method='bfill')
        
        # Apply target transformation if specified
        if self.target_transform == 'log':
            if (y <= 0).any():
                y = y + 1  # Shift to make all values positive
            y_transformed = np.log(y)
            
            # Store transformer for inverse transform
            self.target_transformer_ = {
                'type': 'log',
                'shift': 1 if (y <= 0).any() else 0
            }
            
        elif self.target_transform == 'sqrt':
            if (y < 0).any():
                logger.warning("Negative values found for sqrt transform - taking absolute value")
                y = np.abs(y)
            y_transformed = np.sqrt(y)
            
            self.target_transformer_ = {'type': 'sqrt'}
            
        elif self.target_transform == 'box-cox':
            try:
                from scipy import stats
                y_transformed, lambda_param = stats.boxcox(y + 1)  # +1 to handle zero values
                
                self.target_transformer_ = {
                    'type': 'box-cox',
                    'lambda': lambda_param,
                    'shift': 1
                }
            except Exception as e:
                logger.warning(f"Box-Cox transformation failed: {e}, using original values")
                y_transformed = y.values
                
        else:
            y_transformed = y.values
        
        return y_transformed
    
    def _inverse_transform_targets(self, y_transformed: np.ndarray) -> np.ndarray:
        """Inverse transform predictions back to original scale"""
        
        if self.target_transformer_ is None:
            return y_transformed
        
        try:
            transform_type = self.target_transformer_['type']
            
            if transform_type == 'log':
                y_original = np.exp(y_transformed)
                if self.target_transformer_.get('shift', 0) > 0:
                    y_original -= self.target_transformer_['shift']
                    
            elif transform_type == 'sqrt':
                y_original = np.square(y_transformed)
                
            elif transform_type == 'box-cox':
                from scipy import stats
                lambda_param = self.target_transformer_['lambda']
                shift = self.target_transformer_.get('shift', 0)
                
                y_original = stats.inv_boxcox(y_transformed, lambda_param)
                if shift > 0:
                    y_original -= shift
                    
            else:
                y_original = y_transformed
            
            return y_original
            
        except Exception as e:
            logger.warning(f"Inverse transform failed: {e}, returning transformed values")
            return y_transformed
    
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create the underlying model instance"""
        pass
    
    @time_it("regressor_fit", include_args=True)
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[np.ndarray] = None,
            **fit_params) -> 'BaseFinancialRegressor':
        """
        Fit the regression model
        
        Args:
            X: Feature matrix
            y: Target values
            sample_weight: Sample weights (optional)
            **fit_params: Additional fitting parameters
            
        Returns:
            Fitted model instance
        """
        logger.info(f"Fitting {self.name} regressor on {len(X)} samples with {X.shape[1]} features")
        
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
            self.target_name = y.name or 'target'
            
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
            self.training_duration = fit_duration
            
            # Post-training processing
            self._post_training_processing(X_processed, y_processed)
            
            # Update model metadata
            self.update_metadata({
                'training_samples': len(X),
                'training_features': X.shape[1],
                'training_duration_seconds': fit_duration,
                'target_name': self.target_name,
                'target_transform': self.target_transform,
                'target_range': [float(y.min()), float(y.max())]
            })
            
            # Evaluate on training data
            self._evaluate_training_performance(X_processed, y_processed, y)
            
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
        
        # Estimate prediction uncertainty if requested
        if self.uncertainty_estimation:
            try:
                self._estimate_prediction_uncertainty(X, y)
            except Exception as e:
                logger.warning(f"Failed to estimate prediction uncertainty: {e}")
    
    def _estimate_prediction_uncertainty(self, X: np.ndarray, y: np.ndarray):
        """Estimate prediction uncertainty using residual analysis"""
        
        # Make predictions on training data
        y_pred = self.model.predict(X)
        residuals = y - y_pred
        
        # Calculate residual statistics
        self.prediction_std_ = np.std(residuals)
        
        # Store residual analysis
        self.residual_analysis_ = {
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
            'residual_skew': float(pd.Series(residuals).skew()),
            'residual_kurt': float(pd.Series(residuals).kurtosis())
        }
        
        # Calculate prediction intervals
        self.prediction_intervals_ = {
            'std_1': self.prediction_std_,
            'std_2': 2 * self.prediction_std_,
            'percentile_90': float(np.percentile(np.abs(residuals), 90)),
            'percentile_95': float(np.percentile(np.abs(residuals), 95))
        }
    
    def _evaluate_training_performance(self, X: np.ndarray, y_processed: np.ndarray, y_original: pd.Series):
        """Evaluate model performance on training data"""
        
        try:
            # Make predictions (transformed scale)
            y_pred_transformed = self.model.predict(X)
            
            # Inverse transform predictions
            y_pred = self._inverse_transform_targets(y_pred_transformed)
            
            # Calculate metrics
            self.regression_metrics_ = self._calculate_regression_metrics(
                y_true=y_original.values,
                y_pred=y_pred
            )
            
            # Update training score (use R²)
            self.training_score = self.regression_metrics_.r2_score
            
        except Exception as e:
            logger.warning(f"Failed to evaluate training performance: {e}")
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
        """Calculate comprehensive regression metrics"""
        
        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)
        
        # Additional metrics
        mape = None
        max_err = max_error(y_true, y_pred)
        msle = None
        
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except Exception:
            pass
        
        try:
            if (y_true > 0).all() and (y_pred > 0).all():
                msle = mean_squared_log_error(y_true, y_pred)
        except Exception:
            pass
        
        # Financial-specific metrics
        directional_accuracy = None
        prediction_bias = None
        hit_ratio = None
        
        try:
            # Directional accuracy (for price predictions)
            if len(y_true) > 1:
                actual_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                if len(actual_direction) > 0:
                    directional_accuracy = np.mean(actual_direction == pred_direction)
            
            # Prediction bias
            prediction_bias = np.mean(y_pred - y_true)
            
            # Hit ratio (predictions within 5% of actual)
            tolerance = 0.05
            hits = np.abs((y_pred - y_true) / y_true) <= tolerance
            hit_ratio = np.mean(hits)
            
        except Exception as e:
            logger.debug(f"Could not calculate financial metrics: {e}")
        
        return RegressionMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            explained_variance=explained_var,
            mape=mape,
            max_error=max_err,
            mean_squared_log_error=msle,
            directional_accuracy=directional_accuracy,
            prediction_bias=prediction_bias,
            hit_ratio=hit_ratio
        )
    
    @time_it("regressor_predict", include_args=True)
    def predict(self, X: pd.DataFrame, 
                return_std: bool = False,
                return_intervals: bool = False,
                confidence_level: float = 0.95) -> Union[np.ndarray, PredictionInterval]:
        """
        Make predictions using the fitted model
        
        Args:
            X: Feature matrix for prediction
            return_std: Whether to return prediction standard deviation
            return_intervals: Whether to return prediction intervals
            confidence_level: Confidence level for intervals
            
        Returns:
            Predictions or PredictionInterval object
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        logger.debug(f"Making predictions for {len(X)} samples")
        
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            # Make predictions (in transformed space)
            predictions_transformed = self.model.predict(X_processed)
            
            # Inverse transform predictions
            predictions = self._inverse_transform_targets(predictions_transformed)
            
            # Calculate prediction intervals if requested
            if return_std or return_intervals:
                prediction_std = None
                lower_bounds = None
                upper_bounds = None
                
                if self.prediction_std_ is not None:
                    prediction_std = np.full(len(predictions), self.prediction_std_)
                    
                    if return_intervals:
                        from scipy import stats
                        z_score = stats.norm.ppf((1 + confidence_level) / 2)
                        margin = z_score * prediction_std
                        
                        lower_bounds = predictions - margin
                        upper_bounds = predictions + margin
                
                return PredictionInterval(
                    predictions=predictions,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    prediction_std=prediction_std,
                    confidence_level=confidence_level
                )
            
            # Log prediction
            self.log_prediction()
            
            return predictions
        
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            raise
    
    @time_it("regressor_evaluate")
    def evaluate(self, X: pd.DataFrame, y: pd.Series, 
                sample_weight: Optional[np.ndarray] = None) -> RegressionMetrics:
        """
        Evaluate model performance on test data
        
        Args:
            X: Test features
            y: True target values
            sample_weight: Sample weights (optional)
            
        Returns:
            Regression metrics
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before evaluation")
        
        logger.info(f"Evaluating {self.name} on {len(X)} test samples")
        
        # Make predictions
        predictions = self.predict(X)
        
        # Calculate metrics
        metrics = self._calculate_regression_metrics(y.values, predictions)
        
        # Update validation score
        self.validation_score = metrics.r2_score
        
        logger.info(f"Evaluation complete - R²: {metrics.r2_score:.4f}, RMSE: {metrics.rmse:.4f}")
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      cv: int = 5,
                      scoring: Union[str, List[str]] = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                      n_jobs: int = -1) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Target values
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
                    cv=KFold(n_splits=cv, shuffle=True, random_state=42),
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
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: Optional[str] = None) -> Any:
        """
        Plot predictions vs actual values
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 8))
            
            # Scatter plot
            plt.scatter(y_true, y_pred, alpha=0.6, color='blue', s=20)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Labels and title
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(title or f'Predictions vs Actual - {self.name}')
            plt.legend()
            
            # Add metrics text
            if hasattr(self, 'regression_metrics_') and self.regression_metrics_:
                metrics_text = f'R² = {self.regression_metrics_.r2_score:.3f}\nRMSE = {self.regression_metrics_.rmse:.3f}'
                plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Any:
        """
        Plot residuals analysis
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            residuals = y_true - y_pred
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Residuals vs predictions
            axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 0].axhline(y=0, color='r', linestyle='--')
            axes[0, 0].set_xlabel('Predicted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Predictions')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Histogram of residuals
            axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Residuals')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot of Residuals')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Residuals vs order
            axes[1, 1].plot(range(len(residuals)), residuals, 'o-', alpha=0.6, markersize=3)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Observation Order')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals vs Order')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Residual Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib/SciPy not available for plotting")
            return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add regression-specific information
        summary.update({
            'regression_strategy': self.regression_strategy.value,
            'target_transform': self.target_transform,
            'prediction_horizon': self.prediction_horizon,
            'uncertainty_estimation': self.uncertainty_estimation,
            'has_feature_importance': self.feature_importances_ is not None,
            'has_prediction_intervals': self.prediction_intervals_ is not None
        })
        
        # Add performance metrics
        if self.regression_metrics_:
            summary['performance_metrics'] = self.regression_metrics_.to_dict()
        
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
        
        # Add residual analysis
        if self.residual_analysis_:
            summary['residual_analysis'] = self.residual_analysis_
        
        return summary

# ============================================
# Utility Functions for Regression
# ============================================

def create_price_targets(prices: pd.Series, horizon: int = 1, 
                        strategy: str = 'direct') -> pd.Series:
    """
    Create regression targets from price series
    
    Args:
        prices: Price series (usually closing prices)
        horizon: Number of periods to look ahead
        strategy: Target creation strategy ('direct', 'return', 'log_return')
        
    Returns:
        Target values for regression
    """
    if strategy == 'direct':
        # Direct price prediction
        return prices.shift(-horizon)
    elif strategy == 'return':
        # Return prediction
        future_prices = prices.shift(-horizon)
        return (future_prices - prices) / prices
    elif strategy == 'log_return':
        # Log return prediction
        future_prices = prices.shift(-horizon)
        return np.log(future_prices / prices)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def calculate_prediction_intervals(predictions: np.ndarray, 
                                 residuals: np.ndarray,
                                 confidence_levels: List[float] = [0.68, 0.95]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calculate prediction intervals from residuals
    
    Args:
        predictions: Model predictions
        residuals: Training residuals
        confidence_levels: Confidence levels for intervals
        
    Returns:
        Dictionary of prediction intervals
    """
    from scipy import stats
    
    residual_std = np.std(residuals)
    intervals = {}
    
    for confidence_level in confidence_levels:
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * residual_std
        
        intervals[f'{confidence_level:.2f}'] = {
            'lower': predictions - margin,
            'upper': predictions + margin,
            'margin': margin
        }
    
    return intervals

def evaluate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy for price predictions
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Directional accuracy (0-1)
    """
    if len(y_true) < 2:
        return 0.0
    
    actual_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    return np.mean(actual_direction == pred_direction)

def calculate_hit_ratio(y_true: np.ndarray, y_pred: np.ndarray, 
                       tolerance: float = 0.05) -> float:
    """
    Calculate hit ratio (predictions within tolerance of actual)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        tolerance: Tolerance as fraction of actual value
        
    Returns:
        Hit ratio (0-1)
    """
    relative_errors = np.abs((y_pred - y_true) / y_true)
    return np.mean(relative_errors <= tolerance)
