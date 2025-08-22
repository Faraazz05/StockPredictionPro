# ============================================
# StockPredictionPro - src/models/regression/polynomial.py
# Polynomial regression models for financial prediction with non-linear feature transformations
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import itertools
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_regressor import BaseFinancialRegressor, RegressionStrategy

logger = get_logger('models.regression.polynomial')

# ============================================
# Polynomial Regression Model
# ============================================

class FinancialPolynomialRegression(BaseFinancialRegressor):
    """
    Polynomial regression model optimized for financial data
    
    Features:
    - Multiple polynomial degrees with automatic selection
    - Interaction terms and feature combinations
    - Regularization support (Ridge, Lasso) for high-degree polynomials
    - Feature importance analysis for polynomial terms
    - Cross-validation for degree selection
    - Overfitting detection and prevention
    """
    
    def __init__(self,
                 name: str = "polynomial_regression",
                 degree: Union[int, List[int]] = 2,
                 include_bias: bool = True,
                 interaction_only: bool = False,
                 include_interactions: bool = True,
                 regularization: Optional[str] = None,
                 alpha: float = 1.0,
                 auto_degree: bool = False,
                 max_degree: int = 4,
                 cross_validation_degree: bool = True,
                 feature_subset: Optional[List[str]] = None,
                 max_features: Optional[int] = None,
                 auto_scale: bool = True,
                 **kwargs):
        """
        Initialize Financial Polynomial Regression
        
        Args:
            name: Model name
            degree: Polynomial degree(s) to use
            include_bias: Whether to include bias column
            interaction_only: Whether to include only interaction features
            include_interactions: Whether to include interaction terms
            regularization: Regularization type ('ridge', 'lasso', None)
            alpha: Regularization strength
            auto_degree: Whether to automatically select best degree
            max_degree: Maximum degree to test in auto selection
            cross_validation_degree: Whether to use CV for degree selection
            feature_subset: Subset of features to apply polynomial transformation
            max_features: Maximum number of polynomial features to create
            auto_scale: Whether to automatically scale features
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="polynomial_regression",
            regression_strategy=RegressionStrategy.PRICE_PREDICTION,
            **kwargs
        )
        
        # Polynomial regression parameters
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.include_interactions = include_interactions
        self.regularization = regularization
        self.alpha = alpha
        self.auto_degree = auto_degree
        self.max_degree = max_degree
        self.cross_validation_degree = cross_validation_degree
        self.feature_subset = feature_subset
        self.max_features = max_features
        self.auto_scale = auto_scale
        
        # Polynomial-specific attributes
        self.poly_features_: Optional[PolynomialFeatures] = None
        self.scaler_: Optional[StandardScaler] = None
        self.polynomial_feature_names_: Optional[List[str]] = None
        self.selected_degree_: Optional[int] = None
        self.degree_selection_scores_: Optional[Dict[int, float]] = None
        self.feature_importance_poly_: Optional[Dict[str, float]] = None
        self.overfitting_analysis_: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized Polynomial Regression: {self.name} (degree={degree})")
    
    def _create_model(self) -> Union[LinearRegression, Ridge, Lasso]:
        """Create the underlying regression model with regularization"""
        
        if self.regularization == 'ridge':
            return Ridge(alpha=self.alpha, fit_intercept=not self.include_bias)
        elif self.regularization == 'lasso':
            return Lasso(alpha=self.alpha, fit_intercept=not self.include_bias, max_iter=2000)
        else:
            return LinearRegression(fit_intercept=not self.include_bias)
    
    def _create_polynomial_features(self, degree: int) -> PolynomialFeatures:
        """Create polynomial features transformer"""
        
        return PolynomialFeatures(
            degree=degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
    
    def _select_feature_subset(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select subset of features for polynomial transformation"""
        
        if self.feature_subset is None:
            return X
        
        # Select specified features
        available_features = [f for f in self.feature_subset if f in X.columns]
        if not available_features:
            logger.warning("None of the specified features found, using all features")
            return X
        
        logger.info(f"Using feature subset: {available_features}")
        return X[available_features]
    
    def _limit_polynomial_features(self, poly_features: PolynomialFeatures, 
                                  X_poly: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Limit number of polynomial features if specified"""
        
        feature_names = poly_features.get_feature_names_out(self.feature_names)
        
        if self.max_features is None or X_poly.shape[1] <= self.max_features:
            return X_poly, feature_names.tolist()
        
        logger.info(f"Limiting polynomial features from {X_poly.shape[1]} to {self.max_features}")
        
        # Select features based on variance (keep most variable features)
        feature_variances = np.var(X_poly, axis=0)
        top_indices = np.argsort(feature_variances)[-self.max_features:]
        
        X_poly_limited = X_poly[:, top_indices]
        feature_names_limited = [feature_names[i] for i in top_indices]
        
        return X_poly_limited, feature_names_limited
    
    def _select_optimal_degree(self, X: np.ndarray, y: np.ndarray) -> int:
        """Select optimal polynomial degree using cross-validation"""
        
        if isinstance(self.degree, int) and not self.auto_degree:
            return self.degree
        
        degrees_to_test = []
        
        if self.auto_degree:
            degrees_to_test = list(range(1, self.max_degree + 1))
        elif isinstance(self.degree, list):
            degrees_to_test = self.degree
        else:
            degrees_to_test = [self.degree]
        
        logger.info(f"Testing polynomial degrees: {degrees_to_test}")
        
        best_degree = degrees_to_test[0]
        best_score = -np.inf
        degree_scores = {}
        
        for degree in degrees_to_test:
            try:
                # Create polynomial features
                poly_features = self._create_polynomial_features(degree)
                X_poly = poly_features.fit_transform(X)
                
                # Limit features if necessary
                X_poly, _ = self._limit_polynomial_features(poly_features, X_poly)
                
                # Create model
                model = self._create_model()
                
                if self.cross_validation_degree:
                    # Use cross-validation
                    from sklearn.model_selection import cross_val_score
                    cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
                    score = cv_scores.mean()
                    
                    logger.debug(f"Degree {degree}: CV R² = {score:.4f} ± {cv_scores.std():.4f}")
                else:
                    # Use simple train score
                    model.fit(X_poly, y)
                    score = model.score(X_poly, y)
                    
                    logger.debug(f"Degree {degree}: Train R² = {score:.4f}")
                
                degree_scores[degree] = score
                
                if score > best_score:
                    best_score = score
                    best_degree = degree
                    
            except Exception as e:
                logger.warning(f"Error testing degree {degree}: {e}")
                degree_scores[degree] = -np.inf
        
        self.degree_selection_scores_ = degree_scores
        self.selected_degree_ = best_degree
        
        logger.info(f"Selected optimal polynomial degree: {best_degree} (score: {best_score:.4f})")
        
        return best_degree
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with polynomial transformation and scaling"""
        
        # Select feature subset if specified
        X_subset = self._select_feature_subset(X)
        
        # Basic preprocessing on subset
        X_processed = super()._preprocess_features(X_subset)
        
        # Apply polynomial transformation
        if self.poly_features_ is None:
            # During training - determine optimal degree and fit transformer
            optimal_degree = self._select_optimal_degree(X_processed, None)  # Will be called properly during fit
            self.poly_features_ = self._create_polynomial_features(optimal_degree)
            X_poly = self.poly_features_.fit_transform(X_processed)
            
            # Limit features if necessary
            X_poly, feature_names = self._limit_polynomial_features(self.poly_features_, X_poly)
            self.polynomial_feature_names_ = feature_names
            
        else:
            # During prediction - use fitted transformer
            X_poly = self.poly_features_.transform(X_processed)
            
            # Apply same feature limitation as during training
            if self.max_features is not None and X_poly.shape[1] > self.max_features:
                # Use the same features selected during training
                feature_names = self.poly_features_.get_feature_names_out(
                    X_subset.columns if hasattr(X_subset, 'columns') else 
                    [f'x{i}' for i in range(X_subset.shape[1])]
                )
                
                # Find indices of selected features
                selected_indices = []
                for selected_name in self.polynomial_feature_names_:
                    try:
                        idx = list(feature_names).index(selected_name)
                        selected_indices.append(idx)
                    except ValueError:
                        pass
                
                if selected_indices:
                    X_poly = X_poly[:, selected_indices]
        
        # Apply scaling if enabled
        if self.auto_scale:
            if self.scaler_ is None:
                self.scaler_ = StandardScaler()
                X_scaled = self.scaler_.fit_transform(X_poly)
                logger.debug("Fitted scaler for polynomial features")
            else:
                X_scaled = self.scaler_.transform(X_poly)
            
            return X_scaled
        
        return X_poly
    
    def _analyze_overfitting(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Analyze overfitting by comparing train and validation performance"""
        
        train_score = self.model.score(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            overfitting_gap = train_score - val_score
            
            # Determine overfitting severity
            if overfitting_gap > 0.2:
                severity = "High"
            elif overfitting_gap > 0.1:
                severity = "Moderate"
            elif overfitting_gap > 0.05:
                severity = "Low"
            else:
                severity = "None"
            
            self.overfitting_analysis_ = {
                'train_score': float(train_score),
                'validation_score': float(val_score),
                'overfitting_gap': float(overfitting_gap),
                'severity': severity,
                'degree': self.selected_degree_,
                'n_polynomial_features': len(self.polynomial_feature_names_) if self.polynomial_feature_names_ else 0
            }
            
            if severity in ["High", "Moderate"]:
                logger.warning(f"{severity} overfitting detected (gap: {overfitting_gap:.3f})")
        else:
            self.overfitting_analysis_ = {
                'train_score': float(train_score),
                'validation_score': None,
                'overfitting_gap': None,
                'severity': "Unknown",
                'degree': self.selected_degree_,
                'n_polynomial_features': len(self.polynomial_feature_names_) if self.polynomial_feature_names_ else 0
            }
    
    def _analyze_polynomial_importance(self):
        """Analyze importance of different polynomial terms"""
        
        if not hasattr(self.model, 'coef_') or self.polynomial_feature_names_ is None:
            return
        
        coefficients = self.model.coef_
        feature_names = self.polynomial_feature_names_
        
        # Categorize polynomial terms
        importance_by_type = {
            'linear': {},
            'quadratic': {},
            'cubic': {},
            'higher_order': {},
            'interactions': {}
        }
        
        for name, coef in zip(feature_names, coefficients):
            abs_coef = abs(coef)
            
            # Count degree of each variable in the term
            if '^' in name or ' ' in name:
                # Complex polynomial term
                if ' ' in name:
                    # Interaction term
                    importance_by_type['interactions'][name] = abs_coef
                else:
                    # Single variable with power
                    if '^2' in name:
                        importance_by_type['quadratic'][name] = abs_coef
                    elif '^3' in name:
                        importance_by_type['cubic'][name] = abs_coef
                    else:
                        importance_by_type['higher_order'][name] = abs_coef
            else:
                # Linear term or constant
                if name != '1':  # Not the intercept
                    importance_by_type['linear'][name] = abs_coef
        
        # Get top terms in each category
        self.feature_importance_poly_ = {}
        for term_type, terms in importance_by_type.items():
            if terms:
                sorted_terms = sorted(terms.items(), key=lambda x: x[1], reverse=True)
                self.feature_importance_poly_[term_type] = sorted_terms[:5]  # Top 5 in each category
    
    @time_it("polynomial_fit", include_args=True)
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'FinancialPolynomialRegression':
        """
        Fit the polynomial regression model
        
        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional fitting parameters
            
        Returns:
            Fitted model instance
        """
        logger.info(f"Fitting Polynomial Regression on {len(X)} samples with {X.shape[1]} features")
        
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
            
            # Preprocess targets
            y_processed = self._preprocess_targets(y)
            
            # Select optimal degree first (before feature preprocessing)
            X_subset = self._select_feature_subset(X)
            X_basic = super()._preprocess_features(X_subset)
            optimal_degree = self._select_optimal_degree(X_basic, y_processed)
            
            # Now preprocess features with the selected degree
            X_processed = self._preprocess_features(X)
            
            # Create model if not exists
            if self.model is None:
                self.model = self._create_model()
            
            # Fit the model
            fit_start = datetime.now()
            self.model.fit(X_processed, y_processed)
            fit_duration = (datetime.now() - fit_start).total_seconds()
            self.training_duration = fit_duration
            
            # Post-training analysis
            self._analyze_polynomial_importance()
            
            # Analyze overfitting (split data for validation)
            if len(X_processed) > 50:  # Only if we have enough data
                from sklearn.model_selection import train_test_split
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_processed, y_processed, test_size=0.2, random_state=42
                )
                
                # Refit on training split for overfitting analysis
                temp_model = self._create_model()
                temp_model.fit(X_train_split, y_train_split)
                
                # Analyze overfitting
                train_score = temp_model.score(X_train_split, y_train_split)
                val_score = temp_model.score(X_val_split, y_val_split)
                
                self.overfitting_analysis_ = {
                    'train_score': float(train_score),
                    'validation_score': float(val_score),
                    'overfitting_gap': float(train_score - val_score),
                    'severity': "High" if (train_score - val_score) > 0.2 else 
                               "Moderate" if (train_score - val_score) > 0.1 else
                               "Low" if (train_score - val_score) > 0.05 else "None",
                    'degree': self.selected_degree_,
                    'n_polynomial_features': len(self.polynomial_feature_names_) if self.polynomial_feature_names_ else 0
                }
            
            # Update model metadata
            self.update_metadata({
                'training_samples': len(X),
                'training_features': X.shape[1],
                'polynomial_degree': self.selected_degree_,
                'polynomial_features': len(self.polynomial_feature_names_) if self.polynomial_feature_names_ else 0,
                'training_duration_seconds': fit_duration,
                'target_name': self.target_name,
                'regularization': self.regularization
            })
            
            # Calculate training score
            self.training_score = self.model.score(X_processed, y_processed)
            
            # Update status
            self.status = ModelStatus.TRAINED
            self.is_fitted = True
            
            logger.info(f"Polynomial Regression trained successfully in {fit_duration:.2f}s")
            
            return self
            
        except Exception as e:
            from ..base.base_model import ModelStatus
            self.status = ModelStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Polynomial Regression training failed: {e}")
            raise
    
    def get_polynomial_terms_analysis(self) -> Dict[str, Any]:
        """
        Get analysis of polynomial terms importance
        
        Returns:
            Dictionary with polynomial terms analysis
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get polynomial terms analysis")
        
        analysis = {
            'selected_degree': self.selected_degree_,
            'total_polynomial_features': len(self.polynomial_feature_names_) if self.polynomial_feature_names_ else 0,
            'degree_selection_scores': self.degree_selection_scores_,
            'feature_importance_by_type': self.feature_importance_poly_,
            'overfitting_analysis': self.overfitting_analysis_
        }
        
        return analysis
    
    def get_polynomial_feature_names(self) -> List[str]:
        """
        Get names of polynomial features
        
        Returns:
            List of polynomial feature names
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get polynomial feature names")
        
        return self.polynomial_feature_names_.copy() if self.polynomial_feature_names_ else []
    
    def plot_degree_selection(self) -> Any:
        """
        Plot degree selection analysis
        
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.degree_selection_scores_:
                logger.warning("No degree selection scores available")
                return None
            
            degrees = list(self.degree_selection_scores_.keys())
            scores = list(self.degree_selection_scores_.values())
            
            plt.figure(figsize=(10, 6))
            plt.plot(degrees, scores, 'bo-', linewidth=2, markersize=8)
            
            # Highlight selected degree
            if self.selected_degree_ in degrees:
                selected_idx = degrees.index(self.selected_degree_)
                plt.plot(self.selected_degree_, scores[selected_idx], 'ro', 
                        markersize=12, label=f'Selected Degree: {self.selected_degree_}')
            
            plt.xlabel('Polynomial Degree')
            plt.ylabel('Cross-Validation R² Score')
            plt.title(f'Polynomial Degree Selection - {self.name}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add score values as text
            for degree, score in zip(degrees, scores):
                plt.text(degree, score + 0.01, f'{score:.3f}', 
                        ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_polynomial_importance(self) -> Any:
        """
        Plot polynomial terms importance
        
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.feature_importance_poly_:
                logger.warning("No polynomial importance analysis available")
                return None
            
            # Prepare data for plotting
            all_terms = []
            all_importances = []
            all_types = []
            
            colors = {
                'linear': 'blue',
                'quadratic': 'red',
                'cubic': 'green',
                'higher_order': 'orange',
                'interactions': 'purple'
            }
            
            for term_type, terms in self.feature_importance_poly_.items():
                for term_name, importance in terms:
                    all_terms.append(term_name)
                    all_importances.append(importance)
                    all_types.append(term_type)
            
            if not all_terms:
                logger.warning("No polynomial terms to plot")
                return None
            
            # Sort by importance
            sorted_data = sorted(zip(all_terms, all_importances, all_types), 
                               key=lambda x: x[1], reverse=True)
            
            # Take top terms
            top_n = min(20, len(sorted_data))
            top_terms, top_importances, top_types = zip(*sorted_data[:top_n])
            
            # Create plot
            fig, ax = plt.subplots(figsize=(14, 8))
            
            y_pos = np.arange(len(top_terms))
            bar_colors = [colors[t] for t in top_types]
            
            bars = ax.barh(y_pos, top_importances, color=bar_colors, alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_terms, fontsize=8)
            ax.set_xlabel('Absolute Coefficient Value')
            ax.set_title(f'Polynomial Terms Importance - {self.name}')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add importance values as text
            for i, (bar, importance) in enumerate(zip(bars, top_importances)):
                ax.text(importance + max(top_importances) * 0.01, i, f'{importance:.3f}', 
                       va='center', fontsize=8)
            
            # Create legend
            legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7, label=term_type.title()) 
                             for term_type, color in colors.items() 
                             if term_type in top_types]
            ax.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_overfitting_analysis(self) -> Any:
        """
        Plot overfitting analysis
        
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.overfitting_analysis_:
                logger.warning("No overfitting analysis available")
                return None
            
            analysis = self.overfitting_analysis_
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Train vs Validation Score
            if analysis['validation_score'] is not None:
                scores = [analysis['train_score'], analysis['validation_score']]
                labels = ['Training', 'Validation']
                colors = ['blue', 'red']
                
                bars = ax1.bar(labels, scores, color=colors, alpha=0.7)
                ax1.set_ylabel('R² Score')
                ax1.set_title('Training vs Validation Performance')
                ax1.set_ylim(0, max(scores) * 1.1)
                
                # Add values on bars
                for bar, score in zip(bars, scores):
                    ax1.text(bar.get_x() + bar.get_width()/2, score + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
                
                # Add gap information
                gap = analysis['overfitting_gap']
                ax1.text(0.5, max(scores) * 0.9, f'Overfitting Gap: {gap:.3f}', 
                        ha='center', transform=ax1.transData, 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            else:
                ax1.text(0.5, 0.5, 'Validation Score\nNot Available', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                ax1.set_title('Training vs Validation Performance')
            
            # Plot 2: Overfitting Severity
            severity = analysis['severity']
            degree = analysis['degree']
            n_features = analysis['n_polynomial_features']
            
            severity_colors = {
                'None': 'green',
                'Low': 'yellow',
                'Moderate': 'orange',
                'High': 'red',
                'Unknown': 'gray'
            }
            
            ax2.bar(['Overfitting\nSeverity'], [1], color=severity_colors.get(severity, 'gray'), 
                   alpha=0.7)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Severity Level')
            ax2.set_title('Overfitting Assessment')
            
            # Add text information
            info_text = f'Severity: {severity}\nDegree: {degree}\nFeatures: {n_features}'
            ax2.text(0, 0.5, info_text, ha='center', va='center', fontweight='bold')
            
            plt.suptitle(f'Overfitting Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def get_polynomial_summary(self) -> Dict[str, Any]:
        """Get comprehensive polynomial regression summary"""
        
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get polynomial summary")
        
        summary = {
            'polynomial_config': {
                'selected_degree': self.selected_degree_,
                'auto_degree': self.auto_degree,
                'max_degree': self.max_degree,
                'regularization': self.regularization,
                'alpha': self.alpha if self.regularization else None,
                'include_interactions': self.include_interactions,
                'interaction_only': self.interaction_only
            },
            'feature_analysis': {
                'original_features': len(self.feature_names),
                'polynomial_features': len(self.polynomial_feature_names_) if self.polynomial_feature_names_ else 0,
                'feature_subset': self.feature_subset,
                'max_features_limit': self.max_features
            },
            'degree_selection': {
                'degree_scores': self.degree_selection_scores_,
                'cross_validation_used': self.cross_validation_degree
            },
            'polynomial_importance': self.feature_importance_poly_,
            'overfitting_analysis': self.overfitting_analysis_
        }
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add polynomial-specific information
        summary.update({
            'regression_type': 'Polynomial Regression',
            'polynomial_degree': self.selected_degree_,
            'auto_degree_selection': self.auto_degree,
            'regularization': self.regularization,
            'regularization_alpha': self.alpha if self.regularization else None,
            'include_interactions': self.include_interactions,
            'polynomial_features_count': len(self.polynomial_feature_names_) if self.polynomial_feature_names_ else 0,
            'original_features_count': len(self.feature_names),
            'feature_expansion_ratio': (
                len(self.polynomial_feature_names_) / len(self.feature_names) 
                if self.polynomial_feature_names_ else 1.0
            )
        })
        
        # Add overfitting information
        if self.overfitting_analysis_:
            summary.update({
                'overfitting_severity': self.overfitting_analysis_['severity'],
                'overfitting_gap': self.overfitting_analysis_['overfitting_gap']
            })
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_polynomial_regression(degree: Union[int, str] = 'auto',
                                regularization: Optional[str] = None,
                                complexity: str = 'medium',
                                **kwargs) -> FinancialPolynomialRegression:
    """
    Create a Polynomial Regression model
    
    Args:
        degree: Polynomial degree ('auto' for automatic selection, int for fixed degree)
        regularization: Regularization type ('ridge', 'lasso', None)
        complexity: Model complexity ('low', 'medium', 'high')
        **kwargs: Additional model parameters
        
    Returns:
        Configured Polynomial Regression model
    """
    
    # Complexity presets
    complexity_configs = {
        'low': {
            'max_degree': 2,
            'max_features': 50,
            'alpha': 1.0
        },
        'medium': {
            'max_degree': 3,
            'max_features': 100,
            'alpha': 1.0
        },
        'high': {
            'max_degree': 4,
            'max_features': 200,
            'alpha': 0.1
        }
    }
    
    # Base configuration
    base_config = {
        'name': f'polynomial_regression_{complexity}',
        'auto_degree': degree == 'auto',
        'degree': 2 if degree == 'auto' else degree,
        'regularization': regularization,
        'include_bias': True,
        'include_interactions': True,
        'interaction_only': False,
        'cross_validation_degree': True,
        'auto_scale': True
    }
    
    # Apply complexity settings
    config = {**base_config, **complexity_configs.get(complexity, complexity_configs['medium'])}
    config.update(kwargs)  # Override with user parameters
    
    return FinancialPolynomialRegression(**config)

def create_quadratic_regression(regularization: Optional[str] = None, **kwargs) -> FinancialPolynomialRegression:
    """Create quadratic polynomial regression (degree 2)"""
    
    return create_polynomial_regression(
        degree=2,
        regularization=regularization,
        name='quadratic_regression',
        **kwargs
    )

def create_cubic_regression(regularization: str = 'ridge', **kwargs) -> FinancialPolynomialRegression:
    """Create cubic polynomial regression (degree 3) with Ridge regularization"""
    
    return create_polynomial_regression(
        degree=3,
        regularization=regularization,
        complexity='medium',
        name='cubic_regression',
        **kwargs
    )

def create_interaction_regression(degree: int = 2, **kwargs) -> FinancialPolynomialRegression:
    """Create polynomial regression with interaction terms only"""
    
    return create_polynomial_regression(
        degree=degree,
        name='interaction_regression',
        interaction_only=True,
        regularization='lasso',
        **kwargs
    )

def create_regularized_polynomial(degree: Union[int, str] = 'auto',
                                 regularization: str = 'ridge',
                                 **kwargs) -> FinancialPolynomialRegression:
    """Create polynomial regression with regularization"""
    
    return create_polynomial_regression(
        degree=degree,
        regularization=regularization,
        complexity='high',
        name=f'regularized_polynomial_{regularization}',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def find_optimal_polynomial_degree(X: pd.DataFrame, y: pd.Series,
                                  max_degree: int = 5,
                                  regularization: Optional[str] = None,
                                  cv: int = 5) -> Dict[str, Any]:
    """
    Find optimal polynomial degree using cross-validation
    
    Args:
        X: Feature matrix
        y: Target values
        max_degree: Maximum degree to test
        regularization: Regularization type
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with optimal degree and scores
    """
    
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    logger.info(f"Finding optimal polynomial degree (max degree: {max_degree})")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {
        'degrees': [],
        'train_scores': [],
        'cv_scores_mean': [],
        'cv_scores_std': [],
        'n_features': []
    }
    
    for degree in range(1, max_degree + 1):
        logger.info(f"Testing degree {degree}")
        
        try:
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            X_poly = poly.fit_transform(X_scaled)
            
            # Create model
            if regularization == 'ridge':
                model = Ridge(alpha=1.0)
            elif regularization == 'lasso':
                model = Lasso(alpha=1.0, max_iter=2000)
            else:
                model = LinearRegression()
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_poly, y, cv=cv, scoring='r2')
            
            # Training score
            model.fit(X_poly, y)
            train_score = model.score(X_poly, y)
            
            results['degrees'].append(degree)
            results['train_scores'].append(train_score)
            results['cv_scores_mean'].append(cv_scores.mean())
            results['cv_scores_std'].append(cv_scores.std())
            results['n_features'].append(X_poly.shape[1])
            
            logger.debug(f"Degree {degree}: CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}, "
                        f"Features = {X_poly.shape[1]}")
            
        except Exception as e:
            logger.warning(f"Error testing degree {degree}: {e}")
    
    if results['cv_scores_mean']:
        # Find optimal degree
        best_idx = np.argmax(results['cv_scores_mean'])
        optimal_degree = results['degrees'][best_idx]
        optimal_score = results['cv_scores_mean'][best_idx]
        
        results.update({
            'optimal_degree': optimal_degree,
            'optimal_score': optimal_score,
            'optimal_n_features': results['n_features'][best_idx]
        })
        
        logger.info(f"Optimal polynomial degree: {optimal_degree} (CV R²: {optimal_score:.4f})")
    
    return results

def analyze_polynomial_overfitting(X: pd.DataFrame, y: pd.Series,
                                  degrees: List[int] = [1, 2, 3, 4],
                                  test_size: float = 0.2) -> Dict[str, Any]:
    """
    Analyze overfitting for different polynomial degrees
    
    Args:
        X: Feature matrix
        y: Target values
        degrees: List of degrees to analyze
        test_size: Fraction of data for testing
        
    Returns:
        Dictionary with overfitting analysis
    """
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    logger.info(f"Analyzing polynomial overfitting for degrees: {degrees}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    analysis = {
        'degrees': degrees,
        'train_scores': [],
        'test_scores': [],
        'overfitting_gaps': [],
        'n_features': [],
        'complexity_scores': []
    }
    
    for degree in degrees:
        try:
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            X_train_poly = poly.fit_transform(X_train_scaled)
            X_test_poly = poly.transform(X_test_scaled)
            
            # Fit model
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            
            # Calculate scores
            train_score = model.score(X_train_poly, y_train)
            test_score = model.score(X_test_poly, y_test)
            overfitting_gap = train_score - test_score
            
            # Complexity score (penalize high degree and overfitting)
            complexity_penalty = degree * 0.01 + max(0, overfitting_gap) * 2
            complexity_score = test_score - complexity_penalty
            
            analysis['train_scores'].append(train_score)
            analysis['test_scores'].append(test_score)
            analysis['overfitting_gaps'].append(overfitting_gap)
            analysis['n_features'].append(X_train_poly.shape[1])
            analysis['complexity_scores'].append(complexity_score)
            
            logger.debug(f"Degree {degree}: Train={train_score:.4f}, Test={test_score:.4f}, "
                        f"Gap={overfitting_gap:.4f}, Features={X_train_poly.shape[1]}")
            
        except Exception as e:
            logger.warning(f"Error analyzing degree {degree}: {e}")
    
    # Find recommended degree
    if analysis['complexity_scores']:
        best_idx = np.argmax(analysis['complexity_scores'])
        analysis['recommended_degree'] = analysis['degrees'][best_idx]
        analysis['recommended_score'] = analysis['complexity_scores'][best_idx]
    
    # Identify overfitting issues
    severe_overfitting = [
        (deg, gap) for deg, gap in zip(analysis['degrees'], analysis['overfitting_gaps'])
        if gap > 0.2
    ]
    
    analysis['severe_overfitting'] = severe_overfitting
    analysis['overfitting_detected'] = len(severe_overfitting) > 0
    
    logger.info(f"Overfitting analysis complete. Recommended degree: {analysis.get('recommended_degree', 'None')}")
    
    return analysis

def compare_polynomial_regularization(X: pd.DataFrame, y: pd.Series,
                                    degree: int = 3,
                                    regularization_methods: List[str] = ['none', 'ridge', 'lasso']) -> pd.DataFrame:
    """
    Compare different regularization methods for polynomial regression
    
    Args:
        X: Feature matrix
        y: Target values
        degree: Polynomial degree to use
        regularization_methods: List of regularization methods to compare
        
    Returns:
        DataFrame with comparison results
    """
    
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    logger.info(f"Comparing polynomial regularization methods for degree {degree}")
    
    # Prepare data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_scaled)
    
    results = []
    
    for method in regularization_methods:
        try:
            # Create model
            if method == 'ridge':
                model = Ridge(alpha=1.0)
            elif method == 'lasso':
                model = Lasso(alpha=1.0, max_iter=2000)
            else:  # 'none'
                model = LinearRegression()
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
            
            # Fit for additional metrics
            model.fit(X_poly, y)
            train_score = model.score(X_poly, y)
            
            # Count non-zero coefficients (sparsity)
            if hasattr(model, 'coef_'):
                n_nonzero_coef = np.sum(np.abs(model.coef_) > 1e-6)
                sparsity = 1 - (n_nonzero_coef / len(model.coef_))
            else:
                n_nonzero_coef = len(model.coef_) if hasattr(model, 'coef_') else X_poly.shape[1]
                sparsity = 0.0
            
            results.append({
                'regularization': method,
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'train_score': train_score,
                'overfitting_gap': train_score - cv_scores.mean(),
                'n_nonzero_coef': n_nonzero_coef,
                'sparsity': sparsity,
                'total_features': X_poly.shape[1]
            })
            
        except Exception as e:
            logger.warning(f"Error with {method} regularization: {e}")
    
    comparison_df = pd.DataFrame(results)
    
    # Sort by CV score
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values('cv_score_mean', ascending=False)
    
    logger.info("Polynomial regularization comparison complete")
    
    return comparison_df

def create_polynomial_features_analysis(X: pd.DataFrame, degree: int = 2) -> Dict[str, Any]:
    """
    Analyze polynomial feature creation
    
    Args:
        X: Feature matrix
        degree: Polynomial degree
        
    Returns:
        Dictionary with feature analysis
    """
    
    logger.info(f"Analyzing polynomial feature creation (degree {degree})")
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=True, interaction_only=False)
    X_poly = poly.fit_transform(X)
    
    # Get feature names
    feature_names = poly.get_feature_names_out(X.columns)
    
    # Categorize features
    feature_types = {
        'constant': [],
        'linear': [],
        'quadratic': [],
        'cubic': [],
        'higher_order': [],
        'interactions': []
    }
    
    for name in feature_names:
        if name == '1':
            feature_types['constant'].append(name)
        elif '^' not in name and ' ' not in name:
            feature_types['linear'].append(name)
        elif ' ' in name:
            feature_types['interactions'].append(name)
        elif '^2' in name:
            feature_types['quadratic'].append(name)
        elif '^3' in name:
            feature_types['cubic'].append(name)
        else:
            feature_types['higher_order'].append(name)
    
    # Calculate statistics
    analysis = {
        'original_features': X.shape[1],
        'polynomial_features': X_poly.shape[1],
        'expansion_factor': X_poly.shape[1] / X.shape[1],
        'degree': degree,
        'feature_types_count': {k: len(v) for k, v in feature_types.items()},
        'feature_types': feature_types,
        'feature_names': feature_names.tolist()
    }
    
    # Memory usage estimate
    memory_mb = X_poly.nbytes / (1024 * 1024)
    analysis['memory_usage_mb'] = memory_mb
    
    logger.info(f"Feature analysis complete: {X.shape[1]} → {X_poly.shape[1]} features "
               f"({analysis['expansion_factor']:.1f}x expansion)")
    
    return analysis
