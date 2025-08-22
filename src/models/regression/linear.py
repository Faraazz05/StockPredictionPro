# ============================================
# StockPredictionPro - src/models/regression/linear.py
# Linear regression models for financial prediction
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_regressor import BaseFinancialRegressor, RegressionStrategy

logger = get_logger('models.regression.linear')

# ============================================
# Linear Regression Model
# ============================================

class FinancialLinearRegression(BaseFinancialRegressor):
    """
    Linear regression model optimized for financial data
    
    Features:
    - Multiple linear regression variants
    - Automatic feature scaling
    - Robust regression options
    - Multicollinearity detection
    - Financial domain optimizations
    """
    
    def __init__(self,
                 name: str = "linear_regression",
                 fit_intercept: bool = True,
                 normalize: bool = False,
                 copy_X: bool = True,
                 n_jobs: Optional[int] = None,
                 positive: bool = False,
                 auto_scale: bool = True,
                 robust_variant: str = 'standard',
                 **kwargs):
        """
        Initialize Financial Linear Regression
        
        Args:
            name: Model name
            fit_intercept: Whether to calculate intercept
            normalize: Whether to normalize features (deprecated in sklearn)
            copy_X: Whether to copy X or overwrite
            n_jobs: Number of parallel jobs
            positive: Whether to force positive coefficients
            auto_scale: Whether to automatically scale features
            robust_variant: Type of regression ('standard', 'huber', 'theil_sen')
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="linear_regression",
            regression_strategy=RegressionStrategy.PRICE_PREDICTION,
            **kwargs
        )
        
        # Linear regression parameters
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        self.auto_scale = auto_scale
        self.robust_variant = robust_variant
        
        # Store parameters for model creation
        self.model_params.update({
            'fit_intercept': fit_intercept,
            'copy_X': copy_X,
            'n_jobs': n_jobs,
            'positive': positive
        })
        
        # Additional attributes
        self.scaler_: Optional[StandardScaler] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.multicollinearity_stats_: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized {self.robust_variant} linear regression: {self.name}")
    
    def _create_model(self) -> Union[LinearRegression, HuberRegressor, TheilSenRegressor]:
        """Create the linear regression model"""
        
        if self.robust_variant == 'standard':
            return LinearRegression(**self.model_params)
        
        elif self.robust_variant == 'huber':
            # Huber regressor for robustness to outliers
            huber_params = {
                'fit_intercept': self.fit_intercept,
                'alpha': self.model_params.get('alpha', 0.0001),
                'max_iter': self.model_params.get('max_iter', 100),
                'tol': self.model_params.get('tol', 1e-05),
                'epsilon': self.model_params.get('epsilon', 1.35)
            }
            return HuberRegressor(**huber_params)
        
        elif self.robust_variant == 'theil_sen':
            # Theil-Sen regressor for robustness
            theil_sen_params = {
                'fit_intercept': self.fit_intercept,
                'copy_X': self.copy_X,
                'max_subpopulation': self.model_params.get('max_subpopulation', int(1e4)),
                'n_subsamples': self.model_params.get('n_subsamples', None),
                'max_iter': self.model_params.get('max_iter', 300),
                'tol': self.model_params.get('tol', 1e-3),
                'random_state': self.model_params.get('random_state', 42)
            }
            return TheilSenRegressor(**theil_sen_params)
        
        else:
            raise ValueError(f"Unknown robust_variant: {self.robust_variant}")
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with optional scaling"""
        
        # Basic preprocessing
        X_processed = super()._preprocess_features(X)
        
        # Apply feature scaling if enabled
        if self.auto_scale:
            if self.scaler_ is None:
                self.scaler_ = StandardScaler()
                X_scaled = self.scaler_.fit_transform(X_processed)
                logger.debug("Fitted feature scaler for linear regression")
            else:
                X_scaled = self.scaler_.transform(X_processed)
            
            return X_scaled
        
        return X_processed
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Enhanced post-training processing for linear regression"""
        
        # Call parent processing
        super()._post_training_processing(X, y)
        
        # Extract linear regression coefficients
        self.coefficients_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        # Set feature importances (absolute coefficients)
        self.feature_importances_ = np.abs(self.coefficients_)
        
        # Analyze multicollinearity
        self._analyze_multicollinearity(X)
        
        # Additional linear regression diagnostics
        self._calculate_regression_diagnostics(X, y)
    
    def _analyze_multicollinearity(self, X: np.ndarray):
        """Analyze multicollinearity in features"""
        
        try:
            # Calculate condition number
            if X.shape[1] > 1:
                # Add intercept column if fit_intercept is True
                if self.fit_intercept:
                    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                else:
                    X_with_intercept = X
                
                # Calculate condition number
                _, s, _ = np.linalg.svd(X_with_intercept, full_matrices=False)
                condition_number = s[0] / s[-1] if len(s) > 0 and s[-1] != 0 else np.inf
                
                # Calculate VIF (Variance Inflation Factor) approximation
                # For computational efficiency, use correlation matrix
                try:
                    corr_matrix = np.corrcoef(X.T)
                    if not np.any(np.isnan(corr_matrix)):
                        eigenvals = np.linalg.eigvals(corr_matrix)
                        min_eigenval = np.min(eigenvals)
                        max_vif_approx = 1.0 / min_eigenval if min_eigenval > 1e-10 else np.inf
                    else:
                        max_vif_approx = np.inf
                except:
                    max_vif_approx = np.inf
                
                self.multicollinearity_stats_ = {
                    'condition_number': float(condition_number),
                    'max_vif_approx': float(max_vif_approx),
                    'multicollinearity_risk': 'high' if condition_number > 30 else 'medium' if condition_number > 15 else 'low'
                }
                
                # Log warning for high multicollinearity
                if condition_number > 30:
                    logger.warning(f"High multicollinearity detected (condition number: {condition_number:.2f})")
                    
        except Exception as e:
            logger.debug(f"Could not analyze multicollinearity: {e}")
            self.multicollinearity_stats_ = None
    
    def _calculate_regression_diagnostics(self, X: np.ndarray, y: np.ndarray):
        """Calculate additional regression diagnostics"""
        
        try:
            # Make predictions for diagnostics
            y_pred = self.model.predict(X)
            residuals = y - y_pred
            
            # Calculate diagnostic statistics
            n = len(y)
            p = X.shape[1]
            
            # Degrees of freedom
            df_residual = n - p - (1 if self.fit_intercept else 0)
            
            # Standard error of regression
            mse = np.mean(residuals ** 2)
            std_error = np.sqrt(mse)
            
            # Adjusted R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            if df_residual > 0:
                adj_r2 = 1 - (ss_res / df_residual) / (ss_tot / (n - 1))
            else:
                adj_r2 = r2
            
            # F-statistic
            if p > 0 and df_residual > 0:
                mse_model = ((ss_tot - ss_res) / p)
                mse_residual = ss_res / df_residual
                f_statistic = mse_model / mse_residual if mse_residual > 0 else np.inf
            else:
                f_statistic = np.inf
            
            # Store diagnostics
            self.regression_diagnostics_ = {
                'degrees_of_freedom': int(df_residual),
                'standard_error': float(std_error),
                'adjusted_r2': float(adj_r2),
                'f_statistic': float(f_statistic),
                'aic': float(n * np.log(ss_res / n) + 2 * (p + 1)),
                'bic': float(n * np.log(ss_res / n) + np.log(n) * (p + 1))
            }
            
        except Exception as e:
            logger.debug(f"Could not calculate regression diagnostics: {e}")
            self.regression_diagnostics_ = None
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients with feature names
        
        Returns:
            DataFrame with coefficients and statistics
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get coefficients")
        
        if self.coefficients_ is None:
            raise BusinessLogicError("Coefficients not available")
        
        # Create coefficients DataFrame
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.coefficients_,
            'abs_coefficient': np.abs(self.coefficients_)
        })
        
        # Add intercept if fitted
        if self.fit_intercept and self.intercept_ is not None:
            intercept_row = pd.DataFrame({
                'feature': ['intercept'],
                'coefficient': [self.intercept_],
                'abs_coefficient': [abs(self.intercept_)]
            })
            coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
        
        # Sort by absolute coefficient value
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        return coef_df
    
    def predict_with_components(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions with component breakdown
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with predictions and components
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted before making predictions")
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # Calculate component contributions
        feature_contributions = X_processed * self.coefficients_
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        result = {
            'predictions': predictions,
            'feature_contributions': feature_contributions,
            'intercept_contribution': np.full(len(X), self.intercept_) if self.fit_intercept else np.zeros(len(X)),
            'total_feature_contribution': np.sum(feature_contributions, axis=1)
        }
        
        # Inverse transform if target transformation was applied
        if self.target_transformer_ is not None:
            result['predictions'] = self._inverse_transform_targets(result['predictions'])
        
        return result
    
    def plot_coefficients(self, top_n: int = 20) -> Any:
        """
        Plot model coefficients
        
        Args:
            top_n: Number of top coefficients to plot
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            coef_df = self.get_coefficients()
            
            # Select top coefficients
            if len(coef_df) > top_n:
                coef_df = coef_df.head(top_n)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Color positive and negative coefficients differently
            colors = ['red' if x < 0 else 'blue' for x in coef_df['coefficient']]
            
            bars = plt.barh(range(len(coef_df)), coef_df['coefficient'], color=colors, alpha=0.7)
            
            # Customize plot
            plt.yticks(range(len(coef_df)), coef_df['feature'])
            plt.xlabel('Coefficient Value')
            plt.title(f'Linear Regression Coefficients - {self.name}')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            
            # Add coefficient values as text
            for i, (bar, coef) in enumerate(zip(bars, coef_df['coefficient'])):
                plt.text(coef + (0.01 * np.sign(coef) if coef != 0 else 0.01), i, 
                        f'{coef:.3f}', ha='left' if coef >= 0 else 'right', 
                        va='center', fontsize=8)
            
            plt.tight_layout()
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_regression_diagnostics(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Plot comprehensive regression diagnostics
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from scipy import stats
            
            # Make predictions
            predictions = self.predict(X)
            residuals = y.values - predictions
            
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Predictions vs Actual
            axes[0, 0].scatter(y, predictions, alpha=0.6, color='blue')
            min_val, max_val = min(y.min(), predictions.min()), max(y.max(), predictions.max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Predictions vs Actual')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Residuals vs Fitted
            axes[0, 1].scatter(predictions, residuals, alpha=0.6, color='green')
            axes[0, 1].axhline(y=0, color='red', linestyle='--')
            axes[0, 1].set_xlabel('Fitted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals vs Fitted')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Q-Q plot of residuals
            stats.probplot(residuals, dist="norm", plot=axes[0, 2])
            axes[0, 2].set_title('Q-Q Plot of Residuals')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Histogram of residuals
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Residuals')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Scale-Location plot
            sqrt_abs_residuals = np.sqrt(np.abs(residuals))
            axes[1, 1].scatter(predictions, sqrt_abs_residuals, alpha=0.6, color='orange')
            axes[1, 1].set_xlabel('Fitted Values')
            axes[1, 1].set_ylabel('√|Residuals|')
            axes[1, 1].set_title('Scale-Location Plot')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Feature importance (coefficients)
            if len(self.feature_names) <= 15:  # Only if not too many features
                coef_df = self.get_coefficients()
                if 'intercept' in coef_df['feature'].values:
                    coef_df = coef_df[coef_df['feature'] != 'intercept']
                
                colors = ['red' if x < 0 else 'blue' for x in coef_df['coefficient']]
                axes[1, 2].barh(range(len(coef_df)), coef_df['coefficient'], color=colors, alpha=0.7)
                axes[1, 2].set_yticks(range(len(coef_df)))
                axes[1, 2].set_yticklabels(coef_df['feature'], fontsize=8)
                axes[1, 2].set_xlabel('Coefficient Value')
                axes[1, 2].set_title('Feature Coefficients')
                axes[1, 2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                axes[1, 2].grid(True, alpha=0.3)
            else:
                # Too many features - show summary statistics instead
                axes[1, 2].text(0.1, 0.7, f'Number of features: {len(self.feature_names)}', 
                               transform=axes[1, 2].transAxes, fontsize=12)
                if hasattr(self, 'regression_diagnostics_') and self.regression_diagnostics_:
                    diag = self.regression_diagnostics_
                    text = f"Adj. R²: {diag['adjusted_r2']:.3f}\n"
                    text += f"F-statistic: {diag['f_statistic']:.2f}\n"
                    text += f"AIC: {diag['aic']:.2f}\n"
                    text += f"BIC: {diag['bic']:.2f}"
                    axes[1, 2].text(0.1, 0.3, text, transform=axes[1, 2].transAxes, fontsize=10)
                axes[1, 2].set_title('Model Statistics')
                axes[1, 2].set_xticks([])
                axes[1, 2].set_yticks([])
            
            plt.suptitle(f'Linear Regression Diagnostics - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib/SciPy not available for plotting")
            return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add linear regression specific information
        summary.update({
            'robust_variant': self.robust_variant,
            'fit_intercept': self.fit_intercept,
            'auto_scale': self.auto_scale,
            'positive_coefficients': self.positive,
            'intercept': float(self.intercept_) if self.intercept_ is not None else None,
            'n_coefficients': len(self.coefficients_) if self.coefficients_ is not None else None
        })
        
        # Add multicollinearity analysis
        if self.multicollinearity_stats_:
            summary['multicollinearity'] = self.multicollinearity_stats_
        
        # Add regression diagnostics
        if hasattr(self, 'regression_diagnostics_') and self.regression_diagnostics_:
            summary['diagnostics'] = self.regression_diagnostics_
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_linear_regressor(variant: str = 'standard', **kwargs) -> FinancialLinearRegression:
    """
    Create a linear regression model
    
    Args:
        variant: Type of linear regression ('standard', 'huber', 'theil_sen')
        **kwargs: Additional model parameters
        
    Returns:
        Configured linear regression model
    """
    
    if variant == 'standard':
        default_params = {
            'name': 'linear_regression',
            'robust_variant': 'standard',
            'auto_scale': True,
            'fit_intercept': True
        }
    elif variant == 'huber':
        default_params = {
            'name': 'huber_regression',
            'robust_variant': 'huber',
            'auto_scale': True,
            'fit_intercept': True,
            'alpha': 0.0001,
            'epsilon': 1.35
        }
    elif variant == 'theil_sen':
        default_params = {
            'name': 'theil_sen_regression',
            'robust_variant': 'theil_sen',
            'auto_scale': True,
            'fit_intercept': True,
            'max_iter': 300,
            'random_state': 42
        }
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    # Override with provided kwargs
    default_params.update(kwargs)
    
    return FinancialLinearRegression(**default_params)

def create_robust_linear_regressor(**kwargs) -> FinancialLinearRegression:
    """Create a robust linear regression model (Huber)"""
    return create_linear_regressor('huber', **kwargs)

def create_outlier_resistant_regressor(**kwargs) -> FinancialLinearRegression:
    """Create an outlier-resistant regression model (Theil-Sen)"""
    return create_linear_regressor('theil_sen', **kwargs)

# ============================================
# Utility Functions
# ============================================

def analyze_linear_assumptions(X: pd.DataFrame, y: pd.Series, 
                             model: Optional[FinancialLinearRegression] = None) -> Dict[str, Any]:
    """
    Analyze linear regression assumptions
    
    Args:
        X: Feature matrix
        y: Target values
        model: Fitted model (optional)
        
    Returns:
        Dictionary with assumption analysis
    """
    
    results = {
        'linearity': None,
        'independence': None,
        'homoscedasticity': None,
        'normality': None,
        'multicollinearity': None
    }
    
    try:
        from scipy import stats
        
        # If model is provided, use its predictions
        if model and model.is_fitted:
            y_pred = model.predict(X)
            residuals = y.values - y_pred
        else:
            # Fit a simple linear regression for analysis
            from sklearn.linear_model import LinearRegression
            temp_model = LinearRegression()
            X_processed = StandardScaler().fit_transform(X.values)
            temp_model.fit(X_processed, y)
            y_pred = temp_model.predict(X_processed)
            residuals = y.values - y_pred
        
        # 1. Linearity (Rainbow test approximation)
        # Check if residuals show clear patterns
        sorted_indices = np.argsort(y_pred)
        sorted_residuals = residuals[sorted_indices]
        
        # Simple linearity check: correlation between fitted values and residuals
        linearity_corr = np.corrcoef(y_pred, residuals)[0, 1]
        results['linearity'] = {
            'correlation_with_fitted': float(linearity_corr),
            'assumption_met': abs(linearity_corr) < 0.1
        }
        
        # 2. Independence (Durbin-Watson approximation)
        # Check for autocorrelation in residuals
        if len(residuals) > 1:
            durbin_watson = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
            results['independence'] = {
                'durbin_watson': float(durbin_watson),
                'assumption_met': 1.5 < durbin_watson < 2.5
            }
        
        # 3. Homoscedasticity (Breusch-Pagan test approximation)
        # Check if variance of residuals is constant
        abs_residuals = np.abs(residuals)
        bp_correlation = np.corrcoef(y_pred, abs_residuals)[0, 1]
        
        results['homoscedasticity'] = {
            'bp_correlation': float(bp_correlation),
            'assumption_met': abs(bp_correlation) < 0.2
        }
        
        # 4. Normality (Shapiro-Wilk test)
        if len(residuals) >= 3:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                results['normality'] = {
                    'shapiro_statistic': float(shapiro_stat),
                    'shapiro_p_value': float(shapiro_p),
                    'assumption_met': shapiro_p > 0.05
                }
            except:
                # Fallback to simpler normality check
                results['normality'] = {
                    'skewness': float(stats.skew(residuals)),
                    'kurtosis': float(stats.kurtosis(residuals)),
                    'assumption_met': abs(stats.skew(residuals)) < 2 and abs(stats.kurtosis(residuals)) < 2
                }
        
        # 5. Multicollinearity
        if X.shape[1] > 1:
            try:
                corr_matrix = X.corr()
                max_corr = corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].max()
                
                results['multicollinearity'] = {
                    'max_correlation': float(max_corr),
                    'assumption_met': max_corr < 0.9
                }
            except:
                results['multicollinearity'] = None
        
    except Exception as e:
        logger.warning(f"Could not complete assumption analysis: {e}")
    
    return results

def suggest_model_improvements(assumption_analysis: Dict[str, Any]) -> List[str]:
    """
    Suggest model improvements based on assumption analysis
    
    Args:
        assumption_analysis: Results from analyze_linear_assumptions
        
    Returns:
        List of improvement suggestions
    """
    
    suggestions = []
    
    # Check each assumption
    linearity = assumption_analysis.get('linearity')
    if linearity and not linearity.get('assumption_met', True):
        suggestions.append("Consider polynomial features or non-linear models due to linearity violations")
    
    independence = assumption_analysis.get('independence')
    if independence and not independence.get('assumption_met', True):
        suggestions.append("Consider time series models or add lag features due to autocorrelation")
    
    homoscedasticity = assumption_analysis.get('homoscedasticity')
    if homoscedasticity and not homoscedasticity.get('assumption_met', True):
        suggestions.append("Consider robust regression or weighted least squares due to heteroscedasticity")
    
    normality = assumption_analysis.get('normality')
    if normality and not normality.get('assumption_met', True):
        suggestions.append("Consider robust regression or data transformation due to non-normal residuals")
    
    multicollinearity = assumption_analysis.get('multicollinearity')
    if multicollinearity and not multicollinearity.get('assumption_met', True):
        suggestions.append("Consider Ridge/Lasso regression or feature selection due to multicollinearity")
    
    if not suggestions:
        suggestions.append("Linear regression assumptions appear to be satisfied")
    
    return suggestions
