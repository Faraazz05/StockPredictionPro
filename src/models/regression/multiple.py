# ============================================
# StockPredictionPro - src/models/regression/multiple.py
# Multiple regression models and ensemble combinations for financial prediction
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from scipy import stats
import warnings

from ...utils.exceptions import ModelValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ..base.base_regressor import BaseFinancialRegressor, RegressionStrategy

logger = get_logger('models.regression.multiple')

# ============================================
# Multiple Linear Regression Model
# ============================================

class FinancialMultipleRegression(BaseFinancialRegressor):
    """
    Multiple Linear Regression model optimized for financial data
    
    Features:
    - Multiple predictor variables analysis
    - Statistical significance testing
    - Multicollinearity detection and handling
    - Automatic feature selection
    - Stepwise regression implementation
    - Comprehensive regression diagnostics
    """
    
    def __init__(self,
                 name: str = "multiple_regression",
                 fit_intercept: bool = True,
                 copy_X: bool = True,
                 n_jobs: Optional[int] = None,
                 positive: bool = False,
                 feature_selection: Optional[str] = None,
                 selection_k: Optional[int] = None,
                 stepwise: bool = False,
                 significance_level: float = 0.05,
                 vif_threshold: float = 10.0,
                 auto_scale: bool = True,
                 **kwargs):
        """
        Initialize Financial Multiple Linear Regression
        
        Args:
            name: Model name
            fit_intercept: Whether to calculate intercept
            copy_X: Whether to copy X or overwrite
            n_jobs: Number of parallel jobs
            positive: Whether to force positive coefficients
            feature_selection: Feature selection method ('k_best', 'rfe', 'stepwise')
            selection_k: Number of features to select
            stepwise: Whether to perform stepwise regression
            significance_level: P-value threshold for stepwise regression
            vif_threshold: VIF threshold for multicollinearity detection
            auto_scale: Whether to automatically scale features
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            model_type="multiple_regression",
            regression_strategy=RegressionStrategy.PRICE_PREDICTION,
            **kwargs
        )
        
        # Multiple regression parameters
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        self.feature_selection = feature_selection
        self.selection_k = selection_k
        self.stepwise = stepwise
        self.significance_level = significance_level
        self.vif_threshold = vif_threshold
        self.auto_scale = auto_scale
        
        # Store parameters for model creation
        self.model_params.update({
            'fit_intercept': fit_intercept,
            'copy_X': copy_X,
            'n_jobs': n_jobs,
            'positive': positive
        })
        
        # Multiple regression-specific attributes
        self.scaler_: Optional[StandardScaler] = None
        self.feature_selector_: Optional[Any] = None
        self.selected_features_: Optional[List[str]] = None
        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.statistical_summary_: Optional[Dict[str, Any]] = None
        self.multicollinearity_analysis_: Optional[Dict[str, Any]] = None
        self.regression_diagnostics_: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized Multiple Linear Regression: {self.name}")
    
    def _create_model(self) -> LinearRegression:
        """Create the multiple linear regression model"""
        return LinearRegression(**self.model_params)
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features with optional scaling and selection"""
        
        # Basic preprocessing
        X_processed = super()._preprocess_features(X)
        
        # Apply feature scaling if enabled
        if self.auto_scale:
            if self.scaler_ is None:
                self.scaler_ = StandardScaler()
                X_scaled = self.scaler_.fit_transform(X_processed)
                logger.debug("Fitted feature scaler for Multiple Regression")
            else:
                X_scaled = self.scaler_.transform(X_processed)
            X_processed = X_scaled
        
        # Apply feature selection if fitted
        if self.feature_selector_ is not None:
            X_processed = self.feature_selector_.transform(X_processed)
        
        return X_processed
    
    def _perform_feature_selection(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Perform feature selection based on specified method"""
        
        if self.feature_selection is None:
            return X, self.feature_names.copy()
        
        if self.feature_selection == 'k_best':
            # Select K best features using F-statistic
            k = self.selection_k or min(10, X.shape[1])
            self.feature_selector_ = SelectKBest(score_func=f_regression, k=k)
            X_selected = self.feature_selector_.fit_transform(X, y)
            
            # Get selected feature names
            selected_indices = self.feature_selector_.get_support(indices=True)
            selected_features = [self.feature_names[i] for i in selected_indices]
            
            logger.info(f"Selected {len(selected_features)} best features using F-statistic")
            
        elif self.feature_selection == 'rfe':
            # Recursive Feature Elimination
            estimator = LinearRegression()
            n_features = self.selection_k or min(10, X.shape[1])
            self.feature_selector_ = RFE(estimator=estimator, n_features_to_select=n_features)
            X_selected = self.feature_selector_.fit_transform(X, y)
            
            # Get selected feature names
            selected_indices = self.feature_selector_.get_support(indices=True)
            selected_features = [self.feature_names[i] for i in selected_indices]
            
            logger.info(f"Selected {len(selected_features)} features using RFE")
            
        elif self.feature_selection == 'stepwise':
            # Stepwise selection
            X_selected, selected_features = self._stepwise_selection(X, y)
            
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection}")
        
        self.selected_features_ = selected_features
        return X_selected, selected_features
    
    def _stepwise_selection(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Perform stepwise feature selection"""
        
        from sklearn.feature_selection import f_regression
        
        # Start with no features
        selected_features = []
        remaining_features = list(range(X.shape[1]))
        
        # Forward selection
        while remaining_features:
            # Test each remaining feature
            best_feature = None
            best_p_value = 1.0
            
            for feature_idx in remaining_features:
                # Create feature set with current selected + test feature
                test_features = selected_features + [feature_idx]
                X_test = X[:, test_features]
                
                # Fit model and get p-values
                model = LinearRegression()
                model.fit(X_test, y)
                
                # Calculate F-statistic and p-value for the new feature
                f_stats, p_values = f_regression(X[:, [feature_idx]], y)
                p_value = p_values[0]
                
                if p_value < best_p_value:
                    best_p_value = p_value
                    best_feature = feature_idx
            
            # Add feature if significant
            if best_feature is not None and best_p_value < self.significance_level:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                feature_name = self.feature_names[best_feature]
                logger.debug(f"Added feature {feature_name} (p-value: {best_p_value:.4f})")
            else:
                break
        
        if not selected_features:
            # If no features selected, select the best one
            f_stats, p_values = f_regression(X, y)
            best_idx = np.argmin(p_values)
            selected_features = [best_idx]
            logger.warning(f"No significant features found, selected best: {self.feature_names[best_idx]}")
        
        # Get selected feature names and data
        selected_feature_names = [self.feature_names[i] for i in selected_features]
        X_selected = X[:, selected_features]
        
        logger.info(f"Stepwise selection completed: {len(selected_features)} features selected")
        
        return X_selected, selected_feature_names
    
    def _post_training_processing(self, X: np.ndarray, y: np.ndarray):
        """Enhanced post-training processing for multiple regression"""
        
        # Call parent processing
        super()._post_training_processing(X, y)
        
        # Extract coefficients
        self.coefficients_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        # Set feature importances (absolute coefficients)
        self.feature_importances_ = np.abs(self.coefficients_)
        
        # Calculate statistical summary
        self._calculate_statistical_summary(X, y)
        
        # Analyze multicollinearity
        self._analyze_multicollinearity(X)
        
        # Calculate comprehensive regression diagnostics
        self._calculate_comprehensive_diagnostics(X, y)
    
    def _calculate_statistical_summary(self, X: np.ndarray, y: np.ndarray):
        """Calculate comprehensive statistical summary"""
        
        try:
            # Make predictions
            y_pred = self.model.predict(X)
            residuals = y - y_pred
            
            # Basic statistics
            n = len(y)
            k = X.shape[1]  # Number of features
            df_residual = n - k - 1 if self.fit_intercept else n - k
            
            # Sum of squares
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_reg = ss_tot - ss_res
            
            # Mean square errors
            mse_res = ss_res / df_residual if df_residual > 0 else np.inf
            mse_reg = ss_reg / k if k > 0 else 0
            
            # R-squared and Adjusted R-squared
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            adj_r_squared = 1 - ((ss_res / df_residual) / (ss_tot / (n - 1))) if df_residual > 0 and n > 1 else r_squared
            
            # F-statistic
            f_statistic = (mse_reg / mse_res) if mse_res > 0 else np.inf
            f_p_value = 1 - stats.f.cdf(f_statistic, k, df_residual) if df_residual > 0 else 0
            
            # Standard error of coefficients
            if df_residual > 0:
                # Calculate covariance matrix
                try:
                    X_design = np.column_stack([np.ones(n), X]) if self.fit_intercept else X
                    cov_matrix = mse_res * np.linalg.inv(X_design.T @ X_design)
                    std_errors = np.sqrt(np.diag(cov_matrix))
                    
                    if self.fit_intercept:
                        intercept_se = std_errors[0]
                        coef_se = std_errors[1:]
                    else:
                        intercept_se = 0
                        coef_se = std_errors
                        
                    # T-statistics and p-values for coefficients
                    t_stats = self.coefficients_ / coef_se
                    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_residual))
                    
                except np.linalg.LinAlgError:
                    logger.warning("Could not calculate coefficient standard errors due to singular matrix")
                    coef_se = np.full(len(self.coefficients_), np.nan)
                    t_stats = np.full(len(self.coefficients_), np.nan)
                    p_values = np.full(len(self.coefficients_), np.nan)
                    intercept_se = np.nan
            else:
                coef_se = np.full(len(self.coefficients_), np.nan)
                t_stats = np.full(len(self.coefficients_), np.nan)
                p_values = np.full(len(self.coefficients_), np.nan)
                intercept_se = np.nan
            
            # Information criteria
            log_likelihood = -n/2 * np.log(2 * np.pi * mse_res) - ss_res / (2 * mse_res)
            aic = 2 * (k + 1) - 2 * log_likelihood
            bic = np.log(n) * (k + 1) - 2 * log_likelihood
            
            self.statistical_summary_ = {
                'n_observations': int(n),
                'n_features': int(k),
                'df_residual': int(df_residual),
                'r_squared': float(r_squared),
                'adj_r_squared': float(adj_r_squared),
                'f_statistic': float(f_statistic),
                'f_p_value': float(f_p_value),
                'mse': float(mse_res),
                'rmse': float(np.sqrt(mse_res)),
                'aic': float(aic),
                'bic': float(bic),
                'intercept': float(self.intercept_),
                'intercept_se': float(intercept_se),
                'coefficients': self.coefficients_.tolist(),
                'coefficient_se': coef_se.tolist(),
                't_statistics': t_stats.tolist(),
                'p_values': p_values.tolist(),
                'significant_features': [
                    self.selected_features_[i] if self.selected_features_ else self.feature_names[i]
                    for i, p in enumerate(p_values) if not np.isnan(p) and p < self.significance_level
                ]
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate statistical summary: {e}")
            self.statistical_summary_ = None
    
    def _analyze_multicollinearity(self, X: np.ndarray):
        """Analyze multicollinearity using VIF and condition indices"""
        
        try:
            # Calculate VIF (Variance Inflation Factor)
            vif_values = []
            feature_names = self.selected_features_ or self.feature_names
            
            for i in range(X.shape[1]):
                # Regress feature i on all other features
                X_others = np.delete(X, i, axis=1)
                
                if X_others.shape[1] == 0:  # Only one feature
                    vif = 1.0
                else:
                    try:
                        model = LinearRegression()
                        model.fit(X_others, X[:, i])
                        r_squared = model.score(X_others, X[:, i])
                        vif = 1 / (1 - r_squared) if r_squared < 0.999 else np.inf
                    except:
                        vif = np.inf
                
                vif_values.append(vif)
            
            # Calculate condition number
            if self.fit_intercept:
                X_design = np.column_stack([np.ones(X.shape[0]), X])
            else:
                X_design = X
            
            try:
                _, s, _ = np.linalg.svd(X_design, full_matrices=False)
                condition_number = s[0] / s[-1] if len(s) > 0 and s[-1] != 0 else np.inf
            except:
                condition_number = np.inf
            
            # Identify problematic features
            high_vif_features = [
                feature_names[i] for i, vif in enumerate(vif_values) 
                if vif > self.vif_threshold
            ]
            
            self.multicollinearity_analysis_ = {
                'vif_values': vif_values,
                'vif_threshold': self.vif_threshold,
                'high_vif_features': high_vif_features,
                'condition_number': float(condition_number),
                'multicollinearity_detected': len(high_vif_features) > 0 or condition_number > 30,
                'max_vif': float(max(vif_values)) if vif_values else 1.0,
                'mean_vif': float(np.mean(vif_values)) if vif_values else 1.0
            }
            
            if self.multicollinearity_analysis_['multicollinearity_detected']:
                logger.warning(f"Multicollinearity detected: {len(high_vif_features)} features with high VIF")
            
        except Exception as e:
            logger.debug(f"Could not analyze multicollinearity: {e}")
            self.multicollinearity_analysis_ = None
    
    def _calculate_comprehensive_diagnostics(self, X: np.ndarray, y: np.ndarray):
        """Calculate comprehensive regression diagnostics"""
        
        try:
            # Make predictions and calculate residuals
            y_pred = self.model.predict(X)
            residuals = y - y_pred
            
            # Standardized residuals
            residual_std = np.std(residuals)
            standardized_residuals = residuals / residual_std if residual_std > 0 else residuals
            
            # Calculate leverage (hat values)
            if self.fit_intercept:
                X_design = np.column_stack([np.ones(X.shape[0]), X])
            else:
                X_design = X
            
            try:
                # H = X(X'X)^(-1)X'
                XtX_inv = np.linalg.inv(X_design.T @ X_design)
                leverage = np.diag(X_design @ XtX_inv @ X_design.T)
            except:
                leverage = np.full(len(y), 1.0 / len(y))  # Average leverage
            
            # Cook's distance
            n, p = X.shape[0], X.shape[1] + (1 if self.fit_intercept else 0)
            mse = np.mean(residuals ** 2)
            
            cooks_distance = []
            for i in range(len(residuals)):
                if leverage[i] < 1.0 and mse > 0:
                    d = (standardized_residuals[i] ** 2) * (leverage[i] / (1 - leverage[i])) / p
                    cooks_distance.append(d)
                else:
                    cooks_distance.append(0.0)
            
            cooks_distance = np.array(cooks_distance)
            
            # Outlier detection
            outlier_threshold = 2.5
            outliers = np.abs(standardized_residuals) > outlier_threshold
            
            # High leverage points
            leverage_threshold = 2 * p / n  # Rule of thumb
            high_leverage = leverage > leverage_threshold
            
            # Influential points (high Cook's distance)
            influence_threshold = 4 / n  # Rule of thumb
            influential = cooks_distance > influence_threshold
            
            self.regression_diagnostics_ = {
                'residuals': residuals.tolist(),
                'standardized_residuals': standardized_residuals.tolist(),
                'leverage': leverage.tolist(),
                'cooks_distance': cooks_distance.tolist(),
                'outliers': {
                    'indices': np.where(outliers)[0].tolist(),
                    'count': int(np.sum(outliers)),
                    'threshold': outlier_threshold
                },
                'high_leverage': {
                    'indices': np.where(high_leverage)[0].tolist(),
                    'count': int(np.sum(high_leverage)),
                    'threshold': leverage_threshold
                },
                'influential': {
                    'indices': np.where(influential)[0].tolist(),
                    'count': int(np.sum(influential)),
                    'threshold': influence_threshold
                },
                'residual_stats': {
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'skewness': float(stats.skew(residuals)),
                    'kurtosis': float(stats.kurtosis(residuals))
                }
            }
            
        except Exception as e:
            logger.debug(f"Could not calculate regression diagnostics: {e}")
            self.regression_diagnostics_ = None
    
    @time_it("multiple_regression_fit", include_args=True)
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'FinancialMultipleRegression':
        """
        Fit the multiple regression model
        
        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional fitting parameters
            
        Returns:
            Fitted model instance
        """
        logger.info(f"Fitting Multiple Regression on {len(X)} samples with {X.shape[1]} features")
        
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
            
            # Preprocess features and targets
            X_processed = self._preprocess_features(X)
            y_processed = self._preprocess_targets(y)
            
            # Perform feature selection if specified
            if self.feature_selection or self.stepwise:
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
                'training_duration_seconds': fit_duration,
                'target_name': self.target_name,
                'feature_selection_method': self.feature_selection
            })
            
            # Calculate training score
            self.training_score = self.model.score(X_processed, y_processed)
            
            # Update status
            self.status = ModelStatus.TRAINED
            self.is_fitted = True
            
            logger.info(f"Multiple Regression trained successfully in {fit_duration:.2f}s")
            
            return self
            
        except Exception as e:
            from ..base.base_model import ModelStatus
            self.status = ModelStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Multiple Regression training failed: {e}")
            raise
    
    def get_statistical_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive statistical summary
        
        Returns:
            Dictionary with statistical analysis
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get statistical summary")
        
        return self.statistical_summary_.copy() if self.statistical_summary_ else {}
    
    def get_multicollinearity_analysis(self) -> Dict[str, Any]:
        """
        Get multicollinearity analysis
        
        Returns:
            Dictionary with multicollinearity analysis
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get multicollinearity analysis")
        
        return self.multicollinearity_analysis_.copy() if self.multicollinearity_analysis_ else {}
    
    def get_regression_diagnostics(self) -> Dict[str, Any]:
        """
        Get regression diagnostics
        
        Returns:
            Dictionary with regression diagnostics
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get regression diagnostics")
        
        return self.regression_diagnostics_.copy() if self.regression_diagnostics_ else {}
    
    def get_coefficients_table(self) -> pd.DataFrame:
        """
        Get coefficients table with statistical information
        
        Returns:
            DataFrame with coefficient analysis
        """
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get coefficients table")
        
        if not self.statistical_summary_:
            raise BusinessLogicError("Statistical summary not available")
        
        stats_sum = self.statistical_summary_
        feature_names = self.selected_features_ or self.feature_names
        
        # Create coefficients table
        coef_table = pd.DataFrame({
            'feature': feature_names,
            'coefficient': stats_sum['coefficients'],
            'std_error': stats_sum['coefficient_se'],
            't_statistic': stats_sum['t_statistics'],
            'p_value': stats_sum['p_values'],
            'significant': [p < self.significance_level for p in stats_sum['p_values']]
        })
        
        # Add VIF if available
        if self.multicollinearity_analysis_:
            coef_table['vif'] = self.multicollinearity_analysis_['vif_values']
        
        # Sort by absolute t-statistic
        coef_table['abs_t_stat'] = np.abs(coef_table['t_statistic'])
        coef_table = coef_table.sort_values('abs_t_stat', ascending=False)
        coef_table = coef_table.drop('abs_t_stat', axis=1)
        
        return coef_table
    
    def plot_residuals_analysis(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Plot comprehensive residuals analysis
        
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
            
            # 1. Residuals vs Fitted
            axes[0, 0].scatter(predictions, residuals, alpha=0.6)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Fitted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Q-Q Plot
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot of Residuals')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Scale-Location Plot
            sqrt_abs_residuals = np.sqrt(np.abs(residuals))
            axes[0, 2].scatter(predictions, sqrt_abs_residuals, alpha=0.6)
            axes[0, 2].set_xlabel('Fitted Values')
            axes[0, 2].set_ylabel('√|Residuals|')
            axes[0, 2].set_title('Scale-Location Plot')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Histogram of Residuals
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Residuals')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Cook's Distance
            if self.regression_diagnostics_:
                cooks_d = self.regression_diagnostics_['cooks_distance']
                threshold = self.regression_diagnostics_['influential']['threshold']
                
                axes[1, 1].scatter(range(len(cooks_d)), cooks_d, alpha=0.6)
                axes[1, 1].axhline(y=threshold, color='red', linestyle='--', 
                                  label=f'Threshold: {threshold:.3f}')
                axes[1, 1].set_xlabel('Observation Index')
                axes[1, 1].set_ylabel("Cook's Distance")
                axes[1, 1].set_title("Cook's Distance")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Cook\'s Distance\nNot Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title("Cook's Distance")
            
            # 6. Leverage vs Residuals
            if self.regression_diagnostics_:
                leverage = self.regression_diagnostics_['leverage']
                axes[1, 2].scatter(leverage, residuals, alpha=0.6)
                axes[1, 2].set_xlabel('Leverage')
                axes[1, 2].set_ylabel('Residuals')
                axes[1, 2].set_title('Residuals vs Leverage')
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'Leverage Analysis\nNot Available', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Residuals vs Leverage')
            
            plt.suptitle(f'Multiple Regression Diagnostics - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_feature_significance(self) -> Any:
        """
        Plot feature significance analysis
        
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            coef_table = self.get_coefficients_table()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot 1: Coefficients with confidence intervals
            y_pos = np.arange(len(coef_table))
            coefficients = coef_table['coefficient']
            std_errors = coef_table['std_error']
            
            # Color by significance
            colors = ['red' if sig else 'gray' for sig in coef_table['significant']]
            
            bars = ax1.barh(y_pos, coefficients, color=colors, alpha=0.7)
            ax1.errorbar(coefficients, y_pos, xerr=1.96*std_errors, fmt='none', color='black', capsize=3)
            
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(coef_table['feature'], fontsize=10)
            ax1.set_xlabel('Coefficient Value')
            ax1.set_title('Coefficients with 95% Confidence Intervals')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax1.grid(True, alpha=0.3)
            
            # Add legend
            red_patch = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.7, label='Significant')
            gray_patch = plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.7, label='Not Significant')
            ax1.legend(handles=[red_patch, gray_patch])
            
            # Plot 2: P-values
            p_values = coef_table['p_value']
            bars2 = ax2.barh(y_pos, -np.log10(p_values), color=colors, alpha=0.7)
            
            # Add significance line
            sig_line = -np.log10(self.significance_level)
            ax2.axvline(x=sig_line, color='red', linestyle='--', 
                       label=f'Significance Level (α = {self.significance_level})')
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(coef_table['feature'], fontsize=10)
            ax2.set_xlabel('-log10(p-value)')
            ax2.set_title('Feature Significance (-log10 p-values)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f'Multiple Regression Feature Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def plot_multicollinearity_analysis(self) -> Any:
        """
        Plot multicollinearity analysis
        
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.multicollinearity_analysis_:
                logger.warning("Multicollinearity analysis not available")
                return None
            
            vif_values = self.multicollinearity_analysis_['vif_values']
            feature_names = self.selected_features_ or self.feature_names
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot 1: VIF values
            y_pos = np.arange(len(feature_names))
            colors = ['red' if vif > self.vif_threshold else 'blue' for vif in vif_values]
            
            bars = ax1.barh(y_pos, vif_values, color=colors, alpha=0.7)
            ax1.axvline(x=self.vif_threshold, color='red', linestyle='--', 
                       label=f'VIF Threshold = {self.vif_threshold}')
            
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(feature_names, fontsize=10)
            ax1.set_xlabel('Variance Inflation Factor (VIF)')
            ax1.set_title('Multicollinearity Analysis - VIF Values')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add VIF values as text
            for i, (bar, vif) in enumerate(zip(bars, vif_values)):
                if vif < np.inf:
                    ax1.text(vif + max(vif_values) * 0.01, i, f'{vif:.2f}', 
                            va='center', fontsize=8)
                else:
                    ax1.text(max([v for v in vif_values if v < np.inf]) * 1.1, i, 'inf', 
                            va='center', fontsize=8)
            
            # Plot 2: Correlation matrix heatmap
            if hasattr(self, 'scaler_') and self.scaler_ is not None:
                # We need the original data for correlation - this is a limitation
                ax2.text(0.5, 0.5, 'Correlation Matrix\nRequires Original Data', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Feature Correlation Matrix')
            else:
                ax2.text(0.5, 0.5, 'Correlation Matrix\nNot Available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Feature Correlation Matrix')
            
            plt.suptitle(f'Multicollinearity Analysis - {self.name}', fontsize=16)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
    
    def get_multiple_regression_summary(self) -> Dict[str, Any]:
        """Get comprehensive multiple regression summary"""
        
        if not self.is_fitted:
            raise BusinessLogicError("Model must be fitted to get summary")
        
        summary = {
            'model_info': {
                'n_features_original': len(self.feature_names),
                'n_features_selected': len(self.selected_features_) if self.selected_features_ else len(self.feature_names),
                'feature_selection_method': self.feature_selection,
                'selected_features': self.selected_features_
            },
            'statistical_summary': self.statistical_summary_,
            'multicollinearity_analysis': self.multicollinearity_analysis_,
            'regression_diagnostics': self.regression_diagnostics_
        }
        
        return summary
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        summary = super().get_model_summary()
        
        # Add multiple regression-specific information
        summary.update({
            'regression_type': 'Multiple Linear Regression',
            'feature_selection': self.feature_selection,
            'stepwise_regression': self.stepwise,
            'significance_level': self.significance_level,
            'vif_threshold': self.vif_threshold,
            'n_selected_features': len(self.selected_features_) if self.selected_features_ else len(self.feature_names),
            'intercept': float(self.intercept_) if self.intercept_ is not None else None
        })
        
        # Add statistical information
        if self.statistical_summary_:
            summary.update({
                'r_squared': self.statistical_summary_['r_squared'],
                'adj_r_squared': self.statistical_summary_['adj_r_squared'],
                'f_statistic': self.statistical_summary_['f_statistic'],
                'f_p_value': self.statistical_summary_['f_p_value'],
                'aic': self.statistical_summary_['aic'],
                'bic': self.statistical_summary_['bic']
            })
        
        # Add multicollinearity info
        if self.multicollinearity_analysis_:
            summary['multicollinearity_detected'] = self.multicollinearity_analysis_['multicollinearity_detected']
            summary['max_vif'] = self.multicollinearity_analysis_['max_vif']
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_multiple_regression(feature_selection: Optional[str] = None,
                              stepwise: bool = False,
                              **kwargs) -> FinancialMultipleRegression:
    """
    Create a Multiple Linear Regression model
    
    Args:
        feature_selection: Feature selection method ('k_best', 'rfe', 'stepwise')
        stepwise: Whether to perform stepwise regression
        **kwargs: Additional model parameters
        
    Returns:
        Configured Multiple Linear Regression model
    """
    
    default_params = {
        'name': 'multiple_regression',
        'feature_selection': feature_selection,
        'stepwise': stepwise,
        'fit_intercept': True,
        'auto_scale': True,
        'significance_level': 0.05,
        'vif_threshold': 10.0
    }
    
    default_params.update(kwargs)
    return FinancialMultipleRegression(**default_params)

def create_stepwise_regression(**kwargs) -> FinancialMultipleRegression:
    """Create Multiple Regression with stepwise feature selection"""
    
    return create_multiple_regression(
        feature_selection='stepwise',
        stepwise=True,
        name='stepwise_regression',
        **kwargs
    )

def create_best_subset_regression(k: int = 10, **kwargs) -> FinancialMultipleRegression:
    """Create Multiple Regression with K-best feature selection"""
    
    return create_multiple_regression(
        feature_selection='k_best',
        selection_k=k,
        name=f'best_subset_regression_k{k}',
        **kwargs
    )

def create_rfe_regression(n_features: int = 10, **kwargs) -> FinancialMultipleRegression:
    """Create Multiple Regression with Recursive Feature Elimination"""
    
    return create_multiple_regression(
        feature_selection='rfe',
        selection_k=n_features,
        name=f'rfe_regression_{n_features}features',
        **kwargs
    )

# ============================================
# Utility Functions
# ============================================

def perform_multiple_regression_analysis(X: pd.DataFrame, y: pd.Series,
                                       feature_selection_methods: List[str] = ['none', 'k_best', 'rfe', 'stepwise'],
                                       k_values: List[int] = [5, 10, 15]) -> Dict[str, Any]:
    """
    Perform comprehensive multiple regression analysis with different feature selection methods
    
    Args:
        X: Feature matrix
        y: Target values
        feature_selection_methods: List of feature selection methods to compare
        k_values: List of K values for feature selection methods
        
    Returns:
        Dictionary with analysis results
    """
    
    logger.info("Performing comprehensive multiple regression analysis")
    
    results = {}
    
    for method in feature_selection_methods:
        if method == 'none':
            # No feature selection
            model = create_multiple_regression(feature_selection=None)
            model_name = 'no_selection'
        elif method == 'stepwise':
            # Stepwise selection
            model = create_stepwise_regression()
            model_name = 'stepwise'
        else:
            # K-based methods
            for k in k_values:
                k_adj = min(k, X.shape[1])  # Don't select more features than available
                
                if method == 'k_best':
                    model = create_best_subset_regression(k=k_adj)
                    model_name = f'k_best_{k_adj}'
                elif method == 'rfe':
                    model = create_rfe_regression(n_features=k_adj)
                    model_name = f'rfe_{k_adj}'
                else:
                    continue
                
                logger.info(f"Testing {model_name}")
                
                try:
                    # Fit model
                    model.fit(X, y)
                    
                    # Get comprehensive results
                    model_summary = model.get_multiple_regression_summary()
                    statistical_summary = model_summary['statistical_summary']
                    multicollinearity = model_summary['multicollinearity_analysis']
                    
                    results[model_name] = {
                        'model': model,
                        'n_features_selected': model_summary['model_info']['n_features_selected'],
                        'selected_features': model_summary['model_info']['selected_features'],
                        'r_squared': statistical_summary['r_squared'] if statistical_summary else None,
                        'adj_r_squared': statistical_summary['adj_r_squared'] if statistical_summary else None,
                        'f_statistic': statistical_summary['f_statistic'] if statistical_summary else None,
                        'f_p_value': statistical_summary['f_p_value'] if statistical_summary else None,
                        'aic': statistical_summary['aic'] if statistical_summary else None,
                        'bic': statistical_summary['bic'] if statistical_summary else None,
                        'max_vif': multicollinearity['max_vif'] if multicollinearity else None,
                        'multicollinearity_detected': multicollinearity['multicollinearity_detected'] if multicollinearity else None
                    }
                    
                except Exception as e:
                    logger.warning(f"Error with {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
                
                if method not in ['k_best', 'rfe']:
                    break  # Only test k_values for k-based methods
    
    # Find best model based on different criteria
    valid_results = {k: v for k, v in results.items() if 'error' not in v and v.get('adj_r_squared') is not None}
    
    if valid_results:
        best_adj_r2 = max(valid_results.keys(), key=lambda k: valid_results[k]['adj_r_squared'])
        best_aic = min(valid_results.keys(), key=lambda k: valid_results[k]['aic'])
        best_bic = min(valid_results.keys(), key=lambda k: valid_results[k]['bic'])
        
        results['comparison'] = {
            'best_adj_r_squared': best_adj_r2,
            'best_aic': best_aic,
            'best_bic': best_bic,
            'valid_models': list(valid_results.keys())
        }
    
    logger.info(f"Multiple regression analysis complete. Tested {len(results)} configurations")
    
    return results

def detect_multicollinearity(X: pd.DataFrame, vif_threshold: float = 10.0) -> Dict[str, Any]:
    """
    Detect multicollinearity in features
    
    Args:
        X: Feature matrix
        vif_threshold: VIF threshold for multicollinearity detection
        
    Returns:
        Dictionary with multicollinearity analysis
    """
    
    logger.info("Detecting multicollinearity in features")
    
    from sklearn.linear_model import LinearRegression
    
    # Calculate VIF for each feature
    vif_data = []
    feature_names = X.columns.tolist()
    
    for i, feature in enumerate(feature_names):
        # Regress feature on all other features
        X_others = X.drop(columns=[feature])
        y_feature = X[feature]
        
        if X_others.shape[1] == 0:
            vif = 1.0
        else:
            try:
                model = LinearRegression()
                model.fit(X_others, y_feature)
                r_squared = model.score(X_others, y_feature)
                vif = 1 / (1 - r_squared) if r_squared < 0.999 else np.inf
            except:
                vif = np.inf
        
        vif_data.append({
            'feature': feature,
            'vif': vif,
            'high_vif': vif > vif_threshold
        })
    
    vif_df = pd.DataFrame(vif_data).sort_values('vif', ascending=False)
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.8:  # High correlation threshold
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
    
    # Overall assessment
    n_high_vif = vif_df['high_vif'].sum()
    multicollinearity_detected = n_high_vif > 0 or len(high_corr_pairs) > 0
    
    analysis = {
        'vif_analysis': vif_df,
        'correlation_analysis': high_corr_df,
        'summary': {
            'multicollinearity_detected': multicollinearity_detected,
            'n_high_vif_features': int(n_high_vif),
            'n_high_corr_pairs': len(high_corr_pairs),
            'max_vif': float(vif_df['vif'].max()),
            'mean_vif': float(vif_df['vif'].mean()),
            'vif_threshold': vif_threshold
        }
    }
    
    logger.info(f"Multicollinearity analysis complete. Detected: {multicollinearity_detected}")
    
    return analysis

def compare_regression_models(X: pd.DataFrame, y: pd.Series,
                            models_config: List[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Compare different multiple regression configurations
    
    Args:
        X: Feature matrix
        y: Target values
        models_config: List of model configurations to compare
        
    Returns:
        DataFrame with model comparison
    """
    
    if models_config is None:
        models_config = [
            {'name': 'full_model', 'feature_selection': None},
            {'name': 'stepwise', 'feature_selection': 'stepwise'},
            {'name': 'best_10', 'feature_selection': 'k_best', 'selection_k': 10},
            {'name': 'rfe_10', 'feature_selection': 'rfe', 'selection_k': 10},
        ]
    
    logger.info(f"Comparing {len(models_config)} regression models")
    
    comparison_results = []
    
    for config in models_config:
        try:
            # Create and fit model
            model = create_multiple_regression(**{k: v for k, v in config.items() if k != 'name'})
            model.fit(X, y)
            
            # Get model summary
            summary = model.get_multiple_regression_summary()
            stats_summary = summary.get('statistical_summary', {})
            multicol = summary.get('multicollinearity_analysis', {})
            
            comparison_results.append({
                'model_name': config['name'],
                'n_features': summary['model_info']['n_features_selected'],
                'r_squared': stats_summary.get('r_squared'),
                'adj_r_squared': stats_summary.get('adj_r_squared'),
                'f_statistic': stats_summary.get('f_statistic'),
                'f_p_value': stats_summary.get('f_p_value'),
                'aic': stats_summary.get('aic'),
                'bic': stats_summary.get('bic'),
                'max_vif': multicol.get('max_vif'),
                'multicollinearity': multicol.get('multicollinearity_detected'),
                'feature_selection': config.get('feature_selection', 'none')
            })
            
        except Exception as e:
            logger.warning(f"Error with model {config['name']}: {e}")
            comparison_results.append({
                'model_name': config['name'],
                'error': str(e)
            })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    # Sort by adjusted R-squared (descending)
    if 'adj_r_squared' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('adj_r_squared', ascending=False)
    
    logger.info("Model comparison complete")
    
    return comparison_df
