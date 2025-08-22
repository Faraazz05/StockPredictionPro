# ============================================
# StockPredictionPro - src/utils/exceptions.py
# Comprehensive exception hierarchy for error handling
# ============================================

import traceback
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json

class StockPredBaseException(Exception):
    """
    Base exception class for all StockPredictionPro exceptions
    
    Features:
    - Error codes for programmatic handling
    - Context information for debugging
    - Severity levels for appropriate responses
    - User-friendly messages for UI display
    - Detailed technical info for logging
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "error",
        user_message: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize base exception
        
        Args:
            message: Technical error message for logs
            error_code: Unique error code for programmatic handling
            context: Additional context information
            severity: Error severity (debug, info, warning, error, critical)
            user_message: User-friendly message for UI display
            suggestions: List of suggested solutions
            cause: Original exception that caused this error
        """
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.context = context or {}
        self.severity = severity
        self.user_message = user_message or self._generate_user_message()
        self.suggestions = suggestions or []
        self.cause = cause
        self.timestamp = datetime.utcnow()
        
        # Add exception details to context
        self.context.update({
            'exception_type': self.__class__.__name__,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity
        })
    
    def _generate_error_code(self) -> str:
        """Generate error code based on class name"""
        class_name = self.__class__.__name__
        # Convert CamelCase to UPPER_SNAKE_CASE
        import re
        error_code = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        error_code = re.sub('([a-z0-9])([A-Z])', r'\1_\2', error_code).upper()
        return error_code.replace('_EXCEPTION', '_ERROR')
    
    def _generate_user_message(self) -> str:
        """Generate user-friendly message from technical message"""
        # Override in subclasses for better user messages
        return "An error occurred while processing your request. Please try again."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'user_message': self.user_message,
            'severity': self.severity,
            'context': self.context,
            'suggestions': self.suggestions,
            'timestamp': self.timestamp.isoformat(),
            'traceback': traceback.format_exc() if self.severity in ['error', 'critical'] else None
        }
    
    def to_json(self) -> str:
        """Convert exception to JSON string"""
        return json.dumps(self.to_dict(), default=str, indent=2)
    
    def add_context(self, **kwargs):
        """Add additional context to the exception"""
        self.context.update(kwargs)
    
    def add_suggestion(self, suggestion: str):
        """Add a suggestion for resolving the error"""
        self.suggestions.append(suggestion)

# ============================================
# Data-related Exceptions
# ============================================

class DataError(StockPredBaseException):
    """Base class for data-related errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity="error", **kwargs)

class DataFetchError(DataError):
    """Raised when data fetching fails"""
    
    def __init__(self, message: str, symbol: Optional[str] = None, source: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if symbol:
            context['symbol'] = symbol
        if source:
            context['data_source'] = source
        
        super().__init__(
            message,
            context=context,
            user_message=f"Failed to fetch data{f' for {symbol}' if symbol else ''}. Please check your internet connection and try again.",
            suggestions=[
                "Check your internet connection",
                "Verify the stock symbol is correct",
                "Try again in a few minutes",
                "Check if the data source is available"
            ],
            **kwargs
        )

class DataValidationError(DataError):
    """Raised when data validation fails"""
    
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None, **kwargs):
        context = kwargs.get('context', {})
        if validation_errors:
            context['validation_errors'] = validation_errors
        
        super().__init__(
            message,
            context=context,
            user_message="The data contains errors and cannot be processed.",
            suggestions=[
                "Check the data quality",
                "Try a different date range",
                "Verify the stock symbol exists",
                "Contact support if the issue persists"
            ],
            **kwargs
        )

class InsufficientDataError(DataError):
    """Raised when there's not enough data for analysis"""
    
    def __init__(self, message: str, required_points: Optional[int] = None, available_points: Optional[int] = None, **kwargs):
        context = kwargs.get('context', {})
        if required_points:
            context['required_points'] = required_points
        if available_points:
            context['available_points'] = available_points
        
        super().__init__(
            message,
            context=context,
            severity="warning",
            user_message="Not enough data available for analysis. Please try a longer date range.",
            suggestions=[
                "Extend the date range to get more data points",
                "Try a different stock symbol",
                "Check if the stock has sufficient trading history"
            ],
            **kwargs
        )

class DataCacheError(DataError):
    """Raised when data caching operations fail"""
    
    def __init__(self, message: str, cache_operation: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if cache_operation:
            context['cache_operation'] = cache_operation
        
        super().__init__(
            message,
            context=context,
            severity="warning",
            user_message="Data caching is temporarily unavailable. Performance may be slower.",
            suggestions=[
                "The operation will continue without caching",
                "Check available disk space",
                "Restart the application if issues persist"
            ],
            **kwargs
        )

# ============================================
# Model-related Exceptions
# ============================================

class ModelError(StockPredBaseException):
    """Base class for model-related errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity="error", **kwargs)

class ModelTrainingError(ModelError):
    """Raised when model training fails"""
    
    def __init__(self, message: str, model_type: Optional[str] = None, symbol: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if model_type:
            context['model_type'] = model_type
        if symbol:
            context['symbol'] = symbol
        
        super().__init__(
            message,
            context=context,
            user_message="Model training failed. Please try with different parameters or data.",
            suggestions=[
                "Try a different model type",
                "Check the data quality",
                "Adjust model parameters",
                "Ensure sufficient training data is available"
            ],
            **kwargs
        )

class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found"""
    
    def __init__(self, message: str, model_id: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if model_id:
            context['model_id'] = model_id
        
        super().__init__(
            message,
            context=context,
            severity="warning",
            user_message="The requested model was not found. Please train a model first.",
            suggestions=[
                "Train a new model",
                "Check if the model ID is correct",
                "Verify the model hasn't been deleted"
            ],
            **kwargs
        )

class ModelValidationError(ModelError):
    """Raised when model validation fails"""
    
    def __init__(self, message: str, validation_metric: Optional[str] = None, threshold: Optional[float] = None, **kwargs):
        context = kwargs.get('context', {})
        if validation_metric:
            context['validation_metric'] = validation_metric
        if threshold:
            context['threshold'] = threshold
        
        super().__init__(
            message,
            context=context,
            user_message="Model performance is below acceptable thresholds.",
            suggestions=[
                "Try different model parameters",
                "Use more training data",
                "Try a different model type",
                "Check data quality"
            ],
            **kwargs
        )

class ModelPredictionError(ModelError):
    """Raised when model prediction fails"""
    
    def __init__(self, message: str, model_id: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if model_id:
            context['model_id'] = model_id
        
        super().__init__(
            message,
            context=context,
            user_message="Failed to generate predictions. Please try again.",
            suggestions=[
                "Check if the model is properly trained",
                "Verify input data format",
                "Try retraining the model",
                "Contact support if the issue persists"
            ],
            **kwargs
        )

# ============================================
# Feature Engineering Exceptions
# ============================================

class FeatureError(StockPredBaseException):
    """Base class for feature engineering errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity="error", **kwargs)

class IndicatorCalculationError(FeatureError):
    """Raised when technical indicator calculation fails"""
    
    def __init__(self, message: str, indicator_name: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if indicator_name:
            context['indicator_name'] = indicator_name
        
        super().__init__(
            message,
            context=context,
            user_message="Failed to calculate technical indicators. Please check your data.",
            suggestions=[
                "Ensure sufficient data points for indicator calculation",
                "Check for missing or invalid data",
                "Try with different indicator parameters"
            ],
            **kwargs
        )

class FeatureSelectionError(FeatureError):
    """Raised when feature selection fails"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            user_message="Feature selection failed. Using all available features.",
            suggestions=[
                "Check feature data quality",
                "Try different selection criteria",
                "Ensure features have sufficient variance"
            ],
            **kwargs
        )

# ============================================
# Trading-related Exceptions
# ============================================

class TradingError(StockPredBaseException):
    """Base class for trading-related errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity="error", **kwargs)

class SignalGenerationError(TradingError):
    """Raised when trading signal generation fails"""
    
    def __init__(self, message: str, symbol: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if symbol:
            context['symbol'] = symbol
        
        super().__init__(
            message,
            context=context,
            user_message="Failed to generate trading signals. Please check your models and data.",
            suggestions=[
                "Ensure models are properly trained",
                "Check data availability",
                "Verify signal generation parameters"
            ],
            **kwargs
        )

class RiskManagementError(TradingError):
    """Raised when risk management checks fail"""
    
    def __init__(self, message: str, risk_type: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if risk_type:
            context['risk_type'] = risk_type
        
        super().__init__(
            message,
            context=context,
            severity="warning",
            user_message="Risk limits exceeded. Trade rejected for safety.",
            suggestions=[
                "Reduce position size",
                "Check portfolio risk metrics",
                "Adjust risk parameters",
                "Wait for better market conditions"
            ],
            **kwargs
        )

class BacktestError(TradingError):
    """Raised when backtesting fails"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            user_message="Backtesting failed. Please check your strategy parameters.",
            suggestions=[
                "Verify strategy configuration",
                "Check historical data availability",
                "Ensure sufficient backtest period",
                "Review cost and slippage settings"
            ],
            **kwargs
        )

# ============================================
# API-related Exceptions
# ============================================

class APIError(StockPredBaseException):
    """Base class for API-related errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity="error", **kwargs)

class RateLimitError(APIError):
    """Raised when API rate limits are exceeded"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        context = kwargs.get('context', {})
        if retry_after:
            context['retry_after'] = retry_after
        
        super().__init__(
            message,
            context=context,
            severity="warning",
            user_message="Rate limit exceeded. Please wait before making more requests.",
            suggestions=[
                f"Wait {retry_after} seconds before retrying" if retry_after else "Wait before retrying",
                "Reduce request frequency",
                "Consider upgrading API plan"
            ],
            **kwargs
        )

class AuthenticationError(APIError):
    """Raised when API authentication fails"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            user_message="Authentication failed. Please check your API credentials.",
            suggestions=[
                "Verify API key is correct",
                "Check if API key has expired",
                "Ensure API key has required permissions"
            ],
            **kwargs
        )

class ExternalAPIError(APIError):
    """Raised when external API calls fail"""
    
    def __init__(self, message: str, api_name: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        context = kwargs.get('context', {})
        if api_name:
            context['api_name'] = api_name
        if status_code:
            context['status_code'] = status_code
        
        super().__init__(
            message,
            context=context,
            user_message="External service is temporarily unavailable. Please try again later.",
            suggestions=[
                "Try again in a few minutes",
                "Check external service status",
                "Use alternative data source if available"
            ],
            **kwargs
        )

# ============================================
# Configuration Exceptions
# ============================================

class ConfigurationError(StockPredBaseException):
    """Raised when configuration is invalid or missing"""
    
    def __init__(self, message: str, config_name: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if config_name:
            context['config_name'] = config_name
        
        super().__init__(
            message,
            context=context,
            severity="critical",
            user_message="Application configuration error. Please contact support.",
            suggestions=[
                "Check configuration files",
                "Verify environment variables",
                "Restore default configuration",
                "Contact system administrator"
            ],
            **kwargs
        )

# ============================================
# Business Logic Exceptions
# ============================================

class BusinessLogicError(StockPredBaseException):
    """Raised when business logic validation fails"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity="warning", **kwargs)

class InvalidParameterError(BusinessLogicError):
    """Raised when invalid parameters are provided"""
    
    def __init__(self, message: str, parameter_name: Optional[str] = None, provided_value: Any = None, **kwargs):
        context = kwargs.get('context', {})
        if parameter_name:
            context['parameter_name'] = parameter_name
        if provided_value is not None:
            context['provided_value'] = provided_value
        
        super().__init__(
            message,
            context=context,
            user_message="Invalid input parameters. Please check your inputs and try again.",
            suggestions=[
                "Check parameter values",
                "Refer to documentation for valid ranges",
                "Use default values if unsure"
            ],
            **kwargs
        )

# ============================================
# Utility Functions
# ============================================

def handle_exception(func):
    """
    Decorator to handle exceptions gracefully
    
    Usage:
        @handle_exception
        def my_function():
            # Function code here
            pass
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except StockPredBaseException:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Convert standard exceptions to our format
            raise StockPredBaseException(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                cause=e,
                severity="error",
                user_message="An unexpected error occurred. Please try again.",
                context={
                    'function': func.__name__,
                    'args': str(args)[:200],  # Limit size
                    'kwargs': str(kwargs)[:200]
                }
            )
    
    return wrapper

def log_exception(exception: Exception, logger=None):
    """Log exception with appropriate level and context"""
    from .logger import get_logger
    
    if logger is None:
        logger = get_logger('exceptions')
    
    if isinstance(exception, StockPredBaseException):
        # Use the exception's severity level
        level_map = {
            'debug': logger.debug,
            'info': logger.info,
            'warning': logger.warning,
            'error': logger.error,
            'critical': logger.critical
        }
        
        log_func = level_map.get(exception.severity, logger.error)
        log_func(
            f"[{exception.error_code}] {exception.message}",
            extra=exception.context,
            exc_info=exception.severity in ['error', 'critical']
        )
    else:
        # Standard exception
        logger.error(f"Unexpected exception: {str(exception)}", exc_info=True)

def create_error_response(exception: Exception) -> Dict[str, Any]:
    """Create standardized error response for APIs"""
    if isinstance(exception, StockPredBaseException):
        return exception.to_dict()
    else:
        # Convert standard exception
        return {
            'error_code': 'UNEXPECTED_ERROR',
            'message': str(exception),
            'user_message': 'An unexpected error occurred. Please try again.',
            'severity': 'error',
            'context': {'exception_type': type(exception).__name__},
            'suggestions': ['Try again', 'Contact support if the issue persists'],
            'timestamp': datetime.utcnow().isoformat()
        }

# ============================================
# Exception Registry
# ============================================

EXCEPTION_REGISTRY = {
    # Data exceptions
    'DATA_FETCH_ERROR': DataFetchError,
    'DATA_VALIDATION_ERROR': DataValidationError,
    'INSUFFICIENT_DATA_ERROR': InsufficientDataError,
    'DATA_CACHE_ERROR': DataCacheError,
    
    # Model exceptions
    'MODEL_TRAINING_ERROR': ModelTrainingError,
    'MODEL_NOT_FOUND_ERROR': ModelNotFoundError,
    'MODEL_VALIDATION_ERROR': ModelValidationError,
    'MODEL_PREDICTION_ERROR': ModelPredictionError,
    
    # Feature exceptions
    'INDICATOR_CALCULATION_ERROR': IndicatorCalculationError,
    'FEATURE_SELECTION_ERROR': FeatureSelectionError,
    
    # Trading exceptions
    'SIGNAL_GENERATION_ERROR': SignalGenerationError,
    'RISK_MANAGEMENT_ERROR': RiskManagementError,
    'BACKTEST_ERROR': BacktestError,
    
    # API exceptions
    'RATE_LIMIT_ERROR': RateLimitError,
    'AUTHENTICATION_ERROR': AuthenticationError,
    'EXTERNAL_API_ERROR': ExternalAPIError,
    
    # Configuration exceptions
    'CONFIGURATION_ERROR': ConfigurationError,
    
    # Business logic exceptions
    'INVALID_PARAMETER_ERROR': InvalidParameterError,
}

def get_exception_class(error_code: str) -> type:
    """Get exception class by error code"""
    return EXCEPTION_REGISTRY.get(error_code, StockPredBaseException)
