# ============================================
# StockPredictionPro - src/api/exceptions.py
# Comprehensive exception handling for FastAPI application with custom exceptions and global handlers
# ============================================

import logging
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
from traceback import format_exception

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..utils.logger import get_logger

logger = get_logger('api.exceptions')

# ============================================
# Base Exception Classes
# ============================================

class BaseAPIException(HTTPException):
    """
    Base exception class for all API exceptions.
    
    Provides common structure and functionality for custom exceptions
    with error codes, context data, and structured responses.
    """
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str,
        context: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        
        # Add error code to headers
        if headers is None:
            headers = {}
        headers["X-Error-Code"] = error_code
        headers["X-Error-Timestamp"] = self.timestamp.isoformat()
        
        super().__init__(
            status_code=status_code,
            detail=detail,
            headers=headers
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response"""
        return {
            "error_code": self.error_code,
            "message": self.detail,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "status_code": self.status_code
        }

# ============================================
# Client Error Exceptions (4xx)
# ============================================

class ValidationException(BaseAPIException):
    """Validation error exception (422)"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        context = {}
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message,
            error_code="VALIDATION_ERROR",
            context=context
        )

class NotFoundException(BaseAPIException):
    """Resource not found exception (404)"""
    
    def __init__(self, resource: str, identifier: Union[str, int], field: str = "id"):
        context = {
            "resource": resource,
            "field": field,
            "identifier": str(identifier)
        }
        
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with {field} '{identifier}' not found",
            error_code="RESOURCE_NOT_FOUND",
            context=context
        )

class UnauthorizedException(BaseAPIException):
    """Authentication required exception (401)"""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message,
            error_code="UNAUTHORIZED"
        )

class ForbiddenException(BaseAPIException):
    """Access forbidden exception (403)"""
    
    def __init__(self, message: str = "Access forbidden", required_permission: Optional[str] = None):
        context = {}
        if required_permission:
            context["required_permission"] = required_permission
        
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=message,
            error_code="FORBIDDEN",
            context=context
        )

class ConflictException(BaseAPIException):
    """Resource conflict exception (409)"""
    
    def __init__(self, message: str, conflicting_resource: Optional[str] = None):
        context = {}
        if conflicting_resource:
            context["conflicting_resource"] = conflicting_resource
        
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=message,
            error_code="CONFLICT",
            context=context
        )

class RateLimitException(BaseAPIException):
    """Rate limit exceeded exception (429)"""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        context = {}
        headers = {}
        
        if retry_after:
            context["retry_after"] = retry_after
            headers["Retry-After"] = str(retry_after)
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=message,
            error_code="RATE_LIMIT_EXCEEDED",
            context=context,
            headers=headers
        )

# ============================================
# Server Error Exceptions (5xx)
# ============================================

class InternalServerError(BaseAPIException):
    """Internal server error exception (500)"""
    
    def __init__(self, message: str = "Internal server error", error_id: Optional[str] = None):
        context = {}
        if error_id:
            context["error_id"] = error_id
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
            error_code="INTERNAL_SERVER_ERROR",
            context=context
        )

class ServiceUnavailableException(BaseAPIException):
    """Service unavailable exception (503)"""
    
    def __init__(self, message: str = "Service temporarily unavailable", service_name: Optional[str] = None):
        context = {}
        if service_name:
            context["service_name"] = service_name
        
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=message,
            error_code="SERVICE_UNAVAILABLE",
            context=context
        )

class BadGatewayException(BaseAPIException):
    """Bad gateway exception (502)"""
    
    def __init__(self, message: str = "Bad gateway", upstream_service: Optional[str] = None):
        context = {}
        if upstream_service:
            context["upstream_service"] = upstream_service
        
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=message,
            error_code="BAD_GATEWAY",
            context=context
        )

# ============================================
# Domain-Specific Exceptions
# ============================================

class DataException(BaseAPIException):
    """Data-related exceptions"""
    
    def __init__(self, message: str, data_source: Optional[str] = None, symbol: Optional[str] = None):
        context = {}
        if data_source:
            context["data_source"] = data_source
        if symbol:
            context["symbol"] = symbol
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message,
            error_code="DATA_ERROR",
            context=context
        )

class ModelException(BaseAPIException):
    """Model-related exceptions"""
    
    def __init__(self, message: str, model_type: Optional[str] = None, model_id: Optional[str] = None):
        context = {}
        if model_type:
            context["model_type"] = model_type
        if model_id:
            context["model_id"] = model_id
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message,
            error_code="MODEL_ERROR",
            context=context
        )

class TradingException(BaseAPIException):
    """Trading-related exceptions"""
    
    def __init__(self, message: str, strategy: Optional[str] = None, symbol: Optional[str] = None):
        context = {}
        if strategy:
            context["strategy"] = strategy
        if symbol:
            context["symbol"] = symbol
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message,
            error_code="TRADING_ERROR",
            context=context
        )

class BacktestException(BaseAPIException):
    """Backtesting-related exceptions"""
    
    def __init__(self, message: str, backtest_id: Optional[str] = None, period: Optional[str] = None):
        context = {}
        if backtest_id:
            context["backtest_id"] = backtest_id
        if period:
            context["period"] = period
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message,
            error_code="BACKTEST_ERROR",
            context=context
        )

class ConfigurationException(BaseAPIException):
    """Configuration-related exceptions"""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        context = {}
        if config_key:
            context["config_key"] = config_key
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
            error_code="CONFIGURATION_ERROR",
            context=context
        )

# ============================================
# Exception Handlers
# ============================================

async def base_api_exception_handler(request: Request, exc: BaseAPIException) -> JSONResponse:
    """
    Handler for BaseAPIException and its subclasses
    
    Args:
        request: FastAPI request object
        exc: Exception instance
        
    Returns:
        JSON error response
    """
    
    # Log the exception
    log_level = logging.WARNING if exc.status_code < 500 else logging.ERROR
    logger.log(
        log_level,
        f"API Exception: {exc.error_code} - {exc.detail}",
        extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "path": str(request.url),
            "method": request.method,
            "context": exc.context
        }
    )
    
    # Build response
    response_data = {
        "error": {
            "code": exc.error_code,
            "message": exc.detail,
            "timestamp": exc.timestamp.isoformat()
        },
        "request": {
            "path": str(request.url.path),
            "method": request.method
        }
    }
    
    # Add context if available
    if exc.context:
        response_data["error"]["context"] = exc.context
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
        headers=exc.headers
    )

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handler for standard HTTPException
    
    Args:
        request: FastAPI request object
        exc: HTTPException instance
        
    Returns:
        JSON error response
    """
    
    logger.warning(
        f"HTTP Exception: {exc.status_code} - {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": str(request.url),
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat()
            },
            "request": {
                "path": str(request.url.path),
                "method": request.method
            }
        },
        headers=exc.headers
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handler for Pydantic validation errors
    
    Args:
        request: FastAPI request object
        exc: RequestValidationError instance
        
    Returns:
        JSON error response with validation details
    """
    
    logger.warning(
        f"Validation Error: {len(exc.errors())} validation issues",
        extra={
            "path": str(request.url),
            "method": request.method,
            "validation_errors": exc.errors()
        }
    )
    
    # Process validation errors
    errors = []
    for error in exc.errors():
        error_detail = {
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        }
        
        if "input" in error:
            error_detail["input"] = str(error["input"])
        
        errors.append(error_detail)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": f"Validation failed for {len(errors)} field(s)",
                "timestamp": datetime.utcnow().isoformat(),
                "details": errors
            },
            "request": {
                "path": str(request.url.path),
                "method": request.method
            }
        }
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handler for unhandled exceptions
    
    Args:
        request: FastAPI request object
        exc: Exception instance
        
    Returns:
        JSON error response
    """
    
    # Generate error ID for tracking
    error_id = f"err_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{id(exc)}"
    
    # Log the full exception with stack trace
    logger.error(
        f"Unhandled Exception: {type(exc).__name__} - {str(exc)}",
        extra={
            "error_id": error_id,
            "path": str(request.url),
            "method": request.method,
            "exception_type": type(exc).__name__
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat(),
                "error_id": error_id
            },
            "request": {
                "path": str(request.url.path),
                "method": request.method
            }
        }
    )

# ============================================
# Exception Handler Registration
# ============================================

def register_exception_handlers(app):
    """
    Register all exception handlers with the FastAPI app
    
    Args:
        app: FastAPI application instance
    """
    
    # Custom exception handlers
    app.add_exception_handler(BaseAPIException, base_api_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("All exception handlers registered successfully")

# ============================================
# Convenience Functions
# ============================================

def raise_not_found(resource: str, identifier: Union[str, int], field: str = "id") -> None:
    """Convenience function to raise NotFoundException"""
    raise NotFoundException(resource, identifier, field)

def raise_validation_error(message: str, field: Optional[str] = None, value: Optional[Any] = None) -> None:
    """Convenience function to raise ValidationException"""
    raise ValidationException(message, field, value)

def raise_unauthorized(message: str = "Authentication required") -> None:
    """Convenience function to raise UnauthorizedException"""
    raise UnauthorizedException(message)

def raise_forbidden(message: str = "Access forbidden", required_permission: Optional[str] = None) -> None:
    """Convenience function to raise ForbiddenException"""
    raise ForbiddenException(message, required_permission)

def raise_conflict(message: str, conflicting_resource: Optional[str] = None) -> None:
    """Convenience function to raise ConflictException"""
    raise ConflictException(message, conflicting_resource)

def raise_rate_limit(message: str = "Rate limit exceeded", retry_after: Optional[int] = None) -> None:
    """Convenience function to raise RateLimitException"""
    raise RateLimitException(message, retry_after)

def raise_internal_error(message: str = "Internal server error", error_id: Optional[str] = None) -> None:
    """Convenience function to raise InternalServerError"""
    raise InternalServerError(message, error_id)

def raise_service_unavailable(message: str = "Service temporarily unavailable", service_name: Optional[str] = None) -> None:
    """Convenience function to raise ServiceUnavailableException"""
    raise ServiceUnavailableException(message, service_name)

def raise_data_error(message: str, data_source: Optional[str] = None, symbol: Optional[str] = None) -> None:
    """Convenience function to raise DataException"""
    raise DataException(message, data_source, symbol)

def raise_model_error(message: str, model_type: Optional[str] = None, model_id: Optional[str] = None) -> None:
    """Convenience function to raise ModelException"""
    raise ModelException(message, model_type, model_id)

def raise_trading_error(message: str, strategy: Optional[str] = None, symbol: Optional[str] = None) -> None:
    """Convenience function to raise TradingException"""
    raise TradingException(message, strategy, symbol)

def raise_backtest_error(message: str, backtest_id: Optional[str] = None, period: Optional[str] = None) -> None:
    """Convenience function to raise BacktestException"""
    raise BacktestException(message, backtest_id, period)

def raise_config_error(message: str, config_key: Optional[str] = None) -> None:
    """Convenience function to raise ConfigurationException"""
    raise ConfigurationException(message, config_key)

# ============================================
# Error Response Models (for OpenAPI docs)
# ============================================

class ErrorResponse:
    """Standard error response model"""
    
    @staticmethod
    def model_400():
        return {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Invalid input data",
                            "timestamp": "2023-01-01T12:00:00"
                        },
                        "request": {
                            "path": "/api/v1/predictions",
                            "method": "POST"
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def model_401():
        return {
            "description": "Unauthorized",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "UNAUTHORIZED",
                            "message": "Authentication required",
                            "timestamp": "2023-01-01T12:00:00"
                        },
                        "request": {
                            "path": "/api/v1/protected",
                            "method": "GET"
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def model_403():
        return {
            "description": "Forbidden",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "FORBIDDEN",
                            "message": "Access forbidden",
                            "timestamp": "2023-01-01T12:00:00",
                            "context": {
                                "required_permission": "admin"
                            }
                        },
                        "request": {
                            "path": "/api/v1/admin",
                            "method": "GET"
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def model_404():
        return {
            "description": "Not Found",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "RESOURCE_NOT_FOUND",
                            "message": "Model with id 'abc123' not found",
                            "timestamp": "2023-01-01T12:00:00",
                            "context": {
                                "resource": "Model",
                                "field": "id",
                                "identifier": "abc123"
                            }
                        },
                        "request": {
                            "path": "/api/v1/models/abc123",
                            "method": "GET"
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def model_422():
        return {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Validation failed for 2 field(s)",
                            "timestamp": "2023-01-01T12:00:00",
                            "details": [
                                {
                                    "field": "symbol",
                                    "message": "field required",
                                    "type": "value_error.missing"
                                },
                                {
                                    "field": "start_date",
                                    "message": "invalid date format",
                                    "type": "value_error.date",
                                    "input": "invalid-date"
                                }
                            ]
                        },
                        "request": {
                            "path": "/api/v1/data",
                            "method": "POST"
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def model_429():
        return {
            "description": "Rate Limit Exceeded",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "RATE_LIMIT_EXCEEDED",
                            "message": "Rate limit exceeded",
                            "timestamp": "2023-01-01T12:00:00",
                            "context": {
                                "retry_after": 3600
                            }
                        },
                        "request": {
                            "path": "/api/v1/predictions",
                            "method": "POST"
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def model_500():
        return {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "INTERNAL_SERVER_ERROR",
                            "message": "An unexpected error occurred",
                            "timestamp": "2023-01-01T12:00:00",
                            "error_id": "err_20230101_120000_123456"
                        },
                        "request": {
                            "path": "/api/v1/models/train",
                            "method": "POST"
                        }
                    }
                }
            }
        }

# ============================================
# Export All Components
# ============================================

__all__ = [
    # Base classes
    "BaseAPIException",
    
    # Client error exceptions
    "ValidationException",
    "NotFoundException", 
    "UnauthorizedException",
    "ForbiddenException",
    "ConflictException",
    "RateLimitException",
    
    # Server error exceptions
    "InternalServerError",
    "ServiceUnavailableException",
    "BadGatewayException",
    
    # Domain-specific exceptions
    "DataException",
    "ModelException",
    "TradingException",
    "BacktestException",
    "ConfigurationException",
    
    # Exception handlers
    "base_api_exception_handler",
    "http_exception_handler",
    "validation_exception_handler",
    "general_exception_handler",
    "register_exception_handlers",
    
    # Convenience functions
    "raise_not_found",
    "raise_validation_error",
    "raise_unauthorized",
    "raise_forbidden",
    "raise_conflict",
    "raise_rate_limit",
    "raise_internal_error",
    "raise_service_unavailable",
    "raise_data_error",
    "raise_model_error",
    "raise_trading_error",
    "raise_backtest_error",
    "raise_config_error",
    
    # Error response models
    "ErrorResponse",
]
