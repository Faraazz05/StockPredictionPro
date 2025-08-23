# ============================================
# StockPredictionPro - src/api/schemas/error_schemas.py
# Comprehensive Pydantic schemas for error handling and standardized error responses
# ============================================

from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

# ============================================
# Error Classification and Constants
# ============================================

class ErrorCategory(str, Enum):
    """Error categories for classification"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM = "system"
    RATE_LIMITING = "rate_limiting"
    DATA_QUALITY = "data_quality"
    MODEL = "model"
    TRADING = "trading"
    CONFIGURATION = "configuration"

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorRecovery(str, Enum):
    """Error recovery recommendations"""
    RETRY_IMMEDIATELY = "retry_immediately"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    MODIFY_REQUEST = "modify_request"
    CONTACT_SUPPORT = "contact_support"
    NO_RECOVERY = "no_recovery"

# Common HTTP status codes
HTTP_STATUS_CODES = {
    400: "Bad Request",
    401: "Unauthorized", 
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    406: "Not Acceptable",
    409: "Conflict",
    410: "Gone",
    413: "Request Entity Too Large",
    415: "Unsupported Media Type",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    501: "Not Implemented",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}

# ============================================
# Base Error Schemas
# ============================================

class BaseErrorModel(BaseModel):
    """Base error model with common configuration"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        populate_by_name=True,
        json_schema_extra={
            "examples": []
        }
    )

class ErrorDetail(BaseErrorModel):
    """Individual error detail"""
    
    code: str = Field(
        description="Specific error code",
        examples=["INVALID_SYMBOL", "DATA_NOT_FOUND", "VALIDATION_FAILED"]
    )
    
    message: str = Field(
        description="Human-readable error message",
        examples=["Invalid stock symbol format", "No data found for the specified date range"]
    )
    
    field: Optional[str] = Field(
        default=None,
        description="Field that caused the error (for validation errors)",
        examples=["symbol", "start_date", "email"]
    )
    
    value: Optional[Union[str, int, float, bool, None]] = Field(
        default=None,
        description="Invalid value that caused the error",
        examples=["invalid_symbol", "2025-01-01", -1]
    )
    
    location: Optional[List[Union[str, int]]] = Field(
        default=None,
        description="Location path of the error in the request",
        examples=[["body", "symbol"], ["query", "limit"]]
    )

class ErrorContext(BaseErrorModel):
    """Additional context information for errors"""
    
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request identifier for tracking",
        examples=["req_abc123"]
    )
    
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for distributed tracing",
        examples=["corr_xyz789"]
    )
    
    user_id: Optional[str] = Field(
        default=None,
        description="User ID who made the request",
        examples=["user_123"]
    )
    
    endpoint: Optional[str] = Field(
        default=None,
        description="API endpoint that generated the error",
        examples=["/api/v1/data/AAPL"]
    )
    
    method: Optional[str] = Field(
        default=None,
        description="HTTP method used",
        examples=["GET", "POST", "PUT"]
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred",
        examples=["2023-01-01T12:00:00Z"]
    )
    
    environment: Optional[str] = Field(
        default=None,
        description="Environment where error occurred",
        examples=["production", "staging", "development"]
    )

# ============================================
# Standardized Error Response
# ============================================

class ErrorResponse(BaseErrorModel):
    """Standardized error response format"""
    
    error: Dict[str, Any] = Field(
        description="Error information container"
    )
    
    request: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Request information for debugging"
    )
    
    support: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Support and recovery information"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Request validation failed",
                        "category": "validation",
                        "severity": "medium",
                        "timestamp": "2023-01-01T12:00:00Z",
                        "details": [
                            {
                                "code": "INVALID_SYMBOL",
                                "message": "Symbol must be 1-12 alphanumeric characters",
                                "field": "symbol",
                                "value": "invalid_symbol_123",
                                "location": ["body", "symbol"]
                            }
                        ]
                    },
                    "request": {
                        "path": "/api/v1/data/fetch",
                        "method": "POST",
                        "request_id": "req_abc123"
                    },
                    "support": {
                        "recovery_suggestion": "MODIFY_REQUEST",
                        "documentation_url": "https://api.stockpred.com/docs#symbols"
                    }
                }
            ]
        }
    )

class DetailedError(BaseErrorModel):
    """Detailed error information"""
    
    code: str = Field(
        description="Primary error code",
        examples=["VALIDATION_ERROR", "RESOURCE_NOT_FOUND", "RATE_LIMIT_EXCEEDED"]
    )
    
    message: str = Field(
        description="Primary error message",
        examples=["Request validation failed", "Symbol not found", "Rate limit exceeded"]
    )
    
    category: ErrorCategory = Field(
        description="Error category for classification"
    )
    
    severity: ErrorSeverity = Field(
        description="Error severity level"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error occurrence timestamp"
    )
    
    details: Optional[List[ErrorDetail]] = Field(
        default=None,
        description="Detailed error breakdown",
        max_length=10
    )
    
    context: Optional[ErrorContext] = Field(
        default=None,
        description="Additional error context"
    )
    
    retry_after: Optional[int] = Field(
        default=None,
        description="Seconds to wait before retrying (for rate limiting)",
        examples=[60, 300, 3600]
    )
    
    recovery_suggestion: Optional[ErrorRecovery] = Field(
        default=None,
        description="Suggested recovery action"
    )

# ============================================
# Specific Error Types
# ============================================

class ValidationErrorResponse(BaseErrorModel):
    """Validation error response"""
    
    error: DetailedError = Field(
        description="Validation error details"
    )
    
    request: Optional[Dict[str, str]] = Field(
        default=None,
        description="Request information",
        examples=[{
            "path": "/api/v1/data/fetch",
            "method": "POST",
            "request_id": "req_123"
        }]
    )
    
    support: Dict[str, str] = Field(
        description="Support information",
        examples=[{
            "recovery_suggestion": "MODIFY_REQUEST",
            "documentation_url": "https://api.stockpred.com/docs",
            "examples_url": "https://api.stockpred.com/examples"
        }]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Request validation failed",
                        "category": "validation",
                        "severity": "medium",
                        "timestamp": "2023-01-01T12:00:00Z",
                        "details": [
                            {
                                "code": "FIELD_REQUIRED",
                                "message": "Field is required",
                                "field": "symbol",
                                "location": ["body", "symbol"]
                            },
                            {
                                "code": "VALUE_ERROR",
                                "message": "Invalid date format",
                                "field": "start_date",
                                "value": "invalid-date",
                                "location": ["body", "start_date"]
                            }
                        ],
                        "recovery_suggestion": "MODIFY_REQUEST"
                    }
                }
            ]
        }
    )

class AuthenticationErrorResponse(BaseErrorModel):
    """Authentication error response"""
    
    error: DetailedError = Field(
        description="Authentication error details"
    )
    
    support: Dict[str, str] = Field(
        description="Authentication support information",
        examples=[{
            "recovery_suggestion": "CONTACT_SUPPORT",
            "login_url": "https://app.stockpred.com/login",
            "documentation_url": "https://api.stockpred.com/docs/auth"
        }]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "INVALID_TOKEN",
                        "message": "Authentication token is invalid or expired",
                        "category": "authentication",
                        "severity": "high",
                        "timestamp": "2023-01-01T12:00:00Z",
                        "recovery_suggestion": "CONTACT_SUPPORT"
                    },
                    "support": {
                        "recovery_suggestion": "CONTACT_SUPPORT",
                        "login_url": "https://app.stockpred.com/login"
                    }
                }
            ]
        }
    )

class ResourceNotFoundErrorResponse(BaseErrorModel):
    """Resource not found error response"""
    
    error: DetailedError = Field(
        description="Resource not found error details"
    )
    
    support: Dict[str, str] = Field(
        description="Resource support information",
        examples=[{
            "recovery_suggestion": "MODIFY_REQUEST",
            "search_url": "https://api.stockpred.com/search",
            "documentation_url": "https://api.stockpred.com/docs/resources"
        }]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "RESOURCE_NOT_FOUND", 
                        "message": "Symbol 'INVALID' not found",
                        "category": "resource",
                        "severity": "medium",
                        "timestamp": "2023-01-01T12:00:00Z",
                        "details": [
                            {
                                "code": "SYMBOL_NOT_FOUND",
                                "message": "The requested symbol does not exist in our database",
                                "field": "symbol",
                                "value": "INVALID"
                            }
                        ],
                        "recovery_suggestion": "MODIFY_REQUEST"
                    }
                }
            ]
        }
    )

class RateLimitErrorResponse(BaseErrorModel):
    """Rate limit error response"""
    
    error: DetailedError = Field(
        description="Rate limit error details"
    )
    
    support: Dict[str, Union[str, int]] = Field(
        description="Rate limiting support information",
        examples=[{
            "recovery_suggestion": "RETRY_WITH_BACKOFF",
            "retry_after_seconds": 3600,
            "upgrade_url": "https://app.stockpred.com/upgrade",
            "documentation_url": "https://api.stockpred.com/docs/rate-limits"
        }]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Rate limit exceeded: 1000 requests per hour",
                        "category": "rate_limiting",
                        "severity": "medium",
                        "timestamp": "2023-01-01T12:00:00Z",
                        "retry_after": 3600,
                        "details": [
                            {
                                "code": "HOURLY_LIMIT_EXCEEDED",
                                "message": "You have exceeded your hourly request limit",
                                "value": 1000
                            }
                        ],
                        "recovery_suggestion": "RETRY_WITH_BACKOFF"
                    }
                }
            ]
        }
    )

class InternalServerErrorResponse(BaseErrorModel):
    """Internal server error response"""
    
    error: DetailedError = Field(
        description="Internal server error details"
    )
    
    support: Dict[str, str] = Field(
        description="Internal error support information",
        examples=[{
            "recovery_suggestion": "CONTACT_SUPPORT",
            "incident_id": "inc_abc123",
            "support_email": "support@stockpred.com",
            "documentation_url": "https://api.stockpred.com/docs/errors"
        }]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected error occurred",
                        "category": "system",
                        "severity": "critical",
                        "timestamp": "2023-01-01T12:00:00Z",
                        "context": {
                            "request_id": "req_abc123",
                            "correlation_id": "corr_xyz789"
                        },
                        "recovery_suggestion": "CONTACT_SUPPORT"
                    },
                    "support": {
                        "incident_id": "inc_abc123",
                        "support_email": "support@stockpred.com"
                    }
                }
            ]
        }
    )

# ============================================
# Domain-Specific Error Responses
# ============================================

class DataErrorResponse(BaseErrorModel):
    """Data-related error response"""
    
    error: DetailedError = Field(
        description="Data error details"
    )
    
    support: Dict[str, str] = Field(
        description="Data error support information",
        examples=[{
            "recovery_suggestion": "RETRY_WITH_BACKOFF",
            "alternative_sources": "yahoo_finance,alpha_vantage",
            "documentation_url": "https://api.stockpred.com/docs/data-sources"
        }]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "DATA_SOURCE_UNAVAILABLE",
                        "message": "Primary data source is temporarily unavailable",
                        "category": "external_service",
                        "severity": "high",
                        "timestamp": "2023-01-01T12:00:00Z",
                        "details": [
                            {
                                "code": "SERVICE_TIMEOUT",
                                "message": "Data provider API request timed out",
                                "field": "source",
                                "value": "yahoo_finance"
                            }
                        ],
                        "recovery_suggestion": "RETRY_WITH_BACKOFF"
                    }
                }
            ]
        }
    )

class ModelErrorResponse(BaseErrorModel):
    """Model-related error response"""
    
    error: DetailedError = Field(
        description="Model error details"
    )
    
    support: Dict[str, str] = Field(
        description="Model error support information",
        examples=[{
            "recovery_suggestion": "MODIFY_REQUEST",
            "model_status_url": "https://api.stockpred.com/models/status",
            "documentation_url": "https://api.stockpred.com/docs/models"
        }]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "MODEL_TRAINING_FAILED",
                        "message": "Model training failed due to insufficient data",
                        "category": "model",
                        "severity": "high",
                        "timestamp": "2023-01-01T12:00:00Z",
                        "details": [
                            {
                                "code": "INSUFFICIENT_DATA",
                                "message": "Minimum 100 data points required for training",
                                "field": "training_data",
                                "value": 50
                            }
                        ],
                        "recovery_suggestion": "MODIFY_REQUEST"
                    }
                }
            ]
        }
    )

class TradingErrorResponse(BaseErrorModel):
    """Trading-related error response"""
    
    error: DetailedError = Field(
        description="Trading error details"
    )
    
    support: Dict[str, str] = Field(
        description="Trading error support information",
        examples=[{
            "recovery_suggestion": "MODIFY_REQUEST",
            "trading_status_url": "https://api.stockpred.com/trading/status",
            "documentation_url": "https://api.stockpred.com/docs/trading"
        }]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "INSUFFICIENT_BALANCE",
                        "message": "Insufficient balance for trading operation",
                        "category": "trading",
                        "severity": "medium",
                        "timestamp": "2023-01-01T12:00:00Z",
                        "details": [
                            {
                                "code": "BALANCE_TOO_LOW",
                                "message": "Required balance: $1000, Available: $500",
                                "field": "amount",
                                "value": 1000
                            }
                        ],
                        "recovery_suggestion": "MODIFY_REQUEST"
                    }
                }
            ]
        }
    )

# ============================================
# Batch and Multi-Error Responses
# ============================================

class BatchErrorResponse(BaseErrorModel):
    """Batch operation error response"""
    
    overall_status: Literal["partial_success", "failed"] = Field(
        description="Overall batch operation status"
    )
    
    summary: Dict[str, int] = Field(
        description="Error summary statistics",
        examples=[{
            "total_items": 100,
            "successful": 75,
            "failed": 25,
            "error_rate": 0.25
        }]
    )
    
    errors: List[DetailedError] = Field(
        description="List of errors encountered",
        max_length=100
    )
    
    failed_items: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Items that failed processing"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "overall_status": "partial_success",
                    "summary": {
                        "total_items": 100,
                        "successful": 85,
                        "failed": 15,
                        "error_rate": 0.15
                    },
                    "errors": [
                        {
                            "code": "SYMBOL_NOT_FOUND",
                            "message": "Symbol 'INVALID1' not found",
                            "category": "resource",
                            "severity": "medium"
                        }
                    ],
                    "failed_items": [
                        {"symbol": "INVALID1", "index": 15},
                        {"symbol": "INVALID2", "index": 42}
                    ]
                }
            ]
        }
    )

class MultiErrorResponse(BaseErrorModel):
    """Multiple validation errors response"""
    
    error: Dict[str, Any] = Field(
        description="Primary error information"
    )
    
    validation_errors: List[ErrorDetail] = Field(
        description="All validation errors found",
        max_length=50
    )
    
    error_count: int = Field(
        description="Total number of errors",
        examples=[5]
    )
    
    fields_with_errors: List[str] = Field(
        description="List of fields that have errors",
        examples=[["symbol", "start_date", "amount"]]
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "MULTIPLE_VALIDATION_ERRORS",
                        "message": "Multiple validation errors found",
                        "category": "validation",
                        "severity": "medium",
                        "timestamp": "2023-01-01T12:00:00Z"
                    },
                    "validation_errors": [
                        {
                            "code": "FIELD_REQUIRED",
                            "message": "Symbol is required",
                            "field": "symbol"
                        },
                        {
                            "code": "INVALID_FORMAT",
                            "message": "Invalid date format",
                            "field": "start_date",
                            "value": "invalid-date"
                        }
                    ],
                    "error_count": 2,
                    "fields_with_errors": ["symbol", "start_date"]
                }
            ]
        }
    )

# ============================================
# Health Check and System Status Errors
# ============================================

class HealthCheckErrorResponse(BaseErrorModel):
    """Health check error response"""
    
    status: Literal["unhealthy", "degraded"] = Field(
        description="Overall system health status"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    
    services: Dict[str, Dict[str, Any]] = Field(
        description="Individual service health status",
        examples=[{
            "database": {
                "status": "healthy",
                "response_time_ms": 15
            },
            "data_provider": {
                "status": "unhealthy", 
                "error": "Connection timeout",
                "last_successful": "2023-01-01T11:45:00Z"
            }
        }]
    )
    
    errors: List[ErrorDetail] = Field(
        description="System health errors"
    )
    
    recovery_estimate: Optional[str] = Field(
        default=None,
        description="Estimated recovery time",
        examples=["5 minutes", "unknown"]
    )

# ============================================
# Error Response Factory
# ============================================

class ErrorResponseFactory:
    """Factory for creating standardized error responses"""
    
    @staticmethod
    def create_validation_error(
        message: str,
        details: List[ErrorDetail],
        context: Optional[ErrorContext] = None
    ) -> ValidationErrorResponse:
        """Create a validation error response"""
        
        error = DetailedError(
            code="VALIDATION_ERROR",
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            context=context,
            recovery_suggestion=ErrorRecovery.MODIFY_REQUEST
        )
        
        return ValidationErrorResponse(
            error=error,
            support={
                "recovery_suggestion": "MODIFY_REQUEST",
                "documentation_url": "https://api.stockpred.com/docs"
            }
        )
    
    @staticmethod
    def create_not_found_error(
        resource_type: str,
        identifier: str,
        context: Optional[ErrorContext] = None
    ) -> ResourceNotFoundErrorResponse:
        """Create a resource not found error response"""
        
        error = DetailedError(
            code="RESOURCE_NOT_FOUND",
            message=f"{resource_type} '{identifier}' not found",
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.MEDIUM,
            details=[
                ErrorDetail(
                    code=f"{resource_type.upper()}_NOT_FOUND",
                    message=f"The requested {resource_type} does not exist",
                    value=identifier
                )
            ],
            context=context,
            recovery_suggestion=ErrorRecovery.MODIFY_REQUEST
        )
        
        return ResourceNotFoundErrorResponse(
            error=error,
            support={
                "recovery_suggestion": "MODIFY_REQUEST",
                "search_url": "https://api.stockpred.com/search"
            }
        )
    
    @staticmethod
    def create_rate_limit_error(
        limit: int,
        window: str,
        retry_after: int,
        context: Optional[ErrorContext] = None
    ) -> RateLimitErrorResponse:
        """Create a rate limit error response"""
        
        error = DetailedError(
            code="RATE_LIMIT_EXCEEDED",
            message=f"Rate limit exceeded: {limit} requests per {window}",
            category=ErrorCategory.RATE_LIMITING,
            severity=ErrorSeverity.MEDIUM,
            retry_after=retry_after,
            context=context,
            recovery_suggestion=ErrorRecovery.RETRY_WITH_BACKOFF
        )
        
        return RateLimitErrorResponse(
            error=error,
            support={
                "recovery_suggestion": "RETRY_WITH_BACKOFF",
                "retry_after_seconds": retry_after,
                "upgrade_url": "https://app.stockpred.com/upgrade"
            }
        )
    
    @staticmethod
    def create_internal_server_error(
        incident_id: str,
        context: Optional[ErrorContext] = None
    ) -> InternalServerErrorResponse:
        """Create an internal server error response"""
        
        error = DetailedError(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            recovery_suggestion=ErrorRecovery.CONTACT_SUPPORT
        )
        
        return InternalServerErrorResponse(
            error=error,
            support={
                "recovery_suggestion": "CONTACT_SUPPORT",
                "incident_id": incident_id,
                "support_email": "support@stockpred.com"
            }
        )

# ============================================
# Export All Error Schemas
# ============================================

__all__ = [
    # Enums
    "ErrorCategory",
    "ErrorSeverity", 
    "ErrorRecovery",
    
    # Base models
    "BaseErrorModel",
    "ErrorDetail",
    "ErrorContext",
    "DetailedError",
    
    # Standard responses
    "ErrorResponse",
    "ValidationErrorResponse",
    "AuthenticationErrorResponse",
    "ResourceNotFoundErrorResponse",
    "RateLimitErrorResponse",
    "InternalServerErrorResponse",
    
    # Domain-specific responses
    "DataErrorResponse",
    "ModelErrorResponse", 
    "TradingErrorResponse",
    
    # Multi-error responses
    "BatchErrorResponse",
    "MultiErrorResponse",
    "HealthCheckErrorResponse",
    
    # Factory
    "ErrorResponseFactory",
    
    # Constants
    "HTTP_STATUS_CODES",
]
