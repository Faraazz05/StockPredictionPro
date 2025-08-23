# ============================================
# StockPredictionPro - src/api/middleware/logging.py
# Advanced logging middleware for FastAPI with structured logging, monitoring, and audit trails
# ============================================

import json
import time
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Set
from urllib.parse import urlparse, parse_qs
import asyncio
import traceback

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
import structlog

from ...utils.logger import get_logger, setup_structured_logging
from ...utils.config_loader import load_config

logger = get_logger('api.middleware.logging')

# ============================================
# Logging Configuration
# ============================================

class LoggingConfig:
    """Logging middleware configuration"""
    
    def __init__(self):
        self.config = load_config('app_config.yaml')
        logging_config = self.config.get('logging', {})
        
        # Basic settings
        self.enabled = logging_config.get('enabled', True)
        self.log_level = logging_config.get('log_level', 'INFO')
        self.log_format = logging_config.get('log_format', 'json')
        
        # Request/Response logging
        self.log_requests = logging_config.get('log_requests', True)
        self.log_responses = logging_config.get('log_responses', True)
        self.log_request_body = logging_config.get('log_request_body', False)
        self.log_response_body = logging_config.get('log_response_body', False)
        
        # Performance monitoring
        self.log_performance = logging_config.get('log_performance', True)
        self.slow_request_threshold = logging_config.get('slow_request_threshold', 1000)  # ms
        
        # Security and audit
        self.log_client_info = logging_config.get('log_client_info', True)
        self.log_headers = logging_config.get('log_headers', True)
        self.sensitive_headers = set(logging_config.get('sensitive_headers', [
            'authorization', 'cookie', 'x-api-key', 'x-auth-token'
        ]))
        
        # Filtering
        self.excluded_paths = set(logging_config.get('excluded_paths', [
            '/health', '/metrics', '/favicon.ico'
        ]))
        self.excluded_methods = set(logging_config.get('excluded_methods', []))
        self.log_only_errors = logging_config.get('log_only_errors', False)
        
        # Body logging limits
        self.max_body_size = logging_config.get('max_body_size', 10000)  # bytes
        self.body_content_types = set(logging_config.get('body_content_types', [
            'application/json', 'application/xml', 'text/plain', 'text/csv'
        ]))
        
        # Sampling
        self.sampling_rate = logging_config.get('sampling_rate', 1.0)  # 1.0 = 100%
        
        # Correlation
        self.enable_correlation_id = logging_config.get('enable_correlation_id', True)
        self.correlation_header = logging_config.get('correlation_header', 'X-Correlation-ID')
        
        # Environment-specific settings
        environment = self.config.get('environment', 'development')
        if environment == 'production':
            self._apply_production_settings()
    
    def _apply_production_settings(self):
        """Apply production-optimized logging settings"""
        self.log_request_body = False  # Don't log request bodies in production
        self.log_response_body = False  # Don't log response bodies in production
        self.sampling_rate = 0.1  # Sample only 10% of requests
        self.slow_request_threshold = 500  # Lower threshold for production

# Global logging config
logging_config = LoggingConfig()

# ============================================
# Request Context Manager
# ============================================

class RequestContext:
    """Request context for structured logging"""
    
    def __init__(self, request: Request):
        self.request_id = str(uuid.uuid4())
        self.correlation_id = self._extract_correlation_id(request)
        self.start_time = time.time()
        self.timestamp = datetime.utcnow()
        
        # Request information
        self.method = request.method
        self.path = str(request.url.path)
        self.query_params = dict(request.query_params)
        self.path_params = getattr(request, 'path_params', {})
        
        # Client information
        self.client_ip = self._get_client_ip(request)
        self.user_agent = request.headers.get('user-agent', '')
        self.referer = request.headers.get('referer', '')
        
        # Headers
        self.headers = self._sanitize_headers(dict(request.headers))
        
        # User information (if available from auth middleware)
        self.user_id = None
        self.username = None
        if hasattr(request.state, 'user'):
            user = request.state.user
            self.user_id = getattr(user, 'user_id', None)
            self.username = getattr(user, 'username', None)
        
        # Performance tracking
        self.processing_time = 0
        self.database_time = 0
        self.cache_time = 0
        
        # Error tracking
        self.error_count = 0
        self.warnings = []
        self.exceptions = []
    
    def _extract_correlation_id(self, request: Request) -> str:
        """Extract or generate correlation ID"""
        
        correlation_id = request.headers.get(
            logging_config.correlation_header.lower(),
            str(uuid.uuid4())
        )
        return correlation_id
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address with proxy support"""
        
        # Check for forwarded headers (common in production behind proxies)
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(',')[0].strip()
        
        forwarded = request.headers.get('x-forwarded')
        if forwarded:
            return forwarded.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fall back to direct client
        if request.client:
            return request.client.host
        
        return 'unknown'
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize sensitive headers for logging"""
        
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in logging_config.sensitive_headers:
                # Mask sensitive headers
                sanitized[key] = f"***{value[-4:] if len(value) > 4 else '***'}"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def update_performance(self, database_time: float = 0, cache_time: float = 0):
        """Update performance metrics"""
        self.database_time += database_time
        self.cache_time += cache_time
    
    def add_warning(self, warning: str):
        """Add warning to context"""
        self.warnings.append(warning)
    
    def add_exception(self, exception: Exception):
        """Add exception to context"""
        self.exceptions.append({
            'type': type(exception).__name__,
            'message': str(exception),
            'traceback': traceback.format_exc()
        })
        self.error_count += 1
    
    def finalize(self, response: Response):
        """Finalize context with response information"""
        self.processing_time = (time.time() - self.start_time) * 1000  # Convert to ms
        self.status_code = response.status_code
        self.response_size = len(response.body) if hasattr(response, 'body') else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging"""
        
        base_context = {
            'request_id': self.request_id,
            'correlation_id': self.correlation_id,
            'timestamp': self.timestamp.isoformat(),
            'method': self.method,
            'path': self.path,
            'query_params': self.query_params,
            'client_ip': self.client_ip,
            'processing_time_ms': round(self.processing_time, 2),
            'status_code': getattr(self, 'status_code', None)
        }
        
        # Add optional fields based on configuration
        if logging_config.log_client_info:
            base_context.update({
                'user_agent': self.user_agent,
                'referer': self.referer
            })
        
        if logging_config.log_headers:
            base_context['headers'] = self.headers
        
        if self.user_id:
            base_context.update({
                'user_id': self.user_id,
                'username': self.username
            })
        
        if logging_config.log_performance:
            base_context.update({
                'database_time_ms': round(self.database_time, 2),
                'cache_time_ms': round(self.cache_time, 2),
                'response_size_bytes': getattr(self, 'response_size', 0)
            })
        
        if self.warnings:
            base_context['warnings'] = self.warnings
        
        if self.exceptions:
            base_context['exceptions'] = self.exceptions
        
        return base_context

# ============================================
# Enhanced Logging Middleware
# ============================================

class EnhancedLoggingMiddleware(BaseHTTPMiddleware):
    """
    Enhanced logging middleware with structured logging, performance monitoring,
    and comprehensive audit trails.
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        
        # Initialize structured logging
        setup_structured_logging()
        self.struct_logger = structlog.get_logger("api.requests")
        
        # Statistics tracking
        self.request_count = 0
        self.error_count = 0
        self.slow_request_count = 0
        self.total_processing_time = 0
        
        # Sampling
        self.sample_counter = 0
        
        logger.info("Enhanced logging middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request/response logging
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response object
        """
        
        # Check if logging is enabled
        if not logging_config.enabled:
            return await call_next(request)
        
        # Check if path should be excluded
        if self._should_exclude_path(request.url.path):
            return await call_next(request)
        
        # Check if method should be excluded
        if request.method in logging_config.excluded_methods:
            return await call_next(request)
        
        # Apply sampling
        if not self._should_sample():
            return await call_next(request)
        
        # Create request context
        context = RequestContext(request)
        
        # Add correlation ID to request state for downstream use
        request.state.request_id = context.request_id
        request.state.correlation_id = context.correlation_id
        
        # Log request start
        await self._log_request_start(context, request)
        
        # Process request with error handling
        response = None
        try:
            # Add context to request state for use by other middleware/handlers
            request.state.logging_context = context
            
            response = await call_next(request)
            
            # Finalize context with response info
            context.finalize(response)
            
            # Add correlation ID to response headers
            if logging_config.enable_correlation_id:
                response.headers[logging_config.correlation_header] = context.correlation_id
                response.headers['X-Request-ID'] = context.request_id
            
            # Update statistics
            self._update_statistics(context)
            
            # Log request completion
            await self._log_request_complete(context, request, response)
            
        except Exception as e:
            # Handle exceptions during request processing
            context.add_exception(e)
            context.error_count += 1
            self.error_count += 1
            
            # Log the error
            await self._log_request_error(context, request, e)
            
            # Re-raise the exception
            raise
        
        return response
    
    def _should_exclude_path(self, path: str) -> bool:
        """Check if path should be excluded from logging"""
        return any(path.startswith(excluded) for excluded in logging_config.excluded_paths)
    
    def _should_sample(self) -> bool:
        """Determine if request should be sampled based on sampling rate"""
        if logging_config.sampling_rate >= 1.0:
            return True
        
        self.sample_counter += 1
        return (self.sample_counter % int(1 / logging_config.sampling_rate)) == 0
    
    async def _log_request_start(self, context: RequestContext, request: Request):
        """Log request start"""
        
        if not logging_config.log_requests:
            return
        
        log_data = {
            'event': 'request_started',
            'request_id': context.request_id,
            'correlation_id': context.correlation_id,
            'method': context.method,
            'path': context.path,
            'client_ip': context.client_ip,
            'timestamp': context.timestamp.isoformat()
        }
        
        # Add query parameters
        if context.query_params:
            log_data['query_params'] = context.query_params
        
        # Add user information if available
        if context.user_id:
            log_data['user_id'] = context.user_id
            log_data['username'] = context.username
        
        # Add request body if configured
        if logging_config.log_request_body:
            body = await self._get_request_body(request)
            if body:
                log_data['request_body'] = body
        
        self.struct_logger.info("Request started", **log_data)
    
    async def _log_request_complete(self, context: RequestContext, 
                                   request: Request, response: Response):
        """Log request completion"""
        
        if not logging_config.log_responses:
            return
        
        # Skip logging if only errors should be logged
        if logging_config.log_only_errors and response.status_code < 400:
            return
        
        log_data = context.to_dict()
        log_data['event'] = 'request_completed'
        
        # Determine log level based on response status
        if response.status_code >= 500:
            log_level = 'error'
        elif response.status_code >= 400:
            log_level = 'warning'
        elif context.processing_time > logging_config.slow_request_threshold:
            log_level = 'warning'
            log_data['slow_request'] = True
        else:
            log_level = 'info'
        
        # Add response body if configured
        if logging_config.log_response_body and hasattr(response, 'body'):
            body = await self._get_response_body(response)
            if body:
                log_data['response_body'] = body
        
        getattr(self.struct_logger, log_level)("Request completed", **log_data)
    
    async def _log_request_error(self, context: RequestContext, 
                                request: Request, exception: Exception):
        """Log request error"""
        
        log_data = context.to_dict()
        log_data.update({
            'event': 'request_error',
            'error_type': type(exception).__name__,
            'error_message': str(exception),
            'traceback': traceback.format_exc()
        })
        
        self.struct_logger.error("Request error", **log_data)
    
    async def _get_request_body(self, request: Request) -> Optional[str]:
        """Get request body for logging"""
        
        try:
            # Check content type
            content_type = request.headers.get('content-type', '').lower()
            if not any(ct in content_type for ct in logging_config.body_content_types):
                return None
            
            # Get body
            body = await request.body()
            
            # Check size limit
            if len(body) > logging_config.max_body_size:
                return f"<body too large: {len(body)} bytes>"
            
            # Decode body
            try:
                return body.decode('utf-8')
            except UnicodeDecodeError:
                return f"<binary data: {len(body)} bytes>"
                
        except Exception as e:
            logger.warning(f"Failed to read request body: {e}")
            return None
    
    async def _get_response_body(self, response: Response) -> Optional[str]:
        """Get response body for logging"""
        
        try:
            if not hasattr(response, 'body'):
                return None
            
            body = response.body
            
            # Check size limit
            if len(body) > logging_config.max_body_size:
                return f"<body too large: {len(body)} bytes>"
            
            # Check if it's likely text
            try:
                return body.decode('utf-8')
            except UnicodeDecodeError:
                return f"<binary data: {len(body)} bytes>"
                
        except Exception as e:
            logger.warning(f"Failed to read response body: {e}")
            return None
    
    def _update_statistics(self, context: RequestContext):
        """Update middleware statistics"""
        
        self.request_count += 1
        self.total_processing_time += context.processing_time
        
        if context.processing_time > logging_config.slow_request_threshold:
            self.slow_request_count += 1
        
        if hasattr(context, 'status_code') and context.status_code >= 400:
            self.error_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        
        avg_processing_time = (
            self.total_processing_time / max(self.request_count, 1)
        )
        
        return {
            'total_requests': self.request_count,
            'error_requests': self.error_count,
            'slow_requests': self.slow_request_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'slow_request_rate': self.slow_request_count / max(self.request_count, 1),
            'avg_processing_time_ms': round(avg_processing_time, 2),
            'total_processing_time_ms': round(self.total_processing_time, 2),
            'sampling_rate': logging_config.sampling_rate
        }

# ============================================
# Performance Monitoring Utilities
# ============================================

class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    def __init__(self, context: RequestContext):
        self.context = context
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = (time.time() - self.start_time) * 1000
        self.context.update_performance(database_time=elapsed_time)

def monitor_database_time(context: RequestContext):
    """Context manager for monitoring database query time"""
    return PerformanceMonitor(context)

# ============================================
# Logging Utilities
# ============================================

def get_request_logger(request: Request) -> structlog.BoundLogger:
    """
    Get a logger bound to request context
    
    Args:
        request: FastAPI request object
        
    Returns:
        Bound structlog logger with request context
    """
    
    base_logger = structlog.get_logger("api.request")
    
    # Get context from request state
    if hasattr(request.state, 'logging_context'):
        context = request.state.logging_context
        return base_logger.bind(
            request_id=context.request_id,
            correlation_id=context.correlation_id,
            user_id=context.user_id,
            client_ip=context.client_ip
        )
    
    # Fallback to basic binding
    return base_logger.bind(
        request_id=getattr(request.state, 'request_id', 'unknown'),
        correlation_id=getattr(request.state, 'correlation_id', 'unknown')
    )

def log_business_event(
    request: Request,
    event_name: str,
    event_data: Dict[str, Any],
    level: str = 'info'
):
    """
    Log business/application event with request context
    
    Args:
        request: FastAPI request object
        event_name: Name of the business event
        event_data: Event-specific data
        level: Log level
    """
    
    request_logger = get_request_logger(request)
    
    log_data = {
        'event': event_name,
        'event_type': 'business',
        'timestamp': datetime.utcnow().isoformat(),
        **event_data
    }
    
    getattr(request_logger, level)(f"Business event: {event_name}", **log_data)

def create_audit_log(
    request: Request,
    action: str,
    resource: str,
    resource_id: Optional[str] = None,
    changes: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Create audit log entry
    
    Args:
        request: FastAPI request object
        action: Action performed (CREATE, UPDATE, DELETE, etc.)
        resource: Resource type
        resource_id: Resource identifier
        changes: Changes made (for updates)
        metadata: Additional metadata
    """
    
    request_logger = get_request_logger(request)
    
    audit_data = {
        'event': 'audit_log',
        'action': action,
        'resource': resource,
        'resource_id': resource_id,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if changes:
        audit_data['changes'] = changes
    
    if metadata:
        audit_data['metadata'] = metadata
    
    request_logger.info(f"Audit: {action} {resource}", **audit_data)

# ============================================
# Health and Monitoring Endpoints
# ============================================

def create_logging_health_check() -> Dict[str, Any]:
    """Create health check data for logging system"""
    
    return {
        'status': 'healthy' if logging_config.enabled else 'disabled',
        'config': {
            'enabled': logging_config.enabled,
            'log_level': logging_config.log_level,
            'log_format': logging_config.log_format,
            'sampling_rate': logging_config.sampling_rate,
            'slow_threshold_ms': logging_config.slow_request_threshold
        }
    }

# ============================================
# Export Components
# ============================================

__all__ = [
    # Main middleware
    "EnhancedLoggingMiddleware",
    
    # Configuration
    "LoggingConfig",
    "logging_config",
    
    # Context management
    "RequestContext",
    "PerformanceMonitor",
    "monitor_database_time",
    
    # Logging utilities
    "get_request_logger",
    "log_business_event", 
    "create_audit_log",
    
    # Health monitoring
    "create_logging_health_check",
]
