# ============================================
# StockPredictionPro - src/api/middleware/cors.py
# CORS (Cross-Origin Resource Sharing) middleware for FastAPI with advanced configuration and security
# ============================================

import re
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Pattern
from urllib.parse import urlparse

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware as FastAPICORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from ...utils.logger import get_logger
from ...utils.config_loader import load_config

logger = get_logger('api.middleware.cors')

# ============================================
# CORS Configuration
# ============================================

class CORSConfig:
    """CORS configuration management"""
    
    def __init__(self):
        self.config = load_config('app_config.yaml')
        cors_config = self.config.get('cors', {})
        
        # Origin settings
        self.allowed_origins = cors_config.get('allowed_origins', ['*'])
        self.allowed_origin_regex = cors_config.get('allowed_origin_regex')
        self.allow_origin_regex_compiled = None
        
        if self.allowed_origin_regex:
            try:
                self.allow_origin_regex_compiled = re.compile(self.allowed_origin_regex)
            except re.error as e:
                logger.error(f"Invalid origin regex pattern: {e}")
        
        # Method settings
        self.allowed_methods = cors_config.get('allowed_methods', ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH', 'HEAD'])
        
        # Header settings
        self.allowed_headers = cors_config.get('allowed_headers', ['*'])
        self.exposed_headers = cors_config.get('exposed_headers', [])
        
        # Credential settings
        self.allow_credentials = cors_config.get('allow_credentials', True)
        
        # Cache settings
        self.max_age = cors_config.get('max_age', 3600)  # 1 hour default
        
        # Security settings
        self.strict_mode = cors_config.get('strict_mode', False)
        self.log_blocked_origins = cors_config.get('log_blocked_origins', True)
        self.block_null_origin = cors_config.get('block_null_origin', True)
        
        # Environment-specific settings
        environment = self.config.get('environment', 'development')
        if environment == 'production':
            self._apply_production_defaults()
        elif environment == 'development':
            self._apply_development_defaults()
    
    def _apply_production_defaults(self):
        """Apply secure defaults for production"""
        if '*' in self.allowed_origins:
            logger.warning("Wildcard origins detected in production environment")
            # Remove wildcard if specific origins are also defined
            if len(self.allowed_origins) > 1:
                self.allowed_origins = [origin for origin in self.allowed_origins if origin != '*']
        
        # Ensure credentials are handled securely
        if self.allow_credentials and '*' in self.allowed_origins:
            logger.error("Cannot allow credentials with wildcard origins in production")
            self.allow_credentials = False
        
        self.strict_mode = True
        self.log_blocked_origins = True
    
    def _apply_development_defaults(self):
        """Apply permissive defaults for development"""
        # Add common development origins if not specified
        common_dev_origins = [
            'http://localhost:3000',
            'http://localhost:3001', 
            'http://localhost:8080',
            'http://127.0.0.1:3000',
            'http://127.0.0.1:8080'
        ]
        
        for origin in common_dev_origins:
            if origin not in self.allowed_origins:
                self.allowed_origins.append(origin)

# Global CORS config instance
cors_config = CORSConfig()

# ============================================
# Enhanced CORS Middleware
# ============================================

class EnhancedCORSMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CORS middleware with advanced security features and logging.
    
    Extends the standard FastAPI CORS middleware with:
    - Origin validation and logging
    - Security headers
    - Request monitoring
    - Environment-specific configuration
    """
    
    def __init__(
        self,
        app,
        allow_origins: List[str] = None,
        allow_origin_regex: Optional[str] = None,
        allow_methods: List[str] = None,
        allow_headers: List[str] = None,
        allow_credentials: bool = None,
        expose_headers: List[str] = None,
        max_age: int = None,
        **kwargs
    ):
        super().__init__(app, **kwargs)
        
        # Use provided config or fall back to global config
        self.allowed_origins = allow_origins or cors_config.allowed_origins
        self.allowed_methods = allow_methods or cors_config.allowed_methods
        self.allowed_headers = allow_headers or cors_config.allowed_headers
        self.exposed_headers = expose_headers or cors_config.exposed_headers
        self.allow_credentials = allow_credentials if allow_credentials is not None else cors_config.allow_credentials
        self.max_age = max_age or cors_config.max_age
        
        # Compile origin regex if provided
        self.origin_regex = None
        if allow_origin_regex:
            try:
                self.origin_regex = re.compile(allow_origin_regex)
            except re.error as e:
                logger.error(f"Invalid CORS origin regex: {e}")
        elif cors_config.allow_origin_regex_compiled:
            self.origin_regex = cors_config.allow_origin_regex_compiled
        
        # Initialize tracking
        self.request_count = 0
        self.blocked_requests = 0
        self.allowed_origins_cache = set()
        
        logger.info(f"Enhanced CORS middleware initialized with {len(self.allowed_origins)} allowed origins")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process CORS for incoming requests
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response with CORS headers
        """
        
        self.request_count += 1
        origin = request.headers.get('origin')
        method = request.method
        
        # Log request details
        logger.debug(f"CORS request: {method} {request.url.path} from origin: {origin}")
        
        # Handle preflight requests
        if method == 'OPTIONS':
            return await self._handle_preflight_request(request, origin)
        
        # Validate origin for actual requests
        if origin:
            is_allowed = self._is_origin_allowed(origin)
            
            if not is_allowed:
                self.blocked_requests += 1
                
                if cors_config.log_blocked_origins:
                    logger.warning(
                        f"Blocked CORS request from unauthorized origin: {origin}",
                        extra={
                            "origin": origin,
                            "path": request.url.path,
                            "method": method,
                            "user_agent": request.headers.get("user-agent", "unknown")
                        }
                    )
                
                if cors_config.strict_mode:
                    return self._create_cors_error_response("Origin not allowed", origin)
        
        # Process the request
        response = await call_next(request)
        
        # Add CORS headers to response
        if origin and self._is_origin_allowed(origin):
            self._add_cors_headers(response, origin, request)
        
        return response
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """
        Check if origin is allowed
        
        Args:
            origin: Request origin
            
        Returns:
            True if origin is allowed
        """
        
        if not origin:
            return not cors_config.block_null_origin
        
        # Check cache first
        if origin in self.allowed_origins_cache:
            return True
        
        # Check wildcard
        if '*' in self.allowed_origins:
            self.allowed_origins_cache.add(origin)
            return True
        
        # Check exact match
        if origin in self.allowed_origins:
            self.allowed_origins_cache.add(origin)
            return True
        
        # Check regex pattern
        if self.origin_regex and self.origin_regex.match(origin):
            self.allowed_origins_cache.add(origin)
            return True
        
        # Check for development localhost variations
        if self._is_development_origin(origin):
            return True
        
        return False
    
    def _is_development_origin(self, origin: str) -> bool:
        """Check if origin is a development origin (localhost variations)"""
        
        try:
            parsed = urlparse(origin)
            
            # Allow localhost and 127.0.0.1 with any port in development
            if cors_config.config.get('environment') == 'development':
                if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def _handle_preflight_request(self, request: Request, origin: str) -> Response:
        """
        Handle CORS preflight (OPTIONS) requests
        
        Args:
            request: FastAPI request object
            origin: Request origin
            
        Returns:
            Preflight response
        """
        
        # Get requested method and headers
        requested_method = request.headers.get('access-control-request-method')
        requested_headers = request.headers.get('access-control-request-headers', '')
        
        logger.debug(
            f"CORS preflight request: method={requested_method}, "
            f"headers={requested_headers}, origin={origin}"
        )
        
        # Validate origin
        if origin and not self._is_origin_allowed(origin):
            self.blocked_requests += 1
            
            if cors_config.log_blocked_origins:
                logger.warning(f"Blocked preflight request from unauthorized origin: {origin}")
            
            return self._create_cors_error_response("Origin not allowed for preflight", origin)
        
        # Validate requested method
        if requested_method and requested_method not in self.allowed_methods:
            logger.warning(f"Blocked preflight request with unauthorized method: {requested_method}")
            return self._create_cors_error_response("Method not allowed", origin)
        
        # Create preflight response
        response = Response(status_code=200)
        
        if origin:
            self._add_preflight_headers(response, origin, requested_method, requested_headers)
        
        return response
    
    def _add_cors_headers(self, response: Response, origin: str, request: Request):
        """
        Add CORS headers to response
        
        Args:
            response: Response object
            origin: Request origin
            request: Request object
        """
        
        # Access-Control-Allow-Origin
        if '*' in self.allowed_origins and not self.allow_credentials:
            response.headers['access-control-allow-origin'] = '*'
        else:
            response.headers['access-control-allow-origin'] = origin
        
        # Access-Control-Allow-Credentials
        if self.allow_credentials:
            response.headers['access-control-allow-credentials'] = 'true'
        
        # Access-Control-Expose-Headers
        if self.exposed_headers:
            response.headers['access-control-expose-headers'] = ', '.join(self.exposed_headers)
        
        # Vary header for caching
        vary_headers = []
        if 'vary' in response.headers:
            vary_headers = [h.strip() for h in response.headers['vary'].split(',')]
        
        if 'Origin' not in vary_headers:
            vary_headers.append('Origin')
        
        response.headers['vary'] = ', '.join(vary_headers)
        
        # Add security headers
        self._add_security_headers(response)
    
    def _add_preflight_headers(self, response: Response, origin: str, 
                             requested_method: str, requested_headers: str):
        """
        Add CORS headers to preflight response
        
        Args:
            response: Response object
            origin: Request origin
            requested_method: Requested method
            requested_headers: Requested headers
        """
        
        # Access-Control-Allow-Origin
        if '*' in self.allowed_origins and not self.allow_credentials:
            response.headers['access-control-allow-origin'] = '*'
        else:
            response.headers['access-control-allow-origin'] = origin
        
        # Access-Control-Allow-Methods
        if '*' in self.allowed_methods:
            response.headers['access-control-allow-methods'] = ', '.join([
                'GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH', 'HEAD'
            ])
        else:
            response.headers['access-control-allow-methods'] = ', '.join(self.allowed_methods)
        
        # Access-Control-Allow-Headers
        if self.allowed_headers:
            if '*' in self.allowed_headers:
                # Echo back requested headers if using wildcard
                if requested_headers:
                    response.headers['access-control-allow-headers'] = requested_headers
                else:
                    response.headers['access-control-allow-headers'] = '*'
            else:
                response.headers['access-control-allow-headers'] = ', '.join(self.allowed_headers)
        
        # Access-Control-Allow-Credentials
        if self.allow_credentials:
            response.headers['access-control-allow-credentials'] = 'true'
        
        # Access-Control-Max-Age
        if self.max_age:
            response.headers['access-control-max-age'] = str(self.max_age)
        
        # Add security headers
        self._add_security_headers(response)
    
    def _add_security_headers(self, response: Response):
        """Add security-related headers to response"""
        
        # X-Content-Type-Options
        if 'x-content-type-options' not in response.headers:
            response.headers['x-content-type-options'] = 'nosniff'
        
        # X-Frame-Options (only if not already set)
        if 'x-frame-options' not in response.headers:
            response.headers['x-frame-options'] = 'DENY'
        
        # Referrer-Policy
        if 'referrer-policy' not in response.headers:
            response.headers['referrer-policy'] = 'strict-origin-when-cross-origin'
    
    def _create_cors_error_response(self, message: str, origin: str = None) -> Response:
        """
        Create CORS error response
        
        Args:
            message: Error message
            origin: Request origin
            
        Returns:
            Error response
        """
        
        return Response(
            content=f'{{"error": "{message}", "origin": "{origin or "unknown"}"}}',
            status_code=403,
            media_type='application/json',
            headers={
                'content-type': 'application/json',
                'x-cors-error': 'true'
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get CORS middleware statistics"""
        
        return {
            'total_requests': self.request_count,
            'blocked_requests': self.blocked_requests,
            'allowed_origins_count': len(self.allowed_origins),
            'cached_origins_count': len(self.allowed_origins_cache),
            'allow_credentials': self.allow_credentials,
            'max_age': self.max_age,
            'block_rate': self.blocked_requests / max(self.request_count, 1)
        }

# ============================================
# CORS Utility Functions
# ============================================

def create_cors_middleware(
    allowed_origins: List[str] = None,
    allow_origin_regex: Optional[str] = None,
    allow_credentials: bool = None,
    allowed_methods: List[str] = None,
    allowed_headers: List[str] = None,
    **kwargs
) -> EnhancedCORSMiddleware:
    """
    Create CORS middleware with configuration
    
    Args:
        allowed_origins: List of allowed origins
        allow_origin_regex: Regex pattern for allowed origins
        allow_credentials: Whether to allow credentials
        allowed_methods: List of allowed HTTP methods
        allowed_headers: List of allowed headers
        **kwargs: Additional configuration
        
    Returns:
        Configured CORS middleware
    """
    
    return EnhancedCORSMiddleware(
        app=None,  # Will be set when added to FastAPI
        allow_origins=allowed_origins,
        allow_origin_regex=allow_origin_regex,
        allow_credentials=allow_credentials,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers,
        **kwargs
    )

def get_production_cors_config() -> Dict[str, Any]:
    """Get secure CORS configuration for production"""
    
    return {
        'allow_origins': [
            'https://yourapp.com',
            'https://www.yourapp.com',
            'https://api.yourapp.com'
        ],
        'allow_origin_regex': r'https://.*\.yourapp\.com',
        'allow_credentials': True,
        'allow_methods': ['GET', 'POST', 'PUT', 'DELETE'],
        'allow_headers': [
            'Authorization',
            'Content-Type',
            'X-Requested-With',
            'X-API-Key'
        ],
        'expose_headers': [
            'X-Total-Count',
            'X-Rate-Limit-Remaining',
            'X-Request-ID'
        ],
        'max_age': 3600
    }

def get_development_cors_config() -> Dict[str, Any]:
    """Get permissive CORS configuration for development"""
    
    return {
        'allow_origins': ['*'],
        'allow_credentials': False,  # Can't use credentials with wildcard
        'allow_methods': ['*'],
        'allow_headers': ['*'],
        'max_age': 0  # No caching in development
    }

def validate_cors_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate CORS configuration and return warnings
    
    Args:
        config: CORS configuration dictionary
        
    Returns:
        List of validation warnings
    """
    
    warnings = []
    
    # Check wildcard with credentials
    if (config.get('allow_origins') == ['*'] or '*' in config.get('allow_origins', [])) and \
       config.get('allow_credentials', False):
        warnings.append("Cannot use wildcard origins (*) with allow_credentials=True")
    
    # Check empty origins
    if not config.get('allow_origins'):
        warnings.append("No allowed origins specified - all requests will be blocked")
    
    # Check methods
    allowed_methods = config.get('allow_methods', [])
    if not allowed_methods:
        warnings.append("No allowed methods specified")
    elif 'OPTIONS' not in allowed_methods and '*' not in allowed_methods:
        warnings.append("OPTIONS method not in allowed_methods - preflight requests may fail")
    
    # Check headers
    if config.get('allow_headers') and 'content-type' not in \
       [h.lower() for h in config.get('allow_headers', [])] and '*' not in config.get('allow_headers', []):
        warnings.append("content-type not in allowed_headers - JSON requests may fail")
    
    return warnings

# ============================================
# CORS Debugging Utilities
# ============================================

class CORSDebugger:
    """CORS debugging utilities"""
    
    @staticmethod
    def analyze_request(request: Request) -> Dict[str, Any]:
        """
        Analyze CORS request and provide debugging information
        
        Args:
            request: FastAPI request object
            
        Returns:
            Analysis results
        """
        
        origin = request.headers.get('origin')
        method = request.method
        
        analysis = {
            'request_method': method,
            'origin': origin,
            'is_preflight': method == 'OPTIONS',
            'cors_headers': {},
            'recommendations': []
        }
        
        # Extract CORS-related headers
        for header_name, header_value in request.headers.items():
            if header_name.lower().startswith('access-control-'):
                analysis['cors_headers'][header_name] = header_value
        
        # Analyze and provide recommendations
        if not origin:
            analysis['recommendations'].append("Request has no Origin header - not a CORS request")
        
        if method == 'OPTIONS':
            requested_method = request.headers.get('access-control-request-method')
            if not requested_method:
                analysis['recommendations'].append("Preflight request missing Access-Control-Request-Method header")
        
        # Check if origin would be allowed
        if origin:
            is_allowed = cors_config.allowed_origins == ['*'] or origin in cors_config.allowed_origins
            analysis['origin_allowed'] = is_allowed
            
            if not is_allowed:
                analysis['recommendations'].append(f"Origin '{origin}' not in allowed origins list")
        
        return analysis
    
    @staticmethod
    def simulate_cors_check(origin: str, method: str = 'GET', 
                           headers: List[str] = None) -> Dict[str, Any]:
        """
        Simulate CORS check for debugging
        
        Args:
            origin: Request origin
            method: HTTP method
            headers: Request headers
            
        Returns:
            Simulation results
        """
        
        middleware = EnhancedCORSMiddleware(app=None)
        
        result = {
            'origin': origin,
            'method': method,
            'headers': headers or [],
            'allowed': False,
            'reasons': []
        }
        
        # Check origin
        if middleware._is_origin_allowed(origin):
            result['allowed'] = True
        else:
            result['reasons'].append(f"Origin '{origin}' not allowed")
        
        # Check method
        if method not in cors_config.allowed_methods and '*' not in cors_config.allowed_methods:
            result['allowed'] = False
            result['reasons'].append(f"Method '{method}' not allowed")
        
        # Check headers
        if headers:
            for header in headers:
                if (header not in cors_config.allowed_headers and 
                    '*' not in cors_config.allowed_headers):
                    result['allowed'] = False
                    result['reasons'].append(f"Header '{header}' not allowed")
        
        return result

# ============================================
# Export Components
# ============================================

__all__ = [
    # Main middleware
    "EnhancedCORSMiddleware",
    
    # Configuration
    "CORSConfig",
    "cors_config",
    
    # Utilities
    "create_cors_middleware",
    "get_production_cors_config",
    "get_development_cors_config",
    "validate_cors_config",
    
    # Debugging
    "CORSDebugger",
]
