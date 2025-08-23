# ============================================
# StockPredictionPro - src/api/middleware/auth.py
# Authentication middleware for FastAPI with JWT token validation and user context injection
# ============================================

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
import json
from urllib.parse import urlparse

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from jose import JWTError, jwt
import httpx

from ...utils.logger import get_logger
from ...utils.config_loader import load_config
from ..exceptions import UnauthorizedException, ForbiddenException

logger = get_logger('api.middleware.auth')

# ============================================
# Authentication Configuration
# ============================================

class AuthConfig:
    """Authentication configuration"""
    
    def __init__(self):
        self.config = load_config('app_config.yaml')
        auth_config = self.config.get('authentication', {})
        
        # JWT Settings
        self.jwt_secret_key = auth_config.get('jwt_secret_key', 'your-secret-key-change-in-production')
        self.jwt_algorithm = auth_config.get('jwt_algorithm', 'HS256')
        self.jwt_expiration_hours = auth_config.get('jwt_expiration_hours', 24)
        
        # Token Settings
        self.token_header_name = auth_config.get('token_header_name', 'Authorization')
        self.token_prefix = auth_config.get('token_prefix', 'Bearer')
        
        # Route Settings
        self.protected_routes = set(auth_config.get('protected_routes', ['/api/v1']))
        self.public_routes = set(auth_config.get('public_routes', [
            '/api/v1/health',
            '/api/v1/docs',
            '/api/v1/openapi.json',
            '/api/v1/auth/login',
            '/api/v1/auth/register'
        ]))
        
        # API Key Settings
        self.api_key_header = auth_config.get('api_key_header', 'X-API-Key')
        self.valid_api_keys = set(auth_config.get('valid_api_keys', []))
        
        # External Auth Settings (OAuth2, etc.)
        self.oauth2_config = auth_config.get('oauth2', {})
        self.enable_external_validation = auth_config.get('enable_external_validation', False)
        self.external_validation_url = auth_config.get('external_validation_url')

# Global auth config instance
auth_config = AuthConfig()

# ============================================
# Token Management
# ============================================

class TokenManager:
    """JWT token management utilities"""
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token
        
        Args:
            data: Token payload data
            expires_delta: Token expiration time
            
        Returns:
            Encoded JWT token
        """
        
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=auth_config.jwt_expiration_hours)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access_token"
        })
        
        encoded_jwt = jwt.encode(
            to_encode,
            auth_config.jwt_secret_key,
            algorithm=auth_config.jwt_algorithm
        )
        
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            JWTError: If token is invalid or expired
        """
        
        try:
            payload = jwt.decode(
                token,
                auth_config.jwt_secret_key,
                algorithms=[auth_config.jwt_algorithm]
            )
            
            # Verify token type
            if payload.get("type") != "access_token":
                raise JWTError("Invalid token type")
            
            return payload
            
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            raise
    
    @staticmethod
    def extract_token_from_header(authorization_header: str) -> Optional[str]:
        """
        Extract token from Authorization header
        
        Args:
            authorization_header: Authorization header value
            
        Returns:
            Extracted token or None
        """
        
        if not authorization_header:
            return None
        
        try:
            scheme, token = authorization_header.split(' ', 1)
            if scheme.lower() != auth_config.token_prefix.lower():
                return None
            return token
        except ValueError:
            return None

# ============================================
# User Context
# ============================================

class UserContext:
    """User context for authenticated requests"""
    
    def __init__(
        self,
        user_id: str,
        username: str,
        email: Optional[str] = None,
        roles: List[str] = None,
        permissions: List[str] = None,
        is_active: bool = True,
        is_admin: bool = False,
        is_premium: bool = False,
        metadata: Dict[str, Any] = None
    ):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.roles = roles or []
        self.permissions = permissions or []
        self.is_active = is_active
        self.is_admin = is_admin
        self.is_premium = is_premium
        self.metadata = metadata or {}
        self.authenticated_at = datetime.utcnow()
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions or self.is_admin
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role"""
        return role in self.roles or self.is_admin
    
    def has_any_role(self, roles: List[str]) -> bool:
        """Check if user has any of the specified roles"""
        return any(self.has_role(role) for role in roles)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
            "permissions": self.permissions,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "is_premium": self.is_premium,
            "metadata": self.metadata,
            "authenticated_at": self.authenticated_at.isoformat()
        }

# ============================================
# Authentication Middleware
# ============================================

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for FastAPI applications.
    
    Handles JWT token validation, API key authentication,
    and user context injection for protected routes.
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self.token_manager = TokenManager()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process authentication for incoming requests
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response object
        """
        
        # Get request path
        path = request.url.path
        method = request.method
        
        # Log request
        logger.debug(f"Processing {method} {path}")
        
        # Check if route requires authentication
        if not self._requires_auth(path):
            logger.debug(f"Public route: {path}")
            return await call_next(request)
        
        try:
            # Attempt authentication
            user_context = await self._authenticate_request(request)
            
            if user_context:
                # Inject user context into request
                request.state.user = user_context
                request.state.authenticated = True
                
                logger.debug(f"Authenticated user: {user_context.username}")
                
                # Add authentication headers to response
                response = await call_next(request)
                response.headers["X-Authenticated-User"] = user_context.username
                response.headers["X-User-ID"] = user_context.user_id
                
                return response
            else:
                # Authentication failed
                return await self._create_auth_error_response(
                    "Authentication failed",
                    status.HTTP_401_UNAUTHORIZED
                )
                
        except UnauthorizedException as e:
            return await self._create_auth_error_response(
                e.detail,
                e.status_code
            )
        except ForbiddenException as e:
            return await self._create_auth_error_response(
                e.detail,
                e.status_code
            )
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}", exc_info=True)
            return await self._create_auth_error_response(
                "Internal authentication error",
                status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _requires_auth(self, path: str) -> bool:
        """
        Check if path requires authentication
        
        Args:
            path: Request path
            
        Returns:
            True if authentication required
        """
        
        # Check public routes first
        for public_route in auth_config.public_routes:
            if path.startswith(public_route):
                return False
        
        # Check protected routes
        for protected_route in auth_config.protected_routes:
            if path.startswith(protected_route):
                return True
        
        # Default behavior based on configuration
        return len(auth_config.protected_routes) > 0
    
    async def _authenticate_request(self, request: Request) -> Optional[UserContext]:
        """
        Authenticate request using multiple methods
        
        Args:
            request: FastAPI request object
            
        Returns:
            UserContext if authenticated, None otherwise
        """
        
        # Try JWT token authentication first
        user_context = await self._authenticate_jwt_token(request)
        if user_context:
            return user_context
        
        # Try API key authentication
        user_context = await self._authenticate_api_key(request)
        if user_context:
            return user_context
        
        # Try external authentication if enabled
        if auth_config.enable_external_validation:
            user_context = await self._authenticate_external(request)
            if user_context:
                return user_context
        
        return None
    
    async def _authenticate_jwt_token(self, request: Request) -> Optional[UserContext]:
        """
        Authenticate using JWT token
        
        Args:
            request: FastAPI request object
            
        Returns:
            UserContext if valid token, None otherwise
        """
        
        # Get authorization header
        auth_header = request.headers.get(auth_config.token_header_name)
        if not auth_header:
            return None
        
        # Extract token
        token = self.token_manager.extract_token_from_header(auth_header)
        if not token:
            raise UnauthorizedException("Invalid authorization header format")
        
        try:
            # Verify token
            payload = self.token_manager.verify_token(token)
            
            # Extract user information
            user_id = payload.get("sub")  # Subject
            username = payload.get("username", payload.get("sub"))
            email = payload.get("email")
            roles = payload.get("roles", [])
            permissions = payload.get("permissions", [])
            is_admin = payload.get("is_admin", False)
            is_premium = payload.get("is_premium", False)
            metadata = payload.get("metadata", {})
            
            if not user_id:
                raise UnauthorizedException("Invalid token: missing user ID")
            
            # Create user context
            return UserContext(
                user_id=user_id,
                username=username,
                email=email,
                roles=roles,
                permissions=permissions,
                is_admin=is_admin,
                is_premium=is_premium,
                metadata=metadata
            )
            
        except JWTError:
            raise UnauthorizedException("Invalid or expired token")
    
    async def _authenticate_api_key(self, request: Request) -> Optional[UserContext]:
        """
        Authenticate using API key
        
        Args:
            request: FastAPI request object
            
        Returns:
            UserContext if valid API key, None otherwise
        """
        
        if not auth_config.valid_api_keys:
            return None
        
        # Get API key from header
        api_key = request.headers.get(auth_config.api_key_header)
        if not api_key:
            return None
        
        # Validate API key
        if api_key not in auth_config.valid_api_keys:
            raise UnauthorizedException("Invalid API key")
        
        # Create system user context for API key
        return UserContext(
            user_id=f"api_key:{api_key[:8]}",
            username=f"api_key_user",
            roles=["api_user"],
            permissions=["api_access"],
            is_active=True,
            metadata={"auth_method": "api_key"}
        )
    
    async def _authenticate_external(self, request: Request) -> Optional[UserContext]:
        """
        Authenticate using external service
        
        Args:
            request: FastAPI request object
            
        Returns:
            UserContext if external auth succeeds, None otherwise
        """
        
        if not auth_config.external_validation_url:
            return None
        
        # Get authorization header
        auth_header = request.headers.get(auth_config.token_header_name)
        if not auth_header:
            return None
        
        try:
            # Call external validation service
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    auth_config.external_validation_url,
                    headers={auth_config.token_header_name: auth_header},
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    user_data = response.json()
                    
                    return UserContext(
                        user_id=user_data.get("user_id"),
                        username=user_data.get("username"),
                        email=user_data.get("email"),
                        roles=user_data.get("roles", []),
                        permissions=user_data.get("permissions", []),
                        is_admin=user_data.get("is_admin", False),
                        is_premium=user_data.get("is_premium", False),
                        metadata={
                            "auth_method": "external",
                            "external_data": user_data
                        }
                    )
                else:
                    logger.warning(f"External auth failed: {response.status_code}")
                    return None
                    
        except httpx.RequestError as e:
            logger.error(f"External authentication error: {e}")
            return None
    
    async def _create_auth_error_response(self, message: str, status_code: int) -> JSONResponse:
        """
        Create authentication error response
        
        Args:
            message: Error message
            status_code: HTTP status code
            
        Returns:
            JSON error response
        """
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": "AUTHENTICATION_ERROR",
                    "message": message,
                    "timestamp": datetime.utcnow().isoformat()
                }
            },
            headers={
                "WWW-Authenticate": f"{auth_config.token_prefix} realm=\"API\""
            }
        )

# ============================================
# Permission Decorators
# ============================================

def require_permissions(*permissions: str):
    """
    Decorator to require specific permissions
    
    Args:
        permissions: Required permissions
        
    Returns:
        Decorator function
    """
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be used with dependency injection in FastAPI
            # The actual permission checking happens in the route handler
            return func(*args, **kwargs)
        
        # Add permission metadata for introspection
        wrapper._required_permissions = permissions
        return wrapper
    
    return decorator

def require_roles(*roles: str):
    """
    Decorator to require specific roles
    
    Args:
        roles: Required roles
        
    Returns:
        Decorator function
    """
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Add role metadata for introspection
        wrapper._required_roles = roles
        return wrapper
    
    return decorator

def require_admin():
    """Decorator to require admin privileges"""
    return require_roles("admin")

def require_premium():
    """Decorator to require premium subscription"""
    return require_roles("premium")

# ============================================
# Authentication Dependencies
# ============================================

def get_current_user_from_request(request: Request) -> UserContext:
    """
    Get current user from request state
    
    Args:
        request: FastAPI request object
        
    Returns:
        Current user context
        
    Raises:
        HTTPException: If user not authenticated
    """
    
    if not hasattr(request.state, 'user'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    return request.state.user

def check_user_permissions(user: UserContext, required_permissions: List[str]) -> bool:
    """
    Check if user has all required permissions
    
    Args:
        user: User context
        required_permissions: List of required permissions
        
    Returns:
        True if user has all permissions
    """
    
    if user.is_admin:
        return True
    
    return all(user.has_permission(permission) for permission in required_permissions)

def check_user_roles(user: UserContext, required_roles: List[str]) -> bool:
    """
    Check if user has any of the required roles
    
    Args:
        user: User context
        required_roles: List of required roles
        
    Returns:
        True if user has any required role
    """
    
    if user.is_admin:
        return True
    
    return user.has_any_role(required_roles)

# ============================================
# Authentication Utilities
# ============================================

class AuthUtils:
    """Authentication utility functions"""
    
    @staticmethod
    def create_user_token(
        user_id: str,
        username: str,
        email: Optional[str] = None,
        roles: List[str] = None,
        permissions: List[str] = None,
        is_admin: bool = False,
        is_premium: bool = False,
        metadata: Dict[str, Any] = None,
        expires_in_hours: Optional[int] = None
    ) -> str:
        """
        Create JWT token for user
        
        Args:
            user_id: User ID
            username: Username
            email: User email
            roles: User roles
            permissions: User permissions
            is_admin: Admin flag
            is_premium: Premium flag
            metadata: Additional metadata
            expires_in_hours: Custom expiration time
            
        Returns:
            JWT token
        """
        
        token_data = {
            "sub": user_id,
            "username": username,
            "email": email,
            "roles": roles or [],
            "permissions": permissions or [],
            "is_admin": is_admin,
            "is_premium": is_premium,
            "metadata": metadata or {}
        }
        
        expires_delta = None
        if expires_in_hours:
            expires_delta = timedelta(hours=expires_in_hours)
        
        return TokenManager.create_access_token(token_data, expires_delta)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password (placeholder - use bcrypt in production)"""
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password (placeholder - use bcrypt in production)"""
        return AuthUtils.hash_password(password) == hashed_password

# ============================================
# Export Components
# ============================================

__all__ = [
    # Main middleware
    "AuthenticationMiddleware",
    
    # Configuration
    "AuthConfig",
    "auth_config",
    
    # Token management
    "TokenManager",
    
    # User context
    "UserContext",
    
    # Decorators
    "require_permissions",
    "require_roles",
    "require_admin",
    "require_premium",
    
    # Dependencies
    "get_current_user_from_request",
    "check_user_permissions",
    "check_user_roles",
    
    # Utilities
    "AuthUtils",
]
