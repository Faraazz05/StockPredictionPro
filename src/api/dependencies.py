# ============================================
# StockPredictionPro - src/api/dependencies.py
# Comprehensive dependency injection for FastAPI application with database, caching, and service management
# ============================================

import os
import logging
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator, Optional, Dict, Any
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
import aioredis
import redis
from jose import JWTError, jwt

# CORRECTED IMPORTS - Based on actual project structure
from ..utils.config_loader import load_config  # utils/config_loader.py
from ..utils.logger import get_logger  # utils/logger.py
from ..utils.exceptions import ValidationError, ConfigurationError  # utils/exceptions.py
from ..data.manager import DataManager  # data/manager.py
from ..data.cache import CacheManager  # data/cache.py
from ..models.factory import ModelFactory  # models/factory.py
from ..models.persistence import ModelPersistence  # models/persistence.py
from ..trading.portfolio import PortfolioManager  # trading/portfolio.py
from ..trading.strategies.momentum import MomentumManager  # trading/strategies/momentum.py
from ..trading.strategies.mean_reversion import MeanReversionManager  # trading/strategies/mean_reversion.py
from ..trading.strategies.pairs_trading import PairsTradingManager  # trading/strategies/pairs_trading.py
from ..trading.strategies.trend_following import TrendFollowingManager  # trading/strategies/trend_following.py
from ..features.pipeline import FeaturePipeline  # features/pipeline.py
from ..evaluation.backtesting.engine import BacktestEngine  # evaluation/backtesting/engine.py

logger = get_logger('api.dependencies')

# ============================================
# Configuration Dependencies
# ============================================

@lru_cache()
def get_app_settings() -> Dict[str, Any]:
    """
    Get application settings (cached).
    
    Returns:
        Settings dictionary with all configuration
    """
    return load_config('app_config.yaml')

@lru_cache()
def get_model_config() -> Dict[str, Any]:
    """Get model configuration (cached)"""
    return load_config('model_config.yaml')

@lru_cache()
def get_trading_config() -> Dict[str, Any]:
    """Get trading configuration (cached)"""
    return load_config('trading_config.yaml')

def get_logger_instance() -> logging.Logger:
    """Get configured logger instance"""
    return logger

# ============================================
# Database Dependencies
# ============================================

class DatabaseDependency:
    """Database connection dependency manager"""
    
    def __init__(self):
        self.settings = get_app_settings()
        self._engine = None
        self._async_engine = None
        self._session_factory = None
        self._async_session_factory = None
    
    @property
    def engine(self):
        """Get or create sync SQLAlchemy engine"""
        if self._engine is None:
            database_url = self.settings.get('database', {}).get('url', 'sqlite:///./stockpred.db')
            self._engine = create_engine(
                database_url,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=self.settings.get('debug', False)
            )
        return self._engine
    
    @property
    def async_engine(self):
        """Get or create async SQLAlchemy engine"""
        if self._async_engine is None:
            database_url = self.settings.get('database', {}).get('url', 'sqlite:///./stockpred.db')
            # Convert to async URL if needed
            if database_url.startswith('sqlite:'):
                async_url = database_url.replace('sqlite:', 'sqlite+aiosqlite:')
            else:
                async_url = database_url
            
            self._async_engine = create_async_engine(
                async_url,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=self.settings.get('debug', False)
            )
        return self._async_engine
    
    @property
    def session_factory(self):
        """Get sync session factory"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
        return self._session_factory
    
    @property
    def async_session_factory(self):
        """Get async session factory"""
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._async_session_factory

# Global database dependency instance
_db_dependency = DatabaseDependency()

def get_sync_session() -> Generator[Session, None, None]:
    """
    Dependency for sync database session
    
    Yields:
        Database session
    """
    session = _db_dependency.session_factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for async database session
    
    Yields:
        Async database session
    """
    async with _db_dependency.async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session error: {e}")
            raise

# ============================================
# Redis/Caching Dependencies
# ============================================

class CacheDependency:
    """Redis cache dependency manager"""
    
    def __init__(self):
        self.settings = get_app_settings()
        self._redis_pool = None
        self._async_redis = None
        self._cache_manager = None
    
    @property
    def redis_url(self):
        """Get Redis URL from config"""
        return self.settings.get('cache', {}).get('redis_url', 'redis://localhost:6379')
    
    @property
    def redis_pool(self):
        """Get sync Redis connection pool"""
        if self._redis_pool is None:
            self._redis_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=20
            )
        return self._redis_pool
    
    @property
    async def async_redis(self):
        """Get async Redis connection"""
        if self._async_redis is None:
            self._async_redis = await aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=20
            )
        return self._async_redis
    
    @property
    def cache_manager(self):
        """Get cache manager instance"""
        if self._cache_manager is None:
            self._cache_manager = CacheManager()
        return self._cache_manager

# Global cache dependency instance
_cache_dependency = CacheDependency()

def get_redis_sync() -> Generator[redis.Redis, None, None]:
    """
    Dependency for sync Redis connection
    
    Yields:
        Redis client
    """
    client = redis.Redis(connection_pool=_cache_dependency.redis_pool)
    try:
        yield client
    finally:
        client.close()

async def get_redis_async() -> AsyncGenerator[aioredis.Redis, None]:
    """
    Dependency for async Redis connection
    
    Yields:
        Async Redis client
    """
    redis_client = await _cache_dependency.async_redis
    try:
        yield redis_client
    finally:
        # Connection handled by pool, no explicit close needed
        pass

def get_cache_manager() -> CacheManager:
    """Get cache manager dependency"""
    return _cache_dependency.cache_manager

# ============================================
# Authentication Dependencies
# ============================================

class JWTBearer(HTTPBearer):
    """JWT Bearer token authentication"""
    
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)
        self.settings = get_app_settings()
    
    async def __call__(self, credentials: HTTPAuthorizationCredentials = None):
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme"
                )
            
            if not self.verify_jwt(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid token or expired token"
                )
            
            return credentials.credentials
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authorization code"
            )
    
    def verify_jwt(self, token: str) -> bool:
        """Verify JWT token"""
        try:
            auth_config = self.settings.get('authentication', {})
            secret_key = auth_config.get('jwt_secret_key', 'default-secret-key')
            algorithm = auth_config.get('jwt_algorithm', 'HS256')
            
            payload = jwt.decode(token, secret_key, algorithms=[algorithm])
            return payload is not None
        except JWTError:
            return False

# JWT Bearer instance
jwt_bearer = JWTBearer()

def get_current_user_token(token: str = Depends(jwt_bearer)) -> str:
    """Get current user token from JWT"""
    return token

async def get_current_user(
    token: str = Depends(get_current_user_token),
    db: AsyncSession = Depends(get_async_session)
) -> Dict[str, Any]:
    """
    Get current authenticated user
    
    Args:
        token: JWT token
        db: Database session
        
    Returns:
        Current user data
        
    Raises:
        HTTPException: If user not found or token invalid
    """
    
    try:
        # Decode JWT token
        settings = get_app_settings()
        auth_config = settings.get('authentication', {})
        secret_key = auth_config.get('jwt_secret_key', 'default-secret-key')
        algorithm = auth_config.get('jwt_algorithm', 'HS256')
        
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        
        # Return user data (simplified for now)
        return {
            "username": username,
            "user_id": payload.get("user_id", 1),
            "is_active": True,
            "is_admin": payload.get("is_admin", False),
            "is_premium": payload.get("is_premium", False)
        }
    
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get current active user
    
    Args:
        current_user: Current user from JWT
        
    Returns:
        Active user
        
    Raises:
        HTTPException: If user is inactive
    """
    
    if not current_user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return current_user

def get_optional_current_user(
    token: Optional[str] = Depends(HTTPBearer(auto_error=False))
) -> Optional[str]:
    """Get current user token (optional, for public endpoints)"""
    return token.credentials if token else None

# ============================================
# Service Layer Dependencies
# ============================================

class ServiceDependency:
    """Service layer dependency manager"""
    
    def __init__(self):
        self.settings = get_app_settings()
        self._data_manager = None
        self._cache_manager = None
        self._model_factory = None
        self._model_persistence = None
        self._portfolio_manager = None
        self._momentum_manager = None
        self._mean_reversion_manager = None
        self._pairs_trading_manager = None
        self._trend_following_manager = None
        self._feature_pipeline = None
        self._backtest_engine = None
    
    @property
    def data_manager(self):
        """Get data manager instance"""
        if self._data_manager is None:
            self._data_manager = DataManager()
        return self._data_manager
    
    @property
    def cache_manager(self):
        """Get cache manager instance"""
        if self._cache_manager is None:
            self._cache_manager = CacheManager()
        return self._cache_manager
    
    @property
    def model_factory(self):
        """Get model factory instance"""
        if self._model_factory is None:
            self._model_factory = ModelFactory()
        return self._model_factory
    
    @property
    def model_persistence(self):
        """Get model persistence instance"""
        if self._model_persistence is None:
            self._model_persistence = ModelPersistence()
        return self._model_persistence
    
    @property
    def portfolio_manager(self):
        """Get portfolio manager instance"""
        if self._portfolio_manager is None:
            self._portfolio_manager = PortfolioManager("api_portfolio", 1000000.0)
        return self._portfolio_manager
    
    @property
    def momentum_manager(self):
        """Get momentum strategy manager"""
        if self._momentum_manager is None:
            self._momentum_manager = MomentumManager()
        return self._momentum_manager
    
    @property
    def mean_reversion_manager(self):
        """Get mean reversion strategy manager"""
        if self._mean_reversion_manager is None:
            self._mean_reversion_manager = MeanReversionManager()
        return self._mean_reversion_manager
    
    @property
    def pairs_trading_manager(self):
        """Get pairs trading strategy manager"""
        if self._pairs_trading_manager is None:
            self._pairs_trading_manager = PairsTradingManager()
        return self._pairs_trading_manager
    
    @property
    def trend_following_manager(self):
        """Get trend following strategy manager"""
        if self._trend_following_manager is None:
            self._trend_following_manager = TrendFollowingManager()
        return self._trend_following_manager
    
    @property
    def feature_pipeline(self):
        """Get feature pipeline instance"""
        if self._feature_pipeline is None:
            self._feature_pipeline = FeaturePipeline()
        return self._feature_pipeline
    
    @property
    def backtest_engine(self):
        """Get backtest engine instance"""
        if self._backtest_engine is None:
            self._backtest_engine = BacktestEngine()
        return self._backtest_engine

# Global service dependency instance
_service_dependency = ServiceDependency()

def get_data_manager() -> DataManager:
    """Get data manager dependency"""
    return _service_dependency.data_manager

def get_cache_manager_service() -> CacheManager:
    """Get cache manager service dependency"""
    return _service_dependency.cache_manager

def get_model_factory() -> ModelFactory:
    """Get model factory dependency"""
    return _service_dependency.model_factory

def get_model_persistence() -> ModelPersistence:
    """Get model persistence dependency"""
    return _service_dependency.model_persistence

def get_portfolio_manager() -> PortfolioManager:
    """Get portfolio manager dependency"""
    return _service_dependency.portfolio_manager

def get_momentum_manager() -> MomentumManager:
    """Get momentum strategy manager dependency"""
    return _service_dependency.momentum_manager

def get_mean_reversion_manager() -> MeanReversionManager:
    """Get mean reversion strategy manager dependency"""
    return _service_dependency.mean_reversion_manager

def get_pairs_trading_manager() -> PairsTradingManager:
    """Get pairs trading strategy manager dependency"""
    return _service_dependency.pairs_trading_manager

def get_trend_following_manager() -> TrendFollowingManager:
    """Get trend following strategy manager dependency"""
    return _service_dependency.trend_following_manager

def get_feature_pipeline() -> FeaturePipeline:
    """Get feature pipeline dependency"""
    return _service_dependency.feature_pipeline

def get_backtest_engine() -> BacktestEngine:
    """Get backtest engine dependency"""
    return _service_dependency.backtest_engine

# ============================================
# Permission Dependencies
# ============================================

def require_admin_user(current_user: Dict[str, Any] = Depends(get_current_active_user)) -> Dict[str, Any]:
    """
    Require admin user permissions
    
    Args:
        current_user: Current active user
        
    Returns:
        Admin user
        
    Raises:
        HTTPException: If user is not admin
    """
    
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required"
        )
    
    return current_user

def require_premium_user(current_user: Dict[str, Any] = Depends(get_current_active_user)) -> Dict[str, Any]:
    """
    Require premium user subscription
    
    Args:
        current_user: Current active user
        
    Returns:
        Premium user
        
    Raises:
        HTTPException: If user doesn't have premium subscription
    """
    
    if not current_user.get("is_premium", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    
    return current_user

# ============================================
# Rate Limiting Dependencies
# ============================================

class RateLimitDependency:
    """Rate limiting dependency"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    async def __call__(
        self,
        current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
        redis_client: aioredis.Redis = Depends(get_redis_async)
    ):
        """Check rate limits"""
        
        # Get user identifier
        if current_user:
            user_id = f"user:{current_user.get('user_id', 'unknown')}"
            # Premium users get higher limits
            max_requests = self.max_requests * 5 if current_user.get('is_premium') else self.max_requests
        else:
            # Use IP-based limiting for anonymous users
            user_id = "anonymous"  # Would normally get from request.client.host
            max_requests = self.max_requests // 10  # Lower limit for anonymous
        
        # Check current request count
        key = f"rate_limit:{user_id}:{datetime.now().hour}"
        current_requests = await redis_client.get(key)
        
        if current_requests and int(current_requests) >= max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Increment counter
        await redis_client.incr(key)
        await redis_client.expire(key, self.window_seconds)

# Rate limiting instances
standard_rate_limit = RateLimitDependency(max_requests=100, window_seconds=3600)
strict_rate_limit = RateLimitDependency(max_requests=10, window_seconds=3600)

# ============================================
# Validation Dependencies
# ============================================

def validate_symbol(symbol: str) -> str:
    """
    Validate trading symbol format
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Validated symbol
        
    Raises:
        HTTPException: If symbol is invalid
    """
    
    symbol = symbol.upper().strip()
    
    if not symbol:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Symbol cannot be empty"
        )
    
    if not symbol.replace('.', '').isalnum():  # Allow dots for NSE symbols
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Symbol must be alphanumeric (dots allowed)"
        )
    
    if len(symbol) > 12:  # Allow longer symbols for NSE
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Symbol too long"
        )
    
    return symbol

def validate_date_range(start_date: datetime, end_date: datetime) -> tuple[datetime, datetime]:
    """
    Validate date range
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Validated date range
        
    Raises:
        HTTPException: If date range is invalid
    """
    
    if start_date >= end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Start date must be before end date"
        )
    
    # Limit to reasonable date ranges
    max_days = 365 * 5  # 5 years
    if (end_date - start_date).days > max_days:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Date range cannot exceed {max_days} days"
        )
    
    # Don't allow future dates
    now = datetime.now()
    if start_date > now or end_date > now:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dates cannot be in the future"
        )
    
    return start_date, end_date

# ============================================
# Application Lifecycle Dependencies
# ============================================

@asynccontextmanager
async def lifespan_manager():
    """Application lifespan manager"""
    
    logger.info("Starting StockPredictionPro API")
    
    # Startup tasks
    try:
        # Initialize cache
        await _cache_dependency.async_redis
        
        # Initialize services
        _service_dependency.data_manager.initialize() if hasattr(_service_dependency.data_manager, 'initialize') else None
        
        logger.info("API startup completed successfully")
        
        yield
        
    finally:
        # Shutdown tasks
        logger.info("Shutting down StockPredictionPro API")
        
        # Close database connections
        if _db_dependency._async_engine:
            await _db_dependency._async_engine.dispose()
        
        # Close Redis connections
        if _cache_dependency._async_redis:
            await _cache_dependency._async_redis.close()
        
        logger.info("API shutdown completed")

# ============================================
# Utility Dependencies
# ============================================

def get_pagination_params(
    skip: int = 0,
    limit: int = 100,
    max_limit: int = 1000
) -> tuple[int, int]:
    """
    Get validated pagination parameters
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        max_limit: Maximum allowed limit
        
    Returns:
        Validated skip and limit
        
    Raises:
        HTTPException: If parameters are invalid
    """
    
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip must be non-negative"
        )
    
    if limit < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be positive"
        )
    
    if limit > max_limit:
        limit = max_limit
    
    return skip, limit

def get_request_metadata() -> Dict[str, Any]:
    """Get request metadata (IP, user agent, etc.)"""
    # Would normally extract from FastAPI Request object
    return {
        "timestamp": datetime.now(),
        "ip_address": "127.0.0.1",  # Placeholder
        "user_agent": "API Client"   # Placeholder
    }

# ============================================
# Dependency Collections
# ============================================

class CommonDependencies:
    """Collection of commonly used dependencies"""
    
    # Database
    db = Depends(get_async_session)
    sync_db = Depends(get_sync_session)
    
    # Cache
    cache = Depends(get_redis_async)
    sync_cache = Depends(get_redis_sync)
    
    # Authentication
    current_user = Depends(get_current_active_user)
    optional_user = Depends(get_optional_current_user)
    admin_user = Depends(require_admin_user)
    premium_user = Depends(require_premium_user)
    
    # Core Services
    data_manager = Depends(get_data_manager)
    cache_manager = Depends(get_cache_manager_service)
    model_factory = Depends(get_model_factory)
    model_persistence = Depends(get_model_persistence)
    feature_pipeline = Depends(get_feature_pipeline)
    backtest_engine = Depends(get_backtest_engine)
    
    # Trading Services
    portfolio_manager = Depends(get_portfolio_manager)
    momentum_manager = Depends(get_momentum_manager)
    mean_reversion_manager = Depends(get_mean_reversion_manager)
    pairs_trading_manager = Depends(get_pairs_trading_manager)
    trend_following_manager = Depends(get_trend_following_manager)
    
    # Rate limiting
    rate_limit = Depends(standard_rate_limit)
    strict_rate_limit = Depends(strict_rate_limit)
    
    # Configuration
    settings = Depends(get_app_settings)
    model_config = Depends(get_model_config)
    trading_config = Depends(get_trading_config)
    logger = Depends(get_logger_instance)

# ============================================
# Export Dependencies
# ============================================

__all__ = [
    # Configuration
    "get_app_settings",
    "get_model_config", 
    "get_trading_config",
    "get_logger_instance",
    
    # Database
    "get_sync_session",
    "get_async_session",
    
    # Cache
    "get_redis_sync",
    "get_redis_async",
    "get_cache_manager",
    
    # Authentication
    "JWTBearer",
    "jwt_bearer",
    "get_current_user_token",
    "get_current_user",
    "get_current_active_user",
    "get_optional_current_user",
    
    # Permissions
    "require_admin_user",
    "require_premium_user",
    
    # Core Services
    "get_data_manager",
    "get_cache_manager_service",
    "get_model_factory",
    "get_model_persistence",
    "get_feature_pipeline",
    "get_backtest_engine",
    
    # Trading Services
    "get_portfolio_manager",
    "get_momentum_manager",
    "get_mean_reversion_manager",
    "get_pairs_trading_manager",
    "get_trend_following_manager",
    
    # Rate limiting
    "RateLimitDependency",
    "standard_rate_limit",
    "strict_rate_limit",
    
    # Validation
    "validate_symbol",
    "validate_date_range",
    "get_pagination_params",
    
    # Utilities
    "get_request_metadata",
    "lifespan_manager",
    
    # Collections
    "CommonDependencies",
]
