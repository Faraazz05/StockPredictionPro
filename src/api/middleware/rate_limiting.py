# ============================================
# StockPredictionPro - src/api/middleware/rate_limiting.py
# Advanced rate limiting middleware for FastAPI with multiple algorithms, user-based limits, and monitoring
# ============================================

import asyncio
import hashlib
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import math

from fastapi import Request, Response, status, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import aioredis
import redis

from ...utils.logger import get_logger
from ...utils.config_loader import load_config
from ..exceptions import RateLimitException

logger = get_logger('api.middleware.rate_limiting')

# ============================================
# Rate Limiting Algorithms and Configuration
# ============================================

class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"

class UserTier(Enum):
    """User tiers for rate limiting"""
    ANONYMOUS = "anonymous"
    BASIC = "basic"
    PREMIUM = "premium"
    ADMIN = "admin"

@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests: int  # Number of requests allowed
    window: int    # Time window in seconds
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    burst_multiplier: float = 1.5  # Allow burst requests

@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    remaining: int
    reset_time: int  # Unix timestamp
    retry_after: Optional[int] = None
    limit: int = 0
    current_requests: int = 0

class RateLimitConfig:
    """Rate limiting configuration"""
    
    def __init__(self):
        self.config = load_config('app_config.yaml')
        rate_limit_config = self.config.get('rate_limiting', {})
        
        # Global settings
        self.enabled = rate_limit_config.get('enabled', True)
        self.default_algorithm = RateLimitAlgorithm(
            rate_limit_config.get('default_algorithm', 'sliding_window')
        )
        
        # Storage backend
        self.storage_backend = rate_limit_config.get('storage_backend', 'memory')  # memory, redis
        self.redis_url = rate_limit_config.get('redis_url', 'redis://localhost:6379')
        
        # Default limits by user tier
        default_limits = rate_limit_config.get('default_limits', {})
        self.limits_by_tier = {
            UserTier.ANONYMOUS: RateLimit(
                requests=default_limits.get('anonymous', {}).get('requests', 100),
                window=default_limits.get('anonymous', {}).get('window', 3600),
                algorithm=self.default_algorithm
            ),
            UserTier.BASIC: RateLimit(
                requests=default_limits.get('basic', {}).get('requests', 1000),
                window=default_limits.get('basic', {}).get('window', 3600),
                algorithm=self.default_algorithm
            ),
            UserTier.PREMIUM: RateLimit(
                requests=default_limits.get('premium', {}).get('requests', 5000),
                window=default_limits.get('premium', {}).get('window', 3600),
                algorithm=self.default_algorithm
            ),
            UserTier.ADMIN: RateLimit(
                requests=default_limits.get('admin', {}).get('requests', 50000),
                window=default_limits.get('admin', {}).get('window', 3600),
                algorithm=self.default_algorithm
            )
        }
        
        # Path-specific limits
        path_limits = rate_limit_config.get('path_limits', {})
        self.path_limits = {}
        for path, limit_config in path_limits.items():
            self.path_limits[path] = RateLimit(
                requests=limit_config['requests'],
                window=limit_config['window'],
                algorithm=RateLimitAlgorithm(limit_config.get('algorithm', 'sliding_window'))
            )
        
        # Exempted paths and IPs
        self.exempt_paths = set(rate_limit_config.get('exempt_paths', [
            '/health', '/metrics', '/docs', '/openapi.json'
        ]))
        self.exempt_ips = set(rate_limit_config.get('exempt_ips', ['127.0.0.1']))
        
        # Headers
        self.rate_limit_headers = rate_limit_config.get('rate_limit_headers', True)
        self.custom_headers = rate_limit_config.get('custom_headers', {})
        
        # Monitoring
        self.log_violations = rate_limit_config.get('log_violations', True)
        self.log_all_requests = rate_limit_config.get('log_all_requests', False)
        
        # Advanced settings
        self.enable_burst_protection = rate_limit_config.get('enable_burst_protection', True)
        self.sliding_window_precision = rate_limit_config.get('sliding_window_precision', 60)  # seconds
        self.cleanup_interval = rate_limit_config.get('cleanup_interval', 300)  # seconds

# Global rate limiting config
rate_limit_config = RateLimitConfig()

# ============================================
# Rate Limiting Storage Backends
# ============================================

class RateLimitStorage:
    """Base class for rate limit storage backends"""
    
    async def get_request_count(self, key: str, window: int) -> int:
        """Get current request count for key within window"""
        raise NotImplementedError
    
    async def increment_request_count(self, key: str, window: int) -> int:
        """Increment request count and return new count"""
        raise NotImplementedError
    
    async def get_token_bucket(self, key: str) -> Tuple[int, float]:
        """Get token bucket state (tokens, last_refill)"""
        raise NotImplementedError
    
    async def update_token_bucket(self, key: str, tokens: int, last_refill: float):
        """Update token bucket state"""
        raise NotImplementedError
    
    async def cleanup_expired_keys(self):
        """Cleanup expired keys"""
        raise NotImplementedError

class MemoryRateLimitStorage(RateLimitStorage):
    """In-memory rate limit storage"""
    
    def __init__(self):
        self.request_counts = defaultdict(deque)  # key -> deque of timestamps
        self.token_buckets = {}  # key -> (tokens, last_refill)
        self.fixed_windows = defaultdict(dict)  # key -> {window_start: count}
        self.last_cleanup = time.time()
    
    async def get_request_count(self, key: str, window: int) -> int:
        """Get current request count for sliding window"""
        current_time = time.time()
        cutoff_time = current_time - window
        
        # Clean old requests
        request_times = self.request_counts[key]
        while request_times and request_times[0] < cutoff_time:
            request_times.popleft()
        
        return len(request_times)
    
    async def increment_request_count(self, key: str, window: int) -> int:
        """Increment request count and return new count"""
        current_time = time.time()
        cutoff_time = current_time - window
        
        # Clean old requests
        request_times = self.request_counts[key]
        while request_times and request_times[0] < cutoff_time:
            request_times.popleft()
        
        # Add new request
        request_times.append(current_time)
        
        return len(request_times)
    
    async def get_token_bucket(self, key: str) -> Tuple[int, float]:
        """Get token bucket state"""
        if key not in self.token_buckets:
            # Initialize with full bucket
            return (1000, time.time())  # Default capacity
        
        return self.token_buckets[key]
    
    async def update_token_bucket(self, key: str, tokens: int, last_refill: float):
        """Update token bucket state"""
        self.token_buckets[key] = (tokens, last_refill)
    
    async def get_fixed_window_count(self, key: str, window_start: int) -> int:
        """Get request count for fixed window"""
        return self.fixed_windows[key].get(window_start, 0)
    
    async def increment_fixed_window_count(self, key: str, window_start: int) -> int:
        """Increment fixed window count"""
        count = self.fixed_windows[key].get(window_start, 0) + 1
        self.fixed_windows[key][window_start] = count
        return count
    
    async def cleanup_expired_keys(self):
        """Cleanup expired keys periodically"""
        current_time = time.time()
        
        # Only cleanup periodically
        if current_time - self.last_cleanup < rate_limit_config.cleanup_interval:
            return
        
        # Cleanup sliding window data older than max window
        max_window = 86400  # 24 hours
        cutoff_time = current_time - max_window
        
        keys_to_remove = []
        for key, request_times in self.request_counts.items():
            while request_times and request_times[0] < cutoff_time:
                request_times.popleft()
            
            if not request_times:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.request_counts[key]
        
        # Cleanup fixed window data
        for key, windows in self.fixed_windows.items():
            expired_windows = [
                window_start for window_start in windows.keys()
                if window_start < cutoff_time
            ]
            for window_start in expired_windows:
                del windows[window_start]
        
        # Cleanup old token buckets
        expired_buckets = [
            key for key, (tokens, last_refill) in self.token_buckets.items()
            if current_time - last_refill > max_window
        ]
        for key in expired_buckets:
            del self.token_buckets[key]
        
        self.last_cleanup = current_time
        logger.debug(f"Cleaned up {len(keys_to_remove)} expired rate limit keys")

class RedisRateLimitStorage(RateLimitStorage):
    """Redis-based rate limit storage"""
    
    def __init__(self):
        self.redis_url = rate_limit_config.redis_url
        self.redis_client = None
        self.connection_pool = None
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get Redis connection"""
        if self.redis_client is None:
            try:
                self.redis_client = await aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    max_connections=20
                )
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                # Fallback to memory storage
                return None
        
        return self.redis_client
    
    async def get_request_count(self, key: str, window: int) -> int:
        """Get current request count for sliding window using Redis"""
        redis = await self._get_redis()
        if not redis:
            return 0
        
        try:
            # Use Redis sorted sets for sliding window
            current_time = time.time()
            cutoff_time = current_time - window
            
            # Remove expired entries
            await redis.zremrangebyscore(f"rate_limit:{key}", 0, cutoff_time)
            
            # Get current count
            count = await redis.zcard(f"rate_limit:{key}")
            return count
            
        except Exception as e:
            logger.error(f"Redis error in get_request_count: {e}")
            return 0
    
    async def increment_request_count(self, key: str, window: int) -> int:
        """Increment request count using Redis"""
        redis = await self._get_redis()
        if not redis:
            return 1
        
        try:
            current_time = time.time()
            cutoff_time = current_time - window
            
            pipeline = redis.pipeline()
            
            # Remove expired entries
            pipeline.zremrangebyscore(f"rate_limit:{key}", 0, cutoff_time)
            
            # Add current request
            pipeline.zadd(f"rate_limit:{key}", {str(current_time): current_time})
            
            # Get count
            pipeline.zcard(f"rate_limit:{key}")
            
            # Set expiration
            pipeline.expire(f"rate_limit:{key}", window + 60)  # Add buffer
            
            results = await pipeline.execute()
            return results[2]  # Count result
            
        except Exception as e:
            logger.error(f"Redis error in increment_request_count: {e}")
            return 1
    
    async def get_token_bucket(self, key: str) -> Tuple[int, float]:
        """Get token bucket state from Redis"""
        redis = await self._get_redis()
        if not redis:
            return (1000, time.time())
        
        try:
            bucket_data = await redis.hgetall(f"token_bucket:{key}")
            
            if bucket_data:
                tokens = int(bucket_data.get('tokens', 1000))
                last_refill = float(bucket_data.get('last_refill', time.time()))
                return (tokens, last_refill)
            else:
                # Initialize new bucket
                tokens, last_refill = 1000, time.time()
                await self.update_token_bucket(key, tokens, last_refill)
                return (tokens, last_refill)
                
        except Exception as e:
            logger.error(f"Redis error in get_token_bucket: {e}")
            return (1000, time.time())
    
    async def update_token_bucket(self, key: str, tokens: int, last_refill: float):
        """Update token bucket state in Redis"""
        redis = await self._get_redis()
        if not redis:
            return
        
        try:
            await redis.hset(f"token_bucket:{key}", mapping={
                'tokens': tokens,
                'last_refill': last_refill
            })
            
            # Set expiration
            await redis.expire(f"token_bucket:{key}", 86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Redis error in update_token_bucket: {e}")
    
    async def cleanup_expired_keys(self):
        """Redis automatically handles expiration"""
        pass

# ============================================
# Rate Limiting Algorithms Implementation
# ============================================

class RateLimitingAlgorithms:
    """Implementation of various rate limiting algorithms"""
    
    @staticmethod
    async def sliding_window(storage: RateLimitStorage, key: str, 
                           limit: RateLimit) -> RateLimitResult:
        """Sliding window rate limiting"""
        
        current_count = await storage.increment_request_count(key, limit.window)
        
        allowed = current_count <= limit.requests
        remaining = max(0, limit.requests - current_count)
        reset_time = int(time.time() + limit.window)
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=limit.window if not allowed else None,
            limit=limit.requests,
            current_requests=current_count
        )
    
    @staticmethod
    async def fixed_window(storage: RateLimitStorage, key: str, 
                          limit: RateLimit) -> RateLimitResult:
        """Fixed window rate limiting"""
        
        current_time = time.time()
        window_start = int(current_time // limit.window) * limit.window
        
        if hasattr(storage, 'increment_fixed_window_count'):
            current_count = await storage.increment_fixed_window_count(key, window_start)
        else:
            # Fallback for Redis storage
            current_count = await storage.increment_request_count(key, limit.window)
        
        allowed = current_count <= limit.requests
        remaining = max(0, limit.requests - current_count)
        reset_time = int(window_start + limit.window)
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=reset_time - int(current_time) if not allowed else None,
            limit=limit.requests,
            current_requests=current_count
        )
    
    @staticmethod
    async def token_bucket(storage: RateLimitStorage, key: str, 
                          limit: RateLimit) -> RateLimitResult:
        """Token bucket rate limiting"""
        
        current_time = time.time()
        tokens, last_refill = await storage.get_token_bucket(key)
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = current_time - last_refill
        tokens_to_add = int(time_elapsed * (limit.requests / limit.window))
        
        # Refill tokens up to capacity
        tokens = min(limit.requests, tokens + tokens_to_add)
        
        # Check if request can be processed
        if tokens > 0:
            tokens -= 1
            allowed = True
        else:
            allowed = False
        
        # Update bucket
        await storage.update_token_bucket(key, tokens, current_time)
        
        # Calculate when bucket will have tokens again
        if not allowed:
            time_for_token = limit.window / limit.requests
            retry_after = int(time_for_token)
        else:
            retry_after = None
        
        return RateLimitResult(
            allowed=allowed,
            remaining=tokens,
            reset_time=int(current_time + limit.window),
            retry_after=retry_after,
            limit=limit.requests,
            current_requests=limit.requests - tokens
        )
    
    @staticmethod
    async def leaky_bucket(storage: RateLimitStorage, key: str, 
                          limit: RateLimit) -> RateLimitResult:
        """Leaky bucket rate limiting (similar to token bucket but with constant leak)"""
        
        current_time = time.time()
        tokens, last_refill = await storage.get_token_bucket(key)
        
        # Calculate leak rate
        leak_rate = limit.requests / limit.window
        time_elapsed = current_time - last_refill
        
        # Leak tokens
        tokens_leaked = time_elapsed * leak_rate
        tokens = max(0, tokens - tokens_leaked)
        
        # Add new request
        if tokens < limit.requests:
            tokens += 1
            allowed = True
        else:
            allowed = False
        
        # Update bucket
        await storage.update_token_bucket(key, tokens, current_time)
        
        remaining = max(0, limit.requests - int(tokens))
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=int(current_time + limit.window),
            retry_after=int(1.0 / leak_rate) if not allowed else None,
            limit=limit.requests,
            current_requests=int(tokens)
        )

# ============================================
# Enhanced Rate Limiting Middleware
# ============================================

class EnhancedRateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Enhanced rate limiting middleware with multiple algorithms,
    user-based limits, path-specific limits, and comprehensive monitoring.
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        
        # Initialize storage backend
        if rate_limit_config.storage_backend == 'redis':
            self.storage = RedisRateLimitStorage()
        else:
            self.storage = MemoryRateLimitStorage()
        
        # Statistics
        self.total_requests = 0
        self.blocked_requests = 0
        self.requests_by_tier = defaultdict(int)
        self.blocks_by_tier = defaultdict(int)
        
        # Background cleanup task
        self.cleanup_task = None
        if isinstance(self.storage, MemoryRateLimitStorage):
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        logger.info(f"Enhanced rate limiting middleware initialized with {rate_limit_config.storage_backend} storage")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process rate limiting for incoming requests
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response object or rate limit error response
        """
        
        # Check if rate limiting is enabled
        if not rate_limit_config.enabled:
            return await call_next(request)
        
        # Check if path is exempt
        if self._is_path_exempt(request.url.path):
            return await call_next(request)
        
        # Check if IP is exempt
        client_ip = self._get_client_ip(request)
        if client_ip in rate_limit_config.exempt_ips:
            return await call_next(request)
        
        # Get rate limit configuration for this request
        rate_limit = self._get_rate_limit_for_request(request)
        
        # Generate rate limiting key
        rate_limit_key = self._generate_rate_limit_key(request)
        
        # Check rate limit
        result = await self._check_rate_limit(rate_limit_key, rate_limit)
        
        # Update statistics
        self._update_statistics(request, result)
        
        # Log if configured
        if rate_limit_config.log_all_requests or (rate_limit_config.log_violations and not result.allowed):
            self._log_rate_limit_check(request, result, rate_limit_key)
        
        # Handle rate limit exceeded
        if not result.allowed:
            return self._create_rate_limit_error_response(result, request)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        if rate_limit_config.rate_limit_headers:
            self._add_rate_limit_headers(response, result)
        
        return response
    
    def _is_path_exempt(self, path: str) -> bool:
        """Check if path is exempt from rate limiting"""
        return any(path.startswith(exempt_path) for exempt_path in rate_limit_config.exempt_paths)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address with proxy support"""
        
        # Check forwarded headers
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return 'unknown'
    
    def _get_rate_limit_for_request(self, request: Request) -> RateLimit:
        """Get appropriate rate limit for request"""
        
        # Check for path-specific limits first
        for path, path_limit in rate_limit_config.path_limits.items():
            if request.url.path.startswith(path):
                return path_limit
        
        # Get user tier from request
        user_tier = self._get_user_tier(request)
        
        # Return tier-based limit
        return rate_limit_config.limits_by_tier[user_tier]
    
    def _get_user_tier(self, request: Request) -> UserTier:
        """Determine user tier from request"""
        
        # Check if user is authenticated
        if hasattr(request.state, 'user'):
            user = request.state.user
            
            # Check if admin
            if getattr(user, 'is_admin', False):
                return UserTier.ADMIN
            
            # Check if premium
            if getattr(user, 'is_premium', False):
                return UserTier.PREMIUM
            
            # Authenticated user
            return UserTier.BASIC
        
        # Anonymous user
        return UserTier.ANONYMOUS
    
    def _generate_rate_limit_key(self, request: Request) -> str:
        """Generate unique rate limiting key for request"""
        
        # Get client identifier
        if hasattr(request.state, 'user'):
            # Use user ID for authenticated users
            user = request.state.user
            identifier = f"user:{getattr(user, 'user_id', 'unknown')}"
        else:
            # Use IP for anonymous users
            identifier = f"ip:{self._get_client_ip(request)}"
        
        # Add method and path for more granular limiting
        method_path = f"{request.method}:{request.url.path}"
        
        # Hash for consistent key length
        key_content = f"{identifier}:{method_path}"
        key_hash = hashlib.md5(key_content.encode()).hexdigest()
        
        return f"rate_limit:{key_hash}"
    
    async def _check_rate_limit(self, key: str, limit: RateLimit) -> RateLimitResult:
        """Check rate limit using appropriate algorithm"""
        
        algorithm_map = {
            RateLimitAlgorithm.SLIDING_WINDOW: RateLimitingAlgorithms.sliding_window,
            RateLimitAlgorithm.FIXED_WINDOW: RateLimitingAlgorithms.fixed_window,
            RateLimitAlgorithm.TOKEN_BUCKET: RateLimitingAlgorithms.token_bucket,
            RateLimitAlgorithm.LEAKY_BUCKET: RateLimitingAlgorithms.leaky_bucket
        }
        
        algorithm_func = algorithm_map.get(limit.algorithm, RateLimitingAlgorithms.sliding_window)
        
        try:
            result = await algorithm_func(self.storage, key, limit)
            
            # Apply burst protection if enabled
            if rate_limit_config.enable_burst_protection and result.allowed:
                burst_limit = int(limit.requests * limit.burst_multiplier)
                if result.current_requests > burst_limit:
                    result.allowed = False
                    result.retry_after = 60  # 1 minute burst cooldown
            
            return result
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if rate limiting fails
            return RateLimitResult(
                allowed=True,
                remaining=limit.requests,
                reset_time=int(time.time() + limit.window),
                limit=limit.requests,
                current_requests=0
            )
    
    def _update_statistics(self, request: Request, result: RateLimitResult):
        """Update middleware statistics"""
        
        self.total_requests += 1
        
        user_tier = self._get_user_tier(request)
        self.requests_by_tier[user_tier] += 1
        
        if not result.allowed:
            self.blocked_requests += 1
            self.blocks_by_tier[user_tier] += 1
    
    def _log_rate_limit_check(self, request: Request, result: RateLimitResult, key: str):
        """Log rate limit check"""
        
        log_data = {
            'client_ip': self._get_client_ip(request),
            'method': request.method,
            'path': request.url.path,
            'user_tier': self._get_user_tier(request).value,
            'rate_limit_key': key,
            'allowed': result.allowed,
            'remaining': result.remaining,
            'limit': result.limit,
            'current_requests': result.current_requests
        }
        
        if hasattr(request.state, 'user'):
            user = request.state.user
            log_data['user_id'] = getattr(user, 'user_id', None)
        
        if result.allowed:
            logger.debug("Rate limit check passed", extra=log_data)
        else:
            logger.warning("Rate limit exceeded", extra=log_data)
    
    def _create_rate_limit_error_response(self, result: RateLimitResult, request: Request) -> JSONResponse:
        """Create rate limit error response"""
        
        response_data = {
            'error': {
                'code': 'RATE_LIMIT_EXCEEDED',
                'message': 'Rate limit exceeded',
                'details': {
                    'limit': result.limit,
                    'remaining': result.remaining,
                    'reset_time': result.reset_time,
                    'retry_after': result.retry_after
                }
            },
            'request': {
                'path': request.url.path,
                'method': request.method
            }
        }
        
        headers = {}
        if result.retry_after:
            headers['Retry-After'] = str(result.retry_after)
        
        # Add custom headers
        headers.update(rate_limit_config.custom_headers)
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=response_data,
            headers=headers
        )
    
    def _add_rate_limit_headers(self, response: Response, result: RateLimitResult):
        """Add rate limit headers to response"""
        
        response.headers['X-RateLimit-Limit'] = str(result.limit)
        response.headers['X-RateLimit-Remaining'] = str(result.remaining)
        response.headers['X-RateLimit-Reset'] = str(result.reset_time)
        
        if result.retry_after:
            response.headers['X-RateLimit-Retry-After'] = str(result.retry_after)
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task for memory storage"""
        
        while True:
            try:
                await asyncio.sleep(rate_limit_config.cleanup_interval)
                await self.storage.cleanup_expired_keys()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        
        block_rate = self.blocked_requests / max(self.total_requests, 1)
        
        stats = {
            'total_requests': self.total_requests,
            'blocked_requests': self.blocked_requests,
            'block_rate': round(block_rate, 4),
            'requests_by_tier': dict(self.requests_by_tier),
            'blocks_by_tier': dict(self.blocks_by_tier),
            'storage_backend': rate_limit_config.storage_backend,
            'algorithms_enabled': [alg.value for alg in RateLimitAlgorithm]
        }
        
        return stats
    
    def __del__(self):
        """Cleanup when middleware is destroyed"""
        if self.cleanup_task:
            self.cleanup_task.cancel()

# ============================================
# Rate Limiting Decorators and Utilities
# ============================================

def custom_rate_limit(requests: int, window: int, 
                     algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW):
    """
    Decorator for custom rate limiting on specific endpoints
    
    Args:
        requests: Number of requests allowed
        window: Time window in seconds
        algorithm: Rate limiting algorithm to use
        
    Returns:
        Decorator function
    """
    
    def decorator(func):
        # Store rate limit metadata on function
        func._custom_rate_limit = RateLimit(
            requests=requests,
            window=window,
            algorithm=algorithm
        )
        return func
    
    return decorator

async def check_rate_limit_manual(request: Request, 
                                 rate_limit: RateLimit) -> RateLimitResult:
    """
    Manually check rate limit for custom scenarios
    
    Args:
        request: FastAPI request object
        rate_limit: Rate limit configuration
        
    Returns:
        Rate limit result
    """
    
    # Create temporary storage and middleware for manual checking
    if rate_limit_config.storage_backend == 'redis':
        storage = RedisRateLimitStorage()
    else:
        storage = MemoryRateLimitStorage()
    
    # Generate key
    client_ip = request.client.host if request.client else 'unknown'
    key = f"manual_check:{client_ip}:{request.url.path}"
    
    # Check using specified algorithm
    algorithm_map = {
        RateLimitAlgorithm.SLIDING_WINDOW: RateLimitingAlgorithms.sliding_window,
        RateLimitAlgorithm.FIXED_WINDOW: RateLimitingAlgorithms.fixed_window,
        RateLimitAlgorithm.TOKEN_BUCKET: RateLimitingAlgorithms.token_bucket,
        RateLimitAlgorithm.LEAKY_BUCKET: RateLimitingAlgorithms.leaky_bucket
    }
    
    algorithm_func = algorithm_map.get(rate_limit.algorithm, RateLimitingAlgorithms.sliding_window)
    return await algorithm_func(storage, key, rate_limit)

# ============================================
# Rate Limiting Health and Monitoring
# ============================================

def create_rate_limiting_health_check() -> Dict[str, Any]:
    """Create health check data for rate limiting system"""
    
    return {
        'status': 'enabled' if rate_limit_config.enabled else 'disabled',
        'storage_backend': rate_limit_config.storage_backend,
        'default_algorithm': rate_limit_config.default_algorithm.value,
        'config': {
            'enabled': rate_limit_config.enabled,
            'storage_backend': rate_limit_config.storage_backend,
            'burst_protection': rate_limit_config.enable_burst_protection,
            'cleanup_interval': rate_limit_config.cleanup_interval
        },
        'limits_by_tier': {
            tier.value: {
                'requests': limit.requests,
                'window': limit.window,
                'algorithm': limit.algorithm.value
            }
            for tier, limit in rate_limit_config.limits_by_tier.items()
        }
    }

# ============================================
# Export Components
# ============================================

__all__ = [
    # Main middleware
    "EnhancedRateLimitingMiddleware",
    
    # Configuration
    "RateLimitConfig",
    "rate_limit_config",
    
    # Data structures
    "RateLimit",
    "RateLimitResult", 
    "RateLimitAlgorithm",
    "UserTier",
    
    # Storage backends
    "RateLimitStorage",
    "MemoryRateLimitStorage",
    "RedisRateLimitStorage",
    
    # Algorithms
    "RateLimitingAlgorithms",
    
    # Utilities
    "custom_rate_limit",
    "check_rate_limit_manual",
    "create_rate_limiting_health_check",
]
