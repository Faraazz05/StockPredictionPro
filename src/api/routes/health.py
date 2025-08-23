# ============================================
# StockPredictionPro - src/api/routes/health.py
# Comprehensive health check routes for FastAPI with system monitoring, dependency checks, and diagnostic information
# ============================================

import asyncio
import time
import psutil
import platform
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..dependencies import (
    get_async_session,
    get_data_manager,
    get_cache_manager,
    get_backtest_engine,
    get_portfolio_manager,
    optional_auth  # Health checks shouldn't require authentication by default
)
from ..schemas.error_schemas import HealthCheckErrorResponse
from ...data.manager import DataManager
from ...data.cache import CacheManager
from ...evaluation.backtesting.engine import BacktestEngine
from ...trading.portfolio import PortfolioManager
from ...utils.logger import get_logger

logger = get_logger('api.routes.health')

# ============================================
# Router Configuration
# ============================================

router = APIRouter(
    prefix="/health",
    tags=["Health Checks"],
    responses={
        503: {"description": "Service Unavailable"}
    }
)

# ============================================
# Health Check Schemas
# ============================================

from pydantic import BaseModel, Field
from enum import Enum

class HealthStatus(str, Enum):
    """Health status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ServiceStatus(str, Enum):
    """Individual service status values"""
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class ComponentHealth(BaseModel):
    """Health status of individual component"""
    
    status: ServiceStatus = Field(description="Component status")
    response_time_ms: Optional[float] = Field(default=None, description="Response time in milliseconds")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last health check time")
    error_message: Optional[str] = Field(default=None, description="Error message if unhealthy")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional component metadata")

class SystemMetrics(BaseModel):
    """System resource metrics"""
    
    cpu_percent: float = Field(description="CPU usage percentage")
    memory_percent: float = Field(description="Memory usage percentage")
    disk_percent: float = Field(description="Disk usage percentage")
    load_average: Optional[List[float]] = Field(default=None, description="System load average")
    uptime_seconds: float = Field(description="System uptime in seconds")
    process_count: int = Field(description="Number of running processes")

class ApplicationMetrics(BaseModel):
    """Application-specific metrics"""
    
    version: str = Field(description="Application version")
    environment: str = Field(description="Environment (development, staging, production)")
    start_time: datetime = Field(description="Application start time")
    uptime_seconds: float = Field(description="Application uptime in seconds")
    python_version: str = Field(description="Python version")
    host_info: Dict[str, str] = Field(description="Host information")
    memory_usage_mb: float = Field(description="Application memory usage in MB")

class HealthCheckResponse(BaseModel):
    """Complete health check response"""
    
    status: HealthStatus = Field(description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    
    # Service health
    services: Dict[str, ComponentHealth] = Field(description="Individual service health status")
    
    # System metrics
    system: SystemMetrics = Field(description="System resource metrics")
    
    # Application metrics
    application: ApplicationMetrics = Field(description="Application metrics")
    
    # Summary
    total_checks: int = Field(description="Total number of health checks performed")
    passed_checks: int = Field(description="Number of successful health checks")
    failed_checks: int = Field(description="Number of failed health checks")
    
    # Performance
    check_duration_ms: float = Field(description="Total health check duration in milliseconds")

class LivenessResponse(BaseModel):
    """Simple liveness response"""
    
    status: str = Field(default="alive", description="Liveness status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")

class ReadinessResponse(BaseModel):
    """Readiness check response"""
    
    ready: bool = Field(description="Whether the service is ready to serve requests")
    services: Dict[str, ServiceStatus] = Field(description="Readiness of individual services")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")

# ============================================
# Health Check Functions
# ============================================

async def check_database_health(db: AsyncSession) -> ComponentHealth:
    """Check database connectivity and health"""
    
    start_time = time.time()
    
    try:
        # Simple database connectivity test
        result = await db.execute(text("SELECT 1"))
        result.fetchone()
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            status=ServiceStatus.UP,
            response_time_ms=round(response_time, 2),
            metadata={
                "connection_pool_size": str(db.get_bind().pool.size()),
                "checked_out_connections": str(db.get_bind().pool.checkedout())
            }
        )
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            status=ServiceStatus.DOWN,
            response_time_ms=round(response_time, 2),
            error_message=str(e)
        )

async def check_cache_health(cache_manager: CacheManager) -> ComponentHealth:
    """Check cache (Redis) connectivity and health"""
    
    start_time = time.time()
    
    try:
        # Test cache connectivity
        test_key = "health_check_test"
        test_value = "ping"
        
        await cache_manager.set(test_key, test_value, ttl=10)
        retrieved_value = await cache_manager.get(test_key)
        
        if retrieved_value != test_value:
            raise Exception("Cache read/write test failed")
        
        # Clean up test key
        await cache_manager.delete(test_key)
        
        response_time = (time.time() - start_time) * 1000
        
        # Get cache info
        cache_info = await cache_manager.get_info()
        
        return ComponentHealth(
            status=ServiceStatus.UP,
            response_time_ms=round(response_time, 2),
            metadata={
                "connected_clients": cache_info.get("connected_clients"),
                "used_memory": cache_info.get("used_memory_human"),
                "keyspace_hits": cache_info.get("keyspace_hits"),
                "keyspace_misses": cache_info.get("keyspace_misses")
            }
        )
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            status=ServiceStatus.DOWN,
            response_time_ms=round(response_time, 2),
            error_message=str(e)
        )

async def check_data_manager_health(data_manager: DataManager) -> ComponentHealth:
    """Check data manager health and external data sources"""
    
    start_time = time.time()
    
    try:
        # Test data source connectivity
        health_status = await data_manager.check_health()
        
        response_time = (time.time() - start_time) * 1000
        
        if health_status.get("status") == "healthy":
            return ComponentHealth(
                status=ServiceStatus.UP,
                response_time_ms=round(response_time, 2),
                metadata=health_status.get("details", {})
            )
        else:
            return ComponentHealth(
                status=ServiceStatus.DEGRADED,
                response_time_ms=round(response_time, 2),
                error_message=health_status.get("message", "Data manager unhealthy"),
                metadata=health_status.get("details", {})
            )
            
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            status=ServiceStatus.DOWN,
            response_time_ms=round(response_time, 2),
            error_message=str(e)
        )

async def check_backtest_engine_health(backtest_engine: BacktestEngine) -> ComponentHealth:
    """Check backtest engine health"""
    
    start_time = time.time()
    
    try:
        # Check backtest engine status
        engine_status = await backtest_engine.health_check()
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            status=ServiceStatus.UP if engine_status.get("healthy") else ServiceStatus.DEGRADED,
            response_time_ms=round(response_time, 2),
            metadata={
                "active_backtests": engine_status.get("active_jobs", 0),
                "queue_size": engine_status.get("queue_size", 0)
            }
        )
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            status=ServiceStatus.DOWN,
            response_time_ms=round(response_time, 2),
            error_message=str(e)
        )

def get_system_metrics() -> SystemMetrics:
    """Get system resource metrics"""
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # Disk usage
    disk = psutil.disk_usage('/')
    disk_percent = (disk.used / disk.total) * 100
    
    # Load average (Unix systems only)
    load_average = None
    try:
        if hasattr(psutil, 'getloadavg'):
            load_average = list(psutil.getloadavg())
    except (AttributeError, OSError):
        pass
    
    # System uptime
    boot_time = psutil.boot_time()
    uptime_seconds = time.time() - boot_time
    
    # Process count
    process_count = len(psutil.pids())
    
    return SystemMetrics(
        cpu_percent=round(cpu_percent, 2),
        memory_percent=round(memory_percent, 2),
        disk_percent=round(disk_percent, 2),
        load_average=load_average,
        uptime_seconds=round(uptime_seconds, 2),
        process_count=process_count
    )

def get_application_metrics() -> ApplicationMetrics:
    """Get application-specific metrics"""
    
    # Application start time (would be set globally in real app)
    app_start_time = datetime.utcnow() - timedelta(seconds=3600)  # Placeholder
    uptime_seconds = (datetime.utcnow() - app_start_time).total_seconds()
    
    # Host information
    host_info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor() or "unknown"
    }
    
    # Current process memory usage
    current_process = psutil.Process()
    memory_usage_mb = current_process.memory_info().rss / 1024 / 1024
    
    return ApplicationMetrics(
        version="1.0.0",  # Would come from config
        environment="development",  # Would come from config
        start_time=app_start_time,
        uptime_seconds=round(uptime_seconds, 2),
        python_version=platform.python_version(),
        host_info=host_info,
        memory_usage_mb=round(memory_usage_mb, 2)
    )

# ============================================
# Route Handlers
# ============================================

@router.get("/",
    response_model=HealthCheckResponse,
    summary="Comprehensive health check",
    description="Detailed health check of all system components and dependencies"
)
async def comprehensive_health_check(
    db: AsyncSession = Depends(get_async_session),
    data_manager: DataManager = Depends(get_data_manager),
    cache_manager: CacheManager = Depends(get_cache_manager),
    backtest_engine: BacktestEngine = Depends(get_backtest_engine)
) -> HealthCheckResponse:
    """
    Comprehensive health check endpoint
    
    Checks the health of all system components including:
    - Database connectivity
    - Cache (Redis) connectivity  
    - External data sources
    - Background services
    - System resources
    - Application metrics
    
    Returns detailed status information for monitoring and alerting.
    """
    
    overall_start_time = time.time()
    
    try:
        # Perform all health checks in parallel
        health_checks = await asyncio.gather(
            check_database_health(db),
            check_cache_health(cache_manager),
            check_data_manager_health(data_manager),
            check_backtest_engine_health(backtest_engine),
            return_exceptions=True
        )
        
        # Process results
        services = {
            "database": health_checks[0] if not isinstance(health_checks[0], Exception) 
                       else ComponentHealth(status=ServiceStatus.DOWN, error_message=str(health_checks[0])),
            "cache": health_checks[1] if not isinstance(health_checks[1], Exception)
                    else ComponentHealth(status=ServiceStatus.DOWN, error_message=str(health_checks[1])),
            "data_manager": health_checks[2] if not isinstance(health_checks[2], Exception)
                           else ComponentHealth(status=ServiceStatus.DOWN, error_message=str(health_checks[2])),
            "backtest_engine": health_checks[3] if not isinstance(health_checks[3], Exception)
                              else ComponentHealth(status=ServiceStatus.DOWN, error_message=str(health_checks[3]))
        }
        
        # Get system and application metrics
        system_metrics = get_system_metrics()
        app_metrics = get_application_metrics()
        
        # Calculate overall health status
        total_checks = len(services)
        passed_checks = sum(1 for service in services.values() if service.status == ServiceStatus.UP)
        failed_checks = total_checks - passed_checks
        
        # Determine overall status
        if failed_checks == 0:
            overall_status = HealthStatus.HEALTHY
        elif failed_checks < total_checks / 2:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNHEALTHY
        
        # Calculate total check duration
        check_duration_ms = (time.time() - overall_start_time) * 1000
        
        response = HealthCheckResponse(
            status=overall_status,
            services=services,
            system=system_metrics,
            application=app_metrics,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            check_duration_ms=round(check_duration_ms, 2)
        )
        
        # Return appropriate HTTP status code
        if overall_status == HealthStatus.UNHEALTHY:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.model_dump()
            )
        elif overall_status == HealthStatus.DEGRADED:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=response.model_dump()
            )
        else:
            return response
            
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        
        # Return error response
        error_response = HealthCheckErrorResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            services={},
            errors=[{
                "code": "HEALTH_CHECK_ERROR",
                "message": f"Health check failed: {str(e)}"
            }],
            recovery_estimate="unknown"
        )
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response.model_dump()
        )

@router.get("/liveness",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Simple liveness check for container orchestration"
)
async def liveness_check() -> LivenessResponse:
    """
    Liveness probe endpoint
    
    Simple check to verify that the application is running.
    Used by container orchestrators (Kubernetes, Docker Swarm) 
    to determine if the container should be restarted.
    
    Always returns 200 OK if the application is responding.
    """
    
    return LivenessResponse()

@router.get("/readiness",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Readiness check to determine if service can accept requests"
)
async def readiness_check(
    db: AsyncSession = Depends(get_async_session),
    cache_manager: CacheManager = Depends(get_cache_manager)
) -> ReadinessResponse:
    """
    Readiness probe endpoint
    
    Checks if the service is ready to serve requests by verifying
    that critical dependencies are available:
    - Database connectivity
    - Cache connectivity
    
    Used by load balancers and container orchestrators to determine
    if the service should receive traffic.
    """
    
    try:
        # Quick checks of critical services
        services_status = {}
        ready = True
        
        # Check database
        try:
            await db.execute(text("SELECT 1"))
            services_status["database"] = ServiceStatus.UP
        except Exception:
            services_status["database"] = ServiceStatus.DOWN
            ready = False
        
        # Check cache
        try:
            await cache_manager.ping()
            services_status["cache"] = ServiceStatus.UP
        except Exception:
            services_status["cache"] = ServiceStatus.DOWN
            ready = False
        
        response = ReadinessResponse(
            ready=ready,
            services=services_status
        )
        
        # Return 503 if not ready
        if not ready:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.model_dump()
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "ready": False,
                "services": {},
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.get("/ping",
    response_class=PlainTextResponse,
    summary="Simple ping endpoint",
    description="Minimal health check that returns 'pong'"
)
async def ping() -> PlainTextResponse:
    """
    Simple ping endpoint
    
    Returns plain text 'pong' response.
    Useful for basic connectivity testing and monitoring tools
    that expect simple text responses.
    """
    
    return PlainTextResponse(content="pong")

@router.get("/version",
    summary="Application version information",
    description="Returns application version and build information"
)
async def version_info():
    """
    Application version endpoint
    
    Returns version information including:
    - Application version
    - Build timestamp
    - Git commit (if available)
    - Environment information
    """
    
    return {
        "application": "StockPredictionPro",
        "version": "1.0.0",
        "build_time": "2023-01-01T00:00:00Z",
        "git_commit": "abc123def456",  # Would come from build process
        "environment": "development",
        "python_version": platform.python_version(),
        "platform": platform.platform()
    }

@router.get("/metrics",
    summary="Application metrics",
    description="Prometheus-style metrics for monitoring"
)
async def metrics():
    """
    Application metrics endpoint
    
    Returns metrics in a format suitable for monitoring systems.
    Can be consumed by Prometheus, Grafana, or other monitoring tools.
    """
    
    try:
        # Get system metrics
        system = get_system_metrics()
        app = get_application_metrics()
        
        # Format as simple key-value pairs (Prometheus format would be more complex)
        metrics_data = {
            "system_cpu_percent": system.cpu_percent,
            "system_memory_percent": system.memory_percent,
            "system_disk_percent": system.disk_percent,
            "system_process_count": system.process_count,
            "application_uptime_seconds": app.uptime_seconds,
            "application_memory_usage_mb": app.memory_usage_mb,
            "health_check_timestamp": time.time()
        }
        
        return metrics_data
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to collect metrics"
        )

# ============================================
# Export router
# ============================================

__all__ = ["router"]
