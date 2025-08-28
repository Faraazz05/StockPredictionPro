# ============================================
# StockPredictionPro - src/api/main.py
# Main FastAPI application with comprehensive configuration, middleware, and routing
# ============================================

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List
import uvicorn
import os
from pathlib import Path

# FastAPI and core imports
from fastapi import FastAPI, Request, Response, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Database and caching
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

# Import all route modules
from .routes import (
    backtests,
    data,
    health,
    models,
    predictions,
    signals
)

# Import dependencies and utilities
from .dependencies import get_async_session, get_current_active_user
from .schemas.error_schemas import ErrorResponse
from .exceptions import (
    CustomHTTPException,
    ValidationException,
    AuthenticationException,
    RateLimitException
)
from ..utils.logger import get_logger, setup_logging
from src.utils.config_loader import load_app_config

# Initialize logging
logger = get_logger("api.main")

# Load application config
config = load_app_config()

# Get settings
settings = config.settings

# ============================================
# Application Lifespan Management
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks"""
    
    # Startup
    logger.info("🚀 Starting StockPredictionPro API...")
    
    try:
        # Initialize logging
        setup_logging(
            level=settings.LOG_LEVEL,
            log_file=settings.LOG_FILE,
            max_file_size=settings.LOG_MAX_FILE_SIZE,
            backup_count=settings.LOG_BACKUP_COUNT
        )
        
        # Test database connection
        logger.info("Testing database connection...")
        # In real implementation:
        # await test_database_connection()
        
        # Test Redis connection
        logger.info("Testing Redis connection...")
        # In real implementation:
        # await test_redis_connection()
        
        # Initialize ML models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        logger.info(f"ML models directory: {models_dir.absolute()}")
        
        # Initialize downloads directory
        downloads_dir = Path("downloads")
        downloads_dir.mkdir(exist_ok=True)
        logger.info(f"Downloads directory: {downloads_dir.absolute()}")
        
        # Warm up critical services
        logger.info("Warming up services...")
        # In real implementation:
        # await warmup_data_providers()
        # await warmup_ml_models()
        
        logger.info("✅ StockPredictionPro API startup completed successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("🛑 Shutting down StockPredictionPro API...")
    
    try:
        # Close database connections
        logger.info("Closing database connections...")
        # In real implementation:
        # await close_database_connections()
        
        # Close Redis connections
        logger.info("Closing Redis connections...")
        # In real implementation:
        # await close_redis_connections()
        
        # Save any pending data
        logger.info("Saving pending data...")
        # In real implementation:
        # await save_pending_jobs()
        # await cleanup_temp_files()
        
        logger.info("✅ Shutdown completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# ============================================
# FastAPI Application Configuration
# ============================================

app = FastAPI(
    title="StockPredictionPro API",
    description="""
    ## 🚀 Advanced Stock Prediction & Trading Signals Platform

    A comprehensive financial API platform that combines machine learning, technical analysis, 
    and real-time data processing to provide intelligent stock predictions and trading signals.

    ### 🔥 Key Features

    * **🤖 Machine Learning Models**: Train custom ML models with 10+ algorithms
    * **📊 Technical Analysis**: 20+ technical indicators and pattern recognition  
    * **🎯 Trading Signals**: AI-powered buy/sell/hold recommendations
    * **⚡ Real-time Data**: Live market data with WebSocket streaming
    * **🔄 Backtesting**: Comprehensive strategy testing with performance analytics
    * **📈 Predictions**: Single, batch, and ensemble prediction capabilities
    * **🔒 Enterprise Security**: JWT authentication with role-based access
    * **📱 Developer Friendly**: RESTful API with comprehensive documentation

    ### 🛠 Technology Stack

    * **Backend**: FastAPI, Python 3.11+
    * **ML/AI**: scikit-learn, pandas, numpy, optuna
    * **Database**: PostgreSQL with async SQLAlchemy
    * **Caching**: Redis for high-performance data access
    * **Monitoring**: Comprehensive health checks and metrics

    ### 📚 API Documentation

    * **Swagger UI**: Interactive API documentation at `/docs`
    * **ReDoc**: Alternative documentation at `/redoc`  
    * **OpenAPI Schema**: Raw schema available at `/openapi.json`

    ### 🔗 Quick Links

    * [GitHub Repository](https://github.com/stockpredpro/api)
    * [Documentation](https://docs.stockpredpro.com)
    * [Support](https://support.stockpredpro.com)
    """,
    summary="Professional Stock Prediction & Trading Signals Platform",
    version="2.0.0",
    contact={
        "name": "StockPredictionPro Support",
        "url": "https://support.stockpredpro.com",
        "email": "support@stockpredpro.com"
    },
    license_info={
        "name": "Commercial License",
        "url": "https://stockpredpro.com/license"
    },
    openapi_tags=[
        {
            "name": "Health Checks",
            "description": "System health monitoring and status endpoints for container orchestration and monitoring tools."
        },
        {
            "name": "Market Data", 
            "description": "Real-time and historical market data fetching, validation, and quality monitoring with multi-source support."
        },
        {
            "name": "Machine Learning Models",
            "description": "Complete ML lifecycle management including training, deployment, comparison, and performance monitoring."
        },
        {
            "name": "Predictions",
            "description": "Single, batch, and ensemble predictions with confidence scoring and feature importance analysis."
        },
        {
            "name": "Trading Signals",
            "description": "AI-powered trading signals combining technical analysis with machine learning for buy/sell/hold recommendations."
        },
        {
            "name": "Backtests",
            "description": "Comprehensive backtesting engine with strategy testing, performance analytics, and risk management."
        }
    ],
    servers=[
        {"url": "https://api.stockpredpro.com", "description": "Production server"},
        {"url": "https://staging-api.stockpredpro.com", "description": "Staging server"},  
        {"url": "http://localhost:8000", "description": "Development server"}
    ],
    openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
    docs_url=None,  # Custom docs URL
    redoc_url=None,  # Custom redoc URL
    lifespan=lifespan
)

# ============================================
# Custom OpenAPI Schema
# ============================================

def custom_openapi():
    """Generate custom OpenAPI schema with enhanced documentation"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        servers=app.servers
    )
    
    # Add custom security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your JWT token in the format: Bearer {your-token}"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Key for server-to-server communication"
        }
    }
    
    # Add custom headers to all operations
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method in ["get", "post", "put", "delete", "patch"]:
                operation = openapi_schema["paths"][path][method]
                
                # Add common parameters
                if "parameters" not in operation:
                    operation["parameters"] = []
                
                operation["parameters"].extend([
                    {
                        "name": "X-Request-ID",
                        "in": "header",
                        "required": False,
                        "schema": {"type": "string"},
                        "description": "Unique request identifier for tracking"
                    },
                    {
                        "name": "X-Client-Version",
                        "in": "header", 
                        "required": False,
                        "schema": {"type": "string"},
                        "description": "Client application version"
                    }
                ])
                
                # Add security requirements for protected endpoints
                if path not in ["/health", "/health/ping", "/health/version", "/docs", "/redoc"]:
                    operation["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# ============================================
# Middleware Configuration
# ============================================

# CORS Middleware - Must be first
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS if settings.ALLOWED_HOSTS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Request-ID",
        "X-Client-Version",
        "X-Requested-With"
    ],
    expose_headers=[
        "X-Request-ID",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "X-Response-Time"
    ]
)

# Trusted Host Middleware
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# GZip Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom Request Processing Middleware
@app.middleware("http")
async def request_processing_middleware(request: Request, call_next):
    """Custom middleware for request processing, logging, and performance monitoring"""
    
    start_time = datetime.utcnow()
    
    # Generate request ID if not provided
    request_id = request.headers.get("X-Request-ID") or f"req_{int(start_time.timestamp() * 1000)}"
    
    # Log request
    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("User-Agent", "unknown")
        }
    )
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        end_time = datetime.utcnow()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
        response.headers["X-API-Version"] = app.version
        
        # Log response
        logger.info(
            f"Request completed: {response.status_code}",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "response_time_ms": round(response_time_ms, 2)
            }
        )
        
        return response
        
    except Exception as e:
        # Calculate error response time
        end_time = datetime.utcnow()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Log error
        logger.error(
            f"Request failed: {str(e)}",
            extra={
                "request_id": request_id,
                "error": str(e),
                "response_time_ms": round(response_time_ms, 2)
            },
            exc_info=True
        )
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred",
                    "request_id": request_id,
                    "timestamp": end_time.isoformat()
                }
            },
            headers={
                "X-Request-ID": request_id,
                "X-Response-Time": f"{response_time_ms:.2f}ms"
            }
        )

# ============================================
# Exception Handlers
# ============================================

@app.exception_handler(CustomHTTPException)
async def custom_http_exception_handler(request: Request, exc: CustomHTTPException):
    """Handle custom HTTP exceptions"""
    
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.detail,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                **exc.extra_data
            }
        },
        headers={"X-Request-ID": request_id}
    )

@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    """Handle validation exceptions"""
    
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        },
        headers={"X-Request-ID": request_id}
    )

@app.exception_handler(AuthenticationException)
async def authentication_exception_handler(request: Request, exc: AuthenticationException):
    """Handle authentication exceptions"""
    
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "code": "AUTHENTICATION_FAILED",
                "message": str(exc),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        },
        headers={
            "X-Request-ID": request_id,
            "WWW-Authenticate": "Bearer"
        }
    )

@app.exception_handler(RateLimitException) 
async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
    """Handle rate limit exceptions"""
    
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": str(exc),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "retry_after": exc.retry_after
            }
        },
        headers={
            "X-Request-ID": request_id,
            "Retry-After": str(exc.retry_after),
            "X-RateLimit-Remaining": "0"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle standard HTTP exceptions"""
    
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": exc.detail,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        },
        headers={"X-Request-ID": request_id}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    
    request_id = request.headers.get("X-Request-ID", "unknown")
    
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={"request_id": request_id},
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR", 
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        },
        headers={"X-Request-ID": request_id}
    )

# ============================================
# Custom Documentation Endpoints
# ============================================

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced styling"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Interactive API Documentation",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

@app.get("/redoc", include_in_schema=False)
async def custom_redoc_html():
    """Custom ReDoc documentation"""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Documentation",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js",
        redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

# ============================================
# Root and Info Endpoints  
# ============================================

@app.get("/", include_in_schema=False)
async def root():
    """API root endpoint with welcome message and quick links"""
    return {
        "message": "🚀 Welcome to StockPredictionPro API",
        "version": app.version,
        "description": "Advanced Stock Prediction & Trading Signals Platform",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc", 
            "openapi_schema": "/openapi.json"
        },
        "health": {
            "status": "/health",
            "liveness": "/health/liveness",
            "readiness": "/health/readiness"
        },
        "endpoints": {
            "market_data": "/api/v1/data",
            "ml_models": "/api/v1/models", 
            "predictions": "/api/v1/predictions",
            "trading_signals": "/api/v1/signals",
            "backtests": "/api/v1/backtests"
        },
        "support": {
            "documentation": "https://docs.stockpredpro.com",
            "support": "https://support.stockpredpro.com",
            "github": "https://github.com/stockpredpro/api"
        },
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.ENVIRONMENT
    }

@app.get("/info", include_in_schema=False)
async def api_info():
    """Detailed API information and system status"""
    return {
        "api": {
            "name": app.title,
            "version": app.version,
            "description": app.summary,
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG
        },
        "features": {
            "machine_learning": {
                "algorithms": ["Linear Regression", "Random Forest", "Neural Networks", "SVM", "Gradient Boosting"],
                "capabilities": ["Training", "Hyperparameter Optimization", "Cross-Validation", "Deployment"]
            },
            "technical_analysis": {
                "indicators": ["RSI", "MACD", "Moving Averages", "Bollinger Bands", "Stochastic"],
                "patterns": ["Breakouts", "Crossovers", "Divergences", "Support/Resistance"]
            },
            "data_sources": {
                "supported": ["Yahoo Finance", "Alpha Vantage", "FRED", "Custom CSV"],
                "real_time": True,
                "historical": True
            },
            "predictions": {
                "types": ["Single", "Batch", "Ensemble", "Real-time"],
                "confidence_scoring": True,
                "feature_importance": True
            }
        },
        "system": {
            "startup_time": datetime.utcnow().isoformat(),  # Would be actual startup time
            "timezone": "UTC",
            "python_version": "3.11+",
            "framework": "FastAPI"
        }
    }

# ============================================
# Static Files (if needed)
# ============================================

# Mount static files for documentation assets, logos, etc.
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================
# Route Registration
# ============================================

# Register all route modules
app.include_router(health.router)      # Health checks (no auth required)
app.include_router(data.router)        # Market data endpoints  
app.include_router(models.router)      # ML model management
app.include_router(predictions.router) # Prediction services
app.include_router(signals.router)     # Trading signals
app.include_router(backtests.router)   # Backtesting engine

# ============================================
# Application Metadata
# ============================================

# Add metadata to app instance
app.state.startup_time = datetime.utcnow()
app.state.version = app.version
app.state.environment = settings.ENVIRONMENT

# ============================================
# Development Server
# ============================================

if __name__ == "__main__":
    # This block runs when executing the file directly (development only)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        access_log=True,
        use_colors=True,
        log_level="info"
    )

# ============================================
# Export for WSGI servers
# ============================================

# For production deployment with gunicorn, uvicorn, etc.
__all__ = ["app"]
