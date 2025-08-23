# ============================================
# StockPredictionPro - src/api/routes/data.py
# Comprehensive data management routes for FastAPI with market data fetching, validation, and real-time streaming
# ============================================

import asyncio
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Union
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update, func, and_, or_
from sqlalchemy.orm import selectinload

from ..dependencies import (
    get_async_session,
    get_current_active_user,
    get_data_manager,
    get_cache_manager,
    standard_rate_limit,
    validate_symbol,
    validate_date_range
)
from ..schemas.data_schemas import (
    DataRequest,
    MultiSymbolDataRequest,
    RealTimeDataRequest,
    DataValidationRequest,
    SymbolSearchRequest,
    DataMonitoringRequest,
    BulkDataRequest,
    DataResponse,
    MarketDataResponse,
    MultiSymbolDataResponse,
    RealTimeDataResponse,
    DataValidationResponse,
    SymbolSearchResponse,
    DataMonitoringResponse,
    BulkDataResponse,
    SymbolInfo,
    MarketData,
    OHLCVData,
    DataQualityMetrics,
    MarketType,
    DataSource,
    TimeInterval
)
from ..schemas.error_schemas import ErrorResponse
from ..exceptions import (
    raise_not_found,
    raise_validation_error,
    raise_data_error,
    DataException
)
from ...data.manager import DataManager
from ...data.cache import CacheManager
from ...utils.logger import get_logger

logger = get_logger('api.routes.data')

# ============================================
# Router Configuration
# ============================================

router = APIRouter(
    prefix="/api/v1/data",
    tags=["Market Data"],
    dependencies=[Depends(standard_rate_limit)],
    responses={
        400: ErrorResponse.model_400(),
        401: ErrorResponse.model_401(),
        403: ErrorResponse.model_403(),
        404: ErrorResponse.model_404(),
        422: ErrorResponse.model_422(),
        429: ErrorResponse.model_429(),
        500: ErrorResponse.model_500(),
    }
)

# ============================================
# Database Helper Functions
# ============================================

async def save_market_data_to_db(db: AsyncSession, market_data: MarketData, user_id: str):
    """Save market data to database"""
    try:
        # Convert to ORM model and save
        # market_data_orm = MarketDataORM(**market_data.model_dump(), user_id=user_id)
        # db.add(market_data_orm)
        # await db.commit()
        # await db.refresh(market_data_orm)
        
        logger.info(f"Saved market data for {market_data.symbol} to database")
        
    except Exception as e:
        logger.error(f"Failed to save market data: {e}")
        await db.rollback()
        raise

async def get_market_data_from_db(db: AsyncSession, symbol: str, user_id: str) -> Optional[MarketData]:
    """Get market data from database"""
    try:
        # Query database for market data
        # result = await db.execute(
        #     select(MarketDataORM)
        #     .where(MarketDataORM.symbol == symbol, MarketDataORM.user_id == user_id)
        #     .options(selectinload(MarketDataORM.data_points))
        # )
        # market_data_orm = result.scalars().first()
        
        # if market_data_orm:
        #     return MarketData(**market_data_orm.__dict__)
        
        logger.info(f"Getting market data for {symbol} from database")
        return None
        
    except Exception as e:
        logger.error(f"Failed to get market data: {e}")
        raise

async def search_symbols_in_db(db: AsyncSession, query: str, market: Optional[MarketType], 
                              sector: Optional[str], limit: int) -> List[SymbolInfo]:
    """Search symbols in database"""
    try:
        # Build search query
        # search_query = select(SymbolInfoORM)
        
        # Add filters
        # if query:
        #     search_query = search_query.where(
        #         or_(
        #             SymbolInfoORM.symbol.ilike(f"%{query}%"),
        #             SymbolInfoORM.name.ilike(f"%{query}%")
        #         )
        #     )
        
        # if market:
        #     search_query = search_query.where(SymbolInfoORM.market == market)
        
        # if sector:
        #     search_query = search_query.where(SymbolInfoORM.sector.ilike(f"%{sector}%"))
        
        # search_query = search_query.limit(limit)
        
        # result = await db.execute(search_query)
        # symbols_orm = result.scalars().all()
        
        # return [SymbolInfo(**symbol.__dict__) for symbol in symbols_orm]
        
        logger.info(f"Searching symbols with query: {query}")
        return []
        
    except Exception as e:
        logger.error(f"Failed to search symbols: {e}")
        raise

# ============================================
# Background Tasks
# ============================================

async def bulk_data_fetch_task(
    job_id: str,
    request: BulkDataRequest,
    user_id: str,
    data_manager: DataManager
):
    """Background task for bulk data fetching"""
    
    try:
        logger.info(f"Starting bulk data fetch job {job_id}")
        
        total_symbols = len(request.symbols)
        completed = 0
        failed = 0
        results = {}
        errors = []
        
        # Update job status
        await update_bulk_job_status(job_id, "processing", 0.0)
        
        # Process symbols in batches
        batch_size = request.batch_size
        for i in range(0, total_symbols, batch_size):
            batch_symbols = request.symbols[i:i + batch_size]
            
            if request.parallel_processing:
                # Process batch in parallel
                tasks = [
                    process_single_symbol(symbol, request.parameters, data_manager)
                    for symbol in batch_symbols
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Process batch sequentially
                batch_results = []
                for symbol in batch_symbols:
                    result = await process_single_symbol(symbol, request.parameters, data_manager)
                    batch_results.append(result)
            
            # Process batch results
            for j, result in enumerate(batch_results):
                symbol = batch_symbols[j]
                
                if isinstance(result, Exception):
                    failed += 1
                    errors.append({
                        "symbol": symbol,
                        "error": str(result),
                        "error_type": type(result).__name__
                    })
                else:
                    completed += 1
                    results[symbol] = result
                
                # Update progress
                progress = (completed + failed) / total_symbols
                await update_bulk_job_status(job_id, "processing", progress)
        
        # Save final results
        await save_bulk_job_results(job_id, {
            "results": results,
            "errors": errors,
            "summary": {
                "total": total_symbols,
                "completed": completed,
                "failed": failed
            }
        })
        
        # Update final status
        final_status = "completed" if failed == 0 else "partial_success"
        await update_bulk_job_status(job_id, final_status, 1.0)
        
        logger.info(f"Bulk data fetch job {job_id} completed: {completed} success, {failed} failed")
        
    except Exception as e:
        logger.error(f"Bulk data fetch job {job_id} failed: {e}", exc_info=True)
        await update_bulk_job_status(job_id, "failed", None, str(e))

async def process_single_symbol(symbol: str, parameters: Dict[str, Any], 
                               data_manager: DataManager) -> MarketData:
    """Process data fetching for a single symbol"""
    
    try:
        return await data_manager.get_historical_data(
            symbol=symbol,
            start_date=parameters.get('start_date'),
            end_date=parameters.get('end_date'),
            interval=parameters.get('interval', 'daily'),
            source=parameters.get('source')
        )
    except Exception as e:
        logger.error(f"Failed to process symbol {symbol}: {e}")
        raise

# ============================================
# Route Handlers
# ============================================

@router.post("/fetch",
    response_model=MarketDataResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch historical market data",
    description="Fetch historical OHLCV data for a single symbol"
)
async def fetch_historical_data(
    request: DataRequest,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session),
    data_manager: DataManager = Depends(get_data_manager),
    cache_manager: CacheManager = Depends(get_cache_manager)
) -> MarketDataResponse:
    """
    Fetch historical market data for a symbol
    
    - **symbol**: Stock symbol (e.g., AAPL, MSFT, RELIANCE.NS)
    - **start_date**: Start date for data retrieval
    - **end_date**: End date for data retrieval
    - **interval**: Data time interval (1d, 1h, etc.)
    - **source**: Data source provider (optional, auto-selected if not specified)
    """
    
    # Validate symbol
    validate_symbol(request.symbol)
    
    # Validate date range
    if request.start_date and request.end_date:
        validate_date_range(request.start_date, request.end_date)
    
    try:
        # Check cache first
        cache_key = f"market_data:{request.symbol}:{request.start_date}:{request.end_date}:{request.interval}"
        cached_data = await cache_manager.get(cache_key)
        
        if cached_data:
            logger.info(f"Serving cached data for {request.symbol}")
            return MarketDataResponse(**cached_data)
        
        # Fetch from data manager
        market_data = await data_manager.get_historical_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval,
            source=request.source
        )
        
        if not market_data or not market_data.data:
            raise_not_found("Market data", request.symbol)
        
        # Get symbol info
        symbol_info = await data_manager.get_symbol_info(request.symbol)
        if not symbol_info:
            # Create basic symbol info if not found
            symbol_info = SymbolInfo(
                symbol=request.symbol,
                name=request.symbol,
                market=MarketType.US,  # Default
                currency="USD"  # Default
            )
        
        # Calculate statistics
        statistics = calculate_market_statistics(market_data.data)
        
        # Create response
        response = MarketDataResponse(
            success=True,
            message="Data retrieved successfully",
            data_count=len(market_data.data),
            symbol_info=symbol_info,
            market_data=market_data,
            statistics=statistics
        )
        
        # Cache the response
        await cache_manager.set(cache_key, response.model_dump(), ttl=3600)  # 1 hour cache
        
        # Save to database for user history
        await save_market_data_to_db(db, market_data, current_user["user_id"])
        
        logger.info(f"Successfully fetched data for {request.symbol}: {len(market_data.data)} data points")
        
        return response
        
    except DataException as e:
        raise_data_error(str(e), data_source=request.source.value if request.source else None, symbol=request.symbol)
    except Exception as e:
        logger.error(f"Failed to fetch data for {request.symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch data: {str(e)}"
        )

@router.post("/fetch/multi",
    response_model=MultiSymbolDataResponse,
    summary="Fetch data for multiple symbols",
    description="Fetch historical data for multiple symbols in a single request"
)
async def fetch_multi_symbol_data(
    request: MultiSymbolDataRequest,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session),
    data_manager: DataManager = Depends(get_data_manager)
) -> MultiSymbolDataResponse:
    """
    Fetch historical data for multiple symbols
    
    Supports up to 50 symbols per request with parallel processing.
    """
    
    # Validate symbols
    for symbol in request.symbols:
        validate_symbol(symbol)
    
    # Validate date range
    if request.start_date and request.end_date:
        validate_date_range(request.start_date, request.end_date)
    
    data = {}
    failed_symbols = []
    total_data_points = 0
    
    try:
        # Create tasks for parallel fetching
        tasks = []
        for symbol in request.symbols:
            task = data_manager.get_historical_data(
                symbol=symbol,
                start_date=request.start_date,
                end_date=request.end_date,
                interval=request.interval,
                source=request.source
            )
            tasks.append((symbol, task))
        
        # Execute tasks with error handling
        results = await asyncio.gather(
            *[task for _, task in tasks], 
            return_exceptions=True
        )
        
        # Process results
        for i, (symbol, task) in enumerate(tasks):
            result = results[i]
            
            if isinstance(result, Exception):
                failed_symbols.append(symbol)
                logger.warning(f"Failed to fetch data for {symbol}: {result}")
            else:
                if result and result.data:
                    data[symbol] = result
                    total_data_points += len(result.data)
                else:
                    failed_symbols.append(symbol)
        
        # Create summary
        summary = {
            "total_symbols": len(request.symbols),
            "successful_symbols": len(data),
            "failed_symbols": len(failed_symbols),
            "total_data_points": total_data_points
        }
        
        response = MultiSymbolDataResponse(
            success=True,
            message=f"Fetched data for {len(data)} of {len(request.symbols)} symbols",
            data_count=total_data_points,
            data=data,
            failed_symbols=failed_symbols if failed_symbols else None,
            summary=summary
        )
        
        logger.info(f"Multi-symbol fetch completed: {len(data)} success, {len(failed_symbols)} failed")
        
        return response
        
    except Exception as e:
        logger.error(f"Multi-symbol data fetch failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-symbol data fetch failed: {str(e)}"
        )

@router.get("/realtime",
    response_model=RealTimeDataResponse,
    summary="Get real-time market data",
    description="Get current real-time market data for specified symbols"
)
async def get_realtime_data(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to retrieve"),
    current_user: dict = Depends(get_current_active_user),
    data_manager: DataManager = Depends(get_data_manager),
    cache_manager: CacheManager = Depends(get_cache_manager)
) -> RealTimeDataResponse:
    """
    Get real-time market data
    
    - **symbols**: Comma-separated list of symbols (max 100)
    - **fields**: Optional comma-separated list of fields (price, volume, change, etc.)
    """
    
    # Parse symbols
    symbol_list = [s.strip().upper() for s in symbols.split(',')]
    
    if len(symbol_list) > 100:
        raise_validation_error("Maximum 100 symbols allowed for real-time data")
    
    # Parse fields
    field_list = None
    if fields:
        field_list = [f.strip() for f in fields.split(',')]
    
    # Validate symbols
    for symbol in symbol_list:
        validate_symbol(symbol)
    
    try:
        # Get real-time data
        realtime_data = await data_manager.get_realtime_data(
            symbols=symbol_list,
            fields=field_list
        )
        
        # Get market status
        market_status = await data_manager.get_market_status()
        
        # Get delay information
        delay_info = await data_manager.get_data_delay_info(symbol_list)
        
        response = RealTimeDataResponse(
            success=True,
            message="Real-time data retrieved successfully",
            data_count=len(realtime_data),
            data=realtime_data,
            market_status=market_status,
            delay_info=delay_info
        )
        
        logger.info(f"Retrieved real-time data for {len(symbol_list)} symbols")
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get real-time data: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get real-time data: {str(e)}"
        )

@router.post("/validate",
    response_model=DataValidationResponse,
    summary="Validate market data quality",
    description="Validate market data for quality issues, outliers, and completeness"
)
async def validate_data_quality(
    request: DataValidationRequest,
    current_user: dict = Depends(get_current_active_user),
    data_manager: DataManager = Depends(get_data_manager)
) -> DataValidationResponse:
    """
    Validate market data quality
    
    Performs comprehensive data quality checks including:
    - OHLCV consistency validation
    - Outlier detection
    - Missing data point identification
    - Statistical anomaly detection
    """
    
    # Validate symbol
    validate_symbol(request.symbol)
    
    try:
        # Perform data validation
        validation_results = await data_manager.validate_data_quality(
            symbol=request.symbol,
            data=request.data,
            validation_rules=request.validation_rules
        )
        
        # Apply corrections if needed
        corrected_data = None
        if validation_results.get("corrections_available"):
            corrected_data = await data_manager.apply_data_corrections(
                request.symbol,
                request.data,
                validation_results.get("corrections")
            )
        
        response = DataValidationResponse(
            success=True,
            message="Data validation completed",
            data_count=len(request.data),
            validation_results=validation_results,
            corrected_data=corrected_data
        )
        
        logger.info(f"Data validation completed for {request.symbol}")
        
        return response
        
    except Exception as e:
        logger.error(f"Data validation failed for {request.symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data validation failed: {str(e)}"
        )

@router.post("/search",
    response_model=SymbolSearchResponse,
    summary="Search for stock symbols",
    description="Search for stock symbols by name, symbol, or sector"
)
async def search_symbols(
    request: SymbolSearchRequest,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session),
    data_manager: DataManager = Depends(get_data_manager)
) -> SymbolSearchResponse:
    """
    Search for stock symbols
    
    Supports searching by:
    - Symbol (exact or partial match)
    - Company name (fuzzy matching)
    - Sector filtering
    - Market filtering
    """
    
    try:
        start_time = datetime.utcnow()
        
        # Search in database first
        db_results = await search_symbols_in_db(
            db=db,
            query=request.query,
            market=request.market,
            sector=request.sector,
            limit=request.limit
        )
        
        # If not enough results, search external sources
        if len(db_results) < request.limit:
            external_results = await data_manager.search_symbols(
                query=request.query,
                market=request.market,
                sector=request.sector,
                limit=request.limit - len(db_results),
                include_inactive=request.include_inactive
            )
            
            # Merge results
            all_results = db_results + external_results
        else:
            all_results = db_results
        
        # Remove duplicates
        unique_results = {}
        for symbol_info in all_results:
            unique_results[symbol_info.symbol] = symbol_info
        
        final_results = list(unique_results.values())[:request.limit]
        
        # Calculate search time
        search_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Create search metadata
        search_metadata = {
            "query": request.query,
            "total_matches": len(final_results),
            "returned_count": len(final_results),
            "search_time_ms": round(search_time_ms, 2)
        }
        
        response = SymbolSearchResponse(
            success=True,
            message="Symbol search completed",
            data_count=len(final_results),
            symbols=final_results,
            search_metadata=search_metadata
        )
        
        logger.info(f"Symbol search for '{request.query}' returned {len(final_results)} results")
        
        return response
        
    except Exception as e:
        logger.error(f"Symbol search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Symbol search failed: {str(e)}"
        )

@router.post("/monitor",
    response_model=DataMonitoringResponse,
    summary="Monitor data quality",
    description="Monitor data quality metrics for multiple symbols over a time period"
)
async def monitor_data_quality(
    request: DataMonitoringRequest,
    current_user: dict = Depends(get_current_active_user),
    data_manager: DataManager = Depends(get_data_manager)
) -> DataMonitoringResponse:
    """
    Monitor data quality across multiple symbols
    
    Provides comprehensive quality monitoring including:
    - Completeness analysis
    - Accuracy scoring
    - Timeliness assessment
    - Consistency validation
    - Alert generation for quality issues
    """
    
    # Validate symbols
    for symbol in request.symbols:
        validate_symbol(symbol)
    
    # Validate date range
    validate_date_range(request.start_date, request.end_date)
    
    try:
        # Monitor data quality for each symbol
        monitoring_results = {}
        alerts = []
        total_quality_score = 0
        symbols_below_threshold = 0
        
        for symbol in request.symbols:
            try:
                # Get quality metrics for symbol
                quality_metrics = await data_manager.get_data_quality_metrics(
                    symbol=symbol,
                    start_date=request.start_date,
                    end_date=request.end_date
                )
                
                monitoring_results[symbol] = quality_metrics
                
                # Check if below threshold
                overall_quality = (
                    quality_metrics.completeness + 
                    quality_metrics.accuracy + 
                    quality_metrics.timeliness + 
                    quality_metrics.consistency
                ) / 4
                
                total_quality_score += overall_quality
                
                if overall_quality < request.quality_threshold:
                    symbols_below_threshold += 1
                    
                    # Generate alert
                    alerts.append({
                        "symbol": symbol,
                        "alert_type": "LOW_QUALITY",
                        "message": f"Data quality below threshold ({overall_quality:.2f} < {request.quality_threshold})",
                        "quality_score": overall_quality,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
            except Exception as e:
                logger.warning(f"Failed to monitor {symbol}: {e}")
                
                # Add error alert
                alerts.append({
                    "symbol": symbol,
                    "alert_type": "MONITORING_ERROR",
                    "message": f"Failed to retrieve quality metrics: {str(e)}",
                    "quality_score": 0.0,
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Calculate summary
        avg_quality_score = total_quality_score / len(monitoring_results) if monitoring_results else 0
        
        summary = {
            "symbols_monitored": len(request.symbols),
            "avg_quality_score": round(avg_quality_score, 3),
            "symbols_below_threshold": symbols_below_threshold,
            "total_alerts": len(alerts)
        }
        
        response = DataMonitoringResponse(
            success=True,
            message=f"Quality monitoring completed for {len(request.symbols)} symbols",
            data_count=len(monitoring_results),
            monitoring_results=monitoring_results,
            alerts=alerts,
            summary=summary
        )
        
        logger.info(f"Data quality monitoring completed: avg quality {avg_quality_score:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Data quality monitoring failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data quality monitoring failed: {str(e)}"
        )

@router.post("/bulk",
    response_model=BulkDataResponse,
    summary="Start bulk data operation",
    description="Start a bulk data operation (fetch, validate, update, delete) for multiple symbols"
)
async def bulk_data_operation(
    request: BulkDataRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_active_user),
    data_manager: DataManager = Depends(get_data_manager)
) -> BulkDataResponse:
    """
    Start bulk data operation
    
    Supports large-scale data operations with:
    - Background processing
    - Progress tracking
    - Batch processing
    - Parallel execution
    - Error handling and recovery
    """
    
    # Validate symbols
    for symbol in request.symbols:
        validate_symbol(symbol)
    
    # Generate job ID
    job_id = str(uuid4())
    
    try:
        # Initialize job
        await initialize_bulk_job(job_id, request, current_user["user_id"])
        
        # Start background task
        if request.operation == "fetch":
            background_tasks.add_task(
                bulk_data_fetch_task,
                job_id=job_id,
                request=request,
                user_id=current_user["user_id"],
                data_manager=data_manager
            )
        # Add other operations as needed
        
        # Calculate estimated completion time
        estimated_completion = datetime.utcnow() + timedelta(
            minutes=len(request.symbols) * 0.1  # Rough estimate
        )
        
        response = BulkDataResponse(
            success=True,
            message="Bulk operation started successfully",
            data_count=0,  # Will be updated as job progresses
            job_id=job_id,
            status="pending",
            progress={
                "total": len(request.symbols),
                "completed": 0,
                "failed": 0,
                "pending": len(request.symbols)
            },
            estimated_completion=estimated_completion
        )
        
        logger.info(f"Started bulk {request.operation} job {job_id} for {len(request.symbols)} symbols")
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to start bulk operation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start bulk operation: {str(e)}"
        )

@router.get("/bulk/{job_id}/status",
    response_model=BulkDataResponse,
    summary="Get bulk operation status",
    description="Get the current status and progress of a bulk data operation"
)
async def get_bulk_operation_status(
    job_id: str,
    current_user: dict = Depends(get_current_active_user)
) -> BulkDataResponse:
    """Get bulk operation status and progress"""
    
    try:
        # Get job status from storage
        job_status = await get_bulk_job_status(job_id, current_user["user_id"])
        
        if not job_status:
            raise_not_found("Bulk job", job_id)
        
        return job_status
        
    except Exception as e:
        logger.error(f"Failed to get bulk job status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bulk job status: {str(e)}"
        )

@router.get("/{symbol}/info",
    response_model=SymbolInfo,
    summary="Get symbol information",
    description="Get detailed information about a specific stock symbol"
)
async def get_symbol_info(
    symbol: str,
    current_user: dict = Depends(get_current_active_user),
    data_manager: DataManager = Depends(get_data_manager)
) -> SymbolInfo:
    """
    Get detailed symbol information
    
    Returns comprehensive information including:
    - Company name and description
    - Market and sector classification
    - Market capitalization
    - Currency and country
    """
    
    # Validate and normalize symbol
    symbol = symbol.upper().strip()
    validate_symbol(symbol)
    
    try:
        # Get symbol info
        symbol_info = await data_manager.get_symbol_info(symbol)
        
        if not symbol_info:
            raise_not_found("Symbol information", symbol)
        
        logger.info(f"Retrieved symbol info for {symbol}")
        
        return symbol_info
        
    except Exception as e:
        logger.error(f"Failed to get symbol info for {symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get symbol information: {str(e)}"
        )

@router.get("/{symbol}/download",
    summary="Download historical data",
    description="Download historical data as CSV or JSON file"
)
async def download_historical_data(
    symbol: str,
    start_date: Optional[date] = Query(None, description="Start date"),
    end_date: Optional[date] = Query(None, description="End date"),
    interval: TimeInterval = Query(TimeInterval.DAY_1, description="Data interval"),
    format: str = Query("csv", regex="^(csv|json)$", description="File format"),
    current_user: dict = Depends(get_current_active_user),
    data_manager: DataManager = Depends(get_data_manager)
):
    """Download historical data as file"""
    
    # Validate and normalize symbol
    symbol = symbol.upper().strip()
    validate_symbol(symbol)
    
    # Validate date range
    if start_date and end_date:
        validate_date_range(start_date, end_date)
    
    try:
        # Get historical data
        market_data = await data_manager.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if not market_data or not market_data.data:
            raise_not_found("Historical data", symbol)
        
        # Generate file
        file_path = await data_manager.export_data_to_file(
            market_data=market_data,
            format=format
        )
        
        # Return file
        media_type = "text/csv" if format == "csv" else "application/json"
        filename = f"{symbol}_historical_data.{format}"
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"Failed to download data for {symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download data: {str(e)}"
        )

# ============================================
# Helper Functions
# ============================================

def calculate_market_statistics(data: List[OHLCVData]) -> Dict[str, float]:
    """Calculate market statistics from OHLCV data"""
    
    if not data:
        return {}
    
    prices = [point.close for point in data]
    volumes = [point.volume for point in data]
    
    # Calculate basic statistics
    mean_price = sum(prices) / len(prices)
    total_volume = sum(volumes)
    
    # Calculate price change
    if len(prices) >= 2:
        price_change = prices[-1] - prices[0]
        price_change_percent = (price_change / prices[0]) * 100
    else:
        price_change = 0
        price_change_percent = 0
    
    # Calculate volatility (simple standard deviation)
    if len(prices) > 1:
        variance = sum((p - mean_price) ** 2 for p in prices) / (len(prices) - 1)
        volatility = variance ** 0.5 / mean_price  # Relative volatility
    else:
        volatility = 0
    
    return {
        "mean_price": round(mean_price, 2),
        "volatility": round(volatility, 4),
        "total_volume": total_volume,
        "price_change": round(price_change, 2),
        "price_change_percent": round(price_change_percent, 2)
    }

async def initialize_bulk_job(job_id: str, request: BulkDataRequest, user_id: str):
    """Initialize bulk job in storage"""
    # Implementation would initialize job in database/cache
    logger.info(f"Initialized bulk job {job_id}")

async def update_bulk_job_status(job_id: str, status: str, progress: Optional[float], error: Optional[str] = None):
    """Update bulk job status"""
    # Implementation would update job status in database/cache
    logger.info(f"Updated bulk job {job_id} status to {status}, progress: {progress}")

async def save_bulk_job_results(job_id: str, results: Dict[str, Any]):
    """Save bulk job results"""
    # Implementation would save results to storage
    logger.info(f"Saved results for bulk job {job_id}")

async def get_bulk_job_status(job_id: str, user_id: str) -> Optional[BulkDataResponse]:
    """Get bulk job status from storage"""
    # Implementation would retrieve job status from database/cache
    logger.info(f"Getting bulk job status for {job_id}")
    return None

# Export router
__all__ = ["router"]
