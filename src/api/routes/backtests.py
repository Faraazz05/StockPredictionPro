# ============================================
# StockPredictionPro - src/api/routes/backtests.py
# Comprehensive backtesting routes for FastAPI with strategy testing, performance analytics, and risk management
# ============================================

import asyncio
import uuid
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Union
import logging

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update, func
from sqlalchemy.orm import selectinload

from ..dependencies import (
    get_async_session,
    get_current_active_user,
    get_data_manager,
    get_backtest_engine,
    get_portfolio_manager,
    standard_rate_limit,
    validate_symbol,
    validate_date_range
)
from ..schemas.data_schemas import SymbolStr, TimestampMixin
from ..schemas.error_schemas import ErrorResponseFactory, ErrorResponse
from ..exceptions import (
    raise_not_found,
    raise_validation_error,
    raise_forbidden,
    BacktestException
)
from ...evaluation.backtesting.engine import BacktestEngine
from ...trading.portfolio import PortfolioManager
from ...data.manager import DataManager
from ...utils.logger import get_logger

logger = get_logger('api.routes.backtests')

# ============================================
# Router Configuration
# ============================================

router = APIRouter(
    prefix="/api/v1/backtests",
    tags=["Backtests"],
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
# Backtest Schemas
# ============================================

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum

class BacktestStatus(str, Enum):
    """Backtest execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BacktestStrategy(str, Enum):
    """Available backtest strategies"""
    BUY_AND_HOLD = "buy_and_hold"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    PAIRS_TRADING = "pairs_trading"
    TREND_FOLLOWING = "trend_following"
    CUSTOM = "custom"

class RiskManagementType(str, Enum):
    """Risk management methods"""
    FIXED_STOP_LOSS = "fixed_stop_loss"
    ATR_STOP_LOSS = "atr_stop_loss"
    PERCENTAGE_STOP_LOSS = "percentage_stop_loss"
    TRAILING_STOP = "trailing_stop"
    VOLATILITY_TARGET = "volatility_target"

class BacktestRequest(BaseModel):
    """Backtest configuration request"""
    
    name: str = Field(
        description="Backtest name",
        max_length=100,
        examples=["AAPL Momentum Strategy Q1 2023"]
    )
    
    symbols: List[SymbolStr] = Field(
        description="List of symbols to backtest",
        min_length=1,
        max_length=50,
        examples=[["AAPL", "MSFT", "GOOGL"]]
    )
    
    strategy: BacktestStrategy = Field(
        description="Trading strategy to test"
    )
    
    start_date: date = Field(
        description="Backtest start date"
    )
    
    end_date: date = Field(
        description="Backtest end date"
    )
    
    initial_capital: float = Field(
        gt=0,
        le=100_000_000,
        description="Initial capital for backtest",
        examples=[100000.0]
    )
    
    position_sizing: Dict[str, Any] = Field(
        description="Position sizing configuration",
        examples=[{
            "method": "equal_weight",
            "max_position_size": 0.1,
            "rebalance_frequency": "monthly"
        }]
    )
    
    risk_management: Dict[str, Any] = Field(
        description="Risk management configuration",
        examples=[{
            "stop_loss_type": "percentage_stop_loss",
            "stop_loss_threshold": 0.05,
            "take_profit": 0.15,
            "max_drawdown": 0.20
        }]
    )
    
    transaction_costs: Dict[str, float] = Field(
        default={
            "commission": 0.001,  # 0.1%
            "slippage": 0.0005,   # 0.05%
            "market_impact": 0.0002  # 0.02%
        },
        description="Transaction cost configuration"
    )
    
    benchmark_symbol: Optional[str] = Field(
        default="SPY",
        description="Benchmark for comparison"
    )
    
    strategy_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Strategy-specific parameters",
        examples=[{
            "momentum_lookback": 20,
            "momentum_threshold": 0.02,
            "rebalance_frequency": "weekly"
        }]
    )
    
    include_dividends: bool = Field(
        default=True,
        description="Include dividend payments"
    )
    
    frequency: str = Field(
        default="daily",
        description="Rebalancing frequency",
        examples=["daily", "weekly", "monthly"]
    )
    
    @field_validator('symbols', mode='before')
    @classmethod
    def symbols_to_upper(cls, v: List[str]) -> List[str]:
        """Convert symbols to uppercase"""
        return [s.upper().strip() for s in v] if v else v
    
    @field_validator('benchmark_symbol', mode='before')
    @classmethod
    def benchmark_to_upper(cls, v: Optional[str]) -> Optional[str]:
        """Convert benchmark symbol to uppercase"""
        return v.upper().strip() if v else v
    
    @model_validator(mode='after')
    def validate_backtest_config(self):
        """Validate backtest configuration"""
        
        # Date range validation
        if self.start_date >= self.end_date:
            raise ValueError('Start date must be before end date')
        
        # Limit backtest period to reasonable range
        max_years = 10
        if (self.end_date - self.start_date).days > (365 * max_years):
            raise ValueError(f'Backtest period cannot exceed {max_years} years')
        
        # Don't allow future end dates
        if self.end_date > date.today():
            raise ValueError('End date cannot be in the future')
        
        # Validate position sizing
        position_sizing = self.position_sizing
        if 'max_position_size' in position_sizing:
            if not 0 < position_sizing['max_position_size'] <= 1:
                raise ValueError('max_position_size must be between 0 and 1')
        
        return self
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "AAPL Momentum Strategy",
                    "symbols": ["AAPL"],
                    "strategy": "momentum",
                    "start_date": "2022-01-01",
                    "end_date": "2023-12-31",
                    "initial_capital": 100000.0,
                    "position_sizing": {
                        "method": "equal_weight",
                        "max_position_size": 0.2
                    },
                    "risk_management": {
                        "stop_loss_type": "percentage_stop_loss",
                        "stop_loss_threshold": 0.05,
                        "take_profit": 0.15
                    },
                    "strategy_parameters": {
                        "momentum_lookback": 20,
                        "momentum_threshold": 0.02
                    }
                }
            ]
        }
    )

class BacktestMetrics(BaseModel):
    """Backtest performance metrics"""
    
    # Return metrics
    total_return: float = Field(description="Total return percentage")
    annualized_return: float = Field(description="Annualized return percentage")
    volatility: float = Field(description="Annualized volatility")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    sortino_ratio: float = Field(description="Sortino ratio")
    calmar_ratio: float = Field(description="Calmar ratio")
    
    # Risk metrics
    max_drawdown: float = Field(description="Maximum drawdown")
    var_95: float = Field(description="Value at Risk (95%)")
    cvar_95: float = Field(description="Conditional Value at Risk (95%)")
    
    # Trade metrics
    total_trades: int = Field(description="Total number of trades")
    win_rate: float = Field(description="Win rate percentage")
    profit_factor: float = Field(description="Profit factor")
    avg_trade_return: float = Field(description="Average trade return")
    
    # Benchmark comparison
    benchmark_return: Optional[float] = Field(default=None, description="Benchmark return")
    alpha: Optional[float] = Field(default=None, description="Alpha vs benchmark")
    beta: Optional[float] = Field(default=None, description="Beta vs benchmark")
    information_ratio: Optional[float] = Field(default=None, description="Information ratio")
    
    # Additional metrics
    kelly_criterion: Optional[float] = Field(default=None, description="Kelly criterion optimal bet size")
    expectancy: Optional[float] = Field(default=None, description="Trade expectancy")

class BacktestResult(BaseModel, TimestampMixin):
    """Backtest result"""
    
    backtest_id: str = Field(description="Unique backtest identifier")
    name: str = Field(description="Backtest name")
    user_id: str = Field(description="User who created the backtest")
    status: BacktestStatus = Field(description="Backtest status")
    
    # Configuration
    symbols: List[str] = Field(description="Backtested symbols")
    strategy: BacktestStrategy = Field(description="Strategy used")
    start_date: date = Field(description="Backtest start date")
    end_date: date = Field(description="Backtest end date")
    initial_capital: float = Field(description="Initial capital")
    
    # Results
    final_portfolio_value: Optional[float] = Field(default=None, description="Final portfolio value")
    metrics: Optional[BacktestMetrics] = Field(default=None, description="Performance metrics")
    
    # Execution info
    execution_time_seconds: Optional[float] = Field(default=None, description="Execution time")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # File paths for detailed results
    equity_curve_path: Optional[str] = Field(default=None, description="Equity curve data file")
    trades_log_path: Optional[str] = Field(default=None, description="Trades log file")
    report_path: Optional[str] = Field(default=None, description="HTML report file")

class BacktestListResponse(BaseModel):
    """Backtest list response"""
    
    total: int = Field(description="Total number of backtests")
    backtests: List[BacktestResult] = Field(description="List of backtests")
    
    summary: Dict[str, Any] = Field(
        description="Summary statistics",
        examples=[{
            "total_backtests": 25,
            "by_status": {
                "completed": 20,
                "running": 3,
                "failed": 2
            },
            "by_strategy": {
                "momentum": 10,
                "mean_reversion": 8,
                "pairs_trading": 7
            }
        }]
    )

class BacktestCompareRequest(BaseModel):
    """Request to compare multiple backtests"""
    
    backtest_ids: List[str] = Field(
        min_length=2,
        max_length=10,
        description="Backtest IDs to compare"
    )
    
    metrics: List[str] = Field(
        default=["total_return", "sharpe_ratio", "max_drawdown"],
        description="Metrics to compare"
    )

# ============================================
# Database Helper Functions
# ============================================

async def save_backtest_to_db(db: AsyncSession, backtest: BacktestResult):
    """Save backtest to database"""
    try:
        # Convert to ORM model (assuming Backtest ORM model exists)
        # backtest_orm = BacktestORM(**backtest.model_dump())
        # db.add(backtest_orm)
        # await db.commit()
        # await db.refresh(backtest_orm)
        
        # For now, placeholder implementation
        logger.info(f"Saving backtest {backtest.backtest_id} to database")
        
    except Exception as e:
        logger.error(f"Failed to save backtest: {e}")
        await db.rollback()
        raise

async def get_backtest_from_db(db: AsyncSession, backtest_id: str, user_id: str) -> Optional[BacktestResult]:
    """Get backtest from database"""
    try:
        # Query database for backtest
        # result = await db.execute(
        #     select(BacktestORM).where(
        #         BacktestORM.id == backtest_id,
        #         BacktestORM.user_id == user_id
        #     )
        # )
        # backtest_orm = result.scalars().first()
        
        # if backtest_orm:
        #     return BacktestResult(**backtest_orm.__dict__)
        
        # For now, placeholder implementation
        logger.info(f"Getting backtest {backtest_id} for user {user_id}")
        return None
        
    except Exception as e:
        logger.error(f"Failed to get backtest: {e}")
        raise

async def get_user_backtests(
    db: AsyncSession, 
    user_id: str, 
    status: Optional[BacktestStatus], 
    strategy: Optional[BacktestStrategy], 
    limit: int, 
    offset: int
) -> List[BacktestResult]:
    """Get user backtests with filtering"""
    try:
        # Build query with filters
        # query = select(BacktestORM).where(BacktestORM.user_id == user_id)
        
        # if status:
        #     query = query.where(BacktestORM.status == status)
        
        # if strategy:
        #     query = query.where(BacktestORM.strategy == strategy)
        
        # query = query.offset(offset).limit(limit).order_by(BacktestORM.created_at.desc())
        
        # result = await db.execute(query)
        # backtests_orm = result.scalars().all()
        
        # return [BacktestResult(**bt.__dict__) for bt in backtests_orm]
        
        # For now, placeholder implementation
        logger.info(f"Getting backtests for user {user_id}")
        return []
        
    except Exception as e:
        logger.error(f"Failed to get user backtests: {e}")
        raise

async def count_user_backtests(
    db: AsyncSession, 
    user_id: str, 
    status: Optional[BacktestStatus],
    strategy: Optional[BacktestStrategy]
) -> int:
    """Count user backtests"""
    try:
        # Build count query
        # query = select(func.count(BacktestORM.id)).where(BacktestORM.user_id == user_id)
        
        # if status:
        #     query = query.where(BacktestORM.status == status)
        
        # if strategy:
        #     query = query.where(BacktestORM.strategy == strategy)
        
        # result = await db.execute(query)
        # return result.scalar() or 0
        
        # For now, placeholder implementation
        logger.info(f"Counting backtests for user {user_id}")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to count user backtests: {e}")
        raise

async def get_backtest_summary(db: AsyncSession, user_id: str) -> Dict[str, Any]:
    """Get backtest summary statistics"""
    try:
        # Calculate summary statistics
        # total_result = await db.execute(
        #     select(func.count(BacktestORM.id)).where(BacktestORM.user_id == user_id)
        # )
        # total = total_result.scalar() or 0
        
        # status_result = await db.execute(
        #     select(BacktestORM.status, func.count(BacktestORM.id))
        #     .where(BacktestORM.user_id == user_id)
        #     .group_by(BacktestORM.status)
        # )
        # by_status = {status: count for status, count in status_result.all()}
        
        # strategy_result = await db.execute(
        #     select(BacktestORM.strategy, func.count(BacktestORM.id))
        #     .where(BacktestORM.user_id == user_id)
        #     .group_by(BacktestORM.strategy)
        # )
        # by_strategy = {strategy: count for strategy, count in strategy_result.all()}
        
        # return {
        #     "total_backtests": total,
        #     "by_status": by_status,
        #     "by_strategy": by_strategy
        # }
        
        # For now, placeholder implementation
        return {
            "total_backtests": 0,
            "by_status": {},
            "by_strategy": {}
        }
        
    except Exception as e:
        logger.error(f"Failed to get backtest summary: {e}")
        raise

async def delete_backtest_from_db(db: AsyncSession, backtest_id: str):
    """Delete backtest from database"""
    try:
        # Delete backtest record
        # await db.execute(
        #     delete(BacktestORM).where(BacktestORM.id == backtest_id)
        # )
        # await db.commit()
        
        # For now, placeholder implementation
        logger.info(f"Deleting backtest {backtest_id} from database")
        
    except Exception as e:
        logger.error(f"Failed to delete backtest: {e}")
        await db.rollback()
        raise

async def update_backtest_in_db(db: AsyncSession, backtest: BacktestResult):
    """Update backtest in database"""
    try:
        # Update backtest record
        # await db.execute(
        #     update(BacktestORM)
        #     .where(BacktestORM.id == backtest.backtest_id)
        #     .values(**backtest.model_dump(exclude={'backtest_id', 'created_at'}))
        # )
        # await db.commit()
        
        # For now, placeholder implementation
        logger.info(f"Updating backtest {backtest.backtest_id} in database")
        
    except Exception as e:
        logger.error(f"Failed to update backtest: {e}")
        await db.rollback()
        raise

# ============================================
# Background Tasks
# ============================================

async def run_backtest_task(
    backtest_id: str,
    config: BacktestRequest,
    user_id: str,
    backtest_engine: BacktestEngine,
    data_manager: DataManager
):
    """Background task to run backtest"""
    
    try:
        logger.info(f"Starting backtest {backtest_id}")
        
        # Update status to running
        await update_backtest_status(backtest_id, BacktestStatus.RUNNING)
        
        # Prepare data
        market_data = {}
        for symbol in config.symbols:
            data = await data_manager.get_historical_data(
                symbol=symbol,
                start_date=config.start_date,
                end_date=config.end_date
            )
            market_data[symbol] = data
        
        # Run backtest
        start_time = datetime.utcnow()
        
        results = await backtest_engine.run_backtest(
            symbols=config.symbols,
            strategy=config.strategy.value,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            market_data=market_data,
            position_sizing=config.position_sizing,
            risk_management=config.risk_management,
            transaction_costs=config.transaction_costs,
            strategy_parameters=config.strategy_parameters or {}
        )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Calculate metrics
        metrics = calculate_backtest_metrics(results, config.benchmark_symbol)
        
        # Save results
        await save_backtest_results(
            backtest_id=backtest_id,
            results=results,
            metrics=metrics,
            execution_time=execution_time
        )
        
        # Update status to completed
        await update_backtest_status(backtest_id, BacktestStatus.COMPLETED)
        
        logger.info(f"Backtest {backtest_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Backtest {backtest_id} failed: {e}", exc_info=True)
        
        # Update status to failed
        await update_backtest_status(
            backtest_id, 
            BacktestStatus.FAILED, 
            error_message=str(e)
        )

# ============================================
# Route Handlers
# ============================================

@router.post("/", 
    response_model=BacktestResult,
    status_code=status.HTTP_201_CREATED,
    summary="Create and start a new backtest",
    description="Submit a new backtesting job with strategy configuration"
)
async def create_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session),
    backtest_engine: BacktestEngine = Depends(get_backtest_engine),
    data_manager: DataManager = Depends(get_data_manager)
) -> BacktestResult:
    """
    Create and start a new backtest
    
    - **name**: Descriptive name for the backtest
    - **symbols**: List of symbols to backtest (max 50)
    - **strategy**: Trading strategy to test
    - **start_date**: Backtest start date
    - **end_date**: Backtest end date
    - **initial_capital**: Starting capital amount
    - **position_sizing**: Position sizing configuration
    - **risk_management**: Risk management rules
    - **strategy_parameters**: Strategy-specific parameters
    """
    
    # Validate symbols
    for symbol in request.symbols:
        validate_symbol(symbol)
    
    # Validate date range
    validate_date_range(request.start_date, request.end_date)
    
    # Generate backtest ID
    backtest_id = str(uuid.uuid4())
    
    # Create backtest record
    backtest = BacktestResult(
        backtest_id=backtest_id,
        name=request.name,
        user_id=current_user["user_id"],
        status=BacktestStatus.PENDING,
        symbols=request.symbols,
        strategy=request.strategy,
        start_date=request.start_date,
        end_date=request.end_date,
        initial_capital=request.initial_capital
    )
    
    # Save to database
    await save_backtest_to_db(db, backtest)
    
    # Start background task
    background_tasks.add_task(
        run_backtest_task,
        backtest_id=backtest_id,
        config=request,
        user_id=current_user["user_id"],
        backtest_engine=backtest_engine,
        data_manager=data_manager
    )
    
    logger.info(f"Created backtest {backtest_id} for user {current_user['user_id']}")
    
    return backtest

@router.get("/",
    response_model=BacktestListResponse,
    summary="List user backtests",
    description="Get list of backtests for the current user with filtering options"
)
async def list_backtests(
    status: Optional[BacktestStatus] = Query(None, description="Filter by status"),
    strategy: Optional[BacktestStrategy] = Query(None, description="Filter by strategy"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Results offset for pagination"),
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
) -> BacktestListResponse:
    """
    List backtests for the current user
    
    Supports filtering by status and strategy with pagination.
    """
    
    backtests = await get_user_backtests(
        db=db,
        user_id=current_user["user_id"],
        status=status,
        strategy=strategy,
        limit=limit,
        offset=offset
    )
    
    total = await count_user_backtests(
        db=db,
        user_id=current_user["user_id"],
        status=status,
        strategy=strategy
    )
    
    # Generate summary statistics
    summary = await get_backtest_summary(db, current_user["user_id"])
    
    return BacktestListResponse(
        total=total,
        backtests=backtests,
        summary=summary
    )

@router.get("/{backtest_id}",
    response_model=BacktestResult,
    summary="Get backtest details",
    description="Retrieve detailed information about a specific backtest"
)
async def get_backtest(
    backtest_id: str,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
) -> BacktestResult:
    """Get detailed backtest information"""
    
    backtest = await get_backtest_from_db(db, backtest_id, current_user["user_id"])
    
    if not backtest:
        raise_not_found("Backtest", backtest_id)
    
    return backtest

@router.delete("/{backtest_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete backtest",
    description="Delete a backtest (only if pending or failed)"
)
async def delete_backtest(
    backtest_id: str,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Delete a backtest"""
    
    backtest = await get_backtest_from_db(db, backtest_id, current_user["user_id"])
    
    if not backtest:
        raise_not_found("Backtest", backtest_id)
    
    # Only allow deletion of pending or failed backtests
    if backtest.status in [BacktestStatus.RUNNING, BacktestStatus.COMPLETED]:
        raise_forbidden("Cannot delete running or completed backtests")
    
    await delete_backtest_from_db(db, backtest_id)
    
    logger.info(f"Deleted backtest {backtest_id} for user {current_user['user_id']}")

@router.post("/{backtest_id}/cancel",
    response_model=BacktestResult,
    summary="Cancel running backtest",
    description="Cancel a currently running backtest"
)
async def cancel_backtest(
    backtest_id: str,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
) -> BacktestResult:
    """Cancel a running backtest"""
    
    backtest = await get_backtest_from_db(db, backtest_id, current_user["user_id"])
    
    if not backtest:
        raise_not_found("Backtest", backtest_id)
    
    if backtest.status != BacktestStatus.RUNNING:
        raise_validation_error("Can only cancel running backtests")
    
    # Cancel the backtest
    await cancel_backtest_job(backtest_id)
    
    # Update status
    backtest.status = BacktestStatus.CANCELLED
    await update_backtest_in_db(db, backtest)
    
    logger.info(f"Cancelled backtest {backtest_id} for user {current_user['user_id']}")
    
    return backtest

@router.post("/compare",
    summary="Compare multiple backtests",
    description="Compare performance metrics across multiple backtests"
)
async def compare_backtests(
    request: BacktestCompareRequest,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Compare multiple backtests"""
    
    # Verify user owns all backtests
    backtests = []
    for backtest_id in request.backtest_ids:
        backtest = await get_backtest_from_db(db, backtest_id, current_user["user_id"])
        if not backtest:
            raise_not_found("Backtest", backtest_id)
        
        if backtest.status != BacktestStatus.COMPLETED:
            raise_validation_error(f"Backtest {backtest_id} is not completed")
        
        backtests.append(backtest)
    
    # Generate comparison
    comparison = await generate_backtest_comparison(backtests, request.metrics)
    
    return comparison

@router.get("/{backtest_id}/report",
    summary="Get backtest report",
    description="Download detailed backtest report (HTML/PDF)"
)
async def download_backtest_report(
    backtest_id: str,
    format: str = Query("html", regex="^(html|pdf)$", description="Report format"),
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Download detailed backtest report"""
    
    backtest = await get_backtest_from_db(db, backtest_id, current_user["user_id"])
    
    if not backtest:
        raise_not_found("Backtest", backtest_id)
    
    if backtest.status != BacktestStatus.COMPLETED:
        raise_validation_error("Report only available for completed backtests")
    
    # Generate and return report file
    report_path = await generate_backtest_report(backtest, format)
    
    return FileResponse(
        path=report_path,
        media_type="application/octet-stream",
        filename=f"backtest_{backtest_id}_report.{format}"
    )

@router.get("/{backtest_id}/trades",
    summary="Get backtest trades log",
    description="Download detailed trades log for the backtest"
)
async def download_trades_log(
    backtest_id: str,
    format: str = Query("csv", regex="^(csv|json)$", description="File format"),
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Download backtest trades log"""
    
    backtest = await get_backtest_from_db(db, backtest_id, current_user["user_id"])
    
    if not backtest:
        raise_not_found("Backtest", backtest_id)
    
    if backtest.status != BacktestStatus.COMPLETED:
        raise_validation_error("Trades log only available for completed backtests")
    
    # Get trades log file
    if not backtest.trades_log_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trades log not found"
        )
    
    return FileResponse(
        path=backtest.trades_log_path,
        media_type="application/octet-stream",
        filename=f"backtest_{backtest_id}_trades.{format}"
    )

# ============================================
# Additional Helper Functions
# ============================================

async def update_backtest_status(backtest_id: str, status: BacktestStatus, error_message: Optional[str] = None):
    """Update backtest status"""
    # Implementation would update database status
    logger.info(f"Updating backtest {backtest_id} status to {status}")

async def save_backtest_results(backtest_id: str, results: Any, metrics: BacktestMetrics, execution_time: float):
    """Save backtest results"""
    # Implementation would save results to database and files
    logger.info(f"Saving results for backtest {backtest_id}")

def calculate_backtest_metrics(results: Any, benchmark_symbol: Optional[str]) -> BacktestMetrics:
    """Calculate backtest performance metrics"""
    # Implementation would calculate all metrics
    return BacktestMetrics(
        total_return=0.0,
        annualized_return=0.0,
        volatility=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        max_drawdown=0.0,
        var_95=0.0,
        cvar_95=0.0,
        total_trades=0,
        win_rate=0.0,
        profit_factor=0.0,
        avg_trade_return=0.0
    )

async def cancel_backtest_job(backtest_id: str):
    """Cancel running backtest job"""
    # Implementation would cancel background task
    logger.info(f"Cancelling backtest job {backtest_id}")

async def generate_backtest_comparison(backtests: List[BacktestResult], metrics: List[str]) -> Dict[str, Any]:
    """Generate backtest comparison"""
    # Implementation would compare backtests
    return {"comparison": "placeholder"}

async def generate_backtest_report(backtest: BacktestResult, format: str) -> str:
    """Generate backtest report file"""
    # Implementation would generate report
    return f"/tmp/backtest_{backtest.backtest_id}_report.{format}"

# Export router
__all__ = ["router"]
