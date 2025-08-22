# ============================================
# StockPredictionPro - src/data/manager.py
# Comprehensive data management orchestration system
# ============================================

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import threading
import queue
import time

from ..utils.exceptions import (
    DataFetchError, DataValidationError, BusinessLogicError,
    ExternalAPIError, RateLimitError
)
from ..utils.logger import get_logger
from ..utils.timing import Timer, time_it
from ..utils.config_loader import get
from ..utils.governance import governance, log_governance_event

# Import data components
from .cache import FinancialDataCache, get_cache
from .fetchers.base_fetcher import DataRequest, DataResponse, DataType
from .fetchers.yahoo_finance import YahooFinanceFetcher
from .fetchers.alpha_vantage import AlphaVantageFetcher
from .fetchers.polygon import PolygonFetcher
from .fetchers.fred import FREDFetcher
from .fetchers.quandl import QuandlFetcher
from .processors.cleaner import FinancialDataCleaner, create_data_cleaner
from .processors.transformer import FinancialTransformationPipeline, create_financial_pipeline
from .processors.splitter import FinancialTimeSeriesSplitter, create_financial_splitter
from .processors.resampler import FinancialDataResampler, create_resampler
from .validators import DataQualityValidator

logger = get_logger('data.manager')

# ============================================
# Data Management Enums and Types
# ============================================

class DataSource(Enum):
    """Available data sources"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage" 
    POLYGON = "polygon"
    FRED = "fred"
    QUANDL = "quandl"

class ProcessingStage(Enum):
    """Data processing stages"""
    RAW = "raw"
    CLEANED = "cleaned"
    TRANSFORMED = "transformed"
    SPLIT = "split"
    READY = "ready"

@dataclass
class DataPipeline:
    """
    Data processing pipeline configuration
    """
    name: str
    cleaning_enabled: bool = True
    cleaning_mode: str = 'comprehensive'
    transformation_enabled: bool = True
    transformation_type: str = 'comprehensive'
    splitting_enabled: bool = True
    splitting_type: str = 'research'
    resampling_enabled: bool = False
    target_frequency: Optional[str] = None
    validation_enabled: bool = True
    caching_enabled: bool = True
    cache_ttl: int = 3600

@dataclass
class DataJob:
    """
    Data processing job specification
    """
    job_id: str
    symbols: List[str]
    data_types: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    pipeline: Optional[DataPipeline] = None
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    status: str = 'pending'
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ============================================
# Main Data Manager
# ============================================

class FinancialDataManager:
    """
    Comprehensive financial data management system
    
    Features:
    - Multi-source data fetching with failover
    - Intelligent caching and storage
    - Data processing pipelines
    - Quality assurance and validation
    - Parallel processing and job queues
    - Real-time monitoring and analytics
    """
    
    def __init__(self, 
                 cache_instance: Optional[FinancialDataCache] = None,
                 max_workers: int = 4,
                 default_pipeline: Optional[DataPipeline] = None):
        """
        Initialize financial data manager
        
        Args:
            cache_instance: Cache instance to use
            max_workers: Maximum worker threads
            default_pipeline: Default processing pipeline
        """
        self.cache = cache_instance or get_cache()
        self.max_workers = max_workers
        self.default_pipeline = default_pipeline or self._create_default_pipeline()
        
        # Initialize data sources
        self.data_sources = self._initialize_data_sources()
        
        # Initialize processors
        self.processors = self._initialize_processors()
        
        # Job management
        self.job_queue = queue.PriorityQueue()
        self.active_jobs: Dict[str, DataJob] = {}
        self.completed_jobs: Dict[str, DataJob] = {}
        self.job_lock = threading.RLock()
        
        # Worker thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Statistics and monitoring
        self.stats = {
            'jobs_submitted': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'total_symbols_processed': 0,
            'total_data_points_fetched': 0,
            'start_time': datetime.now()
        }
        
        # Data quality tracking
        self.quality_validator = DataQualityValidator()
        
        logger.info(f"Financial Data Manager initialized with {len(self.data_sources)} sources")
    
    def _create_default_pipeline(self) -> DataPipeline:
        """Create default data processing pipeline"""
        return DataPipeline(
            name='default',
            cleaning_enabled=True,
            cleaning_mode='comprehensive',
            transformation_enabled=True,
            transformation_type='comprehensive',
            splitting_enabled=True,
            splitting_type='research',
            validation_enabled=True,
            caching_enabled=True,
            cache_ttl=3600
        )
    
    def _initialize_data_sources(self) -> Dict[str, Any]:
        """Initialize available data sources"""
        sources = {}
        
        # Yahoo Finance (always available)
        try:
            sources[DataSource.YAHOO_FINANCE.value] = YahooFinanceFetcher()
            logger.info("✅ Yahoo Finance fetcher initialized")
        except Exception as e:
            logger.warning(f"Yahoo Finance fetcher initialization failed: {e}")
        
        # Alpha Vantage (requires API key)
        try:
            sources[DataSource.ALPHA_VANTAGE.value] = AlphaVantageFetcher()
            logger.info("✅ Alpha Vantage fetcher initialized")
        except Exception as e:
            logger.warning(f"Alpha Vantage fetcher initialization failed: {e}")
        
        # Polygon (requires API key)
        try:
            sources[DataSource.POLYGON.value] = PolygonFetcher()  
            logger.info("✅ Polygon fetcher initialized")
        except Exception as e:
            logger.warning(f"Polygon fetcher initialization failed: {e}")
        
        # FRED (requires API key)
        try:
            sources[DataSource.FRED.value] = FREDFetcher()
            logger.info("✅ FRED fetcher initialized")
        except Exception as e:
            logger.warning(f"FRED fetcher initialization failed: {e}")
        
        # Quandl (requires API key)
        try:
            sources[DataSource.QUANDL.value] = QuandlFetcher()
            logger.info("✅ Quandl fetcher initialized")
        except Exception as e:
            logger.warning(f"Quandl fetcher initialization failed: {e}")
        
        return sources
    
    def _initialize_processors(self) -> Dict[str, Any]:
        """Initialize data processors"""
        return {
            'cleaner': create_data_cleaner('comprehensive'),
            'transformer': create_financial_pipeline('comprehensive'),
            'splitter': create_financial_splitter('research'),
            'resampler': create_resampler('standard')
        }
    
    @time_it("data_manager_fetch")
    def fetch_data(self, 
                   symbols: Union[str, List[str]],
                   data_types: Union[str, List[str]] = 'stock_data',
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   sources: Optional[List[str]] = None,
                   pipeline: Optional[DataPipeline] = None,
                   use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch and process financial data
        
        Args:
            symbols: Stock symbols to fetch
            data_types: Types of data to fetch
            start_date: Start date for data
            end_date: End date for data  
            sources: Preferred data sources (ordered by preference)
            pipeline: Processing pipeline to use
            use_cache: Whether to use caching
            
        Returns:
            Dictionary mapping symbols to processed DataFrames
        """
        # Normalize inputs
        if isinstance(symbols, str):
            symbols = [symbols]
        if isinstance(data_types, str):
            data_types = [data_types]
        
        pipeline = pipeline or self.default_pipeline
        
        logger.info(f"Fetching data for {len(symbols)} symbols: {symbols}")
        
        # Create data job
        job_id = f"fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(symbols))}"
        job = DataJob(
            job_id=job_id,
            symbols=symbols,
            data_types=data_types,
            start_date=start_date,
            end_date=end_date,
            pipeline=pipeline
        )
        
        try:
            # Execute job synchronously
            result = self._execute_data_job(job, use_cache=use_cache, sources=sources)
            
            # Log governance event
            log_governance_event(
                event_type='data_fetch',
                action='fetch_data',
                resource=f"symbols:{','.join(symbols)}",
                details={
                    'symbols_count': len(symbols),
                    'data_types': data_types,
                    'date_range': f"{start_date} to {end_date}" if start_date and end_date else None,
                    'pipeline': pipeline.name,
                    'success': True
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Data fetch failed for job {job_id}: {e}")
            
            # Log failure
            log_governance_event(
                event_type='data_fetch',
                action='fetch_data_failed',
                resource=f"symbols:{','.join(symbols)}",
                details={
                    'error': str(e),
                    'symbols_count': len(symbols),
                    'data_types': data_types
                },
                severity='error'
            )
            
            raise DataFetchError(f"Data fetch failed: {str(e)}", context={'job_id': job_id})
    
    def _execute_data_job(self, job: DataJob, use_cache: bool = True, 
                         sources: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Execute a data job"""
        
        with self.job_lock:
            self.active_jobs[job.job_id] = job
            job.status = 'running'
        
        try:
            all_data = {}
            total_steps = len(job.symbols) * len(job.data_types)
            completed_steps = 0
            
            # Process each symbol and data type combination
            for symbol in job.symbols:
                for data_type in job.data_types:
                    
                    # Check cache first
                    if use_cache and job.pipeline.caching_enabled:
                        cached_data = self._get_cached_data(symbol, data_type, job.start_date, job.end_date)
                        if cached_data is not None:
                            logger.debug(f"Cache hit for {symbol}:{data_type}")
                            all_data[f"{symbol}_{data_type}"] = cached_data
                            completed_steps += 1
                            job.progress = completed_steps / total_steps
                            continue
                    
                    # Fetch raw data
                    raw_data = self._fetch_raw_data(symbol, data_type, job.start_date, job.end_date, sources)
                    
                    if raw_data is not None:
                        # Process data through pipeline
                        processed_data = self._process_data_through_pipeline(raw_data, symbol, job.pipeline)
                        
                        # Cache processed data
                        if use_cache and job.pipeline.caching_enabled and processed_data is not None:
                            self._cache_processed_data(processed_data, symbol, data_type, 
                                                     job.start_date, job.end_date, job.pipeline.cache_ttl)
                        
                        if processed_data is not None:
                            all_data[f"{symbol}_{data_type}"] = processed_data
                    
                    completed_steps += 1
                    job.progress = completed_steps / total_steps
            
            # Update job status
            job.status = 'completed'
            job.result = all_data
            
            # Move to completed jobs
            with self.job_lock:
                self.active_jobs.pop(job.job_id, None)
                self.completed_jobs[job.job_id] = job
                self.stats['jobs_completed'] += 1
                self.stats['total_symbols_processed'] += len(job.symbols)
            
            logger.info(f"Data job {job.job_id} completed successfully with {len(all_data)} datasets")
            return all_data
            
        except Exception as e:
            # Update job with error
            job.status = 'failed'
            job.error = str(e)
            
            with self.job_lock:
                self.active_jobs.pop(job.job_id, None)
                self.completed_jobs[job.job_id] = job
                self.stats['jobs_failed'] += 1
            
            raise
    
    def _fetch_raw_data(self, symbol: str, data_type: str, 
                       start_date: Optional[datetime], end_date: Optional[datetime],
                       preferred_sources: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Fetch raw data with source failover"""
        
        # Determine source priority
        if preferred_sources:
            source_order = preferred_sources
        else:
            # Default source priority
            source_order = [
                DataSource.YAHOO_FINANCE.value,
                DataSource.ALPHA_VANTAGE.value,
                DataSource.POLYGON.value,
                DataSource.FRED.value,
                DataSource.QUANDL.value
            ]
        
        # Create data request
        request = DataRequest(
            symbols=[symbol],
            data_type=data_type,
            start_date=start_date,
            end_date=end_date
        )
        
        # Try sources in order
        for source_name in source_order:
            if source_name not in self.data_sources:
                continue
                
            fetcher = self.data_sources[source_name]
            
            try:
                # Check if fetcher can handle this request
                if not fetcher.can_fetch(request):
                    logger.debug(f"{source_name} cannot fetch {symbol}:{data_type}")
                    continue
                
                logger.debug(f"Trying {source_name} for {symbol}:{data_type}")
                
                with Timer(f"fetch_{source_name}_{symbol}") as timer:
                    response = fetcher.fetch_data(request)
                
                if response.has_data():
                    data = response.get_symbol_data(symbol)
                    if data is not None and not data.empty:
                        logger.info(f"✅ Fetched {symbol}:{data_type} from {source_name} in {timer.result.duration_str}")
                        self.stats['total_data_points_fetched'] += len(data)
                        return data
                
            except RateLimitError as e:
                logger.warning(f"Rate limit hit for {source_name}: {e}")
                continue
            except ExternalAPIError as e:
                logger.warning(f"API error from {source_name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error from {source_name}: {e}")
                continue
        
        logger.warning(f"Failed to fetch {symbol}:{data_type} from all sources")
        return None
    
    def _process_data_through_pipeline(self, data: pd.DataFrame, symbol: str, 
                                     pipeline: DataPipeline) -> Optional[pd.DataFrame]:
        """Process data through the specified pipeline"""
        
        current_data = data.copy()
        
        try:
            # Stage 1: Data Cleaning
            if pipeline.cleaning_enabled:
                with Timer(f"clean_{symbol}") as timer:
                    cleaner = create_data_cleaner(pipeline.cleaning_mode)
                    cleaned_data, cleaning_report = cleaner.clean_data(current_data, symbol)
                    current_data = cleaned_data
                    
                    logger.debug(f"Cleaned {symbol} in {timer.result.duration_str}: "
                               f"Quality {cleaning_report['initial_quality_score']:.1f} -> {cleaning_report['final_quality_score']:.1f}")
            
            # Stage 2: Data Transformation
            if pipeline.transformation_enabled:
                with Timer(f"transform_{symbol}") as timer:
                    transformer = create_financial_pipeline(pipeline.transformation_type)
                    
                    # Fit and transform (assuming we have target data - for now just transform features)
                    transformed_data = transformer.fit_transform(current_data)
                    current_data = transformed_data
                    
                    logger.debug(f"Transformed {symbol} in {timer.result.duration_str}: "
                               f"{data.shape[1]} -> {current_data.shape[1]} features")
            
            # Stage 3: Resampling (if enabled)
            if pipeline.resampling_enabled and pipeline.target_frequency:
                with Timer(f"resample_{symbol}") as timer:
                    resampler = create_resampler('standard', target_frequency=pipeline.target_frequency)
                    resampled_data = resampler.resample(current_data, symbol)
                    current_data = resampled_data
                    
                    logger.debug(f"Resampled {symbol} to {pipeline.target_frequency} in {timer.result.duration_str}: "
                               f"{len(data)} -> {len(current_data)} records")
            
            # Stage 4: Data Validation
            if pipeline.validation_enabled:
                validation_result = self.quality_validator.validate_data(current_data, symbol)
                if not validation_result.is_valid:
                    logger.warning(f"Data quality issues for {symbol}: {validation_result.errors}")
            
            return current_data
            
        except Exception as e:
            logger.error(f"Pipeline processing failed for {symbol}: {e}")
            return None
    
    def _get_cached_data(self, symbol: str, data_type: str, 
                        start_date: Optional[datetime], end_date: Optional[datetime]) -> Optional[pd.DataFrame]:
        """Get data from cache"""
        try:
            return self.cache.get(symbol, data_type, start_date, end_date)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    def _cache_processed_data(self, data: pd.DataFrame, symbol: str, data_type: str,
                            start_date: Optional[datetime], end_date: Optional[datetime], ttl: int):
        """Cache processed data"""
        try:
            self.cache.put(data, symbol, data_type, start_date, end_date, ttl=ttl)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def submit_job(self, job: DataJob) -> str:
        """Submit a data job for asynchronous processing"""
        
        with self.job_lock:
            self.job_queue.put((job.priority, job.created_at, job))
            self.stats['jobs_submitted'] += 1
        
        # Submit to thread pool
        future = self.executor.submit(self._execute_data_job, job)
        
        logger.info(f"Submitted job {job.job_id} for processing")
        return job.job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a data job"""
        
        with self.job_lock:
            # Check active jobs
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                return {
                    'job_id': job.job_id,
                    'status': job.status,
                    'progress': job.progress,
                    'symbols': job.symbols,
                    'data_types': job.data_types,
                    'created_at': job.created_at.isoformat(),
                    'error': job.error
                }
            
            # Check completed jobs
            if job_id in self.completed_jobs:
                job = self.completed_jobs[job_id]
                return {
                    'job_id': job.job_id,
                    'status': job.status,
                    'progress': job.progress,
                    'symbols': job.symbols,
                    'data_types': job.data_types,
                    'created_at': job.created_at.isoformat(),
                    'error': job.error,
                    'result_keys': list(job.result.keys()) if job.result else []
                }
        
        return None
    
    def get_job_result(self, job_id: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Get result of a completed job"""
        
        with self.job_lock:
            if job_id in self.completed_jobs:
                job = self.completed_jobs[job_id]
                if job.status == 'completed':
                    return job.result
        
        return None
    
    def batch_fetch(self, symbols: List[str], data_types: List[str],
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   max_workers: Optional[int] = None,
                   pipeline: Optional[DataPipeline] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols in parallel
        
        Args:
            symbols: List of symbols to fetch
            data_types: List of data types to fetch
            start_date: Start date
            end_date: End date
            max_workers: Number of parallel workers
            pipeline: Processing pipeline
            
        Returns:
            Dictionary of fetched and processed data
        """
        
        max_workers = max_workers or self.max_workers
        pipeline = pipeline or self.default_pipeline
        
        logger.info(f"Batch fetching {len(symbols)} symbols with {max_workers} workers")
        
        # Create batch job
        batch_job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        all_results = {}
        
        def fetch_single_symbol(symbol):
            try:
                result = self.fetch_data(
                    symbols=[symbol],
                    data_types=data_types,
                    start_date=start_date,
                    end_date=end_date,
                    pipeline=pipeline
                )
                return symbol, result, None
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                return symbol, None, str(e)
        
        # Process symbols in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_single_symbol, symbol) for symbol in symbols]
            
            for future in as_completed(futures):
                symbol, result, error = future.result()
                
                if result:
                    all_results.update(result)
                elif error:
                    logger.warning(f"Symbol {symbol} failed: {error}")
        
        logger.info(f"Batch fetch completed: {len(all_results)} successful datasets")
        return all_results
    
    def create_data_pipeline(self, name: str, **kwargs) -> DataPipeline:
        """Create a custom data pipeline"""
        
        # Default pipeline settings
        pipeline_config = {
            'name': name,
            'cleaning_enabled': kwargs.get('cleaning_enabled', True),
            'cleaning_mode': kwargs.get('cleaning_mode', 'comprehensive'),
            'transformation_enabled': kwargs.get('transformation_enabled', True),
            'transformation_type': kwargs.get('transformation_type', 'comprehensive'),
            'splitting_enabled': kwargs.get('splitting_enabled', True),
            'splitting_type': kwargs.get('splitting_type', 'research'),
            'resampling_enabled': kwargs.get('resampling_enabled', False),
            'target_frequency': kwargs.get('target_frequency'),
            'validation_enabled': kwargs.get('validation_enabled', True),
            'caching_enabled': kwargs.get('caching_enabled', True),
            'cache_ttl': kwargs.get('cache_ttl', 3600)
        }
        
        pipeline = DataPipeline(**pipeline_config)
        logger.info(f"Created custom pipeline: {name}")
        
        return pipeline
    
    def invalidate_cache(self, symbols: Optional[List[str]] = None,
                        data_types: Optional[List[str]] = None):
        """Invalidate cached data"""
        
        if symbols:
            for symbol in symbols:
                self.cache.invalidate(symbol=symbol)
        
        if data_types:
            for data_type in data_types:
                self.cache.invalidate(data_type=data_type)
        
        if not symbols and not data_types:
            self.cache.clear()
        
        logger.info(f"Cache invalidated for symbols: {symbols}, data_types: {data_types}")
    
    def get_data_sources_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data sources"""
        
        status = {}
        
        for source_name, fetcher in self.data_sources.items():
            try:
                if hasattr(fetcher, 'test_connection'):
                    connection_ok = fetcher.test_connection()
                else:
                    connection_ok = True
                
                if hasattr(fetcher, 'get_status'):
                    source_status = fetcher.get_status()
                else:
                    source_status = {'status': 'available' if connection_ok else 'error'}
                
                status[source_name] = {
                    **source_status,
                    'connection_ok': connection_ok,
                    'last_checked': datetime.now().isoformat()
                }
                
            except Exception as e:
                status[source_name] = {
                    'status': 'error',
                    'error': str(e),
                    'connection_ok': False,
                    'last_checked': datetime.now().isoformat()
                }
        
        return status
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics"""
        
        with self.job_lock:
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()
            
            manager_stats = {
                **self.stats,
                'uptime_seconds': uptime,
                'active_jobs_count': len(self.active_jobs),
                'completed_jobs_count': len(self.completed_jobs),
                'cache_stats': self.cache.get_stats(),
                'data_sources_count': len(self.data_sources),
                'success_rate': (self.stats['jobs_completed'] / max(self.stats['jobs_submitted'], 1)) * 100
            }
        
        return manager_stats
    
    def create_health_report(self) -> Dict[str, Any]:
        """Create comprehensive health report"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'manager_stats': self.get_manager_stats(),
            'data_sources_status': self.get_data_sources_status(),  
            'cache_performance': self.cache.get_stats(),
            'system_health': {
                'memory_cache_healthy': self.cache.memory_cache.current_size_bytes < self.cache.memory_cache.max_size_bytes * 0.9,
                'disk_cache_healthy': True,  # Could add disk space checks
                'job_queue_healthy': self.job_queue.qsize() < 100,  # Arbitrary threshold
                'error_rate_healthy': (self.stats['jobs_failed'] / max(self.stats['jobs_submitted'], 1)) < 0.1
            }
        }
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs"""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self.job_lock:
            jobs_to_remove = [
                job_id for job_id, job in self.completed_jobs.items()
                if job.created_at < cutoff_time
            ]
            
            for job_id in jobs_to_remove:
                self.completed_jobs.pop(job_id)
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old completed jobs")
    
    def shutdown(self):
        """Shutdown the data manager"""
        
        logger.info("Shutting down Financial Data Manager...")
        
        # Wait for active jobs to complete (with timeout)
        self.executor.shutdown(wait=True, timeout=30)
        
        # Clear job queues
        with self.job_lock:
            self.active_jobs.clear()
            
            # Empty the queue
            while not self.job_queue.empty():
                try:
                    self.job_queue.get_nowait()
                except queue.Empty:
                    break
        
        logger.info("Financial Data Manager shutdown complete")

# ============================================
# Convenience Functions and Factory
# ============================================

def create_data_manager(config_type: str = 'standard', **kwargs) -> FinancialDataManager:
    """
    Create pre-configured data manager
    
    Args:
        config_type: Configuration type ('minimal', 'standard', 'comprehensive', 'research')
        **kwargs: Additional configuration options
        
    Returns:
        Configured FinancialDataManager
    """
    
    if config_type == 'minimal':
        default_pipeline = DataPipeline(
            name='minimal',
            cleaning_enabled=True,
            cleaning_mode='minimal',
            transformation_enabled=False,
            splitting_enabled=False,
            validation_enabled=True,
            caching_enabled=True
        )
        max_workers = 2
        
    elif config_type == 'standard':
        default_pipeline = DataPipeline(
            name='standard',
            cleaning_enabled=True,
            cleaning_mode='standard',
            transformation_enabled=True,
            transformation_type='basic',
            splitting_enabled=True,
            splitting_type='basic',
            validation_enabled=True,
            caching_enabled=True
        )
        max_workers = 4
        
    elif config_type == 'comprehensive':
        default_pipeline = DataPipeline(
            name='comprehensive',
            cleaning_enabled=True,
            cleaning_mode='comprehensive',
            transformation_enabled=True,
            transformation_type='comprehensive',
            splitting_enabled=True,
            splitting_type='research',
            validation_enabled=True,
            caching_enabled=True,
            cache_ttl=7200  # 2 hours
        )
        max_workers = 6
        
    elif config_type == 'research':
        default_pipeline = DataPipeline(
            name='research',
            cleaning_enabled=True,
            cleaning_mode='comprehensive',
            transformation_enabled=True,
            transformation_type='comprehensive',
            splitting_enabled=True,
            splitting_type='research',
            resampling_enabled=True,
            target_frequency='1D',
            validation_enabled=True,
            caching_enabled=True,
            cache_ttl=14400  # 4 hours
        )
        max_workers = 8
        
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    # Override with provided kwargs
    if 'max_workers' in kwargs:
        max_workers = kwargs.pop('max_workers')
    
    if 'default_pipeline' in kwargs:
        default_pipeline = kwargs.pop('default_pipeline')
    
    return FinancialDataManager(
        max_workers=max_workers,
        default_pipeline=default_pipeline,
        **kwargs
    )

# Global data manager instance
_global_manager: Optional[FinancialDataManager] = None

def get_data_manager() -> FinancialDataManager:
    """Get global data manager instance"""
    global _global_manager
    
    if _global_manager is None:
        _global_manager = create_data_manager('standard')
    
    return _global_manager

def set_data_manager(manager: FinancialDataManager):
    """Set global data manager instance"""
    global _global_manager
    _global_manager = manager

# Convenience functions using global manager
def fetch_stock_data(symbols: Union[str, List[str]], 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    **kwargs) -> Dict[str, pd.DataFrame]:
    """Fetch stock data using global manager"""
    manager = get_data_manager()
    return manager.fetch_data(symbols, 'stock_data', start_date, end_date, **kwargs)

def fetch_fundamentals(symbols: Union[str, List[str]], **kwargs) -> Dict[str, pd.DataFrame]:
    """Fetch fundamental data using global manager"""
    manager = get_data_manager()
    return manager.fetch_data(symbols, 'fundamentals', **kwargs)

def batch_fetch_stocks(symbols: List[str], 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      **kwargs) -> Dict[str, pd.DataFrame]:
    """Batch fetch stock data using global manager"""
    manager = get_data_manager()
    return manager.batch_fetch(symbols, ['stock_data'], start_date, end_date, **kwargs)
