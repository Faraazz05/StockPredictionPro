# ============================================
# StockPredictionPro - src/utils/timing.py
# Performance timing and monitoring utilities
# ============================================

import time
import functools
import threading
import contextlib
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json

from .logger import get_logger
from .exceptions import BusinessLogicError

logger = get_logger('timing')

# ============================================
# Core Timing Classes
# ============================================

@dataclass
class TimingResult:
    """Container for timing measurement results"""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds"""
        return self.duration * 1000
    
    @property
    def duration_str(self) -> str:
        """Human-readable duration string"""
        return format_duration(self.duration)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'duration_ms': self.duration_ms,
            'duration_str': self.duration_str,
            'metadata': self.metadata,
            'timestamp': datetime.fromtimestamp(self.start_time).isoformat()
        }

class Timer:
    """
    High-precision timer for measuring operation performance
    
    Can be used as context manager or decorator
    """
    
    def __init__(self, operation_name: str, auto_log: bool = True, 
                 log_level: str = 'info', metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize timer
        
        Args:
            operation_name: Name of the operation being timed
            auto_log: Whether to automatically log timing results
            log_level: Log level for timing messages
            metadata: Additional metadata to include
        """
        self.operation_name = operation_name
        self.auto_log = auto_log
        self.log_level = log_level.lower()
        self.metadata = metadata or {}
        
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.result: Optional[TimingResult] = None
        
    def start(self) -> 'Timer':
        """Start timing"""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> TimingResult:
        """Stop timing and return result"""
        if self.start_time is None:
            raise BusinessLogicError("Timer not started")
        
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        
        self.result = TimingResult(
            operation_name=self.operation_name,
            start_time=self.start_time,
            end_time=self.end_time,
            duration=duration,
            metadata=self.metadata
        )
        
        if self.auto_log:
            self._log_result()
        
        return self.result
    
    def _log_result(self):
        """Log timing result"""
        if self.result is None:
            return
        
        message = f"Operation '{self.operation_name}' completed in {self.result.duration_str}"
        
        log_func = getattr(logger, self.log_level, logger.info)
        log_func(message, extra={
            'operation_name': self.operation_name,
            'duration': self.result.duration,
            'duration_ms': self.result.duration_ms,
            'performance_metric': True,
            **self.metadata
        })
    
    def __enter__(self) -> 'Timer':
        """Context manager entry"""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        
        # If exception occurred, log it with timing info
        if exc_type is not None:
            logger.error(
                f"Operation '{self.operation_name}' failed after {self.result.duration_str}",
                extra={
                    'operation_name': self.operation_name,
                    'duration': self.result.duration,
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val)
                },
                exc_info=True
            )

# ============================================
# Performance Monitor
# ============================================

class PerformanceMonitor:
    """
    Advanced performance monitoring and statistics collection
    
    Tracks timing statistics across multiple operations
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor
        
        Args:
            max_history: Maximum number of timing records to keep per operation
        """
        self.max_history = max_history
        self.timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        
    def record_timing(self, timing_result: TimingResult):
        """
        Record timing result
        
        Args:
            timing_result: Timing result to record
        """
        with self.lock:
            operation_name = timing_result.operation_name
            self.timings[operation_name].append(timing_result.duration)
            self.operation_counts[operation_name] += 1
    
    def get_statistics(self, operation_name: str) -> Optional[Dict[str, float]]:
        """
        Get timing statistics for an operation
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary with timing statistics or None if no data
        """
        with self.lock:
            if operation_name not in self.timings or not self.timings[operation_name]:
                return None
            
            durations = list(self.timings[operation_name])
            
            stats = {
                'count': len(durations),
                'total_time': sum(durations),
                'mean': statistics.mean(durations),
                'median': statistics.median(durations),
                'min': min(durations),
                'max': max(durations),
                'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0.0
            }
            
            # Add percentiles
            sorted_durations = sorted(durations)
            n = len(sorted_durations)
            
            stats.update({
                'p25': sorted_durations[int(n * 0.25)],
                'p75': sorted_durations[int(n * 0.75)],
                'p90': sorted_durations[int(n * 0.90)],
                'p95': sorted_durations[int(n * 0.95)],
                'p99': sorted_durations[int(n * 0.99)] if n >= 100 else sorted_durations[-1]
            })
            
            return stats
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations"""
        return {
            operation: self.get_statistics(operation)
            for operation in self.timings.keys()
        }
    
    def get_slow_operations(self, threshold_seconds: float = 1.0) -> List[Dict[str, Any]]:
        """
        Get operations that are slower than threshold
        
        Args:
            threshold_seconds: Threshold in seconds
            
        Returns:
            List of slow operations with statistics
        """
        slow_ops = []
        
        for operation_name in self.timings.keys():
            stats = self.get_statistics(operation_name)
            if stats and stats['mean'] > threshold_seconds:
                slow_ops.append({
                    'operation': operation_name,
                    'mean_duration': stats['mean'],
                    'max_duration': stats['max'],
                    'count': stats['count'],
                    **stats
                })
        
        return sorted(slow_ops, key=lambda x: x['mean_duration'], reverse=True)
    
    def clear_statistics(self, operation_name: Optional[str] = None):
        """
        Clear statistics
        
        Args:
            operation_name: Specific operation to clear, or None for all
        """
        with self.lock:
            if operation_name:
                self.timings[operation_name].clear()
                self.operation_counts[operation_name] = 0
            else:
                self.timings.clear()
                self.operation_counts.clear()
    
    def create_performance_report(self) -> str:
        """Create formatted performance report"""
        all_stats = self.get_all_statistics()
        
        if not all_stats:
            return "No performance data available"
        
        lines = [
            "\nPerformance Report",
            "=" * 50,
            ""
        ]
        
        # Sort operations by mean duration (slowest first)
        sorted_ops = sorted(
            [(op, stats) for op, stats in all_stats.items() if stats],
            key=lambda x: x[1]['mean'],
            reverse=True
        )
        
        for operation, stats in sorted_ops:
            lines.extend([
                f"Operation: {operation}",
                f"  Count: {stats['count']}",
                f"  Total Time: {format_duration(stats['total_time'])}",
                f"  Mean: {format_duration(stats['mean'])}",
                f"  Median: {format_duration(stats['median'])}",
                f"  Min: {format_duration(stats['min'])}",
                f"  Max: {format_duration(stats['max'])}",
                f"  95th Percentile: {format_duration(stats['p95'])}",
                ""
            ])
        
        return "\n".join(lines)

# ============================================
# Decorators
# ============================================

def time_it(operation_name: Optional[str] = None, auto_log: bool = True, 
           log_level: str = 'info', include_args: bool = False):
    """
    Decorator to time function execution
    
    Args:
        operation_name: Custom operation name (defaults to function name)
        auto_log: Whether to automatically log timing results
        log_level: Log level for timing messages
        include_args: Whether to include function arguments in metadata
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metadata = {}
            
            if include_args:
                # Include function arguments (truncated for safety)
                metadata['args_count'] = len(args)
                metadata['kwargs_count'] = len(kwargs)
                
                # Include first few args as strings (truncated)
                if args:
                    metadata['first_args'] = [str(arg)[:50] for arg in args[:3]]
            
            timer = Timer(operation_name, auto_log, log_level, metadata)
            
            with timer:
                result = func(*args, **kwargs)
            
            # Record timing in global monitor
            if hasattr(wrapper, '_monitor') and wrapper._monitor:
                wrapper._monitor.record_timing(timer.result)
            
            return result
        
        # Attach monitor if available
        wrapper._monitor = None
        
        return wrapper
    
    return decorator

def profile_performance(monitor: Optional[PerformanceMonitor] = None):
    """
    Decorator to profile function performance with detailed monitoring
    
    Args:
        monitor: Performance monitor instance (creates new if None)
        
    Returns:
        Decorator function
    """
    if monitor is None:
        monitor = PerformanceMonitor()
    
    def decorator(func: Callable) -> Callable:
        operation_name = f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer = Timer(operation_name, auto_log=False)
            
            with timer:
                result = func(*args, **kwargs)
            
            # Record detailed performance data
            monitor.record_timing(timer.result)
            
            return result
        
        # Attach monitor for access
        wrapper._monitor = monitor
        wrapper._operation_name = operation_name
        
        return wrapper
    
    return decorator

def timeout(seconds: float):
    """
    Decorator to timeout function execution
    
    Args:
        seconds: Maximum execution time in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set timeout signal (Unix only)
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(seconds))
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)  # Cancel alarm
                    signal.signal(signal.SIGALRM, old_handler)
                
                return result
                
            except AttributeError:
                # Windows doesn't have SIGALRM, use basic timing check
                start_time = time.time()
                result = func(*args, **kwargs)
                
                if time.time() - start_time > seconds:
                    logger.warning(f"Function {func.__name__} exceeded timeout ({seconds}s) but couldn't be interrupted")
                
                return result
        
        return wrapper
    
    return decorator

# ============================================
# Context Managers
# ============================================

@contextlib.contextmanager
def timing_context(operation_name: str, auto_log: bool = True, 
                  metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager for timing code blocks
    
    Args:
        operation_name: Name of the operation
        auto_log: Whether to automatically log results
        metadata: Additional metadata
        
    Yields:
        Timer instance
    """
    timer = Timer(operation_name, auto_log=auto_log, metadata=metadata)
    
    try:
        with timer:
            yield timer
    finally:
        pass  # Timer handles cleanup in __exit__

@contextlib.contextmanager
def performance_session(session_name: str, monitor: Optional[PerformanceMonitor] = None):
    """
    Context manager for performance monitoring sessions
    
    Args:
        session_name: Name of the performance session
        monitor: Performance monitor to use
        
    Yields:
        Performance monitor instance
    """
    if monitor is None:
        monitor = PerformanceMonitor()
    
    logger.info(f"Starting performance session: {session_name}")
    session_start = time.time()
    
    try:
        yield monitor
    finally:
        session_duration = time.time() - session_start
        logger.info(
            f"Performance session '{session_name}' completed in {format_duration(session_duration)}",
            extra={
                'session_name': session_name,
                'session_duration': session_duration,
                'operations_count': sum(monitor.operation_counts.values())
            }
        )

# ============================================
# Batch Timing
# ============================================

class BatchTimer:
    """Timer for batch operations with progress tracking"""
    
    def __init__(self, operation_name: str, total_items: int):
        """
        Initialize batch timer
        
        Args:
            operation_name: Name of the batch operation
            total_items: Total number of items to process
        """
        self.operation_name = operation_name
        self.total_items = total_items
        self.completed_items = 0
        
        self.start_time = time.time()
        self.item_times: List[float] = []
        
    def record_item_completion(self, item_duration: Optional[float] = None):
        """
        Record completion of a single item
        
        Args:
            item_duration: Duration of the item processing (optional)
        """
        self.completed_items += 1
        
        if item_duration is not None:
            self.item_times.append(item_duration)
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information"""
        elapsed_time = time.time() - self.start_time
        progress_pct = (self.completed_items / self.total_items) * 100 if self.total_items > 0 else 0
        
        # Estimate remaining time
        if self.completed_items > 0:
            avg_time_per_item = elapsed_time / self.completed_items
            remaining_items = self.total_items - self.completed_items
            estimated_remaining = avg_time_per_item * remaining_items
        else:
            estimated_remaining = 0
        
        info = {
            'operation_name': self.operation_name,
            'completed_items': self.completed_items,
            'total_items': self.total_items,
            'progress_pct': progress_pct,
            'elapsed_time': elapsed_time,
            'estimated_remaining': estimated_remaining,
            'estimated_total': elapsed_time + estimated_remaining
        }
        
        # Add item timing statistics if available
        if self.item_times:
            info.update({
                'avg_item_time': statistics.mean(self.item_times),
                'min_item_time': min(self.item_times),
                'max_item_time': max(self.item_times)
            })
        
        return info
    
    def log_progress(self, log_interval: int = 10):
        """
        Log progress if appropriate
        
        Args:
            log_interval: Log every N items
        """
        if self.completed_items % log_interval == 0 or self.completed_items == self.total_items:
            info = self.get_progress_info()
            
            logger.info(
                f"{self.operation_name}: {self.completed_items}/{self.total_items} "
                f"({info['progress_pct']:.1f}%) - ETA: {format_duration(info['estimated_remaining'])}",
                extra=info
            )

# ============================================
# Utility Functions
# ============================================

def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 0.001:
        return f"{seconds*1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"

def benchmark_function(func: Callable, *args, iterations: int = 100, 
                      warmup_iterations: int = 5, **kwargs) -> Dict[str, float]:
    """
    Benchmark function performance
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with benchmark statistics
    """
    # Warmup
    for _ in range(warmup_iterations):
        func(*args, **kwargs)
    
    # Actual benchmark
    durations = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        durations.append(end_time - start_time)
    
    return {
        'iterations': iterations,
        'total_time': sum(durations),
        'mean': statistics.mean(durations),
        'median': statistics.median(durations),
        'min': min(durations),
        'max': max(durations),
        'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0.0,
        'p95': sorted(durations)[int(iterations * 0.95)],
        'p99': sorted(durations)[int(iterations * 0.99)]
    }

def measure_memory_usage():
    """
    Measure current memory usage
    
    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        logger.warning("psutil not available for memory measurement")
        return 0.0

def create_timing_summary(timing_results: List[TimingResult]) -> Dict[str, Any]:
    """
    Create summary statistics from timing results
    
    Args:
        timing_results: List of timing results
        
    Returns:
        Summary statistics dictionary
    """
    if not timing_results:
        return {}
    
    durations = [result.duration for result in timing_results]
    
    return {
        'count': len(durations),
        'total_time': sum(durations),
        'mean': statistics.mean(durations),
        'median': statistics.median(durations),
        'min': min(durations),
        'max': max(durations),
        'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0.0,
        'operations': [result.operation_name for result in timing_results],
        'start_time': min(result.start_time for result in timing_results),
        'end_time': max(result.end_time for result in timing_results)
    }

# ============================================
# Global Performance Monitor
# ============================================

# Global performance monitor instance
global_monitor = PerformanceMonitor(max_history=10000)

def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return global_monitor

def reset_global_monitor():
    """Reset the global performance monitor"""
    global_monitor.clear_statistics()

def log_performance_summary():
    """Log performance summary from global monitor"""
    report = global_monitor.create_performance_report()
    logger.info("Performance Summary", extra={'performance_report': report})
    print(report)  # Also print to console

# ============================================
# Convenience Functions
# ============================================

def quick_timer(operation_name: str) -> Timer:
    """Create a quick timer instance"""
    return Timer(operation_name, auto_log=True)

def silent_timer(operation_name: str) -> Timer:
    """Create a silent timer instance (no auto-logging)"""
    return Timer(operation_name, auto_log=False)
