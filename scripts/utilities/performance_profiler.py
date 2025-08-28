"""
scripts/utilities/performance_profiler.py

Performance profiling and optimization system for StockPredictionPro.
Provides CPU profiling, memory analysis, I/O monitoring, and performance
benchmarking with detailed reports and optimization recommendations.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import logging
import time
import threading
import psutil
import functools
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
import cProfile
import pstats
import io
import gc
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Performance monitoring libraries
# Performance monitoring libraries
try:
    import memory_profiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

try:
    import line_profiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False

# py-spy is a command-line tool, not a Python module
# Check if py-spy binary is available
import shutil
HAS_PY_SPY = shutil.which('py-spy') is not None

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    mdates = None

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'performance_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StockPredictionPro.Performance')

# Directory configuration
PROJECT_ROOT = Path('.')
PROFILE_DIR = PROJECT_ROOT / 'profiling'
REPORTS_DIR = PROFILE_DIR / 'reports'
BENCHMARKS_DIR = PROFILE_DIR / 'benchmarks'

# Ensure directories exist
for dir_path in [PROFILE_DIR, REPORTS_DIR, BENCHMARKS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class ProfileConfig:
    """Configuration for performance profiling"""
    # Profiling types
    enable_cpu_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_io_profiling: bool = True
    enable_line_profiling: bool = False
    
    # Sampling settings
    cpu_sample_interval: float = 0.01  # 10ms
    memory_sample_interval: float = 0.1  # 100ms
    io_sample_interval: float = 0.1  # 100ms
    
    # Duration and limits
    max_profile_duration: int = 300  # 5 minutes
    max_memory_snapshots: int = 1000
    max_call_stack_depth: int = 50
    
    # Output settings
    generate_reports: bool = True
    generate_visualizations: bool = True
    save_raw_data: bool = True
    
    # Filtering
    include_patterns: List[str] = None
    exclude_patterns: List[str] = None
    min_execution_time: float = 0.001  # 1ms
    
    # System monitoring
    monitor_system_resources: bool = True
    system_sample_interval: float = 1.0  # 1 second
    
    def __post_init__(self):
        if self.include_patterns is None:
            self.include_patterns = ['scripts/', 'api/']
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                'site-packages/', '__pycache__/', '.pytest_cache/',
                'venv/', 'env/', '.git/'
            ]
    def _run_py_spy_profile(self, pid: int, duration: int = 10) -> Optional[str]:
        """Run py-spy profiling via subprocess for given process PID"""
        if not HAS_PY_SPY:
            logger.warning("py-spy binary not available in PATH")
            return None
        
        try:
            import subprocess
            import tempfile
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
                output_file = f.name
            
            # Run py-spy record command
            cmd = [
                'py-spy', 'record',
                '--pid', str(pid),
                '--duration', str(duration),
                '--output', output_file,
                '--format', 'flamegraph'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 10)
            
            if result.returncode == 0:
                logger.info(f"py-spy flamegraph saved to {output_file}")
                return output_file
            else:
                logger.error(f"py-spy failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"py-spy subprocess failed: {e}")
            return None

@dataclass
class ProfileMetrics:
    """Performance metrics from profiling"""
    # Timing metrics
    total_time: float
    cpu_time: float
    wall_time: float
    
    # Memory metrics
    peak_memory_mb: float
    memory_growth_mb: float
    gc_collections: int
    
    # Call statistics
    total_calls: int
    function_calls: int
    primitive_calls: int
    
    # I/O metrics
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0

@dataclass
class FunctionProfile:
    """Profile data for individual function"""
    name: str
    filename: str
    line_number: int
    total_time: float
    cumulative_time: float
    calls: int
    avg_time_per_call: float
    percentage_of_total: float
    memory_usage_mb: float = 0.0

@dataclass
class ProfileResult:
    """Complete profiling result"""
    profile_id: str
    start_time: datetime
    end_time: datetime
    duration: float
    
    # Overall metrics
    metrics: ProfileMetrics
    
    # Function profiles
    function_profiles: List[FunctionProfile]
    
    # System resources
    system_snapshots: List[Dict[str, Any]] = None
    
    # Memory snapshots
    memory_snapshots: List[Dict[str, Any]] = None
    
    # Configuration used
    config: ProfileConfig = None
    
    # Analysis results
    bottlenecks: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.system_snapshots is None:
            self.system_snapshots = []
        if self.memory_snapshots is None:
            self.memory_snapshots = []
        if self.bottlenecks is None:
            self.bottlenecks = []
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ============================================
# PERFORMANCE DECORATORS
# ============================================

class PerformanceProfiler:
    """Main performance profiler class"""
    
    def __init__(self, config: ProfileConfig = None):
        self.config = config or ProfileConfig()
        self.active_profiles = {}
        self.system_monitor = None
        self.memory_tracker = None
        
    def profile(self, name: Optional[str] = None):
        """Decorator for profiling functions"""
        def decorator(func: Callable) -> Callable:
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.profile_function(func, profile_name, *args, **kwargs)
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.profile_async_function(func, profile_name, *args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
        
        return decorator
    
    def profile_function(self, func: Callable, name: str, *args, **kwargs) -> Any:
        """Profile synchronous function execution"""
        profile_id = f"{name}_{int(time.time() * 1000)}"
        
        # Start profiling
        profiler_data = self._start_profiling(profile_id)
        
        try:
            # Execute function
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            # Stop profiling and generate result
            profile_result = self._stop_profiling(
                profile_id, profiler_data, start_time, end_time
            )
            
            # Save results
            if self.config.generate_reports:
                self._save_profile_result(profile_result)
            
            return result
            
        except Exception as e:
            # Stop profiling even on error
            self._stop_profiling(profile_id, profiler_data, time.perf_counter(), time.perf_counter())
            raise
    
    async def profile_async_function(self, func: Callable, name: str, *args, **kwargs) -> Any:
        """Profile asynchronous function execution"""
        profile_id = f"{name}_{int(time.time() * 1000)}"
        
        # Start profiling
        profiler_data = self._start_profiling(profile_id)
        
        try:
            # Execute async function
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            
            # Stop profiling and generate result
            profile_result = self._stop_profiling(
                profile_id, profiler_data, start_time, end_time
            )
            
            # Save results
            if self.config.generate_reports:
                self._save_profile_result(profile_result)
            
            return result
            
        except Exception as e:
            # Stop profiling even on error
            self._stop_profiling(profile_id, profiler_data, time.perf_counter(), time.perf_counter())
            raise
    
    def _start_profiling(self, profile_id: str) -> Dict[str, Any]:
        """Start profiling session"""
        profiler_data = {
            'profile_id': profile_id,
            'start_time': datetime.now(),
            'cpu_profiler': None,
            'memory_tracker': None,
            'system_monitor': None,
            'initial_memory': 0,
            'initial_io': None
        }
        
        # CPU profiling
        if self.config.enable_cpu_profiling:
            profiler_data['cpu_profiler'] = cProfile.Profile()
            profiler_data['cpu_profiler'].enable()
        
        # Memory profiling
        if self.config.enable_memory_profiling:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            profiler_data['initial_memory'] = self._get_memory_usage()
        
        # I/O monitoring
        if self.config.enable_io_profiling:
            profiler_data['initial_io'] = self._get_io_counters()
        
        # System monitoring
        if self.config.monitor_system_resources:
            profiler_data['system_monitor'] = SystemMonitor(self.config.system_sample_interval)
            profiler_data['system_monitor'].start()
        
        self.active_profiles[profile_id] = profiler_data
        return profiler_data
    
    def _stop_profiling(self, profile_id: str, profiler_data: Dict[str, Any], 
                       start_time: float, end_time: float) -> ProfileResult:
        """Stop profiling session and generate result"""
        
        duration = end_time - start_time
        
        # Stop CPU profiler
        cpu_stats = None
        if profiler_data['cpu_profiler']:
            profiler_data['cpu_profiler'].disable()
            cpu_stats = pstats.Stats(profiler_data['cpu_profiler'])
        
        # Get memory information
        final_memory = self._get_memory_usage()
        memory_growth = final_memory - profiler_data['initial_memory']
        
        # Get I/O information
        final_io = self._get_io_counters()
        io_diff = self._calculate_io_diff(profiler_data['initial_io'], final_io)
        
        # Stop system monitoring
        system_snapshots = []
        if profiler_data['system_monitor']:
            profiler_data['system_monitor'].stop()
            system_snapshots = profiler_data['system_monitor'].get_snapshots()
        
        # Process CPU profiling data
        function_profiles = []
        total_cpu_time = 0
        
        if cpu_stats:
            cpu_stats.sort_stats('cumulative')
            total_cpu_time = cpu_stats.total_tt
            
            function_profiles = self._extract_function_profiles(cpu_stats, total_cpu_time)
        
        # Create metrics
        metrics = ProfileMetrics(
            total_time=duration,
            cpu_time=total_cpu_time,
            wall_time=duration,
            peak_memory_mb=final_memory,
            memory_growth_mb=memory_growth,
            gc_collections=gc.get_count()[0],
            total_calls=sum(stats.ncalls for stats in (cpu_stats.stats.values() if cpu_stats else [])),
            function_calls=len(function_profiles),
            primitive_calls=sum(stats.pcalls for stats in (cpu_stats.stats.values() if cpu_stats else [])),
            disk_read_mb=io_diff.get('disk_read_mb', 0),
            disk_write_mb=io_diff.get('disk_write_mb', 0),
            network_sent_mb=io_diff.get('network_sent_mb', 0),
            network_recv_mb=io_diff.get('network_recv_mb', 0)
        )
        
        # Create result
        result = ProfileResult(
            profile_id=profile_id,
            start_time=profiler_data['start_time'],
            end_time=datetime.now(),
            duration=duration,
            metrics=metrics,
            function_profiles=function_profiles,
            system_snapshots=system_snapshots,
            config=self.config
        )
        
        # Analyze performance
        result.bottlenecks = self._identify_bottlenecks(result)
        result.recommendations = self._generate_recommendations(result)
        
        # Clean up
        if profile_id in self.active_profiles:
            del self.active_profiles[profile_id]
        
        return result
    
    def _extract_function_profiles(self, stats: pstats.Stats, total_time: float) -> List[FunctionProfile]:
        """Extract function profiles from pstats"""
        function_profiles = []
        
        # Get top functions by cumulative time
        stats_data = []
        for func_key, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line_num, func_name = func_key
            
            # Apply filters
            if not self._should_include_function(filename):
                continue
            
            if tt < self.config.min_execution_time:
                continue
            
            stats_data.append({
                'name': func_name,
                'filename': filename,
                'line_number': line_num,
                'total_time': tt,
                'cumulative_time': ct,
                'calls': nc,
                'avg_time_per_call': tt / nc if nc > 0 else 0,
                'percentage_of_total': (ct / total_time * 100) if total_time > 0 else 0
            })
        
        # Sort by cumulative time
        stats_data.sort(key=lambda x: x['cumulative_time'], reverse=True)
        
        # Convert to FunctionProfile objects
        for data in stats_data[:50]:  # Top 50 functions
            function_profiles.append(FunctionProfile(**data))
        
        return function_profiles
    
    def _should_include_function(self, filename: str) -> bool:
        """Check if function should be included in profiling"""
        # Check exclude patterns
        for pattern in self.config.exclude_patterns:
            if pattern in filename:
                return False
        
        # Check include patterns
        if self.config.include_patterns:
            for pattern in self.config.include_patterns:
                if pattern in filename:
                    return True
            return False
        
        return True
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _get_io_counters(self) -> Dict[str, float]:
        """Get current I/O counters"""
        try:
            process = psutil.Process()
            io_counters = process.io_counters()
            
            return {
                'read_bytes': io_counters.read_bytes,
                'write_bytes': io_counters.write_bytes,
                'read_count': io_counters.read_count,
                'write_count': io_counters.write_count
            }
        except:
            return {'read_bytes': 0, 'write_bytes': 0, 'read_count': 0, 'write_count': 0}
    
    def _calculate_io_diff(self, initial: Dict[str, float], final: Dict[str, float]) -> Dict[str, float]:
        """Calculate I/O difference"""
        if not initial or not final:
            return {'disk_read_mb': 0, 'disk_write_mb': 0, 'network_sent_mb': 0, 'network_recv_mb': 0}
        
        return {
            'disk_read_mb': (final['read_bytes'] - initial['read_bytes']) / 1024 / 1024,
            'disk_write_mb': (final['write_bytes'] - initial['write_bytes']) / 1024 / 1024,
            'network_sent_mb': 0,  # Would need network monitoring
            'network_recv_mb': 0
        }
    
    def _identify_bottlenecks(self, result: ProfileResult) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # High CPU usage functions
        cpu_threshold = result.metrics.total_time * 0.1  # 10% of total time
        cpu_bottlenecks = [
            f for f in result.function_profiles 
            if f.cumulative_time > cpu_threshold
        ]
        
        if cpu_bottlenecks:
            bottlenecks.append(
                f"CPU bottlenecks: {', '.join([f.name for f in cpu_bottlenecks[:3]])}"
            )
        
        # High memory usage
        if result.metrics.memory_growth_mb > 100:  # 100MB growth
            bottlenecks.append(f"High memory usage: {result.metrics.memory_growth_mb:.1f}MB growth")
        
        # High I/O
        total_io = result.metrics.disk_read_mb + result.metrics.disk_write_mb
        if total_io > 10:  # 10MB I/O
            bottlenecks.append(f"High I/O usage: {total_io:.1f}MB disk I/O")
        
        # Long execution time
        if result.duration > 1.0:  # 1 second
            bottlenecks.append(f"Long execution time: {result.duration:.2f}s")
        
        return bottlenecks
    
    def _generate_recommendations(self, result: ProfileResult) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Function-specific recommendations
        if result.function_profiles:
            top_function = result.function_profiles[0]
            
            if top_function.percentage_of_total > 50:
                recommendations.append(
                    f"Optimize '{top_function.name}' - consuming {top_function.percentage_of_total:.1f}% of execution time"
                )
            
            # Check for functions with many calls
            high_call_functions = [
                f for f in result.function_profiles if f.calls > 1000
            ]
            
            if high_call_functions:
                recommendations.append(
                    f"Consider caching or reducing calls to: {', '.join([f.name for f in high_call_functions[:2]])}"
                )
        
        # Memory recommendations
        if result.metrics.memory_growth_mb > 50:
            recommendations.append("Consider memory optimization - high memory growth detected")
        
        # I/O recommendations
        if result.metrics.disk_read_mb > 5:
            recommendations.append("Consider I/O optimization - high disk read activity")
        
        if result.metrics.disk_write_mb > 5:
            recommendations.append("Consider I/O optimization - high disk write activity")
        
        # General recommendations
        if result.duration > 0.5:
            recommendations.append("Consider algorithm optimization for better performance")
        
        if not recommendations:
            recommendations.append("Performance looks good - no major bottlenecks detected")
        
        return recommendations
    
    def _save_profile_result(self, result: ProfileResult) -> None:
        """Save profiling result to file"""
        try:
            # Save JSON report
            report_file = REPORTS_DIR / f"{result.profile_id}.json"
            with open(report_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            # Generate text report
            text_report = self._generate_text_report(result)
            text_file = REPORTS_DIR / f"{result.profile_id}.txt"
            with open(text_file, 'w') as f:
                f.write(text_report)
            
            logger.info(f"Profile results saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save profile result: {e}")
    
    def _generate_text_report(self, result: ProfileResult) -> str:
        """Generate human-readable text report"""
        report = f"""StockPredictionPro Performance Profile Report
{'='*50}

Profile ID: {result.profile_id}
Start Time: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {result.duration:.3f} seconds

PERFORMANCE METRICS
{'-'*20}
Total Time: {result.metrics.total_time:.3f}s
CPU Time: {result.metrics.cpu_time:.3f}s
Wall Time: {result.metrics.wall_time:.3f}s
Peak Memory: {result.metrics.peak_memory_mb:.1f} MB
Memory Growth: {result.metrics.memory_growth_mb:.1f} MB
Total Calls: {result.metrics.total_calls:,}
Function Calls: {result.metrics.function_calls:,}
GC Collections: {result.metrics.gc_collections:,}

I/O METRICS
{'-'*20}
Disk Read: {result.metrics.disk_read_mb:.2f} MB
Disk Write: {result.metrics.disk_write_mb:.2f} MB

TOP FUNCTIONS BY TIME
{'-'*20}
"""
        
        for i, func in enumerate(result.function_profiles[:10], 1):
            report += f"{i:2d}. {func.name:<40} {func.cumulative_time:8.3f}s ({func.percentage_of_total:5.1f}%)\n"
            report += f"    File: {func.filename}:{func.line_number}\n"
            report += f"    Calls: {func.calls:,}, Avg: {func.avg_time_per_call*1000:.2f}ms\n\n"
        
        if result.bottlenecks:
            report += f"\nPERFORMANCE BOTTLENECKS\n{'-'*20}\n"
            for bottleneck in result.bottlenecks:
                report += f"• {bottleneck}\n"
        
        if result.recommendations:
            report += f"\nRECOMMENDATIONS\n{'-'*20}\n"
            for recommendation in result.recommendations:
                report += f"• {recommendation}\n"
        
        return report

class SystemMonitor:
    """Monitor system resources during profiling"""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.snapshots = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start(self) -> None:
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.snapshots = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self) -> None:
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Get collected system snapshots"""
        return self.snapshots.copy()
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                break
    
    def _take_snapshot(self) -> Dict[str, Any]:
        """Take system resource snapshot"""
        try:
            # Process information
            process = psutil.Process()
            
            # System information
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'process_cpu_percent': process.cpu_percent(),
                'memory_percent': memory.percent,
                'process_memory_mb': process.memory_info().rss / 1024 / 1024,
                'available_memory_mb': memory.available / 1024 / 1024,
                'disk_io_read_bytes': getattr(process.io_counters(), 'read_bytes', 0),
                'disk_io_write_bytes': getattr(process.io_counters(), 'write_bytes', 0),
                'threads': process.num_threads(),
                'open_files': len(process.open_files())
            }
        except Exception as e:
            logger.error(f"Failed to take system snapshot: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

# ============================================
# BENCHMARKING SYSTEM
# ============================================

class BenchmarkSuite:
    """Performance benchmarking system"""
    
    def __init__(self, config: ProfileConfig = None):
        self.config = config or ProfileConfig()
        self.profiler = PerformanceProfiler(config)
        self.benchmarks = {}
    
    def benchmark(self, name: str, iterations: int = 100):
        """Decorator for benchmarking functions"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.run_benchmark(func, name, iterations, *args, **kwargs)
            return wrapper
        return decorator
    
    def run_benchmark(self, func: Callable, name: str, iterations: int, *args, **kwargs) -> Dict[str, Any]:
        """Run benchmark for function"""
        logger.info(f"Running benchmark: {name} ({iterations} iterations)")
        
        results = []
        total_time = 0
        
        # Warmup
        for _ in range(min(10, iterations // 10)):
            func(*args, **kwargs)
        
        # Actual benchmark
        for i in range(iterations):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            results.append(execution_time)
            total_time += execution_time
        
        # Calculate statistics
        results.sort()
        benchmark_stats = {
            'name': name,
            'iterations': iterations,
            'total_time': total_time,
            'average_time': total_time / iterations,
            'min_time': min(results),
            'max_time': max(results),
            'median_time': results[len(results) // 2],
            'p95_time': results[int(len(results) * 0.95)],
            'p99_time': results[int(len(results) * 0.99)],
            'std_dev': self._calculate_std_dev(results),
            'operations_per_second': iterations / total_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save benchmark results
        self._save_benchmark_result(benchmark_stats)
        
        # Store for comparison
        self.benchmarks[name] = benchmark_stats
        
        return benchmark_stats
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _save_benchmark_result(self, stats: Dict[str, Any]) -> None:
        """Save benchmark results"""
        try:
            benchmark_file = BENCHMARKS_DIR / f"{stats['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(benchmark_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            logger.info(f"Benchmark results saved: {benchmark_file}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmark result: {e}")
    
    def compare_benchmarks(self, names: List[str]) -> Dict[str, Any]:
        """Compare multiple benchmark results"""
        comparison = {
            'comparison_time': datetime.now().isoformat(),
            'benchmarks': {},
            'winner': None,
            'performance_ratios': {}
        }
        
        valid_benchmarks = {name: self.benchmarks[name] for name in names if name in self.benchmarks}
        
        if len(valid_benchmarks) < 2:
            logger.warning("Need at least 2 benchmarks for comparison")
            return comparison
        
        comparison['benchmarks'] = valid_benchmarks
        
        # Find fastest benchmark
        fastest = min(valid_benchmarks.values(), key=lambda x: x['average_time'])
        comparison['winner'] = fastest['name']
        
        # Calculate performance ratios
        for name, stats in valid_benchmarks.items():
            ratio = stats['average_time'] / fastest['average_time']
            comparison['performance_ratios'][name] = {
                'ratio': ratio,
                'percentage_slower': (ratio - 1) * 100 if ratio > 1 else 0
            }
        
        return comparison

# ============================================
# VISUALIZATION SYSTEM
# ============================================

class PerformanceVisualizer:
    """Generate performance visualizations"""
    
    def __init__(self, config: ProfileConfig):
        self.config = config
    
    def generate_visualizations(self, result: ProfileResult) -> List[str]:
        """Generate performance visualizations"""
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not available for visualizations")
            return []
        
        generated_files = []
        
        try:
            # Function performance chart
            func_chart = self._create_function_performance_chart(result)
            if func_chart:
                generated_files.append(func_chart)
            
            # Memory usage over time
            if result.system_snapshots:
                memory_chart = self._create_memory_usage_chart(result)
                if memory_chart:
                    generated_files.append(memory_chart)
            
            # System resources chart
            if result.system_snapshots:
                system_chart = self._create_system_resources_chart(result)
                if system_chart:
                    generated_files.append(system_chart)
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
        
        return generated_files
    
    def _create_function_performance_chart(self, result: ProfileResult) -> Optional[str]:
        """Create function performance chart"""
        try:
            if not result.function_profiles:
                return None
            
            # Top 10 functions
            top_functions = result.function_profiles[:10]
            
            names = [f.name[:30] + '...' if len(f.name) > 30 else f.name for f in top_functions]
            times = [f.cumulative_time for f in top_functions]
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(names, times)
            
            # Color bars by performance
            colors = plt.cm.Reds([t / max(times) for t in times])
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.xlabel('Cumulative Time (seconds)')
            plt.title(f'Top Functions by Execution Time - {result.profile_id}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            chart_file = REPORTS_DIR / f"{result.profile_id}_functions.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_file)
            
        except Exception as e:
            logger.error(f"Function chart generation failed: {e}")
            return None
    
    def _create_memory_usage_chart(self, result: ProfileResult) -> Optional[str]:
        """Create memory usage over time chart"""
        try:
            timestamps = []
            memory_usage = []
            
            for snapshot in result.system_snapshots:
                if 'error' in snapshot:
                    continue
                    
                timestamps.append(datetime.fromisoformat(snapshot['timestamp']))
                memory_usage.append(snapshot['process_memory_mb'])
            
            if not timestamps:
                return None
            
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, memory_usage, linewidth=2, color='blue')
            plt.xlabel('Time')
            plt.ylabel('Memory Usage (MB)')
            plt.title(f'Memory Usage Over Time - {result.profile_id}')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            chart_file = REPORTS_DIR / f"{result.profile_id}_memory.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_file)
            
        except Exception as e:
            logger.error(f"Memory chart generation failed: {e}")
            return None
    
    def _create_system_resources_chart(self, result: ProfileResult) -> Optional[str]:
        """Create system resources chart"""
        try:
            timestamps = []
            cpu_usage = []
            memory_percent = []
            
            for snapshot in result.system_snapshots:
                if 'error' in snapshot:
                    continue
                    
                timestamps.append(datetime.fromisoformat(snapshot['timestamp']))
                cpu_usage.append(snapshot['process_cpu_percent'])
                memory_percent.append(snapshot['memory_percent'])
            
            if not timestamps:
                return None
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # CPU usage
            ax1.plot(timestamps, cpu_usage, linewidth=2, color='red', label='Process CPU %')
            ax1.set_ylabel('CPU Usage (%)')
            ax1.set_title(f'System Resources Over Time - {result.profile_id}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Memory usage
            ax2.plot(timestamps, memory_percent, linewidth=2, color='green', label='System Memory %')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Memory Usage (%)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            chart_file = REPORTS_DIR / f"{result.profile_id}_system.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_file)
            
        except Exception as e:
            logger.error(f"System chart generation failed: {e}")
            return None

# ============================================
# CLI INTERFACE
# ============================================

def profile_script(script_path: str, config: ProfileConfig) -> ProfileResult:
    """Profile entire Python script"""
    profiler = PerformanceProfiler(config)
    
    profile_id = f"script_{Path(script_path).stem}_{int(time.time())}"
    
    # Start profiling
    profiler_data = profiler._start_profiling(profile_id)
    
    try:
        # Execute script
        start_time = time.perf_counter()
        
        with open(script_path, 'r') as f:
            script_code = f.read()
        
        # Execute in global namespace
        exec(script_code, globals())
        
        end_time = time.perf_counter()
        
        # Stop profiling and generate result
        result = profiler._stop_profiling(profiler_data, profiler_data, start_time, end_time)
        
        # Generate visualizations
        if config.generate_visualizations:
            visualizer = PerformanceVisualizer(config)
            charts = visualizer.generate_visualizations(result)
            logger.info(f"Generated {len(charts)} visualization charts")
        
        return result
        
    except Exception as e:
        profiler._stop_profiling(profile_id, profiler_data, time.perf_counter(), time.perf_counter())
        raise

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance profiler for StockPredictionPro')
    parser.add_argument('--script', help='Python script to profile')
    parser.add_argument('--config', help='Path to profiling configuration JSON file')
    parser.add_argument('--cpu', action='store_true', default=True, help='Enable CPU profiling')
    parser.add_argument('--memory', action='store_true', default=True, help='Enable memory profiling')
    parser.add_argument('--io', action='store_true', default=True, help='Enable I/O profiling')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--duration', type=int, default=300, help='Max profiling duration in seconds')
    parser.add_argument('--output-dir', default='profiling/reports', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = ProfileConfig(**config_dict)
        except Exception as e:
            logger.warning(f"Could not load config from {args.config}: {e}")
            config = ProfileConfig()
    else:
        config = ProfileConfig()
    
    # Override with CLI arguments
    config.enable_cpu_profiling = args.cpu
    config.enable_memory_profiling = args.memory
    config.enable_io_profiling = args.io
    config.generate_visualizations = args.visualize
    config.max_profile_duration = args.duration
    
    try:
        if args.script:
            # Profile specific script
            if not Path(args.script).exists():
                print(f"❌ Script not found: {args.script}")
                sys.exit(1)
            
            logger.info(f"Profiling script: {args.script}")
            result = profile_script(args.script, config)
            
            print(f"\n✅ Profiling completed!")
            print(f"Profile ID: {result.profile_id}")
            print(f"Duration: {result.duration:.3f} seconds")
            print(f"Peak Memory: {result.metrics.peak_memory_mb:.1f} MB")
            print(f"Total Functions: {result.metrics.function_calls}")
            
            if result.bottlenecks:
                print(f"\nBottlenecks:")
                for bottleneck in result.bottlenecks:
                    print(f"  • {bottleneck}")
            
            if result.recommendations:
                print(f"\nRecommendations:")
                for recommendation in result.recommendations:
                    print(f"  • {recommendation}")
            
        else:
            # Show usage examples
            print("\nStockPredictionPro Performance Profiler")
            print("="*40)
            print("\nUsage Examples:")
            print("  # Profile a specific script")
            print("  python scripts/utilities/performance_profiler.py --script scripts/models/train_all_models.py")
            print("\n  # Profile with visualizations")
            print("  python scripts/utilities/performance_profiler.py --script my_script.py --visualize")
            print("\n  # Use decorator in your code:")
            print("  from scripts.utilities.performance_profiler import PerformanceProfiler")
            print("  profiler = PerformanceProfiler()")
            print("  @profiler.profile('my_function')")
            print("  def my_function():")
            print("      # Your code here")
            print("      pass")
        
    except KeyboardInterrupt:
        print("\n❌ Profiling interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Profiling failed: {e}")
        print(f"❌ Profiling failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
