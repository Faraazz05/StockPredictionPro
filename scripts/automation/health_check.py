"""
scripts/automation/health_check.py

Comprehensive health monitoring system for StockPredictionPro.
Monitors system resources, data freshness, model availability, API endpoints,
and overall platform health with detailed reporting and alerting.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import logging
import time
import traceback
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import socket

# System monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Database connectivity
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'health_check_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StockPredictionPro.HealthCheck')

# Directory configuration
DATA_DIR = Path('./data')
MODELS_DIR = Path('./models')
LOGS_DIR = Path('./logs')
CONFIG_DIR = Path('./config')
OUTPUTS_DIR = Path('./outputs')

# ============================================
# ENUMS AND DATA MODELS
# ============================================

class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    """Individual health check result"""
    check_name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = None
    timestamp: datetime = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.details is None:
            self.details = {}

@dataclass
class SystemHealthReport:
    """Overall system health report"""
    timestamp: str
    overall_status: HealthStatus
    execution_time: float
    
    # Component health results
    system_resources: HealthCheckResult
    data_health: HealthCheckResult
    model_health: HealthCheckResult
    database_health: HealthCheckResult
    api_health: HealthCheckResult
    service_health: HealthCheckResult
    
    # Summary statistics
    total_checks: int
    healthy_checks: int
    warning_checks: int
    critical_checks: int
    
    # Recommendations
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

# ============================================
# HEALTH CHECK COMPONENTS
# ============================================

class SystemResourceMonitor:
    """Monitor system resources (CPU, Memory, Disk)"""
    
    def __init__(self):
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 85.0,
            'memory_warning': 80.0,
            'memory_critical': 90.0,
            'disk_warning': 80.0,
            'disk_critical': 90.0
        }
    
    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource utilization"""
        start_time = time.time()
        
        try:
            if not HAS_PSUTIL:
                return HealthCheckResult(
                    check_name="system_resources",
                    status=HealthStatus.UNKNOWN,
                    message="psutil not available for system monitoring",
                    execution_time=time.time() - start_time
                )
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # Get process count
            process_count = len(psutil.pids())
            
            # Get load averages (Unix only)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except (AttributeError, OSError):
                pass  # Not available on Windows
            
            details = {
                'cpu_percent': round(cpu_percent, 2),
                'memory_percent': round(memory.percent, 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_usage_percent': round((disk.used / disk.total) * 100, 2),
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'process_count': process_count,
                'load_average': load_avg
            }
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent >= self.thresholds['cpu_critical']:
                status = HealthStatus.CRITICAL
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent >= self.thresholds['cpu_warning']:
                status = HealthStatus.WARNING
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            if memory.percent >= self.thresholds['memory_critical']:
                status = HealthStatus.CRITICAL
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent >= self.thresholds['memory_warning']:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"Memory usage high: {memory.percent:.1f}%")
            
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent >= self.thresholds['disk_critical']:
                status = HealthStatus.CRITICAL
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent >= self.thresholds['disk_warning']:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"Disk usage high: {disk_percent:.1f}%")
            
            message = "System resources healthy" if status == HealthStatus.HEALTHY else "; ".join(issues)
            
            return HealthCheckResult(
                check_name="system_resources",
                status=status,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return HealthCheckResult(
                check_name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"System resource check failed: {e}",
                execution_time=time.time() - start_time
            )

class DataHealthMonitor:
    """Monitor data freshness and quality"""
    
    def __init__(self):
        self.data_dirs = {
            'raw': DATA_DIR / 'raw',
            'processed': DATA_DIR / 'processed',
            'cache': DATA_DIR / 'cache'
        }
    
    def check_data_health(self) -> HealthCheckResult:
        """Check data freshness and availability"""
        start_time = time.time()
        
        try:
            details = {}
            issues = []
            status = HealthStatus.HEALTHY
            
            for dir_type, dir_path in self.data_dirs.items():
                dir_info = self._analyze_directory(dir_path, dir_type)
                details[f"{dir_type}_directory"] = dir_info
                
                # Check for issues
                if dir_info['file_count'] == 0:
                    issues.append(f"No files in {dir_type} directory")
                    status = HealthStatus.WARNING
                
                if dir_info['days_since_newest'] > 7:  # Data older than 7 days
                    issues.append(f"{dir_type} data is {dir_info['days_since_newest']} days old")
                    if status == HealthStatus.HEALTHY:
                        status = HealthStatus.WARNING
                
                if dir_info['days_since_newest'] > 30:  # Data older than 30 days
                    status = HealthStatus.CRITICAL
            
            # Check data validation history
            validation_info = self._check_validation_history()
            details['validation_history'] = validation_info
            
            if validation_info['recent_failures'] > 5:
                issues.append(f"High validation failure rate: {validation_info['recent_failures']} recent failures")
                status = HealthStatus.WARNING
            
            message = "Data health good" if status == HealthStatus.HEALTHY else "; ".join(issues)
            
            return HealthCheckResult(
                check_name="data_health",
                status=status,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Data health check failed: {e}")
            return HealthCheckResult(
                check_name="data_health",
                status=HealthStatus.CRITICAL,
                message=f"Data health check failed: {e}",
                execution_time=time.time() - start_time
            )
    
    def _analyze_directory(self, dir_path: Path, dir_type: str) -> Dict[str, Any]:
        """Analyze directory contents and freshness"""
        if not dir_path.exists():
            return {
                'exists': False,
                'file_count': 0,
                'total_size_mb': 0,
                'newest_file': None,
                'days_since_newest': float('inf')
            }
        
        files = list(dir_path.rglob('*'))
        data_files = [f for f in files if f.is_file() and f.suffix in ['.csv', '.parquet', '.json']]
        
        total_size = sum(f.stat().st_size for f in data_files if f.exists())
        
        newest_file = None
        newest_time = None
        
        if data_files:
            newest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            newest_time = datetime.fromtimestamp(newest_file.stat().st_mtime)
        
        days_since_newest = float('inf')
        if newest_time:
            days_since_newest = (datetime.now() - newest_time).days
        
        return {
            'exists': True,
            'file_count': len(data_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'newest_file': newest_file.name if newest_file else None,
            'newest_file_time': newest_time.isoformat() if newest_time else None,
            'days_since_newest': days_since_newest
        }
    
    def _check_validation_history(self) -> Dict[str, Any]:
        """Check recent data validation history"""
        try:
            validation_log = LOGS_DIR / 'data_validation.log'
            
            if not validation_log.exists():
                return {
                    'log_exists': False,
                    'recent_failures': 0,
                    'last_validation': None
                }
            
            # Simple check - count "FAILED" in recent log entries
            with open(validation_log, 'r') as f:
                recent_lines = f.readlines()[-100:]  # Last 100 lines
            
            recent_failures = sum(1 for line in recent_lines if 'FAILED' in line or 'ERROR' in line)
            
            return {
                'log_exists': True,
                'recent_failures': recent_failures,
                'last_validation': datetime.fromtimestamp(validation_log.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            logger.debug(f"Could not check validation history: {e}")
            return {
                'log_exists': False,
                'recent_failures': 0,
                'last_validation': None,
                'error': str(e)
            }

class ModelHealthMonitor:
    """Monitor model availability and performance"""
    
    def __init__(self):
        self.model_dirs = {
            'trained': MODELS_DIR / 'trained',
            'production': MODELS_DIR / 'production',
            'experiments': MODELS_DIR / 'experiments'
        }
    
    def check_model_health(self) -> HealthCheckResult:
        """Check model availability and health"""
        start_time = time.time()
        
        try:
            details = {}
            issues = []
            status = HealthStatus.HEALTHY
            
            # Check model directories
            for dir_type, dir_path in self.model_dirs.items():
                dir_info = self._analyze_model_directory(dir_path)
                details[f"{dir_type}_models"] = dir_info
                
                if dir_type == 'production' and dir_info['model_count'] == 0:
                    issues.append("No production models available")
                    status = HealthStatus.CRITICAL
            
            # Check model registry
            registry_info = self._check_model_registry()
            details['model_registry'] = registry_info
            
            if not registry_info['registry_exists']:
                issues.append("Model registry not found")
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
            
            # Test model loading
            loading_results = self._test_model_loading()
            details['model_loading_tests'] = loading_results
            
            if loading_results['failed_models'] > 0:
                issues.append(f"{loading_results['failed_models']} models failed to load")
                status = HealthStatus.WARNING
            
            message = "Models healthy" if status == HealthStatus.HEALTHY else "; ".join(issues)
            
            return HealthCheckResult(
                check_name="model_health",
                status=status,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return HealthCheckResult(
                check_name="model_health",
                status=HealthStatus.CRITICAL,
                message=f"Model health check failed: {e}",
                execution_time=time.time() - start_time
            )
    
    def _analyze_model_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Analyze model directory contents"""
        if not dir_path.exists():
            return {
                'exists': False,
                'model_count': 0,
                'total_size_mb': 0,
                'newest_model': None,
                'days_since_newest': float('inf')
            }
        
        model_files = list(dir_path.glob('*.pkl')) + list(dir_path.glob('*.joblib'))
        
        total_size = sum(f.stat().st_size for f in model_files if f.exists())
        
        newest_model = None
        newest_time = None
        
        if model_files:
            newest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            newest_time = datetime.fromtimestamp(newest_model.stat().st_mtime)
        
        days_since_newest = float('inf')
        if newest_time:
            days_since_newest = (datetime.now() - newest_time).days
        
        return {
            'exists': True,
            'model_count': len(model_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'newest_model': newest_model.name if newest_model else None,
            'days_since_newest': days_since_newest
        }
    
    def _check_model_registry(self) -> Dict[str, Any]:
        """Check model registry status"""
        registry_path = MODELS_DIR / 'production' / 'model_registry.json'
        
        if not registry_path.exists():
            return {
                'registry_exists': False,
                'active_models': 0,
                'total_models': 0
            }
        
        try:
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            active_models = sum(1 for model_info in registry_data.values() 
                              if model_info.get('deployment_status') == 'active')
            
            return {
                'registry_exists': True,
                'active_models': active_models,
                'total_models': len(registry_data),
                'last_updated': datetime.fromtimestamp(registry_path.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            return {
                'registry_exists': True,
                'error': str(e),
                'active_models': 0,
                'total_models': 0
            }
    
    def _test_model_loading(self) -> Dict[str, Any]:
        """Test loading of production models"""
        try:
            import joblib
            
            production_models = list((MODELS_DIR / 'production').glob('*.pkl'))
            
            if not production_models:
                return {
                    'tested_models': 0,
                    'successful_models': 0,
                    'failed_models': 0,
                    'test_results': []
                }
            
            # Test loading first 3 models to avoid long execution time
            test_models = production_models[:3]
            test_results = []
            successful = 0
            failed = 0
            
            for model_path in test_models:
                try:
                    model = joblib.load(model_path)
                    test_results.append({
                        'model': model_path.name,
                        'status': 'success',
                        'type': type(model).__name__
                    })
                    successful += 1
                    
                except Exception as e:
                    test_results.append({
                        'model': model_path.name,
                        'status': 'failed',
                        'error': str(e)
                    })
                    failed += 1
            
            return {
                'tested_models': len(test_models),
                'successful_models': successful,
                'failed_models': failed,
                'test_results': test_results
            }
            
        except ImportError:
            return {
                'tested_models': 0,
                'successful_models': 0,
                'failed_models': 0,
                'error': 'joblib not available for model testing'
            }

class DatabaseHealthMonitor:
    """Monitor database connectivity and health"""
    
    def __init__(self):
        self.db_configs = self._load_db_configs()
    
    def check_database_health(self) -> HealthCheckResult:
        """Check database connectivity and basic operations"""
        start_time = time.time()
        
        if not HAS_SQLALCHEMY:
            return HealthCheckResult(
                check_name="database_health",
                status=HealthStatus.UNKNOWN,
                message="SQLAlchemy not available for database checks",
                execution_time=time.time() - start_time
            )
        
        try:
            details = {}
            issues = []
            status = HealthStatus.HEALTHY
            
            for db_name, db_url in self.db_configs.items():
                db_result = self._test_database_connection(db_name, db_url)
                details[f"database_{db_name}"] = db_result
                
                if not db_result['connected']:
                    issues.append(f"Cannot connect to {db_name} database")
                    status = HealthStatus.CRITICAL
                elif db_result.get('slow_response', False):
                    issues.append(f"{db_name} database responding slowly")
                    if status == HealthStatus.HEALTHY:
                        status = HealthStatus.WARNING
            
            if not self.db_configs:
                # Check for SQLite files as fallback
                sqlite_files = list(Path('.').glob('*.db'))
                if sqlite_files:
                    for db_file in sqlite_files[:3]:  # Check first 3 SQLite files
                        db_result = self._test_database_connection(db_file.stem, f"sqlite:///{db_file}")
                        details[f"sqlite_{db_file.stem}"] = db_result
                else:
                    details['no_databases'] = True
                    issues.append("No database configurations found")
                    status = HealthStatus.WARNING
            
            message = "Database connections healthy" if status == HealthStatus.HEALTHY else "; ".join(issues)
            
            return HealthCheckResult(
                check_name="database_health",
                status=status,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return HealthCheckResult(
                check_name="database_health",
                status=HealthStatus.CRITICAL,
                message=f"Database health check failed: {e}",
                execution_time=time.time() - start_time
            )
    
    def _load_db_configs(self) -> Dict[str, str]:
        """Load database configurations"""
        configs = {}
        
        # Try to load from environment variables
        if os.getenv('DATABASE_URL'):
            configs['primary'] = os.getenv('DATABASE_URL')
        
        # Try to load from config files
        config_files = [
            CONFIG_DIR / 'database.json',
            CONFIG_DIR / 'config.json',
            Path('.env')
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    if config_file.suffix == '.json':
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                        
                        if 'database' in config_data:
                            db_config = config_data['database']
                            if isinstance(db_config, dict) and 'url' in db_config:
                                configs['config'] = db_config['url']
                            elif isinstance(db_config, str):
                                configs['config'] = db_config
                    
                except Exception as e:
                    logger.debug(f"Could not load config from {config_file}: {e}")
        
        return configs
    
    def _test_database_connection(self, db_name: str, db_url: str) -> Dict[str, Any]:
        """Test individual database connection"""
        try:
            start_time = time.time()
            
            engine = create_engine(db_url, connect_args={'timeout': 10})
            
            with engine.connect() as conn:
                # Test basic query
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            connection_time = time.time() - start_time
            
            return {
                'connected': True,
                'connection_time': round(connection_time, 3),
                'slow_response': connection_time > 2.0,
                'database_type': engine.dialect.name
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'connection_time': None
            }

class ApiHealthMonitor:
    """Monitor API endpoints and external services"""
    
    def __init__(self):
        self.api_endpoints = self._get_api_endpoints()
    
    def check_api_health(self) -> HealthCheckResult:
        """Check API endpoint health"""
        start_time = time.time()
        
        try:
            details = {}
            issues = []
            status = HealthStatus.HEALTHY
            
            if not self.api_endpoints:
                return HealthCheckResult(
                    check_name="api_health",
                    status=HealthStatus.UNKNOWN,
                    message="No API endpoints configured for monitoring",
                    details={'endpoints_configured': 0},
                    execution_time=time.time() - start_time
                )
            
            for endpoint_name, endpoint_config in self.api_endpoints.items():
                api_result = self._test_api_endpoint(endpoint_name, endpoint_config)
                details[f"endpoint_{endpoint_name}"] = api_result
                
                if not api_result['available']:
                    issues.append(f"{endpoint_name} API unavailable")
                    status = HealthStatus.CRITICAL
                elif api_result.get('slow_response', False):
                    issues.append(f"{endpoint_name} API responding slowly")
                    if status == HealthStatus.HEALTHY:
                        status = HealthStatus.WARNING
            
            message = "API endpoints healthy" if status == HealthStatus.HEALTHY else "; ".join(issues)
            
            return HealthCheckResult(
                check_name="api_health",
                status=status,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return HealthCheckResult(
                check_name="api_health",
                status=HealthStatus.CRITICAL,
                message=f"API health check failed: {e}",
                execution_time=time.time() - start_time
            )
    
    def _get_api_endpoints(self) -> Dict[str, Dict[str, str]]:
        """Get configured API endpoints"""
        endpoints = {}
        
        # Common financial data APIs
        default_endpoints = {
            'yahoo_finance': {
                'url': 'https://query1.finance.yahoo.com/v8/finance/chart/AAPL',
                'timeout': 10
            },
            'local_api': {
                'url': 'http://localhost:8000/health',
                'timeout': 5
            }
        }
        
        # Try to load from config
        config_file = CONFIG_DIR / 'api_endpoints.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    endpoints.update(json.load(f))
            except Exception as e:
                logger.debug(f"Could not load API config: {e}")
        
        # Use defaults if no config found
        if not endpoints:
            endpoints = default_endpoints
        
        return endpoints
    
    def _test_api_endpoint(self, endpoint_name: str, endpoint_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test individual API endpoint"""
        try:
            url = endpoint_config['url']
            timeout = endpoint_config.get('timeout', 10)
            
            start_time = time.time()
            
            response = requests.get(url, timeout=timeout)
            
            response_time = time.time() - start_time
            
            return {
                'available': response.status_code < 500,
                'status_code': response.status_code,
                'response_time': round(response_time, 3),
                'slow_response': response_time > 5.0,
                'content_length': len(response.content)
            }
            
        except requests.exceptions.Timeout:
            return {
                'available': False,
                'error': 'Timeout',
                'response_time': None
            }
        except requests.exceptions.ConnectionError:
            return {
                'available': False,
                'error': 'Connection Error',
                'response_time': None
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e),
                'response_time': None
            }

class ServiceHealthMonitor:
    """Monitor system services and processes"""
    
    def __init__(self):
        self.services_to_check = self._get_services_config()
    
    def check_service_health(self) -> HealthCheckResult:
        """Check system services and processes"""
        start_time = time.time()
        
        try:
            details = {}
            issues = []
            status = HealthStatus.HEALTHY
            
            # Check disk space
            disk_info = self._check_disk_space()
            details['disk_space'] = disk_info
            
            if disk_info['critical_partitions']:
                issues.append(f"Critical disk space on {', '.join(disk_info['critical_partitions'])}")
                status = HealthStatus.CRITICAL
            elif disk_info['warning_partitions']:
                issues.append(f"Low disk space on {', '.join(disk_info['warning_partitions'])}")
                status = HealthStatus.WARNING
            
            # Check log file sizes
            log_info = self._check_log_files()
            details['log_files'] = log_info
            
            if log_info['large_logs']:
                issues.append(f"Large log files detected: {', '.join(log_info['large_logs'])}")
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
            
            # Check Python processes
            python_info = self._check_python_processes()
            details['python_processes'] = python_info
            
            # Check network connectivity
            network_info = self._check_network_connectivity()
            details['network'] = network_info
            
            if not network_info['internet_available']:
                issues.append("Internet connectivity issues detected")
                status = HealthStatus.WARNING
            
            message = "Services healthy" if status == HealthStatus.HEALTHY else "; ".join(issues)
            
            return HealthCheckResult(
                check_name="service_health",
                status=status,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Service health check failed: {e}")
            return HealthCheckResult(
                check_name="service_health",
                status=HealthStatus.CRITICAL,
                message=f"Service health check failed: {e}",
                execution_time=time.time() - start_time
            )
    
    def _get_services_config(self) -> List[str]:
        """Get list of services to monitor"""
        # Default services relevant to StockPredictionPro
        return [
            'cron',
            'nginx',
            'redis-server',
            'postgresql',
            'mysql'
        ]
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space on all mounted partitions"""
        if not HAS_PSUTIL:
            return {'error': 'psutil not available'}
        
        try:
            partitions = psutil.disk_partitions()
            warning_partitions = []
            critical_partitions = []
            partition_info = []
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    percent_used = (usage.used / usage.total) * 100
                    
                    partition_info.append({
                        'mountpoint': partition.mountpoint,
                        'device': partition.device,
                        'fstype': partition.fstype,
                        'percent_used': round(percent_used, 2),
                        'free_gb': round(usage.free / (1024**3), 2),
                        'total_gb': round(usage.total / (1024**3), 2)
                    })
                    
                    if percent_used >= 95:
                        critical_partitions.append(partition.mountpoint)
                    elif percent_used >= 80:
                        warning_partitions.append(partition.mountpoint)
                        
                except PermissionError:
                    continue  # Skip partitions we can't access
                    
            return {
                'partitions': partition_info,
                'warning_partitions': warning_partitions,
                'critical_partitions': critical_partitions
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _check_log_files(self) -> Dict[str, Any]:
        """Check log file sizes"""
        try:
            log_files = list(LOGS_DIR.glob('*.log'))
            large_logs = []
            log_info = []
            
            for log_file in log_files:
                try:
                    size_mb = log_file.stat().st_size / (1024 * 1024)
                    
                    log_info.append({
                        'file': log_file.name,
                        'size_mb': round(size_mb, 2),
                        'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                    })
                    
                    if size_mb > 100:  # Files larger than 100MB
                        large_logs.append(log_file.name)
                        
                except Exception:
                    continue
            
            return {
                'log_files': log_info,
                'large_logs': large_logs,
                'total_logs': len(log_files)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _check_python_processes(self) -> Dict[str, Any]:
        """Check Python processes related to the application"""
        if not HAS_PSUTIL:
            return {'error': 'psutil not available'}
        
        try:
            python_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        
                        if 'stockpredictionpro' in cmdline.lower() or 'scripts/' in cmdline.lower():
                            python_processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cmdline': cmdline[:100],  # Limit length
                                'cpu_percent': round(proc.info['cpu_percent'], 2),
                                'memory_percent': round(proc.info['memory_percent'], 2)
                            })
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                'python_processes': python_processes,
                'process_count': len(python_processes)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            # Test DNS resolution
            try:
                socket.gethostbyname('google.com')
                dns_working = True
            except socket.gaierror:
                dns_working = False
            
            # Test internet connectivity
            try:
                response = requests.get('https://httpbin.org/ip', timeout=10)
                internet_available = response.status_code == 200
            except:
                internet_available = False
            
            return {
                'dns_working': dns_working,
                'internet_available': internet_available
            }
            
        except Exception as e:
            return {'error': str(e)}

# ============================================
# MAIN HEALTH CHECK ORCHESTRATOR
# ============================================

class HealthCheckOrchestrator:
    """Main orchestrator for all health checks"""
    
    def __init__(self):
        self.monitors = {
            'system_resources': SystemResourceMonitor(),
            'data_health': DataHealthMonitor(),
            'model_health': ModelHealthMonitor(),
            'database_health': DatabaseHealthMonitor(),
            'api_health': ApiHealthMonitor(),
            'service_health': ServiceHealthMonitor()
        }
    
    def run_all_health_checks(self) -> SystemHealthReport:
        """Run all health checks and generate comprehensive report"""
        logger.info("🏥 Starting comprehensive health check...")
        start_time = time.time()
        
        # Run all checks
        check_results = {}
        
        for check_name, monitor in self.monitors.items():
            try:
                logger.info(f"Running {check_name} check...")
                check_method = getattr(monitor, f'check_{check_name}')
                result = check_method()
                check_results[check_name] = result
                
                status_emoji = {
                    HealthStatus.HEALTHY: "✅",
                    HealthStatus.WARNING: "⚠️",
                    HealthStatus.CRITICAL: "🚨",
                    HealthStatus.UNKNOWN: "❓"
                }
                
                logger.info(f"{status_emoji[result.status]} {check_name}: {result.message}")
                
            except Exception as e:
                logger.error(f"Failed to run {check_name} check: {e}")
                check_results[check_name] = HealthCheckResult(
                    check_name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Check failed: {e}"
                )
        
        # Calculate overall status
        overall_status = self._calculate_overall_status(check_results)
        
        # Count status types
        status_counts = {
            'healthy': sum(1 for r in check_results.values() if r.status == HealthStatus.HEALTHY),
            'warning': sum(1 for r in check_results.values() if r.status == HealthStatus.WARNING),
            'critical': sum(1 for r in check_results.values() if r.status == HealthStatus.CRITICAL)
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(check_results)
        
        # Create comprehensive report
        execution_time = time.time() - start_time
        
        report = SystemHealthReport(
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            execution_time=execution_time,
            system_resources=check_results['system_resources'],
            data_health=check_results['data_health'],
            model_health=check_results['model_health'],
            database_health=check_results['database_health'],
            api_health=check_results['api_health'],
            service_health=check_results['service_health'],
            total_checks=len(check_results),
            healthy_checks=status_counts['healthy'],
            warning_checks=status_counts['warning'],
            critical_checks=status_counts['critical'],
            recommendations=recommendations
        )
        
        # Save report
        self._save_health_report(report)
        
        # Print summary
        self._print_health_summary(report)
        
        logger.info(f"🏥 Health check completed in {execution_time:.2f} seconds")
        
        return report
    
    def _calculate_overall_status(self, check_results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Calculate overall system health status"""
        statuses = [result.status for result in check_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.WARNING  # Treat unknown as warning
        else:
            return HealthStatus.HEALTHY
    
    def _generate_recommendations(self, check_results: Dict[str, HealthCheckResult]) -> List[str]:
        """Generate actionable recommendations based on check results"""
        recommendations = []
        
        for check_name, result in check_results.items():
            if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                if check_name == 'system_resources':
                    if 'CPU usage' in result.message:
                        recommendations.append("Consider scaling CPU resources or optimizing processes")
                    if 'Memory usage' in result.message:
                        recommendations.append("Monitor memory usage and consider adding RAM")
                    if 'Disk usage' in result.message:
                        recommendations.append("Clean up disk space or expand storage capacity")
                
                elif check_name == 'data_health':
                    if 'data is' in result.message and 'days old' in result.message:
                        recommendations.append("Update data sources and check data ingestion pipeline")
                    if 'validation failure' in result.message:
                        recommendations.append("Review data validation rules and data quality")
                
                elif check_name == 'model_health':
                    if 'No production models' in result.message:
                        recommendations.append("Deploy trained models to production environment")
                    if 'failed to load' in result.message:
                        recommendations.append("Check model file integrity and dependencies")
                
                elif check_name == 'database_health':
                    if 'Cannot connect' in result.message:
                        recommendations.append("Check database service status and connection settings")
                    if 'responding slowly' in result.message:
                        recommendations.append("Optimize database queries and consider indexing")
                
                elif check_name == 'api_health':
                    if 'unavailable' in result.message:
                        recommendations.append("Check API service status and network connectivity")
                    if 'responding slowly' in result.message:
                        recommendations.append("Monitor API performance and consider caching")
                
                elif check_name == 'service_health':
                    if 'disk space' in result.message:
                        recommendations.append("Clean up temporary files and implement log rotation")
                    if 'connectivity' in result.message:
                        recommendations.append("Check network configuration and firewall settings")
        
        # Add general recommendations if system is unhealthy
        if not recommendations:
            recommendations.append("System appears healthy - continue regular monitoring")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _save_health_report(self, report: SystemHealthReport) -> None:
        """Save health report to file"""
        try:
            # Save detailed report
            report_file = OUTPUTS_DIR / 'reports' / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report.save(report_file)
            
            # Save latest report (overwrite)
            latest_report = OUTPUTS_DIR / 'reports' / 'health_report_latest.json'
            report.save(latest_report)
            
            logger.info(f"💾 Health report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save health report: {e}")
    
    def _print_health_summary(self, report: SystemHealthReport) -> None:
        """Print health check summary to console"""
        status_colors = {
            HealthStatus.HEALTHY: "🟢",
            HealthStatus.WARNING: "🟡", 
            HealthStatus.CRITICAL: "🔴",
            HealthStatus.UNKNOWN: "⚪"
        }
        
        print("\n" + "="*60)
        print("STOCKPREDICTIONPRO SYSTEM HEALTH CHECK")
        print("="*60)
        print(f"Timestamp: {report.timestamp}")
        print(f"Overall Status: {status_colors[report.overall_status]} {report.overall_status.value.upper()}")
        print(f"Execution Time: {report.execution_time:.2f} seconds")
        
        print(f"\nCheck Results:")
        print("-" * 40)
        
        checks = [
            ("System Resources", report.system_resources),
            ("Data Health", report.data_health),
            ("Model Health", report.model_health),
            ("Database Health", report.database_health),
            ("API Health", report.api_health),
            ("Service Health", report.service_health)
        ]
        
        for check_display_name, check_result in checks:
            status_icon = status_colors[check_result.status]
            print(f"{status_icon} {check_display_name:20}: {check_result.message}")
        
        print(f"\nSummary:")
        print(f"  ✅ Healthy: {report.healthy_checks}")
        print(f"  ⚠️ Warnings: {report.warning_checks}")
        print(f"  🚨 Critical: {report.critical_checks}")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='StockPredictionPro Health Check')
    parser.add_argument('--check', choices=['system', 'data', 'models', 'database', 'api', 'services'],
                       help='Run specific health check only')
    parser.add_argument('--output', help='Output file for health report (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (errors only)')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run health checks
    orchestrator = HealthCheckOrchestrator()
    
    if args.check:
        # Run specific check only
        check_name = args.check + '_health' if not args.check.endswith('_health') else args.check
        if args.check == 'system':
            check_name = 'system_resources'
        
        if check_name in orchestrator.monitors:
            monitor = orchestrator.monitors[check_name]
            check_method = getattr(monitor, f'check_{check_name}')
            result = check_method()
            
            print(f"\n{args.check.title()} Health Check:")
            print(f"Status: {result.status.value.upper()}")
            print(f"Message: {result.message}")
            if args.verbose and result.details:
                print(f"Details: {json.dumps(result.details, indent=2, default=str)}")
        else:
            print(f"Unknown health check: {args.check}")
            sys.exit(1)
    else:
        # Run all health checks
        report = orchestrator.run_all_health_checks()
        
        # Save to custom output file if specified
        if args.output:
            report.save(Path(args.output))
            print(f"\nHealth report saved to: {args.output}")
        
        # Exit with appropriate code
        if report.overall_status == HealthStatus.HEALTHY:
            sys.exit(0)
        elif report.overall_status == HealthStatus.WARNING:
            sys.exit(1)
        else:  # CRITICAL or UNKNOWN
            sys.exit(2)

if __name__ == '__main__':
    main()
