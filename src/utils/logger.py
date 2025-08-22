# ============================================
# StockPredictionPro - src/utils/logger.py
# Advanced logging system with governance and performance tracking
# ============================================

import os
import sys
import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from datetime import datetime
import json
import traceback
from functools import wraps
import time

from .config_loader import get_config

class PerformanceLogger:
    """Performance tracking and timing utilities"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timings: Dict[str, float] = {}
    
    def time_operation(self, operation_name: str):
        """Decorator for timing operations"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.timings[operation_name] = duration
                    
                    # Log performance with custom format
                    self.logger.info(
                        f"PERFORMANCE: {operation_name} completed",
                        extra={
                            'operation': operation_name,
                            'duration': duration,
                            'function': func.__name__,
                            'performance_metric': True
                        }
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.logger.error(
                        f"PERFORMANCE: {operation_name} failed after {duration:.3f}s",
                        extra={
                            'operation': operation_name,
                            'duration': duration,
                            'error': str(e),
                            'performance_metric': True
                        }
                    )
                    raise
            return wrapper
        return decorator
    
    def log_timing(self, operation_name: str, duration: float, **kwargs):
        """Manually log timing information"""
        self.timings[operation_name] = duration
        self.logger.info(
            f"TIMING: {operation_name} - {duration:.3f}s",
            extra={
                'operation': operation_name,
                'duration': duration,
                'performance_metric': True,
                **kwargs
            }
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recorded performance metrics"""
        if not self.timings:
            return {}
        
        return {
            'total_operations': len(self.timings),
            'total_time': sum(self.timings.values()),
            'average_time': sum(self.timings.values()) / len(self.timings),
            'slowest_operation': max(self.timings.items(), key=lambda x: x[1]),
            'fastest_operation': min(self.timings.items(), key=lambda x: x[1]),
            'operations': dict(self.timings)
        }

class AuditLogger:
    """Audit trail logging for governance and compliance"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_user_action(self, user: str, action: str, resource: str, **details):
        """Log user actions for audit trail"""
        self.logger.info(
            f"USER_ACTION: {action} on {resource}",
            extra={
                'audit': True,
                'user': user,
                'action': action,
                'resource': resource,
                'timestamp': datetime.utcnow().isoformat(),
                **details
            }
        )
    
    def log_model_training(self, model_type: str, symbol: str, **metadata):
        """Log model training events"""
        self.logger.info(
            f"MODEL_TRAINING: {model_type} for {symbol}",
            extra={
                'audit': True,
                'event_type': 'model_training',
                'model_type': model_type,
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                **metadata
            }
        )
    
    def log_prediction(self, model_id: str, symbol: str, prediction: Any, **metadata):
        """Log prediction generation"""
        self.logger.info(
            f"PREDICTION: {model_id} for {symbol}",
            extra={
                'audit': True,
                'event_type': 'prediction',
                'model_id': model_id,
                'symbol': symbol,
                'prediction': prediction,
                'timestamp': datetime.utcnow().isoformat(),
                **metadata
            }
        )
    
    def log_trade_signal(self, symbol: str, signal: str, price: float, **metadata):
        """Log trading signals"""
        self.logger.info(
            f"TRADE_SIGNAL: {signal} for {symbol} at {price}",
            extra={
                'audit': True,
                'event_type': 'trade_signal',
                'symbol': symbol,
                'signal': signal,
                'price': price,
                'timestamp': datetime.utcnow().isoformat(),
                **metadata
            }
        )

class ContextFilter(logging.Filter):
    """Add contextual information to log records"""
    
    def __init__(self):
        super().__init__()
        self.context: Dict[str, Any] = {}
    
    def filter(self, record):
        # Add context to log record
        for key, value in self.context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        
        # Add request ID if available
        if not hasattr(record, 'request_id'):
            record.request_id = getattr(record, 'request_id', 'N/A')
        
        # Add module context
        if not hasattr(record, 'component'):
            # Extract component from logger name
            name_parts = record.name.split('.')
            if len(name_parts) >= 2:
                record.component = name_parts[1]  # stockpred.data -> data
            else:
                record.component = 'app'
        
        return True
    
    def set_context(self, **kwargs):
        """Set context for subsequent log messages"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear context"""
        self.context.clear()

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info'):
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)

class StockPredLogger:
    """
    Advanced logging system for StockPredictionPro
    
    Features:
    - Multi-level logging with rotation
    - Performance tracking and timing
    - Audit trail for governance
    - Structured JSON logging for production
    - Context-aware logging
    - Component-specific loggers
    """
    
    def __init__(self):
        self.project_root = self._find_project_root()
        self.logs_dir = self.project_root / "logs"
        self.context_filter = ContextFilter()
        self._ensure_log_directories()
        self._setup_logging()
        
        # Component loggers
        self._loggers: Dict[str, logging.Logger] = {}
        self._performance_loggers: Dict[str, PerformanceLogger] = {}
        self._audit_loggers: Dict[str, AuditLogger] = {}
    
    def _find_project_root(self) -> Path:
        """Find project root directory"""
        current = Path(__file__).resolve()
        markers = ['pyproject.toml', 'setup.py', '.git', 'requirements.txt']
        
        for parent in current.parents:
            if any((parent / marker).exists() for marker in markers):
                return parent
        
        return current.parents[2]
    
    def _ensure_log_directories(self):
        """Create log directories if they don't exist"""
        directories = [
            self.logs_dir,
            self.logs_dir / "audit",
            self.logs_dir / "audit" / "runs",
            self.logs_dir / "audit" / "models"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        try:
            # Try to load logging config from YAML
            logging_config = get_config('logging')
            if logging_config:
                self._setup_from_config(logging_config)
            else:
                self._setup_default_logging()
        except Exception as e:
            print(f"Warning: Failed to load logging config, using defaults: {e}")
            self._setup_default_logging()
    
    def _setup_from_config(self, config: Dict[str, Any]):
        """Setup logging from configuration file"""
        # Update file paths to be absolute
        if 'handlers' in config:
            for handler_name, handler_config in config['handlers'].items():
                if 'filename' in handler_config:
                    filename = handler_config['filename']
                    if not Path(filename).is_absolute():
                        handler_config['filename'] = str(self.logs_dir / filename)
        
        # Apply configuration
        logging.config.dictConfig(config)
        
        # Add context filter to all handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.addFilter(self.context_filter)
    
    def _setup_default_logging(self):
        """Setup default logging configuration"""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(self.context_filter)
        
        # File handler
        file_handler = logging.FileHandler(self.logs_dir / 'app.log')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(self.context_filter)
        
        # Configure root logger
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Set third-party library levels
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger for a specific component
        
        Args:
            name: Logger name (e.g., 'data', 'models', 'trading')
            
        Returns:
            Configured logger instance
        """
        if name in self._loggers:
            return self._loggers[name]
        
        # Create logger with full name
        full_name = f"stockpred.{name}" if not name.startswith('stockpred') else name
        logger = logging.getLogger(full_name)
        
        # Store reference
        self._loggers[name] = logger
        
        return logger
    
    def get_performance_logger(self, name: str) -> PerformanceLogger:
        """Get performance logger for a component"""
        if name not in self._performance_loggers:
            logger = self.get_logger(f"{name}.performance")
            self._performance_loggers[name] = PerformanceLogger(logger)
        
        return self._performance_loggers[name]
    
    def get_audit_logger(self, name: str = "audit") -> AuditLogger:
        """Get audit logger for governance"""
        if name not in self._audit_loggers:
            logger = self.get_logger(f"audit.{name}")
            self._audit_loggers[name] = AuditLogger(logger)
        
        return self._audit_loggers[name]
    
    def set_context(self, **kwargs):
        """Set context for all subsequent log messages"""
        self.context_filter.set_context(**kwargs)
    
    def clear_context(self):
        """Clear logging context"""
        self.context_filter.clear_context()
    
    def log_startup_info(self):
        """Log application startup information"""
        logger = self.get_logger('app')
        
        startup_info = {
            'app_name': 'StockPredictionPro',
            'version': '1.0.0',
            'python_version': sys.version,
            'platform': sys.platform,
            'cwd': os.getcwd(),
            'pid': os.getpid(),
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'logs_directory': str(self.logs_dir)
        }
        
        logger.info("Application starting up", extra=startup_info)
    
    def log_performance_summary(self):
        """Log performance summary for all components"""
        logger = self.get_logger('performance')
        
        for component, perf_logger in self._performance_loggers.items():
            summary = perf_logger.get_performance_summary()
            if summary:
                logger.info(f"Performance summary for {component}", extra=summary)
    
    def setup_request_logging(self, request_id: str, user_id: Optional[str] = None):
        """Setup logging context for a request"""
        context = {'request_id': request_id}
        if user_id:
            context['user_id'] = user_id
        
        self.set_context(**context)
    
    def log_error_with_context(self, logger_name: str, message: str, error: Exception, **context):
        """Log error with full context and stack trace"""
        logger = self.get_logger(logger_name)
        
        error_context = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            **context
        }
        
        logger.error(message, extra=error_context, exc_info=True)

# Global logger instance
logger_system = StockPredLogger()

# Convenience functions for common operations
def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific component"""
    return logger_system.get_logger(name)

def get_performance_logger(name: str) -> PerformanceLogger:
    """Get performance logger for a component"""
    return logger_system.get_performance_logger(name)

def get_audit_logger(name: str = "audit") -> AuditLogger:
    """Get audit logger for governance"""
    return logger_system.get_audit_logger(name)

def set_logging_context(**kwargs):
    """Set context for all subsequent log messages"""
    logger_system.set_context(**kwargs)

def clear_logging_context():
    """Clear logging context"""
    logger_system.clear_context()

def log_startup():
    """Log application startup information"""
    logger_system.log_startup_info()

def time_operation(operation_name: str, logger_name: str = "performance"):
    """Decorator for timing operations"""
    return logger_system.get_performance_logger(logger_name).time_operation(operation_name)

def log_user_action(user: str, action: str, resource: str, **details):
    """Log user actions for audit trail"""
    audit_logger = get_audit_logger()
    audit_logger.log_user_action(user, action, resource, **details)

def log_model_training(model_type: str, symbol: str, **metadata):
    """Log model training events"""
    audit_logger = get_audit_logger()
    audit_logger.log_model_training(model_type, symbol, **metadata)

def log_prediction(model_id: str, symbol: str, prediction: Any, **metadata):
    """Log prediction generation"""
    audit_logger = get_audit_logger()
    audit_logger.log_prediction(model_id, symbol, prediction, **metadata)

def log_trade_signal(symbol: str, signal: str, price: float, **metadata):
    """Log trading signals"""
    audit_logger = get_audit_logger()
    audit_logger.log_trade_signal(symbol, signal, price, **metadata)

# Component-specific logger factories
def get_data_logger() -> logging.Logger:
    """Get logger for data operations"""
    return get_logger('data')

def get_models_logger() -> logging.Logger:
    """Get logger for model operations"""
    return get_logger('models')

def get_trading_logger() -> logging.Logger:
    """Get logger for trading operations"""
    return get_logger('trading')

def get_api_logger() -> logging.Logger:
    """Get logger for API operations"""
    return get_logger('api')

def get_ui_logger() -> logging.Logger:
    """Get logger for UI operations"""
    return get_logger('ui')

# Initialize logging on module import
try:
    log_startup()
except Exception as e:
    print(f"Warning: Failed to initialize logging: {e}")
