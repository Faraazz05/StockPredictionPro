# ============================================
# StockPredictionPro - src/utils/config_loader.py
# Advanced configuration management system
# ============================================

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime
import logging

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ConfigMetadata:
    """Configuration metadata for governance and audit"""
    config_name: str
    file_path: Path
    last_modified: datetime
    version: str = "1.0.0"
    environment: str = "development"
    
class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass

class ConfigLoader:
    """
    Advanced configuration management for StockPredictionPro
    
    Features:
    - Multi-environment support (dev/prod/test)
    - YAML + Environment variable integration
    - Configuration validation and defaults
    - Hot reloading during development
    - Audit trail and governance
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigLoader
        
        Args:
            config_dir: Path to configuration directory. If None, auto-detected.
        """
        self.project_root = self._find_project_root()
        self.config_dir = Path(config_dir) if config_dir else (self.project_root / "config")
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Configuration storage
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, ConfigMetadata] = {}
        
        # Validation schemas
        self._validation_schemas: Dict[str, Dict] = {}
        
        # Logger setup (basic, will be enhanced by logger.py)
        self.logger = logging.getLogger(__name__)
        
        # Initialize configurations
        self._ensure_config_directory()
        self._load_all_configs()
        
    def _find_project_root(self) -> Path:
        """Find project root directory by looking for key files"""
        current = Path(__file__).resolve()
        
        # Look for project markers
        markers = ['pyproject.toml', 'setup.py', '.git', 'requirements.txt']
        
        for parent in current.parents:
            if any((parent / marker).exists() for marker in markers):
                return parent
                
        # Fallback to 3 levels up from this file
        return current.parents[2]
    
    def _ensure_config_directory(self):
        """Ensure config directory exists and create if needed"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep if directory is empty
        if not any(self.config_dir.iterdir()):
            (self.config_dir / '.gitkeep').touch()
    
    def _load_all_configs(self):
        """Load all configuration files"""
        config_files = [
            "app_config.yaml",
            "model_config.yaml",
            "indicators_config.yaml", 
            "trading_config.yaml",
            "api_config.yaml",
            "logging.yaml"
        ]
        
        for config_file in config_files:
            config_name = config_file.replace('.yaml', '').replace('.yml', '')
            try:
                self._load_config_file(config_file, config_name)
            except Exception as e:
                self.logger.warning(f"Failed to load {config_file}: {e}")
                self._create_default_config(config_file, config_name)
    
    def _load_config_file(self, filename: str, config_name: str):
        """Load a single configuration file"""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load YAML content
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML in {filename}: {e}")
        
        # Process environment variables
        config_data = self._process_environment_variables(config_data)
        
        # Validate configuration
        if config_name in self._validation_schemas:
            self._validate_config(config_data, config_name)
        
        # Store configuration
        self.configs[config_name] = config_data
        
        # Store metadata
        stat = config_path.stat()
        self.metadata[config_name] = ConfigMetadata(
            config_name=config_name,
            file_path=config_path,
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            environment=self.environment
        )
        
        self.logger.debug(f"Loaded configuration: {config_name}")
    
    def _process_environment_variables(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process environment variable substitutions in config
        
        Supports formats:
        - ${VAR_NAME}
        - ${VAR_NAME:default_value}
        """
        def process_value(value):
            if isinstance(value, str):
                # Handle environment variable substitution
                if value.startswith('${') and value.endswith('}'):
                    env_spec = value[2:-1]  # Remove ${ and }
                    
                    if ':' in env_spec:
                        var_name, default_value = env_spec.split(':', 1)
                        return os.getenv(var_name, default_value)
                    else:
                        return os.getenv(env_spec, value)  # Return original if not found
                        
                return value
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            else:
                return value
        
        return process_value(config_data)
    
    def _validate_config(self, config_data: Dict[str, Any], config_name: str):
        """Validate configuration against schema"""
        schema = self._validation_schemas[config_name]
        
        # Basic validation implementation
        required_keys = schema.get('required', [])
        for key in required_keys:
            if key not in config_data:
                raise ConfigValidationError(f"Missing required key '{key}' in {config_name}")
    
    def _create_default_config(self, filename: str, config_name: str):
        """Create default configuration file if it doesn't exist"""
        self.logger.info(f"Creating default configuration: {filename}")
        
        # Default configurations for each file
        defaults = {
            "app_config.yaml": self._get_app_config_defaults(),
            "model_config.yaml": self._get_model_config_defaults(),
            "indicators_config.yaml": self._get_indicators_config_defaults(),
            "trading_config.yaml": self._get_trading_config_defaults(),
            "api_config.yaml": self._get_api_config_defaults(),
            "logging.yaml": self._get_logging_config_defaults()
        }
        
        if filename not in defaults:
            self.logger.warning(f"No default configuration available for {filename}")
            return
        
        # Create the configuration file
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(defaults[filename], f, default_flow_style=False, indent=2, sort_keys=False)
            
            # Load the newly created config
            self._load_config_file(filename, config_name)
            self.logger.info(f"Created default configuration: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create default config {filename}: {e}")
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get complete configuration by name
        
        Args:
            config_name: Name of configuration (without .yaml extension)
            
        Returns:
            Complete configuration dictionary
        """
        if config_name not in self.configs:
            self.logger.warning(f"Configuration '{config_name}' not found")
            return {}
        
        return self.configs[config_name].copy()  # Return copy to prevent modification
    
    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        Get specific configuration value using dot notation
        
        Args:
            config_name: Name of configuration
            key_path: Dot-separated path to key (e.g., 'app.name' or 'data.sources.yahoo_finance.enabled')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config = self.get_config(config_name)
        
        # Navigate through nested keys
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, config_name: str, key_path: str, value: Any):
        """
        Set configuration value (runtime only, not persisted)
        
        Args:
            config_name: Name of configuration
            key_path: Dot-separated path to key
            value: Value to set
        """
        if config_name not in self.configs:
            self.configs[config_name] = {}
        
        # Navigate and create nested structure
        keys = key_path.split('.')
        current = self.configs[config_name]
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        self.logger.debug(f"Set config {config_name}.{key_path} = {value}")
    
    def reload_config(self, config_name: Optional[str] = None):
        """
        Reload configuration files from disk
        
        Args:
            config_name: Specific config to reload, or None for all
        """
        if config_name:
            # Reload specific config
            filename = f"{config_name}.yaml"
            try:
                self._load_config_file(filename, config_name)
                self.logger.info(f"Reloaded configuration: {config_name}")
            except Exception as e:
                self.logger.error(f"Failed to reload {config_name}: {e}")
        else:
            # Reload all configs
            self.configs.clear()
            self.metadata.clear()
            self._load_all_configs()
            self.logger.info("Reloaded all configurations")
    
    def get_metadata(self, config_name: str) -> Optional[ConfigMetadata]:
        """Get metadata for configuration"""
        return self.metadata.get(config_name)
    
    def list_configs(self) -> List[str]:
        """Get list of loaded configuration names"""
        return list(self.configs.keys())
    
    def validate_all_configs(self) -> Dict[str, List[str]]:
        """
        Validate all loaded configurations
        
        Returns:
            Dictionary with config names as keys and list of validation errors as values
        """
        validation_results = {}
        
        for config_name in self.configs:
            errors = []
            try:
                if config_name in self._validation_schemas:
                    self._validate_config(self.configs[config_name], config_name)
            except ConfigValidationError as e:
                errors.append(str(e))
            
            validation_results[config_name] = errors
        
        return validation_results
    
    def export_config(self, config_name: str, format: str = 'yaml') -> str:
        """
        Export configuration in specified format
        
        Args:
            config_name: Name of configuration to export
            format: Export format ('yaml', 'json')
            
        Returns:
            Configuration as formatted string
        """
        config = self.get_config(config_name)
        
        if format.lower() == 'json':
            return json.dumps(config, indent=2, default=str)
        elif format.lower() == 'yaml':
            return yaml.dump(config, default_flow_style=False, indent=2, sort_keys=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment and configuration information"""
        return {
            'environment': self.environment,
            'config_directory': str(self.config_dir),
            'project_root': str(self.project_root),
            'loaded_configs': list(self.configs.keys()),
            'python_version': os.sys.version,
            'config_loader_version': '1.0.0'
        }
    
    # Default configuration templates
    def _get_app_config_defaults(self) -> Dict[str, Any]:
        """Get default app configuration"""
        return {
            'app': {
                'name': 'StockPredictionPro',
                'version': '1.0.0',
                'environment': '${ENVIRONMENT:development}',
                'debug': True,
                'timezone': '${APP_TIMEZONE:Asia/Kolkata}'
            },
            'data_sources': {
                'yahoo_finance': {
                    'enabled': True,
                    'timeout': 30,
                    'retries': 3,
                    'cache_ttl': 3600
                },
                'alpha_vantage': {
                    'enabled': True,
                    'api_key': '${ALPHA_VANTAGE_KEY:}',
                    'rate_limit': 5
                }
            },
            'markets': {
                'default_market': '${DEFAULT_MARKET:NSE}',
                'default_symbols': '${DEFAULT_TICKERS:AAPL,MSFT,INFY.NS,TCS.NS}'
            },
            'caching': {
                'enabled': True,
                'backend': {'type': 'memory'},
                'ttl': {'stock_data': 3600}
            }
        }
    
    def _get_model_config_defaults(self) -> Dict[str, Any]:
        """Get default model configuration"""
        return {
            'global': {
                'random_seed': 42,
                'validation': {
                    'method': 'time_series_split',
                    'n_splits': 5,
                    'test_size': 0.2
                }
            },
            'regression': {
                'linear': {'enabled': True, 'parameters': {'fit_intercept': True}},
                'multiple': {'enabled': True, 'parameters': {'fit_intercept': True}},
                'polynomial': {
                    'enabled': True,
                    'polynomial_features': {'degrees': [2, 3]},
                    'regularization': {'alpha_range': [0.001, 0.01, 0.1, 1.0]}
                }
            },
            'classification': {
                'targets': {
                    'ternary': {'enabled': True, 'classes': ['Down', 'Sideways', 'Up'], 'thresholds': [-0.01, 0.01]}
                },
                'logistic': {'enabled': True, 'parameters': {'class_weight': 'balanced'}},
                'svm': {'enabled': True, 'parameters': {'kernel': 'rbf', 'class_weight': 'balanced'}},
                'random_forest': {'enabled': True, 'parameters': {'n_estimators_range': [50, 100, 200]}}
            }
        }
    
    def _get_indicators_config_defaults(self) -> Dict[str, Any]:
        """Get default indicators configuration"""
        return {
            'global': {'calculation': {'precision': 6, 'fill_method': 'forward'}},
            'trend': {
                'sma': {'enabled': True, 'periods': [5, 10, 20, 50], 'default_periods': [20, 50]},
                'ema': {'enabled': True, 'periods': [12, 26, 50], 'default_periods': [12, 26]}
            },
            'momentum': {
                'rsi': {'enabled': True, 'period': 14, 'overbought': 70, 'oversold': 30},
                'macd': {'enabled': True, 'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            },
            'volatility': {
                'bollinger_bands': {'enabled': True, 'period': 20, 'std_dev': 2.0},
                'atr': {'enabled': True, 'period': 14}
            },
            'volume': {
                'obv': {'enabled': True},
                'volume_roc': {'enabled': True, 'period': 12}
            }
        }
    
    def _get_trading_config_defaults(self) -> Dict[str, Any]:
        """Get default trading configuration"""
        return {
            'global': {
                'universe': {'default_symbols': ['AAPL', 'MSFT', 'INFY.NS'], 'max_positions': 10},
                'base_currency': 'INR'
            },
            'signals': {
                'sources': {
                    'model_predictions': {'enabled': True, 'weight': 0.7},
                    'technical_indicators': {'enabled': True, 'weight': 0.3}
                }
            },
            'position_sizing': {'method': 'fixed_percentage', 'fixed_percentage': {'default_size': 0.1}},
            'risk_management': {
                'stop_loss': {'enabled': True, 'percentage_stops': {'long_stop': 0.02}},
                'take_profit': {'enabled': True, 'fixed_targets': {'target_1': 0.03}}
            },
            'costs': {
                'total_cost_estimate': {'buy_side': 0.0015, 'sell_side': 0.0025, 'round_trip': 0.004}
            }
        }
    
    def _get_api_config_defaults(self) -> Dict[str, Any]:
        """Get default API configuration"""
        return {
            'server': {'host': '0.0.0.0', 'port': 8000, 'workers': 4},
            'cors': {'enabled': True, 'allow_origins': ['http://localhost:8501']},
            'rate_limiting': {'enabled': True, 'global': {'calls_per_minute': 60}},
            'docs': {'enabled': True, 'swagger': {'url': '/docs'}}
        }
    
    def _get_logging_config_defaults(self) -> Dict[str, Any]:
        """Get default logging configuration"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'simple': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'simple',
                    'stream': 'ext://sys.stdout'
                }
            },
            'loggers': {
                '': {'level': 'INFO', 'handlers': ['console'], 'propagate': False}
            }
        }

# Global configuration instance
config = ConfigLoader()

# Convenience functions for common operations
def get_config(config_name: str) -> Dict[str, Any]:
    """Get complete configuration by name"""
    return config.get_config(config_name)

def get(config_name: str, key_path: str, default: Any = None) -> Any:
    """Get specific configuration value using dot notation"""
    return config.get(config_name, key_path, default)

def reload_configs():
    """Reload all configurations from disk"""
    config.reload_config()

def get_environment() -> str:
    """Get current environment"""
    return config.environment

# Export commonly used configurations
def get_app_config() -> Dict[str, Any]:
    """Get application configuration"""
    return get_config('app_config')

def get_model_config() -> Dict[str, Any]:
    """Get model configuration"""
    return get_config('model_config')

def get_trading_config() -> Dict[str, Any]:
    """Get trading configuration"""
    return get_config('trading_config')

def get_indicators_config() -> Dict[str, Any]:
    """Get indicators configuration"""
    return get_config('indicators_config')
