"""
setup/setup_config.py

Comprehensive configuration management system for StockPredictionPro.
Handles environment-based configuration, validation, and runtime overrides.
Supports YAML, JSON, and environment variable configuration sources.

Author: StockPredictionPro Team  
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import yaml
import logging
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockPredictionPro.ConfigSetup')

# ============================================
# CONFIGURATION SCHEMAS
# ============================================

@dataclass
class DatabaseConfig:
    """Database configuration schema"""
    url: str = "sqlite:///./data/stockpred.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_recycle: int = 3600
    echo: bool = False
    timeout: int = 30

@dataclass
class ApiConfig:
    """API endpoints and authentication"""
    alpha_vantage_key: str = ""
    yahoo_finance_enabled: bool = True
    polygon_api_key: str = ""
    rate_limit_per_minute: int = 60
    request_timeout: int = 30
    retry_attempts: int = 3

@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    default_train_test_split: float = 0.8
    cross_validation_folds: int = 5
    random_seed: int = 42
    max_training_time: int = 3600  # seconds
    model_save_path: str = "./models/"
    auto_retrain_threshold: float = 0.05  # performance degradation threshold

@dataclass
class TradingConfig:
    """Trading and backtesting configuration"""
    initial_capital: float = 100000.0
    max_position_size: float = 0.3  # 30% of portfolio
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    risk_free_rate: float = 0.02    # 2%
    stop_loss_pct: float = 0.05     # 5%

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "./logs/stockpred.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    console_output: bool = True

@dataclass
class SecurityConfig:
    """Security and authentication configuration"""
    secret_key: str = ""
    jwt_expiration_hours: int = 24
    bcrypt_rounds: int = 12
    api_rate_limit: int = 1000
    allowed_hosts: List[str] = None
    cors_enabled: bool = True

@dataclass
class MonitoringConfig:
    """System monitoring configuration"""
    metrics_enabled: bool = True
    health_check_interval: int = 300  # 5 minutes
    alert_email: str = ""
    prometheus_enabled: bool = False
    grafana_enabled: bool = False

# ============================================
# CONFIGURATION MANAGER
# ============================================

class ConfigurationManager:
    """Comprehensive configuration management system"""
    
    def __init__(self, environment: str = None, config_dir: Path = None):
        self.environment = environment or os.getenv('STOCKPRED_ENV', 'development')
        self.config_dir = config_dir or Path(__file__).parent.parent / 'config'
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration file paths
        self.default_config_file = self.config_dir / 'default.yaml'
        self.env_config_file = self.config_dir / f'{self.environment}.yaml'
        self.local_config_file = self.config_dir / 'local.yaml'
        self.secrets_file = self.config_dir / '.secrets.yaml'
        
        # Loaded configuration
        self.config: Dict[str, Any] = {}
        self.schema_classes = {
            'database': DatabaseConfig,
            'api': ApiConfig,
            'models': ModelConfig,
            'trading': TradingConfig,
            'logging': LoggingConfig,
            'security': SecurityConfig,
            'monitoring': MonitoringConfig
        }
        
        # Initialize configuration
        self._load_configuration()
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Safely load YAML file with error handling"""
        if not file_path.exists():
            logger.debug(f"Configuration file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            logger.info(f"✅ Loaded configuration: {file_path}")
            return data
        except yaml.YAMLError as e:
            logger.error(f"❌ Invalid YAML in {file_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
            return {}
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Database configuration from environment
        if os.getenv('DATABASE_URL'):
            env_config.setdefault('database', {})['url'] = os.getenv('DATABASE_URL')
        
        # API keys from environment
        api_config = {}
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            api_config['alpha_vantage_key'] = os.getenv('ALPHA_VANTAGE_API_KEY')
        if os.getenv('POLYGON_API_KEY'):
            api_config['polygon_api_key'] = os.getenv('POLYGON_API_KEY')
        if api_config:
            env_config['api'] = api_config
        
        # Security configuration
        if os.getenv('SECRET_KEY'):
            env_config.setdefault('security', {})['secret_key'] = os.getenv('SECRET_KEY')
        
        # Logging level
        if os.getenv('LOG_LEVEL'):
            env_config.setdefault('logging', {})['level'] = os.getenv('LOG_LEVEL')
        
        if env_config:
            logger.info("✅ Loaded configuration from environment variables")
        
        return env_config
    
    def _load_configuration(self) -> None:
        """Load configuration from multiple sources in priority order"""
        logger.info("🔧 Loading configuration...")
        
        # 1. Start with default configuration
        config = self._load_yaml_file(self.default_config_file)
        
        # 2. Override with environment-specific configuration
        env_config = self._load_yaml_file(self.env_config_file)
        config = self._deep_merge(config, env_config)
        
        # 3. Override with local configuration (for development)
        local_config = self._load_yaml_file(self.local_config_file)
        config = self._deep_merge(config, local_config)
        
        # 4. Override with secrets (API keys, passwords, etc.)
        secrets_config = self._load_yaml_file(self.secrets_file)
        config = self._deep_merge(config, secrets_config)
        
        # 5. Override with environment variables (highest priority)
        env_vars_config = self._load_environment_variables()
        config = self._deep_merge(config, env_vars_config)
        
        self.config = config
        logger.info(f"✅ Configuration loaded for environment: {self.environment}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'database.url')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'database.url')
            value: Value to set
        """
        keys = key_path.split('.')
        current = self.config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final key
        current[keys[-1]] = value
        logger.info(f"🔧 Configuration updated: {key_path} = {value}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config.get(section, {})
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate configuration against schemas"""
        errors = {}
        
        for section_name, schema_class in self.schema_classes.items():
            section_config = self.get_section(section_name)
            section_errors = []
            
            # Check required fields (basic validation)
            if section_name == 'database' and not section_config.get('url'):
                section_errors.append("Database URL is required")
            
            if section_name == 'security' and not section_config.get('secret_key'):
                section_errors.append("Secret key is required for production")
            
            # Add more validation rules as needed
            
            if section_errors:
                errors[section_name] = section_errors
        
        return errors
    
    def save_configuration(self, file_path: Path = None) -> bool:
        """Save current configuration to file"""
        target_file = file_path or self.env_config_file
        
        try:
            # Ensure directory exists
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(target_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Configuration saved to: {target_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            return False
    
    def create_default_configs(self) -> bool:
        """Create default configuration files"""
        try:
            # Default configuration
            default_config = {
                'database': asdict(DatabaseConfig()),
                'api': asdict(ApiConfig()),
                'models': asdict(ModelConfig()),
                'trading': asdict(TradingConfig()),
                'logging': asdict(LoggingConfig()),
                'security': asdict(SecurityConfig()),
                'monitoring': asdict(MonitoringConfig())
            }
            
            # Save default configuration
            with open(self.default_config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            # Create environment-specific configuration (minimal overrides)
            env_specific_config = {
                'database': {
                    'url': f'sqlite:///./data/stockpred_{self.environment}.db',
                    'echo': self.environment == 'development'
                },
                'logging': {
                    'level': 'DEBUG' if self.environment == 'development' else 'INFO',
                    'file_path': f'./logs/stockpred_{self.environment}.log'
                }
            }
            
            with open(self.env_config_file, 'w', encoding='utf-8') as f:
                yaml.dump(env_specific_config, f, default_flow_style=False, indent=2)
            
            # Create secrets template
            secrets_template = {
                'api': {
                    'alpha_vantage_key': 'your_alpha_vantage_api_key_here',
                    'polygon_api_key': 'your_polygon_api_key_here'
                },
                'security': {
                    'secret_key': 'your_secret_key_here'
                },
                'monitoring': {
                    'alert_email': 'your_email@example.com'
                }
            }
            
            # Only create secrets file if it doesn't exist
            if not self.secrets_file.exists():
                with open(self.secrets_file, 'w', encoding='utf-8') as f:
                    yaml.dump(secrets_template, f, default_flow_style=False, indent=2)
                logger.info(f"📝 Secrets template created: {self.secrets_file}")
                logger.warning("⚠️ Please update the secrets file with actual values")
            
            logger.info("✅ Default configuration files created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create default configurations: {e}")
            return False
    
    def print_configuration_summary(self) -> None:
        """Print configuration summary"""
        print("\n" + "="*60)
        print("STOCKPREDICTIONPRO CONFIGURATION SUMMARY")
        print("="*60)
        
        print(f"Environment: {self.environment}")
        print(f"Configuration Directory: {self.config_dir}")
        
        print(f"\nConfiguration Files:")
        for file_path in [self.default_config_file, self.env_config_file, 
                         self.local_config_file, self.secrets_file]:
            status = "✅" if file_path.exists() else "❌"
            print(f"  {status} {file_path}")
        
        print(f"\nConfiguration Sections:")
        for section in self.schema_classes.keys():
            section_config = self.get_section(section)
            if section_config:
                print(f"  ✅ {section} ({len(section_config)} settings)")
            else:
                print(f"  ❌ {section} (not configured)")
        
        # Validation results
        errors = self.validate_configuration()
        if errors:
            print(f"\n⚠️ Configuration Validation Errors:")
            for section, error_list in errors.items():
                print(f"  {section}:")
                for error in error_list:
                    print(f"    - {error}")
        else:
            print(f"\n✅ Configuration validation passed")

# ============================================
# UTILITY FUNCTIONS
# ============================================

def setup_configuration(environment: str = None, 
                       create_defaults: bool = True,
                       validate: bool = True) -> ConfigurationManager:
    """
    Setup configuration system for StockPredictionPro
    
    Args:
        environment: Target environment (development, testing, production)
        create_defaults: Whether to create default configuration files
        validate: Whether to validate configuration
        
    Returns:
        Configured ConfigurationManager instance
    """
    logger.info("🚀 Setting up StockPredictionPro configuration...")
    
    # Initialize configuration manager
    config_manager = ConfigurationManager(environment=environment)
    
    # Create default configuration files if requested
    if create_defaults:
        config_manager.create_default_configs()
    
    # Reload configuration after creating defaults
    config_manager._load_configuration()
    
    # Validate configuration if requested
    if validate:
        errors = config_manager.validate_configuration()
        if errors:
            logger.warning("⚠️ Configuration validation found issues")
            for section, error_list in errors.items():
                for error in error_list:
                    logger.warning(f"  {section}: {error}")
        else:
            logger.info("✅ Configuration validation passed")
    
    # Print summary
    config_manager.print_configuration_summary()
    
    logger.info("✅ Configuration setup completed")
    return config_manager

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup StockPredictionPro Configuration')
    parser.add_argument('--environment', '-e',
                       choices=['development', 'testing', 'production'],
                       default='development',
                       help='Target environment')
    parser.add_argument('--create-defaults', '-c',
                       action='store_true',
                       help='Create default configuration files')
    parser.add_argument('--no-validate',
                       action='store_true',
                       help='Skip configuration validation')
    parser.add_argument('--show-config', '-s',
                       action='store_true',
                       help='Show current configuration')
    parser.add_argument('--set-config', '-set',
                       nargs=2,
                       metavar=('KEY', 'VALUE'),
                       help='Set configuration value (e.g., --set-config database.url "sqlite:///test.db")')
    
    args = parser.parse_args()
    
    # Setup configuration
    config_manager = setup_configuration(
        environment=args.environment,
        create_defaults=args.create_defaults,
        validate=not args.no_validate
    )
    
    # Set configuration value if requested
    if args.set_config:
        key, value = args.set_config
        # Try to parse as JSON first (for complex values)
        try:
            parsed_value = json.loads(value)
        except:
            parsed_value = value
        
        config_manager.set(key, parsed_value)
        config_manager.save_configuration()
        print(f"✅ Configuration updated: {key} = {parsed_value}")
    
    # Show configuration if requested
    if args.show_config:
        print("\n" + "="*60)
        print("CURRENT CONFIGURATION")
        print("="*60)
        print(yaml.dump(config_manager.config, default_flow_style=False, indent=2))
    
    logger.info("🎉 Configuration setup completed successfully!")

if __name__ == '__main__':
    main()
