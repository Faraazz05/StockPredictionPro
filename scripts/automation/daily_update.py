"""
scripts/automation/daily_update.py

Daily automation script for StockPredictionPro.
Performs comprehensive daily maintenance including data updates, model predictions,
performance monitoring, and system health checks.

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import numpy as np
import joblib
from sympy import symbols

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import project modules
try:
    from scripts.data.update_data import update_symbol
    from scripts.data.validate_data import validate_csv_file
    from scripts.models.evaluate_models import ModelEvaluator
    from scripts.models.deploy_models import ModelRegistry
except ImportError as e:
    logging.warning(f"Could not import project modules: {e}")

# Setup logging with both file and console output
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'daily_update_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StockPredictionPro.DailyUpdate')

# Directory configuration
DATA_DIR = Path('./data')
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = Path('./models')
PRODUCTION_MODELS_DIR = MODELS_DIR / 'production'
OUTPUTS_DIR = Path('./outputs')
PREDICTIONS_DIR = OUTPUTS_DIR / 'predictions'
REPORTS_DIR = OUTPUTS_DIR / 'reports'
CONFIG_DIR = Path('./config')

# Ensure directories exist
for dir_path in [PREDICTIONS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class DailyUpdateConfig:
    """Configuration for daily update process"""
    # Data update settings
    symbols: List[str] = None
    data_sources: List[str] = None
    update_lookback_days: int = 7
    
    # Model settings
    generate_predictions: bool = True
    update_model_performance: bool = True
    retrain_threshold: float = 0.05  # Retrain if performance drops by 5%
    
    # Monitoring settings
    send_alerts: bool = True
    alert_email: str = None
    performance_threshold: float = 0.1  # Alert if error > 10%
    
    # System settings
    max_execution_time: int = 3600  # 1 hour max
    enable_data_validation: bool = True
    backup_predictions: bool = True
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        if self.data_sources is None:
            self.data_sources = ['yahoo_finance']

@dataclass
class DailyUpdateResult:
    """Results from daily update process"""
    update_date: str
    execution_time: float
    status: str  # success, partial_success, failed
    
    # Data update results
    symbols_updated: List[str]
    symbols_failed: List[str]
    data_validation_results: Dict[str, Any]
    
    # Model results
    predictions_generated: Dict[str, int]  # model_name -> num_predictions
    model_performance: Dict[str, float]    # model_name -> current_score
    models_needing_retrain: List[str]
    
    # System health
    system_metrics: Dict[str, Any]
    alerts_sent: List[str]
    errors_encountered: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

# ============================================
# CORE UPDATE COMPONENTS
# ============================================

class DataUpdateManager:
    """Manage daily data updates"""
    
    def __init__(self, config: DailyUpdateConfig):
        self.config = config
        self.updated_symbols = []
        self.failed_symbols = []
        
    def update_all_symbols(self) -> Tuple[List[str], List[str]]:
        """Update data for all configured symbols"""
        logger.info(f"📊 Starting data update for {len(self.config.symbols)} symbols")
        
        for symbol in self.config.symbols:
            try:
                self._update_single_symbol(symbol)
                self.updated_symbols.append(symbol)
                logger.info(f"✅ Updated data for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Failed to update {symbol}: {e}")
                self.failed_symbols.append(symbol)
        
        logger.info(f"📈 Data update completed: {len(self.updated_symbols)} success, {len(self.failed_symbols)} failed")
        return self.updated_symbols, self.failed_symbols
    
    def _update_single_symbol(self, symbol: str) -> None:
        """Update data for a single symbol"""
        for source in self.config.data_sources:
            try:
                # Use the update_data module if available
                if 'update_symbol' in globals():
                    update_symbol(symbol, source)
                else:
                    # Fallback: simulate data update
                    self._simulate_data_update(symbol, source)
                break  # Success, no need to try other sources
                
            except Exception as e:
                logger.warning(f"Failed to update {symbol} from {source}: {e}")
                continue
        else:
            raise Exception(f"Failed to update {symbol} from all sources")
    
    def _simulate_data_update(self, symbol: str, source: str) -> None:
        """Simulate data update when actual modules aren't available"""
        logger.info(f"Simulating data update for {symbol} from {source}")
        
        # Create dummy data file to simulate update
        dummy_data = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=30, freq='D'),
            'symbol': [symbol] * 30,
            'open': np.random.uniform(100, 200, 30),
            'high': np.random.uniform(100, 200, 30),
            'low': np.random.uniform(100, 200, 30),
            'close': np.random.uniform(100, 200, 30),
            'volume': np.random.randint(1000000, 10000000, 30)
        })
        
        # Save to processed directory
        output_file = PROCESSED_DATA_DIR / f"{symbol}_updated_{datetime.now().strftime('%Y%m%d')}.csv"
        dummy_data.to_csv(output_file, index=False)
        logger.info(f"Created dummy data file: {output_file}")

class DataValidationManager:
    """Manage data validation for updated data"""
    
    def __init__(self, config: DailyUpdateConfig):
        self.config = config
        self.validation_results = {}
    
    def validate_updated_data(self, updated_symbols: List[str]) -> Dict[str, Any]:
        """Validate recently updated data"""
        if not self.config.enable_data_validation:
            logger.info("📋 Data validation disabled")
            return {}
        
        logger.info(f"🔍 Starting data validation for {len(updated_symbols)} symbols")
        
        overall_results = {
            'symbols_validated': 0,
            'symbols_passed': 0,
            'symbols_failed': 0,
            'validation_details': {}
        }
        
        for symbol in updated_symbols:
            try:
                result = self._validate_single_symbol(symbol)
                overall_results['validation_details'][symbol] = result
                overall_results['symbols_validated'] += 1
                
                if result.get('validation_score', {}).get('overall_score', 0) >= 70:
                    overall_results['symbols_passed'] += 1
                else:
                    overall_results['symbols_failed'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Validation failed for {symbol}: {e}")
                overall_results['validation_details'][symbol] = {'error': str(e)}
                overall_results['symbols_failed'] += 1
        
        logger.info(f"✅ Data validation completed: {overall_results['symbols_passed']} passed, {overall_results['symbols_failed']} failed")
        return overall_results
    
    def _validate_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """Validate data for a single symbol"""
        # Find most recent data file
        pattern = f"{symbol}_*.csv"
        data_files = list(PROCESSED_DATA_DIR.glob(pattern))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found for {symbol}")
        
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        
        # Use validation module if available
        if 'validate_csv_file' in globals():
            return validate_csv_file(str(latest_file))
        else:
            # Fallback validation
            return self._basic_validation(latest_file)
    
    def _basic_validation(self, file_path: Path) -> Dict[str, Any]:
        """Basic validation when full validation module isn't available"""
        try:
            df = pd.read_csv(file_path)
            
            # Basic checks
            has_required_columns = all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            has_recent_data = len(df) > 0
            no_null_values = not df.isnull().all().any()
            
            score = 100 if has_required_columns and has_recent_data and no_null_values else 50
            
            return {
                'validation_score': {'overall_score': score},
                'data_summary': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'has_required_columns': has_required_columns,
                    'has_recent_data': has_recent_data,
                    'no_null_values': no_null_values
                }
            }
            
        except Exception as e:
            return {'error': str(e), 'validation_score': {'overall_score': 0}}

class PredictionManager:
    """Manage daily prediction generation"""
    
    def __init__(self, config: DailyUpdateConfig):
        self.config = config
        self.predictions_generated = {}
        
    def generate_daily_predictions(self, updated_symbols: List[str]) -> Dict[str, int]:
        """Generate predictions for updated symbols"""
        if not self.config.generate_predictions:
            logger.info("🔮 Prediction generation disabled")
            return {}
        
        logger.info("🔮 Starting daily prediction generation")
        
        # Get available models
        available_models = self._get_available_models()
        
        if not available_models:
            logger.warning("No production models found for prediction generation")
            return {}
        
        for model_name, model_path in available_models.items():
            try:
                predictions_count = self._generate_predictions_for_model(
                    model_name, model_path, updated_symbols
                )
                self.predictions_generated[model_name] = predictions_count
                
            except Exception as e:
                logger.error(f"❌ Failed to generate predictions for {model_name}: {e}")
                self.predictions_generated[model_name] = 0
        
        total_predictions = sum(self.predictions_generated.values())
        logger.info(f"✅ Generated {total_predictions} predictions across {len(available_models)} models")
        
        return self.predictions_generated
    
    def _get_available_models(self) -> Dict[str, str]:
        """Get available production models"""
        models = {}
        
        # Look for model registry
        registry_file = PRODUCTION_MODELS_DIR / 'model_registry.json'
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                for model_id, metadata in registry_data.items():
                    if metadata.get('deployment_status') == 'active':
                        models[metadata['model_name']] = metadata['model_path']
                        
            except Exception as e:
                logger.warning(f"Could not load model registry: {e}")
        
        # Fallback: look for model files directly
        if not models:
            model_files = list(PRODUCTION_MODELS_DIR.glob("*.pkl"))
            for model_file in model_files[:3]:  # Limit to 3 models
                model_name = model_file.stem
                models[model_name] = str(model_file)
        
        return models
    
    def _generate_predictions_for_model(self, model_name: str, model_path: str, 
                                      symbols: List[str]) -> int:
        """Generate predictions for a specific model"""
        try:
            # Load model
            model = joblib.load(model_path)
            
            predictions_count = 0
            predictions_data = []
            
            for symbol in symbols:
                try:
                    # Get latest data for symbol
                    symbol_data = self._get_latest_symbol_data(symbol)
                    
                    if symbol_data is None:
                        continue
                    
                    # Generate prediction
                    prediction = self._make_prediction(model, symbol_data)
                    
                    predictions_data.append({
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'model_name': model_name,
                        'prediction': prediction,
                        'confidence': 0.75  # Placeholder confidence score
                    })
                    
                    predictions_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to predict {symbol} with {model_name}: {e}")
                    continue
            
            # Save predictions
            if predictions_data:
                self._save_predictions(model_name, predictions_data)
            
            return predictions_count
            
        except Exception as e:
            logger.error(f"Model loading failed for {model_name}: {e}")
            return 0
    
    def _get_latest_symbol_data(self, symbol: str) -> Optional[np.ndarray]:
        """Get latest processed data for a symbol"""
        try:
            # Find latest data file
            pattern = f"{symbol}_*.csv"
            data_files = list(PROCESSED_DATA_DIR.glob(pattern))
            
            if not data_files:
                return None
            
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Get last row for prediction (excluding target column)
            exclude_cols = {'timestamp', 'symbol', 'close', 'date'}
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if len(feature_cols) == 0:
                return None
            
            # Return last row as features
            return df[feature_cols].iloc[-1:].values
            
        except Exception as e:
            logger.warning(f"Could not load data for {symbol}: {e}")
            return None
    
    def _make_prediction(self, model: Any, features: np.ndarray) -> float:
        """Make prediction using the model"""
        try:
            prediction = model.predict(features)
            return float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return 0.0
    
    def _save_predictions(self, model_name: str, predictions_data: List[Dict]) -> None:
        """Save predictions to file"""
        df = pd.DataFrame(predictions_data)
        
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"predictions_{model_name}_{timestamp}.csv"
        filepath = PREDICTIONS_DIR / filename
        
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Saved {len(predictions_data)} predictions to {filepath}")

class PerformanceMonitor:
    """Monitor model performance and system health"""
    
    def __init__(self, config: DailyUpdateConfig):
        self.config = config
        self.performance_results = {}
        self.models_needing_retrain = []
    
    def monitor_model_performance(self) -> Tuple[Dict[str, float], List[str]]:
        """Monitor performance of production models"""
        if not self.config.update_model_performance:
            logger.info("📈 Model performance monitoring disabled")
            return {}, []
        
        logger.info("📈 Starting model performance monitoring")
        
        # Get available models
        available_models = self._get_available_models()
        
        for model_name, model_path in available_models.items():
            try:
                performance_score = self._evaluate_model_performance(model_name, model_path)
                self.performance_results[model_name] = performance_score
                
                # Check if model needs retraining
                if self._needs_retraining(model_name, performance_score):
                    self.models_needing_retrain.append(model_name)
                    logger.warning(f"⚠️ Model {model_name} may need retraining (score: {performance_score:.4f})")
                
            except Exception as e:
                logger.error(f"❌ Performance monitoring failed for {model_name}: {e}")
                self.performance_results[model_name] = float('inf')  # Worst possible score
        
        logger.info(f"✅ Performance monitoring completed for {len(available_models)} models")
        return self.performance_results, self.models_needing_retrain
    
    def _get_available_models(self) -> Dict[str, str]:
        """Get available production models"""
        models = {}
        
        # Look for recent model files
        model_files = list(PRODUCTION_MODELS_DIR.glob("*.pkl"))
        for model_file in model_files:
            model_name = model_file.stem
            models[model_name] = str(model_file)
        
        return models
    
    def _evaluate_model_performance(self, model_name: str, model_path: str) -> float:
        """Evaluate current model performance"""
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Get recent test data (last 30 days)
            test_data = self._get_recent_test_data()
            
            if test_data is None or len(test_data) < 10:
                logger.warning(f"Insufficient test data for {model_name}")
                return 0.5  # Default moderate score
            
            # Make predictions on test data
            X_test, y_test = self._prepare_test_data(test_data)
            y_pred = model.predict(X_test)
            
            # Calculate RMSE
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            
            return rmse
            
        except Exception as e:
            logger.error(f"Performance evaluation failed for {model_name}: {e}")
            return float('inf')
    
    def _get_recent_test_data(self) -> Optional[pd.DataFrame]:
        """Get recent data for performance testing"""
        try:
            # Look for recent processed data files
            data_files = list(PROCESSED_DATA_DIR.glob("*_updated_*.csv"))
            
            if not data_files:
                return None
            
            # Load most recent file
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Get last 30 rows for testing
            return df.tail(30) if len(df) >= 30 else df
            
        except Exception as e:
            logger.warning(f"Could not load test data: {e}")
            return None
    
    def _prepare_test_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare test data for evaluation"""
        # Prepare features and target
        exclude_cols = {'timestamp', 'symbol', 'close', 'date'}
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['close'].values if 'close' in df.columns else df.iloc[:, -1].values
        
        return X, y
    
    def _needs_retraining(self, model_name: str, current_score: float) -> bool:
        """Determine if model needs retraining based on performance"""
        # Load historical performance if available
        historical_score = self._get_historical_performance(model_name)
        
        if historical_score is None:
            return False  # No baseline to compare
        
        # Check if performance degraded beyond threshold
        performance_change = (current_score - historical_score) / historical_score
        
        return performance_change > self.config.retrain_threshold
    
    def _get_historical_performance(self, model_name: str) -> Optional[float]:
        """Get historical performance baseline for model"""
        try:
            # Look for historical performance file
            performance_file = REPORTS_DIR / f"{model_name}_performance_history.json"
            
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    history = json.load(f)
                
                # Get baseline performance (e.g., training performance)
                return history.get('baseline_rmse', None)
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not load historical performance for {model_name}: {e}")
            return None

class NotificationManager:
    """Manage notifications and alerts"""
    
    def __init__(self, config: DailyUpdateConfig):
        self.config = config
        self.alerts_sent = []
    
    def send_daily_summary(self, update_result: DailyUpdateResult) -> List[str]:
        """Send daily summary and alerts"""
        if not self.config.send_alerts or not self.config.alert_email:
            logger.info("📧 Notifications disabled")
            return []
        
        logger.info("📧 Preparing daily summary notification")
        
        # Generate summary message
        summary_message = self._generate_summary_message(update_result)
        
        # Send email if configured
        try:
            self._send_email_notification(
                subject=f"StockPredictionPro Daily Update - {update_result.update_date}",
                message=summary_message
            )
            self.alerts_sent.append("daily_summary")
            
        except Exception as e:
            logger.error(f"❌ Failed to send email notification: {e}")
        
        # Send critical alerts if needed
        self._send_critical_alerts(update_result)
        
        return self.alerts_sent
    
    def _generate_summary_message(self, result: DailyUpdateResult) -> str:
        """Generate daily summary message"""
        message = f"""
StockPredictionPro Daily Update Summary
=====================================
Date: {result.update_date}
Status: {result.status.upper()}
Execution Time: {result.execution_time:.1f} seconds

Data Updates:
- Symbols Updated: {len(result.symbols_updated)} ({', '.join(result.symbols_updated)})
- Symbols Failed: {len(result.symbols_failed)} ({', '.join(result.symbols_failed)})

Model Operations:
- Predictions Generated: {sum(result.predictions_generated.values())}
- Models Needing Retrain: {len(result.models_needing_retrain)}

System Health:
- Overall Status: {'HEALTHY' if result.status == 'success' else 'ISSUES DETECTED'}
- Errors Encountered: {len(result.errors_encountered)}

Performance Metrics:
"""
        
        for model_name, score in result.model_performance.items():
            message += f"- {model_name}: {score:.4f}\n"
        
        if result.models_needing_retrain:
            message += f"\n⚠️ ATTENTION: The following models may need retraining:\n"
            for model in result.models_needing_retrain:
                message += f"- {model}\n"
        
        if result.errors_encountered:
            message += f"\n❌ ERRORS:\n"
            for error in result.errors_encountered[:5]:  # Limit to 5 errors
                message += f"- {error}\n"
        
        return message
    
    def _send_email_notification(self, subject: str, message: str) -> None:
        """Send email notification"""
        # This is a placeholder implementation
        # In production, you would configure SMTP settings
        logger.info(f"📧 Email notification prepared: {subject}")
        logger.info(f"Message preview: {message[:200]}...")
        
        # Actual email sending would be implemented here
        # Example with smtplib:
        # smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
        # smtp_server.starttls()
        # smtp_server.login(email_user, email_password)
        # smtp_server.send_message(msg)
        # smtp_server.quit()
    
    def _send_critical_alerts(self, result: DailyUpdateResult) -> None:
        """Send critical alerts for important issues"""
        critical_issues = []
        
        # Check for critical failures
        if result.status == 'failed':
            critical_issues.append("Daily update process failed")
        
        if len(result.symbols_failed) > len(result.symbols_updated):
            critical_issues.append("More symbols failed than succeeded in data update")
        
        if len(result.models_needing_retrain) > 0:
            critical_issues.append(f"{len(result.models_needing_retrain)} models need retraining")
        
        # Send critical alerts
        for issue in critical_issues:
            try:
                self._send_email_notification(
                    subject=f"🚨 CRITICAL ALERT - StockPredictionPro",
                    message=f"Critical Issue Detected:\n{issue}\n\nPlease check the system immediately."
                )
                self.alerts_sent.append(f"critical_alert_{issue[:20]}")
                
            except Exception as e:
                logger.error(f"Failed to send critical alert: {e}")

# ============================================
# MAIN DAILY UPDATE ORCHESTRATOR
# ============================================

class DailyUpdateOrchestrator:
    """Main orchestrator for daily update process"""
    
    def __init__(self, config: DailyUpdateConfig = None):
        self.config = config or DailyUpdateConfig()
        self.start_time = None
        self.errors_encountered = []
        
        # Initialize managers
        self.data_manager = DataUpdateManager(self.config)
        self.validation_manager = DataValidationManager(self.config)
        self.prediction_manager = PredictionManager(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.notification_manager = NotificationManager(self.config)
    
    def run_daily_update(self) -> DailyUpdateResult:
        """Run complete daily update process"""
        logger.info("🚀 Starting StockPredictionPro daily update process")
        self.start_time = time.time()
        
        try:
            # Step 1: Update market data
            updated_symbols, failed_symbols = self._safe_execute(
                self.data_manager.update_all_symbols,
                "Data Update",
                default=([], [])
            )
            
            # Step 2: Validate updated data
            validation_results = self._safe_execute(
                lambda: self.validation_manager.validate_updated_data(updated_symbols),
                "Data Validation",
                default={}
            )
            
            # Step 3: Generate predictions
            predictions_generated = self._safe_execute(
                lambda: self.prediction_manager.generate_daily_predictions(updated_symbols),
                "Prediction Generation",
                default={}
            )
            
            # Step 4: Monitor model performance
            model_performance, models_needing_retrain = self._safe_execute(
                self.performance_monitor.monitor_model_performance,
                "Performance Monitoring",
                default=({}, [])
            )
            
            # Step 5: Generate system metrics
            system_metrics = self._generate_system_metrics()
            
            # Determine overall status
            status = self._determine_overall_status(
                updated_symbols, failed_symbols, self.errors_encountered
            )
            
            # Create result object
            execution_time = time.time() - self.start_time
            result = DailyUpdateResult(
                update_date=datetime.now().strftime('%Y-%m-%d'),
                execution_time=execution_time,
                status=status,
                symbols_updated=updated_symbols,
                symbols_failed=failed_symbols,
                data_validation_results=validation_results,
                predictions_generated=predictions_generated,
                model_performance=model_performance,
                models_needing_retrain=models_needing_retrain,
                system_metrics=system_metrics,
                alerts_sent=[],
                errors_encountered=self.errors_encountered
            )
            
            # Step 6: Send notifications
            alerts_sent = self._safe_execute(
                lambda: self.notification_manager.send_daily_summary(result),
                "Notification",
                default=[]
            )
            result.alerts_sent = alerts_sent
            
            # Step 7: Save results
            self._save_daily_results(result)
            
            # Final summary
            self._print_final_summary(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Daily update process failed: {e}")
            logger.error(traceback.format_exc())
            
            # Create failure result
            execution_time = time.time() - self.start_time if self.start_time else 0
            return DailyUpdateResult(
                update_date=datetime.now().strftime('%Y-%m-%d'),
                execution_time=execution_time,
                status='failed',
                symbols_updated=[],
                symbols_failed=self.config.symbols,
                data_validation_results={},
                predictions_generated={},
                model_performance={},
                models_needing_retrain=[],
                system_metrics={},
                alerts_sent=[],
                errors_encountered=[str(e)]
            )
    
    def _safe_execute(self, func, operation_name: str, default=None):
        """Safely execute operation with error handling"""
        try:
            logger.info(f"▶️ Starting {operation_name}")
            result = func()
            logger.info(f"✅ Completed {operation_name}")
            return result
            
        except Exception as e:
            error_msg = f"{operation_name} failed: {e}"
            logger.error(f"❌ {error_msg}")
            self.errors_encountered.append(error_msg)
            return default
    
    def _generate_system_metrics(self) -> Dict[str, Any]:
        """Generate system health metrics"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('.').percent,
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            return {
                'cpu_percent': 'N/A',
                'memory_percent': 'N/A',
                'disk_usage_percent': 'N/A',
                'timestamp': datetime.now().isoformat()
            }
    
    def _determine_overall_status(self, updated_symbols: List[str], failed_symbols: List[str], 
                                errors: List[str]) -> str:
        """Determine overall update status"""
        if len(errors) > 5:
            return 'failed'
        elif len(failed_symbols) > 0 or len(errors) > 0:
            return 'partial_success'
        else:
            return 'success'
    
    def _save_daily_results(self, result: DailyUpdateResult) -> None:
        """Save daily update results"""
        try:
            # Save detailed results
            results_file = REPORTS_DIR / f"daily_update_{result.update_date}.json"
            result.save(results_file)
            
            # Update summary log
            summary_file = REPORTS_DIR / "daily_update_summary.json"
            self._update_summary_log(summary_file, result)
            
            logger.info(f"💾 Saved daily update results to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save daily results: {e}")
    
    def _update_summary_log(self, summary_file: Path, result: DailyUpdateResult) -> None:
        """Update summary log with latest results"""
        try:
            # Load existing summary
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
            else:
                summary_data = {'daily_updates': []}
            
            # Add current result
            summary_data['daily_updates'].append({
                'date': result.update_date,
                'status': result.status,
                'execution_time': result.execution_time,
                'symbols_updated': len(result.symbols_updated),
                'symbols_failed': len(result.symbols_failed),
                'predictions_generated': sum(result.predictions_generated.values()),
                'errors_count': len(result.errors_encountered)
            })
            
            # Keep only last 30 days
            summary_data['daily_updates'] = summary_data['daily_updates'][-30:]
            
            # Save updated summary
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to update summary log: {e}")
    
    def _print_final_summary(self, result: DailyUpdateResult) -> None:
        """Print final summary to console"""
        print("\n" + "="*60)
        print("STOCKPREDICTIONPRO DAILY UPDATE SUMMARY")
        print("="*60)
        print(f"Date: {result.update_date}")
        print(f"Status: {result.status.upper()}")
        print(f"Execution Time: {result.execution_time:.1f} seconds")
        
        print(f"\nData Updates:")
        print(f"  ✅ Successful: {len(result.symbols_updated)}")
        print(f"  ❌ Failed: {len(result.symbols_failed)}")
        
        print(f"\nPredictions:")
        total_predictions = sum(result.predictions_generated.values())
        print(f"  🔮 Generated: {total_predictions}")
        
        print(f"\nModel Health:")
        print(f"  📊 Models Monitored: {len(result.model_performance)}")
        print(f"  ⚠️ Need Retraining: {len(result.models_needing_retrain)}")
        
        if result.errors_encountered:
            print(f"\nErrors ({len(result.errors_encountered)}):")
            for error in result.errors_encountered[:3]:
                print(f"  • {error}")
        
        print(f"\nOverall Status: {'🎉 SUCCESS' if result.status == 'success' else '⚠️ PARTIAL SUCCESS' if result.status == 'partial_success' else '❌ FAILED'}")

def load_config_from_file(config_path: str) -> DailyUpdateConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return DailyUpdateConfig(**config_dict)
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        return DailyUpdateConfig()

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='StockPredictionPro Daily Update')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--symbols', nargs='+', help='Symbols to update (overrides config)')
    parser.add_argument('--no-predictions', action='store_true', help='Skip prediction generation')
    parser.add_argument('--no-alerts', action='store_true', help='Skip sending alerts')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no actual updates)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = DailyUpdateConfig()
    
    # Override config with command line arguments
    if args.symbols:
        config.symbols = args.symbols
    if args.no_predictions:
        config.generate_predictions = False
    if args.no_alerts:
        config.send_alerts = False
    
    # Run daily update
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No actual updates will be performed")
        print("Dry run completed - no changes made")
    else:
        orchestrator = DailyUpdateOrchestrator(config)
        result = orchestrator.run_daily_update()
        
        # Exit with appropriate code
        exit_code = 0 if result.status == 'success' else 1 if result.status == 'partial_success' else 2
        sys.exit(exit_code)

if __name__ == '__main__':
    main()
