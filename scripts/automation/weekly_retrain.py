"""
scripts/automation/weekly_retrain.py

Automated weekly model retraining pipeline for StockPredictionPro.
Evaluates model performance degradation, triggers retraining when needed,
compares new models with existing ones, and handles model deployment.

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
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import warnings

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import project modules
try:
    from scripts.models.train_all_models import ModelTrainingPipeline, TrainingConfig
    from scripts.models.evaluate_models import ModelEvaluator
    from scripts.models.deploy_models import ModelDeployer, DeploymentConfig
    from scripts.data.validate_data import validate_csv_file
except ImportError as e:
    logging.warning(f"Could not import project modules: {e}")

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'weekly_retrain_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StockPredictionPro.WeeklyRetrain')

# Suppress warnings
warnings.filterwarnings('ignore')

# Directory configuration
DATA_DIR = Path('./data')
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = Path('./models')
PRODUCTION_MODELS_DIR = MODELS_DIR / 'production'
RETRAIN_MODELS_DIR = MODELS_DIR / 'retrain'
BACKUP_MODELS_DIR = MODELS_DIR / 'backup'
REPORTS_DIR = Path('./outputs/reports')

# Ensure directories exist
for dir_path in [RETRAIN_MODELS_DIR, BACKUP_MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class RetrainConfig:
    """Configuration for weekly retraining process"""
    # Performance thresholds
    performance_degradation_threshold: float = 0.10  # 10% degradation triggers retrain
    minimum_performance_threshold: float = 0.05  # RMSE threshold for acceptable performance
    
    # Data requirements
    minimum_data_days: int = 30  # Minimum days of data required
    lookback_days: int = 90  # Days of historical data to include
    validation_split: float = 0.2  # Validation data percentage
    
    # Model selection
    models_to_retrain: List[str] = None  # None means all production models
    max_models_per_symbol: int = 3  # Maximum models to maintain per symbol
    
    # Training parameters
    retrain_all_models: bool = False  # If True, retrain regardless of performance
    enable_hyperparameter_optimization: bool = True
    training_timeout_minutes: int = 120  # 2 hours max training time
    
    # Deployment settings
    auto_deploy_better_models: bool = True
    backup_old_models: bool = True
    require_improvement_threshold: float = 0.02  # 2% improvement required for deployment
    
    # Notification settings
    send_notifications: bool = True
    notification_email: str = None
    
    def __post_init__(self):
        if self.models_to_retrain is None:
            self.models_to_retrain = ['xgboost', 'lightgbm', 'neural_network']

@dataclass
class ModelPerformanceReport:
    """Performance report for a single model"""
    model_name: str
    symbol: str
    current_performance: float
    historical_baseline: float
    performance_change: float
    needs_retraining: bool
    data_available: bool
    last_trained: Optional[str]
    recommendation: str

@dataclass
class RetrainingResult:
    """Results from model retraining"""
    model_name: str
    symbol: str
    training_status: str  # success, failed, skipped
    old_performance: float
    new_performance: float
    improvement: float
    training_time: float
    deployed: bool
    model_path: str = None
    error_message: str = None

@dataclass
class WeeklyRetrainReport:
    """Comprehensive weekly retraining report"""
    execution_date: str
    execution_time: float
    
    # Performance analysis
    models_evaluated: int
    models_needing_retrain: int
    
    # Retraining results
    models_retrained: int
    successful_retrains: int
    failed_retrains: int
    
    # Deployment results
    models_deployed: int
    models_backed_up: int
    
    # Detailed results
    performance_reports: List[ModelPerformanceReport]
    retraining_results: List[RetrainingResult]
    
    # Summary
    overall_status: str  # success, partial_success, failed
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

# ============================================
# PERFORMANCE ANALYSIS
# ============================================

class ModelPerformanceAnalyzer:
    """Analyze model performance and determine retraining needs"""
    
    def __init__(self, config: RetrainConfig):
        self.config = config
        self.performance_reports = []
    
    def analyze_all_models(self) -> List[ModelPerformanceReport]:
        """Analyze performance of all production models"""
        logger.info("📊 Starting model performance analysis...")
        
        # Get production models
        production_models = self._get_production_models()
        
        if not production_models:
            logger.warning("No production models found for analysis")
            return []
        
        for model_info in production_models:
            try:
                report = self._analyze_single_model(model_info)
                self.performance_reports.append(report)
                
                if report.needs_retraining:
                    logger.warning(f"🔄 Model {report.model_name} for {report.symbol} needs retraining")
                else:
                    logger.info(f"✅ Model {report.model_name} for {report.symbol} performing well")
                    
            except Exception as e:
                logger.error(f"❌ Failed to analyze {model_info.get('name', 'unknown')}: {e}")
                # Create error report
                error_report = ModelPerformanceReport(
                    model_name=model_info.get('name', 'unknown'),
                    symbol=model_info.get('symbol', 'unknown'),
                    current_performance=float('inf'),
                    historical_baseline=0.0,
                    performance_change=float('inf'),
                    needs_retraining=True,
                    data_available=False,
                    last_trained=None,
                    recommendation=f"Analysis failed: {e}"
                )
                self.performance_reports.append(error_report)
        
        logger.info(f"📈 Performance analysis completed: {len(self.performance_reports)} models analyzed")
        return self.performance_reports
    
    def _get_production_models(self) -> List[Dict[str, Any]]:
        """Get list of production models to analyze"""
        models = []
        
        # Try to load from model registry
        registry_file = PRODUCTION_MODELS_DIR / 'model_registry.json'
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                for model_id, metadata in registry_data.items():
                    if metadata.get('deployment_status') == 'active':
                        models.append({
                            'id': model_id,
                            'name': metadata.get('model_name', 'unknown'),
                            'path': metadata.get('model_path'),
                            'symbol': self._extract_symbol_from_metadata(metadata),
                            'deployed_at': metadata.get('deployed_at'),
                            'metadata': metadata
                        })
                        
            except Exception as e:
                logger.warning(f"Could not load model registry: {e}")
        
        # Fallback: scan production directory
        if not models:
            model_files = list(PRODUCTION_MODELS_DIR.glob("*.pkl"))
            for model_file in model_files:
                models.append({
                    'id': model_file.stem,
                    'name': self._extract_model_name(model_file.name),
                    'path': str(model_file),
                    'symbol': self._extract_symbol_from_filename(model_file.name),
                    'deployed_at': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                    'metadata': {}
                })
        
        return models
    
    def _analyze_single_model(self, model_info: Dict[str, Any]) -> ModelPerformanceReport:
        """Analyze performance of a single model"""
        model_name = model_info['name']
        symbol = model_info['symbol']
        model_path = model_info['path']
        
        # Get current performance
        current_performance = self._evaluate_current_performance(model_path, symbol)
        
        # Get historical baseline
        historical_baseline = self._get_historical_baseline(model_name, symbol)
        
        # Calculate performance change
        if historical_baseline > 0:
            performance_change = (current_performance - historical_baseline) / historical_baseline
        else:
            performance_change = float('inf')
        
        # Determine if retraining is needed
        needs_retraining = self._determine_retraining_need(
            current_performance, historical_baseline, performance_change
        )
        
        # Check data availability
        data_available = self._check_data_availability(symbol)
        
        # Get last training date
        last_trained = model_info.get('deployed_at')
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            current_performance, performance_change, data_available, needs_retraining
        )
        
        return ModelPerformanceReport(
            model_name=model_name,
            symbol=symbol,
            current_performance=current_performance,
            historical_baseline=historical_baseline,
            performance_change=performance_change,
            needs_retraining=needs_retraining,
            data_available=data_available,
            last_trained=last_trained,
            recommendation=recommendation
        )
    
    def _evaluate_current_performance(self, model_path: str, symbol: str) -> float:
        """Evaluate current model performance on recent data"""
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Get recent test data
            test_data = self._get_recent_test_data(symbol)
            
            if test_data is None or len(test_data) < 10:
                logger.warning(f"Insufficient test data for {symbol}")
                return float('inf')  # Indicate poor performance due to lack of data
            
            # Prepare test data
            X_test, y_test = self._prepare_test_data(test_data)
            
            if X_test is None or len(X_test) == 0:
                return float('inf')
            
            # Make predictions and calculate RMSE
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            return rmse
            
        except Exception as e:
            logger.error(f"Performance evaluation failed for {symbol}: {e}")
            return float('inf')
    
    def _get_recent_test_data(self, symbol: str, days_back: int = 30) -> Optional[pd.DataFrame]:
        """Get recent data for testing model performance"""
        try:
            # Look for recent processed data files
            pattern = f"{symbol}_*.csv"
            data_files = list(PROCESSED_DATA_DIR.glob(pattern))
            
            if not data_files:
                return None
            
            # Get most recent file
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Convert timestamp if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            # Get last N days of data
            if len(df) > days_back:
                df = df.tail(days_back)
            
            return df if len(df) >= 10 else None
            
        except Exception as e:
            logger.error(f"Could not load test data for {symbol}: {e}")
            return None
    
    def _prepare_test_data(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare test data for model evaluation"""
        try:
            # Identify target column
            target_col = 'close'
            if target_col not in df.columns:
                # Try other common target names
                for alt_target in ['price', 'target', df.columns[-1]]:
                    if alt_target in df.columns:
                        target_col = alt_target
                        break
                else:
                    logger.error("Could not identify target column")
                    return None, None
            
            # Prepare features
            exclude_cols = {target_col, 'symbol', 'timestamp', 'date'}
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if not feature_cols:
                logger.error("No feature columns found")
                return None, None
            
            X = df[feature_cols].values
            y = df[target_col].values
            
            # Handle missing values
            if pd.DataFrame(X).isnull().any().any():
                # Simple forward fill for missing values
                X = pd.DataFrame(X).fillna(method='ffill').fillna(method='bfill').values
            
            return X, y
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return None, None
    
    def _get_historical_baseline(self, model_name: str, symbol: str) -> float:
        """Get historical performance baseline for comparison"""
        try:
            # Look for historical performance file
            performance_file = REPORTS_DIR / f"{model_name}_{symbol}_performance_history.json"
            
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    history = json.load(f)
                
                # Get baseline (training or validation performance)
                baseline = history.get('baseline_rmse') or history.get('validation_rmse')
                if baseline:
                    return float(baseline)
            
            # Fallback: use a reasonable default based on typical financial data
            return 0.05  # 5% RMSE as default baseline
            
        except Exception as e:
            logger.debug(f"Could not load historical baseline: {e}")
            return 0.05
    
    def _determine_retraining_need(self, current_perf: float, baseline: float, 
                                 change: float) -> bool:
        """Determine if model needs retraining based on performance metrics"""
        if self.config.retrain_all_models:
            return True
        
        # Check for significant performance degradation
        if change > self.config.performance_degradation_threshold:
            return True
        
        # Check if performance is below minimum acceptable threshold
        if current_perf > self.config.minimum_performance_threshold:
            return True
        
        return False
    
    def _check_data_availability(self, symbol: str) -> bool:
        """Check if sufficient data is available for retraining"""
        try:
            # Look for recent data files
            pattern = f"{symbol}_*.csv"
            data_files = list(PROCESSED_DATA_DIR.glob(pattern))
            
            if not data_files:
                return False
            
            # Check if we have enough recent data
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Check data age and quantity
            if len(df) < self.config.minimum_data_days:
                return False
            
            # Check if data is recent enough (within last 7 days)
            file_age = (datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)).days
            return file_age <= 7
            
        except Exception as e:
            logger.debug(f"Data availability check failed for {symbol}: {e}")
            return False
    
    def _generate_recommendation(self, current_perf: float, perf_change: float, 
                               data_available: bool, needs_retraining: bool) -> str:
        """Generate human-readable recommendation"""
        if not data_available:
            return "Cannot retrain - insufficient or stale data available"
        
        if not needs_retraining:
            return "Model performing well - no retraining needed"
        
        if perf_change > 0.20:  # >20% degradation
            return "URGENT: Significant performance degradation detected - immediate retraining recommended"
        elif perf_change > 0.10:  # >10% degradation
            return "Performance degradation detected - retraining recommended"
        elif current_perf > self.config.minimum_performance_threshold:
            return "Performance below acceptable threshold - retraining needed"
        else:
            return "Scheduled retraining (retrain_all_models enabled)"
    
    def _extract_symbol_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Extract symbol from model metadata"""
        # Try various metadata fields
        for field in ['symbol', 'ticker', 'asset']:
            if field in metadata:
                return metadata[field]
        
        # Try to extract from model name or path
        model_name = metadata.get('model_name', '')
        if '_' in model_name:
            parts = model_name.split('_')
            for part in parts:
                if part.isupper() and 2 <= len(part) <= 5:  # Likely a stock symbol
                    return part
        
        return 'UNKNOWN'
    
    def _extract_symbol_from_filename(self, filename: str) -> str:
        """Extract symbol from filename"""
        # Common patterns: AAPL_model.pkl, model_AAPL.pkl, etc.
        parts = filename.replace('.pkl', '').split('_')
        for part in parts:
            if part.isupper() and 2 <= len(part) <= 5:
                return part
        return 'UNKNOWN'
    
    def _extract_model_name(self, filename: str) -> str:
        """Extract model type from filename"""
        filename_lower = filename.lower()
        if 'xgboost' in filename_lower or 'xgb' in filename_lower:
            return 'xgboost'
        elif 'lightgbm' in filename_lower or 'lgb' in filename_lower:
            return 'lightgbm'
        elif 'neural' in filename_lower or 'nn' in filename_lower:
            return 'neural_network'
        elif 'random' in filename_lower or 'rf' in filename_lower:
            return 'random_forest'
        else:
            return 'unknown'

# ============================================
# MODEL RETRAINING ENGINE
# ============================================

class ModelRetrainingEngine:
    """Engine for retraining models based on performance analysis"""
    
    def __init__(self, config: RetrainConfig):
        self.config = config
        self.retraining_results = []
    
    def retrain_models(self, performance_reports: List[ModelPerformanceReport]) -> List[RetrainingResult]:
        """Retrain models that need it based on performance analysis"""
        logger.info("🔄 Starting model retraining process...")
        
        # Filter models that need retraining and have data available
        models_to_retrain = [
            report for report in performance_reports 
            if report.needs_retraining and report.data_available
        ]
        
        if not models_to_retrain:
            logger.info("No models need retraining")
            return []
        
        logger.info(f"Retraining {len(models_to_retrain)} models...")
        
        for report in models_to_retrain:
            try:
                result = self._retrain_single_model(report)
                self.retraining_results.append(result)
                
                if result.training_status == 'success':
                    logger.info(f"✅ Successfully retrained {result.model_name} for {result.symbol}")
                else:
                    logger.error(f"❌ Failed to retrain {result.model_name} for {result.symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Retraining failed for {report.model_name} ({report.symbol}): {e}")
                self.retraining_results.append(RetrainingResult(
                    model_name=report.model_name,
                    symbol=report.symbol,
                    training_status='failed',
                    old_performance=report.current_performance,
                    new_performance=float('inf'),
                    improvement=0.0,
                    training_time=0.0,
                    deployed=False,
                    error_message=str(e)
                ))
        
        logger.info(f"🔄 Retraining completed: {len(self.retraining_results)} models processed")
        return self.retraining_results
    
    def _retrain_single_model(self, report: ModelPerformanceReport) -> RetrainingResult:
        """Retrain a single model"""
        start_time = time.time()
        
        try:
            # Get training data
            training_data = self._prepare_training_data(report.symbol)
            
            if training_data is None:
                return RetrainingResult(
                    model_name=report.model_name,
                    symbol=report.symbol,
                    training_status='failed',
                    old_performance=report.current_performance,
                    new_performance=float('inf'),
                    improvement=0.0,
                    training_time=time.time() - start_time,
                    deployed=False,
                    error_message="Could not prepare training data"
                )
            
            # Set up training configuration
            training_config = self._create_training_config(report.model_name)
            
            # Train new model
            new_model_path = self._train_model(training_data, training_config, report)
            
            if new_model_path is None:
                return RetrainingResult(
                    model_name=report.model_name,
                    symbol=report.symbol,
                    training_status='failed',
                    old_performance=report.current_performance,
                    new_performance=float('inf'),
                    improvement=0.0,
                    training_time=time.time() - start_time,
                    deployed=False,
                    error_message="Model training failed"
                )
            
            # Evaluate new model
            new_performance = self._evaluate_new_model(new_model_path, report.symbol)
            
            # Calculate improvement
            improvement = (report.current_performance - new_performance) / report.current_performance
            
            # Decide whether to deploy
            should_deploy = self._should_deploy_model(improvement, new_performance)
            
            deployed = False
            if should_deploy:
                deployed = self._deploy_new_model(new_model_path, report)
            
            training_time = time.time() - start_time
            
            return RetrainingResult(
                model_name=report.model_name,
                symbol=report.symbol,
                training_status='success',
                old_performance=report.current_performance,
                new_performance=new_performance,
                improvement=improvement,
                training_time=training_time,
                deployed=deployed,
                model_path=new_model_path
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            return RetrainingResult(
                model_name=report.model_name,
                symbol=report.symbol,
                training_status='failed',
                old_performance=report.current_performance,
                new_performance=float('inf'),
                improvement=0.0,
                training_time=training_time,
                deployed=False,
                error_message=str(e)
            )
    
    def _prepare_training_data(self, symbol: str) -> Optional[str]:
        """Prepare training data for model retraining"""
        try:
            # Find recent data files for the symbol
            pattern = f"{symbol}_*.csv"
            data_files = list(PROCESSED_DATA_DIR.glob(pattern))
            
            if not data_files:
                logger.error(f"No data files found for {symbol}")
                return None
            
            # Get most recent file
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            
            # Load and validate data
            df = pd.read_csv(latest_file)
            
            # Basic validation
            if len(df) < self.config.minimum_data_days:
                logger.error(f"Insufficient data for {symbol}: {len(df)} rows")
                return None
            
            # Ensure data is sorted by time
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            # Get recent data based on lookback period
            if len(df) > self.config.lookback_days:
                df = df.tail(self.config.lookback_days)
            
            # Save prepared training data
            training_file = RETRAIN_MODELS_DIR / f"{symbol}_training_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(training_file, index=False)
            
            logger.info(f"Prepared training data for {symbol}: {len(df)} rows")
            return str(training_file)
            
        except Exception as e:
            logger.error(f"Failed to prepare training data for {symbol}: {e}")
            return None
    
    def _create_training_config(self, model_name: str) -> 'TrainingConfig':
        """Create training configuration for model retraining"""
        # Import here to avoid circular imports
        try:
            from scripts.models.train_all_models import TrainingConfig
        except ImportError:
            # Fallback configuration
            class TrainingConfig:
                def __init__(self):
                    self.target_column = 'close'
                    self.test_size = 0.2
                    self.cv_folds = 3  # Reduced for faster retraining
                    self.n_trials = 25 if self.enable_hyperparameter_optimization else 1
                    self.timeout = self.training_timeout_minutes * 60
                    self.random_state = 42
                    
                    # Set specific model training flags
                    self.train_xgboost = model_name == 'xgboost'
                    self.train_lightgbm = model_name == 'lightgbm'
                    self.train_neural_network = model_name == 'neural_network'
        
        config = TrainingConfig()
        
        # Configure based on our retraining settings
        config.test_size = self.config.validation_split
        config.n_trials = 25 if self.config.enable_hyperparameter_optimization else 1
        config.timeout = self.config.training_timeout_minutes * 60
        config.cv_folds = 3  # Faster retraining with fewer folds
        
        return config
    
    def _train_model(self, training_data_path: str, training_config: 'TrainingConfig', 
                    report: ModelPerformanceReport) -> Optional[str]:
        """Train new model using training pipeline"""
        try:
            # Try to use training pipeline if available
            if 'ModelTrainingPipeline' in globals():
                pipeline = ModelTrainingPipeline(training_config)
                training_report = pipeline.train_all_models(training_data_path)
                
                # Find the best model for our type
                for result in training_report.model_results:
                    if result.model_name.lower() == report.model_name.lower():
                        return result.model_path
            
            # Fallback: basic model training
            return self._basic_model_training(training_data_path, report)
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return None
    
    def _basic_model_training(self, data_path: str, report: ModelPerformanceReport) -> Optional[str]:
        """Fallback basic model training"""
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Prepare features and target
            target_col = 'close'
            exclude_cols = {target_col, 'symbol', 'timestamp', 'date'}
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols].values
            y = df[target_col].values
            
            # Basic train-test split
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train simple model based on type
            if report.model_name == 'xgboost':
                import xgboost as xgb
                model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            elif report.model_name == 'lightgbm':
                import lightgbm as lgb
                model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Save model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = RETRAIN_MODELS_DIR / f"{report.model_name}_{report.symbol}_retrained_{timestamp}.pkl"
            joblib.dump(model, model_path)
            
            logger.info(f"Basic model training completed: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Basic model training failed: {e}")
            return None
    
    def _evaluate_new_model(self, model_path: str, symbol: str) -> float:
        """Evaluate performance of newly trained model"""
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Get test data (same as used in performance analysis)
            test_data = self._get_recent_test_data(symbol)
            
            if test_data is None:
                return float('inf')
            
            # Prepare test data
            X_test, y_test = self._prepare_test_data(test_data)
            
            if X_test is None:
                return float('inf')
            
            # Evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            return rmse
            
        except Exception as e:
            logger.error(f"New model evaluation failed: {e}")
            return float('inf')
    
    def _get_recent_test_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get recent test data (same as in performance analyzer)"""
        try:
            pattern = f"{symbol}_*.csv"
            data_files = list(PROCESSED_DATA_DIR.glob(pattern))
            
            if not data_files:
                return None
            
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            return df.tail(30) if len(df) >= 30 else df
            
        except Exception:
            return None
    
    def _prepare_test_data(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare test data (same as in performance analyzer)"""
        try:
            target_col = 'close'
            exclude_cols = {target_col, 'symbol', 'timestamp', 'date'}
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if not feature_cols:
                return None, None
            
            X = df[feature_cols].values
            y = df[target_col].values
            
            return X, y
            
        except Exception:
            return None, None
    
    def _should_deploy_model(self, improvement: float, new_performance: float) -> bool:
        """Determine if new model should be deployed"""
        if not self.config.auto_deploy_better_models:
            return False
        
        # Must show significant improvement
        if improvement < self.config.require_improvement_threshold:
            return False
        
        # Must meet minimum performance threshold
        if new_performance > self.config.minimum_performance_threshold:
            return False
        
        return True
    
    def _deploy_new_model(self, model_path: str, report: ModelPerformanceReport) -> bool:
        """Deploy new model to production"""
        try:
            # Backup old model if configured
            if self.config.backup_old_models:
                self._backup_old_model(report)
            
            # Copy new model to production
            model_file = Path(model_path)
            production_path = PRODUCTION_MODELS_DIR / f"{report.model_name}_{report.symbol}_production.pkl"
            
            shutil.copy2(model_path, production_path)
            
            # Update model registry if it exists
            self._update_model_registry(production_path, report)
            
            logger.info(f"✅ Deployed new model: {production_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False
    
    def _backup_old_model(self, report: ModelPerformanceReport) -> None:
        """Backup existing production model"""
        try:
            # Find existing production model
            production_files = list(PRODUCTION_MODELS_DIR.glob(f"{report.model_name}_{report.symbol}_*.pkl"))
            
            if production_files:
                for prod_file in production_files:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_name = f"{prod_file.stem}_backup_{timestamp}.pkl"
                    backup_path = BACKUP_MODELS_DIR / backup_name
                    
                    shutil.copy2(prod_file, backup_path)
                    logger.info(f"📦 Backed up model: {backup_path}")
                    
        except Exception as e:
            logger.warning(f"Model backup failed: {e}")
    
    def _update_model_registry(self, model_path: Path, report: ModelPerformanceReport) -> None:
        """Update model registry with new model information"""
        try:
            registry_file = PRODUCTION_MODELS_DIR / 'model_registry.json'
            
            # Load existing registry
            registry_data = {}
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
            
            # Update registry with new model
            model_id = f"{report.model_name}_{report.symbol}"
            registry_data[model_id] = {
                'model_name': report.model_name,
                'symbol': report.symbol,
                'model_path': str(model_path),
                'deployed_at': datetime.now().isoformat(),
                'deployment_status': 'active',
                'retrained': True,
                'previous_performance': report.current_performance
            }
            
            # Save updated registry
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Registry update failed: {e}")

# ============================================
# MAIN WEEKLY RETRAIN ORCHESTRATOR
# ============================================

class WeeklyRetrainOrchestrator:
    """Main orchestrator for weekly retraining process"""
    
    def __init__(self, config: RetrainConfig = None):
        self.config = config or RetrainConfig()
        self.start_time = None
        
        # Initialize components
        self.performance_analyzer = ModelPerformanceAnalyzer(self.config)
        self.retraining_engine = ModelRetrainingEngine(self.config)
    
    def run_weekly_retrain(self) -> WeeklyRetrainReport:
        """Run complete weekly retraining process"""
        logger.info("🚀 Starting weekly model retraining process...")
        self.start_time = time.time()
        
        try:
            # Step 1: Analyze model performance
            logger.info("📊 Step 1: Analyzing model performance...")
            performance_reports = self.performance_analyzer.analyze_all_models()
            
            # Step 2: Retrain models that need it
            logger.info("🔄 Step 2: Retraining models...")
            retraining_results = self.retraining_engine.retrain_models(performance_reports)
            
            # Step 3: Generate comprehensive report
            report = self._generate_final_report(performance_reports, retraining_results)
            
            # Step 4: Save results
            self._save_retrain_results(report)
            
            # Step 5: Send notifications if configured
            if self.config.send_notifications:
                self._send_notifications(report)
            
            # Step 6: Print summary
            self._print_summary(report)
            
            logger.info("✅ Weekly retraining process completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"❌ Weekly retraining process failed: {e}")
            logger.error(traceback.format_exc())
            
            # Create failure report
            execution_time = time.time() - self.start_time if self.start_time else 0
            return WeeklyRetrainReport(
                execution_date=datetime.now().strftime('%Y-%m-%d'),
                execution_time=execution_time,
                models_evaluated=0,
                models_needing_retrain=0,
                models_retrained=0,
                successful_retrains=0,
                failed_retrains=0,
                models_deployed=0,
                models_backed_up=0,
                performance_reports=[],
                retraining_results=[],
                overall_status='failed',
                recommendations=[f"Process failed: {e}"]
            )
    
    def _generate_final_report(self, performance_reports: List[ModelPerformanceReport],
                             retraining_results: List[RetrainingResult]) -> WeeklyRetrainReport:
        """Generate comprehensive final report"""
        execution_time = time.time() - self.start_time
        
        # Count results
        models_needing_retrain = sum(1 for report in performance_reports if report.needs_retraining)
        successful_retrains = sum(1 for result in retraining_results if result.training_status == 'success')
        failed_retrains = sum(1 for result in retraining_results if result.training_status == 'failed')
        models_deployed = sum(1 for result in retraining_results if result.deployed)
        
        # Determine overall status
        if failed_retrains == 0 and len(retraining_results) > 0:
            overall_status = 'success'
        elif successful_retrains > 0:
            overall_status = 'partial_success'
        else:
            overall_status = 'failed'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(performance_reports, retraining_results)
        
        return WeeklyRetrainReport(
            execution_date=datetime.now().strftime('%Y-%m-%d'),
            execution_time=execution_time,
            models_evaluated=len(performance_reports),
            models_needing_retrain=models_needing_retrain,
            models_retrained=len(retraining_results),
            successful_retrains=successful_retrains,
            failed_retrains=failed_retrains,
            models_deployed=models_deployed,
            models_backed_up=models_deployed,  # Assuming all deployed models were backed up
            performance_reports=performance_reports,
            retraining_results=retraining_results,
            overall_status=overall_status,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, performance_reports: List[ModelPerformanceReport],
                                retraining_results: List[RetrainingResult]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        poor_performers = [r for r in performance_reports if r.current_performance > 0.1]  # >10% error
        if poor_performers:
            recommendations.append(f"Review {len(poor_performers)} models with high error rates")
        
        # Data-based recommendations
        no_data = [r for r in performance_reports if not r.data_available]
        if no_data:
            recommendations.append(f"Update data for {len(no_data)} symbols with stale/missing data")
        
        # Training-based recommendations
        training_failures = [r for r in retraining_results if r.training_status == 'failed']
        if training_failures:
            recommendations.append(f"Investigate {len(training_failures)} training failures")
        
        # Deployment recommendations
        undeployed_improvements = [r for r in retraining_results 
                                 if r.improvement > 0.05 and not r.deployed]
        if undeployed_improvements:
            recommendations.append(f"Consider deploying {len(undeployed_improvements)} improved models")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Continue regular monitoring - system performing well")
        
        return recommendations
    
    def _save_retrain_results(self, report: WeeklyRetrainReport) -> None:
        """Save retraining results to files"""
        try:
            # Save detailed report
            report_file = REPORTS_DIR / f"weekly_retrain_{report.execution_date}.json"
            report.save(report_file)
            
            # Save summary for trending
            summary_file = REPORTS_DIR / "weekly_retrain_summary.json"
            self._update_summary_log(summary_file, report)
            
            logger.info(f"💾 Saved retraining report: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _update_summary_log(self, summary_file: Path, report: WeeklyRetrainReport) -> None:
        """Update summary log with latest results"""
        try:
            # Load existing summary
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
            else:
                summary_data = {'weekly_retrains': []}
            
            # Add current report summary
            summary_data['weekly_retrains'].append({
                'date': report.execution_date,
                'status': report.overall_status,
                'execution_time': report.execution_time,
                'models_evaluated': report.models_evaluated,
                'models_retrained': report.models_retrained,
                'successful_retrains': report.successful_retrains,
                'models_deployed': report.models_deployed
            })
            
            # Keep only last 12 weeks
            summary_data['weekly_retrains'] = summary_data['weekly_retrains'][-12:]
            
            # Save updated summary
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to update summary log: {e}")
    
    def _send_notifications(self, report: WeeklyRetrainReport) -> None:
        """Send notifications about retraining results"""
        try:
            # This is a placeholder for notification system
            # In production, integrate with email/Slack/Teams etc.
            
            subject = f"Weekly Retrain Report - {report.overall_status.title()}"
            
            message = f"""
Weekly Model Retraining Summary
==============================
Date: {report.execution_date}
Status: {report.overall_status.upper()}
Execution Time: {report.execution_time/60:.1f} minutes

Results:
- Models Evaluated: {report.models_evaluated}
- Models Needing Retrain: {report.models_needing_retrain}
- Models Successfully Retrained: {report.successful_retrains}
- Models Deployed: {report.models_deployed}

Recommendations:
"""
            
            for i, rec in enumerate(report.recommendations, 1):
                message += f"{i}. {rec}\n"
            
            logger.info(f"📧 Notification prepared: {subject}")
            # In production, send actual email/notification here
            
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
    
    def _print_summary(self, report: WeeklyRetrainReport) -> None:
        """Print summary to console"""
        print("\n" + "="*60)
        print("WEEKLY MODEL RETRAINING SUMMARY")
        print("="*60)
        print(f"Date: {report.execution_date}")
        print(f"Status: {report.overall_status.upper()}")
        print(f"Execution Time: {report.execution_time/60:.1f} minutes")
        
        print(f"\nPerformance Analysis:")
        print(f"  📊 Models Evaluated: {report.models_evaluated}")
        print(f"  🔄 Models Needing Retrain: {report.models_needing_retrain}")
        
        print(f"\nRetraining Results:")
        print(f"  🚀 Models Retrained: {report.models_retrained}")
        print(f"  ✅ Successful: {report.successful_retrains}")
        print(f"  ❌ Failed: {report.failed_retrains}")
        
        print(f"\nDeployment:")
        print(f"  🚀 Models Deployed: {report.models_deployed}")
        print(f"  📦 Models Backed Up: {report.models_backed_up}")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Show individual model results
        if report.retraining_results:
            print(f"\nDetailed Results:")
            print("-" * 40)
            for result in report.retraining_results:
                status_emoji = "✅" if result.training_status == 'success' else "❌"
                improvement_str = f"{result.improvement*100:+.1f}%" if result.improvement != 0 else "N/A"
                deployed_str = "🚀 Deployed" if result.deployed else "⏸️ Not deployed"
                
                print(f"{status_emoji} {result.model_name} ({result.symbol}): {improvement_str} improvement - {deployed_str}")

def load_config_from_file(config_path: str) -> RetrainConfig:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return RetrainConfig(**config_dict)
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        return RetrainConfig()

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='StockPredictionPro Weekly Model Retraining')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--retrain-all', action='store_true', help='Force retrain all models regardless of performance')
    parser.add_argument('--no-deploy', action='store_true', help='Skip automatic deployment of improved models')
    parser.add_argument('--models', nargs='+', help='Specific model types to retrain (xgboost, lightgbm, etc.)')
    parser.add_argument('--dry-run', action='store_true', help='Analyze only, no actual retraining')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = RetrainConfig()
    
    # Override config with command line arguments
    if args.retrain_all:
        config.retrain_all_models = True
    if args.no_deploy:
        config.auto_deploy_better_models = False
    if args.models:
        config.models_to_retrain = args.models
    
    # Run weekly retraining
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - Analysis only, no retraining")
        # Only run performance analysis
        analyzer = ModelPerformanceAnalyzer(config)
        performance_reports = analyzer.analyze_all_models()
        
        print(f"\nDry Run Analysis Results:")
        print(f"Models evaluated: {len(performance_reports)}")
        models_needing_retrain = sum(1 for r in performance_reports if r.needs_retraining)
        print(f"Models needing retrain: {models_needing_retrain}")
        
        for report in performance_reports:
            if report.needs_retraining:
                print(f"  - {report.model_name} ({report.symbol}): {report.recommendation}")
    else:
        orchestrator = WeeklyRetrainOrchestrator(config)
        report = orchestrator.run_weekly_retrain()
        
        # Exit with appropriate code
        if report.overall_status == 'success':
            sys.exit(0)
        elif report.overall_status == 'partial_success':
            sys.exit(1)
        else:
            sys.exit(2)

if __name__ == '__main__':
    main()
