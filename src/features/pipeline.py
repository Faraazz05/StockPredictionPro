# ============================================
# StockPredictionPro - src/features/pipeline.py
# Advanced feature engineering pipeline for financial machine learning
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
import warnings
from datetime import datetime
from pathlib import Path
import joblib
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# CORRECTED IMPORTS - Based on actual file structure
from .feature_store import FeatureStore, FeatureMetadata, create_feature_store

# Indicators imports - from individual files
from .indicators.momentum import MomentumIndicator, RelativeStrengthIndex, RateOfChange, MACD, StochasticOscillator, WilliamsR, CommodityChannelIndex
from .indicators.trend import SimpleMovingAverage, ExponentialMovingAverage, BollingerBands, AverageTrueRange
from .indicators.volatility import VolatilityIndicator, GARCHVolatility
from .indicators.volume import OnBalanceVolume, VWAP
from .indicators.custom import CustomIndicator

# Transformers imports
from .transformers.interactions import PolynomialInteractionTransformer, FinancialRatioTransformer, CorrelationInteractionTransformer, TimeSeriesInteractionTransformer
from .transformers.lags import SimpleLagTransformer, RollingLagTransformer
from .transformers.scalers import FinancialStandardScaler, FinancialRobustScaler
from .transformers.selectors import StatisticalSelector, ModelBasedSelector, FinancialCorrelationSelector

# Targets imports
from .targets.classification import create_classification_target
from .targets.regression import create_regression_target
from .targets.multi_horizon import create_multi_horizon_targets

from ..utils.exceptions import ValidationError, CalculationError
from ..utils.logger import get_logger
from ..utils.timing import time_it

logger = get_logger('features.pipeline')

# ============================================
# Configuration Classes
# ============================================

@dataclass
class FeaturePipelineConfig:
    """Configuration for feature engineering pipeline"""
    # Data sources
    input_data_path: Optional[str] = None
    output_data_path: Optional[str] = None
    
    # Pipeline components
    indicators_config: Dict[str, Any] = field(default_factory=dict)
    transformers_config: Dict[str, Any] = field(default_factory=dict)
    targets_config: Dict[str, Any] = field(default_factory=dict)
    
    # Processing options
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 10000
    memory_limit_mb: int = 2048
    
    # Feature store integration
    use_feature_store: bool = True
    feature_store_config: Dict[str, Any] = field(default_factory=dict)
    save_intermediate: bool = False
    
    # Pipeline behavior
    handle_missing_data: str = 'interpolate'  # 'drop', 'interpolate', 'forward_fill'
    validate_outputs: bool = True
    enable_caching: bool = True
    cache_dir: str = "./pipeline_cache"
    
    # Logging and monitoring
    log_level: str = 'INFO'
    save_pipeline_state: bool = True
    pipeline_state_path: str = "./pipeline_state.pkl"

@dataclass
class PipelineStage:
    """Configuration for a single pipeline stage"""
    name: str
    stage_type: str  # 'indicator', 'transformer', 'target'
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_features: List[str] = field(default_factory=list)
    output_features: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    description: str = ""

# ============================================
# Main Feature Pipeline Class (with corrected mappings)
# ============================================

class FeaturePipeline:
    """
    Advanced feature engineering pipeline for financial machine learning.
    Orchestrates the creation, transformation, and storage of financial features.
    """
    
    def __init__(self, config: Optional[FeaturePipelineConfig] = None):
        self.config = config or FeaturePipelineConfig()
        self.stages: List[PipelineStage] = []
        self.feature_store: Optional[FeatureStore] = None
        self.fitted_transformers: Dict[str, Any] = {}
        self.pipeline_state: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.is_fitted: bool = False
        
        # Initialize feature store if enabled
        if self.config.use_feature_store:
            self._initialize_feature_store()
        
        # Setup caching directory
        if self.config.enable_caching:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _initialize_feature_store(self):
        """Initialize the feature store"""
        try:
            store_config = self.config.feature_store_config
            self.feature_store = create_feature_store(**store_config)
            logger.info("Feature store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize feature store: {e}")
            self.feature_store = None
    
    def add_indicator_stage(self, 
                           name: str,
                           indicator_type: str,
                           parameters: Optional[Dict[str, Any]] = None,
                           input_features: Optional[List[str]] = None,
                           description: str = "") -> 'FeaturePipeline':
        """Add an indicator calculation stage to the pipeline"""
        
        # CORRECTED indicator mapping based on actual file structure
        indicator_map = {
            # From indicators/trend.py
            'sma': SimpleMovingAverage,
            'ema': ExponentialMovingAverage,
            'bollinger': BollingerBands,
            'atr': AverageTrueRange,
            
            # From indicators/momentum.py
            'rsi': RelativeStrengthIndex,
            'macd': MACD,
            'momentum': MomentumIndicator,
            'roc': RateOfChange,
            'stochastic': StochasticOscillator,
            'williams_r': WilliamsR,
            'cci': CommodityChannelIndex,
            
            # From indicators/volume.py
            'obv': OnBalanceVolume,
            'vwap': VWAP,
            
            # From indicators/volatility.py
            'volatility': VolatilityIndicator,
            'garch_vol': GARCHVolatility,
            
            # From indicators/custom.py
            'custom': CustomIndicator,
        }
        
        if indicator_type not in indicator_map:
            raise ValueError(f"Unknown indicator type: {indicator_type}. Available: {list(indicator_map.keys())}")
        
        stage = PipelineStage(
            name=name,
            stage_type='indicator',
            function=indicator_map[indicator_type],
            parameters=parameters or {},
            input_features=input_features or ['open', 'high', 'low', 'close', 'volume'],
            output_features=[name],
            description=description
        )
        
        self.stages.append(stage)
        logger.info(f"Added indicator stage: {name} ({indicator_type})")
        return self
    
    def add_transformer_stage(self,
                             name: str,
                             transformer_type: str,
                             parameters: Optional[Dict[str, Any]] = None,
                             input_features: Optional[List[str]] = None,
                             description: str = "") -> 'FeaturePipeline':
        """Add a feature transformation stage to the pipeline"""
        
        # CORRECTED transformer mapping
        transformer_map = {
            # From transformers/interactions.py
            'polynomial': PolynomialInteractionTransformer,
            'financial_ratios': FinancialRatioTransformer,
            'correlation': CorrelationInteractionTransformer,
            'time_series': TimeSeriesInteractionTransformer,
            
            # From transformers/lags.py
            'lag': SimpleLagTransformer,
            'rolling_lag': RollingLagTransformer,
            
            # From transformers/scalers.py
            'scaler': FinancialStandardScaler,
            'robust_scaler': FinancialRobustScaler,
            
            # From transformers/selectors.py
            'selector': StatisticalSelector,
            'model_selector': ModelBasedSelector,
            'correlation_selector': FinancialCorrelationSelector,
        }
        
        if transformer_type not in transformer_map:
            raise ValueError(f"Unknown transformer type: {transformer_type}. Available: {list(transformer_map.keys())}")
        
        stage = PipelineStage(
            name=name,
            stage_type='transformer',
            function=transformer_map[transformer_type],
            parameters=parameters or {},
            input_features=input_features or [],
            output_features=[f"{name}_output"],
            description=description
        )
        
        self.stages.append(stage)
        logger.info(f"Added transformer stage: {name} ({transformer_type})")
        return self
    
    def add_target_stage(self,
                        name: str,
                        target_type: str,
                        parameters: Optional[Dict[str, Any]] = None,
                        input_features: Optional[List[str]] = None,
                        description: str = "") -> 'FeaturePipeline':
        """Add a target creation stage to the pipeline"""
        
        # CORRECTED target mapping
        target_map = {
            # From targets/regression.py
            'returns': create_regression_target,
            'volatility': create_regression_target,
            
            # From targets/classification.py
            'direction': create_classification_target,
            
            # From targets/multi_horizon.py
            'multi_horizon': create_multi_horizon_targets,
        }
        
        if target_type not in target_map:
            raise ValueError(f"Unknown target type: {target_type}. Available: {list(target_map.keys())}")
        
        stage = PipelineStage(
            name=name,
            stage_type='target',
            function=target_map[target_type],
            parameters=parameters or {},
            input_features=input_features or ['close'],
            output_features=[f"{name}_target"],
            description=description
        )
        
        self.stages.append(stage)
        logger.info(f"Added target stage: {name} ({target_type})")
        return self

    
    def add_target_stage(self,
                        name: str,
                        target_type: str,
                        parameters: Optional[Dict[str, Any]] = None,
                        input_features: Optional[List[str]] = None,
                        description: str = "") -> 'FeaturePipeline':
        """
        Add a target creation stage to the pipeline
        """
        
        # Map target types to functions - FIXED
        target_map = {
            'returns': create_regression_target,
            'direction': create_classification_target,
            'volatility': create_regression_target,
            'multi_horizon': create_multi_horizon_targets,
        }
        
        if target_type not in target_map:
            raise ValueError(f"Unknown target type: {target_type}")
        
        stage = PipelineStage(
            name=name,
            stage_type='target',
            function=target_map[target_type],
            parameters=parameters or {},
            input_features=input_features or ['close'],
            output_features=[f"{name}_target"],
            description=description
        )
        
        self.stages.append(stage)
        logger.info(f"Added target stage: {name} ({target_type})")
        return self
    
    def add_custom_stage(self,
                        name: str,
                        function: Callable,
                        stage_type: str = 'custom',
                        parameters: Optional[Dict[str, Any]] = None,
                        input_features: Optional[List[str]] = None,
                        output_features: Optional[List[str]] = None,
                        dependencies: Optional[List[str]] = None,
                        description: str = "") -> 'FeaturePipeline':
        """
        Add a custom function stage to the pipeline
        
        Args:
            name: Stage name
            function: Custom function to execute
            stage_type: Stage type for organization
            parameters: Function parameters
            input_features: Input feature names
            output_features: Output feature names
            dependencies: Stage dependencies
            description: Stage description
            
        Returns:
            Self for method chaining
        """
        
        stage = PipelineStage(
            name=name,
            stage_type=stage_type,
            function=function,
            parameters=parameters or {},
            input_features=input_features or [],
            output_features=output_features or [name],
            dependencies=dependencies or [],
            description=description
        )
        
        self.stages.append(stage)
        logger.info(f"Added custom stage: {name}")
        return self
    
    def _resolve_dependencies(self) -> List[PipelineStage]:
        """Resolve stage dependencies and return execution order"""
        
        # Simple topological sort for dependency resolution
        stages_by_name = {stage.name: stage for stage in self.stages}
        resolved_order = []
        temporary_mark = set()
        permanent_mark = set()
        
        def visit(stage_name: str):
            if stage_name in permanent_mark:
                return
            if stage_name in temporary_mark:
                raise ValueError(f"Circular dependency detected involving {stage_name}")
            
            temporary_mark.add(stage_name)
            
            stage = stages_by_name.get(stage_name)
            if stage:
                for dependency in stage.dependencies:
                    visit(dependency)
                
                permanent_mark.add(stage_name)
                resolved_order.append(stage)
            
            temporary_mark.remove(stage_name)
        
        # Visit all stages
        for stage in self.stages:
            if stage.name not in permanent_mark:
                visit(stage.name)
        
        return resolved_order
    
    def _execute_stage(self, 
                      stage: PipelineStage, 
                      data: Dict[str, Union[pd.DataFrame, np.ndarray]]) -> Dict[str, Any]:
        """Execute a single pipeline stage"""
        
        stage_start_time = datetime.now()
        
        try:
            # Prepare input data
            if stage.input_features:
                input_data = {feat: data[feat] for feat in stage.input_features if feat in data}
            else:
                input_data = data
            
            # Execute stage based on type
            if stage.stage_type == 'indicator':
                result = self._execute_indicator_stage(stage, input_data)
            elif stage.stage_type == 'transformer':
                result = self._execute_transformer_stage(stage, input_data)
            elif stage.stage_type == 'target':
                result = self._execute_target_stage(stage, input_data)
            else:
                result = self._execute_custom_stage(stage, input_data)
            
            # Store results in feature store if enabled
            if self.feature_store and self.config.save_intermediate:
                self._save_stage_results(stage, result)
            
            # Record execution info
            execution_time = (datetime.now() - stage_start_time).total_seconds()
            execution_info = {
                'stage_name': stage.name,
                'stage_type': stage.stage_type,
                'execution_time': execution_time,
                'success': True,
                'timestamp': stage_start_time,
                'output_shape': self._get_result_shape(result)
            }
            
            self.execution_history.append(execution_info)
            
            logger.info(f"Stage '{stage.name}' completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_info = {
                'stage_name': stage.name,
                'stage_type': stage.stage_type,
                'execution_time': (datetime.now() - stage_start_time).total_seconds(),
                'success': False,
                'error': str(e),
                'timestamp': stage_start_time
            }
            
            self.execution_history.append(execution_info)
            logger.error(f"Stage '{stage.name}' failed: {e}")
            raise
    
    def _execute_indicator_stage(self, stage: PipelineStage, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an indicator stage"""
        
        # Create indicator instance
        indicator_class = stage.function
        indicator = indicator_class(**stage.parameters)
        
        # Prepare data for indicator
        if isinstance(list(input_data.values())[0], pd.DataFrame):
            # DataFrame input
            df = list(input_data.values())[0]
            result = indicator.calculate(df)
        else:
            # Multiple series input
            df = pd.DataFrame(input_data)
            result = indicator.calculate(df)
        
        # Store fitted indicator
        self.fitted_transformers[stage.name] = indicator
        
        return {stage.name: result.values}
    
    def _execute_transformer_stage(self, stage: PipelineStage, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a transformer stage"""
        
        # Create transformer instance
        transformer_class = stage.function
        transformer = transformer_class(**stage.parameters)
        
        # Prepare input data
        if len(input_data) == 1 and isinstance(list(input_data.values())[0], pd.DataFrame):
            X = list(input_data.values())[0].values
        else:
            # Combine multiple features
            feature_arrays = []
            for feat_name in stage.input_features:
                if feat_name in input_data:
                    feat_data = input_data[feat_name]
                    if isinstance(feat_data, pd.Series):
                        feat_data = feat_data.values
                    elif isinstance(feat_data, pd.DataFrame):
                        feat_data = feat_data.values.flatten()
                    feature_arrays.append(feat_data)
            
            if feature_arrays:
                min_length = min(len(arr) for arr in feature_arrays)
                X = np.column_stack([arr[:min_length] for arr in feature_arrays])
            else:
                raise ValueError(f"No valid input features for transformer {stage.name}")
        
        # Fit and transform
        if hasattr(transformer, 'fit_transform'):
            result = transformer.fit_transform(X)
        else:
            transformer.fit(X)
            result = transformer.transform(X)
        
        # Store fitted transformer
        self.fitted_transformers[stage.name] = transformer
        
        # Handle multi-output transformers
        if isinstance(result, dict):
            return result
        else:
            return {stage.name: result}
    
    def _execute_target_stage(self, stage: PipelineStage, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a target creation stage"""
        
        # Get target function and parameters
        target_function = stage.function
        parameters = stage.parameters.copy()
        
        # Prepare input data (usually price data)
        if len(input_data) == 1:
            data = list(input_data.values())[0]
            if isinstance(data, pd.Series):
                data = data.values
            elif isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0].values  # Take first column
        else:
            # Use close price if available
            if 'close' in input_data:
                data = input_data['close']
                if isinstance(data, pd.Series):
                    data = data.values
            else:
                raise ValueError(f"No suitable input data for target {stage.name}")
        
        # Create target
        if 'target_type' not in parameters:
            # Infer target type from stage name
            if 'returns' in stage.name.lower():
                parameters['target_type'] = 'returns'
            elif 'direction' in stage.name.lower():
                parameters['target_type'] = 'direction'
            elif 'volatility' in stage.name.lower():
                parameters['target_type'] = 'volatility'
        
        result = target_function(data, **parameters)
        
        # Handle different result formats
        if isinstance(result, tuple):
            targets, creator = result
            self.fitted_transformers[f"{stage.name}_creator"] = creator
            return {stage.name: targets}
        else:
            return {stage.name: result}
    
    def _execute_custom_stage(self, stage: PipelineStage, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom stage"""
        
        # Call custom function with input data and parameters
        if stage.parameters:
            result = stage.function(input_data, **stage.parameters)
        else:
            result = stage.function(input_data)
        
        # Ensure result is in expected format
        if not isinstance(result, dict):
            result = {stage.name: result}
        
        return result
    
    def _save_stage_results(self, stage: PipelineStage, results: Dict[str, Any]):
        """Save stage results to feature store"""
        
        if not self.feature_store:
            return
        
        for output_name, output_data in results.items():
            try:
                # Convert to numpy array if needed
                if isinstance(output_data, (pd.Series, pd.DataFrame)):
                    data_array = output_data.values
                else:
                    data_array = np.asarray(output_data)
                
                # Create metadata
                metadata = FeatureMetadata(
                    feature_name=output_name,
                    feature_type=stage.stage_type,
                    data_type='numerical',
                    creation_date=datetime.now(),
                    last_updated=datetime.now(),
                    description=f"Output from {stage.name} stage: {stage.description}",
                    parameters=stage.parameters,
                    dependencies=stage.input_features,
                    tags=[stage.stage_type, 'pipeline_output']
                )
                
                # Save to feature store
                success = self.feature_store.save_feature(
                    output_name, 
                    data_array, 
                    stage.stage_type,
                    metadata.description,
                    metadata.parameters,
                    metadata.dependencies,
                    metadata.tags,
                    overwrite=True
                )
                
                if success:
                    logger.debug(f"Saved {output_name} to feature store")
                else:
                    logger.warning(f"Failed to save {output_name} to feature store")
                    
            except Exception as e:
                logger.warning(f"Error saving {output_name} to feature store: {e}")
    
    def _get_result_shape(self, result: Dict[str, Any]) -> Dict[str, Tuple]:
        """Get shapes of result arrays"""
        shapes = {}
        for name, data in result.items():
            if isinstance(data, np.ndarray):
                shapes[name] = data.shape
            elif isinstance(data, (pd.Series, pd.DataFrame)):
                shapes[name] = data.shape
            else:
                shapes[name] = (len(data),) if hasattr(data, '__len__') else None
        return shapes
    
    @time_it("pipeline_fit")
    def fit(self, data: Union[pd.DataFrame, Dict[str, Union[pd.Series, np.ndarray]]]) -> 'FeaturePipeline':
        """
        Fit the feature pipeline
        
        Args:
            data: Input data (DataFrame or dict of series/arrays)
            
        Returns:
            Self for method chaining
        """
        
        logger.info(f"Fitting feature pipeline with {len(self.stages)} stages")
        
        # Convert input data to consistent format
        if isinstance(data, pd.DataFrame):
            working_data = {col: data[col] for col in data.columns}
        else:
            working_data = data.copy()
        
        # Resolve stage dependencies
        execution_order = self._resolve_dependencies()
        
        # Execute stages in order
        for stage in execution_order:
            if not stage.enabled:
                logger.info(f"Skipping disabled stage: {stage.name}")
                continue
            
            logger.info(f"Fitting stage: {stage.name} ({stage.stage_type})")
            
            # Execute stage
            stage_results = self._execute_stage(stage, working_data)
            
            # Add results to working data
            working_data.update(stage_results)
        
        self.is_fitted = True
        
        # Save pipeline state if enabled
        if self.config.save_pipeline_state:
            self._save_pipeline_state()
        
        logger.info("Pipeline fitting completed successfully")
        return self
    
    @time_it("pipeline_transform")
    def transform(self, data: Union[pd.DataFrame, Dict[str, Union[pd.Series, np.ndarray]]]) -> Dict[str, np.ndarray]:
        """
        Transform data using the fitted pipeline
        
        Args:
            data: Input data
            
        Returns:
            Dictionary of transformed features
        """
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming")
        
        logger.info("Transforming data through pipeline")
        
        # Convert input data to consistent format
        if isinstance(data, pd.DataFrame):
            working_data = {col: data[col] for col in data.columns}
        else:
            working_data = data.copy()
        
        # Resolve stage dependencies
        execution_order = self._resolve_dependencies()
        
        # Execute stages in order (transform mode)
        for stage in execution_order:
            if not stage.enabled:
                continue
            
            logger.debug(f"Transforming through stage: {stage.name}")
            
            # Use fitted transformers for transform
            if stage.name in self.fitted_transformers:
                fitted_transformer = self.fitted_transformers[stage.name]
                
                # Prepare input data for transformer
                if stage.input_features:
                    input_data = {feat: working_data[feat] for feat in stage.input_features if feat in working_data}
                else:
                    input_data = working_data
                
                # Apply transformation
                if hasattr(fitted_transformer, 'transform'):
                    # Prepare data format for transformer
                    if len(input_data) == 1 and isinstance(list(input_data.values())[0], pd.Series):
                        X = list(input_data.values())[0].values.reshape(-1, 1)
                    elif len(input_data) == 1 and isinstance(list(input_data.values())[0], pd.DataFrame):
                        X = list(input_data.values())[0].values
                    else:
                        # Multiple features
                        feature_arrays = []
                        for feat_name in stage.input_features:
                            if feat_name in input_data:
                                feat_data = input_data[feat_name]
                                if isinstance(feat_data, pd.Series):
                                    feat_data = feat_data.values
                                feature_arrays.append(feat_data)
                        
                        if feature_arrays:
                            min_length = min(len(arr) for arr in feature_arrays)
                            X = np.column_stack([arr[:min_length] for arr in feature_arrays])
                        else:
                            continue
                    
                    result = fitted_transformer.transform(X)
                    
                    if isinstance(result, dict):
                        working_data.update(result)
                    else:
                        working_data[stage.name] = result
                
                elif hasattr(fitted_transformer, 'calculate'):
                    # Indicator case
                    if isinstance(list(input_data.values())[0], pd.DataFrame):
                        df = list(input_data.values())[0]
                    else:
                        df = pd.DataFrame(input_data)
                    
                    result = fitted_transformer.calculate(df)
                    working_data[stage.name] = result.values
        
        # Return final results (excluding original input features)
        input_features = set(data.columns if isinstance(data, pd.DataFrame) else data.keys())
        output_features = {k: v for k, v in working_data.items() if k not in input_features}
        
        logger.info(f"Pipeline transformation completed, generated {len(output_features)} features")
        return output_features
    
    def fit_transform(self, data: Union[pd.DataFrame, Dict[str, Union[pd.Series, np.ndarray]]]) -> Dict[str, np.ndarray]:
        """Fit the pipeline and transform data in one step"""
        return self.fit(data).transform(data)
    
    def _save_pipeline_state(self):
        """Save pipeline state to disk"""
        try:
            state = {
                'config': self.config,
                'stages': self.stages,
                'fitted_transformers': self.fitted_transformers,
                'execution_history': self.execution_history,
                'is_fitted': self.is_fitted,
                'timestamp': datetime.now()
            }
            
            with open(self.config.pipeline_state_path, 'wb') as f:
                joblib.dump(state, f)
            
            logger.info(f"Pipeline state saved to {self.config.pipeline_state_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save pipeline state: {e}")
    
    def load_pipeline_state(self, state_path: str):
        """Load pipeline state from disk"""
        try:
            with open(state_path, 'rb') as f:
                state = joblib.load(f)
            
            self.config = state['config']
            self.stages = state['stages']
            self.fitted_transformers = state['fitted_transformers']
            self.execution_history = state['execution_history']
            self.is_fitted = state['is_fitted']
            
            logger.info(f"Pipeline state loaded from {state_path}")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline state: {e}")
            raise
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary"""
        
        summary = {
            'total_stages': len(self.stages),
            'enabled_stages': sum(1 for stage in self.stages if stage.enabled),
            'stage_types': {},
            'execution_history': len(self.execution_history),
            'is_fitted': self.is_fitted,
            'feature_store_enabled': self.feature_store is not None
        }
        
        # Count stages by type
        for stage in self.stages:
            stage_type = stage.stage_type
            summary['stage_types'][stage_type] = summary['stage_types'].get(stage_type, 0) + 1
        
        # Execution statistics
        if self.execution_history:
            successful_runs = sum(1 for run in self.execution_history if run['success'])
            total_time = sum(run['execution_time'] for run in self.execution_history)
            
            summary['execution_stats'] = {
                'successful_runs': successful_runs,
                'failed_runs': len(self.execution_history) - successful_runs,
                'total_execution_time': total_time,
                'average_stage_time': total_time / len(self.execution_history) if self.execution_history else 0
            }
        
        return summary
    
    def get_stage_info(self, stage_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific stage"""
        
        stage = next((s for s in self.stages if s.name == stage_name), None)
        
        if not stage:
            raise ValueError(f"Stage '{stage_name}' not found")
        
        info = {
            'name': stage.name,
            'type': stage.stage_type,
            'enabled': stage.enabled,
            'description': stage.description,
            'parameters': stage.parameters,
            'input_features': stage.input_features,
            'output_features': stage.output_features,
            'dependencies': stage.dependencies,
            'is_fitted': stage.name in self.fitted_transformers
        }
        
        # Add execution history for this stage
        stage_executions = [run for run in self.execution_history if run['stage_name'] == stage_name]
        if stage_executions:
            info['execution_history'] = {
                'total_runs': len(stage_executions),
                'successful_runs': sum(1 for run in stage_executions if run['success']),
                'average_time': np.mean([run['execution_time'] for run in stage_executions]),
                'last_run': stage_executions[-1]
            }
        
        return info
    
    def enable_stage(self, stage_name: str):
        """Enable a pipeline stage"""
        stage = next((s for s in self.stages if s.name == stage_name), None)
        if stage:
            stage.enabled = True
            logger.info(f"Enabled stage: {stage_name}")
        else:
            raise ValueError(f"Stage '{stage_name}' not found")
    
    def disable_stage(self, stage_name: str):
        """Disable a pipeline stage"""
        stage = next((s for s in self.stages if s.name == stage_name), None)
        if stage:
            stage.enabled = False
            logger.info(f"Disabled stage: {stage_name}")
        else:
            raise ValueError(f"Stage '{stage_name}' not found")
    
    def remove_stage(self, stage_name: str):
        """Remove a stage from the pipeline"""
        self.stages = [s for s in self.stages if s.name != stage_name]
        
        # Remove fitted transformer if exists
        if stage_name in self.fitted_transformers:
            del self.fitted_transformers[stage_name]
        
        logger.info(f"Removed stage: {stage_name}")

# ============================================
# Pipeline Builder Class
# ============================================

class PipelineBuilder:
    """
    Builder class for creating feature pipelines with fluent interface
    """
    
    def __init__(self, config: Optional[FeaturePipelineConfig] = None):
        self.pipeline = FeaturePipeline(config)
    
    def with_indicators(self, indicators_config: Dict[str, Dict[str, Any]]) -> 'PipelineBuilder':
        """Add multiple indicator stages from configuration"""
        
        for indicator_name, indicator_config in indicators_config.items():
            indicator_type = indicator_config.pop('type')
            self.pipeline.add_indicator_stage(
                name=indicator_name,
                indicator_type=indicator_type,
                **indicator_config
            )
        
        return self
    
    def with_transformers(self, transformers_config: Dict[str, Dict[str, Any]]) -> 'PipelineBuilder':
        """Add multiple transformer stages from configuration"""
        
        for transformer_name, transformer_config in transformers_config.items():
            transformer_type = transformer_config.pop('type')
            self.pipeline.add_transformer_stage(
                name=transformer_name,
                transformer_type=transformer_type,
                **transformer_config
            )
        
        return self
    
    def with_targets(self, targets_config: Dict[str, Dict[str, Any]]) -> 'PipelineBuilder':
        """Add multiple target stages from configuration"""
        
        for target_name, target_config in targets_config.items():
            target_type = target_config.pop('type')
            self.pipeline.add_target_stage(
                name=target_name,
                target_type=target_type,
                **target_config
            )
        
        return self
    
    def build(self) -> FeaturePipeline:
        """Build and return the configured pipeline"""
        return self.pipeline

# ============================================
# Utility Functions
# ============================================

def create_default_pipeline() -> FeaturePipeline:
    """Create a pipeline with common financial features"""
    
    pipeline = FeaturePipeline()
    
    # Add common indicators
    pipeline.add_indicator_stage('sma_20', 'sma', {'period': 20})
    pipeline.add_indicator_stage('sma_50', 'sma', {'period': 50})
    pipeline.add_indicator_stage('ema_12', 'ema', {'period': 12})
    pipeline.add_indicator_stage('ema_26', 'ema', {'period': 26})
    pipeline.add_indicator_stage('rsi_14', 'rsi', {'period': 14})
    pipeline.add_indicator_stage('macd', 'macd', {'fast': 12, 'slow': 26, 'signal': 9})
    pipeline.add_indicator_stage('bollinger', 'bollinger', {'period': 20, 'std_dev': 2})
    pipeline.add_indicator_stage('atr_14', 'atr', {'period': 14})
    
    # Add transformers
    pipeline.add_transformer_stage(
        'polynomial_features', 'polynomial',
        {'degree': 2, 'interaction_only': True},
        input_features=['sma_20', 'ema_12', 'rsi_14']
    )
    
    pipeline.add_transformer_stage(
        'lag_features', 'lag',
        {'max_lag': 5},
        input_features=['close', 'volume']
    )
    
    # Add targets
    pipeline.add_target_stage(
        'returns_1d', 'returns',
        {'target_type': 'returns', 'lookahead_periods': 1},
        input_features=['close']
    )
    
    pipeline.add_target_stage(
        'direction_1d', 'direction',
        {'target_type': 'direction'},
        input_features=['close']
    )
    
    return pipeline

def load_pipeline_from_config(config_path: str) -> FeaturePipeline:
    """Load pipeline configuration from YAML file"""
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create pipeline config
        pipeline_config = FeaturePipelineConfig(**config_data.get('pipeline_config', {}))
        
        # Build pipeline
        builder = PipelineBuilder(pipeline_config)
        
        if 'indicators' in config_data:
            builder.with_indicators(config_data['indicators'])
        
        if 'transformers' in config_data:
            builder.with_transformers(config_data['transformers'])
        
        if 'targets' in config_data:
            builder.with_targets(config_data['targets'])
        
        return builder.build()
        
    except Exception as e:
        logger.error(f"Failed to load pipeline from config: {e}")
        raise

def save_pipeline_config(pipeline: FeaturePipeline, config_path: str):
    """Save pipeline configuration to YAML file"""
    
    config_data = {
        'pipeline_config': {
            'parallel_processing': pipeline.config.parallel_processing,
            'max_workers': pipeline.config.max_workers,
            'use_feature_store': pipeline.config.use_feature_store,
            'save_intermediate': pipeline.config.save_intermediate,
            'validate_outputs': pipeline.config.validate_outputs,
            'enable_caching': pipeline.config.enable_caching
        },
        'indicators': {},
        'transformers': {},
        'targets': {}
    }
    
    # Extract stage configurations
    for stage in pipeline.stages:
        stage_config = {
            'type': stage.stage_type,
            'parameters': stage.parameters,
            'input_features': stage.input_features,
            'description': stage.description
        }
        
        if stage.stage_type == 'indicator':
            config_data['indicators'][stage.name] = stage_config
        elif stage.stage_type == 'transformer':
            config_data['transformers'][stage.name] = stage_config
        elif stage.stage_type == 'target':
            config_data['targets'][stage.name] = stage_config
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Pipeline configuration saved to {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save pipeline configuration: {e}")
        raise

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    print("Testing Feature Pipeline")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic OHLCV data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    returns = np.random.normal(0.001, 0.02, n_samples)
    
    prices = 100 * np.cumprod(1 + returns)
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
    volume = np.random.lognormal(10, 0.5, n_samples)
    
    market_data = pd.DataFrame({
        'date': dates,
        'open': np.roll(prices, 1),
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })
    
    market_data['open'].iloc[0] = market_data['close'].iloc[0]
    
    print(f"Created sample data: {market_data.shape}")
    
    # Test basic pipeline creation
    print("\n1. Testing Basic Pipeline Creation")
    
    pipeline = FeaturePipeline()
    
    # Add indicator stages
    pipeline.add_indicator_stage('sma_20', 'sma', {'period': 20})
    pipeline.add_indicator_stage('rsi_14', 'rsi', {'period': 14})
    pipeline.add_indicator_stage('macd', 'macd', {'fast': 12, 'slow': 26, 'signal': 9})
    
    # Add transformer stages
    pipeline.add_transformer_stage(
        'lag_features', 'lag',
        {'max_lag': 3},
        input_features=['close']
    )
    
    # Add target stages
    pipeline.add_target_stage(
        'returns_1d', 'returns',
        {'target_type': 'returns', 'lookahead_periods': 1},
        input_features=['close']
    )
    
    print(f"Created pipeline with {len(pipeline.stages)} stages")
    
    # Test pipeline fitting
    print("\n2. Testing Pipeline Fitting")
    
    pipeline.fit(market_data)
    
    summary = pipeline.get_pipeline_summary()
    print("Pipeline Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test pipeline transformation
    print("\n3. Testing Pipeline Transformation")
    
    # Create new test data
    test_data = market_data.iloc[-200:].copy()  # Last 200 days
    
    features = pipeline.transform(test_data)
    
    print(f"Generated features: {list(features.keys())}")
    for name, data in features.items():
        if isinstance(data, np.ndarray):
            print(f"  {name}: shape={data.shape}")
        else:
            print(f"  {name}: type={type(data)}")
    
    # Test pipeline builder
    print("\n4. Testing Pipeline Builder")
    
    indicators_config = {
        'sma_10': {'type': 'sma', 'parameters': {'period': 10}},
        'ema_20': {'type': 'ema', 'parameters': {'period': 20}},
        'rsi_14': {'type': 'rsi', 'parameters': {'period': 14}}
    }
    
    transformers_config = {
        'polynomial': {
            'type': 'polynomial',
            'parameters': {'degree': 2},
            'input_features': ['sma_10', 'ema_20']
        }
    }
    
    targets_config = {
        'returns': {
            'type': 'returns',
            'parameters': {'target_type': 'returns'},
            'input_features': ['close']
        }
    }
    
    built_pipeline = (PipelineBuilder()
                     .with_indicators(indicators_config)
                     .with_transformers(transformers_config)
                     .with_targets(targets_config)
                     .build())
    
    print(f"Built pipeline has {len(built_pipeline.stages)} stages")
    
    # Test default pipeline
    print("\n5. Testing Default Pipeline")
    
    default_pipeline = create_default_pipeline()
    default_summary = default_pipeline.get_pipeline_summary()
    
    print("Default Pipeline Summary:")
    for key, value in default_summary.items():
        print(f"  {key}: {value}")
    
    # Test stage management
    print("\n6. Testing Stage Management")
    
    # Get stage info
    stage_info = pipeline.get_stage_info('sma_20')
    print("SMA_20 Stage Info:")
    for key, value in stage_info.items():
        print(f"  {key}: {value}")
    
    # Test stage enable/disable
    pipeline.disable_stage('rsi_14')
    pipeline.enable_stage('rsi_14')
    
    print("Stage management completed")
    
    # Test execution history
    print("\n7. Testing Execution History")
    
    print(f"Execution history: {len(pipeline.execution_history)} entries")
    if pipeline.execution_history:
        latest_execution = pipeline.execution_history[-1]
        print("Latest execution:")
        for key, value in latest_execution.items():
            print(f"  {key}: {value}")
    
    print("\nFeature pipeline testing completed successfully!")
