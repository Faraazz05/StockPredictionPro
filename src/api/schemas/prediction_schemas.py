# ============================================
# StockPredictionPro - src/api/schemas/prediction_schemas.py
# Comprehensive Pydantic schemas for prediction and signal generation API endpoints
# ============================================

from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Union, Literal, Annotated
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# ============================================
# Type Aliases and Constants
# ============================================

SymbolStr = Annotated[str, Field(min_length=1, max_length=12)]
ModelIdStr = Annotated[str, Field(min_length=1, max_length=50)]
PositiveFloat = Annotated[float, Field(gt=0)]
PositiveInt = Annotated[int, Field(gt=0)]
ScoreFloat = Annotated[float, Field(ge=0.0, le=1.0)]
PercentageFloat = Annotated[float, Field(ge=-100.0, le=100.0)]

# ============================================
# Enums and Constants
# ============================================

class PredictionType(str, Enum):
    """Types of predictions"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    PROBABILITY = "probability"
    TIME_SERIES = "time_series"
    MULTI_TARGET = "multi_target"

class PredictionHorizon(str, Enum):
    """Prediction time horizons"""
    INTRADAY = "intraday"
    NEXT_DAY = "next_day"
    NEXT_WEEK = "next_week"
    NEXT_MONTH = "next_month"
    NEXT_QUARTER = "next_quarter"
    CUSTOM = "custom"

class SignalType(str, Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class SignalStrength(str, Enum):
    """Signal strength levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class PredictionStatus(str, Enum):
    """Prediction job status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PredictionMethod(str, Enum):
    """Prediction methods"""
    SINGLE_MODEL = "single_model"
    ENSEMBLE = "ensemble"
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"

# ============================================
# Base Schemas
# ============================================

class BasePredictionModel(BaseModel):
    """Base model for all prediction schemas"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        populate_by_name=True,
        json_schema_extra={
            "examples": []
        }
    )

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp"
    )

# ============================================
# Input/Output Data Schemas
# ============================================

class FeatureInput(BasePredictionModel):
    """Individual feature input"""
    
    name: str = Field(
        description="Feature name",
        examples=["open", "close", "volume", "sma_20", "rsi_14"]
    )
    
    value: Union[float, int] = Field(
        description="Feature value",
        examples=[150.25, 1234567]
    )
    
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Feature timestamp (for time-based features)"
    )

class PredictionInput(BasePredictionModel):
    """Input data for prediction"""
    
    symbol: SymbolStr = Field(
        description="Stock symbol",
        examples=["AAPL", "MSFT", "GOOGL"]
    )
    
    features: Union[Dict[str, Union[float, int]], List[FeatureInput]] = Field(
        description="Feature values for prediction",
        examples=[{
            "open": 150.25,
            "high": 152.75,
            "low": 149.80,
            "close": 151.50,
            "volume": 1234567,
            "sma_20": 148.50,
            "rsi_14": 62.5,
            "macd": 0.75
        }]
    )
    
    data_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the input data"
    )
    
    additional_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context data",
        examples=[{
            "market_conditions": "volatile",
            "sector_performance": 0.02,
            "economic_indicators": {"gdp_growth": 2.5}
        }]
    )
    
    @field_validator('symbol', mode='before')
    @classmethod
    def symbol_to_upper(cls, v: str) -> str:
        """Convert symbol to uppercase"""
        return v.upper().strip() if v else v

class PredictionOutput(BasePredictionModel):
    """Prediction output"""
    
    prediction: Union[float, int, str, List[Union[float, int, str]]] = Field(
        description="Predicted value(s)",
        examples=[155.75, "BUY", [155.75, 158.20, 160.30]]
    )
    
    confidence: Optional[ScoreFloat] = Field(
        default=None,
        description="Confidence score (0.0 to 1.0)",
        examples=[0.85]
    )
    
    probability_distribution: Optional[Dict[str, float]] = Field(
        default=None,
        description="Probability distribution for classification",
        examples=[{
            "BUY": 0.65,
            "HOLD": 0.25,
            "SELL": 0.10
        }]
    )
    
    prediction_interval: Optional[Dict[str, float]] = Field(
        default=None,
        description="Prediction interval bounds",
        examples=[{
            "lower_bound": 152.50,
            "upper_bound": 159.00,
            "confidence_level": 0.95
        }]
    )
    
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None,
        description="Feature importance scores",
        examples=[{
            "close": 0.35,
            "volume": 0.20,
            "sma_20": 0.15,
            "rsi_14": 0.12,
            "macd": 0.18
        }]
    )
    
    prediction_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the prediction was made"
    )
    
    model_version: Optional[str] = Field(
        default=None,
        description="Version of model used",
        examples=["1.0.0", "2.1.3"]
    )

# ============================================
# Prediction Request/Response Schemas
# ============================================

class SinglePredictionRequest(BasePredictionModel):
    """Single prediction request"""
    
    model_id: ModelIdStr = Field(
        description="Model to use for prediction",
        examples=["model_abc123", "ensemble_model_xyz789"]
    )
    
    input_data: PredictionInput = Field(
        description="Input data for prediction"
    )
    
    prediction_type: PredictionType = Field(
        default=PredictionType.REGRESSION,
        description="Type of prediction requested"
    )
    
    prediction_horizon: PredictionHorizon = Field(
        default=PredictionHorizon.NEXT_DAY,
        description="Prediction time horizon"
    )
    
    custom_horizon_days: Optional[PositiveInt] = Field(
        default=None,
        description="Custom horizon in days (if horizon is 'custom')"
    )
    
    include_confidence: bool = Field(
        default=True,
        description="Include confidence scores"
    )
    
    include_feature_importance: bool = Field(
        default=False,
        description="Include feature importance analysis"
    )
    
    @model_validator(mode='after')
    def validate_custom_horizon(self):
        """Validate custom horizon"""
        if self.prediction_horizon == PredictionHorizon.CUSTOM:
            if self.custom_horizon_days is None:
                raise ValueError('custom_horizon_days is required when horizon is "custom"')
        return self
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model_id": "model_abc123",
                    "input_data": {
                        "symbol": "AAPL",
                        "features": {
                            "open": 150.25,
                            "high": 152.75,
                            "low": 149.80,
                            "close": 151.50,
                            "volume": 1234567,
                            "sma_20": 148.50,
                            "rsi_14": 62.5
                        }
                    },
                    "prediction_type": "regression",
                    "prediction_horizon": "next_day",
                    "include_confidence": True,
                    "include_feature_importance": True
                }
            ]
        }
    )

class BatchPredictionRequest(BasePredictionModel):
    """Batch prediction request"""
    
    model_id: ModelIdStr = Field(
        description="Model to use for predictions"
    )
    
    input_data: List[PredictionInput] = Field(
        description="List of input data for batch prediction",
        min_length=1,
        max_length=1000
    )
    
    prediction_type: PredictionType = Field(
        default=PredictionType.REGRESSION,
        description="Type of prediction requested"
    )
    
    prediction_horizon: PredictionHorizon = Field(
        default=PredictionHorizon.NEXT_DAY,
        description="Prediction time horizon"
    )
    
    batch_size: Optional[PositiveInt] = Field(
        default=100,
        description="Processing batch size"
    )
    
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing"
    )
    
    include_confidence: bool = Field(
        default=True,
        description="Include confidence scores"
    )

class EnsemblePredictionRequest(BasePredictionModel):
    """Ensemble prediction request using multiple models"""
    
    model_ids: List[ModelIdStr] = Field(
        description="List of models to use in ensemble",
        min_length=2,
        max_length=10
    )
    
    input_data: PredictionInput = Field(
        description="Input data for prediction"
    )
    
    ensemble_method: PredictionMethod = Field(
        default=PredictionMethod.WEIGHTED_AVERAGE,
        description="Ensemble combination method"
    )
    
    model_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weights for each model (for weighted average)",
        examples=[{
            "model_abc123": 0.4,
            "model_def456": 0.35,
            "model_ghi789": 0.25
        }]
    )
    
    prediction_type: PredictionType = Field(
        description="Type of prediction requested"
    )
    
    @field_validator('model_weights')
    @classmethod
    def validate_weights(cls, v: Optional[Dict[str, float]], info) -> Optional[Dict[str, float]]:
        """Validate that weights sum to 1.0"""
        if v is not None:
            total_weight = sum(v.values())
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError('Model weights must sum to 1.0')
        return v

# ============================================
# Prediction Results Schemas
# ============================================

class PredictionResult(BasePredictionModel):
    """Individual prediction result"""
    
    input_data: PredictionInput = Field(
        description="Input data used for prediction"
    )
    
    output_data: PredictionOutput = Field(
        description="Prediction output"
    )
    
    processing_time_ms: float = Field(
        description="Processing time in milliseconds",
        examples=[45.2]
    )
    
    model_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata about the model used",
        examples=[{
            "model_name": "AAPL_Random_Forest",
            "model_type": "random_forest_regressor",
            "training_date": "2023-01-15",
            "performance_score": 0.87
        }]
    )

class PredictionResponse(BasePredictionModel, TimestampMixin):
    """Standard prediction response"""
    
    prediction_id: str = Field(
        description="Unique prediction identifier",
        examples=["pred_abc123"]
    )
    
    model_id: ModelIdStr = Field(
        description="Model ID used for prediction"
    )
    
    prediction_type: PredictionType = Field(
        description="Type of prediction performed"
    )
    
    results: List[PredictionResult] = Field(
        description="Prediction results"
    )
    
    total_predictions: int = Field(
        description="Total number of predictions made"
    )
    
    total_processing_time_ms: float = Field(
        description="Total processing time in milliseconds"
    )
    
    success_rate: ScoreFloat = Field(
        description="Success rate of predictions (0.0 to 1.0)"
    )
    
    errors: Optional[List[str]] = Field(
        default=None,
        description="Any errors encountered during prediction"
    )

class BatchPredictionResponse(BasePredictionModel, TimestampMixin):
    """Batch prediction response"""
    
    job_id: str = Field(
        description="Batch job identifier",
        examples=["batch_pred_123456"]
    )
    
    status: PredictionStatus = Field(
        description="Batch prediction status"
    )
    
    progress: ScoreFloat = Field(
        description="Job progress (0.0 to 1.0)"
    )
    
    total_items: int = Field(
        description="Total number of items to process"
    )
    
    completed_items: int = Field(
        description="Number of completed items"
    )
    
    failed_items: int = Field(
        description="Number of failed items"
    )
    
    results: Optional[List[PredictionResult]] = Field(
        default=None,
        description="Prediction results (available when completed)"
    )
    
    estimated_completion: Optional[datetime] = Field(
        default=None,
        description="Estimated completion time"
    )
    
    error_summary: Optional[Dict[str, int]] = Field(
        default=None,
        description="Summary of errors encountered",
        examples=[{
            "invalid_features": 5,
            "model_errors": 2,
            "timeout_errors": 1
        }]
    )

# ============================================
# Trading Signals Schemas
# ============================================

class TradingSignal(BasePredictionModel):
    """Trading signal based on prediction"""
    
    symbol: SymbolStr = Field(
        description="Stock symbol"
    )
    
    signal_type: SignalType = Field(
        description="Type of trading signal"
    )
    
    signal_strength: SignalStrength = Field(
        description="Strength of the signal"
    )
    
    confidence: ScoreFloat = Field(
        description="Signal confidence (0.0 to 1.0)"
    )
    
    target_price: Optional[PositiveFloat] = Field(
        default=None,
        description="Target price for the signal",
        examples=[165.50]
    )
    
    stop_loss: Optional[PositiveFloat] = Field(
        default=None,
        description="Suggested stop loss price",
        examples=[145.00]
    )
    
    expected_return: Optional[PercentageFloat] = Field(
        default=None,
        description="Expected return percentage",
        examples=[8.5]
    )
    
    time_horizon: PredictionHorizon = Field(
        description="Signal time horizon"
    )
    
    reasoning: Optional[str] = Field(
        default=None,
        description="Human-readable reasoning for the signal",
        examples=["Strong technical indicators suggest upward momentum"]
    )
    
    supporting_factors: Optional[List[str]] = Field(
        default=None,
        description="Factors supporting the signal",
        examples=[["RSI oversold", "Volume spike", "Breaking resistance level"]]
    )
    
    risk_factors: Optional[List[str]] = Field(
        default=None,
        description="Risk factors to consider",
        examples=[["Market volatility", "Earnings announcement pending"]]
    )
    
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the signal was generated"
    )
    
    expires_at: Optional[datetime] = Field(
        default=None,
        description="When the signal expires"
    )
    
    @field_validator('symbol', mode='before')
    @classmethod
    def symbol_to_upper(cls, v: str) -> str:
        """Convert symbol to uppercase"""
        return v.upper().strip() if v else v

class SignalGenerationRequest(BasePredictionModel):
    """Request for generating trading signals"""
    
    symbols: List[SymbolStr] = Field(
        description="List of symbols to generate signals for",
        min_length=1,
        max_length=100
    )
    
    model_ids: Optional[List[ModelIdStr]] = Field(
        default=None,
        description="Specific models to use (default: auto-select best models)"
    )
    
    signal_types: Optional[List[SignalType]] = Field(
        default=None,
        description="Types of signals to generate (default: all types)"
    )
    
    min_confidence: ScoreFloat = Field(
        default=0.6,
        description="Minimum confidence threshold for signals"
    )
    
    time_horizon: PredictionHorizon = Field(
        default=PredictionHorizon.NEXT_DAY,
        description="Signal time horizon"
    )
    
    include_reasoning: bool = Field(
        default=True,
        description="Include reasoning and supporting factors"
    )
    
    risk_assessment: bool = Field(
        default=True,
        description="Include risk factor analysis"
    )
    
    @field_validator('symbols', mode='before')
    @classmethod
    def symbols_to_upper(cls, v: List[str]) -> List[str]:
        """Convert symbols to uppercase"""
        return [s.upper().strip() for s in v] if v else v

class SignalGenerationResponse(BasePredictionModel, TimestampMixin):
    """Response for signal generation"""
    
    signals: List[TradingSignal] = Field(
        description="Generated trading signals"
    )
    
    total_symbols_analyzed: int = Field(
        description="Total number of symbols analyzed"
    )
    
    signals_generated: int = Field(
        description="Number of signals generated"
    )
    
    average_confidence: ScoreFloat = Field(
        description="Average confidence of generated signals"
    )
    
    signal_distribution: Dict[str, int] = Field(
        description="Distribution of signal types",
        examples=[{
            "BUY": 15,
            "SELL": 8,
            "HOLD": 12,
            "STRONG_BUY": 5
        }]
    )
    
    failed_symbols: Optional[List[str]] = Field(
        default=None,
        description="Symbols that failed analysis"
    )
    
    processing_summary: Dict[str, Any] = Field(
        description="Processing summary",
        examples=[{
            "total_processing_time_ms": 2450.5,
            "avg_processing_time_per_symbol_ms": 61.3,
            "models_used": ["model_123", "model_456"],
            "success_rate": 0.92
        }]
    )

# ============================================
# Prediction Analytics Schemas
# ============================================

class PredictionAccuracy(BasePredictionModel):
    """Prediction accuracy metrics"""
    
    prediction_id: str = Field(
        description="Original prediction ID"
    )
    
    actual_value: Union[float, int, str] = Field(
        description="Actual observed value"
    )
    
    predicted_value: Union[float, int, str] = Field(
        description="Original predicted value"
    )
    
    accuracy_score: Optional[ScoreFloat] = Field(
        default=None,
        description="Accuracy score (for classification)"
    )
    
    absolute_error: Optional[float] = Field(
        default=None,
        description="Absolute error (for regression)"
    )
    
    percentage_error: Optional[PercentageFloat] = Field(
        default=None,
        description="Percentage error"
    )
    
    was_correct: Optional[bool] = Field(
        default=None,
        description="Whether prediction was correct (for classification)"
    )
    
    verification_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the accuracy was verified"
    )

class PredictionPerformanceRequest(BasePredictionModel):
    """Request for prediction performance analysis"""
    
    model_id: Optional[ModelIdStr] = Field(
        default=None,
        description="Model to analyze (default: all models)"
    )
    
    symbol: Optional[SymbolStr] = Field(
        default=None,
        description="Symbol to analyze (default: all symbols)"
    )
    
    start_date: date = Field(
        description="Start date for analysis"
    )
    
    end_date: date = Field(
        description="End date for analysis"
    )
    
    prediction_type: Optional[PredictionType] = Field(
        default=None,
        description="Type of predictions to analyze"
    )
    
    min_confidence: Optional[ScoreFloat] = Field(
        default=None,
        description="Minimum confidence threshold"
    )
    
    @field_validator('symbol', mode='before')
    @classmethod
    def symbol_to_upper(cls, v: Optional[str]) -> Optional[str]:
        """Convert symbol to uppercase"""
        return v.upper().strip() if v else v
    
    @model_validator(mode='after')
    def validate_date_range(self):
        """Validate date range"""
        if self.start_date >= self.end_date:
            raise ValueError('Start date must be before end date')
        
        # Limit analysis to reasonable timeframe
        if (self.end_date - self.start_date).days > 365:
            raise ValueError('Analysis period cannot exceed 1 year')
        
        return self

class PredictionPerformanceResponse(BasePredictionModel):
    """Response for prediction performance analysis"""
    
    total_predictions: int = Field(
        description="Total number of predictions analyzed"
    )
    
    accuracy_metrics: Dict[str, float] = Field(
        description="Accuracy metrics",
        examples=[{
            "overall_accuracy": 0.73,
            "precision": 0.68,
            "recall": 0.75,
            "f1_score": 0.71,
            "mean_absolute_error": 2.45,
            "root_mean_squared_error": 3.21
        }]
    )
    
    performance_by_confidence: Dict[str, Dict[str, float]] = Field(
        description="Performance breakdown by confidence levels",
        examples=[{
            "high_confidence_0.8_1.0": {
                "count": 150,
                "accuracy": 0.85,
                "avg_error": 1.2
            },
            "medium_confidence_0.6_0.8": {
                "count": 200,
                "accuracy": 0.72,
                "avg_error": 2.1
            }
        }]
    )
    
    performance_by_symbol: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Performance breakdown by symbol"
    )
    
    temporal_performance: List[Dict[str, Any]] = Field(
        description="Performance over time",
        examples=[[{
            "date": "2023-01-01",
            "accuracy": 0.75,
            "prediction_count": 25,
            "avg_confidence": 0.68
        }]]
    )
    
    improvement_suggestions: List[str] = Field(
        description="Suggestions for improving prediction performance",
        examples=[["Increase training data", "Add more features", "Tune hyperparameters"]]
    )

# ============================================
# Export All Schemas
# ============================================

__all__ = [
    # Type aliases
    "SymbolStr",
    "ModelIdStr", 
    "PositiveFloat",
    "PositiveInt",
    "ScoreFloat",
    "PercentageFloat",
    
    # Enums
    "PredictionType",
    "PredictionHorizon",
    "SignalType",
    "SignalStrength",
    "PredictionStatus",
    "PredictionMethod",
    
    # Base models
    "BasePredictionModel",
    "TimestampMixin",
    
    # Input/Output schemas
    "FeatureInput",
    "PredictionInput",
    "PredictionOutput",
    
    # Request schemas
    "SinglePredictionRequest",
    "BatchPredictionRequest",
    "EnsemblePredictionRequest",
    
    # Response schemas
    "PredictionResult",
    "PredictionResponse",
    "BatchPredictionResponse",
    
    # Trading signal schemas
    "TradingSignal",
    "SignalGenerationRequest",
    "SignalGenerationResponse",
    
    # Analytics schemas
    "PredictionAccuracy",
    "PredictionPerformanceRequest",
    "PredictionPerformanceResponse",
]
