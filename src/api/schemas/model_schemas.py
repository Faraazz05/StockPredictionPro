# ============================================
# StockPredictionPro - src/api/schemas/model_schemas.py
# Comprehensive Pydantic schemas for machine learning models API endpoints
# ============================================

from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Literal, Annotated
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# ============================================
# Type Aliases and Constants
# ============================================

ModelIdStr = Annotated[str, Field(min_length=1, max_length=50)]
SymbolStr = Annotated[str, Field(min_length=1, max_length=12)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
ScoreFloat = Annotated[float, Field(ge=0.0, le=1.0)]

# ============================================
# Enums and Constants
# ============================================

class ModelType(str, Enum):
    """Types of machine learning models"""
    # Regression models
    LINEAR_REGRESSION = "linear_regression"
    MULTIPLE_REGRESSION = "multiple_regression"
    POLYNOMIAL_REGRESSION = "polynomial_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    ELASTIC_NET = "elastic_net"
    SVR = "svr"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"
    NEURAL_NETWORK_REGRESSOR = "neural_network_regressor"
    
    # Classification models
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM_CLASSIFIER = "svm_classifier"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"
    NAIVE_BAYES = "naive_bayes"
    KNN_CLASSIFIER = "knn_classifier"
    NEURAL_NETWORK_CLASSIFIER = "neural_network_classifier"
    
    # Ensemble models
    VOTING_CLASSIFIER = "voting_classifier"
    VOTING_REGRESSOR = "voting_regressor"
    BAGGING_CLASSIFIER = "bagging_classifier"
    BAGGING_REGRESSOR = "bagging_regressor"
    STACKING_CLASSIFIER = "stacking_classifier"
    STACKING_REGRESSOR = "stacking_regressor"
    BLENDING_ENSEMBLE = "blending_ensemble"

class ModelStatus(str, Enum):
    """Model lifecycle status"""
    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

class ModelCategory(str, Enum):
    """Model categories"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"
    TIME_SERIES = "time_series"

class ModelPriority(str, Enum):
    """Model training priority"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidationMethod(str, Enum):
    """Model validation methods"""
    TRAIN_TEST_SPLIT = "train_test_split"
    K_FOLD_CV = "k_fold_cv"
    TIME_SERIES_CV = "time_series_cv"
    WALK_FORWARD = "walk_forward"
    PURGED_CV = "purged_cv"

class OptimizationMethod(str, Enum):
    """Hyperparameter optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    OPTUNA = "optuna"
    MANUAL = "manual"

# ============================================
# Base Model Schemas
# ============================================

class BaseModelSchema(BaseModel):
    """Base model for all model schemas"""
    
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
        description="Model creation timestamp"
    )
    
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp"
    )
    
    trained_at: Optional[datetime] = Field(
        default=None,
        description="Model training completion timestamp"
    )

# ============================================
# Model Configuration Schemas
# ============================================

class ModelFeatures(BaseModelSchema):
    """Model features configuration"""
    
    feature_columns: List[str] = Field(
        description="List of feature column names",
        examples=[["open", "high", "low", "close", "volume", "sma_20", "rsi_14"]]
    )
    
    target_column: str = Field(
        description="Target column name",
        examples=["next_day_return", "price_direction"]
    )
    
    feature_engineering: Dict[str, Any] = Field(
        default_factory=dict,
        description="Feature engineering configuration",
        examples=[{
            "technical_indicators": ["sma", "ema", "rsi", "macd"],
            "lag_features": [1, 2, 3, 5],
            "polynomial_features": {"degree": 2, "interaction_only": False},
            "scaling": "standard"
        }]
    )
    
    feature_selection: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Feature selection configuration",
        examples=[{
            "method": "rfe",
            "n_features": 20,
            "step": 1
        }]
    )

class ModelHyperparameters(BaseModelSchema):
    """Model hyperparameters"""
    
    parameters: Dict[str, Any] = Field(
        description="Model-specific hyperparameters",
        examples=[{
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        }]
    )
    
    optimization_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hyperparameter optimization configuration",
        examples=[{
            "method": "optuna",
            "n_trials": 100,
            "timeout": 3600,
            "parameter_space": {
                "n_estimators": [50, 100, 200, 500],
                "max_depth": [5, 10, 15, 20],
                "min_samples_split": [2, 5, 10]
            }
        }]
    )

class ModelValidation(BaseModelSchema):
    """Model validation configuration"""
    
    method: ValidationMethod = Field(
        default=ValidationMethod.TIME_SERIES_CV,
        description="Validation method"
    )
    
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Test set size ratio"
    )
    
    cv_folds: Optional[int] = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of cross-validation folds"
    )
    
    validation_metrics: List[str] = Field(
        default=["mse", "mae", "r2"],
        description="Metrics to calculate during validation",
        examples=[["accuracy", "precision", "recall", "f1"], ["mse", "mae", "mape", "r2"]]
    )
    
    early_stopping: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Early stopping configuration",
        examples=[{
            "monitor": "val_loss",
            "patience": 10,
            "min_delta": 0.001
        }]
    )

# ============================================
# Model Performance Schemas
# ============================================

class ModelMetrics(BaseModelSchema):
    """Model performance metrics"""
    
    # Regression metrics
    mse: Optional[float] = Field(default=None, description="Mean Squared Error")
    mae: Optional[float] = Field(default=None, description="Mean Absolute Error")
    rmse: Optional[float] = Field(default=None, description="Root Mean Squared Error")
    mape: Optional[float] = Field(default=None, description="Mean Absolute Percentage Error")
    r2_score: Optional[float] = Field(default=None, description="R-squared score")
    adjusted_r2: Optional[float] = Field(default=None, description="Adjusted R-squared")
    
    # Classification metrics
    accuracy: Optional[ScoreFloat] = Field(default=None, description="Classification accuracy")
    precision: Optional[ScoreFloat] = Field(default=None, description="Precision score")
    recall: Optional[ScoreFloat] = Field(default=None, description="Recall score")
    f1_score: Optional[ScoreFloat] = Field(default=None, description="F1 score")
    roc_auc: Optional[ScoreFloat] = Field(default=None, description="ROC AUC score")
    log_loss: Optional[float] = Field(default=None, description="Logarithmic loss")
    
    # Cross-validation metrics
    cv_scores: Optional[List[float]] = Field(
        default=None,
        description="Cross-validation scores"
    )
    
    cv_mean: Optional[float] = Field(
        default=None,
        description="Mean cross-validation score"
    )
    
    cv_std: Optional[float] = Field(
        default=None,
        description="Standard deviation of cross-validation scores"
    )
    
    # Training metrics
    training_time: Optional[float] = Field(
        default=None,
        description="Training time in seconds"
    )
    
    prediction_time: Optional[float] = Field(
        default=None,
        description="Average prediction time per sample in milliseconds"
    )
    
    model_size_mb: Optional[float] = Field(
        default=None,
        description="Model size in megabytes"
    )
    
    # Custom metrics
    custom_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom domain-specific metrics"
    )

class ModelPerformanceSummary(BaseModelSchema):
    """Model performance summary"""
    
    overall_score: ScoreFloat = Field(
        description="Overall model performance score (0-1)"
    )
    
    performance_grade: Literal["A", "B", "C", "D", "F"] = Field(
        description="Performance grade based on metrics"
    )
    
    strengths: List[str] = Field(
        description="Model strengths",
        examples=[["High accuracy", "Fast prediction", "Good generalization"]]
    )
    
    weaknesses: List[str] = Field(
        description="Model weaknesses", 
        examples=[["Prone to overfitting", "High memory usage", "Poor on edge cases"]]
    )
    
    recommendations: List[str] = Field(
        description="Improvement recommendations",
        examples=[["Increase training data", "Feature engineering", "Hyperparameter tuning"]]
    )

# ============================================
# Model Training Schemas
# ============================================

class ModelTrainingRequest(BaseModelSchema):
    """Model training request"""
    
    symbol: SymbolStr = Field(
        description="Stock symbol for training",
        examples=["AAPL", "MSFT", "GOOGL"]
    )
    
    model_type: ModelType = Field(
        description="Type of model to train"
    )
    
    model_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Custom model name",
        examples=["AAPL_RF_V1", "Price_Predictor_MSFT"]
    )
    
    features: ModelFeatures = Field(
        description="Feature configuration"
    )
    
    hyperparameters: ModelHyperparameters = Field(
        description="Model hyperparameters"
    )
    
    validation: ModelValidation = Field(
        description="Validation configuration"
    )
    
    training_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Training data configuration",
        examples=[{
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "data_source": "yahoo_finance",
            "min_samples": 1000
        }]
    )
    
    priority: ModelPriority = Field(
        default=ModelPriority.MEDIUM,
        description="Training priority"
    )
    
    tags: Optional[List[str]] = Field(
        default=None,
        description="Model tags for organization",
        examples=[["production", "experimental", "v2"]]
    )
    
    @field_validator('symbol', mode='before')
    @classmethod
    def symbol_to_upper(cls, v: str) -> str:
        """Convert symbol to uppercase"""
        return v.upper().strip() if v else v
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "symbol": "AAPL",
                    "model_type": "random_forest_regressor",
                    "model_name": "AAPL_RF_Price_Predictor",
                    "features": {
                        "feature_columns": ["open", "high", "low", "close", "volume", "sma_20", "rsi_14"],
                        "target_column": "next_day_return",
                        "feature_engineering": {
                            "technical_indicators": ["sma", "ema", "rsi"],
                            "lag_features": [1, 2, 3],
                            "scaling": "standard"
                        }
                    },
                    "hyperparameters": {
                        "parameters": {
                            "n_estimators": 100,
                            "max_depth": 10,
                            "random_state": 42
                        }
                    },
                    "validation": {
                        "method": "time_series_cv",
                        "cv_folds": 5,
                        "validation_metrics": ["mse", "mae", "r2"]
                    }
                }
            ]
        }
    )

class ModelTrainingStatus(BaseModelSchema):
    """Model training status"""
    
    job_id: str = Field(
        description="Training job ID",
        examples=["train_job_123456"]
    )
    
    status: ModelStatus = Field(
        description="Current training status"
    )
    
    progress: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        description="Training progress (0.0 to 1.0)"
    )
    
    current_epoch: Optional[int] = Field(
        default=None,
        description="Current training epoch"
    )
    
    total_epochs: Optional[int] = Field(
        default=None,
        description="Total training epochs"
    )
    
    elapsed_time: float = Field(
        description="Elapsed training time in seconds"
    )
    
    estimated_remaining: Optional[float] = Field(
        default=None,
        description="Estimated remaining time in seconds"
    )
    
    current_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Current training metrics"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if training failed"
    )

class ModelTrainingResponse(BaseModelSchema, TimestampMixin):
    """Model training response"""
    
    job_id: str = Field(
        description="Training job ID"
    )
    
    model_id: Optional[ModelIdStr] = Field(
        default=None,
        description="Model ID (available after completion)"
    )
    
    status: ModelStatus = Field(
        description="Training status"
    )
    
    message: str = Field(
        description="Status message",
        examples=["Training started successfully", "Model trained successfully"]
    )
    
    estimated_completion: Optional[datetime] = Field(
        default=None,
        description="Estimated completion time"
    )

# ============================================
# Model Management Schemas
# ============================================

class ModelInfo(BaseModelSchema, TimestampMixin):
    """Comprehensive model information"""
    
    model_id: ModelIdStr = Field(
        description="Unique model identifier"
    )
    
    name: str = Field(
        description="Model name",
        examples=["AAPL_Random_Forest_V1"]
    )
    
    symbol: SymbolStr = Field(
        description="Associated stock symbol"
    )
    
    model_type: ModelType = Field(
        description="Model type"
    )
    
    category: ModelCategory = Field(
        description="Model category"
    )
    
    status: ModelStatus = Field(
        description="Current model status"
    )
    
    version: str = Field(
        description="Model version",
        examples=["1.0.0", "2.1.3"]
    )
    
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Model description"
    )
    
    features: ModelFeatures = Field(
        description="Feature configuration used"
    )
    
    hyperparameters: ModelHyperparameters = Field(
        description="Model hyperparameters used"
    )
    
    metrics: ModelMetrics = Field(
        description="Model performance metrics"
    )
    
    performance_summary: Optional[ModelPerformanceSummary] = Field(
        default=None,
        description="Performance summary"
    )
    
    tags: Optional[List[str]] = Field(
        default=None,
        description="Model tags"
    )
    
    is_deployed: bool = Field(
        default=False,
        description="Whether model is deployed"
    )
    
    deployment_url: Optional[str] = Field(
        default=None,
        description="Model deployment endpoint"
    )
    
    creator: Optional[str] = Field(
        default=None,
        description="Model creator user ID"
    )

class ModelCreateRequest(BaseModelSchema):
    """Request to create a new model (without training)"""
    
    name: str = Field(
        max_length=100,
        description="Model name"
    )
    
    symbol: SymbolStr = Field(
        description="Associated stock symbol"
    )
    
    model_type: ModelType = Field(
        description="Model type"
    )
    
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Model description"
    )
    
    tags: Optional[List[str]] = Field(
        default=None,
        description="Model tags"
    )
    
    @field_validator('symbol', mode='before')
    @classmethod
    def symbol_to_upper(cls, v: str) -> str:
        """Convert symbol to uppercase"""
        return v.upper().strip() if v else v

class ModelUpdateRequest(BaseModelSchema):
    """Request to update model information"""
    
    name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Updated model name"
    )
    
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Updated description"
    )
    
    status: Optional[ModelStatus] = Field(
        default=None,
        description="Updated status"
    )
    
    tags: Optional[List[str]] = Field(
        default=None,
        description="Updated tags"
    )
    
    hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated hyperparameters"
    )

class ModelResponse(BaseModelSchema):
    """Standard model response"""
    
    model_id: ModelIdStr = Field(
        description="Model identifier"
    )
    
    name: str = Field(
        description="Model name"
    )
    
    symbol: SymbolStr = Field(
        description="Associated symbol"
    )
    
    model_type: ModelType = Field(
        description="Model type"
    )
    
    status: ModelStatus = Field(
        description="Model status"
    )
    
    created_at: datetime = Field(
        description="Creation timestamp"
    )
    
    metrics: Optional[ModelMetrics] = Field(
        default=None,
        description="Performance metrics"
    )

class ModelListRequest(BaseModelSchema):
    """Request for listing models"""
    
    symbol: Optional[SymbolStr] = Field(
        default=None,
        description="Filter by symbol"
    )
    
    model_type: Optional[ModelType] = Field(
        default=None,
        description="Filter by model type"
    )
    
    status: Optional[ModelStatus] = Field(
        default=None,
        description="Filter by status"
    )
    
    tags: Optional[List[str]] = Field(
        default=None,
        description="Filter by tags (any match)"
    )
    
    created_after: Optional[datetime] = Field(
        default=None,
        description="Filter by creation date"
    )
    
    limit: Annotated[int, Field(ge=1, le=1000)] = Field(
        default=50,
        description="Maximum number of results"
    )
    
    offset: Annotated[int, Field(ge=0)] = Field(
        default=0,
        description="Results offset for pagination"
    )
    
    sort_by: Optional[Literal["created_at", "updated_at", "name", "performance"]] = Field(
        default="created_at",
        description="Sort field"
    )
    
    sort_order: Literal["asc", "desc"] = Field(
        default="desc",
        description="Sort order"
    )
    
    @field_validator('symbol', mode='before')
    @classmethod
    def symbol_to_upper(cls, v: Optional[str]) -> Optional[str]:
        """Convert symbol to uppercase"""
        return v.upper().strip() if v else v

class ModelListResponse(BaseModelSchema):
    """Response for model listing"""
    
    total: int = Field(
        description="Total number of models matching criteria"
    )
    
    limit: int = Field(
        description="Limit used in query"
    )
    
    offset: int = Field(
        description="Offset used in query"
    )
    
    models: List[ModelResponse] = Field(
        description="List of models"
    )
    
    summary: Dict[str, Any] = Field(
        description="Summary statistics",
        examples=[{
            "total_models": 150,
            "by_status": {
                "trained": 120,
                "training": 15,
                "failed": 10,
                "archived": 5
            },
            "by_type": {
                "random_forest_regressor": 45,
                "neural_network_regressor": 30,
                "linear_regression": 25
            }
        }]
    )

# ============================================
# Model Deployment Schemas
# ============================================

class ModelDeploymentRequest(BaseModelSchema):
    """Model deployment request"""
    
    model_id: ModelIdStr = Field(
        description="Model to deploy"
    )
    
    environment: Literal["staging", "production"] = Field(
        description="Deployment environment"
    )
    
    auto_scale: bool = Field(
        default=True,
        description="Enable auto-scaling"
    )
    
    min_instances: PositiveInt = Field(
        default=1,
        description="Minimum number of instances"
    )
    
    max_instances: PositiveInt = Field(
        default=10,
        description="Maximum number of instances"
    )
    
    resource_requirements: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Resource requirements",
        examples=[{
            "cpu": "1000m",
            "memory": "2Gi",
            "gpu": "false"
        }]
    )

class ModelDeploymentResponse(BaseModelSchema):
    """Model deployment response"""
    
    deployment_id: str = Field(
        description="Deployment identifier"
    )
    
    model_id: ModelIdStr = Field(
        description="Deployed model ID"
    )
    
    status: Literal["deploying", "deployed", "failed", "stopped"] = Field(
        description="Deployment status"
    )
    
    endpoint_url: Optional[str] = Field(
        default=None,
        description="Model endpoint URL"
    )
    
    deployed_at: Optional[datetime] = Field(
        default=None,
        description="Deployment timestamp"
    )

# ============================================
# Model Comparison Schemas
# ============================================

class ModelComparisonRequest(BaseModelSchema):
    """Model comparison request"""
    
    model_ids: List[ModelIdStr] = Field(
        min_length=2,
        max_length=10,
        description="Models to compare"
    )
    
    metrics: List[str] = Field(
        description="Metrics to compare",
        examples=[["accuracy", "f1_score", "training_time"]]
    )
    
    include_details: bool = Field(
        default=False,
        description="Include detailed comparison"
    )

class ModelComparisonResponse(BaseModelSchema):
    """Model comparison response"""
    
    models: List[ModelResponse] = Field(
        description="Compared models"
    )
    
    comparison_matrix: Dict[str, Dict[str, float]] = Field(
        description="Metric comparison matrix",
        examples=[{
            "model_1": {"accuracy": 0.85, "f1_score": 0.82},
            "model_2": {"accuracy": 0.87, "f1_score": 0.84}
        }]
    )
    
    winner: Optional[ModelIdStr] = Field(
        default=None,
        description="Best performing model ID"
    )
    
    ranking: List[Dict[str, Any]] = Field(
        description="Model ranking by performance",
        examples=[[
            {"model_id": "model_1", "rank": 1, "score": 0.85},
            {"model_id": "model_2", "rank": 2, "score": 0.82}
        ]]
    )

# ============================================
# Export All Model Schemas
# ============================================

__all__ = [
    # Type aliases
    "ModelIdStr",
    "SymbolStr",
    "PositiveInt",
    "PositiveFloat",
    "ScoreFloat",
    
    # Enums
    "ModelType",
    "ModelStatus",
    "ModelCategory",
    "ModelPriority",
    "ValidationMethod",
    "OptimizationMethod",
    
    # Base models
    "BaseModelSchema",
    "TimestampMixin",
    
    # Configuration schemas
    "ModelFeatures",
    "ModelHyperparameters",
    "ModelValidation",
    
    # Performance schemas
    "ModelMetrics",
    "ModelPerformanceSummary",
    
    # Training schemas
    "ModelTrainingRequest",
    "ModelTrainingStatus",
    "ModelTrainingResponse",
    
    # Management schemas
    "ModelInfo",
    "ModelCreateRequest",
    "ModelUpdateRequest",
    "ModelResponse",
    "ModelListRequest",
    "ModelListResponse",
    
    # Deployment schemas
    "ModelDeploymentRequest",
    "ModelDeploymentResponse",
    
    # Comparison schemas
    "ModelComparisonRequest",
    "ModelComparisonResponse",
]
