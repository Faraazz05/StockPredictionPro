# ============================================
# StockPredictionPro - src/api/routes/models.py
# Comprehensive ML model management routes for FastAPI
# ============================================

import asyncio
import uuid
import pickle
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Union
import logging
import tempfile

# ML libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# FastAPI imports
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, File, UploadFile, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import (
    get_async_session,
    get_current_active_user,
    get_data_manager,
    standard_rate_limit,
    validate_symbol
)
from ..schemas.model_schemas import (
    ModelTrainingRequest,
    ModelTrainingResponse,
    ModelTrainingStatus,
    ModelCreateRequest,
    ModelUpdateRequest,
    ModelResponse,
    ModelListRequest,
    ModelListResponse,
    ModelInfo,
    ModelDeploymentRequest,
    ModelDeploymentResponse,
    ModelComparisonRequest,
    ModelComparisonResponse,
    ModelType,
    ModelStatus,
    ModelCategory,
    ModelMetrics
)
from ..schemas.error_schemas import ErrorResponse
from ...data.manager import DataManager
from ...utils.logger import get_logger

logger = get_logger('api.routes.models')

# ============================================
# Router Configuration
# ============================================

router = APIRouter(
    prefix="/api/v1/models",
    tags=["Machine Learning Models"],
    dependencies=[Depends(standard_rate_limit)],
    responses={
        400: ErrorResponse.model_400(),
        401: ErrorResponse.model_401(),
        403: ErrorResponse.model_403(),
        404: ErrorResponse.model_404(),
        422: ErrorResponse.model_422(),
        500: ErrorResponse.model_500(),
    }
)

# ============================================
# ML Pipeline Service
# ============================================

class MLModelService:
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.training_jobs = {}
        
        # Model type mapping
        self.model_classes = {
            ModelType.LINEAR_REGRESSION: LinearRegression,
            ModelType.RIDGE_REGRESSION: Ridge,
            ModelType.LASSO_REGRESSION: Lasso,
            ModelType.RANDOM_FOREST_REGRESSOR: RandomForestRegressor,
            ModelType.GRADIENT_BOOSTING_REGRESSOR: GradientBoostingRegressor,
            ModelType.SVR: SVR,
            ModelType.NEURAL_NETWORK_REGRESSOR: MLPRegressor,
            ModelType.LOGISTIC_REGRESSION: LogisticRegression,
            ModelType.RANDOM_FOREST_CLASSIFIER: RandomForestClassifier,
            ModelType.SVM_CLASSIFIER: SVC,
            ModelType.NEURAL_NETWORK_CLASSIFIER: MLPClassifier
        }
        
        # Default parameters
        self.default_params = {
            ModelType.LINEAR_REGRESSION: {},
            ModelType.RIDGE_REGRESSION: {"alpha": 1.0},
            ModelType.LASSO_REGRESSION: {"alpha": 1.0},
            ModelType.RANDOM_FOREST_REGRESSOR: {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            ModelType.GRADIENT_BOOSTING_REGRESSOR: {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            },
            ModelType.SVR: {"kernel": "rbf", "C": 1.0},
            ModelType.NEURAL_NETWORK_REGRESSOR: {
                "hidden_layer_sizes": (100,),
                "max_iter": 1000,
                "random_state": 42
            },
            ModelType.LOGISTIC_REGRESSION: {"max_iter": 1000, "random_state": 42},
            ModelType.RANDOM_FOREST_CLASSIFIER: {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            ModelType.SVM_CLASSIFIER: {"kernel": "rbf", "C": 1.0, "random_state": 42},
            ModelType.NEURAL_NETWORK_CLASSIFIER: {
                "hidden_layer_sizes": (100,),
                "max_iter": 1000,
                "random_state": 42
            }
        }

    def prepare_data(self, market_data, features_config):
        """Prepare training data from market data"""
        try:
            # Convert market data to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': point.timestamp,
                    'open': point.open,
                    'high': point.high,
                    'low': point.low,
                    'close': point.close,
                    'volume': point.volume
                }
                for point in market_data.data
            ])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Create target variable
            df['next_day_return'] = df['close'].pct_change().shift(-1) * 100
            df['price_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Select features and target
            feature_columns = features_config.feature_columns
            target_column = features_config.target_column
            
            # Ensure required columns exist
            missing_cols = set(feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            # Remove NaN values
            df = df.dropna(subset=feature_columns + [target_column])
            
            if len(df) < 50:
                raise ValueError("Insufficient data points after preprocessing")
            
            X = df[feature_columns]
            y = df[target_column]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise

    def add_technical_indicators(self, df):
        """Add technical indicators to DataFrame"""
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Price-based features
        df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['open_close_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        return df

    def create_model(self, model_type, hyperparameters):
        """Create model instance"""
        model_class = self.model_classes.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Merge parameters
        params = self.default_params.get(model_type, {}).copy()
        params.update(hyperparameters.get('parameters', {}))
        
        return model_class(**params)

    def train_model(self, X, y, model_type, hyperparameters, validation_config):
        """Train model with validation"""
        try:
            # Split data
            test_size = validation_config.test_size
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Create and apply scaler for certain models
            scaler = None
            if model_type in [ModelType.SVR, ModelType.SVM_CLASSIFIER, 
                             ModelType.NEURAL_NETWORK_REGRESSOR, ModelType.NEURAL_NETWORK_CLASSIFIER]:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Create and train model
            model = self.create_model(model_type, hyperparameters)
            
            start_time = datetime.utcnow()
            model.fit(X_train_scaled, y_train)
            training_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = self.calculate_metrics(
                y_train, y_train_pred, y_test, y_test_pred, model_type
            )
            
            # Cross-validation if requested
            if validation_config.cv_folds and validation_config.cv_folds > 1:
                cv_scores = self.perform_cross_validation(
                    X_train_scaled, y_train, model, validation_config, model_type
                )
                metrics.update({
                    'cv_scores': cv_scores,
                    'cv_mean': float(np.mean(cv_scores)),
                    'cv_std': float(np.std(cv_scores))
                })
            
            return {
                'model': model,
                'scaler': scaler,
                'metrics': metrics,
                'training_time': training_time,
                'feature_columns': X.columns.tolist()
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

    def perform_cross_validation(self, X, y, model, validation_config, model_type):
        """Perform cross-validation"""
        try:
            cv_folds = validation_config.cv_folds
            
            if validation_config.method.value == "time_series_cv":
                cv = TimeSeriesSplit(n_splits=cv_folds)
            else:
                cv = cv_folds
            
            # Choose scoring metric
            scoring = 'neg_mean_squared_error' if self.is_regression_model(model_type) else 'accuracy'
            
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            # Convert negative MSE to positive
            if scoring == 'neg_mean_squared_error':
                scores = -scores
            
            return scores.tolist()
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return []

    def is_regression_model(self, model_type):
        """Check if model is for regression"""
        regression_types = [
            ModelType.LINEAR_REGRESSION,
            ModelType.RIDGE_REGRESSION,
            ModelType.LASSO_REGRESSION,
            ModelType.RANDOM_FOREST_REGRESSOR,
            ModelType.GRADIENT_BOOSTING_REGRESSOR,
            ModelType.SVR,
            ModelType.NEURAL_NETWORK_REGRESSOR
        ]
        return model_type in regression_types

    def calculate_metrics(self, y_train_true, y_train_pred, y_test_true, y_test_pred, model_type):
        """Calculate performance metrics"""
        metrics = {}
        
        if self.is_regression_model(model_type):
            # Regression metrics
            metrics.update({
                'mse': float(mean_squared_error(y_test_true, y_test_pred)),
                'mae': float(mean_absolute_error(y_test_true, y_test_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test_true, y_test_pred))),
                'r2_score': float(r2_score(y_test_true, y_test_pred)),
                'train_r2_score': float(r2_score(y_train_true, y_train_pred))
            })
            
            # MAPE if no zero values
            if not np.any(y_test_true == 0):
                mape = np.mean(np.abs((y_test_true - y_test_pred) / y_test_true)) * 100
                metrics['mape'] = float(mape)
        else:
            # Classification metrics
            metrics.update({
                'accuracy': float(accuracy_score(y_test_true, y_test_pred)),
                'precision': float(precision_score(y_test_true, y_test_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test_true, y_test_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)),
                'train_accuracy': float(accuracy_score(y_train_true, y_train_pred))
            })
        
        return metrics

    def save_model(self, model_id, model_data, metadata):
        """Save model to disk"""
        try:
            model_dir = self.models_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_data['model'], f)
            
            # Save scaler if exists
            if model_data.get('scaler'):
                scaler_path = model_dir / "scaler.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(model_data['scaler'], f)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            full_metadata = {
                **metadata,
                'metrics': model_data['metrics'],
                'training_time': model_data['training_time'],
                'feature_columns': model_data['feature_columns'],
                'saved_at': datetime.utcnow().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)
            
            logger.info(f"Model {model_id} saved successfully")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            raise

    def load_model(self, model_id):
        """Load model from disk"""
        try:
            model_dir = self.models_dir / model_id
            
            if not model_dir.exists():
                raise FileNotFoundError(f"Model {model_id} not found")
            
            # Load model
            model_path = model_dir / "model.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load scaler if exists
            scaler = None
            scaler_path = model_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                'model': model,
                'scaler': scaler,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def delete_model(self, model_id):
        """Delete model from disk"""
        try:
            model_dir = self.models_dir / model_id
            
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info(f"Model {model_id} deleted successfully")
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            raise

# Global service instance
ml_service = MLModelService()

# ============================================
# Background Tasks
# ============================================

async def model_training_task(job_id: str, training_request: ModelTrainingRequest, 
                             user_id: str, data_manager: DataManager):
    """Background task for model training"""
    try:
        logger.info(f"Starting model training job {job_id}")
        
        # Update status
        ml_service.training_jobs[job_id]['status'] = ModelStatus.TRAINING
        ml_service.training_jobs[job_id]['progress'] = 0.1
        
        # Get market data
        market_data = await data_manager.get_historical_data(
            symbol=training_request.symbol,
            start_date=training_request.training_data.get('start_date') if training_request.training_data else None,
            end_date=training_request.training_data.get('end_date') if training_request.training_data else None,
            interval='1d'
        )
        
        if not market_data or not market_data.data:
            raise ValueError(f"No market data available for {training_request.symbol}")
        
        # Update progress
        ml_service.training_jobs[job_id]['progress'] = 0.3
        
        # Prepare data
        X, y = ml_service.prepare_data(market_data, training_request.features)
        
        # Update progress
        ml_service.training_jobs[job_id]['progress'] = 0.5
        
        # Train model
        training_results = ml_service.train_model(
            X, y, training_request.model_type, 
            training_request.hyperparameters, training_request.validation
        )
        
        # Update progress
        ml_service.training_jobs[job_id]['progress'] = 0.8
        
        # Generate model ID and save
        model_id = f"model_{uuid.uuid4().hex[:12]}"
        
        model_path = ml_service.save_model(
            model_id=model_id,
            model_data=training_results,
            metadata={
                "training_request": training_request.model_dump(),
                "user_id": user_id,
                "job_id": job_id
            }
        )
        
        # Update final status
        ml_service.training_jobs[job_id].update({
            'status': ModelStatus.TRAINED,
            'progress': 1.0,
            'model_id': model_id,
            'completed_at': datetime.utcnow()
        })
        
        logger.info(f"Model training job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Model training job {job_id} failed: {e}")
        ml_service.training_jobs[job_id].update({
            'status': ModelStatus.FAILED,
            'error_message': str(e),
            'failed_at': datetime.utcnow()
        })

# ============================================
# Route Handlers
# ============================================

@router.post("/train", response_model=ModelTrainingResponse, status_code=status.HTTP_201_CREATED)
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_active_user),
    data_manager: DataManager = Depends(get_data_manager)
):
    """Start model training with comprehensive ML capabilities"""
    
    validate_symbol(request.symbol)
    
    # Generate job ID
    job_id = f"train_job_{uuid.uuid4().hex[:12]}"
    
    # Initialize job tracking
    ml_service.training_jobs[job_id] = {
        'job_id': job_id,
        'user_id': current_user["user_id"],
        'status': ModelStatus.PENDING,
        'progress': 0.0,
        'created_at': datetime.utcnow()
    }
    
    # Start background task
    background_tasks.add_task(
        model_training_task,
        job_id=job_id,
        training_request=request,
        user_id=current_user["user_id"],
        data_manager=data_manager
    )
    
    # Estimate completion time
    base_minutes = 10  # Base training time
    if request.validation.cv_folds:
        base_minutes *= request.validation.cv_folds
    
    estimated_completion = datetime.utcnow() + timedelta(minutes=base_minutes)
    
    return ModelTrainingResponse(
        job_id=job_id,
        status=ModelStatus.PENDING,
        message="Model training started successfully",
        estimated_completion=estimated_completion
    )

@router.get("/train/{job_id}/status", response_model=ModelTrainingStatus)
async def get_training_status(job_id: str, current_user: dict = Depends(get_current_active_user)):
    """Get training job status"""
    
    if job_id not in ml_service.training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = ml_service.training_jobs[job_id]
    
    if job['user_id'] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    elapsed_time = (datetime.utcnow() - job['created_at']).total_seconds()
    
    return ModelTrainingStatus(
        job_id=job_id,
        status=job['status'],
        progress=job['progress'],
        elapsed_time=elapsed_time,
        error_message=job.get('error_message')
    )

@router.post("/", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(request: ModelCreateRequest, current_user: dict = Depends(get_current_active_user)):
    """Create a new model entry"""
    
    validate_symbol(request.symbol)
    
    model_id = f"model_{uuid.uuid4().hex[:12]}"
    
    # Create model info (in real app, save to database)
    model_info = {
        'model_id': model_id,
        'name': request.name,
        'symbol': request.symbol.upper(),
        'model_type': request.model_type,
        'status': ModelStatus.CREATED,
        'created_at': datetime.utcnow(),
        'user_id': current_user["user_id"]
    }
    
    logger.info(f"Created model {model_id} for user {current_user['user_id']}")
    
    return ModelResponse(
        model_id=model_id,
        name=request.name,
        symbol=request.symbol.upper(),
        model_type=request.model_type,
        status=ModelStatus.CREATED,
        created_at=datetime.utcnow()
    )

@router.get("/", response_model=ModelListResponse)
async def list_models(
    symbol: Optional[str] = Query(None),
    model_type: Optional[ModelType] = Query(None),
    status: Optional[ModelStatus] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_active_user)
):
    """List user models with filtering"""
    
    # In real app, query database with filters
    # For now, return empty list with summary
    
    summary = {
        "total_models": 0,
        "by_status": {
            "created": 0,
            "training": len([j for j in ml_service.training_jobs.values() 
                           if j.get('status') == ModelStatus.TRAINING and j.get('user_id') == current_user["user_id"]]),
            "trained": 0,
            "failed": len([j for j in ml_service.training_jobs.values() 
                         if j.get('status') == ModelStatus.FAILED and j.get('user_id') == current_user["user_id"]])
        },
        "by_type": {}
    }
    
    return ModelListResponse(
        total=0,
        limit=limit,
        offset=offset,
        models=[],
        summary=summary
    )

@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str, current_user: dict = Depends(get_current_active_user)):
    """Get detailed model information"""
    
    if not model_id.startswith('model_'):
        raise HTTPException(status_code=400, detail="Invalid model ID format")
    
    try:
        # Load model data
        model_data = ml_service.load_model(model_id)
        metadata = model_data['metadata']
        
        # Verify ownership
        if metadata.get('user_id') != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create model info
        training_request = metadata.get('training_request', {})
        
        model_info = ModelInfo(
            model_id=model_id,
            name=training_request.get('model_name', f"{model_id}_model"),
            symbol=training_request.get('symbol', 'UNKNOWN'),
            model_type=ModelType(training_request.get('model_type', 'linear_regression')),
            category=ModelCategory.REGRESSION,  # Simplified
            status=ModelStatus.TRAINED,
            version="1.0.0",
            description="Trained model",
            metrics=ModelMetrics(**metadata.get('metrics', {})),
            creator=current_user["user_id"],
            is_deployed=False
        )
        
        return model_info
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logger.error(f"Failed to get model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model")

@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(model_id: str, current_user: dict = Depends(get_current_active_user)):
    """Delete a model"""
    
    if not model_id.startswith('model_'):
        raise HTTPException(status_code=400, detail="Invalid model ID format")
    
    try:
        # Load model to verify ownership
        model_data = ml_service.load_model(model_id)
        metadata = model_data['metadata']
        
        if metadata.get('user_id') != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete model
        ml_service.delete_model(model_id)
        
        logger.info(f"Deleted model {model_id} for user {current_user['user_id']}")
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete model")

@router.get("/{model_id}/download")
async def download_model(model_id: str, current_user: dict = Depends(get_current_active_user)):
    """Download model package"""
    
    if not model_id.startswith('model_'):
        raise HTTPException(status_code=400, detail="Invalid model ID format")
    
    try:
        # Load model to verify ownership
        model_data = ml_service.load_model(model_id)
        metadata = model_data['metadata']
        
        if metadata.get('user_id') != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create zip package
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                model_dir = ml_service.models_dir / model_id
                
                for file_path in model_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(model_dir)
                        zipf.write(file_path, arcname)
            
            return FileResponse(
                path=tmp_file.name,
                media_type="application/zip",
                filename=f"model_{model_id}.zip"
            )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to download model")

@router.post("/compare", response_model=ModelComparisonResponse)
async def compare_models(
    request: ModelComparisonRequest, 
    current_user: dict = Depends(get_current_active_user)
):
    """Compare multiple models"""
    
    try:
        models = []
        comparison_matrix = {}
        
        # Load all models and verify ownership
        for model_id in request.model_ids:
            model_data = ml_service.load_model(model_id)
            metadata = model_data['metadata']
            
            if metadata.get('user_id') != current_user["user_id"]:
                raise HTTPException(status_code=403, detail=f"Access denied for model {model_id}")
            
            # Extract metrics
            metrics = metadata.get('metrics', {})
            model_metrics = {}
            
            for metric in request.metrics:
                model_metrics[metric] = metrics.get(metric)
            
            comparison_matrix[model_id] = model_metrics
            
            # Create model response
            training_request = metadata.get('training_request', {})
            model_response = ModelResponse(
                model_id=model_id,
                name=training_request.get('model_name', model_id),
                symbol=training_request.get('symbol', 'UNKNOWN'),
                model_type=ModelType(training_request.get('model_type', 'linear_regression')),
                status=ModelStatus.TRAINED,
                created_at=datetime.fromisoformat(metadata.get('saved_at', datetime.utcnow().isoformat())),
                metrics=ModelMetrics(**metrics) if metrics else None
            )
            models.append(model_response)
        
        # Create simple ranking based on first metric
        ranking = []
        if request.metrics:
            first_metric = request.metrics[0]
            
            # Sort models by first metric
            model_scores = []
            for model_id in request.model_ids:
                score = comparison_matrix[model_id].get(first_metric, 0)
                if score is not None:
                    model_scores.append((model_id, score))
            
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (model_id, score) in enumerate(model_scores):
                ranking.append({
                    "model_id": model_id,
                    "rank": i + 1,
                    "score": score
                })
        
        winner = ranking[0]["model_id"] if ranking else None
        
        return ModelComparisonResponse(
            models=models,
            comparison_matrix=comparison_matrix,
            winner=winner,
            ranking=ranking
        )
        
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare models")

# Export router
__all__ = ["router"]
