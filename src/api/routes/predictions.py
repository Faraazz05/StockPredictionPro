# ============================================
# StockPredictionPro - src/api/routes/predictions.py
# Comprehensive prediction routes for FastAPI with real-time and batch predictions
# ============================================

import asyncio
import uuid
import json
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Union
import logging

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import (
    get_async_session,
    get_current_active_user,
    get_data_manager,
    get_cache_manager,
    standard_rate_limit,
    validate_symbol
)
from ..schemas.prediction_schemas import (
    SinglePredictionRequest,
    BatchPredictionRequest,
    EnsemblePredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    PredictionResult,
    PredictionInput,
    PredictionOutput,
    PredictionType,
    PredictionHorizon,
    PredictionStatus,
    PredictionMethod
)
from ..schemas.error_schemas import ErrorResponse
from ...data.manager import DataManager
from ...data.cache import CacheManager
from ...utils.logger import get_logger

# Import ML service from models route
from .models import ml_service

logger = get_logger('api.routes.predictions')

# ============================================
# Router Configuration
# ============================================

router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["Predictions"],
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
# Prediction Service
# ============================================

class PredictionService:
    def __init__(self):
        self.prediction_jobs = {}
        self.cache_ttl = 300  # 5 minutes cache
    
    def prepare_features(self, input_data: PredictionInput, model_feature_columns: List[str]) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            # Convert features to DataFrame
            if isinstance(input_data.features, dict):
                features_dict = input_data.features
            else:
                # Handle list of FeatureInput objects
                features_dict = {f.name: f.value for f in input_data.features}
            
            # Create DataFrame
            df = pd.DataFrame([features_dict])
            
            # Check for missing features
            missing_features = set(model_feature_columns) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Select only required features in correct order
            df = df[model_feature_columns]
            
            return df
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            raise

    def make_single_prediction(self, model_data: Dict[str, Any], features: pd.DataFrame) -> Dict[str, Any]:
        """Make single prediction using loaded model"""
        try:
            model = model_data['model']
            scaler = model_data.get('scaler')
            metadata = model_data['metadata']
            
            # Apply scaling if model has scaler
            if scaler:
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features
            
            # Make prediction
            prediction = model.predict(features_scaled)
            
            # Get prediction confidence if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(features_scaled)
                    confidence = float(np.max(proba[0]))
                except:
                    pass
            elif hasattr(model, 'decision_function'):
                try:
                    decision = model.decision_function(features_scaled)
                    # Convert decision function to confidence-like score
                    confidence = float(1 / (1 + np.exp(-abs(decision[0]))))
                except:
                    pass
            
            # Feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_names = features.columns.tolist()
                importances = model.feature_importances_
                feature_importance = dict(zip(feature_names, importances.tolist()))
            
            return {
                'prediction': float(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
                'confidence': confidence,
                'feature_importance': feature_importance,
                'model_version': metadata.get('saved_at', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def calculate_prediction_interval(self, prediction: float, confidence: float) -> Dict[str, float]:
        """Calculate prediction interval based on confidence"""
        if confidence is None:
            return None
        
        # Simple prediction interval calculation
        # In practice, this would be more sophisticated
        margin = prediction * (1 - confidence) * 0.1  # 10% margin based on uncertainty
        
        return {
            'lower_bound': prediction - margin,
            'upper_bound': prediction + margin,
            'confidence_level': confidence
        }

    async def process_batch_predictions(self, job_id: str, request: BatchPredictionRequest, 
                                      user_id: str, data_manager: DataManager):
        """Background task for batch predictions"""
        try:
            logger.info(f"Starting batch prediction job {job_id}")
            
            # Update job status
            self.prediction_jobs[job_id] = {
                'job_id': job_id,
                'user_id': user_id,
                'status': PredictionStatus.PROCESSING,
                'progress': 0.0,
                'total_items': len(request.input_data),
                'completed_items': 0,
                'failed_items': 0,
                'results': [],
                'errors': [],
                'created_at': datetime.utcnow()
            }
            
            # Load model
            try:
                model_data = ml_service.load_model(request.model_id)
                metadata = model_data['metadata']
                
                # Verify ownership
                if metadata.get('user_id') != user_id:
                    raise ValueError("Access denied to model")
                
                feature_columns = metadata.get('feature_columns', [])
                
            except Exception as e:
                self.prediction_jobs[job_id]['status'] = PredictionStatus.FAILED
                self.prediction_jobs[job_id]['error'] = f"Failed to load model: {str(e)}"
                return
            
            # Process predictions in batches
            batch_size = request.batch_size or 100
            results = []
            errors = []
            
            for i in range(0, len(request.input_data), batch_size):
                batch = request.input_data[i:i + batch_size]
                
                if request.parallel_processing:
                    # Process batch in parallel
                    tasks = []
                    for input_data in batch:
                        task = self.process_single_prediction_async(
                            input_data, model_data, feature_columns
                        )
                        tasks.append(task)
                    
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # Process sequentially
                    batch_results = []
                    for input_data in batch:
                        try:
                            result = await self.process_single_prediction_async(
                                input_data, model_data, feature_columns
                            )
                            batch_results.append(result)
                        except Exception as e:
                            batch_results.append(e)
                
                # Process batch results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        errors.append({
                            'input_index': i + j,
                            'error': str(result),
                            'symbol': batch[j].symbol
                        })
                        self.prediction_jobs[job_id]['failed_items'] += 1
                    else:
                        results.append(result)
                        self.prediction_jobs[job_id]['completed_items'] += 1
                
                # Update progress
                progress = (i + len(batch)) / len(request.input_data)
                self.prediction_jobs[job_id]['progress'] = progress
            
            # Update final status
            self.prediction_jobs[job_id].update({
                'status': PredictionStatus.COMPLETED,
                'progress': 1.0,
                'results': results,
                'errors': errors,
                'completed_at': datetime.utcnow()
            })
            
            logger.info(f"Batch prediction job {job_id} completed: {len(results)} success, {len(errors)} failed")
            
        except Exception as e:
            logger.error(f"Batch prediction job {job_id} failed: {e}")
            self.prediction_jobs[job_id].update({
                'status': PredictionStatus.FAILED,
                'error': str(e),
                'failed_at': datetime.utcnow()
            })

    async def process_single_prediction_async(self, input_data: PredictionInput, 
                                            model_data: Dict[str, Any], 
                                            feature_columns: List[str]) -> PredictionResult:
        """Process single prediction asynchronously"""
        try:
            start_time = datetime.utcnow()
            
            # Prepare features
            features = self.prepare_features(input_data, feature_columns)
            
            # Make prediction
            prediction_data = self.make_single_prediction(model_data, features)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create prediction output
            output = PredictionOutput(
                prediction=prediction_data['prediction'],
                confidence=prediction_data['confidence'],
                prediction_interval=self.calculate_prediction_interval(
                    prediction_data['prediction'], 
                    prediction_data['confidence']
                ),
                feature_importance=prediction_data['feature_importance'],
                prediction_timestamp=datetime.utcnow(),
                model_version=prediction_data['model_version']
            )
            
            return PredictionResult(
                input_data=input_data,
                output_data=output,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            raise

# Global service instance
prediction_service = PredictionService()

# ============================================
# Route Handlers
# ============================================

@router.post("/single", response_model=PredictionResponse)
async def make_single_prediction(
    request: SinglePredictionRequest,
    current_user: dict = Depends(get_current_active_user),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Make a single prediction using trained model"""
    
    validate_symbol(request.input_data.symbol)
    
    try:
        # Check cache first
        cache_key = f"prediction:{request.model_id}:{hash(str(request.input_data.features))}"
        cached_result = await cache_manager.get(cache_key)
        
        if cached_result:
            logger.info(f"Serving cached prediction for {request.input_data.symbol}")
            return PredictionResponse(**cached_result)
        
        # Load model
        try:
            model_data = ml_service.load_model(request.model_id)
            metadata = model_data['metadata']
            
            # Verify ownership
            if metadata.get('user_id') != current_user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied to model")
            
            feature_columns = metadata.get('feature_columns', [])
            
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Process prediction
        start_time = datetime.utcnow()
        
        # Prepare features
        features = prediction_service.prepare_features(request.input_data, feature_columns)
        
        # Make prediction
        prediction_data = prediction_service.make_single_prediction(model_data, features)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Create prediction output
        output = PredictionOutput(
            prediction=prediction_data['prediction'],
            confidence=prediction_data['confidence'],
            prediction_interval=prediction_service.calculate_prediction_interval(
                prediction_data['prediction'], 
                prediction_data['confidence']
            ),
            feature_importance=prediction_data['feature_importance'] if request.include_feature_importance else None,
            prediction_timestamp=datetime.utcnow(),
            model_version=prediction_data['model_version']
        )
        
        # Create result
        result = PredictionResult(
            input_data=request.input_data,
            output_data=output,
            processing_time_ms=processing_time
        )
        
        # Create response
        response = PredictionResponse(
            prediction_id=f"pred_{uuid.uuid4().hex[:12]}",
            model_id=request.model_id,
            prediction_type=request.prediction_type,
            results=[result],
            total_predictions=1,
            total_processing_time_ms=processing_time,
            success_rate=1.0
        )
        
        # Cache the response
        await cache_manager.set(cache_key, response.model_dump(), ttl=prediction_service.cache_ttl)
        
        logger.info(f"Single prediction completed for {request.input_data.symbol}")
        
        return response
        
    except Exception as e:
        logger.error(f"Single prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch", response_model=BatchPredictionResponse)
async def make_batch_predictions(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_active_user)
):
    """Start batch prediction job for multiple inputs"""
    
    # Validate all symbols
    for input_data in request.input_data:
        validate_symbol(input_data.symbol)
    
    # Generate job ID
    job_id = f"batch_pred_{uuid.uuid4().hex[:12]}"
    
    try:
        # Verify model access
        try:
            model_data = ml_service.load_model(request.model_id)
            metadata = model_data['metadata']
            
            if metadata.get('user_id') != current_user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied to model")
                
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Start background task
        background_tasks.add_task(
            prediction_service.process_batch_predictions,
            job_id=job_id,
            request=request,
            user_id=current_user["user_id"],
            data_manager=None  # Will be injected in real implementation
        )
        
        # Calculate estimated completion time
        estimated_minutes = len(request.input_data) * 0.1  # 0.1 minute per prediction
        estimated_completion = datetime.utcnow() + timedelta(minutes=estimated_minutes)
        
        response = BatchPredictionResponse(
            job_id=job_id,
            status=PredictionStatus.PROCESSING,
            progress=0.0,
            total_items=len(request.input_data),
            completed_items=0,
            failed_items=0,
            estimated_completion=estimated_completion
        )
        
        logger.info(f"Started batch prediction job {job_id} for {len(request.input_data)} items")
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to start batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start batch prediction: {str(e)}")

@router.get("/batch/{job_id}/status", response_model=BatchPredictionResponse)
async def get_batch_prediction_status(
    job_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Get batch prediction job status"""
    
    if job_id not in prediction_service.prediction_jobs:
        raise HTTPException(status_code=404, detail="Prediction job not found")
    
    job = prediction_service.prediction_jobs[job_id]
    
    if job['user_id'] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    response = BatchPredictionResponse(
        job_id=job_id,
        status=job['status'],
        progress=job['progress'],
        total_items=job['total_items'],
        completed_items=job['completed_items'],
        failed_items=job['failed_items'],
        results=job.get('results') if job['status'] == PredictionStatus.COMPLETED else None,
        error_summary={"total_errors": len(job.get('errors', []))} if job.get('errors') else None
    )
    
    return response

@router.post("/ensemble", response_model=PredictionResponse)
async def make_ensemble_prediction(
    request: EnsemblePredictionRequest,
    current_user: dict = Depends(get_current_active_user)
):
    """Make ensemble prediction using multiple models"""
    
    validate_symbol(request.input_data.symbol)
    
    try:
        models_data = []
        
        # Load all models and verify ownership
        for model_id in request.model_ids:
            try:
                model_data = ml_service.load_model(model_id)
                metadata = model_data['metadata']
                
                if metadata.get('user_id') != current_user["user_id"]:
                    raise HTTPException(status_code=403, detail=f"Access denied to model {model_id}")
                
                models_data.append((model_id, model_data))
                
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Make predictions with each model
        predictions = []
        processing_times = []
        
        for model_id, model_data in models_data:
            start_time = datetime.utcnow()
            
            feature_columns = model_data['metadata'].get('feature_columns', [])
            features = prediction_service.prepare_features(request.input_data, feature_columns)
            prediction_data = prediction_service.make_single_prediction(model_data, features)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            processing_times.append(processing_time)
            
            predictions.append({
                'model_id': model_id,
                'prediction': prediction_data['prediction'],
                'confidence': prediction_data['confidence']
            })
        
        # Combine predictions based on ensemble method
        if request.ensemble_method == PredictionMethod.WEIGHTED_AVERAGE:
            weights = request.model_weights or {mid: 1.0/len(request.model_ids) for mid in request.model_ids}
            
            # Weighted average
            final_prediction = sum(
                pred['prediction'] * weights.get(pred['model_id'], 1.0/len(predictions))
                for pred in predictions
            )
            
            # Weighted confidence
            final_confidence = sum(
                pred['confidence'] * weights.get(pred['model_id'], 1.0/len(predictions))
                for pred in predictions if pred['confidence']
            ) if any(p['confidence'] for p in predictions) else None
            
        elif request.ensemble_method == PredictionMethod.VOTING:
            # Simple average for voting
            final_prediction = sum(pred['prediction'] for pred in predictions) / len(predictions)
            final_confidence = sum(
                pred['confidence'] for pred in predictions if pred['confidence']
            ) / len([p for p in predictions if p['confidence']]) if any(p['confidence'] for p in predictions) else None
        
        else:
            # Default: simple average
            final_prediction = sum(pred['prediction'] for pred in predictions) / len(predictions)
            final_confidence = None
        
        # Create ensemble output
        output = PredictionOutput(
            prediction=final_prediction,
            confidence=final_confidence,
            prediction_interval=prediction_service.calculate_prediction_interval(
                final_prediction, final_confidence
            ),
            prediction_timestamp=datetime.utcnow(),
            model_version=f"ensemble_{len(request.model_ids)}_models"
        )
        
        # Create result
        result = PredictionResult(
            input_data=request.input_data,
            output_data=output,
            processing_time_ms=sum(processing_times),
            model_metadata={
                'ensemble_method': request.ensemble_method.value,
                'models_used': request.model_ids,
                'individual_predictions': predictions
            }
        )
        
        # Create response
        response = PredictionResponse(
            prediction_id=f"ensemble_{uuid.uuid4().hex[:12]}",
            model_id=f"ensemble_{'+'.join(request.model_ids[:3])}",
            prediction_type=request.prediction_type,
            results=[result],
            total_predictions=1,
            total_processing_time_ms=sum(processing_times),
            success_rate=1.0
        )
        
        logger.info(f"Ensemble prediction completed using {len(request.model_ids)} models")
        
        return response
        
    except Exception as e:
        logger.error(f"Ensemble prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ensemble prediction failed: {str(e)}")

@router.get("/history")
async def get_prediction_history(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    model_id: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    current_user: dict = Depends(get_current_active_user)
):
    """Get prediction history for user"""
    
    # In real implementation, query database for user's prediction history
    # with filtering by model_id, symbol, date range, etc.
    
    history = {
        "total": 0,
        "limit": limit,
        "offset": offset,
        "predictions": [],
        "summary": {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "models_used": [],
            "symbols_predicted": []
        }
    }
    
    return history

@router.post("/validate")
async def validate_prediction_input(
    input_data: PredictionInput,
    model_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Validate prediction input against model requirements"""
    
    try:
        # Load model to get feature requirements
        model_data = ml_service.load_model(model_id)
        metadata = model_data['metadata']
        
        if metadata.get('user_id') != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied to model")
        
        feature_columns = metadata.get('feature_columns', [])
        
        # Validate features
        if isinstance(input_data.features, dict):
            provided_features = set(input_data.features.keys())
        else:
            provided_features = set(f.name for f in input_data.features)
        
        required_features = set(feature_columns)
        
        missing_features = required_features - provided_features
        extra_features = provided_features - required_features
        
        validation_result = {
            "valid": len(missing_features) == 0,
            "required_features": feature_columns,
            "provided_features": list(provided_features),
            "missing_features": list(missing_features),
            "extra_features": list(extra_features),
            "model_info": {
                "model_id": model_id,
                "model_type": metadata.get('training_request', {}).get('model_type', 'unknown'),
                "symbol": metadata.get('training_request', {}).get('symbol', 'unknown')
            }
        }
        
        return validation_result
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/models/{model_id}/info")
async def get_model_prediction_info(
    model_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Get model information relevant for predictions"""
    
    try:
        model_data = ml_service.load_model(model_id)
        metadata = model_data['metadata']
        
        if metadata.get('user_id') != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied to model")
        
        training_request = metadata.get('training_request', {})
        
        model_info = {
            "model_id": model_id,
            "model_type": training_request.get('model_type', 'unknown'),
            "symbol": training_request.get('symbol', 'unknown'),
            "feature_columns": metadata.get('feature_columns', []),
            "target_column": training_request.get('features', {}).get('target_column', 'unknown'),
            "metrics": metadata.get('metrics', {}),
            "training_date": metadata.get('saved_at'),
            "prediction_capabilities": {
                "supports_confidence": hasattr(model_data['model'], 'predict_proba') or hasattr(model_data['model'], 'decision_function'),
                "supports_feature_importance": hasattr(model_data['model'], 'feature_importances_'),
                "requires_scaling": metadata.get('scaler') is not None
            }
        }
        
        return model_info
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# Export router
__all__ = ["router"]
