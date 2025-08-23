# ============================================
# StockPredictionPro - src/api/routes/signals.py
# Comprehensive trading signals routes for FastAPI with technical analysis and ML-based signals
# ============================================

import asyncio
import uuid
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
    SignalGenerationRequest,
    SignalGenerationResponse,
    TradingSignal,
    SignalType,
    SignalStrength,
    PredictionHorizon
)
from ..schemas.error_schemas import ErrorResponse
from ...data.manager import DataManager
from ...data.cache import CacheManager
from ...utils.logger import get_logger

# Import services from other routes
from .models import ml_service
from .predictions import prediction_service

logger = get_logger('api.routes.signals')

# ============================================
# Router Configuration
# ============================================

router = APIRouter(
    prefix="/api/v1/signals",
    tags=["Trading Signals"],
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
# Signal Generation Service
# ============================================

class TradingSignalService:
    def __init__(self):
        self.signal_jobs = {}
        self.cache_ttl = 60  # 1 minute cache for signals
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Sort by timestamp if available
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Moving Averages
            for period in [5, 10, 20, 50, 200]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
            
            # Stochastic Oscillator
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price action indicators
            df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
            df['open_close_pct'] = (df['close'] - df['open']) / df['open'] * 100
            df['price_change_pct'] = df['close'].pct_change() * 100
            
            return df
            
        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {e}")
            raise

    def generate_technical_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """Generate trading signals based on technical analysis"""
        signals = []
        
        if len(df) < 50:  # Need sufficient data
            return signals
        
        try:
            # Get latest data point
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            current_price = latest['close']
            
            # 1. Moving Average Crossover Signals
            if pd.notna(latest['sma_20']) and pd.notna(latest['sma_50']):
                # Golden Cross (bullish)
                if latest['sma_20'] > latest['sma_50'] and prev['sma_20'] <= prev['sma_50']:
                    signals.append({
                        'type': 'technical',
                        'signal': SignalType.BUY,
                        'strength': SignalStrength.MEDIUM,
                        'confidence': 0.7,
                        'reason': 'Golden Cross - 20 SMA crossed above 50 SMA',
                        'indicator': 'MA_CROSSOVER',
                        'target_price': current_price * 1.05,
                        'stop_loss': current_price * 0.95
                    })
                
                # Death Cross (bearish)
                elif latest['sma_20'] < latest['sma_50'] and prev['sma_20'] >= prev['sma_50']:
                    signals.append({
                        'type': 'technical',
                        'signal': SignalType.SELL,
                        'strength': SignalStrength.MEDIUM,
                        'confidence': 0.7,
                        'reason': 'Death Cross - 20 SMA crossed below 50 SMA',
                        'indicator': 'MA_CROSSOVER',
                        'target_price': current_price * 0.95,
                        'stop_loss': current_price * 1.05
                    })
            
            # 2. RSI Signals
            if pd.notna(latest['rsi']):
                # Oversold condition
                if latest['rsi'] < 30 and prev['rsi'] >= 30:
                    signals.append({
                        'type': 'technical',
                        'signal': SignalType.BUY,
                        'strength': SignalStrength.HIGH,
                        'confidence': 0.8,
                        'reason': f'RSI oversold at {latest["rsi"]:.1f}',
                        'indicator': 'RSI',
                        'target_price': current_price * 1.03,
                        'stop_loss': current_price * 0.98
                    })
                
                # Overbought condition
                elif latest['rsi'] > 70 and prev['rsi'] <= 70:
                    signals.append({
                        'type': 'technical',
                        'signal': SignalType.SELL,
                        'strength': SignalStrength.HIGH,
                        'confidence': 0.8,
                        'reason': f'RSI overbought at {latest["rsi"]:.1f}',
                        'indicator': 'RSI',
                        'target_price': current_price * 0.97,
                        'stop_loss': current_price * 1.02
                    })
            
            # 3. MACD Signals
            if pd.notna(latest['macd']) and pd.notna(latest['macd_signal']):
                # MACD bullish crossover
                if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                    signals.append({
                        'type': 'technical',
                        'signal': SignalType.BUY,
                        'strength': SignalStrength.MEDIUM,
                        'confidence': 0.75,
                        'reason': 'MACD bullish crossover',
                        'indicator': 'MACD',
                        'target_price': current_price * 1.04,
                        'stop_loss': current_price * 0.96
                    })
                
                # MACD bearish crossover
                elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                    signals.append({
                        'type': 'technical',
                        'signal': SignalType.SELL,
                        'strength': SignalStrength.MEDIUM,
                        'confidence': 0.75,
                        'reason': 'MACD bearish crossover',
                        'indicator': 'MACD',
                        'target_price': current_price * 0.96,
                        'stop_loss': current_price * 1.04
                    })
            
            # 4. Bollinger Bands Signals
            if pd.notna(latest['bb_position']):
                # Price touching lower band (oversold)
                if latest['bb_position'] < 0.05:
                    signals.append({
                        'type': 'technical',
                        'signal': SignalType.BUY,
                        'strength': SignalStrength.MEDIUM,
                        'confidence': 0.65,
                        'reason': 'Price near Bollinger Band lower limit',
                        'indicator': 'BOLLINGER_BANDS',
                        'target_price': latest['bb_upper'],
                        'stop_loss': current_price * 0.97
                    })
                
                # Price touching upper band (overbought)
                elif latest['bb_position'] > 0.95:
                    signals.append({
                        'type': 'technical',
                        'signal': SignalType.SELL,
                        'strength': SignalStrength.MEDIUM,
                        'confidence': 0.65,
                        'reason': 'Price near Bollinger Band upper limit',
                        'indicator': 'BOLLINGER_BANDS',
                        'target_price': latest['bb_lower'],
                        'stop_loss': current_price * 1.03
                    })
            
            # 5. Volume Breakout Signals
            if pd.notna(latest['volume_ratio']):
                # High volume breakout
                if latest['volume_ratio'] > 2 and latest['price_change_pct'] > 2:
                    signals.append({
                        'type': 'technical',
                        'signal': SignalType.BUY,
                        'strength': SignalStrength.HIGH,
                        'confidence': 0.8,
                        'reason': f'Volume breakout - {latest["volume_ratio"]:.1f}x average volume',
                        'indicator': 'VOLUME_BREAKOUT',
                        'target_price': current_price * 1.08,
                        'stop_loss': current_price * 0.95
                    })
                
                elif latest['volume_ratio'] > 2 and latest['price_change_pct'] < -2:
                    signals.append({
                        'type': 'technical',
                        'signal': SignalType.SELL,
                        'strength': SignalStrength.HIGH,
                        'confidence': 0.8,
                        'reason': f'Volume breakdown - {latest["volume_ratio"]:.1f}x average volume',
                        'indicator': 'VOLUME_BREAKDOWN',
                        'target_price': current_price * 0.92,
                        'stop_loss': current_price * 1.05
                    })
            
            # 6. Stochastic Signals
            if pd.notna(latest['stoch_k']) and pd.notna(latest['stoch_d']):
                # Stochastic oversold
                if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
                    signals.append({
                        'type': 'technical',
                        'signal': SignalType.BUY,
                        'strength': SignalStrength.LOW,
                        'confidence': 0.6,
                        'reason': 'Stochastic oversold condition',
                        'indicator': 'STOCHASTIC',
                        'target_price': current_price * 1.02,
                        'stop_loss': current_price * 0.98
                    })
                
                # Stochastic overbought
                elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
                    signals.append({
                        'type': 'technical',
                        'signal': SignalType.SELL,
                        'strength': SignalStrength.LOW,
                        'confidence': 0.6,
                        'reason': 'Stochastic overbought condition',
                        'indicator': 'STOCHASTIC',
                        'target_price': current_price * 0.98,
                        'stop_loss': current_price * 1.02
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Technical signal generation failed: {e}")
            return []

    async def generate_ml_signals(self, symbol: str, market_data: Dict[str, Any], 
                                user_id: str, model_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Generate ML-based trading signals"""
        signals = []
        
        try:
            # Get available models for the user
            if model_ids:
                available_models = model_ids
            else:
                # In real implementation, query database for user's trained models for this symbol
                available_models = []
            
            if not available_models:
                return signals
            
            # Use the best performing model or ensemble of models
            for model_id in available_models[:3]:  # Limit to top 3 models
                try:
                    # Load model
                    model_data = ml_service.load_model(model_id)
                    metadata = model_data['metadata']
                    
                    # Verify ownership and symbol match
                    if metadata.get('user_id') != user_id:
                        continue
                    
                    training_request = metadata.get('training_request', {})
                    if training_request.get('symbol') != symbol:
                        continue
                    
                    # Prepare features for prediction
                    feature_columns = metadata.get('feature_columns', [])
                    
                    # Create feature DataFrame from market data
                    latest_data = market_data['data'][-1] if market_data['data'] else None
                    if not latest_data:
                        continue
                    
                    features_dict = {
                        'open': latest_data.open,
                        'high': latest_data.high,
                        'low': latest_data.low,
                        'close': latest_data.close,
                        'volume': latest_data.volume
                    }
                    
                    # Add technical indicators if available
                    df = pd.DataFrame([features_dict])
                    df = self.calculate_technical_indicators(df)
                    latest_features = df.iloc[-1]
                    
                    # Select required features
                    available_features = {}
                    for col in feature_columns:
                        if col in latest_features.index:
                            available_features[col] = latest_features[col]
                    
                    if len(available_features) < len(feature_columns) * 0.8:  # Need at least 80% of features
                        continue
                    
                    # Make prediction
                    features_df = pd.DataFrame([available_features])
                    prediction_data = prediction_service.make_single_prediction(model_data, features_df)
                    
                    predicted_value = prediction_data['prediction']
                    confidence = prediction_data['confidence'] or 0.5
                    
                    # Convert prediction to signal
                    current_price = latest_data.close
                    
                    # For regression models, compare predicted return with thresholds
                    if abs(predicted_value) > 1:  # If predicted return > 1%
                        if predicted_value > 0:
                            signal_type = SignalType.BUY
                            target_price = current_price * (1 + predicted_value / 100)
                            stop_loss = current_price * 0.97
                        else:
                            signal_type = SignalType.SELL
                            target_price = current_price * (1 + predicted_value / 100)
                            stop_loss = current_price * 1.03
                        
                        # Determine signal strength based on confidence and predicted magnitude
                        if confidence > 0.8 and abs(predicted_value) > 3:
                            strength = SignalStrength.VERY_HIGH
                        elif confidence > 0.7 and abs(predicted_value) > 2:
                            strength = SignalStrength.HIGH
                        elif confidence > 0.6:
                            strength = SignalStrength.MEDIUM
                        else:
                            strength = SignalStrength.LOW
                        
                        signals.append({
                            'type': 'ml',
                            'signal': signal_type,
                            'strength': strength,
                            'confidence': confidence,
                            'reason': f'ML model predicts {predicted_value:.2f}% return',
                            'indicator': f'ML_MODEL_{model_id[-6:]}',
                            'target_price': target_price,
                            'stop_loss': stop_loss,
                            'model_id': model_id,
                            'predicted_return': predicted_value
                        })
                
                except Exception as e:
                    logger.warning(f"Failed to generate ML signal with model {model_id}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            logger.error(f"ML signal generation failed: {e}")
            return []

    def combine_signals(self, technical_signals: List[Dict[str, Any]], 
                       ml_signals: List[Dict[str, Any]], symbol: str) -> List[TradingSignal]:
        """Combine and prioritize technical and ML signals"""
        
        try:
            combined_signals = []
            
            # Group signals by type
            buy_signals = []
            sell_signals = []
            
            all_signals = technical_signals + ml_signals
            
            for signal_data in all_signals:
                if signal_data['signal'] == SignalType.BUY:
                    buy_signals.append(signal_data)
                elif signal_data['signal'] == SignalType.SELL:
                    sell_signals.append(signal_data)
            
            # Process buy signals
            if buy_signals:
                # Sort by confidence and strength
                buy_signals.sort(key=lambda x: (x['confidence'], self.strength_to_numeric(x['strength'])), reverse=True)
                
                # Take the strongest buy signal
                best_buy = buy_signals[0]
                
                # Combine supporting factors
                supporting_factors = [best_buy['reason']]
                risk_factors = []
                
                # Add other supporting buy signals
                for signal in buy_signals[1:3]:  # Max 2 additional supporting signals
                    supporting_factors.append(signal['reason'])
                
                # Check for conflicting sell signals
                if sell_signals:
                    risk_factors.append(f"{len(sell_signals)} conflicting sell signals detected")
                
                combined_signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    signal_strength=SignalStrength(best_buy['strength']),
                    confidence=best_buy['confidence'],
                    target_price=best_buy.get('target_price'),
                    stop_loss=best_buy.get('stop_loss'),
                    expected_return=((best_buy.get('target_price', 0) / (best_buy.get('stop_loss', 1) or 1)) - 1) * 100 if best_buy.get('target_price') and best_buy.get('stop_loss') else None,
                    time_horizon=PredictionHorizon.NEXT_DAY,
                    reasoning=best_buy['reason'],
                    supporting_factors=supporting_factors,
                    risk_factors=risk_factors if risk_factors else None,
                    generated_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(hours=24)
                ))
            
            # Process sell signals
            if sell_signals:
                sell_signals.sort(key=lambda x: (x['confidence'], self.strength_to_numeric(x['strength'])), reverse=True)
                
                best_sell = sell_signals[0]
                
                supporting_factors = [best_sell['reason']]
                risk_factors = []
                
                for signal in sell_signals[1:3]:
                    supporting_factors.append(signal['reason'])
                
                if buy_signals:
                    risk_factors.append(f"{len(buy_signals)} conflicting buy signals detected")
                
                combined_signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    signal_strength=SignalStrength(best_sell['strength']),
                    confidence=best_sell['confidence'],
                    target_price=best_sell.get('target_price'),
                    stop_loss=best_sell.get('stop_loss'),
                    expected_return=((best_sell.get('target_price', 0) / (best_sell.get('stop_loss', 1) or 1)) - 1) * 100 if best_sell.get('target_price') and best_sell.get('stop_loss') else None,
                    time_horizon=PredictionHorizon.NEXT_DAY,
                    reasoning=best_sell['reason'],
                    supporting_factors=supporting_factors,
                    risk_factors=risk_factors if risk_factors else None,
                    generated_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(hours=24)
                ))
            
            # If no strong signals, create a HOLD signal
            if not combined_signals:
                combined_signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    signal_strength=SignalStrength.MEDIUM,
                    confidence=0.6,
                    time_horizon=PredictionHorizon.NEXT_DAY,
                    reasoning="No strong directional signals detected",
                    supporting_factors=["Market conditions appear neutral"],
                    generated_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(hours=24)
                ))
            
            return combined_signals
            
        except Exception as e:
            logger.error(f"Signal combination failed: {e}")
            return []

    def strength_to_numeric(self, strength: str) -> int:
        """Convert signal strength to numeric for sorting"""
        strength_map = {
            SignalStrength.VERY_LOW: 1,
            SignalStrength.LOW: 2,
            SignalStrength.MEDIUM: 3,
            SignalStrength.HIGH: 4,
            SignalStrength.VERY_HIGH: 5
        }
        return strength_map.get(strength, 3)

# Global service instance
signal_service = TradingSignalService()

# ============================================
# Route Handlers
# ============================================

@router.post("/generate", response_model=SignalGenerationResponse)
async def generate_trading_signals(
    request: SignalGenerationRequest,
    current_user: dict = Depends(get_current_active_user),
    data_manager: DataManager = Depends(get_data_manager),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Generate comprehensive trading signals for specified symbols"""
    
    # Validate symbols
    for symbol in request.symbols:
        validate_symbol(symbol)
    
    try:
        start_time = datetime.utcnow()
        all_signals = []
        failed_symbols = []
        total_analyzed = 0
        
        for symbol in request.symbols:
            try:
                # Check cache first
                cache_key = f"signals:{symbol}:{request.time_horizon.value}:{request.min_confidence}"
                cached_signals = await cache_manager.get(cache_key)
                
                if cached_signals:
                    logger.info(f"Serving cached signals for {symbol}")
                    all_signals.extend([TradingSignal(**sig) for sig in cached_signals])
                    total_analyzed += 1
                    continue
                
                # Get market data
                end_date = date.today()
                start_date = end_date - timedelta(days=100)  # Get 100 days of data
                
                market_data = await data_manager.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval='1d'
                )
                
                if not market_data or not market_data.data or len(market_data.data) < 20:
                    failed_symbols.append(symbol)
                    continue
                
                # Convert to DataFrame for analysis
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
                
                # Calculate technical indicators
                df = signal_service.calculate_technical_indicators(df)
                
                # Generate technical signals
                technical_signals = signal_service.generate_technical_signals(df, symbol)
                
                # Generate ML signals
                ml_signals = await signal_service.generate_ml_signals(
                    symbol, market_data, current_user["user_id"], request.model_ids
                )
                
                # Combine and filter signals
                symbol_signals = signal_service.combine_signals(technical_signals, ml_signals, symbol)
                
                # Apply confidence filter
                filtered_signals = [
                    sig for sig in symbol_signals 
                    if sig.confidence >= request.min_confidence
                ]
                
                # Apply signal type filter if specified
                if request.signal_types:
                    filtered_signals = [
                        sig for sig in filtered_signals
                        if sig.signal_type in request.signal_types
                    ]
                
                all_signals.extend(filtered_signals)
                total_analyzed += 1
                
                # Cache the signals
                if filtered_signals:
                    signal_dicts = [sig.model_dump() for sig in filtered_signals]
                    await cache_manager.set(cache_key, signal_dicts, ttl=signal_service.cache_ttl)
                
            except Exception as e:
                logger.warning(f"Failed to generate signals for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        # Calculate processing summary
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        avg_processing_time = processing_time / len(request.symbols) if request.symbols else 0
        
        # Generate signal distribution
        signal_distribution = {}
        for signal_type in SignalType:
            count = len([s for s in all_signals if s.signal_type == signal_type])
            if count > 0:
                signal_distribution[signal_type.value] = count
        
        # Calculate average confidence
        avg_confidence = sum(s.confidence for s in all_signals) / len(all_signals) if all_signals else 0
        
        response = SignalGenerationResponse(
            signals=all_signals,
            total_symbols_analyzed=total_analyzed,
            signals_generated=len(all_signals),
            average_confidence=round(avg_confidence, 3),
            signal_distribution=signal_distribution,
            failed_symbols=failed_symbols if failed_symbols else None,
            processing_summary={
                "total_processing_time_ms": round(processing_time, 2),
                "avg_processing_time_per_symbol_ms": round(avg_processing_time, 2),
                "models_used": request.model_ids or [],
                "success_rate": round(total_analyzed / len(request.symbols), 3) if request.symbols else 0
            }
        )
        
        logger.info(f"Generated {len(all_signals)} signals for {total_analyzed} symbols")
        
        return response
        
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")

@router.get("/live/{symbol}", response_model=List[TradingSignal])
async def get_live_signals(
    symbol: str,
    min_confidence: float = Query(0.6, ge=0.0, le=1.0),
    include_reasoning: bool = Query(True),
    current_user: dict = Depends(get_current_active_user),
    data_manager: DataManager = Depends(get_data_manager),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Get live trading signals for a single symbol"""
    
    validate_symbol(symbol)
    
    try:
        # Check cache first
        cache_key = f"live_signals:{symbol}:{min_confidence}"
        cached_signals = await cache_manager.get(cache_key)
        
        if cached_signals:
            return [TradingSignal(**sig) for sig in cached_signals]
        
        # Generate fresh signals
        request = SignalGenerationRequest(
            symbols=[symbol],
            min_confidence=min_confidence,
            include_reasoning=include_reasoning
        )
        
        response = await generate_trading_signals(request, current_user, data_manager, cache_manager)
        
        # Cache for shorter time for live signals
        if response.signals:
            signal_dicts = [sig.model_dump() for sig in response.signals]
            await cache_manager.set(cache_key, signal_dicts, ttl=30)  # 30 seconds cache
        
        return response.signals
        
    except Exception as e:
        logger.error(f"Live signal generation failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Live signal generation failed: {str(e)}")

@router.get("/history")
async def get_signal_history(
    symbol: Optional[str] = Query(None),
    signal_type: Optional[SignalType] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_active_user)
):
    """Get historical trading signals with filtering"""
    
    # In real implementation, query database for user's signal history
    history = {
        "total": 0,
        "limit": limit,
        "offset": offset,
        "signals": [],
        "summary": {
            "total_signals": 0,
            "by_type": {},
            "by_symbol": {},
            "avg_confidence": 0.0,
            "success_rate": 0.0  # Would need outcome tracking
        }
    }
    
    return history

@router.get("/performance")
async def get_signal_performance(
    symbol: Optional[str] = Query(None),
    days_back: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(get_current_active_user)
):
    """Get signal performance analytics"""
    
    # In real implementation, calculate actual signal performance
    # by comparing signal recommendations with actual price movements
    
    performance = {
        "period_days": days_back,
        "total_signals": 0,
        "accurate_signals": 0,
        "accuracy_rate": 0.0,
        "avg_return_per_signal": 0.0,
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "by_signal_type": {},
        "by_symbol": {} if symbol else None,
        "monthly_performance": []
    }
    
    return performance

@router.get("/alerts")
async def get_signal_alerts(
    active_only: bool = Query(True),
    min_confidence: float = Query(0.7, ge=0.0, le=1.0),
    current_user: dict = Depends(get_current_active_user)
):
    """Get active signal alerts for user's watchlist"""
    
    # In real implementation, get user's watchlist and check for high-confidence signals
    alerts = {
        "active_alerts": 0,
        "total_alerts": 0,
        "alerts": [],
        "alert_summary": {
            "high_confidence_buy": 0,
            "high_confidence_sell": 0,
            "volume_breakouts": 0,
            "technical_patterns": 0
        }
    }
    
    return alerts

# Export router
__all__ = ["router"]
