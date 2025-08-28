"""
tests/utils/mock_factories.py

Mock data factories for comprehensive testing in StockPredictionPro.
Generates realistic, reproducible test data for stocks, fundamentals, API responses,
machine learning models, and system components.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import json
import random
import string
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
import tempfile
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# ============================================
# STOCK MARKET DATA FACTORIES
# ============================================

class StockDataFactory:
    """Factory for generating realistic stock market data"""
    
    @staticmethod
    def create_ohlcv_data(symbol: str = 'MOCK', days: int = 30, 
                         start_price: float = 100.0, volatility: float = 0.02,
                         trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
        """
        Generate realistic OHLCV stock data with trend and volatility
        
        Args:
            symbol: Stock symbol
            days: Number of trading days
            start_price: Starting price
            volatility: Daily volatility (std dev of returns)
            trend: Daily trend (mean return)
            seed: Random seed for reproducibility
        
        Returns:
            DataFrame with OHLCV data
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate date range (skip weekends)
        start_date = datetime.now() - timedelta(days=days * 2)  # Buffer for weekends
        all_dates = pd.date_range(start=start_date, periods=days * 2, freq='D')
        # Filter to weekdays only
        trading_dates = [d for d in all_dates if d.weekday() < 5][:days]
        
        # Generate price series with geometric Brownian motion
        returns = np.random.normal(trend, volatility, len(trading_dates))
        prices = [start_price]
        
        for return_rate in returns[1:]:
            new_price = prices[-1] * (1 + return_rate)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        data = []
        for i, trade_date in enumerate(trading_dates):
            close_price = prices[i]
            
            # Generate realistic intraday movement
            daily_vol = volatility / 4  # Intraday volatility is lower
            
            # Open price (based on gap from previous close)
            gap = np.random.normal(0, daily_vol / 2)
            if i == 0:
                open_price = start_price
            else:
                open_price = prices[i-1] * (1 + gap)
                open_price = max(open_price, 0.01)
            
            # High and low prices
            intraday_range = abs(np.random.normal(0, daily_vol))
            price_range = [open_price, close_price]
            
            high_price = max(price_range) * (1 + intraday_range)
            low_price = min(price_range) * (1 - intraday_range)
            
            # Ensure OHLC logic
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate realistic volume (log-normal distribution)
            base_volume = 1000000  # 1M shares
            volume = int(np.random.lognormal(np.log(base_volume), 0.5))
            volume = max(volume, 100)  # Minimum volume
            
            data.append({
                'date': trade_date.date(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'Symbol': symbol,
                'Source': 'mock_factory'
            })
        
        df = pd.DataFrame(data)
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    @staticmethod
    def create_multi_stock_data(symbols: List[str], days: int = 30, 
                               correlations: Dict[str, float] = None,
                               seed: int = 42) -> pd.DataFrame:
        """
        Generate correlated multi-stock data
        
        Args:
            symbols: List of stock symbols
            days: Number of trading days
            correlations: Dict mapping symbol pairs to correlation coefficients
            seed: Random seed
        
        Returns:
            Combined DataFrame with all stocks
        """
        if correlations is None:
            correlations = {}
        
        all_data = []
        base_returns = None
        
        for i, symbol in enumerate(symbols):
            # Create base price and volatility variation
            start_price = random.uniform(50, 500)
            volatility = random.uniform(0.015, 0.035)
            trend = random.uniform(-0.002, 0.005)
            
            # Generate correlated returns if specified
            if i == 0 or symbol not in correlations:
                # First stock or uncorrelated stock
                stock_data = StockDataFactory.create_ohlcv_data(
                    symbol, days, start_price, volatility, trend, seed + i
                )
            else:
                # Create correlated returns
                correlation = correlations.get(symbol, 0.0)
                
                np.random.seed(seed + i)
                independent_returns = np.random.normal(0, volatility, days)
                
                if base_returns is not None:
                    # Mix base returns with independent returns
                    corr_returns = (correlation * base_returns + 
                                  np.sqrt(1 - correlation**2) * independent_returns)
                else:
                    corr_returns = independent_returns
                
                # Generate price series manually
                prices = [start_price]
                for ret in corr_returns[1:]:
                    new_price = prices[-1] * (1 + trend + ret)
                    prices.append(max(new_price, 0.01))
                
                # Use the price series to create OHLCV data
                stock_data = StockDataFactory.create_ohlcv_data(
                    symbol, days, start_price, volatility, trend, seed + i
                )
                # Replace close prices with correlated prices
                stock_data['close'] = [round(p, 2) for p in prices]
            
            if base_returns is None:
                # Calculate returns for correlation baseline
                closes = stock_data['close'].values
                base_returns = np.diff(closes) / closes[:-1]
                base_returns = np.concatenate([[0], base_returns])  # Pad for alignment
            
            all_data.append(stock_data)
        
        return pd.concat(all_data, ignore_index=True)

class FundamentalsFactory:
    """Factory for generating fundamental data"""
    
    @staticmethod
    def create_company_fundamentals(symbol: str = 'MOCK', 
                                  company_type: str = 'growth',
                                  seed: int = 42) -> Dict[str, Any]:
        """
        Generate realistic company fundamental data
        
        Args:
            symbol: Stock symbol
            company_type: 'growth', 'value', 'dividend', 'tech', 'utility'
            seed: Random seed
        
        Returns:
            Dictionary with fundamental metrics
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Base company info
        company_names = {
            'tech': f'{symbol} Technologies Inc.',
            'growth': f'{symbol} Growth Corp.',
            'value': f'{symbol} Industries Ltd.',
            'dividend': f'{symbol} Utilities Co.',
            'utility': f'{symbol} Power & Gas'
        }
        
        sectors = {
            'tech': 'Technology',
            'growth': 'Consumer Discretionary', 
            'value': 'Industrials',
            'dividend': 'Utilities',
            'utility': 'Utilities'
        }
        
        industries = {
            'tech': 'Software',
            'growth': 'E-commerce',
            'value': 'Manufacturing',
            'dividend': 'Electric Utilities',
            'utility': 'Gas Utilities'
        }
        
        # Type-specific metrics
        if company_type == 'tech':
            pe_range = (25, 50)
            pb_range = (3, 15)
            roe_range = (0.15, 0.35)
            margin_range = (0.15, 0.35)
            dividend_range = (0, 0.02)
            beta_range = (1.2, 1.8)
        elif company_type == 'growth':
            pe_range = (30, 80)
            pb_range = (2, 8)
            roe_range = (0.12, 0.25)
            margin_range = (0.08, 0.20)
            dividend_range = (0, 0.01)
            beta_range = (1.0, 1.5)
        elif company_type == 'value':
            pe_range = (8, 18)
            pb_range = (0.8, 2.5)
            roe_range = (0.08, 0.18)
            margin_range = (0.05, 0.15)
            dividend_range = (0.02, 0.05)
            beta_range = (0.8, 1.2)
        elif company_type == 'dividend':
            pe_range = (12, 25)
            pb_range = (1.2, 3.0)
            roe_range = (0.10, 0.20)
            margin_range = (0.10, 0.25)
            dividend_range = (0.04, 0.08)
            beta_range = (0.6, 1.0)
        else:  # utility
            pe_range = (15, 25)
            pb_range = (1.0, 2.0)
            roe_range = (0.08, 0.15)
            margin_range = (0.08, 0.18)
            dividend_range = (0.03, 0.06)
            beta_range = (0.5, 0.9)
        
        fundamentals = {
            'symbol': symbol,
            'company_name': company_names.get(company_type, f'{symbol} Corp.'),
            'sector': sectors.get(company_type, 'Diversified'),
            'industry': industries.get(company_type, 'Conglomerates'),
            'country': 'US',
            'currency': 'USD',
            'exchange': random.choice(['NASDAQ', 'NYSE']),
            
            # Valuation metrics
            'market_cap': random.randint(1_000_000_000, 500_000_000_000),  # 1B to 500B
            'pe_ratio': round(random.uniform(*pe_range), 2),
            'forward_pe': round(random.uniform(pe_range[0]*0.9, pe_range[1]*0.9), 2),
            'price_to_book': round(random.uniform(*pb_range), 2),
            'price_to_sales': round(random.uniform(1.0, 8.0), 2),
            'peg_ratio': round(random.uniform(0.5, 3.0), 2),
            
            # Profitability metrics
            'roe': round(random.uniform(*roe_range), 4),
            'roa': round(random.uniform(0.03, 0.15), 4),
            'profit_margin': round(random.uniform(*margin_range), 4),
            'operating_margin': round(random.uniform(margin_range[0]*0.8, margin_range[1]*1.2), 4),
            'gross_margin': round(random.uniform(0.20, 0.70), 4),
            
            # Financial health
            'debt_to_equity': round(random.uniform(0.1, 2.5), 2),
            'current_ratio': round(random.uniform(1.0, 3.5), 2),
            'quick_ratio': round(random.uniform(0.5, 2.5), 2),
            'cash_ratio': round(random.uniform(0.1, 1.5), 2),
            
            # Market metrics
            'beta': round(random.uniform(*beta_range), 2),
            'dividend_yield': round(random.uniform(*dividend_range), 4),
            'payout_ratio': round(random.uniform(0.2, 0.8), 4) if dividend_range[1] > 0 else 0,
            
            # Size metrics
            'shares_outstanding': random.randint(100_000_000, 10_000_000_000),  # 100M to 10B
            'float_shares': random.randint(50_000_000, 8_000_000_000),
            'insider_ownership': round(random.uniform(0.01, 0.30), 4),
            'institutional_ownership': round(random.uniform(0.40, 0.90), 4),
            
            # Performance metrics
            '52_week_high': round(random.uniform(100, 500), 2),
            '52_week_low': round(random.uniform(20, 200), 2),
            'avg_volume': random.randint(500_000, 50_000_000),
            
            # Growth metrics
            'revenue_growth_yoy': round(random.uniform(-0.1, 0.4), 4),
            'earnings_growth_yoy': round(random.uniform(-0.2, 0.6), 4),
            'revenue_growth_qoq': round(random.uniform(-0.05, 0.2), 4),
            'earnings_growth_qoq': round(random.uniform(-0.1, 0.3), 4),
            
            'last_updated': datetime.now().isoformat()
        }
        
        # Ensure consistency
        fundamentals['52_week_low'] = min(fundamentals['52_week_low'], fundamentals['52_week_high'])
        fundamentals['float_shares'] = min(fundamentals['float_shares'], fundamentals['shares_outstanding'])
        
        return fundamentals

# ============================================
# API RESPONSE FACTORIES
# ============================================

class APIResponseFactory:
    """Factory for generating realistic API responses"""
    
    @staticmethod
    def create_alpha_vantage_response(symbol: str = 'MOCK', days: int = 10,
                                    response_type: str = 'success', seed: int = 42) -> Dict[str, Any]:
        """Generate Alpha Vantage API response"""
        if response_type == 'error':
            return {
                'Error Message': 'Invalid API call. Please retry or visit the documentation for TIME_SERIES_DAILY.'
            }
        elif response_type == 'rate_limit':
            return {
                'Note': 'Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day.'
            }
        
        # Generate successful response
        stock_data = StockDataFactory.create_ohlcv_data(symbol, days, seed=seed)
        
        time_series = {}
        for _, row in stock_data.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            time_series[date_str] = {
                '1. open': f"{row['open']:.4f}",
                '2. high': f"{row['high']:.4f}",
                '3. low': f"{row['low']:.4f}",
                '4. close': f"{row['close']:.4f}",
                '5. volume': str(row['volume'])
            }
        
        return {
            'Meta Data': {
                '1. Information': 'Daily Prices (open, high, low, close) and Volumes',
                '2. Symbol': symbol,
                '3. Last Refreshed': datetime.now().strftime('%Y-%m-%d'),
                '4. Output Size': 'Compact',
                '5. Time Zone': 'US/Eastern'
            },
            'Time Series (Daily)': time_series
        }
    
    @staticmethod
    def create_finnhub_profile_response(symbol: str = 'MOCK', 
                                      response_type: str = 'success',
                                      company_type: str = 'tech') -> Dict[str, Any]:
        """Generate Finnhub company profile response"""
        if response_type == 'error':
            return {}  # Finnhub returns empty dict for invalid symbols
        
        fundamentals = FundamentalsFactory.create_company_fundamentals(symbol, company_type)
        
        return {
            'country': fundamentals['country'],
            'currency': fundamentals['currency'],
            'exchange': fundamentals['exchange'],
            'ipo': (datetime.now() - timedelta(days=random.randint(365, 7300))).strftime('%Y-%m-%d'),
            'marketCapitalization': fundamentals['market_cap'] // 1_000_000,  # In millions
            'name': fundamentals['company_name'],
            'phone': f"+1{random.randint(1000000000, 9999999999)}",
            'shareOutstanding': fundamentals['shares_outstanding'] // 1_000_000,  # In millions
            'ticker': symbol,
            'weburl': f'https://www.{symbol.lower()}.com',
            'logo': f'https://logo.clearbit.com/{symbol.lower()}.com',
            'finnhubIndustry': fundamentals['industry']
        }
    
    @staticmethod
    def create_finnhub_sentiment_response(symbol: str = 'MOCK',
                                        sentiment_bias: str = 'neutral') -> Dict[str, Any]:
        """Generate Finnhub news sentiment response"""
        if sentiment_bias == 'bullish':
            bullish_pct = random.uniform(0.6, 0.8)
            bearish_pct = 1.0 - bullish_pct
            news_score = random.uniform(0.6, 0.8)
        elif sentiment_bias == 'bearish':
            bearish_pct = random.uniform(0.6, 0.8)
            bullish_pct = 1.0 - bearish_pct
            news_score = random.uniform(0.2, 0.4)
        else:  # neutral
            bullish_pct = random.uniform(0.4, 0.6)
            bearish_pct = 1.0 - bullish_pct
            news_score = random.uniform(0.4, 0.6)
        
        return {
            'buzz': {
                'articlesInLastWeek': random.randint(5, 150),
                'buzz': round(random.uniform(0.1, 3.0), 2),
                'weeklyAverage': round(random.uniform(0.5, 2.0), 2)
            },
            'companyNewsScore': round(news_score, 2),
            'sectorAverageNewsScore': round(random.uniform(0.4, 0.7), 2),
            'sentiment': {
                'bearishPercent': round(bearish_pct, 2),
                'bullishPercent': round(bullish_pct, 2)
            }
        }
    
    @staticmethod
    def create_polygon_intraday_response(symbol: str = 'MOCK', 
                                       hours: int = 6) -> Dict[str, Any]:
        """Generate Polygon intraday response"""
        results = []
        now = datetime.now()
        
        # Generate hourly data for market hours
        for i in range(hours):
            timestamp = now.replace(hour=9+i, minute=0, second=0, microsecond=0)
            
            base_price = random.uniform(100, 300)
            results.append({
                't': int(timestamp.timestamp() * 1000),  # Timestamp in milliseconds
                'o': round(base_price * random.uniform(0.98, 1.02), 2),
                'h': round(base_price * random.uniform(1.00, 1.05), 2),
                'l': round(base_price * random.uniform(0.95, 1.00), 2),
                'c': round(base_price * random.uniform(0.99, 1.01), 2),
                'v': random.randint(100000, 1000000)
            })
        
        return {
            'ticker': symbol,
            'queryCount': len(results),
            'resultsCount': len(results),
            'adjusted': True,
            'results': results,
            'status': 'OK',
            'request_id': f'mock_{random.randint(100000, 999999)}',
            'count': len(results)
        }

# ============================================
# ML MODEL FACTORIES
# ============================================

class MLModelFactory:
    """Factory for generating ML model outputs and metrics"""
    
    @staticmethod
    def create_predictions(n_samples: int = 100, model_type: str = 'regression',
                          n_classes: int = 3, seed: int = 42) -> np.ndarray:
        """Generate realistic model predictions"""
        np.random.seed(seed)
        
        if model_type == 'regression':
            # Generate realistic stock price predictions
            predictions = np.random.normal(150, 30, n_samples)
            return np.maximum(predictions, 0.01)  # Ensure positive prices
        
        elif model_type == 'classification':
            # Generate class predictions
            return np.random.randint(0, n_classes, n_samples)
        
        elif model_type == 'classification_proba':
            # Generate probability predictions
            logits = np.random.normal(0, 1, (n_samples, n_classes))
            exp_logits = np.exp(logits)
            probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            return probabilities
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    @staticmethod
    def create_model_metrics(model_type: str = 'regression', 
                           performance_level: str = 'good') -> Dict[str, float]:
        """Generate realistic model performance metrics"""
        if performance_level == 'excellent':
            noise_factor = 0.1
        elif performance_level == 'good':
            noise_factor = 0.2
        elif performance_level == 'poor':
            noise_factor = 0.5
        else:
            noise_factor = 0.3
        
        if model_type == 'regression':
            base_r2 = {'excellent': 0.85, 'good': 0.70, 'poor': 0.30}[performance_level]
            r2 = base_r2 + random.uniform(-0.1, 0.1)
            
            # Generate correlated metrics
            mse = random.uniform(100, 1000) * (1 + noise_factor)
            rmse = np.sqrt(mse)
            mae = rmse * random.uniform(0.6, 0.9)
            mape = random.uniform(5, 25) * (1 + noise_factor)
            
            return {
                'r2_score': round(r2, 4),
                'mse': round(mse, 4),
                'rmse': round(rmse, 4),
                'mae': round(mae, 4),
                'mape': round(mape, 4)
            }
        
        elif model_type == 'classification':
            base_acc = {'excellent': 0.90, 'good': 0.75, 'poor': 0.55}[performance_level]
            accuracy = base_acc + random.uniform(-0.05, 0.05)
            
            # Generate correlated metrics
            precision = accuracy + random.uniform(-0.1, 0.1)
            recall = accuracy + random.uniform(-0.1, 0.1)
            f1 = 2 * (precision * recall) / (precision + recall)
            
            # Ensure valid ranges
            precision = max(0.1, min(1.0, precision))
            recall = max(0.1, min(1.0, recall))
            accuracy = max(0.1, min(1.0, accuracy))
            f1 = max(0.1, min(1.0, f1))
            
            return {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'auc_roc': round(random.uniform(0.6, 0.95), 4)
            }
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

# ============================================
# SYSTEM FACTORIES
# ============================================

class SystemFactory:
    """Factory for generating system-level test data"""
    
    @staticmethod
    def create_temp_file(content: str = "", suffix: str = '.txt') -> str:
        """Create temporary file with content"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def create_temp_csv(df: pd.DataFrame) -> str:
        """Create temporary CSV file from DataFrame"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def create_config_dict(config_type: str = 'default') -> Dict[str, Any]:
        """Generate configuration dictionaries"""
        if config_type == 'api':
            return {
                'api_keys': {
                    'alpha_vantage': 'mock_alpha_key',
                    'finnhub': 'mock_finnhub_key',
                    'polygon': 'mock_polygon_key',
                    'twelvedata': 'mock_twelvedata_key'
                },
                'rate_limits': {
                    'alpha_vantage': {'calls_per_minute': 5},
                    'finnhub': {'calls_per_minute': 60},
                    'polygon': {'calls_per_minute': 5},
                    'twelvedata': {'calls_per_minute': 8}
                },
                'timeouts': {
                    'default': 30,
                    'long_running': 120
                }
            }
        
        elif config_type == 'model':
            return {
                'model_params': {
                    'random_state': 42,
                    'test_size': 0.2,
                    'cv_folds': 5
                },
                'feature_engineering': {
                    'lookback_window': 30,
                    'technical_indicators': ['rsi', 'macd', 'sma', 'ema'],
                    'fundamental_features': ['pe_ratio', 'price_to_book', 'roe']
                },
                'training': {
                    'algorithms': ['RandomForest', 'XGBoost', 'LSTM'],
                    'hyperparameter_tuning': True,
                    'early_stopping': True
                }
            }
        
        else:  # default
            return {
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'data_sources': {
                    'primary': 'alpha_vantage',
                    'fallback': ['finnhub', 'polygon']
                },
                'cache': {
                    'enabled': True,
                    'ttl': 3600,
                    'max_size': 1000
                }
            }

# ============================================
# MOCK OBJECT FACTORIES
# ============================================

class MockObjectFactory:
    """Factory for creating mock objects and responses"""
    
    @staticmethod
    def create_mock_response(status_code: int = 200, json_data: Dict[str, Any] = None,
                           text: str = "", headers: Dict[str, str] = None) -> Mock:
        """Create mock HTTP response"""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data or {}
        mock_response.text = text or json.dumps(json_data or {})
        mock_response.headers = headers or {'Content-Type': 'application/json'}
        
        if status_code >= 400:
            mock_response.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
        else:
            mock_response.raise_for_status.return_value = None
        
        return mock_response
    
    @staticmethod
    def create_mock_database_cursor(query_results: List[Tuple] = None) -> Mock:
        """Create mock database cursor"""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = query_results or []
        mock_cursor.fetchone.return_value = query_results[0] if query_results else None
        mock_cursor.rowcount = len(query_results) if query_results else 0
        return mock_cursor
    
    @staticmethod
    def create_mock_file_system(files: Dict[str, str] = None) -> Mock:
        """Create mock file system operations"""
        files = files or {}
        
        def mock_open_func(filename, mode='r', *args, **kwargs):
            if filename in files:
                from io import StringIO
                return StringIO(files[filename])
            else:
                raise FileNotFoundError(f"Mock file not found: {filename}")
        
        mock_fs = Mock()
        mock_fs.open = mock_open_func
        mock_fs.exists = lambda path: path in files
        mock_fs.listdir = lambda path: list(files.keys())
        
        return mock_fs

# ============================================
# UTILITY FUNCTIONS
# ============================================

def generate_random_string(length: int = 8, charset: str = 'alphanumeric') -> str:
    """Generate random string for testing"""
    if charset == 'alphanumeric':
        chars = string.ascii_letters + string.digits
    elif charset == 'alphabetic':
        chars = string.ascii_letters
    elif charset == 'numeric':
        chars = string.digits
    elif charset == 'symbols':
        chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'
    else:
        chars = charset
    
    return ''.join(random.choices(chars, k=length))

def generate_test_symbols(count: int = 5, prefix: str = 'TEST') -> List[str]:
    """Generate list of test stock symbols"""
    return [f"{prefix}{i:03d}" for i in range(1, count + 1)]

def cleanup_temp_files(file_paths: List[str]) -> None:
    """Clean up temporary files created during testing"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not remove temp file {file_path}: {e}")

# ============================================
# TEST FUNCTIONS FOR FACTORIES
# ============================================

def test_stock_data_factory():
    """Test StockDataFactory"""
    # Test single stock data
    df = StockDataFactory.create_ohlcv_data('TEST', 10, 100.0, seed=42)
    assert len(df) == 10
    assert df['Symbol'].iloc[0] == 'TEST'
    assert all(df['high'] >= df[['open', 'low', 'close']].max(axis=1))
    
    # Test multi-stock data
    symbols = ['TEST1', 'TEST2', 'TEST3']
    multi_df = StockDataFactory.create_multi_stock_data(symbols, 5)
    assert len(multi_df) == 15  # 3 stocks * 5 days

def test_api_response_factory():
    """Test APIResponseFactory"""
    # Test Alpha Vantage response
    av_response = APIResponseFactory.create_alpha_vantage_response('TEST', 5)
    assert 'Meta Data' in av_response
    assert 'Time Series (Daily)' in av_response
    assert len(av_response['Time Series (Daily)']) == 5
    
    # Test Finnhub response
    fh_response = APIResponseFactory.create_finnhub_profile_response('TEST')
    assert 'name' in fh_response
    assert 'ticker' in fh_response
    assert fh_response['ticker'] == 'TEST'

def test_ml_model_factory():
    """Test MLModelFactory"""
    # Test regression predictions
    reg_pred = MLModelFactory.create_predictions(100, 'regression')
    assert len(reg_pred) == 100
    assert all(reg_pred > 0)  # Stock prices should be positive
    
    # Test classification probabilities
    class_proba = MLModelFactory.create_predictions(50, 'classification_proba', 3)
    assert class_proba.shape == (50, 3)
    assert np.allclose(class_proba.sum(axis=1), 1.0)  # Probabilities sum to 1

if __name__ == "__main__":
    """Run tests when module is executed directly"""
    print("Running mock factory validation tests...")
    
    try:
        test_stock_data_factory()
        print("✅ StockDataFactory tests passed")
        
        test_api_response_factory()
        print("✅ APIResponseFactory tests passed")
        
        test_ml_model_factory()
        print("✅ MLModelFactory tests passed")
        
        print("\n🎉 All mock factories validated successfully!")
        print("✅ Ready to generate realistic test data")
        
        # Demo usage
        print("\n📊 Demo: Generating sample data...")
        
        demo_stock = StockDataFactory.create_ohlcv_data('DEMO', 5)
        print(f"Generated stock data: {len(demo_stock)} records")
        
        demo_fundamentals = FundamentalsFactory.create_company_fundamentals('DEMO', 'tech')
        print(f"Generated fundamentals: {demo_fundamentals['company_name']}")
        
        demo_api = APIResponseFactory.create_alpha_vantage_response('DEMO', 3)
        print(f"Generated API response: {len(demo_api['Time Series (Daily)'])} days")
        
    except Exception as e:
        print(f"❌ Mock factory validation failed: {e}")
        import traceback
        traceback.print_exc()
