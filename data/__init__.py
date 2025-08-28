"""
data/__init__.py

StockPredictionPro Data Management System
Handles data extraction using Alpha Vantage, Finnhub, Polygon, and TwelveData APIs
Now includes: Top US Stocks (50) + Top Indian Stocks (Nifty 50) + Global Giants
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yfinance as yf
import requests
from dotenv import load_dotenv
import time
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockPredictionPro.Data')

# Data directories
DATA_ROOT = Path(__file__).parent
RAW_DIR = DATA_ROOT / 'raw'
PROCESSED_DIR = DATA_ROOT / 'processed'

# API Configuration from environment
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')

# 🇺🇸 TOP 50 US STOCKS (S&P 500 + NASDAQ 100 Giants)
US_TOP_50 = [
    # Mega Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE',
    # Major Tech & Growth
    'CRM', 'ORCL', 'INTC', 'AMD', 'PYPL', 'UBER', 'ABNB', 'SHOP', 'SNOW', 'PLTR',
    # Finance & Banking
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'MA',
    # Healthcare & Biotech
    'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'ABT', 'MRK', 'LLY', 'AMGN', 'GILD',
    # Consumer & Retail
    'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'PG', 'KO'
]

# 🇮🇳 TOP 50 INDIAN STOCKS (NIFTY 50 + Major BSE)
INDIAN_TOP_50 = [
    # Nifty 50 Major Companies (NSE)
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
    'INFY.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'HCLTECH.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'BAJFINANCE.NS',
    'AXISBANK.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
    'POWERGRID.NS', 'NTPC.NS', 'WIPRO.NS', 'TECHM.NS', 'ONGC.NS',
    'TATAMOTORS.NS', 'M&M.NS', 'BAJAJFINSV.NS', 'DRREDDY.NS', 'COALINDIA.NS',
    'EICHERMOT.NS', 'CIPLA.NS', 'GRASIM.NS', 'JSWSTEEL.NS', 'HINDALCO.NS',
    'INDUSINDBK.NS', 'TATASTEEL.NS', 'ADANIPORTS.NS', 'HEROMOTOCO.NS', 'DIVISLAB.NS',
    'BRITANNIA.NS', 'APOLLOHOSP.NS', 'UPL.NS', 'SHREECEM.NS', 'BPCL.NS',
    'TATACONSUM.NS', 'LTIM.NS', 'ADANIENT.NS', 'SBILIFE.NS', 'HDFCLIFE.NS'
]

# Combined symbols list
ALL_SYMBOLS = US_TOP_50 + INDIAN_TOP_50

class EnhancedStockDataExtractor:
    """Enhanced data extraction for US and Indian markets"""
    
    def __init__(self):
        self.session = requests.Session()
        
        # Validate API keys
        self.api_status = {
            'alpha_vantage': bool(ALPHA_VANTAGE_API_KEY),
            'finnhub': bool(FINNHUB_API_KEY),
            'polygon': bool(POLYGON_API_KEY),
            'twelvedata': bool(TWELVEDATA_API_KEY)
        }
        
        logger.info(f"API Status: {self.api_status}")
    
    def extract_all_data(self, symbols: List[str] = None, market_filter: str = 'all') -> Dict[str, pd.DataFrame]:
        """
        Extract data from all configured APIs
        
        Args:
            symbols: List of symbols to extract (default: ALL_SYMBOLS)
            market_filter: 'all', 'us', 'indian' to filter markets
        """
        if symbols is None:
            if market_filter == 'us':
                symbols = US_TOP_50
            elif market_filter == 'indian':
                symbols = INDIAN_TOP_50
            else:
                symbols = ALL_SYMBOLS
        
        logger.info(f"🚀 Starting data extraction for {len(symbols)} symbols ({market_filter} market)")
        
        all_data = {}
        
        # 1. Extract daily prices (yfinance - works for both US & Indian markets)
        logger.info("📈 Extracting daily price data...")
        daily_data = self.extract_daily_prices(symbols)
        all_data['daily_prices'] = daily_data
        
        # 2. Extract technical indicators (Alpha Vantage - primarily US)
        logger.info("📊 Extracting technical indicators...")
        us_symbols = [s for s in symbols if '.NS' not in s and '.BO' not in s]
        technical_data = self.extract_technical_indicators(us_symbols[:10])  # Limit for API rate limits
        all_data['technical_indicators'] = technical_data
        
        # 3. Extract fundamentals (Finnhub - global)
        logger.info("🏢 Extracting company fundamentals...")
        fundamentals_data = self.extract_fundamentals(symbols[:20])  # Limit for rate limits
        all_data['fundamentals'] = fundamentals_data
        
        # 4. Extract news sentiment (Finnhub - global)
        logger.info("📰 Extracting news sentiment...")
        news_data = self.extract_news_sentiment(symbols[:15])
        all_data['news_sentiment'] = news_data
        
        # 5. Extract intraday data (Polygon - US only)
        logger.info("⏰ Extracting intraday data...")
        intraday_data = self.extract_intraday_data(us_symbols[:5])
        all_data['intraday'] = intraday_data
        
        # 6. Extract market indices
        logger.info("📊 Extracting market indices...")
        indices_data = self.extract_market_indices()
        all_data['market_indices'] = indices_data
        
        # Save all data
        self.save_all_data(all_data, market_filter)
        
        logger.info("✅ Data extraction completed successfully")
        return all_data
    
    def extract_daily_prices(self, symbols: List[str]) -> pd.DataFrame:
        """Extract daily OHLCV data using yfinance (supports both US and Indian markets)"""
        all_price_data = []
        
        logger.info(f"Processing {len(symbols)} symbols for daily data...")
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"[{i+1}/{len(symbols)}] Fetching daily data for {symbol}")
                
                # yfinance works for both US and Indian stocks
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")  # Last 1 year
                
                if not data.empty:
                    data = data.reset_index()
                    data['Symbol'] = symbol
                    data['Market'] = 'Indian' if '.NS' in symbol or '.BO' in symbol else 'US'
                    data['Source'] = 'yfinance'
                    
                    # Standardize column names
                    data = data.rename(columns={
                        'Date': 'date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    
                    all_price_data.append(data)
                    
                    # Save individual symbol data
                    market_subdir = 'indian' if data['Market'].iloc[0] == 'Indian' else 'us'
                    symbol_clean = symbol.replace('.NS', '').replace('.BO', '')
                    symbol_file = RAW_DIR / 'daily' / market_subdir / f'{symbol_clean}_daily.csv'
                    symbol_file.parent.mkdir(parents=True, exist_ok=True)
                    data.to_csv(symbol_file, index=False)
                    
                    logger.info(f"✅ Saved {symbol}: {len(data)} records")
                else:
                    logger.warning(f"⚠️ No data found for {symbol}")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch data for {symbol}: {e}")
                continue
        
        # Combine all data
        if all_price_data:
            combined_df = pd.concat(all_price_data, ignore_index=True)
            
            # Save combined files by market
            us_data = combined_df[combined_df['Market'] == 'US']
            indian_data = combined_df[combined_df['Market'] == 'Indian']
            
            if not us_data.empty:
                us_file = RAW_DIR / 'daily' / 'us' / 'all_us_stocks_daily.csv'
                us_file.parent.mkdir(parents=True, exist_ok=True)
                us_data.to_csv(us_file, index=False)
                logger.info(f"📊 Saved US stocks data: {len(us_data)} records")
            
            if not indian_data.empty:
                indian_file = RAW_DIR / 'daily' / 'indian' / 'all_indian_stocks_daily.csv'  
                indian_file.parent.mkdir(parents=True, exist_ok=True)
                indian_data.to_csv(indian_file, index=False)
                logger.info(f"📊 Saved Indian stocks data: {len(indian_data)} records")
            
            # Save master combined file
            combined_file = RAW_DIR / 'daily' / 'all_global_stocks_daily.csv'
            combined_df.to_csv(combined_file, index=False)
            
            return combined_df
        
        return pd.DataFrame()
    
    def extract_technical_indicators(self, symbols: List[str]) -> pd.DataFrame:
        """Extract technical indicators using Alpha Vantage (US symbols only)"""
        if not ALPHA_VANTAGE_API_KEY:
            logger.warning("Alpha Vantage API key missing - skipping technical indicators")
            return pd.DataFrame()
        
        technical_data = []
        
        for symbol in symbols[:5]:  # Limit to 5 symbols due to API rate limits
            try:
                logger.info(f"Fetching technical indicators for {symbol}")
                
                # RSI (Relative Strength Index)
                rsi_url = "https://www.alphavantage.co/query"
                rsi_params = {
                    'function': 'RSI',
                    'symbol': symbol,
                    'interval': 'daily',
                    'time_period': 14,
                    'series_type': 'close',
                    'apikey': ALPHA_VANTAGE_API_KEY
                }
                
                response = self.session.get(rsi_url, params=rsi_params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'Technical Analysis: RSI' in data:
                        rsi_data = data['Technical Analysis: RSI']
                        latest_date = max(rsi_data.keys())
                        latest_rsi = float(rsi_data[latest_date]['RSI'])
                        
                        technical_data.append({
                            'symbol': symbol,
                            'date': latest_date,
                            'rsi_14': latest_rsi,
                            'source': 'alpha_vantage'
                        })
                        
                        logger.info(f"✅ Got RSI for {symbol}: {latest_rsi:.2f}")
                    else:
                        logger.warning(f"⚠️ No RSI data for {symbol}")
                
                time.sleep(15)  # Alpha Vantage: 5 calls per minute limit
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch technical indicators for {symbol}: {e}")
                continue
        
        if technical_data:
            df = pd.DataFrame(technical_data)
            tech_file = RAW_DIR / 'daily' / 'technical_indicators.csv'
            tech_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(tech_file, index=False)
            return df
        
        return pd.DataFrame()
    
    def extract_fundamentals(self, symbols: List[str]) -> pd.DataFrame:
        """Extract company fundamentals using Finnhub (global markets)"""
        if not FINNHUB_API_KEY:
            logger.warning("Finnhub API key missing - skipping fundamentals")
            return pd.DataFrame()
        
        fundamental_data = []
        
        for symbol in symbols:
            try:
                # Convert Indian symbols to Finnhub format
                finnhub_symbol = symbol.replace('.NS', '').replace('.BO', '')
                if '.NS' in symbol:
                    finnhub_symbol += '.BSE'  # Some Indian stocks need .BSE suffix
                
                logger.info(f"Fetching fundamentals for {symbol} (Finnhub: {finnhub_symbol})")
                
                # Company Profile
                profile_url = "https://finnhub.io/api/v1/stock/profile2"
                params = {
                    'symbol': finnhub_symbol,
                    'token': FINNHUB_API_KEY
                }
                
                response = self.session.get(profile_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    profile_data = response.json()
                    
                    if profile_data and 'name' in profile_data:
                        # Basic metrics
                        metrics_url = "https://finnhub.io/api/v1/stock/metric"
                        metrics_params = {
                            'symbol': finnhub_symbol,
                            'metric': 'all',
                            'token': FINNHUB_API_KEY
                        }
                        
                        metrics_response = self.session.get(metrics_url, params=metrics_params, timeout=30)
                        metrics_data = metrics_response.json() if metrics_response.status_code == 200 else {}
                        
                        fundamental_record = {
                            'symbol': symbol,
                            'finnhub_symbol': finnhub_symbol,
                            'company_name': profile_data.get('name', ''),
                            'country': profile_data.get('country', ''),
                            'currency': profile_data.get('currency', ''),
                            'exchange': profile_data.get('exchange', ''),
                            'ipo_date': profile_data.get('ipo', ''),
                            'market_cap': profile_data.get('marketCapitalization', 0),
                            'shares_outstanding': profile_data.get('shareOutstanding', 0),
                            'industry': profile_data.get('finnhubIndustry', ''),
                            'website': profile_data.get('weburl', ''),
                            'pe_ratio': metrics_data.get('metric', {}).get('peBasicExclExtraTTM', 0),
                            'beta': metrics_data.get('metric', {}).get('beta', 0),
                            'market_type': 'Indian' if '.NS' in symbol or '.BO' in symbol else 'US',
                            'last_updated': datetime.now().isoformat()
                        }
                        
                        fundamental_data.append(fundamental_record)
                        logger.info(f"✅ Got fundamentals for {symbol}: {fundamental_record['company_name']}")
                    else:
                        logger.warning(f"⚠️ No fundamental data for {symbol}")
                
                time.sleep(1.1)  # Finnhub rate limiting
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch fundamentals for {symbol}: {e}")
                continue
        
        if fundamental_data:
            df = pd.DataFrame(fundamental_data)
            
            # Save by market type
            us_fundamentals = df[df['market_type'] == 'US']
            indian_fundamentals = df[df['market_type'] == 'Indian']
            
            if not us_fundamentals.empty:
                us_fund_file = RAW_DIR / 'fundamental' / 'us_company_fundamentals.csv'
                us_fund_file.parent.mkdir(parents=True, exist_ok=True)
                us_fundamentals.to_csv(us_fund_file, index=False)
            
            if not indian_fundamentals.empty:
                indian_fund_file = RAW_DIR / 'fundamental' / 'indian_company_fundamentals.csv'
                indian_fund_file.parent.mkdir(parents=True, exist_ok=True)
                indian_fundamentals.to_csv(indian_fund_file, index=False)
            
            # Save combined
            fund_file = RAW_DIR / 'fundamental' / 'all_company_fundamentals.csv'
            fund_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(fund_file, index=False)
            
            return df
        
        return pd.DataFrame()
    
    def extract_news_sentiment(self, symbols: List[str]) -> pd.DataFrame:
        """Extract news sentiment using Finnhub (global)"""
        if not FINNHUB_API_KEY:
            logger.warning("Finnhub API key missing - skipping news sentiment")
            return pd.DataFrame()
        
        news_data = []
        
        for symbol in symbols:
            try:
                # Convert Indian symbols
                finnhub_symbol = symbol.replace('.NS', '').replace('.BO', '')
                
                logger.info(f"Fetching news sentiment for {symbol}")
                
                url = "https://finnhub.io/api/v1/news-sentiment"
                params = {
                    'symbol': finnhub_symbol,
                    'token': FINNHUB_API_KEY
                }
                
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data and 'buzz' in data:
                        sentiment_record = {
                            'symbol': symbol,
                            'finnhub_symbol': finnhub_symbol,
                            'buzz_articles_week': data.get('buzz', {}).get('articlesInLastWeek', 0),
                            'buzz_score': data.get('buzz', {}).get('buzz', 0),
                            'buzz_weekly_avg': data.get('buzz', {}).get('weeklyAverage', 0),
                            'company_news_score': data.get('companyNewsScore', 0),
                            'sector_avg_news_score': data.get('sectorAverageNewsScore', 0),
                            'bearish_percent': data.get('sentiment', {}).get('bearishPercent', 0),
                            'bullish_percent': data.get('sentiment', {}).get('bullishPercent', 0),
                            'market_type': 'Indian' if '.NS' in symbol or '.BO' in symbol else 'US',
                            'last_updated': datetime.now().isoformat()
                        }
                        
                        news_data.append(sentiment_record)
                        logger.info(f"✅ Got news sentiment for {symbol}: {sentiment_record['bullish_percent']:.1f}% bullish")
                    else:
                        logger.warning(f"⚠️ No sentiment data for {symbol}")
                
                time.sleep(1.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch news sentiment for {symbol}: {e}")
                continue
        
        if news_data:
            df = pd.DataFrame(news_data)
            news_file = RAW_DIR / 'fundamental' / 'news_sentiment.csv'
            news_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(news_file, index=False)
            return df
        
        return pd.DataFrame()
    
    def extract_intraday_data(self, symbols: List[str]) -> pd.DataFrame:
        """Extract intraday data using Polygon (US stocks only)"""
        if not POLYGON_API_KEY:
            logger.warning("Polygon API key missing - skipping intraday data")
            return pd.DataFrame()
        
        intraday_data = []
        today = datetime.now().strftime('%Y-%m-%d')
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching intraday data for {symbol}")
                
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/hour/{today}/{today}"
                params = {
                    'apikey': POLYGON_API_KEY
                }
                
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('results'):
                        for result in data['results']:
                            intraday_record = {
                                'symbol': symbol,
                                'timestamp': pd.to_datetime(result['t'], unit='ms'),
                                'open': result['o'],
                                'high': result['h'],
                                'low': result['l'],
                                'close': result['c'],
                                'volume': result['v'],
                                'source': 'polygon'
                            }
                            intraday_data.append(intraday_record)
                        
                        logger.info(f"✅ Got {len(data['results'])} intraday records for {symbol}")
                    else:
                        logger.warning(f"⚠️ No intraday data for {symbol}")
                
                time.sleep(13)  # Polygon: 5 calls per minute for free tier
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch intraday data for {symbol}: {e}")
                continue
        
        if intraday_data:
            df = pd.DataFrame(intraday_data)
            intraday_file = RAW_DIR / 'intraday' / f'us_intraday_{today}.csv'
            intraday_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(intraday_file, index=False)
            return df
        
        return pd.DataFrame()
    
    def extract_market_indices(self) -> pd.DataFrame:
        """Extract major market indices data"""
        indices_data = []
        
        # Major indices symbols
        indices = {
            # US Indices
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000',
            
            # Indian Indices  
            '^NSEI': 'NIFTY 50',
            '^BSESN': 'SENSEX',
            '^NSEBANK': 'BANK NIFTY'
        }
        
        for symbol, name in indices.items():
            try:
                logger.info(f"Fetching index data for {name} ({symbol})")
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d")  # Last 5 days
                
                if not data.empty:
                    latest = data.iloc[-1]
                    
                    indices_record = {
                        'symbol': symbol,
                        'name': name,
                        'current_price': latest['Close'],
                        'open': latest['Open'],
                        'high': latest['High'],
                        'low': latest['Low'],
                        'volume': latest['Volume'],
                        'date': latest.name.date(),
                        'market': 'Indian' if 'NSE' in symbol or 'BSE' in symbol else 'US',
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    indices_data.append(indices_record)
                    logger.info(f"✅ Got index data for {name}: {latest['Close']:.2f}")
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch index data for {name}: {e}")
                continue
        
        if indices_data:
            df = pd.DataFrame(indices_data)
            indices_file = RAW_DIR / 'daily' / 'market_indices.csv'
            indices_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(indices_file, index=False)
            return df
        
        return pd.DataFrame()
    
    def save_all_data(self, data_dict: Dict[str, pd.DataFrame], market_filter: str) -> None:
        """Save all extracted data with comprehensive metadata"""
        try:
            # Count records by market
            us_count = indian_count = 0
            for df in data_dict.values():
                if not df.empty and 'Market' in df.columns:
                    us_count += len(df[df['Market'] == 'US'])
                    indian_count += len(df[df['Market'] == 'Indian'])
                elif not df.empty and 'market_type' in df.columns:
                    us_count += len(df[df['market_type'] == 'US'])
                    indian_count += len(df[df['market_type'] == 'Indian'])
            
            metadata = {
                'extraction_timestamp': datetime.now().isoformat(),
                'market_filter': market_filter,
                'api_keys_used': self.api_status,
                'symbols_processed': {
                    'us_symbols': len([s for s in ALL_SYMBOLS if '.NS' not in s and '.BO' not in s]),
                    'indian_symbols': len([s for s in ALL_SYMBOLS if '.NS' in s or '.BO' in s]),
                    'total_symbols': len(ALL_SYMBOLS)
                },
                'data_summary': {},
                'market_breakdown': {
                    'us_records': us_count,
                    'indian_records': indian_count,
                    'total_records': us_count + indian_count
                }
            }
            
            total_records = 0
            for data_type, df in data_dict.items():
                if not df.empty:
                    records_count = len(df)
                    total_records += records_count
                    
                    metadata['data_summary'][data_type] = {
                        'records_count': records_count,
                        'columns': list(df.columns),
                        'file_saved': f"{data_type}_{datetime.now().strftime('%Y%m%d')}.csv"
                    }
                    logger.info(f"📊 {data_type}: {records_count} records extracted")
                else:
                    metadata['data_summary'][data_type] = {'records_count': 0, 'status': 'no_data'}
            
            metadata['total_records_extracted'] = total_records
            
            # Save metadata
            metadata_file = RAW_DIR / f'extraction_metadata_{market_filter}_{datetime.now().strftime("%Y%m%d")}.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Extraction metadata saved: {metadata_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {e}")

def create_enhanced_data_directories():
    """Create enhanced data directories for multi-market structure"""
    directories = [
        # Raw data - separated by market
        RAW_DIR / 'daily' / 'us',
        RAW_DIR / 'daily' / 'indian',
        RAW_DIR / 'intraday' / 'us',
        RAW_DIR / 'intraday' / 'indian',
        RAW_DIR / 'fundamental',
        
        # Processed data
        PROCESSED_DIR / 'features' / 'us',
        PROCESSED_DIR / 'features' / 'indian',
        PROCESSED_DIR / 'targets' / 'us',
        PROCESSED_DIR / 'targets' / 'indian',
        
        # Other directories
        DATA_ROOT / 'models' / 'us',
        DATA_ROOT / 'models' / 'indian',
        DATA_ROOT / 'predictions' / 'daily',
        DATA_ROOT / 'predictions' / 'signals',
        DATA_ROOT / 'backtests' / 'us',
        DATA_ROOT / 'backtests' / 'indian',
        DATA_ROOT / 'exports' / 'charts',
        DATA_ROOT / 'exports' / 'reports'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        init_file = directory / '__init__.py'
        if not init_file.exists():
            init_file.write_text('# StockPredictionPro Data Directory\n')
    
    logger.info("📁 Created enhanced multi-market data directories")

def run_data_extraction(market_filter: str = 'all'):
    """
    Main function to run enhanced data extraction
    
    Args:
        market_filter: 'all', 'us', 'indian' to filter markets
    """
    logger.info("🚀 Starting Enhanced StockPredictionPro data extraction...")
    
    print(f"\n{'='*70}")
    print("🌍 ENHANCED STOCKPREDICTIONPRO DATA EXTRACTION")
    print(f"{'='*70}")
    print(f"📊 US Stocks: {len(US_TOP_50)} symbols")
    print(f"🇮🇳 Indian Stocks: {len(INDIAN_TOP_50)} symbols")  
    print(f"🌐 Total Symbols: {len(ALL_SYMBOLS)} symbols")
    print(f"🎯 Market Filter: {market_filter.upper()}")
    print(f"{'='*70}")
    
    # Create directories
    create_enhanced_data_directories()
    
    # Initialize extractor
    extractor = EnhancedStockDataExtractor()
    
    # Extract data based on market filter
    extracted_data = extractor.extract_all_data(market_filter=market_filter)
    
    # Print comprehensive summary
    print(f"\n{'='*70}")
    print("📋 DATA EXTRACTION SUMMARY")
    print(f"{'='*70}")
    
    total_records = 0
    for data_type, df in extracted_data.items():
        if not df.empty:
            print(f"✅ {data_type.replace('_', ' ').title()}: {len(df):,} records")
            total_records += len(df)
        else:
            print(f"⚠️ {data_type.replace('_', ' ').title()}: No data")
    
    print(f"\n🎉 Total Records Extracted: {total_records:,}")
    print(f"📁 Data Location: {RAW_DIR}")
    print(f"⏱️ Extraction completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("✅ Multi-market data extraction completed successfully!")
    print(f"{'='*70}\n")

# Quick extraction functions
def extract_us_data():
    """Extract only US market data"""
    run_data_extraction(market_filter='us')

def extract_indian_data():
    """Extract only Indian market data"""
    run_data_extraction(market_filter='indian')

def extract_all_data():
    """Extract all market data"""
    run_data_extraction(market_filter='all')

if __name__ == "__main__":
    # Default: extract all data
    extract_all_data()
