"""
setup/setup_database.py

Comprehensive database setup script for StockPredictionPro.
Creates database schema, tables, indexes, and handles migrations.
Supports SQLite for development and PostgreSQL for production.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, 
    Boolean, ForeignKey, Index, UniqueConstraint, CheckConstraint,
    MetaData, Table, inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool

try:
    from sqlalchemy_utils import database_exists, create_database, drop_database
except ImportError:
    print("⚠️ sqlalchemy-utils not found. Install with: pip install sqlalchemy-utils")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockPredictionPro.DatabaseSetup')

# Database configuration
BASE_DIR = Path(__file__).parent.parent
DATABASE_CONFIG = {
    'development': f'sqlite:///{BASE_DIR}/data/stockpred_dev.db',
    'testing': f'sqlite:///{BASE_DIR}/data/stockpred_test.db',
    'production': os.getenv('DATABASE_URL', f'sqlite:///{BASE_DIR}/data/stockpred_prod.db')
}

# SQLAlchemy setup
Base = declarative_base()
metadata = MetaData()

# ============================================
# DATABASE MODELS
# ============================================

class MarketData(Base):
    """Raw market data (OHLCV) for stocks"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    adjusted_close = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='unique_symbol_date'),
        CheckConstraint('high_price >= low_price', name='check_high_low'),
        CheckConstraint('volume >= 0', name='check_volume_positive'),
        Index('idx_symbol_date', 'symbol', 'date'),
        Index('idx_date_symbol', 'date', 'symbol'),
    )
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', date='{self.date}', close={self.close_price})>"

class TechnicalIndicators(Base):
    """Technical indicators calculated from market data"""
    __tablename__ = 'technical_indicators'
    
    id = Column(Integer, primary_key=True)
    market_data_id = Column(Integer, ForeignKey('market_data.id', ondelete='CASCADE'), nullable=False)
    indicator_name = Column(String(64), nullable=False)
    indicator_value = Column(Float)
    calculation_params = Column(Text)  # JSON string of parameters
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    market_data = relationship("MarketData", backref="technical_indicators")
    
    __table_args__ = (
        Index('idx_market_indicator', 'market_data_id', 'indicator_name'),
    )

class FeatureStore(Base):
    """Engineered features for ML models"""
    __tablename__ = 'feature_store'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    feature_set_version = Column(String(16), nullable=False)
    features = Column(Text, nullable=False)  # JSON string of all features
    target_variable = Column(Float)  # Target for supervised learning
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', 'feature_set_version', name='unique_feature_entry'),
        Index('idx_symbol_date_version', 'symbol', 'date', 'feature_set_version'),
    )

class ModelRegistry(Base):
    """Registry of trained ML models"""
    __tablename__ = 'model_registry'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(32), nullable=False)
    model_type = Column(String(64), nullable=False)  # 'xgboost', 'lstm', etc.
    algorithm = Column(String(64))
    hyperparameters = Column(Text)  # JSON string
    training_data_version = Column(String(32))
    performance_metrics = Column(Text)  # JSON string
    model_path = Column(String(512))  # Path to serialized model
    is_active = Column(Boolean, default=False)
    trained_at = Column(DateTime, nullable=False)
    created_by = Column(String(64), default='system')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('model_name', 'model_version', name='unique_model_version'),
        Index('idx_model_active', 'model_name', 'is_active'),
    )

class Predictions(Base):
    """Model predictions and forecasts"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('model_registry.id', ondelete='CASCADE'), nullable=False)
    symbol = Column(String(32), nullable=False, index=True)
    prediction_date = Column(DateTime, nullable=False)  # When prediction was made
    target_date = Column(DateTime, nullable=False, index=True)  # Date being predicted
    prediction_value = Column(Float, nullable=False)
    confidence_score = Column(Float)
    prediction_interval_lower = Column(Float)
    prediction_interval_upper = Column(Float)
    actual_value = Column(Float)  # Filled in later for evaluation
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    model = relationship("ModelRegistry", backref="predictions")
    
    __table_args__ = (
        Index('idx_symbol_target_date', 'symbol', 'target_date'),
        Index('idx_model_prediction_date', 'model_id', 'prediction_date'),
    )

class TradingSignals(Base):
    """Trading signals generated from predictions"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(32), nullable=False, index=True)
    signal_date = Column(DateTime, nullable=False, index=True)
    signal_type = Column(String(16), nullable=False)  # 'BUY', 'SELL', 'HOLD'
    signal_strength = Column(Float)  # 0.0 to 1.0
    price_target = Column(Float)
    stop_loss = Column(Float)
    risk_score = Column(Float)
    strategy_name = Column(String(128))
    model_combination = Column(Text)  # JSON array of contributing models
    executed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        CheckConstraint("signal_type IN ('BUY', 'SELL', 'HOLD')", name='check_signal_type'),
        Index('idx_symbol_signal_date', 'symbol', 'signal_date'),
    )

class BacktestResults(Base):
    """Results from strategy backtesting"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    backtest_name = Column(String(128), nullable=False)
    strategy_name = Column(String(128), nullable=False)
    symbol = Column(String(32), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    total_trades = Column(Integer)
    avg_trade_return = Column(Float)
    configuration = Column(Text)  # JSON string of strategy parameters
    detailed_results = Column(Text)  # JSON string of detailed results
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_backtest_strategy', 'backtest_name', 'strategy_name'),
    )

class SystemMetrics(Base):
    """System performance and health metrics"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_name = Column(String(128), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(32))
    component = Column(String(64))  # 'database', 'model_training', etc.
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    tags = Column(Text)  # JSON string of additional tags
    
    __table_args__ = (
        Index('idx_metric_timestamp', 'metric_name', 'timestamp'),
    )

class AuditLog(Base):
    """Audit trail for system operations"""
    __tablename__ = 'audit_log'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    user_id = Column(String(64))
    operation = Column(String(128), nullable=False)
    table_name = Column(String(64))
    record_id = Column(String(64))
    old_values = Column(Text)  # JSON string
    new_values = Column(Text)  # JSON string
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    __table_args__ = (
        Index('idx_audit_operation_timestamp', 'operation', 'timestamp'),
    )

# ============================================
# DATABASE SETUP FUNCTIONS
# ============================================

class DatabaseManager:
    """Comprehensive database management utility"""
    
    def __init__(self, environment: str = 'development'):
        self.environment = environment
        self.database_url = DATABASE_CONFIG.get(environment, DATABASE_CONFIG['development'])
        self.engine = None
        self.session_factory = None
        
    def create_engine(self, echo: bool = False) -> sa.engine.Engine:
        """Create SQLAlchemy engine with appropriate settings"""
        if 'sqlite' in self.database_url:
            # SQLite specific settings
            self.engine = create_engine(
                self.database_url,
                echo=echo,
                poolclass=StaticPool,
                connect_args={
                    'check_same_thread': False,
                    'timeout': 30
                }
            )
        else:
            # PostgreSQL/other database settings
            self.engine = create_engine(
                self.database_url,
                echo=echo,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
        
        logger.info(f"✅ Database engine created for {self.environment}")
        return self.engine
    
    def ensure_database_exists(self) -> bool:
        """Ensure database exists, create if it doesn't"""
        try:
            if not database_exists(self.engine.url):
                logger.info(f"📝 Creating database: {self.database_url}")
                create_database(self.engine.url)
                logger.info("✅ Database created successfully")
                return True
            else:
                logger.info(f"✅ Database already exists: {self.database_url}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to create database: {e}")
            return False
    
    def create_tables(self, drop_existing: bool = False) -> bool:
        """Create all tables defined in Base metadata"""
        try:
            if drop_existing:
                logger.warning("⚠️ Dropping existing tables...")
                Base.metadata.drop_all(self.engine)
                logger.info("✅ Existing tables dropped")
            
            logger.info("📊 Creating database tables...")
            Base.metadata.create_all(self.engine)
            logger.info("✅ All tables created successfully")
            
            # Verify tables were created
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            logger.info(f"📋 Tables in database: {', '.join(tables)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            return False
    
    def create_indexes(self) -> bool:
        """Create additional performance indexes"""
        try:
            with self.engine.connect() as conn:
                # Additional custom indexes for performance
                additional_indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_market_data_volume ON market_data(volume) WHERE volume > 0",
                    "CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON predictions(confidence_score) WHERE confidence_score > 0.5",
                    "CREATE INDEX IF NOT EXISTS idx_signals_strength ON trading_signals(signal_strength) WHERE signal_strength > 0.7",
                ]
                
                for index_sql in additional_indexes:
                    try:
                        conn.execute(sa.text(index_sql))
                        conn.commit()
                    except Exception as e:
                        logger.debug(f"Index creation skipped (may already exist): {e}")
                
            logger.info("✅ Additional indexes created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create additional indexes: {e}")
            return False
    
    def get_session(self) -> Session:
        """Get database session"""
        if not self.session_factory:
            self.session_factory = sessionmaker(bind=self.engine)
        return self.session_factory()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sa.text("SELECT 1"))
                result.fetchone()
            logger.info("✅ Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information"""
        try:
            inspector = inspect(self.engine)
            
            info = {
                'database_url': self.database_url,
                'environment': self.environment,
                'tables': inspector.get_table_names(),
                'total_tables': len(inspector.get_table_names()),
                'database_exists': database_exists(self.engine.url),
                'connection_test': self.test_connection()
            }
            
            # Get table row counts
            table_counts = {}
            try:
                with self.engine.connect() as conn:
                    for table_name in info['tables']:
                        result = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table_name}"))
                        count = result.scalar()
                        table_counts[table_name] = count
                info['table_row_counts'] = table_counts
            except Exception as e:
                logger.debug(f"Could not get table counts: {e}")
                info['table_row_counts'] = {}
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Failed to get database info: {e}")
            return {'error': str(e)}
    
    def seed_initial_data(self) -> bool:
        """Insert initial/sample data if needed"""
        try:
            session = self.get_session()
            
            # Check if we need to seed data
            existing_models = session.query(ModelRegistry).count()
            if existing_models > 0:
                logger.info("🌱 Database already has data, skipping seeding")
                session.close()
                return True
            
            # Seed some initial model registry entries
            initial_models = [
                ModelRegistry(
                    model_name="XGBoost_Baseline",
                    model_version="1.0.0",
                    model_type="xgboost",
                    algorithm="XGBRegressor",
                    hyperparameters='{"n_estimators": 100, "learning_rate": 0.1}',
                    training_data_version="1.0",
                    performance_metrics='{"rmse": 0.05, "r2": 0.85}',
                    trained_at=datetime.utcnow(),
                    is_active=False
                ),
                ModelRegistry(
                    model_name="LSTM_Advanced",
                    model_version="1.0.0",
                    model_type="lstm",
                    algorithm="JAX_LSTM",
                    hyperparameters='{"hidden_size": 64, "num_layers": 2}',
                    training_data_version="1.0",
                    performance_metrics='{"rmse": 0.048, "r2": 0.87}',
                    trained_at=datetime.utcnow(),
                    is_active=True
                )
            ]
            
            session.add_all(initial_models)
            session.commit()
            session.close()
            
            logger.info("🌱 Initial data seeded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to seed initial data: {e}")
            return False

def setup_database(environment: str = 'development', 
                  drop_existing: bool = False,
                  seed_data: bool = True) -> bool:
    """
    Main database setup function
    
    Args:
        environment: Database environment (development, testing, production)
        drop_existing: Whether to drop existing tables
        seed_data: Whether to insert initial sample data
        
    Returns:
        True if setup successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STOCKPREDICTIONPRO DATABASE SETUP")
    logger.info("=" * 60)
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(environment)
        
        # Create engine
        db_manager.create_engine(echo=False)
        
        # Ensure database exists
        if not db_manager.ensure_database_exists():
            return False
        
        # Create tables
        if not db_manager.create_tables(drop_existing=drop_existing):
            return False
        
        # Create additional indexes
        if not db_manager.create_indexes():
            logger.warning("⚠️ Some indexes could not be created, but continuing...")
        
        # Test connection
        if not db_manager.test_connection():
            return False
        
        # Seed initial data if requested
        if seed_data:
            if not db_manager.seed_initial_data():
                logger.warning("⚠️ Could not seed initial data, but database setup complete")
        
        # Print database information
        db_info = db_manager.get_database_info()
        logger.info("📊 DATABASE SETUP SUMMARY:")
        logger.info(f"   • Environment: {db_info.get('environment', 'unknown')}")
        logger.info(f"   • Total tables: {db_info.get('total_tables', 0)}")
        logger.info(f"   • Tables: {', '.join(db_info.get('tables', []))}")
        
        if 'table_row_counts' in db_info:
            total_rows = sum(db_info['table_row_counts'].values())
            logger.info(f"   • Total rows: {total_rows}")
        
        logger.info("✅ Database setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup StockPredictionPro Database')
    parser.add_argument('--environment', '-e', 
                       choices=['development', 'testing', 'production'],
                       default='development',
                       help='Database environment')
    parser.add_argument('--drop-existing', '-d', 
                       action='store_true',
                       help='Drop existing tables before creating')
    parser.add_argument('--no-seed', 
                       action='store_true',
                       help='Skip seeding initial data')
    parser.add_argument('--info', '-i',
                       action='store_true',
                       help='Show database information only')
    
    args = parser.parse_args()
    
    if args.info:
        # Just show database info
        db_manager = DatabaseManager(args.environment)
        db_manager.create_engine()
        
        if database_exists(db_manager.engine.url):
            info = db_manager.get_database_info()
            print("\n" + "="*50)
            print("DATABASE INFORMATION")
            print("="*50)
            print(json.dumps(info, indent=2, default=str))
        else:
            print(f"❌ Database does not exist at: {db_manager.database_url}")
        return
    
    # Setup database
    success = setup_database(
        environment=args.environment,
        drop_existing=args.drop_existing,
        seed_data=not args.no_seed
    )
    
    if success:
        logger.info("🎉 Database setup completed successfully!")
        exit(0)
    else:
        logger.error("❌ Database setup failed!")
        exit(1)

if __name__ == '__main__':
    main()
