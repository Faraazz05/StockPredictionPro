"""
data/clean_data.py

Advanced data cleaning and preprocessing pipeline for StockPredictionPro.
Comprehensive data quality enhancement, outlier handling, feature engineering, and transformation.
Supports multiple cleaning strategies with detailed logging and validation.

Author: StockPredictionPro Team
Date: August 2025
Python Version: 3.13.7 Compatible
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    PowerTransformer, QuantileTransformer
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockPredictionPro.DataCleaner')

# ============================================
# CONFIGURATION AND DATA MODELS
# ============================================

@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations"""
    # Missing value handling
    missing_value_threshold: float = 0.5  # Drop columns with >50% missing
    imputation_strategy: str = 'iterative'  # simple, knn, iterative
    
    # Outlier detection
    outlier_method: str = 'iqr'  # zscore, iqr, isolation_forest
    outlier_threshold: float = 3.0  # For z-score method
    outlier_quantile_range: Tuple[float, float] = (0.01, 0.99)  # Keep 1%-99%
    
    # Feature scaling
    scaling_method: str = 'robust'  # standard, minmax, robust, power, quantile
    
    # Data validation
    validate_ohlc: bool = True  # Validate OHLC relationships
    min_price: float = 0.01  # Minimum valid price
    max_price: float = 50000.0  # Maximum reasonable price
    min_volume: int = 0  # Minimum volume
    
    # Feature engineering
    create_returns: bool = True
    create_log_returns: bool = True
    create_volatility: bool = True
    volatility_window: int = 20
    create_moving_averages: bool = True
    ma_windows: List[int] = None
    
    def __post_init__(self):
        if self.ma_windows is None:
            self.ma_windows = [5, 10, 20, 50]

@dataclass
class CleaningReport:
    """Report of cleaning operations performed"""
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    rows_removed: int
    columns_removed: int
    missing_values_imputed: Dict[str, int]
    outliers_removed: Dict[str, int]
    features_created: List[str]
    processing_time: float
    cleaning_config: CleaningConfig

# ============================================
# ADVANCED DATA CLEANER
# ============================================

class AdvancedDataCleaner:
    """Comprehensive data cleaning and preprocessing system"""
    
    def __init__(self, config: CleaningConfig = None):
        self.config = config or CleaningConfig()
        self.original_df = None
        self.df = None
        self.cleaning_report = None
        self.scalers = {}
        self.imputers = {}
        
        # Processing statistics
        self.stats = {
            'rows_removed': 0,
            'columns_removed': 0,
            'missing_imputed': {},
            'outliers_removed': {},
            'features_created': []
        }
    
    def clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Main cleaning pipeline
        
        Args:
            df: Input DataFrame to clean
            
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        logger.info("🧹 Starting comprehensive data cleaning pipeline...")
        start_time = datetime.now()
        
        # Store original data
        self.original_df = df.copy()
        self.df = df.copy()
        original_shape = self.df.shape
        
        # Reset statistics
        self.stats = {
            'rows_removed': 0,
            'columns_removed': 0,
            'missing_imputed': {},
            'outliers_removed': {},
            'features_created': []
        }
        
        # Execute cleaning pipeline
        self._validate_input_data()
        self._handle_missing_values()
        self._detect_and_handle_outliers()
        self._validate_business_rules()
        self._create_derived_features()
        self._apply_transformations()
        
        # Generate report
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        self.cleaning_report = CleaningReport(
            original_shape=original_shape,
            final_shape=self.df.shape,
            rows_removed=self.stats['rows_removed'],
            columns_removed=self.stats['columns_removed'],
            missing_values_imputed=self.stats['missing_imputed'],
            outliers_removed=self.stats['outliers_removed'],
            features_created=self.stats['features_created'],
            processing_time=processing_time,
            cleaning_config=self.config
        )
        
        logger.info(f"✅ Data cleaning completed in {processing_time:.2f} seconds")
        logger.info(f"📊 Shape changed from {original_shape} to {self.df.shape}")
        
        return self.df, self.cleaning_report
    
    def _validate_input_data(self) -> None:
        """Validate input data structure and content"""
        logger.info("🔍 Validating input data structure...")
        
        if self.df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Check for required columns (flexible for different data sources)
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            logger.warning("No numeric columns found in DataFrame")
        
        # Remove completely empty columns
        empty_columns = self.df.columns[self.df.isnull().all()].tolist()
        if empty_columns:
            self.df = self.df.drop(columns=empty_columns)
            self.stats['columns_removed'] += len(empty_columns)
            logger.info(f"Removed {len(empty_columns)} completely empty columns: {empty_columns}")
        
        # Convert object columns to numeric if possible
        for col in self.df.select_dtypes(include=['object']).columns:
            if col not in ['symbol', 'date', 'timestamp']:  # Skip known non-numeric columns
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    logger.info(f"Converted column '{col}' to numeric")
                except:
                    pass
    
    def _handle_missing_values(self) -> None:
        """Comprehensive missing value handling"""
        logger.info("📊 Handling missing values...")
        
        # Calculate missing value ratios
        missing_ratios = self.df.isnull().mean()
        
        # Drop columns with too many missing values
        columns_to_drop = missing_ratios[missing_ratios > self.config.missing_value_threshold].index.tolist()
        if columns_to_drop:
            self.df = self.df.drop(columns=columns_to_drop)
            self.stats['columns_removed'] += len(columns_to_drop)
            logger.info(f"Dropped {len(columns_to_drop)} columns with >{self.config.missing_value_threshold:.0%} missing values")
        
        # Impute remaining missing values
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        columns_with_missing = [col for col in numeric_columns if self.df[col].isnull().any()]
        
        if columns_with_missing:
            if self.config.imputation_strategy == 'simple':
                self._simple_imputation(columns_with_missing)
            elif self.config.imputation_strategy == 'knn':
                self._knn_imputation(columns_with_missing)
            elif self.config.imputation_strategy == 'iterative':
                self._iterative_imputation(columns_with_missing)
            else:
                logger.warning(f"Unknown imputation strategy: {self.config.imputation_strategy}")
                self._simple_imputation(columns_with_missing)
    
    def _simple_imputation(self, columns: List[str]) -> None:
        """Simple imputation using median/mode"""
        for col in columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                if self.df[col].dtype in ['float64', 'int64']:
                    fill_value = self.df[col].median()
                    strategy = 'median'
                else:
                    fill_value = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 0
                    strategy = 'mode'
                
                self.df[col].fillna(fill_value, inplace=True)
                self.stats['missing_imputed'][col] = missing_count
                logger.info(f"Imputed {missing_count} missing values in '{col}' using {strategy}")
    
    def _knn_imputation(self, columns: List[str]) -> None:
        """KNN-based imputation"""
        try:
            imputer = KNNImputer(n_neighbors=5)
            self.df[columns] = imputer.fit_transform(self.df[columns])
            self.imputers['knn'] = imputer
            
            total_imputed = sum(self.original_df[columns].isnull().sum())
            logger.info(f"KNN imputation completed: {total_imputed} values imputed across {len(columns)} columns")
            
            for col in columns:
                original_missing = self.original_df[col].isnull().sum()
                if original_missing > 0:
                    self.stats['missing_imputed'][col] = original_missing
                    
        except Exception as e:
            logger.error(f"KNN imputation failed: {e}. Falling back to simple imputation.")
            self._simple_imputation(columns)
    
    def _iterative_imputation(self, columns: List[str]) -> None:
        """Iterative imputation using regression"""
        try:
            imputer = IterativeImputer(
                max_iter=10,
                random_state=42,
                initial_strategy='median'
            )
            self.df[columns] = imputer.fit_transform(self.df[columns])
            self.imputers['iterative'] = imputer
            
            total_imputed = sum(self.original_df[columns].isnull().sum())
            logger.info(f"Iterative imputation completed: {total_imputed} values imputed across {len(columns)} columns")
            
            for col in columns:
                original_missing = self.original_df[col].isnull().sum()
                if original_missing > 0:
                    self.stats['missing_imputed'][col] = original_missing
                    
        except Exception as e:
            logger.error(f"Iterative imputation failed: {e}. Falling back to KNN imputation.")
            self._knn_imputation(columns)
    
    def _detect_and_handle_outliers(self) -> None:
        """Detect and handle outliers using multiple methods"""
        logger.info("🎯 Detecting and handling outliers...")
        
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['symbol', 'timestamp', 'date']:  # Skip non-price/volume columns
                continue
                
            original_length = len(self.df)
            
            if self.config.outlier_method == 'zscore':
                self._zscore_outlier_removal(col)
            elif self.config.outlier_method == 'iqr':
                self._iqr_outlier_removal(col)
            elif self.config.outlier_method == 'quantile':
                self._quantile_outlier_removal(col)
            else:
                logger.warning(f"Unknown outlier method: {self.config.outlier_method}")
                continue
            
            outliers_removed = original_length - len(self.df)
            if outliers_removed > 0:
                self.stats['outliers_removed'][col] = outliers_removed
                self.stats['rows_removed'] += outliers_removed
    
    def _zscore_outlier_removal(self, column: str) -> None:
        """Remove outliers using Z-score method"""
        series = self.df[column]
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        outlier_mask = z_scores > self.config.outlier_threshold
        outliers_count = outlier_mask.sum()
        
        if outliers_count > 0:
            self.df = self.df[~outlier_mask]
            logger.info(f"Removed {outliers_count} Z-score outliers from '{column}'")
    
    def _iqr_outlier_removal(self, column: str) -> None:
        """Remove outliers using IQR method"""
        series = self.df[column]
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outliers_count = outlier_mask.sum()
        
        if outliers_count > 0:
            self.df = self.df[~outlier_mask]
            logger.info(f"Removed {outliers_count} IQR outliers from '{column}'")
    
    def _quantile_outlier_removal(self, column: str) -> None:
        """Remove outliers using quantile method"""
        series = self.df[column]
        lower_quantile, upper_quantile = self.config.outlier_quantile_range
        
        lower_bound = series.quantile(lower_quantile)
        upper_bound = series.quantile(upper_quantile)
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outliers_count = outlier_mask.sum()
        
        if outliers_count > 0:
            self.df = self.df[~outlier_mask]
            logger.info(f"Removed {outliers_count} quantile outliers from '{column}' (keeping {lower_quantile:.1%}-{upper_quantile:.1%})")
    
    def _validate_business_rules(self) -> None:
        """Validate financial market business rules"""
        logger.info("💼 Validating business rules...")
        
        # Price validation
        price_columns = ['open', 'high', 'low', 'close', 'adjusted_close']
        for col in price_columns:
            if col in self.df.columns:
                # Remove rows with invalid prices
                invalid_prices = (
                    (self.df[col] < self.config.min_price) | 
                    (self.df[col] > self.config.max_price) |
                    (self.df[col] <= 0)
                ) & self.df[col].notna()
                
                if invalid_prices.any():
                    invalid_count = invalid_prices.sum()
                    self.df = self.df[~invalid_prices]
                    self.stats['rows_removed'] += invalid_count
                    logger.info(f"Removed {invalid_count} rows with invalid prices in '{col}'")
        
        # Volume validation
        if 'volume' in self.df.columns:
            invalid_volume = (self.df['volume'] < self.config.min_volume) & self.df['volume'].notna()
            if invalid_volume.any():
                invalid_count = invalid_volume.sum()
                self.df = self.df[~invalid_volume]
                self.stats['rows_removed'] += invalid_count
                logger.info(f"Removed {invalid_count} rows with invalid volume")
        
        # OHLC relationship validation
        if self.config.validate_ohlc and all(col in self.df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (self.df['high'] < self.df[['open', 'low', 'close']].max(axis=1)) |
                (self.df['low'] > self.df[['open', 'high', 'close']].min(axis=1))
            )
            
            if invalid_ohlc.any():
                invalid_count = invalid_ohlc.sum()
                self.df = self.df[~invalid_ohlc]
                self.stats['rows_removed'] += invalid_count
                logger.info(f"Removed {invalid_count} rows with invalid OHLC relationships")
    
    def _create_derived_features(self) -> None:
        """Create derived features for financial analysis"""
        logger.info("🔧 Creating derived features...")
        
        if 'close' not in self.df.columns:
            logger.warning("'close' column not found. Skipping feature engineering.")
            return
        
        # Returns
        if self.config.create_returns:
            self.df['returns'] = self.df['close'].pct_change()
            self.stats['features_created'].append('returns')
            logger.info("Created 'returns' feature")
        
        # Log returns
        if self.config.create_log_returns:
            self.df['log_returns'] = np.log(self.df['close'] / self.df['close'].shift(1))
            self.stats['features_created'].append('log_returns')
            logger.info("Created 'log_returns' feature")
        
        # Volatility
        if self.config.create_volatility:
            if 'returns' in self.df.columns:
                self.df['volatility'] = self.df['returns'].rolling(window=self.config.volatility_window).std()
                self.stats['features_created'].append('volatility')
                logger.info(f"Created 'volatility' feature with {self.config.volatility_window}-period window")
        
        # Moving averages
        if self.config.create_moving_averages:
            for window in self.config.ma_windows:
                ma_col = f'ma_{window}'
                self.df[ma_col] = self.df['close'].rolling(window=window).mean()
                self.stats['features_created'].append(ma_col)
                logger.info(f"Created '{ma_col}' moving average")
        
        # Price ratios (if OHLC available)
        if all(col in self.df.columns for col in ['open', 'high', 'low', 'close']):
            self.df['price_range'] = (self.df['high'] - self.df['low']) / self.df['close']
            self.df['open_close_ratio'] = self.df['open'] / self.df['close']
            self.stats['features_created'].extend(['price_range', 'open_close_ratio'])
            logger.info("Created price ratio features")
    
    def _apply_transformations(self) -> None:
        """Apply scaling and transformations to numeric features"""
        logger.info("📐 Applying feature transformations...")
        
        # Select numeric columns for scaling (exclude derived time features)
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        columns_to_scale = [col for col in numeric_columns 
                           if col not in ['symbol', 'timestamp', 'date'] and not col.startswith('ma_')]
        
        if not columns_to_scale:
            logger.warning("No numeric columns found for scaling")
            return
        
        # Remove any remaining infinite or extremely large values
        self.df[columns_to_scale] = self.df[columns_to_scale].replace([np.inf, -np.inf], np.nan)
        
        # Apply scaling
        if self.config.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.config.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.config.scaling_method == 'robust':
            scaler = RobustScaler()
        elif self.config.scaling_method == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
        elif self.config.scaling_method == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal')
        else:
            logger.warning(f"Unknown scaling method: {self.config.scaling_method}")
            return
        
        try:
            # Handle any remaining NaN values before scaling
            if self.df[columns_to_scale].isnull().any().any():
                logger.warning("Found NaN values before scaling, filling with median")
                for col in columns_to_scale:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
            
            self.df[columns_to_scale] = scaler.fit_transform(self.df[columns_to_scale])
            self.scalers[self.config.scaling_method] = scaler
            logger.info(f"Applied {self.config.scaling_method} scaling to {len(columns_to_scale)} columns")
            
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
    
    def save_cleaned_data(self, output_path: str, include_report: bool = True) -> None:
        """Save cleaned data and optionally the cleaning report"""
        if self.df is None:
            raise ValueError("No cleaned data available. Run clean_dataframe() first.")
        
        # Save cleaned data
        self.df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved cleaned data to: {output_path}")
        
        # Save cleaning report
        if include_report and self.cleaning_report:
            report_path = output_path.replace('.csv', '_cleaning_report.json')
            with open(report_path, 'w') as f:
                json.dump(asdict(self.cleaning_report), f, indent=2, default=str)
            logger.info(f"📊 Saved cleaning report to: {report_path}")
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get summary of cleaning operations"""
        if not self.cleaning_report:
            return {"error": "No cleaning report available"}
        
        return {
            "original_shape": self.cleaning_report.original_shape,
            "final_shape": self.cleaning_report.final_shape,
            "data_reduction": {
                "rows_removed": self.cleaning_report.rows_removed,
                "columns_removed": self.cleaning_report.columns_removed,
                "reduction_percentage": (
                    (self.cleaning_report.original_shape[0] - self.cleaning_report.final_shape[0]) / 
                    self.cleaning_report.original_shape[0] * 100
                )
            },
            "missing_values_handled": len(self.cleaning_report.missing_values_imputed),
            "outliers_removed": sum(self.cleaning_report.outliers_removed.values()),
            "features_created": len(self.cleaning_report.features_created),
            "processing_time": self.cleaning_report.processing_time
        }

def clean_csv_file(input_path: str, output_path: str, config: CleaningConfig = None) -> Dict[str, Any]:
    """
    Clean a CSV file with market data
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        config: Cleaning configuration
        
    Returns:
        Cleaning summary dictionary
    """
    try:
        # Load data
        df = pd.read_csv(input_path)
        logger.info(f"📂 Loaded data from {input_path}: {df.shape}")
        
        # Initialize cleaner
        cleaner = AdvancedDataCleaner(config or CleaningConfig())
        
        # Clean data
        cleaned_df, report = cleaner.clean_dataframe(df)
        
        # Save results
        cleaner.save_cleaned_data(output_path, include_report=True)
        
        return cleaner.get_cleaning_summary()
        
    except Exception as e:
        logger.error(f"Failed to clean CSV file {input_path}: {e}")
        return {"error": str(e)}

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean market data for StockPredictionPro')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output CSV file path')
    parser.add_argument('--imputation', choices=['simple', 'knn', 'iterative'], 
                       default='iterative', help='Imputation strategy')
    parser.add_argument('--outlier-method', choices=['zscore', 'iqr', 'quantile'], 
                       default='iqr', help='Outlier detection method')
    parser.add_argument('--scaling', choices=['standard', 'minmax', 'robust', 'power', 'quantile'],
                       default='robust', help='Feature scaling method')
    parser.add_argument('--no-features', action='store_true', help='Skip feature engineering')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = CleaningConfig(
        imputation_strategy=args.imputation,
        outlier_method=args.outlier_method,
        scaling_method=args.scaling,
        create_returns=not args.no_features,
        create_log_returns=not args.no_features,
        create_volatility=not args.no_features,
        create_moving_averages=not args.no_features
    )
    
    # Run cleaning
    logger.info(f"🚀 Starting data cleaning: {args.input_file} → {args.output_file}")
    summary = clean_csv_file(args.input_file, args.output_file, config)
    
    # Print summary
    if "error" not in summary:
        print("\n" + "="*50)
        print("DATA CLEANING SUMMARY")
        print("="*50)
        print(f"Original shape: {summary['original_shape']}")
        print(f"Final shape: {summary['final_shape']}")
        print(f"Rows removed: {summary['data_reduction']['rows_removed']}")
        print(f"Columns removed: {summary['data_reduction']['columns_removed']}")
        print(f"Data reduction: {summary['data_reduction']['reduction_percentage']:.1f}%")
        print(f"Features created: {summary['features_created']}")
        print(f"Processing time: {summary['processing_time']:.2f} seconds")
        print("\n✅ Data cleaning completed successfully!")
    else:
        print(f"❌ Data cleaning failed: {summary['error']}")
        exit(1)

if __name__ == '__main__':
    main()
