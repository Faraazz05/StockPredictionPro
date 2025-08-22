# ============================================
# StockPredictionPro - src/data/processors/resampler.py
# Advanced data resampling for financial time series with market awareness
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from datetime import datetime, timedelta, time
import warnings
from functools import partial

from ...utils.exceptions import DataValidationError, BusinessLogicError, InvalidParameterError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ...utils.config_loader import get
from ...utils.helpers import safe_divide, validate_numeric_range

logger = get_logger('data.processors.resampler')

# ============================================
# Financial Time Series Resampler
# ============================================

class FinancialDataResampler:
    """
    Advanced resampling for financial time series data
    
    Features:
    - OHLCV-aware resampling
    - Multiple aggregation methods
    - Market calendar awareness
    - Volume-weighted calculations
    - Trading session alignment
    """
    
    def __init__(self,
                 target_frequency: str = '1H',
                 ohlcv_mapping: Optional[Dict[str, str]] = None,
                 volume_weighted: bool = True,
                 trading_calendar: Optional[str] = None,
                 session_alignment: bool = True,
                 preserve_gaps: bool = True):
        """
        Initialize financial data resampler
        
        Args:
            target_frequency: Target frequency ('1T', '5T', '1H', '1D', etc.)
            ohlcv_mapping: Custom mapping for OHLCV columns
            volume_weighted: Whether to use volume-weighted calculations
            trading_calendar: Trading calendar to use ('NYSE', 'NASDAQ', 'NSE')
            session_alignment: Whether to align with trading sessions
            preserve_gaps: Whether to preserve gaps in non-trading periods
        """
        self.target_frequency = target_frequency
        self.volume_weighted = volume_weighted
        self.trading_calendar = trading_calendar
        self.session_alignment = session_alignment
        self.preserve_gaps = preserve_gaps
        
        # Default OHLCV column mapping
        self.ohlcv_mapping = ohlcv_mapping or {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Adj Close': 'last'
        }
        
        # Custom aggregation functions
        self.custom_agg_funcs = self._create_custom_aggregations()
        
        # Trading session definitions
        self.trading_sessions = {
            'NYSE': {'start': time(9, 30), 'end': time(16, 0)},
            'NASDAQ': {'start': time(9, 30), 'end': time(16, 0)},
            'NSE': {'start': time(9, 15), 'end': time(15, 30)},
            'LSE': {'start': time(8, 0), 'end': time(16, 30)},
            'TSE': {'start': time(9, 0), 'end': time(15, 0)}
        }
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate initialization parameters"""
        # Validate frequency format
        try:
            pd.Timedelta(self.target_frequency)
        except ValueError:
            raise InvalidParameterError(f"Invalid target frequency: {self.target_frequency}")
        
        if self.trading_calendar and self.trading_calendar not in self.trading_sessions:
            logger.warning(f"Unknown trading calendar: {self.trading_calendar}. Available: {list(self.trading_sessions.keys())}")
    
    @time_it("financial_data_resample")
    def resample(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Resample financial data to target frequency
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol for logging context
            
        Returns:
            Resampled DataFrame
        """
        if symbol:
            logger.info(f"Resampling {symbol} data to {self.target_frequency}")
        else:
            logger.info(f"Resampling data to {self.target_frequency}")
        
        # Validate input data
        self._validate_input_data(df)
        
        # Prepare data for resampling
        df_prepared = self._prepare_data_for_resampling(df)
        
        # Apply resampling strategy
        df_resampled = self._apply_resampling_strategy(df_prepared)
        
        # Post-process resampled data
        df_final = self._post_process_resampled_data(df_resampled, df)
        
        # Validate output
        self._validate_output_data(df_final, df)
        
        logger.info(f"Resampling complete: {len(df)} -> {len(df_final)} records")
        return df_final
    
    def _validate_input_data(self, df: pd.DataFrame):
        """Validate input data format"""
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise DataValidationError("Input DataFrame is empty")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError("DataFrame must have DatetimeIndex")
        
        # Check for basic OHLCV columns
        basic_columns = ['Open', 'High', 'Low', 'Close']
        missing_basic = [col for col in basic_columns if col not in df.columns]
        
        if len(missing_basic) == len(basic_columns):
            logger.warning("No standard OHLCV columns found - using generic resampling")
        elif missing_basic:
            logger.warning(f"Missing OHLCV columns: {missing_basic}")
    
    def _prepare_data_for_resampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for resampling"""
        df_prepared = df.copy()
        
        # Ensure data is sorted by time
        if not df_prepared.index.is_monotonic_increasing:
            df_prepared = df_prepared.sort_index()
            logger.info("Sorted data by timestamp")
        
        # Handle timezone issues
        if df_prepared.index.tz is not None:
            # Convert to UTC for consistent resampling, then localize back
            original_tz = df_prepared.index.tz
            df_prepared.index = df_prepared.index.tz_convert('UTC')
            logger.debug("Converted to UTC for resampling")
        
        # Add derived columns for better resampling
        if self.volume_weighted and 'Volume' in df_prepared.columns and 'Close' in df_prepared.columns:
            # Volume-weighted price for better aggregation
            df_prepared['Volume_Price'] = df_prepared['Close'] * df_prepared['Volume']
        
        # Add typical price for better aggregation
        if all(col in df_prepared.columns for col in ['High', 'Low', 'Close']):
            df_prepared['Typical_Price'] = (df_prepared['High'] + df_prepared['Low'] + df_prepared['Close']) / 3
            
            if 'Volume' in df_prepared.columns:
                df_prepared['Volume_Typical'] = df_prepared['Typical_Price'] * df_prepared['Volume']
        
        return df_prepared
    
    def _apply_resampling_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the main resampling strategy"""
        
        # Create aggregation dictionary
        agg_dict = self._create_aggregation_dict(df)
        
        # Apply resampling
        if self.session_alignment and self.trading_calendar:
            df_resampled = self._session_aligned_resample(df, agg_dict)
        else:
            df_resampled = self._standard_resample(df, agg_dict)
        
        return df_resampled
    
    def _standard_resample(self, df: pd.DataFrame, agg_dict: Dict[str, Any]) -> pd.DataFrame:
        """Standard pandas resampling"""
        
        # Apply resampling
        resampler = df.resample(self.target_frequency, label='left', closed='left')
        df_resampled = resampler.agg(agg_dict)
        
        # Remove empty periods if preserve_gaps is False
        if not self.preserve_gaps:
            # Remove rows where all price columns are NaN
            price_columns = ['Open', 'High', 'Low', 'Close']
            available_price_columns = [col for col in price_columns if col in df_resampled.columns]
            
            if available_price_columns:
                mask = df_resampled[available_price_columns].notna().any(axis=1)
                df_resampled = df_resampled[mask]
        
        return df_resampled
    
    def _session_aligned_resample(self, df: pd.DataFrame, agg_dict: Dict[str, Any]) -> pd.DataFrame:
        """Session-aligned resampling for trading hours"""
        
        if self.trading_calendar not in self.trading_sessions:
            logger.warning(f"Unknown trading calendar {self.trading_calendar}, using standard resampling")
            return self._standard_resample(df, agg_dict)
        
        session_info = self.trading_sessions[self.trading_calendar]
        
        # Filter to trading hours
        trading_mask = (
            (df.index.time >= session_info['start']) & 
            (df.index.time <= session_info['end'])
        )
        
        df_trading = df[trading_mask]
        
        if df_trading.empty:
            logger.warning("No data within trading hours, using all data")
            df_trading = df
        
        # Resample trading hours data
        resampler = df_trading.resample(self.target_frequency, label='left', closed='left')
        df_resampled = resampler.agg(agg_dict)
        
        # Add non-trading hours data if preserve_gaps is True
        if self.preserve_gaps and len(df_trading) < len(df):
            df_non_trading = df[~trading_mask]
            
            if not df_non_trading.empty:
                # Resample non-trading data separately
                non_trading_resampled = df_non_trading.resample(self.target_frequency).agg(agg_dict)
                
                # Combine trading and non-trading data
                df_resampled = pd.concat([df_resampled, non_trading_resampled]).sort_index()
        
        return df_resampled
    
    def _create_aggregation_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create aggregation dictionary based on column types"""
        
        agg_dict = {}
        
        # Process each column
        for col in df.columns:
            if col in self.ohlcv_mapping:
                # Use predefined OHLCV aggregation
                agg_dict[col] = self.ohlcv_mapping[col]
            elif col in ['Volume_Price', 'Volume_Typical']:
                # Sum for volume-weighted calculations
                agg_dict[col] = 'sum'
            elif 'volume' in col.lower():
                # Sum for volume-related columns
                agg_dict[col] = 'sum'
            elif 'price' in col.lower() or 'close' in col.lower():
                # Last for price columns
                agg_dict[col] = 'last'
            elif col.startswith(('SMA', 'EMA', 'RSI', 'MACD')):
                # Last for technical indicators
                agg_dict[col] = 'last'
            elif df[col].dtype in ['int64', 'float64']:
                # Mean for other numeric columns
                agg_dict[col] = 'mean'
            else:
                # Last for non-numeric columns
                agg_dict[col] = 'last'
        
        # Add custom aggregations
        agg_dict.update(self.custom_agg_funcs)
        
        return agg_dict
    
    def _create_custom_aggregations(self) -> Dict[str, Callable]:
        """Create custom aggregation functions"""
        
        custom_funcs = {}
        
        # Volume-weighted average price (VWAP)
        def vwap_agg(group_df):
            if 'Volume_Price' in group_df.columns and 'Volume' in group_df.columns:
                total_volume = group_df['Volume'].sum()
                if total_volume > 0:
                    return group_df['Volume_Price'].sum() / total_volume
            return np.nan
        
        # True Range aggregation
        def true_range_agg(group_df):
            if all(col in group_df.columns for col in ['High', 'Low', 'Close']):
                high_low = group_df['High'] - group_df['Low']
                high_close = np.abs(group_df['High'] - group_df['Close'].shift())
                low_close = np.abs(group_df['Low'] - group_df['Close'].shift())
                
                true_ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                return true_ranges.mean()
            return np.nan
        
        # Add custom functions if specific columns exist
        if self.volume_weighted:
            custom_funcs['VWAP'] = vwap_agg
        
        return custom_funcs
    
    def _post_process_resampled_data(self, df_resampled: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
        """Post-process resampled data"""
        
        df_final = df_resampled.copy()
        
        # Calculate volume-weighted prices if enabled
        if self.volume_weighted:
            df_final = self._calculate_volume_weighted_prices(df_final)
        
        # Recalculate derived OHLCV relationships
        df_final = self._recalculate_ohlcv_relationships(df_final)
        
        # Clean up temporary columns
        df_final = self._cleanup_temporary_columns(df_final)
        
        # Forward fill missing values in price columns (optional)
        df_final = self._handle_missing_values(df_final)
        
        # Restore timezone if it was present in original data
        if hasattr(df_original.index, 'tz') and df_original.index.tz is not None:
            df_final.index = df_final.index.tz_localize('UTC').tz_convert(df_original.index.tz)
        
        return df_final
    
    def _calculate_volume_weighted_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-weighted prices"""
        
        # VWAP calculation
        if all(col in df.columns for col in ['Volume_Price', 'Volume']):
            volume_mask = df['Volume'] > 0
            df.loc[volume_mask, 'VWAP'] = safe_divide(
                df.loc[volume_mask, 'Volume_Price'],
                df.loc[volume_mask, 'Volume']
            )
        
        # Volume-weighted typical price
        if all(col in df.columns for col in ['Volume_Typical', 'Volume']):
            volume_mask = df['Volume'] > 0
            df.loc[volume_mask, 'VWTP'] = safe_divide(
                df.loc[volume_mask, 'Volume_Typical'],
                df.loc[volume_mask, 'Volume']
            )
        
        return df
    
    def _recalculate_ohlcv_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recalculate and validate OHLCV relationships"""
        
        # Ensure High is the maximum of O,H,L,C
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        available_ohlc = [col for col in ohlc_cols if col in df.columns]
        
        if len(available_ohlc) >= 3:
            # Recalculate High and Low to ensure consistency
            price_cols_for_high = [col for col in ['Open', 'Close'] if col in df.columns]
            price_cols_for_low = [col for col in ['Open', 'Close'] if col in df.columns]
            
            if 'High' in df.columns and price_cols_for_high:
                df['High'] = df[['High'] + price_cols_for_high].max(axis=1)
            
            if 'Low' in df.columns and price_cols_for_low:
                df['Low'] = df[['Low'] + price_cols_for_low].min(axis=1)
        
        return df
    
    def _cleanup_temporary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove temporary columns created during resampling"""
        
        temp_columns = ['Volume_Price', 'Volume_Typical', 'Typical_Price']
        columns_to_drop = [col for col in temp_columns if col in df.columns]
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logger.debug(f"Removed temporary columns: {columns_to_drop}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in resampled data"""
        
        # Forward fill price columns within reasonable limits
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        
        for col in price_columns:
            if col in df.columns:
                # Forward fill with limit to prevent excessive filling
                df[col] = df[col].fillna(method='ffill', limit=2)
        
        # Volume columns - fill with 0 instead of forward fill
        volume_columns = [col for col in df.columns if 'volume' in col.lower()]
        for col in volume_columns:
            df[col] = df[col].fillna(0)
        
        return df
    
    def _validate_output_data(self, df_resampled: pd.DataFrame, df_original: pd.DataFrame):
        """Validate output data quality"""
        
        # Check that we have data
        if df_resampled.empty:
            raise DataValidationError("Resampling resulted in empty DataFrame")
        
        # Check OHLCV relationships
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        if all(col in df_resampled.columns for col in ohlc_columns):
            
            # Check for impossible OHLC relationships
            invalid_high = (df_resampled['High'] < df_resampled[['Open', 'Low', 'Close']].max(axis=1)).sum()
            invalid_low = (df_resampled['Low'] > df_resampled[['Open', 'High', 'Close']].min(axis=1)).sum()
            
            if invalid_high > 0:
                logger.warning(f"Found {invalid_high} invalid High values after resampling")
            if invalid_low > 0:
                logger.warning(f"Found {invalid_low} invalid Low values after resampling")
        
        # Check for excessive missing data
        total_cells = len(df_resampled) * len(df_resampled.columns)
        missing_cells = df_resampled.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        if missing_percentage > 50:
            logger.warning(f"High percentage of missing data after resampling: {missing_percentage:.1f}%")
        
        # Check frequency consistency
        if len(df_resampled) > 1:
            try:
                inferred_freq = pd.infer_freq(df_resampled.index)
                if inferred_freq != self.target_frequency:
                    logger.debug(f"Inferred frequency {inferred_freq} differs from target {self.target_frequency}")
            except Exception:
                logger.debug("Could not infer frequency from resampled data")

# ============================================
# Multi-Timeframe Resampler
# ============================================

class MultiTimeframeResampler:
    """
    Resample data to multiple timeframes simultaneously
    
    Features:
    - Multiple frequency output
    - Aligned timeframes
    - Memory-efficient processing
    - Consistent aggregation
    """
    
    def __init__(self, 
                 target_frequencies: List[str],
                 base_frequency: Optional[str] = None,
                 volume_weighted: bool = True):
        """
        Initialize multi-timeframe resampler
        
        Args:
            target_frequencies: List of target frequencies
            base_frequency: Base frequency (auto-detected if None)
            volume_weighted: Whether to use volume-weighted calculations
        """
        self.target_frequencies = target_frequencies
        self.base_frequency = base_frequency
        self.volume_weighted = volume_weighted
        
        # Create individual resamplers
        self.resamplers = {
            freq: FinancialDataResampler(
                target_frequency=freq,
                volume_weighted=volume_weighted
            )
            for freq in target_frequencies
        }
    
    @time_it("multi_timeframe_resample")
    def resample(self, df: pd.DataFrame, symbol: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Resample data to multiple timeframes
        
        Args:
            df: Input DataFrame
            symbol: Stock symbol for logging
            
        Returns:
            Dictionary mapping frequencies to resampled DataFrames
        """
        
        logger.info(f"Multi-timeframe resampling for {len(self.target_frequencies)} frequencies")
        
        results = {}
        
        for freq in self.target_frequencies:
            try:
                with Timer(f"resample_{freq}") as timer:
                    resampled_df = self.resamplers[freq].resample(df, symbol)
                    results[freq] = resampled_df
                
                logger.debug(f"Resampled to {freq} in {timer.result.duration_str}: {len(df)} -> {len(resampled_df)} records")
                
            except Exception as e:
                logger.error(f"Failed to resample to {freq}: {e}")
                results[freq] = None
        
        # Filter out failed results
        results = {freq: data for freq, data in results.items() if data is not None}
        
        logger.info(f"Successfully resampled to {len(results)} timeframes")
        return results

# ============================================
# Specialized Resamplers
# ============================================

class VolumeClockResampler:
    """
    Volume-based resampling (volume clock)
    
    Features:
    - Fixed volume intervals
    - Volume imbalance detection
    - Liquidity-aware aggregation
    """
    
    def __init__(self, volume_interval: int = 1000000):
        """
        Initialize volume clock resampler
        
        Args:
            volume_interval: Volume threshold per bar
        """
        self.volume_interval = volume_interval
    
    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample based on volume intervals
        
        Args:
            df: Input DataFrame with Volume column
            
        Returns:
            Volume-clock resampled DataFrame
        """
        if 'Volume' not in df.columns:
            raise DataValidationError("Volume column required for volume clock resampling")
        
        # Calculate cumulative volume
        df_copy = df.copy()
        df_copy['Cumulative_Volume'] = df_copy['Volume'].cumsum()
        
        # Create volume intervals
        max_volume = df_copy['Cumulative_Volume'].iloc[-1]
        volume_breaks = np.arange(0, max_volume, self.volume_interval)
        
        # Assign volume groups
        df_copy['Volume_Group'] = pd.cut(
            df_copy['Cumulative_Volume'], 
            bins=np.append(volume_breaks, max_volume),
            labels=False,
            include_lowest=True
        )
        
        # Aggregate by volume groups
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Add other columns
        for col in df_copy.columns:
            if col not in agg_dict and col not in ['Cumulative_Volume', 'Volume_Group']:
                if df_copy[col].dtype in ['int64', 'float64']:
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'last'
        
        # Group and aggregate
        volume_bars = df_copy.groupby('Volume_Group').agg(agg_dict)
        
        # Set index to timestamp of last trade in each group
        last_timestamps = df_copy.groupby('Volume_Group').apply(lambda x: x.index[-1])
        volume_bars.index = last_timestamps
        
        logger.info(f"Volume clock resampling: {len(df)} -> {len(volume_bars)} bars")
        return volume_bars

class DollarVolumeResampler:
    """
    Dollar volume-based resampling
    
    Features:
    - Fixed dollar volume intervals
    - Market cap awareness
    - Liquidity normalization
    """
    
    def __init__(self, dollar_volume_interval: float = 10000000):  # $10M default
        """
        Initialize dollar volume resampler
        
        Args:
            dollar_volume_interval: Dollar volume threshold per bar
        """
        self.dollar_volume_interval = dollar_volume_interval
    
    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample based on dollar volume intervals
        
        Args:
            df: Input DataFrame with Volume and price columns
            
        Returns:
            Dollar volume resampled DataFrame
        """
        required_columns = ['Volume', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise DataValidationError(f"Missing required columns for dollar volume resampling: {missing_columns}")
        
        # Calculate dollar volume
        df_copy = df.copy()
        df_copy['Dollar_Volume'] = df_copy['Volume'] * df_copy['Close']
        df_copy['Cumulative_Dollar_Volume'] = df_copy['Dollar_Volume'].cumsum()
        
        # Create dollar volume intervals
        max_dollar_volume = df_copy['Cumulative_Dollar_Volume'].iloc[-1]
        dollar_breaks = np.arange(0, max_dollar_volume, self.dollar_volume_interval)
        
        # Assign groups
        df_copy['Dollar_Volume_Group'] = pd.cut(
            df_copy['Cumulative_Dollar_Volume'],
            bins=np.append(dollar_breaks, max_dollar_volume),
            labels=False,
            include_lowest=True
        )
        
        # Aggregate
        agg_dict = {
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Dollar_Volume': 'sum'
        }
        
        # Add other columns
        for col in df_copy.columns:
            if col not in agg_dict and not col.startswith(('Cumulative_', 'Dollar_Volume_Group')):
                if df_copy[col].dtype in ['int64', 'float64']:
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'last'
        
        # Group and aggregate
        dollar_bars = df_copy.groupby('Dollar_Volume_Group').agg(agg_dict)
        
        # Set index to timestamp of last trade in each group
        last_timestamps = df_copy.groupby('Dollar_Volume_Group').apply(lambda x: x.index[-1])
        dollar_bars.index = last_timestamps
        
        logger.info(f"Dollar volume resampling: {len(df)} -> {len(dollar_bars)} bars")
        return dollar_bars

# ============================================
# Factory Functions
# ============================================

def create_resampler(resampler_type: str = 'standard', **kwargs) -> FinancialDataResampler:
    """
    Create pre-configured resampler
    
    Args:
        resampler_type: Type of resampler ('standard', 'trading_hours', 'research')
        **kwargs: Additional arguments
        
    Returns:
        Configured resampler
    """
    
    if resampler_type == 'standard':
        config = {
            'target_frequency': '1H',
            'volume_weighted': True,
            'preserve_gaps': True
        }
    elif resampler_type == 'trading_hours':
        config = {
            'target_frequency': '1H',
            'volume_weighted': True,
            'trading_calendar': 'NYSE',
            'session_alignment': True,
            'preserve_gaps': False
        }
    elif resampler_type == 'research':
        config = {
            'target_frequency': '1D',
            'volume_weighted': True,
            'preserve_gaps': True,
            'session_alignment': False
        }
    else:
        raise ValueError(f"Unknown resampler type: {resampler_type}")
    
    # Override with provided kwargs
    config.update(kwargs)
    
    return FinancialDataResampler(**config)

def create_multi_timeframe_resampler(frequencies: List[str], **kwargs) -> MultiTimeframeResampler:
    """
    Create multi-timeframe resampler
    
    Args:
        frequencies: List of target frequencies
        **kwargs: Additional arguments
        
    Returns:
        Configured multi-timeframe resampler
    """
    return MultiTimeframeResampler(target_frequencies=frequencies, **kwargs)

def resample_to_daily(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Quick function to resample to daily frequency
    
    Args:
        df: Input DataFrame
        symbol: Stock symbol
        
    Returns:
        Daily resampled DataFrame
    """
    resampler = create_resampler('research', target_frequency='1D')
    return resampler.resample(df, symbol)

def resample_to_hourly(df: pd.DataFrame, symbol: Optional[str] = None,
                      trading_hours_only: bool = True) -> pd.DataFrame:
    """
    Quick function to resample to hourly frequency
    
    Args:
        df: Input DataFrame
        symbol: Stock symbol  
        trading_hours_only: Whether to include only trading hours
        
    Returns:
        Hourly resampled DataFrame
    """
    if trading_hours_only:
        resampler = create_resampler('trading_hours', target_frequency='1H')
    else:
        resampler = create_resampler('standard', target_frequency='1H')
    
    return resampler.resample(df, symbol)

# ============================================
# Utility Functions
# ============================================

def validate_resampling_result(df_original: pd.DataFrame, df_resampled: pd.DataFrame,
                              target_frequency: str) -> Dict[str, Any]:
    """
    Validate resampling results
    
    Args:
        df_original: Original DataFrame
        df_resampled: Resampled DataFrame
        target_frequency: Target frequency
        
    Returns:
        Validation report
    """
    
    report = {
        'original_records': len(df_original),
        'resampled_records': len(df_resampled),
        'compression_ratio': len(df_original) / len(df_resampled) if len(df_resampled) > 0 else float('inf'),
        'target_frequency': target_frequency,
        'data_quality': {},
        'warnings': []
    }
    
    # Check data quality
    if df_resampled.empty:
        report['warnings'].append("Resampled data is empty")
        return report
    
    # Check OHLCV relationships
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    if all(col in df_resampled.columns for col in ohlc_cols):
        invalid_high = (df_resampled['High'] < df_resampled[['Open', 'Low', 'Close']].max(axis=1)).sum()
        invalid_low = (df_resampled['Low'] > df_resampled[['Open', 'High', 'Close']].min(axis=1)).sum()
        
        report['data_quality']['ohlc_violations'] = {
            'invalid_high': int(invalid_high),
            'invalid_low': int(invalid_low)
        }
        
        if invalid_high > 0:
            report['warnings'].append(f"{invalid_high} invalid High values detected")
        if invalid_low > 0:
            report['warnings'].append(f"{invalid_low} invalid Low values detected")
    
    # Check missing data
    missing_data = df_resampled.isnull().sum()
    if missing_data.sum() > 0:
        report['data_quality']['missing_data'] = missing_data.to_dict()
        
        missing_pct = (missing_data.sum() / (len(df_resampled) * len(df_resampled.columns))) * 100
        if missing_pct > 10:
            report['warnings'].append(f"High missing data percentage: {missing_pct:.1f}%")
    
    # Check frequency consistency
    if len(df_resampled) > 2:
        try:
            inferred_freq = pd.infer_freq(df_resampled.index)
            report['inferred_frequency'] = inferred_freq
            
            if inferred_freq != target_frequency:
                report['warnings'].append(f"Inferred frequency ({inferred_freq}) differs from target ({target_frequency})")
        except Exception:
            report['warnings'].append("Could not infer frequency from resampled data")
    
    return report

def get_optimal_frequency_for_timespan(start_date: datetime, end_date: datetime,
                                      max_records: int = 10000) -> str:
    """
    Get optimal resampling frequency for a given timespan
    
    Args:
        start_date: Start date
        end_date: End date
        max_records: Maximum desired records
        
    Returns:
        Optimal frequency string
    """
    
    timespan = end_date - start_date
    total_days = timespan.days
    
    # Estimate records for different frequencies
    frequencies = {
        '1T': total_days * 24 * 60,      # 1 minute
        '5T': total_days * 24 * 12,      # 5 minutes  
        '15T': total_days * 24 * 4,      # 15 minutes
        '1H': total_days * 24,           # 1 hour
        '4H': total_days * 6,            # 4 hours
        '1D': total_days,                # 1 day
        '1W': total_days // 7,           # 1 week
        '1M': total_days // 30           # 1 month
    }
    
    # Find the highest frequency that doesn't exceed max_records
    optimal_freq = '1M'  # Default to monthly
    
    for freq, estimated_records in frequencies.items():
        if estimated_records <= max_records:
            optimal_freq = freq
            break
    
    logger.info(f"Optimal frequency for {total_days} days with max {max_records} records: {optimal_freq}")
    return optimal_freq
