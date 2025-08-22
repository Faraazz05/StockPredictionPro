# ============================================
# StockPredictionPro - src/data/processors/cleaner.py
# Advanced data cleaning and quality assurance for financial time series
# ============================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks
import warnings
from scipy.signal import savgol_filter


from ...utils.exceptions import DataValidationError, BusinessLogicError
from ...utils.logger import get_logger
from ...utils.timing import Timer, time_it
from ...utils.config_loader import get
from ...utils.helpers import safe_divide
from ...utils.validators import ValidationResult, validate_data_for_analysis


logger = get_logger('data.processors.cleaner')

# ============================================
# Data Quality Assessment
# ============================================

class DataQualityAssessor:
    """
    Assess data quality for financial time series
    
    Features:
    - Missing data analysis
    - Outlier detection
    - Data consistency checks
    - Quality scoring
    """
    
    def __init__(self):
        self.quality_report = {}
        self.thresholds = {
            'missing_data_threshold': 0.05,  # 5% maximum missing data
            'outlier_threshold': 3.0,        # 3 standard deviations
            'gap_threshold_days': 7,         # Maximum 7-day gaps in daily data
            'zero_volume_threshold': 0.1,    # 10% maximum zero volume days
            'price_jump_threshold': 0.2      # 20% maximum single-day price jump
        }
    
    def assess_quality(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment
        
        Args:
            df: DataFrame to assess
            symbol: Stock symbol for context
            
        Returns:
            Quality assessment report
        """
        logger.info(f"Assessing data quality for {symbol}")
        
        report = {
            'symbol': symbol,
            'assessment_time': datetime.now().isoformat(),
            'total_records': len(df),
            'date_range': self._get_date_range(df),
            'missing_data': self._assess_missing_data(df),
            'outliers': self._assess_outliers(df),
            'data_consistency': self._assess_consistency(df),
            'temporal_issues': self._assess_temporal_issues(df),
            'ohlcv_validation': self._assess_ohlcv_relationships(df),
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # Calculate overall quality score
        report['overall_score'] = self._calculate_quality_score(report)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        self.quality_report[symbol] = report
        
        logger.info(f"Quality assessment complete for {symbol}: Score {report['overall_score']:.2f}/10")
        return report
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get date range information"""
        if isinstance(df.index, pd.DatetimeIndex):
            return {
                'start_date': df.index.min().isoformat(),
                'end_date': df.index.max().isoformat(),
                'total_days': (df.index.max() - df.index.min()).days,
                'trading_days': len(df)
            }
        else:
            return {
                'start_date': None,
                'end_date': None,
                'total_days': 0,
                'trading_days': len(df)
            }
    
    def _assess_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess missing data patterns"""
        missing_analysis = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / len(df) * 100
            
            # Identify missing data patterns
            missing_mask = df[col].isnull()
            consecutive_missing = self._find_consecutive_periods(missing_mask)
            
            missing_analysis[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_pct),
                'max_consecutive_missing': max(consecutive_missing) if consecutive_missing else 0,
                'missing_periods': len(consecutive_missing),
                'quality_impact': 'high' if missing_pct > 10 else 'medium' if missing_pct > 5 else 'low'
            }
        
        return missing_analysis
    
    def _assess_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess outlier patterns"""
        outlier_analysis = {}
        
        # Check price columns for outliers
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        
        for col in price_columns:
            if col in df.columns:
                outliers = self._detect_outliers(df[col])
                
                outlier_analysis[col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(df) * 100,
                    'outlier_indices': outliers.tolist(),
                    'extreme_values': self._get_extreme_values(df[col], outliers)
                }
        
        # Check volume outliers separately
        if 'Volume' in df.columns:
            volume_outliers = self._detect_volume_outliers(df['Volume'])
            outlier_analysis['Volume'] = {
                'outlier_count': len(volume_outliers),
                'outlier_percentage': len(volume_outliers) / len(df) * 100,
                'zero_volume_days': (df['Volume'] == 0).sum(),
                'zero_volume_percentage': (df['Volume'] == 0).sum() / len(df) * 100
            }
        
        return outlier_analysis
    
    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency"""
        consistency_analysis = {
            'duplicate_dates': 0,
            'chronological_order': True,
            'data_type_consistency': True,
            'negative_prices': 0,
            'negative_volume': 0,
            'ohlc_violations': 0
        }
        
        # Check for duplicate dates
        if isinstance(df.index, pd.DatetimeIndex):
            consistency_analysis['duplicate_dates'] = df.index.duplicated().sum()
            consistency_analysis['chronological_order'] = df.index.is_monotonic_increasing
        
        # Check for negative values
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_columns:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                consistency_analysis['negative_prices'] += negative_count
        
        if 'Volume' in df.columns:
            consistency_analysis['negative_volume'] = (df['Volume'] < 0).sum()
        
        # Check OHLC relationships
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            high_violations = (
                (df['High'] < df['Open']) | 
                (df['High'] < df['Low']) | 
                (df['High'] < df['Close'])
            ).sum()
            
            low_violations = (
                (df['Low'] > df['Open']) | 
                (df['Low'] > df['High']) | 
                (df['Low'] > df['Close'])
            ).sum()
            
            consistency_analysis['ohlc_violations'] = high_violations + low_violations
        
        return consistency_analysis
    
    def _assess_temporal_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess temporal data issues"""
        temporal_analysis = {
            'large_gaps': [],
            'irregular_frequency': False,
            'weekend_data': 0,
            'holiday_trading': 0
        }
        
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            # Check for large gaps
            date_diffs = df.index.to_series().diff().dt.days
            large_gaps = date_diffs[date_diffs > self.thresholds['gap_threshold_days']]
            
            temporal_analysis['large_gaps'] = [
                {
                    'start_date': df.index[i-1].isoformat(),
                    'end_date': df.index[i].isoformat(),
                    'gap_days': int(gap)
                }
                for i, gap in large_gaps.items()
            ]
            
            # Check for weekend data (Saturday=5, Sunday=6)
            weekend_mask = df.index.weekday >= 5
            temporal_analysis['weekend_data'] = weekend_mask.sum()
            
            # Check frequency regularity
            most_common_diff = date_diffs.mode().iloc[0] if len(date_diffs.mode()) > 0 else None
            irregular_count = (date_diffs != most_common_diff).sum()
            temporal_analysis['irregular_frequency'] = irregular_count > len(df) * 0.1  # 10% irregular
        
        return temporal_analysis
    
    def _assess_ohlcv_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess OHLCV relationship validity"""
        ohlcv_analysis = {
            'valid_ohlc': True,
            'price_volume_correlation': None,
            'suspicious_patterns': []
        }
        
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Check for impossible OHLC relationships
            invalid_high = (df['High'] < df[['Open', 'Low', 'Close']].max(axis=1)).any()
            invalid_low = (df['Low'] > df[['Open', 'High', 'Close']].min(axis=1)).any()
            
            ohlcv_analysis['valid_ohlc'] = not (invalid_high or invalid_low)
            
            # Check price-volume correlation
            if 'Volume' in df.columns:
                price_change = df['Close'].pct_change().abs()
                correlation = price_change.corr(df['Volume'])
                ohlcv_analysis['price_volume_correlation'] = float(correlation) if not np.isnan(correlation) else None
            
            # Detect suspicious patterns
            ohlcv_analysis['suspicious_patterns'] = self._detect_suspicious_patterns(df)
        
        return ohlcv_analysis
    
    def _detect_outliers(self, series: pd.Series, method: str = 'iqr') -> np.ndarray:
        """Detect outliers using various methods"""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            return series[(series < lower_bound) | (series > upper_bound)].index.values
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series.dropna()))
            threshold = self.thresholds['outlier_threshold']
            outlier_mask = z_scores > threshold
            
            return series.dropna().iloc[outlier_mask].index.values
        
        else:
            return np.array([])
    
    def _detect_volume_outliers(self, volume_series: pd.Series) -> np.ndarray:
        """Detect volume-specific outliers"""
        # Volume outliers are typically extremely high values
        # Use log transformation for better detection
        log_volume = np.log1p(volume_series)  # log1p handles zero values
        z_scores = np.abs(stats.zscore(log_volume.dropna()))
        
        outlier_mask = z_scores > 3.0  # More stringent for volume
        return volume_series.dropna().iloc[outlier_mask].index.values
    
    def _get_extreme_values(self, series: pd.Series, outlier_indices: np.ndarray) -> Dict[str, float]:
        """Get extreme values information"""
        if len(outlier_indices) == 0:
            return {'min': None, 'max': None}
        
        outlier_values = series.loc[outlier_indices]
        return {
            'min': float(outlier_values.min()),
            'max': float(outlier_values.max()),
            'mean': float(outlier_values.mean()),
            'std': float(outlier_values.std())
        }
    
    def _find_consecutive_periods(self, mask: pd.Series) -> List[int]:
        """Find consecutive periods of True values"""
        consecutive_periods = []
        current_period = 0
        
        for value in mask:
            if value:
                current_period += 1
            else:
                if current_period > 0:
                    consecutive_periods.append(current_period)
                current_period = 0
        
        # Add final period if it ends with True values
        if current_period > 0:
            consecutive_periods.append(current_period)
        
        return consecutive_periods
    
    def _detect_suspicious_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect suspicious trading patterns"""
        suspicious_patterns = []
        
        # Pattern 1: Identical OHLC values (suspicious for stocks)
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            identical_ohlc = (
                (df['Open'] == df['High']) & 
                (df['High'] == df['Low']) & 
                (df['Low'] == df['Close'])
            )
            
            if identical_ohlc.any():
                suspicious_patterns.append({
                    'pattern': 'identical_ohlc',
                    'count': int(identical_ohlc.sum()),
                    'percentage': float(identical_ohlc.sum() / len(df) * 100),
                    'description': 'Days with identical OHLC values'
                })
        
        # Pattern 2: Extreme price jumps
        if 'Close' in df.columns:
            price_changes = df['Close'].pct_change().abs()
            extreme_jumps = price_changes > self.thresholds['price_jump_threshold']
            
            if extreme_jumps.any():
                suspicious_patterns.append({
                    'pattern': 'extreme_price_jumps',
                    'count': int(extreme_jumps.sum()),
                    'max_jump': float(price_changes.max()),
                    'description': f'Price jumps > {self.thresholds["price_jump_threshold"]*100}%'
                })
        
        # Pattern 3: Round number bias
        if 'Close' in df.columns:
            # Check for excessive round numbers (ending in .00)
            round_numbers = (df['Close'] % 1 == 0)
            round_pct = round_numbers.sum() / len(df) * 100
            
            if round_pct > 20:  # More than 20% round numbers is suspicious
                suspicious_patterns.append({
                    'pattern': 'round_number_bias',
                    'percentage': float(round_pct),
                    'description': 'Excessive round number prices'
                })
        
        return suspicious_patterns
    
    def _calculate_quality_score(self, report: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-10)"""
        score = 10.0  # Start with perfect score
        
        # Deduct for missing data
        missing_data = report['missing_data']
        for col_info in missing_data.values():
            missing_pct = col_info['missing_percentage']
            if missing_pct > 10:
                score -= 2.0
            elif missing_pct > 5:
                score -= 1.0
            elif missing_pct > 1:
                score -= 0.5
        
        # Deduct for outliers
        outliers = report['outliers']
        for col_info in outliers.values():
            outlier_pct = col_info.get('outlier_percentage', 0)
            if outlier_pct > 5:
                score -= 1.0
            elif outlier_pct > 2:
                score -= 0.5
        
        # Deduct for consistency issues
        consistency = report['data_consistency']
        if not consistency['chronological_order']:
            score -= 2.0
        if consistency['ohlc_violations'] > 0:
            score -= 1.0
        if consistency['negative_prices'] > 0:
            score -= 1.5
        
        # Deduct for temporal issues
        temporal = report['temporal_issues']
        if len(temporal['large_gaps']) > 0:
            score -= 0.5 * len(temporal['large_gaps'])
        if temporal['irregular_frequency']:
            score -= 1.0
        
        # Deduct for OHLCV issues
        ohlcv = report['ohlcv_validation']
        if not ohlcv['valid_ohlc']:
            score -= 2.0
        
        return max(0.0, score)  # Ensure score doesn't go below 0
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate data cleaning recommendations"""
        recommendations = []
        
        # Missing data recommendations
        missing_data = report['missing_data']
        for col, info in missing_data.items():
            if info['missing_percentage'] > 10:
                recommendations.append(f"Consider removing or interpolating missing data in {col} ({info['missing_percentage']:.1f}% missing)")
            elif info['max_consecutive_missing'] > 5:
                recommendations.append(f"Address consecutive missing periods in {col} (max {info['max_consecutive_missing']} consecutive)")
        
        # Outlier recommendations
        outliers = report['outliers']
        for col, info in outliers.items():
            if info.get('outlier_percentage', 0) > 5:
                recommendations.append(f"Review and possibly cap outliers in {col} ({info['outlier_percentage']:.1f}% outliers)")
        
        # Consistency recommendations
        consistency = report['data_consistency']
        if not consistency['chronological_order']:
            recommendations.append("Sort data by date to ensure chronological order")
        if consistency['ohlc_violations'] > 0:
            recommendations.append("Fix OHLC relationship violations")
        if consistency['negative_prices'] > 0:
            recommendations.append("Remove or correct negative price values")
        
        # Temporal recommendations
        temporal = report['temporal_issues']
        if len(temporal['large_gaps']) > 0:
            recommendations.append("Fill or account for large date gaps in the data")
        if temporal['weekend_data'] > 0:
            recommendations.append("Consider removing weekend trading data")
        
        return recommendations

# ============================================
# Data Cleaning Engine
# ============================================

class FinancialDataCleaner:
    """
    Comprehensive data cleaning engine for financial time series
    
    Features:
    - Automated cleaning workflows
    - Configurable cleaning rules
    - Data validation and repair
    - Quality improvement tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data cleaner
        
        Args:
            config: Cleaning configuration parameters
        """
        self.config = config or self._get_default_config()
        self.assessor = DataQualityAssessor()
        self.cleaning_history = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default cleaning configuration"""
        return {
            'missing_data': {
                'method': 'interpolate',  # 'drop', 'interpolate', 'forward_fill', 'backward_fill'
                'max_missing_pct': 10.0,
                'max_consecutive_missing': 5,
                'interpolation_method': 'linear'
            },
            'outliers': {
                'method': 'clip',  # 'remove', 'clip', 'transform'
                'detection_method': 'iqr',  # 'iqr', 'zscore'
                'threshold': 3.0,
                'clip_percentiles': (1, 99)
            },
            'temporal': {
                'ensure_chronological': True,
                'remove_duplicates': True,
                'remove_weekends': False,
                'fill_gaps': True,
                'max_gap_days': 7
            },
            'validation': {
                'fix_ohlc_violations': True,
                'remove_negative_prices': True,
                'remove_zero_volume': False,
                'validate_relationships': True
            },
            'smoothing': {
                'enable_smoothing': False,
                'method': 'savgol',  # 'savgol', 'rolling_mean'
                'window_length': 5,
                'polyorder': 2
            }
        }
    
    @time_it("financial_data_cleaning")
    def clean_data(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform comprehensive data cleaning
        
        Args:
            df: DataFrame to clean
            symbol: Stock symbol for context
            
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        logger.info(f"Starting data cleaning for {symbol}")
        
        # Initial quality assessment
        initial_quality = self.assessor.assess_quality(df.copy(), symbol)
        
        # Create cleaning report
        cleaning_report = {
            'symbol': symbol,
            'cleaning_time': datetime.now().isoformat(),
            'initial_quality_score': initial_quality['overall_score'],
            'initial_records': len(df),
            'steps_applied': [],
            'final_quality_score': 0.0,
            'final_records': 0,
            'improvement': 0.0
        }
        
        # Start with copy of original data
        df_cleaned = df.copy()
        
        try:
            # Step 1: Fix temporal issues
            df_cleaned, temporal_report = self._fix_temporal_issues(df_cleaned, symbol)
            cleaning_report['steps_applied'].append(('temporal_fixes', temporal_report))
            
            # Step 2: Fix data consistency issues
            df_cleaned, consistency_report = self._fix_consistency_issues(df_cleaned, symbol)
            cleaning_report['steps_applied'].append(('consistency_fixes', consistency_report))
            
            # Step 3: Handle missing data
            df_cleaned, missing_report = self._handle_missing_data(df_cleaned, symbol)
            cleaning_report['steps_applied'].append(('missing_data_handling', missing_report))
            
            # Step 4: Handle outliers
            df_cleaned, outlier_report = self._handle_outliers(df_cleaned, symbol)
            cleaning_report['steps_applied'].append(('outlier_handling', outlier_report))
            
            # Step 5: Apply smoothing (optional)
            if self.config['smoothing']['enable_smoothing']:
                df_cleaned, smoothing_report = self._apply_smoothing(df_cleaned, symbol)
                cleaning_report['steps_applied'].append(('smoothing', smoothing_report))
            
            # Step 6: Final validation
            df_cleaned, validation_report = self._final_validation(df_cleaned, symbol)
            cleaning_report['steps_applied'].append(('final_validation', validation_report))
            
            # Final quality assessment
            final_quality = self.assessor.assess_quality(df_cleaned, symbol)
            cleaning_report['final_quality_score'] = final_quality['overall_score']
            cleaning_report['final_records'] = len(df_cleaned)
            cleaning_report['improvement'] = final_quality['overall_score'] - initial_quality['overall_score']
            
            # Store cleaning history
            self.cleaning_history[symbol] = cleaning_report
            
            logger.info(f"Data cleaning complete for {symbol}: "
                       f"Quality improved from {initial_quality['overall_score']:.2f} to {final_quality['overall_score']:.2f}")
            
            return df_cleaned, cleaning_report
            
        except Exception as e:
            logger.error(f"Error during data cleaning for {symbol}: {e}")
            cleaning_report['error'] = str(e)
            return df, cleaning_report
    
    def _fix_temporal_issues(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fix temporal data issues"""
        report = {
            'duplicates_removed': 0,
            'chronological_sorting': False,
            'weekend_data_removed': 0,
            'gaps_filled': 0
        }
        
        original_length = len(df)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            else:
                logger.warning(f"No datetime index or Date column found for {symbol}")
                return df, report
        
        # Remove duplicates
        if self.config['temporal']['remove_duplicates']:
            before_dedup = len(df)
            df = df[~df.index.duplicated(keep='first')]
            report['duplicates_removed'] = before_dedup - len(df)
        
        # Ensure chronological order
        if self.config['temporal']['ensure_chronological']:
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
                report['chronological_sorting'] = True
        
        # Remove weekend data (optional)
        if self.config['temporal']['remove_weekends']:
            weekday_mask = df.index.weekday < 5  # Monday=0, Friday=4
            weekend_count = (~weekday_mask).sum()
            df = df[weekday_mask]
            report['weekend_data_removed'] = weekend_count
        
        # Fill small gaps (optional)
        if self.config['temporal']['fill_gaps'] and len(df) > 1:
            df = self._fill_temporal_gaps(df)
            report['gaps_filled'] = len(df) - original_length + report['duplicates_removed'] + report['weekend_data_removed']
        
        return df, report
    
    def _fix_consistency_issues(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fix data consistency issues"""
        report = {
            'negative_prices_fixed': 0,
            'negative_volume_fixed': 0,
            'ohlc_violations_fixed': 0,
            'zero_values_handled': 0
        }
        
        # Fix negative prices
        if self.config['validation']['remove_negative_prices']:
            price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
            for col in price_columns:
                if col in df.columns:
                    negative_mask = df[col] < 0
                    negative_count = negative_mask.sum()
                    if negative_count > 0:
                        # Set negative prices to NaN for later interpolation
                        df.loc[negative_mask, col] = np.nan
                        report['negative_prices_fixed'] += negative_count
        
        # Fix negative volume
        if 'Volume' in df.columns:
            negative_volume_mask = df['Volume'] < 0
            negative_volume_count = negative_volume_mask.sum()
            if negative_volume_count > 0:
                df.loc[negative_volume_mask, 'Volume'] = 0  # Set to zero instead of NaN
                report['negative_volume_fixed'] = negative_volume_count
        
        # Fix OHLC violations
        if self.config['validation']['fix_ohlc_violations']:
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                violations_fixed = self._fix_ohlc_violations(df)
                report['ohlc_violations_fixed'] = violations_fixed
        
        # Handle zero volume (optional)
        if self.config['validation']['remove_zero_volume'] and 'Volume' in df.columns:
            zero_volume_mask = df['Volume'] == 0
            zero_count = zero_volume_mask.sum()
            if zero_count > 0:
                df.loc[zero_volume_mask, 'Volume'] = np.nan
                report['zero_values_handled'] = zero_count
        
        return df, report
    
    def _handle_missing_data(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing data"""
        report = {
            'method_used': self.config['missing_data']['method'],
            'records_before': len(df),
            'missing_before': df.isnull().sum().sum(),
            'records_after': 0,
            'missing_after': 0,
            'columns_processed': []
        }
        
        method = self.config['missing_data']['method']
        
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            
            # Skip if too much missing data
            if missing_pct > self.config['missing_data']['max_missing_pct']:
                logger.warning(f"Column {col} has {missing_pct:.1f}% missing data, exceeding threshold")
                continue
            
            if df[col].isnull().any():
                if method == 'drop':
                    df = df.dropna(subset=[col])
                elif method == 'interpolate':
                    df[col] = df[col].interpolate(method=self.config['missing_data']['interpolation_method'])
                elif method == 'forward_fill':
                    df[col] = df[col].fillna(method='ffill')
                elif method == 'backward_fill':
                    df[col] = df[col].fillna(method='bfill')
                
                report['columns_processed'].append(col)
        
        report['records_after'] = len(df)
        report['missing_after'] = df.isnull().sum().sum()
        
        return df, report
    
    def _handle_outliers(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle outlier data"""
        report = {
            'method_used': self.config['outliers']['method'],
            'detection_method': self.config['outliers']['detection_method'],
            'outliers_found': 0,
            'outliers_handled': 0,
            'columns_processed': []
        }
        
        method = self.config['outliers']['method']
        detection_method = self.config['outliers']['detection_method']
        
        # Process price columns
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        
        for col in price_columns:
            if col in df.columns:
                outliers = self._detect_outliers_in_column(df[col], detection_method)
                
                if len(outliers) > 0:
                    report['outliers_found'] += len(outliers)
                    
                    if method == 'remove':
                        df = df.drop(outliers)
                        report['outliers_handled'] += len(outliers)
                    elif method == 'clip':
                        lower_pct, upper_pct = self.config['outliers']['clip_percentiles']
                        lower_bound = df[col].quantile(lower_pct / 100)
                        upper_bound = df[col].quantile(upper_pct / 100)
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                        report['outliers_handled'] += len(outliers)
                    elif method == 'transform':
                        # Use log transformation for extreme outliers
                        extreme_outliers = df.loc[outliers, col]
                        median_val = df[col].median()
                        df.loc[outliers, col] = np.sign(extreme_outliers) * np.log1p(np.abs(extreme_outliers - median_val)) + median_val
                        report['outliers_handled'] += len(outliers)
                    
                    report['columns_processed'].append(col)
        
        return df, report
    
    def _apply_smoothing(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply data smoothing"""
        report = {
            'method_used': self.config['smoothing']['method'],
            'columns_smoothed': []
        }
        
        method = self.config['smoothing']['method']
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        
        for col in price_columns:
            if col in df.columns:
                original_col = f"{col}_original"
                df[original_col] = df[col].copy()
                
                if method == 'savgol':
                    window_length = self.config['smoothing']['window_length']
                    polyorder = self.config['smoothing']['polyorder']
                    
                    # Ensure window length is odd and not larger than data
                    if window_length % 2 == 0:
                        window_length += 1
                    window_length = min(window_length, len(df))
                    
                    if window_length >= polyorder + 1:
                        df[col] = savgol_filter(df[col], window_length, polyorder)
                        report['columns_smoothed'].append(col)
                
                elif method == 'rolling_mean':
                    window = self.config['smoothing']['window_length']
                    df[col] = df[col].rolling(window=window, center=True).mean()
                    # Fill NaN values at edges
                    df[col] = df[col].fillna(df[original_col])
                    report['columns_smoothed'].append(col)
        
        return df, report
    
    def _final_validation(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform final validation and cleanup"""
        report = {
            'final_record_count': len(df),
            'final_missing_values': df.isnull().sum().sum(),
            'validation_passed': True,
            'issues_found': []
        }
        
        # Check for remaining issues
        if df.empty:
            report['validation_passed'] = False
            report['issues_found'].append("DataFrame is empty after cleaning")
        
        # Check for excessive missing data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        if missing_pct > 5:
            report['issues_found'].append(f"High missing data percentage: {missing_pct:.1f}%")
        
        # Check OHLCV columns exist
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            report['issues_found'].append(f"Missing required columns: {missing_required}")
        
        # Drop any remaining rows with all NaN values
        initial_length = len(df)
        df = df.dropna(how='all')
        dropped_empty_rows = initial_length - len(df)
        
        if dropped_empty_rows > 0:
            report['issues_found'].append(f"Dropped {dropped_empty_rows} completely empty rows")
        
        report['final_record_count'] = len(df)
        
        return df, report
    
    def _fill_temporal_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill small temporal gaps in the data"""
        if len(df) < 2:
            return df
        
        # Create a complete date range
        start_date = df.index.min()
        end_date = df.index.max()
        
        # For daily data, create business day range
        if isinstance(df.index.freq, type(None)):
            # Infer frequency
            date_diffs = df.index.to_series().diff().dt.days
            most_common_diff = date_diffs.mode().iloc[0] if len(date_diffs.mode()) > 0 else 1
            
            if most_common_diff == 1:
                # Daily data - use business days
                complete_range = pd.bdate_range(start=start_date, end=end_date)
            else:
                # Other frequency - create basic range
                complete_range = pd.date_range(start=start_date, end=end_date, freq=f'{most_common_diff}D')
        else:
            complete_range = pd.date_range(start=start_date, end=end_date, freq=df.index.freq)
        
        # Reindex and forward fill small gaps
        df_complete = df.reindex(complete_range)
        
        # Only fill gaps smaller than max_gap_days
        max_gap = self.config['temporal']['max_gap_days']
        df_filled = df_complete.fillna(method='ffill', limit=max_gap)
        
        # Remove rows that are still NaN (large gaps)
        df_filled = df_filled.dropna()
        
        return df_filled
    
    def _detect_outliers_in_column(self, series: pd.Series, method: str) -> List:
        """Detect outliers in a specific column"""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            return series[outlier_mask].index.tolist()
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series.dropna()))
            threshold = self.config['outliers']['threshold']
            outlier_mask = z_scores > threshold
            
            return series.dropna().iloc[outlier_mask].index.tolist()
        
        return []
    
    def _fix_ohlc_violations(self, df: pd.DataFrame) -> int:
        """Fix OHLC relationship violations"""
        violations_fixed = 0
        
        # Fix High values that are too low
        for idx in df.index:
            o, h, l, c = df.loc[idx, ['Open', 'High', 'Low', 'Close']]
            
            # High should be >= max(Open, Low, Close)
            correct_high = max(o, l, c)
            if h < correct_high:
                df.loc[idx, 'High'] = correct_high
                violations_fixed += 1
            
            # Low should be <= min(Open, High, Close)
            correct_low = min(o, h, c)
            if l > correct_low:
                df.loc[idx, 'Low'] = correct_low
                violations_fixed += 1
        
        return violations_fixed
    
    def get_cleaning_history(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Get cleaning history for symbol(s)"""
        if symbol:
            return self.cleaning_history.get(symbol, {})
        else:
            return self.cleaning_history
    
    def create_cleaning_summary(self) -> Dict[str, Any]:
        """Create summary of all cleaning operations"""
        if not self.cleaning_history:
            return {}
        
        summary = {
            'total_symbols_cleaned': len(self.cleaning_history),
            'average_quality_improvement': 0.0,
            'common_issues': {},
            'cleaning_statistics': {}
        }
        
        improvements = []
        issue_counts = {}
        
        for symbol, history in self.cleaning_history.items():
            improvements.append(history.get('improvement', 0))
            
            # Count common issues
            for step_name, step_report in history.get('steps_applied', []):
                for key, value in step_report.items():
                    if isinstance(value, (int, float)) and value > 0:
                        issue_key = f"{step_name}_{key}"
                        issue_counts[issue_key] = issue_counts.get(issue_key, 0) + 1
        
        summary['average_quality_improvement'] = np.mean(improvements) if improvements else 0.0
        summary['common_issues'] = dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return summary

# ============================================
# Factory Functions
# ============================================

def create_data_cleaner(cleaning_mode: str = 'comprehensive', **kwargs) -> FinancialDataCleaner:
    """
    Create pre-configured data cleaner
    
    Args:
        cleaning_mode: Mode of cleaning ('minimal', 'standard', 'comprehensive', 'aggressive')
        
    Returns:
        Configured FinancialDataCleaner
    """
    if cleaning_mode == 'minimal':
        config = {
            'missing_data': {'method': 'drop', 'max_missing_pct': 5.0},
            'outliers': {'method': 'clip', 'clip_percentiles': (5, 95)},
            'temporal': {'ensure_chronological': True, 'remove_duplicates': True},
            'validation': {'fix_ohlc_violations': True, 'remove_negative_prices': True}
        }
    elif cleaning_mode == 'standard':
        config = {
            'missing_data': {'method': 'interpolate', 'max_missing_pct': 10.0},
            'outliers': {'method': 'clip', 'clip_percentiles': (2, 98)},
            'temporal': {'ensure_chronological': True, 'remove_duplicates': True, 'fill_gaps': True},
            'validation': {'fix_ohlc_violations': True, 'remove_negative_prices': True}
        }
    elif cleaning_mode == 'comprehensive':
        config = {
            'missing_data': {'method': 'interpolate', 'max_missing_pct': 15.0, 'interpolation_method': 'linear'},
            'outliers': {'method': 'clip', 'detection_method': 'iqr', 'clip_percentiles': (1, 99)},
            'temporal': {'ensure_chronological': True, 'remove_duplicates': True, 'fill_gaps': True, 'max_gap_days': 7},
            'validation': {'fix_ohlc_violations': True, 'remove_negative_prices': True, 'validate_relationships': True},
            'smoothing': {'enable_smoothing': False}
        }
    elif cleaning_mode == 'aggressive':
        config = {
            'missing_data': {'method': 'interpolate', 'max_missing_pct': 20.0},
            'outliers': {'method': 'transform', 'detection_method': 'zscore', 'threshold': 2.5},
            'temporal': {'ensure_chronological': True, 'remove_duplicates': True, 'fill_gaps': True, 'remove_weekends': True},
            'validation': {'fix_ohlc_violations': True, 'remove_negative_prices': True, 'remove_zero_volume': True},
            'smoothing': {'enable_smoothing': True, 'method': 'savgol', 'window_length': 5}
        }
    else:
        raise ValueError(f"Unknown cleaning mode: {cleaning_mode}")
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if key in config:
            if isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    
    return FinancialDataCleaner(config=config)

def quick_clean_financial_data(df: pd.DataFrame, symbol: str, mode: str = 'standard') -> pd.DataFrame:
    """
    Quick data cleaning function
    
    Args:
        df: DataFrame to clean
        symbol: Stock symbol
        mode: Cleaning mode
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = create_data_cleaner(mode)
    cleaned_df, _ = cleaner.clean_data(df, symbol)
    return cleaned_df
