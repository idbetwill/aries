"""
Data Preprocessor for Aries Trading Agent

Handles data cleaning, normalization, feature engineering,
and preparation for machine learning models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing pipeline for the Aries trading agent.
    
    Handles cleaning, normalization, feature engineering,
    and preparation of market data for ML models.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self._default_config()
        self.scalers = {}
        self.imputers = {}
        self.feature_columns = []
        
    def _default_config(self) -> Dict:
        """Return default preprocessing configuration."""
        return {
            'missing_data_strategy': 'interpolate',  # 'interpolate', 'forward_fill', 'backward_fill', 'drop'
            'outlier_detection': True,
            'outlier_method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
            'outlier_threshold': 3.0,
            'normalization': 'standard',  # 'standard', 'minmax', 'robust', 'none'
            'feature_engineering': True,
            'time_features': True,
            'lag_features': True,
            'rolling_features': True,
            'lag_periods': [1, 2, 3, 6, 12, 24],
            'rolling_windows': [3, 6, 12, 24, 48],
            'target_column': 'price'
        }
    
    def process_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process market data through the complete preprocessing pipeline.
        
        Args:
            data: Raw market data DataFrame
            
        Returns:
            Processed DataFrame ready for ML models
        """
        logger.info(f"Processing market data: {len(data)} records")
        
        if data.empty:
            logger.warning("Empty dataset provided")
            return data
        
        # Create a copy to avoid modifying original data
        processed = data.copy()
        
        # Step 1: Basic cleaning
        processed = self._clean_data(processed)
        
        # Step 2: Handle missing values
        processed = self._handle_missing_values(processed)
        
        # Step 3: Detect and handle outliers
        if self.config['outlier_detection']:
            processed = self._handle_outliers(processed)
        
        # Step 4: Feature engineering
        if self.config['feature_engineering']:
            processed = self._engineer_features(processed)
        
        # Step 5: Normalization
        if self.config['normalization'] != 'none':
            processed = self._normalize_data(processed)
        
        # Step 6: Final validation
        processed = self._validate_data(processed)
        
        logger.info(f"Processing completed: {len(processed)} records, {len(processed.columns)} features")
        return processed
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean basic data issues."""
        logger.info("Cleaning data")
        
        # Remove duplicate timestamps
        if 'timestamp' in data.columns:
            data = data.drop_duplicates(subset=['timestamp'])
            data = data.sort_values('timestamp')
        
        # Convert numeric columns
        numeric_columns = ['price', 'volume', 'demand', 'supply', 'consumption', 'generation']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values")
        
        strategy = self.config['missing_data_strategy']
        
        if strategy == 'interpolate':
            # Interpolate missing values
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if data[col].isna().any():
                    data[col] = data[col].interpolate(method='linear')
                    # Fill remaining NaN with forward fill
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        elif strategy == 'forward_fill':
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        elif strategy == 'backward_fill':
            data = data.fillna(method='bfill').fillna(method='ffill')
        
        elif strategy == 'drop':
            data = data.dropna()
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in the dataset."""
        logger.info("Handling outliers")
        
        method = self.config['outlier_method']
        threshold = self.config['outlier_threshold']
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col == 'timestamp':
                continue
                
            if method == 'iqr':
                # Interquartile Range method
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                data[col] = data[col].clip(lower_bound, upper_bound)
            
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_mask = z_scores > threshold
                
                if outlier_mask.any():
                    # Replace outliers with median
                    median_value = data[col].median()
                    data.loc[outlier_mask, col] = median_value
            
            elif method == 'isolation_forest':
                # Isolation Forest method (for multivariate outliers)
                try:
                    from sklearn.ensemble import IsolationForest
                    
                    if len(data) > 10:  # Need sufficient data
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outlier_mask = iso_forest.fit_predict(data[[col]]) == -1
                        
                        if outlier_mask.any():
                            # Replace outliers with median
                            median_value = data[col].median()
                            data.loc[outlier_mask, col] = median_value
                
                except ImportError:
                    logger.warning("IsolationForest not available, skipping outlier detection")
        
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features from existing data."""
        logger.info("Engineering features")
        
        # Time-based features
        if self.config['time_features'] and 'timestamp' in data.columns:
            data = self._add_time_features(data)
        
        # Lag features
        if self.config['lag_features']:
            data = self._add_lag_features(data)
        
        # Rolling window features
        if self.config['rolling_features']:
            data = self._add_rolling_features(data)
        
        # Price-based features
        if 'price' in data.columns:
            data = self._add_price_features(data)
        
        # Demand-supply features
        if 'demand' in data.columns and 'supply' in data.columns:
            data = self._add_demand_supply_features(data)
        
        # San Andrés specific features
        if 'consumption' in data.columns and 'generation' in data.columns:
            data = self._add_island_features(data)
        
        return data
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' not in data.columns:
            return data
        
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['day_of_month'] = data['timestamp'].dt.day
        data['month'] = data['timestamp'].dt.month
        data['quarter'] = data['timestamp'].dt.quarter
        data['year'] = data['timestamp'].dt.year
        
        # Cyclical encoding for time features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Business day indicators
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['is_business_hour'] = ((data['hour'] >= 8) & (data['hour'] <= 18)).astype(int)
        data['is_peak_hour'] = ((data['hour'] >= 18) & (data['hour'] <= 22)).astype(int)
        
        return data
    
    def _add_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for time series."""
        lag_periods = self.config['lag_periods']
        target_col = self.config['target_column']
        
        if target_col not in data.columns:
            return data
        
        for lag in lag_periods:
            data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
        
        # Add lag features for other important columns
        other_cols = ['demand', 'supply', 'consumption', 'generation']
        for col in other_cols:
            if col in data.columns:
                for lag in [1, 2, 3, 6]:
                    data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        return data
    
    def _add_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features."""
        rolling_windows = self.config['rolling_windows']
        target_col = self.config['target_column']
        
        if target_col not in data.columns:
            return data
        
        for window in rolling_windows:
            # Rolling statistics
            data[f'{target_col}_ma_{window}'] = data[target_col].rolling(window=window, min_periods=1).mean()
            data[f'{target_col}_std_{window}'] = data[target_col].rolling(window=window, min_periods=1).std()
            data[f'{target_col}_min_{window}'] = data[target_col].rolling(window=window, min_periods=1).min()
            data[f'{target_col}_max_{window}'] = data[target_col].rolling(window=window, min_periods=1).max()
            
            # Rolling ratios
            if window > 1:
                data[f'{target_col}_ratio_{window}'] = data[target_col] / data[f'{target_col}_ma_{window}']
        
        return data
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-specific features."""
        if 'price' not in data.columns:
            return data
        
        # Price changes
        data['price_change'] = data['price'].pct_change()
        data['price_change_abs'] = data['price'].diff()
        
        # Price volatility
        data['price_volatility_24h'] = data['price'].rolling(window=24, min_periods=1).std()
        data['price_volatility_7d'] = data['price'].rolling(window=168, min_periods=1).std()
        
        # Price momentum
        data['price_momentum_1h'] = data['price'] / data['price'].shift(1) - 1
        data['price_momentum_24h'] = data['price'] / data['price'].shift(24) - 1
        
        # Price bands
        data['price_upper_band'] = data['price'].rolling(window=20, min_periods=1).mean() + 2 * data['price'].rolling(window=20, min_periods=1).std()
        data['price_lower_band'] = data['price'].rolling(window=20, min_periods=1).mean() - 2 * data['price'].rolling(window=20, min_periods=1).std()
        data['price_band_position'] = (data['price'] - data['price_lower_band']) / (data['price_upper_band'] - data['price_lower_band'])
        
        return data
    
    def _add_demand_supply_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add demand-supply relationship features."""
        if 'demand' not in data.columns or 'supply' not in data.columns:
            return data
        
        # Basic ratios
        data['demand_supply_ratio'] = data['demand'] / data['supply']
        data['excess_supply'] = data['supply'] - data['demand']
        data['supply_utilization'] = data['demand'] / data['supply']
        
        # Rolling demand-supply features
        data['demand_ma_24h'] = data['demand'].rolling(window=24, min_periods=1).mean()
        data['supply_ma_24h'] = data['supply'].rolling(window=24, min_periods=1).mean()
        data['demand_supply_balance'] = (data['demand_ma_24h'] - data['supply_ma_24h']) / data['supply_ma_24h']
        
        return data
    
    def _add_island_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add San Andrés island-specific features."""
        if 'consumption' not in data.columns or 'generation' not in data.columns:
            return data
        
        # Island energy balance
        data['net_consumption'] = data['consumption'] - data['generation']
        data['renewable_percentage'] = data['generation'] / data['consumption'] * 100
        
        # Storage features (if available)
        if 'storage_level' in data.columns:
            data['storage_percentage'] = data['storage_level'] / data['storage_level'].max() * 100
            data['storage_change'] = data['storage_level'].diff()
            data['storage_autonomy'] = data['storage_level'] / data['consumption'].clip(0.1)
        
        # Island autonomy
        data['island_autonomy_hours'] = data['storage_level'] / data['consumption'].clip(0.1)
        data['is_autonomous'] = (data['island_autonomy_hours'] > 4).astype(int)
        
        return data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data using specified method."""
        logger.info(f"Normalizing data using {self.config['normalization']} method")
        
        method = self.config['normalization']
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        # Skip timestamp and categorical columns
        skip_columns = ['timestamp', 'hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year']
        numeric_columns = [col for col in numeric_columns if col not in skip_columns]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return data
        
        # Fit and transform
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        
        # Store scaler for later use
        self.scalers['main'] = scaler
        
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate processed data."""
        logger.info("Validating processed data")
        
        # Check for infinite values
        inf_mask = np.isinf(data.select_dtypes(include=[np.number])).any(axis=1)
        if inf_mask.any():
            logger.warning(f"Found {inf_mask.sum()} rows with infinite values, removing them")
            data = data[~inf_mask]
        
        # Check for NaN values
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in processed data")
            # Fill remaining NaN with median
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if data[col].isna().any():
                    data[col] = data[col].fillna(data[col].median())
        
        # Check data types
        logger.info(f"Data types: {data.dtypes.value_counts().to_dict()}")
        
        return data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessing parameters and transform data."""
        return self.process_market_data(data)
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted parameters."""
        # This would use stored scalers and imputers
        # For now, just process the data
        return self.process_market_data(data)
    
    def inverse_transform(self, data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Inverse transform normalized data."""
        if 'main' not in self.scalers:
            return data
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        data[columns] = self.scalers['main'].inverse_transform(data[columns])
        return data
    
    def get_feature_importance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get feature importance based on correlation with target."""
        target_col = self.config['target_column']
        
        if target_col not in data.columns:
            return pd.DataFrame()
        
        # Calculate correlations
        correlations = data.corr()[target_col].abs().sort_values(ascending=False)
        
        # Remove target column and NaN values
        correlations = correlations.drop(target_col).dropna()
        
        return pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values,
            'importance': correlations.values / correlations.values.max()
        })
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict:
        """Get summary statistics of processed data."""
        summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isna().sum().to_dict(),
            'numeric_summary': data.describe().to_dict() if not data.empty else {},
            'memory_usage': data.memory_usage(deep=True).sum()
        }
        
        return summary
